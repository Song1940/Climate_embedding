#!/usr/bin/env python3
"""
Step 3: 시간축 압축 — 위치별 시간 TCN 오토인코더 (`step3_model.py`, 양자화 없음).

step2 잠재값 [day, ...]을 입력으로 받습니다. step1·step2 모델은 고정이며 이 단계는
step2 잠재값만 봅니다. 위치마다 전체 기간을 시퀀스 [C, T]로 놓고 [L, T/2] float32
잠재값으로 줄입니다. 압축률(값 개수) = 2C / L 이고, 위치별 시간 평균 μ(하루치)를
기간 전체에 한 번 더 저장합니다.

보관용 압축이므로 검증 분할 없이 모든 위치·모든 날짜로 학습하고, 진행 확인과 조기
종료는 전체 자료의 손실로 합니다.

손실 D: 채널별 정규화 이상값의 오차를 σ_c² 가중으로 합친 값으로, step2 잠재값 기준
에러율의 제곱과 같습니다 (에러율 = RMSE/표준편차, step2_decoder.embedding_metrics 정의).

학습 묶음: 무작위 위치 --batch 개. --crop 0(기본)이면 각 위치의 전체 기간을 넣습니다.
기간이 길면(예: 2년) --crop N 으로 짝수 시작점에서 N일씩 잘라 넣고, 자른 경계 쪽
--margin 일은 손실에서 뺍니다(실제 기간의 처음과 끝은 포함). 저장과 복원은 항상 전체
기간을 한 번에 처리합니다.

출력 (<output-dir>):
    step3_model.pt                     config + state_dict(μ, σ 포함) + step2 정보
    step3_latent_float32.npy           저장값 [ceil(T/2), 6, L, S, S]
    step3_restored_latent_float32.npy  복원한 step2 잠재값 (step2와 같은 형식)
    step3_report.json                  3단계 독립 에러, 압축률

    python step3_temporal.py --step2-dir runs/step2_30day_cnn15 --latent-channels 256 --output-dir runs/step3_cnn15_r2
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from common import cpu_state, pick_device, set_seed, write_json
from data_utils import SEED
from step2_decoder import LATENT_FILE as STEP2_LATENT_FILE, MODEL_FILE as STEP2_MODEL_FILE
from step3_decoder import LATENT_FILE, MODEL_FILE, RESTORED_FILE, compare, decode_latent
from step3_model import TemporalTCNCodec, cube_to_sequences, from_cube, pad_even, to_cube


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step2-dir", type=Path, default=Path("runs/step2_30day_cnn15"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--latent-channels", type=int, default=256, help="L: ratio = 2C / L")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--no-linear-path", action="store_true", help="TCN branches only (stalls at ~10.5%%)")
    p.add_argument("--batch", type=int, default=256, help="positions per step")
    p.add_argument("--crop", type=int, default=0, help="0 = whole period; N = random N-day crops")
    p.add_argument("--margin", type=int, default=4, help="days at crop edges left out of the loss")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=40)
    p.add_argument("--learning-rate", type=float, default=5.0e-4)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true", help="continue from step3_model_partial.pt")
    return p.parse_args()


class Problem:
    """Normalized sequences [P, C, T_even] on the device, plus distortion weights."""

    def __init__(self, sequences: np.ndarray, model: TemporalTCNCodec, device):
        self.positions, self.channels, self.days = sequences.shape
        with torch.no_grad():
            a = model.normalize(torch.as_tensor(sequences, device=device))
        self.anomaly = pad_even(a)
        self.valid = torch.zeros(self.anomaly.shape[-1], device=device)
        self.valid[:self.days] = 1.0                              # padded last day is not scored
        var = model.scale.square()
        self.weight = (var / var.mean())[None, :, None]           # D = error_rate²

    def distortion(self, decoded, target, mask):
        """Weighted MSE over valid (position, channel, day) entries."""
        err = (self.weight * (decoded - target).square()).mean(1)  # [N, T]
        return (err * mask).sum() / mask.sum().clamp_min(1.0) / 1.0


def batch_loss(model, problem: Problem, positions, crop, margin, rng):
    target = problem.anomaly[positions]
    mask = problem.valid[None].expand(len(positions), -1)
    t_even = target.shape[-1]
    if crop and crop < t_even:
        start = 2 * rng.randint(0, (t_even - crop) // 2 + 1)
        target, mask = target[..., start:start + crop], mask[..., start:start + crop].clone()
        if start > 0:
            mask[..., :margin] = 0.0
        if start + crop < t_even:
            mask[..., crop - margin:] = 0.0
    decoded, _ = model(target)
    return problem.distortion(decoded, target, mask)


@torch.no_grad()
def full_check(model, problem: Problem, chunk: int = 512) -> float:
    model.eval()
    total = weight = 0.0
    for start in range(0, problem.positions, chunk):
        target = problem.anomaly[start:start + chunk]
        mask = problem.valid[None].expand(target.shape[0], -1)
        decoded, _ = model(target)
        total += float(problem.distortion(decoded, target, mask)) * float(mask.sum())
        weight += float(mask.sum())
    return total / weight


def train(model, problem: Problem, args, save_partial, resume=None):
    start_epoch = 1 if resume is None else int(resume["epochs_run"]) + 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = None
    if args.lr_schedule == "cosine":
        offset = (start_epoch - 1) * args.steps_per_epoch
        def factor(step):
            warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            progress = min(step + offset, total_steps) / max(total_steps, 1)
            return warm * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.RandomState(SEED + 37 + start_epoch - 1)
    best, best_epoch, bad, best_state, history = float("inf"), 0, 0, None, []
    if resume is not None:
        best, best_epoch = float(resume["best_check_loss"]), int(resume["best_epoch"])
        history, best_state = list(resume["history"]), cpu_state(model)
        print(f"[step3 resume] from epoch {start_epoch} (best epoch {best_epoch}, check {best:.6f})",
              flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            positions = torch.as_tensor(rng.choice(problem.positions, size=min(args.batch, problem.positions),
                                                   replace=False), device=problem.anomaly.device)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, problem, positions, args.crop, args.margin, rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss)
        total /= args.steps_per_epoch
        check = full_check(model, problem)
        history.append({"epoch": epoch, "train": total, "check": check})
        print(f"[step3 epoch {epoch:03d}] train={total:.6f} check={check:.6f} "
              f"(err={100 * math.sqrt(check):.2f}%) lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if check < best - 1.0e-7:
            best, best_epoch, bad, best_state = check, epoch, 0, cpu_state(model)
        else:
            bad += 1
        if epoch % args.checkpoint_every == 0 and best_state is not None:
            save_partial(best_state, {"best_epoch": best_epoch, "best_check_loss": best,
                                      "epochs_run": epoch, "history": history})
        if bad >= args.patience:
            print(f"[step3 early-stop] best epoch={best_epoch} check={best:.6f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_check_loss": best, "epochs_run": len(history),
            "history": history}


@torch.no_grad()
def encode_all(model, problem: Problem, side: int, chunk: int = 512) -> np.ndarray:
    parts = [model.encode(problem.anomaly[s:s + chunk]).cpu().numpy()
             for s in range(0, problem.positions, chunk)]
    seq = np.concatenate(parts)                                    # [P, L, T/2]
    p, l, t2 = seq.shape
    return np.ascontiguousarray(seq.reshape(6, side, side, l, t2).transpose(4, 0, 3, 1, 2))


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    set_seed()
    device = pick_device(args.device)
    started = time.perf_counter()

    step2_ck = torch.load(args.step2_dir / STEP2_MODEL_FILE, map_location="cpu", weights_only=False)
    kind = step2_ck["kind"]
    cube = to_cube(np.load(args.step2_dir / STEP2_LATENT_FILE), kind).astype(np.float32)
    days, _, channels, side, _ = cube.shape
    sequences = cube_to_sequences(cube)                           # [P, C, T]
    mean = sequences.mean(2)                                      # [P, C]
    scale = cube.transpose(2, 0, 1, 3, 4).reshape(channels, -1).std(1)
    model = TemporalTCNCodec(channels, mean, scale, hidden=args.hidden,
                             latent_channels=args.latent_channels, blocks=args.blocks,
                             linear_path=not args.no_linear_path).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[step3] step2={args.step2_dir} kind={kind} cube={cube.shape} L={args.latent_channels} "
          f"value_ratio={2 * channels / args.latent_channels:.3f} parameters={params:,} "
          f"decoder_parameters={model.decoder_parameter_count():,}", flush=True)

    def checkpoint(state, training, partial=False):
        return {"created_utc": datetime.now(timezone.utc).isoformat(), "config": model.config,
                "state_dict": state, "step2_dir": str(args.step2_dir.resolve()), "step2_kind": kind,
                "days": days, "arguments": vars(args), "training": training, "partial": partial}

    def save_partial(state, progress):
        torch.save(checkpoint(state, progress, True), out / "step3_model_partial.pt")

    resume = None
    if args.resume:
        partial = torch.load(out / "step3_model_partial.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(partial["state_dict"])
        resume = partial["training"]
    problem = Problem(sequences, model, device)
    training = train(model, problem, args, save_partial, resume)
    torch.save(checkpoint(cpu_state(model), training), out / MODEL_FILE)

    latent = encode_all(model, problem, side)
    np.save(out / LATENT_FILE, latent)
    restored_cube = decode_latent(model, np.load(out / LATENT_FILE), days, device)
    np.save(out / RESTORED_FILE, from_cube(restored_cube, kind))
    metrics = compare(restored_cube, cube)

    step2_bytes = 4 * cube.size
    latent_bytes = 4 * latent.size
    side_bytes = 4 * (mean.size + scale.size)                     # μ and σ, stored once
    decoder_bytes = 4 * model.decoder_parameter_count()
    report = {
        "step": "step3_temporal (per-position temporal TCN autoencoder, float32, no quantization)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step2_dir": str(args.step2_dir), "step2_kind": kind, "days": days,
        "model": {**model.config, "parameters": params,
                  "decoder_parameters": model.decoder_parameter_count()},
        "training": {k: v for k, v in training.items() if k != "history"},
        "latent": {"shape": list(latent.shape)},
        "step3_error": {k: v for k, v in metrics.items() if k != "per_day"},
        "step3_error_per_day": metrics["per_day"],
        "compression": {
            "value_ratio_latent_only": cube.size / latent.size,
            "step2_latent_bytes": step2_bytes,
            "step3_latent_bytes": latent_bytes,
            "mean_scale_bytes": side_bytes,
            "step3_decoder_bytes": decoder_bytes,
            "ratio_with_mean": step2_bytes / (latent_bytes + side_bytes),
            "ratio_with_mean_and_decoder": step2_bytes / (latent_bytes + side_bytes + decoder_bytes),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "step3_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "step3_error_per_day"},
                     ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[done] {out / 'step3_report.json'}", flush=True)


if __name__ == "__main__":
    main()
