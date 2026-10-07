#!/usr/bin/env python3
"""
Step 2: 공간 지역성을 이용한 압축 (큐브스피어 CNN 또는 ViT).

step1이 만든 임베딩 [day, grid, embed_dim]을 입력으로 받습니다. 격자를 큐브스피어
6개 면의 51×51 이미지로 무손실 재배열한 뒤, CNN이나 ViT로 더 작은 잠재값을
만들도록 학습합니다. step1 코덱은 고정(frozen)이며 학습되지 않습니다.

목적은 보관용 압축이므로 검증 분할 없이 모든 날짜·모든 셀로 학습합니다. 진행
확인과 조기 종료는 전체 자료의 손실로 합니다.

손실 = (1 - w) * 임베딩 MSE + w * 원본 공간 MSE   (둘 다 모든 셀 기준)
  원본 공간 MSE는 복원 임베딩을 고정된 step1 디코더로 풀어 정규화 원본과
  비교합니다. 모든 셀을 한 번에 풀면 GPU 메모리를 넘으므로 셀 묶음 단위로
  기울기를 누적합니다(수학적으로 전체 평균과 같음). --original-loss-samples N을
  주면 매 step 무작위 N개 셀만 쓰는 빠른 근사로 바뀝니다.
  (w = --original-loss-weight, 0이면 임베딩 공간만)

출력 (<output-dir>):
    cube_layout.npz                격자 → 6면 매핑 (디코딩에 필요)
    step2_model.pt                 kind + config + state_dict + step1_dir
    step2_latent_float32.npy       잠재값 (CNN [day, 6, c, 13, 13], ViT [day, 726, d])
    restored_step1_embeddings_float32.npy   step2 디코더로 복원한 임베딩
    decoder_metrics.json / summary.json

    python step2_vertical.py --model cnn --step1-dir runs/step1_day1 --output-dir runs/step2_cnn
    python step2_vertical.py --model vit --step1-dir runs/step1_day1 --output-dir runs/step2_vit
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

from common import autocast_context, cpu_state, pick_device, set_seed, write_json
from data_utils import SEED, build_cube_layout, read_coordinates, save_cube_layout
from step1_decoder import CACHE_FILE, EMBEDDINGS_FILE, load_codec
from step2_decoder import (
    LATENT_FILE, LAYOUT_FILE, MODEL_FILE, RESTORED_FILE, decode_latent, embedding_metrics,
    physical_eval,
)
from step2_model import build_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step1-dir", type=Path, default=Path("runs/step1_day1"))
    p.add_argument("--output-dir", type=Path, default=None, help="default: runs/step2_<model>")
    p.add_argument("--model", choices=("cnn", "vit"), default="cnn")
    # CNN
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--latent-channels", type=int, default=64, help="CNN: stored channels per 13x13 face cell")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--stages", type=int, default=2, choices=(1, 2),
                   help="CNN stride-2 stages: 2 -> 13x13 latent, 1 -> 26x26 latent")
    # ViT
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--latent-dim", type=int, default=90, help="ViT: stored numbers per token")
    p.add_argument("--patch", type=int, default=5, help="ViT: cells per token side (5 -> 726 tokens, 3 -> 2166)")
    p.add_argument("--vit-halo", type=int, default=2, help="ViT: halo nodes (51 + 2*halo must divide by patch)")
    # training
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=20)
    p.add_argument("--days-per-step", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=5.0e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--checkpoint-every", type=int, default=10,
                   help="save the best-so-far weights every N epochs (step2_model_partial.pt)")
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--original-loss-weight", type=float, default=0.30)
    p.add_argument("--original-loss-samples", type=int, default=0,
                   help="0 = exact loss over every cell (chunked); N > 0 = random N cells per step")
    p.add_argument("--original-loss-chunk", type=int, default=2048,
                   help="cells per step1-decoder chunk in the exact mode (GPU memory knob)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip-physical-eval", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="continue from <output-dir>/step2_model_partial.pt")
    return p.parse_args()


class EmbeddingSource:
    """Step1 embeddings, kept on the GPU when they fit comfortably."""

    def __init__(self, path: Path, device, gpu_limit_bytes=4 * 2**30):
        self.memmap = np.load(path, mmap_mode="r")
        self.shape = self.memmap.shape
        self.device = device
        self.on_device = None
        if device.type == "cuda" and self.memmap.nbytes <= gpu_limit_bytes:
            self.on_device = torch.as_tensor(np.array(self.memmap), device=device)

    def days(self, day_ids) -> torch.Tensor:
        if self.on_device is not None:
            return self.on_device[torch.as_tensor(day_ids, device=self.device)]
        return torch.as_tensor(np.array(self.memmap[np.asarray(day_ids)]), device=self.device)


def sampled_original_loss(codec, restored, day_ids, raw_cache, rng, samples, chunk, device,
                          train: bool):
    """Original-space loss on `samples` random restored cells.

    The step1 decode runs in chunks on a detached copy (as in full_original_loss),
    so memory stays at one chunk while value and gradient equal the one-shot loss.
    Returns (value, gathered codes, d(loss)/d(codes) or None).
    """
    t, grid = restored.shape[:2]
    day_local = rng.randint(0, t, size=samples)
    cells = rng.randint(0, grid, size=samples)
    order = np.lexsort((cells, day_ids[day_local]))
    day_local, cells = day_local[order], cells[order]
    codes = restored[torch.as_tensor(day_local, device=device), torch.as_tensor(cells, device=device)]
    detached = codes.detach().float().requires_grad_(train)
    value = 0.0
    for start in range(0, samples, chunk):
        stop = min(samples, start + chunk)
        target = torch.as_tensor(np.array(raw_cache[day_ids[day_local[start:stop]], cells[start:stop]],
                                          dtype=np.float32), device=device)
        with autocast_context(device):
            part = codec.variable_balanced_mse(codec.decode(detached[start:stop]), target)
            part = part * ((stop - start) / samples)
        if train:
            part.backward()
        value += float(part.detach())
    return value, codes, (detached.grad if train else None)


def full_original_loss(codec, restored, day_ids, raw_cache, chunk, device, train: bool):
    """Exact original-space loss over every cell of every day in the batch.

    Decoding all cells at once through the step1 codec does not fit in GPU memory,
    so cells go through in chunks. Each chunk's loss is back-propagated only as far
    as a detached copy of `restored`, accumulating d(loss)/d(restored); the caller
    then pushes that gradient through the step2 model once. The result equals the
    gradient of the full-batch mean loss.
    """
    detached = restored.detach().float().requires_grad_(train)
    t, grid = restored.shape[:2]
    total_cells = t * grid
    value = 0.0
    for ti in range(t):
        for start in range(0, grid, chunk):
            stop = min(grid, start + chunk)
            target = torch.as_tensor(np.array(raw_cache[day_ids[ti], start:stop], dtype=np.float32),
                                     device=device)
            with autocast_context(device):
                part = codec.variable_balanced_mse(codec.decode(detached[ti, start:stop]), target)
                part = part * ((stop - start) / total_cells)
            if train:
                part.backward()
            value += float(part.detach())
    return value, (detached.grad if train else None)


def batch_step(model, codec, source, raw_cache, day_ids, args, rng, device, train: bool):
    """One forward pass (and backward when train=True); returns (loss, code, orig)."""
    day_ids = np.sort(np.asarray(day_ids))
    embeddings = source.days(day_ids)
    w = args.original_loss_weight
    with autocast_context(device):
        restored, _ = model(embeddings)
        code = (restored.float() - embeddings.float()).square().mean()
    if w <= 0:
        orig_value = 0.0
        loss = code
    elif args.original_loss_samples > 0:
        orig_value, codes, grad = sampled_original_loss(
            codec, restored, day_ids, raw_cache, rng, args.original_loss_samples,
            args.original_loss_chunk, device, train)
        surrogate = (codes.float() * grad).sum() if train else torch.zeros((), device=device)
        loss = (1.0 - w) * code + w * surrogate
    else:
        orig_value, grad = full_original_loss(codec, restored, day_ids, raw_cache,
                                              args.original_loss_chunk, device, train)
        # surrogate whose gradient w.r.t. restored equals the accumulated full-loss gradient
        surrogate = (restored.float() * grad).sum() if train else torch.zeros((), device=device)
        loss = (1.0 - w) * code + w * surrogate
    if train:
        loss.backward()
    total = (1.0 - w) * float(code.detach()) + w * orig_value
    return total, float(code.detach()), orig_value


def train(model, codec, source, raw_cache, args, device, save_partial=None, resume=None):
    """Train with early stopping on the full-data check loss.

    `resume` is the `training` dict of a partial checkpoint whose weights are
    already loaded into `model`; training continues from the next epoch with the
    same cosine curve (Adam moments restart, so the warm-up is repeated).
    """
    start_epoch = 1 if resume is None else int(resume["epochs_run"]) + 1
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = None
    if args.lr_schedule == "cosine":
        offset = (start_epoch - 1) * args.steps_per_epoch
        def factor(step):
            warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            progress = min(step + offset, total_steps) / max(total_steps, 1)
            return warm * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    rng = np.random.RandomState(SEED + 17 + start_epoch - 1)
    days_total = source.shape[0]
    best, best_epoch, bad, best_state, history = float("inf"), 0, 0, None, []
    if resume is not None:
        best, best_epoch = float(resume["best_check_loss"]), int(resume["best_epoch"])
        history, best_state = list(resume["history"]), cpu_state(model)
        print(f"[step2 resume] from epoch {start_epoch} (best epoch {best_epoch}, "
              f"check {best:.6f})", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        totals = np.zeros(3)
        for _ in range(args.steps_per_epoch):
            day_ids = rng.choice(days_total, size=min(args.days_per_step, days_total), replace=False)
            optimizer.zero_grad(set_to_none=True)
            loss, code, orig = batch_step(model, codec, source, raw_cache, day_ids, args, rng,
                                          device, train=True)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            totals += (loss, code, orig)
        totals /= args.steps_per_epoch

        # progress check on every day with a fixed sample for the original-space term
        model.eval()
        check_rng = np.random.RandomState(SEED + 29)
        check, chunks = np.zeros(3), 0
        with torch.no_grad():
            for start in range(0, days_total, args.days_per_step):
                day_ids = np.arange(start, min(days_total, start + args.days_per_step))
                loss, code, orig = batch_step(model, codec, source, raw_cache, day_ids, args,
                                              check_rng, device, train=False)
                check += (loss, code, orig)
                chunks += 1
        check /= chunks
        history.append({"epoch": epoch, "train": totals.tolist(), "check": check.tolist()})
        print(f"[step2 epoch {epoch:03d}] train={totals[0]:.6f} (code={totals[1]:.6f} "
              f"orig={totals[2]:.6f}) check={check[0]:.6f} (code={check[1]:.6f} orig={check[2]:.6f}) "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if check[0] < best - 1.0e-6:
            best, best_epoch, bad, best_state = float(check[0]), epoch, 0, cpu_state(model)
        else:
            bad += 1
        if save_partial is not None and epoch % args.checkpoint_every == 0 and best_state is not None:
            save_partial(best_state, {"best_epoch": best_epoch, "best_check_loss": best,
                                      "epochs_run": epoch, "history": history})
        if bad >= args.patience:
            print(f"[step2 early-stop] best epoch={best_epoch} check={best:.6f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_check_loss": best,
            "epochs_run": len(history), "history": history}


def encode_all(model, source, output: Path, device):
    with torch.no_grad(), autocast_context(device):
        probe = model.encode(source.days([0]))
    latent = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32,
                                       shape=(source.shape[0],) + tuple(probe.shape[1:]))
    with torch.no_grad():
        for day in range(source.shape[0]):
            with autocast_context(device):
                latent[day] = model.encode(source.days([day]))[0].float().cpu().numpy()
    latent.flush()
    del latent
    return np.load(output, mmap_mode="r")


def main():
    args = parse_args()
    out = args.output_dir or Path(f"runs/step2_{args.model}")
    args.output_dir = out
    set_seed()
    device = pick_device(args.device)
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)

    codec, step1_checkpoint = load_codec(args.step1_dir, device)
    for p in codec.parameters():
        p.requires_grad_(False)
    raw_cache = np.load(args.step1_dir / CACHE_FILE, mmap_mode="r")
    data_dir = Path(step1_checkpoint["arguments"]["data_dir"])
    source_files = step1_checkpoint["schema"]["source_files"]
    lons, lats = read_coordinates(data_dir / source_files[0])
    layout = build_cube_layout(lons, lats, halo=4)
    save_cube_layout(layout, out / LAYOUT_FILE)

    source = EmbeddingSource(args.step1_dir / EMBEDDINGS_FILE, device)
    days, grid_count, embed_dim = source.shape
    model = build_model(args.model, layout, embed_dim, args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"[step2] embeddings={source.shape} model={args.model} parameters={params:,} "
          f"decoder_parameters={model.decoder_parameter_count():,}", flush=True)

    def save_partial(state, progress):
        torch.save({"kind": model.kind, "config": model.config, "state_dict": state,
                    "step1_dir": str(args.step1_dir.resolve()), "arguments": vars(args),
                    "training": progress, "partial": True}, out / "step2_model_partial.pt")

    resume = None
    if args.resume:
        partial = torch.load(out / "step2_model_partial.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(partial["state_dict"])
        resume = partial["training"]
    training = train(model, codec, source, raw_cache, args, device, save_partial, resume)
    torch.save({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "kind": model.kind, "config": model.config, "state_dict": cpu_state(model),
        "step1_dir": str(args.step1_dir.resolve()), "arguments": vars(args),
        "training": training,
    }, out / MODEL_FILE)

    latent = encode_all(model, source, out / LATENT_FILE, device)
    restored = decode_latent(model, latent, out / RESTORED_FILE, device)
    metrics = {"embedding_space": embedding_metrics(restored, source.memmap)}
    if not args.skip_physical_eval:
        metrics["physical"] = physical_eval(args.step1_dir, restored, device)
    write_json(out / "decoder_metrics.json", metrics)

    latent_values = int(np.prod(latent.shape[1:]))
    step1_values = grid_count * embed_dim
    original_values = grid_count * codec.total_features
    original_bytes = sum((data_dir / name).stat().st_size for name in source_files)
    latent_bytes = (out / LATENT_FILE).stat().st_size
    step1_decoder_params = (sum(p.numel() for p in codec.decoders.parameters())
                            + codec.variable_embedding.numel())
    decoder_bytes = 4 * (model.decoder_parameter_count() + step1_decoder_params)
    summary = {
        "step": f"step2_vertical ({args.model} on cubed-sphere faces)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step1_dir": str(args.step1_dir),
        "model": {**model.config, "kind": model.kind, "parameters": params,
                  "decoder_parameters": model.decoder_parameter_count()},
        "training": {k: v for k, v in training.items() if k != "history"},
        "latent": {"path": str(out / LATENT_FILE), "shape": list(latent.shape),
                   "values_per_day": latent_values},
        "compression": {
            "step2_value_ratio_vs_step1": step1_values / latent_values,
            "total_value_ratio_vs_original": original_values / latent_values,
            "original_netcdf_bytes": original_bytes,
            "latent_file_bytes": latent_bytes,
            "file_ratio_latent_only": original_bytes / latent_bytes,
            "decoder_bytes_step1_plus_step2": decoder_bytes,
            "file_ratio_with_decoders": original_bytes / (latent_bytes + decoder_bytes),
        },
        "reconstruction": {
            "embedding_space": metrics["embedding_space"],
            "physical": None if "physical" not in metrics else {
                k: v for k, v in metrics["physical"]["all_cells"].items() if k != "variables"},
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[done] {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
