#!/usr/bin/env python3
"""
Step 2 디코더: step2 잠재값 → 복원된 step1 임베딩 [day, grid, embed_dim].

복원한 임베딩은 step1 임베딩과 같은 형식의 .npy로 저장되므로, 그대로
`step1_decoder.py --embeddings`에 넣어 물리 단위로 되돌릴 수 있습니다.
`--physical-eval`은 그 과정을 한 번에 수행하고 변수별 R²/에러율을 보고합니다.

    python step2_decoder.py --step2-dir runs/step2_cnn
    python step2_decoder.py --step2-dir runs/step2_cnn --physical-eval
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from common import autocast_context, load_normalization, pick_device, write_json
from data_utils import load_cube_layout, specs_from_report
from step2_model import CubeCodecBase

MODEL_FILE = "step2_model.pt"
LAYOUT_FILE = "cube_layout.npz"
LATENT_FILE = "step2_latent_float32.npy"
RESTORED_FILE = "restored_step1_embeddings_float32.npy"


def load_step2(step2_dir: Path, device: torch.device):
    checkpoint = torch.load(step2_dir / MODEL_FILE, map_location="cpu", weights_only=False)
    layout = load_cube_layout(step2_dir / LAYOUT_FILE)
    return CubeCodecBase.from_checkpoint(checkpoint, layout).to(device), checkpoint


def decode_latent(model: CubeCodecBase, latent, output: Path, device, day_chunk: int = 1):
    """latent [day, ...] -> restored step1 embeddings [day, grid, embed_dim] (.npy memmap)."""
    restored = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(latent.shape[0], model.grid_count, model.embed_dim),
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, latent.shape[0], day_chunk):
            stop = min(latent.shape[0], start + day_chunk)
            code = torch.as_tensor(np.array(latent[start:stop], dtype=np.float32), device=device)
            with autocast_context(device):
                restored[start:stop] = model.decode(code).float().cpu().numpy()
    restored.flush()
    del restored
    return np.load(output, mmap_mode="r")


def embedding_metrics(restored, reference) -> dict:
    """Step2's own error: restored vs step1 embeddings (independent of step1's error).

    R² and error rate (RMSE / std, %) pool all 256 embedding dimensions, cells and
    days; `per_day` repeats them for each day.
    """
    sse = count = 0.0
    sums = np.zeros(reference.shape[-1])
    sums2 = np.zeros(reference.shape[-1])
    per_day = []
    for day in range(reference.shape[0]):
        ref = np.asarray(reference[day], dtype=np.float64)
        rec = np.asarray(restored[day], dtype=np.float64)
        day_sse = float(np.square(rec - ref).sum())
        day_total = float(np.square(ref - ref.mean(0)).sum())
        per_day.append({"day": day, "mse": day_sse / ref.size, "R2": 1.0 - day_sse / max(day_total, 1e-30),
                        "error_rate_percent": 100.0 * (day_sse / max(day_total, 1e-30)) ** 0.5})
        sse += day_sse
        sums += ref.sum(0)
        sums2 += np.square(ref).sum(0)
        count += len(ref)
    total = float((sums2 - sums ** 2 / count).sum())
    return {"mse": sse / (count * reference.shape[-1]), "R2": 1.0 - sse / max(total, 1.0e-30),
            "error_rate_percent": 100.0 * (sse / max(total, 1.0e-30)) ** 0.5, "per_day": per_day}


def physical_eval(step1_dir: Path, restored, device, days=None, batch_size=4096) -> dict:
    from step1_decoder import CACHE_FILE, evaluate, load_codec
    codec, checkpoint = load_codec(step1_dir, device)
    specs = specs_from_report(checkpoint["schema"])
    mean, std = load_normalization(step1_dir / "normalization.npz")
    raw_cache = np.load(step1_dir / CACHE_FILE, mmap_mode="r")
    return evaluate(codec, restored, raw_cache, specs, mean, std, device, batch_size, None, days)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step2-dir", type=Path, required=True)
    p.add_argument("--latent", type=Path, default=None, help=f"default: <step2-dir>/{LATENT_FILE}")
    p.add_argument("--output", type=Path, default=None, help=f"default: <step2-dir>/{RESTORED_FILE}")
    p.add_argument("--physical-eval", action="store_true")
    p.add_argument("--days", type=int, default=None, help="physical eval on first N days only")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    started = time.perf_counter()
    model, checkpoint = load_step2(args.step2_dir, device)
    latent = np.load(args.latent or args.step2_dir / LATENT_FILE, mmap_mode="r")
    step1_dir = Path(checkpoint["step1_dir"])
    reference = np.load(step1_dir / "step1_embeddings_float32.npy", mmap_mode="r")
    output = args.output or args.step2_dir / RESTORED_FILE
    restored = decode_latent(model, latent, output, device)
    metrics = {"embedding_space": embedding_metrics(restored, reference)}
    if args.physical_eval:
        days = None if args.days is None else range(min(args.days, restored.shape[0]))
        metrics["physical"] = physical_eval(step1_dir, restored, device, days)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.step2_dir / "decoder_metrics.json", metrics)
    print(metrics["embedding_space"])
    if "physical" in metrics:
        m = metrics["physical"]["all_cells"]
        print(f"[physical] R2 mean={m['mean_R2']:.4f} median={m['median_R2']:.4f} "
              f"error rate % mean={m['mean_error_rate_percent']:.2f} median={m['median_error_rate_percent']:.2f}")
    print(f"[done] restored embeddings -> {output}")


if __name__ == "__main__":
    main()
