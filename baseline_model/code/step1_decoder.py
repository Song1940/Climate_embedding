#!/usr/bin/env python3
"""
Step 1 디코더: step1 임베딩 [day, grid, embed_dim] → 정규화 값 [grid, 6035] →
물리 단위로 복원하고 변수별 R²를 측정합니다.

step1 자체 품질 확인뿐 아니라, step2 디코더가 복원한 임베딩
(`step2_decoder.py`의 출력)을 넣어 전체 파이프라인 품질을 잴 때도 씁니다.

    python step1_decoder.py --step1-dir runs/step1
    python step1_decoder.py --step1-dir runs/step1 \
        --embeddings runs/step2/restored_step1_embeddings_float32.npy \
        --output runs/step2/metrics_physical.json
    python step1_decoder.py --step1-dir runs/step1 --write-netcdf --days 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import (
    VariableMetrics, autocast_context, load_normalization, pick_device,
    write_json, write_reconstructed_netcdf,
)
from data_utils import holdout_cells, load_patch_layout, specs_from_report
from step1_model import Step1Codec

CODEC_FILE = "step1_codec.pt"
EMBEDDINGS_FILE = "step1_embeddings_float32.npy"
CACHE_FILE = "normalized_float32.npy"


def load_codec(step1_dir: Path, device: torch.device) -> tuple[Step1Codec, dict]:
    checkpoint = torch.load(step1_dir / CODEC_FILE, map_location="cpu", weights_only=False)
    return Step1Codec.from_checkpoint(checkpoint).to(device), checkpoint


def decode_embeddings(codec: Step1Codec, embeddings, device, batch_size: int = 4096) -> np.ndarray:
    """[cells, embed_dim] -> [cells, features] (normalized units)."""
    output = np.empty((len(embeddings), codec.total_features), dtype=np.float32)
    codec.eval()
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            stop = min(len(embeddings), start + batch_size)
            chunk = torch.as_tensor(np.array(embeddings[start:stop], dtype=np.float32),
                                    device=device)
            with autocast_context(device):
                values = codec.decode(chunk)
            output[start:stop] = values.float().cpu().numpy()
    return output


def evaluate(codec, embeddings, raw_cache, specs, mean, std, device,
             batch_size=4096, validation_cells=None, days=None,
             write_dir: Path | None = None, source_files=None) -> dict:
    """Decode every cell of each day and compare against the normalized cache.

    Returns metrics over all cells and, if given, over held-out validation cells.
    """
    days = range(embeddings.shape[0]) if days is None else days
    all_metrics = VariableMetrics(specs, mean, std)
    per_day = []
    val_metrics = VariableMetrics(specs, mean, std) if validation_cells is not None else None
    if write_dir is not None:
        write_dir.mkdir(parents=True, exist_ok=True)
    for day in days:
        print(f"[step1-decode day {day + 1}/{embeddings.shape[0]}]", flush=True)
        prediction = decode_embeddings(codec, embeddings[day], device, batch_size)
        target = np.asarray(raw_cache[day], dtype=np.float32)
        all_metrics.update(prediction, target)
        day_metrics = VariableMetrics(specs, mean, std)
        day_metrics.update(prediction, target)
        summary = day_metrics.result()
        per_day.append({"day": int(day),
                        "file": None if source_files is None else Path(source_files[day]).name,
                        **{k: summary[k] for k in ("mean_R2", "median_R2", "mean_error_rate_percent",
                                                    "median_error_rate_percent", "max_error_rate_percent")}})
        if val_metrics is not None:
            val_metrics.update(prediction[validation_cells], target[validation_cells])
        if write_dir is not None:
            source = Path(source_files[day])
            write_reconstructed_netcdf(source, write_dir / source.name, specs,
                                       prediction * std + mean)
    result = {"all_cells": all_metrics.result(), "per_day": per_day}
    if val_metrics is not None:
        result["validation_cells"] = val_metrics.result()
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step1-dir", type=Path, default=Path("runs/step1"))
    p.add_argument("--embeddings", type=Path, default=None,
                   help=f"default: <step1-dir>/{EMBEDDINGS_FILE}")
    p.add_argument("--output", type=Path, default=None,
                   help="metrics json (default: next to the embeddings file)")
    p.add_argument("--days", type=int, default=None, help="evaluate only the first N days")
    p.add_argument("--write-netcdf", action="store_true",
                   help="also write reconstructed NetCDF files (~700 MB per day)")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    started = time.perf_counter()
    codec, checkpoint = load_codec(args.step1_dir, device)
    report = checkpoint["schema"]
    specs = specs_from_report(report)
    mean, std = load_normalization(args.step1_dir / "normalization.npz")
    raw_cache = np.load(args.step1_dir / CACHE_FILE, mmap_mode="r")
    embeddings_path = args.embeddings or args.step1_dir / EMBEDDINGS_FILE
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.shape[:2] != raw_cache.shape[:2]:
        raise ValueError(f"embeddings {embeddings.shape} do not match cache {raw_cache.shape}")
    layout = load_patch_layout(args.step1_dir / "patch_layout.npz")
    validation_cells = holdout_cells(layout)

    count = embeddings.shape[0] if args.days is None else min(args.days, embeddings.shape[0])
    data_dir = Path(checkpoint["arguments"]["data_dir"])
    source_files = [data_dir / name for name in report["source_files"]]
    write_dir = embeddings_path.parent / "reconstructed_nc" if args.write_netcdf else None
    metrics = evaluate(codec, embeddings, raw_cache, specs, mean, std, device,
                       args.batch_size, validation_cells, range(count),
                       write_dir, source_files)
    metrics["embeddings"] = str(embeddings_path)
    metrics["days_evaluated"] = count
    metrics["elapsed_seconds"] = time.perf_counter() - started
    output = args.output or embeddings_path.parent / f"{embeddings_path.stem}_metrics.json"
    write_json(output, metrics)
    for scope in ("all_cells", "validation_cells"):
        if scope not in metrics:
            continue
        m = metrics[scope]
        print(f"[{scope}] R2 mean={m['mean_R2']:.4f} median={m['median_R2']:.4f} | "
              f"error rate % mean={m['mean_error_rate_percent']:.2f} "
              f"median={m['median_error_rate_percent']:.2f} max={m['max_error_rate_percent']:.2f} | "
              f"R2 undefined={m['variables_R2_undefined']}")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
