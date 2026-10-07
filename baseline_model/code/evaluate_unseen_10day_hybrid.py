#!/usr/bin/env python3
"""Evaluate a trained hybrid model on completely unseen daily NetCDF files.

학습에 쓰지 않은 날짜(예: 31일차 이후 10일)로 모델을 평가하는 스크립트입니다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from all_variable_data import (
    build_normalized_cache,
    discover_daily_files,
    inspect_schema,
    specs_from_report,
)
from base_training_utils import (
    autocast_context,
    encode_grid_cache,
    reconstruct_write_and_measure,
)
from hybrid_patch_model import MultiTokenVariableCodec
from train_hybrid_30day import (
    create_model,
    decode_final_grid_codes,
    encode_final_latent,
    encode_spatial_cache,
    write_latent_netcdf,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/mybook/KCM_output"))
    parser.add_argument("--pattern", default="UP-*.nc")
    parser.add_argument("--test-start-date", default="20000131")
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument(
        "--train-run-dir", type=Path,
        default=Path("runs/all_variable_exchange_then_compress_30day"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/all_variable_exchange_then_compress_30day/unseen_10day"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-vram-fraction", type=float, default=0.55)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--temporal-batch-patches", type=int, default=64)
    return parser.parse_args()


def load_layout(path):
    saved = np.load(path, allow_pickle=False)
    return {name: np.asarray(saved[name]) for name in saved.files}


def namespace_from_training(saved_arguments, evaluation):
    values = dict(saved_arguments)
    values["device"] = evaluation.device
    values["max_vram_fraction"] = evaluation.max_vram_fraction
    values["decode_batch_size"] = evaluation.decode_batch_size
    values["temporal_batch_patches"] = evaluation.temporal_batch_patches
    return argparse.Namespace(**values)


def main():
    evaluation = parse_args()
    if evaluation.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(evaluation.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            evaluation.max_vram_fraction, device
        )
    started = time.perf_counter()
    evaluation.output_dir.mkdir(parents=True, exist_ok=True)

    model_path = evaluation.train_run_dir / "full_model.pt"
    normalization_path = evaluation.train_run_dir / "normalization.npz"
    layout_path = evaluation.train_run_dir / "patch_layout.npz"
    for path in (model_path, normalization_path, layout_path):
        if not path.exists():
            raise FileNotFoundError(path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    train_args = namespace_from_training(checkpoint["arguments"], evaluation)
    layout = load_layout(layout_path)

    files = discover_daily_files(
        evaluation.data_dir,
        evaluation.pattern,
        evaluation.test_start_date,
        evaluation.test_days,
    )
    report, lons, lats = inspect_schema(files[0])
    specs = specs_from_report(report)
    trained_specs = checkpoint["schema"]["modelled_variables"]
    trained_names = [item["name"] for item in trained_specs]
    if [spec.name for spec in specs] != trained_names:
        raise ValueError("test NetCDF schema/order differs from the training schema")

    normalization = np.load(normalization_path, allow_pickle=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    raw_cache_path = evaluation.output_dir / "unseen_normalized_float32.npy"
    raw_cache = build_normalized_cache(
        files, specs, mean, std, report["grid_count"], raw_cache_path
    )

    sparse_indices = checkpoint.get("sparse_indices", [])
    codec = MultiTokenVariableCodec(
        [spec.width for spec in specs],
        token_dim=train_args.input_token_dim,
        grid_slots=train_args.grid_slots,
        heads=train_args.heads,
        sparse_indices=sparse_indices,
    ).to(device)
    codec.load_state_dict(checkpoint["variable_codec"])
    codec.eval()
    model = create_model(train_args, layout, device)
    model.load_state_dict(checkpoint["hybrid_model"])
    model.eval()

    grid_cache_path = evaluation.output_dir / "unseen_grid_codes_float32.npy"
    grid_cache = encode_grid_cache(
        codec, raw_cache, grid_cache_path, device, evaluation.decode_batch_size
    )
    node_features = torch.as_tensor(
        layout["node_features"], dtype=torch.float32, device=device
    )
    spatial_cache_path = evaluation.output_dir / "unseen_spatial_latents_float32.npy"
    spatial_cache = encode_spatial_cache(
        model, grid_cache, node_features, spatial_cache_path, device
    )
    latent_npy = evaluation.output_dir / "unseen_latent_float32.npy"
    latent = encode_final_latent(
        model, spatial_cache, latent_npy, train_args, device
    )
    latent_nc = evaluation.output_dir / "unseen_latent_float32.nc"
    write_latent_netcdf(latent_nc, latent, layout, files)
    restored_path = evaluation.output_dir / "unseen_restored_grid_codes_float32.npy"
    restored = decode_final_grid_codes(
        model, latent, restored_path, len(files), train_args, device
    )
    metrics = reconstruct_write_and_measure(
        files,
        evaluation.output_dir,
        specs,
        mean,
        std,
        codec,
        restored,
        raw_cache,
        device,
        evaluation.decode_batch_size,
    )

    r2 = np.asarray([record["R2"] for record in metrics.values()], dtype=np.float64)
    original_bytes = sum(path.stat().st_size for path in files)
    shared_files = [
        model_path,
        normalization_path,
        layout_path,
        evaluation.train_run_dir / "schema_report.json",
    ]
    shared_bytes = sum(path.stat().st_size for path in shared_files if path.exists())
    latent_bytes = latent_nc.stat().st_size
    summary = {
        "experiment": "all_variable_exchange_then_compress_unseen_10day",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_period": {
            "first": checkpoint["schema"]["source_files"][0],
            "last": checkpoint["schema"]["source_files"][-1],
        },
        "unseen_test_period": {
            "days": len(files), "first": files[0].name, "last": files[-1].name,
        },
        "evaluation_scope": "all unseen days x all 15002 grids x all modelled variables and levels",
        "latent_shape": list(latent.shape),
        "reconstruction": {
            "unweighted_variable_mean_R2": float(np.mean(r2)),
            "median_variable_R2": float(np.median(r2)),
            "variables_R2_ge_0": int(np.sum(r2 >= 0.0)),
            "variables_R2_ge_0_5": int(np.sum(r2 >= 0.5)),
            "variables_R2_ge_0_8": int(np.sum(r2 >= 0.8)),
            "variables_R2_ge_0_9": int(np.sum(r2 >= 0.9)),
            "variables": metrics,
        },
        "storage": {
            "test_original_netcdf_bytes": original_bytes,
            "test_latent_netcdf_bytes": latent_bytes,
            "shared_model_normalization_layout_bytes": shared_bytes,
            "data_only_compression_ratio": original_bytes / max(latent_bytes, 1),
            "including_shared_package_once_ratio": original_bytes / max(latent_bytes + shared_bytes, 1),
            "note": "static/non-modelled template variables are not included in the standalone package size",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = evaluation.output_dir / "unseen_10day_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
