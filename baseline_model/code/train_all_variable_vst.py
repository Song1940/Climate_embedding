#!/usr/bin/env python3
"""Train, serialize, reconstruct, and evaluate the all-variable VST model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from all_variable_data import (
    SEED,
    build_normalized_cache,
    build_spherical_patches,
    compute_normalization,
    discover_daily_files,
    inspect_schema,
    padded_patch_arrays,
    save_schema_report,
    specs_from_report,
    stratified_patch_split,
)
from all_variable_model import (
    SpatialTemporalAutoencoder,
    VariableGridCodec,
    patch_targets,
    scatter_patch_codes,
)
# Re-exported here (not just used internally) because other scripts in this
# project still do `from train_all_variable_vst import autocast_context` etc.
# -- keep these names importable from this module without changes elsewhere.
from vst_core import (
    autocast_context,
    code_mse,
    cpu_state,
    decode_all_variables,
    encode_grid_cache,
    reconstruct_write_and_measure,
    sampled_original_loss,
    set_seed,
    subset_tensors,
    train_variable_codec,
    variable_validation,
    write_latent_netcdf,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/mybook/KCM_output"))
    parser.add_argument("--pattern", default="UP-*.nc")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/all_variable_vst_30day"))
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--val-patch-fraction", type=float, default=0.10)
    parser.add_argument("--patch-neighbours", type=int, default=8)
    parser.add_argument("--patch-slots", type=int, default=4)
    parser.add_argument("--grid-dim", type=int, default=256)
    parser.add_argument("--spatial-dim", type=int, default=128)
    parser.add_argument("--variable-token-dim", type=int, default=96)
    parser.add_argument("--variable-slots", type=int, default=4)
    parser.add_argument("--variable-epochs", type=int, default=40)
    parser.add_argument("--variable-steps-per-epoch", type=int, default=200)
    parser.add_argument("--variable-batch-size", type=int, default=128)
    parser.add_argument("--st-epochs", type=int, default=80)
    parser.add_argument("--original-loss-samples", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    return parser.parse_args()


def evaluate_spatial_temporal(
    model, codec, grid_codes, raw_cache, tensors, device, original_samples
):
    cells, valid, features, neighbours = tensors
    model.eval()
    rng = np.random.RandomState(SEED + 99)
    with torch.no_grad():
        with autocast_context(device):
            prediction, latent = model(grid_codes, cells, valid, features, neighbours)
            target = patch_targets(grid_codes, cells, valid)
            code_loss = code_mse(prediction, target, valid)
            original_loss = sampled_original_loss(
                codec, prediction, raw_cache, cells, valid, rng,
                original_samples, device,
            )
    return float(code_loss), float(original_loss), tuple(latent.shape)


def train_spatial_temporal(
    model, codec, grid_codes, raw_cache, train_tensors, val_tensors, args, device
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    codec.eval()
    rng = np.random.RandomState(SEED + 7)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    train_cells, train_valid, train_features, train_neighbours = train_tensors
    for epoch in range(1, args.st_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            restored, latent = model(
                grid_codes, train_cells, train_valid,
                train_features, train_neighbours,
            )
            target = patch_targets(grid_codes, train_cells, train_valid)
            latent_loss = code_mse(restored, target, train_valid)
            original_loss = sampled_original_loss(
                codec, restored, raw_cache, train_cells, train_valid, rng,
                args.original_loss_samples, device,
            )
            loss = 0.70 * latent_loss + 0.30 * original_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val_code, val_original, val_shape = evaluate_spatial_temporal(
            model, codec, grid_codes, raw_cache, val_tensors, device,
            args.original_loss_samples,
        )
        validation = 0.70 * val_code + 0.30 * val_original
        print(
            f"[spacetime epoch {epoch:03d}] total={float(loss):.6f} "
            f"code={float(latent_loss):.6f} original={float(original_loss):.6f} "
            f"spatial_val={validation:.6f} latent={list(val_shape)}",
            flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(model)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[spacetime early-stop] best epoch={best_epoch} val={best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_spatial_validation_loss": best}


def smoke_test(device):
    print("[smoke] synthetic all-variable model", flush=True)
    codec = VariableGridCodec([60, 1, 4], token_dim=32, grid_dim=48).to(device)
    raw = torch.randn(12, 65, device=device)
    decoded, codes = codec(raw)
    times, grids = 6, 20
    grid_codes = codes[:1].expand(times, grids, -1).contiguous()
    patch_cells = torch.arange(grids, device=device).reshape(4, 5)
    valid = torch.ones(4, 5, dtype=torch.bool, device=device)
    features = torch.randn(4, 5, 4, device=device)
    neighbours = torch.tensor(
        [[1, 2], [0, 2], [1, 3], [2, 1]], dtype=torch.long, device=device
    )
    model = SpatialTemporalAutoencoder(
        grid_dim=48, spatial_dim=32, patch_slots=2, heads=4,
        local_layers=1, temporal_layers=1,
    ).to(device)
    restored, latent = model(grid_codes, patch_cells, valid, features, neighbours)
    loss = codec.variable_balanced_mse(decoded, raw) + restored.float().square().mean()
    loss.backward()
    print({
        "status": "OK", "decoded": list(decoded.shape), "grid": list(codes.shape),
        "restored_patch": list(restored.shape), "latent": list(latent.shape),
        "loss": float(loss.detach()),
    }, flush=True)


def main():
    args = parse_args()
    set_seed()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.smoke_only:
        smoke_test(device)
        return 0

    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = discover_daily_files(args.data_dir, args.pattern, args.start_date, args.days)
    report, lons, lats = inspect_schema(files[0])
    report["source_files"] = [path.name for path in files]
    specs = specs_from_report(report)
    save_schema_report(report, args.output_dir / "schema_report.json")
    print(
        f"[schema] variables={len(specs)} flattened_features="
        f"{report['flattened_feature_count']} grids={report['grid_count']}", flush=True,
    )
    estimated_cache_bytes = (
        len(files) * report["grid_count"] * report["flattened_feature_count"] * 4
    )
    print(
        f"[schema] normalized float32 cache estimate="
        f"{estimated_cache_bytes / 2**30:.3f} GiB",
        flush=True,
    )
    if args.schema_only:
        print(f"[schema-only done] {args.output_dir / 'schema_report.json'}")
        return 0

    patches = build_spherical_patches(lons, lats, args.patch_size)
    patch_cells, patch_valid, patch_features, centroids, centroid_latlon = \
        padded_patch_arrays(patches, lons, lats)
    train_patch_ids, val_patch_ids, split_summary = stratified_patch_split(
        centroid_latlon, args.val_patch_fraction
    )
    training_cells = np.concatenate([patches[index] for index in train_patch_ids])
    validation_cells = np.concatenate([patches[index] for index in val_patch_ids])
    split_summary.update({
        "training_cells": int(training_cells.size),
        "validation_cells": int(validation_cells.size),
        "all_30_days_used_in_both_spatial_splits": True,
        "unseen_date_validation": False,
    })
    print(f"[patches] count={len(patches)} width<={args.patch_size} split={split_summary}")

    normalization_path = args.output_dir / "normalization.npz"
    mean, std, missing = compute_normalization(files, specs, training_cells)
    np.savez(
        normalization_path, mean=mean, std=std,
        variable_names=np.asarray([spec.name for spec in specs]),
        offsets=np.asarray([spec.offset for spec in specs] + [specs[-1].stop]),
    )
    report["normalization_training"] = "training spatial patches only, all selected days"
    report["missing_training_values"] = missing
    save_schema_report(report, args.output_dir / "schema_report.json")

    raw_cache_path = args.output_dir / "all_variables_normalized_float32.npy"
    raw_cache = build_normalized_cache(
        files, specs, mean, std, report["grid_count"], raw_cache_path
    )
    codec = VariableGridCodec(
        [spec.width for spec in specs], token_dim=args.variable_token_dim,
        grid_dim=args.grid_dim, variable_slots=args.variable_slots,
    ).to(device)
    variable_training = train_variable_codec(
        codec, raw_cache, training_cells, validation_cells, args, device
    )
    torch.save(cpu_state(codec), args.output_dir / "variable_codec_best.pt")

    grid_cache_path = args.output_dir / "grid_codes_float32.npy"
    grid_cache = encode_grid_cache(
        codec, raw_cache, grid_cache_path, device, args.decode_batch_size
    )
    grid_codes = torch.as_tensor(np.asarray(grid_cache), device=device)
    train_tensors = subset_tensors(
        patch_cells, patch_valid, patch_features, centroids,
        train_patch_ids, args.patch_neighbours, device,
    )
    val_tensors = subset_tensors(
        patch_cells, patch_valid, patch_features, centroids,
        val_patch_ids, args.patch_neighbours, device,
    )
    st_model = SpatialTemporalAutoencoder(
        grid_dim=args.grid_dim, spatial_dim=args.spatial_dim,
        patch_slots=args.patch_slots,
    ).to(device)
    spacetime_training = train_spatial_temporal(
        st_model, codec, grid_codes, raw_cache,
        train_tensors, val_tensors, args, device,
    )
    torch.save(cpu_state(st_model), args.output_dir / "spatial_temporal_best.pt")

    all_ids = np.arange(len(patches), dtype=np.int64)
    full_tensors = subset_tensors(
        patch_cells, patch_valid, patch_features, centroids,
        all_ids, args.patch_neighbours, device,
    )
    cells_t, valid_t, features_t, neighbours_t = full_tensors
    st_model.eval()
    with torch.no_grad():
        with autocast_context(device):
            restored_patches, latent = st_model(
                grid_codes, cells_t, valid_t, features_t, neighbours_t
            )
        restored_grid = scatter_patch_codes(
            restored_patches.float(), cells_t, valid_t, report["grid_count"]
        )
    latent_path = args.output_dir / "latent_float32.nc"
    write_latent_netcdf(latent_path, latent, patch_cells, patch_valid, files)

    full_model_path = args.output_dir / "full_model.pt"
    torch.save({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": report,
        "arguments": vars(args),
        "variable_codec": cpu_state(codec),
        "spatial_temporal": cpu_state(st_model),
    }, full_model_path)
    metrics = reconstruct_write_and_measure(
        files, args.output_dir, specs, mean, std, codec, restored_grid,
        raw_cache, device, args.decode_batch_size,
    )
    original_bytes = sum(path.stat().st_size for path in files)
    package_files = [
        latent_path, full_model_path, normalization_path,
        args.output_dir / "schema_report.json",
    ]
    compressed_bytes = sum(path.stat().st_size for path in package_files)
    score_values = [value["R2"] for value in metrics.values()]
    summary = {
        "experiment": "all_variable_vertical_spatial_temporal_30day",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {
            "days": len(files), "first_file": files[0].name, "last_file": files[-1].name,
        },
        "schema": {
            "modelled_floating_dynamic_variables": len(specs),
            "flattened_features": report["flattened_feature_count"],
            "losslessly_copied_variable_count": len(report["losslessly_copied_variables"]),
        },
        "validation": split_summary,
        "model": {
            "pipeline": [
                "variable-axis 1D convolution",
                "learned variable-query attention",
                "within-patch Transformer",
                "nearest-patch Transformer",
                "two kernel-3 stride-2 temporal convolutions",
                "bidirectional temporal Transformer",
                "reverse temporal-spatial-variable decoder",
            ],
            "temporal_length": [len(files), (len(files) + 1) // 2,
                                ((len(files) + 1) // 2 + 1) // 2],
            "patches": len(patches),
            "patch_width_max": args.patch_size,
            "patch_slots": args.patch_slots,
            "latent_dimension": args.spatial_dim,
            "variable_codec_parameters": sum(p.numel() for p in codec.parameters()),
            "spatial_temporal_parameters": sum(p.numel() for p in st_model.parameters()),
        },
        "training": {
            "variable_codec": variable_training,
            "spatial_temporal": spacetime_training,
            "spatial_loss": "70% grid-token MSE + 30% equal-variable normalized original MSE",
        },
        "reconstruction": {
            "evaluation_scope": "all selected days and all grids; spatial validation is reported separately during training",
            "unweighted_variable_mean_R2": float(np.mean(score_values)),
            "unweighted_variable_mean_variance_preserved_percent": float(np.mean(score_values) * 100.0),
            "variables": metrics,
        },
        "storage": {
            "method": "actual bytes",
            "original_netcdf_bytes": original_bytes,
            "compressed_package_bytes": compressed_bytes,
            "included": [path.name for path in package_files],
            "compression_ratio": original_bytes / max(compressed_bytes, 1),
            "reduction_percent": 100.0 * (1.0 - compressed_bytes / original_bytes),
            "note": "conservative package includes the full encoder and decoder checkpoint",
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
