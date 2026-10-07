#!/usr/bin/env python3
"""Train and evaluate exchange-before-compression weather autoencoder.

`hybrid_patch_model.py`(core+halo 교환 후 패치 압축)
모델의 실행 진입점입니다. 기존 information-first 실험의 정규화·원본 캐시·
변수 코덱·그리드 코드가 있으면 자동으로 재사용하며, 공간/시간/joint 단계별로
중단 후 재시작(RESUME_SPATIAL 등)을 지원합니다."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import time

from netCDF4 import Dataset
import numpy as np
import torch

from all_variable_data import (
    SEED,
    build_normalized_cache,
    build_spherical_knn_graph,
    build_spherical_patches,
    compute_normalization,
    discover_daily_files,
    inspect_schema,
    padded_patch_arrays,
    save_schema_report,
    specs_from_report,
    stratified_patch_split,
)
from base_training_utils import (
    autocast_context,
    code_mse,
    cpu_state,
    decode_all_variables,
    encode_grid_cache,
    is_sparse_variable,
    reconstruct_write_and_measure,
    sampled_original_losses,
    train_variable_codec,
)
from hybrid_patch_model import (
    HybridExchangeCompressAutoencoder,
    MultiTokenVariableCodec,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/mybook/KCM_output"))
    parser.add_argument("--pattern", default="UP-*.nc")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/all_variable_exchange_then_compress_30day"),
    )
    parser.add_argument(
        "--reuse-precomputed-dir", type=Path, default=None,
        help="reuse normalization/raw cache/variable codec/grid code cache",
    )

    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--halo-cells", type=int, default=8)
    parser.add_argument("--cell-neighbours", type=int, default=8)
    parser.add_argument("--patch-neighbours", type=int, default=8)
    parser.add_argument("--val-patch-fraction", type=float, default=0.10)
    parser.add_argument("--input-token-dim", type=int, default=128)
    parser.add_argument("--grid-slots", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--patch-slots", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--local-layers", type=int, default=2)
    parser.add_argument("--patch-exchange-layers", type=int, default=2)
    parser.add_argument("--temporal-layers", type=int, default=2)
    parser.add_argument("--temporal-decoder-layers", type=int, default=2)
    parser.add_argument("--patch-chunk-size", type=int, default=16)
    parser.add_argument("--temporal-chunk-size", type=int, default=128)
    parser.add_argument("--graph-chunk-size", type=int, default=256)

    # The names below are kept compatible with the reused variable-codec trainer.
    parser.add_argument("--variable-epochs", type=int, default=80)
    parser.add_argument("--variable-steps-per-epoch", type=int, default=200)
    parser.add_argument("--variable-batch-size", type=int, default=128)
    parser.add_argument("--sparse-loss-weight", type=float, default=0.15)
    parser.add_argument("--spatial-epochs", type=int, default=80)
    parser.add_argument("--spatial-steps-per-epoch", type=int, default=4)
    parser.add_argument("--temporal-epochs", type=int, default=100)
    parser.add_argument("--temporal-steps-per-epoch", type=int, default=20)
    parser.add_argument("--temporal-batch-patches", type=int, default=64)
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--joint-steps-per-epoch", type=int, default=8)
    parser.add_argument("--joint-batch-patches", type=int, default=16)
    parser.add_argument("--original-loss-samples", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--joint-patience", type=int, default=10)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-vram-fraction", type=float, default=0.55)
    parser.add_argument("--resume-spatial", action="store_true")
    parser.add_argument("--resume-temporal", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    return parser.parse_args()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unique_parameters(parameters):
    result = []
    seen = set()
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            result.append(parameter)
    return result


def make_patch_layout(lons, lats, patch_size, halo_count, cell_k, patch_k):
    patches = build_spherical_patches(lons, lats, patch_size)
    core_cells, core_valid, core_features, centroids, centroid_latlon = \
        padded_patch_arrays(patches, lons, lats)
    cell_neighbours, node_features, _ = build_spherical_knn_graph(
        lons, lats, cell_k
    )
    halo_cells = np.full((len(patches), halo_count), -1, dtype=np.int64)
    halo_valid = np.zeros((len(patches), halo_count), dtype=bool)
    xyz = node_features[:, :3].astype(np.float64)
    for patch_id, core in enumerate(patches):
        core_set = set(int(value) for value in core)
        candidates = set(int(value) for value in cell_neighbours[core].reshape(-1))
        candidates.difference_update(core_set)
        candidates = np.asarray(sorted(candidates), dtype=np.int64)
        if len(candidates):
            score = xyz[candidates] @ centroids[patch_id].astype(np.float64)
            order = np.argsort(-score, kind="stable")[:halo_count]
            chosen = candidates[order]
            halo_cells[patch_id, :len(chosen)] = chosen
            halo_valid[patch_id, :len(chosen)] = True
    combined_cells = np.concatenate([core_cells, halo_cells], axis=1)
    combined_valid = np.concatenate([core_valid, halo_valid], axis=1)
    patch_neighbours, _, patch_edges = build_spherical_knn_graph(
        centroid_latlon[:, 1], centroid_latlon[:, 0], patch_k
    )
    return {
        "patches": patches,
        "core_cells": core_cells,
        "core_valid": core_valid,
        "core_features": core_features,
        "halo_cells": halo_cells,
        "halo_valid": halo_valid,
        "combined_cells": combined_cells,
        "combined_valid": combined_valid,
        "centroids": centroids,
        "centroid_latlon": centroid_latlon,
        "patch_neighbours": patch_neighbours,
        "patch_edges": patch_edges,
        "node_features": node_features,
    }


def save_layout(layout, path):
    np.savez_compressed(
        path,
        core_cells=layout["core_cells"],
        core_valid=layout["core_valid"],
        core_features=layout["core_features"],
        halo_cells=layout["halo_cells"],
        halo_valid=layout["halo_valid"],
        combined_cells=layout["combined_cells"],
        combined_valid=layout["combined_valid"],
        centroids=layout["centroids"],
        centroid_latlon=layout["centroid_latlon"],
        patch_neighbours=layout["patch_neighbours"],
        patch_edges=layout["patch_edges"],
        node_features=layout["node_features"],
    )


def create_model(args, layout, device):
    return HybridExchangeCompressAutoencoder(
        core_cells=layout["core_cells"],
        core_valid=layout["core_valid"],
        combined_cells=layout["combined_cells"],
        combined_valid=layout["combined_valid"],
        core_features=layout["core_features"],
        patch_neighbours=layout["patch_neighbours"],
        patch_edge_features=layout["patch_edges"],
        input_token_dim=args.input_token_dim,
        grid_slots=args.grid_slots,
        latent_dim=args.latent_dim,
        patch_slots=args.patch_slots,
        heads=args.heads,
        local_layers=args.local_layers,
        patch_exchange_layers=args.patch_exchange_layers,
        temporal_layers=args.temporal_layers,
        temporal_decoder_layers=args.temporal_decoder_layers,
        patch_chunk_size=args.patch_chunk_size,
        temporal_chunk_size=args.temporal_chunk_size,
        graph_chunk_size=args.graph_chunk_size,
    ).to(device)


def evaluate_spatial(model, grid_cache, node_features, validation_cells, device):
    days = np.linspace(
        0, grid_cache.shape[0] - 1, num=min(3, grid_cache.shape[0]), dtype=int
    )
    validation = torch.as_tensor(validation_cells, dtype=torch.long, device=device)
    values = []
    model.eval()
    with torch.no_grad():
        for day in days:
            target = torch.as_tensor(
                np.asarray(grid_cache[day], dtype=np.float32), device=device
            )
            with autocast_context(device):
                prediction, _ = model.spatial_autoencode_day(target, node_features)
                loss = code_mse(prediction[validation], target[validation])
            values.append(float(loss))
    return float(np.mean(values))


def train_spatial(
    model, grid_cache, node_features, training_cells, validation_cells, args, device
):
    parameters = unique_parameters(
        list(model.spatial_encoder_parameters())
        + list(model.spatial_decoder_parameters())
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    train_index = torch.as_tensor(training_cells, dtype=torch.long, device=device)
    rng = np.random.RandomState(SEED + 710)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    for epoch in range(1, args.spatial_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.spatial_steps_per_epoch):
            day = int(rng.randint(0, grid_cache.shape[0]))
            target = torch.as_tensor(
                np.asarray(grid_cache[day], dtype=np.float32), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction, _ = model.spatial_autoencode_day(target, node_features)
                loss = code_mse(prediction[train_index], target[train_index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation = evaluate_spatial(
            model, grid_cache, node_features, validation_cells, device
        )
        print(
            f"[spatial epoch {epoch:03d}] "
            f"train={total / args.spatial_steps_per_epoch:.6f} "
            f"val={validation:.6f}", flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(model)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[spatial early-stop] best={best_epoch} val={best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_validation_code_mse": best}


def encode_spatial_cache(model, grid_cache, node_features, output, device):
    shape = (
        grid_cache.shape[0], model.patch_count, model.patch_slots, model.latent_dim
    )
    cache = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=shape
    )
    model.eval()
    with torch.no_grad():
        for day in range(grid_cache.shape[0]):
            print(f"[spatial-cache {day + 1:02d}/{grid_cache.shape[0]}]", flush=True)
            target = torch.as_tensor(
                np.asarray(grid_cache[day], dtype=np.float32), device=device
            )
            with autocast_context(device):
                latent = model.encode_spatial_day(target, node_features)
            cache[day] = latent.float().cpu().numpy()
            cache.flush()
    return cache


def evaluate_temporal(model, spatial_cache, patch_ids, args, device):
    rng = np.random.RandomState(SEED + 720)
    count = min(args.temporal_batch_patches * 2, len(patch_ids))
    selected = rng.choice(patch_ids, size=count, replace=False)
    target = torch.as_tensor(
        np.asarray(spatial_cache[:, selected], dtype=np.float32), device=device
    )
    model.eval()
    with torch.no_grad(), autocast_context(device):
        prediction, latent = model.temporal_autoencode(target)
        loss = code_mse(prediction, target)
    return float(loss), tuple(latent.shape)


def train_temporal(model, spatial_cache, train_patches, val_patches, args, device):
    parameters = unique_parameters(model.temporal_parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.RandomState(SEED + 730)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    for epoch in range(1, args.temporal_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.temporal_steps_per_epoch):
            selected = rng.choice(
                train_patches,
                size=min(args.temporal_batch_patches, len(train_patches)),
                replace=False,
            )
            target = torch.as_tensor(
                np.asarray(spatial_cache[:, selected], dtype=np.float32), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction, _ = model.temporal_autoencode(target)
                loss = code_mse(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation, shape = evaluate_temporal(
            model, spatial_cache, val_patches, args, device
        )
        print(
            f"[temporal epoch {epoch:03d}] "
            f"train={total / args.temporal_steps_per_epoch:.6f} "
            f"val={validation:.6f} latent={list(shape)}", flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(model)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[temporal early-stop] best={best_epoch} val={best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_validation_code_mse": best}


def patch_core_cells(model, patch_ids):
    patch_ids = torch.as_tensor(
        patch_ids, dtype=torch.long, device=model.core_cells.device
    )
    valid = model.core_valid[patch_ids].reshape(-1)
    return model.core_cells[patch_ids].reshape(-1)[valid]


def joint_batch_loss(
    model, codec, spatial_cache, grid_cache, raw_cache, patch_ids,
    mean_t, std_t, args, device, rng,
):
    patch_ids = np.asarray(patch_ids, dtype=np.int64)
    target_spatial = torch.as_tensor(
        np.asarray(spatial_cache[:, patch_ids], dtype=np.float32), device=device
    )
    restored_spatial, latent = model.temporal_autoencode(target_spatial)
    restored_codes, cells = model.decode_spatial_patches(
        restored_spatial,
        torch.as_tensor(patch_ids, dtype=torch.long, device=device),
    )
    target_codes = torch.as_tensor(
        np.asarray(
            grid_cache[:, cells.detach().cpu().numpy()], dtype=np.float32
        ),
        device=device,
    )
    code = code_mse(restored_codes, target_codes)
    original, _ = sampled_original_losses(
        codec,
        restored_codes,
        raw_cache,
        cells,
        rng,
        args.original_loss_samples,
        mean_t,
        std_t,
        device,
    )
    return 0.10 * code + 0.90 * original, code, original, latent


def evaluate_joint(
    model, codec, spatial_cache, grid_cache, raw_cache, val_patches,
    mean_t, std_t, args, device,
):
    count = min(max(args.joint_batch_patches, 32), len(val_patches))
    selected = np.random.RandomState(SEED + 740).choice(
        val_patches, size=count, replace=False
    )
    model.eval()
    codec.eval()
    rng = np.random.RandomState(SEED + 741)
    with torch.no_grad(), autocast_context(device):
        loss, code, original, latent = joint_batch_loss(
            model, codec, spatial_cache, grid_cache, raw_cache, selected,
            mean_t, std_t, args, device, rng,
        )
    return float(loss), float(code), float(original), tuple(latent.shape)


def train_joint(
    model, codec, spatial_cache, grid_cache, raw_cache,
    train_patches, val_patches, mean_t, std_t, args, device,
):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    for parameter in model.temporal_parameters():
        parameter.requires_grad_(True)
    for parameter in model.spatial_decoder_parameters():
        parameter.requires_grad_(True)
    for module in (codec.grid_to_variable, codec.decoders):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = unique_parameters(
        parameter
        for module in (model, codec)
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate * 0.1, weight_decay=args.weight_decay
    )
    rng = np.random.RandomState(SEED + 750)
    baseline = evaluate_joint(
        model, codec, spatial_cache, grid_cache, raw_cache, val_patches,
        mean_t, std_t, args, device,
    )
    best, best_epoch, bad = baseline[0], 0, 0
    best_model, best_codec = cpu_state(model), cpu_state(codec)
    print(
        f"[joint baseline] val={baseline[0]:.6f} code={baseline[1]:.6f} "
        f"original={baseline[2]:.6f} latent={list(baseline[3])}", flush=True,
    )
    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        codec.train()
        total = 0.0
        for _ in range(args.joint_steps_per_epoch):
            selected = rng.choice(
                train_patches,
                size=min(args.joint_batch_patches, len(train_patches)),
                replace=False,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                loss, _, _, _ = joint_batch_loss(
                    model, codec, spatial_cache, grid_cache, raw_cache, selected,
                    mean_t, std_t, args, device, rng,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation = evaluate_joint(
            model, codec, spatial_cache, grid_cache, raw_cache, val_patches,
            mean_t, std_t, args, device,
        )
        print(
            f"[joint epoch {epoch:03d}] "
            f"train={total / args.joint_steps_per_epoch:.6f} "
            f"val={validation[0]:.6f} code={validation[1]:.6f} "
            f"original={validation[2]:.6f} latent={list(validation[3])}",
            flush=True,
        )
        if validation[0] < best - 1.0e-5:
            best, best_epoch, bad = validation[0], epoch, 0
            best_model, best_codec = cpu_state(model), cpu_state(codec)
        else:
            bad += 1
        if bad >= args.joint_patience:
            print(f"[joint early-stop] best={best_epoch} val={best:.6f}")
            break
    model.load_state_dict(best_model)
    codec.load_state_dict(best_codec)
    model.eval()
    codec.eval()
    return {"best_epoch": best_epoch, "best_validation_loss": best}


def encode_final_latent(model, spatial_cache, output, args, device):
    latent_times = (spatial_cache.shape[0] + 1) // 2
    shape = (
        latent_times, model.patch_count, model.patch_slots, model.latent_dim
    )
    result = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=shape
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, model.patch_count, args.temporal_batch_patches):
            stop = min(model.patch_count, start + args.temporal_batch_patches)
            print(f"[final-encode patches {start}:{stop}]", flush=True)
            values = torch.as_tensor(
                np.asarray(spatial_cache[:, start:stop], dtype=np.float32),
                device=device,
            )
            with autocast_context(device):
                latent = model.encode_temporal(values)
            result[:, start:stop] = latent.float().cpu().numpy()
            result.flush()
    return result


def decode_final_grid_codes(model, latent, output, target_times, args, device):
    grid_count = int(model.core_valid.sum().item())
    shape = (target_times, grid_count, model.grid_dim)
    result = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32, shape=shape
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, model.patch_count, args.temporal_batch_patches):
            stop = min(model.patch_count, start + args.temporal_batch_patches)
            print(f"[final-decode patches {start}:{stop}]", flush=True)
            values = torch.as_tensor(
                np.asarray(latent[:, start:stop], dtype=np.float32), device=device
            )
            ids = torch.arange(start, stop, dtype=torch.long, device=device)
            with autocast_context(device):
                spatial = model.decode_temporal(values, target_times)
                decoded, cells = model.decode_spatial_patches(spatial, ids)
            result[:, cells.cpu().numpy()] = decoded.float().cpu().numpy()
            result.flush()
    return result


def write_latent_netcdf(path, latent, layout, files):
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("latent_time", latent.shape[0])
        dataset.createDimension("patch", latent.shape[1])
        dataset.createDimension("patch_slot", latent.shape[2])
        dataset.createDimension("latent_dim", latent.shape[3])
        dataset.createDimension("core_position", layout["core_cells"].shape[1])
        z = dataset.createVariable(
            "latent", "f4",
            ("latent_time", "patch", "patch_slot", "latent_dim"),
            zlib=False,
        )
        z[:] = np.asarray(latent)
        core = dataset.createVariable(
            "patch_core_grid_index", "i4", ("patch", "core_position")
        )
        core[:] = layout["core_cells"].astype(np.int32)
        valid = dataset.createVariable(
            "patch_core_valid", "i1", ("patch", "core_position")
        )
        valid[:] = layout["core_valid"].astype(np.int8)
        dataset.setncattr("pipeline", "vertical -> spatial exchange -> spatial compression -> temporal exchange -> temporal compression")
        dataset.setncattr("source_days", len(files))
        dataset.setncattr("source_first_file", files[0].name)
        dataset.setncattr("source_last_file", files[-1].name)
        dataset.setncattr("temporal_stride", 2)
        dataset.setncattr("halo_cells_are_context_only", "true")


def model_config(args):
    return {
        "input_token_dim": args.input_token_dim,
        "grid_slots": args.grid_slots,
        "latent_dim": args.latent_dim,
        "patch_slots": args.patch_slots,
        "heads": args.heads,
        "local_layers": args.local_layers,
        "patch_exchange_layers": args.patch_exchange_layers,
        "temporal_layers": args.temporal_layers,
        "temporal_decoder_layers": args.temporal_decoder_layers,
        "patch_chunk_size": args.patch_chunk_size,
        "temporal_chunk_size": args.temporal_chunk_size,
        "graph_chunk_size": args.graph_chunk_size,
    }


def smoke_test(device):
    grids, patches, core_width, halo_width = 12, 4, 3, 2
    core = np.arange(grids, dtype=np.int64).reshape(patches, core_width)
    core_valid = np.ones_like(core, dtype=bool)
    halo = np.stack([
        np.asarray([(3 * p - 1) % grids, (3 * p + 3) % grids])
        for p in range(patches)
    ])
    combined = np.concatenate([core, halo], axis=1)
    combined_valid = np.ones((patches, core_width + halo_width), dtype=bool)
    patch_neighbours = np.stack([
        np.asarray([(p - 1) % patches, (p + 1) % patches])
        for p in range(patches)
    ])
    model = HybridExchangeCompressAutoencoder(
        core, core_valid, combined, combined_valid,
        np.random.randn(patches, core_width, 4).astype(np.float32),
        patch_neighbours,
        np.random.randn(patches, 2, 4).astype(np.float32),
        input_token_dim=8, grid_slots=4, latent_dim=32, patch_slots=3,
        heads=4, local_layers=1, patch_exchange_layers=1,
        temporal_layers=1, temporal_decoder_layers=1,
        patch_chunk_size=2, temporal_chunk_size=4, graph_chunk_size=4,
    ).to(device)
    values = torch.randn(6, grids, 32, device=device)
    node_features = torch.randn(grids, 5, device=device)
    restored, latent = model(values, node_features)
    loss = code_mse(restored, values)
    loss.backward()
    print({
        "status": "OK",
        "input": list(values.shape),
        "latent": list(latent.shape),
        "restored": list(restored.shape),
        "loss": float(loss.detach()),
        "parameters": sum(p.numel() for p in model.parameters()),
    }, flush=True)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if not 0.0 < args.max_vram_fraction <= 1.0:
            raise ValueError("--max-vram-fraction must be in (0,1]")
        torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction, device)
        total = torch.cuda.get_device_properties(device).total_memory / 2**30
        print(
            f"[cuda-memory-limit] {args.max_vram_fraction:.2f} "
            f"≈ {total * args.max_vram_fraction:.2f}/{total:.2f} GiB",
            flush=True,
        )
    set_seed()
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
        f"[schema] variables={len(specs)} features={report['flattened_feature_count']} "
        f"grids={report['grid_count']} days={len(files)}", flush=True,
    )
    if args.schema_only:
        return 0

    layout = make_patch_layout(
        lons, lats, args.patch_size, args.halo_cells,
        args.cell_neighbours, args.patch_neighbours,
    )
    layout_path = args.output_dir / "patch_layout.npz"
    save_layout(layout, layout_path)
    train_patch_ids, val_patch_ids, split = stratified_patch_split(
        layout["centroid_latlon"], args.val_patch_fraction
    )
    training_cells = np.concatenate(
        [layout["patches"][index] for index in train_patch_ids]
    )
    validation_cells = np.concatenate(
        [layout["patches"][index] for index in val_patch_ids]
    )
    split.update({
        "training_cells": int(len(training_cells)),
        "validation_cells": int(len(validation_cells)),
        "patch_count": int(len(layout["patches"])),
        "core_width_max": int(layout["core_cells"].shape[1]),
        "halo_width_max": int(layout["halo_cells"].shape[1]),
    })
    print(f"[patch-layout] {split}", flush=True)

    normalization_path = args.output_dir / "normalization.npz"
    raw_cache_path = args.output_dir / "all_variables_normalized_float32.npy"
    codec_path = args.output_dir / "variable_codec_best.pt"
    grid_cache_path = args.output_dir / "grid_codes_float32.npy"
    sparse_indices = [
        index for index, spec in enumerate(specs) if is_sparse_variable(spec.name)
    ]
    codec = MultiTokenVariableCodec(
        [spec.width for spec in specs],
        token_dim=args.input_token_dim,
        grid_slots=args.grid_slots,
        heads=args.heads,
        sparse_indices=sparse_indices,
    ).to(device)

    reused = args.reuse_precomputed_dir is not None
    if reused:
        source = args.reuse_precomputed_dir
        required = {
            "normalization": source / "normalization.npz",
            "raw": source / "all_variables_normalized_float32.npy",
            "codec": source / "variable_codec_best.pt",
            "grid": source / "grid_codes_float32.npy",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("reuse files missing: " + ", ".join(missing))
        saved = np.load(required["normalization"], allow_pickle=False)
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
        raw_cache = np.load(required["raw"], mmap_mode="r")
        grid_cache = np.load(required["grid"], mmap_mode="r")
        codec.load_state_dict(
            torch.load(required["codec"], map_location="cpu", weights_only=True)
        )
        shutil.copy2(required["normalization"], normalization_path)
        shutil.copy2(required["codec"], codec_path)
        variable_training = {"reused_from": str(source)}
        print(f"[reuse] precomputed vertical codec/cache from {source}", flush=True)
    else:
        mean, std, missing_values = compute_normalization(
            files, specs, training_cells
        )
        np.savez(
            normalization_path,
            mean=mean,
            std=std,
            variable_names=np.asarray([spec.name for spec in specs]),
            offsets=np.asarray([spec.offset for spec in specs] + [specs[-1].stop]),
        )
        report["missing_training_values"] = missing_values
        raw_cache = build_normalized_cache(
            files, specs, mean, std, report["grid_count"], raw_cache_path
        )
        mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
        std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
        variable_training = train_variable_codec(
            codec, raw_cache, training_cells, validation_cells,
            mean_t, std_t, args, device,
        )
        torch.save(cpu_state(codec), codec_path)
        grid_cache = encode_grid_cache(
            codec, raw_cache, grid_cache_path, device, args.decode_batch_size
        )
    expected_raw = (len(files), report["grid_count"], report["flattened_feature_count"])
    expected_grid = (len(files), report["grid_count"], codec.grid_dim)
    if tuple(raw_cache.shape) != expected_raw:
        raise ValueError(f"raw cache {raw_cache.shape} != {expected_raw}")
    if tuple(grid_cache.shape) != expected_grid:
        raise ValueError(f"grid cache {grid_cache.shape} != {expected_grid}")
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    node_features = torch.as_tensor(
        layout["node_features"], dtype=torch.float32, device=device
    )

    model = create_model(args, layout, device)
    spatial_checkpoint = args.output_dir / "spatial_best.pt"
    if args.resume_spatial or args.resume_temporal:
        if not spatial_checkpoint.exists():
            raise FileNotFoundError(spatial_checkpoint)
        model.load_state_dict(
            torch.load(spatial_checkpoint, map_location="cpu", weights_only=True)
        )
        spatial_training = {"resumed": str(spatial_checkpoint)}
    else:
        spatial_training = train_spatial(
            model, grid_cache, node_features, training_cells,
            validation_cells, args, device,
        )
        torch.save(cpu_state(model), spatial_checkpoint)

    spatial_cache_path = args.output_dir / "spatial_patch_latents_float32.npy"
    if args.resume_temporal and spatial_cache_path.exists():
        spatial_cache = np.load(spatial_cache_path, mmap_mode="r")
    else:
        spatial_cache = encode_spatial_cache(
            model, grid_cache, node_features, spatial_cache_path, device
        )

    temporal_checkpoint = args.output_dir / "temporal_best.pt"
    if args.resume_temporal:
        if not temporal_checkpoint.exists():
            raise FileNotFoundError(temporal_checkpoint)
        model.load_state_dict(
            torch.load(temporal_checkpoint, map_location="cpu", weights_only=True)
        )
        temporal_training = {"resumed": str(temporal_checkpoint)}
    else:
        temporal_training = train_temporal(
            model, spatial_cache, train_patch_ids, val_patch_ids, args, device
        )
        torch.save(cpu_state(model), temporal_checkpoint)

    torch.save(
        cpu_state(codec), args.output_dir / "variable_codec_pre_joint.pt"
    )
    joint_training = train_joint(
        model, codec, spatial_cache, grid_cache, raw_cache,
        train_patch_ids, val_patch_ids, mean_t, std_t, args, device,
    )

    full_model_path = args.output_dir / "full_model.pt"
    torch.save({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": report,
        "arguments": vars(args),
        "model_config": model_config(args),
        "variable_codec": cpu_state(codec),
        "hybrid_model": cpu_state(model),
        "sparse_indices": sparse_indices,
    }, full_model_path)
    torch.save(cpu_state(codec), codec_path)

    final_latent_npy = args.output_dir / "latent_float32.npy"
    latent = encode_final_latent(
        model, spatial_cache, final_latent_npy, args, device
    )
    latent_nc = args.output_dir / "latent_float32.nc"
    write_latent_netcdf(latent_nc, latent, layout, files)
    restored_cache_path = args.output_dir / "restored_grid_codes_float32.npy"
    restored_grid = decode_final_grid_codes(
        model, latent, restored_cache_path, len(files), args, device
    )
    metrics = reconstruct_write_and_measure(
        files, args.output_dir, specs, mean, std, codec, restored_grid,
        raw_cache, device, args.decode_batch_size,
    )

    original_bytes = sum(path.stat().st_size for path in files)
    package_files = [
        latent_nc, full_model_path, normalization_path,
        args.output_dir / "schema_report.json", layout_path,
    ]
    r2 = np.asarray([record["R2"] for record in metrics.values()], dtype=np.float64)
    report["hybrid_model"] = {
        "pipeline": [
            "variable/vertical local codec: 6035 -> 8x128 per grid",
            "core+halo local spatial attention before grid removal",
            "learned cross-attention pool to 16x256 per patch",
            "eight-neighbour patch exchange",
            "full-resolution TCN and temporal attention",
            "stride-2 temporal compression after exchange",
        ],
        "stored_shape": list(latent.shape),
    }
    save_schema_report(report, args.output_dir / "schema_report.json")
    compressed_bytes = sum(path.stat().st_size for path in package_files)
    summary = {
        "experiment": "all_variable_exchange_then_compress_30day",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {
            "days": len(files),
            "first_file": files[0].name,
            "last_file": files[-1].name,
        },
        "schema": {
            "modelled_dynamic_floating_variables": len(specs),
            "flattened_features": report["flattened_feature_count"],
            "grid_count": report["grid_count"],
        },
        "validation": split,
        "model": {
            **model_config(args),
            "patch_count": model.patch_count,
            "core_plus_halo_width": model.local_width,
            "latent_shape": list(latent.shape),
            "variable_codec_parameters": sum(p.numel() for p in codec.parameters()),
            "hybrid_parameters": sum(p.numel() for p in model.parameters()),
        },
        "training": {
            "variable_codec": variable_training,
            "spatial_exchange_then_compression": spatial_training,
            "temporal_exchange_then_compression": temporal_training,
            "decoder_joint_finetuning": joint_training,
            "joint_loss": "10% grid-code MSE + 90% equal-variable normalized original MSE",
            "spatial_encoder_frozen_during_joint_stage": True,
        },
        "reconstruction": {
            "scope": "all selected days, all 15002 grids, all modelled variables",
            "unweighted_variable_mean_R2": float(np.mean(r2)),
            "median_variable_R2": float(np.median(r2)),
            "variables_R2_ge_0": int(np.sum(r2 >= 0.0)),
            "variables_R2_ge_0_5": int(np.sum(r2 >= 0.5)),
            "variables_R2_ge_0_8": int(np.sum(r2 >= 0.8)),
            "variables": metrics,
        },
        "storage": {
            "method": "actual file bytes; float32 latent NetCDF",
            "original_netcdf_bytes": original_bytes,
            "compressed_package_bytes": compressed_bytes,
            "included": [path.name for path in package_files],
            "compression_ratio": original_bytes / max(compressed_bytes, 1),
            "reduction_percent": 100.0 * (1.0 - compressed_bytes / original_bytes),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[done] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
