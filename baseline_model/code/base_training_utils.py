#!/usr/bin/env python3
"""Train/evaluate information-first all-variable compression.

상혁씨가 작업중이신 `information_first_model.py`(그래프 기반) 모델을
변수→공간→시간→**joint 미세조정**의 4단계로 학습·평가하는 공용 함수 모음입니다.
강수 등 대부분의 값이 0인 "희소(sparse) 변수"를 위한 별도 발생확률 손실
(`occurrence_bce`)이 추가되어 있습니다. 이 중 `set_seed`, `autocast_context`,
`cpu_state`, `decode_all_variables`, `encode_grid_cache`, `matrix_to_source_shape`,
`assign_reconstructed`, `reconstruct_write_and_measure` 8개 함수는 제 쪽
`vst_core.py`와 완전히 동일함을 확인하여 그것을 그대로 가져다 씁니다."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
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
from information_first_model import (
    InformationFirstAutoencoder,
    MultiTokenVariableCodec,
)
# These 8 are byte-identical to the versions in vst_core.py (confirmed by AST
# diff when this colleague pipeline was merged into the shared project) --
# reused from there instead of kept as a second copy. code_mse,
# train_variable_codec, variable_validation, and write_latent_netcdf are
# genuinely different here (sparse-variable BCE loss, cell-indexed rather
# than patch-indexed latent) and stay defined locally below.
from vst_core import (
    assign_reconstructed,
    autocast_context,
    cpu_state,
    decode_all_variables,
    encode_grid_cache,
    matrix_to_source_shape,
    reconstruct_write_and_measure,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/mybook/KCM_output"))
    parser.add_argument("--pattern", default="UP-*.nc")
    parser.add_argument("--start-date", default="20000101")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/all_variable_information_first_30day"),
    )
    parser.add_argument("--patch-size", type=int, default=15)
    parser.add_argument("--val-patch-fraction", type=float, default=0.10)
    parser.add_argument("--graph-neighbours", type=int, default=8)
    parser.add_argument("--exchange-layers", type=int, default=3)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--graph-chunk-size", type=int, default=512)
    parser.add_argument("--temporal-chunk-size", type=int, default=256)
    parser.add_argument("--slot-chunk-size", type=int, default=2048)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--grid-slots", type=int, default=8)
    parser.add_argument("--exchange-slots", type=int, default=4)
    parser.add_argument("--variable-epochs", type=int, default=40)
    parser.add_argument("--variable-steps-per-epoch", type=int, default=200)
    parser.add_argument("--variable-batch-size", type=int, default=128)
    parser.add_argument("--spatial-epochs", type=int, default=20)
    parser.add_argument("--spatial-steps-per-epoch", type=int, default=10)
    parser.add_argument("--spatial-mask-fraction", type=float, default=0.15)
    parser.add_argument("--temporal-epochs", type=int, default=40)
    parser.add_argument("--temporal-steps-per-epoch", type=int, default=100)
    parser.add_argument("--temporal-batch-grids", type=int, default=128)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--joint-patience", type=int, default=10)
    parser.add_argument("--original-loss-samples", type=int, default=2048)
    parser.add_argument("--sparse-loss-weight", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=3.0e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-vram-fraction", type=float, default=0.80,
        help="maximum fraction of the visible GPU memory usable by this process",
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument(
        "--resume-precomputed", action="store_true",
        help=(
            "reuse normalization, raw cache, variable codec, and grid-code cache "
            "already present in output-dir"
        ),
    )
    parser.add_argument(
        "--resume-prejoint", action="store_true",
        help=(
            "also reuse pre_joint_checkpoint.pt and restart directly at joint "
            "fine-tuning"
        ),
    )
    parser.add_argument(
        "--resume-full-model", action="store_true",
        help=(
            "load output-dir/full_model.pt and continue only joint fine-tuning"
        ),
    )
    return parser.parse_args()


SPARSE_NAMES = {
    "cld", "lcld", "mcld", "hcld", "tcld", "qc", "qi", "qr", "qs",
    "pr", "snowd", "i_qc_sum", "i_qi_sum", "i_qs_g_sum", "i_cldf",
}


def is_sparse_variable(name):
    name = name.lower()
    return (
        name in SPARSE_NAMES
        or name.startswith(("qc_", "qi_", "qr_", "qs_", "cld_", "pr_"))
        or name.endswith(("_qc", "_qi", "_qr", "_qs", "_cld", "_pr"))
        or "precip" in name
    )


def occurrence_bce(codec, logits, target, mean_t, std_t):
    losses = []
    for index, prediction in logits.items():
        start, stop = codec.offsets[index], codec.offsets[index + 1]
        physical = target[:, start:stop].float() * std_t[start:stop] + mean_t[start:stop]
        threshold = torch.clamp(std_t[start:stop].abs() * 1.0e-6, min=1.0e-12)
        label = (physical.abs() > threshold).float()
        positive = label.sum(dim=0)
        negative = label.shape[0] - positive
        pos_weight = torch.clamp(negative / (positive + 1.0), min=1.0, max=20.0)
        losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(), label, pos_weight=pos_weight
        ))
    if not losses:
        return target.new_zeros((), dtype=torch.float32)
    return torch.stack(losses).mean()


def variable_validation(
    codec, cache, validation_cells, mean_t, std_t, device, batch_size,
    sparse_weight, samples=8192,
):
    rng = np.random.RandomState(SEED + 41)
    count = min(samples, cache.shape[0] * len(validation_cells))
    days = rng.randint(0, cache.shape[0], size=count)
    cells = rng.choice(validation_cells, size=count, replace=True)
    total, batches = 0.0, 0
    codec.eval()
    with torch.no_grad():
        for start in range(0, count, batch_size):
            stop = min(count, start + batch_size)
            target = torch.as_tensor(
                np.asarray(cache[days[start:stop], cells[start:stop]], dtype=np.float32),
                device=device,
            )
            with autocast_context(device):
                prediction, _, logits = codec(target)
                regression = codec.variable_balanced_mse(prediction, target)
                sparse = occurrence_bce(codec, logits, target, mean_t, std_t)
                loss = (1.0 - sparse_weight) * regression + sparse_weight * sparse
            total += float(loss) * (stop - start)
            batches += stop - start
    return total / max(batches, 1)


def train_variable_codec(
    codec, cache, training_cells, validation_cells, mean_t, std_t, args, device
):
    optimizer = torch.optim.AdamW(
        codec.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.RandomState(SEED)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    codec.train()
    for epoch in range(1, args.variable_epochs + 1):
        total = 0.0
        for _ in range(args.variable_steps_per_epoch):
            days = rng.randint(0, cache.shape[0], size=args.variable_batch_size)
            cells = rng.choice(training_cells, size=args.variable_batch_size, replace=True)
            target = torch.as_tensor(
                np.asarray(cache[days, cells], dtype=np.float32), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction, _, logits = codec(target)
                regression = codec.variable_balanced_mse(prediction, target)
                sparse = occurrence_bce(codec, logits, target, mean_t, std_t)
                loss = (
                    (1.0 - args.sparse_loss_weight) * regression
                    + args.sparse_loss_weight * sparse
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation = variable_validation(
            codec, cache, validation_cells, mean_t, std_t, device,
            args.variable_batch_size, args.sparse_loss_weight,
        )
        train_loss = total / args.variable_steps_per_epoch
        print(
            f"[variable epoch {epoch:03d}] train={train_loss:.6f} "
            f"spatial_val={validation:.6f} sparse_bce_weight="
            f"{args.sparse_loss_weight:.2f}", flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(codec)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[variable early-stop] best epoch={best_epoch} val={best:.6f}")
            break
        codec.train()
    if best_state is not None:
        codec.load_state_dict(best_state)
    codec.eval()
    return {"best_epoch": best_epoch, "best_spatial_validation_loss": best}


def graph_tensors(lons, lats, cell_indices, neighbours, device):
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    graph_neighbours, node_features, edge_features = build_spherical_knn_graph(
        lons, lats, neighbours, cell_indices=cell_indices
    )
    return (
        torch.as_tensor(cell_indices, dtype=torch.long, device=device),
        torch.as_tensor(node_features, dtype=torch.float32, device=device),
        torch.as_tensor(graph_neighbours, dtype=torch.long, device=device),
        torch.as_tensor(edge_features, dtype=torch.float32, device=device),
    )


def code_mse(prediction, target):
    return (prediction.float() - target.float()).square().mean()


def sampled_original_losses(
    codec, restored, raw_cache, cell_indices, rng, sample_count,
    mean_t, std_t, device,
):
    count = min(sample_count, restored.shape[0] * restored.shape[1])
    chosen = rng.randint(0, restored.shape[1], size=count)
    days_np = rng.randint(0, restored.shape[0], size=count)
    days = torch.as_tensor(days_np, dtype=torch.long, device=device)
    local_cells = torch.as_tensor(chosen, dtype=torch.long, device=device)
    predicted_codes = restored[days, local_cells]
    original_cells = cell_indices[local_cells]
    target_np = np.asarray(
        raw_cache[days_np, original_cells.detach().cpu().numpy()], dtype=np.float32
    )
    target = torch.as_tensor(target_np, device=device)
    prediction, logits = codec.decode_with_occurrence(predicted_codes)
    regression = codec.variable_balanced_mse(prediction, target)
    sparse = occurrence_bce(codec, logits, target, mean_t, std_t)
    return regression, sparse


def _stage_parameters(*modules):
    parameters = []
    for module in modules:
        parameters.extend(parameter for parameter in module.parameters())
    return parameters


def evaluate_spatial_stage(model, grid_codes, tensors, mask_fraction, device):
    cells, node_features, neighbours, edge_features = tensors
    cell_np = cells.detach().cpu().numpy()
    days = np.linspace(0, grid_codes.shape[0] - 1, num=min(4, grid_codes.shape[0]), dtype=int)
    rng = np.random.RandomState(SEED + 101)
    model.eval()
    values = []
    with torch.no_grad():
        for day in days:
            target = torch.as_tensor(
                np.asarray(grid_codes[day, cell_np], dtype=np.float32), device=device
            )
            mask_count = max(1, int(len(cell_np) * mask_fraction))
            chosen = rng.choice(len(cell_np), size=mask_count, replace=False)
            mask = torch.zeros(len(cell_np), dtype=torch.bool, device=device)
            mask[torch.as_tensor(chosen, device=device)] = True
            with autocast_context(device):
                prediction = model.spatial_pretrain(
                    target[None], node_features, neighbours, edge_features, mask
                )[0]
                masked = code_mse(prediction[mask], target[mask])
                full = code_mse(prediction, target)
            values.append(float(masked + 0.05 * full))
    return float(np.mean(values))


def train_spatial_stage(model, grid_codes, train_tensors, val_tensors, args, device):
    parameters = list(model.slot_bottleneck_parameters()) + _stage_parameters(
        model.coordinate_mlp, model.spatial_encoder, model.spatial_pretrain_head
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.RandomState(SEED + 201)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    train_cells, train_node_features, train_neighbours, train_edges = train_tensors
    train_np = train_cells.detach().cpu().numpy()
    for epoch in range(1, args.spatial_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.spatial_steps_per_epoch):
            day = int(rng.randint(0, grid_codes.shape[0]))
            target = torch.as_tensor(
                np.asarray(grid_codes[day, train_np], dtype=np.float32), device=device
            )
            mask_count = max(1, int(len(train_np) * args.spatial_mask_fraction))
            chosen = rng.choice(len(train_np), size=mask_count, replace=False)
            mask = torch.zeros(len(train_np), dtype=torch.bool, device=device)
            mask[torch.as_tensor(chosen, device=device)] = True
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction = model.spatial_pretrain(
                    target[None], train_node_features, train_neighbours, train_edges, mask
                )[0]
                masked_loss = code_mse(prediction[mask], target[mask])
                full_loss = code_mse(prediction, target)
                loss = masked_loss + 0.05 * full_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation = evaluate_spatial_stage(
            model, grid_codes, val_tensors, args.spatial_mask_fraction, device
        )
        print(
            f"[spatial epoch {epoch:03d}] train="
            f"{total / args.spatial_steps_per_epoch:.6f} val={validation:.6f}",
            flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(model)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[spatial early-stop] best epoch={best_epoch} val={best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_masked_validation_loss": best}


def evaluate_temporal_stage(model, grid_codes, validation_cells, args, device):
    rng = np.random.RandomState(SEED + 301)
    count = min(args.temporal_batch_grids * 4, len(validation_cells))
    cells = rng.choice(validation_cells, size=count, replace=False)
    target = torch.as_tensor(
        np.asarray(grid_codes[:, cells], dtype=np.float32), device=device
    )
    model.eval()
    with torch.no_grad(), autocast_context(device):
        prediction, latent = model.temporal_pretrain(target)
        loss = code_mse(prediction, target)
    return float(loss), tuple(latent.shape)


def train_temporal_stage(
    model, grid_codes, training_cells, validation_cells, args, device
):
    parameters = list(model.slot_bottleneck_parameters()) + _stage_parameters(
        model.temporal_encoder, model.temporal_down, model.temporal_refine,
        model.temporal_decoder, model.temporal_pretrain_head,
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    rng = np.random.RandomState(SEED + 401)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    for epoch in range(1, args.temporal_epochs + 1):
        model.train()
        total = 0.0
        for _ in range(args.temporal_steps_per_epoch):
            cells = rng.choice(
                training_cells, size=args.temporal_batch_grids, replace=True
            )
            target = torch.as_tensor(
                np.asarray(grid_codes[:, cells], dtype=np.float32), device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction, _ = model.temporal_pretrain(target)
                loss = code_mse(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.detach())
        validation, latent_shape = evaluate_temporal_stage(
            model, grid_codes, validation_cells, args, device
        )
        print(
            f"[temporal epoch {epoch:03d}] train="
            f"{total / args.temporal_steps_per_epoch:.6f} val={validation:.6f} "
            f"latent={list(latent_shape)}",
            flush=True,
        )
        if validation < best - 1.0e-5:
            best, best_epoch, bad = validation, epoch, 0
            best_state = cpu_state(model)
        else:
            bad += 1
        if bad >= args.patience:
            print(f"[temporal early-stop] best epoch={best_epoch} val={best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "best_temporal_validation_loss": best}


def evaluate_joint_stage(
    model, codec, grid_codes, raw_cache, tensors, mean_t, std_t, args, device
):
    cells, node_features, neighbours, edge_features = tensors
    cell_np = cells.detach().cpu().numpy()
    target = torch.as_tensor(
        np.asarray(grid_codes[:, cell_np], dtype=np.float32), device=device
    )
    rng = np.random.RandomState(SEED + 501)
    model.eval()
    codec.eval()
    with torch.no_grad(), autocast_context(device):
        prediction, latent = model(target, node_features, neighbours, edge_features)
        code = code_mse(prediction, target)
        original, sparse = sampled_original_losses(
            codec, prediction, raw_cache, cells, rng,
            args.original_loss_samples, mean_t, std_t, device,
        )
        total = (
            0.10 * code
            + (0.90 - args.sparse_loss_weight) * original
            + args.sparse_loss_weight * sparse
        )
    return float(total), float(code), float(original), float(sparse), tuple(latent.shape)


def train_joint_stage(
    model, codec, grid_codes, raw_cache, train_tensors, val_tensors,
    mean_t, std_t, args, device,
):
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    for module in (codec.grid_to_variable, codec.decoders, codec.occurrence_heads):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    decoder_parameters = [p for p in codec.parameters() if p.requires_grad]
    parameters = list(model.parameters()) + decoder_parameters
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate * 0.1, weight_decay=args.weight_decay
    )
    cells, node_features, neighbours, edge_features = train_tensors
    cell_np = cells.detach().cpu().numpy()
    target = torch.as_tensor(
        np.asarray(grid_codes[:, cell_np], dtype=np.float32), device=device
    )
    rng = np.random.RandomState(SEED + 601)
    baseline = evaluate_joint_stage(
        model, codec, grid_codes, raw_cache, val_tensors,
        mean_t, std_t, args, device,
    )
    best, best_epoch, bad = baseline[0], 0, 0
    best_model, best_codec = cpu_state(model), cpu_state(codec)
    print(
        f"[joint baseline] val={baseline[0]:.6f} "
        f"code={baseline[1]:.6f} original={baseline[2]:.6f} "
        f"sparse={baseline[3]:.6f} latent={list(baseline[4])}",
        flush=True,
    )
    for epoch in range(1, args.joint_epochs + 1):
        model.train()
        codec.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            prediction, latent = model(target, node_features, neighbours, edge_features)
            code = code_mse(prediction, target)
            original, sparse = sampled_original_losses(
                codec, prediction, raw_cache, cells, rng,
                args.original_loss_samples, mean_t, std_t, device,
            )
            loss = (
                0.10 * code
                + (0.90 - args.sparse_loss_weight) * original
                + args.sparse_loss_weight * sparse
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        validation = evaluate_joint_stage(
            model, codec, grid_codes, raw_cache, val_tensors,
            mean_t, std_t, args, device,
        )
        print(
            f"[joint epoch {epoch:03d}] train={float(loss):.6f} "
            f"val={validation[0]:.6f} code={validation[1]:.6f} "
            f"original={validation[2]:.6f} sparse={validation[3]:.6f} "
            f"latent={list(validation[4])}", flush=True,
        )
        if validation[0] < best - 1.0e-5:
            best, best_epoch, bad = validation[0], epoch, 0
            best_model, best_codec = cpu_state(model), cpu_state(codec)
        else:
            bad += 1
        if bad >= args.joint_patience:
            print(f"[joint early-stop] best epoch={best_epoch} val={best:.6f}")
            break
    if best_model is not None:
        model.load_state_dict(best_model)
        codec.load_state_dict(best_codec)
    model.eval()
    codec.eval()
    return {"best_epoch": best_epoch, "best_joint_validation_loss": best}


def write_latent_netcdf(path, latent, cell_indices, lons, lats, files):
    values = latent.detach().float().cpu().numpy()
    with Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("latent_time", values.shape[0])
        dataset.createDimension("grid", values.shape[1])
        dataset.createDimension("grid_slot", values.shape[2])
        dataset.createDimension("latent_dim", values.shape[3])
        z = dataset.createVariable(
            "latent", "f4", ("latent_time", "grid", "grid_slot", "latent_dim")
        )
        z[:] = values
        grid = dataset.createVariable("grid_index", "i4", ("grid",))
        grid[:] = np.asarray(cell_indices, dtype=np.int32)
        longitude = dataset.createVariable("lons", "f4", ("grid",))
        longitude[:] = np.asarray(lons, dtype=np.float32)[cell_indices]
        latitude = dataset.createVariable("lats", "f4", ("grid",))
        latitude[:] = np.asarray(lats, dtype=np.float32)[cell_indices]
        dataset.setncattr("source_days", len(files))
        dataset.setncattr("source_first_file", files[0].name)
        dataset.setncattr("source_last_file", files[-1].name)
        dataset.setncattr("temporal_stride", "kernel3/stride2 once after context exchange")
        dataset.setncattr("spatial_storage", "multiple latent vectors per original grid")
        dataset.setncattr("spatial_compression", "none")
        dataset.setncattr("slot_compression_before_exchange", "8_to_4")
        dataset.setncattr("temporal_compression_after_exchange", "30_to_15")


def smoke_test(device):
    print("[smoke] synthetic information-first model", flush=True)
    mean_t = torch.zeros(65, device=device)
    std_t = torch.ones(65, device=device)
    codec = MultiTokenVariableCodec(
        [60, 1, 4], token_dim=32, grid_slots=4, heads=4,
        sparse_indices=[2],
    ).to(device)
    raw = torch.randn(12, 65, device=device)
    decoded, codes, logits = codec(raw)
    times, grids = 6, 20
    grid_codes = codes[:1].expand(times, grids, -1).contiguous()
    offsets = torch.tensor([-2, -1, 1, 2], device=device)
    neighbours = (
        torch.arange(grids, device=device)[:, None] + offsets[None]
    ) % grids
    node_features = torch.randn(grids, 5, device=device)
    edge_features = torch.randn(grids, 4, 4, device=device)
    model = InformationFirstAutoencoder(
        token_dim=32, grid_slots=4, exchange_slots=2, heads=4,
        exchange_layers=2, decoder_layers=1,
        graph_chunk_size=8, temporal_chunk_size=8,
        slot_chunk_size=8,
    ).to(device)
    restored, latent = model(
        grid_codes, node_features, neighbours, edge_features
    )
    occurrence = occurrence_bce(codec, logits, raw, mean_t, std_t)
    loss = codec.variable_balanced_mse(decoded, raw) + occurrence + code_mse(
        restored, grid_codes
    )
    loss.backward()
    print({
        "status": "OK", "decoded": list(decoded.shape), "grid": list(codes.shape),
        "restored_grid": list(restored.shape), "latent": list(latent.shape),
        "grid_slots": codec.grid_slots,
        "exchange_slots": model.exchange_slots,
        "loss": float(loss.detach()),
    }, flush=True)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if not 0.0 < args.max_vram_fraction <= 1.0:
            raise ValueError("--max-vram-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(
            args.max_vram_fraction, device=device
        )
        total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        print(
            f"[cuda-memory-limit] fraction={args.max_vram_fraction:.2f} "
            f"allocator_limit≈{total_gib * args.max_vram_fraction:.2f} GiB "
            f"of {total_gib:.2f} GiB", flush=True,
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
    _, _, _, _, centroid_latlon = \
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
    print(
        f"[validation partition] patches={len(patches)} "
        f"width<={args.patch_size} split={split_summary}", flush=True,
    )

    normalization_path = args.output_dir / "normalization.npz"
    raw_cache_path = args.output_dir / "all_variables_normalized_float32.npy"
    codec_path = args.output_dir / "variable_codec_best.pt"
    grid_cache_path = args.output_dir / "grid_codes_float32.npy"
    if args.resume_prejoint or args.resume_full_model:
        args.resume_precomputed = True
    if args.resume_precomputed:
        required = [normalization_path, raw_cache_path, codec_path, grid_cache_path]
        missing_paths = [str(path) for path in required if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "--resume-precomputed requested but files are missing: "
                + ", ".join(missing_paths)
            )
        saved = np.load(normalization_path, allow_pickle=False)
        mean = np.asarray(saved["mean"], dtype=np.float32)
        std = np.asarray(saved["std"], dtype=np.float32)
        if mean.shape != (report["flattened_feature_count"],):
            raise ValueError(
                "reused normalization does not match current schema: "
                f"{mean.shape} != {(report['flattened_feature_count'],)}"
            )
        missing = {"reused_from_previous_attempt": True}
        report["normalization_training"] = (
            "reused from previous attempt; originally fit on training spatial "
            "patches over all selected days"
        )
        print("[resume] normalization and raw/grid caches will be reused", flush=True)
    else:
        mean, std, missing = compute_normalization(files, specs, training_cells)
        np.savez(
            normalization_path, mean=mean, std=std,
            variable_names=np.asarray([spec.name for spec in specs]),
            offsets=np.asarray([spec.offset for spec in specs] + [specs[-1].stop]),
        )
        report["normalization_training"] = (
            "training spatial patches only, all selected days"
        )
    report["missing_training_values"] = missing
    report["spatial_model"] = {
        "type": "information-first spherical grid Graph Transformer",
        "graph_neighbours": args.graph_neighbours,
        "exchange_layers": args.exchange_layers,
        "stored_grid_count": int(report["grid_count"]),
        "spatial_compression": False,
        "slot_compression_before_exchange": [
            args.grid_slots, args.exchange_slots
        ],
        "exchange_before_temporal_compression": True,
    }
    sparse_indices = [
        index for index, spec in enumerate(specs) if is_sparse_variable(spec.name)
    ]
    report["sparse_occurrence_variables"] = [specs[index].name for index in sparse_indices]
    save_schema_report(report, args.output_dir / "schema_report.json")

    if args.resume_precomputed:
        raw_cache = np.load(raw_cache_path, mmap_mode="r")
        expected_shape = (
            len(files), report["grid_count"], report["flattened_feature_count"]
        )
        if raw_cache.shape != expected_shape:
            raise ValueError(
                f"reused raw cache shape {raw_cache.shape} != {expected_shape}"
            )
    else:
        raw_cache = build_normalized_cache(
            files, specs, mean, std, report["grid_count"], raw_cache_path
        )
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    codec = MultiTokenVariableCodec(
        [spec.width for spec in specs], token_dim=args.token_dim,
        grid_slots=args.grid_slots, sparse_indices=sparse_indices,
    ).to(device)
    if args.resume_precomputed:
        codec.load_state_dict(
            torch.load(codec_path, map_location="cpu", weights_only=True)
        )
        codec.eval()
        variable_training = {
            "reused_from_previous_attempt": True,
            "checkpoint": str(codec_path),
        }
        grid_cache = np.load(grid_cache_path, mmap_mode="r")
        expected_grid_shape = (
            len(files), report["grid_count"], codec.grid_dim
        )
        if grid_cache.shape != expected_grid_shape:
            raise ValueError(
                f"reused grid cache shape {grid_cache.shape} != "
                f"{expected_grid_shape}"
            )
        print(
            f"[resume] variable codec and grid codes reused: "
            f"{list(grid_cache.shape)}", flush=True,
        )
    else:
        variable_training = train_variable_codec(
            codec, raw_cache, training_cells, validation_cells,
            mean_t, std_t, args, device,
        )
        torch.save(cpu_state(codec), codec_path)
        grid_cache = encode_grid_cache(
            codec, raw_cache, grid_cache_path, device, args.decode_batch_size
        )
    print("[graph] building separate training/validation spherical graphs", flush=True)
    train_tensors = graph_tensors(
        lons, lats, training_cells, args.graph_neighbours, device,
    )
    val_tensors = graph_tensors(
        lons, lats, validation_cells, args.graph_neighbours, device,
    )
    model = InformationFirstAutoencoder(
        token_dim=args.token_dim, grid_slots=args.grid_slots,
        exchange_slots=args.exchange_slots,
        exchange_layers=args.exchange_layers, decoder_layers=args.decoder_layers,
        graph_chunk_size=args.graph_chunk_size,
        temporal_chunk_size=args.temporal_chunk_size,
        slot_chunk_size=args.slot_chunk_size,
    ).to(device)
    pre_joint_path = args.output_dir / "pre_joint_checkpoint.pt"
    if args.resume_full_model:
        resume_full_path = args.output_dir / "full_model.pt"
        if not resume_full_path.exists():
            raise FileNotFoundError(
                f"--resume-full-model requested but missing {resume_full_path}"
            )
        backup_path = args.output_dir / "full_model_before_continuation.pt"
        if not backup_path.exists():
            shutil.copy2(resume_full_path, backup_path)
        old_summary = args.output_dir / "summary.json"
        old_summary_backup = args.output_dir / "summary_before_continuation.json"
        if old_summary.exists() and not old_summary_backup.exists():
            shutil.copy2(old_summary, old_summary_backup)
        resumed = torch.load(
            resume_full_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(resumed["information_first"])
        codec.load_state_dict(resumed["variable_codec"])
        spatial_training = {"reused_from_full_model": True}
        temporal_training = {"reused_from_full_model": True}
        print(
            "[resume] completed full model restored; continuing joint stage only",
            flush=True,
        )
    elif args.resume_prejoint:
        if not pre_joint_path.exists():
            raise FileNotFoundError(
                f"--resume-prejoint requested but missing {pre_joint_path}"
            )
        pre_joint = torch.load(
            pre_joint_path, map_location="cpu", weights_only=False
        )
        model.load_state_dict(pre_joint["information_first"])
        codec.load_state_dict(pre_joint["variable_codec"])
        spatial_training = pre_joint["spatial_training"]
        temporal_training = pre_joint["temporal_training"]
        print("[resume] pre-joint model restored; starting joint stage", flush=True)
    else:
        spatial_training = train_spatial_stage(
            model, grid_cache, train_tensors, val_tensors, args, device,
        )
        temporal_training = train_temporal_stage(
            model, grid_cache, training_cells, validation_cells, args, device,
        )
        torch.save({
            "information_first": cpu_state(model),
            "variable_codec": cpu_state(codec),
            "spatial_training": spatial_training,
            "temporal_training": temporal_training,
            "arguments": vars(args),
        }, pre_joint_path)
        print(f"[checkpoint] saved before joint stage: {pre_joint_path}", flush=True)
    joint_training = train_joint_stage(
        model, codec, grid_cache, raw_cache, train_tensors, val_tensors,
        mean_t, std_t, args, device,
    )
    torch.save(cpu_state(model), args.output_dir / "information_first_best.pt")
    torch.save(cpu_state(codec), args.output_dir / "variable_codec_best.pt")

    all_cells = np.arange(report["grid_count"], dtype=np.int64)
    print("[graph] building full 15,002-cell spherical graph", flush=True)
    full_tensors = graph_tensors(
        lons, lats, all_cells, args.graph_neighbours, device,
    )
    cells_t, node_features_t, neighbours_t, edge_features_t = full_tensors
    full_codes = torch.as_tensor(np.asarray(grid_cache), device=device)
    model.eval()
    with torch.no_grad():
        with autocast_context(device):
            restored_grid, latent = model(
                full_codes[:, cells_t], node_features_t,
                neighbours_t, edge_features_t,
            )
        restored_grid = restored_grid.float()
    latent_path = args.output_dir / "latent_float32.nc"
    write_latent_netcdf(latent_path, latent, all_cells, lons, lats, files)

    full_model_path = args.output_dir / "full_model.pt"
    torch.save({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": report,
        "arguments": vars(args),
        "variable_codec": cpu_state(codec),
        "information_first": cpu_state(model),
        "sparse_indices": sparse_indices,
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
        "experiment": "all_variable_information_first_30day",
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
                "eight 128-dimensional tokens per grid",
                "learned slot bottleneck from eight to four tokens",
                "alternating spherical spatial and full-length temporal attention",
                "slot compression before context exchange; no grid reduction",
                "one kernel-3 stride-2 temporal bottleneck after context exchange",
                "temporal-spatial-variable-family reconstruction",
            ],
            "temporal_length": [len(files), (len(files) + 1) // 2],
            "validation_partition_patches": len(patches),
            "validation_partition_patch_width_max": args.patch_size,
            "stored_spatial_positions": int(report["grid_count"]),
            "spatial_compression": False,
            "graph_neighbours": args.graph_neighbours,
            "exchange_layers": args.exchange_layers,
            "grid_slots": args.grid_slots,
            "exchange_slots": args.exchange_slots,
            "token_dimension": args.token_dim,
            "latent_shape": list(latent.shape),
            "variable_codec_parameters": sum(p.numel() for p in codec.parameters()),
            "information_first_parameters": sum(p.numel() for p in model.parameters()),
        },
        "training": {
            "variable_codec": variable_training,
            "spatial_masked_pretraining": spatial_training,
            "temporal_pretraining": temporal_training,
            "joint_finetuning": joint_training,
            "joint_loss": (
                f"10% grid-token MSE + "
                f"{100.0 * (0.90 - args.sparse_loss_weight):g}% "
                f"equal-variable normalized original MSE + "
                f"{100.0 * args.sparse_loss_weight:g}% "
                "sparse-variable occurrence BCE"
            ),
            "staged_training": True,
            "continued_from_full_model": bool(args.resume_full_model),
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
            "note": (
                "The package includes the full encoder/decoder checkpoint. "
                "Static or non-modelled variables copied from each source NetCDF "
                "template are not included, so this is not yet a fully standalone "
                "all-variable archive ratio."
            ),
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
