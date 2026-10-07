#!/usr/bin/env python3
"""
Step 1: 격자 셀별 1D 압축 (변수·연직 축).

각 격자 셀의 하루치 벡터(181개 변수, 6,035개 값)를 `Step1Codec`으로 임베딩
하나(기본 256차원)로 압축하도록 학습하고, 모든 날짜·모든 셀의 임베딩을
파일로 저장합니다. 공간·시간 정보는 쓰지 않습니다.

출력 (<output-dir>):
    schema_report.json          변수 스키마 (step2, 디코더가 사용)
    patch_layout.npz            구면 패치와 학습/검증 패치 분할 (step2가 그대로 사용)
    normalization.npz           학습 셀 기준 변수별 평균/표준편차
    normalized_float32.npy      정규화 캐시 [day, grid, 6035] (재실행 시 재사용)
    step1_codec.pt              코덱 체크포인트 (config + state_dict)
    step1_embeddings_float32.npy  임베딩 [day, grid, embed_dim]  ← step2 입력
    summary.json                학습 기록, 변수별 R², 압축률

목적은 보관용 압축이므로 기본은 모든 셀로 학습합니다(검증 분할 없음). 학습
진행은 전체 셀에서 뽑은 고정 표본의 복원 손실로 확인하고, 그 값이 --patience
epoch 동안 나아지지 않으면 멈춥니다. --val-patch-fraction > 0 이면 예전처럼
공간 검증 패치를 떼어 둡니다.

    python step1_horizontal.py --schema-only
    python step1_horizontal.py --days 30 --output-dir runs/step1 --device cuda:0
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

from common import (
    autocast_context, cpu_state, load_normalization, pick_device, set_seed, write_json,
)
from data_utils import (
    SEED, build_cache_one_pass, build_patch_layout, cells_of,
    holdout_cells,
    discover_daily_files, inspect_schema, load_patch_layout, save_patch_layout,
    save_schema_report, specs_from_report,
)
from step1_decoder import CACHE_FILE, CODEC_FILE, EMBEDDINGS_FILE, evaluate
from step1_model import Step1Codec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("/home/jung/climate/data"))
    p.add_argument("--pattern", default="UP-*.nc")
    p.add_argument("--start-date", default="20000101")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=Path("runs/step1"))
    p.add_argument("--patch-size", type=int, default=15,
                   help="spatial patch size; also defines the validation split")
    p.add_argument("--val-patch-fraction", type=float, default=0.0,
                   help="0 = train on every cell (archival compression)")
    # model
    p.add_argument("--token-dim", type=int, default=96)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--variable-slots", type=int, default=4)
    # training (old experiments: 40 epochs was undertrained, 300 still improving)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1.0e-3)
    p.add_argument("--weight-decay", type=float, default=3.0e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--check-samples", type=int, default=8192,
                   help="fixed sample used to track reconstruction loss each epoch")
    p.add_argument("--encode-batch-size", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="constant",
                   help="cosine: linear warm-up then cosine decay to 5%% over all epochs")
    p.add_argument("--warmup-steps", type=int, default=500)
    # control
    p.add_argument("--schema-only", action="store_true", help="read one file header and stop")
    p.add_argument("--cache-workers", type=int, default=8, help="parallel NetCDF readers")
    p.add_argument("--no-cache-in-ram", action="store_true",
                   help="sample from the on-disk memmap instead of loading the cache into RAM")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="recompute normalization/cache even if a matching one exists")
    p.add_argument("--skip-train", action="store_true",
                   help=f"reuse <output-dir>/{CODEC_FILE}; only encode and evaluate")
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- data

def prepare_cache(args, files, specs, report, layout, training_cells):
    """Normalization + normalized cache, reused when inputs are unchanged."""
    out = args.output_dir
    meta_path = out / "cache_meta.json"
    meta = {
        "files": [f.name for f in files],
        "patch_size": args.patch_size,
        "val_patch_fraction": args.val_patch_fraction,
        "features": report["flattened_feature_count"],
    }
    cache_path, norm_path = out / CACHE_FILE, out / "normalization.npz"
    if (not args.rebuild_cache and meta_path.exists() and cache_path.exists()
            and norm_path.exists() and json.loads(meta_path.read_text()) == meta):
        print("[cache] reusing existing normalization and normalized cache", flush=True)
        mean, std = load_normalization(norm_path)
        return mean, std, np.load(cache_path, mmap_mode="r")

    cells = None if len(training_cells) == report["grid_count"] else training_cells
    mean, std, missing = build_cache_one_pass(files, specs, report["grid_count"], cache_path,
                                              cells, args.cache_workers)
    np.savez(norm_path, mean=mean, std=std,
             variable_names=np.asarray([s.name for s in specs]),
             offsets=np.asarray([s.offset for s in specs] + [specs[-1].stop]))
    report["missing_training_values"] = missing
    write_json(meta_path, meta)
    return mean, std, np.load(cache_path, mmap_mode="r")


# ---------------------------------------------------------------- training

def sample_batch(cache, cells, count, rng, device):
    days = rng.randint(0, cache.shape[0], size=count)
    chosen = rng.choice(cells, size=count, replace=True)
    order = np.lexsort((chosen, days))  # sorted access is kinder to the memmap
    values = np.asarray(cache[days[order], chosen[order]], dtype=np.float32)
    return torch.as_tensor(values, device=device)


def check_loss(codec, cache, cells, args, device):
    """Reconstruction loss on a fixed sample (eval mode), used for stopping."""
    rng = np.random.RandomState(SEED + 41)  # same sample every epoch
    total, seen = 0.0, 0
    codec.eval()
    with torch.no_grad():
        for start in range(0, args.check_samples, args.encode_batch_size):
            count = min(args.encode_batch_size, args.check_samples - start)
            target = sample_batch(cache, cells, count, rng, device)
            with autocast_context(device):
                prediction, _ = codec(target)
                loss = codec.variable_balanced_mse(prediction, target)
            total += float(loss) * count
            seen += count
    return total / max(seen, 1)


def train_codec(codec, cache, training_cells, check_cells, args, device):
    optimizer = torch.optim.AdamW(codec.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    if args.lr_schedule == "cosine":
        def factor(step):
            warm = min(1.0, (step + 1) / max(args.warmup_steps, 1))
            progress = min(step, total_steps) / max(total_steps, 1)
            return warm * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    else:
        scheduler = None
    # Single GPU on purpose: torch.nn.DataParallel trained this model far worse
    # (loss 0.36 vs 0.175 after 200 identical steps on the 30-day cache).
    trainer = codec
    rng = np.random.RandomState(SEED)
    best, best_epoch, bad, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, args.epochs + 1):
        codec.train()
        total = 0.0
        for _ in range(args.steps_per_epoch):
            target = sample_batch(cache, training_cells, args.batch_size, rng, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                prediction, _ = trainer(target)
                loss = codec.variable_balanced_mse(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(codec.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach())
        train_loss = total / args.steps_per_epoch
        check = check_loss(codec, cache, check_cells, args, device)
        history.append({"epoch": epoch, "train": train_loss, "check": check})
        print(f"[step1 epoch {epoch:03d}] train={train_loss:.6f} check={check:.6f} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}", flush=True)
        if check < best - 1.0e-5:
            best, best_epoch, bad, best_state = check, epoch, 0, cpu_state(codec)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[step1 early-stop] best epoch={best_epoch} check={best:.6f}", flush=True)
                break
    if best_state is not None:
        codec.load_state_dict(best_state)
    codec.eval()
    return {"best_epoch": best_epoch, "best_check_loss": best,
            "epochs_run": len(history), "history": history}


def encode_all(codec, cache, output, device, batch_size):
    """Write embeddings [day, grid, embed_dim] as a float32 .npy memmap."""
    embeddings = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(cache.shape[0], cache.shape[1], codec.embed_dim),
    )
    codec.eval()
    with torch.no_grad():
        for day in range(cache.shape[0]):
            print(f"[step1-encode day {day + 1}/{cache.shape[0]}]", flush=True)
            for start in range(0, cache.shape[1], batch_size):
                stop = min(cache.shape[1], start + batch_size)
                values = torch.as_tensor(np.array(cache[day, start:stop], dtype=np.float32),
                                         device=device)
                with autocast_context(device):
                    embeddings[day, start:stop] = codec.encode(values).float().cpu().numpy()
        embeddings.flush()
    del embeddings
    return np.load(output, mmap_mode="r")


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    set_seed()
    device = pick_device(args.device)
    started = time.perf_counter()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    files = discover_daily_files(args.data_dir, args.pattern, args.start_date, args.days)
    report, lons, lats = inspect_schema(files[0])
    report["source_files"] = [f.name for f in files]
    specs = specs_from_report(report)
    save_schema_report(report, out / "schema_report.json")
    print(f"[schema] variables={len(specs)} features={report['flattened_feature_count']} "
          f"grids={report['grid_count']} days={len(files)}", flush=True)
    if args.schema_only:
        return

    layout = build_patch_layout(lons, lats, args.patch_size, args.val_patch_fraction)
    save_patch_layout(layout, out / "patch_layout.npz")
    training_cells = cells_of(layout, layout["train_patch_ids"])
    validation_cells = holdout_cells(layout)
    check_cells = training_cells if validation_cells is None else validation_cells
    print(f"[split] {layout['summary']} train_cells={len(training_cells)} "
          f"holdout_cells={0 if validation_cells is None else len(validation_cells)}", flush=True)

    mean, std, cache = prepare_cache(args, files, specs, report, layout, training_cells)
    if not args.no_cache_in_ram:
        started_load = time.perf_counter()
        cache = np.load(args.output_dir / CACHE_FILE)          # ~0.36 GB per day
        print(f"[cache] loaded into RAM in {time.perf_counter() - started_load:.0f}s", flush=True)
    save_schema_report(report, out / "schema_report.json")

    codec = Step1Codec([s.width for s in specs], token_dim=args.token_dim,
                       embed_dim=args.embed_dim, variable_slots=args.variable_slots).to(device)
    codec_path = out / CODEC_FILE
    if args.skip_train:
        checkpoint = torch.load(codec_path, map_location="cpu", weights_only=False)
        codec = Step1Codec.from_checkpoint(checkpoint).to(device)
        training = {"skipped": True, "loaded_from": str(codec_path)}
    else:
        training = train_codec(codec, cache, training_cells, check_cells, args, device)
        torch.save({
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": codec.config, "state_dict": cpu_state(codec),
            "schema": report, "arguments": vars(args), "training": training,
        }, codec_path)

    embeddings_path = out / EMBEDDINGS_FILE
    embeddings = encode_all(codec, cache, embeddings_path, device, args.encode_batch_size)

    metrics = None
    if not args.skip_eval:
        metrics = evaluate(codec, embeddings, cache, specs, mean, std, device,
                           args.encode_batch_size, validation_cells, source_files=files)
        write_json(out / "step1_embeddings_float32_metrics.json", metrics)

    original_bytes = sum(f.stat().st_size for f in files)
    embedding_bytes = embeddings_path.stat().st_size
    summary = {
        "step": "step1_horizontal (per-cell 1D codec)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": {"days": len(files), "first": files[0].name, "last": files[-1].name},
        "split": layout["summary"],
        "model": {**codec.config, "widths": f"{len(specs)} variables",
                  "parameters": sum(p.numel() for p in codec.parameters())},
        "training": {k: v for k, v in training.items() if k != "history"},
        "embeddings": {"path": str(embeddings_path), "shape": list(embeddings.shape)},
        "compression": {
            "values_per_cell": f"{codec.total_features} -> {codec.embed_dim}",
            "value_ratio": codec.total_features / codec.embed_dim,
            "original_netcdf_bytes": original_bytes,
            "embedding_file_bytes": embedding_bytes,
            "file_ratio_excluding_model": original_bytes / embedding_bytes,
        },
        "reconstruction": None if metrics is None else {
            scope: {k: v for k, v in m.items() if k != "variables"}
            for scope, m in metrics.items() if scope != "per_day"
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)
    print(f"[done] {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
