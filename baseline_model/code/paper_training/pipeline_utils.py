"""Shared preprocessing, checkpoint export, CLI and result aggregation.

Model/data files keep their original code/ locations. SE and temporal-normalization
changes are marked explicitly in step1_model.py and step3_model.py.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from component_models import CONDITIONS
from ablation_models import DEFAULT_SETTINGS, load_decoder
from ablation_storage import (atomic_save, csv_write, fingerprint,
                             nc_create, package_metadata,
                             read_json, write_reference)
from ablation_training import encode_archive
from common import write_json
from data_utils import (build_cache_one_pass, build_cube_layout, discover_daily_files,
                        inspect_schema, load_cube_layout, save_cube_layout, specs_from_report)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CLEAR SE model: variable/vertical -> global spatial ViT -> temporal TCN")
    p.add_argument("--data-dir", type=Path, default=Path("/mnt/mybook/KCM_output"))
    p.add_argument("--pattern", default="UP-*.nc")
    p.add_argument("--start-date", default="20000101")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--output-dir", type=Path, default=Path("runs/clear_se_30day"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-workers", type=int, default=4)
    p.add_argument("--reuse-preprocessing", type=Path,
                   help="read-only reuse of compatible completed shared_preprocessing; source/code fingerprint checked")
    p.add_argument("--budget-tolerance", type=float, default=.01)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--evaluate-only", action="store_true")
    p.add_argument("--stop-after", choices=("V", "S", "T"), default="T",
                   help="Stage-file boundary only; model and training settings stay the same")
    p.add_argument("--in-process", action="store_true", help="opt out of fresh-process isolation (default is isolated)")
    p.add_argument("--worker-condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    p.add_argument("--vram-first-updates", type=int, default=3)
    p.add_argument("--vram-every-updates", type=int, default=200)
    for stage, epochs, steps in (("v", 150, 200), ("s", 300, 40), ("t", 300, 40)):
        p.add_argument(f"--{stage}-epochs", type=int, default=epochs)
        p.add_argument(f"--{stage}-steps", type=int, default=steps)
    p.add_argument("--extension-factor", type=float, default=2,
                   help="if not plateaued, allow this multiple of the initial epoch budget")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--relative-improvement", type=float, default=1e-4)
    p.add_argument("--v-batch", type=int, default=4096)
    p.add_argument("--s-targets", type=int, default=2048)
    p.add_argument("--t-batch", type=int, default=256)
    p.add_argument("--vst-days", type=int, default=1)
    p.add_argument("--decoder-chunk", type=int, default=256)
    p.add_argument("--check-grids", type=int, default=64)
    p.add_argument("--check-days", type=int, default=8)
    p.add_argument("--local-neighbors", type=int, default=8, help=argparse.SUPPRESS)
    p.add_argument("--local-query-chunk", type=int, default=128, help=argparse.SUPPRESS)
    p.add_argument("--t-latent", type=int, default=480)
    # Exposed for capacity experiments / local tiny verification, not auto equalized.
    for key, value in DEFAULT_SETTINGS.items():
        p.add_argument("--" + key.replace("_", "-"), type=int, default=value)
    args = p.parse_args(argv)
    positive = ["days", "cache_workers", "v_epochs", "s_epochs", "t_epochs", "v_steps", "s_steps",
                "t_steps", "patience", "v_batch", "s_targets", "t_batch", "vst_days",
                "decoder_chunk", "check_grids", "check_days", "vram_first_updates", "vram_every_updates", "local_neighbors", "local_query_chunk", "t_latent"] + list(DEFAULT_SETTINGS)
    if any(getattr(args, k) < 1 for k in positive) or args.extension_factor < 1:
        p.error("sizes/budgets must be positive; extension-factor >= 1")
    if not 0 < args.budget_tolerance < 1 or not 0 < args.relative_improvement < 1:
        p.error("tolerances must be between 0 and 1")
    args.conditions = list(dict.fromkeys(args.conditions))
    if args.worker_condition and args.worker_condition not in args.conditions:
        p.error("worker-condition is not in the selected conditions")
    return args


def baseline_hashes():
    here = Path(__file__).resolve().parent.parent  # original code/ model files
    return {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
            ("common.py", "data_utils.py", "step1_model.py", "step2_model.py", "step3_model.py")}


def implementation_hashes():
    here = Path(__file__).parent
    result = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in
            ("ablation_models.py", "ablation_training.py", "ablation_storage.py",
             "train_clear_se.py", "pipeline_utils.py", "component_models.py", "vram_monitor.py")}
    for name in ("paper_entry.py", "step1_horizontal.py", "step2_vertical.py", "step3_temporal.py"):
        result["../" + name] = hashlib.sha256((here.parent / name).read_bytes()).hexdigest()
    return result


def prepare(args, root):
    files = discover_daily_files(args.data_dir, args.pattern, args.start_date, args.days)
    source_identity = [dict(path=str(p.resolve()), size=p.stat().st_size,
                            mtime_ns=p.stat().st_mtime_ns) for p in files]
    identity = fingerprint(dict(sources=source_identity, dtype="float32", missing="mean",
                                baseline_sources=baseline_hashes()))
    shared = args.reuse_preprocessing.resolve() if args.reuse_preprocessing else root / "shared_preprocessing"
    if (shared / "preprocessing.json").exists():
        prep = read_json(shared / "preprocessing.json")
        if prep["source_fingerprint"] != identity:
            raise ValueError("shared cache source/preprocessing fingerprint mismatch; use a new run folder")
        required = ("normalized_float32.npy", "normalization.npz", "schema.json", "coordinates.npz",
                    "cube_layout.npz", "reference_float32.nc")
        if not all((shared / name).is_file() for name in required):
            raise FileNotFoundError("incomplete shared preprocessing; do not reuse partial files")
        cache = np.load(shared / "normalized_float32.npy", mmap_mode="r")
        if list(cache.shape) != [prep["days"], prep["grids"], prep["features"]] or cache.dtype != np.float32:
            raise ValueError("shared cache shape/dtype mismatch")
        print(f"[shared preprocessing reuse] {shared}", flush=True)
        return shared, cache, identity
    if args.reuse_preprocessing:
        raise FileNotFoundError("reuse-preprocessing requires a completed preprocessing.json")
    shared.mkdir(parents=True, exist_ok=True)
    if any(shared.iterdir()):
        raise FileExistsError("incomplete preprocessing folder: choose a new --output-dir; source files are untouched")
    schema, lons, lats = inspect_schema(files[0])
    specs = specs_from_report(schema)
    # Confirm every day's modeled variables and coordinates, before reading/normalizing.
    for path in files[1:]:
        other, other_lon, other_lat = inspect_schema(path)
        if other["modelled_variables"] != schema["modelled_variables"]:
            raise ValueError(f"schema changes on {path.name}")
        if not np.allclose(lons, other_lon) or not np.allclose(lats, other_lat):
            raise ValueError(f"grid coordinates/order change on {path.name}")
    write_json(shared / "schema.json", schema)
    np.savez(shared / "coordinates.npz", longitude=lons, latitude=lats)
    save_cube_layout(build_cube_layout(lons, lats, halo=max(4, args.halo)), shared / "cube_layout.npz")
    print(f"[schema] days={len(files)} variables={len(specs)} "
          f"features={sum(s.width for s in specs)} grids={len(lons)}", flush=True)
    mean, std, missing = build_cache_one_pass(files, specs, len(lons), shared / "normalized_float32.npy",
                                              training_cells=None, workers=args.cache_workers)
    np.savez(shared / "normalization.npz", mean=mean, std=std)
    cache = np.load(shared / "normalized_float32.npy", mmap_mode="r")
    prep = dict(days=len(files), grids=len(lons), features=len(mean), source_fingerprint=identity,
                sources=source_identity, missing_values=missing, baseline_sources=baseline_hashes(),
                holdout=False, all_dates_and_grids_used=True)
    # Size planner reads this marker only after prepare() completes successfully.
    reference_bytes = write_reference(cache, shared, files, specs)
    prep["reference_bytes"] = reference_bytes
    write_json(shared / "preprocessing.json", prep)
    return shared, cache, identity


def stage_latents(path, stage, model, source, device, chunk, signature):
    marker = path.with_suffix(".json")
    if marker.exists() and path.exists():
        info = read_json(marker)
        if info["signature"] != signature:
            raise ValueError(f"intermediate latent fingerprint mismatch: {path}")
        values = np.load(path, mmap_mode="r")
        if list(values.shape) != info["shape"]:
            raise ValueError("intermediate latent shape mismatch")
        return values
    if path.exists():
        raise FileExistsError(f"unmarked intermediate latent: {path}; choose a new run folder")
    values = encode_archive(stage, model, source, path, device, chunk)
    write_json(marker, dict(signature=signature, shape=list(values.shape), cache_only=True))
    return values


def save_restore_package(directory, shared, condition, configs, latent, stage_paths, shape):
    if directory.exists():
        raise FileExistsError(directory)
    pending = directory.with_name(directory.name + "_pending")
    # Only our own explicit pending output can be reused after an interrupted write.
    pending.mkdir(parents=True, exist_ok=True)
    package_metadata(pending, shared, condition, configs, shape)
    for stage, stage_dir in stage_paths.items():
        saved = torch.load(stage_dir / "best.pt", map_location="cpu", weights_only=False)
        state = {k: v.float().cpu() for k, v in saved["state_dict"].items()
                 if k.startswith("dec") or (stage == "V" and k == "variable_embedding")
                 or (stage == "T" and k in ("mean", "scale"))}
        atomic_save(dict(config=saved["config"], state_dict=state), pending / f"decoder_{stage}.pt")
    ds, arr = nc_create(pending / "latent.nc", shape)
    for day in range(shape[0]):
        arr[day] = latent[day]
        ds.sync()
    ds.close()
    pending.replace(directory)


def prefix_decoder(stage, path, layout, device):
    saved = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    keys = {k: v for k, v in saved["state_dict"].items()
            if k.startswith("dec") or (stage == "V" and k == "variable_embedding")
            or (stage == "T" and k in ("mean", "scale"))}
    return load_decoder(stage, dict(config=saved["config"], state_dict=keys), layout, device)


def write_comparison(root, protocol, summaries):
    write_json(root / "comparison.json", dict(protocol=protocol, results=summaries))
    rows, variables, memory_rows = [], [], []
    for s in summaries:
        train_only = max((r.get("training_update_peak", {}).get("max_allocated_bytes") or 0
                          for r in s["training"]), default=0)
        rows.append(dict(condition=s["condition"], terminal_dim=s["terminal_dim"],
                         package_MB=s["package"]["total_bytes"] / 1e6,
                         fixed_package_difference_percent=s["budget_relative_difference"] * 100,
                         compression_ratio=s["compression_ratio"], reduction_percent=s["reduction_percent"],
                         mean_R2_percent=s["metrics"]["mean_R2"] * 100 if s["metrics"]["mean_R2"] is not None else None,
                         median_R2_percent=s["metrics"]["median_R2"] * 100 if s["metrics"]["median_R2"] is not None else None,
                         normalized_MSE=s["metrics"]["variable_balanced_normalized_MSE"],
                         full_stage_parameters=s["full_stage_parameter_count"], updates=s["total_stage_updates"],
                         training_hours=s["total_training_seconds"] / 3600,
                         additional_training_hours=s["additional_training_seconds"] / 3600,
                         additional_updates=s["additional_updates"],
                         encoder_change=s["encoder_change"],
                         train_peak_allocated_GiB=(s["training_max_allocated_bytes"] or 0) / 2**30 or None,
                         train_only_peak_allocated_GiB=train_only / 2**30 or None,
                         restore_peak_allocated_GiB=(s["reconstruction_memory"]["max_allocated_bytes"] or 0) / 2**30 or None,
                         convergence_confirmed=s["all_stages_plateaued"],
                         isolated_processes=not protocol["arguments"]["in_process"],
                         exhaustive_values_per_second=s["metrics"].get("exhaustive_values_per_second")))
        for report in s["training"]:
            profile = report.get("vram_profile", {})
            for phase, stats in profile.get("phases", {}).items():
                memory_rows.append(dict(condition=s["condition"], role="training", stage=report["stage"],
                                        shared_prefix=report.get("shared_prefix", False), phase=phase,
                                        pid=profile["metadata"]["pid"], **stats))
        profile = s["metrics"].get("vram_profile", {})
        for phase, stats in profile.get("phases", {}).items():
            memory_rows.append(dict(condition=s["condition"], role="reconstruction", stage="reverse_pipeline",
                                    shared_prefix=False, phase=phase, pid=profile["metadata"]["pid"], **stats))
        for name, metric in s["metrics"]["variables"].items():
            variables.append(dict(condition=s["condition"], variable=name, **metric))
    csv_write(root / "comparison.csv", rows)
    csv_write(root / "variables.csv", variables)
    csv_write(root / "vram_comparison.csv", memory_rows)
    write_json(root / "vram_comparison.json", dict(
        measurement="PyTorch allocator peaks; tensor storage inventory; phase-boundary snapshots",
        phase_rows=memory_rows,
        notes=["phase peaks are not additive; nested phase timings must not be summed",
               "CPU correctness tests have null GPU bytes, not zero-byte GPU efficiency",
               "shared M0 prefix cost is logically included; additional physical training is reported separately",
               "native batches differ: this is implementation memory, not an equal-batch architecture ranking"],
        condition_inventories={s["condition"]: dict(
            training={r["stage"]: r.get("vram_profile", {}).get("inventory_maxima", {}) for r in s["training"]},
            reconstruction=s["metrics"].get("vram_profile", {}).get("inventory_maxima", {}),
            host_intermediate_bytes=s["metrics"].get("vram_profile", {}).get("host_intermediate_peak_logical_bytes"))
                              for s in summaries}))
    print(f"[comparison] {root / 'comparison.json'}; {root / 'vram_comparison.csv'}", flush=True)


