"""CLEAR SE-only handover; the measured M1 model/loss/training policy is unchanged.

M1 is the historical implementation label for CLEAR with its SE variable gate.
This entry point never trains the SE-free M0 or any removal condition.
"""
from __future__ import annotations

import gc
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ablation_models import DEFAULT_SETTINGS, configs, loss_weights
from ablation_storage import exhaustive_metrics, fingerprint, package_size, read_json, size_probe
from ablation_training import StageObjective, global_statistics, train_stage
from common import pick_device, set_seed, write_json
from component_models import CHANGES, FIRST_CHANGED, component_codec
from data_utils import load_cube_layout, specs_from_report
from pipeline_utils import (baseline_hashes, implementation_hashes, parse_args, prepare,
                            prefix_decoder, save_restore_package, stage_latents, write_comparison)


def release(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def fixed_configs(condition, widths, settings, args):
    result = configs("VST", args.t_latent, widths, settings)
    if condition != "M0":
        changed = FIRST_CHANGED[condition]
        result[changed]["ablation_condition"] = condition
        if condition == "M6":
            result[changed]["ablation_neighbors"] = args.local_neighbors
            result[changed]["ablation_query_chunk"] = args.local_query_chunk
    return result


def run_condition(condition, args, root, shared, raw, layout, widths, settings, protocol):
    directory = root / condition
    directory.mkdir(exist_ok=True)
    summary_file = directory / "summary.json"
    if summary_file.exists():
        result = read_json(summary_file)
        if result["protocol_hash"] != protocol["signature"]:
            raise ValueError("completed result protocol mismatch; use a new run directory")
        print(f"[reuse complete {condition}]", flush=True)
        return result
    device = pick_device(args.device)
    weights = loss_weights(widths, device)
    stage_configs = fixed_configs(condition, widths, settings, args)
    stage_paths, reports = {}, []
    package = directory / "restore_package"
    if args.evaluate_only and not package.exists():
        raise FileNotFoundError(f"evaluate-only requires {package}")
    if not package.exists():
        source = raw
        v_decoder = s_decoder = None
        first = "VST".index(FIRST_CHANGED[condition])
        for number, stage in enumerate("VST"):
            # [BASELINE-FORMAT UPDATE] Existing stage entry files can stop at
            # their boundary; completed prefix training/cache is reused later.
            if number > "VST".index(args.stop_after):
                write_json(directory / "partial_stage_training.json", reports)
                return dict(condition=condition, partial=True, completed_through=args.stop_after,
                            stage_configs=stage_configs, training=reports)
            if number < first:
                stage_dir = root / "M0" / ("train_" + stage)
                report = read_json(stage_dir / "training.json")
                baseline_summary = read_json(root / "M0" / "summary.json")
                if baseline_summary["protocol_hash"] != protocol["signature"]:
                    raise ValueError("M0 prefix protocol mismatch")
                marker = read_json((stage_dir / "codes.npy").with_suffix(".json"))
                expected = fingerprint(dict(training=report["signature"],
                                            best=report["best_fixed_training_check_loss"]))
                if marker["signature"] != expected:
                    raise ValueError("M0 prefix latent fingerprint mismatch")
                source = np.load(stage_dir / "codes.npy", mmap_mode="r")
                if list(source.shape) != marker["shape"]:
                    raise ValueError("M0 prefix latent shape mismatch")
                stage_configs[stage] = read_json(root / "M0" / "model_plan.json")["stage_configs"][stage]
                print(f"[{condition}] frozen/reused M0 {stage}", flush=True)
                reports.append(dict(stage=stage, shared_prefix=True, shared_from="M0", **report))
            else:
                stage_dir = directory / ("train_" + stage)
                set_seed(args.seed + number)
                stats = global_statistics(source) if stage == "T" else None
                model = component_codec(stage, stage_configs[stage], condition, layout, stats,
                                        args.local_neighbors, args.local_query_chunk).to(device)
                if hasattr(model, "neighbor_report"):
                    write_json(directory / "local_neighbor_geometry.json", model.neighbor_report)
                print(f"[{condition}/{stage}] parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
                objective = StageObjective(stage, model, raw, source, weights, device, args,
                                           v_decoder, s_decoder)
                print(f"[{condition}] train {stage}: {CHANGES[condition]}", flush=True)
                stage_protocol = fingerprint(dict(protocol=protocol["signature"], condition=condition,
                                                  upstream=[(r["signature"], r["best_fixed_training_check_loss"])
                                                            for r in reports]))
                report = train_stage(stage_dir, stage, model, objective, args, stage_protocol)
                reports.append(dict(stage=stage, shared_prefix=False, **report))
                signature = fingerprint(dict(training=report["signature"],
                                            best=report["best_fixed_training_check_loss"]))
                source = stage_latents(stage_dir / "codes.npy", stage, model, source, device,
                                        args.t_batch if stage == "T" else args.decoder_chunk, signature)
                del objective, model
                release(device)
            stage_paths[stage] = stage_dir
            if stage == "V":
                v_decoder = prefix_decoder(stage, stage_dir, layout, device)
            elif stage == "S":
                s_decoder = prefix_decoder(stage, stage_dir, layout, device)
        write_json(directory / "stage_training.json", reports)
        write_json(directory / "model_plan.json", dict(stage_configs=stage_configs,
                                                       encoder_change=CHANGES[condition]))
        if list(source.shape) != protocol["latent_shape"]:
            raise AssertionError("variant changed the fixed latent shape")
        # Internal restore condition is VST, not the experimental label M0-M6.
        save_restore_package(package, shared, "VST", stage_configs, source, stage_paths, source.shape)
        del v_decoder, s_decoder
        release(device)
    else:
        reports = read_json(directory / "stage_training.json")
    size = package_size(package)
    difference = size["total_bytes"] / protocol["fixed_package_reference_bytes"] - 1
    if abs(difference) > args.budget_tolerance:
        raise ValueError(f"fixed decoder/latent storage check failed: {difference:+.2%}")
    started = time.perf_counter()
    metrics = exhaustive_metrics(package, raw, shared, device, args.decoder_chunk,
                                directory / "reconstruction_vram", args.vram_first_updates,
                                args.vram_every_updates)
    reference_bytes = (shared / "reference_float32.nc").stat().st_size
    fresh = [r for r in reports if not r["shared_prefix"]]
    result = dict(condition=condition, encoder_change=CHANGES[condition], SE=condition=="M1",
                  protocol_hash=protocol["signature"], terminal_dim=args.t_latent,
                  latent_shape=protocol["latent_shape"], package=size,
                  budget_target_bytes=protocol["fixed_package_reference_bytes"],
                  budget_relative_difference=difference, fixed_dimensions=True,
                  reference_bytes=reference_bytes, compression_ratio=reference_bytes / size["total_bytes"],
                  reduction_percent=100*(1-size["total_bytes"]/reference_bytes),
                  metrics=metrics, training=reports,
                  training_max_allocated_bytes=max((r["max_allocated_bytes"] or 0 for r in reports), default=0) or None,
                  training_max_reserved_bytes=max((r["max_reserved_bytes"] or 0 for r in reports), default=0) or None,
                  reconstruction_memory=metrics["vram_profile"]["peaks"],
                  reconstruction_seconds=time.perf_counter()-started,
                  total_stage_updates=sum(r["updates"] for r in reports),
                  total_training_seconds=sum(r["training_seconds"] for r in reports),
                  additional_updates=sum(r["updates"] for r in fresh),
                  additional_training_seconds=sum(r["training_seconds"] for r in fresh),
                  full_stage_parameter_count=sum(r["parameter_count"] for r in reports),
                  all_stages_plateaued=all(r["convergence_status"]=="plateau_under_declared_rule" for r in reports),
                  decoder_architecture="unchanged baseline; S decoder remains global attention",
                  generalization_claim=False)
    write_json(summary_file, result)
    print(f"[done {condition}] mean R2={metrics['mean_R2']} "
          f"package={size['total_bytes']/1e6:.3f} MB CR={result['compression_ratio']:.2f}", flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    # All three M1 stages start from their own initialization: no M0 is needed.
    args.conditions = ["M1"]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = {k: getattr(args, k) for k in DEFAULT_SETTINGS}
    set_seed(args.seed)
    shared, raw, source_hash = prepare(args, root)
    schema = read_json(shared / "schema.json")
    widths = [s.width for s in specs_from_report(schema)]
    layout = load_cube_layout(shared / "cube_layout.npz")
    tokens = 6*((int(layout["n"])+1+2*args.halo)//args.patch)**2
    if args.local_neighbors >= tokens:
        raise ValueError("local-neighbors must be smaller than the token count")
    shape = [(len(raw)+1)//2, tokens, args.t_latent]
    keys = ("v_epochs", "s_epochs", "t_epochs", "v_steps", "s_steps", "t_steps", "extension_factor",
            "patience", "relative_improvement", "seed", "v_batch", "s_targets", "t_batch", "vst_days",
            "decoder_chunk", "check_grids", "check_days", "vram_first_updates", "vram_every_updates")
    training = {k: getattr(args, k) for k in keys}
    training.update(compute_precision="cuda_bfloat16_amp" if args.device.startswith("cuda") else "cpu_float32",
                    isolated_condition_processes=not args.in_process)
    signature = fingerprint(dict(source=source_hash, settings=settings, t_latent=args.t_latent,
                                 conditions=args.conditions, training=training,
                                 local_neighbors=args.local_neighbors, local_query_chunk=args.local_query_chunk,
                                 tolerance=args.budget_tolerance, implementation=implementation_hashes()))
    manifest = root / "experiment_protocol.json"
    if manifest.exists():
        protocol = read_json(manifest)
        if protocol["signature"] != signature:
            raise ValueError("changed data/code/settings/conditions: choose a new --output-dir")
    else:
        # Serialize the true decoder and NC container, not a parameter-count estimate.
        fixed_bytes = size_probe(root, shared, "VST", configs("VST", args.t_latent, widths, settings), shape, layout)
        protocol = dict(signature=signature, conditions=args.conditions, changes=CHANGES,
                        settings=settings, latent_shape=shape, fixed_package_reference_bytes=fixed_bytes,
                        training_configuration=training, baseline_sources=baseline_hashes(),
                        implementation_sources=implementation_hashes(), raw_loss="equal-weight variable normalized MSE",
                        data_split="none: all dates/grids; archive reconstruction, NOT generalization",
                        temporal_statistics="global per-channel: matches stage-ablation reference, not original per-position mean map",
                        decoder_topology_fixed=True, parameter_counts_fixed=False,
                        model_name="CLEAR_SE", legacy_condition_label="M1",
                        local_attention_scope="not used: S encoder and decoder both use GLOBAL attention",
                        equal_updates=False, equal_wall_time=False, arguments=vars(args))
        write_json(manifest, protocol)
    print(f"[fixed shape] {shape}; decoder + latent NC package template={protocol['fixed_package_reference_bytes']/1e6:.3f} MB", flush=True)
    if args.plan_only:
        print("[plan only] preprocessing and actual-file sizing complete; no training", flush=True)
        return 0
    if not args.in_process and args.worker_condition is None:
        original = list(sys.argv[1:] if argv is None else argv)
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), *original,
                   "--conditions", *args.conditions]
        for condition in args.conditions:
            print(f"[isolated worker] {condition}; sequential GPU use", flush=True)
            subprocess.run(command + ["--worker-condition", condition], check=True)
        summaries = [read_json(root/c/"summary.json") for c in args.conditions]
    else:
        selected = [args.worker_condition] if args.worker_condition else args.conditions
        summaries = [run_condition(c, args, root, shared, raw, layout, widths, settings, protocol)
                     for c in selected]
    if args.worker_condition is None:
        if all(not result.get("partial", False) for result in summaries):
            write_comparison(root, protocol, summaries)
        else:
            print(f"[stage boundary] completed through {args.stop_after}; no final VST metrics yet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
