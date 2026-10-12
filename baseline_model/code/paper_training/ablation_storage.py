"""Actual float32 NetCDF / decoder package sizing and exhaustive restoration."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from contextlib import contextmanager, nullcontext

from netCDF4 import Dataset
import numpy as np
import torch

from ablation_models import configs, make_decoder, decoder_bytes, load_decoder
from common import VariableMetrics, autocast_context, write_json
from data_utils import load_cube_layout, specs_from_report, read_grid_matrix
from vram_monitor import VramRecorder


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    torch.save(value, temporary)
    temporary.replace(path)


def nc_create(path, shape, variable="latent"):
    ds = Dataset(path, "w", format="NETCDF4")
    ds.set_fill_off()
    for name, size in zip(("time", "position", "channel"), shape):
        ds.createDimension(name, int(size))
    arr = ds.createVariable(variable, "f4", ("time", "position", "channel"),
                            contiguous=True, zlib=False, fill_value=False)
    ds.storage_dtype = "float32"
    ds.lossless_compression = "none"
    return ds, arr


def latent_shape(condition, d, days, grids, tokens):
    return ((days + 1) // 2 if "T" in condition else days,
            tokens if "S" in condition else grids, d)


def package_metadata(destination, shared, condition, stage_configs, shape):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("schema.json", "normalization.npz", "coordinates.npz"):
        shutil.copy2(shared / name, destination / name)
    if "S" in condition:
        shutil.copy2(shared / "cube_layout.npz", destination / "cube_layout.npz")
    preprocessing = read_json(shared / "preprocessing.json")
    write_json(destination / "restore.json", dict(
        protocol=1, condition=condition, stages=stage_configs, latent_shape=list(shape),
        original_days=preprocessing["days"],
        source_date_labels=[Path(s["path"]).name for s in preprocessing.get("sources", [])],
        dtype="float32", file_compression="none", temporal_statistics="global_per_channel",
        missing_values="training-feature-mean imputation; same as the supplied baseline",
        source_required_for_decoding=False))


def package_size(directory):
    files = {p.name: p.stat().st_size for p in sorted(directory.iterdir()) if p.is_file()}
    return dict(files=files, total_bytes=sum(files.values()))


def size_probe(root, shared, condition, stage_configs, shape, layout):
    """Real serialization, no dummy padding. Probe latent is a size-only sparse file.

    Its contents are never evaluated. A new private temporary folder is automatically
    cleaned up; no input/archive/previous-run paths are removed.
    """
    with tempfile.TemporaryDirectory(prefix="size_probe_", dir=root) as temporary:
        directory = Path(temporary)
        package_metadata(directory, shared, condition, stage_configs, shape)
        for stage, config in stage_configs.items():
            decoder = make_decoder(stage, config, layout)
            torch.save(dict(config=config, state_dict=decoder.state_dict()), directory / f"decoder_{stage}.pt")
            del decoder
        ds, arr = nc_create(directory / "latent.nc", shape)
        arr[tuple(s - 1 for s in shape)] = np.float32(0)
        ds.close()
        return package_size(directory)["total_bytes"]


def plan_budget(root, shared, widths, settings, conditions, target_mb=None, tolerance=.01):
    prep = read_json(shared / "preprocessing.json")
    layout = load_cube_layout(shared / "cube_layout.npz")
    days, grids = prep["days"], prep["grids"]
    tokens = 6 * ((int(layout["n"]) + 1 + 2 * settings["halo"]) // settings["patch"]) ** 2
    refconfigs = configs("VST", settings["t_hidden"], widths, settings)
    refshape = latent_shape("VST", settings["t_hidden"], days, grids, tokens)
    reference = size_probe(root, shared, "VST", refconfigs, refshape, layout)
    target = round(target_mb * 1e6) if target_mb else reference
    result = dict(target_bytes=target, relative_tolerance=tolerance, reference="VST template",
                  reference_template_bytes=reference, conditions={})
    bounds = {"V": max(512, settings["v_dim"]), "S": max(512, settings["s_width"]),
              "VS": max(512, settings["s_width"]), "T": settings["t_hidden"],
              "VT": settings["t_hidden"], "VST": settings["t_hidden"]}
    for condition in conditions:
        def estimated(d, overhead):
            cs = configs(condition, d, widths, settings)
            shape = latent_shape(condition, d, days, grids, tokens)
            return int(np.prod(shape)) * 4 + decoder_bytes(cs, layout) + overhead
        overhead = 100_000
        d = 1
        actual = None
        # Two actual measurements calibrate condition-specific NC/checkpoint headers.
        for _ in range(3):
            lo, hi = 1, bounds[condition]
            while lo < hi:
                mid = (lo + hi) // 2
                if estimated(mid, overhead) < target:
                    lo = mid + 1
                else:
                    hi = mid
            candidates = {max(1, lo - 1), lo}
            d = min(candidates, key=lambda k: abs(estimated(k, overhead) - target))
            cs = configs(condition, d, widths, settings)
            shape = latent_shape(condition, d, days, grids, tokens)
            actual = size_probe(root, shared, condition, cs, shape, layout)
            overhead = actual - int(np.prod(shape)) * 4 - decoder_bytes(cs, layout)
            if abs(actual / target - 1) <= tolerance:
                break
        difference = actual / target - 1
        if abs(difference) > tolerance:
            raise ValueError(f"{condition}: storage budget cannot be matched within {tolerance:.1%}; "
                             f"nearest d={d}, package={actual:,}, target={target:,}. "
                             "Choose --budget-mb / adjust model capacity; no padding is added.")
        result["conditions"][condition] = dict(terminal_dim=d, configs=cs, latent_shape=list(shape),
                                                probe_bytes=actual, relative_difference=difference)
        print(f"[budget {condition}] d={d} actual template={actual / 1e6:.3f} MB "
              f"difference={difference:+.2%}", flush=True)
    return result


def write_reference(cache, shared, files, specs, chunk=512):
    """Lossless flattened packing of the evaluated modeled values, float32, no zlib."""
    path = shared / "reference_float32.nc"
    if path.exists():
        raise FileExistsError(path)
    norm = np.load(shared / "normalization.npz", allow_pickle=False)
    ds, arr = nc_create(path, cache.shape, "values")
    ds.scope = "modeled dynamic floating variables only; flattened losslessly"
    ds.missing_values = "feature mean imputed (same values evaluated in all conditions)"
    for day, source in enumerate(files):
        # Read source values directly, rather than inverse-normalizing the cache:
        # the float32 reference thus does not accumulate normalization roundoff.
        values = np.empty(cache.shape[1:], np.float32)
        with Dataset(source, "r") as original:
            for spec in specs:
                block = read_grid_matrix(original, spec)
                mask = ~np.isfinite(block)
                if mask.any():
                    block = np.where(mask, norm["mean"][spec.offset:spec.stop], block)
                values[:, spec.offset:spec.stop] = block
        for start in range(0, cache.shape[1], chunk):
            stop = min(start + chunk, cache.shape[1])
            arr[day, start:stop] = values[start:stop]
        ds.sync()
        print(f"[reference {day + 1}/{len(cache)}]", flush=True)
    ds.close()
    return path.stat().st_size


def load_package(directory, device):
    info = read_json(directory / "restore.json")
    layout = load_cube_layout(directory / "cube_layout.npz") if "S" in info["condition"] else None
    models = {}
    for stage in info["stages"]:
        saved = torch.load(directory / f"decoder_{stage}.pt", map_location="cpu", weights_only=False)
        models[stage] = load_decoder(stage, saved, layout, device)
    return info, models


@contextmanager
def restore_phase(profiler, name, context=None):
    if profiler is None:
        yield
        return
    calls = profiler.aggregates.get(name, {}).get("count", 0) + 1
    with profiler.section(name, dict(context or {}, phase_call=calls), detailed=profiler.selected(calls)):
        yield


def restore_batches(directory, device, chunk=256, profiler=None):
    """Decode using ONLY package files, yield (day, grid_start, normalized values).

    T is independent per spatial position. S is global: decode all its tokens
    before slicing grid outputs. No truthful upstream latent is supplied as context.
    """
    with restore_phase(profiler, "restore_load_decoders"):
        info, models = load_package(directory, device)
        if profiler is not None:
            profiler.models = {"decoder_" + s: model for s, model in models.items()}
    days = info["original_days"]
    condition = info["condition"]
    with torch.inference_mode(), autocast_context(device), Dataset(directory / "latent.nc") as nc:
        latent = nc.variables["latent"]
        if "S" not in condition:
            positions = latent.shape[1]
            for start in range(0, positions, chunk):
                stop = min(start + chunk, positions)
                with restore_phase(profiler, "restore_read_transfer", dict(position_start=start)):
                    values = torch.as_tensor(np.asarray(latent[:, start:stop]), device=device)
                    if profiler is not None:
                        profiler.tensors(latent_input=values)
                if "T" in condition:
                    with restore_phase(profiler, "restore_T_decode", dict(position_start=start)):
                        values = models["T"].denormalize(models["T"].decode(values.permute(1, 2, 0)))
                        values = values[..., :days].permute(2, 0, 1)
                        if profiler is not None:
                            profiler.tensors(restored_grid_time_codes=values)
                for day in range(days):
                    with restore_phase(profiler, "restore_V_decode" if "V" in condition else "restore_direct_output",
                                       dict(position_start=start, day=day)):
                        out = models["V"].decode(values[day]) if "V" in condition else values[day]
                        if profiler is not None:
                            profiler.tensors(raw_normalized_output=out)
                    with restore_phase(profiler, "restore_output_transfer", dict(position_start=start, day=day)):
                        output = out.float().cpu().numpy()
                    yield day, start, output
        else:
            # Small intermediate reconstruction lives in host RAM, not in the package.
            if "T" in condition:
                c = info["stages"]["T"]["channels"]
                spatial = np.empty((days, latent.shape[1], c), np.float32)
                if profiler is not None:
                    profiler.host_intermediate("restored_spatial_tokens_host_array", spatial.nbytes)
                for start in range(0, latent.shape[1], chunk):
                    stop = min(start + chunk, latent.shape[1])
                    with restore_phase(profiler, "restore_read_transfer", dict(position_start=start)):
                        seq = torch.as_tensor(np.asarray(latent[:, start:stop]), device=device).permute(1, 2, 0)
                        if profiler is not None:
                            profiler.tensors(latent_input=seq)
                    with restore_phase(profiler, "restore_T_decode", dict(position_start=start)):
                        out = models["T"].denormalize(models["T"].decode(seq))[..., :days]
                        if profiler is not None:
                            profiler.tensors(restored_spatial_time_codes=out)
                    with restore_phase(profiler, "restore_output_transfer", dict(position_start=start)):
                        spatial[:, start:stop] = out.permute(2, 0, 1).float().cpu().numpy()
            for day in range(days):
                array = spatial[day:day + 1] if "T" in condition else np.asarray(latent[day:day + 1])
                with restore_phase(profiler, "restore_S_decode", dict(day=day)):
                    grid = models["S"].decode(torch.as_tensor(array, device=device))[0]
                    if profiler is not None:
                        profiler.tensors(full_grid_codes=grid)
                for start in range(0, len(grid), chunk):
                    codes = grid[start:start + chunk]
                    with restore_phase(profiler, "restore_V_decode" if "V" in condition else "restore_direct_output",
                                       dict(position_start=start, day=day)):
                        out = models["V"].decode(codes) if "V" in condition else codes
                        if profiler is not None:
                            profiler.tensors(raw_normalized_output=out)
                    with restore_phase(profiler, "restore_output_transfer", dict(position_start=start, day=day)):
                        output = out.float().cpu().numpy()
                    yield day, start, output
                del grid


def exhaustive_metrics(directory, cache, shared, device, chunk=256, profile_dir=None,
                       profile_first=3, profile_every=200):
    specs = specs_from_report(read_json(shared / "schema.json"))
    with np.load(shared / "normalization.npz") as stats:
        metrics = VariableMetrics(specs, stats["mean"], stats["std"])
    total, loss_sum = 0, 0.0
    weights = np.concatenate([np.full(s.width, 1.0 / (len(specs) * s.width)) for s in specs])
    profiler = (VramRecorder(profile_dir, device, first_updates=profile_first, every_updates=profile_every,
                            metadata=dict(role="exhaustive_reconstruction", package=str(directory),
                                          raw_shape=list(cache.shape), grid_chunk=chunk,
                                          same_complete_output_scope=True)) if profile_dir is not None else None)
    scope = profiler.section("exhaustive_reconstruction") if profiler is not None else nullcontext()
    try:
        with scope, Dataset(shared / "reference_float32.nc", "r") as reference:
            for day, start, prediction in restore_batches(directory, device, chunk, profiler):
                target = np.asarray(cache[day, start:start + len(prediction)])
                physical = np.asarray(reference.variables["values"][day, start:start + len(prediction)], np.float64)
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite reconstruction; evaluation stopped")
                # Same baseline R2 definition, but direct source float32 ground truth.
                for spec in specs:
                    sl = slice(spec.offset, spec.stop)
                    true = physical[:, sl]
                    pred = prediction[:, sl].astype(np.float64) * metrics.std[sl] + metrics.mean[sl]
                    error = pred - true
                    shifted = true - metrics.mean[sl].mean()
                    record = metrics.acc[spec.name]
                    record["n"] += error.size
                    record["sae"] += float(np.abs(error).sum())
                    record["sse"] += float(np.square(error).sum())
                    record["sum"] += float(shifted.sum())
                    record["sum2"] += float(np.square(shifted).sum())
                    record["max"] = max(record["max"], float(np.abs(error).max()))
                loss_sum += float(((prediction.astype(np.float64) - target) ** 2 * weights).sum())
                total += len(prediction)
    except BaseException as error:
        if profiler is not None:
            profiler.metadata["failure"] = dict(type=type(error).__name__, message=str(error))
            profiler.close()
        raise
    profile = profiler.close() if profiler is not None else None
    expected = cache.shape[0] * cache.shape[1]
    if total != expected:
        raise AssertionError(f"incomplete exhaustive evaluation: {total} != {expected}")
    result = metrics.result()
    result.update(evaluated_date_grid_pairs=total, variable_balanced_normalized_MSE=loss_sum / total,
                  evaluation="all dates, all grids, all modeled variable attributes; training archive",
                  decoded_from_saved_package=True, physical_ground_truth="direct source float32 reference NC")
    if profile is not None:
        result["vram_profile"] = profile
        elapsed = profile["phases"]["exhaustive_reconstruction"]["synchronized_wall_seconds"]
        result["exhaustive_date_grid_pairs_per_second"] = total / elapsed if elapsed else None
        result["exhaustive_values_per_second"] = total * cache.shape[-1] / elapsed if elapsed else None
        result["throughput_scope"] = "instrumented decode + NC read + CPU metrics; NOT query latency or pure GPU compute"
    return result


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
