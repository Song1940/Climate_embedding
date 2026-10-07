"""
KIM 일별 NetCDF 데이터 계층 (step1, step2 공용).

1. 스키마 파악: 격자 정렬된 시간 의존 부동소수 변수를 찾아 `VariableSpec`으로
   정리합니다. 각 변수는 격자당 평탄화된 벡터에서 [offset, offset+width) 구간을
   차지합니다 (전체 6,035개 값).
2. 정규화 캐시: 학습 셀 기준 평균/표준편차로 정규화한 float32 memmap
   `[day, grid, feature]` 을 디스크에 만들어, 원본 NetCDF를 매번 다시 읽지 않게 합니다.
3. 구면 패치: 15,002개 셀을 재귀 이분할로 최대 `target`개씩 묶고, 위경도 층화
   방식으로 학습/검증 패치를 나눕니다.

`old/all_variable_data.py`에서 그래프 모델 전용 함수(build_spherical_knn_graph)만
뺀 것입니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import re
from typing import Iterable

from netCDF4 import Dataset
import numpy as np


SEED = 0
LATITUDE_NAMES = ("lats", "lat", "latitude", "clat")
LONGITUDE_NAMES = ("lons", "lon", "longitude", "clon")
COORDINATE_NAMES = {
    "time", "time_ini", "levs", "ilevs", "level", "levels",
    "lats", "lat", "latitude", "clat",
    "lons", "lon", "longitude", "clon",
    "grid", "grid_id", "cell", "cell_id",
}


@dataclass(frozen=True)
class VariableSpec:
    name: str
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    time_dimension: str
    grid_dimension: str
    feature_dimensions: tuple[str, ...]
    feature_shape: tuple[int, ...]
    width: int
    offset: int
    units: str
    long_name: str

    @property
    def stop(self) -> int:
        return self.offset + self.width

    def to_json(self) -> dict:
        value = asdict(self)
        for key in ("dimensions", "shape", "feature_dimensions", "feature_shape"):
            value[key] = list(value[key])
        return value


def _file_date(path: Path):
    match = re.search(r"(\d{8})", path.name)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def discover_daily_files(
    data_dir: Path, pattern: str, start_date: str, days: int
) -> list[Path]:
    start = datetime.strptime(start_date, "%Y%m%d").date()
    by_date = {}
    for path in data_dir.glob(pattern):
        date = _file_date(path)
        if date is not None:
            by_date.setdefault(date, path)
    required = [start + timedelta(days=index) for index in range(days)]
    missing = [date.isoformat() for date in required if date not in by_date]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"missing {len(missing)} daily files from the requested period: {preview}"
        )
    return [by_date[date] for date in required]


def _filled(variable, key=None) -> np.ndarray:
    values = variable[:] if key is None else variable[key]
    if np.ma.isMaskedArray(values):
        values = values.filled(np.nan)
    values = np.asarray(values, dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    values[np.abs(values) > 1.0e30] = np.nan
    return values


def _coordinate(dataset: Dataset, candidates: Iterable[str]):
    for name in candidates:
        if name in dataset.variables:
            values = _filled(dataset.variables[name]).reshape(-1)
            if values.size > 1:
                return name, values
    raise KeyError(f"none of these coordinates were found: {tuple(candidates)}")


def inspect_schema(first_file: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Discover every floating, dynamic, grid-aligned physical variable."""
    with Dataset(first_file, "r") as dataset:
        lon_name, lons = _coordinate(dataset, LONGITUDE_NAMES)
        lat_name, lats = _coordinate(dataset, LATITUDE_NAMES)
        if lons.shape != lats.shape:
            raise ValueError("longitude and latitude shapes differ")
        grid_count = int(lons.size)
        lon_var = dataset.variables[lon_name]
        if lon_var.ndim != 1:
            raise ValueError(f"{lon_name} must be one-dimensional, got {lon_var.shape}")
        grid_dimension = lon_var.dimensions[0]

        specs: list[VariableSpec] = []
        copied = []
        skipped = []
        offset = 0
        for name, variable in dataset.variables.items():
            dimensions = tuple(variable.dimensions)
            dtype = np.dtype(variable.dtype)
            reason = None
            time_dimensions = [dim for dim in dimensions if "time" in dim.lower()]
            if name.lower() in COORDINATE_NAMES or (
                len(dimensions) == 1 and dimensions[0] == name
            ):
                reason = "coordinate"
            elif grid_dimension not in dimensions:
                reason = "not_grid_aligned"
            elif not np.issubdtype(dtype, np.floating):
                reason = "non_floating_or_flag"
            elif not time_dimensions:
                reason = "static_grid_field"
            elif len(time_dimensions) != 1:
                reason = "multiple_time_dimensions"
            elif int(dataset.dimensions[time_dimensions[0]].size) != 1:
                reason = "daily_file_time_not_one"

            if reason is not None:
                copied.append({
                    "name": name,
                    "dimensions": list(dimensions),
                    "shape": list(variable.shape),
                    "dtype": str(dtype),
                    "handling": "copied_losslessly",
                    "reason": reason,
                })
                continue

            time_dimension = time_dimensions[0]
            feature_dimensions = tuple(
                dim for dim in dimensions
                if dim not in (time_dimension, grid_dimension)
            )
            feature_shape = tuple(int(dataset.dimensions[dim].size)
                                  for dim in feature_dimensions)
            width = int(np.prod(feature_shape, dtype=np.int64)) if feature_shape else 1
            if width < 1:
                skipped.append({"name": name, "reason": "empty_feature_axis"})
                continue
            spec = VariableSpec(
                name=name,
                dimensions=dimensions,
                shape=tuple(int(size) for size in variable.shape),
                dtype=str(dtype),
                time_dimension=time_dimension,
                grid_dimension=grid_dimension,
                feature_dimensions=feature_dimensions,
                feature_shape=feature_shape,
                width=width,
                offset=offset,
                units=str(getattr(variable, "units", "")),
                long_name=str(getattr(variable, "long_name", "")),
            )
            specs.append(spec)
            offset += width

        if not specs:
            raise ValueError("no floating, time-dependent, grid-aligned variables found")
        report = {
            "source_file": first_file.name,
            "grid_dimension": grid_dimension,
            "grid_count": grid_count,
            "longitude_variable": lon_name,
            "latitude_variable": lat_name,
            "modelled_variable_count": len(specs),
            "flattened_feature_count": offset,
            "modelled_variables": [spec.to_json() for spec in specs],
            "losslessly_copied_variables": copied,
            "skipped": skipped,
        }
    return report, lons.astype(np.float32), lats.astype(np.float32)


def specs_from_report(report: dict) -> list[VariableSpec]:
    output = []
    for item in report["modelled_variables"]:
        item = dict(item)
        for key in ("dimensions", "shape", "feature_dimensions", "feature_shape"):
            item[key] = tuple(item[key])
        output.append(VariableSpec(**item))
    return output


def read_grid_matrix(dataset: Dataset, spec: VariableSpec) -> np.ndarray:
    variable = dataset.variables[spec.name]
    if tuple(variable.dimensions) != spec.dimensions:
        raise ValueError(
            f"schema changed for {spec.name}: {variable.dimensions} != {spec.dimensions}"
        )
    key = []
    remaining = []
    for dimension in spec.dimensions:
        if dimension == spec.time_dimension:
            key.append(0)
        else:
            key.append(slice(None))
            remaining.append(dimension)
    values = _filled(variable, tuple(key))
    grid_axis = remaining.index(spec.grid_dimension)
    values = np.moveaxis(values, grid_axis, 0)
    if values.shape[0] * int(np.prod(values.shape[1:] or (1,))) == 0:
        raise ValueError(f"empty variable {spec.name}")
    values = values.reshape(values.shape[0], -1)
    if values.shape[1] != spec.width:
        raise ValueError(
            f"flattened width changed for {spec.name}: {values.shape[1]} != {spec.width}"
        )
    return values


def compute_normalization(
    files: list[Path], specs: list[VariableSpec], training_cells: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-feature mean/std over training cells and all given days.

    Sums are accumulated around a per-feature reference value (the first file's
    training-cell mean) so large-magnitude fields such as pressure in Pa do not
    lose their variance to cancellation. Features that are constant relative to
    their variable's scale (e.g. pure-pressure top levels of `p`) would otherwise
    divide float32 rounding noise by ~1e-6; they get the median std of the
    variable's non-constant features instead.
    """
    features = max(spec.stop for spec in specs)
    reference = None
    sums = np.zeros(features, dtype=np.float64)
    squares = np.zeros(features, dtype=np.float64)
    counts = np.zeros(features, dtype=np.int64)
    missing_by_variable = {spec.name: 0 for spec in specs}
    for day, path in enumerate(files):
        print(f"[statistics {day + 1:02d}/{len(files)}] {path.name}", flush=True)
        with Dataset(path, "r") as dataset:
            for spec in specs:
                sl = slice(spec.offset, spec.stop)
                values = read_grid_matrix(dataset, spec)[training_cells].astype(np.float64)
                finite = np.isfinite(values)
                if reference is None:
                    reference = np.zeros(features, dtype=np.float64)
                    reference_set = np.zeros(features, dtype=bool)
                if not reference_set[sl].all():
                    reference[sl] = np.nan_to_num(np.nanmean(np.where(finite, values, np.nan), axis=0))
                    reference_set[sl] = True
                shifted = np.where(finite, values - reference[sl], 0.0)
                sums[sl] += shifted.sum(axis=0)
                squares[sl] += np.square(shifted).sum(axis=0)
                counts[sl] += finite.sum(axis=0)
                missing_by_variable[spec.name] += int(np.count_nonzero(~finite))
    mean, std = finalize_statistics(reference, sums, squares, counts, specs)
    return mean, std, missing_by_variable


def finalize_statistics(reference, sums, squares, counts, specs):
    """Mean/std from sums accumulated around `reference`, with the relative
    near-constant rule (see compute_normalization)."""
    if np.any(counts == 0):
        bad = np.flatnonzero(counts == 0)[:20].tolist()
        raise ValueError(f"features with no finite training values: {bad}")
    shifted_mean = sums / counts
    variance = np.maximum(squares / counts - np.square(shifted_mean), 0.0)
    mean = reference + shifted_mean
    std = np.sqrt(variance)
    for spec in specs:
        sl = slice(spec.offset, spec.stop)
        scale = float(np.max(np.abs(mean[sl]) + std[sl]))
        constant = std[sl] <= 1.0e-6 * max(scale, 1.0e-30)
        if constant.any():
            varying = std[sl][~constant]
            fill = float(np.median(varying)) if varying.size else max(scale, 1.0)
            block = std[sl]
            block[constant] = fill
            std[sl] = block
    std[std <= 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------- one-pass parallel cache

def _cache_worker(job):
    """Read one daily file, write raw float32 values, return shifted sums."""
    day, path, output, shape, specs, reference, cells = job
    cache = np.load(output, mmap_mode="r+")
    features = shape[2]
    sums = np.zeros(features); squares = np.zeros(features)
    counts = np.zeros(features, dtype=np.int64)
    missing = {}
    with Dataset(path, "r") as dataset:
        for spec in specs:
            sl = slice(spec.offset, spec.stop)
            values = read_grid_matrix(dataset, spec)                   # float32 [grid, width]
            cache[day, :, sl] = values
            chosen = values if cells is None else values[cells]
            finite = np.isfinite(chosen)
            shifted = np.where(finite, chosen.astype(np.float64) - reference[sl], 0.0)
            sums[sl] = shifted.sum(0)
            squares[sl] = np.square(shifted).sum(0)
            counts[sl] = finite.sum(0)
            missing[spec.name] = int((~finite).sum())
    cache.flush()
    del cache
    return day, sums, squares, counts, missing


def _normalize_worker(job):
    day, output, mean, std = job
    cache = np.load(output, mmap_mode="r+")
    block = (cache[day] - mean) / std
    block[~np.isfinite(block)] = 0.0
    cache[day] = block
    cache.flush()
    del cache
    return day


def build_cache_one_pass(files, specs, grid_count, output: Path, training_cells=None,
                         workers: int = 8):
    """Normalized float32 cache [day, grid, feature] reading each NetCDF file once.

    Workers read files in parallel, write raw float32 values into the memmap and
    return statistics accumulated around a reference taken from the first file.
    After all files, mean/std are finalized (same rules as compute_normalization)
    and the memmap is normalized in place. Missing values become 0 (= the mean).
    """
    from concurrent.futures import ProcessPoolExecutor
    features = max(spec.stop for spec in specs)
    shape = (len(files), grid_count, features)
    print(f"[cache] one pass, {workers} workers, shape={list(shape)} "
          f"GiB={np.prod(shape) * 4 / 2**30:.2f}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=shape).flush()

    reference = np.zeros(features)
    with Dataset(files[0], "r") as dataset:
        for spec in specs:
            values = read_grid_matrix(dataset, spec)
            if training_cells is not None:
                values = values[training_cells]
            with np.errstate(all="ignore"):
                reference[spec.offset:spec.stop] = np.nan_to_num(np.nanmean(values, axis=0))

    sums = np.zeros(features); squares = np.zeros(features)
    counts = np.zeros(features, dtype=np.int64)
    missing = {spec.name: 0 for spec in specs}
    jobs = [(day, path, output, shape, specs, reference, training_cells)
            for day, path in enumerate(files)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for done, (day, s1, s2, c, miss) in enumerate(pool.map(_cache_worker, jobs), 1):
            sums += s1; squares += s2; counts += c
            for name, value in miss.items():
                missing[name] += value
            print(f"[cache read {done:02d}/{len(files)}] {files[day].name}", flush=True)
    mean, std = finalize_statistics(reference, sums, squares, counts, specs)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_normalize_worker, [(day, output, mean, std) for day in range(len(files))]))
    print("[cache] normalized in place", flush=True)
    return mean, std, missing


def build_normalized_cache(
    files: list[Path], specs: list[VariableSpec], mean: np.ndarray,
    std: np.ndarray, grid_count: int, output: Path,
) -> np.memmap:
    features = int(mean.size)
    estimated = len(files) * grid_count * features * 4
    print(
        f"[cache] shape={[len(files), grid_count, features]} "
        f"estimated_float32_GiB={estimated / 2**30:.3f}", flush=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.float32,
        shape=(len(files), grid_count, features),
    )
    for day, path in enumerate(files):
        print(f"[cache {day + 1:02d}/{len(files)}] {path.name}", flush=True)
        with Dataset(path, "r") as dataset:
            for spec in specs:
                values = read_grid_matrix(dataset, spec)
                sl = slice(spec.offset, spec.stop)
                normalized = (values - mean[sl]) / std[sl]
                normalized[~np.isfinite(normalized)] = 0.0
                cache[day, :, sl] = normalized.astype(np.float32)
        cache.flush()
    return cache


def spherical_xyz(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(np.asarray(lons, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lats, dtype=np.float64))
    return np.stack(
        [np.cos(lat) * np.cos(lon),
         np.cos(lat) * np.sin(lon),
         np.sin(lat)], axis=1,
    )


def build_spherical_patches(lons: np.ndarray, lats: np.ndarray, target: int):
    xyz = spherical_xyz(lons, lats)
    patches = []

    def split(indices):
        if len(indices) <= target:
            patches.append(np.sort(indices).astype(np.int64))
            return
        points = xyz[indices]
        axis = int(np.argmax(points.max(axis=0) - points.min(axis=0)))
        order = indices[np.argsort(points[:, axis], kind="stable")]
        middle = len(order) // 2
        split(order[:middle])
        split(order[middle:])

    split(np.arange(len(lons), dtype=np.int64))
    covered = np.concatenate(patches)
    if not np.array_equal(np.sort(covered), np.arange(len(lons))):
        raise RuntimeError("patches overlap or omit grid cells")
    return patches


def padded_patch_arrays(
    patches: list[np.ndarray], lons: np.ndarray, lats: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = max(len(patch) for patch in patches)
    cells = np.full((len(patches), width), -1, dtype=np.int64)
    valid = np.zeros((len(patches), width), dtype=bool)
    features = np.zeros((len(patches), width, 4), dtype=np.float32)
    xyz = spherical_xyz(lons, lats)
    centroids = np.empty((len(patches), 3), dtype=np.float64)
    radii = []
    raw = []
    for patch_id, patch in enumerate(patches):
        cells[patch_id, :len(patch)] = patch
        valid[patch_id, :len(patch)] = True
        center = xyz[patch].mean(axis=0)
        center /= np.linalg.norm(center)
        centroids[patch_id] = center
        east = np.cross(np.array([0.0, 0.0, 1.0]), center)
        norm = np.linalg.norm(east)
        east = east / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0])
        north = np.cross(center, east)
        dx = xyz[patch] @ east
        dy = xyz[patch] @ north
        radii.append(max(float(np.abs(dx).max()), float(np.abs(dy).max()), 1e-8))
        raw.append((patch, dx, dy))
    scale = float(np.median(radii))
    latitude = np.deg2rad(np.asarray(lats, dtype=np.float64))
    for patch_id, (patch, dx, dy) in enumerate(raw):
        features[patch_id, :len(patch)] = np.stack(
            [dx / scale, dy / scale,
             np.sin(latitude[patch]), np.cos(latitude[patch])], axis=1,
        ).astype(np.float32)
    centroid_lats = np.rad2deg(np.arcsin(np.clip(centroids[:, 2], -1, 1)))
    centroid_lons = np.rad2deg(np.arctan2(centroids[:, 1], centroids[:, 0]))
    return cells, valid, features, centroids.astype(np.float32), np.stack(
        [centroid_lats, centroid_lons], axis=1
    ).astype(np.float32)


def stratified_patch_split(
    centroid_latlon: np.ndarray, validation_fraction: float, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Select validation patches across 6 latitude x 12 longitude bins."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    rng = np.random.RandomState(seed)
    lat = centroid_latlon[:, 0]
    lon = centroid_latlon[:, 1]
    lat_bin = np.clip(np.digitize(lat, [-60, -30, 0, 30, 60]), 0, 5)
    lon_bin = np.clip(((lon + 180.0) // 30.0).astype(int), 0, 11)
    strata = lat_bin * 12 + lon_bin
    target = max(1, int(round(len(lat) * validation_fraction)))
    chosen = []
    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        take = int(round(len(members) * validation_fraction))
        if take > 0:
            chosen.extend(rng.choice(members, take, replace=False).tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < target:
        remaining = np.setdiff1d(np.arange(len(lat)), np.asarray(chosen, dtype=int))
        chosen.extend(rng.choice(remaining, target - len(chosen), replace=False).tolist())
    elif len(chosen) > target:
        chosen = rng.choice(np.asarray(chosen), target, replace=False).tolist()
    validation = np.sort(np.asarray(chosen, dtype=np.int64))
    training = np.setdiff1d(np.arange(len(lat), dtype=np.int64), validation)
    summary = {
        "method": "spatial patches stratified over 6 latitude x 12 longitude bins",
        "seed": seed,
        "fraction": validation_fraction,
        "training_patches": int(training.size),
        "validation_patches": int(validation.size),
        "validation_latitude_minmax": [float(lat[validation].min()), float(lat[validation].max())],
        "validation_longitude_minmax": [float(lon[validation].min()), float(lon[validation].max())],
    }
    return training, validation, summary


def nearest_patch_indices(centroids: np.ndarray, subset: np.ndarray, neighbours: int):
    points = np.asarray(centroids[subset], dtype=np.float64)
    count = len(points)
    if count <= 1:
        return np.empty((count, 0), dtype=np.int64)
    similarity = points @ points.T
    np.fill_diagonal(similarity, -np.inf)
    k = min(neighbours, count - 1)
    indices = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    scores = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(indices, order, axis=1).astype(np.int64)


def save_schema_report(report: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def build_patch_layout(lons: np.ndarray, lats: np.ndarray, patch_size: int,
                       val_patch_fraction: float, seed: int = SEED) -> dict:
    """Patches + optional stratified hold-out split, as one saveable dict.

    val_patch_fraction == 0 (the default for archival compression) trains on every
    patch; `val_patch_ids` is then empty.
    """
    patches = build_spherical_patches(lons, lats, patch_size)
    cells, valid, features, centroids, centroid_latlon = padded_patch_arrays(
        patches, lons, lats
    )
    if val_patch_fraction > 0:
        train_ids, val_ids, summary = stratified_patch_split(
            centroid_latlon, val_patch_fraction, seed
        )
    else:
        train_ids = np.arange(len(patches), dtype=np.int64)
        val_ids = np.empty(0, dtype=np.int64)
        summary = {"method": "no hold-out: every patch is used for training",
                   "training_patches": len(patches), "validation_patches": 0}
    summary.update({"patch_size": int(patch_size), "patch_count": len(patches)})
    return {
        "patch_cells": cells,
        "patch_valid": valid,
        "patch_features": features,
        "centroids": centroids,
        "centroid_latlon": centroid_latlon,
        "train_patch_ids": train_ids,
        "val_patch_ids": val_ids,
        "summary": summary,
    }


def holdout_cells(layout: dict):
    """Held-out validation cells, or None when training used every cell."""
    if len(layout["val_patch_ids"]) == 0:
        return None
    return cells_of(layout, layout["val_patch_ids"])


def cells_of(layout: dict, patch_ids: np.ndarray) -> np.ndarray:
    cells = layout["patch_cells"][patch_ids]
    return np.sort(cells[layout["patch_valid"][patch_ids]]).astype(np.int64)


def save_patch_layout(layout: dict, path: Path):
    arrays = {key: value for key, value in layout.items() if key != "summary"}
    np.savez(path, summary=json.dumps(layout["summary"]), **arrays)


def load_patch_layout(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        layout = {key: saved[key] for key in saved.files if key != "summary"}
        layout["summary"] = json.loads(str(saved["summary"]))
    return layout


# ---------------------------------------------------------------- cubed sphere

def _nearest(queries: np.ndarray, points: np.ndarray, chunk: int = 2048):
    """Index of and angular distance (deg) to the nearest point, chunked."""
    index = np.empty(len(queries), dtype=np.int64)
    distance = np.empty(len(queries), dtype=np.float64)
    points32 = points.astype(np.float32)
    for start in range(0, len(queries), chunk):
        sim = queries[start:start + chunk].astype(np.float32) @ points32.T
        best = sim.argmax(1)
        index[start:start + chunk] = best
        # float32 dot products cannot resolve angles below ~0.03 deg; redo in float64
        exact = (queries[start:start + chunk].astype(np.float64) * points[best]).sum(1)
        distance[start:start + chunk] = np.degrees(np.arccos(np.clip(exact, -1.0, 1.0)))
    return index, distance


def cube_axes(xyz: np.ndarray) -> np.ndarray:
    """Recover the (rotated) cube frame from the 8 cube-corner grid points.

    Corners are the only points with 3 lattice neighbours, which shows up as the
    largest ratio between the 4th and 3rd nearest-neighbour distances.
    """
    points32 = xyz.astype(np.float32)
    top = np.empty((len(xyz), 4), dtype=np.float32)
    for start in range(0, len(xyz), 2048):
        sim = points32[start:start + 2048] @ points32.T
        sim[np.arange(sim.shape[0]), np.arange(start, start + sim.shape[0])] = -2
        top[start:start + 2048] = -np.sort(-sim, axis=1)[:, :4]
    ratio = np.arccos(np.clip(top[:, 3], -1, 1)) / np.arccos(np.clip(top[:, 2], -1, 1))
    corners = xyz[np.argsort(-ratio)[:8]]
    angles = np.degrees(np.arccos(np.clip(corners @ corners.T, -1, 1)))
    expected = np.array([70.53, 109.47, 180.0])
    upper = angles[np.triu_indices(8, 1)]
    if np.abs(upper[:, None] - expected[None]).min(1).max() > 0.5:
        raise ValueError("could not identify the 8 cube corners; is this a cubed-sphere grid?")
    first = corners[0]
    edge_neighbours = corners[np.abs(angles[0] - 70.53) < 0.5]
    axes = np.stack([(first - b) / np.linalg.norm(first - b) for b in edge_neighbours])
    return axes


def cube_face_frames(axes: np.ndarray):
    """Six (normal, u, v) frames: faces +a0, -a0, +a1, -a1, +a2, -a2."""
    frames = []
    for k in range(3):
        for sign in (1.0, -1.0):
            normal = sign * axes[k]
            u = axes[(k + 1) % 3]
            frames.append((normal, u, np.cross(normal, u)))
    return frames


def cube_points(axes: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Equiangular points [6, len, len, 3] for the given per-axis angles."""
    t = np.tan(np.deg2rad(np.asarray(angles_deg, dtype=np.float64)))
    a, b = np.meshgrid(t, t, indexing="ij")
    out = []
    for normal, u, v in cube_face_frames(axes):
        q = normal + a[..., None] * u + b[..., None] * v
        out.append(q / np.linalg.norm(q, axis=-1, keepdims=True))
    return np.stack(out)


def build_cube_layout(lons: np.ndarray, lats: np.ndarray, halo: int = 4) -> dict:
    """Map the KIM cubed-sphere grid (6 N^2 + 2 unique points) to 6 face images.

    Each face becomes an (N+1) x (N+1) node lattice; nodes on shared face edges
    appear on both faces, so every grid cell appears at least once and the map is
    lossless. `node_index` extends each face by `halo` nodes whose values come
    from the nearest cell of the neighbouring face.
    """
    xyz = spherical_xyz(lons, lats)
    count = len(xyz)
    n = int(round(math.sqrt((count - 2) / 6)))
    if 6 * n * n + 2 != count:
        raise ValueError(f"{count} points is not a cubed-sphere grid (6 N^2 + 2)")
    axes = cube_axes(xyz)
    spacing = 90.0 / n
    angles = (np.arange(-halo, n + 1 + halo) * spacing) - 45.0
    nodes = cube_points(axes, angles)
    index, distance = _nearest(nodes.reshape(-1, 3), xyz)
    size = n + 1 + 2 * halo
    index = index.reshape(6, size, size)
    distance = distance.reshape(6, size, size)
    inner = (slice(None), slice(halo, halo + n + 1), slice(halo, halo + n + 1))
    if distance[inner].max() > 1.0e-3:
        raise ValueError(f"face lattice does not match the grid (max {distance[inner].max():.4f} deg)")
    counts = np.bincount(index[inner].ravel(), minlength=count)
    if (counts == 0).any():
        raise ValueError("some grid cells are not on any face lattice")
    return {
        "axes": axes, "n": np.int64(n), "halo": np.int64(halo),
        "node_index": index, "node_halo_distance_deg": distance, "counts": counts,
    }


def cube_center_halo_index(axes: np.ndarray, angles_deg: np.ndarray, interior: slice) -> np.ndarray:
    """Halo map for a coarser per-face grid (e.g. a CNN latent).

    `angles_deg` lists the per-axis angles of the padded grid; `interior` selects
    the entries that are real (stored) positions. Returns [6, L, L] indices into
    the flattened [6, n_int, n_int] interior grid, each pointing at the nearest
    interior position on any face.
    """
    padded = cube_points(axes, angles_deg)                         # [6, L, L, 3]
    real = padded[:, interior, interior]                            # [6, n, n, 3]
    index, _ = _nearest(padded.reshape(-1, 3), real.reshape(-1, 3))
    return index.reshape(padded.shape[:3])


def read_coordinates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Longitudes and latitudes only (a few KB), without touching data variables."""
    with Dataset(path, "r") as dataset:
        _, lons = _coordinate(dataset, LONGITUDE_NAMES)
        _, lats = _coordinate(dataset, LATITUDE_NAMES)
    return lons.astype(np.float64), lats.astype(np.float64)


def save_cube_layout(layout: dict, path: Path):
    np.savez(path, **layout)


def load_cube_layout(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}
