"""
step1/step2 공용 도구: 시드, 혼합정밀도, 체크포인트 도우미, 그리고 공식 평가.

평가(`VariableMetrics`)는 정규화 공간의 예측/정답을 변수별 평균·표준편차로
물리 단위로 되돌린 뒤 변수별 MAE/RMSE/R²를 누적합니다. R²는 변수 하나의 모든
연직층·격자·날짜 값을 합쳐 계산하므로 `old/vst_core.reconstruct_write_and_measure`
와 같은 정의입니다. 에러율(error_rate_percent)은 RMSE를 그 변수 값의 표준편차로
나눈 백분율이며 R² = 1 - (에러율/100)² 관계입니다. 정답은 정규화 캐시에서 읽으므로 원본의 결측값은 캐시를
만들 때처럼 평균값으로 채워진 상태로 비교됩니다.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import shutil

from netCDF4 import Dataset
import numpy as np
import torch

from data_utils import SEED, VariableSpec


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    """Return the requested device; never fall back to CPU silently.

    A training run that quietly lands on the CPU is ~100x slower and looks hung
    (this happened after a GPU fell off the bus). Pass --device cpu explicitly
    to run on the CPU.
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{name} requested but CUDA is unavailable "
                           "(driver error? check nvidia-smi); use --device cpu explicitly")
    return torch.device(name)


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def cpu_state(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def write_json(path: Path, value: dict):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return (np.asarray(saved["mean"], dtype=np.float32),
                np.asarray(saved["std"], dtype=np.float32))


class VariableMetrics:
    """Accumulate per-variable errors in physical units, one day at a time."""

    def __init__(self, specs: list[VariableSpec], mean: np.ndarray, std: np.ndarray):
        self.specs = specs
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.acc = {
            s.name: {"n": 0, "sae": 0.0, "sse": 0.0, "sum": 0.0, "sum2": 0.0, "max": 0.0}
            for s in specs
        }

    def update(self, prediction_norm: np.ndarray, target_norm: np.ndarray):
        """Both arrays: [cells, features] in normalized units."""
        for spec in self.specs:
            sl = slice(spec.offset, spec.stop)
            true = target_norm[:, sl] * self.std[sl] + self.mean[sl]
            pred = prediction_norm[:, sl] * self.std[sl] + self.mean[sl]
            error = pred - true
            # shift by one constant per variable so sum2 - sum^2/n does not cancel badly
            shifted = true - self.mean[sl].mean()
            record = self.acc[spec.name]
            record["n"] += error.size
            record["sae"] += float(np.abs(error).sum())
            record["sse"] += float(np.square(error).sum())
            record["sum"] += float(shifted.sum())
            record["sum2"] += float(np.square(shifted).sum())
            record["max"] = max(record["max"], float(np.abs(error).max()))

    def result(self) -> dict:
        """R² is None when a variable has (numerically) zero variance in the scope."""
        variables = {}
        for spec in self.specs:
            r = self.acc[spec.name]
            n = max(r["n"], 1)
            total_variance = r["sum2"] - r["sum"] ** 2 / n
            defined = total_variance > 1.0e-12 * max(r["sum2"], 1.0e-30)
            variables[spec.name] = {
                "units": spec.units,
                "long_name": spec.long_name,
                "values": r["n"],
                "MAE": r["sae"] / n,
                "RMSE": (r["sse"] / n) ** 0.5,
                "max_absolute_error": r["max"],
                "R2": 1.0 - r["sse"] / total_variance if defined else None,
                # error rate: RMSE as a percentage of the variable's own spread (std)
                "error_rate_percent": 100.0 * (r["sse"] / total_variance) ** 0.5 if defined else None,
            }
        scored = {k: v["R2"] for k, v in variables.items() if v["R2"] is not None}
        r2 = np.asarray(list(scored.values()), dtype=np.float64)
        err = np.asarray([variables[k]["error_rate_percent"] for k in scored], dtype=np.float64)
        return {
            "mean_error_rate_percent": float(err.mean()) if err.size else None,
            "median_error_rate_percent": float(np.median(err)) if err.size else None,
            "max_error_rate_percent": float(err.max()) if err.size else None,
            "mean_R2": float(r2.mean()) if r2.size else None,
            "median_R2": float(np.median(r2)) if r2.size else None,
            "variables_scored": int(r2.size),
            "variables_R2_undefined": sorted(set(variables) - set(scored)),
            "variables_R2_ge_0.9": int((r2 >= 0.9).sum()),
            "variables_R2_ge_0.5": int((r2 >= 0.5).sum()),
            "variables_R2_lt_0": int((r2 < 0.0).sum()),
            "worst_10": sorted(scored, key=scored.get)[:10],
            "variables": variables,
        }


def write_reconstructed_netcdf(source: Path, destination: Path,
                               specs: list[VariableSpec], values_physical: np.ndarray):
    """Copy `source` and overwrite each modelled variable with [grid, features] values.

    Variables that are not modelled (coordinates, static fields) stay byte-identical.
    """
    shutil.copy2(source, destination)
    with Dataset(destination, "r+") as dataset:
        for spec in specs:
            variable = dataset.variables[spec.name]
            grid = values_physical.shape[0]
            block = values_physical[:, spec.offset:spec.stop].reshape(
                (grid,) + tuple(spec.feature_shape)
            )
            remaining = [d for d in spec.dimensions if d != spec.time_dimension]
            block = np.moveaxis(block, 0, remaining.index(spec.grid_dimension))
            key = tuple(0 if d == spec.time_dimension else slice(None)
                        for d in spec.dimensions)
            variable[key] = block.astype(variable.dtype)
