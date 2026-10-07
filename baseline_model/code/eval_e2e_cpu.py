#!/usr/bin/env python3
"""
두 단계를 거친 원본 공간 평가를 GPU 없이 수행합니다 (step1 + step2 → 물리 단위).

복원된 step1 임베딩([day, grid, 256])을 고정된 step1 디코더로 풀어, 정규화 캐시의
정답과 비교합니다. 두 가지 지표를 함께 냅니다.
- ours: common.VariableMetrics (물리 단위, 에러율 = RMSE/표준편차, R² 정의 불가 변수 제외)
- old:  old/pca_spatial_baseline.py 와 같은 방식. 정규화 단위에서 변수별 R²를 계산하고,
        R² > -1 인 변수만 평균 (robust_mean). 중앙값은 전체 변수로.

    python eval_e2e_cpu.py --name vit37 --restored runs/step2_30day_vit37/restored_step1_embeddings_float32.npy
    python eval_e2e_cpu.py --name step1_only --restored runs/step1_30day/step1_embeddings_float32.npy
    python eval_e2e_cpu.py --name cnn37 --model-run runs/step2_30day_cnn37 --partial
    python eval_e2e_cpu.py --name pca1024 --pca-k 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from common import VariableMetrics, load_normalization
from data_utils import (build_spherical_patches, load_cube_layout, padded_patch_arrays,
                        read_coordinates, specs_from_report)
from step1_model import Step1Codec
from step2_model import CubeCodecBase

STEP1 = Path("runs/step1_30day")


def restored_from_model(run: Path, partial: bool, reference):
    name = "step2_model_partial.pt" if partial else "step2_model.pt"
    ck = torch.load(run / name, map_location="cpu", weights_only=False)
    model = CubeCodecBase.from_checkpoint(ck, load_cube_layout(run / "cube_layout.npz"))
    out = np.empty(reference.shape, dtype=np.float32)
    with torch.no_grad():
        for d in range(reference.shape[0]):
            out[d] = model.decode(model.encode(torch.as_tensor(np.array(reference[d:d + 1]))))[0].numpy()
    return out


def restored_from_pca(k: int, reference, data_file: Path):
    lons, lats = read_coordinates(data_file)
    cells, valid, *_ = padded_patch_arrays(build_spherical_patches(lons, lats, 15), lons, lats)
    e = torch.as_tensor(np.array(reference))
    t, g, d = e.shape
    p, w = cells.shape
    ci, vi = torch.as_tensor(cells).clamp_min(0), torch.as_tensor(valid)
    x = (e[:, ci] * vi[None, :, :, None]).reshape(t * p, w * d).double()
    mean = x.mean(0)
    evals, evecs = torch.linalg.eigh((x - mean).T @ (x - mean) / x.shape[0])
    v = evecs[:, torch.argsort(evals, descending=True)[:k]]
    rec = (((x - mean) @ v) @ v.T + mean).float().reshape(t, p, w, d)
    out = torch.zeros_like(e)
    flat_c, flat_v = ci.reshape(-1), vi.reshape(-1)
    out[:, flat_c[flat_v]] = rec.reshape(t, p * w, d)[:, flat_v]
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--restored", type=Path)
    ap.add_argument("--model-run", type=Path)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--pca-k", type=int)
    ap.add_argument("--threads", type=int, default=32)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    ck = torch.load(STEP1 / "step1_codec.pt", map_location="cpu", weights_only=False)
    codec = Step1Codec.from_checkpoint(ck)
    specs = specs_from_report(ck["schema"])
    mean, std = load_normalization(STEP1 / "normalization.npz")
    cache = np.load(STEP1 / "normalized_float32.npy", mmap_mode="r")
    reference = np.load(STEP1 / "step1_embeddings_float32.npy", mmap_mode="r")
    if args.restored:
        restored = np.load(args.restored, mmap_mode="r")
    elif args.model_run:
        restored = restored_from_model(args.model_run, args.partial, reference)
    else:
        data_file = Path(ck["arguments"]["data_dir"]) / ck["schema"]["source_files"][0]
        restored = restored_from_pca(args.pca_k, reference, data_file)

    ours = VariableMetrics(specs, mean, std)
    width = cache.shape[2]
    sse = np.zeros(width); s1 = np.zeros(width); s2 = np.zeros(width); n = 0
    with torch.no_grad():
        for day in range(cache.shape[0]):
            pred = np.empty((cache.shape[1], width), dtype=np.float32)
            for a in range(0, cache.shape[1], 4096):
                pred[a:a + 4096] = codec.decode(torch.as_tensor(np.array(restored[day, a:a + 4096]))).numpy()
            true = np.asarray(cache[day], dtype=np.float32)
            ours.update(pred, true)
            diff = (pred - true).astype(np.float64)
            sse += np.square(diff).sum(0); s1 += true.sum(0, dtype=np.float64)
            s2 += np.square(true, dtype=np.float64).sum(0); n += len(true)
            print(f"[{args.name}] day {day + 1}/{cache.shape[0]}", flush=True)
    res = ours.result()
    old = {}
    for spec in specs:                       # old-style: normalized units, pooled per variable
        sl = slice(spec.offset, spec.stop)
        count = n * spec.width
        mse = sse[sl].sum() / count
        var = s2[sl].sum() / count - (s1[sl].sum() / count) ** 2
        old[spec.name] = 1.0 - mse / max(var, 1e-30)
    kept = [v for v in old.values() if v > -1]
    out = {
        "name": args.name,
        "ours": {k: v for k, v in res.items() if k != "variables"},
        "ours_variables": {k: {"R2": v["R2"], "error_rate_percent": v["error_rate_percent"]}
                           for k, v in res["variables"].items()},
        "old_style": {"robust_mean_R2_percent": 100 * float(np.mean(kept)), "kept_variables": len(kept),
                      "dropped_variables": len(old) - len(kept),
                      "median_R2_percent_all": 100 * float(np.median(list(old.values())))},
        "elapsed_seconds": time.perf_counter() - started,
    }
    Path("runs/e2e_eval").mkdir(exist_ok=True)
    Path(f"runs/e2e_eval/{args.name}.json").write_text(json.dumps(out, indent=1, default=float))
    o, s = out["ours"], out["old_style"]
    print(f"{args.name}: ours err mean {o['mean_error_rate_percent']:.2f}% median {o['median_error_rate_percent']:.2f}% "
          f"R2 mean {100*o['mean_R2']:.2f}% | old-style R2 robust mean {s['robust_mean_R2_percent']:.2f}% "
          f"median {s['median_R2_percent_all']:.2f}% (dropped {s['dropped_variables']})")


if __name__ == "__main__":
    main()
