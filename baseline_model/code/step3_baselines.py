#!/usr/bin/env python3
"""
Step 3 비교 기준 (선형, AI 아님): 이틀 묶음 PCA.

TCN 오토인코더와 같은 저장 구조의 선형판입니다. 위치별 시간 평균 μ를 뺀 뒤, 위치마다
연속 이틀 [2C]를 한 행으로 모아 전체 위치·기간에 공통인 PCA 기저 L개로 줄입니다.
저장값은 [T/2, P, L]로 TCN과 같고, 기저 [2C, L]은 디코더 가중치에 해당합니다.
에러율은 step2_decoder.embedding_metrics와 같은 정의입니다.

    python step3_baselines.py --step2-dir runs/step2_30day_cnn15 --latent-channels 384 320 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import write_json
from step2_decoder import LATENT_FILE, MODEL_FILE as STEP2_MODEL_FILE
from step3_decoder import compare
from step3_model import cube_to_sequences, sequences_to_cube, to_cube


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step2-dir", type=Path, required=True)
    p.add_argument("--latent-channels", type=int, nargs="+", default=[384, 320, 256])
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    kind = torch.load(args.step2_dir / STEP2_MODEL_FILE, map_location="cpu", weights_only=False)["kind"]
    cube = to_cube(np.load(args.step2_dir / LATENT_FILE), kind).astype(np.float32)
    days, _, c, side, _ = cube.shape
    seq = cube_to_sequences(cube).astype(np.float64)              # [P, C, T]
    mean = seq.mean(2, keepdims=True)
    a = seq - mean
    if days % 2:
        a = np.concatenate([a, a[..., -1:]], axis=2)
    t2 = a.shape[2] // 2
    rows = a.reshape(a.shape[0], c, t2, 2).transpose(0, 2, 3, 1).reshape(-1, 2 * c)   # [P·T/2, 2C]
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    result = {"step2_dir": str(args.step2_dir), "shape": list(cube.shape), "pair_pca": []}
    for l in args.latent_channels:
        basis = vt[:l]
        rec_rows = (rows @ basis.T) @ basis
        rec = rec_rows.reshape(a.shape[0], t2, 2, c).transpose(0, 3, 1, 2).reshape(a.shape[0], c, 2 * t2)
        rec = (rec[..., :days] + mean).astype(np.float32)
        metrics = compare(sequences_to_cube(rec, side), cube)
        stored = t2 * a.shape[0] * l + mean.size + c
        row = {"L": l, "value_ratio_latent_only": cube.size / (t2 * a.shape[0] * l),
               "ratio_with_mean": cube.size / stored, "ratio_with_mean_and_basis": cube.size / (stored + basis.size),
               "error_rate_percent": metrics["error_rate_percent"], "R2": metrics["R2"]}
        result["pair_pca"].append(row)
        print(f"[pair-PCA L={l}] err={row['error_rate_percent']:.2f}% ratio={row['ratio_with_mean']:.3f}x "
              f"(+basis {row['ratio_with_mean_and_basis']:.3f}x)", flush=True)
    output = args.output or Path(f"runs/step3_baselines_{args.step2_dir.name}.json")
    write_json(output, result)
    print(f"[done] {output}", flush=True)


if __name__ == "__main__":
    main()
