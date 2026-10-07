#!/usr/bin/env python3
"""
Step 2의 독립 에러(step1 임베딩 vs step2 복원 임베딩)를 GPU 없이 계산합니다.

GPU 드라이버 장애로 학습 스크립트의 마지막 평가 단계가 실패했을 때 씁니다.
- restored_step1_embeddings_float32.npy가 있으면 그대로 비교합니다.
- 없으면 step2_model.pt(없으면 step2_model_partial.pt)로 CPU에서 인코딩·디코딩합니다.
결과는 <run-dir>/step2_error_cpu.json 에 저장합니다.

    python eval_step2_cpu.py runs/step2_30day_vit15 [--use-partial]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_utils import load_cube_layout
from step2_decoder import embedding_metrics
from step2_model import CubeCodecBase


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--use-partial", action="store_true", help="evaluate step2_model_partial.pt")
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    name = "step2_model_partial.pt" if args.use_partial else "step2_model.pt"
    checkpoint = torch.load(args.run_dir / name, map_location="cpu", weights_only=False)
    reference = np.load(Path(checkpoint["step1_dir"]) / "step1_embeddings_float32.npy", mmap_mode="r")
    restored_path = args.run_dir / "restored_step1_embeddings_float32.npy"
    if restored_path.exists() and not args.use_partial:
        restored = np.load(restored_path, mmap_mode="r")
        source = "restored file written by the training run"
    else:
        model = CubeCodecBase.from_checkpoint(checkpoint, load_cube_layout(args.run_dir / "cube_layout.npz"))
        restored = np.empty(reference.shape, dtype=np.float32)
        with torch.no_grad():
            for day in range(reference.shape[0]):
                x = torch.as_tensor(np.array(reference[day:day + 1]))
                restored[day] = model.decode(model.encode(x))[0].numpy()
                print(f"[cpu decode day {day + 1}/{reference.shape[0]}]", flush=True)
        source = f"CPU encode/decode with {name}"
    metrics = embedding_metrics(restored, reference)
    training = checkpoint.get("training", {})
    result = {
        "run": str(args.run_dir), "checkpoint": name, "source": source,
        "kind": checkpoint["kind"], "config": checkpoint["config"],
        "epochs_run": training.get("epochs_run"), "best_epoch": training.get("best_epoch"),
        "step2_error": metrics, "elapsed_seconds": time.perf_counter() - started,
    }
    (args.run_dir / "step2_error_cpu.json").write_text(json.dumps(result, indent=1))
    print(f"{args.run_dir.name}: error {metrics['error_rate_percent']:.2f}%  R2 {100 * metrics['R2']:.2f}%  "
          f"(epochs {result['epochs_run']}, best {result['best_epoch']}, {source})")


if __name__ == "__main__":
    main()
