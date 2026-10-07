#!/usr/bin/env python3
"""
Step 3 디코더: step3 잠재값 → 복원된 step2 잠재값 (step2와 같은 형식의 .npy).

복원 결과는 그대로 `step2_decoder.py --latent`에 넣어 step1 임베딩으로, 다시
step1 디코더로 물리 단위까지 되돌릴 수 있습니다.

    python step3_decoder.py --step3-dir runs/step3_cnn15_r2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import pick_device, write_json
from step2_decoder import LATENT_FILE as STEP2_LATENT_FILE, embedding_metrics
from step3_model import TemporalTCNCodec, from_cube, sequences_to_cube, to_cube

MODEL_FILE = "step3_model.pt"
LATENT_FILE = "step3_latent_float32.npy"
RESTORED_FILE = "step3_restored_latent_float32.npy"


def load_step3(step3_dir: Path, device):
    checkpoint = torch.load(step3_dir / MODEL_FILE, map_location="cpu", weights_only=False)
    return TemporalTCNCodec.from_checkpoint(checkpoint).to(device), checkpoint


@torch.no_grad()
def decode_latent(model: TemporalTCNCodec, latent_cube: np.ndarray, days: int, device,
                  chunk: int = 512) -> np.ndarray:
    """step3 latent [T/2, 6, L, S, S] -> anomaly-free step2 cube [T, 6, C, S, S] (float32)."""
    model.eval()
    side = latent_cube.shape[-1]
    seq = np.ascontiguousarray(latent_cube.transpose(1, 3, 4, 2, 0).reshape(-1, latent_cube.shape[2],
                                                                            latent_cube.shape[0]))
    out = np.empty((seq.shape[0], model.config["channels"], days), dtype=np.float32)
    for start in range(0, seq.shape[0], chunk):
        sl = slice(start, min(seq.shape[0], start + chunk))
        a = model.decode(torch.as_tensor(seq[sl], device=device))[..., :days]
        out[sl] = model.denormalize(a, sl).cpu().numpy()
    return sequences_to_cube(out, side)


def compare(restored_cube: np.ndarray, reference_cube: np.ndarray) -> dict:
    """Step 3 independent error: pooled over channels, like step2_decoder.embedding_metrics."""
    t, _, c = reference_cube.shape[:3]
    flat = lambda x: x.transpose(0, 1, 3, 4, 2).reshape(t, -1, c)
    return embedding_metrics(flat(restored_cube), flat(reference_cube))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step3-dir", type=Path, required=True)
    p.add_argument("--latent", type=Path, default=None, help=f"default: <step3-dir>/{LATENT_FILE}")
    p.add_argument("--output", type=Path, default=None, help=f"default: <step3-dir>/{RESTORED_FILE}")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    device = pick_device(args.device)
    model, checkpoint = load_step3(args.step3_dir, device)
    latent = np.load(args.latent or args.step3_dir / LATENT_FILE)
    kind, days = checkpoint["step2_kind"], int(checkpoint["days"])
    restored_cube = decode_latent(model, latent, days, device)
    output = args.output or args.step3_dir / RESTORED_FILE
    np.save(output, from_cube(restored_cube, kind))
    reference = to_cube(np.load(Path(checkpoint["step2_dir"]) / STEP2_LATENT_FILE), kind)
    metrics = compare(restored_cube, reference)
    print(f"[step3 decode] days={days} error_rate={metrics['error_rate_percent']:.3f}% "
          f"R2={metrics['R2']:.6f} -> {output}", flush=True)
    write_json(args.step3_dir / "decode_check.json", {k: v for k, v in metrics.items() if k != "per_day"})


if __name__ == "__main__":
    main()
