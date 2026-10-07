#!/usr/bin/env python3
"""Decode one requested day/grid/variable/level from one patch latent only.

압축된 전체 latent를 다 읽지 않고, 요청한 격자가
속한 패치의 압축 표현만 읽어서 특정 (날짜, 격자, 변수, 층) 값 하나만 부분
복원해주는 조회 도구입니다. """

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from netCDF4 import Dataset
import numpy as np
import torch

from all_variable_data import specs_from_report
from base_training_utils import autocast_context
from hybrid_patch_model import MultiTokenVariableCodec
from train_hybrid_30day import create_model
from evaluate_unseen_10day_hybrid import load_layout, namespace_from_training


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--latent-nc", type=Path, default=None)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--grid-index", type=int, required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-vram-fraction", type=float, default=0.55)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--temporal-batch-patches", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction, device)
    checkpoint = torch.load(
        args.run_dir / "full_model.pt", map_location="cpu", weights_only=False
    )
    layout = load_layout(args.run_dir / "patch_layout.npz")
    model_args = namespace_from_training(checkpoint["arguments"], args)
    specs = specs_from_report(checkpoint["schema"])
    names = [spec.name for spec in specs]
    if args.variable not in names:
        raise KeyError(f"{args.variable!r} not in modelled variables")
    variable_index = names.index(args.variable)
    spec = specs[variable_index]
    if not 0 <= args.level_index < spec.width:
        raise IndexError(
            f"level-index {args.level_index} outside [0,{spec.width - 1}]"
        )

    codec = MultiTokenVariableCodec(
        [item.width for item in specs],
        token_dim=model_args.input_token_dim,
        grid_slots=model_args.grid_slots,
        heads=model_args.heads,
        sparse_indices=checkpoint.get("sparse_indices", []),
    ).to(device)
    codec.load_state_dict(checkpoint["variable_codec"])
    codec.eval()
    model = create_model(model_args, layout, device)
    model.load_state_dict(checkpoint["hybrid_model"])
    model.eval()

    core = layout["core_cells"]
    valid = layout["core_valid"]
    match = np.argwhere((core == args.grid_index) & valid)
    if len(match) != 1:
        raise ValueError(
            f"grid {args.grid_index} must occur as one core cell; found {len(match)}"
        )
    patch_id = int(match[0, 0])
    latent_path = args.latent_nc or (args.run_dir / "latent_float32.nc")
    started = time.perf_counter()
    with Dataset(latent_path, "r") as dataset:
        source_days = int(dataset.getncattr("source_days"))
        if not 0 <= args.day_index < source_days:
            raise IndexError(
                f"day-index {args.day_index} outside [0,{source_days - 1}]"
            )
        values = np.asarray(
            dataset.variables["latent"][:, patch_id:patch_id + 1, :, :],
            dtype=np.float32,
        )
    io_seconds = time.perf_counter() - started

    latent = torch.as_tensor(values, device=device)
    patch_ids = torch.as_tensor([patch_id], dtype=torch.long, device=device)
    with torch.no_grad(), autocast_context(device):
        temporal = model.decode_temporal(latent, source_days)
        decoded_codes, cells = model.decode_spatial_patches(
            temporal[args.day_index:args.day_index + 1], patch_ids
        )
        local = torch.nonzero(cells == args.grid_index, as_tuple=False).flatten()
        if len(local) != 1:
            raise RuntimeError("decoded patch did not return the requested grid")
        code = decoded_codes[0, local[0]][None]
        hidden = codec._decoder_hidden(code)
        normalized = codec.decoders[variable_index](
            hidden[:, variable_index]
        ).float().cpu().numpy()[0]
    decode_seconds = time.perf_counter() - started - io_seconds

    normalization = np.load(args.run_dir / "normalization.npz", allow_pickle=False)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    physical = normalized * std[spec.offset:spec.stop] + mean[spec.offset:spec.stop]
    result = {
        "day_index": args.day_index,
        "grid_index": args.grid_index,
        "patch_id": patch_id,
        "variable": spec.name,
        "long_name": spec.long_name,
        "units": spec.units,
        "level_index": args.level_index,
        "normalized_value": float(normalized[args.level_index]),
        "physical_value": float(physical[args.level_index]),
        "latent_slice_read": list(values.shape),
        "io_seconds": io_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
