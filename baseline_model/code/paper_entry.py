"""[PAPER UPDATE] Preserve original stage entry names and checkpoint/latent names.

The implementation of the measured raw-variable loss and stopping rule lives in
paper_training/. Original training functions remain in the stage files as
legacy_main; the default executable path below uses the current paper policy.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

HELPERS = Path(__file__).resolve().parent / "paper_training"
if str(HELPERS) not in sys.path:
    sys.path.insert(0, str(HELPERS))

from ablation_storage import atomic_save, read_json
from common import write_json
from pipeline_utils import parse_args as paper_args
import train_clear_se as backend

STAGE_NUMBER = {"V": 1, "S": 2, "T": 3}
MARKER = "clear_paper_run.json"


def link_or_copy(source, destination):
    """Output compatibility view: hard-link immutable files, otherwise copy."""
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        if destination.stat().st_size == source.stat().st_size:
            import hashlib
            def digest(path):
                sha = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(2**20), b""):
                        sha.update(chunk)
                return sha.digest()
            if digest(source) == digest(destination):
                return
        raise FileExistsError(f"Compatibility output already differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_baseline_files(root, destination, stage):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shared, directory = root / "shared_preprocessing", root / "M1" / ("train_" + stage)
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    protocol = read_json(root / "experiment_protocol.json")
    metadata = dict(saved, arguments=protocol["arguments"], training=read_json(directory / "training.json"))
    if stage == "V":
        schema = dict(read_json(shared / "schema.json"))
        schema["source_files"] = [Path(p["path"]).name for p in read_json(shared / "preprocessing.json")["sources"]]
        metadata["schema"] = schema
        write_json(destination / "schema_report.json", schema)
        atomic_save(metadata, destination / "step1_codec.pt")
        for name in ("normalization.npz", "normalized_float32.npy"):
            link_or_copy(shared / name, destination / name)
        link_or_copy(directory / "codes.npy", destination / "step1_embeddings_float32.npy")
    elif stage == "S":
        metadata.update(kind="vit", step1_dir=str(root / "step1"))
        atomic_save(metadata, destination / "step2_model.pt")
        link_or_copy(shared / "cube_layout.npz", destination / "cube_layout.npz")
        link_or_copy(directory / "codes.npy", destination / "step2_latent_float32.npy")
    else:
        metadata.update(step2_kind="vit", step2_dir=str(root / "step2"),
                        days=read_json(shared / "preprocessing.json")["days"])
        atomic_save(metadata, destination / "step3_model.pt")
        path = destination / "step3_latent_float32.npy"
        if not path.exists():
            codes = np.load(directory / "codes.npy", mmap_mode="r")
            days, tokens, channels = codes.shape
            side = int(round((tokens / 6) ** .5))
            if tokens != 6 * side * side:
                raise ValueError("Expected six equal square ViT token faces")
            out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                           shape=(days, 6, channels, side, side))
            for day in range(days):
                out[day] = codes[day].reshape(6, side, side, channels).transpose(0, 3, 1, 2)
            out.flush()
            del out
        link_or_copy(root / "M1" / "summary.json", destination / "summary.json")
    write_json(destination / MARKER, dict(paper_root=str(root), stage=stage,
                                         current_paper_settings=True, legacy_label="M1"))


def run_stage(stage, argv=None):
    p = argparse.ArgumentParser(description=f"Updated baseline step{STAGE_NUMBER[stage]} with CLEAR SE paper settings")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--step1-dir", type=Path)
    p.add_argument("--step2-dir", type=Path)
    p.add_argument("--model", choices=("vit",), default="vit")
    p.add_argument("--run-dir", type=Path, help="Canonical paper run root, shared across all three stages")
    a, extra = p.parse_known_args(argv)
    prefix = a.step1_dir if stage == "S" else a.step2_dir if stage == "T" else None
    if prefix is not None:
        marker = read_json(prefix / MARKER)
        inferred = Path(marker["paper_root"])
        if a.run_dir and a.run_dir.resolve() != inferred.resolve():
            p.error("Input stage belongs to a different --run-dir")
        a.run_dir = inferred
    root = (a.run_dir or (a.output_dir.parent if a.output_dir else Path("runs/clear_se_30day"))).resolve()
    destination = a.output_dir or root / f"step{STAGE_NUMBER[stage]}"
    # Use canonical aliases so original step2/step3 decoder helpers resolve prefixes.
    if destination.resolve() != root / f"step{STAGE_NUMBER[stage]}":
        p.error("Use <run-dir>/step1, step2 and step3 for the original baseline layout")
    manifest = root / "baseline_entry_arguments.json"
    if manifest.exists():
        original = read_json(manifest)["arguments"]
        if extra and extra != original:
            p.error("Settings already fixed by step1; pass them only to step1 or choose a new run directory")
        extra = original
    else:
        if stage != "V":
            p.error("Run step1_horizontal.py first; completed prefix outputs are required")
        root.mkdir(parents=True, exist_ok=True)
        write_json(manifest, dict(arguments=extra))
    arguments = [*extra, "--output-dir", str(root), "--conditions", "M1", "--stop-after", stage, "--in-process"]
    # The same remaining options/defaults are passed to all stages. Stage selection
    # changes only where execution pauses, not the protocol/training fingerprint.
    paper_args(arguments)
    if backend.main(arguments) != 0:
        raise RuntimeError("Paper training failed")
    for completed in "VST"[:STAGE_NUMBER[stage]]:
        export_baseline_files(root, root / f"step{STAGE_NUMBER[completed]}", completed)
    print(f"[baseline output] {destination}", flush=True)
    return 0
