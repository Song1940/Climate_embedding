#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null || { printf 'Activate the CUDA PyTorch environment first.\n' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
run="${RUN_DIR:-runs/clear_se_30day}"
printf '[original baseline filenames + CLEAR SE paper settings]\nphysical GPU=%s\n' "$CUDA_VISIBLE_DEVICES"
"$PYTHON_BIN" -B -u step1_horizontal.py --run-dir "$run" --output-dir "$run/step1" \
  --data-dir "${DATA_DIR:-/mnt/mybook/KCM_output}" --days "${DAYS:-30}" \
  --start-date "${START_DATE:-20000101}" --device cuda:0 "$@"
"$PYTHON_BIN" -B -u step2_vertical.py --step1-dir "$run/step1" --output-dir "$run/step2" --model vit
"$PYTHON_BIN" -B -u step3_temporal.py --step2-dir "$run/step2" --output-dir "$run/step3"
