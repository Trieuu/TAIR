#!/usr/bin/env bash
set -Eeuo pipefail
# Run TeReDiff validation using the local patched config.
# Uses GPU 0 by default; override with: CUDA_VISIBLE_DEVICES=1 bash ...
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
accelerate launch val.py \
  --config configs/val/local_val_terediff.yaml \
  --config_testr testr/configs/TESTR/TESTR_R_50_Polygon.yaml
