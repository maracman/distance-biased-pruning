#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python train.py \
  --phase all \
  --seeds 3 \
  --hidden_sizes 64,128,256,512 \
  --include_2d_variants \
  --device auto

