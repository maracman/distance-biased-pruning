#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conditions=(
  distance_quick
  bio_quick
  distance_balanced
  random_er
  bio_inspired
  snip
)

python train.py \
  --dataset cifar100 \
  --sparsity 0.98 \
  --epochs 200 \
  --seeds 3 \
  --conditions "${conditions[@]}"

python train.py \
  --dataset cifar100 \
  --sparsity 0.99 \
  --epochs 200 \
  --seeds 3 \
  --conditions "${conditions[@]}"

python train.py \
  --transfer \
  --sparsity 0.98 \
  --epochs 200 \
  --seeds 3

