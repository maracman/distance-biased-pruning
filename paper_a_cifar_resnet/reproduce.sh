#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

conditions=(
  distance_dev
  balanced_dev
  distance_prior
  random_er
  balanced_random
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

