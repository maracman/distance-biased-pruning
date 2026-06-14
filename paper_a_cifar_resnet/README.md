# Paper A: CIFAR ResNet Pruning

Reproduction code for **Inverse-Square Distance Priors Improve Network Pruning at Extreme Sparsity**.

The experiment compares six pruning conditions on ResNet-18 / CIFAR-100 at matched channel-pair granularity:

- `distance_dev`: distance-biased initialization, 3 training epochs, then blended magnitude x proximity pruning
- `distance_prior`: zero-cost distance-biased balanced allocation
- `balanced_dev`: balanced allocation plus short magnitude-pruning phase
- `balanced_random`: balanced allocation at target density
- `random_er`: ERK random sparse baseline
- `snip`: SNIP-style initialization baseline

The table below is from CIFAR-100 at 98% and 99% sparsity. The training script
also accepts CIFAR-10 and lower sparsities for quick development checks; those
development runs are not the reported Paper A results.

## Commands

Full reproduction:

```bash
bash reproduce.sh
```

Quick smoke test:

```bash
python train.py --quick --dataset cifar100 --device auto
```

Single full condition group:

```bash
python train.py \
  --dataset cifar100 \
  --sparsity 0.98 \
  --epochs 200 \
  --seeds 3 \
  --conditions distance_dev balanced_dev distance_prior random_er balanced_random snip
```

Results are written under `paper_a_cifar_resnet/results/` unless `--output_dir` is supplied.

## Reported Main Table

| Condition | 98% sparsity | 99% sparsity | Orphans/dead at 99%, seed 42 |
| --- | ---: | ---: | ---: |
| `distance_dev` | 69.74 +/- 0.03 | 65.95 +/- 0.20 | 0 / 0 |
| `balanced_dev` | 69.64 +/- 0.20 | 65.79 +/- 0.26 | 0 / 0 |
| `distance_prior` | 69.22 +/- 0.30 | 65.08 +/- 0.24 | 0 / 0 |
| `random_er` | 69.09 +/- 0.16 | 64.55 +/- 0.60 | 162 / 255 |
| `balanced_random` | 69.00 +/- 0.19 | 64.61 +/- 0.32 | 0 / 0 |
| `snip` | 68.52 +/- 0.16 | 64.10 +/- 0.34 | 1141 / 1469 |
