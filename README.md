# Distance-Biased Pruning

Code for two papers on inverse-square distance priors for sparse neural network pruning.

## Papers

1. **Inverse-Square Distance Priors Improve Network Pruning at Extreme Sparsity**  
   Preprint forthcoming.  
   ResNet-18 on CIFAR-100, testing distance-biased allocation and distance x magnitude pruning at 98% and 99% sparsity.

2. **Emergent Connectivity Structure in Inverse-Square Distance Pruning: Bandwidth, Capacity, and Embedding Geometry**  
   Preprint forthcoming.  
   MNIST single-hidden-layer MLP experiments characterizing how bandwidth, hidden size, and embedding geometry affect the topology produced by the method.

Paper A's reported results are CIFAR-100 runs. The training CLI still supports CIFAR-10 quick/development runs, but those outputs are not the reported Paper A experiments. Paper B's main results use downsample-then-upsample MNIST bandwidth control at 98% sparsity; the separate `bio_developmental_comparison.py` script uses patch-sampled MNIST at 90% sparsity and should be treated as a secondary comparison.

## Code Availability

This repository contains code for reproducing the experiments reported in both papers.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiment scripts download CIFAR and MNIST through `torchvision` into ignored `data/` directories.

## Reproduce

Run tests first:

```bash
python -m pytest tests/ -v
```

Run the full Paper A pipeline:

```bash
bash paper_a_cifar_resnet/reproduce.sh
```

Run the full Paper B pipeline:

```bash
bash paper_b_topology_mnist/reproduce.sh
```

Quick smoke tests:

```bash
python paper_a_cifar_resnet/train.py --quick --dataset cifar100 --device auto
python paper_b_topology_mnist/train.py --quick --phase main --device auto
python paper_b_topology_mnist/bio_developmental_comparison.py --quick --device auto
```

## Layout

```text
distance-biased-pruning/
├── paper_a_cifar_resnet/      # Paper A ResNet-18 / CIFAR experiments
├── paper_b_topology_mnist/    # Paper B MNIST single-hidden-layer MLP experiments
├── shared/                    # Shared topology, model, training, and analysis utilities
└── tests/                     # Unit tests for topology generation, pruning, and metrics
```

## Reported Results

Paper A main result, CIFAR-100 ResNet-18, mean accuracy over 3 seeds:

| Condition | 98% sparsity | 99% sparsity | Orphans/dead at 99%, seed 42 |
| --- | ---: | ---: | ---: |
| `distance_quick` | 69.74 +/- 0.03 | 65.95 +/- 0.20 | 0 / 0 |
| `bio_quick` | 69.64 +/- 0.20 | 65.79 +/- 0.26 | 0 / 0 |
| `distance_balanced` | 69.22 +/- 0.30 | 65.08 +/- 0.24 | 0 / 0 |
| `random_er` | 69.09 +/- 0.16 | 64.55 +/- 0.60 | 162 / 255 |
| `bio_inspired` | 69.00 +/- 0.19 | 64.61 +/- 0.32 | 0 / 0 |
| `snip` | 68.52 +/- 0.16 | 64.10 +/- 0.34 | 1141 / 1469 |

Paper B key characterization result, clustering range across bandwidths:

| Method | H=64 | H=128 | H=256 | H=512 |
| --- | ---: | ---: | ---: | ---: |
| `random_prune` | 0.000 | 0.000 | 0.000 | 0.000 |
| `distance_only` | 0.000 | 0.000 | 0.000 | 0.000 |
| `distance_only_2d` | 0.000 | 0.000 | 0.000 | 0.000 |
| `magnitude_only` | 0.002 | 0.025 | 0.074 | 0.105 |
| `bio_inspired` | 0.001 | 0.018 | 0.067 | 0.113 |
| `bio_inspired_2d` | 0.003 | 0.017 | 0.078 | 0.148 |

## Citation

```bibtex
@misc{anderson2026inverse_square_pruning,
  author = {Anderson, Marcus},
  title = {Inverse-Square Distance Priors Improve Network Pruning at Extreme Sparsity},
  year = {2026},
  note = {Manuscript in preparation}
}

@misc{anderson2026emergent_connectivity,
  author = {Anderson, Marcus},
  title = {Emergent Connectivity Structure in Inverse-Square Distance Pruning: Bandwidth, Capacity, and Embedding Geometry},
  year = {2026},
  note = {Manuscript in preparation}
}
```

## License

MIT.
