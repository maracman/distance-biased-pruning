# Paper B: MNIST Topology Characterization

Reproduction code for **Emergent Connectivity Structure in Inverse-Square Distance Pruning: Bandwidth, Capacity, and Embedding Geometry**.

The main script characterizes topology across bandwidth, hidden size, pruning method, and embedding geometry using MNIST MLPs at 98% sparsity.

## Commands

Full reproduction:

```bash
bash reproduce.sh
```

Quick smoke test:

```bash
python train.py --quick --phase main --device auto
```

Main experiment only:

```bash
python train.py \
  --phase main \
  --seeds 3 \
  --hidden_sizes 64,128,256,512 \
  --include_2d_variants \
  --device auto
```

The secondary `bio_developmental_comparison.py` script runs the checkpointed 90% sparsity comparison between two-phase bio-developmental topology and lambda-mixture baselines:

```bash
python bio_developmental_comparison.py --quick --device auto
```

Results are written under `paper_b_topology_mnist/results_progressive/` or `paper_b_topology_mnist/results/` depending on the script.

## Reported Accuracy Table

Final test accuracy at 98% sparsity, mean percent over 3 seeds:

| Method | H | bw=4 | bw=7 | bw=14 | bw=28 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `random_prune` | 256 | 46.00 | 84.54 | 88.84 | 81.17 |
| `random_prune` | 512 | 46.49 | 86.94 | 92.74 | 88.79 |
| `magnitude_only` | 256 | 45.58 | 84.77 | 92.68 | 89.88 |
| `magnitude_only` | 512 | 46.89 | 88.20 | 95.11 | 93.82 |
| `bio_inspired` | 256 | 44.72 | 81.64 | 90.10 | 84.87 |
| `bio_inspired` | 512 | 46.63 | 86.87 | 94.18 | 92.09 |
| `bio_inspired_2d` | 256 | 45.57 | 84.31 | 91.96 | 89.22 |
| `bio_inspired_2d` | 512 | 46.78 | 88.67 | 95.02 | 93.40 |

## Reported Clustering Range

Clustering coefficient range across bandwidths:

| Method | H=64 | H=128 | H=256 | H=512 |
| --- | ---: | ---: | ---: | ---: |
| `random_prune` | 0.000 | 0.000 | 0.000 | 0.000 |
| `distance_only` | 0.000 | 0.000 | 0.000 | 0.000 |
| `magnitude_only` | 0.002 | 0.025 | 0.074 | 0.105 |
| `bio_inspired` | 0.001 | 0.018 | 0.067 | 0.113 |
| `bio_inspired_2d` | 0.003 | 0.017 | 0.078 | 0.148 |

