#!/usr/bin/env python3
"""
Paper 2: Emergent Connectivity Structure from Bio-Inspired Developmental Pruning

Core question: Does bio-inspired pruning (inverse-square distance × magnitude)
automatically discover different connectivity structures depending on input
bandwidth — and are those emergent structures optimal and transferable?

Design:
  - Same MLP architecture, starts fully connected
  - 4 bandwidth conditions (4×4, 7×7, 14×14, 28×28 MNIST)
  - 4 pruning methods (2×2 factorial: {no spatial, spatial} × {no training, training})
      1. random_prune       — random removal, balanced fan-in (null model)
      2. distance_only      — prune by proximity (α=0.0), no training signal
      3. magnitude_only     — train, prune by magnitude (α=1.0)
      4. bio_inspired       — train, prune by α×magnitude + (1-α)×proximity
  - Progressive developmental schedule: full → 50% → 75% → 90% → 95% → 98%
      with 3-epoch retraining between each pruning stage.  This lets weight
      magnitudes differentiate across intermediate sparsities so the training
      signal has real leverage at the aggressive final prune — a fairer test
      of whether the method's TOPOLOGY (not just its initialisation) adapts
      to the data.
  - Graph metrics measured at each stage
  - Transfer: cross-bandwidth and cross-category with topology+weights and topology-only
  - 3 seeds

Usage:
    python paper2_run.py                          # Full experiment
    python paper2_run.py --quick                  # Smoke test
    python paper2_run.py --phase main             # Main experiment only
    python paper2_run.py --phase transfer         # Transfer only (needs main results)

Self-contained — only requires torch, torchvision, numpy.
"""

import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from collections import OrderedDict


# =============================================================================
# Graph metrics — measure the emergent connectivity structure
# =============================================================================

def compute_graph_metrics(mask):
    """Compute connectivity metrics from a binary mask (n_out × n_in).

    Always reports 1D-circular effective locality as the primary metric for
    backward compatibility.  When the input has natural 2D structure (n_in == 784),
    additionally reports 2D-pixel locality so 1D-pruned and 2D-pruned topologies
    can be compared on a common scale.

    Returns dict with:
        - effective_locality: mean normalised 1D-circular distance of active
          connections (0 = nearest neighbours; 1 = maximally distant)
        - effective_locality_2d: same but using 2D pixel embedding
          (only present when n_in == 784)
        - clustering_coefficient: average local clustering (bipartite projection,
          embedding-independent)
        - fan_in_mean / fan_in_std: fan-in statistics
        - fan_out_mean / fan_out_std: fan-out statistics
        - orphaned_inputs: count of inputs with zero connections
        - dead_outputs: count of outputs with zero connections
        - distance_distribution: 1D-circular histogram (10 bins)
        - distance_distribution_2d: 2D-pixel histogram (only when n_in == 784)
        - density: actual density of the mask
    """
    mask = np.asarray(mask, dtype=np.float32)
    n_out, n_in = mask.shape

    fan_in = mask.sum(axis=0)
    fan_out = mask.sum(axis=1)

    # Primary 1D-circular metric (always)
    dist_1d = _compute_distance_matrix(n_out, n_in, scheme="1d_circular")
    active = mask > 0
    if active.sum() > 0:
        d1 = dist_1d[active]
        d_max_1 = dist_1d.max()
        effective_locality = float(np.mean(d1) / d_max_1) if d_max_1 > 0 else 0.0
        hist1, _ = np.histogram(d1, bins=10, range=(0, d_max_1))
        distance_distribution = (hist1 / hist1.sum()).tolist()
    else:
        effective_locality = 0.0
        distance_distribution = [0.0] * 10

    # Optional 2D-pixel metric for input layer
    extras = {}
    if n_in == 784:
        dist_2d = _compute_distance_matrix(n_out, n_in, scheme="2d_pixel")
        if active.sum() > 0:
            d2 = dist_2d[active]
            d_max_2 = dist_2d.max()
            extras["effective_locality_2d"] = float(np.mean(d2) / d_max_2) if d_max_2 > 0 else 0.0
            hist2, _ = np.histogram(d2, bins=10, range=(0, d_max_2))
            extras["distance_distribution_2d"] = (hist2 / hist2.sum()).tolist()
        else:
            extras["effective_locality_2d"] = 0.0
            extras["distance_distribution_2d"] = [0.0] * 10

    # Clustering coefficient via bipartite projection
    # Project onto input-input co-occurrence: C = M^T @ M
    co_occur = mask.T @ mask  # (n_in, n_in)
    np.fill_diagonal(co_occur, 0)
    degrees = (co_occur > 0).sum(axis=1)

    clustering_vals = []
    for i in range(n_in):
        neighbours = np.where(co_occur[i] > 0)[0]
        k = len(neighbours)
        if k < 2:
            continue
        sub = co_occur[np.ix_(neighbours, neighbours)]
        n_edges = (sub > 0).sum() / 2
        clustering_vals.append(n_edges / (k * (k - 1) / 2))

    clustering = float(np.mean(clustering_vals)) if clustering_vals else 0.0

    out = {
        "effective_locality": effective_locality,
        "clustering_coefficient": clustering,
        "fan_in_mean": float(fan_in.mean()),
        "fan_in_std": float(fan_in.std()),
        "fan_out_mean": float(fan_out.mean()),
        "fan_out_std": float(fan_out.std()),
        "orphaned_inputs": int((fan_in == 0).sum()),
        "dead_outputs": int((fan_out == 0).sum()),
        "distance_distribution": distance_distribution,
        "density": float(mask.mean()),
        "total_connections": int(mask.sum()),
    }
    out.update(extras)
    return out


# =============================================================================
# Model: sparse MLP
# =============================================================================

class SparseLinear(nn.Module):
    """Linear layer with a fixed binary sparsity mask."""

    def __init__(self, in_features, out_features, mask=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        if mask is not None:
            self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        else:
            self.register_buffer("mask", torch.ones(out_features, in_features))

        # Kaiming init using actual fan-in from mask
        fan_in = max(float(self.mask.sum(axis=1).mean()), 1.0)
        std = np.sqrt(2.0 / fan_in)
        nn.init.normal_(self.weight, 0, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        with torch.no_grad():
            self.weight.data *= self.mask

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)

    def enforce_mask(self):
        with torch.no_grad():
            self.weight.data *= self.mask

    def set_mask(self, new_mask):
        self.mask.copy_(torch.tensor(new_mask, dtype=torch.float32))
        self.enforce_mask()


class SparseMLP(nn.Module):
    """MLP with sparse connectivity."""

    def __init__(self, layer_sizes, masks=None):
        super().__init__()
        self.layer_sizes = layer_sizes
        n_layers = len(layer_sizes) - 1
        if masks is None:
            masks = [None] * n_layers

        layers = []
        for i in range(n_layers):
            layers.append(SparseLinear(
                layer_sizes[i], layer_sizes[i + 1], mask=masks[i]
            ))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_sparse_layers(self):
        return [m for m in self.modules() if isinstance(m, SparseLinear)]

    def enforce_masks(self):
        for layer in self.get_sparse_layers():
            layer.enforce_mask()

    def reinit_weights(self):
        """Reinitialise weights using Kaiming with actual fan-in from masks."""
        for layer in self.get_sparse_layers():
            fan_in = max(float(layer.mask.sum(axis=1).mean()), 1.0)
            std = np.sqrt(2.0 / fan_in)
            nn.init.normal_(layer.weight, 0, std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            layer.enforce_mask()


# =============================================================================
# Data loading with bandwidth control
# =============================================================================

def load_mnist():
    """Load MNIST, return numpy arrays (normalised)."""
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST("data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST("data", train=False, download=True, transform=transform)

    X_train = train_data.data.float().view(-1, 784).numpy() / 255.0
    y_train = train_data.targets.numpy()
    X_test = test_data.data.float().view(-1, 784).numpy() / 255.0
    y_test = test_data.targets.numpy()

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def downsample_images(X, target_side):
    """Downsample 28×28 to target resolution then upsample back to 784.
    Controls effective bandwidth while keeping input dimension fixed.
    """
    if target_side >= 28:
        return X
    n = len(X)
    imgs = X.reshape(n, 28, 28)
    t = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
    small = F.interpolate(t, size=target_side, mode='bilinear', align_corners=False)
    big = F.interpolate(small, size=28, mode='bilinear', align_corners=False)
    return big.squeeze(1).numpy().reshape(n, 784)


def make_dataloaders(X_train, y_train, X_test, y_test, bandwidth_side,
                     class_subset=None, seed=42, batch_size=128):
    """Create train/val/test loaders with bandwidth control and optional class subset."""
    rng = np.random.RandomState(seed)

    if class_subset is not None:
        train_mask = np.isin(y_train, class_subset)
        test_mask = np.isin(y_test, class_subset)
        Xtr, ytr = X_train[train_mask], y_train[train_mask]
        Xte, yte = X_test[test_mask], y_test[test_mask]
        label_map = {c: i for i, c in enumerate(sorted(class_subset))}
        ytr = np.array([label_map[c] for c in ytr])
        yte = np.array([label_map[c] for c in yte])
    else:
        Xtr, ytr, Xte, yte = X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy()

    Xtr = downsample_images(Xtr, bandwidth_side)
    Xte = downsample_images(Xte, bandwidth_side)

    n_val = int(0.1 * len(Xtr))
    idx = rng.permutation(len(Xtr))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr[train_idx], dtype=torch.float32),
        torch.tensor(ytr[train_idx], dtype=torch.long)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr[val_idx], dtype=torch.float32),
        torch.tensor(ytr[val_idx], dtype=torch.long)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.tensor(Xte, dtype=torch.float32),
        torch.tensor(yte, dtype=torch.long)
    )

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


# =============================================================================
# Position embeddings and distance matrices
# =============================================================================
#
# Two embedding schemes:
#   "1d_circular" — original, treats each layer as a circular 1D channel arc
#   "2d_pixel"    — 2D embedding for layers whose input has natural 2D structure
#                   (n_in == 784 → 28×28 pixel grid).  Outputs are tiled in a
#                   sqrt grid in the same [0,1]² space.  For layers where n_in
#                   is not 784, falls back to 1d_circular.
#
# 2D scheme is meant to test whether the bandwidth-locality dependence reported
# in the v1 results is partly an artefact of the 1D row-major flattening (which
# misaligns with the 2D pixel topology of MNIST images).

def _compute_distance_matrix(n_out, n_in, scheme="1d_circular"):
    """Return a (n_out, n_in) distance matrix in [0, 1]-ish range.

    1d_circular: positions on a unit circle, distance is min(|d|, 1-|d|).
    2d_pixel:    only for n_in == 784. Inputs at (i//28, i%28)/27.0; outputs
                 tiled in the smallest sqrt-grid covering n_out, normalised to
                 [0, 1]^2. Euclidean distance.  Falls back to 1d_circular
                 otherwise.
    """
    if scheme == "2d_pixel" and n_in == 784:
        in_y = (np.arange(n_in) // 28) / 27.0
        in_x = (np.arange(n_in) % 28) / 27.0
        side = int(np.ceil(np.sqrt(n_out)))
        norm = max(side - 1, 1)
        out_y = (np.arange(n_out) // side) / norm
        out_x = (np.arange(n_out) % side) / norm
        dy = out_y[:, None] - in_y[None, :]
        dx = out_x[:, None] - in_x[None, :]
        return np.sqrt(dy * dy + dx * dx)
    # default: 1D circular
    out_pos = np.linspace(0, 1, n_out)
    in_pos = np.linspace(0, 1, n_in)
    dist = np.abs(out_pos[:, None] - in_pos[None, :])
    return np.minimum(dist, 1.0 - dist)


def _proximity_from_distance(dist_matrix):
    """Convert distance matrix → proximity in [0, 1] (1 = closest)."""
    d_max = dist_matrix.max()
    if d_max <= 0:
        return np.zeros_like(dist_matrix)
    return 1.0 - dist_matrix / d_max


# =============================================================================
# Pruning methods — the 2×2 factorial (plus 2D embedding variants)
# =============================================================================

def _repair_orphans(mask, rng):
    """In-place orphan repair: ensure every input feeds at least one output."""
    in_usage = mask.sum(axis=0)
    orphaned = np.where(in_usage == 0)[0]
    for orphan_in in orphaned:
        most_used = np.argmax(in_usage)
        if in_usage[most_used] <= 1:
            break
        users = np.where(mask[:, most_used] > 0)[0]
        swap_out = rng.choice(users)
        mask[swap_out, most_used] = 0.0
        mask[swap_out, orphan_in] = 1.0
        in_usage[most_used] -= 1
        in_usage[orphan_in] += 1


def prune_random(mask, target_density, seed=42):
    """Random pruning with balanced fan-in. No data, no distance."""
    rng = np.random.RandomState(seed)
    n_out, n_in = mask.shape
    n_keep = max(1, int(target_density * n_out * n_in))
    k_per_out = n_keep // n_out
    remainder = n_keep - k_per_out * n_out

    new_mask = np.zeros_like(mask)
    for o in range(n_out):
        k = k_per_out + (1 if o < remainder else 0)
        active_inputs = np.where(mask[o] > 0)[0]
        if len(active_inputs) <= k:
            new_mask[o, active_inputs] = 1.0
        else:
            chosen = rng.choice(active_inputs, k, replace=False)
            new_mask[o, chosen] = 1.0

    _repair_orphans(new_mask, rng)
    return new_mask


def prune_distance_only(mask, target_density, seed=42, dist_scheme="1d_circular"):
    """Prune by proximity only (α=0.0). No training signal."""
    n_out, n_in = mask.shape
    rng = np.random.RandomState(seed)

    dist_matrix = _compute_distance_matrix(n_out, n_in, scheme=dist_scheme)
    proximity = _proximity_from_distance(dist_matrix)

    n_keep = max(1, int(target_density * n_out * n_in))
    k_per_out = n_keep // n_out
    remainder = n_keep - k_per_out * n_out

    new_mask = np.zeros_like(mask)
    for o in range(n_out):
        k = k_per_out + (1 if o < remainder else 0)
        k = min(k, n_in)
        active_inputs = np.where(mask[o] > 0)[0]
        if len(active_inputs) <= k:
            new_mask[o, active_inputs] = 1.0
        else:
            scores = proximity[o, active_inputs]
            top_idx = active_inputs[np.argpartition(scores, -k)[-k:]]
            new_mask[o, top_idx] = 1.0

    _repair_orphans(new_mask, rng)
    return new_mask


def prune_magnitude_only(mask, weight_matrix, target_density, seed=42):
    """Prune by weight magnitude only (α=1.0). Training signal, no distance."""
    n_out, n_in = mask.shape
    rng = np.random.RandomState(seed)

    weight_mag = np.abs(weight_matrix)
    mag_min = weight_mag.min(axis=1, keepdims=True)
    mag_max = weight_mag.max(axis=1, keepdims=True)
    mag_range = mag_max - mag_min
    mag_range[mag_range == 0] = 1.0
    mag_norm = (weight_mag - mag_min) / mag_range

    n_keep = max(1, int(target_density * n_out * n_in))
    k_per_out = n_keep // n_out
    remainder = n_keep - k_per_out * n_out

    new_mask = np.zeros_like(mask)
    for o in range(n_out):
        k = k_per_out + (1 if o < remainder else 0)
        k = min(k, n_in)
        active_inputs = np.where(mask[o] > 0)[0]
        if len(active_inputs) <= k:
            new_mask[o, active_inputs] = 1.0
        else:
            scores = mag_norm[o, active_inputs]
            top_idx = active_inputs[np.argpartition(scores, -k)[-k:]]
            new_mask[o, top_idx] = 1.0

    _repair_orphans(new_mask, rng)
    return new_mask


def prune_bio_inspired(mask, weight_matrix, target_density, alpha=0.5, seed=42,
                       dist_scheme="1d_circular"):
    """Prune by blended magnitude × proximity (α=0.5). Full bio-inspired method."""
    n_out, n_in = mask.shape
    rng = np.random.RandomState(seed)

    # Score 1: weight magnitude (normalised per output)
    weight_mag = np.abs(weight_matrix)
    mag_min = weight_mag.min(axis=1, keepdims=True)
    mag_max = weight_mag.max(axis=1, keepdims=True)
    mag_range = mag_max - mag_min
    mag_range[mag_range == 0] = 1.0
    mag_norm = (weight_mag - mag_min) / mag_range

    # Score 2: proximity (1 - normalised distance under chosen scheme)
    dist_matrix = _compute_distance_matrix(n_out, n_in, scheme=dist_scheme)
    proximity = _proximity_from_distance(dist_matrix)

    score = alpha * mag_norm + (1 - alpha) * proximity

    n_keep = max(1, int(target_density * n_out * n_in))
    k_per_out = n_keep // n_out
    remainder = n_keep - k_per_out * n_out

    new_mask = np.zeros_like(mask)
    for o in range(n_out):
        k = k_per_out + (1 if o < remainder else 0)
        k = min(k, n_in)
        active_inputs = np.where(mask[o] > 0)[0]
        if len(active_inputs) <= k:
            new_mask[o, active_inputs] = 1.0
        else:
            scores = score[o, active_inputs]
            top_idx = active_inputs[np.argpartition(scores, -k)[-k:]]
            new_mask[o, top_idx] = 1.0

    _repair_orphans(new_mask, rng)
    return new_mask


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=20, lr=0.001):
    """Train a sparse MLP. Returns test accuracy."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            model.enforce_masks()

    return evaluate(model, test_loader, device)


def train_discovery(model, train_loader, device, epochs=3, lr=0.001):
    """Short training for weight development (discovery phase)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            model.enforce_masks()


def evaluate(model, loader, device):
    """Evaluate accuracy."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
    return correct / total


# =============================================================================
# Main experiment
# =============================================================================

PRUNING_METHODS = ["random_prune", "distance_only", "magnitude_only", "bio_inspired"]
PRUNING_METHODS_2D = ["distance_only_2d", "bio_inspired_2d"]  # only added when --include_2d_variants
BANDWIDTHS = [4, 7, 14, 28]      # pixels per side
# Progressive developmental pruning schedule: full → 50% → 75% → 90% → 95% → 98%.
# Retraining between stages lets magnitudes differentiate so the training signal
# has real leverage at the aggressive final prune, making topology emerge rather
# than being dictated by the proximity term at moderate sparsity.
SPARSITY_STAGES = [0.50, 0.75, 0.90, 0.95, 0.98]
DISCOVERY_EPOCHS = 3        # initial exuberant training before first prune
RETRAIN_EPOCHS = 3          # retrain between each intermediate prune stage
TRAINING_EPOCHS = 20        # final training at target sparsity


def _apply_pruning(method, current_mask, weight_np, target_density, seed):
    """Dispatch to the right pruning rule for a given method.

    Methods ending in `_2d` use the 2D pixel embedding for layers with n_in==784;
    other layers fall back to 1D circular automatically (handled inside
    _compute_distance_matrix)."""
    if method == "random_prune":
        return prune_random(current_mask, target_density, seed=seed)
    elif method == "distance_only":
        return prune_distance_only(current_mask, target_density, seed=seed,
                                   dist_scheme="1d_circular")
    elif method == "distance_only_2d":
        return prune_distance_only(current_mask, target_density, seed=seed,
                                   dist_scheme="2d_pixel")
    elif method == "magnitude_only":
        return prune_magnitude_only(current_mask, weight_np, target_density, seed=seed)
    elif method == "bio_inspired":
        return prune_bio_inspired(current_mask, weight_np, target_density,
                                  alpha=0.5, seed=seed, dist_scheme="1d_circular")
    elif method == "bio_inspired_2d":
        return prune_bio_inspired(current_mask, weight_np, target_density,
                                  alpha=0.5, seed=seed, dist_scheme="2d_pixel")
    else:
        raise ValueError(f"Unknown method: {method}")


def run_main_experiment(hidden_size=256, n_seeds=3, device="cpu",
                        results_dir=None, quick=False,
                        hidden_sizes=None, include_2d_variants=False):
    """Main experiment with PROGRESSIVE pruning: 4 methods × 4 bandwidths × 3 seeds.

    All start fully connected. Trained methods get DISCOVERY_EPOCHS of initial
    training before the first prune. Then the network is progressively pruned
    through SPARSITY_STAGES, with RETRAIN_EPOCHS of training between each
    intermediate prune (retrain omitted for untrained methods random_prune
    and distance_only). Graph metrics are recorded at each stage. After the
    final stage, weights are reinitialised (Kaiming using the pruned fan-in)
    and the network is trained to completion at the final sparsity.

    The progressive schedule is the key change from the single-shot version:
    at each intermediate sparsity, magnitudes have room to differentiate
    before the next prune, so the training signal has leverage even at
    aggressive final sparsities.
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
    os.makedirs(results_dir, exist_ok=True)

    bandwidths = [4, 28] if quick else BANDWIDTHS
    methods = list(PRUNING_METHODS)
    if include_2d_variants:
        methods = methods + list(PRUNING_METHODS_2D)
    stages = [0.50, 0.95] if quick else SPARSITY_STAGES
    n_seeds_run = 1 if quick else n_seeds

    # Hidden-size sweep support: if hidden_sizes is given, loop over all of them.
    # Otherwise use the single hidden_size argument (backward-compatible default).
    if hidden_sizes is None:
        hidden_sizes = [hidden_size]
    elif quick:
        hidden_sizes = hidden_sizes[:1]  # quick mode: just the first size

    outpath = os.path.join(results_dir, "main_experiment.json")

    # Resume support
    if os.path.exists(outpath):
        with open(outpath) as f:
            results = json.load(f)
        print(f"Loaded existing results from {outpath}")
    else:
        results = {}

    print(f"\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    total = len(hidden_sizes) * len(bandwidths) * len(methods) * n_seeds_run
    done = 0

    print(f"Grid: {len(hidden_sizes)} sizes × {len(bandwidths)} bw × "
          f"{len(methods)} methods × {n_seeds_run} seeds = {total} cells")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Methods: {methods}")
    print(f"Progressive schedule: {stages}")
    print(f"Retrain epochs between stages: {RETRAIN_EPOCHS}")
    print(f"Device: {device}")

    for H in hidden_sizes:
        layer_sizes = [784, H, 10]
        for bw in bandwidths:
            train_loader, val_loader, test_loader = make_dataloaders(
                X_train, y_train, X_test, y_test,
                bandwidth_side=bw, seed=42
            )

            for method in methods:
                for seed in range(n_seeds_run):
                    # Cell key includes hidden size when sweeping. We retain the
                    # legacy "bw{bw}_{method}_seed{seed}" form for H==256 so old
                    # results in the same dir continue to resume cleanly.
                    if len(hidden_sizes) == 1 and H == 256:
                        cell_key = f"bw{bw}_{method}_seed{seed}"
                    else:
                        cell_key = f"h{H}_bw{bw}_{method}_seed{seed}"

                    if cell_key in results:
                        print(f"[SKIP] {cell_key}")
                        done += 1
                        continue

                    print(f"\n{'='*60}")
                    print(f"  H={H}  BW={bw}x{bw}  METHOD={method}  SEED={seed}")
                    print(f"{'='*60}")

                    torch.manual_seed(seed * 1000)
                    np.random.seed(seed * 1000)

                    # Start fully connected
                    model = SparseMLP(layer_sizes)
                    model = model.to(device)

                    # Initial discovery phase for trained methods
                    # (any method that uses weight magnitude in its score)
                    needs_training = method in (
                        "magnitude_only", "bio_inspired", "bio_inspired_2d"
                    )
                    if needs_training:
                        print(f"  Initial discovery: {DISCOVERY_EPOCHS} epochs...")
                        train_discovery(model, train_loader, device,
                                        epochs=DISCOVERY_EPOCHS)

                    # Progressive pruning with retraining between stages
                    stage_results = {}
                    layers = model.get_sparse_layers()

                    for stage_idx, target_sparsity in enumerate(stages):
                        target_density = 1.0 - target_sparsity

                        all_metrics = []
                        for layer in layers:
                            current_mask = layer.mask.cpu().numpy()
                            weight_np = layer.weight.data.cpu().numpy()
                            new_mask = _apply_pruning(
                                method, current_mask, weight_np,
                                target_density, seed=seed*1000
                            )
                            layer.set_mask(new_mask)
                            metrics = compute_graph_metrics(new_mask)
                            all_metrics.append(metrics)

                        stage_label = f"s{int(target_sparsity*100)}"
                        stage_results[stage_label] = {
                            "layer_metrics": all_metrics,
                            "target_sparsity": target_sparsity,
                        }
                        print(f"  Stage {stage_label}: "
                              f"locality={all_metrics[0]['effective_locality']:.3f}, "
                              f"clustering={all_metrics[0]['clustering_coefficient']:.3f}, "
                              f"orphans={all_metrics[0]['orphaned_inputs']}")

                        # Retrain between stages (skip after final stage — final
                        # training happens after reinit). Only trained methods
                        # retrain.
                        is_final_stage = (stage_idx == len(stages) - 1)
                        if needs_training and not is_final_stage:
                            print(f"    Retrain: {RETRAIN_EPOCHS} epochs at sparsity {target_sparsity}...")
                            train_discovery(model, train_loader, device,
                                            epochs=RETRAIN_EPOCHS)

                    # Save discovered masks and weights BEFORE reinit
                    discovered_masks = [l.mask.cpu().numpy() for l in layers]
                    discovered_weights = [l.weight.data.cpu().numpy() for l in layers]

                    # Reinitialise weights, train to completion at final sparsity
                    model.reinit_weights()
                    print(f"  Training {TRAINING_EPOCHS} epochs at final sparsity {stages[-1]}...")

                    final_acc = train_model(
                        model, train_loader, val_loader, test_loader,
                        device=device, epochs=TRAINING_EPOCHS
                    )
                    print(f"  Final accuracy: {final_acc:.4f}")

                    # Save model for transfer
                    model_path = os.path.join(results_dir, f"{cell_key}_model.pt")
                    torch.save({
                        "model_state": model.state_dict(),
                        "discovered_masks": [m.tolist() for m in discovered_masks],
                        "discovered_weights": [w.tolist() for w in discovered_weights],
                        "layer_sizes": layer_sizes,
                    }, model_path)

                    results[cell_key] = {
                        "stages": stage_results,
                        "final_accuracy": final_acc,
                        "bandwidth": bw,
                        "method": method,
                        "seed": seed,
                        "hidden_size": H,
                    }

                    with open(outpath, "w") as f:
                        json.dump(results, f, indent=2)

                    done += 1
                    print(f"  [SAVED] {cell_key} ({done}/{total})")

    print(f"\nMain experiment complete. Results: {outpath}")
    return results


# =============================================================================
# Transfer experiment
# =============================================================================

def run_transfer_experiment(hidden_size=256, n_seeds=3, device="cpu",
                            results_dir=None, quick=False):
    """Transfer experiment: cross-bandwidth and cross-category.

    For each {source_bandwidth, method}, discover topology on digits 0-4,
    then transfer to digits 5-9 at each target_bandwidth.

    Three transfer modes:
      - topology_and_weights: transfer mask + 3-epoch discovery weights
      - topology_only: transfer mask, reinitialise weights
      - direct: no transfer, fresh discovery+prune on target
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_v2")
    os.makedirs(results_dir, exist_ok=True)

    outpath = os.path.join(results_dir, "transfer_experiment.json")

    if os.path.exists(outpath):
        with open(outpath) as f:
            transfer_results = json.load(f)
        print(f"Loaded existing transfer results")
    else:
        transfer_results = {}

    print(f"\nLoading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    source_classes = list(range(0, 5))
    target_classes = list(range(5, 10))

    bandwidths = [4, 28] if quick else BANDWIDTHS
    methods = ["bio_inspired", "magnitude_only"]  # the two trained methods
    transfer_modes = ["topology_and_weights", "topology_only", "direct"]
    n_seeds_run = 1 if quick else n_seeds

    stages = SPARSITY_STAGES
    target_sparsity = stages[-1]
    target_density = 1.0 - target_sparsity
    n_classes = 5
    layer_sizes = [784, hidden_size, n_classes]

    total = len(bandwidths) * len(methods) * n_seeds_run * len(bandwidths) * len(transfer_modes)
    done = 0

    def _progressive_discover(model_, train_loader_):
        """Run initial discovery, then progressively prune through SPARSITY_STAGES
        with retraining between each stage. Returns the final mask list and the
        final discovery weights (before any reinit)."""
        train_discovery(model_, train_loader_, device, epochs=DISCOVERY_EPOCHS)
        layers_ = model_.get_sparse_layers()
        for stage_idx, ts in enumerate(stages):
            td = 1.0 - ts
            for layer in layers_:
                cm = layer.mask.cpu().numpy()
                wn = layer.weight.data.cpu().numpy()
                nm = _apply_pruning(method, cm, wn, td, seed=seed*1000)
                layer.set_mask(nm)
            is_final_stage = (stage_idx == len(stages) - 1)
            if not is_final_stage:
                train_discovery(model_, train_loader_, device, epochs=RETRAIN_EPOCHS)
        masks_ = [l.mask.cpu().numpy() for l in layers_]
        weights_ = [l.weight.data.cpu().numpy() for l in layers_]
        return masks_, weights_

    for source_bw in bandwidths:
        source_train, source_val, source_test = make_dataloaders(
            X_train, y_train, X_test, y_test,
            bandwidth_side=source_bw, class_subset=source_classes, seed=42
        )

        for method in methods:
            for seed in range(n_seeds_run):
                # === PROGRESSIVE DISCOVERY on source ===
                print(f"\n  Discovery: bw={source_bw}, {method}, seed={seed}")

                torch.manual_seed(seed * 1000)
                np.random.seed(seed * 1000)

                model = SparseMLP(layer_sizes)
                model = model.to(device)
                source_masks, source_weights = _progressive_discover(model, source_train)
                source_metrics = [compute_graph_metrics(m) for m in source_masks]

                # === TRANSFER to each target bandwidth ===
                for target_bw in bandwidths:
                    target_train, target_val, target_test = make_dataloaders(
                        X_train, y_train, X_test, y_test,
                        bandwidth_side=target_bw, class_subset=target_classes,
                        seed=seed*1000
                    )

                    for mode in transfer_modes:
                        cell_key = (f"src{source_bw}_{method}_s{seed}"
                                    f"_tgt{target_bw}_{mode}")

                        if cell_key in transfer_results:
                            done += 1
                            continue

                        torch.manual_seed(seed * 2000 + target_bw)

                        if mode == "topology_and_weights":
                            t_model = SparseMLP(layer_sizes, masks=source_masks)
                            t_model = t_model.to(device)
                            for i, layer in enumerate(t_model.get_sparse_layers()):
                                layer.weight.data = torch.tensor(
                                    source_weights[i], dtype=torch.float32
                                ).to(device)
                                layer.enforce_mask()

                        elif mode == "topology_only":
                            t_model = SparseMLP(layer_sizes, masks=source_masks)
                            t_model = t_model.to(device)

                        elif mode == "direct":
                            # Fresh progressive discovery + prune on target data,
                            # matching the source-side schedule for fair comparison.
                            t_model = SparseMLP(layer_sizes)
                            t_model = t_model.to(device)
                            _progressive_discover(t_model, target_train)
                            t_model.reinit_weights()

                        acc = train_model(
                            t_model, target_train, target_val, target_test,
                            device=device, epochs=TRAINING_EPOCHS
                        )

                        final_metrics = [compute_graph_metrics(l.mask.cpu().numpy())
                                         for l in t_model.get_sparse_layers()]

                        transfer_results[cell_key] = {
                            "accuracy": acc,
                            "source_bw": source_bw,
                            "target_bw": target_bw,
                            "method": method,
                            "mode": mode,
                            "seed": seed,
                            "source_metrics": source_metrics,
                            "target_metrics": final_metrics,
                        }

                        done += 1
                        print(f"    {cell_key}: {acc:.4f} ({done}/{total})")

                        with open(outpath, "w") as f:
                            json.dump(transfer_results, f, indent=2)

    print(f"\nTransfer experiment complete. Results: {outpath}")
    return transfer_results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper 2: Emergent Connectivity from Bio-Inspired Pruning"
    )
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory. Default: results_progressive/")
    parser.add_argument("--phase", choices=["main", "transfer", "all"], default="all")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated sparsity schedule, e.g. '0.5,0.75,0.9,0.95,0.98'. "
                             "Overrides SPARSITY_STAGES.")
    parser.add_argument("--retrain_epochs", type=int, default=None,
                        help="Epochs of retraining between stages. Overrides RETRAIN_EPOCHS.")
    parser.add_argument("--hidden_sizes", type=str, default=None,
                        help="Comma-separated list of hidden sizes for the network-size sweep, "
                             "e.g. '64,128,256,512'. Overrides --hidden_size.")
    parser.add_argument("--include_2d_variants", action="store_true",
                        help="Add distance_only_2d and bio_inspired_2d methods (only affects the "
                             "input layer where n_in=784; other layers fall back to 1D circular).")
    args = parser.parse_args()

    if args.stages is not None:
        SPARSITY_STAGES = [float(x) for x in args.stages.split(",")]
        print(f"Using custom sparsity schedule: {SPARSITY_STAGES}")
    if args.retrain_epochs is not None:
        RETRAIN_EPOCHS = args.retrain_epochs
        print(f"Using custom retrain epochs: {RETRAIN_EPOCHS}")

    hidden_sizes_list = None
    if args.hidden_sizes is not None:
        hidden_sizes_list = [int(x) for x in args.hidden_sizes.split(",")]
        print(f"Hidden-size sweep: {hidden_sizes_list}")
    if args.include_2d_variants:
        print(f"Including 2D embedding variants: {PRUNING_METHODS_2D}")

    if args.results_dir is None:
        args.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results_progressive"
        )

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Paper 2 — Emergent Connectivity from Bio-Inspired Pruning")
    print(f"Device: {device}, Phase: {args.phase}")

    if args.phase in ("main", "all"):
        run_main_experiment(
            hidden_size=args.hidden_size, n_seeds=args.seeds,
            device=device, results_dir=args.results_dir, quick=args.quick,
            hidden_sizes=hidden_sizes_list,
            include_2d_variants=args.include_2d_variants,
        )

    if args.phase in ("transfer", "all"):
        run_transfer_experiment(
            hidden_size=args.hidden_size, n_seeds=args.seeds,
            device=device, results_dir=args.results_dir, quick=args.quick,
        )
