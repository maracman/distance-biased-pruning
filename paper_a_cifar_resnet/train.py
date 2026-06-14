#!/usr/bin/env python3
"""
Paper A / Paper 1: Inverse-Square Distance Priors Improve Network Pruning at Extreme Sparsity

Experiment driver for CIFAR-adapted ResNet-18 channel-pair pruning.

Manuscript runs use CIFAR-100, target sparsities 98% and 99%, 200 epochs,
and seeds 42, 43, 44. The reported six-condition factorial is:
distance_dev, distance_prior, balanced_dev, balanced_random, random_er, snip.

This script also supports older CIFAR-10 / 90% development and smoke-test
runs via its CLI defaults. Those runs are useful for quick checks but are not
the reported Paper A experiments.

Usage examples:
    python paper_a_cifar_resnet/train.py --dataset cifar100 --sparsity 0.98 --seeds 3
    python paper_a_cifar_resnet/train.py --dataset cifar100 --sparsity 0.99 --seeds 3
    python paper_a_cifar_resnet/train.py --quick

Results saved to: paper_a_cifar_resnet/results/ unless --output_dir is supplied.
"""

import os
import sys
import json
import time
import copy
import csv
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


# =============================================================================
# ResNet-18 with structured sparsity support
# =============================================================================

class MaskedConv2d(nn.Conv2d):
    """Conv2d with a binary mask applied to weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

    def apply_mask(self, mask):
        self.mask = mask.to(self.weight.device)
        self.weight.data *= self.mask

    @property
    def sparsity(self):
        return 1.0 - self.mask.sum().item() / self.mask.numel()


def make_resnet18_sparse(num_classes=10):
    """Create ResNet-18 with MaskedConv2d layers for 32x32 CIFAR inputs."""
    import torchvision.models as models
    model = models.resnet18(weights=None, num_classes=num_classes)

    # CIFAR-style adaptation: 3x3 conv1 with stride 1, no maxpool.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    # Replace all Conv2d in residual blocks with MaskedConv2d
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and name != "conv1":
            replacements.append((name, module))

    for name, module in replacements:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        masked = MaskedConv2d(
            module.in_channels, module.out_channels,
            module.kernel_size, stride=module.stride,
            padding=module.padding, bias=module.bias is not None,
        )
        masked.weight.data = module.weight.data.clone()
        if module.bias is not None:
            masked.bias.data = module.bias.data.clone()
        setattr(parent, parts[-1], masked)

    return model


def get_masked_layers(model):
    """Return list of (name, MaskedConv2d) pairs."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MaskedConv2d)]


# =============================================================================
# ERK density distribution
# =============================================================================

def compute_erk_densities(model, target_density):
    """Per-layer densities using Erdos-Renyi-Kernel (ERK) distribution.

    ERK allocates more connections to layers with fewer parameters,
    producing more uniform information flow across layers.
    """
    layers = get_masked_layers(model)
    erk_scores = []
    total_params = 0
    for name, layer in layers:
        n_in, n_out = layer.in_channels, layer.out_channels
        k = layer.kernel_size[0] * layer.kernel_size[1]
        score = (n_in + n_out + k) / (n_in * n_out * k)
        erk_scores.append(score)
        total_params += layer.weight.numel()

    target_nonzero = target_density * total_params

    # Binary search for scale factor
    lo, hi = 0.0, 1e6
    for _ in range(100):
        mid = (lo + hi) / 2
        nonzero = sum(
            min(1.0, erk_scores[i] * mid) * layers[i][1].weight.numel()
            for i in range(len(layers))
        )
        if nonzero < target_nonzero:
            lo = mid
        else:
            hi = mid

    scale = (lo + hi) / 2
    densities = {}
    for i, (name, layer) in enumerate(layers):
        densities[name] = min(1.0, erk_scores[i] * scale)

    return densities


# =============================================================================
# Mask generation: balanced allocation (no orphans)
# =============================================================================

def generate_balanced_masks(model, target_density, seed=42):
    """Balanced random sparse masks: no dead outputs, no orphaned inputs.

    For each layer:
      1. Each output filter gets exactly k input channels (balanced fan-out→in).
      2. After allocation, verify every input channel is used by at least one
         output. If any input is orphaned, swap it in for a random connection
         from the most-connected input.

    This guarantees:
      - Uniform fan-in across output filters (std ≈ 0)
      - No orphaned input channels (every channel contributes)
      - Exact target sparsity via ERK distribution
    """
    rng = np.random.RandomState(seed)
    densities = compute_erk_densities(model, target_density)
    masks = {}

    for name, layer in get_masked_layers(model):
        density = densities[name]
        n_out = layer.out_channels
        n_in = layer.in_channels
        k_h, k_w = layer.kernel_size

        if density >= 1.0:
            masks[name] = torch.ones_like(layer.weight)
            continue

        # Number of input channels each output filter connects to
        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        k_per_out = n_keep // n_out
        remainder = n_keep - k_per_out * n_out

        # Step 1: balanced random allocation per output
        pair_mask = np.zeros((n_out, n_in), dtype=np.float32)
        for o in range(n_out):
            k = k_per_out + (1 if o < remainder else 0)
            k = min(k, n_in)
            if k > 0:
                chosen = rng.choice(n_in, k, replace=False)
                pair_mask[o, chosen] = 1.0

        # Step 2: fix orphaned inputs (input channels with zero outgoing connections)
        in_usage = pair_mask.sum(axis=0)  # how many outputs each input feeds
        orphaned = np.where(in_usage == 0)[0]

        for orphan_in in orphaned:
            # Find the input channel with the most connections (most redundant)
            most_used_in = np.argmax(in_usage)
            if in_usage[most_used_in] <= 1:
                break  # can't swap without creating a new orphan

            # Find an output that uses most_used_in and swap
            users = np.where(pair_mask[:, most_used_in] > 0)[0]
            swap_out = rng.choice(users)
            pair_mask[swap_out, most_used_in] = 0.0
            pair_mask[swap_out, orphan_in] = 1.0
            in_usage[most_used_in] -= 1
            in_usage[orphan_in] += 1

        # Expand to full weight shape (n_out, n_in, k_h, k_w)
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)

        masks[name] = torch.tensor(full_mask)

    return masks, densities


def generate_distance_balanced_masks(model, target_density, seed=42, distance_exponent=2.0):
    """Distance-biased balanced sparse masks at CHANNEL-PAIR granularity.

    The actual bio-inspired method: channels are embedded in 1D space
    (normalised channel index), and connection probability is biased by
    inverse-distance-power-law.  Combined with balanced fan-in and orphan
    repair so every output gets exactly k inputs and every input is used.

    This models biological wiring where:
      - Nearby neurons are more likely to connect (distance decay)
      - Every neuron still receives input (homeostatic balancing)
      - No neurons are orphaned (developmental repair)

    The selection within each output's budget is weighted by 1/d^exponent
    rather than uniform random, so nearby channels are preferred.
    """
    rng = np.random.RandomState(seed)
    densities = compute_erk_densities(model, target_density)
    masks = {}

    for name, layer in get_masked_layers(model):
        density = densities[name]
        n_out = layer.out_channels
        n_in = layer.in_channels
        k_h, k_w = layer.kernel_size

        if density >= 1.0:
            masks[name] = torch.ones_like(layer.weight)
            continue

        # Embed channels in 1D space (normalised index)
        out_pos = np.linspace(0, 1, n_out)
        in_pos = np.linspace(0, 1, n_in)

        # Distance matrix (n_out, n_in) — wrap-around for circular topology
        dist_matrix = np.abs(out_pos[:, None] - in_pos[None, :])
        dist_matrix = np.minimum(dist_matrix, 1.0 - dist_matrix)  # circular
        dist_matrix = np.maximum(dist_matrix, 1e-6)  # avoid div by zero

        # Connection probability weights: inverse distance power law
        prob_weights = 1.0 / (dist_matrix ** distance_exponent)

        # Number of input channels each output filter connects to
        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        k_per_out = n_keep // n_out
        remainder = n_keep - k_per_out * n_out

        # Step 1: distance-biased balanced allocation per output
        pair_mask = np.zeros((n_out, n_in), dtype=np.float32)
        for o in range(n_out):
            k = k_per_out + (1 if o < remainder else 0)
            k = min(k, n_in)
            if k > 0:
                # Weighted sampling without replacement
                w = prob_weights[o].copy()
                w /= w.sum()
                chosen = rng.choice(n_in, k, replace=False, p=w)
                pair_mask[o, chosen] = 1.0

        # Step 2: fix orphaned inputs (same as balanced_random)
        in_usage = pair_mask.sum(axis=0)
        orphaned = np.where(in_usage == 0)[0]

        for orphan_in in orphaned:
            most_used_in = np.argmax(in_usage)
            if in_usage[most_used_in] <= 1:
                break
            users = np.where(pair_mask[:, most_used_in] > 0)[0]
            swap_out = rng.choice(users)
            pair_mask[swap_out, most_used_in] = 0.0
            pair_mask[swap_out, orphan_in] = 1.0
            in_usage[most_used_in] -= 1
            in_usage[orphan_in] += 1

        # Expand to full weight shape
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)
        masks[name] = torch.tensor(full_mask)

    return masks, densities


def generate_random_er_masks(model, target_density, seed=42):
    """Random ERK sparse masks at CHANNEL-PAIR granularity.

    Same granularity as balanced_random: each selected (out, in) channel pair
    keeps its full spatial kernel (all k_h × k_w weights).  The only
    difference from balanced_random is that pairs are sampled i.i.d. Bernoulli
    (no balancing, no orphan repair).

    This ensures the comparison isolates allocation strategy (balanced vs
    random) rather than confounding it with sparsity granularity (structured
    vs unstructured).
    """
    rng = np.random.RandomState(seed)
    densities = compute_erk_densities(model, target_density)
    masks = {}

    for name, layer in get_masked_layers(model):
        density = densities[name]
        n_out, n_in = layer.out_channels, layer.in_channels
        k_h, k_w = layer.kernel_size

        if density >= 1.0:
            masks[name] = torch.ones_like(layer.weight)
            continue

        # Bernoulli sampling at channel-pair level
        pair_mask = (rng.random((n_out, n_in)) < density).astype(np.float32)

        # Expand to full kernel: active pairs keep entire spatial kernel
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)

        masks[name] = torch.tensor(full_mask)

    return masks, densities


def generate_snip_masks(model, target_density, train_loader, device, seed=42):
    """SNIP at CHANNEL-PAIR granularity (Lee et al., 2019).

    Scores are computed per-weight as |grad * weight|, then averaged across
    the spatial kernel dimensions to produce one score per (out, in) channel
    pair. Top-k pairs are kept per layer (ERK budget), and each kept pair
    retains its full spatial kernel.

    This matches the granularity of balanced_random and random_er so that the
    comparison isolates allocation strategy, not sparsity structure.
    """
    densities = compute_erk_densities(model, target_density)

    model.to(device)
    model.train()

    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)

    output = model(inputs)
    loss = nn.CrossEntropyLoss()(output, targets)
    loss.backward()

    masks = {}
    for name, layer in get_masked_layers(model):
        density = densities[name]
        if density >= 1.0:
            masks[name] = torch.ones_like(layer.weight)
            continue

        n_out, n_in = layer.out_channels, layer.in_channels
        k_h, k_w = layer.kernel_size

        # Per-weight sensitivity, averaged to channel-pair score
        score = (layer.weight.grad * layer.weight).abs()       # (n_out, n_in, k_h, k_w)
        pair_score = score.mean(dim=(2, 3))                     # (n_out, n_in)

        # Keep top-k channel pairs per ERK density
        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        threshold = torch.topk(pair_score.flatten(), n_keep).values[-1]
        pair_mask = (pair_score >= threshold).float().cpu().numpy()

        # Expand to full kernel: kept pairs retain entire spatial kernel
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)

        masks[name] = torch.tensor(full_mask)

    model.zero_grad()
    return masks, densities


# =============================================================================
# Balanced magnitude pruning (for balanced_dev / balanced_dev_matched)
# =============================================================================

def balanced_magnitude_prune(model, target_density):
    """Magnitude pruning with balanced fan-in and no orphaned inputs.

    Like standard magnitude pruning, but:
      1. Per-output top-k by |weight| (not global) → balanced fan-in
      2. Post-hoc orphan repair → no dead input channels
    """
    rng = np.random.RandomState(0)
    densities = compute_erk_densities(model, target_density)

    for name, layer in get_masked_layers(model):
        density = densities[name]
        if density >= 1.0:
            continue

        n_out = layer.out_channels
        n_in = layer.in_channels
        k_h, k_w = layer.kernel_size

        # Score: mean |weight| across kernel spatial dims per (out, in) pair
        weight_mag = layer.weight.data.abs().mean(dim=(2, 3)).cpu().numpy()

        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        k_per_out = n_keep // n_out
        remainder = n_keep - k_per_out * n_out

        # Per-output top-k by magnitude
        pair_mask = np.zeros((n_out, n_in), dtype=np.float32)
        for o in range(n_out):
            k = k_per_out + (1 if o < remainder else 0)
            k = min(k, n_in)
            if k > 0:
                top_idx = np.argpartition(weight_mag[o], -k)[-k:]
                pair_mask[o, top_idx] = 1.0

        # Fix orphaned inputs
        in_usage = pair_mask.sum(axis=0)
        orphaned = np.where(in_usage == 0)[0]
        for orphan_in in orphaned:
            most_used = np.argmax(in_usage)
            if in_usage[most_used] <= 1:
                break
            users = np.where(pair_mask[:, most_used] > 0)[0]
            # Swap the weakest connection to most_used for the orphan
            user_scores = weight_mag[users, most_used]
            weakest = users[np.argmin(user_scores)]
            pair_mask[weakest, most_used] = 0.0
            pair_mask[weakest, orphan_in] = 1.0
            in_usage[most_used] -= 1
            in_usage[orphan_in] += 1

        # Expand to full mask
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)

        layer.apply_mask(torch.tensor(full_mask).to(layer.weight.device))


def distance_magnitude_prune(model, target_density, alpha=0.5):
    """Distance-weighted magnitude pruning with balanced fan-in.

    Survival score per channel pair blends learned weight magnitude with
    spatial proximity (inverse distance between channel indices):

        score = alpha * |weight|_norm + (1 - alpha) * proximity_norm

    This mirrors the biological pruning model from the developmental
    notebooks: connections survive based on both activity (weight magnitude)
    AND spatial proximity (wiring cost / distance).

    Args:
        model: The model to prune.
        target_density: Target fraction of channel pairs to keep.
        alpha: Blend factor. 1.0 = pure magnitude, 0.0 = pure distance.
    """
    rng = np.random.RandomState(0)
    densities = compute_erk_densities(model, target_density)

    for name, layer in get_masked_layers(model):
        density = densities[name]
        if density >= 1.0:
            continue

        n_out = layer.out_channels
        n_in = layer.in_channels
        k_h, k_w = layer.kernel_size

        # Score 1: learned weight magnitude (mean |w| across spatial kernel)
        weight_mag = layer.weight.data.abs().mean(dim=(2, 3)).cpu().numpy()
        # Normalise per output to [0, 1]
        mag_min = weight_mag.min(axis=1, keepdims=True)
        mag_max = weight_mag.max(axis=1, keepdims=True)
        mag_range = mag_max - mag_min
        mag_range[mag_range == 0] = 1.0
        mag_norm = (weight_mag - mag_min) / mag_range

        # Score 2: spatial proximity (inverse distance on circular 1D channel space)
        out_pos = np.linspace(0, 1, n_out)
        in_pos = np.linspace(0, 1, n_in)
        dist_matrix = np.abs(out_pos[:, None] - in_pos[None, :])
        dist_matrix = np.minimum(dist_matrix, 1.0 - dist_matrix)  # circular
        # Proximity = 1 - normalised distance (close = high score)
        d_max = dist_matrix.max()
        proximity = 1.0 - (dist_matrix / d_max if d_max > 0 else dist_matrix)

        # Combined survival score
        score = alpha * mag_norm + (1 - alpha) * proximity

        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        k_per_out = n_keep // n_out
        remainder = n_keep - k_per_out * n_out

        # Per-output top-k by blended score (balanced fan-in)
        pair_mask = np.zeros((n_out, n_in), dtype=np.float32)
        for o in range(n_out):
            k = k_per_out + (1 if o < remainder else 0)
            k = min(k, n_in)
            if k > 0:
                top_idx = np.argpartition(score[o], -k)[-k:]
                pair_mask[o, top_idx] = 1.0

        # Fix orphaned inputs
        in_usage = pair_mask.sum(axis=0)
        orphaned = np.where(in_usage == 0)[0]
        for orphan_in in orphaned:
            most_used = np.argmax(in_usage)
            if in_usage[most_used] <= 1:
                break
            users = np.where(pair_mask[:, most_used] > 0)[0]
            user_scores = score[users, most_used]
            weakest = users[np.argmin(user_scores)]
            pair_mask[weakest, most_used] = 0.0
            pair_mask[weakest, orphan_in] = 1.0
            in_usage[most_used] -= 1
            in_usage[orphan_in] += 1

        # Expand to full mask
        full_mask = pair_mask[:, :, None, None] * np.ones((1, 1, k_h, k_w), dtype=np.float32)
        layer.apply_mask(torch.tensor(full_mask).to(layer.weight.device))


def magnitude_prune_step(model, target_density):
    """Magnitude pruning at CHANNEL-PAIR granularity (ERK budget).

    Scores each (out, in) pair by mean |weight| across spatial dims,
    keeps top-k pairs per layer, and retains full kernels for kept pairs.
    Matches granularity of all other methods.
    """
    densities = compute_erk_densities(model, target_density)
    for name, layer in get_masked_layers(model):
        density = densities[name]
        if density >= 1.0:
            continue

        n_out, n_in = layer.out_channels, layer.in_channels
        k_h, k_w = layer.kernel_size

        # Mean |weight| per channel pair
        pair_score = layer.weight.data.abs().mean(dim=(2, 3))  # (n_out, n_in)

        total_pairs = n_out * n_in
        n_keep = max(1, int(density * total_pairs))
        threshold = torch.topk(pair_score.flatten(), n_keep).values[-1]
        pair_mask = (pair_score >= threshold).float()

        # Expand to full kernel
        full_mask = pair_mask.unsqueeze(-1).unsqueeze(-1).expand(n_out, n_in, k_h, k_w)
        layer.apply_mask(full_mask.to(layer.weight.device))


# =============================================================================
# Metrics
# =============================================================================

def compute_graph_metrics(model):
    """Topology metrics from sparse connectivity."""
    layers = get_masked_layers(model)
    total_edges = 0
    degrees_out = []  # fan-in per output filter
    degrees_in = []   # fan-out per input channel
    orphaned_inputs = 0
    dead_outputs = 0

    for name, layer in layers:
        pair_mask = (layer.mask.sum(dim=(2, 3)) > 0).float()
        n_out, n_in = pair_mask.shape

        total_edges += pair_mask.sum().item()

        out_deg = pair_mask.sum(dim=1).cpu().numpy()
        in_deg = pair_mask.sum(dim=0).cpu().numpy()

        degrees_out.extend(out_deg.tolist())
        degrees_in.extend(in_deg.tolist())

        orphaned_inputs += (in_deg == 0).sum()
        dead_outputs += (out_deg == 0).sum()

    degrees_out = np.array(degrees_out)
    degrees_in = np.array(degrees_in)

    # Clustering: cosine similarity of adjacency rows (per layer, averaged)
    clustering = 0.0
    n_layers = 0
    for name, layer in layers:
        pm = (layer.mask.sum(dim=(2, 3)) > 0).float().cpu().numpy()
        if pm.shape[0] < 2:
            continue
        norms = np.sqrt((pm ** 2).sum(axis=1, keepdims=True)) + 1e-10
        normed = pm / norms
        sim = normed @ normed.T
        np.fill_diagonal(sim, 0)
        n = pm.shape[0]
        clustering += sim.sum() / (n * (n - 1))
        n_layers += 1

    if n_layers > 0:
        clustering /= n_layers

    return {
        "clustering": float(clustering),
        "avg_degree_out": float(degrees_out.mean()),
        "std_degree_out": float(degrees_out.std()),
        "avg_degree_in": float(degrees_in.mean()),
        "std_degree_in": float(degrees_in.std()),
        "orphaned_inputs": int(orphaned_inputs),
        "dead_outputs": int(dead_outputs),
        "total_edges": int(total_edges),
    }


def compute_sparsity(model):
    """Overall sparsity of masked layers."""
    total, zeros = 0, 0
    for _, layer in get_masked_layers(model):
        total += layer.mask.numel()
        zeros += (layer.mask == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def compute_flops_per_sample(model):
    """Estimate FLOPs for one forward pass (sparse and dense)."""
    sparse_flops = 0
    dense_flops = 0

    for name, layer in get_masked_layers(model):
        density = 1.0 - layer.sparsity
        k = layer.kernel_size[0] * layer.kernel_size[1]
        # Output spatial size for CIFAR-style 32x32 ResNet-18.
        if layer.in_channels <= 64:
            out_hw = 32
        elif layer.in_channels <= 128:
            out_hw = 16
        elif layer.in_channels <= 256:
            out_hw = 8
        else:
            out_hw = 4

        ops_per_output = layer.in_channels * k
        n_outputs = layer.out_channels * out_hw * out_hw
        layer_dense = 2 * n_outputs * ops_per_output
        dense_flops += layer_dense
        sparse_flops += layer_dense * density

    return sparse_flops, dense_flops


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        # Re-apply masks after optimizer step
        for _, layer in get_masked_layers(model):
            layer.weight.data *= layer.mask
        total_loss += loss.item() * inputs.size(0)
        correct += (output.argmax(1) == targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            correct += (output.argmax(1) == targets).sum().item()
            total += inputs.size(0)
    return correct / total


# =============================================================================
# Condition runner
# =============================================================================

def run_condition(condition, model, train_loader, test_loader, epochs, device, seed=42,
                  target_density=0.10):
    """Run one experimental condition. Returns results dict.

    target_density: final keep fraction (e.g. 0.10 => 90% sparse, 0.02 => 98% sparse).
    """

    print(f"\n{'='*70}")
    print(f"CONDITION: {condition}  target_density={target_density:.4f}")
    print(f"{'='*70}")

    TARGET_DENSITY = target_density                         # final keep fraction
    # Warmup density for balanced_dev/matched scales with target so the dense phase
    # stays a fixed factor (5x) denser than the final mask across sparsity levels.
    QUICK_INIT_DENSITY = min(0.50, max(TARGET_DENSITY * 5.0, TARGET_DENSITY + 0.05))
    QUICK_PRUNE_EPOCH = 3        # prune to final sparsity after this many epochs

    # --- Phase 1: Initial mask ---
    t0 = time.time()
    prune_callback = None  # called at start of each epoch

    if condition == "balanced_random":
        # Balanced random at target density from the start. Zero cost.
        masks, _ = generate_balanced_masks(model, TARGET_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

    elif condition == "distance_prior":
        # Distance-biased balanced allocation at target density. Zero cost.
        # Channels embedded in 1D space; connection probability ~ 1/d^2.
        # Combined with balanced fan-in and orphan repair.
        # This is the "bio-inspired from scratch" condition: spatial bias
        # without any training signal.
        masks, _ = generate_distance_balanced_masks(model, TARGET_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

    elif condition == "distance_dev":
        # The full bio-inspired developmental method:
        # 1. Start at higher density with distance-biased allocation
        #    (nearby channels more likely to connect — models initial
        #    spatially-biased wiring in development)
        # 2. Train 3 epochs (activity-dependent reinforcement)
        # 3. Prune using blended score: α × |weight| + (1-α) × proximity
        #    (models developmental pruning where BOTH activity AND spatial
        #    proximity determine which connections survive)
        masks, _ = generate_distance_balanced_masks(model, QUICK_INIT_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

        def prune_callback(epoch, mdl):
            if epoch == QUICK_PRUNE_EPOCH + 1:
                print(f"  >>> Pruning to {TARGET_DENSITY:.0%} density "
                      f"(distance × magnitude, alpha=0.5)")
                distance_magnitude_prune(mdl, TARGET_DENSITY, alpha=0.5)

    elif condition == "balanced_dev":
        # Start balanced at the warmup density, train 3 epochs, then
        # balanced-magnitude prune to the requested target density.
        masks, _ = generate_balanced_masks(model, QUICK_INIT_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

        def prune_callback(epoch, mdl):
            if epoch == QUICK_PRUNE_EPOCH + 1:
                print(f"  >>> Pruning to {TARGET_DENSITY:.0%} density (balanced magnitude)")
                balanced_magnitude_prune(mdl, TARGET_DENSITY)

    elif condition == "balanced_dev_matched":
        # Same as balanced_dev but target higher sparsity to match FLOPs budget
        # Extra FLOPs from 3 epochs at 50% ≈ 3 × (0.5/0.1) = 15 epoch-equivalents
        # Compensate by targeting ~95% sparsity (density=0.05) for remaining epochs
        # So total = 3*5 + 197*1 ≈ 212 epoch-equivalents at 10% density
        # vs random_er = 200*1 = 200 epoch-equivalents
        # Target density that makes total FLOPs equal:
        # 3 * 0.5 + (E-3) * d_final = E * 0.1
        # d_final = (E * 0.1 - 1.5) / (E - 3) ≈ 0.0924 for E=200
        matched_final_density = (epochs * TARGET_DENSITY - QUICK_PRUNE_EPOCH * QUICK_INIT_DENSITY) / (epochs - QUICK_PRUNE_EPOCH)
        matched_final_density = max(0.01, matched_final_density)  # floor

        masks, _ = generate_balanced_masks(model, QUICK_INIT_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

        def prune_callback(epoch, mdl, _d=matched_final_density):
            if epoch == QUICK_PRUNE_EPOCH + 1:
                print(f"  >>> Pruning to {_d:.4f} density (balanced magnitude, FLOPs-matched)")
                balanced_magnitude_prune(mdl, _d)

    elif condition == "random_er":
        masks, _ = generate_random_er_masks(model, TARGET_DENSITY, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

    elif condition == "snip":
        masks, _ = generate_snip_masks(model, TARGET_DENSITY, train_loader, device, seed=seed)
        for name, layer in get_masked_layers(model):
            layer.apply_mask(masks[name].to(device))

    elif condition == "iterative_mag":
        # Start dense, magnitude prune each epoch over first half
        n_prune_steps = epochs // 2
        density_schedule = []
        for i in range(1, n_prune_steps + 1):
            d = 1.0 - (1.0 - TARGET_DENSITY) * (i / n_prune_steps)
            density_schedule.append(d)

        def prune_callback(epoch, mdl, _sched=density_schedule):
            if epoch <= len(_sched):
                magnitude_prune_step(mdl, _sched[epoch - 1])

    elif condition == "dense_baseline":
        pass  # no pruning

    else:
        raise ValueError(f"Unknown condition: {condition}")

    mask_time = time.time() - t0

    # --- Report initial state ---
    sp0 = compute_sparsity(model)
    metrics0 = compute_graph_metrics(model)
    print(f"  Mask generation: {mask_time:.2f}s")
    print(f"  Initial sparsity: {sp0:.4f}")
    print(f"  Fan-in: {metrics0['avg_degree_out']:.1f}±{metrics0['std_degree_out']:.1f}  "
          f"Fan-out: {metrics0['avg_degree_in']:.1f}±{metrics0['std_degree_in']:.1f}")
    print(f"  Orphaned inputs: {metrics0['orphaned_inputs']}  Dead outputs: {metrics0['dead_outputs']}")
    print(f"  Clustering: {metrics0['clustering']:.4f}")

    # --- Training ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [],
        "lr": [], "sparsity": [],
    }
    best_acc = 0.0
    t_train = time.time()

    # Track cumulative FLOPs
    _, dense_per_sample = compute_flops_per_sample(model)
    n_train = len(train_loader.dataset)
    cumulative_gflops = 0.0

    for epoch in range(1, epochs + 1):
        # Pruning callback (before training this epoch)
        if prune_callback is not None:
            prune_callback(epoch, model)

        # Track FLOPs for this epoch using current sparsity
        sp = compute_sparsity(model)
        density = 1.0 - sp
        epoch_flops = dense_per_sample * density * n_train * 3  # fwd + bwd ≈ 3× fwd
        cumulative_gflops += epoch_flops / 1e9

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)
        history["sparsity"].append(sp)

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch <= 5 or epoch % 20 == 0 or epoch == epochs:
            elapsed = time.time() - t_train
            print(f"  Epoch {epoch:>3d}/{epochs}  train={train_loss:.4f}/{train_acc:.4f}  "
                  f"test={test_acc:.4f}  best={best_acc:.4f}  sp={sp:.3f}  "
                  f"lr={lr:.6f}  [{elapsed:.0f}s]")

    train_time = time.time() - t_train

    # --- Final metrics ---
    final_sp = compute_sparsity(model)
    final_metrics = compute_graph_metrics(model)

    # Convergence milestones
    convergence = {}
    for thresh in [0.70, 0.80, 0.85, 0.90, 0.92, 0.94]:
        key = f"epochs_to_{int(thresh*100)}"
        convergence[key] = next(
            (i+1 for i, a in enumerate(history["test_acc"]) if a >= thresh), None
        )

    print(f"\n  Done: {train_time:.0f}s  best={best_acc:.4f}  final_sp={final_sp:.4f}")
    print(f"  Total GFLOPs: {cumulative_gflops:.0f}")
    print(f"  Final fan-in: {final_metrics['avg_degree_out']:.1f}±{final_metrics['std_degree_out']:.1f}  "
          f"orphans={final_metrics['orphaned_inputs']}  dead={final_metrics['dead_outputs']}")

    return {
        "condition": condition,
        "mask_time": mask_time,
        "train_time": train_time,
        "best_acc": best_acc,
        "final_test_acc": history["test_acc"][-1],
        "final_sparsity": final_sp,
        "total_gflops": cumulative_gflops,
        "graph_metrics_init": metrics0,
        "graph_metrics_final": final_metrics,
        "convergence": convergence,
        "history": history,
    }


# =============================================================================
# Experiment runner
# =============================================================================

ALL_CONDITIONS = [
    "balanced_random",
    "distance_prior",
    "distance_dev",
    "balanced_dev",
    "balanced_dev_matched",
    "random_er",
    "snip",
    "iterative_mag",
    "dense_baseline",
]


def run_experiment(conditions=None, epochs=200, n_seeds=1, device="cuda",
                   seed_start=42, results_dir=None,
                   dataset="cifar10", target_sparsity=0.90):
    """Run all conditions (optionally with multiple seeds) and save results."""
    if conditions is None:
        conditions = ALL_CONDITIONS
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    target_density = 1.0 - target_sparsity

    # --- Data ---
    if dataset == "cifar10":
        print("Loading CIFAR-10...")
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)
        ds_cls = torchvision.datasets.CIFAR10
        num_classes = 10
    elif dataset == "cifar100":
        print("Loading CIFAR-100...")
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)
        ds_cls = torchvision.datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_data = ds_cls("data", train=True, download=True, transform=transform_train)
    test_data  = ds_cls("data", train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    print(f"Dataset: {dataset}  Classes: {num_classes}  Train: {len(train_data)}  "
          f"Test: {len(test_data)}  Epochs: {epochs}  Seeds: {n_seeds}  "
          f"Target sparsity: {target_sparsity:.4f}")

    # --- Save config ---
    config = {
        "conditions": conditions, "epochs": epochs, "n_seeds": n_seeds,
        "seed_start": seed_start, "dataset": dataset, "model": "resnet18",
        "num_classes": num_classes,
        "target_sparsity": target_sparsity, "target_density": target_density,
        "optimizer": "sgd", "lr": 0.1,
        "weight_decay": 5e-4, "scheduler": "cosine",
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    all_results = {}
    summary_rows = []

    for cond in conditions:
        seed_results = []
        for s in range(n_seeds):
            seed = seed_start + s
            print(f"\n{'#'*70}")
            print(f"# {cond}  seed={seed}  ({s+1}/{n_seeds})")
            print(f"{'#'*70}")

            torch.manual_seed(seed)
            np.random.seed(seed)
            model = make_resnet18_sparse(num_classes=num_classes).to(device)

            result = run_condition(cond, model, train_loader, test_loader,
                                   epochs, device, seed,
                                   target_density=target_density)
            seed_results.append(result)

            # Save per-run result
            torch.save(result, os.path.join(results_dir, f"result_{cond}_seed{seed}.pt"))

        all_results[cond] = seed_results

        # Aggregate across seeds
        best_accs = [r["best_acc"] for r in seed_results]
        gflops = [r["total_gflops"] for r in seed_results]
        final_metrics = seed_results[-1]["graph_metrics_final"]

        summary_rows.append({
            "condition": cond,
            "best_acc_mean": np.mean(best_accs),
            "best_acc_std": np.std(best_accs),
            "total_gflops_mean": np.mean(gflops),
            "final_sparsity": seed_results[-1]["final_sparsity"],
            "avg_degree_out": final_metrics["avg_degree_out"],
            "std_degree_out": final_metrics["std_degree_out"],
            "orphaned_inputs": final_metrics["orphaned_inputs"],
            "dead_outputs": final_metrics["dead_outputs"],
            "clustering": final_metrics["clustering"],
            "n_seeds": n_seeds,
        })

    # --- Save summary ---
    summary_path = os.path.join(results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # --- Save curves (first seed per condition) ---
    curves_path = os.path.join(results_dir, "curves.csv")
    with open(curves_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "seed", "epoch", "train_loss", "train_acc",
                          "test_acc", "lr", "sparsity"])
        for cond, seed_results in all_results.items():
            for r in seed_results:
                seed = r.get("seed", seed_start)
                for ep in range(len(r["history"]["train_loss"])):
                    writer.writerow([
                        cond, seed, ep + 1,
                        f"{r['history']['train_loss'][ep]:.6f}",
                        f"{r['history']['train_acc'][ep]:.6f}",
                        f"{r['history']['test_acc'][ep]:.6f}",
                        f"{r['history']['lr'][ep]:.8f}",
                        f"{r['history']['sparsity'][ep]:.6f}",
                    ])

    # --- Print summary ---
    print(f"\n{'='*100}")
    print(f"SUMMARY ({n_seeds} seed{'s' if n_seeds > 1 else ''})")
    print(f"{'='*100}")
    print(f"{'Condition':<18s} {'Best Acc':>14s} {'GFLOPs':>10s} {'Sparsity':>9s} "
          f"{'Fan-in':>12s} {'Orphans':>8s} {'Dead':>6s} {'Clust':>7s}")
    print("-" * 100)
    for row in summary_rows:
        acc_str = f"{row['best_acc_mean']:.4f}±{row['best_acc_std']:.4f}" if n_seeds > 1 else f"{row['best_acc_mean']:.4f}"
        print(f"{row['condition']:<18s} {acc_str:>14s} {row['total_gflops_mean']:>10.0f} "
              f"{row['final_sparsity']:>9.4f} "
              f"{row['avg_degree_out']:>5.1f}±{row['std_degree_out']:>4.1f} "
              f"{row['orphaned_inputs']:>8d} {row['dead_outputs']:>6d} "
              f"{row['clustering']:>7.4f}")

    print(f"\nResults saved to: {results_dir}")
    return all_results


# =============================================================================
# Transfer experiment: topology discovered on one class split, evaluated on another
# =============================================================================

def subset_dataset(dataset, class_indices):
    """Filter dataset to only include samples from the given class indices.
    Remaps labels to 0..len(class_indices)-1."""
    class_set = set(class_indices)
    label_map = {c: i for i, c in enumerate(sorted(class_indices))}
    indices = [i for i, (_, y) in enumerate(dataset) if y in class_set]
    subset = torch.utils.data.Subset(dataset, indices)
    # Wrap to remap labels
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, label_map):
            self.subset = subset
            self.label_map = label_map
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            return x, self.label_map[y]
    return RemappedDataset(subset, label_map)


def run_transfer_experiment(epochs=200, n_seeds=1, device="cuda",
                            seed_start=42, results_dir=None,
                            target_sparsity=0.98):
    """Transfer experiment on CIFAR-100: discover topology on classes 0-49,
    evaluate on classes 50-99.

    Conditions:
      balanced_dev_transfer       - topology + weights from source (balanced mag prune)
      balanced_dev_topo_only      - topology from source, weights reinit (balanced mag prune)
      distance_dev_transfer  - topology + weights from source (distance × mag prune)
      distance_dev_topo_only - topology from source, weights reinit (distance × mag prune)
      balanced_dev_direct         - balanced_dev end-to-end on target classes only
      distance_dev_direct    - distance_dev end-to-end on target classes only
      balanced_random_direct      - balanced random on target (structure, no selection)
      distance_prior_direct - distance-biased balanced on target (structure, no selection)
      random_er_direct         - random ER on target (baseline)
    """
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results_transfer")
    os.makedirs(results_dir, exist_ok=True)

    target_density = 1.0 - target_sparsity
    num_classes_per_split = 50  # half of CIFAR-100

    # --- Data ---
    print("Loading CIFAR-100 for transfer experiment...")
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    full_train = torchvision.datasets.CIFAR100("data", train=True, download=True, transform=transform_train)
    full_test  = torchvision.datasets.CIFAR100("data", train=False, download=True, transform=transform_test)

    source_classes = list(range(0, 50))
    target_classes = list(range(50, 100))

    source_train = subset_dataset(full_train, source_classes)
    target_train = subset_dataset(full_train, target_classes)
    target_test  = subset_dataset(full_test, target_classes)

    source_loader = torch.utils.data.DataLoader(source_train, batch_size=128, shuffle=True, num_workers=2)
    target_train_loader = torch.utils.data.DataLoader(target_train, batch_size=128, shuffle=True, num_workers=2)
    target_test_loader  = torch.utils.data.DataLoader(target_test, batch_size=128, shuffle=False, num_workers=2)

    print(f"Source classes: 0-49 ({len(source_train)} train)")
    print(f"Target classes: 50-99 ({len(target_train)} train, {len(target_test)} test)")
    print(f"Target sparsity: {target_sparsity:.4f}  Epochs: {epochs}  Seeds: {n_seeds}")

    # balanced_dev warmup density (same formula as run_condition)
    QUICK_INIT_DENSITY = min(0.50, max(target_density * 5.0, target_density + 0.05))
    QUICK_PRUNE_EPOCH = 3

    # --- Save config ---
    config = {
        "experiment": "transfer",
        "source_classes": "0-49", "target_classes": "50-99",
        "dataset": "cifar100", "model": "resnet18",
        "num_classes_per_split": num_classes_per_split,
        "target_sparsity": target_sparsity, "target_density": target_density,
        "quick_init_density": QUICK_INIT_DENSITY,
        "quick_prune_epoch": QUICK_PRUNE_EPOCH,
        "epochs": epochs, "n_seeds": n_seeds, "seed_start": seed_start,
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    conditions = [
        "balanced_dev_transfer",        # topology + weights from source (mag prune)
        "balanced_dev_topo_only",       # topology from source, reinit (mag prune)
        "distance_dev_transfer",   # topology + weights from source (dist × mag prune)
        "distance_dev_topo_only",  # topology from source, reinit (dist × mag prune)
        "balanced_dev_direct",          # balanced_dev on target only
        "distance_dev_direct",     # distance_dev on target only
        "balanced_random_direct",       # balanced random on target only
        "distance_prior_direct",  # distance-biased balanced on target only
        "random_er_direct",          # random ER on target only
    ]

    for cond in conditions:
        for s in range(n_seeds):
            seed = seed_start + s

            # Skip if result already exists (resume support)
            result_path = os.path.join(results_dir, f"result_{cond}_seed{seed}.pt")
            if os.path.exists(result_path):
                print(f"\n[SKIP] {cond} seed={seed} — already exists at {result_path}")
                continue

            print(f"\n{'#'*70}")
            print(f"# {cond}  seed={seed}  ({s+1}/{n_seeds})")
            print(f"{'#'*70}")

            torch.manual_seed(seed)
            np.random.seed(seed)

            if cond in ("balanced_dev_transfer", "balanced_dev_topo_only"):
                # --- Phase 1: discover topology on source classes ---
                print("  Phase 1: Topology discovery on source classes 0-49 (balanced mag)")
                model_source = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)

                masks, _ = generate_balanced_masks(model_source, QUICK_INIT_DENSITY, seed=seed)
                for name, layer in get_masked_layers(model_source):
                    layer.apply_mask(masks[name].to(device))

                criterion = nn.CrossEntropyLoss()
                opt_src = optim.SGD(model_source.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                for ep in range(1, QUICK_PRUNE_EPOCH + 1):
                    loss, acc = train_epoch(model_source, source_loader, opt_src, criterion, device)
                    print(f"    Source ep {ep}: loss={loss:.4f}  acc={acc:.4f}")

                print(f"    Pruning to {target_density:.4f} density (balanced magnitude)")
                balanced_magnitude_prune(model_source, target_density)
                sp = compute_sparsity(model_source)
                gm = compute_graph_metrics(model_source)
                print(f"    Post-prune sparsity: {sp:.4f}  "
                      f"orphans={gm['orphaned_inputs']}  dead={gm['dead_outputs']}")

                discovered_masks = {}
                for name, layer in get_masked_layers(model_source):
                    discovered_masks[name] = layer.mask.clone().cpu()

                print("  Phase 2: Transfer to target classes 50-99")
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(discovered_masks[name].to(device))

                if cond == "balanced_dev_transfer":
                    src_state = model_source.state_dict()
                    tgt_state = model.state_dict()
                    for k, v in src_state.items():
                        if k in tgt_state and tgt_state[k].shape == v.shape:
                            tgt_state[k] = v
                    model.load_state_dict(tgt_state)
                    for name, layer in get_masked_layers(model):
                        layer.apply_mask(discovered_masks[name].to(device))
                    print("    Transferred conv weights + topology.")
                else:
                    print("    Transferred topology only. Weights reinitialized.")

                del model_source

            elif cond in ("distance_dev_transfer", "distance_dev_topo_only"):
                # --- Phase 1: discover topology on source classes (distance × mag) ---
                print("  Phase 1: Topology discovery on source classes 0-49 (distance × mag)")
                model_source = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)

                # Start with distance-biased balanced allocation
                masks, _ = generate_distance_balanced_masks(model_source, QUICK_INIT_DENSITY, seed=seed)
                for name, layer in get_masked_layers(model_source):
                    layer.apply_mask(masks[name].to(device))

                criterion = nn.CrossEntropyLoss()
                opt_src = optim.SGD(model_source.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
                for ep in range(1, QUICK_PRUNE_EPOCH + 1):
                    loss, acc = train_epoch(model_source, source_loader, opt_src, criterion, device)
                    print(f"    Source ep {ep}: loss={loss:.4f}  acc={acc:.4f}")

                # Prune with blended distance × magnitude score
                print(f"    Pruning to {target_density:.4f} density (distance × magnitude)")
                distance_magnitude_prune(model_source, target_density, alpha=0.5)
                sp = compute_sparsity(model_source)
                gm = compute_graph_metrics(model_source)
                print(f"    Post-prune sparsity: {sp:.4f}  "
                      f"orphans={gm['orphaned_inputs']}  dead={gm['dead_outputs']}")

                discovered_masks = {}
                for name, layer in get_masked_layers(model_source):
                    discovered_masks[name] = layer.mask.clone().cpu()

                print("  Phase 2: Transfer to target classes 50-99")
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(discovered_masks[name].to(device))

                if cond == "distance_dev_transfer":
                    src_state = model_source.state_dict()
                    tgt_state = model.state_dict()
                    for k, v in src_state.items():
                        if k in tgt_state and tgt_state[k].shape == v.shape:
                            tgt_state[k] = v
                    model.load_state_dict(tgt_state)
                    for name, layer in get_masked_layers(model):
                        layer.apply_mask(discovered_masks[name].to(device))
                    print("    Transferred conv weights + topology.")
                else:
                    print("    Transferred topology only. Weights reinitialized.")

                del model_source

            elif cond == "balanced_dev_direct":
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                masks, _ = generate_balanced_masks(model, QUICK_INIT_DENSITY, seed=seed)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(masks[name].to(device))

            elif cond == "distance_dev_direct":
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                masks, _ = generate_distance_balanced_masks(model, QUICK_INIT_DENSITY, seed=seed)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(masks[name].to(device))

            elif cond == "balanced_random_direct":
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                masks, _ = generate_balanced_masks(model, target_density, seed=seed)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(masks[name].to(device))

            elif cond == "distance_prior_direct":
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                masks, _ = generate_distance_balanced_masks(model, target_density, seed=seed)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(masks[name].to(device))

            elif cond == "random_er_direct":
                model = make_resnet18_sparse(num_classes=num_classes_per_split).to(device)
                masks, _ = generate_random_er_masks(model, target_density, seed=seed)
                for name, layer in get_masked_layers(model):
                    layer.apply_mask(masks[name].to(device))

            # --- Set up pruning callback for *_direct conditions that need it ---
            prune_callback = None
            if cond == "balanced_dev_direct":
                def prune_callback(epoch, mdl, _d=target_density):
                    if epoch == QUICK_PRUNE_EPOCH + 1:
                        print(f"  >>> Pruning to {_d:.4f} density (balanced magnitude)")
                        balanced_magnitude_prune(mdl, _d)
            elif cond == "distance_dev_direct":
                def prune_callback(epoch, mdl, _d=target_density):
                    if epoch == QUICK_PRUNE_EPOCH + 1:
                        print(f"  >>> Pruning to {_d:.4f} density (distance × magnitude)")
                        distance_magnitude_prune(mdl, _d, alpha=0.5)

            # --- Train on target classes ---
            sp0 = compute_sparsity(model)
            gm0 = compute_graph_metrics(model)
            print(f"  Init sparsity: {sp0:.4f}  "
                  f"orphans={gm0['orphaned_inputs']}  dead={gm0['dead_outputs']}")

            result = run_condition_inner(
                cond, model, target_train_loader, target_test_loader,
                epochs, device, prune_callback=prune_callback
            )
            result["graph_metrics_init"] = gm0
            result["condition"] = cond
            result["seed"] = seed

            torch.save(result, os.path.join(results_dir, f"result_{cond}_seed{seed}.pt"))
            print(f"  Saved: result_{cond}_seed{seed}.pt  "
                  f"best={result['best_acc']:.4f}")

    print(f"\nTransfer experiment complete. Results in: {results_dir}")


def run_condition_inner(condition, model, train_loader, test_loader,
                        epochs, device, prune_callback=None):
    """Inner training loop (shared by both main and transfer experiments)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [], "train_acc": [], "test_acc": [],
        "lr": [], "sparsity": [],
    }
    best_acc = 0.0
    t_train = time.time()
    mask_time = 0.0

    _, dense_per_sample = compute_flops_per_sample(model)
    n_train = len(train_loader.dataset)
    cumulative_gflops = 0.0

    for epoch in range(1, epochs + 1):
        if prune_callback is not None:
            prune_callback(epoch, model)

        sp = compute_sparsity(model)
        density = 1.0 - sp
        epoch_flops = dense_per_sample * density * n_train * 3
        cumulative_gflops += epoch_flops / 1e9

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate(model, test_loader, device)
        lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)
        history["sparsity"].append(sp)

        if test_acc > best_acc:
            best_acc = test_acc

        if epoch <= 5 or epoch % 20 == 0 or epoch == epochs:
            elapsed = time.time() - t_train
            print(f"  Epoch {epoch:>3d}/{epochs}  train={train_loss:.4f}/{train_acc:.4f}  "
                  f"test={test_acc:.4f}  best={best_acc:.4f}  sp={sp:.3f}  "
                  f"lr={lr:.6f}  [{elapsed:.0f}s]")

    train_time = time.time() - t_train
    final_sp = compute_sparsity(model)
    final_metrics = compute_graph_metrics(model)

    convergence = {}
    for thresh in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        key = f"epochs_to_{int(thresh*100)}"
        convergence[key] = next(
            (i+1 for i, a in enumerate(history["test_acc"]) if a >= thresh), None
        )

    return {
        "condition": condition,
        "mask_time": mask_time,
        "train_time": train_time,
        "best_acc": best_acc,
        "final_test_acc": history["test_acc"][-1],
        "final_sparsity": final_sp,
        "total_gflops": cumulative_gflops,
        "graph_metrics_final": final_metrics,
        "convergence": convergence,
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper 1: Pruning strategy comparison (CIFAR ResNet-18)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", default=None,
                        help=f"Subset of: {', '.join(ALL_CONDITIONS)}")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 20 epochs, 3 conditions, 1 seed")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100"],
                        help="Training dataset. Paper A reported runs use cifar100; cifar10 is a legacy smoke/dev default.")
    parser.add_argument("--sparsity", type=float, default=0.90,
                        help="Target sparsity (fraction of weights pruned), e.g. 0.98")
    parser.add_argument("--transfer", action="store_true",
                        help="Run transfer experiment (CIFAR-100, split 0-49/50-99)")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Paper 1 — Pruning Strategy Comparison (CIFAR ResNet-18)")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    if args.transfer:
        run_transfer_experiment(
            epochs=args.epochs, n_seeds=args.seeds, device=device,
            seed_start=args.seed_start, results_dir=args.output_dir,
            target_sparsity=args.sparsity,
        )
    elif args.quick:
        run_experiment(
            conditions=args.conditions or ["balanced_random", "random_er", "snip"],
            epochs=20, n_seeds=1, device=device,
            seed_start=args.seed_start, results_dir=args.output_dir,
            dataset=args.dataset, target_sparsity=args.sparsity,
        )
    else:
        run_experiment(
            conditions=args.conditions, epochs=args.epochs,
            n_seeds=args.seeds, device=device,
            seed_start=args.seed_start, results_dir=args.output_dir,
            dataset=args.dataset, target_sparsity=args.sparsity,
        )
