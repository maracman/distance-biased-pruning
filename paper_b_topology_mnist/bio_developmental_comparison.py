#!/usr/bin/env python3
"""
Paper 2 — Bio-Inspired Topology vs λ-Sweep Comparison

Tests whether the Bio-Inspired developmental topology (Paper 1) outperforms
the best simple λ-mixture topology under Paper 2's bandwidth-constrained
framework. The λ-sweep maps the simple topology landscape; this script asks
whether richer, hierarchically-modular structure can beat that landscape's
optimum.

Conditions (all at 90% sparsity, [784, 256, 10] architecture):
  1. bio_developmental  — Inverse-square distance → 10 connectivity-preserving
                          developmental cycles (decay, Pareto reinforce, sigmoid
                          prune, connectivity repair after each cycle)
  2. spatial_proximity  — Inverse-square distance pruning, balanced fan-in,
                          NO developmental cycles (ablation: spatial vs dynamics)
  3. lambda_0.9         — 90% local / 10% random (near-optimal from sweep)
  4. lambda_0.5         — 50% local / 50% random (mid-range reference)
  5. lambda_0.0         — Pure random connections (ER baseline)

Fairness constraints:
  - All conditions use the SAME pixel-grid positions for the input layer
  - All conditions get fresh Kaiming-scaled weights AFTER mask generation
    (Bio-Inspired contribution is purely graph structure, not initial weights)
  - Same Trainer, hyperparameters, and data loaders across all conditions

Disconnect-safe (RunPod / cloud):
  - Results are saved after EVERY individual run (not just at the end)
  - On restart, completed runs are loaded from checkpoint and skipped
  - Atomic writes (temp file → os.replace) prevent corruption from crashes
  - Use --fresh to discard checkpoint and start over

Usage:
    python paper_b_topology_mnist/bio_developmental_comparison.py
    python paper_b_topology_mnist/bio_developmental_comparison.py --quick
    python paper_b_topology_mnist/bio_developmental_comparison.py --bandwidths 16 49 196 784
    python paper_b_topology_mnist/bio_developmental_comparison.py --epochs 30 --seeds 5
    python paper_b_topology_mnist/bio_developmental_comparison.py --fresh

Results saved to:    paper_b_topology_mnist/results/bio_comparison.json
Checkpoint file:     paper_b_topology_mnist/results/bio_comparison_checkpoint.json
"""
import os
import sys
import json
import time
import argparse
import tempfile
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from shared.topology.generators import (
    generate_locality_mask,
    generate_distance_pruned_mask,
    make_grid_positions,
)
from shared.topology.pruning import (
    developmental_pruning_connected,
    repair_connectivity,
)
from shared.topology.metrics import compute_projected_metrics
from shared.models.sparse_mlp import SparseMLP
from shared.training.trainer import Trainer


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint / atomic I/O
# ═══════════════════════════════════════════════════════════════════════════

def atomic_save(data, filepath):
    """Write JSON atomically: temp file in the same dir → os.replace().

    os.replace() is atomic on POSIX (Linux/RunPod).  If the process is
    killed mid-write, the previous checkpoint remains intact — the temp
    file is simply orphaned and harmless.
    """
    dirpath = os.path.dirname(filepath) or "."
    fd, tmp_path = tempfile.mkstemp(dir=dirpath, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, filepath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint(checkpoint_path):
    """Load an existing checkpoint, or return an empty structure."""
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            # Minimal validation
            if "runs" in ckpt:
                return ckpt
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠ Corrupt checkpoint ({e}), starting fresh.")
    return {"runs": {}, "config": {}}


def make_run_key(bw_pixels, condition, seed):
    """Deterministic string key for a single (bandwidth, condition, seed) run."""
    return f"{bw_pixels}px|{condition}|seed_{seed}"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading  (mirrors paper2/run.py — kept self-contained for clarity)
# ═══════════════════════════════════════════════════════════════════════════

class PatchSampledDataset(torch.utils.data.Dataset):
    """MNIST with bandwidth control via spatial patch sampling.

    Keeps the full 784-dim input but on each training access randomly reveals
    only one spatial patch (the rest are zeroed). At evaluation time the full
    image is shown.  This models a low-bandwidth sensory channel: the network
    sees the complete image over many training exposures, but through a narrow
    spatial pipe each time.

    Bandwidth = patch_size² / 784.

    Args:
        X: Images, shape (N, 784).
        y: Labels, shape (N,).
        n_patches: How many non-overlapping patches to divide the 28×28 image
            into.  E.g. n_patches=49 → 7×7 grid of 4×4-pixel patches (16 px
            per exposure).  n_patches=1 → full image.
        stochastic: If True, randomly sample one patch per __getitem__
            (training mode).  If False, show the full image (eval mode).
    """

    def __init__(self, X, y, n_patches=1, stochastic=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.n_patches = n_patches
        self.stochastic = stochastic

        if n_patches > 1:
            patches_per_side = int(np.sqrt(n_patches))
            assert patches_per_side ** 2 == n_patches, \
                f"n_patches must be a perfect square, got {n_patches}"
            patch_h = 28 // patches_per_side
            patch_w = 28 // patches_per_side
            self.patch_masks = []
            for pi in range(patches_per_side):
                for pj in range(patches_per_side):
                    mask = np.zeros((28, 28), dtype=np.float32)
                    mask[pi*patch_h:(pi+1)*patch_h,
                         pj*patch_w:(pj+1)*patch_w] = 1.0
                    self.patch_masks.append(torch.tensor(mask.flatten()))
        else:
            self.patch_masks = [torch.ones(784)]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.stochastic and self.n_patches > 1:
            patch_idx = torch.randint(0, len(self.patch_masks), (1,)).item()
            x = x * self.patch_masks[patch_idx]
        return x, self.y[idx]


def load_mnist():
    """Load MNIST, normalise, and return numpy arrays.

    Returns:
        X_train (N_train, 784), y_train (N_train,),
        X_test  (N_test,  784), y_test  (N_test,)
    """
    transform = transforms.Compose([transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(
        "data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.MNIST(
        "data", train=False, download=True, transform=transform)

    X_train = train_data.data.float().view(-1, 784).numpy() / 255.0
    y_train = train_data.targets.numpy()
    X_test = test_data.data.float().view(-1, 784).numpy() / 255.0
    y_test = test_data.targets.numpy()

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def get_dataloaders(X_train, y_train, X_test, y_test,
                    bandwidth_pixels, seed=42):
    """Create train / val / test DataLoaders for a bandwidth level.

    Args:
        bandwidth_pixels: Pixels visible per training exposure.
            784 → full image, 196 → 2×2 grid, 49 → 4×4 grid, 16 → 7×7 grid.

    Returns:
        train_loader, val_loader, test_loader
    """
    rng = np.random.RandomState(seed)
    n_val = int(0.1 * len(X_train))
    idx = rng.permutation(len(X_train))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    # Map bandwidth → patch grid
    if bandwidth_pixels >= 784:
        n_patches = 1
    elif bandwidth_pixels >= 196:
        n_patches = 4      # 2×2 grid, 196 px per patch
    elif bandwidth_pixels >= 49:
        n_patches = 16     # 4×4 grid, 49 px per patch
    elif bandwidth_pixels >= 16:
        n_patches = 49     # 7×7 grid, 16 px per patch
    else:
        n_patches = 196    # 14×14 grid, 4 px per patch

    train_ds = PatchSampledDataset(
        X_train[train_idx], y_train[train_idx],
        n_patches=n_patches, stochastic=True)
    val_ds = PatchSampledDataset(
        X_train[val_idx], y_train[val_idx],
        n_patches=1, stochastic=False)
    test_ds = PatchSampledDataset(
        X_test, y_test,
        n_patches=1, stochastic=False)

    return (DataLoader(train_ds, batch_size=128, shuffle=True),
            DataLoader(val_ds,   batch_size=128, shuffle=False),
            DataLoader(test_ds,  batch_size=128, shuffle=False))


# ═══════════════════════════════════════════════════════════════════════════
# Bio-Inspired developmental mask generation
# ═══════════════════════════════════════════════════════════════════════════

def generate_bio_developmental_mask(
    n_out: int,
    n_in: int,
    target_sparsity: float = 0.9,
    n_dev_cycles: int = 10,
    distance_exponent: float = 2.0,
    input_positions: np.ndarray = None,
    seed: int = 42,
    return_weights: bool = True,
):
    """Generate a sparse mask via the Bio-Inspired developmental process.

    Pipeline (two-phase, connectivity-preserving):
        1. Embed neurons spatially (grid positions for inputs, random for outputs).
        2. Compute pairwise inverse-square-distance connection probabilities.
        3. Sample an initial mask at 50% sparsity (dense enough to guarantee
           near-complete coverage, then explicitly repaired).
        4. Repair initial connectivity: ensure every input and output node
           has at least one connection.
        5. Create random developmental weights: mask * |N(0,1)|.
        6. Apply connectivity-preserving developmental pruning — 10 iterative
           cycles of: decay -> Pareto reinforce -> sigmoid prune -> repair.
           After each cycle, orphaned nodes are reconnected via minimum-
           distance edges and disconnected components are bridged.
        7. Connectivity-safe adjustment to exactly hit target_sparsity:
           remove farthest-distance edges, but skip any edge whose removal
           would orphan a node.
        8. Generate fresh Kaiming-scaled weights (developmental weights are
           discarded — only the topology is kept).

    This two-phase approach mirrors biological neurodevelopment: networks
    are over-built then refined, but connectivity is never destroyed.

    Args:
        n_out: Number of output neurons.
        n_in: Number of input neurons.
        target_sparsity: Desired final sparsity.
        n_dev_cycles: Developmental pruning iterations (default: 10).
        distance_exponent: Exponent for spatial decay.
        input_positions: (n_in, 2) positions. Grid for input layer.
        seed: Random seed.
        return_weights: If True, return (mask, kaiming_weights).

    Returns:
        mask: Binary (n_out, n_in) at exactly target_sparsity.
        weights (optional): Kaiming-scaled float array.
    """
    rng = np.random.RandomState(seed)
    epsilon = 1e-6

    # --- 1. Spatial embedding ---
    if input_positions is not None:
        input_pos = input_positions.copy()
    else:
        input_pos = rng.uniform(0, 1, (n_in, 2))
    output_pos = rng.uniform(0, 1, (n_out, 2))

    # --- 2. Pairwise distances ---
    distances = cdist(output_pos, input_pos)

    # --- 3. Initial denser mask (Phase 1: 50% sparsity) ---
    # Starting at 50% density with 10 cycles at 15% removal:
    # 0.50 * 0.85^10 ≈ 0.098 → ~90% sparsity naturally
    initial_sparsity = 0.50
    connection_prob = 1.0 / (distances ** distance_exponent + epsilon)
    target_density = 1.0 - initial_sparsity
    scale = target_density / connection_prob.mean()
    connection_prob = np.clip(connection_prob * scale, 0, 1)
    initial_mask = (rng.random((n_out, n_in)) < connection_prob).astype(np.float32)

    # --- 4. Repair initial connectivity ---
    initial_mask, n_init_repaired = repair_connectivity(
        initial_mask, distances, min_fan_in=1, min_fan_out=1)

    # --- 5. Random developmental weights ---
    dev_weights = initial_mask * np.abs(
        rng.normal(0, 1, initial_mask.shape)).astype(np.float32)

    # --- 6. Connectivity-preserving developmental pruning ---
    result = developmental_pruning_connected(
        dev_weights, initial_mask, distances,
        n_cycles=n_dev_cycles,
        decay_rate=0.05,
        reinforcement_total=1.0,
        pareto_fraction=0.2,
        prune_fraction=0.15,
        steepness=10.0,
        seed=seed + 500,
        min_fan_in=1,
        min_fan_out=1,
        check_components=True,
    )
    dev_mask = result["mask"]

    # --- 7. Connectivity-safe sparsity adjustment ---
    n_total = n_out * n_in
    target_active = int((1.0 - target_sparsity) * n_total)
    current_active = int(dev_mask.sum())

    if current_active > target_active:
        # Too dense: remove farthest edges, but skip if it would orphan a node
        ones = np.argwhere(dev_mask > 0)
        one_dists = np.array([distances[r, c] for r, c in ones])
        remove_order = np.argsort(-one_dists)  # farthest first
        n_removed = 0
        for idx in remove_order:
            if n_removed >= (current_active - target_active):
                break
            r, c = ones[idx]
            if dev_mask[r].sum() > 1 and dev_mask[:, c].sum() > 1:
                dev_mask[r, c] = 0.0
                n_removed += 1

    elif current_active < target_active:
        # Too sparse: add closest edges
        zeros = np.argwhere(dev_mask == 0)
        zero_dists = np.array([distances[r, c] for r, c in zeros])
        n_add = target_active - current_active
        add_order = np.argsort(zero_dists)[:n_add]
        for i in add_order:
            dev_mask[zeros[i][0], zeros[i][1]] = 1.0

    # --- 8. Fresh Kaiming weights (topology-only principle) ---
    if return_weights:
        active_per_row = dev_mask.sum(axis=1)
        mean_fan_in = max(float(active_per_row.mean()), 1.0)
        target_std = np.sqrt(2.0 / mean_fan_in)
        init_weights = rng.normal(0, target_std, (n_out, n_in)).astype(np.float32)
        init_weights *= dev_mask
        return dev_mask, init_weights

    return dev_mask



# ═══════════════════════════════════════════════════════════════════════════
# Mask generation dispatcher (one function per condition)
# ═══════════════════════════════════════════════════════════════════════════

def make_masks_for_condition(condition_name, seed, input_pos, target_sparsity=0.9):
    """Generate both layers' masks and weights for a given experimental condition.

    All conditions produce the same shape outputs:
        masks  = [mask_layer1 (256,784), mask_layer2 (10,256)]
        weights = [w_layer1   (256,784), w_layer2   (10,256)]

    The only variable is HOW the mask topology is generated.

    Args:
        condition_name: One of 'bio_developmental', 'spatial_proximity',
            'lambda_0.9', 'lambda_0.5', 'lambda_0.0'.
        seed: Base random seed.  Layer 1 uses seed*1000, layer 2 uses
            seed*1000+100  (matches paper2/run.py convention).
        input_pos: (784, 2) pixel-grid positions for the input layer.
        target_sparsity: Fraction of zero connections.

    Returns:
        masks: list of 2 numpy arrays.
        weights: list of 2 numpy arrays.
        metrics: dict of topology metrics for layer 1 mask.
    """
    seed1 = seed * 1000
    seed2 = seed * 1000 + 100

    if condition_name == "bio_developmental":
        mask1, w1 = generate_bio_developmental_mask(
            256, 784,
            target_sparsity=target_sparsity,
            input_positions=input_pos,
            seed=seed1,
        )
        # Layer 2: no special positions (hidden→output, abstract)
        mask2, w2 = generate_bio_developmental_mask(
            10, 256,
            target_sparsity=target_sparsity,
            input_positions=None,
            seed=seed2,
        )

    elif condition_name == "spatial_proximity":
        # Pure inverse-square distance, balanced fan-in, NO developmental
        # cycles.  This isolates the spatial embedding contribution from
        # the developmental dynamics.
        mask1, w1 = generate_distance_pruned_mask(
            256, 784,
            target_sparsity=target_sparsity,
            weight_variability=0.0,
            balanced=True,
            input_positions=input_pos,
            seed=seed1,
            return_weights=True,
        )
        mask2, w2 = generate_distance_pruned_mask(
            10, 256,
            target_sparsity=target_sparsity,
            weight_variability=0.0,
            balanced=True,
            seed=seed2,
            return_weights=True,
        )

    elif condition_name.startswith("lambda_"):
        locality = float(condition_name.split("_")[1])
        mask1, w1 = generate_locality_mask(
            256, 784,
            target_sparsity=target_sparsity,
            locality=locality,
            input_positions=input_pos,
            seed=seed1,
            return_weights=True,
        )
        mask2, w2 = generate_locality_mask(
            10, 256,
            target_sparsity=target_sparsity,
            locality=locality,
            seed=seed2,
            return_weights=True,
        )

    else:
        raise ValueError(f"Unknown condition: {condition_name}")

    # Topology metrics for layer 1 (projected, not raw bipartite)
    metrics = compute_projected_metrics(
        mask1, project_to="output", max_nodes_for_expensive=600)

    return [mask1, mask2], [w1, w2], metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main experiment
# ═══════════════════════════════════════════════════════════════════════════

# Condition display names for the summary table
CONDITION_LABELS = {
    "bio_developmental": "Bio-Inspired (dev)",
    "spatial_proximity": "Spatial proximity",
    "lambda_0.9":        "λ=0.9 (mostly local)",
    "lambda_0.5":        "λ=0.5 (half-half)",
    "lambda_0.0":        "λ=0.0 (random/ER)",
}

CONDITION_ORDER = [
    "bio_developmental",
    "spatial_proximity",
    "lambda_0.9",
    "lambda_0.5",
    "lambda_0.0",
]


def run_bio_comparison(
    bandwidths=None,
    n_seeds=3,
    epochs=20,
    device="mps",
    results_dir=None,
    target_sparsity=0.9,
    fresh=False,
):
    """Run the Bio-Inspired vs λ-sweep comparison experiment.

    **Disconnect-safe:** after each individual (bandwidth, condition, seed)
    run, results are checkpointed atomically to disk.  On restart, completed
    runs are loaded from the checkpoint and skipped.  A disconnect only loses
    the single run that was in progress — everything else is preserved.

    Args:
        bandwidths: List of bandwidth_pixels values (default: [16, 784]).
        n_seeds: Random seeds per condition.
        epochs: Training epochs.
        device: PyTorch device string.
        results_dir: Directory for JSON output (default: paper2/results/).
        target_sparsity: Network sparsity level.
        fresh: If True, discard any existing checkpoint and start over.

    Returns:
        results: Nested dict  {bw_label → {condition → {mean, std, ...}}}.
    """
    if bandwidths is None:
        bandwidths = [16, 784]
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, "bio_comparison_checkpoint.json")
    final_path = os.path.join(results_dir, "bio_comparison.json")

    # ── Resume or fresh start ──────────────────────────────────────────
    if fresh and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("Cleared previous checkpoint (--fresh).\n")

    checkpoint = load_checkpoint(checkpoint_path)
    completed_runs = checkpoint.get("runs", {})

    # Warn if config changed since last checkpoint
    old_cfg = checkpoint.get("config", {})
    if old_cfg and not fresh:
        if old_cfg.get("epochs") and old_cfg["epochs"] != epochs:
            print(f"⚠ WARNING: Checkpoint used {old_cfg['epochs']} epochs, "
                  f"current run uses {epochs}.")
            print(f"  Cached runs will NOT be re-run with the new epoch count.")
            print(f"  Use --fresh to discard and restart.\n")

    # Enumerate all runs for this experiment
    all_run_keys = []
    for bw in bandwidths:
        for cond in CONDITION_ORDER:
            for seed in range(n_seeds):
                all_run_keys.append(make_run_key(bw, cond, seed))

    total_runs = len(all_run_keys)
    already_done = sum(1 for k in all_run_keys if k in completed_runs)

    if already_done == total_runs:
        print(f"All {total_runs} runs already completed. Jumping to summary.\n")
    elif already_done > 0:
        print(f"Resuming: {already_done}/{total_runs} runs already completed.")
        print(f"  (Use --fresh to discard and restart.)\n")

    # ── Data loading ───────────────────────────────────────────────────
    input_pos = make_grid_positions(784, 28)

    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    t_global = time.time()
    run_idx = 0

    for bw_pixels in bandwidths:
        bw_label = f"{bw_pixels}px"
        print(f"\n{'=' * 70}")
        print(f"BANDWIDTH: {bw_label}  |  {n_seeds} seeds × {epochs} epochs")
        print(f"{'=' * 70}")

        train_loader, val_loader, test_loader = get_dataloaders(
            X_train, y_train, X_test, y_test, bw_pixels)

        for cond in CONDITION_ORDER:
            for seed in range(n_seeds):
                run_idx += 1
                run_key = make_run_key(bw_pixels, cond, seed)
                label = CONDITION_LABELS[cond]

                # ── Skip completed runs ────────────────────────────
                if run_key in completed_runs:
                    cached = completed_runs[run_key]
                    print(f"  [{run_idx:>2}/{total_runs}] {label:<25s} "
                          f"seed={seed} — cached "
                          f"(acc={cached['test_acc']:.4f}) ✓")
                    continue

                # ── Run this triple ────────────────────────────────
                print(f"  [{run_idx:>2}/{total_runs}] {label:<25s} "
                      f"seed={seed} — running...", end="", flush=True)
                t_run = time.time()

                masks, weights, metrics = make_masks_for_condition(
                    cond, seed, input_pos, target_sparsity)

                total_params = sum(m.size for m in masks)
                total_active = sum(m.sum() for m in masks)
                actual_sparsity = 1.0 - total_active / total_params

                model = SparseMLP(
                    [784, 256, 10], masks,
                    initial_weights=weights)
                trainer = Trainer(
                    model, train_loader, val_loader, test_loader,
                    device=device, lr=0.001,
                    track_topology_every=epochs)
                history = trainer.train(epochs, verbose=False)
                test_acc = history.get("test_acc", 0)

                run_elapsed = time.time() - t_run

                # ── Save this run to checkpoint (atomic) ───────────
                completed_runs[run_key] = {
                    "test_acc": float(test_acc),
                    "actual_sparsity": float(actual_sparsity),
                    "clustering": float(metrics.get("clustering_local", 0)),
                    "modularity": float(metrics.get("modularity", 0)),
                    "elapsed_seconds": round(run_elapsed, 1),
                }

                checkpoint["runs"] = completed_runs
                checkpoint["config"] = {
                    "conditions": CONDITION_ORDER,
                    "bandwidths": bandwidths,
                    "n_seeds": n_seeds,
                    "epochs": epochs,
                    "target_sparsity": target_sparsity,
                    "architecture": [784, 256, 10],
                }
                atomic_save(checkpoint, checkpoint_path)

                print(f" acc={test_acc:.4f} ({run_elapsed:.0f}s) — saved ✓")

    # ── Aggregate into final results format ────────────────────────────
    elapsed_total = time.time() - t_global
    results = _aggregate_results(
        completed_runs, bandwidths, n_seeds, epochs,
        target_sparsity, elapsed_total)

    atomic_save(results, final_path)
    print(f"\nFinal results saved to {final_path}")

    # ── Summary tables ─────────────────────────────────────────────────
    _print_summary(results, bandwidths, elapsed_total)

    return results


def _aggregate_results(completed_runs, bandwidths, n_seeds, epochs,
                       target_sparsity, elapsed_total):
    """Convert flat checkpoint runs → nested {bw → {cond → stats}} format.

    This is called once at the end to produce the clean final JSON that
    matches the format expected by downstream analysis notebooks.
    """
    results = {}

    for bw_pixels in bandwidths:
        bw_label = f"{bw_pixels}px"
        bw_results = {}

        for cond in CONDITION_ORDER:
            accs = []
            clustering = 0.0
            modularity = 0.0
            actual_sparsity = 0.0

            for seed in range(n_seeds):
                run_key = make_run_key(bw_pixels, cond, seed)
                run_data = completed_runs.get(run_key, {})
                accs.append(run_data.get("test_acc", 0.0))
                # Use seed 0's topology metrics (topology is seed-dependent
                # but we only need one representative per condition)
                if seed == 0:
                    clustering = run_data.get("clustering", 0.0)
                    modularity = run_data.get("modularity", 0.0)
                    actual_sparsity = run_data.get("actual_sparsity", 0.0)

            bw_results[cond] = {
                "label": CONDITION_LABELS[cond],
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "individual": [float(a) for a in accs],
                "clustering": float(clustering),
                "modularity": float(modularity),
                "actual_sparsity": float(actual_sparsity),
            }

        results[bw_label] = bw_results

    results["_meta"] = {
        "conditions": CONDITION_ORDER,
        "bandwidths": bandwidths,
        "n_seeds": n_seeds,
        "epochs": epochs,
        "target_sparsity": target_sparsity,
        "architecture": [784, 256, 10],
        "elapsed_seconds": round(elapsed_total, 1),
    }

    return results


def _print_summary(results, bandwidths, elapsed_total):
    """Print the final comparison tables to stdout."""
    print(f"\n{'=' * 70}")
    print(f"SUMMARY TABLE  ({elapsed_total:.0f}s total)")
    print(f"{'=' * 70}")
    print(f"{'Condition':<25s}", end="")
    for bw in bandwidths:
        print(f"  {bw}px{'':>16}", end="")
    print()
    print("-" * (25 + 22 * len(bandwidths)))

    for cond in CONDITION_ORDER:
        label = CONDITION_LABELS[cond]
        row = f"{label:<25s}"
        for bw in bandwidths:
            bw_label = f"{bw}px"
            r = results[bw_label][cond]
            row += f"  {r['mean']:.4f}±{r['std']:.4f}       "
        print(row)

    # ── Topology metrics table ─────────────────────────────────────────
    print(f"\n{'Condition':<25s}  {'Clustering':<12s}  {'Modularity':<12s}")
    print("-" * 53)
    first_bw = f"{bandwidths[0]}px"
    for cond in CONDITION_ORDER:
        r = results[first_bw][cond]
        label = CONDITION_LABELS[cond]
        print(f"{label:<25s}  {r['clustering']:<12.4f}  {r['modularity']:<12.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper 2: Bio-Inspired topology vs λ-sweep comparison")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per run (default: 20)")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Random seeds per condition (default: 3)")
    parser.add_argument("--bandwidths", type=int, nargs="+", default=[16, 784],
                        help="Bandwidth levels in pixels (default: 16 784)")
    parser.add_argument("--device", default="auto",
                        help="PyTorch device (default: auto-detect)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke-test: 1 seed, 10 epochs")
    parser.add_argument("--fresh", action="store_true",
                        help="Discard any checkpoint and start from scratch")
    args = parser.parse_args()

    # Auto-detect device (CUDA first for RunPod / cloud GPUs)
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print("Paper 2 — Bio-Inspired vs λ-Sweep Comparison")
    print(f"Device: {device}")
    print(f"Conditions: {', '.join(CONDITION_LABELS[c] for c in CONDITION_ORDER)}")

    if args.quick:
        print("MODE: quick smoke-test (1 seed, 10 epochs)\n")
        run_bio_comparison(
            bandwidths=args.bandwidths,
            n_seeds=1,
            epochs=10,
            device=device,
            fresh=args.fresh,
        )
    else:
        print()
        run_bio_comparison(
            bandwidths=args.bandwidths,
            n_seeds=args.seeds,
            epochs=args.epochs,
            device=device,
            fresh=args.fresh,
        )
