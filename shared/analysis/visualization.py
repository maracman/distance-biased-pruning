"""
Publication-quality visualization for NeurIPS figures.

Generates figures for:
1. Topology metric comparisons across initializations
2. Training dynamics (loss/accuracy curves)
3. Topology evolution during training (signal degradation experiment)
4. RSA heatmaps and RDM comparisons
5. Weight/degree distribution comparisons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional

# NeurIPS style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

TOPOLOGY_COLORS = {
    "bio_inspired": "#2E86AB",
    "erdos_renyi": "#A23B72",
    "watts_strogatz": "#F18F01",
    "barabasi_albert": "#C73E1D",
    "lottery_ticket": "#6B4226",
    "dense": "#888888",
}

TOPOLOGY_LABELS = {
    "bio_inspired": "Bio-Inspired (Ours)",
    "erdos_renyi": "Erdős-Rényi",
    "watts_strogatz": "Watts-Strogatz",
    "barabasi_albert": "Barabási-Albert",
    "lottery_ticket": "Lottery Ticket",
    "dense": "Dense",
}


def plot_training_curves(
    results: dict, save_path: Optional[str] = None, title: str = ""
):
    """Plot training/validation loss and accuracy for multiple models.

    Args:
        results: Dict mapping model_name -> training history dict.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for name, hist in results.items():
        color = TOPOLOGY_COLORS.get(name, "#333333")
        label = TOPOLOGY_LABELS.get(name, name)

        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], color=color, alpha=0.4, linewidth=0.8)
        axes[0].plot(epochs, hist["val_loss"], color=color, label=label, linewidth=1.5)

        axes[1].plot(epochs, hist["train_acc"], color=color, alpha=0.4, linewidth=0.8)
        axes[1].plot(epochs, hist["val_acc"], color=color, label=label, linewidth=1.5)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(frameon=False)

    if title:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_topology_evolution(
    tracker_df: pd.DataFrame,
    metrics: list = None,
    save_path: Optional[str] = None,
):
    """Plot topology metrics over training epochs for multiple conditions.

    Args:
        tracker_df: DataFrame from TopologyTracker.to_dataframe() or
            compare_topology_trajectories().
        metrics: List of metric names to plot. Defaults to key metrics.
    """
    if metrics is None:
        metrics = ["clustering_local", "modularity", "sparsity", "degree_mean"]

    available = [m for m in metrics if m in tracker_df.columns]
    n_metrics = len(available)
    if n_metrics == 0:
        return None

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3.5))
    if n_metrics == 1:
        axes = [axes]

    conditions = tracker_df["label"].unique() if "label" in tracker_df.columns else [""]

    for idx, metric in enumerate(available):
        ax = axes[idx]
        for cond in conditions:
            if "label" in tracker_df.columns:
                subset = tracker_df[tracker_df["label"] == cond]
            else:
                subset = tracker_df
            color = TOPOLOGY_COLORS.get(cond, None)
            label = TOPOLOGY_LABELS.get(cond, cond)
            ax.plot(
                subset["epoch"], subset[metric],
                label=label, color=color, linewidth=1.5, marker="o", markersize=3,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_topology_comparison_bar(
    metrics_dict: dict, save_path: Optional[str] = None
):
    """Bar chart comparing topology metrics across initialization methods.

    Args:
        metrics_dict: Dict mapping topology_name -> metrics dict.
    """
    metric_names = ["clustering_local", "modularity", "small_world_sigma", "degree_std"]
    metric_labels = ["Clustering\nCoefficient", "Modularity\n(Q)", "Small-World\n(σ)", "Degree\nStd Dev"]

    available = []
    for m in metric_names:
        if all(m in d and d[m] is not None for d in metrics_dict.values()):
            available.append(m)

    if not available:
        return None

    labels = [metric_labels[metric_names.index(m)] for m in available]
    n_metrics = len(available)
    n_methods = len(metrics_dict)
    x = np.arange(n_metrics)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, (name, metrics) in enumerate(metrics_dict.items()):
        values = [metrics.get(m, 0) or 0 for m in available]
        color = TOPOLOGY_COLORS.get(name, "#333333")
        label = TOPOLOGY_LABELS.get(name, name)
        ax.bar(x + i * width, values, width, color=color, label=label, alpha=0.85)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Network Topology Properties at Initialization")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_signal_degradation_effects(
    noise_levels: list,
    metrics_by_noise: dict,
    metric_name: str = "clustering_local",
    save_path: Optional[str] = None,
):
    """Plot how a topology metric changes as a function of noise level.

    Args:
        noise_levels: List of noise levels tested.
        metrics_by_noise: Dict mapping topology_name -> list of metric values
            (one per noise level).
        metric_name: Name of the metric being plotted.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for name, values in metrics_by_noise.items():
        color = TOPOLOGY_COLORS.get(name, "#333333")
        label = TOPOLOGY_LABELS.get(name, name)
        ax.plot(noise_levels, values, "o-", color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Noise Level (Signal Degradation)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Effect of Signal Degradation on {metric_name.replace('_', ' ').title()}")
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_rsa_matrix(
    rsa_matrix: np.ndarray,
    model_names: list,
    save_path: Optional[str] = None,
):
    """Plot RSA correlation matrix as heatmap."""
    labels = [TOPOLOGY_LABELS.get(n, n) for n in model_names]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        rsa_matrix, xticklabels=labels, yticklabels=labels,
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        vmin=-1, vmax=1, ax=ax, square=True,
    )
    ax.set_title("Representational Similarity (RSA)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


def plot_experiment_results(
    results: dict, experiment_name: str, save_dir: str = "figures"
):
    """Generate all figures for a given experiment."""
    import os
    os.makedirs(save_dir, exist_ok=True)

    if "training_histories" in results:
        plot_training_curves(
            results["training_histories"],
            save_path=os.path.join(save_dir, f"{experiment_name}_training.pdf"),
        )

    if "topology_metrics" in results:
        plot_topology_comparison_bar(
            results["topology_metrics"],
            save_path=os.path.join(save_dir, f"{experiment_name}_topology.pdf"),
        )

    if "rsa" in results:
        plot_rsa_matrix(
            results["rsa"]["rsa_matrix"],
            results["rsa"]["model_names"],
            save_path=os.path.join(save_dir, f"{experiment_name}_rsa.pdf"),
        )
