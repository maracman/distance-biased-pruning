"""
Topology tracking during training.

Extracts graph-theoretic metrics from sparse neural network weights at
regular intervals during training. Enables analysis of how network topology
evolves under different conditions (clean vs degraded input, different
initialization topologies, pruning interventions).

Tracks clustering coefficient, small-worldedness, number of clusters,
and communicability across training phases to characterize how topology
responds to signal degradation and pruning interventions.
"""

import numpy as np
import pandas as pd
from typing import Optional
import torch

from ..topology.metrics import compute_topology_metrics, compute_multilayer_metrics, compute_projected_metrics


class TopologyTracker:
    """Track topology metrics across training epochs.

    Uses full multi-layer graph metrics by default, since single-layer
    bipartite graphs always have clustering=0 (no triangles possible).
    The full network graph (input→hidden1→hidden2→output) can form
    triangles through skip-like paths and thus gives meaningful clustering,
    small-world, and modularity measurements.
    """

    def __init__(self, compute_expensive: bool = True, use_multilayer: bool = True):
        self.snapshots = []
        self.compute_expensive = compute_expensive
        self.use_multilayer = use_multilayer

    def snapshot(self, model, epoch: int, label: str = ""):
        """Take a topology snapshot of the current model state.

        Uses projected (functional connectivity) graph metrics for meaningful
        clustering values. Direct bipartite layer graphs always have
        clustering=0 since no triangles can form in a bipartite graph.
        """
        from ..models.sparse_linear import SparseLinear

        sparse_layers = [m for m in model.modules() if isinstance(m, SparseLinear)]

        # Use first (largest) hidden layer for projected metrics
        # This gives the most informative topology snapshot
        if sparse_layers:
            layer = sparse_layers[0]
            mask = layer.get_mask_numpy()
            weights = layer.get_weights_numpy()

            metrics = compute_projected_metrics(
                mask, weights, project_to="output",
                max_nodes_for_expensive=0,  # skip expensive for speed
                projection_density=0.1,
            )
            metrics["epoch"] = epoch
            metrics["layer"] = "projected"
            metrics["label"] = label

            # Weight dynamics across all layers
            all_active = []
            total_params = 0
            total_active = 0
            for l in sparse_layers:
                m = l.get_mask_numpy()
                w = l.get_weights_numpy()
                all_active.extend(np.abs(w[m > 0]).tolist())
                total_params += m.size
                total_active += np.count_nonzero(m)

            if all_active:
                active_arr = np.array(all_active)
                metrics["weight_gini"] = self._gini_coefficient(active_arr)
                metrics["weight_entropy"] = self._weight_entropy(active_arr)

            metrics["sparsity"] = 1.0 - total_active / total_params if total_params > 0 else 0.0

            self.snapshots.append(metrics)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all snapshots to a pandas DataFrame for analysis."""
        return pd.DataFrame(self.snapshots)

    def get_metric_trajectory(self, metric: str, layer: int = 0) -> dict:
        """Get a single metric's trajectory over epochs for one layer."""
        df = self.to_dataframe()
        layer_df = df[df["layer"] == layer]
        return {
            "epochs": layer_df["epoch"].tolist(),
            "values": layer_df[metric].tolist(),
            "label": layer_df["label"].iloc[0] if len(layer_df) > 0 else "",
        }

    @staticmethod
    def _gini_coefficient(values: np.ndarray) -> float:
        """Gini coefficient of weight distribution (0=equal, 1=maximally unequal).
        High Gini indicates Pareto-like distribution (few strong, many weak)."""
        sorted_vals = np.sort(values)
        n = len(sorted_vals)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_vals) / (n * np.sum(sorted_vals))) - (n + 1) / n

    @staticmethod
    def _weight_entropy(values: np.ndarray, n_bins: int = 50) -> float:
        """Shannon entropy of the weight distribution."""
        hist, _ = np.histogram(values, bins=n_bins, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        return -np.sum(hist * np.log2(hist + 1e-10))


def compare_topology_trajectories(
    trajectories: dict, metric: str
) -> pd.DataFrame:
    """Compare topology metric trajectories across conditions.

    Args:
        trajectories: Dict mapping condition_name -> TopologyTracker.
        metric: Metric name to compare.

    Returns:
        DataFrame with columns [epoch, condition, value].
    """
    rows = []
    for condition, tracker in trajectories.items():
        df = tracker.to_dataframe()
        for _, row in df.iterrows():
            if metric in row and row[metric] is not None:
                rows.append({
                    "epoch": row["epoch"],
                    "condition": condition,
                    "layer": row["layer"],
                    "value": row[metric],
                })
    return pd.DataFrame(rows)
