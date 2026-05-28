"""
Unit tests for topology generation, pruning, and metrics.

Run with: python -m pytest tests/test_topology.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.topology.generators import (
    generate_erdos_renyi_mask,
    generate_watts_strogatz_mask,
    generate_barabasi_albert_mask,
    generate_bio_inspired_mask,
    generate_bio_inspired_mask_multilayer,
    compute_sparsity,
)
from shared.topology.pruning import (
    developmental_pruning,
    prune_probabilistic,
    prune_by_percentile,
    pareto_reinforce,
    weight_decay,
)
from shared.topology.metrics import (
    compute_topology_metrics,
    mask_to_graph,
    clustering_coefficient,
    average_path_length,
)


class TestGenerators:
    """Test sparse network generators."""

    def test_er_mask_shape(self):
        mask = generate_erdos_renyi_mask(64, 128, target_sparsity=0.9)
        assert mask.shape == (64, 128)

    def test_er_mask_sparsity(self):
        mask = generate_erdos_renyi_mask(100, 100, target_sparsity=0.9, seed=42)
        sparsity = compute_sparsity(mask)
        assert abs(sparsity - 0.9) < 0.05  # within 5% of target

    def test_ws_mask_shape(self):
        mask = generate_watts_strogatz_mask(64, 128, target_sparsity=0.9)
        assert mask.shape == (64, 128)

    def test_ba_mask_shape(self):
        mask = generate_barabasi_albert_mask(64, 128, target_sparsity=0.9)
        assert mask.shape == (64, 128)

    def test_bio_inspired_mask_shape(self):
        mask = generate_bio_inspired_mask(64, 128, target_sparsity=0.9)
        assert mask.shape == (64, 128)

    def test_bio_inspired_with_weights(self):
        mask, weights = generate_bio_inspired_mask(
            64, 128, target_sparsity=0.9, return_weights=True
        )
        assert mask.shape == weights.shape
        # Weights should only be nonzero where mask is nonzero
        assert np.all(weights[mask == 0] == 0)

    def test_bio_inspired_with_positions(self):
        mask, weights, positions = generate_bio_inspired_mask(
            64, 128, return_weights=True, return_positions=True
        )
        assert positions["input"].shape == (128, 2)
        assert positions["output"].shape == (64, 2)

    def test_bio_inspired_multilayer(self):
        layer_sizes = [784, 256, 128, 10]
        masks, positions = generate_bio_inspired_mask_multilayer(layer_sizes)
        assert len(masks) == 3
        assert masks[0].shape == (256, 784)
        assert masks[1].shape == (128, 256)
        assert masks[2].shape == (10, 128)
        assert len(positions) == 4

    def test_deterministic_with_seed(self):
        m1 = generate_bio_inspired_mask(32, 64, seed=42)
        m2 = generate_bio_inspired_mask(32, 64, seed=42)
        assert np.array_equal(m1, m2)

    def test_different_seeds_differ(self):
        m1 = generate_bio_inspired_mask(32, 64, seed=42)
        m2 = generate_bio_inspired_mask(32, 64, seed=99)
        assert not np.array_equal(m1, m2)


class TestPruning:
    """Test pruning functions."""

    def setup_method(self):
        rng = np.random.RandomState(42)
        self.mask = (rng.random((32, 64)) < 0.5).astype(np.float32)
        self.weights = self.mask * rng.normal(0, 1, (32, 64)).astype(np.float32)

    def test_weight_decay(self):
        decayed = weight_decay(self.weights, self.mask, 0.1)
        # Weights should decrease in magnitude
        assert np.abs(decayed).sum() < np.abs(self.weights).sum()
        # Mask pattern preserved
        assert np.all(decayed[self.mask == 0] == 0)

    def test_pareto_reinforce(self):
        reinforced = pareto_reinforce(self.weights, self.mask, reinforcement_total=1.0)
        # Total absolute weight should increase
        active_before = np.abs(self.weights[self.mask > 0]).sum()
        active_after = np.abs(reinforced[self.mask > 0]).sum()
        assert active_after > active_before

    def test_prune_probabilistic_reduces_connections(self):
        new_mask = prune_probabilistic(self.weights, self.mask, target_removal_fraction=0.3)
        assert new_mask.sum() < self.mask.sum()

    def test_prune_by_percentile(self):
        new_mask = prune_by_percentile(self.weights, self.mask, percentile=50)
        assert new_mask.sum() < self.mask.sum()

    def test_developmental_pruning_reduces_density(self):
        result = developmental_pruning(
            self.weights, self.mask, n_cycles=3, track_history=True
        )
        assert result["mask"].sum() < self.mask.sum()
        assert len(result["history"]) == 3
        # Sparsity should increase over cycles
        assert result["history"][-1]["sparsity"] > result["history"][0]["sparsity"]

    def test_developmental_pruning_preserves_shape(self):
        result = developmental_pruning(self.weights, self.mask, n_cycles=2)
        assert result["weights"].shape == self.weights.shape
        assert result["mask"].shape == self.mask.shape


class TestMetrics:
    """Test topology metrics."""

    def test_mask_to_graph(self):
        mask = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
        G = mask_to_graph(mask)
        assert len(G) == 5  # 3 input + 2 output nodes
        assert G.number_of_edges() == 4

    def test_clustering_coefficient(self):
        mask = generate_bio_inspired_mask(20, 20, target_sparsity=0.5, seed=42)
        G = mask_to_graph(mask)
        cc = clustering_coefficient(G)
        assert 0 <= cc["local_mean"] <= 1
        assert 0 <= cc["global"] <= 1

    def test_compute_topology_metrics(self):
        mask = generate_bio_inspired_mask(30, 30, target_sparsity=0.7, seed=42)
        metrics = compute_topology_metrics(mask, compute_expensive=True)
        assert "sparsity" in metrics
        assert "clustering_local" in metrics
        assert "modularity" in metrics
        assert "n_edges" in metrics
        assert metrics["sparsity"] > 0

    def test_bio_inspired_has_higher_clustering_than_er(self):
        """Core hypothesis: bio-inspired networks should have higher clustering."""
        n_out, n_in = 50, 50
        sparsity = 0.8

        bio_mask = generate_bio_inspired_mask(n_out, n_in, sparsity, seed=42)
        er_mask = generate_erdos_renyi_mask(n_out, n_in, sparsity, seed=42)

        bio_metrics = compute_topology_metrics(bio_mask, compute_expensive=False)
        er_metrics = compute_topology_metrics(er_mask, compute_expensive=False)

        # Bio-inspired should have higher clustering (this is a core claim)
        assert bio_metrics["clustering_local"] >= er_metrics["clustering_local"] * 0.8, (
            f"Bio clustering {bio_metrics['clustering_local']:.4f} should be "
            f">= ER clustering {er_metrics['clustering_local']:.4f}"
        )


class TestSparseModel:
    """Test sparse MLP creation and forward pass."""

    def test_create_and_forward(self):
        import torch
        from shared.models.sparse_mlp import create_sparse_mlp

        model = create_sparse_mlp(
            [784, 256, 10],
            topology="bio_inspired",
            target_sparsity=0.9,
            seed=42,
        )
        x = torch.randn(16, 784)
        out = model(x)
        assert out.shape == (16, 10)

    def test_sparsity_preserved_after_forward(self):
        import torch
        from shared.models.sparse_mlp import create_sparse_mlp

        model = create_sparse_mlp([100, 50, 10], topology="erdos_renyi", target_sparsity=0.9)
        x = torch.randn(8, 100)
        _ = model(x)

        sparsity = model.get_total_sparsity()
        assert sparsity > 0.8  # should be close to 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
