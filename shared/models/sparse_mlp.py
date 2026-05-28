"""
Multi-layer perceptron with sparse topology initialization.

Supports arbitrary layer sizes with per-layer sparsity masks generated
by any topology generator. The key abstraction: topology (which connections
exist) is separated from weight initialization (what values those connections
take).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .sparse_linear import SparseLinear


class SparseMLP(nn.Module):
    """MLP with fixed-topology sparse layers."""

    def __init__(
        self,
        layer_sizes: list,
        masks: list,
        initial_weights: Optional[list] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """
        Args:
            layer_sizes: [input_dim, hidden1, ..., output_dim]
            masks: List of (layer_sizes[i+1], layer_sizes[i]) binary masks.
            initial_weights: Optional list of initial weight arrays.
            activation: Activation function name.
            dropout: Dropout rate between layers.
        """
        super().__init__()
        assert len(masks) == len(layer_sizes) - 1

        self.layer_sizes = layer_sizes
        self.n_layers = len(masks)

        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = []
        for i, mask in enumerate(masks):
            w = initial_weights[i] if initial_weights else None
            layers.append(SparseLinear(
                layer_sizes[i], layer_sizes[i + 1], mask,
                bias=True, initial_weights=w,
            ))
            if i < len(masks) - 1:  # no activation/dropout after last layer
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def get_sparse_layers(self) -> list:
        """Return all SparseLinear layers."""
        return [m for m in self.network if isinstance(m, SparseLinear)]

    def get_all_masks(self) -> list:
        """Return masks for all sparse layers."""
        return [layer.get_mask_numpy() for layer in self.get_sparse_layers()]

    def get_all_weights(self) -> list:
        """Return weight matrices for all sparse layers."""
        return [layer.get_weights_numpy() for layer in self.get_sparse_layers()]

    def get_total_sparsity(self) -> float:
        """Overall sparsity across all layers."""
        total_params = 0
        total_active = 0
        for layer in self.get_sparse_layers():
            total_params += layer.mask.numel()
            total_active += layer.mask.sum().item()
        return 1.0 - total_active / total_params if total_params > 0 else 0.0

    def apply_pruning_to_layer(self, layer_idx: int, new_mask: np.ndarray):
        """Apply additional pruning to a specific layer."""
        sparse_layers = self.get_sparse_layers()
        sparse_layers[layer_idx].apply_additional_pruning(new_mask)

    def get_layer_activations(self, x: torch.Tensor) -> list:
        """Forward pass returning intermediate activations for RSA."""
        activations = [x]
        for module in self.network:
            x = module(x)
            if isinstance(module, SparseLinear):
                activations.append(x)
        return activations


def create_sparse_mlp(
    layer_sizes: list,
    topology: str = "bio_inspired",
    target_sparsity: float = 0.9,
    distance_exponent: float = 2.0,
    n_dev_cycles: int = 5,
    seed: int = 42,
    activation: str = "relu",
    dropout: float = 0.0,
) -> SparseMLP:
    """Factory function to create a SparseMLP with specified topology.

    Args:
        layer_sizes: [input_dim, hidden1, ..., output_dim]
        topology: One of 'bio_inspired', 'erdos_renyi', 'watts_strogatz', 'barabasi_albert'
        target_sparsity: Target sparsity level.
        distance_exponent: For bio_inspired topology.
        n_dev_cycles: Number of developmental pruning cycles (bio_inspired only).
        seed: Random seed.

    Returns:
        SparseMLP with the specified topology.
    """
    from ..topology.generators import (
        generate_erdos_renyi_mask,
        generate_watts_strogatz_mask,
        generate_barabasi_albert_mask,
        generate_bio_inspired_mask_multilayer,
    )
    from ..topology.pruning import developmental_pruning

    if topology == "bio_inspired":
        # Generate with lower sparsity, then prune to target
        initial_sparsity = max(0.0, target_sparsity - 0.3)
        masks, positions = generate_bio_inspired_mask_multilayer(
            layer_sizes, initial_sparsity, distance_exponent, seed=seed
        )

        # Apply developmental pruning to each layer
        pruned_masks = []
        for i, mask in enumerate(masks):
            rng = np.random.RandomState(seed + i)
            weights = mask * np.abs(rng.normal(0, 1, mask.shape)).astype(np.float32)
            result = developmental_pruning(
                weights, mask,
                n_cycles=n_dev_cycles,
                seed=seed + i * 1000,
            )
            pruned_masks.append(result["mask"])
        masks = pruned_masks

    elif topology == "erdos_renyi":
        masks = []
        for i in range(len(layer_sizes) - 1):
            m = generate_erdos_renyi_mask(
                layer_sizes[i + 1], layer_sizes[i], target_sparsity, seed + i
            )
            masks.append(m)

    elif topology == "watts_strogatz":
        masks = []
        for i in range(len(layer_sizes) - 1):
            m = generate_watts_strogatz_mask(
                layer_sizes[i + 1], layer_sizes[i], target_sparsity, seed=seed + i
            )
            masks.append(m)

    elif topology == "barabasi_albert":
        masks = []
        for i in range(len(layer_sizes) - 1):
            m = generate_barabasi_albert_mask(
                layer_sizes[i + 1], layer_sizes[i], target_sparsity, seed=seed + i
            )
            masks.append(m)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    return SparseMLP(layer_sizes, masks, activation=activation, dropout=dropout)
