"""
Baseline models for comparison.

Dense MLP and Lottery Ticket (magnitude pruning after training) baselines.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from .sparse_linear import SparseLinear
from .sparse_mlp import SparseMLP


class DenseMLP(nn.Module):
    """Standard dense MLP baseline."""

    def __init__(
        self,
        layer_sizes: list,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(act_fn())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_dense_mlp(
    layer_sizes: list, activation: str = "relu", dropout: float = 0.0
) -> DenseMLP:
    return DenseMLP(layer_sizes, activation, dropout)


def create_lottery_ticket_mlp(
    layer_sizes: list,
    target_sparsity: float = 0.9,
    train_fn=None,
    train_loader=None,
    device: str = "cpu",
    seed: int = 42,
    activation: str = "relu",
    n_prune_rounds: int = 5,
) -> SparseMLP:
    """Create a sparse MLP via iterative magnitude pruning (lottery ticket method).

    Trains a dense network, prunes smallest-magnitude weights, rewinds to
    initial weights, repeats. This is the standard lottery ticket baseline
    from Frankle & Carlin (2019).

    Args:
        layer_sizes: Network architecture.
        target_sparsity: Final target sparsity.
        train_fn: Function(model, loader, device, epochs) that trains the model.
        train_loader: DataLoader for training.
        device: Device to train on.
        seed: Random seed.
        activation: Activation function.
        n_prune_rounds: Number of prune-retrain rounds.
    """
    torch.manual_seed(seed)

    # Create dense model and save initial weights
    dense = DenseMLP(layer_sizes, activation)
    initial_state = {
        k: v.clone() for k, v in dense.state_dict().items()
    }

    # Current masks (start fully connected)
    masks = [np.ones((layer_sizes[i + 1], layer_sizes[i]), dtype=np.float32)
             for i in range(len(layer_sizes) - 1)]

    # Iterative magnitude pruning
    prune_per_round = 1.0 - (1.0 - target_sparsity) ** (1.0 / n_prune_rounds)

    for round_idx in range(n_prune_rounds):
        # Train
        dense = dense.to(device)
        if train_fn and train_loader:
            train_fn(dense, train_loader, device, epochs=10)

        # Prune: remove smallest magnitude weights per layer
        linear_layers = [m for m in dense.network if isinstance(m, nn.Linear)]
        for layer_idx, layer in enumerate(linear_layers):
            w = layer.weight.data.cpu().numpy()
            w_masked = np.abs(w) * masks[layer_idx]
            nonzero = w_masked[masks[layer_idx] > 0]
            if len(nonzero) == 0:
                continue
            threshold = np.percentile(nonzero, prune_per_round * 100)
            masks[layer_idx][w_masked < threshold] = 0.0

        # Rewind to initial weights
        dense.load_state_dict(initial_state)

    # Create SparseMLP with the found masks
    return SparseMLP(layer_sizes, masks, activation=activation)
