"""
Sparse linear layer with topology-constrained connectivity.

The mask is a fixed binary tensor that gates which connections exist.
During training, only unmasked weights receive gradients. This implements
the "fixed topology, learned weights" paradigm from bio-inspired sparse
network research.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SparseLinear(nn.Module):
    """Linear layer with a fixed sparsity mask.

    Weights are dense tensors, but a binary mask zeros out pruned connections
    during forward pass. Gradients only flow through active connections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: np.ndarray,
        bias: bool = True,
        initial_weights: np.ndarray = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Register mask as a buffer (not a parameter — no gradients)
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))

        if initial_weights is not None:
            self.weight.data = torch.tensor(initial_weights, dtype=torch.float32)
        else:
            # Kaiming init scaled for sparse connectivity
            fan_in = mask.sum(axis=1).mean()  # average active inputs per output
            fan_in = max(fan_in, 1.0)
            std = np.sqrt(2.0 / fan_in)
            nn.init.normal_(self.weight, 0, std)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

        # Apply mask to initial weights
        with torch.no_grad():
            self.weight.data *= self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply mask to enforce sparsity
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)

    def get_active_weights(self) -> torch.Tensor:
        """Return only the active (unmasked) weight values."""
        return self.weight.data[self.mask > 0]

    def get_sparsity(self) -> float:
        """Return current sparsity level."""
        return 1.0 - self.mask.sum().item() / self.mask.numel()

    def get_mask_numpy(self) -> np.ndarray:
        return self.mask.cpu().numpy()

    def get_weights_numpy(self) -> np.ndarray:
        return (self.weight.data * self.mask).detach().cpu().numpy()

    def apply_additional_pruning(self, new_mask: np.ndarray):
        """Apply additional pruning mask (intersection with existing mask)."""
        new_mask_t = torch.tensor(new_mask, dtype=torch.float32, device=self.mask.device)
        self.mask.data = self.mask * new_mask_t
        with torch.no_grad():
            self.weight.data *= self.mask

    def extra_repr(self) -> str:
        sparsity = self.get_sparsity()
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"sparsity={sparsity:.3f}, bias={self.bias is not None}"
        )
