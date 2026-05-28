"""
Training loop with topology tracking and signal degradation support.

Trains sparse MLPs while periodically snapshotting topology metrics,
enabling analysis of how network structure evolves during learning.
Supports mid-training interventions: signal degradation (unreliable-input experiment)
and developmental pruning (timing experiment).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Optional, Callable
from tqdm import tqdm

from ..topology.metrics import compute_topology_metrics, compute_projected_metrics
from .signal_degradation import SignalDegrader


class Trainer:
    """Training loop with topology tracking and mid-training interventions."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        device: str = "cpu",
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        track_topology_every: int = 5,
        signal_degrader: Optional[SignalDegrader] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.track_every = track_topology_every
        self.signal_degrader = signal_degrader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "topology_snapshots": [],
            "epoch_times": [],
        }

    def train_epoch(self, epoch: int) -> tuple:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            # Flatten images if needed
            if batch_x.dim() > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)

            # Apply signal degradation if configured
            if self.signal_degrader:
                batch_x = self.signal_degrader(batch_x, epoch)

            self.optimizer.zero_grad()
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            loss.backward()

            # Re-apply masks after gradient update (for SparseLinear layers)
            self.optimizer.step()
            self._enforce_masks()

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

            if batch_x.dim() > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)

            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        return total_loss / total, correct / total

    def _enforce_masks(self):
        """Re-zero pruned weights after optimizer step."""
        from ..models.sparse_linear import SparseLinear
        for module in self.model.modules():
            if isinstance(module, SparseLinear):
                with torch.no_grad():
                    module.weight.data *= module.mask

    def snapshot_topology(self, epoch: int):
        """Capture topology metrics at current training state.

        Uses projected (functional connectivity) metrics for the first layer
        to get meaningful clustering/modularity values. Bipartite layer graphs
        have clustering=0 by construction, so projection is necessary.
        """
        from ..models.sparse_linear import SparseLinear
        sparse_layers = [m for m in self.model.modules() if isinstance(m, SparseLinear)]

        layer_metrics = []
        for i, layer in enumerate(sparse_layers):
            mask = layer.get_mask_numpy()
            weights = layer.get_weights_numpy()

            # Use projected metrics for meaningful clustering
            metrics = compute_projected_metrics(
                mask, weights, project_to="output",
                max_nodes_for_expensive=0,  # skip expensive for training snapshots
                projection_density=0.1,
            )
            metrics["layer"] = i
            metrics["epoch"] = epoch
            layer_metrics.append(metrics)

        self.history["topology_snapshots"].append({
            "epoch": epoch,
            "layers": layer_metrics,
        })

    def train(
        self,
        epochs: int,
        pruning_callback: Optional[Callable] = None,
        verbose: bool = True,
    ) -> dict:
        """Full training loop.

        Args:
            epochs: Number of training epochs.
            pruning_callback: Optional function(model, epoch) called each epoch.
                Used for developmental pruning timing experiments.
            verbose: Print progress.
        """
        # Initial topology snapshot
        self.snapshot_topology(0)

        iterator = range(1, epochs + 1)
        if verbose:
            iterator = tqdm(iterator, desc="Training")

        for epoch in iterator:
            t0 = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.evaluate(self.val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch_times"].append(time.time() - t0)

            # Track per-epoch sparsity (needed for accurate FLOPs accounting
            # when sparsity changes during training, e.g., iterative pruning)
            if hasattr(self.model, "get_total_sparsity"):
                if "epoch_sparsities" not in self.history:
                    self.history["epoch_sparsities"] = []
                self.history["epoch_sparsities"].append(
                    float(self.model.get_total_sparsity())
                )

            # Topology snapshot
            if epoch % self.track_every == 0 or epoch == epochs:
                self.snapshot_topology(epoch)

            # Mid-training pruning callback
            if pruning_callback:
                pruning_callback(self.model, epoch)

            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix(
                    loss=f"{train_loss:.4f}",
                    acc=f"{train_acc:.3f}",
                    val=f"{val_acc:.3f}",
                )

        # Final test evaluation
        if self.test_loader:
            test_loss, test_acc = self.evaluate(self.test_loader)
            self.history["test_loss"] = test_loss
            self.history["test_acc"] = test_acc

        return self.history


def train_model_simple(model, loader, device, epochs=10):
    """Simple training function for lottery ticket baseline."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            if batch_x.dim() > 2:
                batch_x = batch_x.view(batch_x.size(0), -1)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
