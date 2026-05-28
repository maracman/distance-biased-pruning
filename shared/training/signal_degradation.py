"""
Signal degradation for the social signal hypothesis experiment.

Models the effect of unreliable social signals on neural processing by
corrupting input signals during training. Three degradation modes mirror
the signal degradation hypothesis scenarios:

1. Sudden: noise jumps to target level at a specific epoch (social trauma)
2. Gradual: noise increases linearly over epochs (progressive isolation)
3. Intermittent: noise fluctuates (unreliable/inconsistent social signals)

The signal degradation is applied to input data during training, and the
resulting changes in network topology are tracked to test the hypothesis
that unreliable input produces reduced clustering and small-worldedness.
"""

import torch
import numpy as np


class SignalDegrader:
    """Applies configurable noise to input signals during training.

    Supports Gaussian noise, salt-and-pepper noise, and structured noise
    (zeroing out contiguous input regions, modeling loss of specific input
    modalities like social vs non-social information).
    """

    def __init__(
        self,
        noise_level: float = 0.0,
        noise_type: str = "gaussian",
        schedule: str = "sudden",
        onset_epoch: int = 0,
        ramp_epochs: int = 20,
        social_input_fraction: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            noise_level: Target noise level (0 = none, 1 = fully corrupted).
            noise_type: 'gaussian', 'salt_pepper', or 'structured'.
            schedule: 'sudden', 'gradual', or 'intermittent'.
            onset_epoch: Epoch when degradation begins.
            ramp_epochs: For 'gradual', number of epochs to reach full noise.
            social_input_fraction: For 'structured', fraction of inputs
                designated as "social" (corrupted preferentially).
            seed: Random seed.
        """
        self.target_noise = noise_level
        self.noise_type = noise_type
        self.schedule = schedule
        self.onset_epoch = onset_epoch
        self.ramp_epochs = ramp_epochs
        self.social_fraction = social_input_fraction
        self.rng = np.random.RandomState(seed)
        self._social_mask = None

    def _get_noise_level(self, epoch: int) -> float:
        """Compute current noise level based on schedule."""
        if epoch < self.onset_epoch:
            return 0.0

        elapsed = epoch - self.onset_epoch

        if self.schedule == "sudden":
            return self.target_noise

        elif self.schedule == "gradual":
            progress = min(1.0, elapsed / max(1, self.ramp_epochs))
            return self.target_noise * progress

        elif self.schedule == "intermittent":
            # Oscillating noise with increasing baseline
            baseline = self.target_noise * min(1.0, elapsed / max(1, self.ramp_epochs))
            oscillation = 0.3 * self.target_noise * np.sin(elapsed * 0.5)
            return max(0.0, min(1.0, baseline + oscillation))

        return self.target_noise

    def _get_social_mask(self, input_dim: int, device: torch.device) -> torch.Tensor:
        """Create a binary mask identifying 'social' input dimensions."""
        if self._social_mask is None or self._social_mask.shape[-1] != input_dim:
            n_social = int(input_dim * self.social_fraction)
            mask = torch.zeros(input_dim, device=device)
            # Social inputs are a contiguous block (can be made random)
            social_idx = self.rng.choice(input_dim, n_social, replace=False)
            mask[social_idx] = 1.0
            self._social_mask = mask
        return self._social_mask.to(device)

    def __call__(self, x: torch.Tensor, epoch: int) -> torch.Tensor:
        """Apply signal degradation to input batch."""
        noise_level = self._get_noise_level(epoch)
        if noise_level <= 0:
            return x

        if self.noise_type == "gaussian":
            noise = torch.randn_like(x) * noise_level
            return x + noise

        elif self.noise_type == "salt_pepper":
            mask = torch.rand_like(x)
            corrupted = x.clone()
            corrupted[mask < noise_level / 2] = 0.0
            corrupted[mask > 1 - noise_level / 2] = 1.0
            return corrupted

        elif self.noise_type == "structured":
            # Corrupt only "social" input dimensions more heavily
            social_mask = self._get_social_mask(x.shape[-1], x.device)
            noise = torch.randn_like(x) * noise_level
            # Social dimensions get full noise; others get reduced noise
            noise_weights = social_mask * 1.0 + (1 - social_mask) * 0.2
            return x + noise * noise_weights

        return x


def create_degradation_schedule(
    noise_levels: list,
    noise_type: str = "gaussian",
    schedule: str = "sudden",
    onset_epoch: int = 25,
    seed: int = 42,
) -> list:
    """Create a list of SignalDegraders for sweeping noise levels."""
    return [
        SignalDegrader(
            noise_level=nl,
            noise_type=noise_type,
            schedule=schedule,
            onset_epoch=onset_epoch,
            seed=seed + i,
        )
        for i, nl in enumerate(noise_levels)
    ]
