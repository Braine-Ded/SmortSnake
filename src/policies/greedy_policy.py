from __future__ import annotations

import numpy as np
import torch

from .base import Policy


class GreedyPolicy(Policy):
    """Always pick the action with the highest predicted Q-value.

    Usable directly as the `action_fn` of `run_simulated`:
        run_simulated(env, policy.select_action)
    """

    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu"):
        # The model is expected to already live on `device`.
        self.model = model
        self.device = torch.device(device)

    def select_action(self, observation, *, explore: bool = True) -> int:
        """Return argmax_a Q(observation, a).

        `explore` is accepted for interface compatibility and ignored:
        a greedy policy never explores.
        """
        obs = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), device=self.device
        ).unsqueeze(0)

        # Inference only: no gradients, and restore the model's train/eval mode.
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            q_values = self.model(obs)
        if was_training:
            self.model.train()

        return int(q_values.argmax(dim=1).item())
