from __future__ import annotations

import math
import random

import numpy as np
import torch

from .base import Policy


class SoftmaxPolicy(Policy):
    """Sample actions from softmax(Q / temperature) (Boltzmann exploration).

    Unlike epsilon-greedy, exploration follows the Q-values: an action that
    looks fatal is almost never tried once the network knows it is fatal, while
    near-equal actions are tried about equally often. Temperature is decayed
    once per episode via `end_episode()`; explore=False picks the argmax.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        num_actions: int,
        temperature_start: float = 1.0,
        temperature_min: float = 0.01,
        temperature_decay: float = 0.995,
        seed: int | None = 67,
        device: str | torch.device = "cpu",
    ):
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        if not 0.0 < temperature_min <= temperature_start:
            raise ValueError("Need 0 < temperature_min <= temperature_start")
        if not 0.0 < temperature_decay <= 1.0:
            raise ValueError("temperature_decay must be in (0, 1]")

        # The model is expected to already live on `device`.
        self.model = model
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.temperature = temperature_start
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay

        # Private RNG so exploration is reproducible and independent of
        # the global `random` state.
        self._random = random.Random(seed)

    def _q_values(self, observation) -> np.ndarray:
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

        return q_values.squeeze(0).double().cpu().numpy()

    def select_action(self, observation, *, explore: bool = True) -> int:
        q_values = self._q_values(observation)

        # Evaluation: no sampling, same as the greedy policy.
        if not explore:
            return int(q_values.argmax())

        # Subtract the max before exp() so large Q / small temperature can't overflow.
        logits = (q_values - q_values.max()) / self.temperature
        weights = [math.exp(logit) for logit in logits]
        return self._random.choices(range(self.num_actions), weights=weights)[0]

    def decay(self) -> None:
        """Shrink the temperature by one decay step, never going below temperature_min."""
        self.temperature = max(self.temperature_min, self.temperature * self.temperature_decay)

    def end_episode(self) -> None:
        self.decay()

    def metrics(self) -> dict:
        return {"temperature": self.temperature}

    def state_dict(self) -> dict:
        return {"temperature": self.temperature, "rng_state": self._random.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self.temperature = state["temperature"]
        self._random.setstate(state["rng_state"])
