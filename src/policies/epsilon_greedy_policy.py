from __future__ import annotations

import random

from .base import Policy


class EpsilonGreedyPolicy(Policy):
    """With probability epsilon take a random action, otherwise defer to a greedy policy.

    Epsilon is decayed once per episode via `end_episode()`.
    """

    def __init__(
        self,
        greedy_policy: Policy,
        num_actions: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int | None = 67,
    ):
        if num_actions < 1:
            raise ValueError("num_actions must be at least 1")
        if not 0.0 <= epsilon_min <= epsilon_start <= 1.0:
            raise ValueError("Need 0 <= epsilon_min <= epsilon_start <= 1")
        if not 0.0 < epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")

        self.greedy_policy = greedy_policy
        self.num_actions = num_actions
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Private RNG so exploration is reproducible and independent of
        # the global `random` state.
        self._random = random.Random(seed)

    def select_action(self, observation, *, explore: bool = True) -> int:
        # Exploration is skipped entirely when explore=False (evaluation).
        if explore and self._random.random() < self.epsilon:
            return self._random.randrange(self.num_actions)

        return self.greedy_policy.select_action(observation, explore=False)

    def decay(self) -> None:
        """Shrink epsilon by one decay step, never going below epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def end_episode(self) -> None:
        self.decay()

    def metrics(self) -> dict:
        return {"epsilon": self.epsilon}

    def state_dict(self) -> dict:
        return {"epsilon": self.epsilon, "rng_state": self._random.getstate()}

    def load_state_dict(self, state: dict) -> None:
        self.epsilon = state["epsilon"]
        self._random.setstate(state["rng_state"])
