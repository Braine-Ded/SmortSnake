from __future__ import annotations

import random
from collections import deque

import numpy as np

from ..metadata import Transition, TransitionBatch
from .base import Buffer


class ReplayBuffer(Buffer):
    """Fixed-capacity replay memory for DQN training"""

    def __init__(self, capacity: int, seed: int | None = 67):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.memory: deque[Transition] = deque(maxlen=capacity)

        # Private RNG: reproducible, but successive batches still differ,
        # and the global `random` state is left untouched.
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self.memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Store one transition, copying observations to prevent mutation."""
        transition = Transition(
            state=np.array(state, dtype=np.float32, copy=True),
            action=int(action),
            reward=float(reward),
            next_state=np.array(next_state, dtype=np.float32, copy=True),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        self.memory.append(transition)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random minibatch, without replacement"""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self.memory):
            raise ValueError(f"Cannot sample {batch_size} transitions from buffer containing {len(self.memory)}")

        transitions = self._random.sample(list(self.memory), batch_size)

        return TransitionBatch(
            states=np.stack([t.state for t in transitions]),
            actions=np.asarray(
                [t.action for t in transitions], dtype=np.int64
            ),
            rewards=np.asarray(
                [t.reward for t in transitions], dtype=np.float32
            ),
            next_states=np.stack([t.next_state for t in transitions]),
            terminated=np.asarray(
                [t.terminated for t in transitions], dtype=np.float32
            ),
            truncated=np.asarray(
                [t.truncated for t in transitions], dtype=np.float32
            ),
        )
