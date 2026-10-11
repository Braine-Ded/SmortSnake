from __future__ import annotations

import numpy as np

from ..metadata import Transition, TransitionBatch
from .base import Buffer


class ReplayBuffer(Buffer):
    """Fixed-capacity replay memory for DQN training.

    Backed by preallocated numpy arrays used as a ring buffer, so both
    `push` and `sample` cost O(1) / O(batch_size) regardless of how many
    transitions are stored. (The previous deque version copied the whole
    buffer into a list on every sample, which is O(N) per training step.)

    Arrays are allocated on the first push, once the observation shape is known.
    """

    _FIELDS = (
        "states", "actions", "rewards", "next_states", "terminated", "truncated",
    )

    def __init__(self, capacity: int, seed: int | None = 67):
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer")
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self._rng = np.random.default_rng(seed)

        self._index = 0   # next slot to write
        self._size = 0    # number of valid transitions stored

        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._terminated = np.zeros(capacity, dtype=np.float32)
        self._truncated = np.zeros(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    def _allocate(self, obs_shape: tuple[int, ...]) -> None:
        self._states = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((self.capacity, *obs_shape), dtype=np.float32)

    def push(self, transition: Transition) -> None:
        """Store one transition, overwriting the oldest once full."""
        if self._states is None:
            self._allocate(np.shape(transition.state))

        i = self._index
        self._states[i] = transition.state
        self._next_states[i] = transition.next_state
        self._actions[i] = transition.action
        self._rewards[i] = transition.reward
        self._terminated[i] = transition.terminated
        self._truncated[i] = transition.truncated

        self._index = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random minibatch, without replacement."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer containing {self._size}"
            )

        ix = self._rng.choice(self._size, size=batch_size, replace=False)

        return TransitionBatch(
            states=self._states[ix],
            actions=self._actions[ix],
            rewards=self._rewards[ix],
            next_states=self._next_states[ix],
            terminated=self._terminated[ix],
            truncated=self._truncated[ix],
        )

    # ---------- checkpointing ----------

    def state_dict(self) -> dict:
        """Contents in chronological order (oldest first), plus the RNG state."""
        n = self._size
        if n == 0:
            return {"empty": True}

        # When full, the oldest entry sits at `_index`; roll it to the front.
        shift = -self._index if n == self.capacity else 0

        def ordered(arr):
            return np.roll(arr[:n], shift, axis=0).copy()

        return {
            "empty": False,
            "states": ordered(self._states),
            "next_states": ordered(self._next_states),
            "actions": ordered(self._actions),
            "rewards": ordered(self._rewards),
            "terminated": ordered(self._terminated),
            "truncated": ordered(self._truncated),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("empty", True):
            return

        n = len(state["actions"])
        keep = min(n, self.capacity)  # if capacity shrank, keep the newest

        if self._states is None:
            self._allocate(state["states"].shape[1:])

        self._states[:keep] = state["states"][n - keep:]
        self._next_states[:keep] = state["next_states"][n - keep:]
        self._actions[:keep] = state["actions"][n - keep:]
        self._rewards[:keep] = state["rewards"][n - keep:]
        self._terminated[:keep] = state["terminated"][n - keep:]
        self._truncated[:keep] = state["truncated"][n - keep:]

        self._size = keep
        self._index = keep % self.capacity
        self._rng.bit_generator.state = state["rng_state"]
