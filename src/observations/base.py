from abc import ABC, abstractmethod

import numpy as np

from .spec import ObservationSpec


class ObservationEncoder(ABC):
    """Turns the environment's current state into an observation array.

    Encoders only read `env.snake`, `env.food`, `env.direction` and
    `env.grid_size` (plus the env's public helpers); they must never mutate it.
    """

    @property
    @abstractmethod
    def spec(self) -> ObservationSpec:
        """Shape of every array returned by `encode`."""

    @abstractmethod
    def encode(self, env) -> np.ndarray:
        """Return the float32 observation for the env's current state."""

    def check_compatible(self, env) -> None:
        """Raise ValueError if this encoder can't describe `env` (e.g. wrong grid size)."""
