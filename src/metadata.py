from enum import Enum, auto
from dataclasses import dataclass

import numpy as np


class Collision(Enum):
    WALL = auto()
    BODY = auto()


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool

@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
