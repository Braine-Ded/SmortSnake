from enum import Enum, auto
from dataclasses import dataclass

class Collision(Enum):
    WALL = auto()
    BODY = auto()


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: list
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
    terminated: bool
    truncated: bool

