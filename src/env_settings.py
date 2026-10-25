from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rewards:
    """Reward per event. The defaults are the original values."""

    step: float = -0.01     # every move that doesn't eat or die
    food: float = 10.0
    death: float = -10.0
    win: float = 10.0       # added on top of `food` when the board fills


@dataclass(frozen=True)
class EnvSettings:
    """Everything a config may change about the environment.

    Run names list only the fields that differ from these defaults
    (see src/naming.py), so keep the defaults equal to the original setup.
    """

    grid_size: tuple[int, int] = (10, 10)
    start_size: int = 3
    max_steps: int | None = 1000
    rewards: Rewards = field(default_factory=Rewards)
