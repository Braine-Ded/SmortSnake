from __future__ import annotations

from enum import IntEnum

import numpy as np

from .base import ObservationEncoder
from .spec import ObservationSpec


class Cell(IntEnum):
    EMPTY = 0
    BODY = 1
    # The head gets its own code: actions are relative to the heading,
    # and the head/neck pair is the only way to read the heading off the board.
    HEAD = 2
    FOOD = 3


class GridObservation(ObservationEncoder):
    """The board as an (H, W) array of `Cell` codes, indexed [y, x].

    The codes are categories, not magnitudes (FOOD is not "3x BODY"), so don't
    feed this straight into a Conv2d. It suits rendering, debugging, tabular
    methods, or a network that embeds/one-hots it first; `PlaneObservation`
    is the CNN-ready version.
    """

    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = tuple(grid_size)
        width, height = grid_size
        self._spec = ObservationSpec((height, width))

    @property
    def spec(self) -> ObservationSpec:
        return self._spec

    def check_compatible(self, env) -> None:
        if tuple(env.grid_size) != self.grid_size:
            raise ValueError(
                f"{type(self).__name__} was built for grid {self.grid_size}, "
                f"but the environment is {tuple(env.grid_size)}."
            )

    def encode(self, env) -> np.ndarray:
        # float32 so it flows through the replay buffer and agent like any
        # other observation; cast back with .long() if a network embeds it.
        board = np.full(self._spec.shape, Cell.EMPTY, dtype=np.float32)

        for x, y in env.snake[1:]:
            board[y, x] = Cell.BODY

        head_x, head_y = env.snake[0]
        board[head_y, head_x] = Cell.HEAD

        if env.food is not None:
            food_x, food_y = env.food
            board[food_y, food_x] = Cell.FOOD

        return board
