from __future__ import annotations

import numpy as np

from .base import ObservationEncoder
from .spec import ObservationSpec


class PlaneObservation(ObservationEncoder):
    """CNN input: (4, H + 2, W + 2) one-hot-style planes, indexed [channel, y, x].

    Channels:
        0 wall - the 1-cell border around the board. Without it a conv's zero
                 padding would look exactly like an empty cell.
        1 body - head 1.0 fading towards the tail, so the network can see the
                 heading and which cells free up next.
        2 head
        3 food
    """

    WALL, BODY, HEAD, FOOD = range(4)
    NUM_CHANNELS = 4

    def __init__(self, grid_size: tuple[int, int]):
        self.grid_size = tuple(grid_size)
        width, height = grid_size
        self._spec = ObservationSpec((self.NUM_CHANNELS, height + 2, width + 2))

        # The wall plane never changes; build it once.
        self._wall = np.ones((height + 2, width + 2), dtype=np.float32)
        self._wall[1:-1, 1:-1] = 0.0

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
        planes = np.zeros(self._spec.shape, dtype=np.float32)
        planes[self.WALL] = self._wall

        # +1 everywhere below: board coordinates shift by the wall border.
        length = len(env.snake)
        for i, (x, y) in enumerate(env.snake):
            planes[self.BODY, y + 1, x + 1] = 1.0 - i / length

        head_x, head_y = env.snake[0]
        planes[self.HEAD, head_y + 1, head_x + 1] = 1.0

        if env.food is not None:
            food_x, food_y = env.food
            planes[self.FOOD, food_y + 1, food_x + 1] = 1.0

        return planes
