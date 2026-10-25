from __future__ import annotations

import numpy as np

from .plane_observation import PlaneObservation


class EgocentricPlaneObservation(PlaneObservation):
    """PlaneObservation rotated so the snake's heading always points up.

    The actions are relative (forward / right / left), so in this frame
    "forward" is always up and "right" is always right on the image. The
    network learns each rule once instead of once per absolute heading.

    Same (4, H + 2, W + 2) channels as PlaneObservation. The board must be
    square so every rotation keeps the same shape.
    """

    # Quarter turns (np.rot90, counter-clockwise) that bring each heading to up.
    # Screen coordinates: y increases downward, so up is (0, -1).
    _TURNS = {
        (0, -1): 0,   # up: already up
        (1, 0): 1,    # right: east edge -> top
        (0, 1): 2,    # down
        (-1, 0): 3,   # left: west edge -> top
    }

    def __init__(self, grid_size: tuple[int, int]):
        width, height = grid_size
        if width != height:
            raise ValueError(
                f"{type(self).__name__} needs a square board, got {tuple(grid_size)}."
            )
        super().__init__(grid_size)

    def encode(self, env) -> np.ndarray:
        planes = super().encode(env)
        turns = self._TURNS[tuple(env.direction)]
        # rot90 returns a negatively strided view; copy so torch can take it.
        return np.ascontiguousarray(np.rot90(planes, k=turns, axes=(1, 2)))
