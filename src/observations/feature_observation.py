from __future__ import annotations

import numpy as np

from .base import ObservationEncoder
from .spec import ObservationSpec


class FeatureObservation(ObservationEncoder):
    """Hand-crafted features: 3 danger + 4 direction + 4 food-position flags."""

    _SPEC = ObservationSpec((11,))

    @property
    def spec(self) -> ObservationSpec:
        return self._SPEC

    def encode(self, env) -> np.ndarray:
        head_x, head_y = env.snake[0]

        straight = env.direction
        right = env.dir_after_action(env.direction, env.ACTION_RIGHT)
        left = env.dir_after_action(env.direction, env.ACTION_LEFT)

        food_left = env.food is not None and env.food[0] < head_x
        food_right = env.food is not None and env.food[0] > head_x
        food_up = env.food is not None and env.food[1] < head_y
        food_down = env.food is not None and env.food[1] > head_y

        return np.asarray(
            [
                env.is_danger(straight),
                env.is_danger(right),
                env.is_danger(left),

                env.direction == (-1, 0),
                env.direction == (1, 0),
                env.direction == (0, -1),
                env.direction == (0, 1),

                food_left,
                food_right,
                food_up,
                food_down,
            ],
            dtype=np.float32,
        )
