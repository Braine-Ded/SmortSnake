from __future__ import annotations

import random
from numbers import Integral

import numpy as np

from .env_settings import EnvSettings, Rewards
from .metadata import Collision
from .observations import FeatureObservation, ObservationEncoder, ObservationSpec


class SnakeEnv:
    ACTION_FORWARD = 0
    ACTION_RIGHT = 1
    ACTION_LEFT = 2
    NUM_ACTIONS = 3

    def __init__(
        self,
        grid_size: tuple[int, int],
        start_size: int,
        start_pos: tuple[int, int],
        start_horizontal: bool = True,
        seed: int = 67,
        max_steps: int | None = 1000,
        observation: ObservationEncoder | None = None,
        rewards: Rewards | None = None,
    ):
        width, height = grid_size

        if width <= 0 or height <= 0:
            raise ValueError("Grid dimensions must be positive.")

        if not (0 <= start_pos[0] < width and
                0 <= start_pos[1] < height):
            raise ValueError(f"Start position {start_pos} is outside the grid.")

        if start_size < 1:
            raise ValueError("start_size must be at least 1.")

        if start_size >= width * height:
            raise ValueError(
                "The initial snake must leave at least one free cell."
            )

        # The body extends left of the head when horizontal,
        # and above the head when vertical.
        available_space = (
            start_pos[0] + 1
            if start_horizontal
            else start_pos[1] + 1
        )

        if start_size > available_space:
            raise ValueError(
                "The initial snake would extend outside the board."
            )

        if max_steps is not None and (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, Integral)
            or max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer or None.")

        self.grid_size = grid_size
        self._start_size = start_size
        self._start_pos = start_pos
        self._start_horizontal = start_horizontal

        self.max_steps = (
            None if max_steps is None else int(max_steps)
        )

        self.rewards = Rewards() if rewards is None else rewards

        self.seed = seed
        self._random = random.Random(seed)

        # What the agent sees. Defaults to the original 11 features.
        self.observation = (
            FeatureObservation() if observation is None else observation
        )

        # Catch an encoder built for a different grid size up front.
        self.observation.check_compatible(self)

        self.reset()

    @classmethod
    def from_settings(
        cls,
        settings: EnvSettings,
        *,
        seed: int,
        observation: ObservationEncoder | None = None,
    ) -> SnakeEnv:
        """Build the env a config describes; the snake starts mid-board."""
        return cls(
            settings.grid_size,
            settings.start_size,
            tuple(size // 2 for size in settings.grid_size),
            seed=seed,
            max_steps=settings.max_steps,
            observation=observation,
            rewards=settings.rewards,
        )

    @property
    def observation_spec(self) -> ObservationSpec:
        return self.observation.spec

    @property
    def grid_width(self) -> int:
        return self.grid_size[0]

    @property
    def grid_height(self) -> int:
        return self.grid_size[1]

    def _generate_snake(
        self,
        start_size: int,
        start_pos: tuple[int, int],
        start_horizontal: bool,
    ) -> list[tuple[int, int]]:
        x, y = start_pos

        if start_horizontal:
            return [(x - i, y) for i in range(start_size)]

        return [(x, y - i) for i in range(start_size)]

    def _generate_food(self) -> tuple[int, int] | None:
        """Choose an unoccupied cell, or None if the board is full."""
        occupied = set(self.snake)

        empty_cells = [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in occupied
        ]

        if not empty_cells:
            return None

        return self._random.choice(empty_cells)

    @staticmethod
    def dir_after_action(
        current: tuple[int, int],
        action: int,
    ) -> tuple[int, int]:
        """Convert a relative action to an absolute direction."""
        dx, dy = current

        if action == SnakeEnv.ACTION_FORWARD:
            return current

        if action == SnakeEnv.ACTION_RIGHT:
            # Screen coordinates: y increases downward.
            return (-dy, dx)

        if action == SnakeEnv.ACTION_LEFT:
            return (dy, -dx)

        raise ValueError(f"Invalid action: {action}")

    def is_danger(self, direction: tuple[int, int]) -> bool:
        """Would moving in this direction hit a wall or body segment?"""
        head_x, head_y = self.snake[0]
        dx, dy = direction

        next_head = (head_x + dx, head_y + dy)
        x, y = next_head

        if not (0 <= x < self.grid_width and
                0 <= y < self.grid_height):
            return True

        # On a normal move, the current tail vacates its cell.
        return next_head in self.snake[:-1]

    def _get_obs(self) -> np.ndarray:
        """Return the observation produced by the configured encoder."""
        return self.observation.encode(self)

    def _check_collision(
        self,
        new_head: tuple[int, int],
        growing: bool,
    ) -> Collision | None:
        """Check a potential head position without changing the state."""
        x, y = new_head

        if not (0 <= x < self.grid_width and
                0 <= y < self.grid_height):
            return Collision.WALL

        # if growing the tail stays in place
        # otherwise the tail leaves its current cell
        occupied = self.snake if growing else self.snake[:-1]

        if new_head in occupied:
            return Collision.BODY

        return None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode and return (observation, info)."""
        if options:
            raise ValueError("This environment does not support reset options.")

        # Only reseed when explicitly asked. Ordinary resets continue
        # the random sequence rather than repeating the same food.
        if seed is not None:
            self.seed = seed
            self._random = random.Random(seed)

        self.score = 0
        self.steps = 0
        self._episode_done = False

        self.snake = self._generate_snake(
            self._start_size,
            self._start_pos,
            self._start_horizontal,
        )

        # The body is behind the head, so the snake moves away from it.
        self.direction = (1, 0) if self._start_horizontal else (0, 1)

        self.food = self._generate_food()

        observation = self._get_obs()
        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "termination_reason": None,
        }

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one step and return the five transition values."""
        if self._episode_done:
            raise RuntimeError("Episode ended. Call reset() before step().")

        if (
            isinstance(action, bool)
            or not isinstance(action, Integral)
            or action not in range(self.NUM_ACTIONS)
        ):
            raise ValueError("action must be 0 (forward), 1 (right), or 2 (left).")

        action = int(action)

        # Calculate the candidate move before mutating the environment.
        new_direction = self.dir_after_action(self.direction, action)

        head_x, head_y = self.snake[0]
        dx, dy = new_direction
        new_head = (head_x + dx, head_y + dy)

        will_grow = self.food is not None and new_head == self.food

        self.steps += 1

        reward = self.rewards.step
        terminated = False
        truncated = False
        ate_food = False
        collision = None
        reason = None

        collision = self._check_collision(new_head, growing=will_grow)

        if collision is not None:
            # Keep the last valid snake configuration.
            terminated = True
            reward = self.rewards.death

            if collision == Collision.WALL:
                reason = "wall_collision"
            else:
                reason = "self_collision"

        else:
            # Commit the new direction only after validating the move.
            self.direction = new_direction

            if will_grow:
                # Keep the old tail: the snake grows by one segment.
                self.snake = [new_head, *self.snake]
                self.score += 1
                ate_food = True
                reward = self.rewards.food

                # Check victory BEFORE spawning another food item.
                if len(self.snake) == self.grid_width * self.grid_height:
                    self.food = None
                    terminated = True
                    reason = "board_filled"
                    reward += self.rewards.win
                else:
                    self.food = self._generate_food()

            else:
                # Normal movement: discard the old tail.
                self.snake = [new_head, *self.snake[:-1]]

        # A true terminal event takes precedence over the time limit.
        if (
            not terminated
            and self.max_steps is not None
            and self.steps >= self.max_steps
        ):
            truncated = True
            reason = "max_steps"

        self._episode_done = terminated or truncated

        observation = self._get_obs()
        info = {
            "score": self.score,
            "steps": self.steps,
            "snake_length": len(self.snake),
            "ate_food": ate_food,
            "collision": collision.name.lower() if collision else None,
            "termination_reason": reason,
        }

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

