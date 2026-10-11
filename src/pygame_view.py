"""Pygame rendering and manual controls for the Snake environment.

This module does not implement the environment's transition logic.
It expects an environment 'env' to expose the following interface/have the following attributes:

-> (grid_width, grid_height) : Grid dimensions in cells
-> snake : Ordered sequence of (x, y) coordinates starting from the snake head
-> food : (x, y) coordinate
-> direction : current direction as (dirx, diry)
-> score : current score
-> reset() : function to reset the episode
-> step(action) : function that returns (observation, reward, terminated, truncated, info) in the exact format
"""

from __future__ import annotations

import pygame

import csv
from pathlib import Path
from typing import Any, Callable

# ---------- Display configuration ----------

CELL_SIZE = 24
HUD_HEIGHT = 56
FPS = 60                # manual play
SIM_FPS = 120           # policy evaluation window
MOVE_INTERVAL_MS = 120
END_PAUSE_MS = 150      # pause on the game-over overlay between evaluation episodes

BACKGROUND = (17, 24, 32)
BOARD_COLOR = (23, 33, 43)
GRID_COLOR = (37, 49, 61)
SNAKE_HEAD_COLOR = (95, 210, 130)
SNAKE_BODY_COLOR = (48, 160, 100)
FOOD_COLOR = (235, 90, 90)
TEXT_COLOR = (235, 240, 245)
MUTED_COLOR = (155, 170, 185)


KEY_TO_DIR = {
    pygame.K_UP: (0, -1),
    pygame.K_RIGHT: (1, 0),
    pygame.K_DOWN: (0, 1),
    pygame.K_LEFT: (-1, 0),
    pygame.K_w: (0, -1),
    pygame.K_d: (1, 0),
    pygame.K_s: (0, 1),
    pygame.K_a: (-1, 0),
}


def action_for_direction(
    current: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    """Convert an absolute direction into a relative turn action.

    Returns:
        0: straight
        1: right
        2: left
        None: reverse direction; ignore
    """
    if target == current:
        return 0

    dx, dy = current

    # Coordinates use screen convention: y increases downward.
    right = (-dy, dx)
    left = (dy, -dx)

    if target == right:
        return 1
    if target == left:
        return 2

    return None


def draw_game(
    screen: pygame.Surface,
    env,
    fonts: tuple[pygame.font.Font, pygame.font.Font],
    game_over: bool,
    interactive: bool = True,
) -> None:
    """Draw the current environment state.

    `interactive` selects the HUD/overlay hints: manual play shows the
    movement and restart keys, policy evaluation only shows how to quit.
    """
    title_font, body_font = fonts

    board_width = env.grid_width * CELL_SIZE
    board_height = env.grid_height * CELL_SIZE

    screen.fill(BACKGROUND)

    # HUD
    score_surface = title_font.render(
        f"Score: {env.score}", True, TEXT_COLOR
    )
    controls_text = (
        "Arrows/WASD: move   R: restart   Esc/Q: quit"
        if interactive
        else "Esc/Q: quit"
    )
    controls_surface = body_font.render(
        controls_text,
        True,
        MUTED_COLOR,
    )

    screen.blit(score_surface, (14, 7))
    screen.blit(controls_surface, (14, 32))

    # Board
    board = pygame.Rect(0, HUD_HEIGHT, board_width, board_height)
    pygame.draw.rect(screen, BOARD_COLOR, board)

    # Grid lines
    for x in range(env.grid_width + 1):
        px = x * CELL_SIZE
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (px, HUD_HEIGHT),
            (px, HUD_HEIGHT + board_height),
        )

    for y in range(env.grid_height + 1):
        py = HUD_HEIGHT + y * CELL_SIZE
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (0, py),
            (board_width, py),
        )

    # Food
    if env.food is not None:
        food_x, food_y = env.food
        food_rect = pygame.Rect(
            food_x * CELL_SIZE + 4,
            HUD_HEIGHT + food_y * CELL_SIZE + 4,
            CELL_SIZE - 8,
            CELL_SIZE - 8,
        )
        pygame.draw.ellipse(screen, FOOD_COLOR, food_rect)

    # Snake: first segment is the head
    for index, (x, y) in enumerate(env.snake):
        color = SNAKE_HEAD_COLOR if index == 0 else SNAKE_BODY_COLOR

        segment = pygame.Rect(
            x * CELL_SIZE + 2,
            HUD_HEIGHT + y * CELL_SIZE + 2,
            CELL_SIZE - 4,
            CELL_SIZE - 4,
        )
        pygame.draw.rect(screen, color, segment, border_radius=5)

    # Game-over overlay
    if game_over:
        overlay = pygame.Surface(
            (board_width, board_height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 170))
        screen.blit(overlay, (0, HUD_HEIGHT))

        message = title_font.render("GAME OVER", True, TEXT_COLOR)
        restart_text = (
            "Press R to restart or Esc/Q to quit"
            if interactive
            else "Episode finished"
        )
        restart = body_font.render(
            restart_text,
            True,
            TEXT_COLOR,
        )

        screen.blit(
            message,
            message.get_rect(
                center=(board_width // 2, HUD_HEIGHT + board_height // 2 - 15)
            ),
        )
        screen.blit(
            restart,
            restart.get_rect(
                center=(board_width // 2, HUD_HEIGHT + board_height // 2 + 20)
            ),
        )

    pygame.display.flip()


def run_manual(env) -> None:
    """Open a window and let a person play through env.step()."""
    pygame.init()

    window_size = (
        env.grid_width * CELL_SIZE,
        HUD_HEIGHT + env.grid_height * CELL_SIZE,
    )

    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("Snake RL - Environment Viewer")
    clock = pygame.time.Clock()

    fonts = (
        pygame.font.SysFont("monospace", 22, bold=True),
        pygame.font.SysFont("monospace", 13),
    )

    env.reset()

    running = True
    game_over = False
    pending_action = None
    elapsed_ms = 0

    try:
        while running:
            # Process events every frame.
            # Pygame requires the event queue to be serviced regularly.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

                    elif event.key == pygame.K_r:
                        env.reset()
                        game_over = False
                        pending_action = None
                        elapsed_ms = 0

                    elif not game_over and event.key in KEY_TO_DIR:
                        target = KEY_TO_DIR[event.key]

                        action = action_for_direction(
                            env.direction,
                            target,
                        )

                        # Pressing the current heading again (action 0) must
                        # not cancel a turn that is already queued.
                        if action is not None and not (
                            action == 0 and pending_action is not None
                        ):
                            pending_action = action

            # The clock controls how often the game advances, independently of the window's rendering FPS.
            elapsed_ms += clock.tick(FPS)

            if not game_over and elapsed_ms >= MOVE_INTERVAL_MS:
                elapsed_ms -= MOVE_INTERVAL_MS

                action = pending_action if pending_action is not None else 0
                pending_action = None

                # Environment transitions are implemented elsewhere.
                _, _, terminated, truncated, _ = env.step(action)
                game_over = bool(terminated or truncated)

            draw_game(screen, env, fonts, game_over)

    finally:
        print("Quitting pygame...")
        pygame.quit()




def run_simulated(
    env,
    action_fn: Callable[[Any], int],
    *,
    episodes: int = 1,
    move_interval_ms: int = MOVE_INTERVAL_MS,
    fps: int = SIM_FPS,
    csv_path: str | Path | None = None,
    checkpoint: int | str | None = None,
) -> list[dict]:
    """Visualize a policy playing Snake and record episode metrics.

    Args:
        env:
            Environment implementing reset() and step(action).
        action_fn:
            Callable mapping an observation to an action (0, 1, or 2).
            It should use greedy/deterministic inference, without exploration.
        episodes:
            Number of evaluation episodes to run.
        move_interval_ms:
            Milliseconds between environment steps. When this is shorter than
            one frame, several steps run per rendered frame, so the frames in
            between are skipped rather than drawn.
        fps:
            Rendering frame rate of the evaluation window.
        csv_path:
            Optional CSV path for appending evaluation results.
        checkpoint:
            Training episode/step at which this evaluation was performed.

    Returns:
        One metrics dictionary per episode.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")

    if move_interval_ms < 1:
        raise ValueError("move_interval_ms must be positive")

    pygame.init()

    screen = pygame.display.set_mode(
        (
            env.grid_width * CELL_SIZE,
            HUD_HEIGHT + env.grid_height * CELL_SIZE,
        )
    )
    pygame.display.set_caption("Snake RL - Policy Evaluation")

    clock = pygame.time.Clock()
    fonts = (
        pygame.font.SysFont("monospace", 22, bold=True),
        pygame.font.SysFont("monospace", 13),
    )

    results: list[dict] = []
    running = True

    try:
        for episode in range(1, episodes + 1):
            reset_result = env.reset()

            # Support both Gymnasium-style reset() and reset() methods that return only an observation.
            if (
                isinstance(reset_result, tuple)
                and len(reset_result) == 2
                and isinstance(reset_result[1], dict)
            ):
                observation, _ = reset_result
            else:
                observation = reset_result

            terminated = False
            truncated = False
            interrupted = False
            total_reward = 0.0
            steps = 0
            elapsed_ms = 0
            info = {}

            while running and not (terminated or truncated):
                elapsed_ms += clock.tick(fps)

                # Keep the event queue responsive even while the model is playing automatically.
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif (
                        event.type == pygame.KEYDOWN
                        and event.key in (pygame.K_ESCAPE, pygame.K_q)
                    ):
                        running = False

                if not running:
                    interrupted = True
                    break

                # Run every step that has come due since the last frame, then
                # draw once. At high speeds that means many steps per frame.
                due, elapsed_ms = divmod(elapsed_ms, move_interval_ms)

                for _ in range(int(due)):
                    # Inference only. The agent's implementation
                    # and the environment's transition logic are separate.
                    action = action_fn(observation)

                    # Support scalar tensor actions as well as integers.
                    if hasattr(action, "item"):
                        action = action.item()
                    action = int(action)

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    total_reward += float(reward)
                    steps += 1

                    if terminated or truncated:
                        break

                draw_game(
                    screen,
                    env,
                    fonts,
                    game_over=bool(terminated or truncated),
                    interactive=False,
                )

            end_reason = info.get(
                "termination_reason",
                info.get("reason"),
            )

            if interrupted:
                end_reason = "interrupted"
            elif end_reason is None:
                if terminated:
                    end_reason = "terminated"
                elif truncated:
                    end_reason = "truncated"

            record = {
                "checkpoint": checkpoint,
                "episode": episode,
                "score": getattr(env, "score", info.get("score", 0)),
                "total_reward": total_reward,
                "steps": steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "end_reason": end_reason,
                "interrupted": interrupted,
            }

            results.append(record)

            # Briefly show the final board before starting the next evaluation episode.
            if running and not interrupted:
                pygame.time.wait(END_PAUSE_MS)
                clock.tick()  # don't count the pause as elapsed play time

            if not running:
                break

    finally:
        pygame.quit()

    # Append metrics so evaluations from multiple checkpoints can be compared in one file.
    if csv_path is not None and results:
        path = Path(csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        fields = list(results[0].keys())
        needs_header = not path.exists() or path.stat().st_size == 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)

            if needs_header:
                writer.writeheader()

            writer.writerows(results)

    return results
