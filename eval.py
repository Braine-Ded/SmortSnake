"""Watch (or batch-evaluate) a trained agent and log its performance.

Examples (run from the project root):

    python eval.py                                    # asks for a model (blank = best.pt), 10 episodes
    python eval.py --checkpoint my_model              # checkpoints/my_model.pt, no prompt
    python eval.py --episodes 25 --speed 3            # more episodes, 3x playback speed
    python eval.py --speed 0.5                        # slow motion
    python eval.py --headless --episodes 500          # no window: as fast as possible
    python eval.py --checkpoint checkpoints/best.pt checkpoints/final.pt
    python eval.py --no-log                           # no CSV (and no file-name prompt)

Without --checkpoint you are asked for a model name (blank = best.pt). Names are
looked up under ./checkpoints (".pt" is added if missing), or used as a path if
the file exists as given. A missing model is an error; there is no silent
substitution, so results are never logged under the wrong model.

You are asked for a log file name (blank = eval.csv, paths are relative to ./logs).
The agent plays greedily (no exploration). Every run appends one row per
episode to the log CSV, tagged with the checkpoint name, so evaluations of
different checkpoints can be compared in a single file.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

import config
from src import SnakeEnv
from src.logs import ask_log_path
from src.networks import MLP
from src.policies import GreedyPolicy

ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_CHECKPOINT_NAME = "best.pt"

# Evaluation uses a different food sequence than training: config.SEED + offset.
# (With the default SEED = 67 this gives the old fixed eval seed of 123.)
EVAL_SEED_OFFSET = 56


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RLSnake agent.")
    parser.add_argument(
        "--checkpoint", nargs="+", default=None, metavar="MODEL",
        help="one or more models (name under ./checkpoints, or a path); each is "
             f"evaluated in turn. Omit to be asked (blank = {DEFAULT_CHECKPOINT_NAME})",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="number of simulated episodes per checkpoint",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="playback speed multiplier: 2 = twice as fast, 0.5 = half speed. "
             "Above ~8x several steps run per rendered frame (frames are "
             "skipped). Ignored with --headless.",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="skip the pygame window and simulate as fast as possible",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="environment seed (each checkpoint sees the same food sequence); "
             f"default: config.SEED + {EVAL_SEED_OFFSET}",
    )
    parser.add_argument("--no-log", action="store_true", help="do not write a CSV")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.seed is None:
        args.seed = config.SEED + EVAL_SEED_OFFSET

    return args


def resolve_checkpoint(name: str) -> Path:
    """Find a weight file from a typed name or path, or exit with an error."""
    given = Path(name).expanduser()
    candidates = [given]

    # A bare name like "model" or "model.pt" is looked up in ./checkpoints.
    if not given.is_absolute() and given.parent == Path("."):
        candidates.append(CHECKPOINT_DIR / (given if given.suffix else given.with_suffix(".pt")))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    hint = "\nRun train.py first." if name == DEFAULT_CHECKPOINT_NAME else ""
    raise SystemExit(f"Checkpoint not found: {name} (looked in: {searched}){hint}")


def make_env(seed: int) -> SnakeEnv:
    return SnakeEnv(
        config.GRID_SIZE,
        3,
        tuple(size // 2 for size in config.GRID_SIZE),
        seed=seed,
    )


def load_policy(path: Path, env: SnakeEnv) -> GreedyPolicy:
    # One observation at a time is faster on the CPU than on a GPU.
    device = torch.device("cpu")

    model = MLP(
        input_size=env.OBSERVATION_SIZE,
        output_size=env.NUM_ACTIONS,
    ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return GreedyPolicy(model, device=device)


def run_headless(env, action_fn, *, episodes: int, checkpoint: str) -> list[dict]:
    """Same metrics as run_simulated, but with no rendering or frame limit."""
    results = []

    for episode in range(1, episodes + 1):
        observation, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        info = {}

        while not (terminated or truncated):
            action = int(action_fn(observation))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        results.append({
            "checkpoint": checkpoint,
            "episode": episode,
            "score": env.score,
            "total_reward": total_reward,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_reason": info.get("termination_reason"),
            "interrupted": False,
        })

    return results


def append_csv(path: Path, results: list[dict]) -> None:
    """Append rows in the same format run_simulated uses."""
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        if needs_header:
            writer.writeheader()
        writer.writerows(results)


def summarize(label: str, results: list[dict]) -> None:
    finished = [r for r in results if not r["interrupted"]]
    if not finished:
        print(f"[{label}] no completed episodes.")
        return

    n = len(finished)
    mean_score = sum(r["score"] for r in finished) / n
    mean_steps = sum(r["steps"] for r in finished) / n
    best = max(r["score"] for r in finished)
    endings = dict(Counter(r["end_reason"] for r in finished))

    print(
        f"[{label}] {n} episodes | mean score {mean_score:.2f} | best {best} | "
        f"mean steps {mean_steps:.1f} | endings {endings}"
    )


def main() -> None:
    args = parse_args()

    names = args.checkpoint
    if names is None:
        typed = input(f"Model checkpoint(s) [{DEFAULT_CHECKPOINT_NAME}]: ").split()
        names = typed or [DEFAULT_CHECKPOINT_NAME]

    checkpoints = [resolve_checkpoint(name) for name in names]

    log_path = None if args.no_log else ask_log_path("eval.csv")

    if not args.headless:
        # Imported lazily so --headless works without opening pygame at all.
        from src.pygame_view import MOVE_INTERVAL_MS, run_simulated

        interval_ms = max(1, round(MOVE_INTERVAL_MS / args.speed))

    for path in checkpoints:
        label = path.stem

        # A fresh env with the same seed per checkpoint => identical conditions.
        env = make_env(args.seed)
        policy = load_policy(path, env)

        if args.headless:
            results = run_headless(
                env, policy.select_action,
                episodes=args.episodes, checkpoint=label,
            )
            if log_path is not None:
                append_csv(log_path, results)
        else:
            results = run_simulated(
                env, policy.select_action,
                episodes=args.episodes,
                move_interval_ms=interval_ms,
                csv_path=log_path,
                checkpoint=label,
            )

        summarize(label, results)

        # Window closed / Esc pressed: stop instead of opening the next one.
        if any(r["interrupted"] for r in results):
            break

    if log_path is not None:
        print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
