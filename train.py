import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from src import Transition
from src.config_loader import load_config
from src.logs import ask_log_path, log_path as resolve_log_path

# The whole setup (env, observation, network, optimizer, loss, buffer, policy,
# agent) comes from the config selected in config.py; nothing is built here.
config = load_config()

SEED = config.SEED
# One folder per config, so training one setup never overwrites another's weights.
CHECKPOINT_DIR = Path("checkpoints") / config.NAME
STATE_PATH = CHECKPOINT_DIR / "state.pt"   # full training state, used by --resume
LOG_EVERY = 10      # print a progress line every N episodes
SAVE_EVERY = 50     # write the resumable state every N episodes
WINDOW = 100        # moving-average window used for "best model" tracking


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Snake agent.")
    parser.add_argument(
        "--resume", nargs="?", const=str(STATE_PATH), default=None, metavar="PATH",
        help=f"resume from a training-state checkpoint (default: {STATE_PATH})",
    )
    parser.add_argument(
        "--log", default=None,
        help="log file name under ./logs (skips the interactive prompt). "
             "When resuming, rows are appended to this file.",
    )
    return parser.parse_args()


def save_state(path, *, episode, agent, recent_scores, best_avg_score):
    """Atomically write everything needed to continue training later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "config": config.NAME,
            "episode": episode,                      # last *completed* episode
            "agent": agent.state_dict(),
            "recent_scores": list(recent_scores),
            "best_avg_score": best_avg_score,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash mid-write can't corrupt the old file


def main():
    args = parse_args()

    if args.log:
        log_path = resolve_log_path(args.log)
    else:
        log_path = ask_log_path(f"training_{config.NAME}.csv")

    # Seed every global RNG (network init, anything using random/np.random).
    # The env, buffer and policy additionally get their own private seeded RNGs.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Load a checkpoint first (if resuming) ----------
    state = None
    start_episode = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"Cannot resume: {resume_path} does not exist.")
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if "agent" not in state:
            raise SystemExit(f"Cannot resume: {resume_path} predates config-based setups.")
        if state["config"] != config.NAME:
            raise SystemExit(
                f"Cannot resume: {resume_path} was trained with config "
                f"'{state['config']}', but config.py selects '{config.NAME}'."
            )
        start_episode = state["episode"] + 1
        print(f"Resuming {config.NAME} from {resume_path} "
              f"(completed episode {state['episode']})")

    # Different seed per resume so the env/policy don't replay the same
    # random sequence as the start of the first run.
    run_seed = SEED + (start_episode - 1)

    # ---------- Build the pieces ----------
    env = config.make_env(run_seed)
    agent = config.make_agent(env, device, run_seed)

    recent_scores = deque(maxlen=WINDOW)
    best_avg_score = float("-inf")

    # ---------- Restore state ----------
    if state is not None:
        agent.load_state_dict(state["agent"])
        recent_scores.extend(state["recent_scores"])
        best_avg_score = state["best_avg_score"]
        del state

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Agent-specific columns (e.g. epsilon) sit where epsilon always did.
    log_fields = [
        "episode", "score", "total_reward", "steps",
        *agent.metrics(), "mean_loss", "avg_score",
    ]

    # Append to the existing log when resuming; otherwise start a fresh one.
    append = args.resume is not None and log_path.exists() and log_path.stat().st_size > 0
    completed = start_episode - 1  # last fully finished episode

    print(f"Training config: {config.NAME}")

    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=log_fields)
        if not append:
            writer.writeheader()

        try:
            for episode in range(start_episode, config.MAX_EPISODES + 1):
                observation, _ = env.reset()
                done = False

                total_reward = 0.0
                steps = 0
                losses = []
                info = {}

                while not done:
                    # 1. choose, 2. act in the world
                    action = agent.act(observation)
                    next_observation, reward, terminated, truncated, info = env.step(action)

                    # 3. remember, 4. learn (does nothing until the buffer is full enough)
                    agent.observe(
                        Transition(
                            observation, action, reward,
                            next_observation, terminated, truncated,
                        )
                    )
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)

                    # 5. the new observation is the next step's input
                    observation = next_observation
                    total_reward += reward
                    steps += 1
                    done = terminated or truncated

                # Per-episode bookkeeping, e.g. epsilon decay.
                agent.end_episode()
                metrics = agent.metrics()

                score = info["score"]
                recent_scores.append(score)
                avg_score = sum(recent_scores) / len(recent_scores)
                mean_loss = sum(losses) / len(losses) if losses else None

                writer.writerow({
                    "episode": episode,
                    "score": score,
                    "total_reward": round(total_reward, 4),
                    "steps": steps,
                    **{name: round(value, 5) for name, value in metrics.items()},
                    "mean_loss": mean_loss,
                    "avg_score": round(avg_score, 4),
                })
                log_file.flush()  # so the CSV is intact if the run is killed
                completed = episode

                # Keep the best model, judged on a full moving-average window.
                if len(recent_scores) == WINDOW and avg_score > best_avg_score:
                    best_avg_score = avg_score
                    torch.save(agent.weights(), CHECKPOINT_DIR / "best.pt")

                if episode % SAVE_EVERY == 0:
                    save_state(
                        STATE_PATH, episode=completed, agent=agent,
                        recent_scores=recent_scores, best_avg_score=best_avg_score,
                    )

                if episode % LOG_EVERY == 0:
                    loss_text = "n/a" if mean_loss is None else f"{mean_loss:.4f}"
                    metrics_text = "".join(
                        f" | {name} {value:.3f}" for name, value in metrics.items()
                    )
                    print(
                        f"ep {episode:4d} | score {score:3d} | "
                        f"avg{WINDOW} {avg_score:6.2f}{metrics_text} | "
                        f"steps {steps:4d} | loss {loss_text}"
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted; saving the current state.")

        finally:
            torch.save(agent.weights(), CHECKPOINT_DIR / "final.pt")
            save_state(
                STATE_PATH, episode=completed, agent=agent,
                recent_scores=recent_scores, best_avg_score=best_avg_score,
            )

    print(f"Done. Best moving-average score: {best_avg_score:.2f}")
    print(f"Weights: {CHECKPOINT_DIR}/  |  metrics: {log_path}")
    print(f"Resume with: python train.py --resume --log {log_path.name}")


if __name__ == "__main__":
    main()
