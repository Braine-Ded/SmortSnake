import argparse
import csv
from collections import deque
from pathlib import Path

import torch

import config
from src import SnakeEnv, Transition
from src.agents import DQNAgent
from src.buffers import ReplayBuffer
from src.logs import ask_log_path, log_path as resolve_log_path
from src.networks import MLP
from src.policies import EpsilonGreedyPolicy, GreedyPolicy

SEED = 67
CHECKPOINT_DIR = Path("checkpoints")
STATE_PATH = CHECKPOINT_DIR / "state.pt"   # full training state, used by --resume
LOG_EVERY = 10      # print a progress line every N episodes
SAVE_EVERY = 50     # write the resumable state every N episodes
WINDOW = 100        # moving-average window used for "best model" tracking

LOG_FIELDS = [
    "episode", "score", "total_reward", "steps",
    "epsilon", "mean_loss", "avg_score",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Snake DQN agent.")
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


def save_state(path, *, episode, model, agent, optimizer, policy, buffer,
               recent_scores, best_avg_score):
    """Atomically write everything needed to continue training later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "episode": episode,                      # last *completed* episode
            "model": model.state_dict(),
            "target_model": agent.target_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "update_count": agent._update_count,
            "epsilon": policy.epsilon,
            "recent_scores": list(recent_scores),
            "best_avg_score": best_avg_score,
            "buffer": buffer.state_dict(),
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash mid-write can't corrupt the old file


def main():
    args = parse_args()

    if args.log:
        log_path = resolve_log_path(args.log)
    else:
        log_path = ask_log_path("training.csv")

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
        start_episode = state["episode"] + 1
        print(f"Resuming from {resume_path} (completed episode {state['episode']}, "
              f"epsilon {state['epsilon']:.4f})")

    # Different seed per resume so the env/policy don't replay the same
    # random sequence as the start of the first run.
    run_seed = SEED + (start_episode - 1)

    # ---------- Build the pieces ----------
    env = SnakeEnv(
        config.GRID_SIZE,
        3,
        tuple(size // 2 for size in config.GRID_SIZE),
        seed=run_seed,
    )

    model = MLP(
        input_size=env.OBSERVATION_SIZE,
        output_size=env.NUM_ACTIONS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=config.REPLAY_CAPACITY, seed=run_seed)

    # The epsilon-greedy policy wraps a greedy one that reads the *online* model.
    policy = EpsilonGreedyPolicy(
        GreedyPolicy(model, device=device),
        num_actions=env.NUM_ACTIONS,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        seed=run_seed,
    )

    agent = DQNAgent(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        buffer=buffer,
        policy=policy,
        gamma=config.GAMMA,
        batch_size=config.BATCH_SIZE,
        target_update_freq=config.TARGET_UPDATE_FREQ,
        device=device,
    )

    recent_scores = deque(maxlen=WINDOW)
    best_avg_score = float("-inf")

    # ---------- Restore state ----------
    if state is not None:
        model.load_state_dict(state["model"])
        agent.target_model.load_state_dict(state["target_model"])
        optimizer.load_state_dict(state["optimizer"])
        agent._update_count = state["update_count"]
        policy.epsilon = state["epsilon"]
        recent_scores.extend(state["recent_scores"])
        best_avg_score = state["best_avg_score"]
        buffer.load_state_dict(state["buffer"])
        del state

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append to the existing log when resuming; otherwise start a fresh one.
    append = args.resume is not None and log_path.exists() and log_path.stat().st_size > 0
    completed = start_episode - 1  # last fully finished episode

    with log_path.open("a" if append else "w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
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

                # Epsilon decays once per episode.
                policy.decay()

                score = info["score"]
                recent_scores.append(score)
                avg_score = sum(recent_scores) / len(recent_scores)
                mean_loss = sum(losses) / len(losses) if losses else None

                writer.writerow({
                    "episode": episode,
                    "score": score,
                    "total_reward": round(total_reward, 4),
                    "steps": steps,
                    "epsilon": round(policy.epsilon, 5),
                    "mean_loss": mean_loss,
                    "avg_score": round(avg_score, 4),
                })
                log_file.flush()  # so the CSV is intact if the run is killed
                completed = episode

                # Keep the best model, judged on a full moving-average window.
                if len(recent_scores) == WINDOW and avg_score > best_avg_score:
                    best_avg_score = avg_score
                    torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")

                if episode % SAVE_EVERY == 0:
                    save_state(
                        STATE_PATH, episode=completed, model=model, agent=agent,
                        optimizer=optimizer, policy=policy, buffer=buffer,
                        recent_scores=recent_scores, best_avg_score=best_avg_score,
                    )

                if episode % LOG_EVERY == 0:
                    loss_text = "n/a" if mean_loss is None else f"{mean_loss:.4f}"
                    print(
                        f"ep {episode:4d} | score {score:3d} | "
                        f"avg{WINDOW} {avg_score:6.2f} | eps {policy.epsilon:.3f} | "
                        f"steps {steps:4d} | loss {loss_text}"
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted; saving the current state.")

        finally:
            torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")
            save_state(
                STATE_PATH, episode=completed, model=model, agent=agent,
                optimizer=optimizer, policy=policy, buffer=buffer,
                recent_scores=recent_scores, best_avg_score=best_avg_score,
            )

    print(f"Done. Best moving-average score: {best_avg_score:.2f}")
    print(f"Weights: {CHECKPOINT_DIR}/  |  metrics: {log_path}")
    print(f"Resume with: python train.py --resume --log {log_path.name}")


if __name__ == "__main__":
    main()
