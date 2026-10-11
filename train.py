import csv
from collections import deque
from pathlib import Path

import torch

import config
from src import SnakeEnv, Transition
from src.agents import DQNAgent
from src.buffers import ReplayBuffer
from src.networks import MLP
from src.policies import EpsilonGreedyPolicy, GreedyPolicy

SEED = 67
CHECKPOINT_DIR = Path("checkpoints")
LOG_PATH = Path("logs") / "training.csv"
LOG_EVERY = 10      # print a progress line every N episodes
WINDOW = 100        # moving-average window used for "best model" tracking

LOG_FIELDS = [
    "episode", "score", "total_reward", "steps",
    "epsilon", "mean_loss", "avg_score",
]


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Build the pieces ----------
    env = SnakeEnv(
        config.GRID_SIZE,
        3,
        tuple(size // 2 for size in config.GRID_SIZE),
        seed=SEED,
    )

    model = MLP(
        input_size=env.OBSERVATION_SIZE,
        output_size=env.NUM_ACTIONS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=config.REPLAY_CAPACITY, seed=SEED)

    # The epsilon-greedy policy wraps a greedy one that reads the *online* model.
    policy = EpsilonGreedyPolicy(
        GreedyPolicy(model, device=device),
        num_actions=env.NUM_ACTIONS,
        epsilon_start=config.EPSILON_START,
        epsilon_min=config.EPSILON_MIN,
        epsilon_decay=config.EPSILON_DECAY,
        seed=SEED,
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

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_PATH.parent.mkdir(exist_ok=True)

    recent_scores = deque(maxlen=WINDOW)
    best_avg_score = float("-inf")

    with LOG_PATH.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        writer.writeheader()

        try:
            for episode in range(1, config.MAX_EPISODES + 1):
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

                # Keep the best model, judged on a full moving-average window.
                if len(recent_scores) == WINDOW and avg_score > best_avg_score:
                    best_avg_score = avg_score
                    torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")

                if episode % LOG_EVERY == 0:
                    loss_text = "n/a" if mean_loss is None else f"{mean_loss:.4f}"
                    print(
                        f"ep {episode:4d} | score {score:3d} | "
                        f"avg{WINDOW} {avg_score:6.2f} | eps {policy.epsilon:.3f} | "
                        f"steps {steps:4d} | loss {loss_text}"
                    )

        except KeyboardInterrupt:
            print("\nTraining interrupted; saving the current model.")

        finally:
            torch.save(model.state_dict(), CHECKPOINT_DIR / "final.pt")

    print(f"Done. Best moving-average score: {best_avg_score:.2f}")
    print(f"Weights: {CHECKPOINT_DIR}/  |  metrics: {LOG_PATH}")


if __name__ == "__main__":
    main()
