"""DQN + CNN on the full board (PlaneObservation: wall/body/head/food planes).

Select this config from config.py with:

    from configs.dqn_cnn import *
"""

import torch

from src import SnakeEnv
from src.agents import DQNAgent
from src.buffers import ReplayBuffer
from src.networks import CNN
from src.observations import PlaneObservation
from src.policies import EpsilonGreedyPolicy, GreedyPolicy

__all__ = ["NAME", "SEED", "MAX_EPISODES", "make_env", "make_agent"]

NAME = "dqn_cnn"

# ---------- Reproducibility ----------
SEED = 67

# ---------- Environment ----------
GRID_SIZE = (10, 10)
START_SIZE = 3

# ---------- Optimisation ----------
LEARNING_RATE = 1e-4
# Longer horizon than the feature config: seeing the whole board is only
# useful if the agent plans far enough ahead to avoid trapping itself.
GAMMA = 0.95

# ---------- Exploration (epsilon decays once per episode) ----------
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.998          # slower: pixels take longer to learn from

# ---------- DQN ----------
BATCH_SIZE = 64
# Each stored transition is 2 x (4, 12, 12) float32 planes (~4.6 KB),
# so 50k transitions is roughly 230 MB of replay memory.
REPLAY_CAPACITY = 50_000
TARGET_UPDATE_FREQ = 500       # in gradient updates

# ---------- Run length ----------
MAX_EPISODES = 10_000


def make_env(seed: int) -> SnakeEnv:
    return SnakeEnv(
        GRID_SIZE,
        START_SIZE,
        tuple(size // 2 for size in GRID_SIZE),
        seed=seed,
        observation=PlaneObservation(GRID_SIZE),
    )


def make_agent(env: SnakeEnv, device: torch.device, seed: int) -> DQNAgent:
    model = CNN(
        input_shape=env.observation_spec.shape,
        output_size=env.NUM_ACTIONS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=REPLAY_CAPACITY, seed=seed)

    policy = EpsilonGreedyPolicy(
        GreedyPolicy(model, device=device),
        num_actions=env.NUM_ACTIONS,
        epsilon_start=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=seed,
    )

    return DQNAgent(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        buffer=buffer,
        policy=policy,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        device=device,
    )
