"""DQN + MLP on the 11 hand-crafted features (the original setup).

Every component the run uses is built here; train.py / eval.py / play.py only
call `make_env` and `make_agent`. Select this config from config.py with:

    from configs.dqn_features import *
"""

import torch

from src import SnakeEnv
from src.agents import DQNAgent
from src.buffers import ReplayBuffer
from src.networks import MLP
from src.observations import FeatureObservation
from src.policies import EpsilonGreedyPolicy, GreedyPolicy

# What train.py / eval.py / play.py rely on. Everything else is private.
__all__ = ["NAME", "SEED", "MAX_EPISODES", "make_env", "make_agent"]

NAME = "dqn_features"   # stored in checkpoints so a resume can't mix configs

# ---------- Reproducibility ----------
# Global seed: drives the env, replay buffer, exploration policy, network init
# and python/numpy/torch RNGs in train.py (eval/play/tests derive from it too).
SEED = 67

# ---------- Environment ----------
GRID_SIZE = (10, 10)
START_SIZE = 3

# ---------- Optimisation ----------
LEARNING_RATE = 1e-3
GAMMA = 0.9

# ---------- Exploration (epsilon decays once per episode) ----------
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# ---------- DQN ----------
BATCH_SIZE = 64
REPLAY_CAPACITY = 100_000      # keep this bounded; the buffer is preallocated
TARGET_UPDATE_FREQ = 200       # in gradient updates

# ---------- Run length ----------
MAX_EPISODES = 5000


def make_env(seed: int) -> SnakeEnv:
    return SnakeEnv(
        GRID_SIZE,
        START_SIZE,
        tuple(size // 2 for size in GRID_SIZE),
        seed=seed,
        observation=FeatureObservation(),
    )


def make_agent(env: SnakeEnv, device: torch.device, seed: int) -> DQNAgent:
    model = MLP(
        input_size=env.observation_spec.size,
        output_size=env.NUM_ACTIONS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.SmoothL1Loss()
    buffer = ReplayBuffer(capacity=REPLAY_CAPACITY, seed=seed)

    # The epsilon-greedy policy wraps a greedy one that reads the *online* model.
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
