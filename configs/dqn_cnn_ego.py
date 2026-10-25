"""DQN + CNN on the egocentric board (EgocentricPlaneObservation: planes rotated
so the heading always points up).

Select this config from config.py with:

    from configs.dqn_cnn_ego import *
"""

import torch

from src import EnvSettings, Rewards, SnakeEnv
from src.agents import DQNAgent
from src.buffers import ReplayBuffer
from src.networks import CNN
from src.observations import EgocentricPlaneObservation
from src.policies import EpsilonGreedyPolicy, GreedyPolicy

__all__ = [
    "NAME", "SEED", "MAX_EPISODES", "make_env", "make_agent",
    "ALGO", "POLICY", "NET", "OBS", "ARCH", "GAMMA", "ENV",
]

NAME = "dqn_cnn_ego"

# ---------- Run name labels (see src/naming.py) ----------
ALGO = "dqn"
POLICY = "egp"
NET = "cnn"
OBS = "ego"

# ---------- Reproducibility ----------
SEED = 67

# ---------- Environment (grid, episode limit, rewards) ----------
# A loop the agent expects to last forever is worth rewards.step / (1 - gamma);
# at the default -0.01 that was ~free, so circling beat any risky path to food.
ENV = EnvSettings(rewards=Rewards(step=-0.1))

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

# ---------- CNN Architecture ----------
CNN_CONV1_CHANNELS = 16
CNN_CONV2_CHANNELS = 32
CNN_HIDDEN_SIZE = 64
ARCH = (CNN_CONV1_CHANNELS, CNN_CONV2_CHANNELS, CNN_HIDDEN_SIZE)


def make_env(seed: int) -> SnakeEnv:
    return SnakeEnv.from_settings(
        ENV, seed=seed, observation=EgocentricPlaneObservation(ENV.grid_size),
    )


def make_agent(env: SnakeEnv, device: torch.device, seed: int) -> DQNAgent:
    model = CNN(
        input_shape=env.observation_spec.shape,
        conv1_channels=CNN_CONV1_CHANNELS,
        conv2_channels=CNN_CONV2_CHANNELS,
        hidden_size=CNN_HIDDEN_SIZE,
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
