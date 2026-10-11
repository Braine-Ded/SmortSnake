from replay_buffer import ReplayBuffer

import pytorch

class DQNAgent:

    def __init__(
        self,
        model: pytorch.nn.Module,
        optimizer: pytorch.optim.Optimizer,
        replay_buf: ReplayBuffer,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        gamma: float,
        batch_size: int,
        target_model,
        target_update_freq: int,
        learn_steps: int,
        device: str
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buf = replay_buf

