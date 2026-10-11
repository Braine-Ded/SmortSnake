from __future__ import annotations

import copy

import torch

from .base import Agent
from ..buffers.base import Buffer
from ..metadata import Transition
from ..policies.base import Policy

class DQNAgent(Agent):

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        buffer: Buffer,
        policy: Policy,
        gamma: float,
        batch_size: int,
        target_update_freq: int,
        learn_steps: int,
        device: str | torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.buffer = buffer
        self.policy = policy

        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_steps = learn_steps
        self.device = device

        # Number of gradient updates performed so far (drives target syncing).
        self._update_count = 0

        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.target_model.eval()
        self.target_model.requires_grad_(False)


    def act(self, observation, *, explore=True):
        # The policy owns the model call: it turns the observation into a
        # batched tensor, runs the network and returns a plain int action.
        return self.policy.select_action(observation, explore=explore)

    def observe(self, transition: Transition):
        self.buffer.push(transition)

    def learn(self):
        # Not enough experience yet to form a batch.
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)
        # batch.truncated is intentionally unused: a time-limit ending is not
        # a real terminal state, so the target still bootstraps through it.

        # Prediction: Q(s, a) from the online network, only for the action
        # that was actually taken.   (B, 3) -> (B,)
        predictions = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a'), no gradient.   (B,)
        with torch.no_grad():
            next_q = self.target_model(next_states).max(dim=1).values
            targets = rewards + self.gamma * (1.0 - terminated) * next_q

        loss = self.loss_fn(predictions, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically copy the online weights into the target network.
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()
