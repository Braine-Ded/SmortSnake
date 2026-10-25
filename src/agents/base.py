from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, observation, *, explore=True):
        """Select an action using the configured policy."""

    @abstractmethod
    def observe(self, transition):
        """Receive a transition from the environment."""

    @abstractmethod
    def learn(self):
        """Perform learning if the algorithm is ready."""

    def end_episode(self) -> None:
        """Called once after every training episode."""

    def metrics(self) -> dict:
        """Extra per-episode values for the training log."""
        return {}

    @abstractmethod
    def weights(self) -> dict:
        """What is needed to act: saved as best_<run>.pt / final_<run>.pt and loaded by eval."""

    @abstractmethod
    def load_weights(self, weights: dict) -> None:
        """Restore the output of `weights()`."""

    @abstractmethod
    def state_dict(self) -> dict:
        """Full training state (networks, optimizer, buffer, policy) for --resume."""

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        """Restore the output of `state_dict()`."""
