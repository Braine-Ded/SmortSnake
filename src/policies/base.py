from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select_action(self, observation, *, explore=True):
        """Return a valid environment action."""

    def end_episode(self) -> None:
        """Called once after every training episode (e.g. to decay exploration)."""

    def metrics(self) -> dict:
        """Values worth logging each episode, e.g. {"epsilon": 0.3}."""
        return {}

    def state_dict(self) -> dict:
        """Mutable state to checkpoint; stateless policies return {}."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass
