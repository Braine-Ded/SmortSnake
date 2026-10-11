from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def select_action(self, observation, *, explore=True):
        """Return a valid environment action."""
