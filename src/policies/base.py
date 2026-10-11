from ABC import ABC

class Policy(ABC):
    def select_action(self, observation, *, explore=True):
        """Return a valid environment action."""
