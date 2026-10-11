from ABC import ABC

class QNetwork(ABC):
    def forward(self, observations):
        """Return Q-values for a batch of observations."""
