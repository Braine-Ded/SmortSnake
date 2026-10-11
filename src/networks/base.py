from abc import ABC, abstractmethod

class QNetwork(ABC):
    @abstractmethod
    def forward(self, observations):
        """Return Q-values for a batch of observations."""
