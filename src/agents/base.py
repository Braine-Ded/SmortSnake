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
