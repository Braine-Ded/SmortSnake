from ABC import ABC

class Agent(ABC):
    def act(self, observation, *, explore=True):
        """Select an action using the configured policy."""

    def observe(self, transition):
        """Receive a transition from the environment."""

    def learn(self):
        """Perform learning if the algorithm is ready."""
