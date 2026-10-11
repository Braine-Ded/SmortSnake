from ABC import ABC

class ReplayBuffer(ABC):
    def __init__(self, capacity: int):
        """Create bounded experience storage."""

    def add(self, transition: Transition) -> None:
        """Store one transition."""

    def sample(self, batch_size: int) -> TransitionBatch:
        """Return a random batch without replacement."""

    def __len__(self) -> int:
        """Return the number of stored transitions."""
