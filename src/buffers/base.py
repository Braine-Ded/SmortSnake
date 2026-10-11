from abc import ABC, abstractmethod

from ..metadata import Transition, TransitionBatch

class Buffer(ABC):
    def __init__(self, capacity: int):
        """Create bounded experience storage."""

    @abstractmethod
    def push(self, transition: Transition) -> None:
        """Store one transition."""

    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
        """Return a random batch without replacement."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of stored transitions."""
