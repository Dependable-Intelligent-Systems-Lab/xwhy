"""Base surrogate abstractions."""

from abc import ABC, abstractmethod


class BaseSurrogate(ABC):
    """Base class for surrogate component.

    Full implementation in later phases.
    """

    @abstractmethod
    def __placeholder_method__(self, *args: object, **kwargs: object) -> None:
        """Implement this method in subclasses."""
        raise NotImplementedError("To be implemented in later phases.")
