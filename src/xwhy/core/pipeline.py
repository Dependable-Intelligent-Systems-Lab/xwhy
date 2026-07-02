"""Explanation pipeline abstractions."""

from abc import ABC, abstractmethod

from xwhy.core.result import XWhyResult


class ExplanationPipeline(ABC):
    """Abstract pipeline orchestrator for explanation process.

    Full implementation in later phases.
    """

    @abstractmethod
    def run(self, instance: object, **kwargs: object) -> XWhyResult:
        """Run the full explanation pipeline."""
        raise NotImplementedError("Subclasses must implement run method.")
