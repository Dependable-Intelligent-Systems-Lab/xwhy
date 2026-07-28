"""Explanation pipeline abstractions."""

from abc import ABC, abstractmethod
from typing import Any

from xwhy.core.result import BaseXWhyResult


class ExplanationPipeline(ABC):
    """Abstract pipeline orchestrator for explanation process.

    Full implementation in later phases.
    """

    @abstractmethod
    def run(self, instance: Any, **kwargs: Any) -> BaseXWhyResult:  # noqa: ANN401
        """Run the full explanation pipeline."""
        raise NotImplementedError("Subclasses must implement run method.")
