"""Base perturbation abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class BasePerturbation(ABC):
    """Abstract base class for perturbation strategies."""

    @abstractmethod
    def generate(
        self,
        *,
        text: str,
        num_perturbations: int = 64,
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        """Generate perturbed samples."""

    @abstractmethod
    def apply_mask(
        self,
        *,
        words: Sequence[str],
        mask: Sequence[int],
    ) -> list[str]:
        """Apply a perturbation mask."""
