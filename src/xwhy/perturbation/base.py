"""Base perturbation abstractions."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

T_Input = TypeVar("T_Input")
T_Mask = TypeVar("T_Mask")
T_Output = TypeVar("T_Output")


class BasePerturbation[T_Input, T_Mask, T_Output](ABC):
    """Abstract base class for perturbation strategies."""

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Generate perturbed samples or masks."""

    @abstractmethod
    def apply_mask(
        self,
        item: T_Input,
        mask: T_Mask,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> T_Output:
        """Apply a perturbation mask to the input item."""
