"""Types for perturbation strategies."""

from __future__ import annotations

from enum import StrEnum


class SuperpixelType(StrEnum):
    """Supported superpixel segmentation algorithms for image perturbation."""

    QUICKSHIFT = "quickshift"
    SLIC = "slic"
    FELZENSZWALB = "felzenszwalb"

    @classmethod
    def from_str(cls, value: str | SuperpixelType) -> SuperpixelType:
        """Resolve a string or enum instance into a SuperpixelType.

        Args:
            value: Algorithm name as string or SuperpixelType enum.

        Returns:
            SuperpixelType: The corresponding enum member.

        Raises:
            ValueError: If the provided string does not match any SuperpixelType.

        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            for member in cls:
                if member.value == normalized:
                    return member
        valid_values = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Invalid superpixel type '{value}'. Must be one of: {valid_values}."
        )
