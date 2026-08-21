"""Type definitions for point cloud models."""

from __future__ import annotations

from enum import StrEnum


class PointCloudModelType(StrEnum):
    """Supported point cloud model types."""

    CUSTOM = "custom"
    HUGGINGFACE = "huggingface"

    @classmethod
    def from_str(cls, value: str | PointCloudModelType) -> PointCloudModelType:
        """Safely convert string or enum to PointCloudModelType.

        Args:
            value: String or PointCloudModelType instance.

        Returns:
            PointCloudModelType enum member.

        Raises:
            ValueError: If string is invalid.

        """
        try:
            return cls(value)
        except ValueError as err:
            valid_options = ", ".join([item.value for item in cls])
            raise ValueError(
                f"'{value}' is not a valid PointCloudModelType. "
                f"Supported options are: [{valid_options}]"
            ) from err
