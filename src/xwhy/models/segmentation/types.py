"""Segmentation type definitions."""

from __future__ import annotations

from enum import StrEnum


class SegmentationType(StrEnum):
    """Supported segmentation backends."""

    DEEPLABV3_RESNET101 = "deeplabv3_resnet101"
    DEEPLABV3_RESNET50 = "deeplabv3_resnet50"
    DEEPLABV3_MOBILENET_V3 = "deeplabv3_mobilenet_v3_large"
    FCN_RESNET50 = "fcn_resnet50"
    LRASPP_MOBILENET_V3 = "lraspp_mobilenet_v3_large"

    @classmethod
    def from_str(cls, value: str | SegmentationType) -> SegmentationType:
        """Safely convert a string or enum instance to SegmentationType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = ", ".join([item.value for item in cls])
            raise ValueError(
                f"'{value}' is not a valid SegmentationType. "
                f"Supported options are: [{valid_options}]"
            ) from err
