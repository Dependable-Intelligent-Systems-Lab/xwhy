"""Unit tests for segmentation types."""

import pytest

from xwhy.models.segmentation.types import SegmentationType


def test_segmentation_type_from_str_success() -> None:
    """Test successful conversion from valid strings."""
    assert (
        SegmentationType.from_str("deeplabv3_resnet101")
        == SegmentationType.DEEPLABV3_RESNET101
    )
    assert (
        SegmentationType.from_str("deeplabv3_resnet50")
        == SegmentationType.DEEPLABV3_RESNET50
    )
    assert (
        SegmentationType.from_str("deeplabv3_mobilenet_v3_large")
        == SegmentationType.DEEPLABV3_MOBILENET_V3
    )
    assert SegmentationType.from_str("fcn_resnet50") == SegmentationType.FCN_RESNET50
    assert (
        SegmentationType.from_str("lraspp_mobilenet_v3_large")
        == SegmentationType.LRASPP_MOBILENET_V3
    )


def test_segmentation_type_from_str_invalid() -> None:
    """Test that invalid input raises ValueError with a clear message."""
    invalid_input = "invalid_segmentation"

    with pytest.raises(
        ValueError, match=f"'{invalid_input}' is not a valid SegmentationType"
    ):
        SegmentationType.from_str(invalid_input)
