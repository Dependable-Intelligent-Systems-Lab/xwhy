"""Unit tests for PointCloudModelType."""

from __future__ import annotations

import pytest

from xwhy.models.point_cloud.types import PointCloudModelType


def test_from_str_with_valid_string() -> None:
    """Convert a valid string into the matching enum member."""
    result = PointCloudModelType.from_str("custom")
    assert result is PointCloudModelType.CUSTOM


def test_from_str_with_enum_instance() -> None:
    """Return the same enum member when an instance is supplied."""
    result = PointCloudModelType.from_str(PointCloudModelType.CUSTOM)
    assert result is PointCloudModelType.CUSTOM


def test_from_str_with_invalid_string_raises() -> None:
    """Raise ValueError for a string that is not a known model type."""
    with pytest.raises(ValueError, match="is not a valid PointCloudModelType"):
        PointCloudModelType.from_str("unknown_model")


def test_enum_value() -> None:
    """Expose the expected string value for the CUSTOM member."""
    assert PointCloudModelType.CUSTOM.value == "custom"
