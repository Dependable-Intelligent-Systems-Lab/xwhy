"""Test distance types module."""

import re

import pytest

from xwhy.distance.types import DistanceType


def test_distancetype_from_str_valid_string() -> None:
    """Verify that a valid string is successfully converted to Enum."""
    assert DistanceType.from_str("cosine") == DistanceType.COSINE


def test_distancetype_from_str_valid_enum() -> None:
    """Verify that passing an Enum directly returns the same Enum."""
    assert DistanceType.from_str(DistanceType.WASSERSTEIN) == DistanceType.WASSERSTEIN


def test_distancetype_from_str_invalid() -> None:
    """Ensure ValueError is raised for invalid strings."""
    with pytest.raises(ValueError, match=re.escape("is not a valid DistanceType")):
        DistanceType.from_str("invalid_metric")


def test_distancetype_is_text_metric() -> None:
    """WMD is a text metric; numeric metrics are not."""
    assert DistanceType.WMD.is_text_metric is True
    assert DistanceType.COSINE.is_text_metric is False


def test_distancetype_is_numeric_metric() -> None:
    """Numeric metrics return True; WMD returns False."""
    assert DistanceType.COSINE.is_numeric_metric is True
    assert DistanceType.WMD.is_numeric_metric is False
