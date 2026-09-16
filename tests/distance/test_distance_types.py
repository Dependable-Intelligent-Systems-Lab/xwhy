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
