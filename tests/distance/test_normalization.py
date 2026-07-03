"""Tests for distance normalization."""

from __future__ import annotations

from xwhy.distance.normalization import DistanceNormalizer


def test_min_max_normalization() -> None:
    """Normalize different distances."""
    scores = [
        ("a", 0.0),
        ("b", 5.0),
        ("c", 10.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores)

    assert result == [
        ("a", 1.0),
        ("b", 0.5),
        ("c", 0.0),
    ]


def test_min_max_equal_distances() -> None:
    """Return similarity of one when all distances are equal."""
    scores = [
        ("a", 2.0),
        ("b", 2.0),
        ("c", 2.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores)

    assert result == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
    ]
