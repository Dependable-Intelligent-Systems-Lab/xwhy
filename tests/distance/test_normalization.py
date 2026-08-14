"""Tests for distance normalization."""

from __future__ import annotations

from xwhy.distance.normalization import DistanceNormalizer


def test_min_max_empty_scores() -> None:
    """Test empty scores list returns empty list."""
    result = DistanceNormalizer.min_max(scores=[])
    assert result == []


def test_min_max_normalization() -> None:
    """Normalize different distances using linear mode."""
    scores = [
        ("a", 0.0),
        ("b", 5.0),
        ("c", 10.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores, mode="linear")

    assert result == [
        ("a", 1.0),
        ("b", 0.5),
        ("c", 0.0),
    ]


def test_min_max_equal_distances() -> None:
    """Return similarity of one when linear distances are equal."""
    scores = [
        ("a", 2.0),
        ("b", 2.0),
        ("c", 2.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores, mode="linear")

    assert result == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
    ]


def test_min_max_inverse_normalization() -> None:
    """Normalize distances using inverse mode."""
    scores = [
        ("a", 0.0),
        ("b", 5.0),
        ("c", 10.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores, mode="inverse")

    assert len(result) == 3
    assert result[0] == ("a", 1.0)
    assert result[2] == ("c", 0.0)


def test_min_max_inverse_equal_distances() -> None:
    """Return similarity of one when inverse distances are equal."""
    scores = [
        ("a", 2.0),
        ("b", 2.0),
        ("c", 2.0),
    ]

    result = DistanceNormalizer.min_max(scores=scores, mode="inverse")

    assert result == [
        ("a", 1.0),
        ("b", 1.0),
        ("c", 1.0),
    ]
