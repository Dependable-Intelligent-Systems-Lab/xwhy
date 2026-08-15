"""Tests for Word Mover's Distance."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import numpy as np
import pytest
from gensim.models import KeyedVectors

from xwhy.distance.wmd import WMDDistance


class FakeModel(KeyedVectors):  # type: ignore[misc]
    """Fake Word2Vec model for testing purposes."""

    def __init__(self) -> None:
        """Initialize the fake Word2Vec model."""
        pass

    def __contains__(self, key: str) -> bool:
        """Return whether a word exists in the fake vocabulary.

        Args:
            key: The text token to check for presence.

        Returns:
            bool: True if the word is in the fake vocabulary, False otherwise.

        """
        return key in {
            "hello",
            "world",
            "python",
        }

    def wmdistance(
        self,
        words1: list[str],
        words2: list[str],
    ) -> float:
        """Return a fake constant distance for two lists of words.

        Args:
            words1: First list of words to compare.
            words2: Second list of words to compare.

        Returns:
            float: A predefined float distance value.

        """
        assert words1
        assert words2

        return 0.42


def test_clean_text() -> None:
    """Test text cleaning correctly removes punctuation and applies lowercase."""
    distance = WMDDistance()

    result = distance.clean_text(
        text=" Hello, WORLD!! ",
    )

    assert result == "hello world"


def test_compute_distance() -> None:
    """Test compute calculates the standard WMD distance appropriately."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="Hello world",
        target="Python",
    )

    assert result == 0.42


def test_compute_returns_default_for_unknown_words() -> None:
    """Test compute safely returns the default distance for unknown vocabulary."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="xxxx",
        target="yyyy",
    )

    assert result == 1.0


def test_compute_returns_default_when_first_is_empty() -> None:
    """Test compute safely returns default distance when the source has no tokens."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="!!!",
        target="hello",
    )

    assert result == 1.0


def test_compute_returns_default_when_second_is_empty() -> None:
    """Test compute safely returns default distance when the target has no tokens."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="hello",
        target="!!!",
    )

    assert result == 1.0


def test_compute_batch() -> None:
    """Test compute_batch calculating distances sequentially without sanitization."""
    distance = WMDDistance()

    distance.compute = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            0.1,
            0.2,
            0.3,
        ],
    )

    result = distance.compute_batch(
        model=FakeModel(),
        original="original",
        perturbed_texts=[
            "one",
            "two",
            "three",
        ],
    )

    assert result == [
        ("one", 0.1),
        ("two", 0.2),
        ("three", 0.3),
    ]

    assert distance.compute.call_count == 3


def test_wmd_compute_missing_model() -> None:
    """Test that compute correctly raises a ValueError for an invalid/missing model."""
    wmd = WMDDistance()
    with pytest.raises(ValueError, match=re.escape("requires a gensim KeyedVectors")):
        wmd.compute(source="a", target="b", model=None)


def test_sanitize_distances_mixed() -> None:
    """Test sanitization smoothly handles a mix of finite, infinite, and NaN values."""
    wmd = WMDDistance()
    raw_distances = np.array([1.0, 5.0, np.inf, np.nan])

    result = wmd.sanitize_distances(raw_distances)

    # Max finite is 5.0, meaning replacement fallback should equal 55.0
    np.testing.assert_array_equal(result, np.array([1.0, 5.0, 55.0, 55.0]))


def test_sanitize_distances_all_non_finite() -> None:
    """Test sanitization with all non-finite values."""
    wmd = WMDDistance()
    raw_distances = np.array([np.inf, np.nan, -np.inf])

    result = wmd.sanitize_distances(raw_distances)

    np.testing.assert_array_equal(result, np.array([100.0, 100.0, 100.0]))


def test_sanitize_distances_all_finite() -> None:
    """Test sanitization bypasses arrays already composed entirely of finite values."""
    wmd = WMDDistance()
    raw_distances = np.array([1.0, 2.0, 3.0])

    result = wmd.sanitize_distances(raw_distances)

    np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_compute_batch_with_sanitize() -> None:
    """Test compute_batch actively applies sanitization parameters correctly."""
    distance = WMDDistance()

    distance.compute = MagicMock(  # type: ignore[method-assign]
        side_effect=[
            2.0,
            float("inf"),
        ],
    )

    result = distance.compute_batch(
        model=FakeModel(),
        original="original",
        perturbed_texts=[
            "valid_text",
            "invalid_text",
        ],
        sanitize=True,
    )

    # Given max finite 2.0, sanitization uses max + 50 = 52.0 for replacing inf value
    assert result == [
        ("valid_text", 2.0),
        ("invalid_text", 52.0),
    ]

    assert distance.compute.call_count == 2
