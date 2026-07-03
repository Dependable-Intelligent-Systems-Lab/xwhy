"""Tests for Word Mover's Distance."""

from __future__ import annotations

from unittest.mock import MagicMock

from xwhy.distance.wmd import WMDDistance


class FakeModel:
    """Fake Word2Vec model."""

    def __contains__(self, key: str) -> bool:
        """Return whether a word exists."""
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
        """Return fake distance."""
        assert words1
        assert words2

        return 0.42


def test_clean_text() -> None:
    """Clean punctuation and lowercase."""
    distance = WMDDistance()

    result = distance.clean_text(
        text=" Hello, WORLD!! ",
    )

    assert result == "hello world"


def test_compute_distance() -> None:
    """Compute WMD distance."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="Hello world",
        target="Python",
    )

    assert result == 0.42


def test_compute_returns_default_for_unknown_words() -> None:
    """Return default distance for unknown vocabulary."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="xxxx",
        target="yyyy",
    )

    assert result == 1.0


def test_compute_returns_default_when_first_is_empty() -> None:
    """Return default distance when source has no valid tokens."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="!!!",
        target="hello",
    )

    assert result == 1.0


def test_compute_returns_default_when_second_is_empty() -> None:
    """Return default distance when target has no valid tokens."""
    distance = WMDDistance()

    result = distance.compute(
        model=FakeModel(),
        source="hello",
        target="!!!",
    )

    assert result == 1.0


def test_compute_batch() -> None:
    """Compute distances for multiple texts."""
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
