"""Tests for text perturbation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import NDArray

from xwhy.perturbation.text import TextPerturbation


def fake_choice(
    items: Sequence[str] | Sequence[tuple[int, ...]],
    size: int | None = None,
    replace: bool = False,
) -> NDArray[Any]:
    """Mock implementation of ``numpy.random.Generator.choice``.

    Returns a string array when selecting words for ``apply_mask()``
    and an integer array when selecting perturbation masks for
    ``generate()``.

    Args:
        items:
            Sequence passed to ``Generator.choice``.
        size:
            Requested sample size. Included for API compatibility.
        replace:
            Whether sampling is performed with replacement.
            Included for API compatibility.

    Returns:
        NumPy array containing the mocked selection.

    """
    del size
    del replace

    if items and isinstance(items[0], str):
        return np.array(["two"])

    return np.array([1, 0, 0])


# ---------------------------------------------------------------------
# apply_mask
# ---------------------------------------------------------------------
def test_apply_mask_returns_selected_words() -> None:
    """Return selected words when mask keeps enough tokens."""
    perturbation = TextPerturbation(seed=42)

    result = perturbation.apply_mask(
        words=["hello", "beautiful", "world"],
        mask=(1, 0, 1),
    )

    assert result == ["hello", "world"]


def test_apply_mask_adds_random_words_if_needed() -> None:
    """Ensure at least two words remain after masking."""
    perturbation = TextPerturbation(seed=42)

    result = perturbation.apply_mask(
        words=["hello", "beautiful", "world"],
        mask=(1, 0, 0),
    )

    assert len(result) >= 2
    assert "hello" in result


# ---------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------
def test_generate_returns_requested_number() -> None:
    """Generate requested number of perturbations."""
    perturbation = TextPerturbation(seed=42)

    texts, masks = perturbation.generate(
        text="hello beautiful world",
        num_perturbations=10,
    )

    assert len(texts) == 10
    assert len(masks) == 10

    assert all(isinstance(text, str) for text in texts)
    assert all(isinstance(mask, tuple) for mask in masks)


def test_generate_masks_match_text_length() -> None:
    """Generated masks should match input length."""
    perturbation = TextPerturbation(seed=42)

    _, masks = perturbation.generate(
        text="one two three four",
        num_perturbations=8,
    )

    assert all(len(mask) == 4 for mask in masks)


def test_generate_never_returns_empty_text() -> None:
    """Generated perturbations should never be empty."""
    perturbation = TextPerturbation(seed=42)

    texts, _ = perturbation.generate(
        text="one two three",
        num_perturbations=12,
    )

    assert all(text.strip() for text in texts)


# ---------------------------------------------------------------------
# reuse branch
# ---------------------------------------------------------------------
def test_generate_reuses_masks_when_unique_exhausted() -> None:
    """Reuse existing masks when unique generation is exhausted."""
    perturbation = TextPerturbation(seed=42)

    fake_rng = MagicMock()

    fake_rng.choice.side_effect = fake_choice

    fake_rng.choice.return_value = np.array([1, 0, 0])

    with patch.object(perturbation, "_rng", fake_rng):
        texts, masks = perturbation.generate(
            text="one two three",
            num_perturbations=5,
        )

    assert len(texts) == 5
    assert len(masks) == 5

    assert all(mask == (1, 0, 0) for mask in masks)


def test_apply_mask_removes_duplicates() -> None:
    """Duplicate selected words should be removed."""
    perturbation = TextPerturbation(seed=42)

    result = perturbation.apply_mask(
        words=["hello", "hello", "world"],
        mask=(1, 1, 1),
    )

    assert result == ["hello", "world"]


def test_apply_mask_preserves_word_order() -> None:
    """Selected words should preserve original order."""
    perturbation = TextPerturbation(seed=42)

    result = perturbation.apply_mask(
        words=["a", "b", "c", "d"],
        mask=(1, 0, 1, 1),
    )

    assert result == ["a", "c", "d"]


def test_apply_mask_adds_missing_words() -> None:
    """Add random words when fewer than minimum words are selected."""
    perturbation = TextPerturbation(seed=42)

    fake_rng = MagicMock()
    fake_rng.choice.return_value = np.array(["world"])

    with patch.object(perturbation, "_rng", fake_rng):
        result = perturbation.apply_mask(
            words=["hello", "world", "python"],
            mask=(1, 0, 0),
        )

    # One word selected + one random word added
    assert result == ["hello", "world"]

    # Ensure random selection happened
    fake_rng.choice.assert_called_once()

    _, kwargs = fake_rng.choice.call_args

    assert kwargs["size"] == 1
    assert kwargs["replace"] is False


def test_apply_mask_returns_empty_when_no_candidates() -> None:
    """Return empty list when no selectable words exist."""
    perturbation = TextPerturbation(seed=42)

    result = perturbation.apply_mask(
        words=["", "   "],
        mask=(0, 0),
    )

    assert result == []
