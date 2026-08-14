"""Tests for text utility functions."""

import re

import pytest

from xwhy.utils.text import inject_text_at_position


def test_inject_text_with_custom_inject_text() -> None:
    """Test injecting a custom string into text."""
    result = inject_text_at_position(
        original_text="hello world",
        position="start",
        inject_text="custom",
    )
    assert result == "custom hello world"


def test_inject_text_position_start() -> None:
    """Test injecting default word at the start."""
    result = inject_text_at_position(
        original_text="hello world",
        position="start",
        default_index=0,  # "could"
    )
    assert result == "could hello world"


def test_inject_text_position_middle() -> None:
    """Test injecting default word in the middle."""
    result = inject_text_at_position(
        original_text="hello brave new world",
        position="middle",
        default_index=4,  # "please"
    )
    assert result == "hello brave please new world"


def test_inject_text_position_end() -> None:
    """Test injecting default word at the end."""
    result = inject_text_at_position(
        original_text="hello world",
        position="end",
        default_index=3,  # "###"
    )
    assert result == "hello world ###"


def test_inject_text_invalid_default_index_low() -> None:
    """Test ValueError is raised when default_index is below 0."""
    with pytest.raises(ValueError, match="default_index must be between 0 and"):
        inject_text_at_position(
            original_text="hello world",
            position="start",
            default_index=-1,
        )


def test_inject_text_invalid_default_index_high() -> None:
    """Test ValueError is raised when default_index exceeds available options."""
    with pytest.raises(ValueError, match="default_index must be between 0 and"):
        inject_text_at_position(
            original_text="hello world",
            position="start",
            default_index=10,
        )


def test_inject_text_invalid_position() -> None:
    """Test ValueError is raised for unsupported position arguments."""
    with pytest.raises(
        ValueError, match=re.escape("Position must be 'start', 'middle', or 'end'.")
    ):
        inject_text_at_position(
            original_text="hello world",
            position="invalid",  # type: ignore[arg-type]
            default_index=0,
        )
