"""Tests for perturbation types."""

import re

import pytest

from xwhy.perturbation.types import SuperpixelType


def test_superpixel_type_from_str() -> None:
    """Test SuperpixelType.from_str method."""
    # Valid enum instance
    assert (
        SuperpixelType.from_str(SuperpixelType.QUICKSHIFT) == SuperpixelType.QUICKSHIFT
    )
    assert SuperpixelType.from_str(SuperpixelType.SLIC) == SuperpixelType.SLIC
    assert (
        SuperpixelType.from_str(SuperpixelType.FELZENSZWALB)
        == SuperpixelType.FELZENSZWALB
    )

    # Valid strings
    assert SuperpixelType.from_str("quickshift") == SuperpixelType.QUICKSHIFT
    assert SuperpixelType.from_str(" SLIC ") == SuperpixelType.SLIC
    assert SuperpixelType.from_str("felzenszwalb") == SuperpixelType.FELZENSZWALB

    # Invalid string
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid superpixel type 'invalid'. Must be one of:"),
    ):
        SuperpixelType.from_str("invalid")

    # Invalid type
    with pytest.raises(
        ValueError, match=re.escape("Invalid superpixel type '123'. Must be one of:")
    ):
        SuperpixelType.from_str(123)  # type: ignore
