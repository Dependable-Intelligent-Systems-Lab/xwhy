"""Tests for surrogate factory."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from xwhy.surrogate.base import BaseSurrogate
from xwhy.surrogate.factory import SurrogateFactory
from xwhy.surrogate.types import SurrogateType


def test_register_and_create_surrogate() -> None:
    """Register and create a surrogate successfully."""
    SurrogateFactory.clear()

    def mock_builder(**kwargs: object) -> BaseSurrogate:
        return MagicMock(spec=BaseSurrogate)

    SurrogateFactory.register(
        method=SurrogateType.LIME,
        builder=mock_builder,
    )

    surrogate = SurrogateFactory.create(
        method=SurrogateType.LIME,
        some_param="value",
    )

    assert isinstance(surrogate, MagicMock)


def test_register_duplicate_surrogate() -> None:
    """Raise an error for duplicate surrogate registration."""
    SurrogateFactory.clear()

    def builder_func(**kwargs: object) -> BaseSurrogate:
        return MagicMock(spec=BaseSurrogate)

    SurrogateFactory.register(
        method=SurrogateType.LIME,
        builder=builder_func,
    )

    SurrogateFactory.register(method=SurrogateType.LIME, builder=builder_func)

    with pytest.raises(
        ValueError,
        match="Surrogate method already registered",
    ):
        SurrogateFactory.register(
            method=SurrogateType.LIME,
            builder=lambda: MagicMock(spec=BaseSurrogate),
        )


def test_create_unknown_surrogate() -> None:
    """Raise an error for an unknown surrogate method."""
    SurrogateFactory.clear()

    with pytest.raises(
        ValueError,
        match="Unsupported surrogate method",
    ):
        SurrogateFactory.create(
            method="unknown_method",  # type: ignore[arg-type]
        )
