"""Tests for provider factory."""

from __future__ import annotations

import pytest

from xwhy.providers.factory import ProviderFactory
from xwhy.providers.openai import OpenAIProvider
from xwhy.providers.types import ProviderType


def test_register_and_create_provider() -> None:
    """Register and create a provider successfully."""
    ProviderFactory.register(
        provider=ProviderType.OPENAI,
        provider_cls=OpenAIProvider,
    )

    client = object()

    provider = ProviderFactory.create(
        provider=ProviderType.OPENAI,
        client=client,
    )

    assert isinstance(provider, OpenAIProvider)


def test_register_duplicate_provider() -> None:
    """Raise an error for duplicate provider registration."""
    ProviderFactory.register(
        provider=ProviderType.OPENAI,
        provider_cls=OpenAIProvider,
    )

    with pytest.raises(
        ValueError,
        match="Provider already registered",
    ):
        ProviderFactory.register(
            provider=ProviderType.OPENAI,
            provider_cls=OpenAIProvider,
        )


def test_create_unknown_provider() -> None:
    """Raise an error for an unknown provider."""
    with pytest.raises(
        ValueError,
        match="Unsupported provider",
    ):
        ProviderFactory.create(
            provider=ProviderType.OPENAI,
            client=object(),
        )
