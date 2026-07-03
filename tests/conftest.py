"""Pytest configuration and shared fixtures."""

import pytest

from xwhy.embeddings.factory import EmbeddingFactory
from xwhy.providers.factory import ProviderFactory


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Reset all global registries before each test."""
    ProviderFactory.clear()
    EmbeddingFactory.clear()
