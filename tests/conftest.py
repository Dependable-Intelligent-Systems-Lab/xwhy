"""Pytest configuration and shared fixtures."""

import pytest

from xwhy.bootstrap import register_all
from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.embeddings.factory import EmbeddingFactory
from xwhy.models.segmentation.factory import SegmentationFactory
from xwhy.providers.factory import ProviderFactory
from xwhy.providers.resolver import ProviderResolver
from xwhy.surrogate.factory import SurrogateFactory


@pytest.fixture(autouse=True)
def reset_registry() -> None:
    """Reset all global registries before each test and reload defaults."""
    ProviderFactory.clear()
    EmbeddingFactory.clear()
    ClassificationFactory.clear()
    SegmentationFactory.clear()
    ProviderResolver.clear()
    SurrogateFactory.clear()

    register_all()
