"""Unit tests for provider resolver."""

from collections.abc import Iterator

import pytest

from xwhy.providers.base import BaseProvider
from xwhy.providers.resolver import ProviderResolver
from xwhy.providers.types import ProviderType


class MockProvider(BaseProvider):
    """Implement a concrete BaseProvider for testing."""

    def answer(self, prompt: str) -> str:
        """Return a dummy response for testing purposes."""
        return "mock_response"

    def score(self, text: str) -> float:
        """Return a dummy score for testing purposes."""
        return 1.0


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Ensure a clean registry before and after every test."""
    ProviderResolver.clear()
    yield
    ProviderResolver.clear()


def test_resolve_with_instance() -> None:
    """Verify that an instance is returned directly."""
    instance = MockProvider()
    result = ProviderResolver.resolve(instance)
    assert result is instance


def test_resolve_success_with_enum() -> None:
    """Verify successful resolution using a registered ProviderType."""
    ProviderResolver.register(ProviderType.OPENAI, MockProvider)
    result = ProviderResolver.resolve(ProviderType.OPENAI)
    assert isinstance(result, MockProvider)


def test_resolve_success_with_string() -> None:
    """Verify successful resolution using a string identifier."""
    ProviderResolver.register(ProviderType.OPENAI, MockProvider)
    result = ProviderResolver.resolve("openai")
    assert isinstance(result, MockProvider)


def test_resolve_fails_when_no_builder_registered() -> None:
    """Verify error when type is valid but no builder is registered."""
    with pytest.raises(ValueError, match="No default builder registered"):
        ProviderResolver.resolve(ProviderType.OPENAI)


def test_resolve_fails_with_invalid_type() -> None:
    """Verify error when the provided string is not a valid ProviderType."""
    # Use regex match to satisfy PT011 (too broad exception)
    with pytest.raises(ValueError, match=r".*"):
        ProviderResolver.resolve("non_existent_provider_type")


def test_register_and_clear() -> None:
    """Verify registration and clearing works correctly."""
    ProviderResolver.register(ProviderType.OPENAI, MockProvider)

    # Check registration worked
    assert ProviderResolver.resolve(ProviderType.OPENAI) is not None

    # Clear and check
    ProviderResolver.clear()
    with pytest.raises(ValueError, match="No default builder registered"):
        ProviderResolver.resolve(ProviderType.OPENAI)
