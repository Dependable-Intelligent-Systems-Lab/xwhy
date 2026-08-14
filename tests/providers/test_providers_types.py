"""Unit tests for ProviderType properties in types.py."""

import pytest

from xwhy.providers.types import ProviderType


def test_provider_type_is_text_only() -> None:
    """Verify text-only property for designated providers."""
    assert ProviderType.ANTHROPIC.is_text_only is True
    assert ProviderType.OPENAI.is_text_only is False


def test_provider_type_is_image_only() -> None:
    """Verify image-only property returns false for all providers."""
    for provider in ProviderType:
        assert provider.is_image_only is False


def test_provider_type_supports_both() -> None:
    """Verify supports-both property for multimodal providers."""
    assert ProviderType.OPENAI.supports_both is True
    assert ProviderType.ANTHROPIC.supports_both is False


def test_provider_type_from_str() -> None:
    """Ensure correct parsing from string or enum values."""
    assert ProviderType.from_str("openai") == ProviderType.OPENAI
    assert ProviderType.from_str(ProviderType.GEMINI) == ProviderType.GEMINI

    with pytest.raises(ValueError, match="is not a valid ProviderType"):
        ProviderType.from_str("invalid_provider")
