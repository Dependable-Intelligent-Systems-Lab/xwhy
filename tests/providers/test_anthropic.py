"""Tests for the Anthropic provider."""

import re
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from xwhy.providers.anthropic import AnthropicProvider

# -------------------------------------------------------------------------
# Text Generation Tests
# -------------------------------------------------------------------------


def test_anthropic_provider_success() -> None:
    """Test successful text generation with Anthropic."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Mocking the specific content block structure returned by Anthropic
    mock_content_block = MagicMock()
    mock_content_block.text = "Claude's generated response"
    mock_response.content = [mock_content_block]

    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(client=mock_client)
    result = provider.answer(
        prompt="Hello Claude",
        model="claude-opus-4-8",
        max_tokens=500,
        temperature=0.8,
    )

    assert result == "Claude's generated response"
    mock_client.messages.create.assert_called_once_with(
        model="claude-opus-4-8",
        max_tokens=500,
        temperature=0.8,
        messages=[{"role": "user", "content": "Hello Claude"}],
    )


@patch("time.sleep", return_value=None)
def test_anthropic_provider_api_error_max_retries(
    mock_sleep: MagicMock,
) -> None:
    """Test generic exception handling during Anthropic API calls with retries."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("Invalid API Key or Limit")

    provider = AnthropicProvider(client=mock_client)

    with pytest.raises(
        RuntimeError, match="Anthropic request failed: Invalid API Key or Limit"
    ):
        provider.answer(prompt="Will fail", max_retries=3)

    assert mock_client.messages.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
def test_anthropic_retry_then_success(mock_sleep: MagicMock) -> None:
    """Test retry logic when API fails transiently before succeeding."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_content_block = MagicMock()
    mock_content_block.text = "Success after retry"
    mock_response.content = [mock_content_block]

    mock_client.messages.create.side_effect = [
        Exception("Temporary network glitch"),
        mock_response,
    ]

    provider = AnthropicProvider(client=mock_client)
    result = provider.answer(prompt="Test retry", max_retries=3, delay=1.0)

    assert result == "Success after retry"
    assert mock_client.messages.create.call_count == 2
    mock_sleep.assert_called_once_with(1.0)


@patch("time.sleep", return_value=None)
def test_anthropic_direct_runtime_error_raises_immediately(
    mock_sleep: MagicMock,
) -> None:
    """RuntimeError raised during API execution should re-raise without retrying."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = RuntimeError("Direct RuntimeError")

    provider = AnthropicProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="Direct RuntimeError"):
        provider.answer(prompt="Will fail immediately", max_retries=3)

    assert mock_client.messages.create.call_count == 1
    mock_sleep.assert_not_called()


@pytest.mark.parametrize(
    "empty_content",
    [
        [],
        [MagicMock(text="   \n ")],
    ],
)
@patch("time.sleep", return_value=None)
def test_anthropic_empty_response_content_raises_error(
    mock_sleep: MagicMock,
    empty_content: Any,  # noqa: ANN401
) -> None:
    """Test RuntimeError is raised immediately when Anthropic returns empty content."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = empty_content
    mock_client.messages.create.return_value = mock_response

    provider = AnthropicProvider(client=mock_client)

    expected_error = "empty response from the Anthropic API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response")

    mock_client.messages.create.assert_called_once()
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
def test_anthropic_exponential_backoff(mock_sleep: MagicMock) -> None:
    """Ensure retries respect exponential backoff capped at 30s."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API Error")

    provider = AnthropicProvider(client=mock_client)
    with pytest.raises(RuntimeError, match="Anthropic request failed"):
        provider.answer(prompt="Backoff test", max_retries=6)

    expected_calls = [call(2), call(4), call(8), call(16), call(30)]
    mock_sleep.assert_has_calls(expected_calls)
    assert mock_sleep.call_count == 5


@patch("time.sleep", return_value=None)
def test_anthropic_custom_delay(mock_sleep: MagicMock) -> None:
    """Ensure custom delay overrides exponential backoff."""
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API Error")

    provider = AnthropicProvider(client=mock_client)
    with pytest.raises(RuntimeError, match="Anthropic request failed"):
        provider.answer(prompt="Custom delay test", max_retries=3, delay=2.5)

    expected_calls = [call(2.5), call(2.5)]
    mock_sleep.assert_has_calls(expected_calls)
    assert mock_sleep.call_count == 2


def test_anthropic_zero_retries_raises_fallback() -> None:
    """Hit the end-of-function fallback RuntimeError by supplying max_retries=0."""
    mock_client = MagicMock()
    provider = AnthropicProvider(mock_client)

    with pytest.raises(
        RuntimeError,
        match=re.escape("Anthropic text generation failed after max retries."),
    ):
        provider.answer("prompt", max_retries=0)
