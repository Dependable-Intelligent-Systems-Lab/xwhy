"""Unit tests for the Anthropic provider functionality."""

from unittest.mock import MagicMock

from xwhy.providers.anthropic import AnthropicProvider


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


def test_anthropic_provider_api_error() -> None:
    """Test general exception handling during Anthropic API calls."""
    mock_client = MagicMock()

    # Simulate a network or authentication error
    mock_client.messages.create.side_effect = Exception("Invalid API Key or Rate Limit")

    provider = AnthropicProvider(client=mock_client)
    result = provider.answer(prompt="Will fail")

    assert result == ""
    mock_client.messages.create.assert_called_once_with(
        model="claude-opus-4-8",
        max_tokens=1024,
        temperature=0.0,
        messages=[{"role": "user", "content": "Will fail"}],
    )
