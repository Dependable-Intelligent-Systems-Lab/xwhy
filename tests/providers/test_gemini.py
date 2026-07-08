"""Unit tests for the Gemini provider functionality."""

from unittest.mock import MagicMock, PropertyMock, patch

from xwhy.providers.gemini import GeminiProvider


@patch("xwhy.providers.gemini.types")
def test_gemini_provider_success(mock_types: MagicMock) -> None:
    """Test successful text generation with Gemini using the new SDK."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Create mock objects for the types module to verify correct parameter passing
    mock_part = MagicMock()
    mock_config = MagicMock()
    mock_types.Part.from_text.return_value = mock_part
    mock_types.GenerateContentConfig.return_value = mock_config

    # Explicitly mock the .text property access
    type(mock_response).text = PropertyMock(return_value="Gemini output")
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    result = provider.answer(
        prompt="Test prompt",
        model="gemini-2.5-flash",
        max_tokens=100,
        temperature=0.7,
    )

    assert result == "Gemini output"

    # Verify the exact parameters passed to the SDK types
    mock_types.Part.from_text.assert_called_once_with(text="Test prompt")
    mock_types.GenerateContentConfig.assert_called_once_with(
        max_output_tokens=100,
        temperature=0.7,
    )

    # Verify the main client generation call
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents=mock_part,
        config=mock_config,
    )


@patch("xwhy.providers.gemini.types")
def test_gemini_provider_safety_block_fallback(mock_types: MagicMock) -> None:
    """Test fallback when Gemini response is blocked by safety filters."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Simulate ValueError when attempting to read the response.text property
    type(mock_response).text = PropertyMock(
        side_effect=ValueError("The `response.text` quick accessor only works...")
    )

    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)
    result = provider.answer(prompt="Blocked prompt test")

    assert result == ""
    mock_client.models.generate_content.assert_called_once()
    mock_types.Part.from_text.assert_called_once_with(text="Blocked prompt test")


@patch("xwhy.providers.gemini.types")
def test_gemini_provider_api_error(mock_types: MagicMock) -> None:
    """Test general exception handling during API calls."""
    mock_client = MagicMock()

    # Simulate a network or API validation error
    mock_client.models.generate_content.side_effect = Exception(
        "API rate limit exceeded"
    )

    provider = GeminiProvider(client=mock_client)
    result = provider.answer(prompt="Error prompt test")

    assert result == ""
    mock_client.models.generate_content.assert_called_once()
