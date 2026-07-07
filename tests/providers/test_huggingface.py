"""Unit tests for the HuggingFace provider functionality."""

from unittest.mock import MagicMock

from xwhy.providers.huggingface import HuggingFaceProvider


def test_huggingface_provider_success() -> None:
    """Test successful text generation with HuggingFace."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Mock the choices[0].message.content structure
    mock_message = MagicMock()
    mock_message.content = "HuggingFace generated output"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_response

    provider = HuggingFaceProvider(client=mock_client)
    result = provider.answer(
        prompt="Test prompt",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        max_tokens=256,
        temperature=0.7,
    )

    assert result == "HuggingFace generated output"
    mock_client.chat.completions.create.assert_called_once_with(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        messages=[{"role": "user", "content": "Test prompt"}],
        max_tokens=256,
        temperature=0.7,
    )


def test_huggingface_provider_api_error() -> None:
    """Test general exception handling during HuggingFace API calls."""
    mock_client = MagicMock()

    # Simulate a network or validation error
    mock_client.chat.completions.create.side_effect = Exception(
        "Model loading or API error"
    )

    provider = HuggingFaceProvider(client=mock_client)
    result = provider.answer(prompt="Error prompt test")

    assert result == ""
    mock_client.chat.completions.create.assert_called_once_with(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Error prompt test"}],
        max_tokens=512,
        temperature=0.1,
    )
