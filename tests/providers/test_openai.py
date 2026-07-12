"""Tests for the OpenAI provider."""

from unittest.mock import MagicMock, patch

import pytest

from xwhy.providers.openai import OpenAIProvider


def test_answer_uses_completion_api() -> None:
    """Completion models should use the Completions API."""
    client = MagicMock()

    response = MagicMock()
    response.choices = [MagicMock(text="hello")]

    client.completions.create.return_value = response

    provider = OpenAIProvider(client)

    result = provider.answer("prompt")

    assert result == "hello"

    client.completions.create.assert_called_once()
    client.responses.create.assert_not_called()


def test_answer_uses_responses_api() -> None:
    """Reasoning models should use the Responses API."""
    client = MagicMock()

    response = MagicMock()
    response.output_text = "reasoning"

    client.responses.create.return_value = response

    provider = OpenAIProvider(client)

    result = provider.answer(
        "prompt",
        model="gpt-5-mini",
    )

    assert result == "reasoning"

    client.responses.create.assert_called_once()
    client.completions.create.assert_not_called()


def test_answer_raises_runtime_error_when_client_fails() -> None:
    """RuntimeError from the client should propagate and raise RuntimeError."""
    client = MagicMock()

    client.completions.create.side_effect = RuntimeError("boom")

    provider = OpenAIProvider(client)

    with pytest.raises(RuntimeError, match="boom"):
        provider.answer("prompt")


def test_answer_raises_runtime_error_on_generic_exception() -> None:
    """Generic exceptions should be caught, logged, and raise a RuntimeError."""
    client = MagicMock()
    client.completions.create.side_effect = ValueError("generic error")

    provider = OpenAIProvider(client)

    with pytest.raises(RuntimeError, match="OpenAI request failed: generic error"):
        provider.answer("prompt")


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-3.5-turbo-instruct", False),
        ("gpt-4", False),
        ("gpt-5", True),
        ("gpt-5-mini", True),
        ("o1-mini", True),
        ("o3-mini", True),
        ("o4-mini", True),
    ],
)
def test_is_reasoning_model(
    model: str,
    expected: bool,
) -> None:
    """Reasoning models should be correctly detected."""
    assert OpenAIProvider._is_reasoning_model(model) is expected


def test_generate_regex_dynamic_fix() -> None:
    """Test that regex correctly extracts min token requirement and retries."""
    client = MagicMock()
    provider = OpenAIProvider(client)

    error_message = (
        "Error: max_output_tokens is an integer below minimum value. "
        "Expected a value >= 50"
    )
    client.completions.create.side_effect = [
        Exception(error_message),
        MagicMock(choices=[MagicMock(text="fixed_response")]),
    ]

    result = provider._generate(
        prompt="test", model="gpt-3.5-turbo-instruct", max_tokens=10, temperature=0.0
    )

    assert result == "fixed_response"

    assert client.completions.create.call_count == 2

    retry_call = client.completions.create.call_args_list[1]
    assert retry_call.kwargs["max_tokens"] == 50


def test_generate_regex_no_match_fallback() -> None:
    """Raise RuntimeError when regex matching fails on tokens error."""
    client = MagicMock()

    error_message = (
        "Error: max_output_tokens is an integer below minimum value. Expected a value."
    )
    client.completions.create.side_effect = Exception(error_message)

    provider = OpenAIProvider(client)

    with patch("xwhy.providers.openai.logger") as mock_logger:
        with pytest.raises(RuntimeError, match="Expected a value"):
            provider._generate(
                prompt="test",
                model="gpt-3.5-turbo-instruct",
                max_tokens=10,
                temperature=0.0,
            )

        assert client.completions.create.call_count == 1
        mock_logger.error.assert_called()
        mock_logger.warning.assert_not_called()


def test_openai_provider_reasoning_model_with_temperature() -> None:
    """Test that reasoning models receive the temperature parameter successfully."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Reasoning output"
    mock_client.responses.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)
    result = provider.answer(prompt="Test", model="o1-mini", temperature=0.7)

    assert result == "Reasoning output"
    mock_client.responses.create.assert_called_once_with(
        model="o1-mini",
        input="Test",
        max_output_tokens=200,
        reasoning={"effort": "low"},
        temperature=0.7,
    )


def test_openai_provider_reasoning_model_temperature_fallback() -> None:
    """Test the dynamic fallback when a reasoning model rejects custom temperature."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Fallback success"

    # First call raises temperature error, second call succeeds with temperature=1.0
    mock_client.responses.create.side_effect = [
        Exception("The temperature parameter is not supported with this model."),
        mock_response,
    ]

    provider = OpenAIProvider(client=mock_client)
    result = provider.answer(prompt="Test", model="o1-preview", temperature=0.0)

    assert result == "Fallback success"
    assert mock_client.responses.create.call_count == 2


def test_openai_provider_max_tokens_lowercase_regex() -> None:
    """Test that token limitation errors are handled with the new lowercase regex."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Token fix success"

    mock_client.responses.create.side_effect = [
        Exception(
            "max_output_tokens: integer below minimum value. expected a value >= 150"
        ),
        mock_response,
    ]

    provider = OpenAIProvider(client=mock_client)
    result = provider.answer(prompt="Test", model="o3-mini", max_tokens=10)

    assert result == "Token fix success"
    assert mock_client.responses.create.call_count == 2


def test_openai_provider_temperature_already_one_no_retry() -> None:
    """Test that no retry occurs if temperature is already 1.0.

    This covers the False branch of 'if temperature != 1.0:' and verifies
    that a RuntimeError is raised instead of an empty string.
    """
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception(
        "The temperature parameter is not supported with this model."
    )

    provider = OpenAIProvider(client=mock_client)

    # Sending temperature=1.0 initially triggers the False branch of the inner IF
    with pytest.raises(RuntimeError, match="temperature parameter is not supported"):
        provider.answer(prompt="Test", model="o1-preview", temperature=1.0)

    # Should only call once and fail gracefully without infinite recursion
    assert mock_client.responses.create.call_count == 1


def test_openai_provider_max_tokens_no_regex_match() -> None:
    """Test that no retry occurs if token error message format is unexpected.

    This covers the False branch of 'if match:' and ensures a RuntimeError is
    properly raised rather than returning an empty string.
    """
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception(
        "max_output_tokens: integer below minimum value. unexpected error format."
    )

    provider = OpenAIProvider(client=mock_client)

    # The outer IF matches, but regex search fails (returns None), raising error
    with pytest.raises(RuntimeError, match="unexpected error format"):
        provider.answer(prompt="Test", model="o3-mini", max_tokens=5)

    # Should skip retry logic and raise the error immediately
    assert mock_client.responses.create.call_count == 1


def test_openai_empty_text_response_raises_error() -> None:
    """Test RuntimeError is raised when OpenAI returns empty text.

    This covers the 'if not result_text:' block, ensuring that safety
    or filter-related empty string results are explicitly raised.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()

    mock_choice.text = ""
    mock_response.choices = [mock_choice]
    mock_client.completions.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)

    expected_error = "empty response from the OpenAI API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response", model="gpt-3.5-turbo-instruct")

    mock_client.completions.create.assert_called_once()
