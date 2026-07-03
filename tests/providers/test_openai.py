"""Tests for the OpenAI provider."""

from unittest.mock import MagicMock

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


def test_answer_returns_empty_string_when_exception_occurs() -> None:
    """Errors should return an empty string."""
    client = MagicMock()

    client.completions.create.side_effect = RuntimeError("boom")

    provider = OpenAIProvider(client)

    assert provider.answer("prompt") == ""


def test_score_uses_completion_api() -> None:
    """Score generation should use the Completions API."""
    client = MagicMock()

    response = MagicMock()
    response.choices = [MagicMock(text="0.9")]

    client.completions.create.return_value = response

    provider = OpenAIProvider(client)

    result = provider.score("prompt")

    assert result == "0.9"


def test_score_uses_responses_api() -> None:
    """Reasoning models should use the Responses API."""
    client = MagicMock()

    response = MagicMock()
    response.output_text = "0.95"

    client.responses.create.return_value = response

    provider = OpenAIProvider(client)

    result = provider.score(
        "prompt",
        model="gpt-5-mini",
    )

    assert result == "0.95"


def test_score_returns_zero_when_exception_occurs() -> None:
    """Errors should return zero."""
    client = MagicMock()

    client.completions.create.side_effect = RuntimeError("boom")

    provider = OpenAIProvider(client)

    assert provider.score("prompt") == "0"


def test_score_appends_numeric_instruction() -> None:
    """The score prompt should contain the numeric instruction."""
    client = MagicMock()

    response = MagicMock()
    response.choices = [MagicMock(text="1")]

    client.completions.create.return_value = response

    provider = OpenAIProvider(client)

    provider.score("my prompt")

    prompt = client.completions.create.call_args.kwargs["prompt"]

    assert prompt.startswith("my prompt")
    assert "Respond with a single number only." in prompt


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
