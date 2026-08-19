"""Tests for the OpenAI provider."""

import re
from typing import Any
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from PIL import Image

from xwhy.providers.openai import OpenAIProvider

# -------------------------------------------------------------------------
# Text Generation & Reasoning Tests
# -------------------------------------------------------------------------


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
    result = provider.answer("prompt", model="gpt-5-mini")

    assert result == "reasoning"
    client.responses.create.assert_called_once()
    client.completions.create.assert_not_called()


def test_answer_raises_runtime_error_when_client_fails() -> None:
    """RuntimeError from the client should propagate directly."""
    client = MagicMock()
    client.completions.create.side_effect = RuntimeError("boom")

    provider = OpenAIProvider(client)
    with pytest.raises(RuntimeError, match="boom"):
        provider.answer("prompt")


@patch("time.sleep", return_value=None)
def test_answer_raises_runtime_error_on_generic_exception(
    mock_sleep: MagicMock,
) -> None:
    """Generic exceptions should be caught, logged, and raise a RuntimeError."""
    client = MagicMock()
    client.completions.create.side_effect = ValueError("generic error")

    provider = OpenAIProvider(client)
    with pytest.raises(RuntimeError, match="OpenAI request failed: generic error"):
        provider.answer("prompt", max_retries=3)

    assert client.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


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
def test_is_reasoning_model(model: str, expected: bool) -> None:
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
                max_retries=1,
            )

        assert client.completions.create.call_count == 1
        mock_logger.error.assert_called()


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

    mock_client.responses.create.side_effect = [
        Exception("The temperature parameter is not supported with this model."),
        mock_response,
    ]

    provider = OpenAIProvider(client=mock_client)
    result = provider.answer(prompt="Test", model="o1-preview", temperature=0.0)

    assert result == "Fallback success"
    assert mock_client.responses.create.call_count == 2
    retry_call = mock_client.responses.create.call_args_list[1]
    assert retry_call.kwargs["temperature"] == 1.0


def test_openai_provider_max_tokens_lowercase_regex() -> None:
    """Test that token limitation errors are handled with the lowercase regex."""
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
    """Test that no retry occurs if temperature is already 1.0."""
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception(
        "The temperature parameter is not supported with this model."
    )

    provider = OpenAIProvider(client=mock_client)
    with pytest.raises(RuntimeError, match="temperature parameter is not supported"):
        provider.answer(
            prompt="Test", model="o1-preview", temperature=1.0, max_retries=1
        )

    assert mock_client.responses.create.call_count == 1


def test_openai_provider_max_tokens_no_regex_match() -> None:
    """Test that no retry occurs if token error message format is unexpected."""
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = Exception(
        "max_output_tokens: integer below minimum value. unexpected error format."
    )

    provider = OpenAIProvider(client=mock_client)
    with pytest.raises(RuntimeError, match="unexpected error format"):
        provider.answer(prompt="Test", model="o3-mini", max_tokens=5, max_retries=1)

    assert mock_client.responses.create.call_count == 1


@patch("time.sleep", return_value=None)
def test_openai_empty_text_response_raises_error(mock_sleep: MagicMock) -> None:
    """Test RuntimeError is raised when OpenAI returns empty text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.text = "   \n"
    mock_response.choices = [mock_choice]
    mock_client.completions.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)

    expected_error = "Received an empty response from the OpenAI API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response", model="gpt-3.5-turbo-instruct")

    mock_client.completions.create.assert_called_once()
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
def test_generate_text_exponential_backoff(mock_sleep: MagicMock) -> None:
    """Ensure text generation retries respect exponential backoff caps at 30s."""
    client = MagicMock()
    client.completions.create.side_effect = Exception("Random API failure")

    provider = OpenAIProvider(client)
    with pytest.raises(RuntimeError, match="OpenAI request failed"):
        provider.answer("test", max_retries=6)

    expected_calls = [call(2), call(4), call(8), call(16), call(30)]
    mock_sleep.assert_has_calls(expected_calls)
    assert mock_sleep.call_count == 5


@patch("time.sleep", return_value=None)
def test_generate_text_custom_delay(mock_sleep: MagicMock) -> None:
    """Ensure custom delay overrides default exponential backoff."""
    client = MagicMock()
    client.completions.create.side_effect = [
        Exception("Temporary failure"),
        MagicMock(choices=[MagicMock(text="success")]),
    ]

    provider = OpenAIProvider(client)
    result = provider.answer("test", max_retries=3, delay=1.5)

    assert result == "success"
    mock_sleep.assert_called_once_with(1.5)


def test_generate_zero_retries_raises_fallback() -> None:
    """Hit the end-of-function fallback RuntimeError by supplying max_retries=0."""
    client = MagicMock()
    provider = OpenAIProvider(client)

    with pytest.raises(
        RuntimeError,
        match=re.escape("OpenAI text generation failed after max retries."),
    ):
        provider.answer("prompt", max_retries=0)


# -------------------------------------------------------------------------
# Image Generation & Editing Tests
# -------------------------------------------------------------------------


@patch("xwhy.providers.openai.Image.open")
def test_generate_image_b64_json_success(
    mock_image_open: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test successful image generation using b64_json response format."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="  # Base64 for "test"
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, path = provider.generate_image(
        prompt="test prompt", output_dir=str(tmp_path), model_name="test-model"
    )

    assert success is True
    assert "openai_generated_" in path
    assert path.endswith(".png")
    mock_img_instance.save.assert_called_once()
    client.images.generate.assert_called_once_with(
        model="test-model",
        prompt="test prompt",
        response_format="b64_json",
    )


@patch("xwhy.providers.openai.requests.get")
@patch("xwhy.providers.openai.Image.open")
def test_generate_image_url_success(
    mock_image_open: MagicMock,
    mock_requests_get: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test successful image generation using url response format."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = None
    mock_img_obj.url = "http://example.com/image.png"
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_request_response = MagicMock()
    mock_request_response.content = b"fake_image_bytes"
    mock_requests_get.return_value = mock_request_response

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, path = provider.generate_image(
        prompt="test prompt",
        output_dir=str(tmp_path),
        response_format="url",
    )

    assert success is True
    assert path.startswith(str(tmp_path))
    mock_requests_get.assert_called_once_with(
        "http://example.com/image.png", timeout=30
    )
    mock_request_response.raise_for_status.assert_called_once()


def test_edit_image_raises_file_not_found_error() -> None:
    """Ensure FileNotFoundError is raised if the image path does not exist."""
    client = MagicMock()
    provider = OpenAIProvider(client)

    with (
        patch("os.path.exists", return_value=False),
        pytest.raises(FileNotFoundError, match="Input image not found"),
    ):
        provider.edit_image("prompt", "invalid/path.png", "out_dir")


@patch("builtins.open")
@patch("xwhy.providers.openai.Image.open")
@patch("os.path.exists", return_value=True)
def test_edit_image_standard_api(
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
    mock_open_func: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test image editing using the standard images.edit endpoint with None format."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.edit.return_value = mock_response

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.edit_image(
        prompt="edit prompt",
        image_path="test.png",
        output_dir=str(tmp_path),
        response_format=None,
    )

    assert success is True
    client.images.edit.assert_called_once()
    kwargs = client.images.edit.call_args.kwargs
    assert "response_format" not in kwargs


@patch("xwhy.providers.openai.image_to_base64")
@patch("xwhy.providers.openai.Image.open")
@patch("os.path.exists", return_value=True)
def test_edit_image_use_generate_endpoint_with_url(
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
    mock_image_to_base64: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test editing image by routing through generation API with image_url payload."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_image_to_base64.return_value = "data:image/png;base64,fake"
    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.edit_image(
        prompt="edit prompt",
        image_path="test.png",
        output_dir=str(tmp_path),
        use_generate_for_edit=True,
        use_image_url=True,
    )

    assert success is True
    client.images.generate.assert_called_once()
    kwargs = client.images.generate.call_args.kwargs
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["image_url"] == "data:image/png;base64,fake"


@patch("xwhy.providers.openai.image_to_base64")
@patch("xwhy.providers.openai.Image.open")
@patch("os.path.exists", return_value=True)
def test_edit_image_use_generate_endpoint_with_image_key(
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
    mock_image_to_base64: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test editing image by routing through generation API with default image key."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_image_to_base64.return_value = "data:image/png;base64,fake"
    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.edit_image(
        prompt="edit prompt",
        image_path="test.png",
        output_dir=str(tmp_path),
        use_generate_for_edit=True,
        use_image_url=False,
    )

    assert success is True
    kwargs = client.images.generate.call_args.kwargs
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["image"] == "data:image/png;base64,fake"


@patch("time.sleep", return_value=None)
def test_execute_image_request_empty_data(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test handling of empty image data returned from provider."""
    client = MagicMock()
    mock_response = MagicMock(data=[])
    client.images.generate.return_value = mock_response

    provider = OpenAIProvider(client)
    mock_fallback = MagicMock(spec=Image.Image)
    provider._create_placeholder_image = MagicMock(return_value=mock_fallback)  # type: ignore[method-assign]

    success, path = provider.generate_image("prompt", str(tmp_path), max_retries=2)

    assert success is False
    assert path.startswith(str(tmp_path))
    provider._create_placeholder_image.assert_called_once()
    mock_fallback.save.assert_called_once()
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_execute_image_request_invalid_data(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test handling of image data with no valid format fields."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = None
    mock_img_obj.url = None
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    provider = OpenAIProvider(client)
    mock_fallback = MagicMock(spec=Image.Image)
    provider._create_placeholder_image = MagicMock(return_value=mock_fallback)  # type: ignore[method-assign]

    success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=1)

    assert success is False
    provider._create_placeholder_image.assert_called_once()
    mock_fallback.save.assert_called_once()
    mock_sleep.assert_not_called()


@patch("time.sleep", return_value=None)
def test_execute_image_request_exception_and_fallback(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test that API exceptions are caught and fallback image is generated."""
    client = MagicMock()
    client.images.generate.side_effect = Exception("API failure")

    provider = OpenAIProvider(client)
    mock_fallback = MagicMock(spec=Image.Image)
    provider._create_placeholder_image = MagicMock(return_value=mock_fallback)  # type: ignore[method-assign]

    success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=2)

    assert success is False
    provider._create_placeholder_image.assert_called_once()
    mock_fallback.save.assert_called_once()
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_execute_image_request_exception_no_fallback(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test that API exceptions are caught and gracefully fail if no fallback."""
    client = MagicMock()
    client.images.generate.side_effect = Exception("API failure")

    provider = OpenAIProvider(client)
    provider._create_placeholder_image = MagicMock(return_value=None)  # type: ignore[method-assign]

    success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=1)

    assert success is False
    provider._create_placeholder_image.assert_called_once()


@patch("xwhy.providers.openai.image_to_base64")
@patch("xwhy.providers.openai.Image.open")
@patch("os.path.exists", return_value=True)
def test_edit_image_extra_body_image_url_exists(
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
    mock_image_to_base64: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test editing image when image_url is already provided in extra_body."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_image_to_base64.return_value = "data:image/png;base64,fake"
    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.edit_image(
        prompt="edit prompt",
        image_path="test.png",
        output_dir=str(tmp_path),
        use_generate_for_edit=True,
        use_image_url=True,
        extra_body={"image_url": "pre_existing_url"},
    )

    assert success is True
    kwargs = client.images.generate.call_args.kwargs
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["image_url"] == "pre_existing_url"


@patch("xwhy.providers.openai.image_to_base64")
@patch("xwhy.providers.openai.Image.open")
@patch("os.path.exists", return_value=True)
def test_edit_image_extra_body_image_exists(
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
    mock_image_to_base64: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test editing image when image key is already provided in extra_body."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    mock_image_to_base64.return_value = "data:image/png;base64,fake"
    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.edit_image(
        prompt="edit prompt",
        image_path="test.png",
        output_dir=str(tmp_path),
        use_generate_for_edit=True,
        use_image_url=False,
        extra_body={"image": "pre_existing_image"},
    )

    assert success is True
    kwargs = client.images.generate.call_args.kwargs
    assert "extra_body" in kwargs
    assert kwargs["extra_body"]["image"] == "pre_existing_image"


@patch("time.sleep", return_value=None)
def test_execute_image_request_no_placeholder_attr(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test handling failure when provider lacks placeholder generator attribute."""
    client = MagicMock()
    client.images.generate.side_effect = Exception("API failure")

    provider = OpenAIProvider(client)

    with patch.object(
        OpenAIProvider,
        "_create_placeholder_image",
        new_callable=PropertyMock,
    ) as mock_prop:
        mock_prop.side_effect = AttributeError("Does not exist")
        success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=1)

    assert success is False


@patch("time.sleep", return_value=None)
def test_execute_image_request_placeholder_not_an_image(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test failure fallback when placeholder generator returns non-Image."""
    client = MagicMock()
    client.images.generate.side_effect = Exception("API failure")

    provider = OpenAIProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="not_an_image")  # type: ignore[method-assign]

    success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=1)

    assert success is False
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.openai.Image.open")
def test_execute_image_request_retry_success(
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test image generation successfully recovers after a failure with a delay."""
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])

    client.images.generate.side_effect = [
        Exception("Temporary API glitch"),
        mock_response,
    ]

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = OpenAIProvider(client)
    success, _ = provider.generate_image(
        prompt="retry test",
        output_dir=str(tmp_path),
        max_retries=3,
        delay=2.5,
    )

    assert success is True
    assert client.images.generate.call_count == 2
    mock_sleep.assert_called_once_with(2.5)


@patch("time.sleep", return_value=None)
def test_execute_image_request_exponential_backoff(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Ensure image generation retries correctly scale exponential sleep delays."""
    client = MagicMock()
    client.images.generate.side_effect = Exception("Fatal Error")

    provider = OpenAIProvider(client)
    provider._create_placeholder_image = MagicMock(return_value=None)  # type: ignore[method-assign]

    success, _ = provider.generate_image("prompt", str(tmp_path), max_retries=5)

    assert success is False
    expected_calls = [call(2), call(4), call(8), call(16)]
    mock_sleep.assert_has_calls(expected_calls)
    assert mock_sleep.call_count == 4


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.openai.Image.open")
def test_execute_image_request_image_open_returns_none(
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test retry continuation when Image.open yields None.

    Verifies that if Image.open returns None on valid data, the loop
    continues rather than breaking prematurely.

    Args:
        mock_image_open: Mock for PIL.Image.open.
        mock_sleep: Mock for time.sleep to avoid real delays.
        tmp_path: Pytest temporary directory fixture.

    """
    client = MagicMock()
    mock_img_obj = MagicMock()
    mock_img_obj.b64_json = "dGVzdA=="
    mock_response = MagicMock(data=[mock_img_obj])
    client.images.generate.return_value = mock_response

    # Force generated_img to remain None after a "successful" response.
    mock_image_open.return_value = None

    provider = OpenAIProvider(client)
    mock_fallback = MagicMock(spec=Image.Image)
    provider._create_placeholder_image = MagicMock(  # type: ignore[method-assign]
        return_value=mock_fallback,
    )

    success, path = provider.generate_image(
        prompt="prompt",
        output_dir=str(tmp_path),
        max_retries=2,
    )

    assert success is False
    assert path.startswith(str(tmp_path))
    # Both attempts ran (no break occurred).
    assert client.images.generate.call_count == 2
    mock_sleep.assert_called_once()
    provider._create_placeholder_image.assert_called_once()
    mock_fallback.save.assert_called_once()
