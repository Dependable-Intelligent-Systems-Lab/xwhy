"""Unit tests for the Gemini provider functionality."""

import json
from unittest.mock import MagicMock, PropertyMock, call, mock_open, patch

import pytest
from PIL import Image

from xwhy.providers.gemini import GeminiProvider

# -------------------------------------------------------------------------
# Text Generation Tests
# -------------------------------------------------------------------------


@patch("xwhy.providers.gemini.types")
def test_gemini_provider_success(mock_types: MagicMock) -> None:
    """Test successful text generation with Gemini using the new SDK."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    mock_part = MagicMock()
    mock_config = MagicMock()
    mock_types.Part.from_text.return_value = mock_part
    mock_types.GenerateContentConfig.return_value = mock_config

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
    mock_types.Part.from_text.assert_called_once_with(text="Test prompt")
    mock_types.GenerateContentConfig.assert_called_once_with(
        max_output_tokens=100,
        temperature=0.7,
    )
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-2.5-flash",
        contents=mock_part,
        config=mock_config,
    )


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.types")
def test_gemini_provider_safety_block_fallback(
    mock_types: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test that Gemini provider raises RuntimeError when blocked by safety filters."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    type(mock_response).text = PropertyMock(
        side_effect=ValueError("The `response.text` quick accessor only works...")
    )

    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)

    with pytest.raises(
        RuntimeError, match="blocked \\(likely due to safety filters\\)"
    ):
        provider.answer(prompt="Blocked prompt test", max_retries=2)

    # Asserts retries happened and sleep was called once before failing on 2nd try
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()
    mock_types.Part.from_text.assert_called_with(text="Blocked prompt test")


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.types")
def test_gemini_provider_api_error(
    mock_types: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test general exception handling and retries during API calls."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")

    provider = GeminiProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="API error"):
        provider.answer(prompt="Error prompt test", max_retries=3)

    assert mock_client.models.generate_content.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.types")
def test_gemini_empty_text_response_raises_error(
    mock_types: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test RuntimeError is raised when Gemini returns empty text directly."""
    mock_client = MagicMock()
    mock_response = MagicMock()

    type(mock_response).text = PropertyMock(return_value="   ")
    mock_client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client=mock_client)

    expected_error = "empty response from the Gemini API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response", max_retries=2)

    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once()


def test_gemini_provider_zero_retries() -> None:
    """Test text generation fails immediately if max_retries is less than 1."""
    provider = GeminiProvider(client=MagicMock())

    with pytest.raises(RuntimeError, match="max_retries must be at least 1"):
        provider.answer(prompt="Test", max_retries=0)


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.types")
def test_gemini_generate_success_after_retries(
    mock_types: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test generation succeeds on a subsequent retry with an explicit delay."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    type(mock_response).text = PropertyMock(return_value="Delayed success")

    mock_client.models.generate_content.side_effect = [
        Exception("Temporary failure"),
        mock_response,
    ]

    provider = GeminiProvider(client=mock_client)
    result = provider.answer(prompt="Test", max_retries=3, delay=5.5)

    assert result == "Delayed success"
    assert mock_client.models.generate_content.call_count == 2
    mock_sleep.assert_called_once_with(5.5)


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.types")
def test_gemini_generate_exponential_backoff_max(
    mock_types: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test exponential backoff correctly caps at 30 seconds across retries."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("Fail")

    provider = GeminiProvider(client=mock_client)

    with pytest.raises(RuntimeError):
        # 6 retries mean 5 sleeps: 2, 4, 8, 16, 30 (cap)
        provider.answer(prompt="Test", max_retries=6)

    expected_sleep_calls = [call(2), call(4), call(8), call(16), call(30)]
    mock_sleep.assert_has_calls(expected_sleep_calls)
    assert mock_sleep.call_count == 5


# -------------------------------------------------------------------------
# Image Generation & Execution Tests
# -------------------------------------------------------------------------


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
def test_generate_image_stream_no_inline_data(
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test stream generation where chunks/parts lack inline_data."""
    client = MagicMock()
    mock_chunk1 = MagicMock(parts=[])
    mock_part2 = MagicMock(inline_data=None)
    mock_chunk2 = MagicMock(parts=[mock_part2])

    client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value=None)  # type: ignore[method-assign]

    success, _ = provider.generate_image(
        prompt="Test", output_dir="fake_dir", stream=True, max_retries=2
    )

    assert success is False
    assert mock_sleep.call_count == 1
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
def test_generate_image_no_stream_no_inline_data(
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test non-stream generation where response parts lack inline_data."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_part = MagicMock(inline_data=None)
    mock_response.parts = [mock_part]

    client.models.generate_content.return_value = mock_response

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value=None)  # type: ignore[method-assign]

    success, _ = provider.generate_image(
        prompt="Test", output_dir="fake_dir", stream=False, max_retries=2
    )

    assert success is False
    assert mock_sleep.call_count == 1
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("os.makedirs")
def test_execute_image_request_fallback_not_pil_image(
    mock_makedirs: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test when fallback image is not a PIL Image instance."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("API error")

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value=12345)  # type: ignore[method-assign]

    success, _ = provider.generate_image("Test", "out", stream=False, max_retries=1)

    assert success is False
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
@patch("time.time", return_value=1234567.89)
def test_generate_image_stream_success(
    mock_time: MagicMock,
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test successful image generation with stream enabled."""
    client = MagicMock()
    mock_chunk = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"img", mime_type="image/jpeg")
    mock_chunk.parts = [mock_part]

    client.models.generate_content_stream.return_value = [mock_chunk]

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = GeminiProvider(client)
    success, path = provider.generate_image(
        prompt="Test", output_dir="fake_dir", stream=True
    )

    assert success is True
    assert "gemini_generated_1234567890.jpg" in path
    mock_img_instance.save.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
def test_generate_image_no_stream_success(
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test successful image generation with stream disabled."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"img", mime_type="image/png")
    mock_response.parts = [mock_part]

    client.models.generate_content.return_value = mock_response

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = GeminiProvider(client)
    success, path = provider.generate_image(
        prompt="Test", output_dir="fake_dir", stream=False
    )

    assert success is True
    assert ".png" in path
    mock_img_instance.save.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("os.makedirs")
def test_generate_image_exception_fallback(
    mock_makedirs: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test exception during API call triggers the placeholder fallback logic."""
    client = MagicMock()
    client.models.generate_content_stream.side_effect = Exception("Stream fail")

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="not_an_image")  # type: ignore[method-assign]

    success, _ = provider.generate_image("Test", "out", stream=True, max_retries=2)

    assert success is False
    assert mock_sleep.call_count == 1
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
def test_generate_image_success_after_retry_with_delay(
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test image generation success on a subsequent retry with an explicit delay."""
    client = MagicMock()

    mock_response = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"img", mime_type="image/png")
    mock_response.parts = [mock_part]

    client.models.generate_content.side_effect = [Exception("Fail"), mock_response]

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = GeminiProvider(client)
    success, _ = provider.generate_image(
        prompt="Test", output_dir="out", stream=False, max_retries=2, delay=4.2
    )

    assert success is True
    mock_sleep.assert_called_once_with(4.2)
    assert client.models.generate_content.call_count == 2


# -------------------------------------------------------------------------
# Image Editing Tests
# -------------------------------------------------------------------------


def test_edit_image_raises_file_not_found() -> None:
    """Ensure FileNotFoundError is raised if the edit input image is missing."""
    provider = GeminiProvider(MagicMock())

    with (
        patch("os.path.exists", return_value=False),
        pytest.raises(FileNotFoundError, match="Input image not found"),
    ):
        provider.edit_image("prompt", "invalid.png", "out_dir")


@patch("xwhy.providers.gemini.Image.open")
@patch("os.path.exists", return_value=True)
@patch("os.makedirs")
def test_edit_image_jpg_success(
    mock_makedirs: MagicMock,
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
) -> None:
    """Test editing an existing JPG image ensures correct mime type routing."""
    client = MagicMock()
    mock_chunk = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"img", mime_type="image/jpeg")
    mock_chunk.parts = [mock_part]
    client.models.generate_content_stream.return_value = [mock_chunk]

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = GeminiProvider(client)
    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        success, path = provider.edit_image(
            prompt="Test", image_path="test.JPG", output_dir="out"
        )

    assert success is True
    assert "gemini_edited_" in path


@patch("xwhy.providers.gemini.Image.open")
@patch("os.path.exists", return_value=True)
@patch("os.makedirs")
def test_edit_image_png_success(
    mock_makedirs: MagicMock,
    mock_path_exists: MagicMock,
    mock_image_open: MagicMock,
) -> None:
    """Test editing an existing PNG image ensures correct mime type routing."""
    client = MagicMock()
    mock_chunk = MagicMock()
    mock_part = MagicMock()
    mock_part.inline_data = MagicMock(data=b"img", mime_type="image/png")
    mock_chunk.parts = [mock_part]
    client.models.generate_content_stream.return_value = [mock_chunk]

    mock_img_instance = MagicMock(spec=Image.Image)
    mock_image_open.return_value = mock_img_instance

    provider = GeminiProvider(client)
    with patch("builtins.open", mock_open(read_data=b"fake_image_bytes")):
        success, path = provider.edit_image(
            prompt="Test", image_path="test.png", output_dir="out"
        )

    assert success is True
    assert "gemini_edited_" in path


# -------------------------------------------------------------------------
# Batch Image Job Tests
# -------------------------------------------------------------------------


@patch("os.remove")
@patch("builtins.open")
def test_submit_image_batch_with_image_and_seed(
    mock_open_func: MagicMock, mock_remove: MagicMock
) -> None:
    """Test batch submission with a base image and a deterministic seed."""
    client = MagicMock()
    mock_file = MagicMock(name="remote_file", uri="uri", mime_type="image/png")
    client.files.upload.return_value = mock_file
    mock_job = MagicMock(name="job_123")
    client.batches.create.return_value = mock_job

    provider = GeminiProvider(client)
    job_name = provider.submit_image_batch(
        image_path="test.png", text_list=["prompt1"], seed=42
    )

    assert job_name == str(mock_job.name)
    assert client.files.upload.call_count == 2
    mock_remove.assert_called_once()


@patch("os.remove")
@patch("builtins.open")
def test_submit_image_batch_no_image_os_error(
    mock_open_func: MagicMock, mock_remove: MagicMock
) -> None:
    """Test batch submission without base image, catching OSError on cleanup."""
    client = MagicMock()
    mock_job = MagicMock(name="job_123")
    client.batches.create.return_value = mock_job

    mock_remove.side_effect = OSError("Permission denied")

    provider = GeminiProvider(client)
    job_name = provider.submit_image_batch(
        image_path=None, text_list=["prompt1"], seed=None
    )

    assert job_name == str(mock_job.name)
    client.files.upload.assert_called_once()


@patch("os.makedirs")
@patch("builtins.open")
def test_retrieve_image_batch_success_found_image(
    mock_open_func: MagicMock, mock_makedirs: MagicMock
) -> None:
    """Test polling success and processing of JSONL with valid PNG and JPG data."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": "bW9jaw==",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "key": "gemini_request_1_image",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": "bW9jaw==",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ),
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines) + b"\n"

    provider = GeminiProvider(client)
    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1", "t2"])

    assert len(results) == 2
    assert results[0][0] is True
    assert results[1][0] is True
    assert ".png" in results[0][1]
    assert ".jpg" in results[1][1]


@patch("os.makedirs")
def test_retrieve_image_batch_no_inline_data_and_missing_result(
    mock_makedirs: MagicMock,
) -> None:
    """Handle successful jobs with missing inline data or candidates."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "refused"}]}}]
                },
            }
        )
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines)

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(side_effect=["out.png", None])  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1", "t2"])

    assert len(results) == 1
    assert results[0][0] is False
    assert results[0][1] == "out.png"


@patch("os.makedirs")
def test_retrieve_image_batch_json_parse_error(
    mock_makedirs: MagicMock,
) -> None:
    """Test JSON parsing failures during job retrieval."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value = mock_job

    content = (
        b'{"custom_id": "gemini_request_0_image", invalid\n'
        b"completely bad string format\n"
    )
    client.files.download.return_value = content

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1", "t2"])

    assert len(results) == 2
    assert results[0][0] is False
    assert results[1][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_failed_state(
    mock_makedirs: MagicMock,
) -> None:
    """Test job retrieval logic when the batch job enters a FAILED state."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_FAILED"
    client.batches.get.return_value = mock_job

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1"])

    assert len(results) == 1
    assert results[0][0] is False
    assert results[0][1] == "placeholder.png"
    client.files.download.assert_not_called()


@patch("os.makedirs")
def test_retrieve_image_batch_empty_lines_and_non_string_placeholder(
    mock_makedirs: MagicMock,
) -> None:
    """Handle empty JSONL lines and non-string placeholders."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    client.batches.get.return_value = mock_job

    client.files.download.return_value = b"\n\n"

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(  # type: ignore[method-assign]
        return_value=Image.new("RGB", (10, 10))
    )

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1"])

    assert len(results) == 0
    provider._create_placeholder_image.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("xwhy.providers.gemini.Image.open")
@patch("os.makedirs")
def test_execute_image_request_fallback_is_pil_image(
    mock_makedirs: MagicMock,
    mock_image_open: MagicMock,
    mock_sleep: MagicMock,
) -> None:
    """Test when API fails and fallback image returns a valid PIL Image instance."""
    client = MagicMock()
    client.models.generate_content.side_effect = Exception("API error")

    provider = GeminiProvider(client)
    dummy_img = MagicMock(spec=Image.Image)
    provider._create_placeholder_image = MagicMock(return_value=dummy_img)  # type: ignore[method-assign]

    success, _ = provider.generate_image("Test", "out", stream=False, max_retries=1)

    assert success is False
    dummy_img.save.assert_called_once()


@patch("time.sleep", return_value=None)
@patch("os.makedirs")
@patch("builtins.open")
def test_retrieve_image_batch_polls_with_sleep(
    mock_open_func: MagicMock, mock_makedirs: MagicMock, mock_sleep: MagicMock
) -> None:
    """Test batch retrieval loops with time.sleep(30) when state is pending."""
    client = MagicMock()
    job_running = MagicMock()
    job_running.state.name = "JOB_STATE_RUNNING"

    job_succeeded = MagicMock()
    job_succeeded.state.name = "JOB_STATE_SUCCEEDED"
    job_succeeded.dest.file_name = "results.jsonl"

    client.batches.get.side_effect = [job_running, job_succeeded]
    client.files.download.return_value = b""

    provider = GeminiProvider(client)
    results = provider.retrieve_image_batch(job_name="job1", text_list=[])

    assert results == []
    mock_sleep.assert_called_once_with(30)


@patch("os.makedirs")
@patch("builtins.open")
def test_retrieve_image_batch_comprehensive_coverage(
    mock_open_func: MagicMock, mock_makedirs: MagicMock
) -> None:
    """Verify batch retrieval with various inline data and fallbacks."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "response": {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": "bW9jaw==",
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
            }
        ),
        json.dumps(
            {
                "custom_id": "gemini_request_1_image",
                "response": {
                    "candidates": [
                        {"content": {"parts": [{"text": "just text, no image"}]}}
                    ]
                },
            }
        ),
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines)

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(
        job_name="job1", text_list=["prompt0", "prompt1", "prompt2"]
    )

    assert len(results) == 3
    assert results[0][0] is True
    assert results[1][0] is False
    assert results[2][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_missing_response_and_candidates(
    mock_makedirs: MagicMock,
) -> None:
    """Test batch retrieval with missing response or candidates keys."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "other_key": "value",
            }
        ),
        json.dumps(
            {
                "custom_id": "gemini_request_1_image",
                "response": {
                    "other_field": "value",
                },
            }
        ),
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines)

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(
        job_name="job1", text_list=["prompt0", "prompt1"]
    )

    assert len(results) == 2
    assert results[0][0] is False
    assert results[1][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_json_parse_error_with_custom_id_branch(
    mock_makedirs: MagicMock,
) -> None:
    """Test JSON parse error with custom_id present."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    content = b'{"custom_id": "gemini_request_0_image", invalid_json}\n'
    client.files.download.return_value = content

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1"])

    assert len(results) == 1
    assert results[0][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_candidate_missing_content_or_parts(
    mock_makedirs: MagicMock,
) -> None:
    """Test candidate missing content or parts structure."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "response": {"candidates": [{"other_field": "value"}]},
            }
        )
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines)

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["prompt0"])

    assert len(results) == 1
    assert results[0][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_json_parse_error_without_custom_id(
    mock_makedirs: MagicMock,
) -> None:
    """Test JSON parse error without custom_id key."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    content = b"completely malformed line without custom id key\n"
    client.files.download.return_value = content

    provider = GeminiProvider(client)
    provider._create_placeholder_image = MagicMock(return_value="placeholder.png")  # type: ignore[method-assign]

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1"])

    assert len(results) == 1
    assert results[0][0] is False


@patch("os.makedirs")
def test_retrieve_image_batch_path_none_non_string_placeholder(
    mock_makedirs: MagicMock,
) -> None:
    """Cover path is None with non-string placeholder."""
    client = MagicMock()
    mock_job = MagicMock()
    mock_job.state.name = "JOB_STATE_SUCCEEDED"
    mock_job.dest.file_name = "results.jsonl"
    client.batches.get.return_value = mock_job

    lines = [
        json.dumps(
            {
                "custom_id": "gemini_request_0_image",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "refused"}]}}]
                },
            }
        )
    ]
    client.files.download.return_value = b"\n".join(x.encode() for x in lines)

    provider = GeminiProvider(client)
    # Non-string return covers the false branch of isinstance(placeholder, str)
    # at the path-is-None arm inside the custom_id-in-results block.
    provider._create_placeholder_image = MagicMock(  # type: ignore[method-assign]
        return_value=Image.new("RGB", (10, 10))
    )

    results = provider.retrieve_image_batch(job_name="job1", text_list=["t1"])

    assert len(results) == 0
    provider._create_placeholder_image.assert_called_once()
