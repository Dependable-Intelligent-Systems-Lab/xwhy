"""Unit tests for the HuggingFace provider functionality."""

import re
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from PIL import Image

from xwhy.providers.huggingface import HuggingFaceProvider


def _make_pipe(class_name: str) -> MagicMock:
    """Build a MagicMock whose type(pipe).__name__ equals class_name.

    Args:
        class_name: Desired value of ``type(pipe).__name__``.

    Returns:
        A MagicMock instance whose type name is ``class_name``.

    """
    return type(class_name, (MagicMock,), {})()  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Text-generation tests
# ---------------------------------------------------------------------------


def test_huggingface_provider_success() -> None:
    """Test successful text generation with HuggingFace."""
    mock_client = MagicMock()
    mock_response = MagicMock()

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


@patch("time.sleep", return_value=None)
def test_huggingface_provider_api_error(mock_sleep: MagicMock) -> None:
    """Test general exception handling and retries during HuggingFace API calls."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Model loading or API error"
    )

    provider = HuggingFaceProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="Model loading or API error"):
        provider.answer(prompt="Error prompt test", max_retries=3)

    assert mock_client.chat.completions.create.call_count == 3
    assert mock_sleep.call_count == 2


@patch("time.sleep", return_value=None)
def test_huggingface_empty_text_response_raises_error(
    mock_sleep: MagicMock,
) -> None:
    """Test RuntimeError is raised when HuggingFace returns empty text."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()

    mock_choice.message.content = "   \n  "
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    provider = HuggingFaceProvider(client=mock_client)

    expected_error = "empty response from the HuggingFace API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response", max_retries=2)

    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_huggingface_generate_retry_success(mock_sleep: MagicMock) -> None:
    """Test text generation succeeds after retry with explicit delay."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Recovered text"
    mock_response.choices = [mock_choice]

    mock_client.chat.completions.create.side_effect = [
        Exception("Temporary failure"),
        mock_response,
    ]

    provider = HuggingFaceProvider(client=mock_client)
    result = provider.answer(prompt="Retry test", max_retries=3, delay=2.5)

    assert result == "Recovered text"
    assert mock_client.chat.completions.create.call_count == 2
    mock_sleep.assert_called_once_with(2.5)


@patch("time.sleep", return_value=None)
def test_huggingface_generate_exponential_backoff(
    mock_sleep: MagicMock,
) -> None:
    """Test exponential backoff for text generation retries up to the 30s cap."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    provider = HuggingFaceProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="HuggingFace request failed"):
        provider.answer(prompt="Test backoff", max_retries=6)

    expected_sleep_calls = [call(2), call(4), call(8), call(16), call(30)]
    mock_sleep.assert_has_calls(expected_sleep_calls)
    assert mock_sleep.call_count == 5


def test_huggingface_generate_zero_retries_raises_fallback() -> None:
    """Test that zero retries triggers the fallback RuntimeError.

    The loop never executes when max_retries is set to zero, so control
    falls through to the final exception raise.

    """
    mock_client = MagicMock()
    provider = HuggingFaceProvider(client=mock_client)

    with pytest.raises(
        RuntimeError,
        match=re.escape("HuggingFace text generation failed after max retries."),
    ):
        provider.answer(prompt="prompt", max_retries=0)

    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# __init__ / configuration paths
# ---------------------------------------------------------------------------


def test_init_with_pipe_kwarg() -> None:
    """Use a pre-loaded pipeline supplied via the pipe keyword argument."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    provider = HuggingFaceProvider(client=mock_client, pipe=mock_pipe)

    assert provider.pipe is mock_pipe
    assert provider._client is mock_client


def test_init_with_pipe_in_kwargs() -> None:
    """Accept pipe when it is only present inside **kwargs."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    provider = HuggingFaceProvider(client=mock_client, **{"pipe": mock_pipe})

    assert provider.pipe is mock_pipe


def test_init_extracts_options_from_config() -> None:
    """Read model_name, device and use_segmentation_model from a config object."""
    mock_client = MagicMock()
    mock_config = MagicMock()
    mock_config.model_name = "some/model"
    mock_config.device = "cuda"
    mock_config.use_segmentation_model = True

    with patch.object(
        HuggingFaceProvider, "_initialize_pipeline", return_value=None
    ) as mock_init:
        provider = HuggingFaceProvider(client=mock_client, config=mock_config)

    assert provider.model_name == "some/model"
    assert provider.device == "cuda"
    assert provider.use_segmentation_model is True
    mock_init.assert_called_once()


def test_init_prefers_kwargs_over_config() -> None:
    """Keyword arguments override values that would come from config."""
    mock_client = MagicMock()
    mock_config = MagicMock()
    mock_config.model_name = "config/model"
    mock_config.device = "cpu"
    mock_config.use_segmentation_model = False

    with patch.object(HuggingFaceProvider, "_initialize_pipeline", return_value=None):
        provider = HuggingFaceProvider(
            client=mock_client,
            config=mock_config,
            model_name="kwarg/model",
            device="cuda:0",
            use_segmentation_model=True,
        )

    assert provider.model_name == "kwarg/model"
    assert provider.device == "cuda:0"
    assert provider.use_segmentation_model is True


def test_init_defaults_when_no_config_or_kwargs() -> None:
    """Fall back to safe defaults when neither config nor kwargs supply options."""
    mock_client = MagicMock()
    with patch.object(HuggingFaceProvider, "_initialize_pipeline", return_value=None):
        provider = HuggingFaceProvider(client=mock_client)

    assert provider.model_name is None
    assert provider.device == "cpu"
    assert provider.use_segmentation_model is False
    assert provider.pipe is None


# ---------------------------------------------------------------------------
# _initialize_pipeline branches
# ---------------------------------------------------------------------------


def test_initialize_pipeline_no_model_name() -> None:
    """Return None when neither model_name nor pipe is provided."""
    mock_client = MagicMock()
    provider = HuggingFaceProvider(client=mock_client)

    assert provider.pipe is None


def test_initialize_pipeline_instruct_pix2pix() -> None:
    """Build an InstructPix2Pix pipeline when the model name contains that token."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.scheduler.config = {}

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
        patch(
            "diffusers.EulerAncestralDiscreteScheduler.from_config",
            return_value=MagicMock(),
        ),
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="timbrooks/instruct-pix2pix",
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["dtype"] == torch.float32
    mock_pipe.to.assert_called_once()
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_instruct_pix2pix_cuda() -> None:
    """Build an InstructPix2Pix pipeline when CUDA is available (float16)."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.scheduler.config = {}

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
        patch(
            "diffusers.EulerAncestralDiscreteScheduler.from_config",
            return_value=MagicMock(),
        ),
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="timbrooks/instruct-pix2pix",
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["dtype"] == torch.float16
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_inpaint_requires_segmentation() -> None:
    """Raise RuntimeError when an inpaint model is requested without segmentation."""
    mock_client = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="use_segmentation_model=True"),
    ):
        HuggingFaceProvider(
            client=mock_client,
            model_name="runwayml/stable-diffusion-inpainting",
            use_segmentation_model=False,
        )


def test_initialize_pipeline_inpaint_success() -> None:
    """Build an inpainting pipeline when use_segmentation_model is True."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "diffusers.StableDiffusionInpaintPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="runwayml/stable-diffusion-inpainting",
            use_segmentation_model=True,
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["dtype"] == torch.float32
    mock_pipe.to.assert_called_once()
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_inpaint_cuda() -> None:
    """Build an inpainting pipeline when CUDA is available (float16)."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "diffusers.StableDiffusionInpaintPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="runwayml/stable-diffusion-inpainting",
            use_segmentation_model=True,
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["dtype"] == torch.float16
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_auto_text2image_success() -> None:
    """Use AutoPipelineForText2Image for a generic diffusers model name."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "diffusers.AutoPipelineForText2Image.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="stabilityai/stable-diffusion-2",
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["torch_dtype"] == torch.float32
    mock_pipe.to.assert_called_once()
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_auto_text2image_cuda() -> None:
    """Use AutoPipelineForText2Image with CUDA enabled (float16)."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "diffusers.AutoPipelineForText2Image.from_pretrained",
            return_value=mock_pipe,
        ) as mock_from_pretrained,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="stabilityai/stable-diffusion-2",
        )

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["torch_dtype"] == torch.float16
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_falls_back_to_diffusion_pipeline() -> None:
    """Fall back to DiffusionPipeline when AutoPipelineForText2Image fails."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "diffusers.AutoPipelineForText2Image.from_pretrained",
            side_effect=Exception("not a text2image model"),
        ),
        patch(
            "diffusers.DiffusionPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_diffusion,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="some/other-model",
        )

    mock_diffusion.assert_called_once()
    assert mock_diffusion.call_args.kwargs["torch_dtype"] == torch.float32
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_falls_back_to_diffusion_pipeline_cuda() -> None:
    """Fall back to DiffusionPipeline when CUDA is available (float16)."""
    mock_client = MagicMock()
    mock_pipe = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch(
            "diffusers.AutoPipelineForText2Image.from_pretrained",
            side_effect=Exception("not a text2image model"),
        ),
        patch(
            "diffusers.DiffusionPipeline.from_pretrained",
            return_value=mock_pipe,
        ) as mock_diffusion,
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="some/other-model",
        )

    mock_diffusion.assert_called_once()
    assert mock_diffusion.call_args.kwargs["torch_dtype"] == torch.float16
    assert provider.pipe is mock_pipe


def test_initialize_pipeline_all_loads_fail() -> None:
    """Return None and log a warning when every pipeline loader fails."""
    mock_client = MagicMock()

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch(
            "diffusers.AutoPipelineForText2Image.from_pretrained",
            side_effect=Exception("auto failed"),
        ),
        patch(
            "diffusers.DiffusionPipeline.from_pretrained",
            side_effect=Exception("diffusion failed"),
        ),
    ):
        provider = HuggingFaceProvider(
            client=mock_client,
            model_name="completely/broken-model",
        )

    assert provider.pipe is None


# ---------------------------------------------------------------------------
# supports_mask property
# ---------------------------------------------------------------------------


def test_supports_mask_false_when_pipe_is_none() -> None:
    """Return False when no pipeline has been initialised."""
    provider = HuggingFaceProvider(client=MagicMock())
    assert provider.supports_mask is False


def test_supports_mask_true_for_inpaint_pipe() -> None:
    """Return True when the pipeline class name contains 'Inpaint'."""
    mock_pipe = _make_pipe("StableDiffusionInpaintPipeline")
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    assert provider.supports_mask is True


def test_supports_mask_true_for_mask_pipe() -> None:
    """Return True when the pipeline class name contains 'Mask'."""
    mock_pipe = _make_pipe("SomeMaskPipeline")
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    assert provider.supports_mask is True


def test_supports_mask_false_for_generic_pipe() -> None:
    """Return False for ordinary text-to-image pipelines."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    assert provider.supports_mask is False


# ---------------------------------------------------------------------------
# _execute_image_request / generate_image / edit_image
# ---------------------------------------------------------------------------


def test_execute_image_request_raises_when_pipe_none() -> None:
    """Raise RuntimeError when the pipeline was never initialised."""
    provider = HuggingFaceProvider(client=MagicMock())
    with pytest.raises(RuntimeError, match="pipeline is not initialized"):
        provider.generate_image(prompt="a cat", output_dir="/tmp")


def test_generate_image_success(tmp_path: Any) -> None:  # noqa: ANN401
    """Successfully generate an image and write it to disk."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_image])

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    success, path = provider.generate_image(
        prompt="a red cube",
        output_dir=str(tmp_path),
        num_inference_steps=20,
    )

    assert success is True
    assert path.startswith(str(tmp_path))
    assert path.endswith(".png")
    mock_image.save.assert_called_once()
    assert mock_pipe.call_args.kwargs["num_inference_steps"] == 20


def test_generate_image_output_as_list(tmp_path: Any) -> None:  # noqa: ANN401
    """Accept a bare list return value from the pipeline."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = [mock_image]

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    success, path = provider.generate_image(
        prompt="a blue sphere",
        output_dir=str(tmp_path),
    )

    assert success is True
    mock_image.save.assert_called_once()
    assert path.endswith(".png")


@patch("time.sleep", return_value=None)
def test_generate_image_no_valid_output_uses_fallback(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Fall back to a placeholder image when pipeline returns no valid image."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.return_value = MagicMock(images=[])

    placeholder = MagicMock(spec=Image.Image)
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(
        provider,
        "_create_placeholder_image",
        return_value=placeholder,
    ):
        success, path = provider.generate_image(
            prompt="broken",
            output_dir=str(tmp_path),
            max_retries=2,
        )

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")
    mock_sleep.assert_called_once()


@patch("time.sleep", return_value=None)
def test_generate_image_retry_success(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test image generation retry success on second attempt with explicit delay."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.side_effect = [
        Exception("Temporary failure"),
        MagicMock(images=[mock_image]),
    ]

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    success, path = provider.generate_image(
        prompt="a cat",
        output_dir=str(tmp_path),
        max_retries=3,
        delay=3.5,
    )

    assert success is True
    assert path.endswith(".png")
    mock_sleep.assert_called_once_with(3.5)
    assert mock_pipe.call_count == 2


def test_edit_image_file_not_found() -> None:
    """Raise FileNotFoundError when the source image path does not exist."""
    mock_pipe = MagicMock()
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with pytest.raises(FileNotFoundError, match="Input image not found"):
        provider.edit_image(
            prompt="make it blue",
            image_path="/nonexistent/image.png",
            output_dir="/tmp",
        )


@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_success(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Edit an existing image with a generic (non-inpaint) pipeline."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")

    mock_pil = MagicMock(spec=Image.Image)
    mock_pil.convert.return_value = mock_pil
    mock_open.return_value = mock_pil
    mock_exif.return_value = mock_pil

    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_out_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_out_image])

    mock_img2img = MagicMock()
    mock_img2img.return_value = MagicMock(images=[mock_out_image])

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch(
        "diffusers.AutoPipelineForImage2Image.from_pipe",
        return_value=mock_img2img,
    ):
        success, path = provider.edit_image(
            prompt="add a hat",
            image_path=str(src_path),
            output_dir=str(tmp_path),
        )

    assert success is True
    assert path.endswith(".png")
    mock_out_image.save.assert_called_once()


@patch("xwhy.providers.huggingface.get_binary_mask")
@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_inpaint_with_segmentation(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    mock_get_mask: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Run an inpainting pipeline when a segmentation model is supplied."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")

    mock_pil = MagicMock(spec=Image.Image)
    mock_pil.convert.return_value = mock_pil
    mock_open.return_value = mock_pil
    mock_exif.return_value = mock_pil
    mock_mask = MagicMock(spec=Image.Image)
    mock_get_mask.return_value = mock_mask

    mock_pipe = _make_pipe("StableDiffusionInpaintPipeline")
    mock_out_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_out_image])

    provider = HuggingFaceProvider(
        client=MagicMock(),
        pipe=mock_pipe,
        use_segmentation_model=True,
    )

    seg_model: Callable[[torch.Tensor], Any] = MagicMock()
    success, path = provider.edit_image(
        prompt="fill the hole",
        image_path=str(src_path),
        output_dir=str(tmp_path),
        segmentation_model=seg_model,
    )

    assert success is True
    mock_get_mask.assert_called_once()
    call_kwargs = mock_pipe.call_args.kwargs
    assert call_kwargs.get("mask_image") is mock_mask
    assert path.endswith(".png")


@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_pix2pix_sets_guidance(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Inject default image_guidance_scale for InstructPix2Pix pipelines."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")

    mock_pil = MagicMock(spec=Image.Image)
    mock_pil.convert.return_value = mock_pil
    mock_open.return_value = mock_pil
    mock_exif.return_value = mock_pil

    mock_pipe = _make_pipe("StableDiffusionInstructPix2PixPipeline")
    mock_out_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_out_image])

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    success, _path = provider.edit_image(
        prompt="make it night",
        image_path=str(src_path),
        output_dir=str(tmp_path),
    )

    assert success is True
    call_kwargs = mock_pipe.call_args.kwargs
    assert call_kwargs.get("image_guidance_scale") == 1.0


@patch("time.sleep", return_value=None)
def test_execute_image_request_pipeline_exception_uses_fallback(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Catch pipeline exceptions, set success=False and save a placeholder."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.side_effect = RuntimeError("CUDA OOM")

    placeholder = MagicMock(spec=Image.Image)
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(
        provider,
        "_create_placeholder_image",
        return_value=placeholder,
    ):
        success, path = provider.generate_image(
            prompt="boom",
            output_dir=str(tmp_path),
            max_retries=2,
        )

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")
    mock_sleep.assert_called_once()


@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_inpaint_without_segmentation_raises(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Raise ValueError when inpaint pipe is used without segmentation model."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")

    mock_pil = MagicMock(spec=Image.Image)
    mock_pil.convert.return_value = mock_pil
    mock_open.return_value = mock_pil
    mock_exif.return_value = mock_pil

    mock_pipe = _make_pipe("StableDiffusionInpaintPipeline")

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with pytest.raises(
        ValueError,
        match="segmentation_model is required for Inpainting pipelines",
    ):
        provider.edit_image(
            prompt="fill",
            image_path=str(src_path),
            output_dir=str(tmp_path),
        )


def test_generate_image_default_inference_steps(
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Inject num_inference_steps=30 for non-inpaint pipelines."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_image])

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    provider.generate_image(prompt="a cat", output_dir=str(tmp_path))

    assert mock_pipe.call_args.kwargs["num_inference_steps"] == 30


@patch("xwhy.providers.huggingface.get_binary_mask")
@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_inpaint_default_inference_steps(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    mock_get_mask: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Inject num_inference_steps=50 for inpaint pipelines."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")

    mock_pil = MagicMock(spec=Image.Image)
    mock_pil.convert.return_value = mock_pil
    mock_open.return_value = mock_pil
    mock_exif.return_value = mock_pil
    mock_get_mask.return_value = MagicMock(spec=Image.Image)

    mock_pipe = _make_pipe("StableDiffusionInpaintPipeline")
    mock_out = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_out])

    provider = HuggingFaceProvider(
        client=MagicMock(),
        pipe=mock_pipe,
        use_segmentation_model=True,
    )
    provider.edit_image(
        prompt="fill",
        image_path=str(src_path),
        output_dir=str(tmp_path),
        segmentation_model=MagicMock(),
    )

    assert mock_pipe.call_args.kwargs["num_inference_steps"] == 50


@patch("time.sleep", return_value=None)
def test_pipeline_exception_fallback_is_image(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Assign placeholder when _create_placeholder_image returns an Image."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.side_effect = RuntimeError("boom")

    placeholder = MagicMock(spec=Image.Image)
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(provider, "_create_placeholder_image", return_value=placeholder):
        success, path = provider.generate_image(
            prompt="x", output_dir=str(tmp_path), max_retries=1
        )

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")


@patch("time.sleep", return_value=None)
def test_pipeline_exception_fallback_not_image(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Skip save when fallback is not a PIL Image."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.side_effect = RuntimeError("boom")

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(
        provider, "_create_placeholder_image", return_value="not-an-image"
    ):
        success, path = provider.generate_image(
            prompt="x", output_dir=str(tmp_path), max_retries=1
        )

    assert success is False
    assert path.endswith(".png")


@patch("os.makedirs")
def test_generate_image_inpaint_pipe_no_mask_required(
    mock_makedirs: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Fall through supports_mask block when input_image_path is None."""
    mock_pipe = _make_pipe("StableDiffusionInpaintPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_image])

    provider = HuggingFaceProvider(
        client=MagicMock(),
        pipe=mock_pipe,
        use_segmentation_model=True,
    )
    success, path = provider.generate_image(
        prompt="a landscape",
        output_dir=str(tmp_path),
    )

    assert success is True
    assert path.endswith(".png")
    mock_image.save.assert_called_once()


@patch("time.sleep", return_value=None)
def test_execute_image_request_generated_img_none_no_break(
    mock_sleep: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Test retry continuation when generated image is None.

    Verifies that if the pipeline returns a list where the first element
    is None, the retry loop continues instead of breaking.

    Args:
        mock_sleep: Mock for time.sleep to avoid real delays.
        tmp_path: Pytest temporary directory fixture.

    """
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.return_value = MagicMock(images=[None])

    placeholder = MagicMock(spec=Image.Image)
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(
        provider,
        "_create_placeholder_image",
        return_value=placeholder,
    ):
        success, path = provider.generate_image(
            prompt="none image",
            output_dir=str(tmp_path),
            max_retries=2,
        )

    assert success is False
    assert path.endswith(".png")
    # Both attempts ran (no break occurred).
    assert mock_pipe.call_count == 2
    mock_sleep.assert_called_once()
    placeholder.save.assert_called_once()
