"""Unit tests for the HuggingFace provider functionality."""

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock, patch

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


def test_huggingface_provider_api_error() -> None:
    """Test general exception handling during HuggingFace API calls."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception(
        "Model loading or API error"
    )

    provider = HuggingFaceProvider(client=mock_client)

    with pytest.raises(RuntimeError, match="Model loading or API error"):
        provider.answer(prompt="Error prompt test")

    mock_client.chat.completions.create.assert_called_once_with(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Error prompt test"}],
        max_tokens=512,
        temperature=0.1,
    )


def test_huggingface_empty_text_response_raises_error() -> None:
    """Test RuntimeError is raised when HuggingFace returns empty text.

    This covers the 'if not result_text:' block for the HuggingFace API.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()

    mock_choice.message.content = "   \n  "
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    provider = HuggingFaceProvider(client=mock_client)

    expected_error = "empty response from the HuggingFace API"
    with pytest.raises(RuntimeError, match=expected_error):
        provider.answer(prompt="Test empty response")

    mock_client.chat.completions.create.assert_called_once()


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
    mock_pipe.to.assert_called_once()
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
    mock_pipe.to.assert_called_once()
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
    mock_pipe.to.assert_called_once()
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
    """Successfully generate an image and write it to disk.

    Explicitly supplies num_inference_steps so that the
    ``if "num_inference_steps" not in kwargs`` branch evaluates to False.
    """
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_image = MagicMock(spec=Image.Image)
    mock_pipe.return_value = MagicMock(images=[mock_image])

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)
    success, path = provider.generate_image(
        prompt="a red cube",
        output_dir=str(tmp_path),
        num_inference_steps=20,  # already present → if-branch skipped
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


def test_generate_image_no_valid_output_uses_fallback(
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Fall back to a placeholder image when the pipeline returns nothing useful."""
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
        )

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")


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
    src_path.write_bytes(b"")  # satisfy os.path.exists

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
    """Inject the default image_guidance_scale for InstructPix2Pix pipelines."""
    src_path = tmp_path / "src.png"
    src_path.write_bytes(b"")  # satisfy os.path.exists

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


def test_execute_image_request_pipeline_exception_uses_fallback(
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
        )

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")


@patch("xwhy.providers.huggingface.Image.open")
@patch("xwhy.providers.huggingface.ImageOps.exif_transpose")
def test_edit_image_inpaint_without_segmentation_raises(
    mock_exif: MagicMock,
    mock_open: MagicMock,
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Raise ValueError when an inpaint pipe is used without a segmentation model.

    Covers the exact branch:
        elif is_inpaint and input_image_path is not None:
            raise ValueError(
                "segmentation_model is required for Inpainting pipelines."
            )
    """
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
            # segmentation_model deliberately omitted
        )


def test_generate_image_default_inference_steps(
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Inject num_inference_steps=30 for non-inpaint pipelines.

    Covers the True arm of:
        if "num_inference_steps" not in kwargs:
            kwargs["num_inference_steps"] = 50 if is_inpaint else 30
    """
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
    """Inject num_inference_steps=50 for inpaint pipelines.

    Covers the True arm of the same if-statement and the True arm of the
    ternary ``50 if is_inpaint else 30``.
    """
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


def test_pipeline_exception_fallback_is_image(
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Assign the placeholder when _create_placeholder_image returns an Image."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.side_effect = RuntimeError("boom")

    placeholder = MagicMock(spec=Image.Image)
    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(provider, "_create_placeholder_image", return_value=placeholder):
        success, path = provider.generate_image(prompt="x", output_dir=str(tmp_path))

    assert success is False
    placeholder.save.assert_called_once()
    assert path.endswith(".png")


def test_pipeline_exception_fallback_not_image(
    tmp_path: Any,  # noqa: ANN401
) -> None:
    """Skip save when the fallback is not a PIL Image."""
    mock_pipe = _make_pipe("StableDiffusionPipeline")
    mock_pipe.side_effect = RuntimeError("boom")

    provider = HuggingFaceProvider(client=MagicMock(), pipe=mock_pipe)

    with patch.object(
        provider, "_create_placeholder_image", return_value="not-an-image"
    ):
        success, path = provider.generate_image(prompt="x", output_dir=str(tmp_path))

    assert success is False
    # path is still built, but no .save() occurred
    assert path.endswith(".png")
