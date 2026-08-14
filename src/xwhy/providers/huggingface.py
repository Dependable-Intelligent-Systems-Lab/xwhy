"""HuggingFace provider implementation."""

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL import Image, ImageOps

from xwhy.core.types import BaseImageGenerationAndEditing
from xwhy.logger import logger
from xwhy.providers.base import BaseProvider
from xwhy.utils.image import get_binary_mask


class HuggingFaceProvider(BaseImageGenerationAndEditing, BaseProvider):
    """Implement HuggingFace provider interface for text and image tasks."""

    def __init__(
        self,
        client: Any,  # noqa: ANN401
        config: Any | None = None,  # noqa: ANN401
        pipe: Any | None = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the provider with API client, config, or custom pipeline.

        Args:
            client: Configured HuggingFace InferenceClient for text generation.
            config: Optional application configuration containing settings.
            pipe: Optional pre-initialized Diffusers pipeline object.
            **kwargs: Extra parameters (e.g., model_name, device, etc.).

        """
        super().__init__(client)
        self._client = client
        self.config = config
        self._provider_kwargs = kwargs

        # Extract options from config or kwargs for maximum flexibility
        self.model_name: str | None = kwargs.get("model_name") or (
            getattr(config, "model_name", None) if config else None
        )
        self.device: str | torch.device = kwargs.get("device") or (  # type: ignore[assignment]
            getattr(config, "device", "cpu") if config else "cpu"
        )
        self.use_segmentation_model: bool = bool(
            kwargs.get("use_segmentation_model")
            or (getattr(config, "use_segmentation_model", False) if config else False)
        )

        # Scenario B: User provided a custom pre-loaded pipeline
        if pipe is not None or "pipe" in kwargs:
            logger.debug("Using custom pre-loaded HuggingFace pipeline.")
            self.pipe = pipe if pipe is not None else kwargs["pipe"]
        else:
            # Scenario A: Build pipeline dynamically from model_name
            self.pipe = self._initialize_pipeline()

    def _initialize_pipeline(self) -> Any:  # noqa: ANN401
        """Initialize HuggingFace pipeline dynamically based on model_name.

        Returns:
            The initialized Diffusers pipeline, or None if no model_name is set.

        Raises:
            RuntimeError: If model requirements (like segmentation) are missing.

        """
        if not self.model_name:
            logger.debug("No model_name or pipe provided to HuggingFaceProvider.")
            return None

        model_name_lower = self.model_name.lower()

        # Special Case 1: Instruct-Pix2Pix
        if "instruct-pix2pix" in model_name_lower:
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_instruct_pix2pix import (  # noqa: E501
                StableDiffusionInstructPix2PixPipeline,
            )
            from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
                EulerAncestralDiscreteScheduler,
            )

            logger.debug(
                "Initializing InstructPix2Pix pipeline for '%s'...",
                self.model_name,
            )
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                safety_checker=None,
            )
            pipe.to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config
            )
            return pipe

        # Special Case 2: Inpainting models
        if "inpaint" in model_name_lower:
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (  # noqa: E501
                StableDiffusionInpaintPipeline,
            )

            if not self.use_segmentation_model:
                raise RuntimeError(
                    "To use inpainting models, you must set "
                    "`use_segmentation_model=True`."
                )

            logger.debug(
                "Initializing Inpainting pipeline for '%s'...",
                self.model_name,
            )
            pipe = StableDiffusionInpaintPipeline.from_pretrained(  # type: ignore[assignment]
                self.model_name,
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            pipe.to(self.device)
            pipe.enable_attention_slicing()
            return pipe

        # General Case: Any HuggingFace Diffusers model
        logger.debug(
            "Attempting to initialize Diffusers pipeline for '%s'...", self.model_name
        )
        try:
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

            pipe = AutoPipelineForText2Image.from_pretrained(  # type: ignore[no-untyped-call]
                self.model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
            )
            pipe.to(self.device)
            return pipe

        except Exception as exc:
            logger.debug(
                "Could not load '%s' as AutoPipelineForText2Image (This is expected"
                " if it's an LLM). Fallback to DiffusionPipeline. Error: %s",
                self.model_name,
                exc,
            )
            try:
                from diffusers.pipelines.pipeline_utils import DiffusionPipeline

                pipe = DiffusionPipeline.from_pretrained(  # type: ignore[assignment]
                    self.model_name,
                    torch_dtype=torch.float16
                    if torch.cuda.is_available()
                    else torch.float32,
                )
                pipe.to(self.device)
                return pipe

            except Exception as inner_exc:
                logger.warning(
                    "Failed to load '%s' as a Diffusers pipeline. "
                    "Image generation will be disabled, "
                    "but text generation will still work. Reason: %s",
                    self.model_name,
                    inner_exc,
                )
                return None

    def _generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text from HuggingFace.

        Args:
            prompt: Input prompt.
            model: HuggingFace model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If the API returns an empty response or fails.

        """
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )

            result_text = str(response.choices[0].message.content).strip()

            if not result_text:
                error_message = (
                    "Received an empty response from the HuggingFace API. "
                    "This could be due to guardrails or network filtering."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except Exception as exc:
            logger.error("HuggingFace request failed: %s", exc)
            raise RuntimeError(f"HuggingFace request failed: {exc}") from exc

    def answer(
        self,
        prompt: str,
        *,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: HuggingFace model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated response text.

        """
        return self._generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # -------------------------------------------------------------------------
    # Image Generation & Editing Methods
    # -------------------------------------------------------------------------

    @property
    def supports_mask(self) -> bool:
        """Check if the underlying diffusers pipeline supports/requires a mask image."""
        if self.pipe is None:
            return False
        pipe_class_name = type(self.pipe).__name__
        return "Inpaint" in pipe_class_name or "Mask" in pipe_class_name

    def _execute_image_request(
        self,
        prompt: str,
        output_dir: str,
        input_image_path: str | None = None,
        segmentation_model: Callable[[torch.Tensor], Any] | None = None,
        transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
        class_names: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Execute core image generation or editing logic for HuggingFace pipelines.

        Args:
            prompt: Text instruction describing the image modification.
            output_dir: Directory where the generated image will be saved.
            input_image_path: Path to source image if editing, None otherwise.
            segmentation_model: Optional model for generating segmentation masks.
            transform_fn: Preprocessing transform for PIL image.
            class_names: Optional sequence of class names for mask generation.
            **kwargs: Additional pipeline-specific parameters.

        Returns:
            A tuple containing a boolean success flag and the generated file path.

        Raises:
            RuntimeError: If the pipeline is not initialized.
            ValueError: If an inpainting model is used without a segmentation model.

        """
        if self.pipe is None:
            raise RuntimeError("HuggingFace pipeline is not initialized.")

        gen_img_flag = True
        generated_img: Image.Image | None = None

        image: Image.Image | None = None
        if input_image_path is not None:
            image = Image.open(input_image_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

        mask_image: Image.Image | None = None
        pipe_class_name = type(self.pipe).__name__
        is_inpaint = "Inpaint" in pipe_class_name
        is_pix2pix = "InstructPix2Pix" in pipe_class_name

        if self.supports_mask:
            if segmentation_model is not None and input_image_path is not None:
                mask_image = get_binary_mask(
                    image_path=input_image_path,
                    segmentation_model=segmentation_model,
                    transform_fn=transform_fn,
                    device=self.device,
                    class_names=class_names,
                )
            elif is_inpaint and input_image_path is not None:
                raise ValueError(
                    "segmentation_model is required for Inpainting pipelines."
                )

        try:
            # Auto-adapt general Text2Image pipeline to Image2Image if editing
            current_pipe = self.pipe
            if (
                input_image_path is not None
                and not is_pix2pix
                and not is_inpaint
                and "Image2Image" not in pipe_class_name
            ):
                from diffusers.pipelines.auto_pipeline import AutoPipelineForImage2Image

                logger.debug("Converting pipeline to AutoPipelineForImage2Image...")
                current_pipe = AutoPipelineForImage2Image.from_pipe(self.pipe)  # type: ignore[no-untyped-call]

            call_kwargs: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                call_kwargs["image"] = image
            if mask_image is not None and self.supports_mask:
                call_kwargs["mask_image"] = mask_image

            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 50 if is_inpaint else 30

            if is_pix2pix and "image_guidance_scale" not in kwargs:
                kwargs["image_guidance_scale"] = 1.0

            call_kwargs.update(kwargs)

            output = current_pipe(**call_kwargs)

            if hasattr(output, "images") and output.images:
                generated_img = output.images[0]
            elif isinstance(output, list) and output:
                generated_img = output[0]
            else:
                raise RuntimeError("No valid images found in pipeline output.")

        except Exception as exc:
            gen_img_flag = False
            logger.exception(
                "Pipeline execution failed for prompt '%s': %s", prompt, exc
            )
            fallback_img = self._create_placeholder_image(
                prompt=prompt, output_dir=output_dir, save=False
            )
            if isinstance(fallback_img, Image.Image):
                generated_img = fallback_img

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = "hf_edited" if input_image_path else "hf_generated"
        filename = f"{prefix}_{timestamp}.png"
        gen_path = os.path.join(output_dir, filename)

        if isinstance(generated_img, Image.Image):
            generated_img.save(gen_path)

        logger.debug(
            "----- '%s' generated! (Success: %s) -----", gen_path, gen_img_flag
        )

        return gen_img_flag, gen_path

    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image using the initialized Diffusers pipeline.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            **kwargs: Extra parameters for the pipeline.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            input_image_path=None,
            **kwargs,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        *,
        segmentation_model: Callable[[torch.Tensor], Any] | None = None,
        transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
        class_names: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an edited image using the initialized Diffusers pipeline.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            segmentation_model: Optional model for generating segmentation masks.
            transform_fn: Preprocessing transform for PIL image.
            class_names: Optional sequence of class names for mask generation.
            **kwargs: Extra parameters for the pipeline.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        Raises:
            FileNotFoundError: If the provided input image is not found.

        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            input_image_path=image_path,
            segmentation_model=segmentation_model,
            transform_fn=transform_fn,
            class_names=class_names,
            **kwargs,
        )
