"""HuggingFace provider implementation."""

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from PIL import Image, ImageOps

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider
from xwhy.utils.image import get_binary_mask


class HuggingFaceProvider(BaseProvider):
    """HuggingFace implementation of the provider interface."""

    def __init__(self, client: object) -> None:
        """Initialize the provider.

        Args:
            client: Configured HuggingFace InferenceClient for text generation.

        """
        super().__init__(client)
        self._client = client

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
            RuntimeError: If the API returns an empty response.

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
                    "This could be due to model-specific guardrails, network "
                    "filtering (anti-filter), or provider-side anomalies."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except RuntimeError:
            raise

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

    def _execute_image_request(
        self,
        prompt: str,
        output_dir: str,
        pipe: Any,  # noqa: ANN401
        input_image_path: str | None = None,
        segmentation_model: Callable[[torch.Tensor], Any] | None = None,
        transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
        device: str | torch.device = "cpu",
        class_names: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Execute core image generation or editing logic for HuggingFace pipelines.

        Args:
            prompt: Text instruction describing the image modification or generation.
            output_dir: Directory where the generated image will be saved.
            pipe: Pre-initialized HuggingFace Diffusers pipeline object.
            input_image_path: Path to source image if editing, None otherwise.
            segmentation_model: Optional model for generating segmentation masks.
            transform_fn: Preprocessing transform for PIL image. Uses default if None.
            device: Target device for mask generation inference.
            class_names: Optional sequence of class names for mask generation.
            **kwargs: Additional pipeline-specific parameters.

        Returns:
            A tuple containing a boolean success flag and the generated file path.

        """
        gen_img_flag = True
        generated_img: Image.Image | None = None

        image: Image.Image | None = None
        if input_image_path is not None:
            image = Image.open(input_image_path)
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")

        mask_image: Image.Image | None = None
        pipe_class_name = type(pipe).__name__
        is_inpaint = "Inpaint" in pipe_class_name
        is_pix2pix = "InstructPix2Pix" in pipe_class_name

        if segmentation_model is not None:
            if input_image_path is not None:
                mask_image = get_binary_mask(
                    image_path=input_image_path,
                    segmentation_model=segmentation_model,
                    transform_fn=transform_fn,
                    device=device,
                    class_names=class_names,
                )
        elif is_inpaint and input_image_path is not None:
            raise ValueError("segmentation_model is required for Inpainting pipelines.")

        try:
            call_kwargs: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                call_kwargs["image"] = image
            if mask_image is not None:
                call_kwargs["mask_image"] = mask_image

            # Handle default inference steps based on pipeline type
            if "num_inference_steps" not in kwargs:
                kwargs["num_inference_steps"] = 50 if is_inpaint else 30

            # Handle specific default for InstructPix2Pix if not provided
            if is_pix2pix and "image_guidance_scale" not in kwargs:
                kwargs["image_guidance_scale"] = 1.0

            call_kwargs.update(kwargs)

            output = pipe(**call_kwargs)
            if hasattr(output, "images") and output.images:
                generated_img = output.images[0]
            elif isinstance(output, list) and output:
                generated_img = output[0]
            else:
                raise RuntimeError("No valid images found in pipeline output.")

        except Exception as e:
            gen_img_flag = False
            logger.exception(
                "Error generating image with HuggingFace pipeline for prompt '%s': %s",
                prompt,
                e,
            )
            fallback_img = self._create_placeholder_image(
                prompt=prompt, output_dir=output_dir, save=False
            )
            if isinstance(fallback_img, Image.Image):
                generated_img = fallback_img

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = (
            "huggingface_edited"
            if input_image_path is not None
            else "huggingface_generated"
        )
        filename = f"{prefix}_{timestamp}.png"
        gen_path = os.path.join(output_dir, filename)

        if isinstance(generated_img, Image.Image):
            generated_img.save(gen_path)

        logger.debug(
            '------------------- "%s" generated! (Success: %s) -------------------',
            gen_path,
            gen_img_flag,
        )

        return gen_img_flag, gen_path

    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        *,
        pipe: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image using a HuggingFace Diffusers pipeline.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            pipe: Pre-initialized HuggingFace Diffusers generation pipeline.
            **kwargs: Extra parameters for the pipeline.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            pipe=pipe,
            input_image_path=None,
            **kwargs,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        *,
        pipe: Any,  # noqa: ANN401
        segmentation_model: Callable[[torch.Tensor], Any] | None = None,
        transform_fn: Callable[[Image.Image], torch.Tensor] | None = None,
        device: str | torch.device = "cpu",
        class_names: Sequence[str] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an edited image using a HuggingFace Diffusers pipeline.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            pipe: Pre-initialized HuggingFace Diffusers editing pipeline.
            segmentation_model: Optional model for generating segmentation masks.
            transform_fn: Preprocessing transform for PIL image. Uses default if None.
            device: Target device for mask generation inference.
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
            pipe=pipe,
            input_image_path=image_path,
            segmentation_model=segmentation_model,
            transform_fn=transform_fn,
            device=device,
            class_names=class_names,
            **kwargs,
        )
