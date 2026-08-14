"""Custom model wrapper for image generation and editing tasks."""

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from PIL import Image

from xwhy.core.types import BaseImageGenerationAndEditing
from xwhy.logger import logger


class CustomImageGenerationAndEditingModel(BaseImageGenerationAndEditing):
    """Wrap user-defined custom image generation and editing logic."""

    def __init__(
        self,
        generate_fn: Callable[..., Any],
        model: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the custom image generation model wrapper.

        Args:
            generate_fn: User-provided callable that executes image generation
                or editing.
            model: Optional underlying model instance (e.g., PyTorch Module).
            **kwargs: Extra static keyword arguments passed to generate_fn.

        """
        self.generate_fn = generate_fn
        self.model = model
        self.kwargs = kwargs

    def _process_execution_result(
        self,
        result: Any,  # noqa: ANN401
        output_dir: str,
        is_edit: bool,
    ) -> str:
        """Process and save output result returned from user function.

        Args:
            result: Return value from generate_fn (path, PIL Image, or Tensor).
            output_dir: Directory path to save generated output image.
            is_edit: Flag indicating whether operation is image editing.

        Returns:
            The string file path of the saved output image.

        Raises:
            ValueError: If result type is not supported.

        """
        if isinstance(result, str):
            return result

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = "custom_edited" if is_edit else "custom_generated"
        gen_path = os.path.join(output_dir, f"{prefix}_{timestamp}.png")

        if isinstance(result, Image.Image):
            result.save(gen_path)
            return gen_path

        if isinstance(result, torch.Tensor):
            tensor = result.detach().cpu()
            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 0)

            tensor_np = tensor.numpy()
            if tensor_np.dtype != np.uint8:
                if tensor_np.max() <= 1.0:
                    tensor_np = (tensor_np * 255).clip(0, 255).astype(np.uint8)
                else:
                    tensor_np = tensor_np.clip(0, 255).astype(np.uint8)

            img = Image.fromarray(tensor_np)
            img.save(gen_path)
            return gen_path

        if isinstance(result, np.ndarray):
            arr = result
            if arr.ndim == 4:
                arr = arr[0]
            if (
                arr.ndim == 3
                and arr.shape[0] in (1, 3, 4)
                and arr.shape[2] not in (1, 3, 4)
            ):
                arr = arr.transpose(1, 2, 0)

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)

            img = Image.fromarray(arr)
            img.save(gen_path)
            return gen_path

        raise ValueError(
            f"Unsupported return type '{type(result)}' from custom generate_fn."
        )

    def _execute_custom_request(
        self,
        prompt: str,
        output_dir: str,
        input_image_path: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Execute core user-defined image generation or editing request.

        Args:
            prompt: Text prompt describing the image or edit operation.
            output_dir: Target directory to save the output image.
            input_image_path: Optional input image path for edit requests.
            **kwargs: Extra dynamic keyword arguments.

        Returns:
            A tuple containing a boolean success flag and the output file path.

        """
        merged_kwargs = {**self.kwargs, **kwargs}
        if self.model is not None:
            merged_kwargs["model"] = self.model

        if input_image_path is not None:
            merged_kwargs["input_image_path"] = input_image_path

        try:
            logger.debug("Executing custom generate_fn for prompt: '%s'", prompt)
            result = self.generate_fn(
                prompt=prompt,
                output_dir=output_dir,
                **merged_kwargs,
            )

            gen_path = self._process_execution_result(
                result=result,
                output_dir=output_dir,
                is_edit=input_image_path is not None,
            )
            logger.debug("Successfully processed custom image output at '%s'", gen_path)
            return True, gen_path

        except Exception as e:
            logger.debug("Error during custom model execution: %s", e)
            return False, str(e)

    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image using the user-defined callable.

        Args:
            prompt: Text prompt for image generation.
            output_dir: Directory where the output image should be stored.
            **kwargs: Additional parameters passed to custom generate_fn.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        return self._execute_custom_request(
            prompt=prompt,
            output_dir=output_dir,
            **kwargs,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Edit an existing image using the user-defined callable.

        Args:
            prompt: Text prompt describing desired image edits.
            image_path: Path to the original source image file.
            output_dir: Directory where the edited image will be stored.
            **kwargs: Additional parameters passed to custom generate_fn.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        Raises:
            FileNotFoundError: If the input image file does not exist.

        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        return self._execute_custom_request(
            prompt=prompt,
            output_dir=output_dir,
            input_image_path=image_path,
            **kwargs,
        )
