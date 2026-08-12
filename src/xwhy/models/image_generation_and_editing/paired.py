"""Paired inference model implementation."""

import time
from pathlib import Path
from typing import Any

from img2img_turbo import run_inference_paired

from xwhy.core.types import BaseImageGenerationAndEditing
from xwhy.logger import logger


class PairedInferenceModel(BaseImageGenerationAndEditing):
    """Implement paired inference engine for image editing tasks."""

    def __init__(self, model_name: str) -> None:
        """Initialize the paired inference engine.

        Args:
            model_name: Name of the paired inference model to be executed.

        """
        self.model_name = model_name

    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image from scratch.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            **kwargs: Extra parameters.

        Raises:
            NotImplementedError: Since paired inference only supports editing.

        """
        raise NotImplementedError(
            "Paired inference does not support generation from scratch. "
            "Please use the `edit_image` method instead."
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Edit an existing image using the paired inference function.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            **kwargs: Extra parameters for the underlying inference function.

        Returns:
            A tuple containing a boolean success flag and the generated file path.

        """
        try:
            logger.debug("Running paired inference for '%s'...", self.model_name)

            # Strip model_name from kwargs to prevent keyword collision
            kwargs.pop("model_name", None)

            # Execute the local paired inference function
            run_inference_paired(
                model_name=self.model_name,
                input_image=image_path,
                prompt=prompt,
                output_dir=output_dir,
                **kwargs,
            )

            output_path = Path(output_dir)
            if not output_path.exists() or not output_path.is_dir():
                raise FileNotFoundError(f"Output directory not found: {output_dir}")

            # Filter valid files inside the output directory
            files = [p for p in output_path.iterdir() if p.is_file()]
            if not files:
                raise FileNotFoundError(f"No files found in directory: {output_dir}")

            # Find the most recently modified file
            generated_file = max(files, key=lambda p: p.stat().st_mtime)
            timestamp = int(time.time() * 1000)

            # Construct the new filename and rename the file
            new_filename = (
                f"{self.model_name}_edited_{timestamp}{generated_file.suffix}"
            )
            new_file_path = generated_file.rename(
                generated_file.with_name(new_filename)
            )

            logger.debug("Paired inference succeeded! Output: %s", new_file_path)
            return True, str(new_file_path)

        except Exception as exc:
            # Use logger.exception to capture the full traceback for debugging
            logger.exception("Unexpected error during paired image editing: %s", exc)
            return False, ""
