"""OpenAI provider implementation."""

import base64
import os
import re
import time
from io import BytesIO
from typing import Any

import requests
from openai import OpenAI
from PIL import Image

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class OpenAIProvider(BaseProvider):
    """OpenAI implementation of the provider interface."""

    def __init__(self, client: OpenAI) -> None:
        """Initialize the provider.

        Args:
            client: Configured OpenAI client.

        """
        super().__init__(client)
        self._client = client

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Return whether the model uses the Responses API.

        Args:
            model: OpenAI model name.

        Returns:
            ``True`` if the model is a reasoning model.

        """
        return model.startswith(("o1", "o3", "o4", "gpt-5"))

    def _generate(
        self,
        *,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate text from OpenAI.

        Args:
            prompt: Input prompt.
            model: OpenAI model.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If the API returns an empty response.

        """
        try:
            if self._is_reasoning_model(model):
                reasoning_response = self._client.responses.create(
                    model=model,
                    input=prompt,
                    max_output_tokens=max_tokens,
                    reasoning={"effort": "low"},
                    temperature=temperature,
                )
                result_text = str(reasoning_response.output_text).strip()
            else:
                completion_response = self._client.completions.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result_text = str(completion_response.choices[0].text).strip()

            if not result_text:
                error_message = (
                    "Received an empty response from the OpenAI API. "
                    "This could be due to safety guardrails, network "
                    "filtering (anti-filter), or provider-side anomalies."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except RuntimeError:
            raise

        except Exception as exc:
            error_msg = str(exc).lower()

            if "temperature" in error_msg and (
                "support" in error_msg or "value" in error_msg or "allowed" in error_msg
            ):
                logger.warning(
                    "Dynamic fix applied: temperature=%f is not supported for model "
                    "'%s'. Retrying automatically with default temperature (1.0).",
                    temperature,
                    model,
                )
                if temperature != 1.0:
                    return self._generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=1.0,
                    )

            if (
                "max_output_tokens" in error_msg
                and "integer below minimum value" in error_msg
            ):
                match = re.search(r"expected a value >= (\d+)", error_msg)

                if match:
                    required_min = int(match.group(1))
                    logger.warning(
                        "Dynamic fix applied: max_tokens=%d is too low for model '%s'. "
                        "Retrying automatically with required minimum: %d.",
                        max_tokens,
                        model,
                        required_min,
                    )

                    return self._generate(
                        prompt=prompt,
                        model=model,
                        max_tokens=required_min,
                        temperature=temperature,
                    )

            logger.error("OpenAI request failed: %s", exc)
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: OpenAI model name.
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
        model_name: str,
        size: str,
        quality: str,
        n: int,
        input_image_path: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Execute core generation or editing logic for OpenAI-compatible APIs.

        Args:
            prompt: The text prompt provided by the user.
            output_dir: Directory to save the final generated output.
            model_name: Target OpenAI model identifier.
            size: Desired dimensions of the output image.
            quality: Quality level of the generated image.
            n: Number of images requested.
            input_image_path: Path to base image if editing, None otherwise.
            **kwargs: Additional parameters (e.g., `extra_body`, `response_format`).

        Returns:
            A tuple containing a boolean success flag and the file path.

        """
        # Default to b64_json to avoid network overhead, unless overridden by user
        kwargs.setdefault("response_format", "b64_json")

        # Extract flag for providers (like ByteDance) that use /generations for edits
        use_generate_for_edit = kwargs.pop("use_generate_for_edit", False)

        generated_img: Image.Image | None = None
        gen_img_flag = True

        try:
            # Standard OpenAI Edit Request
            if input_image_path is not None and not use_generate_for_edit:
                with open(input_image_path, "rb") as img_file:
                    response = self._client.images.edit(
                        model=model_name,
                        image=img_file,
                        prompt=prompt,
                        size=size,
                        quality=quality,
                        n=n,
                        **kwargs,
                    )
            else:
                # Handle ByteDance-style image-to-image via generate endpoint
                if input_image_path is not None and use_generate_for_edit:
                    with open(input_image_path, "rb") as img_file:
                        b64_str = base64.b64encode(img_file.read()).decode("utf-8")

                    extra_body = kwargs.get("extra_body", {})
                    if "image" not in extra_body:
                        extra_body["image"] = b64_str
                    kwargs["extra_body"] = extra_body

                # Standard or Alternative Generation Request
                response = self._client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=n,
                    **kwargs,
                )

            # Support both b64_json and url formats seamlessly
            if response.data:
                img_data_obj = response.data[0]
                if hasattr(img_data_obj, "b64_json") and img_data_obj.b64_json:
                    img_bytes = BytesIO(base64.b64decode(img_data_obj.b64_json))
                    generated_img = Image.open(img_bytes)
                elif hasattr(img_data_obj, "url") and img_data_obj.url:
                    req_response = requests.get(img_data_obj.url, timeout=30)
                    req_response.raise_for_status()
                    generated_img = Image.open(BytesIO(req_response.content))
                else:
                    raise RuntimeError(
                        "No valid image data (b64_json or url) found in response."
                    )
            else:
                raise RuntimeError("Empty image data returned from provider.")

        except Exception as e:
            logger.exception(f"Error during OpenAI-compatible API call: {e}")

        # Fallback to placeholder if everything failed
        if generated_img is None:
            gen_img_flag = False
            logger.debug(
                f"Failed to generate image for prompt: '{prompt}'. "
                "Creating placeholder."
            )
            fallback_img = self._create_placeholder_image(
                prompt=prompt, output_dir=output_dir, save=False
            )
            if isinstance(fallback_img, Image.Image):
                generated_img = fallback_img

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = "openai_edited" if input_image_path else "openai_generated"
        filename = f"{prefix}_{timestamp}.png"
        gen_path = os.path.join(output_dir, filename)

        if isinstance(generated_img, Image.Image):
            generated_img.save(gen_path)

        logger.debug(
            f'------------------- "{gen_path}" generated! '
            f"(Success: {gen_img_flag}) -------------------"
        )

        return gen_img_flag, gen_path

    def generate_image(
        self,
        prompt: str,
        output_dir: str,
        *,
        model_name: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image using the OpenAI API based on a text prompt.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            model_name: OpenAI model name for image generation.
            size: Sampling size specification for the output.
            quality: Quality configuration ("low", "medium", "high", "auto").
            n: Number of output images to generate.
            **kwargs: Extra parameters (e.g., `output_format`, `extra_body`).

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            model_name=model_name,
            size=size,
            quality=quality,
            n=n,
            **kwargs,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        *,
        model_name: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "auto",
        n: int = 1,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an edited image using OpenAI API and an input image.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            model_name: OpenAI model name for image editing.
            size: Sampling size specification for the output.
            quality: Quality configuration ("low", "medium", "high", "auto").
            n: Number of output images to generate.
            **kwargs: Extra parameters. Pass `use_generate_for_edit=True` for
                providers like ByteDance that use the /generations endpoint
                for image-to-image editing.

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
            model_name=model_name,
            size=size,
            quality=quality,
            n=n,
            input_image_path=image_path,
            **kwargs,
        )
