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

from xwhy.core.types import BaseImageGenerationAndEditing
from xwhy.logger import logger
from xwhy.providers.base import BaseProvider
from xwhy.utils.image import image_to_base64


class OpenAIProvider(BaseImageGenerationAndEditing, BaseProvider):
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
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Generate text from OpenAI with built-in retries and error handling.

        Args:
            prompt: Input prompt.
            model: OpenAI model identifier.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            **kwargs: Extra parameters (supports 'max_retries' and 'delay').

        Returns:
            Generated text string.

        Raises:
            RuntimeError: If the API returns an empty response or fails
                after all retries.

        """
        max_retries: int = kwargs.get("max_retries", 7)
        delay_override: float | None = kwargs.get("delay")

        for retry_number in range(1, max_retries + 1):
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
                    "support" in error_msg
                    or "value" in error_msg
                    or "allowed" in error_msg
                ):
                    logger.warning(
                        "Dynamic fix applied: temperature=%f is not supported "
                        "for model '%s'. Retrying automatically with default "
                        "temperature (1.0).",
                        temperature,
                        model,
                    )
                    if temperature != 1.0:
                        return self._generate(
                            prompt=prompt,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=1.0,
                            **kwargs,
                        )

                if (
                    "max_output_tokens" in error_msg
                    and "integer below minimum value" in error_msg
                ):
                    match = re.search(r"expected a value >= (\d+)", error_msg)

                    if match:
                        required_min = int(match.group(1))
                        logger.warning(
                            "Dynamic fix applied: max_tokens=%d is too low "
                            "for model '%s'. Retrying automatically with "
                            "required minimum: %d.",
                            max_tokens,
                            model,
                            required_min,
                        )

                        return self._generate(
                            prompt=prompt,
                            model=model,
                            max_tokens=required_min,
                            temperature=temperature,
                            **kwargs,
                        )

                if retry_number == max_retries:
                    logger.error(
                        "OpenAI request failed after %d retries: %s",
                        max_retries,
                        exc,
                    )
                    raise RuntimeError(f"OpenAI request failed: {exc}") from exc

                delay: float = (
                    delay_override
                    if delay_override is not None
                    else min(2**retry_number, 30)
                )
                logger.warning(
                    "Retry %d/%d for OpenAI text generation. Waiting %s seconds...",
                    retry_number,
                    max_retries,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("OpenAI text generation failed after max retries.")

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gpt-3.5-turbo-instruct",
        max_tokens: int = 200,
        temperature: float = 0.0,
        **kwargs: Any,  # noqa: ANN401
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: OpenAI model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.
            **kwargs: Extra parameters (supports 'max_retries' and 'delay').

        Returns:
            Generated response text string.

        """
        return self._generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    # -------------------------------------------------------------------------
    # Image Generation & Editing Methods
    # -------------------------------------------------------------------------

    def _execute_image_request(
        self,
        prompt: str,
        output_dir: str,
        model_name: str,
        input_image_path: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Execute core generation or editing logic for OpenAI-compatible APIs.

        Args:
            prompt: The text prompt provided by the user.
            output_dir: Directory to save the final generated output.
            model_name: Target OpenAI model identifier.
            input_image_path: Path to base image if editing, None otherwise.
            **kwargs: Additional parameters (e.g., extra_body, response_format,
                max_retries, delay).

        Returns:
            A tuple containing a boolean success flag and the file path.

        Raises:
            RuntimeError: If image data is missing or empty.
            FileNotFoundError: If the input image path does not exist.

        """
        # Safely extract configuration flags to prevent them from reaching API
        provider_name: str = kwargs.pop("provider_name", "openai")
        output_format: str = kwargs.pop("output_format", "png")
        use_generate_for_edit: bool = kwargs.pop("use_generate_for_edit", False)
        use_image_data_uri: bool = kwargs.pop("use_image_data_uri", False)
        use_image_url: bool = kwargs.pop("use_image_url", False)
        max_retries: int = kwargs.pop("max_retries", 7)
        delay_override: float | None = kwargs.pop("delay", None)

        # Handle response_format logic cleanly
        if "response_format" in kwargs and kwargs["response_format"] is None:
            kwargs.pop("response_format")
        else:
            kwargs.setdefault("response_format", "b64_json")

        generated_img: Image.Image | None = None

        for retry_number in range(1, max_retries + 1):
            try:
                # Standard OpenAI Edit Request
                if input_image_path is not None and not use_generate_for_edit:
                    with open(input_image_path, "rb") as img_file:
                        response = self._client.images.edit(
                            model=model_name,
                            image=img_file,
                            prompt=prompt,
                            **kwargs,
                        )
                else:
                    # Handle ByteDance-style image-to-image via generate
                    if input_image_path is not None and use_generate_for_edit:
                        input_image_data_uri = image_to_base64(
                            image_path=input_image_path,
                            include_data_uri=use_image_data_uri,
                        )

                        extra_body = kwargs.pop("extra_body", {})

                        if use_image_url:
                            if "image_url" not in extra_body:
                                extra_body["image_url"] = input_image_data_uri
                        else:
                            if "image" not in extra_body:
                                extra_body["image"] = input_image_data_uri

                        kwargs["extra_body"] = extra_body

                    # Standard or Alternative Generation Request
                    response = self._client.images.generate(
                        model=model_name,
                        prompt=prompt,
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
                            "No valid image data (b64_json or url) found."
                        )
                else:
                    raise RuntimeError("Empty image data returned from provider.")

                if generated_img is not None:
                    break

            except Exception as e:
                logger.warning(
                    "Error during OpenAI API call on attempt %d/%d: %s",
                    retry_number,
                    max_retries,
                    e,
                )
                if retry_number == max_retries:
                    logger.exception("All image generation retries exhausted.")

            if generated_img is None and retry_number < max_retries:
                delay = (
                    delay_override
                    if delay_override is not None
                    else min(2**retry_number, 30)
                )
                logger.warning(
                    "Retry %d/%d for image generation. Waiting %s seconds...",
                    retry_number,
                    max_retries,
                    delay,
                )
                time.sleep(delay)

        gen_img_flag = True
        # Fallback to placeholder if everything failed after all retries
        if generated_img is None:
            gen_img_flag = False
            logger.debug(
                "Failed to generate image for prompt: '%s' after %d retries. "
                "Creating placeholder.",
                prompt,
                max_retries,
            )
            if hasattr(self, "_create_placeholder_image"):
                fallback_img = self._create_placeholder_image(
                    prompt=prompt, output_dir=output_dir, save=False
                )
                if isinstance(fallback_img, Image.Image):
                    generated_img = fallback_img

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        prefix = (
            f"{provider_name}_edited"
            if input_image_path
            else f"{provider_name}_generated"
        )
        filename = f"{prefix}_{timestamp}.{output_format}"
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
        model_name: str = "gpt-image-1",
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an image using the OpenAI API based on a text prompt.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            model_name: OpenAI model name for image generation.
            **kwargs: Extra parameters (e.g., output_format, extra_body,
                max_retries, delay).

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            model_name=model_name,
            **kwargs,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        *,
        model_name: str = "gpt-image-1",
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[bool, str]:
        """Generate an edited image using OpenAI API and an input image.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            model_name: OpenAI model name for image editing.
            **kwargs: Extra parameters (e.g., output_format, extra_body,
                use_generate_for_edit, max_retries, delay).

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
            input_image_path=image_path,
            **kwargs,
        )
