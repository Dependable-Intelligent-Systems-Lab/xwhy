"""Gemini provider implementation."""

import base64
import json
import os
import time
from io import BytesIO

from google.genai import types
from PIL import Image

from xwhy.logger import logger
from xwhy.providers.base import BaseProvider


class GeminiProvider(BaseProvider):
    """Gemini implementation of the provider interface."""

    def __init__(self, client: object) -> None:
        """Initialize the provider.

        Args:
            client: Configured Gemini client (typically the generativeai module).

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
        """Generate text from Gemini.

        Args:
            prompt: Input prompt.
            model: Gemini model name.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            Generated text.

        Raises:
            RuntimeError: If the API returns an empty response or is
                          blocked by safety filters.

        """
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=types.Part.from_text(text=prompt),
                config=types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )

            try:
                result_text = str(response.text).strip()
            except ValueError as val_err:
                # Gemini throws ValueError on .text access if the response was
                # blocked by safety filters.
                error_message = (
                    f"Gemini generation was blocked (likely due to safety filters) "
                    f"for model '{model}'. No content returned."
                )
                logger.error(error_message)
                raise RuntimeError(error_message) from val_err

            if not result_text:
                error_message = (
                    "Received an empty response from the Gemini API. "
                    "This could be due to network filtering (anti-filter) "
                    "or provider-side anomalies."
                )
                logger.error(error_message)
                raise RuntimeError(error_message)

            return result_text

        except RuntimeError:
            raise

        except Exception as exc:
            logger.error("Gemini request failed: %s", exc)
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

    def answer(
        self,
        prompt: str,
        *,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> str:
        """Generate a natural-language answer.

        Args:
            prompt: Input prompt.
            model: Gemini model name.
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
        contents: list[types.Content],
        model_name: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_output_tokens: int,
        stream: bool,
        seed: int | None,
        default_mime_type: str = "image/png",
    ) -> tuple[bool, str]:
        """Execute the core generation logic for image requests.

        Args:
            prompt: The text prompt provided by the user.
            output_dir: Directory to save the final generated output.
            contents: Formatted content list to send to the API.
            model_name: Target Gemini model identifier.
            temperature: Sampling temperature for generation.
            top_p: Top-p sampling configuration.
            top_k: Top-k sampling configuration.
            max_output_tokens: Maximum output tokens allowed.
            stream: Boolean flag to enable or disable streaming.
            seed: Seed for deterministic generation.
            default_mime_type: Fallback MIME type.

        Returns:
            A tuple containing a boolean success flag and the file path.

        """
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            response_modalities=["image", "text"],
            response_mime_type="text/plain",
            seed=seed,
        )

        generated_img: Image.Image | None = None
        gen_img_flag = True
        final_mime = default_mime_type

        try:
            if stream:
                response_iter = self._client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                for chunk in response_iter:
                    for part in chunk.parts:
                        if part.inline_data is not None:
                            img_data = BytesIO(part.inline_data.data)
                            generated_img = Image.open(img_data)
                            final_mime = part.inline_data.mime_type
                            break
            else:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=generate_content_config,
                )
                for part in response.parts:
                    if part.inline_data is not None:
                        img_data = BytesIO(part.inline_data.data)
                        generated_img = Image.open(img_data)
                        final_mime = part.inline_data.mime_type
                        break
        except Exception as e:
            logger.exception(f"Error during API call: {e}")

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
            final_mime = "image/png"

        # Determine file extension based on MIME type
        ext = ".jpg" if final_mime == "image/jpeg" else ".png"

        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        filename = f"gemini_generated_{timestamp}{ext}"
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
        model_name: str = "gemini-2.5-flash-image",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        stream: bool = True,
        seed: int | None = None,
    ) -> tuple[bool, str]:
        """Generate an image using the Gemini API based on a text prompt.

        Args:
            prompt: Text description of the desired image.
            output_dir: Directory where the image will be stored.
            model_name: Gemini model name for image generation.
            temperature: Sampling temperature configuration.
            top_p: Top-p sampling configuration.
            top_k: Top-k sampling configuration.
            max_output_tokens: Token limit for the response.
            stream: Boolean to indicate if stream mode should be used.
            seed: Random seed to ensure deterministic output.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        """
        text_part = types.Part.from_text(text=prompt)
        contents = [types.Content(role="user", parts=[text_part])]

        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            contents=contents,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            stream=stream,
            seed=seed,
        )

    def edit_image(
        self,
        prompt: str,
        image_path: str,
        output_dir: str,
        *,
        model_name: str = "gemini-2.5-flash-image",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        stream: bool = True,
        seed: int | None = None,
    ) -> tuple[bool, str]:
        """Generate an edited image using the Gemini API and an input image.

        Args:
            prompt: Text instructions for the image editing process.
            image_path: Path to the source image file.
            output_dir: Directory where the edited image will be saved.
            model_name: Gemini model name for image editing.
            temperature: Sampling temperature configuration.
            top_p: Top-p sampling configuration.
            top_k: Top-k sampling configuration.
            max_output_tokens: Maximum output tokens allowed.
            stream: Boolean to indicate if stream mode should be used.
            seed: Optional integer seed for deterministic output.

        Returns:
            A tuple of a boolean success flag and the generated file path.

        Raises:
            FileNotFoundError: If the provided input image is not found.

        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        if image_path.lower().endswith((".jpg", ".jpeg")):
            mime_type = "image/jpeg"
        else:
            mime_type = "image/png"

        text_part = types.Part.from_text(text=prompt)
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        contents = [types.Content(role="user", parts=[image_part, text_part])]

        return self._execute_image_request(
            prompt=prompt,
            output_dir=output_dir,
            contents=contents,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            stream=stream,
            seed=seed,
            default_mime_type=mime_type,
        )

    def submit_image_batch(
        self,
        image_path: str,
        text_list: list[str],
        *,
        model_name: str = "gemini-2.5-flash-image",
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        seed: int | None = None,
        response_mime_type: str = "text/plain",
    ) -> str:
        """Submit a batch image editing job to the Gemini API.

        Args:
            image_path: Path to the local base image file.
            text_list: A list of string prompts to apply to the image.
            model_name: Name of the Gemini multimodal batch model.
            temperature: Sampling temperature for generation.
            top_p: Top-p sampling constraint.
            top_k: Top-k sampling constraint.
            max_output_tokens: Maximum tokens for the generated outputs.
            seed: Random seed for deterministic batch outputs.
            response_mime_type: The expected MIME type from the API.

        Returns:
            The unique string name of the created batch job.

        """
        logger.debug(f"Uploading image file: {image_path}")
        image_file = self._client.files.upload(file=image_path)
        logger.debug(
            f"Uploaded image file: {image_file.name} (MIME: {image_file.mime_type})"
        )

        requests_data = []
        for ix, text in enumerate(text_list):
            custom_id = f"request_{ix}_image"

            gen_config_dict = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
                "response_mime_type": response_mime_type,
            }
            if seed is not None:
                gen_config_dict["seed"] = seed

            requests_data.append(
                {
                    "custom_id": custom_id,
                    "request": {
                        "contents": [
                            {
                                "parts": [
                                    {"text": text},
                                    {
                                        "file_data": {
                                            "file_uri": image_file.uri,
                                            "mime_type": image_file.mime_type,
                                        }
                                    },
                                ]
                            }
                        ],
                        "generation_config": gen_config_dict,
                    },
                }
            )

        json_file_path = "batch_image_gen_requests.json"

        logger.debug(f"Creating JSONL file: {json_file_path}")
        with open(json_file_path, "w") as f:
            for req in requests_data:
                f.write(json.dumps(req) + "\n")

        logger.debug(f"Uploading JSONL file: {json_file_path}")
        batch_input_file = self._client.files.upload(file=json_file_path)
        logger.debug(f"Uploaded JSONL file: {batch_input_file.name}")

        logger.debug("Creating batch job...")
        batch_multimodal_job = self._client.batches.create(
            model=model_name,
            src=batch_input_file.name,
            config={"display_name": "xwhy-batch-image-job"},
        )
        logger.debug(f"Created batch job: {batch_multimodal_job.name}")

        return str(batch_multimodal_job.name)

    def retrieve_image_batch(
        self,
        job_name: str,
        text_list: list[str],
        output_dir: str = "outputs",
        *,
        model_name: str = "gemini-2.5-flash-image",
    ) -> list[tuple[bool, str]]:
        """Poll for completion of a Gemini batch job and save the results.

        Args:
            job_name: The batch job identifier returned by the API.
            text_list: The original list of prompts submitted to the job.
            output_dir: Directory where the output images will be saved.
            model_name: The Gemini model associated with the batch job.

        Returns:
            A list of tuples, each containing a success flag and file path.

        """
        logger.debug(f"Polling status for job: {job_name}")

        while True:
            batch_multimodal_job = self._client.batches.get(name=job_name)
            state = batch_multimodal_job.state.name

            if state in [
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            ]:
                logger.debug(f"Job finished with state: {state}")
                break

            time.sleep(30)

        processed_results: dict[str, tuple[bool, str | None]] = {}
        os.makedirs(output_dir, exist_ok=True)

        if batch_multimodal_job.state.name == "JOB_STATE_SUCCEEDED":
            result_file_name = batch_multimodal_job.dest.file_name
            logger.debug(f"Results available in file: {result_file_name}")

            file_content_bytes = self._client.files.download(file=result_file_name)
            file_content = file_content_bytes.decode("utf-8")

            for line in file_content.splitlines():
                if not line:
                    continue

                try:
                    parsed_response = json.loads(line)
                    custom_id = parsed_response.get("custom_id") or parsed_response.get(
                        "key"
                    )

                    found_image = False
                    if (
                        "response" in parsed_response
                        and "candidates" in parsed_response["response"]
                        and parsed_response["response"]["candidates"]
                    ):
                        candidates = parsed_response["response"]["candidates"][0]
                        if "content" in candidates and "parts" in candidates["content"]:
                            for part in candidates["content"]["parts"]:
                                if "inlineData" in part:
                                    mime = part["inlineData"]["mimeType"]
                                    data_bytes = part["inlineData"]["data"]
                                    data = base64.b64decode(data_bytes)

                                    ext = ".png" if "png" in mime else ".jpg"
                                    timestamp = int(time.time() * 1000)
                                    filename = f"{custom_id}_{timestamp}{ext}"
                                    save_path = os.path.join(output_dir, filename)

                                    with open(save_path, "wb") as img_f:
                                        img_f.write(data)

                                    processed_results[custom_id] = (True, save_path)
                                    found_image = True
                                    break

                    if not found_image:
                        # API returned a candidate but no inlineData (likely
                        # text refusal or filter)
                        logger.warning(f"No image found in response for {custom_id}")
                        processed_results[custom_id] = (False, None)
                except Exception as e:
                    logger.error(f"Error parsing line: {e}")
                    custom_id_from_error = (
                        line.split('"custom_id": "')[1].split('"')[0]
                        if '"custom_id":' in line
                        else f"unknown_error_{int(time.time())}"
                    )
                    processed_results[custom_id_from_error] = (False, None)
        else:
            logger.warning(
                "Job failed or was cancelled. Final state: %s",
                batch_multimodal_job.state.name,
            )

        final_output_list: list[tuple[bool, str]] = []
        for ix, text in enumerate(text_list):
            custom_id = f"request_{ix}_image"

            if custom_id in processed_results:
                flag, path = processed_results[custom_id]
                if path is None:  # If path wasn't set (e.g., API refusal),
                    placeholder = self._create_placeholder_image(
                        prompt=text,
                        output_dir=output_dir,
                        filename_prefix=custom_id,
                        save=True,
                    )
                    if isinstance(placeholder, str):
                        final_output_list.append((False, placeholder))
                else:
                    final_output_list.append((flag, path))
            else:
                logger.warning(
                    f"Generating placeholder for missing result: {custom_id}"
                )
                placeholder = self._create_placeholder_image(
                    prompt=text,
                    output_dir=output_dir,
                    filename_prefix=custom_id,
                    save=True,
                )
                if isinstance(placeholder, str):
                    final_output_list.append((False, placeholder))

        return final_output_list
