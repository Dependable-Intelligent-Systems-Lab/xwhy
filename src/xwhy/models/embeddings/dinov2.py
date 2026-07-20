"""DINOv2 embedding implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from xwhy.config.settings import Settings
from xwhy.logger import logger
from xwhy.models.embeddings.base import BaseEmbedding


class Dinov2Embedding(BaseEmbedding):
    """DINOv2 image embedding backend.

    This class loads the Facebook DINOv2 model using the Hugging Face
    transformers library, caching the model weights locally. It provides
    capabilities to encode both image paths (to satisfy BaseEmbedding)
    and raw PIL Images directly.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        model_name: str = "facebook/dinov2-base",
        seed: int = 42,
        device: torch.device | str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the DINOv2 embedding backend.

        Args:
            settings: Global application settings for cache directories.
            model_name: The Hugging Face model identifier.
            seed: Random seed for reproducible inference.
            device: Target computation device (e.g., 'cpu', 'cuda').
            **kwargs: Additional arbitrary keyword arguments.

        """
        self._settings = settings
        self._model_name = model_name
        self._seed = seed

        self._rng = np.random.default_rng(self._seed)

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

        # Transformers objects are typed as Any because they generate dynamically
        self._processor: Any | None = None
        self._model: Any | None = None

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Read-only property to access the Hugging Face DINOv2 model.

        Raises:
            RuntimeError: If the model has not been loaded yet.

        Returns:
            The loaded Hugging Face model object.

        """
        if self._model is None:
            raise RuntimeError(
                f"Model '{self._model_name}' is not loaded. Call .load() first."
            )
        return self._model

    @property
    def processor(self) -> Any:  # noqa: ANN401
        """Read-only property to access the Hugging Face image processor.

        Raises:
            RuntimeError: If the processor has not been loaded yet.

        Returns:
            The loaded Hugging Face processor object.

        """
        if self._processor is None:
            raise RuntimeError(
                f"Processor for '{self._model_name}' is not loaded. Call .load() first."
            )
        return self._processor

    def __call__(
        self, inputs: Image.Image | torch.Tensor | dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Execute the forward pass to extract and pool image embeddings.

        Args:
            inputs: Can be a PIL Image object, a raw PyTorch tensor, or a
                dictionary of tensors generated directly by the AutoImageProcessor.

        Raises:
            RuntimeError: If the model or processor has not been loaded yet.

        Returns:
            A numpy array representing the mean-pooled embedding vector.

        """
        if self._processor is None or self._model is None:
            processor, model = self.load()
        else:
            processor = self._processor
            model = self._model

        # Handle different input types dynamically and robustly
        if isinstance(inputs, Image.Image):
            processed_inputs = processor(
                images=inputs,
                return_tensors="pt",
                do_rescale=False,
            ).to(self._device)
            with torch.no_grad():
                outputs = model(**processed_inputs)
        elif isinstance(inputs, dict):
            processed_inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**processed_inputs)
        else:
            tensor_inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(tensor_inputs)

        # Average pooling over the token sequence
        emb = outputs.last_hidden_state.mean(dim=1)
        emb_array: np.ndarray = emb.squeeze().cpu().numpy()

        return emb_array

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        logger.debug(f"Setting seeds to {self._seed} for reproducibility...")
        self._rng = np.random.default_rng(self._seed)
        torch.manual_seed(self._seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _get_cache_dir(self) -> Path:
        """Retrieve the caching directory from application settings.

        Returns:
            The Path object pointing to the embedding cache directory.

        """
        cache_dir = self._settings.embedding_cache_dir
        if not cache_dir:
            cache_dir = Path.home() / ".cache" / "xwhy" / "embeddings"
        return cache_dir

    def load(self) -> tuple[Any, Any]:
        """Load the DINOv2 processor and model into memory.

        This method leverages Hugging Face's built-in caching mechanism,
        placing the models exactly where the settings specify.

        Returns:
            A tuple containing the initialized (processor, model).

        """
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        self._set_seed()
        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Setup {self._model_name} processor & model...")

        self._processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
            self._model_name,
            backend="torchvision",
            cache_dir=str(cache_dir),
        )

        self._model = AutoModel.from_pretrained(
            self._model_name,
            cache_dir=str(cache_dir),
        ).to(self._device)

        self._model.eval()

        return self._processor, self._model

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Extract an image embedding using the DINOv2 model.

        This is a specialized method optimized for the Explainer pipeline
        where the image is already preprocessed and loaded in memory.

        Args:
            image: The PIL Image object to be encoded.

        Returns:
            A numpy array representing the extracted embedding vector.

        """
        return self.__call__(image)

    def encode(self, text: str) -> list[float]:
        """Encode an image into a list of floats from a file path.

        This method fulfills the `BaseEmbedding` contract. It loads the image
        from the given path and delegates the core logic to `encode_image`.

        Args:
            text: A string representing the absolute or relative path
                to the image file.

        Returns:
            A flat list of floats representing the image embedding.

        Raises:
            FileNotFoundError: If the provided image path does not exist.
            RuntimeError: If encoding fails due to model/tensor issues.

        """
        image_path = Path(text)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        try:
            with Image.open(image_path) as img:
                # Convert to RGB to ensure 3 channels (handles RGBA/Grayscale safely)
                image = img.convert("RGB")

            embedding_array = self.encode_image(image)
            return embedding_array.tolist()  # type: ignore[no-any-return]

        except Exception as err:
            raise RuntimeError(f"Failed to encode image '{image_path}': {err}") from err
