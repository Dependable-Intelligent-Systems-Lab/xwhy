"""Torchvision models classification implementation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torchvision.models import (
    Inception_V3_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    inception_v3,
    mobilenet_v3_large,
    resnet18,
    resnet50,
    vit_b_16,
)

from xwhy.config.settings import Settings
from xwhy.logger import logger
from xwhy.models.classification.base import BaseClassification


class TorchvisionClassification(BaseClassification):
    """Classification backend for standard torchvision models.

    Supports dynamic loading of various models like Inception V3, ResNet,
    MobileNet, and Vision Transformers.
    """

    # Map model names to their respective initialization functions and default weights
    _MODEL_REGISTRY: ClassVar[Mapping[str, tuple[Callable[..., Any], Any]]] = {
        "inception_v3": (inception_v3, Inception_V3_Weights.DEFAULT),
        "resnet18": (resnet18, ResNet18_Weights.DEFAULT),
        "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
        "mobilenet_v3": (mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT),
        "vit_base": (vit_b_16, ViT_B_16_Weights.DEFAULT),
    }

    def __init__(
        self,
        *,
        settings: Settings,
        model_name: str = "inception_v3",
        seed: int = 42,
        device: torch.device | str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Torchvision classification backend.

        Args:
            settings: Global application settings for cache directories.
            model_name: The torchvision model identifier.
            seed: Random seed for reproducible inference.
            device: Target computation device.
            **kwargs: Additional arbitrary keyword arguments.

        """
        self._settings = settings
        self._model_name = model_name
        self._seed = seed

        if self._model_name not in self._MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model '{self._model_name}'. "
                f"Available models: {list(self._MODEL_REGISTRY.keys())}"
            )

        self._rng = np.random.default_rng(self._seed)

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

        # Torchvision resources
        self._weights: Any | None = None
        self._model: Any | None = None
        self._preprocess: Any | None = None

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Read-only property to access the underlying raw model.

        Raises:
            RuntimeError: If the model has not been loaded yet.

        Returns:
            The loaded torchvision model.

        """
        if self._model is None:
            raise RuntimeError(
                f"Model '{self._model_name}' is not loaded. Call .load() first."
            )
        return self._model

    @property
    def weights(self) -> Any:  # noqa: ANN401
        """Read-only property to access the underlying raw weights.

        Raises:
            RuntimeError: If the weights has not been loaded yet.

        Returns:
            The loaded torchvision weights.

        """
        if self._weights is None:
            raise RuntimeError("Weights is not loaded. Call .load() first.")
        return self._weights

    @property
    def preprocess_fn(self) -> Callable[..., Any] | None:
        """Read-only property to access the preprocessing transform function.

        Returns:
            The torchvision transform function configured for the model.

        """
        if self._model is None:
            raise RuntimeError("Preprocess is not loaded. Call .load() first.")

        return self._preprocess

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass of the classification model.

        Args:
            inputs: A PyTorch tensor containing the preprocessed images.
                Expected shape is typically (B, C, H, W).

        Raises:
            RuntimeError: If the model has not been loaded yet.

        Returns:
            A PyTorch tensor containing the classification logits.

        """
        if self._model is None:
            raise RuntimeError(
                f"Model '{self._model_name}' is not loaded. Call .load() first."
            )

        inputs = inputs.to(self._device)

        return self._model(inputs)  # type: ignore[no-any-return]

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
        """Retrieve the caching directory from application settings."""
        cache_dir = getattr(self._settings, "classification_cache_dir", None)
        if not cache_dir:
            cache_dir = Path.home() / ".cache" / "xwhy" / "classification"
        return cache_dir

    def load(self) -> tuple[Any, Any]:
        """Load the specified model and preprocessing transforms into memory.

        Returns:
            A tuple containing the initialized (preprocess_transforms, model).

        """
        if self._model is not None and self._preprocess is not None:
            return self._preprocess, self._model

        self._set_seed()
        cache_dir = self._get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Point torch cache to our internal directory
        os.environ["TORCH_HOME"] = str(cache_dir)

        logger.debug(f"Setup {self._model_name} preprocessing & model...")

        # Dynamically fetch the model builder and weights from the registry
        model_builder, self._weights = self._MODEL_REGISTRY[self._model_name]

        # Load model and explicitly set to eval mode
        self._model = model_builder(weights=self._weights).to(self._device)
        self._model.eval()

        # Extract the correct preprocessing pipeline for this specific model
        self._preprocess = self._weights.transforms()

        return self._preprocess, self._model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run classification on preprocessed tensor inputs.

        Args:
            inputs: A preprocessed image tensor of shape (B, C, H, W).

        Returns:
            A tensor of logits (B, num_classes).

        """
        if self._model is None:
            _, model = self.load()
        else:
            model = self._model

        inputs = inputs.to(self._device)

        with torch.no_grad():
            logits = model(inputs)

        return logits  # type: ignore[no-any-return]
