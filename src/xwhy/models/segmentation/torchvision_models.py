"""Torchvision models segmentation implementation."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    FCN_ResNet50_Weights,
    LRASPP_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
    fcn_resnet50,
    lraspp_mobilenet_v3_large,
)

from xwhy.config.settings import Settings
from xwhy.logger import logger
from xwhy.models.segmentation.base import BaseSegmentation


class TorchvisionSegmentation(BaseSegmentation):
    """Segmentation backend for standard torchvision models.

    Supports dynamic loading of models like DeepLabV3+, FCN, and LRASPP.
    """

    # Map model names to their respective initialization functions and default weights
    _MODEL_REGISTRY: ClassVar[Mapping[str, tuple[Callable[..., Any], Any]]] = {
        "deeplabv3_resnet101": (
            deeplabv3_resnet101,
            DeepLabV3_ResNet101_Weights.DEFAULT,
        ),
        "deeplabv3_resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT),
        "deeplabv3_mobilenet_v3_large": (
            deeplabv3_mobilenet_v3_large,
            DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        ),
        "fcn_resnet50": (fcn_resnet50, FCN_ResNet50_Weights.DEFAULT),
        "lraspp_mobilenet_v3_large": (
            lraspp_mobilenet_v3_large,
            LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
        ),
    }

    def __init__(
        self,
        *,
        settings: Settings,
        model_name: str = "deeplabv3_resnet101",
        seed: int = 222,
        device: torch.device | str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the Torchvision segmentation backend.

        Args:
            settings: Global application settings for cache directories.
            model_name: The torchvision segmentation model identifier.
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
        self._class_names: list[str] = []

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names (categories) supported by the model."""
        if not self._class_names:
            logger.warning("Model not loaded yet. Loading now to fetch class names.")
            self.load()
        return self._class_names

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
        cache_dir = getattr(self._settings, "segmentation_cache_dir", None)
        if not cache_dir:
            cache_dir = Path.home() / ".cache" / "xwhy" / "segmentation"
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

        # Point torch cache to our internal directory to prevent hub subfolder mismatch
        os.environ["TORCH_HOME"] = str(cache_dir)

        logger.info(f"Loading {self._model_name} segmentation model...")

        # Dynamically fetch the model builder and weights from the registry
        model_builder, self._weights = self._MODEL_REGISTRY[self._model_name]

        # Load model and explicitly set to eval mode
        self._model = model_builder(weights=self._weights).to(self._device)
        self._model.eval()

        # Extract the correct preprocessing pipeline and classes
        self._preprocess = self._weights.transforms()

        # Meta dictionary safely fallback to empty list if categories are absent
        self._class_names = getattr(self._weights, "meta", {}).get("categories", [])
        logger.info(
            f"Segmentation model classes loaded: {len(self._class_names)} classes."
        )

        return self._preprocess, self._model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run segmentation on preprocessed tensor inputs.

        Args:
            inputs: A preprocessed image tensor of shape (B, C, H, W).

        Returns:
            A tensor of logits/masks (B, num_classes, H, W).

        """
        if self._model is None:
            _, model = self.load()
        else:
            model = self._model

        inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = model(inputs)

            # Torchvision segmentation models return an OrderedDict.
            # The main output is stored in the "out" key.
            logits = outputs["out"]

        return logits  # type: ignore[no-any-return]
