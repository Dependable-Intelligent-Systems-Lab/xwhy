"""Custom PyTorch classification model adapter implementation."""

from collections.abc import Callable
from contextlib import suppress
from typing import Any

import torch
from torchvision import transforms

from xwhy.models.classification.base import BaseClassification
from xwhy.utils.image import get_default_transform


class DynamicCategories:
    """A dictionary-like helper returning 'Class {idx}' if categories are absent."""

    def __init__(self, categories: list[str] | None = None) -> None:
        """Initialize dynamic categories with an optional list of names.

        Args:
            categories: Optional list of class name strings.

        """
        self.categories = categories

    def __getitem__(self, idx: int) -> str:
        """Retrieve class name by index or generate a fallback string.

        Args:
            idx: The class index.

        Returns:
            The class name string.

        """
        if self.categories and idx < len(self.categories):
            return self.categories[idx]

        return f"Class {idx}"


class MockWeights:
    """Mocks or wraps the weights structure to prevent AttributeErrors."""

    def __init__(
        self,
        weights_obj: Any = None,  # noqa: ANN401
        categories: list[str] | None = None,
    ) -> None:
        """Initialize weights, extracting metadata categories if available.

        Args:
            weights_obj: Original weights object if provided by the model.
            categories: Optional list of category names.

        """
        resolved_categories = categories

        # Automatically extract categories from weights_obj if not explicitly provided
        if resolved_categories is None and weights_obj is not None:
            if hasattr(weights_obj, "meta") and isinstance(weights_obj.meta, dict):
                if "categories" in weights_obj.meta:
                    resolved_categories = weights_obj.meta["categories"]
            elif hasattr(weights_obj, "categories"):
                resolved_categories = weights_obj.categories

        if weights_obj is not None:
            self._weights = weights_obj
            if not hasattr(self._weights, "meta") or self._weights.meta is None:
                self._weights.meta = {}
            if "categories" not in self._weights.meta:
                self._weights.meta["categories"] = DynamicCategories(
                    resolved_categories
                )
        else:
            self._weights = type("DummyWeights", (), {})()
            self._weights.meta = {"categories": DynamicCategories(resolved_categories)}

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute access to the underlying weights object."""
        return getattr(self._weights, name)


class PreprocessWrapper:
    """Wraps a transform function and extracts mean/std dynamically."""

    def __init__(self, transform_fn: Callable[..., Any], model: Any = None) -> None:  # noqa: ANN401
        """Initialize wrapper, checking model and transform attributes for stats.

        Args:
            transform_fn: The base transformation function.
            model: The custom PyTorch model instance to inspect.

        """
        self.transform_fn = transform_fn
        self.mean = [0.0, 0.0, 0.0]
        self.std = [1.0, 1.0, 1.0]

        # 1. Check if model explicitly exposes mean and std
        if model is not None:
            if hasattr(model, "mean") and model.mean is not None:
                self.mean = model.mean
            if hasattr(model, "std") and model.std is not None:
                self.std = model.std

        # 2. Check if the transform function itself exposes mean and std
        if hasattr(transform_fn, "mean") and transform_fn.mean is not None:
            self.mean = transform_fn.mean
        if hasattr(transform_fn, "std") and transform_fn.std is not None:
            self.std = transform_fn.std

        # 3. Parse transforms.Compose for Normalize layers if not found yet
        if hasattr(transform_fn, "transforms"):
            for t in transform_fn.transforms:
                if isinstance(t, transforms.Normalize):
                    self.mean = t.mean
                    self.std = t.std
                    break

    def __call__(self, x: Any) -> Any:  # noqa: ANN401
        """Apply the preprocessing transform to input data.

        Args:
            x: The input image data.

        Returns:
            The transformed output.

        """
        return self.transform_fn(x)


class CustomTorchClassification(BaseClassification):
    """Wrapper for user-defined custom PyTorch classification models."""

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess_fn: Callable[..., Any] | None = None,
        categories: list[str] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize the custom PyTorch classification model wrapper.

        Args:
            model: The user-defined PyTorch neural network module.
            preprocess_fn: Optional custom image transformation pipeline.
            categories: Optional list of readable class labels.
            device: Target computation device.

        """
        self._model = model

        # Check if the model exposes native weights or categories attributes
        native_weights = getattr(model, "weights", None)
        extracted_categories = categories
        if not extracted_categories and hasattr(model, "categories"):
            extracted_categories = model.categories  # type: ignore[assignment]

        self._weights = MockWeights(native_weights, extracted_categories)

        # Resolve preprocessing function hierarchically:
        # 1. Explicit argument -> 2. Model attribute ->
        # 3. Weights transforms -> 4. Default fallback
        resolved_preprocess = preprocess_fn
        if resolved_preprocess is None:
            if hasattr(model, "preprocess_fn") and model.preprocess_fn is not None:
                resolved_preprocess = model.preprocess_fn  # type: ignore[assignment]
            elif hasattr(model, "preprocess") and model.preprocess is not None:
                resolved_preprocess = model.preprocess  # type: ignore[assignment]
            elif hasattr(model, "transforms") and model.transforms is not None:
                resolved_preprocess = model.transforms  # type: ignore[assignment]
            elif hasattr(native_weights, "transforms") and callable(
                native_weights.transforms  # type: ignore[union-attr]
            ):
                with suppress(Exception):
                    resolved_preprocess = native_weights.transforms()  # type: ignore[union-attr]

        if resolved_preprocess is None:
            resolved_preprocess = get_default_transform()

        self._preprocess_fn = PreprocessWrapper(resolved_preprocess, model=model)

        if device is None:
            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self._device = torch.device(device) if isinstance(device, str) else device

    @property
    def weights(self) -> Any:  # noqa: ANN401
        """Get the model weights or mock structure."""
        return self._weights

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Get the underlying raw PyTorch model instance."""
        return self._model

    @property
    def preprocess_fn(self) -> Callable[..., Any] | None:
        """Get the resolved preprocessing transformation pipeline."""
        return self._preprocess_fn

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Execute the forward pass of the model.

        Args:
            inputs: Preprocessed input tensor.

        Returns:
            Output logits tensor.

        """
        inputs = inputs.to(self._device)
        return self._model(inputs)  # type: ignore[no-any-return]

    def load(self) -> tuple[Callable[..., Any] | None, Any]:
        """Prepare the custom model and transforms for inference.

        Returns:
            A tuple containing the preprocessing function and the model.

        """
        self._model = self._model.to(self._device)
        self._model.eval()
        return self._preprocess_fn, self._model

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run inference and return logits without tracking gradients.

        Args:
            inputs: Preprocessed input tensor.

        Returns:
            Model output logits tensor.

        """
        inputs = inputs.to(self._device)
        with torch.no_grad():
            logits = self._model(inputs)
        return logits  # type: ignore[no-any-return]
