"""Result data structures for explanations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots.visualisation import Explanation


@dataclass
class BaseXWhyResult(ABC):
    """Abstract base container for shared explanation results.

    Future implementations like ImageXWhyResult or TabularXWhyResult
    will inherit from this class to ensure API consistency.
    """

    coefficients: np.ndarray
    metrics: RegressionMetricResult
    raw_data: dict[str, Any] = field(default_factory=dict)
    base_values: float | np.ndarray = 0.0

    @property
    @abstractmethod
    def feature_names(self) -> Sequence[str] | np.ndarray | None:
        """Feature names corresponding to the explanation attributions."""

    @property
    @abstractmethod
    def data(self) -> np.ndarray | Sequence[Any] | None:
        """The underlying raw data instance associated with the explanation."""

    def to_explanation(self) -> Explanation:
        """Convert the XWhy result into an :class:`Explanation`.

        Returns:
            Explanation: A fully initialized explanation container.

        """
        return Explanation(
            values=self.coefficients,
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
        )

    def to_shap(self) -> Explanation:
        """Convert the result into an explanation object.

        Deprecated alias of :meth:`to_explanation`, kept so notebooks written
        against the old SHAP-backed API keep working. XWhy no longer depends
        on ``shap``, so this returns an :class:`Explanation`, not a
        ``shap.Explanation``.

        Returns:
            Explanation: A fully initialized explanation container.

        """
        return self.to_explanation()

    def plot(
        self, save_path: str | Path | None = None, show: bool = True
    ) -> str | None:
        """Plot the actual vs predicted fidelity for the surrogate model.

        Args:
            save_path: Optional path to save the generated plot image.
            show: If True, displays the plot interactively.

        Returns:
            str | None: The path to the saved image, or None if not saved.

        Raises:
            KeyError: If required data arrays are missing from raw_data.

        """
        from xwhy.plots.metrics import plot_fidelity

        required_keys = ["y_target", "y_pred", "weights"]
        for key in required_keys:
            if key not in self.raw_data:
                raise KeyError(
                    f"'{key}' must be present in raw_data to generate"
                    " the fidelity plot."
                )

        return plot_fidelity(
            metrics=self.metrics,
            y_target=self.raw_data["y_target"],
            y_pred=self.raw_data["y_pred"],
            weights=self.raw_data["weights"],
            save_path=save_path,
            show=show,
        )


@dataclass
class TextXWhyResult(BaseXWhyResult):
    """Container for text-specific explanation results."""

    original_output: str = ""
    words: Sequence[str] = field(default_factory=list)

    @property
    def feature_names(self) -> Sequence[str]:
        """Sequence of feature names corresponding to the text tokens."""
        return self.words

    @property
    def data(self) -> np.ndarray:
        """The underlying raw data tokens as a numpy array."""
        return np.array(self.words)


@dataclass
class ImageClassificationXWhyResult(BaseXWhyResult):
    """Container for image classification explanation results."""

    original_image: np.ndarray = field(default_factory=lambda: np.zeros(0))
    superpixels: np.ndarray = field(default_factory=lambda: np.zeros(0))
    top_features: np.ndarray = field(default_factory=lambda: np.zeros(0))
    coverage: float = 0.0
    weighted_coverage: float = 0.0

    @property
    def feature_names(self) -> Sequence[str]:
        """Sequence of feature names corresponding to superpixels."""
        return [f"Superpixel {i}" for i in range(len(self.coefficients))]

    @property
    def data(self) -> np.ndarray:
        """The underlying original image as a numpy array."""
        return self.original_image

    def to_explanation(self) -> Explanation:
        """Convert the XWhy image result into an :class:`Explanation`."""
        from xwhy.plots.image import create_image_heat_mask

        if self.superpixels.size > 0:
            heat_mask = create_image_heat_mask(self.superpixels, self.coefficients)
            data_arr = self.original_image

            if data_arr.ndim == 3:
                data_arr = np.expand_dims(data_arr, axis=0)  # (1, H, W, C)
            elif data_arr.ndim == 2:
                data_arr = np.expand_dims(data_arr, axis=0)  # (1, H, W)

            if data_arr.ndim == 4:
                # Add batch and channel dimention (1, H, W, 1)
                values_arr = np.expand_dims(heat_mask, axis=(0, -1))
            else:
                values_arr = np.expand_dims(heat_mask, axis=0)

            final_shap_values = values_arr

        else:
            data_arr = self.original_image
            final_shap_values = np.asarray(self.coefficients)

        return Explanation(
            values=final_shap_values,
            base_values=self.base_values,
            data=data_arr,
            feature_names=self.feature_names,
        )


@dataclass
class TabularXWhyResult(BaseXWhyResult):
    """Container for tabular-specific explanation results."""

    feature_list: Sequence[str] = field(default_factory=list)
    instance: np.ndarray | None = None

    @property
    def feature_names(self) -> Sequence[str] | np.ndarray:
        """Sequence of feature names corresponding to the tabular columns."""
        if not self.feature_list and self.instance is not None:
            return [f"Feature_{i}" for i in range(len(self.instance))]
        return self.feature_list

    @property
    def data(self) -> np.ndarray | Sequence[Any] | None:
        """The underlying raw data instance (the explained sample)."""
        return self.instance


@dataclass
class ImageGenerationAndEditingXWhyResult(BaseXWhyResult):
    """Container for image generation and editing explanation results.

    Attributes:
        words: Sequence of prompt words/tokens corresponding to the
            explanation features.
        instance: The underlying input instance, path, or text prompt data.

    """

    words: Sequence[str] = field(default_factory=list)
    instance: np.ndarray | str | None = None

    @property
    def feature_names(self) -> Sequence[str]:
        """Return sequence of feature names corresponding to prompt words."""
        return self.words

    @property
    def data(self) -> np.ndarray:
        """Return the underlying token array required for explanation visualization."""
        if self.words:
            return np.array(self.words)
        if isinstance(self.instance, str):
            return np.array(self.instance.split())
        return (
            np.array([str(self.instance)])
            if self.instance is not None
            else np.array([])
        )
