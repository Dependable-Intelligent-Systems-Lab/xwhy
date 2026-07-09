"""Result data structures for explanations."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shap

from xwhy.metrics.regression import RegressionMetricResult


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

    def to_shap(self) -> object:
        """Convert the XWhy result into a standard SHAP Explanation object.

        Returns:
            object: A fully initialized shap.Explanation instance.

        """
        return shap.Explanation(
            values=self.coefficients,
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
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
