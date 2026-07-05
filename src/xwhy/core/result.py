"""Result data structures for explanations."""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from xwhy.metrics.regression import RegressionMetricResult
from xwhy.visualization.factory import TextVisualizerFactory
from xwhy.visualization.types import TextVisualizerType


@dataclass
class BaseXWhyResult(ABC):
    """Abstract base container for shared explanation results.

    Future implementations like ImageXWhyResult or TabularXWhyResult
    will inherit from this class to ensure API consistency for plots.
    """

    coefficients: np.ndarray
    metrics: RegressionMetricResult
    raw_data: dict[str, object] = field(default_factory=dict)


@dataclass
class TextXWhyResult(BaseXWhyResult):
    """Container for text-specific explanation results."""

    original_output: str = ""
    words: Sequence[str] = field(default_factory=list)

    def heatmap(
        self,
        title: str = "Text Heatmap",
        backend: TextVisualizerType = TextVisualizerType.NATIVE_HEATMAP,
        **kwargs: object,
    ) -> None:
        """Plot a text heatmap visualization.

        Args:
            title: Title of the plot.
            backend: Visualization backend to use (Native or SHAP).
            **kwargs: Additional backend-specific arguments.

        """
        visualizer = TextVisualizerFactory.create(method=backend)
        visualizer.plot(
            words=self.words,
            scores=self.coefficients,
            title=title,
            **kwargs,
        )
