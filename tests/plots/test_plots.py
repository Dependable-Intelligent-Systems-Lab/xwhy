"""Unit tests for plotting functions and singledispatch routing."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots import heatmap
from xwhy.visualization.types import TextVisualizerType


class DummyResult(BaseXWhyResult):
    """Dummy unsupported result type for testing dispatch fallback."""

    @property
    def feature_names(self) -> list[str]:
        """Mock feature names."""
        return []

    @property
    def data(self) -> np.ndarray:
        """Mock data instance."""
        return np.array([])


@pytest.fixture
def mock_metrics() -> RegressionMetricResult:
    """Fixture to provide a dummy metric result."""
    return RegressionMetricResult(
        weighted_mse=0.1,
        weighted_mae=0.1,
        weighted_r2=0.9,
        weighted_adj_r2=0.85,
        mean_loss=0.05,
        mean_l1_loss=0.1,
        mean_l2_loss=0.05,
        weighted_l1_norm=0.1,
        weighted_l2_norm=0.05,
    )


def test_heatmap_unsupported_type(mock_metrics: RegressionMetricResult) -> None:
    """Verify heatmap raises TypeError for unsupported BaseXWhyResult subclasses."""
    result = DummyResult(coefficients=np.array([]), metrics=mock_metrics)
    with pytest.raises(TypeError, match="not supported for DummyResult"):
        heatmap(result)


@patch("xwhy.plots.TextVisualizerFactory")
def test_text_heatmap_success(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap calls visualizer successfully with default kwargs logic."""
    mock_visualizer = MagicMock()
    mock_factory.create.return_value = mock_visualizer

    coeffs = np.array([0.1, 0.2])
    words = ["a", "b"]
    result = TextXWhyResult(coefficients=coeffs, metrics=mock_metrics, words=words)

    # Execute with an extra generic kwarg to ensure it passes through correctly
    heatmap(result, custom_kwarg=123)

    mock_factory.create.assert_called_once_with(
        method=TextVisualizerType.NATIVE_HEATMAP
    )
    mock_visualizer.plot.assert_called_once_with(
        words=words, scores=coeffs, title="Text Heatmap", custom_kwarg=123
    )


@patch("xwhy.plots.TextVisualizerFactory")
def test_text_heatmap_custom_args(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap handles explicit 'title' and backend type mapping."""
    mock_visualizer = MagicMock()
    mock_factory.create.return_value = mock_visualizer

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    heatmap(
        result,
        title="Custom Title",
        backend=TextVisualizerType.NATIVE_HEATMAP,
    )

    mock_factory.create.assert_called_once_with(
        method=TextVisualizerType.NATIVE_HEATMAP
    )
    mock_visualizer.plot.assert_called_once_with(
        words=["a"], scores=result.coefficients, title="Custom Title"
    )


@patch("xwhy.plots.TextVisualizerFactory")
def test_text_heatmap_str_backend(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify backend parsing from string works appropriately."""
    mock_visualizer = MagicMock()
    mock_factory.create.return_value = mock_visualizer

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    heatmap(result, backend="native_heatmap")

    mock_factory.create.assert_called_once_with(
        method=TextVisualizerType.NATIVE_HEATMAP
    )


def test_text_heatmap_invalid_backend(mock_metrics: RegressionMetricResult) -> None:
    """Verify ValueError is safely raised for completely invalid backend data types."""
    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    with pytest.raises(ValueError, match="Unsupported backend type: int"):
        heatmap(result, backend=123)
