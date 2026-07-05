"""Unit tests for core results."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.metrics.regression import RegressionMetricResult
from xwhy.visualization.types import TextVisualizerType


class ConcreteResult(BaseXWhyResult):
    """Concrete implementation of BaseXWhyResult for testing."""

    pass

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

def test_base_result_initialization(mock_metrics: RegressionMetricResult) -> None:
    """Verify BaseXWhyResult initializes correctly."""
    coeffs = np.array([0.1, 0.2])
    result = ConcreteResult(coefficients=coeffs, metrics=mock_metrics)

    assert np.array_equal(result.coefficients, coeffs)
    assert result.metrics == mock_metrics
    assert result.raw_data == {}  # Check default factory

def test_text_result_initialization(mock_metrics: RegressionMetricResult) -> None:
    """Verify TextXWhyResult initializes with correct defaults."""
    coeffs = np.array([0.1, 0.2])
    words = ["test", "case"]
    result = TextXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        original_output="test case",
        words=words
    )

    assert result.original_output == "test case"
    assert result.words == words

@patch("xwhy.core.result.TextVisualizerFactory")
def test_text_result_heatmap_success(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify heatmap calls the visualizer with correct arguments."""
    # Setup mocks
    mock_visualizer = MagicMock()
    mock_factory.create.return_value = mock_visualizer

    coeffs = np.array([0.1, 0.2])
    words = ["a", "b"]
    result = TextXWhyResult(
        coefficients=coeffs,
        metrics=mock_metrics,
        words=words
    )

    # Execute
    result.heatmap(
        title="Custom Title",
        backend=TextVisualizerType.NATIVE_HEATMAP,
        custom_kwarg=123
    )

    # Verify Factory interaction
    mock_factory.create.assert_called_once_with(method=TextVisualizerType.NATIVE_HEATMAP)

    # Verify Visualizer.plot interaction
    mock_visualizer.plot.assert_called_once_with(
        words=words,
        scores=coeffs,
        title="Custom Title",
        custom_kwarg=123
    )

@patch("xwhy.core.result.TextVisualizerFactory")
def test_text_result_heatmap_default_args(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify heatmap works with default arguments."""
    mock_visualizer = MagicMock()
    mock_factory.create.return_value = mock_visualizer

    result = TextXWhyResult(
        coefficients=np.array([0.0]),
        metrics=mock_metrics,
        words=[]
    )

    result.heatmap()

    # Verify defaults
    mock_factory.create.assert_called_once_with(method=TextVisualizerType.NATIVE_HEATMAP)
    mock_visualizer.plot.assert_called_once_with(
        words=[],
        scores=np.array([0.0]),
        title="Text Heatmap"
    )

def test_raw_data_mutation() -> None:
    """Verify raw_data can be updated."""
    result = ConcreteResult(
        coefficients=np.array([0]),
        metrics=MagicMock(),
        raw_data={"key": "value"}
    )
    result.raw_data["new"] = "data"
    assert result.raw_data["new"] == "data"
