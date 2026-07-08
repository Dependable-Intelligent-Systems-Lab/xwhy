"""Unit tests for the plots module."""

from typing import Any, cast
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots import heatmap
from xwhy.plots.types import TextPlotterType

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xwhy.plots.factory import TextPlotterFactory


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


@patch("xwhy.plots.TextPlotterFactory")
def test_text_heatmap_success(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap calls plotter successfully with default kwargs logic."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    coeffs = np.array([0.1, 0.2])
    words = ["a", "b"]
    result = TextXWhyResult(coefficients=coeffs, metrics=mock_metrics, words=words)

    # Execute with an extra generic kwarg to ensure it passes through correctly
    heatmap(result, custom_kwarg=123)

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)
    mock_plotter.plot.assert_called_once_with(
        words=words, scores=coeffs, title="Text Heatmap", custom_kwarg=123
    )


@patch("xwhy.plots.TextPlotterFactory")
def test_text_heatmap_custom_args(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap handles explicit 'title' and backend type mapping."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    heatmap(
        result,
        title="Custom Title",
        backend=TextPlotterType.NATIVE_HEATMAP,
    )

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)
    mock_plotter.plot.assert_called_once_with(
        words=["a"], scores=result.coefficients, title="Custom Title"
    )


@patch("xwhy.plots.TextPlotterFactory")
def test_text_heatmap_str_backend(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify backend parsing from string works appropriately."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    heatmap(result, backend="native_heatmap")

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)


def test_text_heatmap_invalid_backend(mock_metrics: RegressionMetricResult) -> None:
    """Verify ValueError is safely raised for completely invalid backend data types."""
    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    with pytest.raises(ValueError, match="Unsupported backend type: int"):
        heatmap(result, backend=123)


@patch("xwhy.plots.text.plt.show")
@patch("xwhy.plots.text.plt.savefig")
def test_native_heatmap_plotter(mock_savefig: object, mock_show: object) -> None:
    """Test standard execution of NativeHeatmapPlotter."""
    plotter = TextPlotterFactory.create(TextPlotterType.NATIVE_HEATMAP)
    words = ["This", "is", "a", "test"]
    scores = np.array([0.1, -0.5, 0.8, 0.0])

    plotter.plot(words=words, scores=scores, title="Test Plot", verbose=1)

    assert mock_show.called  # type: ignore

    plt.close("all")


@patch("xwhy.plots.text.plt.close")
@patch("xwhy.plots.text.plt.show")
@patch("xwhy.plots.text.plt.savefig")
def test_native_heatmap_plotter_save_path(
    mock_savefig: MagicMock, mock_show: MagicMock, mock_close: MagicMock
) -> None:
    """Test saving functionality of NativeHeatmapPlotter."""
    plotter = TextPlotterFactory.create(TextPlotterType.NATIVE_HEATMAP)
    words = ["test"]
    scores = np.array([1.0])

    plotter.plot(words=words, scores=scores, save_path="dummy.png", verbose=0)

    mock_savefig.assert_called_once_with("dummy.png", bbox_inches="tight")

    assert not mock_show.called

    assert mock_close.called


def test_plotter_factory_invalid() -> None:
    """Test factory raises error on invalid input."""
    with pytest.raises(ValueError, match="Unsupported plotter method"):
        TextPlotterFactory.create(cast(Any, "invalid_method"))


def test_plot_denom_handling() -> None:
    """Test that plot handles denom=0 by using a small epsilon."""
    plotter = TextPlotterFactory.create(TextPlotterType.NATIVE_HEATMAP)
    words = ["word"]
    scores = np.array([0.0])

    with patch("xwhy.plots.text.plt.show"), patch("matplotlib.text.Text.draw"):
        plotter.plot(words=words, scores=scores)

    assert True


def test_plot_new_line_logic() -> None:
    """Verify line breaking logic for multi-line heatmap plots."""
    plotter = TextPlotterFactory.create(TextPlotterType.NATIVE_HEATMAP)
    words = ["word1", "word2"]
    scores = np.array([0.1, 0.2])

    with (
        patch("xwhy.plots.text.plt.show"),
        patch("matplotlib.pyplot.tight_layout"),
        patch("matplotlib.text.Text.draw"),
        patch("matplotlib.text.Text.get_window_extent") as mock_extent,
    ):
        bbox = MagicMock()
        bbox.width = 10.0
        mock_extent.return_value = bbox

        plotter.plot(words=words, scores=scores, max_word_per_line=1)

    assert True
