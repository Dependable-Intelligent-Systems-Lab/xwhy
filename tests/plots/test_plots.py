"""Unit tests for the plots module."""

import re
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast
from unittest.mock import ANY, MagicMock, PropertyMock, patch

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import xwhy.plots.plots
from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.metrics.regression import RegressionMetricResult
from xwhy.plots import text_heatmap
from xwhy.plots.plots import (
    bar,
    beeswarm,
    decision,
    embedding,
    force,
    group_difference,
    heatmap,
    image,
    image_to_text,
    initjs,
    monitoring,
    partial_dependence,
    plot_feature_bar_chart,
    plot_feature_box_plot,
    replace_shap_label,
    scatter,
    text,
    violin,
    waterfall,
)
from xwhy.plots.types import TextPlotterType

matplotlib.use("Agg")

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


class EmptyFeatureDummyResult(BaseXWhyResult):
    """Dummy result with None feature names for testing fallback branches."""

    @property
    def feature_names(self) -> list[str] | None:
        """Mock feature names."""
        return None

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


@pytest.fixture(autouse=True)
def clean_plots() -> Generator[None, None, None]:
    """Ensure all matplotlib figures are closed before and after each test."""
    plt.close("all")
    yield
    plt.close("all")


def test_text_heatmap_unsupported_type(mock_metrics: RegressionMetricResult) -> None:
    """Ensure TypeError for unsupported result types."""
    result = DummyResult(coefficients=np.array([]), metrics=mock_metrics)
    with pytest.raises(TypeError, match="not supported for DummyResult"):
        text_heatmap(result)


@patch("xwhy.plots.plots.TextPlotterFactory")
def test_text_heatmap_success(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap calls plotter successfully with default kwargs logic."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    coeffs = np.array([0.1, 0.2])
    words = ["a", "b"]
    result = TextXWhyResult(coefficients=coeffs, metrics=mock_metrics, words=words)

    text_heatmap(result, custom_kwarg=123)

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)
    mock_plotter.plot.assert_called_once_with(
        words=words, scores=coeffs, title="Text Heatmap", custom_kwarg=123
    )


@patch("xwhy.plots.plots.TextPlotterFactory")
def test_text_heatmap_custom_args(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify text heatmap handles explicit 'title' and backend type mapping."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    text_heatmap(
        result,
        title="Custom Title",
        backend=TextPlotterType.NATIVE_HEATMAP,
    )

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)
    mock_plotter.plot.assert_called_once_with(
        words=["a"], scores=result.coefficients, title="Custom Title"
    )


@patch("xwhy.plots.plots.TextPlotterFactory")
def test_text_heatmap_str_backend(
    mock_factory: MagicMock, mock_metrics: RegressionMetricResult
) -> None:
    """Verify backend parsing from string works appropriately."""
    mock_plotter = MagicMock()
    mock_factory.create.return_value = mock_plotter

    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    text_heatmap(result, backend="native_heatmap")

    mock_factory.create.assert_called_once_with(method=TextPlotterType.NATIVE_HEATMAP)


def test_text_heatmap_invalid_backend(mock_metrics: RegressionMetricResult) -> None:
    """Verify ValueError is safely raised for completely invalid backend data types."""
    result = TextXWhyResult(
        coefficients=np.array([0.1]), metrics=mock_metrics, words=["a"]
    )

    with pytest.raises(ValueError, match="Unsupported backend type: int"):
        text_heatmap(result, backend=123)


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
    """Verify line breaking logic for multi-line text heatmap plots."""
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


# ==============================================================================
# VISUALISATION WRAPPER TESTS
# ==============================================================================


@pytest.fixture
def mock_shap_explanation() -> MagicMock:
    """Provide a mock Explanation object."""
    explanation = MagicMock()
    explanation.values.ndim = 5
    return explanation


@pytest.fixture
def mock_xwhy_result_1d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 1D data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros(5)
    result.to_explanation.return_value = mock_shap_explanation
    return result


@pytest.fixture
def mock_xwhy_result_2d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 2D data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros((10, 5))
    result.to_explanation.return_value = mock_shap_explanation
    return result


@pytest.fixture
def mock_xwhy_result_3d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 3D/4D multimodal data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros((1, 28, 28, 3))
    result.to_explanation.return_value = mock_shap_explanation
    return result


# ==============================================================================
# LOCAL PLOTS (Require 1D)
# ==============================================================================


@patch("xwhy.plots.plots.viz.bar")
def test_bar_wrapper(
    mock_shap_bar: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify bar plot wrapper triggers SHAP underlying implementation."""
    bar(mock_xwhy_result_1d, max_display=10)
    mock_shap_bar.assert_called_once_with(mock_shap_explanation, max_display=10)


@patch("xwhy.plots.plots.viz.waterfall")
def test_waterfall_wrapper(
    mock_shap_waterfall: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify waterfall plot wrapper triggers SHAP underlying implementation."""
    waterfall(mock_xwhy_result_1d, alpha=0.5)
    mock_shap_waterfall.assert_called_once_with(mock_shap_explanation, alpha=0.5)


@patch("xwhy.plots.plots.viz.text")
def test_text_wrapper(
    mock_shap_text: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify text plot wrapper triggers SHAP underlying implementation."""
    text(mock_xwhy_result_1d)
    mock_shap_text.assert_called_once_with(mock_shap_explanation)


@patch("xwhy.plots.plots.viz.force")
def test_force_wrapper(
    mock_shap_force: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify force plot wrapper triggers SHAP underlying implementation."""
    mock_shap_force.return_value = "force_html_mock"
    res = force(mock_xwhy_result_1d, link="logit")

    mock_shap_force.assert_called_once_with(mock_shap_explanation, link="logit")
    assert res == "force_html_mock"


@patch("xwhy.plots.plots.viz.decision")
def test_decision_wrapper_none_and_float(
    mock_shap_decision: MagicMock,
    mock_xwhy_result_1d: MagicMock,
) -> None:
    """Verify decision plot triggers SHAP implementation with nulls/floats."""
    mock_xwhy_result_1d.data = None
    mock_xwhy_result_1d.feature_names = None
    mock_xwhy_result_1d.base_values = 1.5

    decision(mock_xwhy_result_1d, min_percentile=0.95)

    mock_shap_decision.assert_called_once_with(
        base_value=1.5,
        shap_values=mock_xwhy_result_1d.coefficients,
        features=None,
        feature_names=None,
        min_percentile=0.95,
    )


@patch("xwhy.plots.plots.viz.decision")
def test_decision_wrapper_list_and_array(
    mock_shap_decision: MagicMock,
    mock_xwhy_result_1d: MagicMock,
) -> None:
    """Verify decision plot triggers SHAP implementation with arrays/lists."""
    mock_xwhy_result_1d.data = np.array([1, 2])
    mock_xwhy_result_1d.feature_names = ("f1", "f2")
    mock_xwhy_result_1d.base_values = np.array([0.5, 0.5])

    decision(mock_xwhy_result_1d, min_percentile=0.95)

    mock_shap_decision.assert_called_once_with(
        base_value=mock_xwhy_result_1d.base_values,
        shap_values=mock_xwhy_result_1d.coefficients,
        features=mock_xwhy_result_1d.data,
        feature_names=["f1", "f2"],
        min_percentile=0.95,
    )


# ==============================================================================
# GLOBAL PLOTS (Require 2D)
# ==============================================================================


@patch("xwhy.plots.plots.viz.scatter")
def test_scatter_wrapper(
    mock_shap_scatter: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify scatter plot wrapper triggers SHAP underlying implementation."""
    scatter(mock_xwhy_result_2d, color="blue")
    mock_shap_scatter.assert_called_once_with(mock_shap_explanation, color="blue")


@patch("xwhy.plots.plots.viz.heatmap")
def test_heatmap_wrapper(
    mock_shap_heatmap: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify native SHAP heatmap wrapper triggers SHAP underlying implementation."""
    heatmap(mock_xwhy_result_2d, show=False)
    mock_shap_heatmap.assert_called_once_with(mock_shap_explanation, show=False)


@patch("xwhy.plots.plots.viz.beeswarm")
def test_beeswarm_wrapper(
    mock_shap_beeswarm: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify beeswarm plot wrapper triggers SHAP underlying implementation."""
    beeswarm(mock_xwhy_result_2d, max_display=5)
    mock_shap_beeswarm.assert_called_once_with(mock_shap_explanation, max_display=5)


@patch("xwhy.plots.plots.viz.violin")
def test_violin_wrapper(
    mock_shap_violin: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify violin plot wrapper triggers SHAP underlying implementation."""
    violin(mock_xwhy_result_2d)
    mock_shap_violin.assert_called_once_with(mock_shap_explanation)


@patch("xwhy.plots.plots.viz.embedding")
def test_embedding_wrapper(
    mock_shap_embedding: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify embedding plot wrapper triggers SHAP underlying implementation."""
    embedding("feature_1", mock_xwhy_result_2d, alpha=0.8)
    mock_shap_embedding.assert_called_once_with(
        "feature_1", mock_shap_explanation, alpha=0.8
    )


@patch("xwhy.plots.plots.viz.group_difference")
def test_group_difference_wrapper(
    mock_shap_gd: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify group difference plot triggers SHAP underlying implementation."""
    mask = np.array([True, False])
    group_difference(mock_xwhy_result_2d, group_mask=mask)
    mock_shap_gd.assert_called_once_with(mock_shap_explanation, mask)


@patch("xwhy.plots.plots.viz.monitoring")
def test_monitoring_wrapper(
    mock_shap_monitoring: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify monitoring plot wrapper triggers SHAP underlying implementation."""
    feats = np.array([[1, 2]])
    monitoring(0, mock_xwhy_result_2d, feats)
    mock_shap_monitoring.assert_called_once_with(0, mock_shap_explanation, feats)


# ==============================================================================
# IMAGE PLOTS (Require 3D)
# ==============================================================================


@patch("xwhy.plots.plots.viz.image")
def test_image_wrapper(
    mock_shap_image: MagicMock,
    mock_xwhy_result_3d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify image plot wrapper triggers SHAP underlying implementation."""
    pixels = np.ones((28, 28))
    image(mock_xwhy_result_3d, pixels, label="test")
    expected_pixels = np.expand_dims(pixels, axis=0)

    mock_shap_image.assert_called_once()
    args, kwargs = mock_shap_image.call_args
    assert args[0] == mock_shap_explanation
    np.testing.assert_array_equal(kwargs["pixel_values"], expected_pixels)
    assert kwargs["label"] == "test"


@patch("xwhy.plots.plots.viz.image_to_text")
def test_image_to_text_wrapper(
    mock_shap_itt: MagicMock,
    mock_xwhy_result_3d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify image to text plot wrapper triggers SHAP underlying implementation."""
    mock_shap_explanation.values.ndim = 5
    image_to_text(mock_xwhy_result_3d)
    mock_shap_itt.assert_called_once_with(mock_shap_explanation)


def test_image_value_error_low_dim(mock_xwhy_result_1d: MagicMock) -> None:
    """Raise ValueError for low-dim result without image structure."""
    with pytest.raises(ValueError, match="requires image-structured explanations"):
        image(mock_xwhy_result_1d)


def test_image_with_superpixels_low_dim(
    mock_xwhy_result_1d: MagicMock, mock_shap_explanation: MagicMock
) -> None:
    """Bypass dimension check if superpixels attribute exists."""
    mock_xwhy_result_1d.coefficients = np.zeros(5)
    mock_xwhy_result_1d.superpixels = np.zeros((28, 28))

    with patch("xwhy.plots.plots.viz.image") as mock_shap_image:
        image(mock_xwhy_result_1d, pixel_values=None)
        mock_shap_image.assert_called_once()


@patch("xwhy.plots.plots.viz.image")
def test_image_pixel_values_none(
    mock_shap_image: MagicMock, mock_xwhy_result_3d: MagicMock
) -> None:
    """Trigger image plot with pixel_values set to None."""
    image(mock_xwhy_result_3d, pixel_values=None)
    mock_shap_image.assert_called_once()
    _, kwargs = mock_shap_image.call_args
    assert kwargs["pixel_values"] is None


@patch("xwhy.plots.plots.viz.image")
def test_image_pixel_values_3d(
    mock_shap_image: MagicMock, mock_xwhy_result_3d: MagicMock
) -> None:
    """Expand 3D pixel_values array to 4D in image plot."""
    pixels = np.ones((28, 28, 3))
    image(mock_xwhy_result_3d, pixel_values=pixels)
    _, kwargs = mock_shap_image.call_args
    assert kwargs["pixel_values"].shape == (1, 28, 28, 3)


@patch("xwhy.plots.plots.viz.image")
def test_image_pixel_values_2d(
    mock_shap_image: MagicMock, mock_xwhy_result_3d: MagicMock
) -> None:
    """Expand 2D pixel_values array to 3D in image plot."""
    pixels = np.ones((28, 28))
    image(mock_xwhy_result_3d, pixel_values=pixels)
    _, kwargs = mock_shap_image.call_args
    assert kwargs["pixel_values"].shape == (1, 28, 28)


@patch("xwhy.plots.plots.viz.image")
def test_image_pixel_values_4d(
    mock_shap_image: MagicMock, mock_xwhy_result_3d: MagicMock
) -> None:
    """Bypass expansion branches when pixel_values is already 4D."""
    pixels = np.ones((1, 28, 28, 3))
    image(mock_xwhy_result_3d, pixel_values=pixels)
    _, kwargs = mock_shap_image.call_args
    assert kwargs["pixel_values"].shape == (1, 28, 28, 3)


def test_image_to_text_value_error_low_dim(
    mock_xwhy_result_1d: MagicMock,
) -> None:
    """Raise ValueError for low-dim coefficients in image_to_text."""
    with pytest.raises(ValueError, match="requires multimodal"):
        image_to_text(mock_xwhy_result_1d)


def test_image_to_text_value_error_shap_dim(
    mock_xwhy_result_3d: MagicMock, mock_shap_explanation: MagicMock
) -> None:
    """Raise ValueError when shap explanation values ndim is less than 5."""
    mock_shap_explanation.values.ndim = 4
    with pytest.raises(ValueError, match="requires 5D explanations"):
        image_to_text(mock_xwhy_result_3d)


@patch("xwhy.plots.plots.viz.image_to_text")
def test_image_to_text_success(
    mock_shap_itt: MagicMock,
    mock_xwhy_result_3d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify successful execution of image_to_text plot wrapper."""
    mock_shap_explanation.values.ndim = 5
    image_to_text(mock_xwhy_result_3d, label="test")
    mock_shap_itt.assert_called_once_with(mock_shap_explanation, label="test")


@patch("xwhy.plots.plots.viz.initjs")
def test_initjs_wrapper(mock_initjs: MagicMock) -> None:
    """Verify initjs delegates to the visualisation engine."""
    xwhy.plots.plots.initjs()

    mock_initjs.assert_called_once_with()


@patch("xwhy.plots.plots.viz.partial_dependence")
def test_partial_dependence_wrapper(mock_pd: MagicMock) -> None:
    """Verify partial dependence forwards its arguments to the engine."""
    data = np.array([[1, 2]])

    xwhy.plots.plots.partial_dependence(0, "model", data, ice=False)

    mock_pd.assert_called_once_with(0, "model", ANY, ice=False)
    assert mock_pd.call_args.args[2] is data


# ==============================================================================
# GUARD TESTS (1D Failures)
# ==============================================================================


def test_plots_raise_error_for_1d_text_result(
    mock_metrics: RegressionMetricResult,
) -> None:
    """Verify that 2D/Image plots gracefully raise ValueError for 1D LLM results."""
    result_1d = TextXWhyResult(
        coefficients=np.array([0.1, 0.4, 0.2]),  # 1D Array
        metrics=mock_metrics,
        words=["Hello", "world", "!"],
    )

    # 3D/4D Multi-modal Plot Checks
    with pytest.raises(ValueError, match="requires image-structured"):
        image(result_1d)

    with pytest.raises(ValueError, match="requires multimodal"):
        image_to_text(result_1d)

    # 2D Global Plot Checks
    with pytest.raises(ValueError, match="requires a 2D matrix"):
        scatter(result_1d)

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        heatmap(result_1d)

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        beeswarm(result_1d)

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        violin(result_1d)

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        embedding(0, result_1d)

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        group_difference(result_1d, group_mask=np.array([True]))

    with pytest.raises(ValueError, match="requires a 2D matrix"):
        monitoring(0, result_1d, features=None)


# ==============================================================================
# COMPATIBILITY DECORATOR TESTS
# ==============================================================================


def test_decorator_modifies_matplotlib_xlabel() -> None:
    """Ensure matplotlib x-axis labels are intercepted and converted."""

    @replace_shap_label
    def bar(show: bool = True) -> None:
        _, ax = plt.subplots()
        ax.set_xlabel("Average SHAP value magnitude")
        if show:
            plt.show()

    with patch("shap.plots") as mock_shap_plots:
        # Real callable (no ``.called``) that accepts ``show`` so the
        # decorator enters the label-rewrite branch.
        def real_shap_bar(*, show: bool = True) -> None:
            """Accept a show flag so the decorator enters the label-rewrite path."""

        mock_shap_plots.bar = real_shap_bar
        decorated_bar = replace_shap_label(bar)

        with patch("matplotlib.pyplot.show") as mock_show:
            decorated_bar(show=True)
            ax = plt.gca()
            assert ax.get_xlabel() == "Average XWhy value magnitude"
            mock_show.assert_called_once()
            plt.close("all")


def test_decorator_bypasses_non_show_functions() -> None:
    """Ensure functions without a 'show' argument pass safely through."""

    @replace_shap_label
    def text(*args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        return "mocked_html_output"

    # Should execute seamlessly without injecting kwargs["show"]
    result = text()
    assert result == "mocked_html_output"


def test_replace_shap_label_returns_wrapped_result() -> None:
    """Verify the decorator forwards the wrapped function's return value."""

    def dummy_plot(show: bool = True) -> str:
        plt.figure()
        plt.xlabel("Average SHAP value (impact)")
        return "success"

    with patch("matplotlib.pyplot.show") as mock_show:
        result = replace_shap_label(dummy_plot)(show=True)

    assert result == "success"
    assert plt.gca().get_xlabel() == "Average XWhy value (impact)"
    mock_show.assert_called_once()


def test_replace_shap_label_inspect_raises() -> None:
    """Verify a non-introspectable callable falls back to a passthrough."""

    def weird_plot() -> str:
        return "handled"

    with patch("inspect.signature", side_effect=ValueError("builtin")):
        decorated = replace_shap_label(weird_plot)

    assert decorated() == "handled"


def test_replace_shap_label_no_figures() -> None:
    """Verify the decorator copes with a plot that opens no figure."""

    def empty_plot(show: bool = True) -> None:
        pass

    replace_shap_label(empty_plot)(show=False)

    assert len(plt.get_fignums()) == 0


def test_replace_shap_label_different_xlabel() -> None:
    """Verify an unrelated x-axis label is left untouched."""

    def diff_plot(show: bool = True) -> None:
        plt.figure()
        plt.xlabel("Feature Importance")

    replace_shap_label(diff_plot)(show=False)

    assert plt.gca().get_xlabel() == "Feature Importance"


def test_replace_shap_label_original_show_false() -> None:
    """Verify show=False relabels the axis without rendering."""

    def silent_plot(show: bool = True) -> None:
        plt.figure()
        plt.xlabel("SHAP value")

    with patch("matplotlib.pyplot.show") as mock_show:
        replace_shap_label(silent_plot)(show=False)

    assert plt.gca().get_xlabel() == "XWhy value"
    mock_show.assert_not_called()


# ==============================================================================
# INDEPENDENCE FROM SHAP
# ==============================================================================


def test_package_does_not_import_shap() -> None:
    """Verify importing xwhy never pulls shap into the interpreter."""
    import xwhy  # noqa: F401

    assert "shap" not in sys.modules, (
        "xwhy imported shap; the package is meant to be shap-free."
    )


def test_no_source_file_imports_shap() -> None:
    """Verify no module under src/xwhy imports shap."""
    source_root = Path(__file__).resolve().parents[2] / "src" / "xwhy"
    offenders = [
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
        if re.search(
            r"^\s*(import\s+shap|from\s+shap[\s.])",
            path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
    ]

    assert not offenders, f"These modules still import shap: {offenders}"
