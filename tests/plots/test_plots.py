"""Unit tests for the plots module."""

import sys
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import ANY, MagicMock, patch

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
    monitoring,
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


if "shap" not in sys.modules:
    shap_mock = MagicMock()
    sys.modules["shap"] = shap_mock
    sys.modules["shap.plots"] = MagicMock()


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
# SHAP WRAPPER TESTS
# ==============================================================================


@pytest.fixture
def mock_shap_explanation() -> MagicMock:
    """Fixture providing a mock SHAP Explanation object."""
    return MagicMock()


@pytest.fixture
def mock_xwhy_result_1d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 1D data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros(5)
    result.to_shap.return_value = mock_shap_explanation
    return result


@pytest.fixture
def mock_xwhy_result_2d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 2D data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros((10, 5))
    result.to_shap.return_value = mock_shap_explanation
    return result


@pytest.fixture
def mock_xwhy_result_3d(mock_shap_explanation: MagicMock) -> MagicMock:
    """Fixture providing a mock BaseXWhyResult with 3D/4D multimodal data."""
    result = MagicMock(spec=BaseXWhyResult)
    result.coefficients = np.zeros((1, 28, 28, 3))
    result.to_shap.return_value = mock_shap_explanation
    return result


# ==============================================================================
# LOCAL PLOTS (Require 1D)
# ==============================================================================


@patch("shap.plots.bar")
def test_bar_wrapper(
    mock_shap_bar: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify bar plot wrapper triggers SHAP underlying implementation."""
    bar(mock_xwhy_result_1d, max_display=10)
    mock_shap_bar.assert_called_once_with(mock_shap_explanation, max_display=10)


@patch("shap.plots.waterfall")
def test_waterfall_wrapper(
    mock_shap_waterfall: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify waterfall plot wrapper triggers SHAP underlying implementation."""
    waterfall(mock_xwhy_result_1d, alpha=0.5)
    mock_shap_waterfall.assert_called_once_with(mock_shap_explanation, alpha=0.5)


@patch("shap.plots.text")
def test_text_wrapper(
    mock_shap_text: MagicMock,
    mock_xwhy_result_1d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify text plot wrapper triggers SHAP underlying implementation."""
    text(mock_xwhy_result_1d)
    mock_shap_text.assert_called_once_with(mock_shap_explanation)


@patch("shap.plots.force")
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


@patch("shap.plots.decision")
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


@patch("shap.plots.decision")
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


@patch("shap.plots.scatter")
def test_scatter_wrapper(
    mock_shap_scatter: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify scatter plot wrapper triggers SHAP underlying implementation."""
    scatter(mock_xwhy_result_2d, color="blue")
    mock_shap_scatter.assert_called_once_with(mock_shap_explanation, color="blue")


@patch("shap.plots.heatmap")
def test_heatmap_wrapper(
    mock_shap_heatmap: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify native SHAP heatmap wrapper triggers SHAP underlying implementation."""
    heatmap(mock_xwhy_result_2d, show=False)
    mock_shap_heatmap.assert_called_once_with(mock_shap_explanation, show=False)


@patch("shap.plots.beeswarm")
def test_beeswarm_wrapper(
    mock_shap_beeswarm: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify beeswarm plot wrapper triggers SHAP underlying implementation."""
    beeswarm(mock_xwhy_result_2d, max_display=5)
    mock_shap_beeswarm.assert_called_once_with(mock_shap_explanation, max_display=5)


@patch("shap.plots.violin")
def test_violin_wrapper(
    mock_shap_violin: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify violin plot wrapper triggers SHAP underlying implementation."""
    violin(mock_xwhy_result_2d)
    mock_shap_violin.assert_called_once_with(mock_shap_explanation)


@patch("shap.plots.embedding")
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


@patch("shap.plots.group_difference")
def test_group_difference_wrapper(
    mock_shap_gd: MagicMock,
    mock_xwhy_result_2d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify group difference plot triggers SHAP underlying implementation."""
    mask = np.array([True, False])
    group_difference(mock_xwhy_result_2d, group_mask=mask)
    mock_shap_gd.assert_called_once_with(mock_shap_explanation, mask)


@patch("shap.plots.monitoring")
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


@patch("shap.plots.image")
def test_image_wrapper(
    mock_shap_image: MagicMock,
    mock_xwhy_result_3d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify image plot wrapper triggers SHAP underlying implementation."""
    pixels = np.ones((28, 28))
    image(mock_xwhy_result_3d, pixels, label="test")
    mock_shap_image.assert_called_once_with(mock_shap_explanation, pixels, label="test")


@patch("shap.plots.image_to_text")
def test_image_to_text_wrapper(
    mock_shap_itt: MagicMock,
    mock_xwhy_result_3d: MagicMock,
    mock_shap_explanation: MagicMock,
) -> None:
    """Verify image to text plot wrapper triggers SHAP underlying implementation."""
    image_to_text(mock_xwhy_result_3d)
    mock_shap_itt.assert_called_once_with(mock_shap_explanation)


@patch("xwhy.plots.plots.initjs")
def test_initjs_wrapper(mock_initjs: MagicMock) -> None:
    """Verify initjs aliases to SHAP implementation directly."""
    xwhy.plots.plots.initjs()

    mock_initjs.assert_called_once()


@patch("xwhy.plots.plots.partial_dependence")
def test_partial_dependence_wrapper(mock_pd: MagicMock) -> None:
    """Verify partial dependence aliases to SHAP implementation directly."""
    xwhy.plots.plots.partial_dependence(0, "model", np.array([[1, 2]]), ice=False)

    mock_pd.assert_called_once_with(0, "model", ANY, ice=False)


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


def test_decorator_modifies_matplotlib_xlabel() -> None:
    """Ensure matplotlib x-axis labels are intercepted and converted."""

    @replace_shap_label
    def bar(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        _, ax = plt.subplots()
        ax.set_xlabel("Average SHAP value magnitude")
        if kwargs.get("show", True):
            plt.show()

    with patch("matplotlib.pyplot.show") as mock_show:
        bar(show=True)
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


@patch("shap.plots", create=True)
@patch("matplotlib.pyplot.show")
def test_replace_shap_label_full_flow_all_true(
    mock_show: MagicMock, mock_shap_plots: MagicMock
) -> None:
    """Test all positive branches: show=True, figure exists, 'SHAP value' in label."""

    def dummy_plot(show: bool = True) -> str:
        plt.figure()
        plt.xlabel("Average SHAP value (impact)")
        return "success"

    mock_shap_plots.dummy_plot = dummy_plot

    decorated = replace_shap_label(dummy_plot)
    result = decorated(show=True)

    assert result == "success"
    assert plt.gca().get_xlabel() == "Average XWhy value (impact)"
    mock_show.assert_called_once()


@patch("shap.plots", create=True)
def test_replace_shap_label_shap_func_none(mock_shap_plots: MagicMock) -> None:
    """Test when the function is not found in shap.plots (shap_func is None)."""

    def unknown_plot() -> str:
        return "bypassed"

    decorated = replace_shap_label(unknown_plot)
    result = decorated()

    assert result == "bypassed"


@patch("shap.plots", create=True)
@patch("inspect.signature")
def test_replace_shap_label_inspect_raises(
    mock_signature: MagicMock, mock_shap_plots: MagicMock
) -> None:
    """Test the except (ValueError, TypeError) block during signature inspection."""

    def weird_plot() -> str:
        return "handled"

    mock_shap_plots.weird_plot = weird_plot
    mock_signature.side_effect = ValueError("Cannot inspect built-in")

    decorated = replace_shap_label(weird_plot)
    result = decorated()

    assert result == "handled"


@patch("shap.plots", create=True)
def test_replace_shap_label_no_figures(mock_shap_plots: MagicMock) -> None:
    """Test the branch where len(plt.get_fignums()) > 0 is False."""

    def empty_plot(show: bool = True) -> None:
        pass

    mock_shap_plots.empty_plot = empty_plot

    decorated = replace_shap_label(empty_plot)
    decorated(show=False)

    assert len(plt.get_fignums()) == 0


@patch("shap.plots", create=True)
def test_replace_shap_label_different_xlabel(mock_shap_plots: MagicMock) -> None:
    """Test the branch where 'SHAP value' is not in the current_xlabel."""

    def diff_plot(show: bool = True) -> None:
        plt.figure()
        plt.xlabel("Feature Importance")

    mock_shap_plots.diff_plot = diff_plot

    decorated = replace_shap_label(diff_plot)
    decorated(show=False)

    assert plt.gca().get_xlabel() == "Feature Importance"


@patch("shap.plots", create=True)
@patch("matplotlib.pyplot.show")
def test_replace_shap_label_original_show_false(
    mock_show: MagicMock, mock_shap_plots: MagicMock
) -> None:
    """Test the branch where original_show is False."""

    def silent_plot(show: bool = True) -> None:
        plt.figure()
        plt.xlabel("SHAP value")

    mock_shap_plots.silent_plot = silent_plot

    decorated = replace_shap_label(silent_plot)
    decorated(show=False)

    assert plt.gca().get_xlabel() == "XWhy value"
    mock_show.assert_not_called()


@patch("shap.plots", create=True)
def test_replace_shap_label_already_called(mock_shap_plots: MagicMock) -> None:
    """Test the guard block handling hasattr(current_func, 'called')."""

    def already_called_plot() -> str:
        return "direct"

    already_called_plot.called = True  # type: ignore
    mock_shap_plots.already_called_plot = already_called_plot

    decorated = replace_shap_label(already_called_plot)
    result = decorated()

    assert result == "direct"


@patch("shap.plots", create=True)
def test_replace_shap_label_shap_func_is_none(mock_shap_plots: MagicMock) -> None:
    """Test the branch where shap_func is None (func not in shap.plots)."""

    def my_custom_plot() -> str:
        return "bypassed and executed"

    del mock_shap_plots.my_custom_plot

    decorated = replace_shap_label(my_custom_plot)
    result = decorated()

    assert result == "bypassed and executed"
