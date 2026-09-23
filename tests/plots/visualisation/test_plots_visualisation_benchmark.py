"""Test benchmark module."""

import matplotlib

matplotlib.use("Agg")

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xwhy.plots.visualisation.benchmark import (
    _method_colors,
    _plot_bar_list,
    _plot_curve_list,
    _plot_multi_metric,
    _plot_single_curve,
    _shap_benchmark,
    _style_axes_bottom_left,
    benchmark,
)


@dataclass
class MockBenchmarkResult:
    """Mock benchmark result for testing."""

    method: str
    metric: str
    value: float
    value_sign: float
    full_name: str
    curve_x: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    curve_y: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None
    curve_y_std: np.ndarray[Any, np.dtype[np.floating[Any]]] | None = None


@pytest.fixture(autouse=True)
def _close_plots() -> Any:  # noqa: ANN401
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _make_curve(
    method: str = "MethodA",
    metric: str = "metric1",
    value: float = 0.5,
) -> MockBenchmarkResult:
    return MockBenchmarkResult(
        method=method,
        metric=metric,
        value=value,
        value_sign=1.0,
        full_name=f"{method}_{metric}",
        curve_x=np.array([1.0, 2.0, 3.0]),
        curve_y=np.array([0.1, 0.2, 0.3]),
        curve_y_std=np.array([0.01, 0.02, 0.03]),
    )


def _make_bar(
    method: str = "MethodB",
    metric: str = "metric1",
    value: float = 0.8,
) -> MockBenchmarkResult:
    return MockBenchmarkResult(
        method=method,
        metric=metric,
        value=value,
        value_sign=-1.0,
        full_name=f"{method}_{metric}",
    )


# ---------- _method_colors ----------


def test_method_colors_with_cmap() -> None:
    """Each method gets a distinct colour from the RED_BLUE_CIRCLE colourmap."""
    colors = _method_colors(["A", "B"])
    assert len(colors) == 2
    assert "A" in colors
    assert "B" in colors


def test_method_colors_without_cmap() -> None:
    """Fall back to BLUE when RED_BLUE_CIRCLE is None."""
    with patch("xwhy.plots.visualisation.benchmark.RED_BLUE_CIRCLE", None):
        colors = _method_colors(["X", "Y"])
        assert len(colors) == 2


# ---------- _style_axes_bottom_left ----------


def test_style_axes_bottom_left() -> None:
    """Only bottom and left spines should be visible."""
    _fig, ax = plt.subplots()
    _style_axes_bottom_left(ax)
    assert not ax.spines["right"].get_visible()
    assert not ax.spines["top"].get_visible()


# ---------- _plot_single_curve ----------


def test_plot_single_curve_with_color() -> None:
    """Curve with explicit colour and std band."""
    _plot_single_curve(_make_curve(), color="red")


def test_plot_single_curve_default_color() -> None:
    """Curve with default colour (colour=None -> BLUE)."""
    _plot_single_curve(_make_curve())


# ---------- _plot_curve_list ----------


def test_plot_curve_list() -> None:
    """Two methods plotted as overlapping curves."""
    _plot_curve_list([_make_curve(), _make_curve(method="M2")], "m1")


# ---------- _plot_bar_list ----------


def test_plot_bar_list() -> None:
    """Two methods plotted as horizontal bars."""
    _plot_bar_list([_make_bar(), _make_bar(method="M2")], "m1")


# ---------- _plot_multi_metric ----------


def test_plot_multi_metric_two_metrics() -> None:
    """Two metrics: all methods have values for both."""
    b1 = MockBenchmarkResult(
        method="M1", metric="m1", value=0.5, value_sign=1.0, full_name="M1_m1"
    )
    b2 = MockBenchmarkResult(
        method="M2", metric="m1", value=0.7, value_sign=1.0, full_name="M2_m1"
    )
    b3 = MockBenchmarkResult(
        method="M1", metric="m2", value=0.8, value_sign=-1.0, full_name="M1_m2"
    )
    b4 = MockBenchmarkResult(
        method="M2", metric="m2", value=0.9, value_sign=-1.0, full_name="M2_m2"
    )
    _plot_multi_metric([b1, b2, b3, b4])


def test_plot_multi_metric_span_zero() -> None:
    """When all values for a metric are equal, span == 0 -> norm = 0.0."""
    b1 = MockBenchmarkResult(
        method="M1", metric="m1", value=0.5, value_sign=1.0, full_name="M1_m1"
    )
    b2 = MockBenchmarkResult(
        method="M2", metric="m1", value=0.5, value_sign=1.0, full_name="M2_m1"
    )
    b3 = MockBenchmarkResult(
        method="M1", metric="m2", value=0.3, value_sign=1.0, full_name="M1_m2"
    )
    b4 = MockBenchmarkResult(
        method="M2", metric="m2", value=0.7, value_sign=1.0, full_name="M2_m2"
    )
    _plot_multi_metric([b1, b2, b3, b4])


def test_plot_multi_metric_three_metrics_branch() -> None:
    """Three metrics: a benchmark whose metric is neither metrics[0] nor metrics[1].

    Covers the 162->159 branch (elif skipped, back to for-loop).
    """
    b1 = MockBenchmarkResult(
        method="M1", metric="m1", value=0.5, value_sign=1.0, full_name="M1_m1"
    )
    b2 = MockBenchmarkResult(
        method="M1", metric="m2", value=0.6, value_sign=1.0, full_name="M1_m2"
    )
    b3 = MockBenchmarkResult(
        method="M1", metric="m3", value=0.7, value_sign=1.0, full_name="M1_m3"
    )
    _plot_multi_metric([b1, b2, b3])


# ---------- _shap_benchmark ----------


def test_shap_benchmark_iterable_curves() -> None:
    """Iterable with single metric + curves -> _plot_curve_list."""
    _shap_benchmark([_make_curve(), _make_curve(method="M2")], show=False)


def test_shap_benchmark_iterable_bars() -> None:
    """Iterable with single metric, no curves -> _plot_bar_list."""
    _shap_benchmark([_make_bar(), _make_bar(method="M3")], show=False)


def test_shap_benchmark_iterable_multi() -> None:
    """Iterable with multiple metrics -> _plot_multi_metric."""
    b1 = _make_bar(method="M1", metric="m1")
    b2 = _make_bar(method="M2", metric="m1")
    b3 = _make_bar(method="M1", metric="m2")
    b4 = _make_bar(method="M2", metric="m2")
    _shap_benchmark([b1, b2, b3, b4], show=False)


def test_shap_benchmark_single() -> None:
    """Single (non-iterable) benchmark result -> _plot_single_curve + show."""
    with patch("matplotlib.pyplot.show"):
        _shap_benchmark(_make_curve(), show=True)


def test_shap_benchmark_single_no_show() -> None:
    """Single benchmark, show=False -> no plt.show() call."""
    _shap_benchmark(_make_curve(), show=False)


# ---------- benchmark (public entry-point) ----------


def test_benchmark_with_finish_matplotlib() -> None:
    """When _finish_matplotlib is available, delegate to it."""
    with patch(
        "xwhy.plots.visualisation.benchmark._finish_matplotlib",
        return_value="finished",
    ):
        result = benchmark(_make_curve(), show=True, title="Title", figsize=(10, 5))
        assert result == "finished"  # type: ignore[comparison-overlap]


def test_benchmark_show_true_no_finish() -> None:
    """show=True without _finish_matplotlib -> plt.show(), return None."""
    with (
        patch("xwhy.plots.visualisation.benchmark._finish_matplotlib", new=None),
        patch("matplotlib.pyplot.show"),
    ):
        assert benchmark(_make_curve(), show=True) is None


def test_benchmark_show_false_no_finish() -> None:
    """show=False without _finish_matplotlib -> return Figure."""
    with patch("xwhy.plots.visualisation.benchmark._finish_matplotlib", new=None):
        fig = benchmark(_make_curve(), show=False)
        assert fig is not None


def test_benchmark_no_figsize() -> None:
    """figsize=None -> plt.figure() instead of plt.subplots(figsize=...)."""
    with patch("xwhy.plots.visualisation.benchmark._finish_matplotlib", new=None):
        fig = benchmark(_make_curve(), show=False)
        assert fig is not None


def test_benchmark_no_title() -> None:
    """title=None -> no suptitle call."""
    with patch("xwhy.plots.visualisation.benchmark._finish_matplotlib", new=None):
        fig = benchmark(_make_curve(), show=False, title=None)
        assert fig is not None
