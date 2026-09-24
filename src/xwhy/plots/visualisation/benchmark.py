"""Benchmark result curves and multi-metric comparison plots."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .base import (
    BLUE,
    RED_BLUE_CIRCLE,
    _finish_matplotlib,
)

XLABEL_NAMES: dict[str, str] = {
    "remove absolute": "Fraction removed",
    "remove positive": "Fraction removed",
    "remove negative": "Fraction removed",
    "keep absolute": "Fraction kept",
    "keep positive": "Fraction kept",
    "keep negative": "Fraction kept",
    "explanation error": "Explanation error as std dev.",
    "compute time": "Seconds per. sample",
}


def _method_colors(methods: Sequence[str]) -> dict[str, Any]:
    """Assign a distinct colour to each method name."""
    n = max(len(methods), 1)
    colors: dict[str, Any] = {}
    for i, method in enumerate(methods):
        if RED_BLUE_CIRCLE is not None:
            colors[method] = RED_BLUE_CIRCLE(i / n)
        else:
            colors[method] = BLUE
    return colors


def _style_axes_bottom_left(ax: Any) -> None:  # noqa: ANN401
    """Show only bottom and left spines/ticks."""
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def _plot_single_curve(result: Any, color: Any = None) -> None:  # noqa: ANN401
    """Draw one benchmark curve with ±1 std band."""
    if color is None:
        color = BLUE
    plt.fill_between(
        result.curve_x,
        result.curve_y - result.curve_y_std,
        result.curve_y + result.curve_y_std,
        color=color,
        alpha=0.1,
        linewidth=0,
    )
    plt.plot(
        result.curve_x,
        result.curve_y,
        color=color,
        linewidth=2,
        label=f"{result.method} ({result.value:0.3})",
    )


def _plot_curve_list(benchmark: list[Any], metric_name: str) -> None:
    """Plot several methods that share a single metric with curves."""
    methods = sorted({b.method for b in benchmark})
    method_color = _method_colors(methods)
    benchmark.sort(key=lambda b: -b.value_sign * b.value)

    for b in benchmark:
        plt.fill_between(
            b.curve_x,
            b.curve_y - b.curve_y_std,
            b.curve_y + b.curve_y_std,
            color=method_color[b.method],
            alpha=0.1,
            linewidth=0,
        )
    for b in benchmark:
        plt.plot(
            b.curve_x,
            b.curve_y,
            color=method_color[b.method],
            linewidth=2,
            label=f"{b.method} ({b.value:0.3})",
        )

    ax = plt.gca()
    ax.set_xlabel(XLABEL_NAMES.get(metric_name, metric_name), fontsize=13)
    ax.set_ylabel("Model output", fontsize=13)
    _style_axes_bottom_left(ax)
    plt.title(metric_name.capitalize())
    plt.legend(fontsize=11)


def _plot_bar_list(benchmark: list[Any], metric_name: str) -> None:
    """Horizontal bar chart for a single metric without curves."""
    methods = sorted({b.method for b in benchmark})
    method_color = _method_colors(methods)
    benchmark.sort(key=lambda b: -b.value_sign * b.value)

    values = np.array([b.value for b in benchmark])
    plt.barh(
        np.arange(len(values)),
        values,
        0.7,
        align="center",
        color=[method_color[b.method] for b in benchmark],
        edgecolor=(1, 1, 1, 0.8),
    )

    ax = plt.gca()
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([b.method for b in benchmark], rotation=0, fontsize=11)
    ax.set_xlabel(XLABEL_NAMES.get(metric_name, metric_name), fontsize=13)
    _style_axes_bottom_left(ax)
    plt.title(metric_name.capitalize())
    plt.gca().invert_yaxis()


def _plot_multi_metric(benchmark: list[Any]) -> None:
    """Parallel-coordinates style plot across multiple metrics."""
    metrics: list[str] = []
    for b in benchmark:
        if b.metric not in metrics:
            metrics.append(b.metric)

    methods = sorted({b.method for b in benchmark})
    method_color = _method_colors(methods)

    max_value = dict.fromkeys(metrics, -np.inf)
    min_value = dict.fromkeys(metrics, np.inf)
    for b in benchmark:
        signed = b.value_sign * b.value
        max_value[b.metric] = max(max_value[b.metric], signed)
        min_value[b.metric] = min(min_value[b.metric], signed)

    norm_values: dict[str, float] = {}
    for b in benchmark:
        span = max_value[b.metric] - min_value[b.metric]
        if span == 0:
            norm_values[b.full_name] = 0.0
        else:
            norm_values[b.full_name] = (
                b.value_sign * b.value - min_value[b.metric]
            ) / span

    metric_0: dict[str, float] = {}
    metric_1: dict[str, float] = {}
    for b in benchmark:
        if b.metric == metrics[0]:
            metric_0[b.method] = b.value
        elif len(metrics) > 1 and b.metric == metrics[1]:
            metric_1[b.method] = b.value

    methods.sort(
        key=lambda method: (
            np.round(metric_0.get(method, 0.0), 3),
            metric_1.get(method, 0.0),
        )
    )

    denom = max(len(methods) - 1, 1)
    xs = [-0.03 * denom, *list(range(len(metrics) + 1))]

    for i, method in enumerate(methods):
        scores: list[float] = [1 - i / denom, 1 - i / denom]
        values: list[float | None] = [None, None]
        for metric in metrics:
            for b in benchmark:
                if b.method == method and b.metric == metric:
                    scores.append(norm_values[b.full_name])
                    values.append(b.value)
        plt.plot(xs, scores, color=method_color[method], label=method)

        for x, y, value in zip(xs, scores, values, strict=False):
            if value is None:
                continue
            label = f"{value:.2f}"
            txt = plt.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, -3),
                ha="center",
                color=method_color[method],
                fontsize=9,
            )
            txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="w")])

    ax = plt.gca()
    ax.set_yticks([1 - i / denom for i in range(len(methods))])
    ax.set_yticklabels(methods, rotation=0, fontsize=11)
    ax.set_xticks(np.arange(len(metrics) + 1))
    ax.set_xticklabels(
        [""] + [m.capitalize() for m in metrics],
        rotation=45,
        ha="left",
        fontsize=11,
    )
    ax.xaxis.tick_top()
    plt.grid(which="major", axis="x", linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.yaxis.set_ticks_position("none")
    ax.xaxis.set_ticks_position("none")
    plt.xlim(xs[0], len(metrics))
    ax.get_xticklabels()[1].set_fontweight("bold")


def _shap_benchmark(
    benchmark: Any,  # noqa: ANN401
    show: bool = True,
) -> None:
    """Plot a BenchmarkResult or a list of such results.

    Parameters
    ----------
    benchmark :
        A single result object or an iterable of results. Each item is
        expected to expose ``method``, ``metric``, ``value``, ``value_sign``,
        and optionally ``curve_x`` / ``curve_y`` / ``curve_y_std``.
    show :
        When ``True`` and a single non-iterable result is plotted, call
        ``plt.show()``.

    """
    if isinstance(benchmark, Iterable) and not isinstance(benchmark, (str, bytes)):
        results = list(benchmark)

        single_metric = True
        metric_name: str | None = None
        has_curves = True
        for b in results:
            if metric_name is None:
                metric_name = b.metric
            elif metric_name != b.metric:
                single_metric = False
            if b.curve_x is None or b.curve_y is None:
                has_curves = False

        if single_metric and has_curves and metric_name is not None:
            _plot_curve_list(results, metric_name)
        elif single_metric and metric_name is not None:
            _plot_bar_list(results, metric_name)
        else:
            _plot_multi_metric(results)
        return

    # Single BenchmarkResult
    _plot_single_curve(benchmark, color=BLUE)
    ax = plt.gca()
    ax.set_xlabel(XLABEL_NAMES.get(benchmark.metric, benchmark.metric), fontsize=13)
    ax.set_ylabel("Model output", fontsize=13)
    _style_axes_bottom_left(ax)
    plt.legend(fontsize=11)
    if show:
        plt.show()


def benchmark(
    benchmark_result: Any,  # noqa: ANN401
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Plot explanation-method benchmark results.

    Accepts a single result or a sequence of results and chooses a curve,
    bar, or multi-metric layout based on the metrics present.

    Parameters
    ----------
    benchmark_result :
        One BenchmarkResult-like object or an iterable of them.
    show :
        Whether to display the figure.
    save_path :
        Optional path to write the figure to.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches.
    **kwargs :
        Accepted for call compatibility; currently ignored.

    Returns
    -------
    Figure | None
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    _ = kwargs
    if figsize is not None:
        plt.subplots(figsize=figsize)
    else:
        plt.figure()

    _shap_benchmark(benchmark_result, show=False)

    if title:
        plt.gcf().suptitle(title, fontsize=13, fontweight="bold")

    fig = plt.gcf()
    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig
