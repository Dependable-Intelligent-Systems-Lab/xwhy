"""Bar summary plots of feature attribution magnitudes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import (
    BLUE,
    RED,
    VALUE_LABEL,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    _finish_plotly,
    _format_value,
    _global_importance,
    _group_minor_features,
    _resolve_names,
    _style_axes,
)


def _bar_colors(values: np.ndarray) -> list[str]:
    """Map attribution signs onto the XWhy red/blue palette."""
    return [RED if v >= 0 else BLUE for v in values]


def bar(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Create a bar plot of attribution values.

    A one-dimensional explanation is drawn as a signed local attribution bar
    chart. A two-dimensional explanation is collapsed to ``mean(|value|)``
    per feature, giving a global importance ranking.

    Args:
        explanation: An :class:`Explanation` or XWhy result.
        max_display: Maximum rows to draw. Remaining features are folded into
            a single "Sum of N other features" row. ``None`` shows everything.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        ax: Optional existing matplotlib axes to draw into.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | go.Figure | None: The figure when ``show`` is False and no
        ``save_path`` was given, otherwise ``None``.

    """
    del kwargs
    engine = _check_backend(backend, frozenset({"matplotlib", "plotly"}))
    exp = _as_explanation(explanation)

    values = np.asarray(exp.values, dtype=float)
    is_global = values.ndim > 1
    scores = _global_importance(values) if is_global else values.ravel()
    names = _resolve_names(exp.feature_names, scores.shape[0])

    plot_values, plot_names = _group_minor_features(scores, names, max_display)
    xlabel = f"mean(|{VALUE_LABEL}|)" if is_global else VALUE_LABEL

    if engine == "plotly":
        fig = go.Figure(
            go.Bar(
                x=plot_values,
                y=plot_names,
                orientation="h",
                marker_color=_bar_colors(plot_values),
                text=[_format_value(v) for v in plot_values],
                textposition="outside",
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            title=title or "Feature importance",
            xaxis_title=xlabel,
            template="plotly_white",
            showlegend=False,
            height=max(320, 32 * len(plot_names) + 140),
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    height = figsize[1] if figsize else (0.5 * len(plot_names) + 1.5)
    width = figsize[0] if figsize else 8.0

    if ax is None:
        ax = plt.gca()
        fig = plt.gcf()
        fig.set_size_inches(width, height)
    else:
        fig = ax.get_figure()

    positions = np.arange(len(plot_values))
    ax.barh(positions, plot_values, color=_bar_colors(plot_values), height=0.7)

    span = float(np.max(np.abs(plot_values))) if plot_values.size else 1.0
    offset = span * 0.02 if span else 0.01

    for pos, value in zip(positions, plot_values, strict=True):
        aligned_right = value >= 0
        ax.text(
            value + (offset if aligned_right else -offset),
            float(pos),
            _format_value(value, "%+0.02f"),
            va="center",
            ha="left" if aligned_right else "right",
            fontsize=12,
            color=RED if aligned_right else BLUE,
        )

    # Double-plot y-ticks to match SHAP's exact bolding hack
    ax.set_yticks(list(positions) + list(positions + 1e-8))
    ax.set_yticklabels(
        plot_names + [t.split("=")[-1] for t in plot_names],
        fontsize=13,
        color="#000000",
    )

    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(len(positions)):
        tick_labels[i].set_color("#999999")

    ax.set_xlabel(xlabel, fontsize=13, color="#000000")
    ax.tick_params("x", labelsize=11)

    xmin, xmax = ax.get_xlim()
    x_buffer = (xmax - xmin) * 0.05
    negative_values_present = (
        float(plot_values.min()) < 0 if plot_values.size else False
    )
    if negative_values_present:
        lower, upper = xmin - x_buffer, xmax + x_buffer
    else:
        lower, upper = xmin, xmax + x_buffer

    if lower == upper:
        lower, upper = lower - 1.0, upper + 1.0
    ax.set_xlim(lower, upper)

    if title:
        ax.set_title(title, fontsize=13, loc="left")

    _style_axes(ax)
    if not is_global:
        ax.axvline(0, color="#333333", linewidth=0.9)

    # fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)
