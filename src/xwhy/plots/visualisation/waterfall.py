"""Waterfall plots of ordered feature contributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure

from .base import (
    BLUE,
    RED,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    _finish_plotly,
    _format_value,
    _group_minor_features,
    _single_instance,
)


def waterfall(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Plot a single prediction as a waterfall of additive contributions.

    The chart starts at the model's expected output ``E[f(X)]`` and walks
    feature by feature to the prediction ``f(x)``, so the bars sum exactly to
    the gap between the two.

    Args:
        explanation: A single-instance :class:`Explanation` or XWhy result.
        max_display: Maximum rows to draw before grouping the remainder.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | go.Figure | None: The figure when ``show`` is False and no
        ``save_path`` was given, otherwise ``None``.

    """
    del kwargs
    engine = _check_backend(backend, frozenset({"matplotlib", "plotly"}))
    exp = _as_explanation(explanation)
    values, base_value, names, data = _single_instance(exp)
    if max_display is None:
        max_display = len(values)

    # Annotate each row with the feature's actual value when we have one.
    labels = list(names)
    if data is not None:
        raw = np.asarray(data, dtype=object).ravel()
        if raw.shape[0] == len(names):
            labels = [
                f"{_format_value(raw[i])} = {names[i]}" for i in range(len(names))
            ]

    plot_values, plot_labels = _group_minor_features(values, labels, max_display)
    prediction = base_value + float(values.sum())

    if engine == "plotly":
        fig = go.Figure(
            go.Waterfall(
                orientation="h",
                y=plot_labels,
                x=plot_values,
                base=base_value,
                measure=["relative"] * len(plot_values),
                decreasing={"marker": {"color": BLUE}},
                increasing={"marker": {"color": RED}},
                connector={"line": {"color": "#bbbbbb"}},
                text=[_format_value(v, "%+.2f") for v in plot_values],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=title
            or (
                f"E[f(X)] = {_format_value(base_value)} &#8594; "
                f"f(x) = {_format_value(prediction)}"
            ),
            xaxis_title="Model output",
            template="plotly_white",
            showlegend=False,
            height=max(360, 34 * len(plot_labels) + 160),
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    if figsize:
        fig = plt.figure(figsize=figsize)
    else:
        num_features = min(max_display, len(values))
        fig = plt.figure(figsize=(8, num_features * 0.5 + 1.5))

    num_features = min(max_display, len(values))
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_value + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    lower_bounds = getattr(exp, "lower_bounds", None)
    upper_bounds = getattr(exp, "upper_bounds", None)

    num_individual = num_features if num_features == len(values) else num_features - 1

    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])  # type: ignore[index]
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])  # type: ignore[index]
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color="#bbbbbb",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if data is None:
            yticklabels[rng[i]] = names[order[i]]
        else:
            if np.issubdtype(type(data[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    _format_value(float(data[order[i]]), "%0.03f")
                    + " = "
                    + str(names[order[i]])
                )
            else:
                yticklabels[rng[i]] = str(data[order[i]]) + " = " + str(names[order[i]])

    if num_features < len(values):
        yticklabels[0] = f"{len(values) - num_features + 1} other features"
        remaining_impact = base_value - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=RED,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=BLUE,
        alpha=0,
    )

    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]

    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            dist - hl_scaled,
            0,
            head_length=min(dist, hl_scaled),
            color=RED,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array(
                    [[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]
                ),
                ecolor=RED,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            _format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="#ffffff",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()
            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                _format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=RED,
                fontsize=12,
            )

    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -(-dist - hl_scaled),
            0,
            head_length=min(-dist, hl_scaled),
            color=BLUE,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array(
                    [[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]
                ),
                ecolor=BLUE,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            _format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="#ffffff",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()
            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                _format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=BLUE,
                fontsize=12,
            )

    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(
        ytick_pos,
        yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]],
        fontsize=13,
    )

    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    plt.axvline(
        base_value,
        0,
        1 / num_features,
        color="#bbbbbb",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    fx = base_value + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)

    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([base_value, base_value + min(1e-8, xmax * 1e-10)])
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + _format_value(base_value, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks([fx, fx + min(1e-8, xmax * 1e-10)])
    ax3.set_xticklabels(
        ["$f(x)$", "$ = " + _format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left"
    )

    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color("#999999")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(
            22 / 72.0, -1 / 72.0, fig.dpi_scale_trans
        )
    )
    tick_labels[1].set_color("#999999")

    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if title:
        plt.title(title)

    return _finish_matplotlib(fig, show=show, save_path=save_path)
