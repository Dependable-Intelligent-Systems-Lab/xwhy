"""Force plots of additive feature contributions."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib import lines
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MatplotlibPath
from numpy.typing import NDArray

from .base import (
    BLUE,
    RED,
    _as_explanation,
    _check_backend,
    _finish_html,
    _finish_matplotlib,
    _format_value,
    _single_instance,
)

# ---------------------------------------------------------------------------
# Pure data helpers
# ---------------------------------------------------------------------------


def _format_data(
    data: dict[str, Any],
) -> tuple[
    NDArray[Any],
    float,
    NDArray[Any],
    float,
]:
    """Split and convert feature effects for the additive force layout.

    Parameters
    ----------
    data :
        Mapping with keys ``features``, ``featureNames``, ``link``,
        ``outValue``, and ``baseValue``.

    Returns
    -------
    neg_features, total_neg, pos_features, total_pos
        Sorted feature arrays (effect, value, name) and total spans after
        applying the link function.

    """
    neg_features = np.array(
        [
            [
                data["features"][x]["effect"],
                data["features"][x]["value"],
                data["featureNames"][x],
            ]
            for x in data["features"]
            if data["features"][x]["effect"] < 0
        ],
        dtype=object,
    )
    if len(neg_features) > 0:
        neg_features = np.array(
            sorted(neg_features, key=lambda x: float(x[0]), reverse=False),
            dtype=object,
        )

    pos_features = np.array(
        [
            [
                data["features"][x]["effect"],
                data["features"][x]["value"],
                data["featureNames"][x],
            ]
            for x in data["features"]
            if data["features"][x]["effect"] >= 0
        ],
        dtype=object,
    )
    if len(pos_features) > 0:
        pos_features = np.array(
            sorted(pos_features, key=lambda x: float(x[0]), reverse=True),
            dtype=object,
        )

    link = data["link"]
    if link == "identity":

        def convert_func(x: float) -> float:
            return x

    elif link == "logit":

        def convert_func(x: float) -> float:
            return float(1 / (1 + np.exp(-x)))

    else:
        msg = f"ERROR: Unrecognized link function: {link}"
        raise ValueError(msg)

    neg_val = float(data["outValue"])
    for i in neg_features:
        val = float(i[0])
        neg_val = neg_val + abs(val)
        i[0] = convert_func(neg_val)

    if len(neg_features) > 0:
        total_neg = float(
            np.max(neg_features[:, 0].astype(float))
            - np.min(neg_features[:, 0].astype(float))
        )
    else:
        total_neg = 0.0

    pos_val = float(data["outValue"])
    for i in pos_features:
        val = float(i[0])
        pos_val = pos_val - abs(val)
        i[0] = convert_func(pos_val)

    if len(pos_features) > 0:
        total_pos = float(
            np.max(pos_features[:, 0].astype(float))
            - np.min(pos_features[:, 0].astype(float))
        )
    else:
        total_pos = 0.0

    data["outValue"] = convert_func(float(data["outValue"]))
    data["baseValue"] = convert_func(float(data["baseValue"]))

    return neg_features, total_neg, pos_features, total_pos


def _force_html(
    positives: list[tuple[str, float]],
    negatives: list[tuple[str, float]],
    base_value: float,
    prediction: float,
    title: str | None = None,
) -> str:
    """Build a static HTML force bar (no JavaScript required).

    Parameters
    ----------
    positives :
        ``(label, value)`` pairs that increase the prediction.
    negatives :
        ``(label, value)`` pairs that decrease the prediction.
    base_value :
        Model expected value.
    prediction :
        Model output for the instance.
    title :
        Optional heading.

    Returns
    -------
    str
        Self-contained HTML markup.

    """
    total = sum(v for _, v in positives) + sum(abs(v) for _, v in negatives)
    total = total if total > 0 else 1.0

    def segment(label: str, value: float, color: str) -> str:
        pct = abs(value) / total * 100
        caption = escape(label) if pct > 8 else ""
        return (
            f"<div title='{escape(label)}: {value:+.4f}' "
            f'style="flex:0 0 {pct:.4f}%;background:{color};'
            "color:#fff;font-size:11px;line-height:34px;"
            'text-align:center;overflow:hidden;white-space:nowrap;">'
            f"{caption}</div>"
        )

    blocks = [segment(label, value, RED) for label, value in positives]
    blocks += [segment(label, value, BLUE) for label, value in negatives]

    heading = (
        f"<div style='font-size:14px;font-weight:600;"
        f"margin-bottom:10px;'>{escape(title)}</div>"
        if title
        else ""
    )

    return (
        "<div class='xwhy-force' style=\"font-family:-apple-system,"
        "BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:14px;"
        'border:1px solid #e6e6e6;border-radius:8px;">'
        f"{heading}"
        "<div style='font-size:13px;margin-bottom:6px;color:#333;'>"
        f"E[f(X)] = {_format_value(base_value, '%.4f')} "
        "&nbsp;&#8594;&nbsp; "
        f"<b>f(x) = {_format_value(prediction, '%.4f')}</b></div>"
        "<div style='display:flex;width:100%;border-radius:5px;"
        f"overflow:hidden;'>{''.join(blocks)}</div>"
        "<div style='margin-top:8px;font-size:11px;color:#666;'>"
        f"<span style='color:{RED};'>&#9632;</span> "
        "increases the prediction"
        f" &nbsp;&nbsp;<span style='color:{BLUE};'>&#9632;</span> "
        "decreases it"
        "</div></div>"
    )


# ---------------------------------------------------------------------------
# Matplotlib drawing helpers
# ---------------------------------------------------------------------------


def draw_output_element(
    out_name: str,
    out_value: float,
    ax: Any,  # noqa: ANN401
) -> None:
    """Draw the final prediction marker and label on *ax*."""
    x_coords = np.array([out_value, out_value])
    y_coords = np.array([0.0, 0.24])
    line = lines.Line2D(x_coords, y_coords, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)

    font0 = FontProperties()
    font = font0.copy()
    font.set_weight("bold")
    text_out_val = plt.text(
        out_value,
        0.25,
        f"{out_value:.2f}",
        fontproperties=font,
        fontsize=14,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})

    text_out_val = plt.text(
        out_value,
        0.33,
        out_name,
        fontsize=12,
        alpha=0.5,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})


def draw_base_element(
    base_value: float,
    ax: Any,  # noqa: ANN401
) -> None:
    """Draw the base-value marker and label on *ax*."""
    x_coords = np.array([base_value, base_value])
    y_coords = np.array([0.13, 0.25])
    line = lines.Line2D(x_coords, y_coords, lw=2.0, color="#F2F2F2")
    line.set_clip_on(False)
    ax.add_line(line)

    text_out_val = plt.text(
        base_value,
        0.33,
        "base value",
        fontsize=12,
        alpha=0.5,
        horizontalalignment="center",
    )
    text_out_val.set_bbox({"facecolor": "white", "edgecolor": "white"})


def draw_higher_lower_element(out_value: float, offset_text: float) -> None:
    """Draw the higher / lower legend arrows around the output value."""
    plt.text(
        out_value - offset_text,
        0.405,
        "higher",
        fontsize=13,
        color="#FF0D57",
        horizontalalignment="right",
    )
    plt.text(
        out_value + offset_text,
        0.405,
        "lower",
        fontsize=13,
        color="#1E88E5",
        horizontalalignment="left",
    )
    plt.text(
        out_value,
        0.4,
        r"$\leftarrow$",
        fontsize=13,
        color="#1E88E5",
        horizontalalignment="center",
    )
    plt.text(
        out_value,
        0.425,
        r"$\rightarrow$",
        fontsize=13,
        color="#FF0D57",
        horizontalalignment="center",
    )


def _update_axis_limits(
    ax: Any,  # noqa: ANN401
    total_pos: float,
    pos_features: NDArray[Any],
    total_neg: float,
    neg_features: NDArray[Any],
    base_value: float,
    out_value: float,
) -> None:
    """Set x/y limits and hide non-top spines for the force layout."""
    ax.set_ylim(-0.5, 0.15)
    padding = float(np.max([abs(total_pos) * 0.2, abs(total_neg) * 0.2]))
    if padding == 0.0:
        padding = 0.1

    if len(pos_features) > 0:
        min_x = min(float(np.min(pos_features[:, 0].astype(float))), base_value)
        min_x -= padding
    else:
        min_x = out_value - padding

    if len(neg_features) > 0:
        max_x = max(float(np.max(neg_features[:, 0].astype(float))), base_value)
        max_x += padding
    else:
        max_x = out_value + padding

    ax.set_xlim(min_x, max_x)

    plt.tick_params(
        top=True,
        bottom=False,
        left=False,
        right=False,
        labelleft=False,
        labeltop=True,
        labelbottom=False,
    )
    plt.locator_params(axis="x", nbins=12)

    for key, spine in zip(
        plt.gca().spines.keys(), plt.gca().spines.values(), strict=False
    ):
        if key != "top":
            spine.set_visible(False)


def _draw_bars(
    out_value: float,
    features: NDArray[Any],
    feature_type: str,
    width_separators: float,
    width_bar: float,
) -> tuple[list[Any], list[Any]]:
    """Draw the contribution bars and separators.

    Parameters
    ----------
    out_value :
        Model output (link space).
    features :
        Rows of ``(plot_position, feature_value, name)``.
    feature_type :
        ``"positive"`` or ``"negative"``.
    width_separators :
        Horizontal size of the chevron separators.
    width_bar :
        Vertical thickness of the bars.

    Returns
    -------
    rectangle_list, separator_list
        Matplotlib polygon patches.

    """
    rectangle_list: list[Any] = []
    separator_list: list[Any] = []
    pre_val = out_value

    for index, feature_values in enumerate(features):
        if feature_type == "positive":
            left_bound = float(feature_values[0])
            right_bound = pre_val
            pre_val = left_bound
            separator_indent = abs(width_separators)
            separator_pos = left_bound
            colors = [RED, "#ffb2c6"]
        else:
            left_bound = pre_val
            right_bound = float(feature_values[0])
            pre_val = right_bound
            separator_indent = -abs(width_separators)
            separator_pos = right_bound
            colors = [BLUE, "#b2dcff"]

        if index == 0:
            if feature_type == "positive":
                points_rectangle = [
                    [left_bound, 0],
                    [right_bound, 0],
                    [right_bound, width_bar],
                    [left_bound, width_bar],
                    [left_bound + separator_indent, (width_bar / 2)],
                ]
            else:
                points_rectangle = [
                    [right_bound, 0],
                    [left_bound, 0],
                    [left_bound, width_bar],
                    [right_bound, width_bar],
                    [right_bound + separator_indent, (width_bar / 2)],
                ]
        else:
            points_rectangle = [
                [left_bound, 0],
                [right_bound, 0],
                [right_bound + separator_indent * 0.90, (width_bar / 2)],
                [right_bound, width_bar],
                [left_bound, width_bar],
                [left_bound + separator_indent * 0.90, (width_bar / 2)],
            ]

        poly = plt.Polygon(  # type: ignore[attr-defined]
            points_rectangle,
            closed=True,
            fill=True,
            facecolor=colors[0],
            linewidth=0,
        )
        rectangle_list.append(poly)

        points_separator = [
            [separator_pos, 0],
            [separator_pos + separator_indent, (width_bar / 2)],
            [separator_pos, width_bar],
        ]
        sep = plt.Polygon(  # type: ignore[attr-defined]
            points_separator,
            closed=None,  # type: ignore[arg-type]
            fill=None,
            edgecolor=colors[1],
            lw=3,
        )
        separator_list.append(sep)

    return rectangle_list, separator_list


def _draw_labels(
    fig: Figure,
    ax: Any,  # noqa: ANN401
    out_value: float,
    features: NDArray[Any],
    feature_type: str,
    offset_text: float,
    total_effect: float = 0.0,
    min_perc: float = 0.05,
    text_rotation: float = 0.0,
    max_display: int | None = None,
) -> tuple[Figure, Any]:
    """Draw feature name labels under the force bars."""
    start_text = out_value
    pre_val = out_value

    if feature_type == "positive":
        colors = [RED, "#ffb2c6"]
        alignment = "right"
        sign = 1
    else:
        colors = [BLUE, "#b2dcff"]
        alignment = "left"
        sign = -1

    if feature_type == "positive":
        x_coords = np.array([pre_val, pre_val])
        y_coords = np.array([0.0, -0.18])
        line = lines.Line2D(x_coords, y_coords, lw=1.0, alpha=0.5, color=colors[0])
        line.set_clip_on(False)
        ax.add_line(line)
        start_text = pre_val

    box_end = out_value
    val = out_value
    for _i, feature in enumerate(features):
        if max_display is not None and _i >= max_display:
            break
        if abs(total_effect) > 0.0:
            feature_contribution = abs(float(feature[0]) - pre_val) / abs(total_effect)
        else:
            feature_contribution = 0.0
        if feature_contribution < min_perc:
            break

        val = float(feature[0])
        text = str(feature[2]) if feature[1] == "" else f"{feature[2]} = {feature[1]}"

        va_alignment = "top" if text_rotation != 0 else "baseline"

        text_out_val = plt.text(
            start_text - sign * offset_text,
            -0.15,
            text,
            fontsize=12,
            color=colors[0],
            horizontalalignment=alignment,
            va=va_alignment,
            rotation=text_rotation,
        )
        text_out_val.set_bbox({"facecolor": "none", "edgecolor": "none"})

        fig.canvas.draw()
        box_size = (
            text_out_val.get_bbox_patch()  # type: ignore[union-attr]
            .get_extents()
            .transformed(ax.transData.inverted())
        )
        if feature_type == "positive":
            box_end_ = box_size.get_points()[0][0]
        else:
            box_end_ = box_size.get_points()[1][0]

        if (sign * box_end_) > (sign * val):
            x_coords = np.array([val, val])
            y_coords = np.array([0.0, -0.18])
            line = lines.Line2D(x_coords, y_coords, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = val
            box_end = val
        else:
            box_end = box_end_ - sign * offset_text
            x_coords = np.array([val, box_end, box_end])
            y_coords = np.array([0.0, -0.08, -0.18])
            line = lines.Line2D(x_coords, y_coords, lw=1.0, alpha=0.5, color=colors[0])
            line.set_clip_on(False)
            ax.add_line(line)
            start_text = box_end

        pre_val = float(feature[0])

    path_pts = [
        [out_value, 0],
        [pre_val, 0],
        [box_end, -0.08],
        [box_end, -0.2],
        [out_value, -0.2],
        [out_value, 0],
    ]
    path = MatplotlibPath(path_pts)
    patch = PathPatch(path, facecolor="none", edgecolor="none")
    ax.add_patch(patch)

    lower_lim, upper_lim = ax.get_xlim()
    if box_end < lower_lim:
        ax.set_xlim(box_end, upper_lim)
    if box_end > upper_lim:
        ax.set_xlim(lower_lim, box_end)

    if feature_type == "positive":
        cmap_colors = np.array([(255, 13, 87), (255, 255, 255)]) / 255.0
    else:
        cmap_colors = np.array([(30, 136, 229), (255, 255, 255)]) / 255.0

    cm = matplotlib.colors.LinearSegmentedColormap.from_list("cm", cmap_colors)
    _, z2 = np.meshgrid(np.linspace(0, 10), np.linspace(-10, 10))
    extent_shading = [out_value, box_end, 0, -0.31]
    if extent_shading[0] == extent_shading[1]:
        extent_shading[1] += 1e-6
    im = plt.imshow(
        z2,
        interpolation="quadric",
        cmap=cm,
        vmax=0.01,
        alpha=0.3,
        origin="lower",
        extent=extent_shading,  # type: ignore[arg-type]
        clip_path=patch,
        clip_on=True,
        aspect="auto",
    )
    im.set_clip_path(patch)

    return fig, ax


def _draw_additive_plot(
    data: dict[str, Any],
    figsize: tuple[float, float],
    show: bool,
    text_rotation: float = 0.0,
    min_perc: float = 0.05,
    max_display: int | None = None,
) -> Figure:
    """Compose the full matplotlib additive force plot.

    Parameters
    ----------
    data :
        Force-plot data dict (see ``_format_data``).
    figsize :
        Figure size in inches.
    show :
        Accepted for API compatibility; display is handled by the caller.
    text_rotation :
        Degrees to rotate feature labels.
    min_perc :
        Minimum contribution fraction required to draw a label.
    max_display :
        Maximum number of feature labels to draw per side.

    Returns
    -------
    Figure
        The constructed matplotlib figure.

    """
    _ = show
    neg_features, total_neg, pos_features, total_pos = _format_data(data)

    base_value = float(data["baseValue"])
    out_value = float(data["outValue"])
    offset_text = (abs(total_neg) + abs(total_pos)) * 0.04

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.patch.set_facecolor("none")
    ax.patch.set_alpha(0.0)

    _update_axis_limits(
        ax, total_pos, pos_features, total_neg, neg_features, base_value, out_value
    )

    width_bar = 0.1
    width_separators = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 200

    rectangle_list, separator_list = _draw_bars(
        out_value, neg_features, "negative", width_separators, width_bar
    )
    for patch in rectangle_list:
        ax.add_patch(patch)
    for patch in separator_list:
        ax.add_patch(patch)

    rectangle_list, separator_list = _draw_bars(
        out_value, pos_features, "positive", width_separators, width_bar
    )
    for patch in rectangle_list:
        ax.add_patch(patch)
    for patch in separator_list:
        ax.add_patch(patch)

    total_effect = abs(total_neg) + total_pos
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value,
        neg_features,
        "negative",
        offset_text,
        total_effect,
        min_perc=min_perc,
        text_rotation=text_rotation,
        max_display=max_display,
    )
    fig, ax = _draw_labels(
        fig,
        ax,
        out_value,
        pos_features,
        "positive",
        offset_text,
        total_effect,
        min_perc=min_perc,
        text_rotation=text_rotation,
        max_display=max_display,
    )

    draw_higher_lower_element(out_value, offset_text)
    draw_base_element(base_value, ax)

    out_names = data["outNames"][0]
    draw_output_element(out_names, out_value, ax)

    if data["link"] == "logit":
        plt.xscale("logit")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.ticklabel_format(style="plain")

    return fig


def force(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | str | None:
    """Visualise attributions with an additive force layout.

    Features pushing the prediction higher are drawn in red to the left of the
    output marker; features pushing it lower are drawn in blue to the right.

    The ``"html"`` backend replaces SHAP's JavaScript force plot with a static,
    self-contained HTML bar that needs no ``initjs()``.

    Parameters
    ----------
    explanation :
        A single-instance Explanation or XWhy result.
    max_display :
        Maximum number of features to label.
    show :
        Whether to display the result.
    save_path :
        Optional path to write the figure or HTML document to.
    backend :
        ``"matplotlib"`` or ``"html"``.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches.
    **kwargs :
        Ignored extras accepted for SHAP call compatibility
        (``text_rotation``, ``contribution_threshold``).

    Returns
    -------
    Figure | str | None
        HTML markup for the ``"html"`` backend, otherwise the matplotlib
        figure when ``show`` is False, else ``None``.

    """
    if _check_backend is not None:
        engine = _check_backend(backend, frozenset({"matplotlib", "html"}))
    else:
        if backend not in {"matplotlib", "html"}:
            msg = f"Unsupported backend: {backend}"
            raise ValueError(msg)
        engine = backend

    exp = _as_explanation(explanation) if _as_explanation is not None else explanation
    if _single_instance is not None:
        values, base_value, names, data = _single_instance(exp)
    else:
        values = np.asarray(exp.values, dtype=float).ravel()
        base_value = float(exp.base_values)
        names = list(exp.feature_names)
        data = getattr(exp, "data", None)

    prediction = float(base_value) + float(values.sum())

    labels = list(names)
    if data is not None:
        raw = np.asarray(data, dtype=object).ravel()
        if raw.shape[0] == len(names):
            labels = [
                f"{names[i]} = {_format_value(raw[i])}" for i in range(len(names))
            ]

    order = np.argsort(np.abs(values))[::-1]
    if max_display is not None and 0 < max_display < len(order):
        order = order[:max_display]

    positives = [(labels[i], float(values[i])) for i in order if values[i] > 0]
    negatives = [(labels[i], float(values[i])) for i in order if values[i] < 0]

    if engine == "html":
        html = _force_html(
            positives, negatives, float(base_value), prediction, title=title
        )
        if _finish_html is not None:
            return _finish_html(html, show=show, save_path=save_path)
        return None if show else html

    features_dict: dict[str, dict[str, float | str]] = {}
    if data is not None:
        raw = np.asarray(data, dtype=object).ravel()
    else:
        raw = np.array([], dtype=object)

    for i in range(len(names)):
        if len(raw) == len(names) and isinstance(raw[i], (int, float, np.number)):
            feat_value: float | str = float(raw[i])
        elif len(raw) == len(names):
            feat_value = str(raw[i])
        else:
            feat_value = ""
        features_dict[str(i)] = {
            "effect": float(values[i]),
            "value": feat_value,
        }

    force_data: dict[str, Any] = {
        "outNames": ["f(x)"],
        "baseValue": float(base_value),
        "outValue": float(prediction),
        "link": "identity",
        "featureNames": {str(i): names[i] for i in range(len(names))},
        "features": features_dict,
    }

    text_rotation = float(kwargs.get("text_rotation", 0))
    min_perc = float(kwargs.get("contribution_threshold", 0.05))

    fig_w = figsize[0] if figsize else 20.0
    fig_h = figsize[1] if figsize else 3.0
    fig = _draw_additive_plot(
        force_data,
        figsize=(fig_w, fig_h),
        show=show,
        text_rotation=text_rotation,
        min_perc=min_perc,
        max_display=max_display,
    )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig
