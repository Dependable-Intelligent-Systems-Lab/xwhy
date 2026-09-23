"""Text token attribution visualizations."""

from __future__ import annotations

import random
import string
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import (
    RED_TRANSPARENT_BLUE,
    _as_explanation,
    _finish_html,
)


def _css_rgba(r: float, g: float, b: float, a: float) -> str:
    """Format an ``rgba()`` color for HTML/CSS.

    NumPy 2 scalar types stringify as ``np.float64(...)``, which browsers reject;
    CSS requires plain numeric literals.

    Parameters
    ----------
    r, g, b :
        Color channel values in the range ``[0, 255]`` (or any float convertible
        to that range).
    a :
        Alpha channel in the range ``[0, 1]``.

    Returns
    -------
    str
        A CSS ``rgba(r, g, b, a)`` string with plain float literals.

    """
    return f"rgba({float(r)}, {float(g)}, {float(b)}, {float(a)})"


def unpack_shap_explanation_contents(
    shap_values: Any,  # noqa: ANN401
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]] | None]:
    """Extract attribution values and optional clustering from a SHAP object.

    Prefers ``hierarchical_values`` when present; otherwise falls back to
    ``values``.  Clustering is returned unchanged (may be ``None``).

    Parameters
    ----------
    shap_values :
        An object exposing ``.values`` (and optionally ``.hierarchical_values``
        and ``.clustering``).

    Returns
    -------
    tuple
        ``(values_array, clustering_or_none)``.

    """
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)
    return np.asarray(values, dtype=float), clustering


def process_shap_values(
    tokens: list[str] | NDArray[Any],
    values: NDArray[np.floating[Any]],
    grouping_threshold: float,
    separator: str,
    clustering: NDArray[np.floating[Any]] | None = None,
    *,
    return_meta_data: bool = False,
) -> (
    tuple[NDArray[Any], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
    | tuple[
        NDArray[Any],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
        NDArray[np.floating[Any]],
    ]
):
    """Group tokens according to hierarchical SHAP interaction structure.

    When ``len(values) != len(tokens)`` a partition tree (``clustering``) is
    required.  Sub-trees whose interaction effects dominate are collapsed into
    single visual tokens.

    Parameters
    ----------
    tokens :
        Sequence of token strings (length ``M``).
    values :
        Attribution values.  May be longer than ``M`` when hierarchical.
    grouping_threshold :
        Collapse a group when the max child effect is smaller than
        ``interaction / group_size * grouping_threshold``.
    separator :
        String used to join tokens that are collapsed together.
    clustering :
        Optional ``(n_merges, 2)`` array of left/right child indices.
    return_meta_data :
        When ``True`` also return token-to-node mapping and collapsed node ids.

    Returns
    -------
    tokens, values, group_sizes
        Possibly regrouped tokens, corresponding values and sizes.
        With ``return_meta_data=True`` two extra arrays are appended.

    """
    token_list = list(tokens)
    m = len(token_list)

    if len(values) != m:
        if clustering is None:
            msg = (
                "The length of the attribution values must match the number of "
                "tokens if shap_values.clustering is None! When passing hierarchical "
                "attributions the clustering is also required."
            )
            raise ValueError(msg)

        groups: list[list[int]] = [[i] for i in range(m)]
        lower_values = np.zeros(len(values), dtype=float)
        lower_values[:m] = values[:m]
        max_values = np.zeros(len(values), dtype=float)
        max_values[:m] = np.abs(values[:m])

        for i in range(clustering.shape[0]):
            li = int(clustering[i, 0])
            ri = int(clustering[i, 1])
            groups.append(groups[li] + groups[ri])
            lower_values[m + i] = lower_values[li] + lower_values[ri] + values[m + i]
            max_values[i + m] = max(
                abs(values[m + i]) / len(groups[m + i]),
                max_values[li],
                max_values[ri],
            )

        upper_values = np.zeros(len(values), dtype=float)

        def lower_credit(
            upper: NDArray[np.floating[Any]],
            clust: NDArray[np.floating[Any]],
            node: int,
            value: float = 0.0,
        ) -> None:
            if node < m:
                upper[node] = value
                return
            li_c = int(clust[node - m, 0])
            ri_c = int(clust[node - m, 1])
            upper[node] = value
            value += float(values[node])
            lower_credit(upper, clust, li_c, value * 0.5)
            lower_credit(upper, clust, ri_c, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)
        group_values = lower_values + upper_values

        new_tokens: list[str] = []
        new_values: list[float] = []
        group_sizes_list: list[int] = []
        token_id_to_node_id_mapping = np.zeros(m, dtype=float)
        collapsed_node_ids: list[int] = []

        def merge_tokens(node: int) -> None:
            if 0 <= node < m:
                new_tokens.append(token_list[node])
                new_values.append(float(group_values[node]))
                group_sizes_list.append(1)
                collapsed_node_ids.append(node)
                token_id_to_node_id_mapping[node] = node
                return

            li = int(clustering[node - m, 0])
            ri = int(clustering[node - m, 1])
            dv = abs(float(values[node])) / len(groups[node])

            if max(max_values[li], max_values[ri]) < dv * grouping_threshold:
                joined = (
                    separator.join(token_list[g] for g in groups[li])
                    + separator
                    + separator.join(token_list[g] for g in groups[ri])
                )
                new_tokens.append(joined)
                new_values.append(float(group_values[node]))
                group_sizes_list.append(len(groups[node]))
                collapsed_node_ids.append(node)
                for g in groups[li]:
                    token_id_to_node_id_mapping[g] = node
                for g in groups[ri]:
                    token_id_to_node_id_mapping[g] = node
            else:
                merge_tokens(li)
                merge_tokens(ri)

        merge_tokens(len(group_values) - 1)

        out_tokens = np.array(new_tokens, dtype=object)
        out_values = np.array(new_values, dtype=float)
        out_sizes = np.array(group_sizes_list, dtype=float)
        mapping = np.asarray(token_id_to_node_id_mapping, dtype=float)
        collapsed = np.asarray(collapsed_node_ids, dtype=float)

        if return_meta_data:
            return out_tokens, out_values, out_sizes, mapping, collapsed
        return out_tokens, out_values, out_sizes

    # Non-hierarchical path
    group_sizes = np.ones(m, dtype=float)
    mapping = np.arange(m, dtype=float)
    collapsed = np.arange(m, dtype=float)

    tokens_arr = np.asarray(token_list, dtype=object)
    values_arr = np.asarray(values, dtype=float)

    if return_meta_data:
        return tokens_arr, values_arr, group_sizes, mapping, collapsed
    return tokens_arr, values_arr, group_sizes


def _values_min_max(
    values: NDArray[np.floating[Any]],
    base_values: float,
) -> tuple[float, float, float]:
    """Compute axis limits and colour scale from SHAP values.

    Parameters
    ----------
    values :
        1-D array of token attributions.
    base_values :
        Model base value for the explanation.

    Returns
    -------
    xmin, xmax, cmax
        Padded axis bounds and maximum absolute value for colour scaling.

    """
    fx = float(base_values) + float(values.sum())
    xmin = fx - float(values[values > 0].sum())
    xmax = fx - float(values[values < 0].sum())
    cmax = max(abs(float(values.min())), abs(float(values.max())))
    span = xmax - xmin
    xmin -= 0.1 * span
    xmax += 0.1 * span
    return xmin, xmax, cmax


def _encode_token(token: str) -> str:
    """HTML-escape a token and strip BERT-style ``##`` prefixes."""
    return token.replace("<", "&lt;").replace(">", "&gt;").replace(" ##", "")


# ---------------------------------------------------------------------------
# SVG force-plot renderer (presentation concern)
# ---------------------------------------------------------------------------


def svg_force_plot(
    values: NDArray[np.floating[Any]],
    base_values: float,
    fx: float,
    tokens: list[str] | NDArray[Any],
    uuid: str,
    xmin: float,
    xmax: float,
    output_name: str,
) -> str:
    """Render an interactive SVG force plot for one explanation.

    Parameters
    ----------
    values :
        Token-level SHAP values (1-D).
    base_values :
        Model expected value.
    fx :
        Model output for the instance (``base_values + values.sum()``).
    tokens :
        Token strings (already HTML-safe preferred).
    uuid :
        Unique prefix for DOM element ids.
    xmin, xmax :
        Axis bounds.
    output_name :
        Label shown next to the final prediction tick.

    Returns
    -------
    str
        SVG markup as a string.

    """
    token_list = [_encode_token(str(t)) for t in tokens]

    def xpos(xval: float) -> float:
        return 100.0 * (xval - xmin) / (xmax - xmin + 1e-8)

    parts: list[str] = ['<svg width="100%" height="80px">']

    # ---- x-axis -----------------------------------------------------------
    parts.append(
        '<line x1="0" y1="33" x2="100%" y2="33" '
        'style="stroke:rgb(150,150,150);stroke-width:1" />'
    )

    def draw_tick_mark(
        xval: float,
        label: str | None = None,
        *,
        bold: bool = False,
        backing: bool = False,
    ) -> str:
        x_pct = xpos(xval)
        tick = (
            f'<line x1="{x_pct}%" y1="33" x2="{x_pct}%" y2="37" '
            'style="stroke:rgb(150,150,150);stroke-width:1" />'
        )
        if bold:
            # White stroke behind bold value for contrast on any background.
            tick += (
                f'<text x="{x_pct}%" y="27" font-size="13px" '
                'style="stroke:#ffffff;stroke-width:8px;" '
                'font-weight="bold" fill="rgb(255,255,255)" '
                f'dominant-baseline="bottom" text-anchor="middle">'
                f"{xval:g}</text>"
            )
            tick += (
                f'<text x="{x_pct}%" y="27" font-size="13px" '
                'font-weight="bold" fill="rgb(0,0,0)" '
                f'dominant-baseline="bottom" text-anchor="middle">'
                f"{xval:g}</text>"
            )
        else:
            if backing:
                tick += (
                    f'<text x="{x_pct}%" y="27" font-size="13px" '
                    'style="stroke:#ffffff;stroke-width:8px;" '
                    'fill="rgb(255,255,255)" '
                    f'dominant-baseline="bottom" text-anchor="middle">'
                    f"{xval:g}</text>"
                )
            tick += (
                f'<text x="{x_pct}%" y="27" font-size="12px" '
                'fill="rgb(120,120,120)" '
                f'dominant-baseline="bottom" text-anchor="middle">'
                f"{xval:g}</text>"
            )
        if label is not None:
            tick += (
                f'<text x="{x_pct}%" y="10" font-size="12px" '
                'fill="rgb(120,120,120)" '
                f'dominant-baseline="bottom" text-anchor="middle">'
                f"{label}</text>"
            )
        return tick

    log_span = round(1 - float(np.log10(xmax - xmin + 1e-8)))
    xcenter = round((xmax + xmin) / 2, log_span)
    parts.append(draw_tick_mark(xcenter))

    tick_interval = round((xmax - xmin) / 7, log_span)
    side_buffer = (xmax - xmin) / 14
    for i in range(1, 10):
        pos = xcenter - i * tick_interval
        if pos < xmin + side_buffer:
            break
        parts.append(draw_tick_mark(pos))
    for i in range(1, 10):
        pos = xcenter + i * tick_interval
        if pos > xmax - side_buffer:
            break
        parts.append(draw_tick_mark(pos))

    parts.append(draw_tick_mark(base_values, label="base value", backing=True))
    parts.append(
        draw_tick_mark(
            fx,
            bold=True,
            label=(
                f'f<tspan baseline-shift="sub" font-size="8px">{output_name}</tspan>'
                "(inputs)"
            ),
            backing=True,
        )
    )

    # ---- positive contributions (red) -------------------------------------
    red = tuple(float(x) for x in (np.array([1.0, 0.0, 81.0 / 255.0]) * 255))
    light_red = (255, 195, 213)

    pos_sum = float(values[values > 0].sum())
    x_start = fx - pos_sum
    w = 100.0 * pos_sum / (xmax - xmin + 1e-8)
    parts.append(
        f'<rect x="{xpos(x_start)}%" width="{w}%" y="40" height="18" '
        f'style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'
    )

    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for ind in inds:
        v = float(values[ind])
        pos -= v
        parts.append(
            f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" '
            'y1="60" y2="60" '
            f'id="_fb_{uuid}_ind_{ind}" '
            f'style="stroke:rgb{red};stroke-width:2; '
            'opacity: 0"/>'
        )
        mid = (xpos(last_pos) + xpos(pos)) / 2
        parts.append(
            f'<text x="{mid}%" y="71" font-size="12px" '
            f'id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" '
            'style="opacity: 0" '
            'dominant-baseline="middle" text-anchor="middle">'
            f"{values[ind].round(3)}</text>"
        )
        w_pct = xpos(last_pos) - xpos(pos)
        parts.append(
            f'<svg x="{xpos(pos)}%" y="40" height="20" '
            f'width="{w_pct}%">'
            '<svg x="0" y="0" width="100%" height="100%">'
            '<text x="50%" y="9" font-size="12px" '
            'fill="rgb(255,255,255)" '
            'dominant-baseline="middle" text-anchor="middle">'
            f"{token_list[ind].strip()}</text></svg></svg>"
        )
        last_pos = pos

    # divider padding (positive)
    pos = fx
    for i, ind in enumerate(inds):
        v = float(values[ind])
        pos -= v
        if i != 0:
            lp = xpos(last_pos)
            parts.extend(
                (
                    f'<g transform="translate({2 * j - 8},0)">'
                    f'<svg x="{lp}%" y="40" height="18" '
                    'overflow="visible" width="30">'
                    '<path d="M 0 -9 l 6 18 L 0 25" fill="none" '
                    f'style="stroke:rgb{red};stroke-width:2" />'
                    "</svg></g>"
                )
                for j in range(4)
            )
        if i + 1 != len(inds):
            pp = xpos(pos)
            parts.extend(
                (
                    f'<g transform="translate({2 * j},0)">'
                    f'<svg x="{pp}%" y="40" height="18" '
                    'overflow="visible" width="30">'
                    '<path d="M 0 -9 l 6 18 L 0 25" fill="none" '
                    f'style="stroke:rgb{red};stroke-width:2" />'
                    "</svg></g>"
                )
                for j in range(4)
            )
        last_pos = pos

    parts.append(
        f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" '
        f'height="18" style="fill:rgb{red}"/>'
    )

    pos = fx - pos_sum
    parts.append(
        '<g transform="translate(-11.5,0)">'
        f'<svg x="{xpos(pos)}%" y="40" height="18" '
        'overflow="visible" width="30">'
        '<path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" '
        'style="stroke:rgb(255,255,255);stroke-width:2" /></svg></g>'
    )

    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = float(values[ind])
        pos -= v
        if i + 1 != len(inds):
            lp = xpos(last_pos)
            parts.append(
                '<g transform="translate(-1.5,0)">'
                f'<svg x="{lp}%" y="40" height="18" '
                'overflow="visible" width="30">'
                '<path d="M 0 -9 l 6 18 L 0 25" fill="none" '
                f'style="stroke:rgb{light_red};stroke-width:2" />'
                "</svg></g>"
            )
        tp = f"_tp_{uuid}_ind_{ind}"
        fs = f"_fs_{uuid}_ind_{ind}"
        fb = f"_fb_{uuid}_ind_{ind}"
        parts.append(
            f'<rect x="{xpos(pos)}%" y="40" height="20" '
            f'width="{xpos(last_pos) - xpos(pos)}%" '
            f"onmouseover=\"document.getElementById('{tp}').style"
            ".textDecoration = 'underline';"
            f"document.getElementById('{fs}').style.opacity = 1;"
            f"document.getElementById('{fb}').style.opacity = 1;\" "
            f"onmouseout=\"document.getElementById('{tp}').style"
            ".textDecoration = 'none';"
            f"document.getElementById('{fs}').style.opacity = 0;"
            f"document.getElementById('{fb}').style.opacity = 0;\" "
            'style="fill:rgb(0,0,0,0)" />'
        )
        last_pos = pos

    # ---- negative contributions (blue) ------------------------------------
    blue = tuple(
        float(x) for x in (np.array([0.0, 138.0 / 255.0, 251.0 / 255.0]) * 255)
    )
    light_blue = (208, 230, 250)

    neg_sum = -float(values[values < 0].sum())
    w = 100.0 * neg_sum / (xmax - xmin + 1e-8)
    parts.append(
        f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" '
        f'style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'
    )

    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for ind in inds:
        v = float(values[ind])
        pos -= v
        parts.append(
            f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" '
            'y1="60" y2="60" '
            f'id="_fb_{uuid}_ind_{ind}" '
            f'style="stroke:rgb{blue};stroke-width:2; '
            'opacity: 0"/>'
        )
        mid = (xpos(last_pos) + xpos(pos)) / 2
        parts.append(
            f'<text x="{mid}%" y="71" font-size="12px" '
            f'fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" '
            'style="opacity: 0" '
            'dominant-baseline="middle" text-anchor="middle">'
            f"{values[ind].round(3)}</text>"
        )
        w_pct = xpos(pos) - xpos(last_pos)
        parts.append(
            f'<svg x="{xpos(last_pos)}%" y="40" height="20" '
            f'width="{w_pct}%">'
            '<svg x="0" y="0" width="100%" height="100%">'
            '<text x="50%" y="9" font-size="12px" '
            'fill="rgb(255,255,255)" '
            'dominant-baseline="middle" text-anchor="middle">'
            f"{token_list[ind].strip()}</text></svg></svg>"
        )
        last_pos = pos

    pos = fx
    for i, ind in enumerate(inds):
        v = float(values[ind])
        pos -= v
        if i != 0:
            lp = xpos(last_pos)
            parts.extend(
                (
                    f'<g transform="translate({-2 * j + 2},0)">'
                    f'<svg x="{lp}%" y="40" height="18" '
                    'overflow="visible" width="30">'
                    '<path d="M 8 -9 l -6 18 L 8 25" fill="none" '
                    f'style="stroke:rgb{blue};stroke-width:2" />'
                    "</svg></g>"
                )
                for j in range(4)
            )
        if i + 1 != len(inds):
            pp = xpos(pos)
            parts.extend(
                (
                    f'<g transform="translate(-{2 * j + 8},0)">'
                    f'<svg x="{pp}%" y="40" height="18" '
                    'overflow="visible" width="30">'
                    '<path d="M 8 -9 l -6 18 L 8 25" fill="none" '
                    f'style="stroke:rgb{blue};stroke-width:2" />'
                    "</svg></g>"
                )
                for j in range(4)
            )
        last_pos = pos

    parts.append(
        f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" '
        f'height="18" style="fill:rgb{blue}"/>'
    )

    pos = fx - float(values[values < 0].sum())
    parts.append(
        '<g transform="translate(-6.0,0)">'
        f'<svg x="{xpos(pos)}%" y="40" height="18" '
        'overflow="visible" width="30">'
        '<path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" '
        'style="stroke:rgb(255,255,255);stroke-width:2" /></svg></g>'
    )

    pos = fx
    last_pos = pos
    for i, ind in enumerate(inds):
        v = float(values[ind])
        pos -= v
        if i + 1 != len(inds):
            pp = xpos(pos)
            parts.append(
                '<g transform="translate(-6.0,0)">'
                f'<svg x="{pp}%" y="40" height="18" '
                'overflow="visible" width="30">'
                '<path d="M 8 -9 l -6 18 L 8 25" fill="none" '
                f'style="stroke:rgb{light_blue};stroke-width:2" />'
                "</svg></g>"
            )
        tp = f"_tp_{uuid}_ind_{ind}"
        fs = f"_fs_{uuid}_ind_{ind}"
        fb = f"_fb_{uuid}_ind_{ind}"
        parts.append(
            f'<rect x="{xpos(last_pos)}%" y="40" height="20" '
            f'width="{xpos(pos) - xpos(last_pos)}%" '
            f"onmouseover=\"document.getElementById('{tp}').style"
            ".textDecoration = 'underline';"
            f"document.getElementById('{fs}').style.opacity = 1;"
            f"document.getElementById('{fb}').style.opacity = 1;\" "
            f"onmouseout=\"document.getElementById('{tp}').style"
            ".textDecoration = 'none';"
            f"document.getElementById('{fs}').style.opacity = 0;"
            f"document.getElementById('{fb}').style.opacity = 0;\" "
            'style="fill:rgb(0,0,0,0)" />'
        )
        last_pos = pos

    parts.append("</svg>")
    return "".join(parts)


def text(
    explanation: Any,  # noqa: ANN401
    num_starting_labels: int = 0,
    grouping_threshold: float = 0.01,
    separator: str = "",
    xmin: float | None = None,
    xmax: float | None = None,
    cmax: float | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> str | None:
    """Plot an explanation of a string of text using colouring and interactive labels.

    The output is interactive HTML; click any token to toggle its SHAP value.

    Parameters
    ----------
    explanation :
        A SHAP ``Explanation`` (or compatible) object.
    num_starting_labels :
        Number of highest-|SHAP| tokens whose value labels are shown initially.
        ``0`` hides all labels until clicked.
    grouping_threshold :
        Collapse interaction-dominated sub-trees when the child effect is smaller
        than this fraction of the parent interaction (Partition explainer).
    separator :
        Joiner for tokens collapsed by interaction grouping.
    xmin, xmax :
        Optional fixed axis bounds for the force plot.
    cmax :
        Optional fixed colour-scale maximum (absolute SHAP value).
    show :
        When ``True`` display the HTML (via ``_finish_html``); otherwise return it.
    save_path :
        Optional path to write the HTML.
    title :
        Optional heading rendered above the plot.
    **kwargs :
        Accepted for forward compatibility; currently ignored.

    Returns
    -------
    str or None
        HTML string when ``show=False``, otherwise ``None``.

    """
    _ = kwargs  # reserved for future options
    shap_values = (
        _as_explanation(explanation) if _as_explanation is not None else explanation
    )
    if getattr(shap_values, "data", None) is None:
        shap_values.data = shap_values.feature_names

    uuid = "".join(random.choices(string.ascii_lowercase, k=20))

    # ---- multi-row (batch) ------------------------------------------------
    if len(shap_values.shape) == 2 and (
        shap_values.output_names is None or isinstance(shap_values.output_names, str)
    ):
        xmin_b, xmax_b, cmax_b = 0.0, 0.0, 0.0
        for i, v in enumerate(shap_values):  # type: ignore[var-annotated, arg-type]
            values, clustering = unpack_shap_explanation_contents(v)
            tokens, values, _gs = process_shap_values(  # type: ignore[misc]
                v.data, values, grouping_threshold, separator, clustering
            )
            if i == 0:
                xmin_b, xmax_b, cmax_b = _values_min_max(values, float(v.base_values))
                continue
            xmin_i, xmax_i, cmax_i = _values_min_max(values, float(v.base_values))
            xmin_b = min(xmin_b, xmin_i)
            xmax_b = max(xmax_b, xmax_i)
            cmax_b = max(cmax_b, cmax_i)

        out = ""
        for i, v in enumerate(shap_values):  # type: ignore[arg-type]
            out += (
                "<br>\n"
                '<hr style="height: 1px; background-color: #fff; border: none; '
                "margin-top: 18px; margin-bottom: 18px; "
                'border-top: 1px dashed #ccc;">\n'
                '<div align="center" style="margin-top: -35px;">'
                '<div style="display: inline-block; background: #fff; padding: 5px; '
                f'color: #999; font-family: monospace">[{i}]</div></div>\n'
            )
            out += (
                text(
                    v,
                    num_starting_labels=num_starting_labels,
                    grouping_threshold=grouping_threshold,
                    separator=separator,
                    xmin=xmin_b,
                    xmax=xmax_b,
                    cmax=cmax_b,
                    show=False,
                )
                or ""
            )
        if title:
            out = (
                f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
                f"{escape(title)}</div>" + out
            )
        if _finish_html is not None:
            _finish_html(out, show=show, save_path=save_path)
        return None if show else out

    # ---- multi-output -----------------------------------------------------
    if len(shap_values.shape) == 2 and shap_values.output_names is not None:
        xmin_c: float | None = None
        xmax_c: float | None = None
        cmax_c: float | None = None

        for i in range(shap_values.shape[-1]):
            values, clustering = unpack_shap_explanation_contents(shap_values[:, i])
            tokens, values, _gs = process_shap_values(  # type: ignore[misc]
                shap_values[:, i].data,  # type: ignore[arg-type]
                values,
                grouping_threshold,
                separator,
                clustering,
            )
            xmin_i, xmax_i, cmax_i = _values_min_max(
                values, float(shap_values[:, i].base_values)
            )
            xmin_c = xmin_i if xmin_c is None else min(xmin_c, xmin_i)
            xmax_c = xmax_i if xmax_c is None else max(xmax_c, xmax_i)
            cmax_c = cmax_i if cmax_c is None else max(cmax_c, cmax_i)

        if xmin is None:
            xmin = xmin_c
        if xmax is None:
            xmax = xmax_c
        if cmax is None:
            cmax = cmax_c

        out = f"""<div align='center'>
<script>
    document._hover_{uuid} = '_tp_{uuid}_output_0';
    document._zoom_{uuid} = undefined;
    function _output_onclick_{uuid}(i) {{
        var next_id = undefined;
        if (document._zoom_{uuid} !== undefined) {{
            document.getElementById(
                document._zoom_{uuid}+ '_zoom'
            ).style.display = 'none';
            if (document._zoom_{uuid} === '_tp_{uuid}_output_' + i) {{
                document.getElementById(document._zoom_{uuid}).style.display = 'block';
                document.getElementById(
                    document._zoom_{uuid}+'_name'
                ).style.borderBottom = '3px solid #000000';
            }} else {{
                document.getElementById(document._zoom_{uuid}).style.display = 'none';
                document.getElementById(
                    document._zoom_{uuid}+'_name'
                ).style.borderBottom = 'none';
            }}
        }}
        if (document._zoom_{uuid} !== '_tp_{uuid}_output_' + i) {{
            next_id = '_tp_{uuid}_output_' + i;
            document.getElementById(next_id).style.display = 'none';
            document.getElementById(next_id + '_zoom').style.display = 'block';
            document.getElementById(
                next_id+'_name'
            ).style.borderBottom = '3px solid #000000';
        }}
        document._zoom_{uuid} = next_id;
    }}
    function _output_onmouseover_{uuid}(i, el) {{
        if (document._zoom_{uuid} !== undefined) {{ return; }}
        if (document._hover_{uuid} !== undefined) {{
            document.getElementById(
                document._hover_{uuid} + '_name'
            ).style.borderBottom = 'none';
            document.getElementById(document._hover_{uuid}).style.display = 'none';
        }}
        document.getElementById('_tp_{uuid}_output_' + i).style.display = 'block';
        el.style.borderBottom = '3px solid #000000';
        document._hover_{uuid} = '_tp_{uuid}_output_' + i;
    }}
</script>
<div style="color: rgb(120,120,120); font-size: 12px;">outputs</div>"""

        output_values = shap_values.values.sum(0) + shap_values.base_values
        output_max = float(np.max(np.abs(output_values)))
        for i, name in enumerate(shap_values.output_names):
            scaled = 0.5 + 0.5 * float(output_values[i]) / (output_max + 1e-8)
            color = (
                RED_TRANSPARENT_BLUE(scaled)
                if RED_TRANSPARENT_BLUE
                else (1.0, 0.0, 0.0, 0.5)
            )
            rgba_css = _css_rgba(
                color[0] * 255, color[1] * 255, color[2] * 255, color[3]
            )
            border = "3px solid #000000" if i == 0 else "none"
            out += (
                f'<div style="display: inline; border-bottom: {border}; '
                f'background: {rgba_css}; border-radius: 3px; padding: 0px" '
                f'id="_tp_{uuid}_output_{i}_name" '
                f'onclick="_output_onclick_{uuid}({i})" '
                f'onmouseover="_output_onmouseover_{uuid}({i}, this);">{name}</div>'
            )
        out += "<br><br>"
        for i, _name in enumerate(shap_values.output_names):
            display = "block" if i == 0 else "none"
            out += f"<div id='_tp_{uuid}_output_{i}' style='display: {display};'>"
            out += (
                text(
                    shap_values[:, i],
                    num_starting_labels=num_starting_labels,
                    grouping_threshold=grouping_threshold,
                    separator=separator,
                    xmin=xmin,
                    xmax=xmax,
                    cmax=cmax,
                    show=False,
                )
                or ""
            )
            out += "</div>"
            out += f"<div id='_tp_{uuid}_output_{i}_zoom' style='display: none;'>"
            out += (
                text(
                    shap_values[:, i],
                    num_starting_labels=num_starting_labels,
                    grouping_threshold=grouping_threshold,
                    separator=separator,
                    show=False,
                )
                or ""
            )
            out += "</div>"
        out += "</div>"
        if title:
            out = (
                f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
                f"{escape(title)}</div>" + out
            )
        if _finish_html is not None:
            _finish_html(out, show=show, save_path=save_path)
        return None if show else out

    # ---- 3-D (batch x tokens x outputs) -----------------------------------
    if len(shap_values.shape) == 3:
        xmin_d: float | None = None
        xmax_d: float | None = None
        cmax_d: float | None = None

        for i in range(shap_values.shape[-1]):
            for j in range(shap_values.shape[0]):
                values, clustering = unpack_shap_explanation_contents(
                    shap_values[j, :, i]
                )
                tokens, values, _gs = process_shap_values(  # type: ignore[misc]
                    shap_values[j, :, i].data,  # type: ignore[arg-type]
                    values,
                    grouping_threshold,
                    separator,
                    clustering,
                )
                xmin_i, xmax_i, cmax_i = _values_min_max(
                    values, float(shap_values[j, :, i].base_values)
                )
                xmin_d = xmin_i if xmin_d is None else min(xmin_d, xmin_i)
                xmax_d = xmax_i if xmax_d is None else max(xmax_d, xmax_i)
                cmax_d = cmax_i if cmax_d is None else max(cmax_d, cmax_i)

        if xmin is None:
            xmin = xmin_d
        if xmax is None:
            xmax = xmax_d
        if cmax is None:
            cmax = cmax_d

        out = ""
        for i, v in enumerate(shap_values):  # type: ignore[arg-type]
            out += (
                "<br>\n"
                '<hr style="height: 1px; background-color: #fff; border: none; '
                "margin-top: 18px; margin-bottom: 18px; "
                'border-top: 1px dashed #ccc;">\n'
                '<div align="center" style="margin-top: -35px;">'
                '<div style="display: inline-block; background: #fff; padding: 5px; '
                f'color: #999; font-family: monospace">[{i}]</div></div>\n'
            )
            out += (
                text(
                    v,
                    num_starting_labels=num_starting_labels,
                    grouping_threshold=grouping_threshold,
                    separator=separator,
                    xmin=xmin,
                    xmax=xmax,
                    cmax=cmax,
                    show=False,
                )
                or ""
            )
        if title:
            out = (
                f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
                f"{escape(title)}</div>" + out
            )
        if _finish_html is not None:
            _finish_html(out, show=show, save_path=save_path)
        return None if show else out

    # ---- single explanation -----------------------------------------------
    xmin_new, xmax_new, cmax_new = _values_min_max(
        shap_values.values, float(shap_values.base_values)
    )
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new

    values, clustering = unpack_shap_explanation_contents(shap_values)
    tokens, values, group_sizes = process_shap_values(  # type: ignore[misc]
        shap_values.data,  # type: ignore[arg-type]
        values,
        grouping_threshold,
        separator,
        clustering,
    )

    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    encoded_tokens = [_encode_token(str(t)) for t in tokens]
    output_name = (
        shap_values.output_names if isinstance(shap_values.output_names, str) else ""
    )

    out = svg_force_plot(
        values,
        float(shap_values.base_values),
        float(shap_values.base_values) + float(values.sum()),
        encoded_tokens,
        uuid,
        xmin,
        xmax,
        output_name,
    )
    out += (
        "<div align='center'><div style=\"color: rgb(120,120,120); "
        'font-size: 12px; margin-top: -15px;">inputs</div>'
    )

    for i, token in enumerate(tokens):
        scaled = 0.5 + 0.5 * float(values[i]) / (cmax + 1e-8)
        color = (
            RED_TRANSPARENT_BLUE(scaled)
            if RED_TRANSPARENT_BLUE
            else (1.0, 0.0, 0.0, 0.5)
        )
        rgba_css = _css_rgba(color[0] * 255, color[1] * 255, color[2] * 255, color[3])

        label_display = "block" if i in top_inds else "none"
        wrapper_display = "inline-block" if i in top_inds else "inline"

        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))
        else:
            value_label = f"{values[i].round(3)} / {int(group_sizes[i])}"

        safe_token = _encode_token(str(token))
        fb = f"_fb_{uuid}_ind_{i}"
        fs = f"_fs_{uuid}_ind_{i}"
        out += (
            f"<div style='display: {wrapper_display}; "
            "text-align: center;'>"
            f"<div style='display: {label_display}; color: #999; "
            f"padding-top: 0px; font-size: 12px;'>{value_label}</div>"
            f"<div id='_tp_{uuid}_ind_{i}' "
            f"style='display: inline; background: {rgba_css}; "
            "border-radius: 3px; padding: 0px' "
            'onclick="'
            "if (this.previousSibling.style.display == 'none') {"
            "this.previousSibling.style.display = 'block';"
            "this.parentNode.style.display = 'inline-block';"
            "} else {"
            "this.previousSibling.style.display = 'none';"
            "this.parentNode.style.display = 'inline';"
            '}" '
            f"onmouseover=\"document.getElementById('{fb}')"
            ".style.opacity = 1; "
            f"document.getElementById('{fs}').style.opacity = 1;\" "
            f"onmouseout=\"document.getElementById('{fb}')"
            ".style.opacity = 0; "
            f"document.getElementById('{fs}').style.opacity = 0;\" "
            f">{safe_token}</div></div>"
        )
    out += "</div>"

    if title:
        out = (
            f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
            f"{escape(title)}</div>" + out
        )
    if _finish_html is not None:
        _finish_html(out, show=show, save_path=save_path)
    return None if show else out
