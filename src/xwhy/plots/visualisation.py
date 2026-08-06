"""Native visualisation engine for XWhy explanation results.

This module is a self-contained reimplementation of the plotting surface that
XWhy previously delegated to ``shap``. Every public function mirrors the SHAP
call signature it replaces, so existing notebooks keep working, but nothing
here imports ``shap``.

Design notes:
    * **matplotlib** is the default backend for every static figure.
    * **plotly** is offered on the plots where interactivity pays off, via
      ``backend="plotly"``.
    * **HTML** replaces SHAP's JavaScript bundles. :func:`text` and
      :func:`force` return plain, self-contained HTML strings, so there is no
      ``initjs`` handshake and no ``bundle.js`` to load. They render inline in
      notebooks and can be written straight to disk with ``save_path=...``.

Every plotting function follows the same output convention:
    * ``save_path`` given: the figure is written to disk and ``None`` returned.
    * ``show=True`` (default): the figure is displayed and ``None`` returned.
    * ``show=False``: the figure object is returned for further composition.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

__all__ = [
    "BLUE",
    "GRAY",
    "RED",
    "RED_BLUE",
    "RED_TRANSPARENT_BLUE",
    "Explanation",
    "bar",
    "beeswarm",
    "decision",
    "embedding",
    "force",
    "group_difference",
    "heatmap",
    "image",
    "image_to_text",
    "initjs",
    "monitoring",
    "partial_dependence",
    "scatter",
    "text",
    "violin",
    "waterfall",
]

# ==============================================================================
# PALETTE
# ==============================================================================

RED = "#ff0051"
BLUE = "#008afb"
GRAY = "#777777"
LIGHT_GRAY = "#c8c8c8"

RED_BLUE = LinearSegmentedColormap.from_list(
    "xwhy_red_blue", [BLUE, LIGHT_GRAY, RED], N=256
)

RED_TRANSPARENT_BLUE = LinearSegmentedColormap.from_list(
    "xwhy_red_transparent_blue",
    [
        (0.0, (0.0, 0.541, 0.984, 1.0)),
        (0.5, (1.0, 1.0, 1.0, 0.0)),
        (1.0, (1.0, 0.0, 0.318, 1.0)),
    ],
    N=256,
)

PLOTLY_RED_BLUE = [
    [0.0, BLUE],
    [0.5, LIGHT_GRAY],
    [1.0, RED],
]

#: Axis label used wherever SHAP would have written "SHAP value".
VALUE_LABEL = "XWhy value"

_MATPLOTLIB_BACKENDS = frozenset({"matplotlib", "mpl"})
_PLOTLY_BACKENDS = frozenset({"plotly", "px", "go"})
_HTML_BACKENDS = frozenset({"html"})


# ==============================================================================
# EXPLANATION CONTAINER
# ==============================================================================


@dataclass
class Explanation:
    """Container for attribution values, mirroring ``shap.Explanation``.

    This is the interchange format between :mod:`xwhy.core.result` and this
    module. It supports the slicing and reduction idioms that SHAP users
    expect (``exp[0]``, ``exp[:, 2]``, ``exp.abs.mean(0)``) without depending
    on ``shap``.

    Attributes:
        values: Attribution values. Shape is ``(n_features,)`` for a single
            explained instance, ``(n_instances, n_features)`` for a batch, or
            higher-rank for image and multimodal explanations.
        base_values: The model's expected output, i.e. the value the
            attributions are measured against.
        data: The underlying instance(s) the attributions describe.
        feature_names: Names aligned with the last axis of ``values``.
        display_data: Optional human-readable stand-in for ``data``.
        output_names: Optional names for the model outputs.

    """

    values: np.ndarray
    base_values: float | np.ndarray = 0.0
    data: np.ndarray | Sequence[Any] | None = None
    feature_names: Sequence[str] | np.ndarray | None = None
    display_data: np.ndarray | None = None
    output_names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        """Coerce ``values`` to a numpy array so downstream maths is safe."""
        self.values = np.asarray(self.values)

    # -- shape helpers ---------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying attribution array."""
        return cast(tuple[int, ...], self.values.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying attribution array."""
        return int(self.values.ndim)

    def __len__(self) -> int:
        """Return the size of the leading axis."""
        return int(self.values.shape[0]) if self.values.ndim else 0

    def __getitem__(self, item: Any) -> Explanation:  # noqa: ANN401
        """Slice the explanation, keeping metadata aligned.

        Args:
            item: Any numpy-compatible index. A tuple index whose second
                element selects features also slices ``feature_names``.

        Returns:
            Explanation: A new explanation covering the selected subset.

        """
        names = self.feature_names
        data = self.data

        if isinstance(item, tuple) and len(item) > 1 and names is not None:
            names = np.asarray(names)[item[1]]

        if data is not None:
            data_arr = np.asarray(data, dtype=object)
            try:
                data = data_arr[item]
            except (IndexError, ValueError):
                data = None

        base = self.base_values
        if isinstance(base, np.ndarray) and base.ndim > 0:
            try:
                base = base[item[0] if isinstance(item, tuple) else item]
            except (IndexError, ValueError):
                base = self.base_values

        return Explanation(
            values=self.values[item],
            base_values=base,
            data=data,
            feature_names=names,
            output_names=self.output_names,
        )

    # -- reductions ------------------------------------------------------

    @property
    def abs(self) -> Explanation:
        """Explanation holding the absolute value of every attribution."""
        return Explanation(
            values=np.abs(self.values),
            base_values=self.base_values,
            data=self.data,
            feature_names=self.feature_names,
            output_names=self.output_names,
        )

    def _reduce(self, func: Callable[..., Any], axis: int | None) -> Explanation:
        """Apply a numpy reduction while preserving feature metadata."""
        reduced = func(self.values, axis=axis)
        keeps_features = axis == 0 and self.values.ndim > 1
        return Explanation(
            values=np.asarray(reduced),
            base_values=self.base_values,
            feature_names=self.feature_names if keeps_features else None,
            output_names=self.output_names,
        )

    def mean(self, axis: int | None = None) -> Explanation:
        """Return the mean of the attributions along ``axis``."""
        return self._reduce(np.mean, axis)

    def sum(self, axis: int | None = None) -> Explanation:
        """Return the sum of the attributions along ``axis``."""
        return self._reduce(np.sum, axis)

    def max(self, axis: int | None = None) -> Explanation:
        """Return the maximum of the attributions along ``axis``."""
        return self._reduce(np.max, axis)

    def min(self, axis: int | None = None) -> Explanation:
        """Return the minimum of the attributions along ``axis``."""
        return self._reduce(np.min, axis)

    def __repr__(self) -> str:
        """Return a short, informative representation."""
        return (
            f"Explanation(values={self.values.shape}, "
            f"base_values={np.shape(self.base_values)}, "
            f"n_names={0 if self.feature_names is None else len(self.feature_names)})"
        )


# ==============================================================================
# INTERNAL HELPERS
# ==============================================================================


def _as_explanation(obj: Any) -> Explanation:  # noqa: ANN401
    """Normalise an explanation-like object into an :class:`Explanation`.

    Args:
        obj: An :class:`Explanation`, an object exposing ``to_explanation()``
            or ``to_shap()`` (e.g. a ``BaseXWhyResult``), or a raw array.

    Returns:
        Explanation: The normalised explanation.

    Raises:
        TypeError: If the object cannot be interpreted as an explanation.

    """
    if isinstance(obj, Explanation):
        return obj

    for attr in ("to_explanation", "to_shap"):
        converter = getattr(obj, attr, None)
        if callable(converter):
            converted = converter()
            if isinstance(converted, Explanation):
                return converted
            # A foreign Explanation-like object: copy the fields we need.
            return Explanation(
                values=np.asarray(getattr(converted, "values", converted)),
                base_values=getattr(converted, "base_values", 0.0),
                data=getattr(converted, "data", None),
                feature_names=getattr(converted, "feature_names", None),
            )

    if isinstance(obj, np.ndarray):
        return Explanation(values=obj)

    raise TypeError(
        f"Cannot build an Explanation from {type(obj).__name__}. Pass an "
        "Explanation, an XWhy result, or a numpy array."
    )


def _resolve_names(
    names: Sequence[str] | np.ndarray | None,
    n_features: int,
) -> list[str]:
    """Return exactly ``n_features`` display names, generating any that are missing."""
    if names is None:
        return [f"Feature {i}" for i in range(n_features)]

    resolved = [str(n) for n in np.asarray(names).ravel().tolist()]
    if len(resolved) >= n_features:
        return resolved[:n_features]

    resolved.extend(f"Feature {i}" for i in range(len(resolved), n_features))
    return resolved


def _format_value(value: Any, fmt: str = "%.2f") -> str:  # noqa: ANN401
    """Format a feature value for display, trimming redundant trailing zeros."""
    if value is None:
        return ""
    if isinstance(value, (str, np.str_)):
        return str(value)
    try:
        formatted = fmt % float(value)
    except (TypeError, ValueError):
        return str(value)

    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return "0" if formatted in {"-0", ""} else formatted


def _global_importance(values: np.ndarray) -> np.ndarray:
    """Collapse an attribution array to one importance score per feature."""
    if values.ndim == 1:
        return cast(np.ndarray, np.abs(values))
    axes = tuple(range(values.ndim - 1))
    return cast(np.ndarray, np.abs(values).mean(axis=axes))


def _group_minor_features(
    values: np.ndarray,
    names: Sequence[str],
    max_display: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Keep the strongest features and fold the remainder into one summary row.

    Args:
        values: One score per feature.
        names: Feature names aligned with ``values``.
        max_display: Maximum number of rows to draw, including the summary
            row. ``None`` disables grouping.

    Returns:
        tuple: ``(values, names)`` sorted ascending by magnitude, ready for a
        bottom-to-top horizontal bar chart.

    """
    values = np.asarray(values, dtype=float).ravel()
    names = list(names)
    order = np.argsort(np.abs(values))[::-1]

    if max_display is not None and 0 < max_display < len(order):
        keep, rest = order[: max_display - 1], order[max_display - 1 :]
        kept_values = np.append(values[keep], values[rest].sum())
        kept_names = [names[i] for i in keep]
        kept_names.append(f"Sum of {len(rest)} other features")
    else:
        kept_values = values[order]
        kept_names = [names[i] for i in order]

    return kept_values[::-1], kept_names[::-1]


def _check_backend(backend: str, allowed: frozenset[str]) -> str:
    """Validate a backend name and return its canonical form.

    Args:
        backend: User-supplied backend name.
        allowed: The set of canonical backends this plot supports.

    Returns:
        str: One of ``"matplotlib"``, ``"plotly"`` or ``"html"``.

    Raises:
        ValueError: If the backend is unknown or unsupported for this plot.

    """
    key = backend.lower().strip()

    if key in _MATPLOTLIB_BACKENDS:
        canonical = "matplotlib"
    elif key in _PLOTLY_BACKENDS:
        canonical = "plotly"
    elif key in _HTML_BACKENDS:
        canonical = "html"
    else:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from 'matplotlib', "
            "'plotly' or 'html'."
        )

    if canonical not in allowed:
        raise ValueError(
            f"Backend {backend!r} is not supported by this plot. "
            f"Supported backends: {', '.join(sorted(allowed))}."
        )
    return canonical


def _finish_matplotlib(
    fig: Figure,
    *,
    show: bool,
    save_path: str | Path | None,
) -> Figure | None:
    """Save, display or hand back a matplotlib figure."""
    if save_path is not None:
        fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close(fig)
        return None

    if show:
        plt.show()
        return None

    return fig


def _finish_plotly(
    fig: go.Figure,
    *,
    show: bool,
    save_path: str | Path | None,
) -> go.Figure | None:
    """Save, display or hand back a plotly figure."""
    if save_path is not None:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)
        return None

    if show:
        fig.show()
        return None

    return fig


def _finish_html(
    html: str,
    *,
    show: bool,
    save_path: str | Path | None,
) -> str:
    """Save and/or display an HTML fragment, always returning the markup."""
    if save_path is not None:
        Path(save_path).write_text(_wrap_html_document(html), encoding="utf-8")
        return html

    if show:
        _display_html(html)

    return html


def _display_html(html: str) -> None:
    """Render HTML inline when running inside IPython, otherwise do nothing."""
    try:  # pragma: no cover - depends on the runtime environment
        from IPython.display import HTML, display
    except ImportError:  # pragma: no cover - plain interpreter
        return

    display(HTML(html))  # type: ignore[no-untyped-call]


def _wrap_html_document(fragment: str, title: str = "XWhy explanation") -> str:
    """Wrap an HTML fragment in a minimal standalone document."""
    return (
        "<!doctype html>\n<html><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title></head><body>{fragment}</body></html>"
    )


def _rgba(cmap: LinearSegmentedColormap, value: float) -> str:
    """Convert a normalised value into a CSS ``rgba()`` string."""
    red, green, blue, alpha = cmap(float(np.clip(value, 0.0, 1.0)))
    return f"rgba({int(red * 255)}, {int(green * 255)}, {int(blue * 255)}, {alpha:.3f})"


def _add_colorbar(
    fig: Figure,
    ax: Axes,
    vmin: float,
    vmax: float,
    label: str,
    cmap: LinearSegmentedColormap = RED_BLUE,
) -> None:
    """Attach a colour bar describing the feature-value gradient."""
    mappable = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    mappable.set_array(np.array([]))
    colorbar = fig.colorbar(mappable, ax=ax, aspect=40, pad=0.02)
    colorbar.set_label(label, size=11)
    colorbar.ax.spines[["outline"]].set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=9)


def _style_axes(ax: Axes) -> None:
    """Apply the shared XWhy axis styling (minimal chrome, soft gridlines)."""
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(visible=True, color="#eeeeee", linewidth=0.8)


def _single_instance(exp: Explanation) -> tuple[np.ndarray, float, list[str], Any]:
    """Reduce an explanation to one instance for the local plots.

    Args:
        exp: The explanation to reduce.

    Returns:
        tuple: ``(values, base_value, feature_names, data)`` for one instance.

    Raises:
        ValueError: If the explanation holds more than one instance.

    """
    values = np.asarray(exp.values, dtype=float)
    data = exp.data
    base = exp.base_values

    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
        if data is not None and len(np.shape(data)) == 2:
            data = np.asarray(data, dtype=object)[0]
        if isinstance(base, np.ndarray) and base.size == 1:
            base = float(base.reshape(-1)[0])
    elif values.ndim > 1:
        raise ValueError(
            "This plot explains a single instance but received values with "
            f"shape {values.shape}. Index the explanation first, e.g. exp[0]."
        )

    if isinstance(base, np.ndarray):
        base = float(base.reshape(-1)[0]) if base.size else 0.0

    names = _resolve_names(exp.feature_names, values.shape[0])
    return values, float(base), names, data


def _bar_colors(values: np.ndarray) -> list[str]:
    """Map attribution signs onto the XWhy red/blue palette."""
    return [RED if v >= 0 else BLUE for v in values]


# ==============================================================================
# LOCAL PLOTS
# ==============================================================================


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

    height = figsize[1] if figsize else max(3.0, 0.4 * len(plot_names) + 1.4)
    width = figsize[0] if figsize else 8.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
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
            _format_value(value),
            va="center",
            ha="left" if aligned_right else "right",
            fontsize=10,
            color=RED if aligned_right else BLUE,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_names, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12)

    lower = min(0.0, float(plot_values.min()) * 1.25) if plot_values.size else 0.0
    upper = max(0.0, float(plot_values.max()) * 1.25) if plot_values.size else 1.0
    if lower == upper:  # every attribution is zero
        lower, upper = lower - 1.0, upper + 1.0
    ax.set_xlim(lower, upper)

    if title:
        ax.set_title(title, fontsize=13, loc="left")

    _style_axes(ax)
    if not is_global:
        ax.axvline(0, color="#333333", linewidth=0.9)

    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


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

    height = figsize[1] if figsize else max(3.2, 0.45 * len(plot_values) + 1.8)
    width = figsize[0] if figsize else 9.0
    fig, ax = plt.subplots(figsize=(width, height))

    # Walk bottom-to-top so the cumulative total lands on the prediction.
    running = base_value
    starts = np.empty(len(plot_values))
    for i, value in enumerate(plot_values):
        starts[i] = running
        running += value

    positions = np.arange(len(plot_values))
    ax.barh(
        positions,
        plot_values,
        left=starts,
        color=_bar_colors(plot_values),
        height=0.65,
    )

    # Connectors between consecutive steps.
    for i in range(len(plot_values) - 1):
        ax.plot(
            [starts[i] + plot_values[i]] * 2,
            [positions[i], positions[i + 1]],
            color="#bbbbbb",
            linewidth=0.9,
            zorder=0,
        )

    span = float(np.max(np.abs(plot_values))) if plot_values.size else 1.0
    offset = span * 0.04 if span else 0.01

    for pos, value, start in zip(positions, plot_values, starts, strict=True):
        end = start + value
        ax.text(
            end + (offset if value >= 0 else -offset),
            float(pos),
            _format_value(value, "%+.2f"),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=10,
            color=RED if value >= 0 else BLUE,
        )

    ax.axvline(base_value, color="#999999", linestyle="--", linewidth=1.0, zorder=0)
    ax.axvline(prediction, color="#333333", linestyle="-", linewidth=1.0, zorder=0)

    # Leave room for the value labels sitting outside each bar, otherwise the
    # ones on short negative bars collide with the y-axis tick labels.
    edges = np.concatenate([starts, starts + plot_values, [base_value, prediction]])
    low, high = float(edges.min()), float(edges.max())
    margin = max(high - low, 1e-9) * 0.18
    ax.set_xlim(low - margin, high + margin)

    ax.set_yticks(positions)
    ax.set_yticklabels(plot_labels, fontsize=11)
    ax.set_xlabel("Model output", fontsize=12)
    ax.set_title(
        title
        or (
            f"E[f(X)] = {_format_value(base_value)}    "
            f"f(x) = {_format_value(prediction)}"
        ),
        fontsize=12,
        loc="left",
    )

    _style_axes(ax)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def text(
    explanation: Any,  # noqa: ANN401
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    separator: str = " ",
    **kwargs: Any,  # noqa: ANN401
) -> str:
    """Render a token-level explanation as self-contained HTML.

    This replaces SHAP's JavaScript text plot. The output is plain HTML with
    inline styles, so it needs no ``initjs()`` call, no bundled JavaScript,
    and renders identically in notebooks, static exports and saved files.

    Args:
        explanation: A token-level :class:`Explanation` or XWhy result.
        show: Whether to display the markup inline when inside IPython.
        save_path: Optional path to write a standalone HTML document to.
        title: Optional heading rendered above the tokens.
        separator: String placed between tokens.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        str: The generated HTML fragment.

    """
    del kwargs
    exp = _as_explanation(explanation)
    values, base_value, names, _ = _single_instance(exp)

    span = float(np.max(np.abs(values))) if values.size else 0.0
    denom = span if span > 0 else 1.0
    prediction = base_value + float(values.sum())

    pieces: list[str] = []
    for token, value in zip(names, values, strict=True):
        normalised = 0.5 * float(value) / denom + 0.5
        text_color = "#ffffff" if abs(float(value)) > 0.65 * denom else "#111111"
        pieces.append(
            f"<span title='{escape(token)}: {value:+.4f}' "
            f'style="background:{_rgba(RED_BLUE, normalised)};color:{text_color};'
            "border-radius:4px;padding:2px 5px;margin:2px 1px;"
            'display:inline-block;font-family:inherit;">'
            f"{escape(token)}</span>"
        )

    legend = (
        "<div style='margin-top:14px;font-size:12px;color:#555;'>"
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{BLUE};border-radius:3px;vertical-align:middle;"></span>'
        f" negative &nbsp;&nbsp;"
        f'<span style="display:inline-block;width:14px;height:14px;'
        f'background:{RED};border-radius:3px;vertical-align:middle;"></span>'
        " positive &nbsp;&nbsp;"
        f"E[f(X)] = {_format_value(base_value, '%.4f')} &nbsp;&nbsp;"
        f"f(x) = {_format_value(prediction, '%.4f')}"
        "</div>"
    )

    heading = (
        f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
        f"{escape(title)}</div>"
        if title
        else ""
    )

    html = (
        "<div class='xwhy-text' style=\"font-family:-apple-system,BlinkMacSystemFont,"
        "'Segoe UI',Roboto,sans-serif;line-height:2.1;padding:14px;"
        'border:1px solid #e6e6e6;border-radius:8px;">'
        f"{heading}<div>{separator.join(pieces)}</div>{legend}</div>"
    )

    return _finish_html(html, show=show, save_path=save_path)


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

    Args:
        explanation: A single-instance :class:`Explanation` or XWhy result.
        max_display: Maximum number of features to label.
        show: Whether to display the result.
        save_path: Optional path to write the figure or HTML document to.
        backend: ``"matplotlib"`` or ``"html"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | str | None: HTML markup for the ``"html"`` backend, otherwise
        the matplotlib figure when ``show`` is False, else ``None``.

    """
    del kwargs
    engine = _check_backend(backend, frozenset({"matplotlib", "html"}))
    exp = _as_explanation(explanation)
    values, base_value, names, data = _single_instance(exp)
    prediction = base_value + float(values.sum())

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
        html = _force_html(positives, negatives, base_value, prediction, title=title)
        return _finish_html(html, show=show, save_path=save_path)

    width = figsize[0] if figsize else 11.0
    height = figsize[1] if figsize else 2.6
    fig, ax = plt.subplots(figsize=(width, height))

    total_pos = sum(v for _, v in positives)
    total_neg = sum(abs(v) for _, v in negatives)
    span = total_pos + total_neg

    # Red blocks end at f(x); blue blocks start at f(x).
    cursor = prediction
    for label, value in positives:
        ax.barh(0, value, left=cursor - value, color=RED, height=0.42)
        _force_annotate(ax, cursor - value / 2, value, label, span)
        cursor -= value

    cursor = prediction
    for label, value in negatives:
        ax.barh(0, abs(value), left=cursor, color=BLUE, height=0.42)
        _force_annotate(ax, cursor + abs(value) / 2, value, label, span)
        cursor += abs(value)

    ax.axvline(prediction, color="#222222", linewidth=1.6)
    ax.text(
        prediction,
        0.34,
        f"f(x) = {_format_value(prediction)}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        base_value,
        -0.38,
        f"E[f(X)] = {_format_value(base_value)}",
        ha="center",
        va="top",
        fontsize=10,
        color="#555555",
    )
    ax.axvline(base_value, color="#999999", linestyle="--", linewidth=1.0)

    padding = max(total_pos + total_neg, 1e-9) * 0.12
    ax.set_xlim(
        min(prediction - total_pos, base_value) - padding,
        max(prediction + total_neg, base_value) + padding,
    )
    ax.set_ylim(-0.75, 0.75)
    ax.set_yticks([])
    ax.spines[["left", "right", "top"]].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, loc="left")

    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def _force_annotate(
    ax: Axes,
    x: float,
    value: float,
    label: str,
    span: float,
) -> None:
    """Label a force-plot block, but only when the block is wide enough to read.

    Narrow blocks are left unlabelled rather than overprinted; hovering is not
    available in a static figure, so an unreadable label is worse than none.

    Args:
        ax: Axes to draw on.
        x: Centre of the block in data coordinates.
        value: The attribution the block represents.
        label: Text to draw.
        span: Total width of the force layout, used as the legibility yardstick.

    """
    if span <= 0 or abs(value) < 0.07 * span:
        return

    # Roughly one character per 1.1% of the layout width at this font size.
    budget = max(3, int(abs(value) / span * 90))
    trimmed = label if len(label) <= budget else f"{label[: budget - 1]}…"

    ax.text(
        x,
        0.0,
        trimmed,
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        clip_on=True,
    )


def _force_html(
    positives: list[tuple[str, float]],
    negatives: list[tuple[str, float]],
    base_value: float,
    prediction: float,
    title: str | None = None,
) -> str:
    """Build the static HTML replacement for SHAP's JavaScript force plot."""
    total = sum(v for _, v in positives) + sum(abs(v) for _, v in negatives)
    total = total if total > 0 else 1.0

    def segment(label: str, value: float, color: str) -> str:
        pct = abs(value) / total * 100
        caption = escape(label) if pct > 8 else ""
        return (
            f"<div title='{escape(label)}: {value:+.4f}' "
            f'style="flex:0 0 {pct:.4f}%;background:{color};color:#fff;'
            "font-size:11px;line-height:34px;text-align:center;overflow:hidden;"
            'white-space:nowrap;">'
            f"{caption}</div>"
        )

    blocks = [segment(label, value, RED) for label, value in positives]
    blocks += [segment(label, value, BLUE) for label, value in negatives]

    heading = (
        f"<div style='font-size:14px;font-weight:600;margin-bottom:10px;'>"
        f"{escape(title)}</div>"
        if title
        else ""
    )

    return (
        "<div class='xwhy-force' style=\"font-family:-apple-system,"
        "BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:14px;"
        'border:1px solid #e6e6e6;border-radius:8px;">'
        f"{heading}"
        "<div style='font-size:13px;margin-bottom:6px;color:#333;'>"
        f"E[f(X)] = {_format_value(base_value, '%.4f')} &nbsp;&#8594;&nbsp; "
        f"<b>f(x) = {_format_value(prediction, '%.4f')}</b></div>"
        "<div style='display:flex;width:100%;border-radius:5px;"
        f"overflow:hidden;'>{''.join(blocks)}</div>"
        "<div style='margin-top:8px;font-size:11px;color:#666;'>"
        f"<span style='color:{RED};'>&#9632;</span> increases the prediction"
        f" &nbsp;&nbsp;<span style='color:{BLUE};'>&#9632;</span> decreases it"
        "</div></div>"
    )


def decision(
    base_value: float | np.ndarray,
    shap_values: np.ndarray,
    features: np.ndarray | Sequence[Any] | None = None,
    feature_names: Sequence[str] | None = None,
    *,
    max_display: int | None = 20,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    alpha: float | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Visualise cumulative attributions as decision paths.

    Each observation is drawn as a line that starts at the expected value and
    accumulates one feature contribution per row, ending at the prediction.

    Args:
        base_value: The model's expected output.
        shap_values: Attributions of shape ``(n_features,)`` or
            ``(n_instances, n_features)``.
        features: Optional raw feature values, used for the row labels.
        feature_names: Optional feature names.
        max_display: Maximum number of feature rows to draw.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        alpha: Line opacity. Defaults to a density-aware value.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | go.Figure | None: The figure when ``show`` is False and no
        ``save_path`` was given, otherwise ``None``.

    """
    del kwargs
    engine = _check_backend(backend, frozenset({"matplotlib", "plotly"}))

    values = np.atleast_2d(np.asarray(shap_values, dtype=float))
    n_instances, n_features = values.shape
    names = _resolve_names(feature_names, n_features)

    if isinstance(base_value, np.ndarray):
        base = float(np.asarray(base_value).reshape(-1)[0])
    else:
        base = float(base_value)

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    if max_display is not None and 0 < max_display < len(order):
        order = order[:max_display]

    # Rows are drawn bottom-to-top, weakest feature first.
    order = order[::-1]
    ordered = values[:, order]
    labels = [names[i] for i in order]

    if features is not None:
        raw = np.atleast_2d(np.asarray(features, dtype=object))
        if raw.shape[1] == n_features:
            labels = [
                f"{names[i]} = {_format_value(raw[0, i])}"
                if n_instances == 1
                else names[i]
                for i in order
            ]

    # Cumulative path: start at the base value, add one feature per row.
    paths = np.concatenate(
        [np.full((n_instances, 1), base), base + np.cumsum(ordered, axis=1)],
        axis=1,
    )
    predictions = paths[:, -1]

    vmin, vmax = float(predictions.min()), float(predictions.max())
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
    y_positions = np.arange(len(order) + 1) - 0.5

    if engine == "plotly":
        fig = go.Figure()
        for i in range(n_instances):
            color = RED_BLUE(norm(predictions[i]))
            fig.add_trace(
                go.Scatter(
                    x=paths[i],
                    y=y_positions,
                    mode="lines",
                    line={
                        "color": f"rgb({int(color[0] * 255)},"
                        f"{int(color[1] * 255)},{int(color[2] * 255)})",
                        "width": 1.4,
                    },
                    opacity=alpha or max(0.15, min(1.0, 30.0 / n_instances)),
                    showlegend=False,
                    hovertemplate="output: %{x:.4f}<extra></extra>",
                )
            )
        fig.update_layout(
            title=title or "Decision plot",
            xaxis_title="Model output",
            template="plotly_white",
            height=max(380, 28 * len(order) + 180),
            yaxis={
                "tickmode": "array",
                "tickvals": y_positions[:-1] + 0.5,
                "ticktext": labels,
            },
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    height = figsize[1] if figsize else max(4.0, 0.34 * len(order) + 2.0)
    width = figsize[0] if figsize else 9.0
    fig, ax = plt.subplots(figsize=(width, height))

    line_alpha = alpha or max(0.12, min(1.0, 30.0 / n_instances))
    for i in range(n_instances):
        ax.plot(
            paths[i],
            y_positions,
            color=RED_BLUE(norm(predictions[i])),
            linewidth=1.3,
            alpha=line_alpha,
        )

    ax.axvline(base, color="#999999", linestyle="--", linewidth=1.0, zorder=0)
    ax.set_yticks(y_positions[:-1] + 0.5)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.set_xlabel("Model output", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, loc="left")

    _add_colorbar(fig, ax, vmin, vmax, "Model output")
    _style_axes(ax)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


# ==============================================================================
# GLOBAL PLOTS
# ==============================================================================


def scatter(
    explanation: Any,  # noqa: ANN401
    *,
    ind: int | str | None = None,
    color: int | str | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Create a dependence scatter plot for one feature.

    The horizontal axis holds the feature's raw value and the vertical axis
    its attribution, revealing the shape of the learned relationship.

    Args:
        explanation: A batched :class:`Explanation` or XWhy result.
        ind: Feature to plot, by index or name. Defaults to the most
            important feature.
        color: Optional second feature used to colour the points, by index or
            name. Defaults to the plotted feature's own attribution.
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    index = _feature_index(ind, names, values)
    y = values[:, index]

    data = _numeric_data(exp.data, values.shape)
    x = data[:, index] if data is not None else np.arange(values.shape[0], dtype=float)
    xlabel = names[index] if data is not None else "Instance index"

    if color is None:
        color_values, color_label = y, f"{VALUE_LABEL} for {names[index]}"
    else:
        color_index = _feature_index(color, names, values)
        if data is not None:
            color_values = data[:, color_index]
        else:
            color_values = values[:, color_index]
        color_label = names[color_index]

    vmin, vmax = _robust_limits(color_values)

    if engine == "plotly":
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={
                    "color": color_values,
                    "colorscale": PLOTLY_RED_BLUE,
                    "cmin": vmin,
                    "cmax": vmax,
                    "size": 8,
                    "opacity": 0.85,
                    "colorbar": {"title": color_label},
                },
                hovertemplate=(
                    f"{names[index]}: %{{x}}<br>{VALUE_LABEL}: "
                    "%{y:.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            title=title or f"Dependence: {names[index]}",
            xaxis_title=xlabel,
            yaxis_title=f"{VALUE_LABEL} for {names[index]}",
            template="plotly_white",
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    fig, ax = plt.subplots(figsize=figsize or (8.0, 5.5))
    ax.scatter(
        x,
        y,
        c=color_values,
        cmap=RED_BLUE,
        vmin=vmin,
        vmax=vmax,
        s=22,
        alpha=0.85,
        linewidths=0,
    )
    ax.axhline(0, color="#999999", linewidth=0.9, zorder=0)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(f"{VALUE_LABEL} for {names[index]}", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, loc="left")

    _add_colorbar(fig, ax, vmin, vmax, color_label)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def heatmap(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    instance_order: str = "hclust",
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Plot attributions as an instance-by-feature heatmap.

    Args:
        explanation: A batched :class:`Explanation` or XWhy result.
        max_display: Maximum number of feature rows before grouping.
        instance_order: ``"hclust"`` to cluster similar instances together,
            ``"output"`` to sort by model output, or ``"none"``.
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]

    if max_display is not None and 0 < max_display < len(order):
        keep, rest = order[: max_display - 1], order[max_display - 1 :]
        matrix = np.column_stack([values[:, keep], values[:, rest].sum(axis=1)])
        labels = [names[i] for i in keep]
        labels.append(f"Sum of {len(rest)} other features")
    else:
        matrix = values[:, order]
        labels = [names[i] for i in order]

    instances = _order_instances(values, instance_order)
    matrix = matrix[instances]
    outputs = np.asarray(exp.base_values, dtype=float).ravel()
    base = float(outputs[0]) if outputs.size else 0.0
    predictions = base + values[instances].sum(axis=1)

    limit = float(np.percentile(np.abs(matrix), 99.5)) if matrix.size else 1.0
    limit = limit if limit > 0 else 1.0

    if engine == "plotly":
        fig = go.Figure(
            go.Heatmap(
                z=matrix.T,
                y=labels,
                colorscale=PLOTLY_RED_BLUE,
                zmid=0,
                zmin=-limit,
                zmax=limit,
                colorbar={"title": VALUE_LABEL},
                hovertemplate=("instance %{x}<br>%{y}: %{z:.4f}<extra></extra>"),
            )
        )
        fig.update_layout(
            title=title or "Attribution heatmap",
            xaxis_title="Instances",
            template="plotly_white",
            height=max(380, 28 * len(labels) + 200),
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    height = figsize[1] if figsize else max(4.0, 0.32 * len(labels) + 2.6)
    width = figsize[0] if figsize else 9.0
    fig, (top_ax, ax) = plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.05},
    )

    top_ax.plot(
        np.arange(len(predictions)), predictions, color="#333333", linewidth=1.0
    )
    top_ax.set_ylabel("f(x)", fontsize=10)
    top_ax.tick_params(labelbottom=False, length=0)
    top_ax.spines[["top", "right", "bottom"]].set_visible(False)

    ax.imshow(
        matrix.T,
        aspect="auto",
        cmap=RED_BLUE,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Instances", fontsize=12)
    ax.tick_params(length=0)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)

    if title:
        top_ax.set_title(title, fontsize=13, loc="left")

    _add_colorbar(fig, ax, -limit, limit, VALUE_LABEL)
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def beeswarm(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    seed: int = 0,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Create a beeswarm summary plot.

    Every instance contributes one dot per feature. Dots are spread vertically
    by local density and coloured by the feature's own value, so both the
    magnitude and the direction of each effect are visible at a glance.

    Args:
        explanation: A batched :class:`Explanation` or XWhy result.
        max_display: Maximum number of feature rows to draw.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        seed: Seed for the jitter applied when breaking density ties.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | go.Figure | None: The figure when ``show`` is False and no
        ``save_path`` was given, otherwise ``None``.

    """
    del kwargs
    engine = _check_backend(backend, frozenset({"matplotlib", "plotly"}))
    exp = _as_explanation(explanation)
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])
    data = _numeric_data(exp.data, values.shape)

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    if max_display is not None and 0 < max_display < len(order):
        order = order[:max_display]
    order = order[::-1]  # bottom-to-top

    rng = np.random.default_rng(seed)

    if engine == "plotly":
        fig = go.Figure()
        for row, feature in enumerate(order):
            column = data[:, feature] if data is not None else None
            vmin, vmax = _robust_limits(column) if column is not None else (0.0, 1.0)
            fig.add_trace(
                go.Scatter(
                    x=values[:, feature],
                    y=row + _density_offsets(values[:, feature], rng),
                    mode="markers",
                    name=names[feature],
                    marker={
                        "color": column if column is not None else GRAY,
                        "colorscale": PLOTLY_RED_BLUE if column is not None else None,
                        "cmin": vmin,
                        "cmax": vmax,
                        "size": 6,
                        "opacity": 0.8,
                        "showscale": column is not None and row == len(order) - 1,
                        "colorbar": {"title": "Feature value"},
                    },
                    showlegend=False,
                    hovertemplate=(
                        f"{names[feature]}<br>{VALUE_LABEL}: %{{x:.4f}}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            title=title or "Beeswarm summary",
            xaxis_title=VALUE_LABEL,
            template="plotly_white",
            height=max(380, 40 * len(order) + 160),
            yaxis={
                "tickmode": "array",
                "tickvals": list(range(len(order))),
                "ticktext": [names[i] for i in order],
            },
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    height = figsize[1] if figsize else max(3.4, 0.5 * len(order) + 1.6)
    width = figsize[0] if figsize else 8.5
    fig, ax = plt.subplots(figsize=(width, height))

    global_min, global_max = 0.0, 1.0
    if data is not None:
        global_min, global_max = _robust_limits(data[:, order])

    for row, feature in enumerate(order):
        y = row + _density_offsets(values[:, feature], rng)
        if data is not None:
            ax.scatter(
                values[:, feature],
                y,
                c=data[:, feature],
                cmap=RED_BLUE,
                vmin=global_min,
                vmax=global_max,
                s=16,
                alpha=0.85,
                linewidths=0,
            )
        else:
            ax.scatter(values[:, feature], y, color=GRAY, s=16, alpha=0.7, linewidths=0)

    ax.axvline(0, color="#999999", linewidth=0.9, zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=11)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel(VALUE_LABEL, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, loc="left")

    if data is not None:
        _add_colorbar(fig, ax, global_min, global_max, "Feature value")

    _style_axes(ax)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def violin(
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
    """Create a violin summary plot of the attribution distributions.

    Args:
        explanation: A batched :class:`Explanation` or XWhy result.
        max_display: Maximum number of feature rows to draw.
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    importance = np.abs(values).mean(axis=0)
    order = np.argsort(importance)[::-1]
    if max_display is not None and 0 < max_display < len(order):
        order = order[:max_display]
    order = order[::-1]

    if engine == "plotly":
        fig = go.Figure()
        for feature in order:
            column = values[:, feature]
            fig.add_trace(
                go.Violin(
                    x=column,
                    name=names[feature],
                    orientation="h",
                    side="positive",
                    points=False,
                    fillcolor=RED if column.mean() >= 0 else BLUE,
                    line={"color": "#555555", "width": 1},
                    opacity=0.75,
                )
            )
        fig.update_layout(
            title=title or "Attribution distributions",
            xaxis_title=VALUE_LABEL,
            template="plotly_white",
            showlegend=False,
            height=max(380, 42 * len(order) + 160),
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    height = figsize[1] if figsize else max(3.4, 0.5 * len(order) + 1.6)
    width = figsize[0] if figsize else 8.5
    fig, ax = plt.subplots(figsize=(width, height))

    # A degenerate (zero-variance) column makes gaussian_kde fail, so nudge it.
    columns = []
    for feature in order:
        column = values[:, feature].astype(float)
        if np.allclose(column, column[0]):
            column = column + np.linspace(-1e-9, 1e-9, column.shape[0])
        columns.append(column)

    parts = ax.violinplot(
        columns,
        positions=np.arange(len(order)),
        orientation="horizontal",
        showextrema=False,
        widths=0.85,
    )

    bodies = cast(list[Any], parts["bodies"])
    for body, feature in zip(bodies, order, strict=True):
        body.set_facecolor(RED if values[:, feature].mean() >= 0 else BLUE)
        body.set_alpha(0.75)
        body.set_edgecolor("#555555")

    ax.axvline(0, color="#999999", linewidth=0.9, zorder=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=11)
    ax.set_xlabel(VALUE_LABEL, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, loc="left")

    _style_axes(ax)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def embedding(
    ind: int | str,
    explanation: Any,  # noqa: ANN401
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Project the attributions to 2D and colour by one feature.

    The projection is the first two principal components of the attribution
    matrix, computed with a plain numpy SVD, so instances that the model
    explains in similar ways sit near each other.

    Args:
        ind: Feature whose attribution colours the points, by index or name.
        explanation: A batched :class:`Explanation` or XWhy result.
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    index = _feature_index(ind, names, values)
    coords = _pca_2d(values)
    color_values = values[:, index]
    vmin, vmax = _robust_limits(color_values)
    color_label = f"{VALUE_LABEL} for {names[index]}"

    if engine == "plotly":
        fig = go.Figure(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                marker={
                    "color": color_values,
                    "colorscale": PLOTLY_RED_BLUE,
                    "cmin": vmin,
                    "cmax": vmax,
                    "size": 8,
                    "opacity": 0.85,
                    "colorbar": {"title": color_label},
                },
            )
        )
        fig.update_layout(
            title=title or "Attribution embedding",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            template="plotly_white",
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    fig, ax = plt.subplots(figsize=figsize or (7.5, 6.0))
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=color_values,
        cmap=RED_BLUE,
        vmin=vmin,
        vmax=vmax,
        s=24,
        alpha=0.85,
        linewidths=0,
    )
    ax.set_xlabel("Component 1", fontsize=12)
    ax.set_ylabel("Component 2", fontsize=12)
    ax.set_title(title or "Attribution embedding", fontsize=13, loc="left")

    _add_colorbar(fig, ax, vmin, vmax, color_label)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def group_difference(
    explanation: Any,  # noqa: ANN401
    group_mask: np.ndarray,
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Plot the mean attribution difference between two groups.

    Args:
        explanation: A batched :class:`Explanation` or XWhy result.
        group_mask: Boolean array selecting the first group. Instances where
            the mask is False form the second group.
        max_display: Maximum number of feature rows to draw.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: ``"matplotlib"`` or ``"plotly"``.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | go.Figure | None: The figure when ``show`` is False and no
        ``save_path`` was given, otherwise ``None``.

    Raises:
        ValueError: If the mask length does not match the instance count, or
            if either group is empty.

    """
    del kwargs
    exp = _as_explanation(explanation)
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    mask = np.asarray(group_mask).astype(bool).ravel()
    if mask.shape[0] != values.shape[0]:
        raise ValueError(
            f"group_mask has length {mask.shape[0]} but the explanation holds "
            f"{values.shape[0]} instances."
        )
    if not mask.any() or mask.all():
        raise ValueError("group_mask must select a non-empty proper subset.")

    differences = values[mask].mean(axis=0) - values[~mask].mean(axis=0)

    return bar(
        Explanation(values=differences, feature_names=names),
        max_display=max_display,
        show=show,
        save_path=save_path,
        backend=backend,
        title=title or "Group difference in attributions",
        figsize=figsize,
    )


def monitoring(
    ind: int | str,
    explanation: Any,  # noqa: ANN401
    features: np.ndarray | None = None,
    *,
    n_splits: int = 50,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Monitor one feature's attribution across the instance ordering.

    Instances are assumed to arrive in order (for example over time). The plot
    marks split points where the attribution distribution before and after the
    split differs significantly under a Welch t-test, which is how attribution
    drift shows up in production.

    Args:
        ind: Feature to monitor, by index or name.
        explanation: A batched :class:`Explanation` or XWhy result.
        features: Optional raw feature values used to colour the points.
        n_splits: Number of candidate split points to test.
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    names = _resolve_names(exp.feature_names, values.shape[1])

    index = _feature_index(ind, names, values)
    series = values[:, index]
    positions = np.arange(series.shape[0])
    breaks = _significant_splits(series, n_splits)

    color_values = None
    if features is not None:
        raw = np.atleast_2d(np.asarray(features))
        if raw.shape[0] == series.shape[0] and raw.shape[1] > index:
            with np.errstate(invalid="ignore"):
                color_values = np.asarray(raw[:, index], dtype=float)

    if engine == "plotly":
        fig = go.Figure(
            go.Scatter(
                x=positions,
                y=series,
                mode="markers",
                marker={
                    "color": color_values if color_values is not None else GRAY,
                    "colorscale": (
                        PLOTLY_RED_BLUE if color_values is not None else None
                    ),
                    "size": 6,
                    "opacity": 0.8,
                    "showscale": color_values is not None,
                    "colorbar": {"title": names[index]},
                },
            )
        )
        for position in breaks:
            fig.add_vline(x=position, line={"color": "#333333", "dash": "dash"})
        fig.update_layout(
            title=title or f"Monitoring: {names[index]}",
            xaxis_title="Instance index",
            yaxis_title=f"{VALUE_LABEL} for {names[index]}",
            template="plotly_white",
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    fig, ax = plt.subplots(figsize=figsize or (9.0, 4.5))
    if color_values is not None:
        vmin, vmax = _robust_limits(color_values)
        ax.scatter(
            positions,
            series,
            c=color_values,
            cmap=RED_BLUE,
            vmin=vmin,
            vmax=vmax,
            s=16,
            alpha=0.85,
            linewidths=0,
        )
        _add_colorbar(fig, ax, vmin, vmax, names[index])
    else:
        ax.scatter(positions, series, color=GRAY, s=16, alpha=0.7, linewidths=0)

    for position in breaks:
        ax.axvline(position, color="#333333", linestyle="--", linewidth=1.1)
        ax.text(
            position,
            ax.get_ylim()[1],
            " drift",
            fontsize=9,
            color="#333333",
            va="top",
        )

    ax.axhline(0, color="#999999", linewidth=0.9, zorder=0)
    ax.set_xlabel("Instance index", fontsize=12)
    ax.set_ylabel(f"{VALUE_LABEL} for {names[index]}", fontsize=12)
    ax.set_title(title or f"Monitoring: {names[index]}", fontsize=13, loc="left")

    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


# ==============================================================================
# IMAGE AND MULTIMODAL PLOTS
# ==============================================================================


def image(
    explanation: Any,  # noqa: ANN401
    pixel_values: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    *,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Overlay attributions on the images they explain.

    The original image is shown desaturated underneath a red/blue attribution
    layer, so positive and negative evidence stays readable against the
    picture.

    Args:
        explanation: An image :class:`Explanation` or XWhy result.
        pixel_values: Optional images to draw underneath the attributions. If
            omitted, ``explanation.data`` is used.
        labels: Optional per-column titles, one per explained output.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | None: The figure when ``show`` is False and no ``save_path``
        was given, otherwise ``None``.

    """
    del kwargs
    exp = _as_explanation(explanation)
    values, backgrounds = _prepare_image_arrays(exp, pixel_values)

    n_rows = values.shape[0]
    n_outputs = values.shape[-1]
    n_cols = n_outputs + 1

    limit = float(np.nanpercentile(np.abs(values), 99.9)) if values.size else 1.0
    limit = limit if limit > 0 else 1.0

    width = figsize[0] if figsize else 2.6 * n_cols + 1.4
    height = figsize[1] if figsize else 2.7 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)

    for row in range(n_rows):
        background = backgrounds[row]
        axes[row][0].imshow(background)
        axes[row][0].axis("off")
        if row == 0:
            axes[row][0].set_title("Input", fontsize=11)

        grayscale = background.mean(axis=2)
        for output in range(n_outputs):
            ax = axes[row][output + 1]
            ax.imshow(grayscale, cmap="gray", alpha=0.28, vmin=0, vmax=1)
            ax.imshow(
                values[row, :, :, output],
                cmap=RED_TRANSPARENT_BLUE,
                vmin=-limit,
                vmax=limit,
            )
            ax.axis("off")
            if row == 0:
                column_label = (
                    labels[output]
                    if labels is not None and output < len(labels)
                    else f"Output {output}"
                )
                ax.set_title(str(column_label), fontsize=11)

    mappable = ScalarMappable(
        norm=Normalize(vmin=-limit, vmax=limit), cmap=RED_TRANSPARENT_BLUE
    )
    mappable.set_array(np.array([]))
    colorbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.04,
        aspect=60,
    )
    colorbar.set_label(VALUE_LABEL, size=11)
    colorbar.ax.spines[["outline"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13)

    return _finish_matplotlib(fig, show=show, save_path=save_path)


def image_to_text(
    explanation: Any,  # noqa: ANN401
    *,
    max_tokens: int = 8,
    show: bool = True,
    save_path: str | Path | None = None,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Plot image attributions for each generated output token.

    Args:
        explanation: A multimodal :class:`Explanation` whose values have shape
            ``(n_samples, height, width, channels, n_tokens)``.
        max_tokens: Maximum number of output tokens to draw.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Ignored, accepted for SHAP call compatibility.

    Returns:
        Figure | None: The figure when ``show`` is False and no ``save_path``
        was given, otherwise ``None``.

    Raises:
        ValueError: If the explanation is not 5-dimensional.

    """
    del kwargs
    exp = _as_explanation(explanation)
    values = np.asarray(exp.values, dtype=float)

    if values.ndim < 5:
        raise ValueError(
            "image_to_text expects 5D explanations of shape (n, H, W, C, "
            f"n_tokens) but received shape {values.shape}. For image "
            "classification use xwhy.plots.image() instead."
        )

    # Collapse the channel axis; attribution maps are per pixel, not per channel.
    maps = values.mean(axis=3)
    n_tokens = min(maps.shape[-1], max_tokens)
    tokens = exp.output_names or [f"Token {i}" for i in range(n_tokens)]

    background = _background_for(exp, maps.shape[1:3])
    limit = float(np.nanpercentile(np.abs(maps), 99.9)) if maps.size else 1.0
    limit = limit if limit > 0 else 1.0

    n_cols = min(4, n_tokens)
    n_rows = int(np.ceil(n_tokens / n_cols))
    width = figsize[0] if figsize else 3.0 * n_cols
    height = figsize[1] if figsize else 3.1 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)

    grayscale = background.mean(axis=2)
    for position in range(n_rows * n_cols):
        ax = axes[position // n_cols][position % n_cols]
        ax.axis("off")
        if position >= n_tokens:
            continue
        ax.imshow(grayscale, cmap="gray", alpha=0.28, vmin=0, vmax=1)
        ax.imshow(
            maps[0, :, :, position],
            cmap=RED_TRANSPARENT_BLUE,
            vmin=-limit,
            vmax=limit,
        )
        ax.set_title(str(tokens[position]), fontsize=11)

    mappable = ScalarMappable(
        norm=Normalize(vmin=-limit, vmax=limit), cmap=RED_TRANSPARENT_BLUE
    )
    mappable.set_array(np.array([]))
    colorbar = fig.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.04,
        aspect=60,
    )
    colorbar.set_label(VALUE_LABEL, size=11)
    colorbar.ax.spines[["outline"]].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=13)

    return _finish_matplotlib(fig, show=show, save_path=save_path)


# ==============================================================================
# MODEL INSPECTION
# ==============================================================================


def partial_dependence(
    ind: int | str,
    model: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
    npoints: int | None = None,
    ice: bool = True,
    max_ice_lines: int = 100,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | go.Figure | None:
    """Plot the partial dependence of the model on one feature.

    Args:
        ind: Feature to sweep, by index or name.
        model: Callable mapping a 2D array of instances to predictions.
        data: Background dataset of shape ``(n_instances, n_features)``.
        feature_names: Optional feature names.
        npoints: Number of grid points. Defaults to 100, or the number of
            distinct values when the feature is low-cardinality.
        ice: Whether to draw individual conditional expectation lines.
        max_ice_lines: Maximum number of ICE lines to draw.
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
    background = np.atleast_2d(np.asarray(data, dtype=float))
    names = _resolve_names(feature_names, background.shape[1])
    index = _feature_index(ind, names, background)

    column = background[:, index]
    distinct = np.unique(column)
    if npoints is None:
        npoints = int(distinct.shape[0]) if distinct.shape[0] <= 20 else 100
    grid = (
        distinct
        if distinct.shape[0] <= npoints
        else np.linspace(column.min(), column.max(), npoints)
    )

    # One forward pass per grid point over the whole background set.
    curves = np.empty((background.shape[0], grid.shape[0]))
    for position, value in enumerate(grid):
        perturbed = background.copy()
        perturbed[:, index] = value
        curves[:, position] = np.asarray(model(perturbed), dtype=float).ravel()

    average = curves.mean(axis=0)
    rng = np.random.default_rng(0)
    if curves.shape[0] > max_ice_lines:
        sample = rng.choice(curves.shape[0], size=max_ice_lines, replace=False)
    else:
        sample = np.arange(curves.shape[0])

    if engine == "plotly":
        fig = go.Figure()
        if ice:
            for row in sample:
                fig.add_trace(
                    go.Scatter(
                        x=grid,
                        y=curves[row],
                        mode="lines",
                        line={"color": GRAY, "width": 1},
                        opacity=0.2,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=average,
                mode="lines",
                name="Partial dependence",
                line={"color": BLUE, "width": 3},
            )
        )
        fig.update_layout(
            title=title or f"Partial dependence: {names[index]}",
            xaxis_title=names[index],
            yaxis_title="Model output",
            template="plotly_white",
        )
        return _finish_plotly(fig, show=show, save_path=save_path)

    fig, ax = plt.subplots(figsize=figsize or (8.0, 5.0))
    if ice:
        for row in sample:
            ax.plot(grid, curves[row], color=GRAY, linewidth=0.8, alpha=0.2)
    ax.plot(grid, average, color=BLUE, linewidth=2.6, label="Partial dependence")

    ax.set_xlabel(names[index], fontsize=12)
    ax.set_ylabel("Model output", fontsize=12)
    ax.set_title(
        title or f"Partial dependence: {names[index]}", fontsize=13, loc="left"
    )
    ax.legend(frameon=False, fontsize=10)

    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    return _finish_matplotlib(fig, show=show, save_path=save_path)


def initjs() -> None:
    """Do nothing; kept so SHAP-style notebooks keep running unchanged.

    SHAP required this call to inject its JavaScript bundle before the force
    and text plots would render. XWhy renders those plots as static HTML and
    matplotlib figures, so there is no JavaScript to initialise.
    """
    return None


# ==============================================================================
# NUMERIC HELPERS
# ==============================================================================


def _feature_index(
    ind: int | str | None,
    names: Sequence[str],
    values: np.ndarray,
) -> int:
    """Resolve a feature reference to a column index.

    Args:
        ind: Feature index, feature name, or ``None`` for the most important.
        names: Feature names.
        values: Attribution or data matrix used to rank importance.

    Returns:
        int: The resolved column index.

    Raises:
        ValueError: If the name is unknown or the index is out of range.

    """
    n_features = np.atleast_2d(values).shape[1]

    if ind is None:
        return int(np.argmax(_global_importance(np.atleast_2d(values))))

    if isinstance(ind, str):
        lookup = list(names)
        if ind not in lookup:
            raise ValueError(
                f"Unknown feature {ind!r}. Available features: "
                f"{', '.join(lookup[:10])}"
                f"{'...' if len(lookup) > 10 else ''}"
            )
        return lookup.index(ind)

    index = int(ind)
    if not -n_features <= index < n_features:
        raise ValueError(
            f"Feature index {index} is out of range for {n_features} features."
        )
    return int(index % n_features)


def _numeric_data(
    data: np.ndarray | Sequence[Any] | None,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    """Coerce the raw data to a numeric matrix matching ``shape``, if possible."""
    if data is None:
        return None

    try:
        numeric = np.asarray(data, dtype=float)
    except (TypeError, ValueError):
        return None

    numeric = np.atleast_2d(numeric)
    if numeric.shape != shape:
        return None
    return numeric


def _robust_limits(values: np.ndarray | None) -> tuple[float, float]:
    """Return 5th/95th percentile limits, guarding against degenerate ranges."""
    if values is None:
        return 0.0, 1.0

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0

    low, high = np.percentile(finite, [5, 95])
    if low == high:
        low, high = float(finite.min()), float(finite.max())
    if low == high:
        return float(low) - 0.5, float(high) + 0.5
    return float(low), float(high)


def _density_offsets(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Compute vertical beeswarm offsets that spread overlapping points.

    Points falling in the same value bin are stacked alternately above and
    below the row centre, which is what gives the beeswarm its shape.

    Args:
        values: Attribution values for one feature.
        rng: Random generator used to break ties deterministically.

    Returns:
        np.ndarray: Vertical offsets in the range ``[-0.4, 0.4]``.

    """
    values = np.asarray(values, dtype=float)
    n_points = values.shape[0]
    if n_points == 0:
        return np.zeros(0)

    vmin, vmax = float(values.min()), float(values.max())
    span = vmax - vmin
    if span <= 0:
        return np.zeros(n_points)

    n_bins = 100
    quantised = np.round(n_bins * (values - vmin) / span)
    order = np.argsort(quantised + rng.normal(0, 1e-6, n_points))

    offsets = np.zeros(n_points)
    layer, last_bin = 0, -1.0
    for position in order:
        if quantised[position] != last_bin:
            layer = 0
        offsets[position] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
        last_bin = quantised[position]

    peak = float(np.max(np.abs(offsets)))
    if peak > 0:
        offsets = offsets / (peak + 1) * 0.8
    return offsets


def _pca_2d(values: np.ndarray) -> np.ndarray:
    """Project a matrix onto its first two principal components via SVD."""
    centred = values - values.mean(axis=0, keepdims=True)
    if centred.shape[1] < 2:
        return np.column_stack([centred[:, 0], np.zeros(centred.shape[0])])

    _, _, components = np.linalg.svd(centred, full_matrices=False)
    return cast(np.ndarray, centred @ components[:2].T)


def _order_instances(values: np.ndarray, strategy: str) -> np.ndarray:
    """Order instances for the heatmap according to ``strategy``.

    Args:
        values: Attribution matrix of shape ``(n_instances, n_features)``.
        strategy: ``"hclust"``, ``"output"`` or ``"none"``.

    Returns:
        np.ndarray: Instance indices in display order.

    """
    n_instances = values.shape[0]

    if strategy == "output":
        return np.argsort(values.sum(axis=1))

    # Hierarchical clustering is O(n^2) in memory, so fall back when large.
    if strategy == "hclust" and 2 < n_instances <= 2000:
        try:
            from scipy.cluster.hierarchy import leaves_list, linkage

            return np.asarray(leaves_list(linkage(values, method="average")))
        except (ImportError, ValueError):  # pragma: no cover - scipy edge cases
            return np.argsort(values.sum(axis=1))

    if strategy == "hclust":
        return np.argsort(values.sum(axis=1))

    return np.arange(n_instances)


def _significant_splits(series: np.ndarray, n_splits: int) -> list[int]:
    """Find split points where the attribution distribution shifts.

    Args:
        series: Attribution values in arrival order.
        n_splits: Number of candidate split points to test.

    Returns:
        list[int]: Instance indices at which the shift is significant after a
        Bonferroni correction.

    """
    n_points = series.shape[0]
    if n_points < 20 or n_splits < 1:
        return []

    try:
        from scipy.stats import ttest_ind
    except ImportError:  # pragma: no cover - scipy is a hard dependency
        return []

    threshold = 0.05 / n_splits
    step = max(1, n_points // n_splits)
    breaks: list[int] = []

    for split in range(step, n_points - step + 1, step):
        left, right = series[:split], series[split:]
        if left.size < 2 or right.size < 2:
            continue

        # Two constant segments make the t-statistic degenerate, so compare the
        # means directly instead of letting scipy divide by a zero variance.
        if np.isclose(left.var(), 0.0) and np.isclose(right.var(), 0.0):
            if not np.isclose(left.mean(), right.mean()):
                breaks.append(int(split))
            continue

        # A near-constant segment makes scipy warn about precision loss; the
        # p-value is still usable and the finiteness check below guards it.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, p_value = ttest_ind(left, right, equal_var=False)

        if np.isfinite(p_value) and p_value < threshold:
            breaks.append(int(split))

    return breaks


def _prepare_image_arrays(
    exp: Explanation,
    pixel_values: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise image attributions and backgrounds to a common layout.

    Args:
        exp: The image explanation.
        pixel_values: Optional background images.

    Returns:
        tuple: ``(values, backgrounds)`` where ``values`` has shape
        ``(n, H, W, n_outputs)`` and ``backgrounds`` has shape
        ``(n, H, W, 3)`` scaled to ``[0, 1]``.

    Raises:
        ValueError: If the attributions are not image-shaped.

    """
    values = np.asarray(exp.values, dtype=float)

    if values.ndim == 2:  # (H, W)
        values = values[np.newaxis, :, :, np.newaxis]
    elif values.ndim == 3:  # (n, H, W)
        values = values[..., np.newaxis]
    elif values.ndim == 4:  # (n, H, W, C) -> collapse channels to one map
        values = values.mean(axis=3)[..., np.newaxis]
    elif values.ndim == 5:  # (n, H, W, C, outputs)
        values = values.mean(axis=3)
    else:
        raise ValueError(
            "image() expects image-structured attributions of rank 2 to 5 but "
            f"received shape {values.shape}."
        )

    source = pixel_values if pixel_values is not None else exp.data
    backgrounds = _normalise_backgrounds(source, values.shape[0], values.shape[1:3])
    return values, backgrounds


def _normalise_backgrounds(
    source: Any,  # noqa: ANN401
    n_rows: int,
    hw: tuple[int, int],
) -> np.ndarray:
    """Coerce background images to ``(n_rows, H, W, 3)`` floats in ``[0, 1]``."""
    if source is None:
        return np.ones((n_rows, hw[0], hw[1], 3), dtype=float)

    try:
        images = np.asarray(source, dtype=float)
    except (TypeError, ValueError):
        return np.ones((n_rows, hw[0], hw[1], 3), dtype=float)

    if images.ndim == 2:
        images = images[np.newaxis, :, :, np.newaxis]
    elif images.ndim == 3:
        # Ambiguous: (H, W, C) for one image, or (n, H, W) for a grayscale batch.
        images = (
            images[np.newaxis]
            if images.shape[-1] in {1, 3, 4}
            else images[..., np.newaxis]
        )

    if images.ndim != 4 or images.shape[1:3] != hw:
        return np.ones((n_rows, hw[0], hw[1], 3), dtype=float)

    if images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    images = images[..., :3]

    peak = float(np.nanmax(np.abs(images))) if images.size else 0.0
    if peak > 1.0:
        images = images / 255.0 if peak > 2.0 else images / peak
    if float(np.nanmin(images)) < 0.0:
        images = (images + 1.0) / 2.0

    images = np.clip(np.nan_to_num(images), 0.0, 1.0)

    if images.shape[0] < n_rows:
        images = np.repeat(images[:1], n_rows, axis=0)
    return images[:n_rows]


def _background_for(exp: Explanation, hw: tuple[int, int]) -> np.ndarray:
    """Return a single background image sized ``hw`` for multimodal plots."""
    return cast(np.ndarray, _normalise_backgrounds(exp.data, 1, hw)[0])
