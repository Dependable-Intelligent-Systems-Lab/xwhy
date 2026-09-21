"""Decision (cumulative attribution path) plots."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objects as go
import scipy.cluster.hierarchy
import scipy.spatial
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .base import (
    RED_BLUE,
    _check_backend,
    _finish_matplotlib,
    _finish_plotly,
    _format_value,
    _resolve_names,
)

# ---------------------------------------------------------------------------
# Link transforms
# ---------------------------------------------------------------------------


class Link:
    """Base class for x-axis link functions."""

    def __str__(self) -> str:
        """Return a human-readable name for the link."""
        return self.__class__.__name__


class IdentityLink(Link):
    """Identity (no-op) link."""

    def __str__(self) -> str:
        r"""Return ``\"identity\"``."""
        return "identity"

    @staticmethod
    def f(x: float | NDArray[Any]) -> float | NDArray[Any]:
        """Forward transform (identity)."""
        return x

    @staticmethod
    def finv(x: float | NDArray[Any]) -> float | NDArray[Any]:
        """Inverse transform (identity)."""
        return x


class LogitLink(Link):
    """Logit link mapping probabilities to log-odds and back."""

    def __str__(self) -> str:
        r"""Return ``\"logit\"``."""
        return "logit"

    @staticmethod
    def f(
        x: float | NDArray[Any],
        epsilon: float = 1e-15,
    ) -> float | NDArray[Any]:
        """Map probability to log-odds."""
        x_clipped = np.clip(x, epsilon, 1 - epsilon)
        return np.log(x_clipped / (1 - x_clipped))  # type: ignore[no-any-return]

    @staticmethod
    def finv(x: float | NDArray[Any]) -> float | NDArray[Any]:
        """Map log-odds to probability."""
        return 1 / (1 + np.exp(-x))


def convert_to_link(val: str | Link) -> Link:
    r"""Convert a string or :class:`Link` instance into a concrete link.

    Parameters
    ----------
    val :
        ``\"identity\"``, ``\"logit\"``, or an existing :class:`Link`.

    Returns
    -------
    Link
        Concrete link instance.

    Raises
    ------
    TypeError
        If *val* is not recognised.

    """
    if isinstance(val, Link):
        return val
    if val == "identity":
        return IdentityLink()
    if val == "logit":
        return LogitLink()
    msg = "Passed link object must be a subclass of Link"
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def hclust_ordering(
    x: npt.NDArray[Any],
    metric: str = "sqeuclidean",
    *,
    anchor_first: bool = False,
) -> npt.NDArray[Any]:
    """Return an optimal leaf ordering from hierarchical clustering.

    Parameters
    ----------
    x :
        2-D array of samples (rows) x features (columns), or the transpose
        used by the decision plot (features as rows).
    metric :
        Distance metric passed to :func:`scipy.spatial.distance.pdist`.
    anchor_first :
        Reserved for API compatibility; currently unused.

    Returns
    -------
    NDArray
        Integer leaf order.

    """
    _ = anchor_first
    dist = scipy.spatial.distance.pdist(x, metric)
    cluster_matrix = scipy.cluster.hierarchy.complete(dist)
    return scipy.cluster.hierarchy.leaves_list(  # type: ignore[no-any-return]
        scipy.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, dist)
    )


def _change_shap_base_value(
    base_value: float,
    new_base_value: float,
    shap_values: NDArray[Any],
) -> NDArray[Any]:
    """Shift SHAP values so they are relative to *new_base_value*.

    Assumes *base_value* / *new_base_value* are scalars and *shap_values* is
    2-D (matrix) or 3-D (interaction cube).
    """
    if shap_values.ndim == 2:
        return shap_values + (base_value - new_base_value) / shap_values.shape[1]  # type: ignore[no-any-return]

    main_effects = shap_values.shape[1]
    all_effects = main_effects * (main_effects + 1) // 2
    # Interaction effects are halved, so divide by 2 as well.
    temp = (base_value - new_base_value) / all_effects / 2
    shifted = shap_values + temp
    idx = np.diag_indices_from(shifted[0])
    shifted[:, idx[0], idx[1]] += temp
    return shifted  # type: ignore[no-any-return]


class DecisionPlotResult:
    """Optional return value of :func:`_shap_decision`.

    Attributes can be reused to apply the same scale and feature ordering to
    subsequent decision plots.
    """

    def __init__(
        self,
        base_value: float,
        shap_values: NDArray[Any],
        feature_names: list[str],
        feature_idx: NDArray[Any],
        xlim: tuple[float, float],
    ) -> None:
        """Store plot scale and ordering metadata.

        Parameters
        ----------
        base_value :
            Base value used in the plot (after ``new_base_value`` if set).
        shap_values :
            SHAP values re-ordered by ``feature_order``.
        feature_names :
            Feature names in the ordered positions.
        feature_idx :
            Index used to order features; pass as ``feature_order`` later.
        xlim :
            X-axis limits for consistent scaling across plots.

        """
        self.base_value = base_value
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.feature_idx = feature_idx
        self.xlim = xlim


# ---------------------------------------------------------------------------
# Matplotlib rendering
# ---------------------------------------------------------------------------


def _decision_plot_matplotlib(
    base_value: float,
    cumsum: NDArray[Any],
    ascending: bool,
    feature_display_count: int,
    features: NDArray[Any] | None,
    feature_names: list[str],
    highlight: Any,  # noqa: ANN401
    plot_color: Any,  # noqa: ANN401
    axis_color: str,
    y_demarc_color: str,
    xlim: tuple[float, float],
    alpha: float,
    color_bar: bool,
    auto_size_plot: bool,
    title: str | None,
    show: bool,
    legend_labels: list[str] | None,
    legend_location: str,
) -> None:
    """Render a decision plot with matplotlib (side-effect on current axes)."""
    _ = show
    row_height = 0.4
    if auto_size_plot:
        plt.gcf().set_size_inches(8, feature_display_count * row_height + 1.5)

    plt.axvline(x=base_value, color="#999999", zorder=-1)

    for i in range(1, feature_display_count):
        plt.axhline(y=i, color=y_demarc_color, lw=0.5, dashes=(1, 5), zorder=-1)

    linestyle = np.array("-", dtype=object)
    linestyle = np.repeat(linestyle, cumsum.shape[0])
    linewidth = np.repeat(1, cumsum.shape[0])
    if highlight is not None:
        linestyle[highlight] = "-."
        linewidth[highlight] = 2

    ax = plt.gca()
    ax.set_xlim(xlim)
    mappable = ScalarMappable(cmap=plot_color)
    mappable.set_clim(xlim)
    y_pos = np.arange(0, feature_display_count + 1)
    lines = []
    for i in range(cumsum.shape[0]):
        drawn = plt.plot(
            cumsum[i, :],
            y_pos,
            color=mappable.to_rgba(cumsum[i, -1], alpha),
            linewidth=linewidth[i],
            linestyle=linestyle[i],
        )
        lines.append(drawn[0])

    interaction_label = next((s for s in feature_names if " *\n" in s), None)
    fontsize = 13 if interaction_label is None else 9

    if (cumsum.shape[0] == 1) and (features is not None):
        renderer = plt.gcf().canvas.get_renderer()  # type: ignore[attr-defined]
        inverter = plt.gca().transData.inverted()
        y_pos_labels = y_pos + 0.5
        for i in range(feature_display_count):
            v: Any = features[0, i]
            if isinstance(v, str):
                text_v = f"({str(v).strip()})"
            else:
                text_v = f"({float(v):,.3f}".rstrip("0").rstrip(".") + ")"
            text_artist = ax.text(
                float(np.max(cumsum[0, i : (i + 2)])),
                y_pos_labels[i],
                "  " + text_v,
                fontsize=fontsize,
                horizontalalignment="left",
                verticalalignment="center_baseline",
                color="#666666",
            )
            bbox = inverter.transform_bbox(
                text_artist.get_window_extent(renderer=renderer)
            )
            if bbox.xmax > xlim[1]:
                text_artist.set_text(text_v + "  ")
                text_artist.set_x(float(np.min(cumsum[0, i : (i + 2)])))
                text_artist.set_horizontalalignment("right")
                bbox = inverter.transform_bbox(
                    text_artist.get_window_extent(renderer=renderer)
                )
                if bbox.xmin < xlim[0]:
                    text_artist.set_text(text_v)
                    text_artist.set_x(xlim[0])
                    text_artist.set_horizontalalignment("left")

    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labeltop=True)
    plt.yticks(
        np.arange(feature_display_count) + 0.5,
        feature_names,
        fontsize=fontsize,
    )
    ax.tick_params("x", labelsize=11)
    plt.ylim(0, feature_display_count)
    plt.xlabel("Model output value", fontsize=13)

    if color_bar:
        bar_mappable = ScalarMappable(cmap=plot_color)
        bar_mappable.set_array(np.array([0, 1]))
        plt.ylim(0, feature_display_count + 0.25)
        ax_cb = ax.inset_axes(
            (xlim[0], feature_display_count, xlim[1] - xlim[0], 0.25),
            transform=ax.transData,
        )
        colorbar = plt.colorbar(
            bar_mappable, ticks=[0, 1], orientation="horizontal", cax=ax_cb
        )
        colorbar.set_ticklabels([])
        colorbar.ax.tick_params(labelsize=11, length=0)
        colorbar.set_alpha(alpha)
        colorbar.outline.set_visible(False)
        plt.sca(ax)

    if title:
        plt.title(title)

    if ascending:
        plt.gca().invert_yaxis()

    if legend_labels is not None:
        ax.legend(handles=lines, labels=legend_labels, loc=legend_location)  # type: ignore[call-overload]


def _shap_decision(
    base_value: float | NDArray[Any],
    shap_values: NDArray[Any],
    features: NDArray[Any] | pd.Series | pd.DataFrame | list[Any] | None = None,
    feature_names: Sequence[str] | NDArray[Any] | None = None,
    feature_order: str | list[int] | NDArray[Any] | None = "importance",
    feature_display_range: slice | range | None = None,
    highlight: Any = None,  # noqa: ANN401
    link: str | Link = "identity",
    plot_color: Any = None,  # noqa: ANN401
    axis_color: str = "#333333",
    y_demarc_color: str = "#333333",
    alpha: float | None = None,
    color_bar: bool = True,
    auto_size_plot: bool = True,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    show: bool = True,
    return_objects: bool = False,
    ignore_warnings: bool = False,
    new_base_value: float | None = None,
    legend_labels: list[str] | None = None,
    legend_location: str = "best",
) -> DecisionPlotResult | None:
    """Visualize model decisions using cumulative SHAP values.

    Each plotted line explains a single model prediction. See the public
    :func:`decision` wrapper for a higher-level API.

    Returns
    -------
    DecisionPlotResult or None
        Metadata when ``return_objects=True``, otherwise ``None``.

    """
    if isinstance(base_value, np.ndarray) and len(base_value) == 1:
        base_value = float(base_value[0])

    if isinstance(base_value, list) or isinstance(shap_values, list):
        msg = (
            "Looks like multi output. Try base_value[i] and shap_values[i], "
            "or use multioutput_decision_plot()."
        )
        raise TypeError(msg)

    if not isinstance(shap_values, np.ndarray):
        msg = "The shap_values arg is the wrong type. Try explainer.shap_values()."
        raise TypeError(msg)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    observation_count = shap_values.shape[0]
    feature_count = shap_values.shape[1]

    # Normalise features / names from pandas and list inputs
    names_list: list[str] | None
    if isinstance(feature_names, np.ndarray):
        names_list = feature_names.tolist()
    elif feature_names is not None:
        names_list = list(feature_names)
    else:
        names_list = None

    features_arr: NDArray[Any] | None
    if isinstance(features, pd.DataFrame):
        if names_list is None:
            names_list = features.columns.to_list()
        features_arr = features.values
    elif isinstance(features, pd.Series):
        if names_list is None:
            names_list = features.index.to_list()
        features_arr = features.values
    elif isinstance(features, list):
        if names_list is None:
            names_list = [str(x) for x in features]
        features_arr = None
    elif features is not None and features.ndim == 1 and names_list is None:
        names_list = features.tolist()
        features_arr = None
    elif features is None:
        features_arr = None
    else:
        features_arr = np.asarray(features)

    if features_arr is not None and not isinstance(features_arr, np.ndarray):
        msg = "The features arg uses an unsupported type."
        raise TypeError(msg)
    if features_arr is not None and features_arr.ndim == 1:
        features_arr = features_arr.reshape(1, -1)

    if names_list is None:
        names_list = [f"Feature {i}" for i in range(feature_count)]
    elif len(names_list) != feature_count:
        msg = (
            "The feature_names arg must include all features represented "
            "in shap_values."
        )
        raise ValueError(msg)

    # Flatten interaction cube to matrix + interaction names
    if shap_values.ndim == 3:
        triu_count = feature_count * (feature_count - 1) // 2
        idx_diag = np.diag_indices_from(shap_values[0])
        idx_triu = np.triu_indices_from(shap_values[0], 1)
        flat = np.empty(
            (observation_count, feature_count + triu_count),
            dtype=shap_values.dtype,
        )
        flat[:, :feature_count] = shap_values[:, idx_diag[0], idx_diag[1]]
        flat[:, feature_count:] = shap_values[:, idx_triu[0], idx_triu[1]] * 2
        shap_values = flat

        interaction_names: list[str] = [""] * shap_values.shape[1]
        interaction_names[:feature_count] = names_list
        for i, row, col in zip(
            range(feature_count, shap_values.shape[1]),
            idx_triu[0],
            idx_triu[1],
            strict=False,
        ):
            interaction_names[i] = f"{names_list[row]} *\n{names_list[col]}"
        names_list = interaction_names
        feature_count = shap_values.shape[1]
        features_arr = None

    # Feature order
    if isinstance(feature_order, list):
        feature_idx = np.array(feature_order)
    elif isinstance(feature_order, np.ndarray):
        feature_idx = feature_order
    elif feature_order is None or (
        isinstance(feature_order, str) and feature_order.lower() == "none"
    ):
        feature_idx = np.arange(feature_count)
    elif feature_order == "importance":
        feature_idx = np.argsort(np.sum(np.abs(shap_values), axis=0))
    elif feature_order == "hclust":
        feature_idx = np.array(hclust_ordering(shap_values.transpose()))
    else:
        msg = (
            "The feature_order arg requires 'importance', 'hclust', 'none', "
            "or an integer list/array of feature indices."
        )
        raise ValueError(msg)

    if (feature_idx.shape != (feature_count,)) or (
        not np.issubdtype(feature_idx.dtype, np.integer)
    ):
        msg = (
            "A list or array has been specified for the feature_order arg. "
            "The length must match the feature count and the data type must "
            "be integer."
        )
        raise ValueError(msg)

    # feature_display_range -> slice
    if feature_display_range is None:
        feature_display_range = slice(-1, -21, -1)
    elif not isinstance(feature_display_range, (slice, range)):
        msg = "The feature_display_range arg requires a slice or a range."
        raise TypeError(msg)
    elif feature_display_range.step not in (-1, 1, None):
        msg = "The feature_display_range arg supports a step of 1, -1, or None."
        raise ValueError(msg)
    elif isinstance(feature_display_range, range):
        c_min = int(np.iinfo(np.intp).min)
        feature_display_range = slice(
            feature_display_range.start if feature_display_range.start >= 0 else c_min,
            feature_display_range.stop if feature_display_range.stop >= 0 else c_min,
            feature_display_range.step,
        )

    if new_base_value is not None:
        base_scalar = float(base_value)
        shap_values = _change_shap_base_value(base_scalar, new_base_value, shap_values)
        base_value = new_base_value

    base_scalar = float(base_value)
    display_indices = feature_display_range.indices(feature_count)
    ascending = True
    start_i, stop_i, step_i = display_indices
    if step_i == -1:
        ascending = False
        start_i, stop_i, step_i = stop_i + 1, start_i + 1, 1
    feature_display_count = stop_i - start_i
    shap_values = shap_values[:, feature_idx]

    if start_i == 0:
        cumsum = np.empty(
            (observation_count, feature_display_count + 1),
            dtype=shap_values.dtype,
        )
        cumsum[:, 0] = base_scalar
        cumsum[:, 1:] = base_scalar + np.nancumsum(shap_values[:, 0:stop_i], axis=1)
    else:
        cumsum = (
            base_scalar + np.nancumsum(shap_values, axis=1)[:, (start_i - 1) : stop_i]
        )

    names_arr = np.array(names_list, dtype=object)
    feature_names_display = names_arr[feature_idx[start_i:stop_i]].tolist()
    ordered_names = names_arr[feature_idx].tolist()
    features_display = (
        None if features_arr is None else features_arr[:, feature_idx[start_i:stop_i]]
    )

    if not ignore_warnings:
        if observation_count > 2000:
            msg = (
                f"Plotting {observation_count} observations may be slow. "
                "Consider subsampling or set ignore_warnings=True to ignore "
                "this message."
            )
            raise RuntimeError(msg)
        if feature_display_count > 200:
            msg = (
                f"Plotting {feature_display_count} features may create a very "
                "large plot. Set ignore_warnings=True to ignore this message."
            )
            raise RuntimeError(msg)
        if feature_count * observation_count > 100_000_000:
            msg = (
                f"Processing SHAP values for {feature_count} features over "
                f"{observation_count} observations may be slow. Set "
                "ignore_warnings=True to ignore this message."
            )
            raise RuntimeError(msg)

    create_xlim = xlim is None
    link_obj = convert_to_link(link)
    base_value_saved = base_scalar
    if isinstance(link_obj, LogitLink):
        base_scalar = float(link_obj.finv(base_scalar))
        cumsum = np.asarray(link_obj.finv(cumsum))
        if create_xlim:
            xlim = (-0.02, 1.02)
    elif create_xlim:
        xmin = float(min(float(cumsum.min()), base_scalar))
        xmax = float(max(float(cumsum.max()), base_scalar))
        n_left, m_right = (base_scalar - xmin), (xmax - base_scalar)
        if n_left > m_right:
            xlim = (base_scalar - n_left, base_scalar + m_right)
        else:
            xlim = (base_scalar - m_right, base_scalar + m_right)
        margin = (xlim[1] - xlim[0]) * 0.02
        xlim = (xlim[0] - margin, xlim[1] + margin)

    assert xlim is not None

    if alpha is None:
        alpha = 1.0

    if plot_color is None:
        plot_color = RED_BLUE

    _decision_plot_matplotlib(
        base_scalar,
        cumsum,
        ascending,
        feature_display_count,
        features_display,
        feature_names_display,
        highlight,
        plot_color,
        axis_color,
        y_demarc_color,
        xlim,
        alpha,
        color_bar,
        auto_size_plot,
        title,
        show,
        legend_labels,
        legend_location,
    )

    if not return_objects:
        return None

    return DecisionPlotResult(
        base_value_saved,
        shap_values,
        [str(n) for n in ordered_names],
        feature_idx,
        xlim,
    )


def decision(
    base_value: float | NDArray[Any],
    shap_values: NDArray[Any],
    features: NDArray[Any] | Sequence[Any] | None = None,
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
) -> Figure | Any | None:  # noqa: ANN401
    r"""Visualise cumulative attributions as decision paths.

    Each observation is drawn as a line that starts at the expected value and
    accumulates one feature contribution per row, ending at the prediction.

    Parameters
    ----------
    base_value :
        The model's expected output.
    shap_values :
        Attributions of shape ``(n_features,)`` or ``(n_instances, n_features)``.
    features :
        Optional raw feature values, used for row labels when a single
        instance is plotted.
    feature_names :
        Optional feature names.
    max_display :
        Maximum number of feature rows to draw.
    show :
        Whether to display the figure.
    save_path :
        Optional path to write the figure to.
    backend :
        ``\"matplotlib\"`` or ``\"plotly\"``.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches.
    alpha :
        Line opacity. Defaults to a density-aware value for plotly.
    **kwargs :
        Extra options forwarded to :func:`_shap_decision`
        (``feature_order``, ``link``, ``highlight``, …).

    Returns
    -------
    Figure | plotly.graph_objects.Figure | None
        The figure when ``show`` is False (and no host finish-helper swallows
        it), otherwise ``None``.

    """
    allowed = frozenset({"matplotlib", "plotly"})
    if _check_backend is not None:
        engine = _check_backend(backend, allowed)
    else:
        if backend not in allowed:
            msg = f"Unsupported backend: {backend}"
            raise ValueError(msg)
        engine = backend

    values = np.atleast_2d(np.asarray(shap_values, dtype=float))
    n_instances, n_features = values.shape

    if _resolve_names is not None:
        names = _resolve_names(feature_names, n_features)
    elif feature_names is not None:
        names = list(feature_names)
    else:
        names = [f"Feature {i}" for i in range(n_features)]

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
            fmt = _format_value if _format_value is not None else str
            labels = [
                f"{names[i]} = {fmt(raw[0, i])}" if n_instances == 1 else names[i]
                for i in order
            ]

    paths = np.concatenate(
        [np.full((n_instances, 1), base), base + np.cumsum(ordered, axis=1)],
        axis=1,
    )
    predictions = paths[:, -1]
    vmin, vmax = float(predictions.min()), float(predictions.max())
    norm = Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1e-9)
    y_positions = np.arange(len(order) + 1) - 0.5

    if engine == "plotly":
        if go is None:
            msg = "plotly is required for the plotly backend"
            raise ImportError(msg)
        fig_pl = go.Figure()
        for i in range(n_instances):
            if callable(RED_BLUE):
                color = RED_BLUE(norm(predictions[i]))
                rgb = (
                    f"rgb({int(color[0] * 255)},"
                    f"{int(color[1] * 255)},{int(color[2] * 255)})"
                )
            else:
                rgb = "rgb(30,136,229)"
            fig_pl.add_trace(
                go.Scatter(
                    x=paths[i],
                    y=y_positions,
                    mode="lines",
                    line={"color": rgb, "width": 1.4},
                    opacity=alpha or max(0.15, min(1.0, 30.0 / n_instances)),
                    showlegend=False,
                    hovertemplate="output: %{x:.4f}<extra></extra>",
                )
            )
        fig_pl.update_layout(
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
        if _finish_plotly is not None:
            return _finish_plotly(fig_pl, show=show, save_path=save_path)
        return None if show else fig_pl

    if figsize is not None:
        fig, _ax = plt.subplots(figsize=figsize)
        auto_size_plot = False
    else:
        fig, _ax = plt.subplots()
        auto_size_plot = True

    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    _ax.patch.set_facecolor("none")
    _ax.patch.set_alpha(0.0)

    feature_display_range = (
        slice(-1, -max_display - 1, -1)
        if max_display is not None
        else slice(None, None, -1)
    )

    _shap_decision(
        base_value=base_value,
        shap_values=shap_values,
        features=features,
        feature_names=feature_names,
        feature_order=kwargs.get("feature_order", "importance"),
        feature_display_range=feature_display_range,
        highlight=kwargs.get("highlight"),
        link=kwargs.get("link", "identity"),
        plot_color=kwargs.get("plot_color"),
        axis_color=kwargs.get("axis_color", "#333333"),
        y_demarc_color=kwargs.get("y_demarc_color", "#333333"),
        alpha=alpha,
        color_bar=kwargs.get("color_bar", True),
        auto_size_plot=auto_size_plot,
        title=title,
        xlim=kwargs.get("xlim"),
        show=False,
        return_objects=False,
        ignore_warnings=kwargs.get("ignore_warnings", False),
        new_base_value=kwargs.get("new_base_value"),
        legend_labels=kwargs.get("legend_labels"),
        legend_location=kwargs.get("legend_location", "best"),
    )

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig
