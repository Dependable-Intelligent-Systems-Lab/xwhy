"""Dependence scatter plots of feature values versus attributions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.markers import MarkerStyle
from numpy.typing import NDArray

from .base import (
    RED_BLUE,
    DimensionError,
    Explanation,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    convert_name,
)


def parse_axis_limit(
    ax_limit: Any,  # noqa: ANN401
    ax_values: NDArray[Any],
    *,
    is_shap_axis: bool,
) -> float | None:
    r"""Resolve an axis limit from percentile strings or Explanation objects.

    Parameters
    ----------
    ax_limit :
        Lower or upper bound: a float, ``None``, ``\"percentile(x)\"``, or an
        aggregated Explanation column.
    ax_values :
        Values on the axis (SHAP or feature values).
    is_shap_axis :
        When *ax_limit* is an Explanation, use ``.values`` if True else
        ``.data``.

    Returns
    -------
    float | None
        Resolved numeric limit, or ``None`` if unrestricted.

    """
    if isinstance(ax_limit, str):
        try:
            percentage = float(ax_limit.removeprefix("percentile(").removesuffix(")"))
        except ValueError as err:
            msg = "Only strings of the format `percentile(x)` are supported."
            raise ValueError(msg) from err
        return float(np.nanpercentile(ax_values, percentage))
    if Explanation is not None and isinstance(ax_limit, Explanation):
        return float(ax_limit.values) if is_shap_axis else float(ax_limit.data)  # type: ignore[arg-type]
    if type(ax_limit).__name__ == "Explanation":
        return float(ax_limit.values) if is_shap_axis else float(ax_limit.data)
    return ax_limit  # type: ignore[no-any-return]


def _suggest_buffered_limits(
    ax_min: float | None,
    ax_max: float | None,
    values: NDArray[Any],
) -> tuple[float, float]:
    """Fill missing limits with a small buffer beyond the data range."""
    nan_max = float(np.nanmax(values)) if ax_max is None else ax_max
    nan_min = float(np.nanmin(values)) if ax_min is None else ax_min
    buffer = (nan_max - nan_min) / 20
    if ax_min is None:
        ax_min = nan_min - buffer
    if ax_max is None:
        ax_max = nan_max + buffer
    return ax_min, ax_max


def _suggest_x_jitter(values: NDArray[Any]) -> float:
    """Suggest x-axis jitter from the spacing of unique feature values."""
    unique_vals = np.sort(np.unique(values))
    if len(unique_vals) < 2:
        return 0.0
    try:
        diffs = np.diff(unique_vals)
        positive = diffs[diffs > 1e-8]
        min_dist = float(np.min(positive)) if len(positive) else 1.0
    except (TypeError, ValueError):
        min_dist = 1.0

    num_points_per_value = len(values) / len(unique_vals)
    if num_points_per_value < 10:
        return 0.0
    if num_points_per_value < 100:
        return min_dist * 0.1
    return min_dist * 0.2


def encode_array_if_needed(
    arr: npt.NDArray[Any],
    dtype: type[Any] = np.float64,
) -> npt.NDArray[Any]:
    """Cast *arr* to *dtype*, or label-encode non-numeric values."""
    try:
        return arr.astype(dtype)
    except (ValueError, TypeError):
        unique_values = np.unique(arr)
        encoding = {string: index for index, string in enumerate(unique_values)}
        return np.array([encoding[string] for string in arr], dtype=dtype)


def approximate_interactions(
    index: str | int,
    shap_values: npt.NDArray[Any],
    x_data: npt.NDArray[Any] | pd.DataFrame,
    feature_names: list[str] | npt.NDArray[Any] | pd.Index | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[Any]:
    """Order features by approximate interaction strength with *index*.

    Bins SHAP values along the target feature and measures correlation with
    each other feature. Prefer true interaction values when available.

    Parameters
    ----------
    index :
        Target feature name or column index.
    shap_values :
        Attribution matrix (``n_samples x n_features``).
    x_data :
        Feature matrix (array or DataFrame).
    feature_names :
        Optional names used to resolve string *index*.
    rng :
        Optional NumPy random generator for subsampling large matrices.

    Returns
    -------
    NDArray
        Feature indices sorted by descending interaction strength.

    """
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(x_data, pd.DataFrame):
        if feature_names is None:
            feature_names = x_data.columns
        matrix = x_data.values
    else:
        matrix = np.asarray(x_data)

    col = convert_name(index, shap_values, feature_names)
    if not isinstance(col, int):
        msg = f"Could not resolve feature index from {index!r}"
        raise TypeError(msg)

    n_rows = matrix.shape[0]
    if n_rows > 10_000:
        inds = rng.choice(n_rows, size=10_000, replace=False)
    else:
        inds = np.arange(n_rows)

    x_col = matrix[inds, col]
    srt = np.argsort(x_col)
    shap_ref = shap_values[inds, col][srt]
    inc = max(min(int(len(x_col) / 10.0), 50), 1)

    interactions: list[float] = []
    for i in range(matrix.shape[1]):
        encoded = encode_array_if_needed(matrix[inds, i][srt], dtype=float)

        def _corr_score(
            other: NDArray[Any],
            *,
            feature_i: int = i,
        ) -> float:
            score = 0.0
            if feature_i == col or float(np.sum(np.abs(other))) < 1e-8:
                return score
            for j in range(0, len(x_col), inc):
                chunk_o = other[j : j + inc]
                chunk_s = shap_ref[j : j + inc]
                if np.std(chunk_o) > 0 and np.std(chunk_s) > 0:
                    score += abs(float(np.corrcoef(chunk_s, chunk_o)[0, 1]))
            return score

        val_v = _corr_score(encoded)
        nan_v = _corr_score(np.isnan(encoded).astype(float))
        interactions.append(max(val_v, nan_v))

    return np.argsort(-np.abs(interactions))


def _plot_histogram(
    ax: Any,  # noqa: ANN401
    xv: NDArray[Any],
    xv_no_jitter: NDArray[Any],
) -> None:
    """Add a density histogram of *xv* on a twin axes."""
    ax2 = cast("Any", ax.twinx())
    xlim = ax.get_xlim()
    xvals = np.unique(xv_no_jitter)

    bins: list[float] | int
    if (
        len(xvals) / max(len(xv_no_jitter), 1) < 0.2
        and len(xvals) < 75
        and np.max(xvals) < 75
        and np.min(xvals) >= 0
    ):
        max_x = int(np.max(xvals))
        bins = [i - 0.5 for i in range(max_x + 1)]
        bins.append(max_x + 0.5)
        lim = (
            np.floor(np.min(xvals) - 0.5) + 0.5,
            np.ceil(np.max(xvals) + 0.5) - 0.5,
        )
        ax.set_xlim(lim)
    elif len(xv_no_jitter) >= 500:
        bins = 50
    elif len(xv_no_jitter) >= 200:
        bins = 20
    elif len(xv_no_jitter) >= 100:
        bins = 10
    else:
        bins = 5

    ax2.hist(
        xv[~np.isnan(xv)],
        bins,
        density=False,
        facecolor="#000000",
        alpha=0.1,
        range=(xlim[0], xlim[1]),
        zorder=-1,
    )
    ax2.set_ylim(0, len(xv))
    ax2.xaxis.set_ticks_position("bottom")
    ax2.yaxis.set_ticks_position("left")
    ax2.yaxis.set_ticks([])
    for side in ("right", "top", "left", "bottom"):
        ax2.spines[side].set_visible(False)


def _feature_mask_excluding(
    names: NDArray[Any] | list[Any],
    exclude: str | int | Any,  # noqa: ANN401
) -> NDArray[Any]:
    """Boolean mask selecting entries of *names* that are not *exclude*.

    Uses element-wise inequality so a single-element comparison never
    produces a Python ``bool`` (avoids deprecated ``~bool`` on 3.16+).
    """
    names_arr = np.asarray(names, dtype=object)
    return names_arr != exclude  # type: ignore[no-any-return]


def _shap_scatter(
    shap_values: Any,  # noqa: ANN401
    color: str | Any | None = "#1E88E5",  # noqa: ANN401
    hist: bool = True,
    axis_color: str = "#333333",
    cmap: Any = None,  # noqa: ANN401
    dot_size: float = 16,
    x_jitter: float | Literal["auto"] = "auto",
    alpha: float = 1.0,
    title: str | None = None,
    xmin: Any = None,  # noqa: ANN401
    xmax: Any = None,  # noqa: ANN401
    ymin: Any = None,  # noqa: ANN401
    ymax: Any = None,  # noqa: ANN401
    overlay: dict[str, Any] | None = None,
    ax: Any = None,  # noqa: ANN401
    ylabel: str = "XWhy value",
    show: bool = True,
    *,
    rng: np.random.Generator | None = None,
) -> Any:  # noqa: ANN401
    r"""Create a SHAP dependence scatter plot, optionally coloured by interaction.

    Parameters
    ----------
    shap_values :
        Single-column Explanation, or multi-column for a subplot grid.
    color :
        Fixed colour string or Explanation used to colour points by interaction.
    hist :
        Draw a light x-axis density histogram.
    axis_color :
        Tick and spine colour.
    cmap :
        Colormap for interaction colouring (default ``RED_BLUE``).
    dot_size :
        Scatter marker size.
    x_jitter :
        ``\"auto\"`` or a float scale for categorical jitter.
    alpha :
        Marker opacity.
    title :
        Optional axes title.
    xmin, xmax, ymin, ymax :
        Axis limits (float, percentile string, or aggregated Explanation).
    overlay :
        Optional named overlay curves.
    ax :
        Optional matplotlib Axes (single-feature only).
    ylabel :
        Y-axis label for multi-feature layouts.
    show :
        Call ``plt.show()`` when True.
    rng :
        Optional NumPy random generator for shuffle and jitter.

    Returns
    -------
    Axes | None
        Axes when ``show`` is False; otherwise ``None``.

    """
    if cmap is None:
        cmap = RED_BLUE
    if rng is None:
        rng = np.random.default_rng()

    is_explanation = (
        (Explanation is not None and isinstance(shap_values, Explanation))
        or type(shap_values).__name__ == "Explanation"
        or hasattr(shap_values, "values")
    )
    if not is_explanation:
        msg = "The shap_values parameter must be a xwhy.plots.visualization.Explanation"
        " object!"
        raise TypeError(msg)

    # Multi-column layout
    feat_names = shap_values.feature_names
    if (
        not isinstance(feat_names, str)
        and feat_names is not None
        and len(feat_names) > 1
    ):
        if ax is not None:
            msg = "The ax parameter is not supported when plotting multiple features"
            raise ValueError(msg)
        inds = np.argsort(np.abs(shap_values.values).mean(0))
        ymin_b = parse_axis_limit(ymin, shap_values.values, is_shap_axis=True)
        ymax_b = parse_axis_limit(ymax, shap_values.values, is_shap_axis=True)
        ymin_b, ymax_b = _suggest_buffered_limits(ymin_b, ymax_b, shap_values.values)
        plt.subplots(1, len(inds), figsize=(min(6 * len(inds), 15), 5))
        for i in inds:
            sub_ax = plt.subplot(1, len(inds), int(i) + 1)
            _shap_scatter(
                shap_values[:, i],
                color=color,
                show=False,
                ax=sub_ax,
                ymin=ymin_b,
                ymax=ymax_b,
                cmap=cmap,
                rng=rng,
            )
            if overlay is not None:
                line_styles = ["solid", "dotted", "dashed"]
                for j, name in enumerate(overlay):
                    vals = overlay[name]
                    if isinstance(vals[i][0][0], (float, int)):
                        plt.plot(
                            vals[i][0],
                            vals[i][1],
                            color="#000000",
                            linestyle=line_styles[j],
                            label=name,
                        )
            if i == inds[0]:
                sub_ax.set_ylabel(ylabel)
            else:
                sub_ax.set_ylabel("")
                sub_ax.set_yticks([])
                sub_ax.spines["left"].set_visible(False)
        if overlay is not None:
            plt.legend()
        if show:
            plt.show()
        return None

    if len(shap_values.shape) != 1:
        msg = (
            "The passed Explanation object has multiple columns. Please pass "
            "a single feature column to scatter like: shap_values[:, column]"
        )
        raise DimensionError(msg)

    feature_names: list[Any] = [shap_values.feature_names]
    ind = 0
    shap_values_arr = shap_values.values.reshape(-1, 1)
    features = shap_values.data.reshape(-1, 1)
    if shap_values.display_data is None:
        display_features = features
    else:
        display_features = shap_values.display_data.reshape(-1, 1)
    interaction_index: str | int | None = None

    if isinstance(color, np.ndarray):
        color = (
            Explanation(values=color, base_values=None, data=color)  # type: ignore[arg-type]
            if Explanation is not None
            else color
        )

    color_is_explanation = (
        Explanation is not None and isinstance(color, Explanation)
    ) or type(color).__name__ == "Explanation"
    if color_is_explanation:
        shap_values2 = color
        if issubclass(type(shap_values2.feature_names), (str, int)):  # type: ignore[union-attr]
            feature_names.append(shap_values2.feature_names)  # type: ignore[union-attr]
            shap_values_arr = np.hstack(
                [
                    shap_values_arr,
                    shap_values2.values.reshape(-1, len(feature_names) - 1),  # type: ignore[union-attr]
                ]
            )
            features = np.hstack(
                [features, shap_values2.data.reshape(-1, len(feature_names) - 1)]  # type: ignore[union-attr]
            )
            if shap_values2.display_data is None:  # type: ignore[union-attr]
                display_features = np.hstack(
                    [
                        display_features,
                        shap_values2.data.reshape(-1, len(feature_names) - 1),  # type: ignore[union-attr]
                    ]
                )
            else:
                display_features = np.hstack(
                    [
                        display_features,
                        shap_values2.display_data.reshape(-1, len(feature_names) - 1),  # type: ignore[union-attr]
                    ]
                )
        else:
            feature_names2 = np.asarray(shap_values2.feature_names)  # type: ignore[union-attr]
            # Element-wise != avoids deprecated ~bool when lengths differ
            mask = _feature_mask_excluding(feature_names2, feature_names[0])
            feature_names.extend(list(feature_names2[mask]))
            shap_values_arr = np.hstack([shap_values_arr, shap_values2.values[:, mask]])  # type: ignore[union-attr]
            features = np.hstack([features, shap_values2.data[:, mask]])  # type: ignore[union-attr, call-overload, index, list-item]
            if shap_values2.display_data is None:  # type: ignore[union-attr]
                display_features = np.hstack(
                    [display_features, shap_values2.data[:, mask]]  # type: ignore[union-attr, call-overload, index, list-item]
                )
            else:
                display_features = np.hstack(
                    [display_features, shap_values2.display_data[:, mask]]  # type: ignore[union-attr]
                )
        color = None
        interaction_index = "auto"

    if isinstance(shap_values_arr, list):
        msg = (
            "The passed shap_values are a list not an array! If you have a "
            "list of explanations try passing shap_values[0] instead."
        )
        raise TypeError(msg)

    # ``features`` is always an ndarray after the reshape above; coerce any
    # remaining 1-D vectors to column form for downstream indexing.
    if len(shap_values_arr.shape) == 1:
        shap_values_arr = np.reshape(shap_values_arr, (len(shap_values_arr), 1))
    if len(features.shape) == 1:
        features = np.reshape(features, (len(features), 1))

    # Normalise feature names to a flat list (handles None, scalar, sequence).
    if not feature_names or feature_names[0] is None:
        feature_names = [f"Feature {i}" for i in range(shap_values_arr.shape[1])]
    elif len(feature_names) == 1 and isinstance(feature_names[0], (list, tuple)):
        feature_names = list(feature_names[0])

    jitter_amount_scale: float
    if x_jitter == "auto":
        jitter_amount_scale = _suggest_x_jitter(features[:, ind])
    else:
        jitter_amount_scale = float(x_jitter)

    if interaction_index == "auto":
        interaction_index = int(
            approximate_interactions(ind, shap_values_arr, features, rng=rng)[0]
        )
    interaction_index = convert_name(interaction_index, shap_values_arr, feature_names)
    categorical_interaction = False

    if ax is None:
        figsize = (
            (7.5, 5)
            if interaction_index != ind and interaction_index is not None
            else (6, 5)
        )
        _, ax = plt.subplots(figsize=figsize)

    if shap_values_arr.shape[0] != features.shape[0]:
        msg = "'shap_values' and 'features' must have the same number of rows!"
        raise AssertionError(msg)
    if shap_values_arr.shape[1] != features.shape[1]:
        msg = "'shap_values' must have the same number of columns as 'features'!"
        raise AssertionError(msg)

    oinds = np.arange(shap_values_arr.shape[0])
    rng.shuffle(oinds)
    xv = encode_array_if_needed(features[oinds, ind])
    xd = display_features[oinds, ind]
    s_vals = shap_values_arr[oinds, ind]

    name_map: dict[Any, Any] = {}
    xnames: list[Any] = []
    if isinstance(xd[0], str):
        for i in range(len(xv)):
            name_map[xd[i]] = xv[i]
        xnames = list(name_map.keys())

    name = feature_names[ind]

    color_norm = None
    clow = chigh = 0.0
    cvals: NDArray[Any] | None = None
    cvals_imp: NDArray[Any] | None = None
    cd: NDArray[Any] | None = None
    cnames: list[Any] = []
    cname_map: dict[Any, Any] = {}

    if interaction_index is not None:
        interaction_feature_values = encode_array_if_needed(
            features[:, interaction_index]
        )
        cv = interaction_feature_values
        cd = display_features[:, interaction_index]
        clow = float(np.nanpercentile(cv.astype(float), 5))
        chigh = float(np.nanpercentile(cv.astype(float), 95))
        if clow == chigh:
            clow = float(np.nanmin(cv.astype(float)))
            chigh = float(np.nanmax(cv.astype(float)))
        if isinstance(cd[0], str):
            for i in range(len(cv)):
                cname_map[cd[i]] = cv[i]
            cnames = list(cname_map.keys())
            categorical_interaction = True
        elif clow % 1 == 0 and chigh % 1 == 0 and chigh - clow < 10:
            categorical_interaction = True

        if categorical_interaction and clow != chigh:
            clow = float(np.nanmin(cv.astype(float)))
            chigh = float(np.nanmax(cv.astype(float)))
            n_bounds = min(int(chigh - clow + 2), getattr(cmap, "N", 256) - 1)
            bounds = np.linspace(clow, chigh, n_bounds)
            color_norm = matplotlib.colors.BoundaryNorm(
                bounds, getattr(cmap, "N", 256) - 1
            )

    xv_no_jitter = xv.copy()
    if jitter_amount_scale > 0:
        if jitter_amount_scale > 1:
            jitter_amount_scale = 1.0
        # ``encode_array_if_needed`` always yields a numeric float array.
        xvals = xv.astype(float)
        xvals = xvals[~np.isnan(xvals)]
        xvals = np.unique(xvals)
        if len(xvals) >= 2:
            smallest_diff = float(np.min(np.diff(xvals)))
            jitter_amount = jitter_amount_scale * smallest_diff
            xv = xv + (rng.random(size=len(xv)) * jitter_amount - jitter_amount / 2)

    xv_nan = np.isnan(xv)
    xv_notnan = ~xv_nan

    if interaction_index is not None:
        cvals = encode_array_if_needed(features[oinds, interaction_index]).astype(
            np.float64
        )
        cvals_imp = cvals.copy()
        cvals_imp[np.isnan(cvals)] = (clow + chigh) / 2.0
        cvals[cvals_imp > chigh] = chigh
        cvals[cvals_imp < clow] = clow
        if color_norm is None:
            vmin: float | None = clow
            vmax: float | None = chigh
        else:
            vmin = vmax = None
        ax.axhline(0, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)
        scatter_pts = ax.scatter(
            xv[xv_notnan],
            s_vals[xv_notnan],
            s=dot_size,
            linewidth=0,
            c=cvals[xv_notnan],
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            norm=color_norm,
            rasterized=len(xv) > 500,
        )
        scatter_pts.set_array(cvals[xv_notnan])
    else:
        scatter_pts = ax.scatter(
            xv,
            s_vals,
            s=dot_size,
            linewidth=0,
            color=color,
            alpha=alpha,
            rasterized=len(xv) > 500,
        )

    if interaction_index != ind and interaction_index is not None:
        assert cd is not None
        assert cvals is not None
        if isinstance(cd[0], str):
            tick_positions = np.array([cname_map[n] for n in cnames])
            tick_positions *= 1 - 1 / len(cnames)
            tick_positions += 0.5 * (chigh - clow) / (chigh - clow + 1)
            colorbar = plt.colorbar(scatter_pts, ticks=tick_positions, ax=ax, aspect=80)
            colorbar.set_ticklabels(cnames)
        else:
            colorbar = plt.colorbar(scatter_pts, ax=ax, aspect=80)

        # ``convert_name`` resolves to an integer column index.
        colorbar.set_label(feature_names[int(interaction_index)], size=13)
        colorbar.ax.tick_params(labelsize=11)
        if categorical_interaction:
            colorbar.ax.tick_params(length=0)
        colorbar.set_alpha(1)
        colorbar.outline.set_visible(False)

    xmin_r = parse_axis_limit(xmin, xv, is_shap_axis=False)
    xmax_r = parse_axis_limit(xmax, xv, is_shap_axis=False)
    ymin_r = parse_axis_limit(ymin, s_vals, is_shap_axis=True)
    ymax_r = parse_axis_limit(ymax, s_vals, is_shap_axis=True)
    if xmin_r is not None or xmax_r is not None:
        ax.set_xlim(*_suggest_buffered_limits(xmin_r, xmax_r, xv))
    if ymin_r is not None or ymax_r is not None:
        ax.set_ylim(*_suggest_buffered_limits(ymin_r, ymax_r, s_vals))

    xlim = ax.get_xlim()
    if interaction_index is not None and cvals is not None and cvals_imp is not None:
        nan_pts = ax.scatter(
            xlim[0] * np.ones(int(xv_nan.sum())),
            s_vals[xv_nan],
            marker=MarkerStyle(1),
            linewidth=2,
            c=cvals_imp[xv_nan],
            cmap=cmap,
            alpha=alpha,
            vmin=clow,
            vmax=chigh,
        )
        nan_pts.set_array(cvals[xv_nan])
    else:
        ax.scatter(
            xlim[0] * np.ones(int(xv_nan.sum())),
            s_vals[xv_nan],
            marker=MarkerStyle(1),
            linewidth=2,
            color=color,
            alpha=alpha,
        )
    ax.set_xlim(xlim)

    if hist:
        _plot_histogram(ax, xv, xv_no_jitter)

    plt.sca(ax)
    ax.set_xlabel(name, color=axis_color, fontsize=13)
    ax.set_ylabel(f"XWhy value for\n{name}", color=axis_color, fontsize=13)
    if title is not None:
        ax.set_title(title, color=axis_color, fontsize=13)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor(axis_color)
    if isinstance(xd[0], str):
        ax.set_xticks([name_map[n] for n in xnames])
        ax.set_xticklabels(xnames, fontdict={"rotation": "vertical", "fontsize": 11})
    if show:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            plt.show()
        return None
    return ax


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
    seed: int = 42,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    r"""Plot a dependence scatter of feature values vs attributions.

    Parameters
    ----------
    explanation :
        Explanation or XWhy result (optionally multi-feature).
    ind :
        Feature index or name to plot when *explanation* has multiple columns.
    color :
        Fixed colour or interaction feature specifier.
    show :
        Whether to display the figure.
    save_path :
        Optional path to write the figure to.
    backend :
        Currently only ``\"matplotlib\"`` is supported.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches.
    seed :
        Seed for the random generator used for jitter and row shuffle.
    **kwargs :
        Extra options forwarded to :func:`_shap_scatter`.

    Returns
    -------
    Figure | None
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    allowed = frozenset({"matplotlib"})
    if _check_backend is not None:
        _check_backend(backend, allowed)
    elif backend not in allowed:
        msg = f"Unsupported backend: {backend}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    exp = _as_explanation(explanation) if _as_explanation is not None else explanation

    if ind is not None and len(exp.shape) > 1 and exp.shape[1] > 1:
        if isinstance(ind, int):
            exp = exp[:, ind]
        else:
            names = exp.feature_names
            if names is not None and ind in names:
                idx = list(names).index(ind)
                exp = exp[:, idx]

    if figsize is not None:
        _fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = None

    _shap_scatter(
        shap_values=exp,
        color=color if color is not None else "#1E88E5",
        hist=bool(kwargs.get("hist", True)),
        axis_color=kwargs.get("axis_color", "#333333"),
        cmap=kwargs.get("cmap"),
        dot_size=float(kwargs.get("dot_size", 16)),
        x_jitter=kwargs.get("x_jitter", "auto"),
        alpha=float(kwargs.get("alpha", 1.0)),
        title=title,
        xmin=kwargs.get("xmin"),
        xmax=kwargs.get("xmax"),
        ymin=kwargs.get("ymin"),
        ymax=kwargs.get("ymax"),
        overlay=kwargs.get("overlay"),
        ax=ax,
        ylabel=kwargs.get("ylabel", "XWhy value"),
        show=False,
        rng=rng,
    )

    if title:
        plt.gcf().suptitle(title, fontsize=13, fontweight="bold")

    fig = plt.gcf()
    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig
