"""Violin and layered-violin summary plots of attributions."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from packaging import version
from scipy.stats import gaussian_kde

from .base import (
    RED_BLUE,
    DimensionError,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
)


def _violin_orientation_kwargs() -> dict[str, str | bool]:
    """Return ``violinplot`` orientation kwargs for the installed matplotlib.

    Matplotlib 3.10 renamed the ``vert`` flag to ``orientation``.  This helper
    isolates the version branch so callers stay free of version checks
    (single responsibility) and the branch is unit-testable without reloading
    the module.

    Returns
    -------
    dict[str, str | bool]
        ``{"orientation": "horizontal"}`` on matplotlib >= 3.10, otherwise
        ``{"vert": False}``.

    """
    # TODO: drop the <3.10 branch when minimum matplotlib is 3.10+
    if version.parse(matplotlib.__version__) >= version.parse("3.10"):
        return {"orientation": "horizontal"}
    return {"vert": False}


ORIENTATION_KWARG: dict[str, str | bool] = _violin_orientation_kwargs()


def _trim_crange(
    values: NDArray[Any],
    nan_mask: NDArray[Any],
) -> tuple[float, float, NDArray[Any]]:
    """Trim the color range, but prevent the color range from collapsing."""
    vmin = float(np.nanpercentile(values, 5))
    vmax = float(np.nanpercentile(values, 95))
    if vmin == vmax:
        vmin = float(np.nanpercentile(values, 1))
        vmax = float(np.nanpercentile(values, 99))
        if vmin == vmax:
            vmin = float(np.min(values))
            vmax = float(np.max(values))

    if vmin > vmax:
        vmin = vmax

    cvals = values[~nan_mask].astype(np.float64)
    cvals_imp = cvals.copy()
    cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
    cvals[cvals_imp > vmax] = vmax
    cvals[cvals_imp < vmin] = vmin

    return vmin, vmax, cvals


def _shap_violin(
    shap_values: Any,  # noqa: ANN401
    features: Any = None,  # noqa: ANN401
    feature_names: Any = None,  # noqa: ANN401
    max_display: int | None = None,
    plot_type: str = "violin",
    color: Any = None,  # noqa: ANN401
    axis_color: str = "#333333",
    title: str | None = None,
    alpha: float = 1.0,
    show: bool = True,
    sort: bool = True,
    color_bar: bool = True,
    plot_size: Literal["auto"] | float | tuple[float, float] | None = "auto",
    layered_violin_max_num_bins: int = 20,
    class_names: Any = None,  # noqa: ANN401
    class_inds: Any = None,  # noqa: ANN401
    color_bar_label: str = "Feature value",
    cmap: Any = None,  # noqa: ANN401
    color_bar_label_size: int = 12,
    color_bar_tick_size: int = 11,
    axhline_lw: float = 0.5,
    use_log_scale: bool = False,
    *,
    rng: np.random.Generator | None = None,
) -> None:
    """Create a SHAP violin plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation or numpy.ndarray
        For single output explanations, this is a matrix of SHAP values
        (# samples x # features).
    features : numpy.ndarray or pandas.DataFrame or list, optional
        Matrix of feature values (# samples x # features), or a
        ``feature_names`` list as shorthand.
    feature_names : list, optional
        Names of the features (length: # features).
    max_display : int, optional
        How many top features to include in the plot (default is 20).
    plot_type : {"violin", "layered_violin"}, optional
        What type of summary plot to produce.
    color : str or None, optional
        Color or colormap to use for the plot.
    axis_color : str, optional
        Color for the plot axes.
    title : str or None, optional
        Plot title (currently unused).
    alpha : float, optional
        Opacity of the plot elements.
    show : bool, optional
        Whether matplotlib show is called before returning.
    sort : bool, optional
        Whether to sort features by the sum of their effect magnitudes.
    color_bar : bool, optional
        Whether to draw the color bar (legend).
    plot_size : {"auto", float, (float, float), None}, optional
        Plot size control.
    layered_violin_max_num_bins : int, optional
        Maximum number of bins for layered violin plots.
    class_names :
        Reserved for multi-class layouts; currently unused.
    class_inds :
        Reserved for multi-class layouts; currently unused.
    color_bar_label : str, optional
        Label for the color bar.
    cmap : str or Colormap, optional
        Colormap to use for coloring points by feature value.
    color_bar_label_size : int, optional
        Font size for the color bar label.
    color_bar_tick_size : int, optional
        Font size for the color bar ticks.
    axhline_lw : float, optional
        Line width for horizontal lines in the plot.
    use_log_scale : bool, optional
        Whether to use a symmetric log scale for the x-axis.
    rng : np.random.Generator, optional
        Random generator for KDE jitter.

    """
    _ = show, class_names, class_inds
    if cmap is None:
        cmap = RED_BLUE
    if rng is None:
        rng = np.random.default_rng()

    if title is not None:
        warnings.warn(
            "The `title` argument is unused and will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names

    if isinstance(shap_values, list):
        msg = (
            "Violin plots don't support multi-output explanations! "
            "Use 'xwhy.plots.bar' instead."
        )
        raise TypeError(msg)

    if plot_type is None:
        plot_type = "violin"
    if plot_type not in {"violin", "layered_violin"}:
        msg = (
            f"plot_type: Expected one of ('violin','layered_violin'), "
            f"received {plot_type} instead."
        )
        raise ValueError(msg)

    if len(shap_values.shape) == 1:
        msg = "Violin summary plots need a matrix of shap_values, not a vector."
        raise AssertionError(msg)

    # default color:
    if color is None:
        color = "coolwarm" if plot_type == "layered_violin" else "#1E88E5"

    # convert from a DataFrame or other types
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = shap_values.shape[1]

    if features is not None:
        shape_msg = (
            "The shape of the shap_values matrix does not match the shape "
            "of the provided data matrix."
        )
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(num_features)])

    if use_log_scale:
        plt.xscale("symlog")

    if max_display is None:
        max_display = 20

    if sort:
        # order features by the sum of their effect magnitudes
        feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    row_height = 0.4
    if plot_size == "auto":
        plt.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        plt.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        plt.gcf().set_size_inches(8, len(feature_order) * float(plot_size) + 1.5)
    plt.axvline(x=0, color="#999999", zorder=-1)

    if plot_type == "violin":
        for pos in range(len(feature_order)):
            plt.axhline(y=pos, color="#cccccc", lw=axhline_lw, dashes=(1, 5), zorder=-1)

        if features is not None:
            global_low = np.nanpercentile(
                shap_values[:, : len(feature_names)].flatten(), 1
            )
            global_high = np.nanpercentile(
                shap_values[:, : len(feature_names)].flatten(), 99
            )
            for pos, i in enumerate(feature_order):
                shaps = shap_values[:, i]
                shap_min, shap_max = np.min(shaps), np.max(shaps)
                span = shap_max - shap_min
                xs = np.linspace(
                    np.min(shaps) - span * 0.2,
                    np.max(shaps) + span * 0.2,
                    100,
                )
                if np.std(shaps) < (global_high - global_low) / 100:
                    noise = rng.standard_normal(len(shaps)) * (
                        (global_high - global_low) / 100
                    )
                    dens = gaussian_kde(shaps + noise)(xs)
                else:
                    dens = gaussian_kde(shaps)(xs)
                dens /= np.max(dens) * 3

                values = features[:, i]
                smooth_values = np.zeros(len(xs) - 1)
                sort_inds = np.argsort(shaps)
                trailing_pos = 0
                leading_pos = 0
                running_sum = 0.0
                back_fill = 0
                for j in range(len(xs) - 1):
                    while (
                        leading_pos < len(shaps)
                        and xs[j] >= shaps[sort_inds[leading_pos]]
                    ):
                        running_sum += values[sort_inds[leading_pos]]
                        leading_pos += 1
                        if leading_pos - trailing_pos > 20:
                            running_sum -= values[sort_inds[trailing_pos]]
                            trailing_pos += 1
                    if leading_pos - trailing_pos > 0:
                        smooth_values[j] = running_sum / (leading_pos - trailing_pos)
                        for k in range(back_fill):
                            smooth_values[j - k - 1] = smooth_values[j]
                    else:
                        back_fill += 1

                nan_mask = np.isnan(values)
                vmin, vmax, cvals = _trim_crange(values, nan_mask)

                plt.scatter(
                    shaps[nan_mask],
                    np.ones(shap_values[nan_mask].shape[0]) * pos,
                    color="#777777",
                    s=9,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )
                plt.scatter(
                    shaps[~nan_mask],
                    np.ones(shap_values[~nan_mask].shape[0]) * pos,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    s=9,
                    c=cvals,
                    alpha=alpha,
                    linewidth=0,
                    zorder=1,
                )

                smooth_values -= vmin
                if vmax - vmin > 0:
                    smooth_values /= vmax - vmin
                for j in range(len(xs) - 1):
                    if dens[j] > 0.05 or dens[j + 1] > 0.05:
                        plt.fill_between(
                            [xs[j], xs[j + 1]],
                            [pos + dens[j], pos + dens[j + 1]],
                            [pos - dens[j], pos - dens[j + 1]],
                            color=RED_BLUE(smooth_values[j]),
                            zorder=2,
                        )

        else:
            parts = plt.violinplot(
                shap_values[:, feature_order],
                range(len(feature_order)),
                points=200,
                **ORIENTATION_KWARG,  # type: ignore[arg-type]
                widths=0.7,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )

            for pc in parts["bodies"]:  # type: ignore[attr-defined]
                pc.set_facecolor(color)
                pc.set_edgecolor("none")
                pc.set_alpha(alpha)

    else:
        # plot_type is "layered_violin" — the only remaining valid value
        # after the earlier membership check (elif-false is unreachable).
        # courtesy of @kodonnell
        num_x_points = 200
        bins = (
            np.linspace(0, features.shape[0], layered_violin_max_num_bins + 1)
            .round(0)
            .astype("int")
        )
        shap_min, shap_max = np.min(shap_values), np.max(shap_values)
        x_points = np.linspace(shap_min, shap_max, num_x_points)

        for pos, ind in enumerate(feature_order):
            feature = features[:, ind]
            unique, counts = np.unique(feature, return_counts=True)
            if unique.shape[0] <= layered_violin_max_num_bins:
                order = np.argsort(unique)
                thesebins = np.cumsum(counts[order])
                thesebins = np.insert(thesebins, 0, 0)
            else:
                thesebins = bins
            nbins = thesebins.shape[0] - 1
            order = np.argsort(feature)
            ys = np.zeros((nbins, num_x_points))
            for i in range(nbins):
                shaps = shap_values[order[thesebins[i] : thesebins[i + 1]], ind]
                if shaps.shape[0] == 1:
                    warnings.warn(
                        f"Not enough data in bin #{i} for feature "
                        f"{feature_names[ind]}, so it'll be ignored. "
                        "Try increasing the number of records to plot.",
                        stacklevel=2,
                    )
                    if i > 0:
                        ys[i, :] = ys[i - 1, :]
                    continue
                noise = rng.normal(loc=0, scale=0.001, size=shaps.shape[0])
                ys[i, :] = gaussian_kde(shaps + noise)(x_points)
                size = thesebins[i + 1] - thesebins[i]
                bin_size_if_even = features.shape[0] / nbins
                relative_bin_size = size / bin_size_if_even
                ys[i, :] *= relative_bin_size
            ys = np.cumsum(ys, axis=0)
            width = 0.8
            # Guard against all-zero density (e.g. every bin ignored)
            scale = max(float(ys.max()) * 2 / width, 1e-12)
            # Avoid ZeroDivisionError when a feature collapses to a single bin
            denom = max(nbins - 1, 1)
            for i in range(nbins - 1, -1, -1):
                y = ys[i, :] / scale
                c = plt.get_cmap(color)(i / denom) if color in plt.colormaps else color
                plt.fill_between(
                    x_points, pos - y, pos + y, facecolor=c, edgecolor="face"
                )
        plt.xlim(shap_min, shap_max)

    # draw the color bar
    if (
        color_bar
        and features is not None
        and (plot_type != "layered_violin" or color in plt.colormaps)
    ):
        mappable = cm.ScalarMappable(
            cmap=cmap if plot_type != "layered_violin" else plt.get_cmap(color)
        )
        mappable.set_array([0, 1])
        colorbar = plt.colorbar(mappable, ax=plt.gca(), ticks=[0, 1], aspect=80)
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label(color_bar_label, size=color_bar_label_size, labelpad=0)
        colorbar.ax.tick_params(labelsize=color_bar_tick_size, length=0)
        colorbar.set_alpha(1)
        colorbar.outline.set_visible(False)

    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().tick_params(color=axis_color, labelcolor=axis_color)
    plt.yticks(
        range(len(feature_order)),
        [feature_names[i] for i in feature_order],
        fontsize=13,
    )
    plt.gca().tick_params("y", length=20, width=0.5, which="major")
    plt.gca().tick_params("x", labelsize=11)
    plt.ylim(-1, len(feature_order))
    plt.xlabel("XWhy values (impact on model output)", fontsize=13)


def violin(
    explanation: Any,  # noqa: ANN401
    *,
    max_display: int | None = 10,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    seed: int = 42,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Plot a violin summary of feature attributions."""
    allowed = frozenset({"matplotlib"})
    if _check_backend is not None:
        _check_backend(backend, allowed)
    elif backend not in allowed:
        msg = f"Unsupported backend: {backend}"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    exp = _as_explanation(explanation) if _as_explanation is not None else explanation

    if figsize is not None:
        plt.subplots(figsize=figsize)

    _shap_violin(
        shap_values=exp,
        max_display=max_display if max_display is not None else 10,
        show=False,
        features=kwargs.get("features"),
        feature_names=kwargs.get("feature_names"),
        plot_type=kwargs.get("plot_type", "violin"),
        color=kwargs.get("color"),
        axis_color=kwargs.get("axis_color", "#333333"),
        alpha=float(kwargs.get("alpha", 1.0)),
        sort=bool(kwargs.get("sort", True)),
        color_bar=bool(kwargs.get("color_bar", True)),
        plot_size=kwargs.get("plot_size", "auto"),
        layered_violin_max_num_bins=int(kwargs.get("layered_violin_max_num_bins", 20)),
        color_bar_label=kwargs.get("color_bar_label", "Feature value"),
        cmap=kwargs.get("cmap"),
        use_log_scale=bool(kwargs.get("use_log_scale", False)),
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
