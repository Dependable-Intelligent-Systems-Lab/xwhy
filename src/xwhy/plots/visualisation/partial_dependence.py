"""Partial dependence and ICE curves for model responses."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .base import (
    BLUE_RGB,
    LIGHT_BLUE_RGB,
    RED_RGB,
    RED_TRANSPARENT_BLUE,
    Explanation,
    _check_backend,
    _finish_matplotlib,
    convert_name,
)


def compute_bounds(
    xmin: float | str | None,
    xmax: float | str | None,
    xv: NDArray[Any],
) -> tuple[float | None, float | None]:
    r"""Resolve xmin/xmax from None, float, or ``\"percentile(x)\"`` strings.

    Args:
        xmin: Lower bound specification.
        xmax: Upper bound specification.
        xv: Feature values used for percentile / auto padding.

    Returns:
        Pair of resolved numeric bounds (may still be None).

    """
    if xmin is not None or xmax is not None:
        if isinstance(xmin, str) and xmin.startswith("percentile"):
            xmin = float(np.nanpercentile(xv, float(xmin[11:-1])))
        if isinstance(xmax, str) and xmax.startswith("percentile"):
            xmax = float(np.nanpercentile(xv, float(xmax[11:-1])))

        if xmin is None or xmin == np.nanmin(xv):
            xmin = float(np.nanmin(xv) - (xmax - np.nanmin(xv)) / 20)
        if xmax is None or xmax == np.nanmax(xv):
            xmax = float(np.nanmax(xv) + (np.nanmax(xv) - xmin) / 20)

    return xmin, xmax  # type: ignore[return-value]


def _shap_partial_dependence(
    ind: int | str | tuple[Any, ...],
    model: Callable[..., Any],
    data: Any,  # noqa: ANN401
    xmin: Any = "percentile(0)",  # noqa: ANN401
    xmax: Any = "percentile(100)",  # noqa: ANN401
    npoints: int | None = None,
    feature_names: Sequence[str] | None = None,
    hist: bool = True,
    model_expected_value: bool | float = False,
    feature_expected_value: bool = False,
    shap_values: Any = None,  # noqa: ANN401
    ylabel: str | None = None,
    ice: bool = True,
    ace_opacity: float = 1.0,
    pd_opacity: float = 1.0,
    pd_linewidth: float = 2.0,
    ace_linewidth: float | str = "auto",
    ax: Any = None,  # noqa: ANN401
    show: bool = True,
) -> tuple[Figure, Any] | None:
    r"""Draw a basic partial dependence plot (1-D or 2-D).

    Args:
        ind: Feature index/name, or a pair of indices for a 2-D surface.
        model: Callable mapping a feature matrix to predictions.
        data: Feature matrix, DataFrame, or Explanation.
        xmin: Lower x bound (float or ``\"percentile(x)\"``).
        xmax: Upper x bound (float or ``\"percentile(x)\"``).
        npoints: Number of grid points (default 100 for 1-D, 20 for 2-D).
        feature_names: Optional feature names.
        hist: Draw a feature-value histogram on a twin axis.
        model_expected_value: Show E[f(x)] line, or a fixed float.
        feature_expected_value: Show E[feature] vertical line.
        shap_values: Optional Explanation for stem markers.
        ylabel: Y-axis label override.
        ice: Draw individual conditional expectation lines.
        ace_opacity: Opacity of ICE lines.
        pd_opacity: Opacity of the mean PD curve.
        pd_linewidth: Width of the mean PD curve.
        ace_linewidth: Width of ICE lines, or ``\"auto\"``.
        ax: Optional matplotlib Axes.
        show: Call ``plt.show()`` when True.

    Returns:
        ``(fig, ax)`` when *show* is False, otherwise ``None``.

    """
    if (Explanation is not None and isinstance(data, Explanation)) or (
        type(data).__name__ == "Explanation"
    ):
        features = data.data
        shap_values = data
    else:
        features = data

    use_dataframe = False
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = list(features.columns)
        features = features.values
        use_dataframe = True

    features = np.asarray(features)
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[1])]

    # ----- 1-D partial dependence -----
    if not isinstance(ind, tuple):
        col = convert_name(ind, None, feature_names)  # type: ignore[arg-type]
        xv = features[:, col]  # type: ignore[index]
        xmin_r, xmax_r = compute_bounds(xmin, xmax, xv)
        npoints_1d = 100 if npoints is None else npoints
        xs = np.linspace(float(xmin_r), float(xmax_r), npoints_1d)  # type: ignore[arg-type]

        ice_vals: NDArray[Any] | None = None
        if ice:
            features_tmp = features.copy()
            ice_vals = np.zeros((npoints_1d, features.shape[0]))
            for i in range(npoints_1d):
                features_tmp[:, col] = xs[i]  # type: ignore[index]
                if use_dataframe:
                    ice_vals[i, :] = model(
                        pd.DataFrame(features_tmp, columns=feature_names)
                    )
                else:
                    ice_vals[i, :] = model(features_tmp)

        features_tmp = features.copy()
        vals = np.zeros(npoints_1d)
        for i in range(npoints_1d):
            features_tmp[:, col] = xs[i]  # type: ignore[index]
            if use_dataframe:
                vals[i] = model(
                    pd.DataFrame(features_tmp, columns=feature_names)
                ).mean()
            else:
                vals[i] = model(features_tmp).mean()

        if ax is None:
            fig = plt.figure()
            ax1 = plt.gca()
        else:
            fig = plt.gcf()
            ax1 = plt.gca()

        ax2 = cast("Any", ax1.twinx())

        if hist:
            ax2.hist(
                xv,
                50,
                density=False,
                facecolor="black",
                alpha=0.1,
                range=(xmin_r, xmax_r),
            )

        if ice and ice_vals is not None:
            line_w: float
            if ace_linewidth == "auto":
                line_w = min(1.0, 50 / ice_vals.shape[1])
            else:
                line_w = float(ace_linewidth)
            ax1.plot(
                xs,
                ice_vals,
                color=LIGHT_BLUE_RGB,
                linewidth=line_w,
                alpha=ace_opacity,
            )

        ax1.plot(
            xs,
            vals,
            color=BLUE_RGB,
            linewidth=pd_linewidth,
            alpha=pd_opacity,
        )

        ax2.set_ylim(0, features.shape[0])
        ax1.set_xlabel(str(feature_names[col]), fontsize=13)  # type: ignore[index]
        if ylabel is None:
            if not ice:
                ylabel = f"E[f(x) | {feature_names[col]}]"  # type: ignore[index]
            else:
                ylabel = f"f(x) | {feature_names[col]}"  # type: ignore[index]
        ax1.set_ylabel(ylabel, fontsize=13)
        ax1.xaxis.set_ticks_position("bottom")
        ax1.yaxis.set_ticks_position("left")
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.tick_params(labelsize=11)

        ax2.xaxis.set_ticks_position("bottom")
        ax2.yaxis.set_ticks_position("left")
        ax2.yaxis.set_ticks([])
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)

        if feature_expected_value is not False:
            ax3 = ax2.twiny()
            ax3.set_xlim(xmin_r, xmax_r)
            mval = float(xv.mean())
            ax3.set_xticks([mval])
            ax3.set_xticklabels([f"E[{feature_names[col]}]"])  # type: ignore[index]
            ax3.spines["right"].set_visible(False)
            ax3.spines["top"].set_visible(False)
            ax3.tick_params(length=0, labelsize=11)
            ax1.axvline(
                mval,
                color="#999999",
                zorder=-1,
                linestyle="--",
                linewidth=1,
            )

        if model_expected_value is not False or shap_values is not None:
            mev: float
            if model_expected_value is True:
                if use_dataframe:
                    mev = float(
                        model(pd.DataFrame(features, columns=feature_names)).mean()
                    )
                else:
                    mev = float(model(features).mean())
            elif isinstance(model_expected_value, (int, float)):
                mev = float(model_expected_value)
            else:
                mev = float(shap_values.base_values)
            ymin, ymax = ax1.get_ylim()
            ax4 = ax2.twinx()
            ax4.set_ylim(ymin, ymax)
            ax4.set_yticks([mev])
            ax4.set_yticklabels(["E[f(x)]"])
            ax4.spines["right"].set_visible(False)
            ax4.spines["top"].set_visible(False)
            ax4.tick_params(length=0, labelsize=11)
            ax1.axhline(
                mev,
                color="#999999",
                zorder=-1,
                linestyle="--",
                linewidth=1,
            )

        if shap_values is not None:
            markerline, stemlines, _ = ax1.stem(
                shap_values.data[:, col],
                shap_values.base_values + shap_values.values[:, col],
                bottom=shap_values.base_values,
                markerfmt="o",
                basefmt=" ",
            )
            stemlines.set_edgecolors([RED_RGB if v > 0 else BLUE_RGB for v in vals])
            plt.setp(stemlines, "zorder", -1)
            plt.setp(stemlines, "linewidth", 2)
            plt.setp(markerline, "color", "black")
            plt.setp(markerline, "markersize", 4)

        if show:
            plt.show()
            return None
        return fig, ax1

    # ----- 2-D partial dependence -----
    ind0 = convert_name(ind[0], None, feature_names)  # type: ignore[arg-type]
    ind1 = convert_name(ind[1], None, feature_names)  # type: ignore[arg-type]
    xv0 = features[:, ind0]  # type: ignore[index]
    xv1 = features[:, ind1]  # type: ignore[index]

    xmin0 = xmin[0] if isinstance(xmin, tuple) else xmin
    xmin1 = xmin[1] if isinstance(xmin, tuple) else xmin
    xmax0 = xmax[0] if isinstance(xmax, tuple) else xmax
    xmax1 = xmax[1] if isinstance(xmax, tuple) else xmax

    xmin0, xmax0 = compute_bounds(xmin0, xmax0, xv0)
    xmin1, xmax1 = compute_bounds(xmin1, xmax1, xv1)
    npoints_2d = 20 if npoints is None else npoints
    xs0 = np.linspace(float(xmin0), float(xmax0), npoints_2d)  # type: ignore[arg-type]
    xs1 = np.linspace(float(xmin1), float(xmax1), npoints_2d)  # type: ignore[arg-type]

    features_tmp = features.copy()
    x0 = np.zeros((npoints_2d, npoints_2d))
    x1 = np.zeros((npoints_2d, npoints_2d))
    vals_2d = np.zeros((npoints_2d, npoints_2d))
    for i in range(npoints_2d):
        for j in range(npoints_2d):
            features_tmp[:, ind0] = xs0[i]  # type: ignore[index]
            features_tmp[:, ind1] = xs1[j]  # type: ignore[index]
            x0[i, j] = xs0[i]
            x1[i, j] = xs1[j]
            vals_2d[i, j] = model(features_tmp).mean()

    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection="3d")
    ax3d.plot_surface(x0, x1, vals_2d, cmap=RED_TRANSPARENT_BLUE)
    ax3d.set_xlabel(str(feature_names[ind0]), fontsize=13)  # type: ignore[index]
    ax3d.set_ylabel(str(feature_names[ind1]), fontsize=13)  # type: ignore[index]
    ax3d.set_zlabel(
        f"E[f(x) | {feature_names[ind0]}, {feature_names[ind1]}]",  # type: ignore[index]
        fontsize=13,
    )

    if show:
        plt.show()
        return None
    return fig, ax3d


def partial_dependence(
    ind: int | str | tuple[Any, ...],
    model: Callable[[NDArray[Any]], NDArray[Any]],
    data: NDArray[Any],
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
) -> Figure | None:
    r"""Plot partial dependence (and optional ICE) for one or two features.

    Args:
        ind: Feature index/name, or a pair for a 2-D surface.
        model: Callable mapping a feature matrix to predictions.
        data: Feature matrix used for the grid and ICE lines.
        feature_names: Optional feature names.
        npoints: Number of grid points along each axis.
        ice: Draw individual conditional expectation lines (1-D only).
        max_ice_lines: Reserved for future ICE subsampling.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: Currently only ``\"matplotlib\"`` is supported.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Extra options forwarded to
            :func:`_shap_partial_dependence`.

    Returns:
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    _ = max_ice_lines
    allowed = frozenset({"matplotlib"})
    if _check_backend is not None:
        _check_backend(backend, allowed)
    elif backend not in allowed:
        msg = f"Unsupported backend: {backend}"
        raise ValueError(msg)

    if figsize is not None:
        plt.subplots(figsize=figsize)

    _shap_partial_dependence(
        ind=ind,
        model=model,
        data=data,
        feature_names=feature_names,
        npoints=npoints,
        ice=ice,
        hist=bool(kwargs.get("hist", True)),
        model_expected_value=kwargs.get("model_expected_value", False),
        feature_expected_value=bool(kwargs.get("feature_expected_value", False)),
        shap_values=kwargs.get("shap_values"),
        ylabel=kwargs.get("ylabel"),
        ace_opacity=float(kwargs.get("ace_opacity", 1.0)),
        pd_opacity=float(kwargs.get("pd_opacity", 1.0)),
        pd_linewidth=float(kwargs.get("pd_linewidth", 2.0)),
        ace_linewidth=kwargs.get("ace_linewidth", "auto"),
        xmin=kwargs.get("xmin", "percentile(0)"),
        xmax=kwargs.get("xmax", "percentile(100)"),
        ax=None,
        show=False,
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
