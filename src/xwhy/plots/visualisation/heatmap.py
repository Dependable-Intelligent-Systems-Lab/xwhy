"""Heatmaps of attributions across instances and features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy
import scipy.spatial
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage

from .base import (
    RED_WHITE_BLUE,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
)


def _order_instances(values: NDArray[Any], strategy: str) -> NDArray[Any]:
    r"""Order instances for the heatmap according to *strategy*.

    Parameters
    ----------
    values :
        Attribution matrix of shape ``(n_instances, n_features)``.
    strategy :
        ``\"hclust\"``, ``\"output\"``, or ``\"none\"``.

    Returns
    -------
    NDArray
        Instance indices in display order.

    """
    n_instances = values.shape[0]

    if strategy == "output":
        return np.argsort(values.sum(axis=1))

    # Hierarchical clustering is O(n^2) in memory; fall back when large.
    if strategy == "hclust" and 2 < n_instances <= 2000:
        try:
            return np.asarray(leaves_list(linkage(values, method="average")))
        except (ImportError, ValueError):  # pragma: no cover
            return np.argsort(values.sum(axis=1))

    if strategy == "hclust":
        return np.argsort(values.sum(axis=1))

    return np.arange(n_instances)


def _resolve_instance_order(
    values: NDArray[Any],
    instance_order: str | NDArray[Any],
) -> NDArray[Any]:
    """Turn an instance-order specification into an index array."""
    if isinstance(instance_order, np.ndarray):
        return instance_order
    if instance_order == "hclust":
        n_instances = values.shape[0]
        if 2 < n_instances <= 2000:
            try:
                dist = scipy.spatial.distance.pdist(values, "sqeuclidean")
                cluster_matrix = scipy.cluster.hierarchy.complete(dist)
                return scipy.cluster.hierarchy.leaves_list(  # type: ignore[no-any-return]
                    scipy.cluster.hierarchy.optimal_leaf_ordering(cluster_matrix, dist)
                )
            except (ImportError, ValueError):  # pragma: no cover
                return np.argsort(values.sum(axis=1))
        return np.argsort(values.sum(axis=1))
    if isinstance(instance_order, str):
        return _order_instances(values, instance_order)
    return np.asarray(instance_order)


def _collapse_excess_features(
    values: NDArray[Any],
    feature_values: NDArray[Any],
    feature_names: NDArray[Any],
    max_display: int,
) -> tuple[NDArray[Any], NDArray[Any], list[str]]:
    """Group features beyond *max_display* into a single summed column."""
    if values.shape[1] <= max_display:
        return values, feature_values, [str(n) for n in feature_names]

    new_values = np.zeros((values.shape[0], max_display))
    new_values[:, :-1] = values[:, : max_display - 1]
    new_values[:, -1] = values[:, max_display - 1 :].sum(1)

    new_feature_values = np.zeros(max_display)
    new_feature_values[:-1] = feature_values[: max_display - 1]
    new_feature_values[-1] = feature_values[max_display - 1 :].sum()

    names = [
        *list(feature_names[: max_display - 1]),
        f"Sum of {values.shape[1] - max_display + 1} other features",
    ]
    return new_values, new_feature_values, [str(n) for n in names]


def _shap_heatmap(
    shap_values: Any,  # noqa: ANN401
    instance_order: str | NDArray[Any] = "hclust",
    feature_values: NDArray[Any] | None = None,
    feature_order: NDArray[Any] | None = None,
    max_display: int = 10,
    cmap: Any = None,  # noqa: ANN401
    show: bool = True,
    plot_width: float = 8,
    ax: Any = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    r"""Create a heatmap of SHAP values with supervised instance clustering.

    Parameters
    ----------
    shap_values :
        Multi-row Explanation (or compatible) to visualise.
    instance_order :
        ``\"hclust\"``, ``\"output\"``, ``\"none\"``, or an index array.
    feature_values :
        Global summary per feature; defaults to mean |SHAP|.
    feature_order :
        Feature index order; defaults to sorting by *feature_values*.
    max_display :
        Maximum number of feature rows (excess are summed).
    cmap :
        Matplotlib colormap (default ``RED_WHITE_BLUE``).
    show :
        Accepted for API compatibility.
    plot_width :
        Figure width in inches when *ax* is not supplied.
    ax :
        Optional axes to draw onto.

    Returns
    -------
    Axes
        The axes the plot was drawn onto.

    """
    _ = show
    if cmap is None:
        cmap = RED_WHITE_BLUE

    values = np.asarray(shap_values.values, dtype=float)
    if feature_values is None:
        feature_values = np.abs(values).mean(0)
    else:
        feature_values = np.asarray(feature_values, dtype=float)

    if feature_order is None:
        feature_order = np.argsort(-feature_values)
    else:
        feature_order = np.asarray(feature_order)

    instance_order = _resolve_instance_order(values, instance_order)

    names_src = getattr(shap_values, "feature_names", None)
    if names_src is None:
        names_src = [f"Feature {i}" for i in range(values.shape[1])]
    feature_names = np.array(names_src)[feature_order]
    values = values[instance_order][:, feature_order]
    feature_values = feature_values[feature_order]

    values, feature_values, name_list = _collapse_excess_features(
        values, feature_values, feature_names, max_display
    )

    row_height = 0.5
    if ax is None:
        plt.gcf().set_size_inches(plot_width, values.shape[1] * row_height + 2.5)
        ax = plt.gca()

    vmin, vmax = np.nanpercentile(values.flatten(), [1, 99])
    v_lo = min(float(vmin), -float(vmax))
    v_hi = max(-float(vmin), float(vmax))
    ax.imshow(
        values.T,
        aspect=0.7 * values.shape[0] / max(values.shape[1], 1),
        interpolation="nearest",
        vmin=v_lo,
        vmax=v_hi,
        cmap=cmap,
    )

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    ax.spines[["left", "right"]].set_visible(True)
    ax.spines[["left", "right"]].set_bounds(values.shape[1] - row_height, -row_height)
    ax.spines[["top", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", direction="out")

    ax.set_ylim(values.shape[1] - row_height, -3)
    heatmap_yticks_pos = np.arange(values.shape[1])
    ax.yaxis.set_ticks(
        [-1.5, *heatmap_yticks_pos],
        [r"$f(x)$", *name_list],
        fontsize=13,
    )
    ax.yaxis.get_ticklines()[0].set_visible(False)

    ax.set_xlim(-0.5, values.shape[0] - 0.5)
    ax.set_xlabel("Instances")

    ax.axhline(-1.5, color="#aaaaaa", linestyle="--", linewidth=0.5)
    fx = values.T.sum(0)
    fx_max = float(np.abs(fx).max())
    fx_normalized = fx / fx_max if fx_max > 0 else fx
    ax.plot(-fx_normalized - 1.5, color="#000000", linewidth=1)

    fv_max = float(np.abs(feature_values).max())
    if fv_max > 0:
        bar_widths = (feature_values / fv_max) * values.shape[0] / 20
    else:
        bar_widths = np.zeros_like(feature_values)

    bar_container = ax.barh(
        heatmap_yticks_pos,
        bar_widths,
        height=0.7,
        align="center",
        color="#000000",
        left=values.shape[0] * 1.0 - 0.5,
    )
    for bar in bar_container:
        bar.set_clip_on(False)

    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([v_lo, v_hi])
    colorbar = plt.colorbar(
        mappable,
        ticks=[v_lo, v_hi],
        ax=ax,
        aspect=80,
        fraction=0.01,
        pad=0.10,
    )
    colorbar.set_label("XWhy values (impact on model output)", size=12, labelpad=-10)
    colorbar.ax.tick_params(labelsize=11, length=0)
    colorbar.set_alpha(1)
    colorbar.outline.set_visible(False)

    return ax


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
) -> Figure | None:
    r"""Plot a heatmap of attributions across instances and features.

    Instances are ordered by hierarchical clustering (or another strategy)
    so that similar explanation patterns sit side by side. Feature rows are
    ranked by mean absolute attribution, with excess features collapsed.

    Parameters
    ----------
    explanation :
        A multi-row Explanation or XWhy result.
    max_display :
        Maximum number of feature rows to show.
    instance_order :
        ``\"hclust\"``, ``\"output\"``, or ``\"none\"``.
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
    **kwargs :
        Extra options forwarded to :func:`_shap_heatmap`
        (``feature_values``, ``feature_order``, ``cmap``, ``plot_width``).

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

    exp = _as_explanation(explanation) if _as_explanation is not None else explanation
    max_d = max_display if max_display is not None else 10

    if figsize is not None:
        _fig, ax = plt.subplots(figsize=figsize)
    else:
        plt.figure(figsize=(8, max_d * 0.7 + 1.5))
        ax = plt.gca()

    _shap_heatmap(
        shap_values=exp,
        instance_order=instance_order,
        feature_values=kwargs.get("feature_values"),
        feature_order=kwargs.get("feature_order"),
        max_display=max_d,
        cmap=kwargs.get("cmap"),
        show=False,
        plot_width=float(kwargs.get("plot_width", 8)),
        ax=ax,
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
