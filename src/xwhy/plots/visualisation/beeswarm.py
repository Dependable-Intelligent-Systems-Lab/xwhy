"""Beeswarm summary plots of per-instance feature attributions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy
import scipy.sparse
import scipy.spatial
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .base import (
    BLUE,
    RED_BLUE,
    DimensionError,
    Explanation,
    _as_explanation,
    _finish_matplotlib,
)


def safe_isinstance(obj: object, class_path_str: str | list[str]) -> bool:
    r"""Return whether *obj*'s MRO contains a class matching *class_path_str*.

    Parameters
    ----------
    obj :
        Any Python object.
    class_path_str :
        Fully-qualified class name(s), e.g. ``\"matplotlib.colors.Colormap\"``.

    Returns
    -------
    bool
        ``True`` if any trailing component of the path appears in the MRO.

    """
    paths = [class_path_str] if isinstance(class_path_str, str) else class_path_str
    class_names = {c.__name__ for c in type(obj).__mro__}
    return any(path.split(".")[-1] in class_names for path in paths)


def convert_color(
    color: str | NDArray[Any] | Colormap,
) -> NDArray[Any] | Colormap | str:
    r"""Resolve a colour alias into a concrete representation.

    Parameters
    ----------
    color :
        Array of RGBA values, a matplotlib colormap name / object, or the
        aliases ``\"shap_red\"`` / ``\"shap_blue\"``.

    Returns
    -------
    ndarray | Colormap | str
        Resolved colour specification.

    """
    if isinstance(color, np.ndarray):
        return color
    if isinstance(color, str) and color == "shap_red":
        return "#FF0D57"
    if isinstance(color, str) and color == "shap_blue":
        return BLUE
    if isinstance(color, Colormap):
        return color
    try:
        return plt.get_cmap(color)
    except ValueError:
        return color


# ---------------------------------------------------------------------------
# Partition-tree helpers (pure)
# ---------------------------------------------------------------------------


def fill_internal_max_values(
    partition_tree: NDArray[Any],
    leaf_values: NDArray[Any],
) -> NDArray[Any]:
    """Fill column 3 of *partition_tree* with the max absolute leaf value.

    Parameters
    ----------
    partition_tree :
        Hierarchical clustering matrix (``n_merges x 4``).
    leaf_values :
        Per-leaf values (length = number of original features).

    Returns
    -------
    NDArray
        Copy of the tree with column 3 updated.

    """
    n_leaves = partition_tree.shape[0] + 1
    new_tree = partition_tree.copy()
    for i in range(new_tree.shape[0]):
        val = 0.0
        for col in (0, 1):
            child = int(new_tree[i, col])
            if child < n_leaves:
                val = max(val, abs(float(leaf_values[child])))
            else:
                ind = child - n_leaves
                val = max(val, abs(float(new_tree[ind, 3])))
        new_tree[i, 3] = val
    return new_tree


def fill_counts(partition_tree: NDArray[Any]) -> None:
    """Update column 3 of *partition_tree* with subtree leaf counts (in-place)."""
    n_leaves = partition_tree.shape[0] + 1
    for i in range(partition_tree.shape[0]):
        val = 0.0
        for col in (0, 1):
            child = int(partition_tree[i, col])
            if child < n_leaves:
                val += 1
            else:
                val += float(partition_tree[child - n_leaves, 3])
        partition_tree[i, 3] = val


def sort_inds(
    partition_tree: NDArray[Any],
    leaf_values: NDArray[Any],
    pos: int | None = None,
    inds: list[int] | None = None,
) -> list[int]:
    """Return a leaf ordering that respects the partition tree.

    Larger-value subtrees are visited first at each internal node.
    """
    if inds is None:
        inds = []

    if pos is None:
        partition_tree = fill_internal_max_values(partition_tree, leaf_values)
        pos = partition_tree.shape[0] - 1

    n_leaves = partition_tree.shape[0] + 1

    if pos < 0:
        inds.append(pos + n_leaves)
        return inds

    left = int(partition_tree[pos, 0]) - n_leaves
    right = int(partition_tree[pos, 1]) - n_leaves

    left_val = (
        float(partition_tree[left, 3])
        if left >= 0
        else float(leaf_values[left + n_leaves])
    )
    right_val = (
        float(partition_tree[right, 3])
        if right >= 0
        else float(leaf_values[right + n_leaves])
    )

    if left_val < right_val:
        left, right = right, left

    sort_inds(partition_tree, leaf_values, left, inds)
    sort_inds(partition_tree, leaf_values, right, inds)
    return inds


def convert_ordering(
    ordering: Any,  # noqa: ANN401
    shap_values: NDArray[Any],
) -> NDArray[Any]:
    """Normalise an ordering specification to an integer index array.

    Accepts raw arrays, SHAP ``OpChain`` objects, or ``Explanation`` objects
    with an argsort history.
    """
    if type(ordering).__name__ == "OpChain":
        ordering = ordering.apply(Explanation(shap_values))
    if type(ordering).__name__ == "Explanation":
        if any(op.name == "argsort" for op in ordering.op_history):
            ordering = ordering.values
        else:
            ordering = ordering.argsort.flip.values
    return np.asarray(ordering)


def get_sort_order(
    dist: NDArray[Any],
    clust_order: list[int] | NDArray[Any],
    cluster_threshold: float,
    feature_order: NDArray[Any],
) -> NDArray[Any]:
    """Sort features while respecting clustering below *cluster_threshold*.

    Nearby features in the partition tree (distance ≤ threshold) stay
    adjacent when possible, ordered by *clust_order*.
    """
    clust_inds = np.argsort(clust_order)
    order = feature_order.copy()

    for i in range(len(order) - 1):
        ind1 = int(order[i])
        next_ind = int(order[i + 1])
        next_ind_pos = i + 1
        for j in range(i + 1, len(order)):
            ind2 = int(order[j])
            if dist[ind1, ind2] <= cluster_threshold and (
                dist[ind1, next_ind] > cluster_threshold
                or clust_inds[ind2] < clust_inds[next_ind]
            ):
                next_ind = ind2
                next_ind_pos = j
        for j in range(next_ind_pos, i + 1, -1):
            order[j] = order[j - 1]
        order[i + 1] = next_ind

    return order


def merge_nodes(
    values: NDArray[Any],
    partition_tree: NDArray[Any],
) -> tuple[NDArray[Any], int, int]:
    """Merge the two clustered leaf nodes with the smallest total value.

    Returns
    -------
    partition_tree_new, ind1, ind2
        Updated tree and the two leaf indices that were merged (``ind1`` kept).

    """
    n_leaves = partition_tree.shape[0] + 1
    ptind = 0
    min_val = np.inf
    for i in range(partition_tree.shape[0]):
        ind1 = int(partition_tree[i, 0])
        ind2 = int(partition_tree[i, 1])
        if ind1 < n_leaves and ind2 < n_leaves:
            val = abs(float(values[ind1])) + abs(float(values[ind2]))
            if val < min_val:
                min_val = val
                ptind = i

    ind1 = int(partition_tree[ptind, 0])
    ind2 = int(partition_tree[ptind, 1])
    if ind1 > ind2:
        ind1, ind2 = ind2, ind1

    tree_new = partition_tree.copy()
    for i in range(tree_new.shape[0]):
        for col in (0, 1):
            child = int(tree_new[i, col])
            if child == ind2:
                tree_new[i, col] = ind1
            elif child > ind2:
                tree_new[i, col] -= 1
                if child == ptind + n_leaves:
                    tree_new[i, col] = ind1
                elif child > ptind + n_leaves:
                    tree_new[i, col] -= 1

    tree_new = np.delete(tree_new, ptind, axis=0)
    fill_counts(tree_new)
    return tree_new, ind1, ind2


# ---------------------------------------------------------------------------
# Beeswarm rendering
# ---------------------------------------------------------------------------


def _shap_beeswarm(
    shap_values: Any,  # noqa: ANN401
    max_display: int | None = 10,
    order: Any = None,  # noqa: ANN401
    clustering: Any = None,  # noqa: ANN401
    cluster_threshold: float = 0.5,
    color: Any = None,  # noqa: ANN401
    axis_color: str = "#333333",
    alpha: float = 1.0,
    ax: Any = None,  # noqa: ANN401
    show: bool = True,
    log_scale: bool = False,
    color_bar: bool = True,
    s: float = 16,
    plot_size: Literal["auto"] | float | tuple[float, float] | None = "auto",
    color_bar_label: str = "Feature value",
    group_remaining_features: bool = True,
    rng: np.random.Generator | None = None,
) -> Any:  # noqa: ANN401
    r"""Create a SHAP beeswarm plot coloured by feature values when provided.

    Parameters
    ----------
    shap_values :
        An ``Explanation`` with a 2-D values matrix (``# samples x # features``).
    max_display :
        Maximum number of feature rows to show.
    order :
        Feature ordering (array, OpChain, or Explanation).
    clustering :
        Optional partition tree; ``False`` disables clustering from the
        Explanation object.
    cluster_threshold :
        Distance below which clustered features stay adjacent.
    color :
        Scatter colour or colormap.
    axis_color :
        Axis tick / spine colour.
    alpha :
        Marker opacity.
    ax :
        Optional matplotlib Axes; when set, *plot_size* must be ``None``.
    show :
        Accepted for API compatibility.
    log_scale :
        Use a symmetric log x-axis when ``True``.
    color_bar :
        Draw a low/high colour bar when a colormap is used.
    s :
        Marker size.
    plot_size :
        ``\"auto\"``, row height in inches, ``(width, height)``, or ``None``.
    color_bar_label :
        Label for the colour bar.
    group_remaining_features :
        Collapse features beyond *max_display* into a single summed row.
    rng :
        Optional NumPy random generator used for marker jitter and row
        shuffling. When omitted, a fresh default generator is created.

    Returns
    -------
    Axes
        The axes the plot was drawn onto.

    """
    _ = show
    if rng is None:
        rng = np.random.default_rng()
    if (
        Explanation is not None
        and not isinstance(shap_values, Explanation)
        and not hasattr(shap_values, "values")
    ):
        msg = (
            "The beeswarm plot requires an `Explanation` object as the "
            "`shap_values` argument."
        )
        raise TypeError(msg)

    sv_shape = shap_values.values.shape
    if len(sv_shape) == 1:
        msg = (
            "The beeswarm plot does not support plotting a single instance, "
            "please pass an explanation matrix with many instances!"
        )
        raise ValueError(msg)
    if len(sv_shape) > 2:
        msg = (
            "The beeswarm plot does not support plotting explanations with "
            "instances that have more than one dimension!"
        )
        raise ValueError(msg)

    if ax is not None and plot_size is not None:
        msg = (
            "The beeswarm plot does not support passing an axis and adjusting "
            "the plot size. To adjust the size of the plot, set plot_size to "
            "None and adjust the size on the original figure the axes was "
            "part of."
        )
        raise ValueError(msg)

    values = np.copy(shap_values.values)
    features = shap_values.data
    if scipy.sparse.issparse(features):
        features = features.toarray()
    feature_names: Any = shap_values.feature_names

    if order is None:
        order = np.argsort(np.abs(values).mean(0))[::-1]
    order = convert_ordering(order, values)

    if color is None:
        color = RED_BLUE if features is not None else BLUE
    color = convert_color(color)

    idx2cat: list[bool] | None = None
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = values.shape[1]

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

    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    if not isinstance(fig, Figure):
        msg = "Expected a matplotlib Figure from ax.get_figure()"
        raise TypeError(msg)

    if log_scale:
        ax.set_xscale("symlog")

    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        partition_tree = None
    else:
        partition_tree = clustering

    if partition_tree is not None and partition_tree.shape[1] != 4:
        msg = (
            "The clustering provided by the Explanation object does not seem "
            "to be a partition tree (which is all xwhy.plots.bar supports)!"
        )
        raise ValueError(msg)

    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(feature_names))

    orig_inds: list[list[int]] = [[i] for i in range(len(feature_names))]
    orig_values = values.copy()

    while True:
        feature_order = convert_ordering(
            order,
            Explanation(np.abs(values)) if Explanation is not None else np.abs(values),  # type: ignore[arg-type]
        )
        if partition_tree is not None:
            clust_order = sort_inds(partition_tree, np.abs(values))
            dist = scipy.spatial.distance.squareform(
                scipy.cluster.hierarchy.cophenet(partition_tree)
            )
            feature_order = get_sort_order(
                dist, clust_order, cluster_threshold, feature_order
            )
            if max_display < len(feature_order) and (
                dist[
                    feature_order[max_display - 1],
                    feature_order[max_display - 2],
                ]
                <= cluster_threshold
            ):
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)
                for _ in range(len(values)):
                    values[:, ind1] += values[:, ind2]
                    values = np.delete(values, ind2, 1)
                    orig_inds[ind1] += orig_inds[ind2]
                    del orig_inds[ind2]
            else:
                break
        else:
            break

    feature_inds = feature_order[:max_display]
    feature_names_new: list[str] = []
    for inds in orig_inds:
        if len(inds) == 1:
            feature_names_new.append(str(feature_names[inds[0]]))
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join(str(feature_names[i]) for i in inds))
        else:
            max_ind = int(np.argmax(np.abs(orig_values).mean(0)[inds]))
            feature_names_new.append(
                f"{feature_names[inds[max_ind]]} + {len(inds) - 1} other features"
            )
    feature_names = feature_names_new

    include_grouped_remaining = (
        num_features < len(values[0]) and group_remaining_features
    )
    num_cut = 0
    if include_grouped_remaining:
        num_cut = int(
            np.sum(
                [
                    len(orig_inds[feature_order[i]])
                    for i in range(num_features - 1, len(values[0]))
                ]
            )
        )
        values[:, feature_order[num_features - 1]] = np.sum(
            [
                values[:, feature_order[i]]
                for i in range(num_features - 1, len(values[0]))
            ],
            0,
        )

    yticklabels = [feature_names[i] for i in feature_inds]
    if include_grouped_remaining:
        yticklabels[-1] = f"Sum of {num_cut} other features"

    row_height = 0.4
    if plot_size == "auto":
        fig.set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        fig.set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        fig.set_size_inches(
            8, min(len(feature_order), max_display) * float(plot_size) + 1.5
        )

    ax.axvline(x=0, color="#999999", zorder=-1)

    for pos, i in enumerate(reversed(list(feature_inds))):
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)
        shaps = values[:, i]
        fvalues = None if features is None else features[:, i]

        f_inds = np.arange(len(shaps))
        rng.shuffle(f_inds)
        if fvalues is not None:
            fvalues = fvalues[f_inds]
        shaps = shaps[f_inds]
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:
                colored_feature = False
            else:
                fvalues = np.array(fvalues, dtype=np.float64)
        except Exception:
            colored_feature = False

        n_pts = len(shaps)
        nbins = 100
        quant = np.round(
            nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8)
        )
        inds_ = np.argsort(quant + rng.standard_normal(n_pts) * 1e-6)
        layer = 0
        last_bin = -1.0
        ys = np.zeros(n_pts)
        for ind in inds_:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]
        ys *= 0.9 * (row_height / np.max(ys + 1))

        is_cmap = safe_isinstance(color, "matplotlib.colors.Colormap")
        if is_cmap and fvalues is not None and colored_feature:
            vmin = float(np.nanpercentile(fvalues, 5))
            vmax = float(np.nanpercentile(fvalues, 95))
            if vmin == vmax:
                vmin = float(np.nanpercentile(fvalues, 1))
                vmax = float(np.nanpercentile(fvalues, 99))
                if vmin == vmax:
                    vmin = float(np.min(fvalues))
                    vmax = float(np.max(fvalues))
            if vmin > vmax:
                vmin = vmax

            if features is not None and features.shape[0] != len(shaps):
                msg = "Feature and SHAP matrices must have the same number of rows!"
                raise DimensionError(msg)

            nan_mask = np.isnan(fvalues)
            ax.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )
            cvals = fvalues[np.invert(nan_mask)].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[np.invert(nan_mask)],
                pos + ys[np.invert(nan_mask)],
                cmap=color,
                vmin=vmin,
                vmax=vmax,
                s=s,
                c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )
        else:
            scatter_color: Any = color
            if is_cmap and hasattr(color, "colors"):
                scatter_color = color.colors
            ax.scatter(
                shaps,
                pos + ys,
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color=scatter_color if colored_feature else "#777777",
                rasterized=len(shaps) > 500,
            )

    if is_cmap and color_bar and features is not None:
        mappable = cm.ScalarMappable(cmap=color)
        mappable.set_array([0, 1])
        colorbar = fig.colorbar(mappable, ax=ax, ticks=[0, 1], aspect=80)
        colorbar.set_ticklabels(["Low", "High"])
        colorbar.set_label(color_bar_label, size=12, labelpad=0)
        colorbar.ax.tick_params(labelsize=11, length=0)
        colorbar.set_alpha(1)
        colorbar.outline.set_visible(False)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(range(len(feature_inds)), list(reversed(yticklabels)), fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(feature_inds))
    ax.set_xlabel("XWhy value (impact on model output)", fontsize=13)
    return ax


def beeswarm(
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
) -> Figure | Any | None:  # noqa: ANN401
    r"""Create a beeswarm summary plot.

    Every instance contributes one dot per feature. Dots are spread vertically
    by local density and coloured by the feature's own value.

    Parameters
    ----------
    explanation :
        A batched Explanation or XWhy result.
    max_display :
        Maximum number of feature rows to draw.
    show :
        Whether to display the figure.
    save_path :
        Optional path to write the figure to.
    backend :
        Currently only ``\"matplotlib\"`` is implemented in this module.
    title :
        Optional figure title.
    figsize :
        Optional matplotlib figure size in inches.
    seed :
        Seed for the jitter applied when breaking density ties.
    **kwargs :
        Extra options forwarded to :func:`_shap_beeswarm`.

    Returns
    -------
    Figure | None
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    _ = backend  # plotly path reserved for future use
    rng = np.random.default_rng(seed)

    exp = _as_explanation(explanation) if _as_explanation is not None else explanation

    if figsize is not None:
        fig, _ax_dummy = plt.subplots(figsize=figsize)
        plot_size: Literal["auto"] | float | tuple[float, float] | None = None
    else:
        fig = plt.gcf()
        plot_size = kwargs.get("plot_size", "auto")

    ax = _shap_beeswarm(
        shap_values=exp,
        max_display=max_display if max_display is not None else 10,
        order=kwargs.get("order"),
        clustering=kwargs.get("clustering"),
        cluster_threshold=float(kwargs.get("cluster_threshold", 0.5)),
        color=kwargs.get("color"),
        axis_color=kwargs.get("axis_color", "#333333"),
        alpha=float(kwargs.get("alpha", 1.0)),
        show=False,
        log_scale=bool(kwargs.get("log_scale", False)),
        color_bar=bool(kwargs.get("color_bar", True)),
        s=float(kwargs.get("s", 16)),
        plot_size=plot_size,
        color_bar_label=kwargs.get("color_bar_label", "Feature value"),
        group_remaining_features=bool(kwargs.get("group_remaining_features", True)),
        rng=rng,
    )

    fig = ax.figure if ax is not None else plt.gcf()

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    if _finish_matplotlib is not None:
        return _finish_matplotlib(fig, show=show, save_path=save_path)
    if show:
        plt.show()
        return None
    return fig
