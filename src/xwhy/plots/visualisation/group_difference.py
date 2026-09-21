"""Mean attribution differences between two sample groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .base import (
    BLUE_RGB,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    _resolve_names,
)


def _shap_group_difference(
    shap_values: NDArray[Any],
    group_mask: NDArray[Any],
    feature_names: list[str] | NDArray[Any] | None = None,
    xlabel: str | None = None,
    xmin: float | None = None,
    xmax: float | None = None,
    max_display: int | None = None,
    sort: bool = True,
    show: bool = True,
    ax: Any = None,  # noqa: ANN401
    *,
    rng: np.random.Generator | None = None,
) -> None:
    r"""Plot the difference in mean XWhy values between two groups.

    Useful for decomposing group-level metrics (e.g. fairness) across
    input features.

    Args:
        shap_values: Matrix of XWhy values (``# samples x # features``)
            or a vector of model outputs (``# samples``).
        group_mask: Boolean mask; ``True`` is the first group, ``False``
            the second.
        feature_names: Optional feature names.
        xlabel: X-axis label (default ``\"Group XWhy value difference\"``).
        xmin: Optional lower x-limit.
        xmax: Optional upper x-limit.
        max_display: Maximum number of features to show.
        sort: Sort features by absolute mean difference.
        show: Call ``plt.show()`` when True and no *ax* was given.
        ax: Optional matplotlib Axes.
        rng: Optional NumPy random generator for bootstrap resampling.

    """
    if rng is None:
        rng = np.random.default_rng()

    # Bootstrap confidence bounds for the group difference
    vs: list[NDArray[Any]] = []
    gmean = float(group_mask.mean())
    for _ in range(200):
        r = rng.random(shap_values.shape[0]) > gmean
        vs.append(shap_values[r].mean(0) - shap_values[~r].mean(0))
    vs_arr = np.array(vs)
    xerr = np.vstack(
        [
            np.percentile(vs_arr, 95, axis=0),
            np.percentile(vs_arr, 5, axis=0),
        ]
    )

    # Single model-output vector → column matrix
    if len(shap_values.shape) == 1:
        shap_values = shap_values.reshape(1, -1).T
        if feature_names is None:
            feature_names = [""]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

    diff = shap_values[group_mask].mean(0) - shap_values[~group_mask].mean(0)

    inds = np.argsort(-np.abs(diff)).astype(int) if sort else np.arange(len(diff))

    if max_display is not None:
        inds = inds[:max_display]

    if ax is not None:
        show = False
    else:
        figsize = (6.4, 0.2 + 0.9 * len(inds))
        _, ax = plt.subplots(figsize=figsize)

    ticks = range(len(inds) - 1, -1, -1)
    ax.axvline(0, color="#999999", linewidth=0.5)
    ax.barh(
        ticks,
        diff[inds],
        color=BLUE_RGB,
        capsize=3,
        xerr=np.abs(xerr[:, inds]),
    )

    for i in range(len(inds)):
        ax.axhline(y=i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(feature_names[i]) for i in inds], fontsize=13)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(labelsize=11)
    if xlabel is None:
        xlabel = "Group XWhy value difference"
    ax.set_xlabel(xlabel, fontsize=13, color="#000000")
    ax.tick_params("x", labelsize=11)
    ax.set_xlim(xmin, xmax)
    if show:
        plt.show()


def group_difference(
    explanation: Any,  # noqa: ANN401
    group_mask: NDArray[Any],
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
    r"""Plot mean attribution differences between two groups.

    Args:
        explanation: Explanation or XWhy result.
        group_mask: Boolean mask selecting the first group.
        max_display: Maximum number of feature rows.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: Currently only ``\"matplotlib\"`` is supported.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        seed: Seed for the bootstrap random generator.
        **kwargs: Extra options forwarded to
            :func:`_shap_group_difference` (``xlabel``, ``xmin``,
            ``xmax``, ``sort``).

    Returns:
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
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    if _resolve_names is not None:
        names = _resolve_names(exp.feature_names, values.shape[1])
    else:
        names = getattr(exp, "feature_names", None)

    if figsize is not None:
        _fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = None

    _shap_group_difference(
        shap_values=values,
        group_mask=np.asarray(group_mask, dtype=bool),
        feature_names=names,
        xlabel=kwargs.get("xlabel"),
        xmin=kwargs.get("xmin"),
        xmax=kwargs.get("xmax"),
        max_display=max_display,
        sort=bool(kwargs.get("sort", True)),
        show=False,
        ax=ax,
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
