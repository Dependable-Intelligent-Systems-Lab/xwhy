"""Temporal monitoring scatters of feature attributions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.stats import ttest_ind

from .base import (
    RED_BLUE,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    _global_importance,
    _resolve_names,
)


def truncate_text(text: str, max_len: int) -> str:
    """Truncate *text* to *max_len* characters with an ellipsis in the middle.

    Args:
        text: Input string.
        max_len: Maximum length of the result.

    Returns:
        The original string if short enough, otherwise a truncated form.

    """
    if len(text) > max_len:
        half = int(max_len / 2)
        return text[: half - 2] + "..." + text[-half + 1 :]
    return text


def _feature_index(
    ind: int | str | None,
    names: Sequence[str],
    values: NDArray[Any],
) -> int:
    """Resolve a feature reference to a column index.

    Args:
        ind: Feature index, feature name, or ``None`` for the most
            important feature.
        names: Feature names.
        values: Attribution or data matrix used to rank importance.

    Returns:
        The resolved column index.

    Raises:
        ValueError: If the name is unknown or the index is out of range.

    """
    n_features = np.atleast_2d(values).shape[1]

    if ind is None:
        if _global_importance is not None:
            return int(np.argmax(_global_importance(np.atleast_2d(values))))
        return int(np.argmax(np.abs(np.atleast_2d(values)).mean(0)))

    if isinstance(ind, str):
        lookup = list(names)
        if ind not in lookup:
            preview = ", ".join(lookup[:10])
            suffix = "..." if len(lookup) > 10 else ""
            msg = f"Unknown feature {ind!r}. Available features: {preview}{suffix}"
            raise ValueError(msg)
        return lookup.index(ind)

    index = int(ind)
    if not -n_features <= index < n_features:
        msg = f"Feature index {index} is out of range for {n_features} features."
        raise ValueError(msg)
    return int(index % n_features)


def _shap_monitoring(
    ind: int,
    shap_values: NDArray[Any],
    features: NDArray[Any] | pd.DataFrame,
    feature_names: list[str] | NDArray[Any] | None = None,
    show: bool = True,
) -> None:
    """Create a SHAP monitoring plot (preliminary API).

    Displays model behaviour over sample order. Attributions often explain
    loss, so changes in a feature's impact over time can help monitor
    performance.

    Args:
        ind: Index of the feature to plot.
        shap_values: Matrix of XWhy values (``# samples x # features``).
        features: Matrix of feature values (array or DataFrame).
        feature_names: Optional feature names.
        show: Call ``plt.show()`` when True.

    """
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = list(features.columns)
        features = features.values

    num_features = shap_values.shape[1]
    if feature_names is None:
        feature_names = np.array([f"Feature {i}" for i in range(num_features)])

    plt.figure(figsize=(10, 3))
    ys = shap_values[:, ind]
    xs = np.arange(len(ys))

    pvals: list[float] = []
    inc = 50
    for i in range(inc, len(ys) - inc, inc):
        _, pval = ttest_ind(ys[:i], ys[i:])
        pvals.append(float(pval))

    if len(pvals) > 0:
        min_pval = float(np.min(pvals))
        min_pval_ind = float(np.argmin(pvals) * inc + inc)
    else:
        min_pval = 1.0
        min_pval_ind = 0.0

    if min_pval < 0.05 / shap_values.shape[1]:
        plt.axvline(min_pval_ind, linestyle="dashed", color="#666666", alpha=0.2)

    plt.scatter(xs, ys, s=10, c=features[:, ind], cmap=RED_BLUE)

    label = truncate_text(str(feature_names[ind]), 30)
    plt.xlabel("Sample index")
    plt.ylabel(f"{label}\nXWhy value", size=13)
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("left")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    colorbar = plt.colorbar()
    colorbar.outline.set_visible(False)
    bbox = colorbar.ax.get_window_extent().transformed(
        plt.gcf().dpi_scale_trans.inverted()
    )
    colorbar.ax.set_aspect((bbox.height - 0.7) * 20)
    colorbar.set_label(label, size=13)
    if show:
        plt.show()


def monitoring(
    ind: int | str,
    explanation: Any,  # noqa: ANN401
    features: NDArray[Any] | None = None,
    *,
    n_splits: int = 50,
    show: bool = True,
    save_path: str | Path | None = None,
    backend: str = "matplotlib",
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    r"""Plot feature attributions against sample index for monitoring.

    Args:
        ind: Feature index or name to plot.
        explanation: Explanation or XWhy result.
        features: Optional feature matrix; defaults to ``explanation.data``.
        n_splits: Reserved for future change-point granularity.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: Currently only ``\"matplotlib\"`` is supported.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Accepted for call compatibility; currently ignored.

    Returns:
        The figure when the host finish-helper returns it, otherwise
        ``None`` when ``show`` is True.

    """
    _ = n_splits, kwargs
    allowed = frozenset({"matplotlib"})
    if _check_backend is not None:
        _check_backend(backend, allowed)
    elif backend not in allowed:
        msg = f"Unsupported backend: {backend}"
        raise ValueError(msg)

    exp = _as_explanation(explanation) if _as_explanation is not None else explanation
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    if _resolve_names is not None:
        names = _resolve_names(exp.feature_names, values.shape[1])
    else:
        names = getattr(exp, "feature_names", None) or [
            f"Feature {i}" for i in range(values.shape[1])
        ]

    if features is None:
        if getattr(exp, "data", None) is not None:
            features = np.asarray(exp.data)
        else:
            features = np.zeros(values.shape)

    if figsize is not None:
        plt.subplots(figsize=figsize)

    resolved = _feature_index(ind, list(names), values)
    _shap_monitoring(
        ind=resolved,
        shap_values=values,
        features=features,
        feature_names=names,
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
