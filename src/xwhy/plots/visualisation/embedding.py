"""2-D embeddings of sample-level attribution vectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import sklearn.decomposition
from matplotlib.figure import Figure
from numpy.typing import NDArray

from .base import (
    RED_BLUE,
    _as_explanation,
    _check_backend,
    _finish_matplotlib,
    _resolve_names,
    convert_name,
)


def _shap_embedding(
    ind: int | str,
    shap_values: NDArray[Any],
    feature_names: list[str] | NDArray[Any] | None = None,
    method: str | NDArray[Any] = "pca",
    alpha: float = 1.0,
    show: bool = True,
) -> None:
    r"""Scatter samples in a 2-D embedding of their SHAP vectors.

    Args:
        ind: Feature index, name, or ``\"sum()\"`` used to colour points.
        shap_values: Attribution matrix (``n_samples x n_features``).
        feature_names: Optional feature names (length ``n_features``).
        method: ``\"pca\"`` or a precomputed ``(n_samples, 2)`` array.
        alpha: Marker opacity.
        show: Accepted for API compatibility; display is left to the caller.

    """
    _ = show
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(shap_values.shape[1])]

    resolved = convert_name(ind, shap_values, feature_names)
    if resolved == "sum()":
        cvals = shap_values.sum(1)
        fname = "sum(XWhy values)"
    else:
        cvals = shap_values[:, resolved]  # type: ignore[index]
        fname = str(feature_names[resolved])  # type: ignore[index]

    if isinstance(method, str) and method == "pca":
        pca = sklearn.decomposition.PCA(2)
        embedding_values = pca.fit_transform(shap_values)
    elif hasattr(method, "shape") and method.shape[1] == 2:
        embedding_values = method
    else:
        msg = f"Unsupported embedding method: {method}"
        raise ValueError(msg)

    plt.scatter(
        embedding_values[:, 0],
        embedding_values[:, 1],
        c=cvals,
        cmap=RED_BLUE,
        alpha=alpha,
        linewidth=0,
    )
    plt.axis("off")
    colorbar = plt.colorbar()
    colorbar.set_label(f"XWhy value for\n{fname}", size=13)
    colorbar.outline.set_visible(False)

    plt.gcf().set_size_inches(7.5, 5)
    bbox = colorbar.ax.get_window_extent().transformed(
        plt.gcf().dpi_scale_trans.inverted()
    )
    colorbar.ax.set_aspect((bbox.height - 0.7) * 10)
    colorbar.set_alpha(1)


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
) -> Figure | None:
    r"""Plot a 2-D embedding of sample attribution vectors.

    Args:
        ind: Feature index, name, or ``\"sum()\"`` for colouring.
        explanation: Explanation or XWhy result.
        show: Whether to display the figure.
        save_path: Optional path to write the figure to.
        backend: Currently only ``\"matplotlib\"`` is supported.
        title: Optional figure title.
        figsize: Optional matplotlib figure size in inches.
        **kwargs: Extra options forwarded to :func:`_shap_embedding`
            (``method``, ``alpha``).

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

    exp = _as_explanation(explanation) if _as_explanation is not None else explanation
    values = np.atleast_2d(np.asarray(exp.values, dtype=float))
    if _resolve_names is not None:
        names = _resolve_names(exp.feature_names, values.shape[1])
    else:
        names = getattr(exp, "feature_names", None)

    if figsize is not None:
        plt.subplots(figsize=figsize)

    _shap_embedding(
        ind=ind,
        shap_values=values,
        feature_names=names,
        method=kwargs.get("method", "pca"),
        alpha=float(kwargs.get("alpha", 1.0)),
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
