"""Tabular plotting utilities for the xwhy library."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from xwhy.core.result import TabularXWhyResult

GRAY_CMAP = LinearSegmentedColormap.from_list(
    "gray_map", [(0.3, 0.3, 0.3), (0.8, 0.8, 0.8)], N=2
)


def set_plot_style() -> None:
    """Set consistent plot style for 2D visualization."""
    plt.axis((-2, 2, -2, 2))
    plt.xlabel("x1")
    plt.ylabel("x2")


def plot_dataset(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float] | None = None,
    *,
    cmap: LinearSegmentedColormap = GRAY_CMAP,
    point: np.ndarray | Sequence[float] | None = None,
    point_style: dict[str, Any] | None = None,
    scatter_kwargs: dict[str, Any] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> None:
    """Plot dataset or single point with flexible matplotlib-style arguments.

    Supports:
    - Full dataset: shape (n_samples, 2)
    - Single point: shape (2,)

    Args:
        x: Input data array.
        y: Optional labels or second coordinate.
        cmap: Colormap for matplotlib.
        point: Additional highlighted point.
        point_style: Style configuration for highlighted point.
        scatter_kwargs: Extra kwargs for plt.scatter.
        show: Whether to display the plot interactively.
        save_path: Optional path to save the generated plot.

    """
    set_plot_style()

    scatter_kwargs = scatter_kwargs or {}
    x_arr = np.asarray(x)

    # ==============================
    # CASE 1: Single point (shape: (2,))
    # ==============================
    if x_arr.ndim == 1 and x_arr.shape[0] == 2:
        plt.scatter(
            x_arr[0],
            x_arr[1],
            **scatter_kwargs,
        )

    # ==============================
    # CASE 2: Dataset (shape: (n, 2))
    # ==============================
    elif x_arr.ndim == 2 and x_arr.shape[1] == 2:
        if y is not None:
            plt.scatter(
                x_arr[:, 0],
                x_arr[:, 1],
                c=y,
                cmap=cmap,
                **scatter_kwargs,
            )
        else:
            plt.scatter(
                x_arr[:, 0],
                x_arr[:, 1],
                **scatter_kwargs,
            )
    else:
        raise ValueError("x must be either shape (n_samples, 2) or (2,)")

    # ==============================
    # OPTIONAL EXTRA POINT
    # ==============================
    if point is not None:
        default_style: dict[str, Any] = {"c": "blue", "marker": "o", "s": 70}
        if point_style:
            default_style.update(point_style)

        plt.scatter(point[0], point[1], **default_style)

    if save_path:
        plt.savefig(str(save_path), bbox_inches="tight")
    elif show:
        plt.show()


def plot_feature_contributions(
    result: TabularXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Visualize feature contributions using a horizontal bar chart.

    Supports any number of features and separates positive and
    negative contributions dynamically.

    Args:
        result: Tabular explanation result containing coefficients and features.
        **kwargs: Additional plotting arguments (e.g., title, save_path).

    """
    coeffs = np.asarray(result.coefficients)
    feature_names = result.feature_list
    num_features = len(coeffs)

    title: str = str(kwargs.pop("title", "Feature Contributions"))
    save_path: str | Path | None = kwargs.pop("save_path", None)

    # ==============================
    # Feature names handling
    # ==============================
    if not feature_names:
        feature_names = [f"x{i}" for i in range(num_features)]

    # ==============================
    # Vectorized split
    # ==============================
    neg = np.minimum(coeffs, 0)
    pos = np.maximum(coeffs, 0)

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "negative": neg,
            "positive": pos,
        }
    )

    # ==============================
    # Plot (both sides)
    # ==============================
    fig = px.bar(
        df,
        x=["negative", "positive"],
        y="feature",
        orientation="h",
        barmode="relative",
        title=title,
    )

    if save_path:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)
    else:
        fig.show()


def plot_method_contributions(
    result: TabularXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Visualize feature contributions for a given explanation method.

    This function creates a horizontal bar plot using Plotly,
    supporting any number of features and methods.

    Args:
        result: Tabular explanation result.
        **kwargs: Additional arguments including 'title', 'method_name',
            and 'save_path'.

    """
    coeffs = np.asarray(result.coefficients)
    feature_names = result.feature_list
    num_features = len(coeffs)

    # Automatically resolve method name from result raw data or kwargs
    method_name: str = kwargs.pop("method_name", None)
    if not method_name and result.raw_data and "surrogate_method" in result.raw_data:
        method_name = str(result.raw_data["surrogate_method"].value)
    elif not method_name:
        method_name = "SMILE"

    title: str = str(kwargs.pop("title", f"{method_name} Feature Contributions"))
    save_path: str | Path | None = kwargs.pop("save_path", None)

    if not feature_names:
        feature_names = [f"x{i}" for i in range(num_features)]

    # ==============================
    # Create DataFrame
    # ==============================
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": coeffs,
        }
    )

    df = df.sort_values("importance", key=np.abs, ascending=True)

    # ==============================
    # Plot
    # ==============================
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        color="feature",
        title=title,
    )

    if save_path:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)
    else:
        fig.show()


def plot_explanation_waterfall(
    result: TabularXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Create a dynamic waterfall plot for explanation method coefficients.

    Args:
        result: Tabular explanation result.
        **kwargs: Additional arguments (title, orientation, save_path).

    """
    coeffs = np.asarray(result.coefficients).flatten()
    feature_names = result.feature_list
    num_features = len(coeffs)

    title: str = str(kwargs.pop("title", "Model Explanation (Waterfall)"))
    orientation: str = str(kwargs.pop("orientation", "h"))
    save_path: str | Path | None = kwargs.pop("save_path", None)

    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(num_features)]

    # ==============================
    # Build dataframe (clean structure)
    # ==============================
    df = pd.DataFrame(
        {
            "coefficient": coeffs,
            "feature": feature_names,
        }
    )

    # ==============================
    # Create waterfall plot
    # ==============================
    fig = go.Figure(
        go.Waterfall(
            name="explanation",
            orientation=orientation,
            x=df["coefficient"],
            y=df["feature"],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        )
    )

    # ==============================
    # Layout
    # ==============================
    fig.update_layout(
        title=title,
        showlegend=False,
    )

    if save_path:
        path_str = str(save_path)
        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)
    else:
        fig.show()
