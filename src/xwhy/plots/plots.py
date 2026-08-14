"""Plotting interface for XWhy results.

The visualisations themselves live in :mod:`xwhy.plots.visualisation`, a native
matplotlib/plotly/HTML engine. This module is the thin public surface over it:
it validates the shape of each result and forwards to the engine, so the call
signatures stay stable for existing notebooks.
"""

import functools
import inspect
import os
from collections.abc import Callable
from functools import singledispatch
from pathlib import Path
from typing import Any, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure

from xwhy.core.result import (
    BaseXWhyResult,
    ImageGenerationAndEditingXWhyResult,
    TextXWhyResult,
)
from xwhy.logger import logger
from xwhy.plots import visualisation as viz
from xwhy.plots.factory import TextPlotterFactory
from xwhy.plots.image import image_heatmap, plot_image  # noqa: F401
from xwhy.plots.tabular import (
    plot_dataset,  # noqa: F401
    plot_explanation_waterfall,  # noqa: F401
    plot_feature_contributions,  # noqa: F401
    plot_method_contributions,  # noqa: F401
)
from xwhy.plots.types import TextPlotterType
from xwhy.plots.visualisation import Explanation  # noqa: F401

F = TypeVar("F", bound=Callable[..., Any])

#: Return type shared by the plotting wrappers.
type PlotResult = Figure | go.Figure | None


@singledispatch
def text_heatmap(
    result: BaseXWhyResult | ImageGenerationAndEditingXWhyResult, **kwargs: object
) -> None:
    """Plot a heatmap visualization for the given explanation result.

    This function automatically delegates plotting to the appropriate plotter
    based on the concrete type of the result (e.g., TextXWhyResult).

    Args:
        result: The explanation result object.
        **kwargs: Additional plotting arguments (e.g., title, backend).

    Raises:
        TypeError: If the given result type does not support heatmap visualization.

    """
    msg = f"Heatmap visualization is not supported for {type(result).__name__}."
    raise TypeError(msg)


@text_heatmap.register
def _text_heatmap(
    result: TextXWhyResult | ImageGenerationAndEditingXWhyResult, **kwargs: object
) -> None:
    """Plot a text heatmap visualization.

    Args:
        result: Text explanation result.
        **kwargs: Can include 'title' (str) and 'backend' (TextPlotterType or str),
            as well as any other arguments passed to the visualizer plot method.

    Raises:
        ValueError: If an unsupported backend type is provided.

    """
    title = str(kwargs.pop("title", "Text Heatmap"))
    backend_kwarg = kwargs.pop("backend", TextPlotterType.NATIVE_HEATMAP)

    if isinstance(backend_kwarg, TextPlotterType):
        backend = backend_kwarg
    elif isinstance(backend_kwarg, str):
        backend = TextPlotterType(backend_kwarg)
    else:
        msg = f"Unsupported backend type: {type(backend_kwarg).__name__}"
        raise ValueError(msg)

    plotter = TextPlotterFactory.create(method=backend)
    plotter.plot(
        words=result.words,
        scores=result.coefficients,
        title=title,
        **kwargs,
    )


def plot_feature_bar_chart(
    result: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Generate and optionally save a Plotly bar chart for feature contributions.

    Args:
        result: The explanation result containing feature names and coefficients.
        **kwargs: Additional arguments including 'title', 'xaxis_title',
            'yaxis_title', 'width', 'height', and 'save_path'.

    """
    coeffs = np.asarray(result.coefficients).flatten()
    feature_names = result.feature_names
    num_features = len(coeffs)

    if not feature_names:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    title: str = str(kwargs.get("title", "Feature Contributions"))
    xaxis_title: str = str(kwargs.get("xaxis_title", "Features"))
    yaxis_title: str = str(kwargs.get("yaxis_title", "Contribution Value"))
    width: int = int(kwargs.get("width", 800))
    height: int = int(kwargs.get("height", 600))
    save_path: str | Path | None = kwargs.get("save_path")

    fig = go.Figure(
        data=[
            go.Bar(
                x=list(feature_names),
                y=coeffs.tolist(),
                marker_color="skyblue",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
        xaxis={
            "tickangle": 45,
            "tickfont": {
                "size": 14,
                "family": "Arial",
                "color": "black",
                "weight": "bold",
            },
        },
        yaxis={
            "tickfont": {
                "size": 12,
                "family": "Arial",
                "color": "black",
                "weight": "bold",
            },
        },
        title_font={
            "size": 16,
            "family": "Arial",
            "color": "black",
            "weight": "bold",
        },
        width=width,
        height=height,
    )

    if save_path:
        path_str = str(save_path)
        save_dir = os.path.dirname(path_str)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)

        logger.debug("Bar chart saved to: %s", path_str)
    else:
        fig.show()


def plot_feature_box_plot(
    result: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Generate and optionally save a Plotly box plot for feature contributions.

    Args:
        result: The explanation result containing feature names and coefficients.
        **kwargs: Additional arguments including 'title', 'xaxis_title',
            'yaxis_title', 'width', 'height', and 'save_path'.

    """
    coeffs = np.asarray(result.coefficients)
    feature_names = result.feature_names
    num_features = len(coeffs)

    if not feature_names:
        feature_names = [f"Feature {i}" for i in range(num_features)]

    title: str = str(kwargs.get("title", "Feature Contributions Box Plot"))
    xaxis_title: str = str(kwargs.get("xaxis_title", "Features"))
    yaxis_title: str = str(kwargs.get("yaxis_title", "Contribution Value"))
    width: int = int(kwargs.get("width", 800))
    height: int = int(kwargs.get("height", 800))
    save_path: str | Path | None = kwargs.get("save_path")

    fig = go.Figure()

    # Handle both 1D arrays (single value per feature) and 2D arrays
    if coeffs.ndim == 1:
        for name, val in zip(feature_names, coeffs, strict=False):
            fig.add_trace(go.Box(y=[val], name=str(name), boxpoints="all"))
    else:
        for idx, name in enumerate(feature_names):
            fig.add_trace(go.Box(y=coeffs[:, idx], name=str(name), boxpoints="all"))

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=False,
        xaxis={
            "tickangle": 45,
            "tickfont": {
                "size": 14,
                "family": "Arial",
                "color": "black",
                "weight": "bold",
            },
        },
        yaxis={
            "tickfont": {
                "size": 12,
                "family": "Arial",
                "color": "black",
                "weight": "bold",
            },
        },
        title_font={
            "size": 16,
            "family": "Arial",
            "color": "black",
            "weight": "bold",
        },
        width=width,
        height=height,
    )

    if save_path:
        path_str = str(save_path)
        save_dir = os.path.dirname(path_str)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if path_str.endswith(".html"):
            fig.write_html(path_str)
        else:
            fig.write_image(path_str)

        logger.debug("Box plot saved to: %s", path_str)
    else:
        fig.show()


# ==============================================================================
# COMPATIBILITY HELPERS
# ==============================================================================


def replace_shap_label[F: Callable[..., Any]](plot_func: F) -> F:
    """Rewrite a lingering 'SHAP value' axis label to 'XWhy value'.

    XWhy's own plots already label their axes correctly, so this decorator is
    no longer applied internally. It is kept, and still exported, so that user
    code wrapping a third-party plotting function keeps working.

    It detects whether the wrapped function accepts a ``show`` parameter. If it
    does, rendering is suppressed long enough to relabel the axis, then the
    original ``show`` intent is honoured. Otherwise arguments pass through
    untouched.

    Args:
        plot_func: The plotting function to wrap.

    Returns:
        F: The wrapped function.

    """
    try:
        accepts_show = "show" in inspect.signature(plot_func).parameters
    except (ValueError, TypeError):
        accepts_show = False

    @functools.wraps(plot_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if not accepts_show:
            return plot_func(*args, **kwargs)

        original_show = kwargs.get("show", True)
        kwargs["show"] = False

        result = plot_func(*args, **kwargs)

        if len(plt.get_fignums()) > 0:
            ax = plt.gca()
            current_xlabel = ax.get_xlabel()

            if current_xlabel and "SHAP value" in current_xlabel:
                new_xlabel = current_xlabel.replace("SHAP value", "XWhy value")
                ax.set_xlabel(new_xlabel)

            if original_show:
                plt.show()

        return result

    return cast(F, wrapper)


def _ensure_2d(result: BaseXWhyResult, plot_name: str) -> None:
    """Ensure the result has multiple instances (2D)."""
    if result.coefficients.ndim < 2:
        raise ValueError(
            f"The '{plot_name}' plot requires a 2D matrix of "
            f"explanations (multiple instances). "
            f"It is not supported for single-instance explainers "
            f"like LLMExplainer."
        )


def _is_image_result(result: BaseXWhyResult) -> bool:
    """Report whether a result carries image structure."""
    return hasattr(result, "superpixels") or hasattr(result, "original_image")


# ==============================================================================
# LOCAL PLOTS (Support 1D Data)
# ==============================================================================


def bar(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Create a bar plot of a set of XWhy values."""
    return viz.bar(result.to_explanation(), **kwargs)


def waterfall(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Plot an explanation of a single prediction as a waterfall plot."""
    return viz.waterfall(result.to_explanation(), **kwargs)


def text(result: BaseXWhyResult, **kwargs: Any) -> str:  # noqa: ANN401
    """Plot a text explanation using coloured, self-contained HTML."""
    return viz.text(result.to_explanation(), **kwargs)


def force(result: BaseXWhyResult, **kwargs: Any) -> Figure | str | None:  # noqa: ANN401
    """Visualize the given XWhy values with an additive force layout."""
    # Note: base_value is deliberately omitted from arguments because
    # it is inherently encapsulated within the `result.to_explanation()` object.
    return viz.force(result.to_explanation(), **kwargs)


def decision(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Visualize model decisions using cumulative XWhy values."""
    # The decision plot works off raw arrays rather than an Explanation, so the
    # result is unpacked here.
    features = result.data if result.data is not None else None
    feature_names = (
        list(result.feature_names) if result.feature_names is not None else None
    )

    return viz.decision(
        base_value=float(result.base_values)
        if isinstance(result.base_values, float)
        else result.base_values,
        shap_values=result.coefficients,
        features=features,
        feature_names=feature_names,
        **kwargs,
    )


# ==============================================================================
# SUMMARY & GLOBAL PLOTS (Require 2D Data)
# ==============================================================================


def scatter(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Create a dependence scatter plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "scatter")
    return viz.scatter(result.to_explanation(), **kwargs)


def heatmap(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Create a heatmap plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "heatmap")
    return viz.heatmap(result.to_explanation(), **kwargs)


def beeswarm(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Create a beeswarm plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "beeswarm")
    return viz.beeswarm(result.to_explanation(), **kwargs)


def violin(result: BaseXWhyResult, **kwargs: Any) -> PlotResult:  # noqa: ANN401
    """Create a violin plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "violin")
    return viz.violin(result.to_explanation(), **kwargs)


def embedding(
    ind: Any,  # noqa: ANN401
    result: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> PlotResult:
    """Use the XWhy values as an embedding projected to 2D (requires 2D data)."""
    _ensure_2d(result, "embedding")
    return viz.embedding(ind, result.to_explanation(), **kwargs)


def group_difference(
    result: BaseXWhyResult,
    group_mask: np.ndarray,
    **kwargs: Any,  # noqa: ANN401
) -> PlotResult:
    """Plot the difference in mean XWhy values between two groups (2D data)."""
    _ensure_2d(result, "group_difference")
    # group_mask is a required boolean array indicating group membership
    return viz.group_difference(result.to_explanation(), group_mask, **kwargs)


def monitoring(
    ind: Any,  # noqa: ANN401
    result: BaseXWhyResult,
    features: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> PlotResult:
    """Create a monitoring plot over time or indices (requires 2D data)."""
    _ensure_2d(result, "monitoring")
    return viz.monitoring(ind, result.to_explanation(), features, **kwargs)


# ==============================================================================
# Multimodal or CV (Require 3D/4D image arrays)
# ==============================================================================


def image(
    result: BaseXWhyResult,
    pixel_values: Any = None,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Plot XWhy values for image inputs."""
    if result.coefficients.ndim < 3 and not _is_image_result(result):
        raise ValueError(
            "The 'image' plot requires image-structured explanations "
            "(3D or 4D arrays) or a result containing superpixels. "
            "It is not supported for 1D text explanations from LLMExplainer."
        )

    explanation = result.to_explanation()

    if pixel_values is not None:
        pixel_values = np.asarray(pixel_values)
        if pixel_values.ndim == 3:  # (H, W, C)
            pixel_values = np.expand_dims(pixel_values, axis=0)  # (1, H, W, C)
        elif pixel_values.ndim == 2:  # (H, W)
            pixel_values = np.expand_dims(pixel_values, axis=0)  # (1, H, W)

    return viz.image(explanation, pixel_values=pixel_values, **kwargs)


def image_to_text(
    result: BaseXWhyResult,
    **kwargs: Any,  # noqa: ANN401
) -> Figure | None:
    """Plot XWhy values for image inputs with text outputs.

    Requires multimodal data.
    """
    if result.coefficients.ndim < 3 and not _is_image_result(result):
        raise ValueError(
            "The 'image_to_text' plot requires multimodal "
            "image-to-text explanations (3D+ arrays) or a result containing "
            "superpixels. It is not supported for 1D text explanations from "
            "LLMExplainer."
        )

    explanation = result.to_explanation()

    if explanation.values.ndim < 5:
        raise ValueError(
            "The 'image_to_text' plot is designed for multimodal text generation "
            "models (e.g., Image captioning) and requires 5D explanations. "
            "For Image Classification models, please use `xwhy.plots.image()` instead."
        )

    return viz.image_to_text(explanation, **kwargs)


# ==============================================================================
# MODEL INSPECTION
# ==============================================================================


def partial_dependence(
    ind: Any,  # noqa: ANN401
    model: Any,  # noqa: ANN401
    data: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> PlotResult:
    """Plot the partial dependence of a model on a single feature."""
    return viz.partial_dependence(ind, model, data, **kwargs)


def initjs() -> None:
    """Do nothing; kept so SHAP-style notebooks keep running unchanged.

    XWhy renders its text and force plots as static HTML, so there is no
    JavaScript bundle to initialise.
    """
    viz.initjs()
