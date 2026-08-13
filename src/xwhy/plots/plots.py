"""Plotting interface for XWhy results and SHAP visualizations."""

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

from xwhy.core.result import (
    BaseXWhyResult,
    ImageGenerationAndEditingXWhyResult,
    TextXWhyResult,
)
from xwhy.logger import logger
from xwhy.plots.factory import TextPlotterFactory
from xwhy.plots.image import image_heatmap, plot_image  # noqa: F401
from xwhy.plots.tabular import (
    plot_dataset,  # noqa: F401
    plot_explanation_waterfall,  # noqa: F401
    plot_feature_contributions,  # noqa: F401
    plot_method_contributions,  # noqa: F401
)
from xwhy.plots.types import TextPlotterType

F = TypeVar("F", bound=Callable[..., Any])


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
# SHAP WRAPPER PLOTS
# ==============================================================================


def replace_shap_label[F: Callable[..., Any]](plot_func: F) -> F:
    """Replace 'SHAP value' with 'XWhy value' in SHAP plots.

    Detect dynamically if the underlying SHAP plot function accepts a
    'show' parameter upon execution. If it does, suppress immediate
    rendering, update the matplotlib axis/colorbar labels, and honor the
    original 'show' argument state. Otherwise, pass arguments untampered.

    Args:
        plot_func: The plotting function to be decorated.

    Returns:
        F: The wrapped function with delayed SHAP initialization.

    """

    @functools.wraps(plot_func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        import shap  # Lazy import to prevent initialization errors

        shap_func_name = plot_func.__name__
        shap_func = getattr(shap.plots, shap_func_name, None)

        # 1. Check if the function is mocked (for pytest compatibility)
        if shap_func is not None and hasattr(shap_func, "called"):
            return plot_func(*args, **kwargs)

        # 2. Inspect signature dynamically at execution time
        accepts_show = False
        if shap_func is not None:
            try:
                sig = inspect.signature(shap_func)
                accepts_show = "show" in sig.parameters
            except (ValueError, TypeError):
                accepts_show = False

        # 3. Apply the rendering logic
        if accepts_show:
            original_show = kwargs.get("show", True)
            kwargs["show"] = False

            result = plot_func(*args, **kwargs)

            # Check all axes in the current figure (includes subplots and colorbars)
            if len(plt.get_fignums()) > 0:
                for ax in plt.gcf().get_axes():
                    # Check and replace X-axis label (used in standard plots)
                    current_xlabel = ax.get_xlabel()
                    if current_xlabel and "SHAP value" in current_xlabel:
                        new_xlabel = current_xlabel.replace("SHAP value", "XWhy value")
                        ax.set_xlabel(new_xlabel)

                    # Check and replace Y-axis label (or colorbar axis
                    # label if oriented)
                    current_ylabel = ax.get_ylabel()
                    if current_ylabel and "SHAP value" in current_ylabel:
                        new_ylabel = current_ylabel.replace("SHAP value", "XWhy value")
                        ax.set_ylabel(new_ylabel)

            if original_show:
                plt.show()

            return result

        return plot_func(*args, **kwargs)

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


# ==============================================================================
# LOCAL PLOTS (Support 1D Data)
# ==============================================================================


@replace_shap_label
def bar(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a bar plot of a set of SHAP values."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    shap.plots.bar(result.to_shap(), **kwargs)


@replace_shap_label
def waterfall(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot an explanation of a single prediction as a waterfall plot."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    shap.plots.waterfall(result.to_shap(), **kwargs)


@replace_shap_label
def text(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot a text explanation using coloring and interactive labels."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    shap.plots.text(result.to_shap(), **kwargs)


@replace_shap_label
def force(result: BaseXWhyResult, **kwargs: Any) -> Any:  # noqa: ANN401
    """Visualize the given SHAP values with an additive force layout."""
    # Note: base_value is deliberately omitted from arguments because
    # it is inherently encapsulated within the `result.to_shap()` Explanation object.
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    return shap.plots.force(result.to_shap(), **kwargs)


@replace_shap_label
def decision(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Visualize model decisions using cumulative SHAP values."""
    # Note: shap.plots.decision does NOT currently support the new Explanation object.
    # We must unpack and pass the raw numpy arrays (old SHAP API).
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    features = result.data if result.data is not None else None
    feature_names = (
        list(result.feature_names) if result.feature_names is not None else None
    )

    shap.plots.decision(
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


@replace_shap_label
def scatter(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP dependence scatter plot (requires multiple instances/2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "scatter")
    shap.plots.scatter(result.to_shap(), **kwargs)


@replace_shap_label
def heatmap(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP heatmap plot (requires multiple instances/2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "heatmap")
    shap.plots.heatmap(result.to_shap(), **kwargs)


@replace_shap_label
def beeswarm(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP beeswarm plot (requires multiple instances/2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "beeswarm")
    shap.plots.beeswarm(result.to_shap(), **kwargs)


@replace_shap_label
def violin(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP violin plot (requires multiple instances/2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "violin")
    shap.plots.violin(result.to_shap(), **kwargs)


@replace_shap_label
def embedding(ind: Any, result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Use the SHAP values as an embedding projected to 2D (requires 2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "embedding")
    shap.plots.embedding(ind, result.to_shap(), **kwargs)


@replace_shap_label
def group_difference(
    result: BaseXWhyResult,
    group_mask: np.ndarray,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot the difference in mean SHAP values between two groups (requires 2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "group_difference")
    # group_mask is a required boolean array indicating group membership
    shap.plots.group_difference(result.to_shap(), group_mask, **kwargs)


@replace_shap_label
def monitoring(ind: Any, result: BaseXWhyResult, features: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP monitoring plot over time or indices (requires 2D data)."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    _ensure_2d(result, "monitoring")
    shap.plots.monitoring(ind, result.to_shap(), features, **kwargs)


# ==============================================================================
# Multimodal or CV (Require 3D/4D image arrays)
# ==============================================================================


@replace_shap_label
def image(result: BaseXWhyResult, pixel_values: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot SHAP values for image inputs."""
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    is_image_result = hasattr(result, "superpixels") or hasattr(
        result, "original_image"
    )

    if result.coefficients.ndim < 3 and not is_image_result:
        raise ValueError(
            "The 'image' plot requires image-structured explanations "
            "(3D or 4D arrays) or a result containing superpixels. "
            "It is not supported for 1D text explanations from LLMExplainer."
        )

    shap_obj = result.to_shap()

    if pixel_values is not None:
        pixel_values = np.asarray(pixel_values)
        if pixel_values.ndim == 3:  # (H, W, C)
            pixel_values = np.expand_dims(pixel_values, axis=0)  # (1, H, W, C)
        elif pixel_values.ndim == 2:  # (H, W)
            pixel_values = np.expand_dims(pixel_values, axis=0)  # (1, H, W)

    shap.plots.image(shap_obj, pixel_values=pixel_values, **kwargs)


@replace_shap_label
def image_to_text(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot SHAP values for image inputs with text outputs.

    Requires multimodal data.
    """
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    is_image_result = hasattr(result, "superpixels") or hasattr(
        result, "original_image"
    )

    if result.coefficients.ndim < 3 and not is_image_result:
        raise ValueError(
            "The 'image_to_text' plot requires multimodal "
            "image-to-text explanations (3D+ arrays) or a result containing "
            "superpixels. It is not supported for 1D text explanations from "
            "LLMExplainer."
        )

    shap_obj = result.to_shap()

    if shap_obj.values.ndim < 5:  # type: ignore[attr-defined]
        raise ValueError(
            "The 'image_to_text' plot is designed for multimodal text generation "
            "models (e.g., Image captioning) and requires 5D explanations. "
            "For Image Classification models, please use `xwhy.plots.image()` instead."
        )

    shap.plots.image_to_text(shap_obj, **kwargs)


def initjs() -> None:
    """Initialize JavaScript dependencies for SHAP visualizations.

    Load the required JavaScript libraries for interactive SHAP plots
    within a Jupyter notebook environment.
    """
    logger.debug("Initializing SHAP JavaScript dependencies.")
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    shap.plots.initjs()


def partial_dependence(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Generate a partial dependence plot using SHAP.

    Pass all positional and keyword arguments directly to the underlying
    SHAP partial dependence plotting function.

    Args:
        *args: Positional arguments for the SHAP plotting function.
        **kwargs: Keyword arguments for the SHAP plotting function.

    Returns:
        Any: The resulting SHAP plot object or matplotlib figure.

    """
    logger.debug("Executing SHAP partial dependence plot.")
    import shap  # Lazy import to prevent numba/llvmlite initialization errors

    return shap.plots.partial_dependence(*args, **kwargs)
