"""Plotting interface for XWhy results and SHAP visualizations."""

from functools import singledispatch
from typing import Any

import numpy as np
import shap

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.plots.factory import TextPlotterFactory
from xwhy.plots.types import TextPlotterType


@singledispatch
def text_heatmap(result: BaseXWhyResult, **kwargs: object) -> None:
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
def _text_heatmap(result: TextXWhyResult, **kwargs: object) -> None:
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


# ==============================================================================
# SHAP WRAPPER PLOTS
# ==============================================================================


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


def bar(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a bar plot of a set of SHAP values."""
    shap.plots.bar(result.to_shap(), **kwargs)


def waterfall(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot an explanation of a single prediction as a waterfall plot."""
    shap.plots.waterfall(result.to_shap(), **kwargs)


def text(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot a text explanation using coloring and interactive labels."""
    shap.plots.text(result.to_shap(), **kwargs)


def force(result: BaseXWhyResult, **kwargs: Any) -> Any:  # noqa: ANN401
    """Visualize the given SHAP values with an additive force layout."""
    # Note: base_value is deliberately omitted from arguments because
    # it is inherently encapsulated within the `result.to_shap()` Explanation object.
    return shap.plots.force(result.to_shap(), **kwargs)


def decision(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Visualize model decisions using cumulative SHAP values."""
    # Note: shap.plots.decision does NOT currently support the new Explanation object.
    # We must unpack and pass the raw numpy arrays (old SHAP API).
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


def scatter(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP dependence scatter plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "scatter")
    shap.plots.scatter(result.to_shap(), **kwargs)


def heatmap(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP heatmap plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "heatmap")
    shap.plots.heatmap(result.to_shap(), **kwargs)


def beeswarm(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP beeswarm plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "beeswarm")
    shap.plots.beeswarm(result.to_shap(), **kwargs)


def violin(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP violin plot (requires multiple instances/2D data)."""
    _ensure_2d(result, "violin")
    shap.plots.violin(result.to_shap(), **kwargs)


def embedding(ind: Any, result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Use the SHAP values as an embedding projected to 2D (requires 2D data)."""
    _ensure_2d(result, "embedding")
    shap.plots.embedding(ind, result.to_shap(), **kwargs)


def group_difference(
    result: BaseXWhyResult,
    group_mask: np.ndarray,
    **kwargs: Any,  # noqa: ANN401
) -> None:
    """Plot the difference in mean SHAP values between two groups (requires 2D data)."""
    _ensure_2d(result, "group_difference")
    # group_mask is a required boolean array indicating group membership
    shap.plots.group_difference(result.to_shap(), group_mask, **kwargs)


def monitoring(ind: Any, result: BaseXWhyResult, features: Any, **kwargs: Any) -> None:  # noqa: ANN401
    """Create a SHAP monitoring plot over time or indices (requires 2D data)."""
    _ensure_2d(result, "monitoring")
    shap.plots.monitoring(ind, result.to_shap(), features, **kwargs)


# ==============================================================================
# Multimodal or CV (Require 3D/4D image arrays)
# ==============================================================================


def image(result: BaseXWhyResult, pixel_values: Any = None, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot SHAP values for image inputs (requires 3D/4D image arrays)."""
    if result.coefficients.ndim < 3:
        raise ValueError(
            "The 'image' plot requires image-structured "
            "explanations (3D or 4D arrays). "
            "It is not supported for 1D text explanations from LLMExplainer."
        )
    shap.plots.image(result.to_shap(), pixel_values, **kwargs)


def image_to_text(result: BaseXWhyResult, **kwargs: Any) -> None:  # noqa: ANN401
    """Plot SHAP values for image inputs with text outputs.

    Requires multimodal data.
    """
    if result.coefficients.ndim < 3:
        raise ValueError(
            "The 'image_to_text' plot requires multimodal "
            "image-to-text explanations (3D+ arrays). "
            "It is not supported for 1D text explanations from LLMExplainer."
        )
    shap.plots.image_to_text(result.to_shap(), **kwargs)


initjs = shap.plots.initjs
partial_dependence = shap.plots.partial_dependence
