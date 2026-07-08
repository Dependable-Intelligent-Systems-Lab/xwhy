"""Visualization plotting functions for XWhy results."""

from functools import singledispatch

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.plots.base import BaseTextPlotter
from xwhy.plots.factory import TextPlotterFactory
from xwhy.plots.text import NativeHeatmapPlotter
from xwhy.plots.types import TextPlotterType

__all__ = [
    "BaseTextPlotter",
    "NativeHeatmapPlotter",
    "TextPlotterFactory",
    "TextPlotterType",
]


@singledispatch
def heatmap(result: BaseXWhyResult, **kwargs: object) -> None:
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


@heatmap.register
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
