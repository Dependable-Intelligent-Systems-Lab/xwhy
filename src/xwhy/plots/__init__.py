"""Visualization plotting functions for XWhy results."""

from functools import singledispatch

from xwhy.core.result import BaseXWhyResult, TextXWhyResult
from xwhy.visualization.factory import TextVisualizerFactory
from xwhy.visualization.types import TextVisualizerType


@singledispatch
def heatmap(result: BaseXWhyResult, **kwargs: object) -> None:
    """Plot a heatmap visualization for the given explanation result.

    This function automatically delegates plotting to the appropriate visualizer
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
        **kwargs: Can include 'title' (str) and 'backend' (TextVisualizerType or str),
            as well as any other arguments passed to the visualizer plot method.

    Raises:
        ValueError: If an unsupported backend type is provided.

    """
    title = str(kwargs.pop("title", "Text Heatmap"))
    backend_kwarg = kwargs.pop("backend", TextVisualizerType.NATIVE_HEATMAP)

    if isinstance(backend_kwarg, TextVisualizerType):
        backend = backend_kwarg
    elif isinstance(backend_kwarg, str):
        backend = TextVisualizerType(backend_kwarg)
    else:
        msg = f"Unsupported backend type: {type(backend_kwarg).__name__}"
        raise ValueError(msg)

    visualizer = TextVisualizerFactory.create(method=backend)
    visualizer.plot(
        words=result.words,
        scores=result.coefficients,
        title=title,
        **kwargs,
    )
