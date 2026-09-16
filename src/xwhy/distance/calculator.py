"""Centralized router for calculating distances."""

from __future__ import annotations

from typing import Any

import numpy as np

from xwhy.distance.distances import (
    AndersonDarlingDistance,
    CosineDistance,
    CvMDistance,
    DTSDistance,
    KSDistance,
    KuiperDistance,
    WassersteinDistance,
)
from xwhy.distance.types import DistanceType

_DISTANCE_MAP = {
    DistanceType.COSINE: CosineDistance,
    DistanceType.WASSERSTEIN: WassersteinDistance,
    DistanceType.KS: KSDistance,
    DistanceType.CRAMER_VON_MISES: CvMDistance,
    DistanceType.ANDERSON_DARLING: AndersonDarlingDistance,
    DistanceType.KUIPER: KuiperDistance,
    DistanceType.DTS: DTSDistance,
}


def calculate_distance(
    metric: str | DistanceType,
    source: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    return_p_value: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> float | tuple[float, float]:
    """Compute the distance between source and target arrays/texts.

    Includes automatic validation to ensure Text data only uses text metrics
    and Image/Tabular data uses numeric metrics.

    Args:
        metric: The distance metric to use.
        source: Source data (numpy array or string).
        target: Target data (numpy array or string).
        return_p_value: If True, calculates statistical significance using bootstrap.
            Returns (p_value, distance_value).
        **kwargs: Additional arguments passed to the underlying compute methods
            (e.g., `mode`, `n_bootstrap`).

    """
    metric_type = DistanceType.from_str(metric)

    # Convert PyTorch Tensors to NumPy arrays automatically if passed
    if hasattr(source, "detach"):
        source = source.detach().cpu().numpy()
    if hasattr(target, "detach"):
        target = target.detach().cpu().numpy()

    # Type verification
    is_source_text = isinstance(source, str)
    is_source_numeric = isinstance(source, np.ndarray)

    if not (is_source_text or is_source_numeric):
        raise TypeError("Source data must be either a string or a numpy array.")

    if type(source) is not type(target):
        raise TypeError("Source and target must be of the exact same data type.")

    # Dispatch
    distance_class = _DISTANCE_MAP[metric_type]
    calculator = distance_class()

    if return_p_value:
        return calculator.compute_with_p_value(source=source, target=target, **kwargs)

    return calculator.compute(source=source, target=target, **kwargs)
