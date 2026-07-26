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
from xwhy.distance.wmd import WMDDistance

_DISTANCE_MAP = {
    DistanceType.COSINE: CosineDistance,
    DistanceType.WASSERSTEIN: WassersteinDistance,
    DistanceType.KS: KSDistance,
    DistanceType.CRAMER_VON_MISES: CvMDistance,
    DistanceType.ANDERSON_DARLING: AndersonDarlingDistance,
    DistanceType.KUIPER: KuiperDistance,
    DistanceType.WMD: WMDDistance,
    DistanceType.DTS: DTSDistance,
}


def calculate_distance(
    metric: str | DistanceType,
    source: Any,  # noqa: ANN401
    target: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> float:
    """Compute the distance between source and target arrays/texts.

    Includes automatic validation to ensure Text data only uses text metrics
    and Image/Tabular data uses numeric metrics.
    """
    metric_type = DistanceType.from_str(metric)

    # Type verification
    is_source_text = isinstance(source, str)
    is_source_numeric = isinstance(source, np.ndarray)

    if not (is_source_text or is_source_numeric):
        raise TypeError("Source data must be either a string or a numpy array.")

    if type(source) is not type(target):
        raise TypeError("Source and target must be of the exact same data type.")

    # Metric compatibility validation
    if is_source_text and not metric_type.is_text_metric:
        raise ValueError(
            f"Text data requires a text-based metric like WMD. "
            f"Received: {metric_type.value}"
        )

    if is_source_numeric and not metric_type.is_numeric_metric:
        raise ValueError(
            f"Numerical data (e.g., images) cannot use text-based metrics. "
            f"Received: {metric_type.value}"
        )

    # Dispatch
    distance_class = _DISTANCE_MAP[metric_type]
    calculator = distance_class()  # type: ignore[abstract]
    return calculator.compute(source=source, target=target, **kwargs)
