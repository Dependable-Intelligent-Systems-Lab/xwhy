"""Test for Unified statistical distance metrics."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np

from xwhy.distance.distances import (
    AndersonDarlingDistance,
    BaseNumericDistance,
    CosineDistance,
    CvMDistance,
    DTSDistance,
    KSDistance,
    KuiperDistance,
    WassersteinDistance,
)


class MockDistance(BaseNumericDistance):
    """Mock implementation for testing BaseNumericDistance logic."""

    def _compute_1d(self, a: Any, b: Any) -> float:  # noqa: ANN401
        return 1.0


def test_extract_statistic() -> None:
    """Verify statistic extraction from float, tuple, and objects."""
    dist = MockDistance()
    # Test float
    assert dist._extract_statistic(1.5) == 1.5
    # Test tuple
    assert dist._extract_statistic((0.5, 0.1)) == 0.5
    # Test object with .statistic
    obj = MagicMock()
    obj.statistic = 2.5
    assert dist._extract_statistic(obj) == 2.5


def test_compute_dimensionality_branches() -> None:
    """Test distance computation across 1D, 3D, and mismatch dimensions."""
    dist = MockDistance()

    # Branch: Shape mismatch
    assert dist.compute(np.array([1]), np.array([1, 2])) == float("inf")

    # Branch: Ndim == 1
    assert dist.compute(np.array([1]), np.array([2])) == 1.0

    # Branch: Ndim == 3 (Channels)
    img1 = np.zeros((10, 10, 3))
    img2 = np.zeros((10, 10, 3))
    # Returns 1.0 per channel (3 channels) -> 3.0
    assert dist.compute(img1, img2) == 3.0

    # Branch: Fallback (2D)
    arr1 = np.zeros((5, 5))
    arr2 = np.zeros((5, 5))
    assert dist.compute(arr1, arr2) == 1.0


def test_specific_metrics_execution() -> None:
    """Verify that all concrete distance implementations execute correctly."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 3.0, 4.0])

    assert isinstance(CosineDistance().compute(a, b), float)
    assert isinstance(WassersteinDistance().compute(a, b), float)
    assert isinstance(KSDistance().compute(a, b), float)
    assert isinstance(CvMDistance().compute(a, b), float)
    assert isinstance(AndersonDarlingDistance().compute(a, b), float)
    assert isinstance(KuiperDistance().compute(a, b), float)
    assert isinstance(DTSDistance().compute(a, b), float)


def test_wasserstein_p_value() -> None:
    """Verify Wasserstein p-value calculation logic."""
    dist = WassersteinDistance()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 2.1, 3.1])
    p, wd = dist.compute_with_p_value(a, b, n_bootstrap=10)
    assert 0 <= p <= 1
    assert wd >= 0


def test_dts_distance_combination() -> None:
    """Verify that DTSDistance correctly sums Anderson-Darling and CvM statistics."""
    a = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    b = np.array([1.2, 2.3, 3.4, 4.5, 5.6])

    # Compute individually
    ad_val = AndersonDarlingDistance().compute(a, b)
    cvm_val = CvMDistance().compute(a, b)

    # Compute using DTS
    dts_val = DTSDistance().compute(a, b)

    # Check if DTS is exactly the sum of AD and CvM (using np.isclose for float safety)
    assert np.isclose(dts_val, ad_val + cvm_val)
