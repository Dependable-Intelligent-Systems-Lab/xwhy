"""Unit tests for Unified statistical distance metrics (custom implementations)."""

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

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


def test_compute_with_p_value_branches() -> None:
    """Test the shared bootstrap p-value computation branches."""
    dist = MockDistance()
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])

    # Branch 1: n_bootstrap = 0 (should safely return 0.0 for p_value)
    p_val_0, dist_val_0 = dist.compute_with_p_value(a, b, n_bootstrap=0)
    assert p_val_0 == 0.0
    assert dist_val_0 == 1.0

    # Branch 2: boost_dist > dist_val
    # We mock _compute_1d so that the bootstrap iterations return a higher
    # distance (2.0)
    dist._compute_1d = MagicMock(side_effect=[1.0, 2.0, 2.0, 2.0])  # type: ignore[method-assign]
    p_val_high, dist_val_high = dist.compute_with_p_value(a, b, n_bootstrap=3)
    # n_bootstrap=3 loops 2 times (range(1, 3)). Both loops return 2.0 > 1.0
    # bigger = 2 -> p_value = 2 / 3
    assert p_val_high == (2 / 3)
    assert dist_val_high == 1.0

    # Branch 3: boost_dist <= dist_val
    # We mock _compute_1d so bootstrap iterations return lower or equal distance (1.0)
    dist._compute_1d = MagicMock(side_effect=[2.0, 1.0, 1.0, 1.0])  # type: ignore[method-assign]
    p_val_low, dist_val_low = dist.compute_with_p_value(a, b, n_bootstrap=3)
    # None of the bootstraps are strictly greater than 2.0
    assert p_val_low == 0.0
    assert dist_val_low == 2.0


def test_specific_metrics_execution_and_branches() -> None:
    """Verify that all concrete distance implementations execute correctly.

    The arrays are carefully designed to hit all internal mathematical branches:
    - Duplicates (e.g., 1.0, 1.0): Hits `xy_sorted[i+1] == xy_sorted[i]`
      False branches.
    - Unique jumps (e.g., 3.0 to 4.0): Hits `xy_sorted[i+1] != xy_sorted[i]`
      True branches.
    - Crossing CDFs: Makes `height` fluctuate between positive and negative to hit
      `height > up` and `height < down` in Kuiper distance.
    """
    a = np.array([1.0, 1.0, 3.0, 4.0, 6.0])
    b = np.array([1.0, 2.0, 2.0, 4.0, 5.0])

    assert isinstance(CosineDistance().compute(a, b), float)
    assert isinstance(WassersteinDistance().compute(a, b), float)
    assert isinstance(KSDistance().compute(a, b), float)
    assert isinstance(CvMDistance().compute(a, b), float)
    assert isinstance(AndersonDarlingDistance().compute(a, b), float)
    assert isinstance(KuiperDistance().compute(a, b), float)
    assert isinstance(DTSDistance().compute(a, b), float)


def test_dts_distance_logic() -> None:
    """Verify that DTSDistance calculates successfully and returns a positive float.

    Replaces the old 'DTS == AD + CVM' test because the custom DTS script
    uses a unique formula involving `width` and `sd` that is not a simple sum.
    """
    a = np.array([1.1, 2.2, 3.3, 4.4, 5.5])
    b = np.array([1.2, 2.3, 3.4, 4.5, 5.6])

    # Compute using DTS
    dts_val = DTSDistance().compute(a, b)

    # Distance should be a non-negative float mathematically
    assert isinstance(dts_val, float)
    assert dts_val >= 0.0


def test_kuiper_and_ks_extreme_branches() -> None:
    """Force specific boundary conditions for KS and Kuiper."""
    # Force a case where f_cdf is always >= e_cdf to ensure `up` / `down`
    # branches receive extreme updates.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    assert isinstance(KuiperDistance().compute(a, b), float)
    assert isinstance(KSDistance().compute(a, b), float)


def test_base_compute_1d_raises() -> None:
    """Calling _compute_1d on the abstract base must raise."""
    base = BaseNumericDistance()
    with pytest.raises(NotImplementedError):
        base._compute_1d(np.array([1.0]), np.array([2.0]))
