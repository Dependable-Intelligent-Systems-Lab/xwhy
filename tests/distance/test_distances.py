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
    """Test distance computation across 1D, 2D-spatial, 3D, and mismatch cases."""
    dist = MockDistance()

    # Branch: 1D shape mismatch => inf
    assert dist.compute(np.array([1.0]), np.array([1.0, 2.0])) == float("inf")

    # Branch: Ndim == 1 success
    assert dist.compute(np.array([1.0]), np.array([2.0])) == 1.0

    # Branch: Ndim == 3 success (channel-wise)
    img1 = np.zeros((10, 10, 3), dtype=np.float64)
    img2 = np.zeros((10, 10, 3), dtype=np.float64)
    assert dist.compute(img1, img2) == 3.0

    # Branch: Ndim == 3 shape mismatch => inf
    img3 = np.zeros((8, 8, 3), dtype=np.float64)
    assert dist.compute(img1, img3) == float("inf")

    # Branch: 2D latent / fallback (default mode)
    arr1 = np.zeros((5, 5), dtype=np.float64)
    arr2 = np.zeros((5, 5), dtype=np.float64)
    assert dist.compute(arr1, arr2) == 1.0

    # Branch: 2D spatial mode success (feature dims match)
    pc1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    pc2 = np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float64)
    # 2 axes => _compute_1d called twice => 2.0
    assert dist.compute(pc1, pc2, mode="spatial") == 2.0

    # Branch: 2D spatial mode feature-dim mismatch => inf
    pc3 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    assert dist.compute(pc1, pc3, mode="spatial") == float("inf")


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


def test_compute_tensor_conversion() -> None:
    """Verify PyTorch-like tensors are converted before dimensionality checks.

    Covers both source-only and target-only conversion branches as well as
    the combined case.
    """
    dist = MockDistance()

    def _make_tensor(arr: np.ndarray) -> MagicMock:
        """Create a mock that behaves like a torch.Tensor."""
        tensor = MagicMock()
        tensor.detach.return_value.cpu.return_value.numpy.return_value = arr
        return tensor

    src_arr = np.array([1.0, 2.0], dtype=np.float64)
    tgt_arr = np.array([3.0, 4.0], dtype=np.float64)

    # Source is tensor-like
    assert dist.compute(_make_tensor(src_arr), tgt_arr) == 1.0

    # Target is tensor-like
    assert dist.compute(src_arr, _make_tensor(tgt_arr)) == 1.0

    # Both are tensor-like
    assert dist.compute(_make_tensor(src_arr), _make_tensor(tgt_arr)) == 1.0


def test_compute_higher_ndim_fallback() -> None:
    """Verify N-D arrays (ndim > 3) fall through to the flatten path."""
    dist = MockDistance()
    vol1 = np.zeros((2, 3, 4, 5), dtype=np.float64)
    vol2 = np.zeros((2, 3, 4, 5), dtype=np.float64)
    assert dist.compute(vol1, vol2) == 1.0


def test_wmd_distance_requires_keyed_vectors() -> None:
    """Raise ValueError when model is missing or not KeyedVectors."""
    from xwhy.distance.distances import WMDDistance

    wmd = WMDDistance()
    with pytest.raises(ValueError, match="KeyedVectors"):
        wmd.compute("hello world", "foo bar")

    with pytest.raises(ValueError, match="KeyedVectors"):
        wmd.compute("hello world", "foo bar", model="not-a-model")


def test_wmd_distance_empty_vocab_returns_one() -> None:
    """Return 1.0 when no words remain after vocab filtering."""
    from gensim.models import KeyedVectors

    from xwhy.distance.distances import WMDDistance

    # Must pass isinstance(..., KeyedVectors); membership always False
    model = MagicMock(spec=KeyedVectors)
    model.__contains__ = MagicMock(return_value=False)

    wmd = WMDDistance()
    result = wmd.compute("hello world", "foo bar", model=model)
    assert result == 1.0


def test_wmd_distance_success_path() -> None:
    """Call model.wmdistance when both sides have in-vocab words."""
    from gensim.models import KeyedVectors

    from xwhy.distance.distances import WMDDistance

    model = MagicMock(spec=KeyedVectors)
    model.__contains__ = MagicMock(return_value=True)
    model.wmdistance.return_value = 0.42

    wmd = WMDDistance()
    result = wmd.compute("hello world", "foo bar", model=model)

    assert result == 0.42
    model.wmdistance.assert_called_once()
