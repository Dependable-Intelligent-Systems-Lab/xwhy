"""Unified statistical distance metrics for Numerical (Image/Tabular) data."""

from __future__ import annotations

import random
from typing import Any, cast

import numpy as np
from scipy.spatial.distance import cosine

from xwhy.distance.base import BaseDistance
from xwhy.logger import logger


class BaseNumericDistance(BaseDistance):
    """Base class for handling dimensionality of numerical distances.

    Automatically handles 1D arrays (Tabular/Embeddings) and 3D arrays (Images)
    by computing channel-wise distances and aggregating them.
    """

    def _prepare_ecdf_data(
        self, a: np.ndarray, b: np.ndarray
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare common Empirical CDF data for distance calculations (DRY).

        Args:
            a: First array.
            b: Second array.

        Returns:
            tuple: Total length, sorted combined array, sorted X weights,
                   sorted Y weights.

        """
        nx = len(a)
        ny = len(b)
        n = nx + ny

        xy_combined = np.concatenate([a, b])
        x_weights = np.concatenate([np.repeat(1 / nx, nx), np.repeat(0, ny)])
        y_weights = np.concatenate([np.repeat(0, nx), np.repeat(1 / ny, ny)])

        sort_indices = np.argsort(xy_combined)
        xy_sorted = xy_combined[sort_indices]
        x2_sorted = x_weights[sort_indices]
        y2_sorted = y_weights[sort_indices]

        return n, xy_sorted, x2_sorted, y2_sorted

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two 1D arrays. Must be implemented by subclasses."""
        raise NotImplementedError

    def compute(
        self,
        source: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,  # noqa: ANN401
    ) -> float:
        """Compute distance robustly regardless of array dimensionality."""
        if source.shape != target.shape:
            logger.warning(f"Shape mismatch: {source.shape} vs {target.shape}")
            return float("inf")

        # Case 1: 1D Array (Tabular Data or Embedding Vector)
        if source.ndim == 1:
            return self._compute_1d(source, target)

        # Case 2: 3D Image (H, W, C) - Channel-wise computation
        elif source.ndim == 3:
            dist_total = 0.0
            channels = source.shape[2]
            for i in range(channels):
                hist1 = source[:, :, i].flatten()
                hist2 = target[:, :, i].flatten()
                dist_total += self._compute_1d(hist1, hist2)
            return dist_total

        # Case 3: 2D or General N-D Fallback (Flatten all)
        else:
            return self._compute_1d(source.flatten(), target.flatten())

    def compute_with_p_value(
        self, source: np.ndarray, target: np.ndarray, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """Compute distance with bootstrap-based p-value.

        Inherited by all numerical distance metrics.
        Uses bootstrap sampling to compute statistical significance.

        Args:
            source (np.ndarray): First sample.
            target (np.ndarray): Second sample.
            n_bootstrap (int): Number of bootstrap iterations. Default is 1000.

        Returns:
            tuple: (p_value, distance_value)

        """
        dist_val = self.compute(source, target)

        na = len(source)
        nb = len(target)
        n = na + nb
        combined = np.concatenate([source, target])
        bigger = 0

        for _ in range(1, n_bootstrap):
            idx_a = random.sample(range(n), na)
            idx_b = random.sample(range(n), nb)
            # Direct calculation on 1D subsets for efficiency
            boost_dist = self._compute_1d(combined[idx_a], combined[idx_b])
            if boost_dist > dist_val:
                bigger += 1

        p_value = bigger / n_bootstrap if n_bootstrap > 0 else 0.0
        return p_value, dist_val


class CosineDistance(BaseNumericDistance):
    """Cosine distance metric."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        return cast(float, cosine(a, b))


class AndersonDarlingDistance(BaseNumericDistance):
    """Anderson-Darling distance metric (Custom Implementation)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        res = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        g_cdf = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]
            g_cdf += 1 / n
            sd = (n * g_cdf * (1 - g_cdf)) ** 0.5
            height = abs(f_cdf - e_cdf)

            if xy_sorted[i + 1] != xy_sorted[i] and sd > 0:
                res += (height / sd) ** power

        return float(res)


class CvMDistance(BaseNumericDistance):
    """Cramer-Von Mises distance metric (Custom Implementation)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        res = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]
            height = abs(f_cdf - e_cdf)

            if xy_sorted[i + 1] != xy_sorted[i]:
                res += height**power

        return float(res)


class DTSDistance(BaseNumericDistance):
    """DTS distance metric (Custom Implementation: Combination of AD and CVM)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        res = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        g_cdf = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]
            g_cdf += 1 / n
            sd = (n * g_cdf * (1 - g_cdf)) ** 0.5
            height = abs(f_cdf - e_cdf)
            width = xy_sorted[i + 1] - xy_sorted[i]

            res += ((height / sd) ** power) * width

        return float(res)


class KSDistance(BaseNumericDistance):
    """Kolmogorov-Smirnov distance metric (Custom Implementation)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        res = 0.0
        height = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]

            if xy_sorted[i + 1] != xy_sorted[i]:
                height = abs(f_cdf - e_cdf)
            if height > res:
                res = height

        return float(res**power)


class KuiperDistance(BaseNumericDistance):
    """Kuiper distance metric (Custom Implementation)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        up = 0.0
        down = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        height = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]

            if xy_sorted[i + 1] != xy_sorted[i]:
                height = f_cdf - e_cdf
            if height > up:
                up = height
            if height < down:
                down = height

        return float(abs(down) ** power + abs(up) ** power)


class WassersteinDistance(BaseNumericDistance):
    """Wasserstein distance metric (Custom Implementation)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        n, xy_sorted, x2_sorted, y2_sorted = self._prepare_ecdf_data(a, b)

        res = 0.0
        e_cdf = 0.0
        f_cdf = 0.0
        power = 1

        for i in range(n - 2):
            e_cdf += x2_sorted[i]
            f_cdf += y2_sorted[i]
            height = abs(f_cdf - e_cdf)
            width = xy_sorted[i + 1] - xy_sorted[i]
            res += (height**power) * width

        return float(res)
