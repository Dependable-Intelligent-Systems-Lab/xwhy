"""Unified statistical distance metrics for Numerical (Image/Tabular) data."""

from __future__ import annotations

import random
import warnings
from typing import Any, cast

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import (
    anderson_ksamp,
    cramervonmises_2samp,
    ks_2samp,
    wasserstein_distance,
)

from xwhy.distance.base import BaseDistance
from xwhy.logger import logger


class BaseNumericDistance(BaseDistance):
    """Base class for handling dimensionality of numerical distances.

    Automatically handles 1D arrays (Tabular/Embeddings) and 3D arrays (Images)
    by computing channel-wise distances and aggregating them.
    """

    def _extract_statistic(self, result: Any) -> float:  # noqa: ANN401
        """Help to extract float value from various scipy result formats."""
        if isinstance(result, (float, int, np.number)):
            return float(result)
        elif isinstance(result, tuple):
            return float(result[0])
        else:
            return float(result.statistic)

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> Any:  # noqa: ANN401
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
            res = self._compute_1d(source, target)
            return self._extract_statistic(res)

        # Case 2: 3D Image (H, W, C) - Channel-wise computation
        elif source.ndim == 3:
            dist_total = 0.0
            channels = source.shape[2]
            for i in range(channels):
                hist1 = source[:, :, i].flatten()
                hist2 = target[:, :, i].flatten()
                res = self._compute_1d(hist1, hist2)
                dist_total += self._extract_statistic(res)
            return dist_total

        # Case 3: 2D or General N-D Fallback (Flatten all)
        else:
            res = self._compute_1d(source.flatten(), target.flatten())
            return self._extract_statistic(res)


class CosineDistance(BaseNumericDistance):
    """Cosine distance metric."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        return cast(float, cosine(a, b))


class WassersteinDistance(BaseNumericDistance):
    """Wasserstein distance metric (handles Image & Tabular)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        return cast(float, wasserstein_distance(a, b))

    def compute_with_p_value(
        self, source: np.ndarray, target: np.ndarray, n_bootstrap: int = 1000
    ) -> tuple[float, float]:
        """Compute Wasserstein distance with bootstrap-based p-value (Tabular spec).

        Args:
            source (np.ndarray): First sample.
            target (np.ndarray): Second sample.
            n_bootstrap (int): Number of bootstrap iterations.

        Returns:
            tuple: (p_value, wasserstein_distance)

        """
        wd = self.compute(source, target)

        na, nb = len(source), len(target)
        n = na + nb
        combined = np.concatenate([source, target])
        bigger = 0

        for _ in range(n_bootstrap):
            idx_x = random.sample(range(n), na)
            idx_y = random.sample(range(n), nb)
            wd_boot = wasserstein_distance(combined[idx_x], combined[idx_y])
            if wd_boot > wd:
                bigger += 1

        p_value = bigger / n_bootstrap if n_bootstrap > 0 else 0.0
        return p_value, wd


class KSDistance(BaseNumericDistance):
    """Kolmogorov-Smirnov (KS) distance metric."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> Any:  # noqa: ANN401
        return ks_2samp(a, b)


class CvMDistance(BaseNumericDistance):
    """Cramer-Von Mises (CvM) distance metric."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> Any:  # noqa: ANN401
        return cramervonmises_2samp(a, b)


class AndersonDarlingDistance(BaseNumericDistance):
    """Anderson-Darling (k-sample) distance metric."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> Any:  # noqa: ANN401
        # Omit the 'method' parameter to prevent Out-Of-Memory (OOM) errors
        # caused by PermutationMethod on large inputs (e.g., flattened images).
        # We use warnings.catch_warnings() to cleanly suppress SciPy's
        # "p-value floored/capped" UserWarning. This warning is irrelevant
        # for our XAI pipeline because we solely rely on the distance statistic
        # to weight the neighborhood, not the statistical significance (p-value).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return anderson_ksamp([a, b], variant="midrank")


class DTSDistance(BaseNumericDistance):
    """DTS distance metric (Combination of Anderson-Darling and Cramer-Von Mises)."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute the combined DTS distance using AD and CvM statistics.

        Args:
            a (np.ndarray): First sample.
            b (np.ndarray): Second sample.

        Returns:
            float: The sum of the Anderson-Darling and Cramer-Von Mises statistics.

        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            ad_result = anderson_ksamp([a, b], variant="midrank")

        cvm_result = cramervonmises_2samp(a, b)

        ad_stat = self._extract_statistic(ad_result)
        cvm_stat = self._extract_statistic(cvm_result)

        return float(ad_stat + cvm_stat)


class KuiperDistance(BaseNumericDistance):
    """Kuiper distance metric using custom numpy optimization."""

    def _compute_1d(self, a: np.ndarray, b: np.ndarray) -> float:
        # Sort data
        d1 = np.sort(a)
        d2 = np.sort(b)
        n1, n2 = len(d1), len(d2)

        # Concatenate and sort all data points to find the common domain
        all_val = np.concatenate([d1, d2])
        all_val.sort()

        # Compute Empirical CDFs at the common points
        # searchsorted finds the index where values would be inserted
        cdf1 = np.searchsorted(d1, all_val, side="right") / n1
        cdf2 = np.searchsorted(d2, all_val, side="right") / n2

        # Kuiper statistic = max(cdf1 - cdf2) + max(cdf2 - cdf1)
        diff = cdf1 - cdf2
        d_plus = np.max(diff)
        d_minus = np.max(-diff)

        return float(d_plus + d_minus)
