"""Utilities for converting distance values into similarity scores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from xwhy.logger import logger


class DistanceNormalizer:
    """Normalize distance values into similarity scores."""

    @staticmethod
    def min_max(
        *,
        scores: Sequence[tuple[str, float]],
        mode: Literal["linear", "inverse"] = "linear",
    ) -> list[tuple[str, float]]:
        """Normalize distances using Min-Max normalization.

        Convert distance values into similarity scores in the range
        ``[0.0, 1.0]`` where larger values indicate greater similarity.

        This method supports two normalization strategies:

        1. 'linear':
            Computes similarity via linear scaling:
            similarity = 1 - MinMax(distance)
            Preserves linear proportionality and avoids non-linear
            distortion, making it recommended for regression-based
            explainability.

        2. 'inverse':
            Computes similarity via inverse distance scaling:
            similarity = MinMax(1 / (distance + epsilon))
            Emphasizes small distances more aggressively through a
            non-linear boost.

        Args:
            scores:
                Sequence of ``(text, distance)`` pairs.
            mode:
                Normalization mode, either "linear" or "inverse".

        Returns:
            List of ``(text, similarity)`` pairs.

        Raises:
            ValueError: If the scores sequence is empty.

        """
        if not scores:
            logger.error("Distance list is empty. Cannot normalize similarities.")
            return []

        texts = [text for text, _ in scores]
        distances = np.array([distance for _, distance in scores], dtype=float)

        if mode == "inverse":
            epsilon = 1e-8
            inv = 1.0 / (distances + epsilon)
            min_v = inv.min()
            max_v = inv.max()

            if max_v == min_v:
                sim_vals = np.ones_like(inv)
            else:
                sim_vals = (inv - min_v) / (max_v - min_v)
        else:  # linear
            min_v = distances.min()
            max_v = distances.max()

            if max_v == min_v:
                sim_vals = np.ones_like(distances)
            else:
                norm = (distances - min_v) / (max_v - min_v)
                sim_vals = 1.0 - norm

        normalized = [
            (text, float(sim)) for text, sim in zip(texts, sim_vals, strict=False)
        ]

        for text, similarity in normalized:
            logger.debug(
                "Mode: %s | Perturbed Text: %s",
                mode,
                text,
            )
            logger.debug(
                "Similarity Score: %.4f",
                similarity,
            )
            logger.debug("-" * 50)

        return normalized
