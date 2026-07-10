"""Utilities for converting distance values into similarity scores."""

from __future__ import annotations

from collections.abc import Sequence

from xwhy.logger import logger


class DistanceNormalizer:
    """Normalize distance values into similarity scores."""

    @staticmethod
    def min_max(
        *,
        scores: Sequence[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Normalize distances using Min-Max normalization.

        Distances are converted into similarity values in the range
        ``[0.0, 1.0]`` where larger values indicate greater similarity.

        Args:
            scores:
                Sequence of ``(text, distance)`` pairs.

        Returns:
            List of ``(text, similarity)`` pairs.

        """
        distances = [distance for _, distance in scores]

        min_distance = min(distances)
        max_distance = max(distances)

        normalized: list[tuple[str, float]] = []

        for text, distance in scores:
            if max_distance == min_distance:
                similarity = 1.0
            else:
                similarity = 1.0 - (
                    (distance - min_distance) / (max_distance - min_distance)
                )

            normalized.append(
                (
                    text,
                    similarity,
                )
            )

        for text, similarity in normalized:
            logger.debug(
                "Perturbed Text: %s",
                text,
            )
            logger.debug(
                "Similarity Score: %.4f",
                similarity,
            )
            logger.debug("-" * 50)

        return normalized
