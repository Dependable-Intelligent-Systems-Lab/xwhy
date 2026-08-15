"""Word Mover's Distance implementation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
from gensim.models import KeyedVectors

from xwhy.distance.base import BaseDistance


class WMDDistance(BaseDistance):
    """Word Mover's Distance implementation for Text Data."""

    def clean_text(
        self,
        *,
        text: str,
    ) -> str:
        """Normalize text before computing distances.

        The normalization removes punctuation, converts text to lowercase,
        and trims surrounding whitespace.

        Args:
            text: Input text to be normalized.

        Returns:
            Cleaned and normalized text string.

        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower())

        return cleaned.strip()

    def compute(
        self,
        source: str,
        target: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> float:
        """Compute Word Mover's Distance between two text instances.

        Args:
            source: Source text string.
            target: Target text string.
            **kwargs: Must contain 'model' (loaded Word2Vec KeyedVectors).

        Returns:
            float: Calculated Word Mover's Distance value.

        Raises:
            ValueError: If 'model' is missing or not an instance of KeyedVectors.

        """
        model = kwargs.get("model")
        if not isinstance(model, KeyedVectors):
            raise ValueError(
                "WMDDistance requires a gensim KeyedVectors 'model' passed via kwargs."
            )

        words1 = [
            word for word in self.clean_text(text=source).split() if word in model
        ]
        words2 = [
            word for word in self.clean_text(text=target).split() if word in model
        ]

        if not words1 or not words2:
            return 1.0

        return float(model.wmdistance(words1, words2))

    @staticmethod
    def sanitize_distances(distances: np.ndarray) -> np.ndarray:
        """Sanitize raw distance array by safely handling NaN and Infinite values.

        Non-finite values are replaced by a fallback distance equal to the maximum
        finite distance plus 50.0, or 100.0 if all values are non-finite.

        Args:
            distances: 1D array of raw distance values.

        Returns:
            np.ndarray: Sanitized 1D array guaranteed to contain finite float values.

        """
        cleaned = np.array(distances, dtype=float, copy=True)
        finite_mask = np.isfinite(cleaned)
        finite_distances = cleaned[finite_mask]

        if finite_distances.size > 0:
            max_finite_distance = float(np.max(finite_distances))
            safe_fallback = max_finite_distance + 50.0
            cleaned[~finite_mask] = safe_fallback
        else:
            cleaned[:] = 100.0

        return cleaned

    def compute_batch(
        self,
        *,
        model: KeyedVectors,
        original: str,
        perturbed_texts: Sequence[str],
        sanitize: bool = False,
    ) -> list[tuple[str, float]]:
        """Compute WMD scores for a batch of perturbed texts.

        Args:
            model: Loaded Word2Vec model.
            original: Original input text string.
            perturbed_texts: Sequence of perturbed text samples.
            sanitize: If True, applies sanitize_distances to clean non-finite values.

        Returns:
            list[tuple[str, float]]: List of (perturbed_text, distance) tuples.

        """
        raw_distances = [
            self.compute(
                source=original,
                target=text,
                model=model,
            )
            for text in perturbed_texts
        ]

        if sanitize:
            distances = self.sanitize_distances(
                np.array(raw_distances, dtype=float)
            ).tolist()
        else:
            distances = raw_distances

        return list(zip(perturbed_texts, distances, strict=True))
