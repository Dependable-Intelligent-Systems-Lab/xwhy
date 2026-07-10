"""Word Mover's Distance implementation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from gensim.models import KeyedVectors

from xwhy.distance.base import BaseDistance


class WMDDistance(BaseDistance):
    """Word Mover's Distance implementation."""

    def clean_text(
        self,
        *,
        text: str,
    ) -> str:
        """Normalize text before computing distances.

        The normalization removes punctuation, converts text to lowercase,
        and trims surrounding whitespace.

        Args:
            text:
                Input text.

        Returns:
            Cleaned text.

        """
        cleaned = re.sub(r"[^\w\s]", "", text.lower())

        return cleaned.strip()

    def compute(
        self,
        *,
        model: KeyedVectors,
        source: str,
        target: str,
    ) -> float:
        """Compute Word Mover's Distance.

        Only words that exist in the embedding vocabulary are used.
        If either text contains no valid words, ``1.0`` is returned.

        Args:
            model:
                Loaded Word2Vec model.

            source:
                Source text.

            target:
                Target text.

        Returns:
            Word Mover's Distance.

        """
        words1 = [
            word for word in self.clean_text(text=source).split() if word in model
        ]

        words2 = [
            word for word in self.clean_text(text=target).split() if word in model
        ]

        if not words1 or not words2:
            return 1.0

        return float(model.wmdistance(words1, words2))

    def compute_batch(
        self,
        *,
        model: KeyedVectors,
        original: str,
        perturbed_texts: Sequence[str],
    ) -> list[tuple[str, float]]:
        """Compute WMD scores for perturbed texts.

        Args:
            model:
                Loaded Word2Vec model.

            original:
                Original text.

            perturbed_texts:
                Perturbed text samples.

        Returns:
            List of (text, distance).

        """
        scores: list[tuple[str, float]] = []

        for text in perturbed_texts:
            distance = self.compute(
                model=model,
                source=original,
                target=text,
            )

            scores.append((text, distance))

        return scores
