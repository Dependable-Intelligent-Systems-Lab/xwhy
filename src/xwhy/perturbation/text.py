"""Text perturbation implementation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import Generator

from xwhy.logger import logger
from xwhy.perturbation.base import BasePerturbation


class TextPerturbation(BasePerturbation):
    """Generate binary perturbations for text.

    This class creates binary perturbation masks and applies them to an
    input sentence by randomly removing words.

    A minimum number of words is always preserved to avoid generating
    completely meaningless samples.
    """

    _MIN_WORDS: int = 2
    _KEEP_PROBABILITY: float = 0.5
    _MAX_ATTEMPT_FACTOR: int = 10

    def __init__(self, *, seed: int = 1024) -> None:
        """Initialize the perturbation generator.

        Args:
            seed:
                Random seed for reproducible perturbations.

        """
        self._rng: Generator = np.random.default_rng(seed)

    def apply_mask(
        self,
        *,
        words: Sequence[str],
        mask: Sequence[int],
    ) -> list[str]:
        """Apply a binary mask to tokenized text.

        The mask determines which words remain in the output.
        If fewer than two words survive, random words are added until the
        minimum size is reached.

        Args:
            words:
                Original tokenized text.
            mask:
                Binary mask.

        Returns:
            List of selected words.

        """
        selected = [
            word
            for word, flag in zip(words, mask, strict=False)
            if flag == 1 and word.strip()
        ]

        selected = list(dict.fromkeys(selected))

        if len(selected) >= self._MIN_WORDS:
            return selected

        candidates = [word for word in words if word.strip() and word not in selected]

        needed = min(
            self._MIN_WORDS - len(selected),
            len(candidates),
        )

        if needed > 0:
            extra = self._rng.choice(
                candidates,
                size=needed,
                replace=False,
            ).tolist()

            selected.extend(extra)

        return selected

    def generate(
        self,
        *,
        text: str,
        num_perturbations: int = 64,
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        """Generate perturbed text samples.

        Args:
            text:
                Original input text.

            num_perturbations:
                Number of perturbations to generate.

        Returns:
            Tuple containing:

            - perturbed texts
            - binary masks

        """
        words = text.split()

        num_words = len(words)

        responses: list[str] = []
        perturbations: list[tuple[int, ...]] = []

        unique_masks: set[tuple[int, ...]] = set()

        attempts = 0
        max_attempts = num_perturbations * self._MAX_ATTEMPT_FACTOR

        while len(unique_masks) < num_perturbations and attempts < max_attempts:
            mask = tuple(
                int(value)
                for value in self._rng.binomial(
                    1,
                    self._KEEP_PROBABILITY,
                    size=num_words,
                )
            )

            if mask not in unique_masks and sum(mask) > 0:
                unique_masks.add(mask)

                perturbed = self.apply_mask(
                    words=words,
                    mask=mask,
                )

                corpus = " ".join(perturbed)

                responses.append(corpus)
                perturbations.append(mask)

                logger.info(
                    "Perturbation: %s, Perturbed Text: %s",
                    mask,
                    corpus,
                )

            attempts += 1

        while len(responses) < num_perturbations:
            mask = tuple(self._rng.choice(list(unique_masks)))

            perturbed = self.apply_mask(
                words=words,
                mask=mask,
            )

            corpus = " ".join(perturbed)

            responses.append(corpus)
            perturbations.append(mask)

            logger.info(
                "Perturbation (reused): %s, Perturbed Text: %s",
                mask,
                corpus,
            )

        return responses, perturbations
