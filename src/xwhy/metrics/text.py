"""Text-specific evaluation metrics for explainability."""

import numpy as np
from sklearn.metrics import roc_auc_score

from xwhy.core.result import BaseXWhyResult


def calculate_stability_score(
    result_one: BaseXWhyResult,
    result_two: BaseXWhyResult,
) -> tuple[float, float]:
    """Calculate stability metrics between two explanation results.

    This function aligns the feature contributions of two similar results based
    on their feature names (tokens) and computes both the Generalized Jaccard
    Similarity and Jaccard Distance between their contribution vectors.

    Args:
        result_one: The first explanation result object.
        result_two: The second explanation result object.

    Returns:
        tuple[float, float]: A tuple containing:
            - Jaccard Similarity: A value between 0 and 1 indicating similarity
              (1.0 means identical importance distributions).
            - Jaccard Distance: A value between 0 and 1 indicating dissimilarity
              (0.0 means identical importance distributions).

    Raises:
        ValueError: If the length of features and contribution vectors mismatch.

    """
    words_one = result_one.feature_names
    words_two = result_two.feature_names
    coeffs_one = np.asarray(result_one.coefficients).flatten()
    coeffs_two = np.asarray(result_two.coefficients).flatten()

    if len(words_one) != len(coeffs_one) or len(words_two) != len(coeffs_two):  # type: ignore[arg-type]
        raise ValueError("Length of features and contribution vectors must match.")

    # Map words to their contributions
    # Note: Duplicate words will take the value of the last occurrence.
    map_one: dict[str, float] = dict(zip(words_one, coeffs_one, strict=False))  # type: ignore[arg-type]
    map_two: dict[str, float] = dict(zip(words_two, coeffs_two, strict=False))  # type: ignore[arg-type]

    # Create the union of all unique words from both prompts
    unique_words = sorted(set(words_one) | set(words_two))  # type: ignore[arg-type]

    # Align vectors based on the union of words.
    # We use absolute values to measure the magnitude of importance.
    vec_one = np.array([abs(map_one.get(w, 0.0)) for w in unique_words])
    vec_two = np.array([abs(map_two.get(w, 0.0)) for w in unique_words])

    # Compute Generalized Jaccard Similarity (Ruzicka similarity)
    numerator = np.sum(np.minimum(vec_one, vec_two))
    denominator = np.sum(np.maximum(vec_one, vec_two))

    if denominator == 0:
        # Both vectors are zero, implying they are identical.
        return 1.0, 0.0

    jaccard_similarity = float(numerator / denominator)
    jaccard_distance = 1.0 - jaccard_similarity

    return jaccard_similarity, jaccard_distance


def calculate_token_auc(
    result: BaseXWhyResult,
    truth: list[int],
) -> float:
    """Calculate the Area Under the ROC Curve (AUC) for token importance.

    Evaluate how well the provided contribution scores align with the
    ground truth binary labels.

    Args:
        result: The explanation result object containing features and scores.
        truth: Ground truth binary labels (1 for relevant, 0 otherwise).

    Returns:
        float: The calculated ROC AUC score. Returns 0.5 if only one class
            is present in the truth labels (as AUC is undefined).

    Raises:
        ValueError: If the number of tokens does not match the lengths
            of the truth labels.

    """
    tokens = result.feature_names
    scores = np.asarray(result.coefficients).flatten()

    # Validation of input lengths
    if len(tokens) != len(scores) or len(tokens) != len(truth):  # type: ignore[arg-type]
        raise ValueError(
            f"Dimension mismatch: Found {len(tokens)} tokens, "  # type: ignore[arg-type]
            f"{len(scores)} scores, and {len(truth)} truth labels."
        )

    # AUC requires at least one positive and one negative sample
    unique_classes = np.unique(truth)
    if len(unique_classes) < 2:
        return 0.5

    y_true = np.array(truth)
    y_scores = np.array(scores)

    return float(roc_auc_score(y_true, y_scores))
