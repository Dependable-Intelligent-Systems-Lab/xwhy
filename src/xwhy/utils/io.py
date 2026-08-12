"""Input/Output utility functions for loading and saving data."""

import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from xwhy.logger import logger


def save_data_to_pickle(
    *,
    output_path: str = "pickle_data.pkl",
    **data: Any,  # noqa: ANN401
) -> None:
    """Save keyword arguments directly to a pickle file.

    Args:
        output_path: Full path including filename where the pickle
            file should be saved.
        **data: Arbitrary named data items to persist.

    Raises:
        Exception: If an error occurs during file writing.

    """
    try:
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("Data successfully saved to: %s", output_path)
    except Exception as exc:
        logger.exception("Error saving data to pickle file: %s", exc)
        raise


def load_data_from_pickle(
    file_path: str,
) -> dict[str, Any]:
    """Load and return the object stored in a pickle file.

    Args:
        file_path: Path to the target pickle file.

    Returns:
        A dictionary containing the persisted data items.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: If an error occurs during unpickling.

    """
    try:
        with open(file_path, "rb") as f:
            loaded_data = pickle.load(f)
        logger.debug("Data successfully loaded from: %s", file_path)
        return loaded_data  # type: ignore[no-any-return]
    except FileNotFoundError:
        logger.exception("Error: File not found at %s", file_path)
        raise
    except Exception as exc:
        logger.exception("Error loading data from pickle file: %s", exc)
        raise


def save_perturbation_data_to_csv(
    *,
    perturbations: list[np.ndarray],
    similarities: list[tuple[str, float]] | None = None,
    wmd_scores: list[tuple[str, float]] | None = None,
    output_path: str = "perturbation_data.csv",
    **extra_columns: list[Any],
) -> str:
    """Consolidate perturbation data and arbitrary metrics into a CSV file.

    Args:
        perturbations: List of binary perturbation vectors.
        similarities: Optional list of text and similarity score pairs.
        wmd_scores: Optional list of text and distance score pairs.
        output_path: Full path including filename for the CSV output.
        **extra_columns: Additional named columns to include in the CSV.

    Returns:
        The full path to the saved CSV file.

    Raises:
        ValueError: If perturbations list is empty or data lengths mismatch.

    """
    if not perturbations:
        raise ValueError("Perturbations list must be non-empty.")

    x_perturbations = np.vstack(perturbations)
    n_features = x_perturbations.shape[1]
    feature_cols = [f"x_{i + 1}" for i in range(n_features)]

    df = pd.DataFrame(x_perturbations, columns=feature_cols)

    # Handle legacy similarities if provided
    if similarities:
        perturbed_texts_sim = [text for text, _ in similarities]
        similarity_scores = [score for _, score in similarities]
        df.insert(0, "Perturbed Text", perturbed_texts_sim)
        df["Similarity_Score"] = similarity_scores

    # Handle legacy wmd_scores if provided
    if wmd_scores:
        wmd_distances = [distance for _, distance in wmd_scores]
        if not similarities and "Perturbed Text" not in df.columns:
            perturbed_texts_wmd = [text for text, _ in wmd_scores]
            df.insert(0, "Perturbed Text", perturbed_texts_wmd)
        df["WMD_Distance"] = wmd_distances

    # Handle any arbitrary extra columns passed dynamically
    for col_name, col_data in extra_columns.items():
        if len(col_data) != len(df):
            raise ValueError(
                f"Length of '{col_name}' ({len(col_data)}) does not match "
                f"number of rows ({len(df)})."
            )
        df[col_name] = col_data

    save_dir = os.path.dirname(output_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    df.to_csv(output_path, index=False)

    logger.debug("Data successfully saved to: %s", output_path)
    return output_path
