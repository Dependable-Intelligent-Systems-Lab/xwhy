"""Tests for input/output utility functions."""

import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from xwhy.utils.io import (
    load_data_from_pickle,
    save_data_to_pickle,
    save_perturbation_data_to_csv,
)


def test_save_data_to_pickle_success(tmp_path: Path) -> None:
    """Test saving data to a pickle file successfully."""
    file_path = str(tmp_path / "test.pkl")
    save_data_to_pickle(output_path=file_path, key="value")
    assert os.path.exists(file_path)


@patch("xwhy.utils.io.open", side_effect=PermissionError("Access denied"))
def test_save_data_to_pickle_exception(mock_open: MagicMock) -> None:
    """Test save_data_to_pickle raises and logs exception on write failure."""
    with pytest.raises(PermissionError, match="Access denied"):
        save_data_to_pickle(output_path="dummy.pkl", key="value")


def test_load_data_from_pickle_success(tmp_path: Path) -> None:
    """Test loading data from a pickle file successfully."""
    file_path = str(tmp_path / "test.pkl")
    save_data_to_pickle(output_path=file_path, foo="bar")

    data = load_data_from_pickle(file_path)
    assert data == {"foo": "bar"}


def test_load_data_from_pickle_file_not_found() -> None:
    """Test load_data_from_pickle raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        load_data_from_pickle("nonexistent_file.pkl")


@patch("xwhy.utils.io.open", side_effect=Exception("Corrupted pickle"))
def test_load_data_from_pickle_general_exception(mock_open: MagicMock) -> None:
    """Test load_data_from_pickle raises generic exceptions on unpickling failure."""
    with pytest.raises(Exception, match="Corrupted pickle"):
        load_data_from_pickle("dummy.pkl")


def test_save_perturbation_data_to_csv_empty_perturbations() -> None:
    """Test ValueError is raised when perturbations list is empty."""
    with pytest.raises(
        ValueError, match=re.escape("Perturbations list must be non-empty.")
    ):
        save_perturbation_data_to_csv(perturbations=[])


def test_save_perturbation_data_to_csv_with_similarities_and_wmd(
    tmp_path: Path,
) -> None:
    """Test saving CSV with both similarities and wmd_scores."""
    output_file = str(tmp_path / "sub" / "perturbations.csv")
    perturbations = [np.array([0, 1]), np.array([1, 0])]
    similarities = [("text1", 0.9), ("text2", 0.8)]
    wmd_scores = [("text1", 0.1), ("text2", 0.2)]

    path = save_perturbation_data_to_csv(
        perturbations=perturbations,
        similarities=similarities,
        wmd_scores=wmd_scores,
        output_path=output_file,
        extra_col=[10, 20],
    )

    assert os.path.exists(path)


def test_save_perturbation_data_to_csv_with_wmd_only(tmp_path: Path) -> None:
    """Test saving CSV with wmd_scores only (without similarities)."""
    output_file = str(tmp_path / "perturbations_wmd.csv")
    perturbations = [np.array([1, 1])]
    wmd_scores = [("text_wmd", 0.5)]

    path = save_perturbation_data_to_csv(
        perturbations=perturbations,
        wmd_scores=wmd_scores,
        output_path=output_file,
    )

    assert os.path.exists(path)


def test_save_perturbation_data_to_csv_extra_column_length_mismatch(
    tmp_path: Path,
) -> None:
    """Test ValueError is raised when extra column length does not match row count."""
    output_file = str(tmp_path / "fail.csv")
    perturbations = [np.array([0]), np.array([1])]

    with pytest.raises(ValueError, match="does not match number of rows"):
        save_perturbation_data_to_csv(
            perturbations=perturbations,
            output_path=output_file,
            bad_col=[1],  # Length 1 for 2 rows
        )
