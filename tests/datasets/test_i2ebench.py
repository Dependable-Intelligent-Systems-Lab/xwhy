"""Tests for I2EBench dataset downloader and loader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xwhy.datasets.i2ebench import (
    download_i2ebench_dataset,
    load_i2ebench_data,
)


@patch("zipfile.ZipFile")
@patch("gdown.download")
def test_download_i2ebench_dataset(
    mock_gdown: MagicMock, mock_zipfile: MagicMock, tmp_path: Path
) -> None:
    """Test downloading and extracting I2EBench dataset."""
    extract_dir = tmp_path / "extracted"
    result = download_i2ebench_dataset(
        url="http://dummy-url",
        output_filename=str(tmp_path / "test.zip"),
        extract_dir=str(extract_dir),
    )
    mock_gdown.assert_called_once()
    mock_zipfile.assert_called_once()
    assert result == str(extract_dir)


def test_load_i2ebench_data_missing_edit_data(tmp_path: Path) -> None:
    """Test FileNotFoundError when EditData directory is missing."""
    with pytest.raises(FileNotFoundError):
        load_i2ebench_data(root_dir=str(tmp_path))


def test_load_i2ebench_data_limit_length_mismatch(tmp_path: Path) -> None:
    """Test ValueError on limits and categories length mismatch."""
    edit_data = tmp_path / "EditBench" / "EditData" / "Deblurring"
    edit_data.mkdir(parents=True)

    with pytest.raises(ValueError, match="Length mismatch"):
        load_i2ebench_data(
            root_dir=str(tmp_path),
            categories=["Deblurring"],
            limits_per_category=[1, 2],
        )


def test_load_i2ebench_data_missing_category_dir(tmp_path: Path) -> None:
    """Test FileNotFoundError when a category directory is missing."""
    edit_data = tmp_path / "EditBench" / "EditData"
    edit_data.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_i2ebench_data(
            root_dir=str(tmp_path),
            categories=["NonExistentCategory"],
            limits_per_category=1,
        )


def test_load_i2ebench_data_missing_json(tmp_path: Path) -> None:
    """Test FileNotFoundError when category JSON file is missing."""
    cat_dir = tmp_path / "EditBench" / "EditData" / "Deblurring"
    cat_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        load_i2ebench_data(
            root_dir=str(tmp_path),
            categories=["Deblurring"],
            limits_per_category=1,
        )


def test_load_i2ebench_data_success(tmp_path: Path) -> None:
    """Test successful loading of I2EBench dataset with branches."""
    cat_dir = tmp_path / "EditBench" / "EditData" / "Deblurring"
    input_dir = cat_dir / "input"
    input_dir.mkdir(parents=True)

    # Create a valid image file
    img_file = input_dir / "img1.png"
    img_file.write_bytes(b"fake image data")

    # JSON with various branches:
    # 1. Valid entry (image exists)
    # 2. Missing image file (triggers `if not os.path.exists` continue branch)
    # 3. Missing image key (triggers `if image_filename and prompt` false branch)
    # 4. Missing prompt key (triggers `if image_filename and prompt` false branch)
    json_data = {
        "item1": {"image": "img1.png", "ori_exp": "enhance image"},
        "item2": {"image": "nonexistent.png", "ori_exp": "missing file"},
        "item3": {"ori_exp": "missing image key"},
        "item4": {"image": "img1.png"},
    }

    json_path = cat_dir / "Deblurring.json"
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    # Test loading with specific category and explicit list limits
    result = load_i2ebench_data(
        root_dir=str(tmp_path),
        categories=["Deblurring"],
        limits_per_category=[2],
    )

    assert "Deblurring" in result
    assert len(result["Deblurring"]) == 1
    assert result["Deblurring"][0][1] == "enhance image"


def test_load_i2ebench_data_default_categories(tmp_path: Path) -> None:
    """Test loading with default categories list and integer limit."""
    edit_data = tmp_path / "EditBench" / "EditData"
    default_categories = [
        "Deblurring",
        "HazeRemoval",
        "Lowlight",
        "NoiseRemoval",
        "RainRemoval",
        "ShadowRemoval",
        "SnowRemoval",
        "WatermarkRemoval",
    ]

    for cat in default_categories:
        cat_dir = edit_data / cat
        input_dir = cat_dir / "input"
        input_dir.mkdir(parents=True)
        img_file = input_dir / "img.png"
        img_file.write_bytes(b"data")

        json_path = cat_dir / f"{cat}.json"
        json_path.write_text(
            json.dumps({"1": {"image": "img.png", "ori_exp": "test"}}),
            encoding="utf-8",
        )

    result = load_i2ebench_data(root_dir=str(tmp_path), limits_per_category=1)
    assert len(result) == 8
    assert len(result["Deblurring"]) == 1


def test_load_i2ebench_data_limit_break_condition(tmp_path: Path) -> None:
    """Test loop breaks early when limit per category is reached."""
    cat_dir = tmp_path / "EditBench" / "EditData" / "Deblurring"
    input_dir = cat_dir / "input"
    input_dir.mkdir(parents=True)

    # Create multiple valid image files
    (input_dir / "img1.png").write_bytes(b"data1")
    (input_dir / "img2.png").write_bytes(b"data2")

    json_data = {
        "item1": {"image": "img1.png", "ori_exp": "prompt1"},
        "item2": {"image": "img2.png", "ori_exp": "prompt2"},
    }

    json_path = cat_dir / "Deblurring.json"
    json_path.write_text(json.dumps(json_data), encoding="utf-8")

    # Limit is 1, so processing item2 will trigger the break condition
    result = load_i2ebench_data(
        root_dir=str(tmp_path),
        categories=["Deblurring"],
        limits_per_category=1,
    )

    assert len(result["Deblurring"]) == 1
    assert result["Deblurring"][0][1] == "prompt1"
