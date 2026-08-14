"""Tests for PairedInferenceModel wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from xwhy.models.image_generation_and_editing.paired import PairedInferenceModel


def test_generate_image_raises_not_implemented() -> None:
    """Test generate_image raises NotImplementedError."""
    model = PairedInferenceModel(model_name="test_model")
    with pytest.raises(
        NotImplementedError,
        match="Paired inference does not support generation from scratch",
    ):
        model.generate_image(prompt="test", output_dir="dummy")


@patch("xwhy.models.image_generation_and_editing.paired.run_inference_paired")
def test_edit_image_success(mock_run: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """Test successful image editing with paired inference."""
    output_dir = tmp_path / "output"
    input_path = tmp_path / "input.png"
    input_path.touch()

    def side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.png").touch()

    mock_run.side_effect = side_effect

    model = PairedInferenceModel(model_name="test_model")
    success, path = model.edit_image(
        prompt="edit prompt",
        image_path=str(input_path),
        output_dir=str(output_dir),
        model_name="collision_name",
    )

    assert success is True
    assert Path(path).exists()
    assert "test_model_edited_" in path
    mock_run.assert_called_once()


@patch("xwhy.models.image_generation_and_editing.paired.run_inference_paired")
def test_edit_image_output_dir_missing(mock_run: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """Test edit_image handles non-existent output directory after execution."""
    output_dir = tmp_path / "nonexistent"
    input_path = tmp_path / "input.png"
    input_path.touch()

    model = PairedInferenceModel(model_name="test_model")
    success, path = model.edit_image(
        prompt="edit prompt",
        image_path=str(input_path),
        output_dir=str(output_dir),
    )

    assert success is False
    assert path == ""


@patch("xwhy.models.image_generation_and_editing.paired.run_inference_paired")
def test_edit_image_output_dir_is_file(mock_run: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """Test edit_image handles output_dir being a file instead of a directory."""
    output_file = tmp_path / "file_as_dir"
    output_file.touch()
    input_path = tmp_path / "input.png"
    input_path.touch()

    model = PairedInferenceModel(model_name="test_model")
    success, path = model.edit_image(
        prompt="edit prompt",
        image_path=str(input_path),
        output_dir=str(output_file),
    )

    assert success is False
    assert path == ""


@patch("xwhy.models.image_generation_and_editing.paired.run_inference_paired")
def test_edit_image_no_files_found(mock_run: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """Test edit_image handles empty output directory."""
    output_dir = tmp_path / "output"
    input_path = tmp_path / "input.png"
    input_path.touch()

    def side_effect(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        output_dir.mkdir(parents=True, exist_ok=True)

    mock_run.side_effect = side_effect

    model = PairedInferenceModel(model_name="test_model")
    success, path = model.edit_image(
        prompt="edit prompt",
        image_path=str(input_path),
        output_dir=str(output_dir),
    )

    assert success is False
    assert path == ""


@patch("xwhy.models.image_generation_and_editing.paired.run_inference_paired")
def test_edit_image_exception_handling(mock_run: Any, tmp_path: Path) -> None:  # noqa: ANN401
    """Test edit_image handles exceptions raised by run_inference_paired."""
    input_path = tmp_path / "input.png"
    input_path.touch()

    mock_run.side_effect = RuntimeError("Inference failed")

    model = PairedInferenceModel(model_name="test_model")
    success, path = model.edit_image(
        prompt="edit prompt",
        image_path=str(input_path),
        output_dir=str(tmp_path / "output"),
    )

    assert success is False
    assert path == ""
