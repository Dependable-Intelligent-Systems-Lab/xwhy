"""Tests for CustomImageGenerationAndEditingModel wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from xwhy.models.image_generation_and_editing.custom import (
    CustomImageGenerationAndEditingModel,
)


def test_custom_model_str_return(tmp_path: Path) -> None:
    """Test generation returning a string path directly."""
    output_dir = tmp_path / "output"
    expected_path = str(output_dir / "custom.png")

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> str:  # noqa: ANN401
        return expected_path

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(
        prompt="test prompt", output_dir=str(output_dir)
    )

    assert success is True
    assert path == expected_path


def test_custom_model_pil_image_generation(tmp_path: Path) -> None:
    """Test generation returning a PIL Image with generated prefix."""
    output_dir = tmp_path / "output"
    img = Image.new("RGB", (10, 10), color="red")

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> Image.Image:  # noqa: ANN401
        return img

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert os.path.exists(path)
    assert "custom_generated" in path


def test_edit_image_missing_file(tmp_path: Path) -> None:
    """Test edit_image raises FileNotFoundError when input image is missing."""
    model = CustomImageGenerationAndEditingModel(generate_fn=lambda **kw: "path")
    with pytest.raises(FileNotFoundError, match="Input image not found"):
        model.edit_image(
            prompt="edit",
            image_path=str(tmp_path / "nonexistent.png"),
            output_dir=str(tmp_path / "output"),
        )


def test_edit_image_pil_image_editing(tmp_path: Path) -> None:
    """Test editing returning a PIL Image with edited prefix."""
    output_dir = tmp_path / "output"
    input_img = tmp_path / "input.png"
    Image.new("RGB", (10, 10)).save(input_img)

    img = Image.new("RGB", (10, 10), color="blue")

    def dummy_generate(
        prompt: str,
        output_dir: str,
        input_image_path: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> Image.Image:
        assert input_image_path == str(input_img)
        return img

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.edit_image(
        prompt="edit",
        image_path=str(input_img),
        output_dir=str(output_dir),
    )

    assert success is True
    assert os.path.exists(path)
    assert "custom_edited" in path


def test_process_torch_tensor_result(tmp_path: Path) -> None:
    """Test processing torch tensor output with various dimensions and values."""
    output_dir = tmp_path / "output"

    # Test 4D tensor with float values <= 1.0 (channels first 3D inside 4D)
    tensor_4d = torch.rand(1, 3, 16, 16)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_4d

    model = CustomImageGenerationAndEditingModel(
        generate_fn=dummy_generate, model="dummy_model"
    )
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))
    assert success is True
    assert os.path.exists(path)

    # Test 3D tensor with integer dtype (not uint8) to cover
    # tensor_np.dtype != np.uint8 and max > 1.0
    tensor_3d_int = torch.randint(0, 256, (3, 16, 16), dtype=torch.long)

    def dummy_generate_3d_int(
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> torch.Tensor:
        return tensor_3d_int

    model_3d_int = CustomImageGenerationAndEditingModel(
        generate_fn=dummy_generate_3d_int
    )
    success, path = model_3d_int.generate_image(
        prompt="test", output_dir=str(output_dir)
    )
    assert success is True
    assert os.path.exists(path)


def test_process_numpy_array_result(tmp_path: Path) -> None:
    """Test processing numpy array output with various dimensions and types."""
    output_dir = tmp_path / "output"

    # Test 4D array
    rng = np.random.default_rng()
    arr_4d = rng.random((1, 16, 16, 3), dtype=np.float32)

    def dummy_generate_4d(prompt: str, output_dir: str, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        return arr_4d

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate_4d)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))
    assert success is True
    assert os.path.exists(path)

    # Test channels-first 3D array with integer dtype to cover
    # arr.dtype != np.uint8 and max > 1.0
    arr_3d_int = np.full((3, 16, 16), 150, dtype=np.int32)

    def dummy_generate_3d_int(
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        return arr_3d_int

    model_int = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate_3d_int)
    success, path = model_int.generate_image(prompt="test", output_dir=str(output_dir))
    assert success is True
    assert os.path.exists(path)


def test_unsupported_return_type_raises_value_error(tmp_path: Path) -> None:
    """Test ValueError is caught for unsupported return types from generate_fn."""
    model = CustomImageGenerationAndEditingModel(generate_fn=lambda **kw: 12345)
    success, path = model.generate_image(
        prompt="test", output_dir=str(tmp_path / "output")
    )

    assert success is False
    assert "Unsupported return type" in path


def test_generate_fn_exception_handling(tmp_path: Path) -> None:
    """Test exception during execution returns failure status and error string."""

    def faulty_generate(**kw: Any) -> Any:  # noqa: ANN401
        raise RuntimeError("Model execution failed")

    model = CustomImageGenerationAndEditingModel(generate_fn=faulty_generate)
    success, error_msg = model.generate_image(
        prompt="test", output_dir=str(tmp_path / "output")
    )

    assert success is False
    assert "Model execution failed" in error_msg


def test_torch_tensor_3d_permute_and_max_greater_than_one(tmp_path: Path) -> None:
    """Test 3D torch tensor with channels-first shape and max > 1.0."""
    output_dir = tmp_path / "output"
    tensor_data = torch.full((3, 16, 16), 128.0, dtype=torch.float32)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_torch_tensor_3d_max_less_than_or_equal_to_one(tmp_path: Path) -> None:
    """Test 3D torch tensor with channels-first shape and max <= 1.0."""
    output_dir = tmp_path / "output"
    tensor_data = torch.full((3, 16, 16), 0.5, dtype=torch.float32)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_numpy_array_3d_transpose_and_max_greater_than_one(tmp_path: Path) -> None:
    """Test 3D numpy array with channels-first shape and max > 1.0."""
    output_dir = tmp_path / "output"
    arr_data = np.full((3, 16, 16), 200.0, dtype=np.float32)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        return arr_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_numpy_array_3d_max_less_than_or_equal_to_one(tmp_path: Path) -> None:
    """Test 3D numpy array with channels-first shape and max <= 1.0."""
    output_dir = tmp_path / "output"
    arr_data = np.full((3, 16, 16), 0.8, dtype=np.float32)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> np.ndarray:  # noqa: ANN401
        return arr_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_torch_tensor_3d_shape_not_in_allowed_channels(tmp_path: Path) -> None:
    """Test 3D torch tensor with shape[0] not in (1, 3, 4) (channels-last format)."""
    output_dir = tmp_path / "output"
    tensor_data = torch.randint(0, 256, (16, 16, 3), dtype=torch.uint8)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_torch_tensor_2d(tmp_path: Path) -> None:
    """Test 2D torch tensor to cover ndim != 3 branch."""
    output_dir = tmp_path / "output"
    tensor_data = torch.rand(16, 16)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_torch_tensor_uint8_dtype(tmp_path: Path) -> None:
    """Test torch tensor with uint8 dtype to cover dtype == np.uint8 branch."""
    output_dir = tmp_path / "output"
    tensor_data = torch.randint(0, 256, (16, 16), dtype=torch.uint8)

    def dummy_generate(prompt: str, output_dir: str, **kwargs: Any) -> torch.Tensor:  # noqa: ANN401
        return tensor_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()


def test_numpy_array_uint8_dtype(tmp_path: Path) -> None:
    """Test numpy array with uint8 dtype to cover dtype == np.uint8 branch.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None

    """
    output_dir = tmp_path / "output"
    rng = np.random.default_rng()
    arr_data = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

    def dummy_generate(
        prompt: str,
        output_dir: str,
        **kwargs: Any,  # noqa: ANN401
    ) -> np.ndarray:
        return arr_data

    model = CustomImageGenerationAndEditingModel(generate_fn=dummy_generate)
    success, path = model.generate_image(prompt="test", output_dir=str(output_dir))

    assert success is True
    assert Path(path).exists()
