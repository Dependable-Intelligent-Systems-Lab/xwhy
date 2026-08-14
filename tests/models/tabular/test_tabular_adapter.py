"""Test suite for the tabular model adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import transformers

from xwhy.models.tabular.adapter import TabularModelAdapter


class MockPredictModel:
    """Mock model implementing the predict method."""

    def predict(self, x: np.ndarray) -> list[int]:
        """Return dummy predictions."""
        return [1, 0]


class MockPredictProbaModel:
    """Mock model implementing the predict_proba method."""

    def predict_proba(self, x: np.ndarray) -> list[list[float]]:
        """Return dummy prediction probabilities."""
        return [[0.1, 0.9]]


class MockCallableModel:
    """Mock model implementing the __call__ method."""

    def __call__(self, x: np.ndarray) -> list[int]:
        """Return dummy predictions."""
        return [1, 1]


class MockUnsupportedModel:
    """Mock model that is not supported by the adapter."""


def test_pytorch_model_with_tensor_output() -> None:
    """Test PyTorch module wrapper when output is a torch.Tensor."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.return_value = torch.tensor([1.0, 2.0])

    adapter = TabularModelAdapter(model=mock_model, device="cpu")
    result = adapter.predict(np.array([[1.0, 2.0]]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))
    mock_model.to.assert_called_once_with("cpu")
    mock_model.eval.assert_called_once()


def test_pytorch_model_with_non_tensor_output() -> None:
    """Test PyTorch module wrapper when output is not a tensor."""
    mock_model = MagicMock(spec=nn.Module)
    mock_model.return_value = [1.0, 2.0]

    adapter = TabularModelAdapter(model=mock_model, device="cpu")
    result = adapter.predict(np.array([[1.0, 2.0]]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1.0, 2.0]))


def test_hf_pipeline_with_ndarray_input() -> None:
    """Test Hugging Face pipeline wrapper with np.ndarray input."""
    mock_pipe = MagicMock(spec=transformers.Pipeline)
    mock_pipe.return_value = [{"score": 0.9}]

    adapter = TabularModelAdapter(model=mock_pipe)

    result = adapter.predict(np.array(["test string"]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([0.9]))


def test_hf_pipeline_with_sequence_input() -> None:
    """Test Hugging Face pipeline wrapper with sequence input."""
    mock_pipe = MagicMock(spec=transformers.Pipeline)
    mock_pipe.return_value = [{"score": 0.8}]

    adapter = TabularModelAdapter(model=mock_pipe)

    result = adapter.predict(["test string"])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([0.8]))


def test_hf_pipeline_complex_outputs() -> None:
    """Test Hugging Face pipeline with varied complex output structures."""
    mock_pipe = MagicMock(spec=transformers.Pipeline)

    mock_pipe.return_value = [
        [{"score": 0.9}],
        {"score": 0.8},
        [],
        [1, 2],
        {"label": "A"},
        "raw_text",
    ]

    adapter = TabularModelAdapter(model=mock_pipe)
    result = adapter.predict(["test"])

    assert isinstance(result, np.ndarray)
    assert len(result) == 6
    assert result[0] == [0.9]
    assert result[1] == 0.8
    assert result[2] == []
    assert result[3] == [1, 2]
    assert result[4] == {"label": "A"}
    assert result[5] == "raw_text"


def test_hf_pipeline_non_list_output() -> None:
    """Test Hugging Face pipeline when output is not a list."""
    mock_pipe = MagicMock(spec=transformers.Pipeline)
    mock_pipe.return_value = np.array([1, 2])

    adapter = TabularModelAdapter(model=mock_pipe)
    result = adapter.predict(["test"])

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2]))


def test_predict_method_model() -> None:
    """Test wrapper for models with a standard predict method."""
    model = MockPredictModel()
    adapter = TabularModelAdapter(model=model)
    result = adapter.predict(np.array([[1.0, 2.0]]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 0]))


def test_predict_proba_method_model() -> None:
    """Test wrapper for models with a predict_proba method."""
    model = MockPredictProbaModel()
    adapter = TabularModelAdapter(model=model)
    result = adapter.predict(np.array([[1.0, 2.0]]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[0.1, 0.9]]))


def test_callable_model() -> None:
    """Test wrapper for callable models or functions."""
    model = MockCallableModel()
    adapter = TabularModelAdapter(model=model)
    result = adapter.predict(np.array([[1.0, 2.0]]))

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 1]))


def test_unsupported_model() -> None:
    """Test that an unsupported model raises a TypeError."""
    model = MockUnsupportedModel()
    adapter = TabularModelAdapter(model=model)

    with pytest.raises(TypeError, match="not supported"):
        adapter.predict(np.array([[1.0, 2.0]]))


def test_hf_pipeline_non_list_inhomogeneous_output() -> None:
    """Test Hugging Face pipeline when non-list output raises ValueError."""
    mock_pipe = MagicMock(spec=transformers.Pipeline)
    # Return a tuple of ragged lists so isinstance(raw_preds, list) is False
    # and np.asarray(raw_preds) triggers ValueError, falling back to dtype=object
    mock_pipe.return_value = ([1, 2], [3, 4, 5])

    adapter = TabularModelAdapter(model=mock_pipe)
    result = adapter.predict(["test"])

    assert isinstance(result, np.ndarray)
    assert result.dtype == object
    assert len(result) == 2
    assert result[0] == [1, 2]
    assert result[1] == [3, 4, 5]
