"""Tabular model adapter module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import transformers


class TabularModelAdapter:
    """Wrap tabular models to provide a unified prediction interface.

    This adapter standardizes various model types (PyTorch, Hugging Face,
    scikit-learn, XGBoost, and callables) into a consistent `.predict()`
    method that receives and returns NumPy arrays.
    """

    def __init__(self, model: Any, device: str = "cpu") -> None:  # noqa: ANN401
        """Initialize the tabular model adapter.

        Args:
            model: The black-box model or pipeline to wrap.
            device: Computation device for tensor-based models.

        """
        self.model: Any = model
        self.device: str = device

        self._is_pytorch: bool = isinstance(model, nn.Module)
        self._is_hf_pipeline: bool = isinstance(model, transformers.Pipeline)

        if self._is_pytorch:
            self.model.to(self.device)
            self.model.eval()

    def predict(self, x: np.ndarray | Sequence[Any]) -> np.ndarray:
        """Execute model inference and return predictions as a NumPy array.

        Args:
            x: Input feature array or sequence of samples.

        Returns:
            np.ndarray: Model prediction outputs.

        Raises:
            TypeError: If the model type is not supported.

        """
        x_arr = np.asarray(x)

        # 1. Handle PyTorch nn.Module
        if self._is_pytorch:
            return self._predict_pytorch(x_arr)

        # 2. Handle Hugging Face Pipeline
        if self._is_hf_pipeline:
            return self._predict_hf_pipeline(x)

        # 3. Handle models with .predict() method (scikit-learn, XGBoost, etc.)
        if hasattr(self.model, "predict") and callable(self.model.predict):
            preds = self.model.predict(x_arr)
            return np.asarray(preds)

        # 4. Handle models with .predict_proba() method
        if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
            preds = self.model.predict_proba(x_arr)
            return np.asarray(preds)

        # 5. Handle callable objects / functions
        if callable(self.model):
            preds = self.model(x_arr)
            return np.asarray(preds)

        raise TypeError(
            f"The provided model of type '{type(self.model).__name__}' is "
            "not supported. Must be a PyTorch nn.Module, Hugging Face "
            "Pipeline, have a `.predict()` method, or be callable."
        )

    def _predict_pytorch(self, x_arr: np.ndarray) -> np.ndarray:
        """Perform inference using a PyTorch nn.Module.

        Args:
            x_arr: Input array of shape [n_samples, n_features].

        Returns:
            np.ndarray: Predictions as a NumPy array.

        """
        tensor_x = torch.tensor(x_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            preds = self.model(tensor_x)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
        return np.asarray(preds)

    def _predict_hf_pipeline(self, x: np.ndarray | Sequence[Any]) -> np.ndarray:
        """Perform inference using a Hugging Face Pipeline.

        Args:
            x: Input data array or sequence of inputs.

        Returns:
            np.ndarray: Extracted predictions or probabilities.

        """
        inputs = x.tolist() if isinstance(x, np.ndarray) else list(x)

        raw_preds = self.model(inputs)

        if isinstance(raw_preds, list):
            extracted = []
            for item in raw_preds:
                if isinstance(item, list) and item and isinstance(item[0], dict):
                    extracted.append([sub["score"] for sub in item])
                elif isinstance(item, dict) and "score" in item:
                    extracted.append(item["score"])
                else:
                    extracted.append(item)

            try:
                return np.asarray(extracted)
            except ValueError:
                return np.asarray(extracted, dtype=object)

        try:
            return np.asarray(raw_preds)
        except ValueError:
            return np.asarray(raw_preds, dtype=object)
