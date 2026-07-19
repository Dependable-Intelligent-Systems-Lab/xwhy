"""Unit tests for classification torchvision model."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from xwhy.config.settings import Settings
from xwhy.models.classification.torchvision_models import TorchvisionClassification


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture to provide a mocked settings object."""
    settings = MagicMock()
    settings.classification_cache_dir = None
    return settings


class TestTorchvisionClassificationInit:
    """Test initialization and conditional device assignments."""

    def test_init_invalid_model(self, mock_settings: MagicMock) -> None:
        """Test if a ValueError is raised for unsupported models."""
        with pytest.raises(ValueError, match="Unsupported model 'invalid_model'"):
            TorchvisionClassification(
                settings=mock_settings, model_name="invalid_model"
            )

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_none_with_cuda(
        self, mock_cuda: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test device fallback to cuda when available and device is None."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=None
        )
        assert model._device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_device_none_without_cuda(
        self, mock_cuda: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test device fallback to cpu when cuda is unavailable."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=None
        )
        assert model._device.type == "cpu"

    def test_init_device_as_string(self, mock_settings: MagicMock) -> None:
        """Test device initialization using a string."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device="cpu"
        )
        assert model._device == torch.device("cpu")

    def test_init_device_as_torch_device(self, mock_settings: MagicMock) -> None:
        """Test device initialization using a torch.device instance."""
        device = torch.device("cpu")
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=device
        )
        assert model._device == device


class TestTorchvisionClassificationMethods:
    """Test internal methods, load logic, and prediction paths."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.manual_seed")
    def test_set_seed_with_cuda(
        self,
        mock_manual_seed: MagicMock,
        mock_cuda_seed: MagicMock,
        mock_cuda_avail: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test seed setting logic when CUDA is available."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", seed=123
        )
        model._set_seed()

        mock_manual_seed.assert_called_once_with(123)
        mock_cuda_seed.assert_called_once_with(123)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.manual_seed")
    def test_set_seed_without_cuda(
        self,
        mock_manual_seed: MagicMock,
        mock_cuda_seed: MagicMock,
        mock_cuda_avail: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test seed setting logic when CUDA is not available."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", seed=123
        )
        model._set_seed()

        mock_manual_seed.assert_called_once_with(123)
        mock_cuda_seed.assert_not_called()

    def test_predict_already_loaded(self, mock_settings: MagicMock) -> None:
        """Test predict method when model is already initialized."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device="cpu"
        )

        mock_model = MagicMock()
        expected_logits = torch.tensor([[1.0, 0.5]])
        mock_model.return_value = expected_logits
        model._model = mock_model

        dummy_input = torch.randn(1, 3, 224, 224)
        result = model.predict(dummy_input)

        mock_model.assert_called_once()

        assert torch.equal(result, expected_logits)

    @patch.object(TorchvisionClassification, "load")
    def test_predict_calls_load_if_missing(
        self, mock_load: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test predict method calls load() if model is None."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device="cpu"
        )

        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.0, 1.0]])

        mock_load.return_value = (MagicMock(), mock_model)

        dummy_input = torch.randn(1, 3, 224, 224)
        model.predict(dummy_input)

        mock_load.assert_called_once()
        mock_model.assert_called_once()

    # ==========================================
    # Tests for __init__ (All conditions covered)
    # ==========================================
    def test_init_invalid_model(self, mock_settings: MagicMock) -> None:
        """Test if a ValueError is raised for unsupported models."""
        with pytest.raises(ValueError, match="Unsupported model 'invalid_model'"):
            TorchvisionClassification(
                settings=mock_settings, model_name="invalid_model"
            )

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_none_with_cuda(
        self, mock_cuda: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test device fallback to cuda when available and device is None."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=None
        )
        assert model._device.type == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_init_device_none_without_cuda(
        self, mock_cuda: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test device fallback to cpu when cuda is unavailable."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=None
        )
        assert model._device.type == "cpu"

    def test_init_device_as_string(self, mock_settings: MagicMock) -> None:
        """Test device initialization using a string."""
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device="cuda:1"
        )
        assert model._device == torch.device("cuda:1")

    def test_init_device_as_torch_device(self, mock_settings: MagicMock) -> None:
        """Test device initialization using a torch.device instance."""
        device = torch.device("cpu")
        model = TorchvisionClassification(
            settings=mock_settings, model_name="resnet18", device=device
        )
        assert model._device == device

    # ==========================================
    # Tests for _get_cache_dir (All conditions covered)
    # ==========================================
    def test_get_cache_dir_from_settings(self, mock_settings: MagicMock) -> None:
        """Test cache dir retrieval when specified in settings."""
        expected_path = Path("/custom/cache/path")
        mock_settings.classification_cache_dir = expected_path
        model = TorchvisionClassification(settings=mock_settings, model_name="resnet18")
        assert model._get_cache_dir() == expected_path

    @patch.object(Path, "home", return_value=Path("/fake/home"))
    def test_get_cache_dir_fallback(
        self, mock_home: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test cache dir fallback to default path when not in settings."""
        mock_settings.classification_cache_dir = None
        model = TorchvisionClassification(settings=mock_settings, model_name="resnet18")
        expected_path = Path("/fake/home") / ".cache" / "xwhy" / "classification"
        assert model._get_cache_dir() == expected_path

    # ==========================================
    # Tests for load (All conditions covered)
    # ==========================================
    def test_load_already_loaded(self, mock_settings: MagicMock) -> None:
        """Test that load returns early if model is already loaded."""
        model = TorchvisionClassification(settings=mock_settings, model_name="resnet18")
        model._model = MagicMock()
        model._preprocess = MagicMock()

        with patch.object(model, "_set_seed") as mock_set_seed:
            prep, mod = model.load()
            assert prep == model._preprocess
            assert mod == model._model
            mock_set_seed.assert_not_called()

    @patch.dict(os.environ, {}, clear=False)
    def test_load_fresh(self, mock_settings: MagicMock) -> None:
        """Test loading a model from scratch securely."""
        mock_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_builder.return_value.to.return_value = mock_model_instance

        mock_weights = MagicMock()
        mock_transforms = MagicMock()
        mock_weights.transforms.return_value = mock_transforms

        dummy_registry = {"dummy_model": (mock_builder, mock_weights)}

        with patch.object(TorchvisionClassification, "_MODEL_REGISTRY", dummy_registry):
            model = TorchvisionClassification(
                settings=mock_settings, model_name="dummy_model", device="cpu"
            )

            with patch.object(model, "_get_cache_dir") as mock_get_cache:
                mock_cache_dir = MagicMock()
                mock_get_cache.return_value = mock_cache_dir

                with patch.object(model, "_set_seed") as mock_set_seed:
                    prep, mod = model.load()

                    mock_set_seed.assert_called_once()
                    mock_get_cache.assert_called_once()
                    mock_cache_dir.mkdir.assert_called_once_with(
                        parents=True, exist_ok=True
                    )

                    assert os.environ.get("TORCH_HOME") == str(mock_cache_dir)

                    mock_builder.assert_called_once_with(weights=mock_weights)
                    mock_model_instance.eval.assert_called_once()

                    assert prep == mock_transforms
                    assert mod == mock_model_instance
                    assert model._model == mock_model_instance
                    assert model._preprocess == mock_transforms
