"""Unit tests for segmentation torchvision model."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from xwhy.config.settings import Settings
from xwhy.models.segmentation.torchvision_models import TorchvisionSegmentation


@pytest.fixture
def mock_settings() -> Settings:
    """Fixture to provide a mocked settings object."""
    settings = MagicMock()
    settings.segmentation_cache_dir = None
    return settings


class TestTorchvisionSegmentation:
    """Test class to test all methods."""

    def test_init_invalid_model(self, mock_settings: MagicMock) -> None:
        """Verify initialization fails for invalid model names."""
        with pytest.raises(ValueError, match="Unsupported model 'invalid_model'"):
            TorchvisionSegmentation(settings=mock_settings, model_name="invalid_model")

    @patch("torch.cuda.is_available", return_value=True)
    def test_init_device_none_with_cuda(
        self, mock_cuda: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Check device is 'cuda' when cuda is available."""
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", device=None
        )
        assert model._device.type == "cuda"

    # ==========================================
    # Tests for class_names property
    # ==========================================
    def test_class_names_loads_if_empty(self, mock_settings: MagicMock) -> None:
        """Ensure load() is called if class names are empty."""
        model = TorchvisionSegmentation(settings=mock_settings)
        # Mocking load to avoid real logic
        with patch.object(model, "load") as mock_load:
            model._class_names = []  # Ensure it triggers load
            _ = model.class_names
            mock_load.assert_called_once()

    # ==========================================
    # Tests for _get_cache_dir
    # ==========================================
    def test_get_cache_dir_from_settings(self, mock_settings: MagicMock) -> None:
        """Verify cache directory is retrieved from settings."""
        expected_path = Path("/custom/segmentation/cache")
        mock_settings.segmentation_cache_dir = expected_path
        model = TorchvisionSegmentation(settings=mock_settings)
        assert model._get_cache_dir() == expected_path

    @patch.object(Path, "home", return_value=Path("/fake/home"))
    def test_get_cache_dir_fallback(
        self, mock_home: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Ensure correct fallback path if settings are missing."""
        model = TorchvisionSegmentation(settings=mock_settings)
        expected_path = Path("/fake/home") / ".cache" / "xwhy" / "segmentation"
        assert model._get_cache_dir() == expected_path

    # ==========================================
    # Tests for load
    # ==========================================
    @patch.dict(os.environ, {}, clear=False)
    def test_load_fresh(self, mock_settings: MagicMock) -> None:
        """Test loading a model from scratch."""
        mock_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_builder.return_value.to.return_value = mock_model_instance

        # Mock weights with meta categories
        mock_weights = MagicMock()
        mock_weights.transforms.return_value = "transforms"
        mock_weights.meta = {"categories": ["background", "person"]}

        dummy_registry = {"dummy_seg": (mock_builder, mock_weights)}

        with patch.object(TorchvisionSegmentation, "_MODEL_REGISTRY", dummy_registry):
            model = TorchvisionSegmentation(
                settings=mock_settings, model_name="dummy_seg", device="cpu"
            )

            with patch.object(model, "_get_cache_dir") as mock_get_cache:
                mock_cache = MagicMock()
                mock_get_cache.return_value = mock_cache

                prep, mod = model.load()

                assert prep == "transforms"
                assert mod == mock_model_instance
                assert model._class_names == ["background", "person"]
                mock_cache.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # ==========================================
    # Tests for predict
    # ==========================================
    def test_predict_returns_out_key(self, mock_settings: MagicMock) -> None:
        """Test predict returns out key."""
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", device="cpu"
        )

        mock_model = MagicMock()
        # Torchvision segmentation models return a dict with an "out" key
        expected_logits = torch.tensor([[[[1.0]]]])
        mock_model.return_value = {"out": expected_logits}
        model._model = mock_model

        dummy_input = MagicMock()
        dummy_input.to.return_value = dummy_input

        result = model.predict(dummy_input)

        assert torch.equal(result, expected_logits)

    def test_class_names_already_populated(self, mock_settings: MagicMock) -> None:
        """Test that class_names returns immediately if already populated."""
        model = TorchvisionSegmentation(settings=mock_settings)
        model._class_names = ["class1", "class2"]

        with patch.object(model, "load") as mock_load:
            result = model.class_names
            assert result == ["class1", "class2"]
            mock_load.assert_not_called()

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    def test_set_seed_no_cuda(
        self,
        mock_cuda_seed: MagicMock,
        mock_seed: MagicMock,
        mock_cuda_avail: MagicMock,
        mock_settings: MagicMock,
    ) -> None:
        """Test _set_seed does not attempt to configure CUDA if unavailable."""
        model = TorchvisionSegmentation(settings=mock_settings, seed=42)
        model._set_seed()

        mock_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_not_called()

    def test_load_already_loaded_returns_early(self, mock_settings: MagicMock) -> None:
        """Test load returns early if model and preprocess are already set."""
        model = TorchvisionSegmentation(settings=mock_settings)
        # Mocking existence of resources
        model._model = "fake_model"
        model._preprocess = "fake_preprocess"

        with patch.object(model, "_set_seed") as mock_set_seed:
            prep, mod = model.load()

            assert prep == "fake_preprocess"
            assert mod == "fake_model"
            mock_set_seed.assert_not_called()

    def test_predict_model_already_exists(self, mock_settings: MagicMock) -> None:
        """Test predict does not call load() if model is already set."""
        model = TorchvisionSegmentation(settings=mock_settings, device="cpu")

        # Setup existing model
        mock_model = MagicMock()
        mock_model.return_value = {"out": torch.tensor([[[[1.0]]]])}
        model._model = mock_model

        dummy_input = torch.randn(1, 3, 32, 32)

        with patch.object(model, "load") as mock_load:
            model.predict(dummy_input)

            # Verify load was NOT called because model exists
            mock_load.assert_not_called()
            # Verify model was used
            mock_model.assert_called_once()


class TestTorchvisionSegmentationMethods:
    """Test internal methods, load logic, and prediction paths."""

    @pytest.fixture
    def instance(self) -> TorchvisionSegmentation:
        """Provide a mocked TorchvisionSegmentation instance."""
        obj = TorchvisionSegmentation.__new__(TorchvisionSegmentation)
        obj._model = None
        obj._model_name = "test_seg"
        obj._preprocess = MagicMock()
        obj._device = torch.device("cpu")
        return obj

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
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", seed=123
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
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", seed=123
        )
        model._set_seed()

        mock_manual_seed.assert_called_once_with(123)
        mock_cuda_seed.assert_not_called()

    def test_predict_already_loaded(self, mock_settings: MagicMock) -> None:
        """Test predict method when model is already initialized."""
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", device="cpu"
        )

        mock_model = MagicMock()

        expected_logits = torch.randn(1, 21, 224, 224)
        mock_model.return_value = {"out": expected_logits}

        model._model = mock_model

        dummy_input = torch.randn(1, 3, 224, 224)
        result = model.predict(dummy_input)

        mock_model.assert_called_once()

        assert torch.equal(result, expected_logits)

    @patch.object(TorchvisionSegmentation, "load")
    def test_predict_calls_load_if_missing(
        self, mock_load: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test predict method calls load() if model is None."""
        model = TorchvisionSegmentation(
            settings=mock_settings, model_name="deeplabv3_resnet101", device="cpu"
        )

        mock_model = MagicMock()

        expected_logits = torch.randn(1, 21, 224, 224)
        mock_model.return_value = {"out": expected_logits}

        mock_load.return_value = (MagicMock(), mock_model)

        dummy_input = torch.randn(1, 3, 224, 224)
        result = model.predict(dummy_input)

        mock_load.assert_called_once()
        mock_model.assert_called_once()
        assert torch.equal(result, expected_logits)

    def test_model_property_unloaded(self, instance: TorchvisionSegmentation) -> None:
        """Test model property raises RuntimeError when unloaded."""
        with pytest.raises(RuntimeError, match="not loaded"):
            _ = instance.model

    def test_model_property_loaded(self, instance: TorchvisionSegmentation) -> None:
        """Test model property returns the loaded model."""
        instance._model = "dummy_model"
        assert instance.model == "dummy_model"

    def test_preprocess_fn(self, instance: TorchvisionSegmentation) -> None:
        """Test preprocess_fn returns the internal preprocess function."""
        assert instance.preprocess_fn == instance._preprocess

    def test_call_unloaded_dict_output(self, instance: TorchvisionSegmentation) -> None:
        """Test __call__ loads model and extracts 'out' key from dict."""
        mock_model = MagicMock(return_value={"out": "logits"})
        instance.load = MagicMock(return_value=(None, mock_model))  # type: ignore[method-assign]

        mock_input = MagicMock()
        mock_input.to.return_value = "device_inputs"

        result = instance(inputs=mock_input)

        instance.load.assert_called_once()
        mock_input.to.assert_called_once_with(torch.device("cpu"))
        assert result == "logits"

    def test_call_loaded_tensor_output(self, instance: TorchvisionSegmentation) -> None:
        """Test __call__ uses existing model and returns tensor directly."""
        mock_model = MagicMock(return_value="tensor_logits")
        instance._model = mock_model

        mock_input = MagicMock()
        mock_input.to.return_value = "device_inputs"

        result = instance(inputs=mock_input)

        mock_model.assert_called_once_with("device_inputs")
        assert result == "tensor_logits"
