"""Unit tests for custom PyTorch classification model adapter implementation."""

from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import torch
from torchvision import transforms

from xwhy.models.classification.custom_torch import (
    CustomTorchClassification,
    DynamicCategories,
    MockWeights,
    PreprocessWrapper,
)


class DummyModel(torch.nn.Module):
    """Dummy PyTorch model for testing purposes."""

    def __init__(self) -> None:
        """Initialize dummy model layers."""
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Execute forward pass."""
        return self.linear(x)  # type: ignore[no-any-return]


def test_dynamic_categories_with_valid_index() -> None:
    """Verify category retrieval with a valid index."""
    cats = ["Cat", "Dog"]
    dyn = DynamicCategories(cats)
    assert dyn[0] == "Cat"
    assert dyn[1] == "Dog"


def test_dynamic_categories_out_of_bounds() -> None:
    """Verify fallback string generation for out-of-bounds index."""
    cats = ["Cat"]
    dyn = DynamicCategories(cats)
    assert dyn[5] == "Class 5"


def test_dynamic_categories_none() -> None:
    """Verify fallback string generation when categories are None."""
    dyn = DynamicCategories(None)
    assert dyn[0] == "Class 0"


def test_mock_weights_with_meta_categories() -> None:
    """Initialize MockWeights using weights object with meta categories."""
    weights_obj = MagicMock()
    weights_obj.meta = {"categories": ["A", "B"]}
    mw = MockWeights(weights_obj)
    assert mw.meta["categories"][0] == "A"


def test_mock_weights_with_categories_attr() -> None:
    """Initialize MockWeights using weights object with categories attribute."""
    weights_obj = MagicMock(spec=[])
    weights_obj.categories = ["X", "Y"]
    mw = MockWeights(weights_obj)
    assert mw.meta["categories"][0] == "X"


def test_mock_weights_with_missing_meta() -> None:
    """Initialize MockWeights when weights object has no meta or categories."""
    weights_obj = MagicMock(spec=[])
    weights_obj.meta = None
    mw = MockWeights(weights_obj, categories=["Fallback"])
    assert mw.meta["categories"][0] == "Fallback"


def test_mock_weights_none_with_categories() -> None:
    """Initialize MockWeights when weights_obj is None with categories."""
    mw = MockWeights(None, categories=["One", "Two"])
    assert mw.meta["categories"][0] == "One"


def test_mock_weights_none_without_categories() -> None:
    """Initialize MockWeights when weights_obj is None without categories."""
    mw = MockWeights(None, None)
    assert mw.meta["categories"][0] == "Class 0"


def test_mock_weights_getattr_proxy() -> None:
    """Verify attribute access proxying to underlying weights object."""
    weights_obj = MagicMock()
    weights_obj.custom_attr = "test_value"
    mw = MockWeights(weights_obj)
    assert mw.custom_attr == "test_value"


def test_preprocess_wrapper_model_stats() -> None:
    """Extract mean and std directly from model attributes."""
    model = MagicMock()
    model.mean = [0.1, 0.2, 0.3]
    model.std = [0.4, 0.5, 0.6]
    transform_fn = lambda x: x  # noqa: E731
    wrapper = PreprocessWrapper(transform_fn, model=model)
    assert wrapper.mean == [0.1, 0.2, 0.3]
    assert wrapper.std == [0.4, 0.5, 0.6]


def test_preprocess_wrapper_transform_stats() -> None:
    """Extract mean and std directly from transform attributes."""
    transform_fn = MagicMock()
    transform_fn.mean = [0.7, 0.8, 0.9]
    transform_fn.std = [0.1, 0.2, 0.3]
    wrapper = PreprocessWrapper(transform_fn)
    assert wrapper.mean == [0.7, 0.8, 0.9]
    assert wrapper.std == [0.1, 0.2, 0.3]


def test_preprocess_wrapper_normalize_layer() -> None:
    """Extract mean and std from transforms.Normalize layer in Compose."""
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform_fn = transforms.Compose([transforms.ToTensor(), normalize])
    wrapper = PreprocessWrapper(transform_fn)
    assert wrapper.mean == [0.5, 0.5, 0.5]
    assert wrapper.std == [0.5, 0.5, 0.5]


def test_preprocess_wrapper_defaults_and_call() -> None:
    """Verify default stats and call execution on PreprocessWrapper."""
    call_tracker: list[Any] = []

    def transform_fn(x: Any) -> str:  # noqa: ANN401
        call_tracker.append(x)
        return "transformed"

    wrapper = PreprocessWrapper(transform_fn)
    assert wrapper.mean == [0.0, 0.0, 0.0]
    assert wrapper.std == [1.0, 1.0, 1.0]
    assert wrapper("input") == "transformed"
    assert call_tracker == ["input"]


@patch("torch.cuda.is_available", return_value=True)
def test_custom_torch_classification_init_cuda(mock_cuda: MagicMock) -> None:
    """Initialize CustomTorchClassification with CUDA device auto-selection."""
    model = DummyModel()
    adapter = CustomTorchClassification(model=model, device=None)
    assert adapter._device.type == "cuda"
    mock_cuda.assert_called_once()


def test_custom_torch_classification_init_device_string() -> None:
    """Initialize CustomTorchClassification with a device string."""
    model = DummyModel()
    adapter = CustomTorchClassification(model=model, device="cpu")
    assert adapter._device.type == "cpu"


def test_custom_torch_classification_init_device_object() -> None:
    """Initialize CustomTorchClassification with a torch.device object."""
    model = DummyModel()
    dev = torch.device("cpu")
    adapter = CustomTorchClassification(model=model, device=dev)
    assert adapter._device == dev


def test_custom_torch_classification_model_attributes() -> None:
    """Resolve preprocessing and categories from model attributes."""
    model = DummyModel()
    model.categories = ["A", "B"]  # type: ignore[assignment]
    model.preprocess_fn = lambda x: x  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model)
    assert adapter.weights.meta["categories"][0] == "A"
    assert adapter.model == model
    assert adapter.preprocess_fn is not None


def test_custom_torch_classification_model_preprocess_and_transforms() -> None:
    """Resolve preprocessing using preprocess and transforms attributes."""
    model1 = DummyModel()
    model1.preprocess = lambda x: x  # type: ignore[assignment]
    adapter1 = CustomTorchClassification(model=model1)
    assert adapter1.preprocess_fn is not None

    model2 = DummyModel()
    model2.transforms = lambda x: x  # type: ignore[assignment]
    adapter2 = CustomTorchClassification(model=model2)
    assert adapter2.preprocess_fn is not None


def test_custom_torch_classification_native_weights_transforms() -> None:
    """Resolve preprocessing from native weights transforms successfully or failing."""
    model = DummyModel()
    native_weights = MagicMock()
    native_weights.transforms.return_value = lambda x: x
    model.weights = native_weights
    adapter = CustomTorchClassification(model=model)
    assert adapter.preprocess_fn is not None

    # Test failure branch in suppress
    native_weights.transforms.side_effect = Exception("Failed")
    adapter_fail = CustomTorchClassification(model=model)
    assert adapter_fail.preprocess_fn is not None


def test_custom_torch_classification_execution_methods() -> None:
    """Test call, load, and predict methods on CustomTorchClassification."""
    model = DummyModel()
    adapter = CustomTorchClassification(model=model, device="cpu")

    inputs = torch.randn(1, 2)
    outputs = adapter(inputs)
    assert outputs.shape == torch.Size([1, 2])

    prep, mdl = adapter.load()
    assert prep is not None
    assert mdl == model
    assert not model.training

    logits = adapter.predict(inputs)
    assert logits.shape == torch.Size([1, 2])


class ObjectWithoutMeta:
    """Helper class without meta attribute for testing categories extraction."""

    categories: ClassVar[list[str]] = ["Class1", "Class2"]


class NonCallableTransformsWeights:
    """Helper class with non-callable transforms attribute."""

    transforms = "not_a_callable"


def test_mock_weights_extraction_via_categories_attr_explicit() -> None:
    """Extract categories using a custom object without meta attribute."""
    obj = ObjectWithoutMeta()
    mw = MockWeights(obj, categories=None)
    assert mw.meta["categories"][0] == "Class1"


def test_custom_torch_classification_non_callable_transforms() -> None:
    """Handle non-callable transforms attribute in native weights gracefully."""
    model = DummyModel()
    model.weights = NonCallableTransformsWeights()  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn is not None


def test_custom_torch_classification_all_preprocess_attributes_none() -> None:
    """Test default transform fallback when all attributes are None."""
    model = DummyModel()
    model.preprocess_fn = None  # type: ignore[assignment]
    model.preprocess = None  # type: ignore[assignment]
    model.transforms = None  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn is not None


def test_mock_weights_meta_contains_categories() -> None:
    """Verify categories extraction when meta has categories key."""
    weights_obj = MagicMock()
    weights_obj.meta = {"categories": ["Cat1", "Cat2"]}
    mw = MockWeights(weights_obj, categories=None)
    assert mw.meta["categories"][0] == "Cat1"


def test_mock_weights_meta_lacks_categories() -> None:
    """Verify fallback when meta dictionary lacks categories key."""
    weights_obj = MagicMock()
    weights_obj.meta = {"other_key": 123}
    mw = MockWeights(weights_obj, categories=None)
    assert mw.meta["categories"][0] == "Class 0"


def test_custom_torch_classification_preprocess_fn_attr() -> None:
    """Resolve preprocessing using model.preprocess_fn attribute."""
    model = DummyModel()
    custom_func = lambda x: x  # noqa: E731
    model.preprocess_fn = custom_func  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn.transform_fn == custom_func  # type: ignore[union-attr]


def test_custom_torch_classification_preprocess_attr() -> None:
    """Resolve preprocessing using model.preprocess attribute."""
    model = DummyModel()
    custom_func = lambda x: x  # noqa: E731
    model.preprocess = custom_func  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn.transform_fn == custom_func  # type: ignore[union-attr]


def test_custom_torch_classification_transforms_attr() -> None:
    """Resolve preprocessing using model.transforms attribute."""
    model = DummyModel()
    custom_func = lambda x: x  # noqa: E731
    model.transforms = custom_func  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn.transform_fn == custom_func  # type: ignore[union-attr]


def test_custom_torch_classification_native_weights_transforms_success() -> None:
    """Resolve preprocessing using native_weights.transforms callable."""
    model = DummyModel()
    custom_func = lambda x: x  # noqa: E731
    native_weights = MagicMock()
    native_weights.transforms.return_value = custom_func
    model.weights = native_weights
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn.transform_fn == custom_func  # type: ignore[union-attr]


def test_custom_torch_classification_preprocess_explicit() -> None:
    """Test explicit preprocess_fn argument bypasses resolution block."""
    model = DummyModel()
    custom_func = lambda x: x  # noqa: E731
    adapter = CustomTorchClassification(model=model, preprocess_fn=custom_func)
    assert adapter.preprocess_fn.transform_fn == custom_func  # type: ignore[union-attr]


def test_custom_torch_classification_attributes_explicitly_none() -> None:
    """Test resolution when model attributes are explicitly set to None."""
    model = DummyModel()
    model.preprocess_fn = None  # type: ignore[assignment]
    model.preprocess = None  # type: ignore[assignment]
    model.transforms = None  # type: ignore[assignment]
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn is not None


def test_custom_torch_classification_native_weights_transforms_none() -> None:
    """Test resolution when native_weights.transforms is explicitly None."""
    model = DummyModel()
    native_weights = MagicMock()
    native_weights.transforms = None
    model.weights = native_weights
    adapter = CustomTorchClassification(model=model, preprocess_fn=None)
    assert adapter.preprocess_fn is not None
