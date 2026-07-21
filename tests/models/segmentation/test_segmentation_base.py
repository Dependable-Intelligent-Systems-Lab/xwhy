"""Test segmentation base module."""

from collections.abc import Callable
from typing import Any

from xwhy.models.segmentation.base import BaseSegmentation


class DummySegmentation(BaseSegmentation):
    """Dummy subclass to test BaseSegmentation properties and methods."""

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Return a dummy model."""
        return "dummy_model"

    @property
    def preprocess_fn(self) -> Callable[..., Any] | None:
        """Return a dummy preprocess function."""
        return None

    @property
    def class_names(self) -> list[str]:
        """Dummy implementation of abstract class_names property."""
        return ["dummy_class"]

    def __call__(self, inputs: Any) -> Any:  # noqa: ANN401
        """Execute a dummy call method."""
        return "called"

    def load(self, force_download: bool = False) -> Any:  # noqa: ANN401
        """Mock the abstract load method."""
        return None

    def predict(self, inputs: Any) -> Any:  # noqa: ANN401
        """Mock the abstract predict method."""
        return None


def test_base_segmentation_methods() -> None:
    """Test properties and call method of BaseSegmentation."""
    obj = DummySegmentation()

    assert obj.model == "dummy_model"
    assert obj.preprocess_fn is None
    assert obj(inputs="test") == "called"


def test_base_segmentation_abstract_methods() -> None:
    """Test abstract methods in BaseSegmentation to cover pass bodies."""
    # Call original abstract properties and methods directly to execute 'pass'
    assert BaseSegmentation.model.fget(None) is None  # type: ignore[attr-defined]
    assert BaseSegmentation.preprocess_fn.fget(None) is None  # type: ignore[attr-defined]
    assert BaseSegmentation.__call__(None, inputs="dummy") is None  # type: ignore[arg-type]
