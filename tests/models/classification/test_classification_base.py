"""Test base classification module."""

from collections.abc import Callable
from typing import Any

from xwhy.models.classification.base import BaseClassification


class DummyClassification(BaseClassification):
    """Dummy subclass to test BaseClassification properties and methods."""

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Return a dummy model."""
        return "dummy_model"

    @property
    def weights(self) -> Any:  # noqa: ANN401
        """Return a dummy weights."""
        return "dummy_weights"

    @property
    def preprocess_fn(self) -> Callable[..., Any] | None:
        """Return a dummy preprocess function."""
        return None

    def __call__(self, inputs: Any) -> Any:  # noqa: ANN401
        """Execute a dummy call method."""
        return "called"

    def load(self, force_download: bool = False) -> Any:  # noqa: ANN401
        """Mock the abstract load method."""
        return None

    def predict(self, inputs: Any) -> Any:  # noqa: ANN401
        """Mock the abstract predict method."""
        return None


def test_base_classification_methods() -> None:
    """Test properties and call method of BaseClassification."""
    obj = DummyClassification()

    assert obj.model == "dummy_model"
    assert obj.weights == "dummy_weights"
    assert obj.preprocess_fn is None
    assert obj(inputs="test") == "called"


def test_base_classification_abstract_methods() -> None:
    """Test abstract methods in BaseClassification to cover pass bodies."""
    # Call original abstract properties and methods directly to execute 'pass'
    assert BaseClassification.weights.fget(None) is None  # type: ignore[attr-defined]
    assert BaseClassification.model.fget(None) is None  # type: ignore[attr-defined]
    assert BaseClassification.preprocess_fn.fget(None) is None  # type: ignore[attr-defined]
    assert BaseClassification.__call__(None, inputs="dummy") is None  # type: ignore[arg-type]
