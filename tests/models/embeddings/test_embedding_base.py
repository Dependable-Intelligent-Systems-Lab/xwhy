"""Test embeddings base module."""

from typing import Any

from xwhy.models.embeddings.base import BaseEmbedding


class DummyEmbedding(BaseEmbedding):
    """Dummy subclass to test BaseEmbedding properties and methods."""

    @property
    def model(self) -> Any:  # noqa: ANN401
        """Return a dummy model."""
        return "dummy_model"

    @property
    def processor(self) -> Any:  # noqa: ANN401
        """Return a dummy processor."""
        return "dummy_processor"

    def __call__(self, inputs: Any) -> Any:  # noqa: ANN401
        """Execute a dummy call method."""
        return "called"

    def load(self, force_download: bool = False) -> Any:  # noqa: ANN401
        """Mock the abstract load method."""
        return None

    def encode(self, inputs: Any) -> Any:  # noqa: ANN401
        """Mock the abstract encode method."""
        return None


def test_base_embedding_methods() -> None:
    """Test properties and call method of BaseEmbedding."""
    obj = DummyEmbedding()

    assert obj.model == "dummy_model"
    assert obj.processor == "dummy_processor"
    assert obj(inputs="test") == "called"


def test_base_embedding_abstract_methods() -> None:
    """Test abstract methods in BaseEmbedding to cover pass bodies."""
    # Call original abstract properties and methods directly to execute 'pass'
    assert BaseEmbedding.model.fget(None) is None  # type: ignore[attr-defined]
    assert BaseEmbedding.processor.fget(None) is None  # type: ignore[attr-defined]
    assert BaseEmbedding.__call__(None, inputs="dummy") is None  # type: ignore[arg-type]
