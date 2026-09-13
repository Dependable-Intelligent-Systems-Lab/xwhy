"""Unit tests for PointCloudModelFactory."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

from xwhy.models.point_cloud.base import BasePointCloudModel
from xwhy.models.point_cloud.factory import PointCloudModelFactory
from xwhy.models.point_cloud.types import PointCloudModelType


class DummyModel(BasePointCloudModel):
    """Minimal concrete model used only for factory tests."""

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        """Store arbitrary kwargs for later inspection."""
        self.kwargs = kwargs

    def predict(
        self,
        sample_input: Any,  # noqa: ANN401
        sample_label: int | None = None,
    ) -> tuple[int, Any, list[int]]:
        """Return a dummy prediction tuple."""
        return 0, None, [0]

    def get_output_probabilities(
        self,
        samples: list[Any],
        device: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Return a dummy probability tensor."""
        return None


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None, None, None]:
    """Ensure the factory registry is empty before and after every test."""
    PointCloudModelFactory.clear()
    yield
    PointCloudModelFactory.clear()


def test_register_and_create_success() -> None:
    """Register a builder and instantiate the corresponding model."""

    def builder(**kwargs: Any) -> BasePointCloudModel:  # noqa: ANN401
        return DummyModel(**kwargs)

    PointCloudModelFactory.register(PointCloudModelType.CUSTOM, builder)
    model = PointCloudModelFactory.create(PointCloudModelType.CUSTOM, alpha=1.5)

    assert isinstance(model, DummyModel)
    assert model.kwargs == {"alpha": 1.5}


def test_register_duplicate_raises() -> None:
    """Raise ValueError when the same model type is registered twice."""

    def builder(**_: Any) -> BasePointCloudModel:  # noqa: ANN401
        return DummyModel()

    PointCloudModelFactory.register(PointCloudModelType.CUSTOM, builder)
    with pytest.raises(ValueError, match="Model type already registered"):
        PointCloudModelFactory.register(PointCloudModelType.CUSTOM, builder)


def test_create_unregistered_raises() -> None:
    """Raise ValueError when create is called for an unknown type."""
    with pytest.raises(ValueError, match="Unsupported point cloud model type"):
        PointCloudModelFactory.create(PointCloudModelType.CUSTOM)


def test_clear_empties_registry() -> None:
    """Remove every registered builder from the registry."""

    def builder(**_: Any) -> BasePointCloudModel:  # noqa: ANN401
        return DummyModel()

    PointCloudModelFactory.register(PointCloudModelType.CUSTOM, builder)
    assert PointCloudModelType.CUSTOM in PointCloudModelFactory._registry

    PointCloudModelFactory.clear()
    assert PointCloudModelFactory._registry == {}
