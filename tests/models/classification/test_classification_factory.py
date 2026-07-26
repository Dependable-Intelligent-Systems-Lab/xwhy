"""Tests for ClassificationFactory."""

from __future__ import annotations

import pytest

from xwhy.models.classification.factory import ClassificationFactory
from xwhy.models.classification.torchvision_models import TorchvisionClassification
from xwhy.models.classification.types import ClassificationType


def test_register_and_create_classification() -> None:
    """Register and create classification successfully."""
    ClassificationFactory.clear()

    def _builder(**kwargs: object) -> TorchvisionClassification:
        return TorchvisionClassification(settings=kwargs["settings"])  # type: ignore[arg-type]

    ClassificationFactory.register(
        classification=ClassificationType.INCEPTION_V3,
        builder=_builder,
    )

    classification = ClassificationFactory.create(
        classification=ClassificationType.INCEPTION_V3,
        settings=object(),
    )

    assert isinstance(classification, TorchvisionClassification)


def test_register_duplicate_classification() -> None:
    """Ensure duplicate registration raises error."""
    ClassificationFactory.clear()

    ClassificationFactory.register(
        classification=ClassificationType.INCEPTION_V3,
        builder=lambda **kwargs: TorchvisionClassification(settings=kwargs["settings"]),
    )

    with pytest.raises(ValueError, match="already registered"):
        ClassificationFactory.register(
            classification=ClassificationType.INCEPTION_V3,
            builder=lambda **kwargs: TorchvisionClassification(
                settings=kwargs["settings"]
            ),
        )


def test_unsupported_classification_raises() -> None:
    """Test that unsupported classification type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported classification"):
        ClassificationFactory.create(
            classification="non_existent_classification",  # type: ignore[arg-type]
            settings=object(),
        )
