"""Unit tests for classification types."""

import pytest

from xwhy.models.classification.types import ClassificationType


def test_classification_type_from_str_success() -> None:
    """Test successful conversion from valid strings."""
    assert (
        ClassificationType.from_str("inception_v3") == ClassificationType.INCEPTION_V3
    )
    assert ClassificationType.from_str("resnet18") == ClassificationType.RESNET18
    assert ClassificationType.from_str("resnet50") == ClassificationType.RESNET50
    assert (
        ClassificationType.from_str("mobilenet_v3") == ClassificationType.MOBILENET_V3
    )
    assert ClassificationType.from_str("vit_base") == ClassificationType.VIT_BASE


def test_classification_type_from_str_invalid() -> None:
    """Test that invalid input raises ValueError with a clear message."""
    invalid_input = "invalid_classification"

    with pytest.raises(
        ValueError, match=f"'{invalid_input}' is not a valid ClassificationType"
    ):
        ClassificationType.from_str(invalid_input)


class TestClassificationTypeProperties:
    """Test class for testing classification properties."""

    @pytest.mark.parametrize(
        ("model_type", "expected_is_cnn", "expected_is_transformer"),
        [
            (ClassificationType.INCEPTION_V3, True, False),
            (ClassificationType.RESNET18, True, False),
            (ClassificationType.RESNET50, True, False),
            (ClassificationType.MOBILENET_V3, True, False),
            (ClassificationType.VIT_BASE, False, True),
        ],
    )
    def test_classification_properties(
        self,
        model_type: ClassificationType,
        expected_is_cnn: bool,
        expected_is_transformer: bool,
    ) -> None:
        """Test is_cnn and is_transformer properties for all classification types."""
        assert model_type.is_cnn is expected_is_cnn
        assert model_type.is_transformer is expected_is_transformer
