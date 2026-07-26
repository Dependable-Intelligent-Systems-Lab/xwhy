"""Classification type definitions."""

from __future__ import annotations

from enum import StrEnum


class ClassificationType(StrEnum):
    """Supported classification backends."""

    INCEPTION_V3 = "inception_v3"
    RESNET18 = "resnet18"
    RESNET50 = "resnet50"
    MOBILENET_V3 = "mobilenet_v3"
    VIT_BASE = "vit_base"

    @classmethod
    def from_str(cls, value: str | ClassificationType) -> ClassificationType:
        """Safely convert a string or enum instance to ClassificationType."""
        try:
            return cls(value)
        except ValueError as err:
            valid_options = ", ".join([item.value for item in cls])
            raise ValueError(
                f"'{value}' is not a valid ClassificationType. "
                f"Supported options are: [{valid_options}]"
            ) from err

    @property
    def is_cnn(self) -> bool:
        """Check if the model uses a Convolutional Neural Network architecture."""
        return self in {
            ClassificationType.INCEPTION_V3,
            ClassificationType.RESNET18,
            ClassificationType.RESNET50,
            ClassificationType.MOBILENET_V3,
        }

    @property
    def is_transformer(self) -> bool:
        """Check if the model uses a Vision Transformer architecture."""
        return self in {ClassificationType.VIT_BASE}
