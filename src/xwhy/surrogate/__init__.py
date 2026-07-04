"""Surrogate module."""

from .base import BaseSurrogate
from .factory import SurrogateFactory
from .linear import LinearRegressionSurrogate
from .trainer import SurrogateTrainer
from .tree import TreeBasedSurrogate
from .types import SurrogateType

__all__ = [
    "BaseSurrogate",
    "LinearRegressionSurrogate",
    "SurrogateFactory",
    "SurrogateTrainer",
    "SurrogateType",
    "TreeBasedSurrogate",
]
