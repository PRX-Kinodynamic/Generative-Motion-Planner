"""Evaluation utilities for genMoPlan."""

from genMoPlan.eval.classifier import Classifier
from genMoPlan.eval.types import PredictionLabel

# Backward-compatible alias
ROAEstimator = Classifier

__all__ = ["Classifier", "ROAEstimator", "PredictionLabel"]
