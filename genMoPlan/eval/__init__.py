"""Evaluation utilities for genMoPlan."""

from genMoPlan.eval.classifier import Classifier

# Backward-compatible alias
ROAEstimator = Classifier

__all__ = ["Classifier", "ROAEstimator"]
