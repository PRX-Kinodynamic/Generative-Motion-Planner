"""Shared prediction label types for threshold and conformal prediction."""

from enum import IntEnum


class PredictionLabel(IntEnum):
    """Labels produced by threshold optimization and conformal prediction.

    FAILURE = 0: Predicted outside RoA (negative class)
    SUCCESS = 1: Predicted inside RoA (positive class)
    UNCERTAIN = -1: Model not confident (in lambda +/- delta band,
                    or non-singleton prediction set)
    INVALID = -2: High p_invalid (endpoint physically meaningless)
    """

    FAILURE = 0
    SUCCESS = 1
    UNCERTAIN = -1
    INVALID = -2
