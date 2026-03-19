"""Tests for genMoPlan.eval.types."""

import pytest
from genMoPlan.eval.types import PredictionLabel


def test_prediction_label_values():
    assert PredictionLabel.FAILURE == 0
    assert PredictionLabel.SUCCESS == 1
    assert PredictionLabel.UNCERTAIN == -1
    assert PredictionLabel.INVALID == -2


def test_prediction_label_is_int():
    assert isinstance(PredictionLabel.SUCCESS, int)
    assert int(PredictionLabel.SUCCESS) == 1
    assert int(PredictionLabel.FAILURE) == 0
