"""Tests for genMoPlan.eval.metrics."""

import numpy as np
import pytest

from genMoPlan.eval.types import PredictionLabel
from genMoPlan.eval.metrics import (
    compute_classification_metrics,
    compute_coverage,
    make_conservative_labels,
)


class TestComputeClassificationMetrics:
    def test_perfect_classification(self):
        pred = np.array([1, 1, 0, 0], dtype=np.int32)
        expected = np.array([1, 1, 0, 0], dtype=np.int32)
        result = compute_classification_metrics(pred, expected)

        assert result["true_positives"] == 2
        assert result["true_negatives"] == 2
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["accuracy"] == 1.0
        assert result["f1_score"] == 1.0

    def test_all_wrong(self):
        pred = np.array([0, 0, 1, 1], dtype=np.int32)
        expected = np.array([1, 1, 0, 0], dtype=np.int32)
        result = compute_classification_metrics(pred, expected)

        assert result["true_positives"] == 0
        assert result["true_negatives"] == 0
        assert result["false_positives"] == 2
        assert result["false_negatives"] == 2
        assert result["accuracy"] == 0.0

    def test_with_uncertain_and_invalid(self):
        pred = np.array(
            [1, 0, PredictionLabel.UNCERTAIN, PredictionLabel.INVALID, 1],
            dtype=np.int32,
        )
        expected = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        result = compute_classification_metrics(pred, expected)

        assert result["true_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["false_positives"] == 1  # pred=1, actual=0
        assert result["false_negatives"] == 0
        assert result["separatrix_uncertain"] == 1
        assert result["separatrix_invalid"] == 1
        assert result["separatrix"] == 2

    def test_confusion_matrix_shape(self):
        pred = np.array([1, 0, -1, -2], dtype=np.int32)
        expected = np.array([1, 0, 1, 0], dtype=np.int32)
        result = compute_classification_metrics(pred, expected)

        cm = result["confusion_matrix"]
        assert len(cm) == 2  # 2 rows (actual neg, actual pos)
        assert len(cm[0]) == 4  # 4 cols (neg, pos, uncertain, invalid)
        assert len(cm[1]) == 4

    def test_empty_inputs(self):
        pred = np.array([], dtype=np.int32)
        expected = np.array([], dtype=np.int32)
        result = compute_classification_metrics(pred, expected)
        assert result["true_positives"] == 0
        assert result["separatrix"] == 0


class TestComputeCoverage:
    def test_perfect_coverage(self):
        # All true labels are in their prediction sets
        pred_sets = np.array([[True, False], [True, False], [False, True]], dtype=bool)
        true_labels = np.array([1, 1, 0], dtype=np.int32)
        assert compute_coverage(pred_sets, true_labels) == 1.0

    def test_no_coverage(self):
        pred_sets = np.array([[False, True], [False, True], [True, False]], dtype=bool)
        true_labels = np.array([1, 1, 0], dtype=np.int32)
        assert compute_coverage(pred_sets, true_labels) == 0.0

    def test_partial_coverage(self):
        pred_sets = np.array(
            [[True, False], [False, False], [False, True], [True, True]], dtype=bool
        )
        true_labels = np.array([1, 1, 0, 0], dtype=np.int32)
        # Point 0: success=True, label=1 -> covered
        # Point 1: success=False, label=1 -> NOT covered
        # Point 2: failure=True, label=0 -> covered
        # Point 3: failure=True, label=0 -> covered
        assert compute_coverage(pred_sets, true_labels) == 0.75

    def test_empty(self):
        pred_sets = np.empty((0, 2), dtype=bool)
        true_labels = np.array([], dtype=np.int32)
        result = compute_coverage(pred_sets, true_labels)
        assert np.isnan(result)


class TestMakeConservativeLabels:
    def test_uncertain_becomes_failure(self):
        labels = np.array(
            [PredictionLabel.SUCCESS, PredictionLabel.UNCERTAIN, PredictionLabel.FAILURE],
            dtype=np.int32,
        )
        result = make_conservative_labels(labels)
        assert result[0] == PredictionLabel.SUCCESS
        assert result[1] == PredictionLabel.FAILURE
        assert result[2] == PredictionLabel.FAILURE

    def test_invalid_becomes_failure(self):
        labels = np.array([PredictionLabel.INVALID], dtype=np.int32)
        result = make_conservative_labels(labels)
        assert result[0] == PredictionLabel.FAILURE

    def test_does_not_modify_original(self):
        labels = np.array([PredictionLabel.UNCERTAIN], dtype=np.int32)
        result = make_conservative_labels(labels)
        assert labels[0] == PredictionLabel.UNCERTAIN  # Original unchanged
        assert result[0] == PredictionLabel.FAILURE
