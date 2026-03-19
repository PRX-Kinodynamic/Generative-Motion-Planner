"""Reusable metric computation for classification and conformal prediction."""

import numpy as np

from genMoPlan.eval.types import PredictionLabel


def compute_classification_metrics(
    predicted_labels: np.ndarray,
    expected_labels: np.ndarray,
) -> dict:
    """Compute classification metrics from predicted vs expected labels.

    Handles 4-way labels: SUCCESS (1), FAILURE (0), UNCERTAIN (-1), INVALID (-2).
    Ground truth is always binary {0, 1}.

    Args:
        predicted_labels: (N,) array with PredictionLabel values.
        expected_labels: (N,) array with ground truth labels {0, 1}.

    Returns:
        dict with TP/TN/FP/FN, uncertain/invalid counts, rates,
        accuracy, precision, recall, specificity, f1_score,
        confusion matrix, and separatrix metrics.
    """
    gt_success = expected_labels == 1
    gt_failure = expected_labels == 0

    pred_success = predicted_labels == PredictionLabel.SUCCESS
    pred_failure = predicted_labels == PredictionLabel.FAILURE
    pred_uncertain = predicted_labels == PredictionLabel.UNCERTAIN
    pred_invalid = predicted_labels == PredictionLabel.INVALID

    tp = int(np.sum(pred_success & gt_success))
    tn = int(np.sum(pred_failure & gt_failure))
    fp = int(np.sum(pred_success & gt_failure))
    fn = int(np.sum(pred_failure & gt_success))

    n_uncertain = int(np.sum(pred_uncertain))
    n_invalid = int(np.sum(pred_invalid))
    n_separatrix = n_uncertain + n_invalid

    # Uncertain/invalid broken down by actual class
    uncertain_for_actual_pos = int(np.sum(pred_uncertain & gt_success))
    uncertain_for_actual_neg = int(np.sum(pred_uncertain & gt_failure))
    invalid_for_actual_pos = int(np.sum(pred_invalid & gt_success))
    invalid_for_actual_neg = int(np.sum(pred_invalid & gt_failure))

    n_total = tp + tn + fp + fn + n_separatrix

    # Rates
    tp_rate = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    tn_rate = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    separatrix_rate = n_separatrix / n_total if n_total > 0 else float("nan")
    uncertain_rate = n_uncertain / n_total if n_total > 0 else float("nan")
    invalid_rate = n_invalid / n_total if n_total > 0 else float("nan")

    # Metrics (excluding separatrix from accuracy)
    valid_decisions = tp + tn + fp + fn
    accuracy = (tp + tn) / valid_decisions if valid_decisions > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else float("nan")
    )

    # Confusion matrix: rows = actual, cols = predicted
    # [Negative, Positive, Uncertain, Invalid]
    confusion_matrix = [
        [tn, fp, uncertain_for_actual_neg, invalid_for_actual_neg],
        [fn, tp, uncertain_for_actual_pos, invalid_for_actual_pos],
    ]

    return {
        "confusion_matrix": confusion_matrix,
        "confusion_matrix_labels": [
            "Negative",
            "Positive",
            "Uncertain",
            "Invalid",
        ],
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "separatrix": n_separatrix,
        "separatrix_uncertain": n_uncertain,
        "separatrix_invalid": n_invalid,
        "tp_rate": float(tp_rate),
        "tn_rate": float(tn_rate),
        "fp_rate": float(fp_rate),
        "fn_rate": float(fn_rate),
        "separatrix_rate": float(separatrix_rate),
        "uncertain_rate": float(uncertain_rate),
        "invalid_rate": float(invalid_rate),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1_score": float(f1_score),
    }


def compute_coverage(
    prediction_sets: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute coverage: fraction of points whose true label is in prediction set.

    Args:
        prediction_sets: (N, 2) bool array [success_in_set, failure_in_set].
        true_labels: (N,) array of ground truth labels {0, 1}.

    Returns:
        Coverage as a float in [0, 1].
    """
    n = len(true_labels)
    if n == 0:
        return float("nan")

    success_mask = true_labels == 1
    failure_mask = true_labels == 0

    covered = np.zeros(n, dtype=bool)
    covered[success_mask] = prediction_sets[success_mask, 0]
    covered[failure_mask] = prediction_sets[failure_mask, 1]

    return float(np.mean(covered))


def make_conservative_labels(labels: np.ndarray) -> np.ndarray:
    """Convert uncertain/invalid predictions to FAILURE (conservative).

    Args:
        labels: (N,) array of PredictionLabel values.

    Returns:
        (N,) array where UNCERTAIN and INVALID are mapped to FAILURE.
    """
    result = labels.copy()
    mask = (result == PredictionLabel.UNCERTAIN) | (result == PredictionLabel.INVALID)
    result[mask] = PredictionLabel.FAILURE
    return result
