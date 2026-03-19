"""Conformal predictor for coverage-guaranteed classification."""

import numpy as np

from genMoPlan.eval.types import PredictionLabel
from genMoPlan.eval.conformal.config import ConformalConfig
from genMoPlan.eval.conformal.types import ConformalResult
from genMoPlan.eval.threshold.types import ThresholdResult


class ConformalPredictor:
    """Calibrates q-hat on calibration data and produces prediction sets.

    Non-conformity scores are two-sided, using lambda +/- delta from
    threshold optimization:
        u = lambda + delta (success threshold)
        v = (1 - lambda) + delta (failure threshold)

    For true label SUCCESS: score = max(0, u - p_success)
    For true label FAILURE: score = max(0, v - p_failure)
    """

    def __init__(self, config: ConformalConfig):
        self.config = config

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_true_labels: np.ndarray,
        threshold_result: ThresholdResult,
    ) -> ConformalResult:
        """Calibrate q-hat on calibration data.

        Always performs BOTH single-class and multi-class calibration.

        Args:
            cal_probs: (N_cal, 3) array of [p_success, p_failure, p_invalid].
            cal_true_labels: (N_cal,) array of ground truth labels {0, 1}.
            threshold_result: ThresholdResult providing lambda* and delta*.

        Returns:
            ConformalResult with all q-hat values.
        """
        alpha = self.config.alpha
        n_cal = len(cal_true_labels)

        # Compute non-conformity scores for each calibration point
        scores = self.compute_nonconformity_scores(
            cal_probs, cal_true_labels, threshold_result
        )

        # Single-class q-hat: quantile over all calibration scores
        q_hat = self._calibration_quantile(scores, alpha, n_cal)

        # Multi-class q-hat: separate quantiles for SUCCESS and FAILURE
        success_mask = cal_true_labels == 1
        failure_mask = cal_true_labels == 0

        q_hat_success = None
        q_hat_failure = None

        if np.sum(success_mask) > 0:
            q_hat_success = self._calibration_quantile(
                scores[success_mask], alpha, int(np.sum(success_mask))
            )

        if np.sum(failure_mask) > 0:
            q_hat_failure = self._calibration_quantile(
                scores[failure_mask], alpha, int(np.sum(failure_mask))
            )

        return ConformalResult(
            q_hat=float(q_hat),
            q_hat_success=float(q_hat_success) if q_hat_success is not None else None,
            q_hat_failure=float(q_hat_failure) if q_hat_failure is not None else None,
            lambda_star=threshold_result.lambda_star,
            delta_star=threshold_result.delta_star,
            alpha=alpha,
            calibration_set_size=n_cal,
        )

    def predict(
        self,
        test_probs: np.ndarray,
        threshold_result: ThresholdResult,
        conformal_result: ConformalResult,
    ) -> dict:
        """Produce prediction sets for ALL conformal variants.

        Args:
            test_probs: (N_test, 3) array of [p_success, p_failure, p_invalid].
            threshold_result: ThresholdResult providing lambda* and delta*.
            conformal_result: ConformalResult with calibrated q-hat values.

        Returns:
            dict with keys:
            - "single_class": prediction_sets, labels, coverage info
            - "multi_class_min": uses min(q_hat_success, q_hat_failure)
            - "multi_class_max": uses max(q_hat_success, q_hat_failure)
            - "multi_class_skip": uses per-class q-hat directly
        """
        lam = threshold_result.lambda_star
        delta = threshold_result.delta_star
        invalid_threshold = max(0.0, lam - delta)

        n_test = len(test_probs)
        p_s = test_probs[:, 0]
        p_f = test_probs[:, 1]
        p_inv = test_probs[:, 2]

        results = {}

        # --- Single-class conformal ---
        q = conformal_result.q_hat
        if q is not None:
            pred_sets = self._build_prediction_sets_single(
                p_s, p_f, threshold_result, q
            )
            labels = self.prediction_sets_to_labels(pred_sets, p_inv, invalid_threshold)
            results["single_class"] = {
                "prediction_sets": pred_sets,
                "labels": labels,
                "q_hat_used": q,
            }

        # --- Multi-class conformal variants ---
        q_s = conformal_result.q_hat_success
        q_f = conformal_result.q_hat_failure

        if q_s is not None and q_f is not None:
            # min variant
            q_min = min(q_s, q_f)
            pred_sets_min = self._build_prediction_sets_single(
                p_s, p_f, threshold_result, q_min
            )
            labels_min = self.prediction_sets_to_labels(
                pred_sets_min, p_inv, invalid_threshold
            )
            results["multi_class_min"] = {
                "prediction_sets": pred_sets_min,
                "labels": labels_min,
                "q_hat_used": q_min,
            }

            # max variant
            q_max = max(q_s, q_f)
            pred_sets_max = self._build_prediction_sets_single(
                p_s, p_f, threshold_result, q_max
            )
            labels_max = self.prediction_sets_to_labels(
                pred_sets_max, p_inv, invalid_threshold
            )
            results["multi_class_max"] = {
                "prediction_sets": pred_sets_max,
                "labels": labels_max,
                "q_hat_used": q_max,
            }

            # skip variant: per-class q-hat
            pred_sets_skip = self._build_prediction_sets_multi(
                p_s, p_f, threshold_result, q_s, q_f
            )
            labels_skip = self.prediction_sets_to_labels(
                pred_sets_skip, p_inv, invalid_threshold
            )
            results["multi_class_skip"] = {
                "prediction_sets": pred_sets_skip,
                "labels": labels_skip,
                "q_hat_success": q_s,
                "q_hat_failure": q_f,
            }

        return results

    def compute_nonconformity_scores(
        self,
        probs: np.ndarray,
        true_labels: np.ndarray,
        threshold_result: ThresholdResult,
    ) -> np.ndarray:
        """Compute non-conformity scores for calibration data.

        Two-sided scores:
            u = lambda + delta, v = (1 - lambda) + delta
            SUCCESS: max(0, u - p_success)
            FAILURE: max(0, v - p_failure)

        Args:
            probs: (N, 3) array of [p_success, p_failure, p_invalid].
            true_labels: (N,) array of ground truth labels {0, 1}.
            threshold_result: ThresholdResult providing lambda* and delta*.

        Returns:
            (N,) array of non-conformity scores.
        """
        lam = threshold_result.lambda_star
        delta = threshold_result.delta_star

        u = lam + delta
        v = (1.0 - lam) + delta

        p_s = probs[:, 0]
        p_f = probs[:, 1]

        scores = np.zeros(len(probs), dtype=np.float64)

        success_mask = true_labels == 1
        failure_mask = true_labels == 0

        scores[success_mask] = np.maximum(0.0, u - p_s[success_mask])
        scores[failure_mask] = np.maximum(0.0, v - p_f[failure_mask])

        return scores

    @staticmethod
    def prediction_sets_to_labels(
        prediction_sets: np.ndarray,
        p_invalid: np.ndarray,
        invalid_threshold: float,
    ) -> np.ndarray:
        """Convert prediction sets to point labels.

        Singleton {SUCCESS} -> SUCCESS
        Singleton {FAILURE} -> FAILURE
        Multi/empty set -> UNCERTAIN
        High p_invalid -> INVALID (overrides everything)

        Args:
            prediction_sets: (N, 2) bool array [success_in_set, failure_in_set].
            p_invalid: (N,) array of p_invalid values.
            invalid_threshold: Threshold for INVALID veto.

        Returns:
            (N,) array of PredictionLabel values.
        """
        n = len(prediction_sets)
        labels = np.full(n, PredictionLabel.UNCERTAIN, dtype=np.int32)

        success_in = prediction_sets[:, 0]
        failure_in = prediction_sets[:, 1]

        # Singleton success
        singleton_success = success_in & ~failure_in
        labels[singleton_success] = PredictionLabel.SUCCESS

        # Singleton failure
        singleton_failure = failure_in & ~success_in
        labels[singleton_failure] = PredictionLabel.FAILURE

        # INVALID veto (overrides everything)
        invalid_mask = p_invalid >= invalid_threshold
        labels[invalid_mask] = PredictionLabel.INVALID

        return labels

    @staticmethod
    def _calibration_quantile(
        scores: np.ndarray, alpha: float, n: int
    ) -> float:
        """Compute conformal calibration quantile.

        q_hat = ceil((n+1)(1-alpha)) / n -th quantile of scores.
        """
        level = np.ceil((n + 1) * (1 - alpha)) / n
        level = min(level, 1.0)
        return float(np.quantile(scores, level))

    @staticmethod
    def _build_prediction_sets_single(
        p_s: np.ndarray,
        p_f: np.ndarray,
        threshold_result: ThresholdResult,
        q_hat: float,
    ) -> np.ndarray:
        """Build prediction sets using a single q-hat for all classes.

        A label y is in the prediction set if its non-conformity score <= q_hat.
        For SUCCESS candidate: score = max(0, u - p_s), include if score <= q_hat
            => p_s >= u - q_hat
        For FAILURE candidate: score = max(0, v - p_f), include if score <= q_hat
            => p_f >= v - q_hat

        Returns:
            (N, 2) bool array [success_in_set, failure_in_set].
        """
        lam = threshold_result.lambda_star
        delta = threshold_result.delta_star
        u = lam + delta
        v = (1.0 - lam) + delta

        success_in = p_s >= (u - q_hat)
        failure_in = p_f >= (v - q_hat)

        return np.column_stack([success_in, failure_in])

    @staticmethod
    def _build_prediction_sets_multi(
        p_s: np.ndarray,
        p_f: np.ndarray,
        threshold_result: ThresholdResult,
        q_hat_success: float,
        q_hat_failure: float,
    ) -> np.ndarray:
        """Build prediction sets using per-class q-hat values.

        Returns:
            (N, 2) bool array [success_in_set, failure_in_set].
        """
        lam = threshold_result.lambda_star
        delta = threshold_result.delta_star
        u = lam + delta
        v = (1.0 - lam) + delta

        success_in = p_s >= (u - q_hat_success)
        failure_in = p_f >= (v - q_hat_failure)

        return np.column_stack([success_in, failure_in])
