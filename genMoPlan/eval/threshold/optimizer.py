"""Threshold optimizer for finding optimal lambda* and delta*."""

import numpy as np

from genMoPlan.eval.types import PredictionLabel
from genMoPlan.eval.threshold.config import ThresholdConfig
from genMoPlan.eval.threshold.types import ThresholdResult


class ThresholdOptimizer:
    """Finds optimal lambda* and delta* via grid search on validation data.

    The classification rule is two-sided:
    - SUCCESS if p_success >= lambda + delta
    - FAILURE if p_failure >= (1 - lambda) + delta
    - INVALID if p_invalid >= lambda - delta (and use_p_invalid_veto is True)
    - UNCERTAIN otherwise (in the lambda +/- delta band)

    All four threshold modes are always run during evaluation:
    - fixed: Uses fixed_lambda_star and fixed_delta_star directly
    - joint: Grid search over both lambda and delta
    - lambda_only: Optimize lambda only, use fixed delta
    - delta_only: Optimize delta only, use fixed lambda
    """

    def __init__(self, config: ThresholdConfig):
        self.config = config

    def get_fixed_result(self) -> ThresholdResult:
        """Return fixed thresholds (no optimization)."""
        return ThresholdResult(
            lambda_star=self.config.fixed_lambda_star,
            delta_star=self.config.fixed_delta_star,
            optimize_objective=None,
            optimization_loss=None,
        )

    def optimize_joint(
        self, outcome_probs: np.ndarray, true_labels: np.ndarray
    ) -> ThresholdResult:
        """Grid search over both lambda and delta.

        Args:
            outcome_probs: (N, 3) array of [p_success, p_failure, p_invalid].
            true_labels: (N,) array of ground truth labels {0, 1}.

        Returns:
            ThresholdResult with optimized lambda* and delta*.
        """
        cfg = self.config

        lambda_grid = np.linspace(0.0, 1.0, cfg.lambda_grid_size + 2)[1:-1]
        delta_grid = np.linspace(cfg.delta_min, cfg.delta_max, cfg.delta_grid_size)

        best_obj = np.inf if cfg.optimize_objective == "loss" else -np.inf
        best_lam = cfg.fixed_lambda_star
        best_delta = cfg.fixed_delta_star

        for lam in lambda_grid:
            for delta in delta_grid:
                # Skip invalid combinations
                if lam + delta > 1.0 or lam - delta < 0.0:
                    continue

                tr = ThresholdResult(lambda_star=lam, delta_star=delta)
                obj = self._compute_objective(outcome_probs, true_labels, tr)

                if obj is None:
                    continue

                if cfg.optimize_objective == "loss":
                    if obj < best_obj:
                        best_obj = obj
                        best_lam = lam
                        best_delta = delta
                else:
                    if obj > best_obj:
                        best_obj = obj
                        best_lam = lam
                        best_delta = delta

        return ThresholdResult(
            lambda_star=float(best_lam),
            delta_star=float(best_delta),
            optimize_objective=cfg.optimize_objective,
            optimization_loss=float(best_obj) if np.isfinite(best_obj) else None,
        )

    def optimize_lambda(
        self, outcome_probs: np.ndarray, true_labels: np.ndarray
    ) -> ThresholdResult:
        """Optimize lambda only, use fixed delta.

        Args:
            outcome_probs: (N, 3) array of [p_success, p_failure, p_invalid].
            true_labels: (N,) array of ground truth labels {0, 1}.

        Returns:
            ThresholdResult with optimized lambda* and fixed delta.
        """
        cfg = self.config
        delta = cfg.fixed_delta_star

        lambda_grid = np.linspace(0.0, 1.0, cfg.lambda_grid_size + 2)[1:-1]

        best_obj = np.inf if cfg.optimize_objective == "loss" else -np.inf
        best_lam = cfg.fixed_lambda_star

        for lam in lambda_grid:
            if lam + delta > 1.0 or lam - delta < 0.0:
                continue

            tr = ThresholdResult(lambda_star=lam, delta_star=delta)
            obj = self._compute_objective(outcome_probs, true_labels, tr)

            if obj is None:
                continue

            if cfg.optimize_objective == "loss":
                if obj < best_obj:
                    best_obj = obj
                    best_lam = lam
            else:
                if obj > best_obj:
                    best_obj = obj
                    best_lam = lam

        return ThresholdResult(
            lambda_star=float(best_lam),
            delta_star=float(delta),
            optimize_objective=cfg.optimize_objective,
            optimization_loss=float(best_obj) if np.isfinite(best_obj) else None,
        )

    def optimize_delta(
        self, outcome_probs: np.ndarray, true_labels: np.ndarray
    ) -> ThresholdResult:
        """Optimize delta only, use fixed lambda.

        Args:
            outcome_probs: (N, 3) array of [p_success, p_failure, p_invalid].
            true_labels: (N,) array of ground truth labels {0, 1}.

        Returns:
            ThresholdResult with fixed lambda and optimized delta*.
        """
        cfg = self.config
        lam = cfg.fixed_lambda_star

        delta_grid = np.linspace(cfg.delta_min, cfg.delta_max, cfg.delta_grid_size)

        best_obj = np.inf if cfg.optimize_objective == "loss" else -np.inf
        best_delta = cfg.fixed_delta_star

        for delta in delta_grid:
            if lam + delta > 1.0 or lam - delta < 0.0:
                continue

            tr = ThresholdResult(lambda_star=lam, delta_star=delta)
            obj = self._compute_objective(outcome_probs, true_labels, tr)

            if obj is None:
                continue

            if cfg.optimize_objective == "loss":
                if obj < best_obj:
                    best_obj = obj
                    best_delta = delta
            else:
                if obj > best_obj:
                    best_obj = obj
                    best_delta = delta

        return ThresholdResult(
            lambda_star=float(lam),
            delta_star=float(best_delta),
            optimize_objective=cfg.optimize_objective,
            optimization_loss=float(best_obj) if np.isfinite(best_obj) else None,
        )

    @staticmethod
    def classify_pointwise(
        outcome_probs: np.ndarray,
        threshold_result: ThresholdResult,
        use_p_invalid_veto: bool = True,
    ) -> np.ndarray:
        """Point-wise classification using lambda +/- delta band. Always two-sided.

        Args:
            outcome_probs: (N, 3) array of [p_success, p_failure, p_invalid].
            threshold_result: ThresholdResult with lambda_star and delta_star.
            use_p_invalid_veto: Whether to check p_invalid for INVALID label.

        Returns:
            (N,) array of PredictionLabel values.
        """
        lam = threshold_result.lambda_star
        delta = threshold_result.delta_star

        p_s = outcome_probs[:, 0]  # p_success
        p_f = outcome_probs[:, 1]  # p_failure
        p_inv = outcome_probs[:, 2]  # p_invalid

        u = lam + delta  # Upper threshold for success
        v = (1.0 - lam) + delta  # Upper threshold for failure

        labels = np.full(len(outcome_probs), PredictionLabel.UNCERTAIN, dtype=np.int32)

        # INVALID: p_invalid >= lambda - delta (veto)
        if use_p_invalid_veto:
            invalid_threshold = max(0.0, lam - delta)
            invalid_mask = p_inv >= invalid_threshold
            labels[invalid_mask] = PredictionLabel.INVALID

        # SUCCESS: p_success >= u (overrides UNCERTAIN, not INVALID)
        success_mask = (p_s >= u) & (labels != PredictionLabel.INVALID)
        labels[success_mask] = PredictionLabel.SUCCESS

        # FAILURE: p_failure >= v (overrides UNCERTAIN, not INVALID)
        failure_mask = (p_f >= v) & (labels != PredictionLabel.INVALID)
        labels[failure_mask] = PredictionLabel.FAILURE

        return labels

    def _compute_objective(
        self,
        outcome_probs: np.ndarray,
        true_labels: np.ndarray,
        threshold_result: ThresholdResult,
    ) -> float:
        """Compute objective value for a given (lambda, delta) pair.

        Returns:
            Objective value (lower is better for "loss", higher for "jstat"/"f1").
            None if the configuration is degenerate.
        """
        pred = self.classify_pointwise(
            outcome_probs, threshold_result, self.config.use_p_invalid_veto
        )

        # Map to binary for comparison with true_labels {0, 1}
        # UNCERTAIN and INVALID are treated as "unknown" (separatrix)
        is_success = pred == PredictionLabel.SUCCESS
        is_failure = pred == PredictionLabel.FAILURE
        is_unknown = (pred == PredictionLabel.UNCERTAIN) | (
            pred == PredictionLabel.INVALID
        )

        gt_success = true_labels == 1
        gt_failure = true_labels == 0

        tp = np.sum(is_success & gt_success)
        tn = np.sum(is_failure & gt_failure)
        fp = np.sum(is_success & gt_failure)
        fn = np.sum(is_failure & gt_success)
        n_unknown = np.sum(is_unknown)
        n_total = len(true_labels)

        obj = self.config.optimize_objective

        if obj == "loss":
            n_decided = tp + tn + fp + fn
            misclass_rate = (fp + fn) / n_total if n_total > 0 else 0.0
            unknown_rate = n_unknown / n_total if n_total > 0 else 0.0
            return self.config.w * misclass_rate + (1.0 - self.config.w) * unknown_rate

        elif obj == "jstat":
            # Youden's J statistic = sensitivity + specificity - 1
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return sensitivity + specificity - 1.0

        elif obj == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision + recall == 0:
                return 0.0
            f1 = 2 * precision * recall / (precision + recall)
            return f1

        return None
