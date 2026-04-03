"""Tests for genMoPlan.eval.threshold.optimizer."""

import numpy as np
import pytest

from genMoPlan.eval.types import PredictionLabel
from genMoPlan.eval.threshold.config import ThresholdConfig
from genMoPlan.eval.threshold.types import ThresholdResult
from genMoPlan.eval.threshold.optimizer import ThresholdOptimizer


class TestThresholdConfig:
    def test_default_values(self):
        cfg = ThresholdConfig()
        assert cfg.optimize_objective == "loss"
        assert cfg.w == 0.9
        assert cfg.fixed_lambda_star == 0.5
        assert cfg.fixed_delta_star == 0.1

    def test_invalid_objective(self):
        with pytest.raises(ValueError, match="optimize_objective"):
            ThresholdConfig(optimize_objective="invalid")

    def test_from_dict(self):
        cfg = ThresholdConfig.from_dict({
            "optimize_objective": "f1",
            "w": 0.8,
            "unknown_key": True,  # Should be ignored
        })
        assert cfg.optimize_objective == "f1"
        assert cfg.w == 0.8

    def test_from_dict_ignores_optimize_mode(self):
        """Old configs with optimize_mode should be silently ignored."""
        cfg = ThresholdConfig.from_dict({
            "optimize_mode": "joint",  # Legacy key, should be ignored
            "optimize_objective": "loss",
        })
        assert cfg.optimize_objective == "loss"

    def test_from_empty_dict(self):
        cfg = ThresholdConfig.from_dict({})
        assert cfg.optimize_objective == "loss"


class TestThresholdResult:
    def test_to_dict(self):
        tr = ThresholdResult(lambda_star=0.6, delta_star=0.15)
        d = tr.to_dict()
        assert d["lambda_star"] == 0.6
        assert d["delta_star"] == 0.15
        assert "optimize_mode" not in d

    def test_from_dict(self):
        tr = ThresholdResult.from_dict({"lambda_star": 0.7, "delta_star": 0.05})
        assert tr.lambda_star == 0.7
        assert tr.delta_star == 0.05


class TestClassifyPointwise:
    def _make_probs(self, p_s, p_f):
        """Helper: create (N, 3) probs from success and failure probs."""
        p_inv = 1.0 - p_s - p_f
        return np.column_stack([p_s, p_f, np.maximum(0, p_inv)])

    def test_clear_success(self):
        # p_success = 0.9 >> lambda + delta = 0.6
        probs = self._make_probs(np.array([0.9]), np.array([0.05]))
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr)
        assert labels[0] == PredictionLabel.SUCCESS

    def test_clear_failure(self):
        # p_failure = 0.8 >> (1 - lambda) + delta = 0.6
        probs = self._make_probs(np.array([0.05]), np.array([0.8]))
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr)
        assert labels[0] == PredictionLabel.FAILURE

    def test_uncertain(self):
        # p_success = 0.55, neither above u=0.6 nor p_failure above v=0.6
        probs = self._make_probs(np.array([0.55]), np.array([0.4]))
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr)
        assert labels[0] == PredictionLabel.UNCERTAIN

    def test_invalid_veto(self):
        # p_invalid = 0.5 >= lambda - delta = 0.4
        probs = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr, use_p_invalid_veto=True)
        assert labels[0] == PredictionLabel.INVALID

    def test_invalid_veto_disabled(self):
        probs = np.array([[0.2, 0.3, 0.5]], dtype=np.float32)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr, use_p_invalid_veto=False)
        assert labels[0] != PredictionLabel.INVALID

    def test_batch(self):
        probs = np.array([
            [0.9, 0.05, 0.05],  # SUCCESS
            [0.05, 0.9, 0.05],  # FAILURE
            [0.4, 0.4, 0.2],    # UNCERTAIN
        ], dtype=np.float32)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        labels = ThresholdOptimizer.classify_pointwise(probs, tr)
        assert labels[0] == PredictionLabel.SUCCESS
        assert labels[1] == PredictionLabel.FAILURE
        assert labels[2] == PredictionLabel.UNCERTAIN


class TestThresholdOptimizer:
    def _make_synthetic_data(self, n=200, seed=42):
        """Create synthetic validation data with clear structure."""
        rng = np.random.RandomState(seed)

        # Half success, half failure
        labels = np.zeros(n, dtype=np.int32)
        labels[: n // 2] = 1

        probs = np.zeros((n, 3), dtype=np.float32)
        # Success points: high p_success
        probs[: n // 2, 0] = rng.uniform(0.6, 0.95, n // 2)
        probs[: n // 2, 1] = rng.uniform(0.0, 0.2, n // 2)
        # Failure points: high p_failure
        probs[n // 2 :, 0] = rng.uniform(0.0, 0.2, n // 2)
        probs[n // 2 :, 1] = rng.uniform(0.6, 0.95, n // 2)
        # Remainder is p_invalid
        probs[:, 2] = np.maximum(0, 1.0 - probs[:, 0] - probs[:, 1])

        return probs, labels

    def test_get_fixed_result(self):
        cfg = ThresholdConfig(fixed_lambda_star=0.6, fixed_delta_star=0.05)
        opt = ThresholdOptimizer(cfg)
        result = opt.get_fixed_result()
        assert result.lambda_star == 0.6
        assert result.delta_star == 0.05
        assert result.optimize_objective is None
        assert result.optimization_loss is None

    def test_optimize_joint(self):
        cfg = ThresholdConfig(
            optimize_objective="loss",
            lambda_grid_size=20,
            delta_grid_size=20,
        )
        opt = ThresholdOptimizer(cfg)
        probs, labels = self._make_synthetic_data()
        result = opt.optimize_joint(probs, labels)

        assert result.optimize_objective == "loss"
        assert 0.0 < result.lambda_star < 1.0
        assert result.delta_star > 0.0
        assert result.optimization_loss is not None

    def test_optimize_lambda(self):
        cfg = ThresholdConfig(
            optimize_objective="loss",
            fixed_delta_star=0.05,
            lambda_grid_size=20,
        )
        opt = ThresholdOptimizer(cfg)
        probs, labels = self._make_synthetic_data()
        result = opt.optimize_lambda(probs, labels)

        assert result.optimize_objective == "loss"
        assert result.delta_star == 0.05  # Fixed

    def test_optimize_delta(self):
        cfg = ThresholdConfig(
            optimize_objective="loss",
            fixed_lambda_star=0.5,
            delta_grid_size=20,
        )
        opt = ThresholdOptimizer(cfg)
        probs, labels = self._make_synthetic_data()
        result = opt.optimize_delta(probs, labels)

        assert result.optimize_objective == "loss"
        assert result.lambda_star == 0.5  # Fixed

    def test_jstat_objective(self):
        cfg = ThresholdConfig(
            optimize_objective="jstat",
            lambda_grid_size=20,
            delta_grid_size=20,
        )
        opt = ThresholdOptimizer(cfg)
        probs, labels = self._make_synthetic_data()
        result = opt.optimize_joint(probs, labels)
        assert result.optimize_objective == "jstat"

    def test_f1_objective(self):
        cfg = ThresholdConfig(
            optimize_objective="f1",
            lambda_grid_size=20,
            delta_grid_size=20,
        )
        opt = ThresholdOptimizer(cfg)
        probs, labels = self._make_synthetic_data()
        result = opt.optimize_joint(probs, labels)
        assert result.optimize_objective == "f1"

    def test_optimized_beats_default(self):
        """Optimized thresholds should perform at least as well as defaults."""
        probs, labels = self._make_synthetic_data()

        # Default
        cfg = ThresholdConfig(lambda_grid_size=30, delta_grid_size=30)
        opt = ThresholdOptimizer(cfg)

        default_tr = opt.get_fixed_result()
        default_labels = ThresholdOptimizer.classify_pointwise(probs, default_tr)
        default_correct = np.sum(
            (default_labels == PredictionLabel.SUCCESS) & (labels == 1)
        ) + np.sum(
            (default_labels == PredictionLabel.FAILURE) & (labels == 0)
        )

        # Optimized (joint)
        result = opt.optimize_joint(probs, labels)
        opt_labels = ThresholdOptimizer.classify_pointwise(probs, result)
        opt_correct = np.sum(
            (opt_labels == PredictionLabel.SUCCESS) & (labels == 1)
        ) + np.sum(
            (opt_labels == PredictionLabel.FAILURE) & (labels == 0)
        )

        # Should be at least as good (might be equal for well-separated data)
        assert opt_correct >= default_correct
