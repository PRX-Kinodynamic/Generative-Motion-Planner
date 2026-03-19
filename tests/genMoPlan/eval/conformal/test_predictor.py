"""Tests for genMoPlan.eval.conformal.predictor."""

import numpy as np
import pytest

from genMoPlan.eval.types import PredictionLabel
from genMoPlan.eval.conformal.config import ConformalConfig
from genMoPlan.eval.conformal.types import ConformalResult
from genMoPlan.eval.conformal.predictor import ConformalPredictor
from genMoPlan.eval.threshold.types import ThresholdResult


class TestConformalConfig:
    def test_default(self):
        cfg = ConformalConfig()
        assert cfg.alpha == 0.1

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            ConformalConfig(alpha=0.0)
        with pytest.raises(ValueError):
            ConformalConfig(alpha=1.0)

    def test_from_dict(self):
        cfg = ConformalConfig.from_dict({"alpha": 0.05, "ignored": True})
        assert cfg.alpha == 0.05


class TestConformalResult:
    def test_to_dict(self):
        cr = ConformalResult(q_hat=0.15, alpha=0.1)
        d = cr.to_dict()
        assert d["q_hat"] == 0.15
        assert d["alpha"] == 0.1

    def test_from_dict(self):
        cr = ConformalResult.from_dict({"q_hat": 0.2, "q_hat_success": 0.18})
        assert cr.q_hat == 0.2
        assert cr.q_hat_success == 0.18


class TestNonconformityScores:
    def test_perfect_scores_zero(self):
        """When probs are clearly above thresholds, scores should be zero."""
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        # u = 0.6, v = 0.6
        probs = np.array([
            [0.9, 0.05, 0.05],  # SUCCESS, p_s=0.9 >> u=0.6
            [0.05, 0.9, 0.05],  # FAILURE, p_f=0.9 >> v=0.6
        ], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int32)

        predictor = ConformalPredictor(ConformalConfig())
        scores = predictor.compute_nonconformity_scores(probs, labels, tr)

        assert scores[0] == 0.0  # p_s=0.9 > u=0.6 => max(0, 0.6-0.9) = 0
        assert scores[1] == 0.0  # p_f=0.9 > v=0.6 => max(0, 0.6-0.9) = 0

    def test_positive_scores(self):
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        # u = 0.6, v = 0.6
        probs = np.array([
            [0.4, 0.3, 0.3],  # SUCCESS, but p_s=0.4 < u=0.6
            [0.3, 0.4, 0.3],  # FAILURE, but p_f=0.4 < v=0.6
        ], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int32)

        predictor = ConformalPredictor(ConformalConfig())
        scores = predictor.compute_nonconformity_scores(probs, labels, tr)

        assert np.isclose(scores[0], 0.2)  # max(0, 0.6-0.4) = 0.2
        assert np.isclose(scores[1], 0.2)  # max(0, 0.6-0.4) = 0.2


class TestCalibration:
    def _make_cal_data(self, n=200, seed=42):
        rng = np.random.RandomState(seed)
        labels = np.zeros(n, dtype=np.int32)
        labels[: n // 2] = 1

        probs = np.zeros((n, 3), dtype=np.float32)
        probs[: n // 2, 0] = rng.uniform(0.5, 0.95, n // 2)
        probs[: n // 2, 1] = rng.uniform(0.0, 0.3, n // 2)
        probs[n // 2 :, 0] = rng.uniform(0.0, 0.3, n // 2)
        probs[n // 2 :, 1] = rng.uniform(0.5, 0.95, n // 2)
        probs[:, 2] = np.maximum(0, 1.0 - probs[:, 0] - probs[:, 1])

        return probs, labels

    def test_calibrate_returns_all_qhats(self):
        cfg = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        probs, labels = self._make_cal_data()

        result = predictor.calibrate(probs, labels, tr)

        assert result.q_hat is not None
        assert result.q_hat_success is not None
        assert result.q_hat_failure is not None
        assert result.alpha == 0.1
        assert result.calibration_set_size == len(labels)

    def test_qhat_nonnegative(self):
        cfg = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        probs, labels = self._make_cal_data()

        result = predictor.calibrate(probs, labels, tr)
        assert result.q_hat >= 0.0

    def test_stores_lambda_delta(self):
        cfg = ConformalConfig()
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.65, delta_star=0.08)
        probs, labels = self._make_cal_data()

        result = predictor.calibrate(probs, labels, tr)
        assert result.lambda_star == 0.65
        assert result.delta_star == 0.08


class TestPredict:
    def _make_data(self, n=300, seed=42):
        rng = np.random.RandomState(seed)
        labels = np.zeros(n, dtype=np.int32)
        labels[: n // 2] = 1

        probs = np.zeros((n, 3), dtype=np.float32)
        probs[: n // 2, 0] = rng.uniform(0.5, 0.95, n // 2)
        probs[: n // 2, 1] = rng.uniform(0.0, 0.3, n // 2)
        probs[n // 2 :, 0] = rng.uniform(0.0, 0.3, n // 2)
        probs[n // 2 :, 1] = rng.uniform(0.5, 0.95, n // 2)
        probs[:, 2] = np.maximum(0, 1.0 - probs[:, 0] - probs[:, 1])

        return probs, labels

    def test_predict_returns_all_variants(self):
        cfg = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)

        cal_probs, cal_labels = self._make_data(n=200, seed=42)
        test_probs, test_labels = self._make_data(n=100, seed=99)

        cr = predictor.calibrate(cal_probs, cal_labels, tr)
        results = predictor.predict(test_probs, tr, cr)

        assert "single_class" in results
        assert "multi_class_min" in results
        assert "multi_class_max" in results
        assert "multi_class_skip" in results

    def test_prediction_sets_shape(self):
        cfg = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)

        cal_probs, cal_labels = self._make_data(n=200, seed=42)
        test_probs, test_labels = self._make_data(n=100, seed=99)

        cr = predictor.calibrate(cal_probs, cal_labels, tr)
        results = predictor.predict(test_probs, tr, cr)

        for variant_name, variant_data in results.items():
            assert variant_data["prediction_sets"].shape == (100, 2)
            assert variant_data["labels"].shape == (100,)

    def test_labels_are_valid(self):
        cfg = ConformalConfig(alpha=0.1)
        predictor = ConformalPredictor(cfg)
        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)

        cal_probs, cal_labels = self._make_data(n=200, seed=42)
        test_probs, test_labels = self._make_data(n=100, seed=99)

        cr = predictor.calibrate(cal_probs, cal_labels, tr)
        results = predictor.predict(test_probs, tr, cr)

        valid_values = {
            PredictionLabel.SUCCESS,
            PredictionLabel.FAILURE,
            PredictionLabel.UNCERTAIN,
            PredictionLabel.INVALID,
        }

        for variant_name, variant_data in results.items():
            unique_labels = set(variant_data["labels"].tolist())
            assert unique_labels.issubset(valid_values), (
                f"Variant {variant_name} has invalid labels: {unique_labels - valid_values}"
            )


class TestPredictionSetsToLabels:
    def test_singleton_success(self):
        pred_sets = np.array([[True, False]], dtype=bool)
        p_inv = np.array([0.0])
        labels = ConformalPredictor.prediction_sets_to_labels(pred_sets, p_inv, 0.5)
        assert labels[0] == PredictionLabel.SUCCESS

    def test_singleton_failure(self):
        pred_sets = np.array([[False, True]], dtype=bool)
        p_inv = np.array([0.0])
        labels = ConformalPredictor.prediction_sets_to_labels(pred_sets, p_inv, 0.5)
        assert labels[0] == PredictionLabel.FAILURE

    def test_multi_set_is_uncertain(self):
        pred_sets = np.array([[True, True]], dtype=bool)
        p_inv = np.array([0.0])
        labels = ConformalPredictor.prediction_sets_to_labels(pred_sets, p_inv, 0.5)
        assert labels[0] == PredictionLabel.UNCERTAIN

    def test_empty_set_is_uncertain(self):
        pred_sets = np.array([[False, False]], dtype=bool)
        p_inv = np.array([0.0])
        labels = ConformalPredictor.prediction_sets_to_labels(pred_sets, p_inv, 0.5)
        assert labels[0] == PredictionLabel.UNCERTAIN

    def test_invalid_overrides(self):
        pred_sets = np.array([[True, False]], dtype=bool)
        p_inv = np.array([0.8])
        labels = ConformalPredictor.prediction_sets_to_labels(pred_sets, p_inv, 0.5)
        assert labels[0] == PredictionLabel.INVALID


class TestCoverageGuarantee:
    """Test that conformal prediction achieves approximately 1-alpha coverage."""

    def test_coverage_on_synthetic_data(self):
        """Coverage on test data should be approximately >= 1 - alpha."""
        rng = np.random.RandomState(123)
        n_cal = 500
        n_test = 500
        alpha = 0.1

        def make_data(n, seed):
            r = np.random.RandomState(seed)
            labels = np.zeros(n, dtype=np.int32)
            labels[: n // 2] = 1
            probs = np.zeros((n, 3), dtype=np.float32)
            probs[: n // 2, 0] = r.uniform(0.3, 0.95, n // 2)
            probs[: n // 2, 1] = r.uniform(0.0, 0.4, n // 2)
            probs[n // 2 :, 0] = r.uniform(0.0, 0.4, n // 2)
            probs[n // 2 :, 1] = r.uniform(0.3, 0.95, n // 2)
            probs[:, 2] = np.maximum(0, 1.0 - probs[:, 0] - probs[:, 1])
            return probs, labels

        cal_probs, cal_labels = make_data(n_cal, 42)
        test_probs, test_labels = make_data(n_test, 99)

        tr = ThresholdResult(lambda_star=0.5, delta_star=0.1)
        cfg = ConformalConfig(alpha=alpha)
        predictor = ConformalPredictor(cfg)

        cr = predictor.calibrate(cal_probs, cal_labels, tr)
        results = predictor.predict(test_probs, tr, cr)

        from genMoPlan.eval.metrics import compute_coverage

        for variant_name, variant_data in results.items():
            coverage = compute_coverage(variant_data["prediction_sets"], test_labels)
            # Coverage should be roughly >= 1 - alpha (with some statistical slack)
            assert coverage >= (1 - alpha) - 0.05, (
                f"Coverage for {variant_name} is {coverage:.3f}, "
                f"expected >= {1 - alpha - 0.05:.3f}"
            )
