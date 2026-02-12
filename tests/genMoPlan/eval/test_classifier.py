"""Tests for Classifier (ROA evaluator) - vectorized compute_outcome_labels and related methods."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch

from genMoPlan.systems import Outcome


class MockNormalizer:
    """Mock normalizer for testing."""
    def __call__(self, data):
        return data * 0.1

    def unnormalize(self, data):
        return data * 10.0


class MockSystem:
    """Mock system for testing Classifier."""
    def __init__(self, state_dim=2):
        self.state_dim = state_dim
        self.state_names = ["x", "y"]
        self.metadata = {
            "invalid_label": -1,
            "invalid_labels": [-1],
            "invalid_outcomes": [],
        }
        self.valid_outcomes = [Outcome.SUCCESS, Outcome.FAILURE]
        # Required attributes for TrajectoryGenerator/Classifier
        self.history_length = 2
        self.horizon_length = 3
        self.stride = 1
        self.max_path_length = 50
        self.normalizer = MockNormalizer()

    def evaluate_final_state(self, state):
        """Single state evaluation (for comparison)."""
        if np.linalg.norm(state) < 1.0:
            return Outcome.SUCCESS
        return Outcome.FAILURE

    def evaluate_final_states(self, states):
        """Vectorized evaluation of final states."""
        norms = np.linalg.norm(states, axis=1)
        outcomes = np.where(
            norms < 1.0,
            Outcome.SUCCESS.value,
            Outcome.FAILURE.value
        )
        return outcomes.astype(np.int32)

    def outcome_to_index(self, outcome):
        """Convert outcome to index."""
        if outcome == Outcome.SUCCESS:
            return 0
        return 1


class TestComputeOutcomeLabelsVectorized:
    """Tests for vectorized compute_outcome_labels method."""

    def test_vectorized_matches_loop_result(self):
        """Test that vectorized evaluation matches loop-based evaluation."""
        system = MockSystem()

        # Create test final states: [n_states, n_runs, dim]
        np.random.seed(42)
        n_states = 100
        n_runs = 5
        dim = 2
        final_states = np.random.randn(n_states, n_runs, dim)

        # Reshape for evaluation
        reshaped_states = final_states.reshape(-1, dim)

        # Vectorized evaluation
        vectorized_results = system.evaluate_final_states(reshaped_states)

        # Loop-based evaluation (for comparison)
        loop_results = np.array([
            system.outcome_to_index(system.evaluate_final_state(s))
            for s in reshaped_states
        ], dtype=np.int32)

        np.testing.assert_array_equal(vectorized_results, loop_results)

    def test_vectorized_handles_mixed_outcomes(self):
        """Test vectorized evaluation handles mix of SUCCESS and FAILURE."""
        system = MockSystem()

        # States with norm < 1 should be SUCCESS, >= 1 should be FAILURE
        states = np.array([
            [0.1, 0.1],   # norm = 0.14 < 1 -> SUCCESS
            [1.0, 1.0],   # norm = 1.41 >= 1 -> FAILURE
            [0.5, 0.5],   # norm = 0.71 < 1 -> SUCCESS
            [2.0, 0.0],   # norm = 2.0 >= 1 -> FAILURE
        ], dtype=np.float32)

        results = system.evaluate_final_states(states)

        assert results[0] == Outcome.SUCCESS.value
        assert results[1] == Outcome.FAILURE.value
        assert results[2] == Outcome.SUCCESS.value
        assert results[3] == Outcome.FAILURE.value

    def test_vectorized_handles_large_batch(self):
        """Test vectorized evaluation handles large batches efficiently."""
        system = MockSystem()

        # Large batch
        np.random.seed(42)
        states = np.random.randn(10000, 2)

        # Should complete without error
        results = system.evaluate_final_states(states)

        assert results.shape == (10000,)
        assert results.dtype == np.int32

    def test_vectorized_all_success(self):
        """Test vectorized evaluation with all SUCCESS outcomes."""
        system = MockSystem()

        # All states with small norm
        states = np.array([
            [0.1, 0.1],
            [0.2, 0.1],
            [0.1, 0.3],
        ], dtype=np.float32)

        results = system.evaluate_final_states(states)

        assert np.all(results == Outcome.SUCCESS.value)

    def test_vectorized_all_failure(self):
        """Test vectorized evaluation with all FAILURE outcomes."""
        system = MockSystem()

        # All states with large norm
        states = np.array([
            [5.0, 5.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ], dtype=np.float32)

        results = system.evaluate_final_states(states)

        assert np.all(results == Outcome.FAILURE.value)

    def test_vectorized_returns_correct_dtype(self):
        """Test vectorized evaluation returns int32 dtype."""
        system = MockSystem()

        states = np.random.randn(10, 2)
        results = system.evaluate_final_states(states)

        assert results.dtype == np.int32


class TestClassifierIntegration:
    """Integration tests for Classifier using vectorized evaluation."""

    @patch('genMoPlan.eval.classifier.load_inference_params')
    @patch('genMoPlan.eval.classifier.TrajectoryGenerator')
    def test_compute_outcome_labels_uses_vectorized(self, mock_traj_gen, mock_load_params):
        """Test Classifier.compute_outcome_labels uses vectorized system.evaluate_final_states."""
        from genMoPlan.eval.classifier import Classifier

        # Setup mocks
        mock_load_params.return_value = {
            "max_path_length": 50,
            "batch_size": 10,
            "n_runs": 3,
            "final_state_directory": "final_states",
            "results_name": "test_results",
        }

        mock_traj_gen_instance = MagicMock()
        mock_traj_gen_instance.model_args = Mock()
        mock_traj_gen_instance.method_name = "flow_matching"
        mock_traj_gen_instance.horizon_length = 3
        mock_traj_gen_instance.history_length = 2
        mock_traj_gen_instance.stride = 1
        mock_traj_gen_instance.conditional_sample_kwargs = {}
        mock_traj_gen.return_value = mock_traj_gen_instance

        system = MockSystem()

        # Create classifier
        classifier = Classifier(
            dataset="test_dataset",
            model_path="/common/home/st1122/Projects/genMoPlan/tmp/test_model",
            model_state_name="best.pt",
            system=system,
            verbose=False,
        )

        # Set final states directly
        classifier.final_states = np.random.randn(10, 3, 2)  # [n_states, n_runs, dim]

        # Compute outcome labels
        labels = classifier.compute_outcome_labels()

        # Should have computed labels for all state-run combinations
        assert labels.shape == (10, 3)

    @patch('genMoPlan.eval.classifier.load_inference_params')
    def test_compute_outcome_labels_without_model(self, mock_load_params):
        """Test compute_outcome_labels works when only analyzing pre-computed final states."""
        from genMoPlan.eval.classifier import Classifier

        mock_load_params.return_value = {
            "max_path_length": 50,
            "batch_size": 10,
            "n_runs": 3,
            "final_state_directory": "final_states",
            "results_name": "test_results",
        }

        system = MockSystem()

        # Create classifier without model (for analysis only)
        classifier = Classifier(
            dataset="test_dataset",
            model_path=None,
            system=system,
            verbose=False,
        )

        # Set final states directly
        classifier.final_states = np.random.randn(5, 2, 2)

        # Should work without model loaded
        labels = classifier.compute_outcome_labels()

        assert labels.shape == (5, 2)


class TestSystemEvaluateFinalStatesInterface:
    """Tests for the system.evaluate_final_states interface."""

    def test_interface_accepts_2d_array(self):
        """Test evaluate_final_states accepts [batch, state_dim] array."""
        system = MockSystem()

        states = np.random.randn(10, 2)
        results = system.evaluate_final_states(states)

        assert results.shape == (10,)

    def test_interface_returns_outcome_values(self):
        """Test evaluate_final_states returns Outcome enum values."""
        system = MockSystem()

        states = np.array([[0.1, 0.1], [5.0, 5.0]])
        results = system.evaluate_final_states(states)

        # Results should be valid Outcome values
        assert results[0] in [o.value for o in Outcome]
        assert results[1] in [o.value for o in Outcome]
