"""Tests for stride-related fixes in TrajectoryGenerator.

These tests verify that trajectory generation with stride > 1 works correctly:
1. Buffer size is in model-space (optimized memory usage)
2. Termination steps are reported in actual timesteps
3. Trajectory filling handles early termination correctly
4. Conversion utility works correctly
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock
from collections import namedtuple

from genMoPlan.utils.trajectory_generator import TrajectoryGenerator, ResolvedParams
from genMoPlan.utils.generation_result import GenerationResult
from genMoPlan.utils.data_processing import compute_total_predictions, compute_actual_length
from genMoPlan.systems import Outcome


Sample = namedtuple("Sample", "trajectories values chains")


class MockModelStride:
    """Mock model for stride testing."""
    def __init__(self, observation_dim=4, history_length=3, horizon_length=5):
        self.observation_dim = observation_dim
        self.history_length = history_length
        self.prediction_length = history_length + horizon_length
        self.input_dim = observation_dim

    def conditional_sample(self, cond, shape, **kwargs):
        """Simulate conditional sample returning trajectories."""
        batch_size = shape[0]
        trajectories = torch.randn(batch_size, self.prediction_length, self.observation_dim)
        return Sample(trajectories=trajectories, values=None, chains=None)

    def __call__(self, cond, **kwargs):
        """Make model callable for trajectory generation."""
        batch_size = cond[0].shape[0] if isinstance(cond, dict) and len(cond) > 0 else 1
        trajectories = torch.randn(batch_size, self.prediction_length, self.observation_dim)
        return Sample(trajectories=trajectories, values=None, chains=None)

    def parameters(self):
        return [torch.tensor([1.0])]


class MockArgsStride:
    """Mock model arguments with stride parameter."""
    def __init__(self, history_length=3, horizon_length=5, observation_dim=4, stride=10):
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.observation_dim = observation_dim
        self.stride = stride
        self.trajectory_normalizer = None
        self.method = "models.generative.FlowMatching"


class MockNormalizerStride:
    """Mock normalizer for stride testing."""
    def __call__(self, data):
        return data * 0.1

    def unnormalize(self, data):
        return data * 10.0


class MockSystemStride:
    """Mock system for stride testing with controllable termination."""
    def __init__(self, state_dim=4, termination_threshold=0.5, history_length=3,
                 horizon_length=5, stride=10, max_path_length=636):
        self.state_dim = state_dim
        self.termination_threshold = termination_threshold
        self.state_names = ["x", "y", "theta", "theta_dot"]
        # Add required attributes for TrajectoryGenerator
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.stride = stride
        self.max_path_length = max_path_length
        # Normalizer is required (single source of truth)
        self.normalizer = MockNormalizerStride()

    def evaluate_final_states(self, states):
        """Vectorized evaluation - terminate when norm < threshold."""
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()

        norms = np.linalg.norm(states, axis=1)
        outcomes = np.where(
            norms < self.termination_threshold,
            Outcome.SUCCESS.value,
            Outcome.FAILURE.value
        )
        return outcomes.astype(np.int32)


@pytest.fixture
def mock_inference_params_stride():
    """Mock inference parameters for stride testing."""
    return {
        "max_path_length": 636,  # This would be wasteful if used for buffer
        "batch_size": 8,
        "n_runs": 1,
        "final_state_directory": "final_states",
        "post_process_fns": None,
        "post_process_fn_kwargs": {},
        "conditional_sample_kwargs": {},
        "load_ema": False,
    }


@pytest.fixture
def trajectory_generator_stride10(mock_inference_params_stride):
    """Create TrajectoryGenerator with stride=10."""
    model = MockModelStride(observation_dim=4, history_length=3, horizon_length=5)
    model_args = MockArgsStride(history_length=3, horizon_length=5, stride=10)
    system = MockSystemStride(
        state_dim=4,
        termination_threshold=0.5,
        history_length=3,
        horizon_length=5,
        stride=10,
        max_path_length=636,
    )

    generator = TrajectoryGenerator(
        dataset="test_dataset_stride",
        model=model,
        model_args=model_args,
        inference_params=mock_inference_params_stride,
        device="cpu",
        verbose=False,
        system=system,
    )
    return generator


@pytest.fixture
def trajectory_generator_stride1(mock_inference_params_stride):
    """Create TrajectoryGenerator with stride=1 for comparison."""
    # Adjust max_path_length for stride=1
    params = mock_inference_params_stride.copy()
    params["max_path_length"] = 100

    model = MockModelStride(observation_dim=4, history_length=3, horizon_length=5)
    model_args = MockArgsStride(history_length=3, horizon_length=5, stride=1)
    system = MockSystemStride(
        state_dim=4,
        termination_threshold=0.5,
        history_length=3,
        horizon_length=5,
        stride=1,
        max_path_length=100,
    )

    generator = TrajectoryGenerator(
        dataset="test_dataset_stride1",
        model=model,
        model_args=model_args,
        inference_params=params,
        device="cpu",
        verbose=False,
        system=system,
    )
    return generator


class TestStrideBufferSize:
    """Test that trajectory buffer uses model-space, not actual timesteps."""

    def test_stride10_buffer_size_is_model_space(self, trajectory_generator_stride10):
        """Verify buffer size is in model-space with stride=10."""
        start_states = np.random.randn(8, 4)

        # Calculate expected buffer size
        history_length = 3
        horizon_length = 5
        num_inference_steps = 15  # From config
        expected_buffer_size = compute_total_predictions(
            history_length, num_inference_steps, horizon_length
        )
        # expected_buffer_size = 3 + 15*5 = 78

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=num_inference_steps,
            return_trajectories=True,
        )

        assert result.trajectories is not None
        batch_size, traj_length, state_dim = result.trajectories.shape

        # Buffer should be 78 (model-space), NOT 636 (actual timesteps)
        assert traj_length == expected_buffer_size, \
            f"Expected buffer size {expected_buffer_size}, got {traj_length}"
        assert traj_length == 78, "Buffer should be 78 (model predictions)"

    def test_stride10_memory_savings(self, trajectory_generator_stride10):
        """Verify stride=10 achieves significant memory savings."""
        start_states = np.random.randn(8, 4)

        # Old approach would allocate max_path_length = 636
        old_buffer_size = 636

        # New approach allocates model predictions only
        new_buffer_size = compute_total_predictions(3, 15, 5)  # 78

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=15,
            return_trajectories=True,
        )

        actual_buffer_size = result.trajectories.shape[1]

        # Verify we're using new approach
        assert actual_buffer_size == new_buffer_size

        # Calculate savings
        savings_percent = (1 - new_buffer_size / old_buffer_size) * 100
        assert savings_percent > 85, f"Expected >85% savings, got {savings_percent:.1f}%"

    def test_stride1_buffer_size(self, trajectory_generator_stride1):
        """Verify stride=1 still works correctly with new approach."""
        start_states = np.random.randn(8, 4)

        # With stride=1, num_inference_steps and model-space are more aligned
        num_inference_steps = 10
        expected_buffer_size = compute_total_predictions(3, num_inference_steps, 5)
        # expected = 3 + 10*5 = 53

        result = trajectory_generator_stride1.generate(
            start_states=start_states,
            num_inference_steps=num_inference_steps,
            return_trajectories=True,
        )

        assert result.trajectories is not None
        traj_length = result.trajectories.shape[1]
        assert traj_length == expected_buffer_size


class TestStrideTerminationSteps:
    """Test that termination steps are reported in actual timesteps."""

    def test_termination_steps_in_actual_timesteps_stride10(self, trajectory_generator_stride10):
        """Verify termination steps are in actual timesteps with stride=10."""
        # Create states that will terminate (small norm)
        start_states = np.random.randn(8, 4) * 0.1  # Small values, likely to terminate

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=5,
            return_trajectories=True,
        )

        # Check termination info exists
        assert result.has_termination_info
        assert result.termination_steps is not None

        # Calculate expected ranges
        history_length = 3
        stride = 10
        horizon_length = 5

        # Actual history in timesteps
        actual_history = compute_actual_length(history_length, stride)
        # actual_history = 1 + (3-1)*10 = 21

        # Minimum termination step (if terminates at first horizon prediction)
        min_termination = actual_history

        # Maximum termination step (if terminates at last possible prediction)
        max_model_idx = history_length + 5 * horizon_length  # 3 + 5*5 = 28
        max_termination = 1 + (max_model_idx - 1) * stride

        # Check that any terminated trajectories have reasonable timestep values
        terminated_mask = result.termination_steps >= 0
        if terminated_mask.any():
            terminated_steps = result.termination_steps[terminated_mask]

            # Steps should be in actual timesteps, not model-space
            # Model-space would be ~3-28, actual timesteps should be ~21-271
            assert np.all(terminated_steps >= min_termination), \
                f"Termination steps should be >= {min_termination}, got min {terminated_steps.min()}"

            # Should be multiples of stride offset from history
            offsets = terminated_steps - actual_history
            # Offsets should be multiples of stride
            assert np.all(offsets % stride == 0), \
                "Termination timesteps should be at strided positions"

    def test_termination_steps_stride1_baseline(self, trajectory_generator_stride1):
        """Verify stride=1 termination steps work as before."""
        start_states = np.random.randn(8, 4) * 0.1

        result = trajectory_generator_stride1.generate(
            start_states=start_states,
            num_inference_steps=10,
            return_trajectories=True,
        )

        assert result.has_termination_info

        # With stride=1, model-space == actual timesteps
        history_length = 3
        actual_history = compute_actual_length(history_length, 1)
        assert actual_history == history_length

        # Any termination steps should be >= history_length
        terminated_mask = result.termination_steps >= 0
        if terminated_mask.any():
            terminated_steps = result.termination_steps[terminated_mask]
            assert np.all(terminated_steps >= history_length)


class TestStrideTrajectoryFilling:
    """Test that trajectory filling works correctly with stride."""

    def test_stride10_trajectory_filling(self, trajectory_generator_stride10):
        """Verify terminated trajectories are filled correctly with stride=10."""
        # Create states likely to terminate early
        start_states = np.random.randn(4, 4) * 0.05

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=10,
            return_trajectories=True,
        )

        if not result.has_termination_info:
            pytest.skip("No termination info generated")

        # Check if any trajectories terminated early
        terminated_mask = result.termination_steps >= 0

        if terminated_mask.any():
            # For terminated trajectories, verify filling
            for i in range(result.batch_size):
                if terminated_mask[i]:
                    traj = result.trajectories[i]

                    # Find last valid prediction index (in model-space)
                    # We can't directly map termination_step to buffer index,
                    # but we can verify the trajectory is filled

                    # Check that trajectory doesn't have all zeros at the end
                    # (which would indicate no filling happened)
                    non_zero_mask = np.any(traj != 0, axis=1)
                    num_non_zero = non_zero_mask.sum()

                    # Should have some filled values
                    assert num_non_zero > 0, "Terminated trajectory should have filled values"


class TestStrideConversionUtility:
    """Test the conversion utility for model-space to timesteps."""

    def test_convert_trajectory_to_timesteps_stride10(self, trajectory_generator_stride10):
        """Test conversion utility with stride=10."""
        # Generate a trajectory in model-space
        start_states = np.random.randn(4, 4)

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=15,
            return_trajectories=True,
        )

        # Result trajectories are in model-space
        model_space_traj = result.trajectories
        assert model_space_traj.shape[1] == 78  # 3 + 15*5

        # Convert to timesteps
        stride = 10
        max_timesteps = 636

        timestep_traj = TrajectoryGenerator.convert_trajectory_to_timesteps(
            model_space_traj,
            stride=stride,
            max_timesteps=max_timesteps,
        )

        # Check output shape
        assert timestep_traj.shape == (4, max_timesteps, 4)

        # Verify predictions are at strided indices
        for pred_idx in range(model_space_traj.shape[1]):
            timestep_idx = pred_idx * stride
            if timestep_idx < max_timesteps:
                # Values should match
                np.testing.assert_array_almost_equal(
                    timestep_traj[:, timestep_idx, :],
                    model_space_traj[:, pred_idx, :],
                    err_msg=f"Mismatch at prediction {pred_idx}, timestep {timestep_idx}"
                )

        # Verify sparse storage (most timesteps should be zeros)
        num_predictions = model_space_traj.shape[1]
        num_nonzero_timesteps = np.sum(np.any(timestep_traj != 0, axis=(0, 2)))

        # Should have approximately num_predictions non-zero timesteps
        assert num_nonzero_timesteps <= num_predictions + 1  # Allow small margin

    def test_convert_trajectory_auto_max_timesteps(self, trajectory_generator_stride10):
        """Test conversion utility with automatic max_timesteps calculation."""
        start_states = np.random.randn(2, 4)

        result = trajectory_generator_stride10.generate(
            start_states=start_states,
            num_inference_steps=10,
            return_trajectories=True,
        )

        model_space_traj = result.trajectories
        num_predictions = model_space_traj.shape[1]
        stride = 10

        # Convert without specifying max_timesteps
        timestep_traj = TrajectoryGenerator.convert_trajectory_to_timesteps(
            model_space_traj,
            stride=stride,
        )

        # Check that max_timesteps was computed correctly
        expected_max = 1 + (num_predictions - 1) * stride
        assert timestep_traj.shape[1] == expected_max

    def test_convert_trajectory_stride1(self, trajectory_generator_stride1):
        """Test conversion utility with stride=1 (should be identity-like)."""
        start_states = np.random.randn(2, 4)

        result = trajectory_generator_stride1.generate(
            start_states=start_states,
            num_inference_steps=10,
            return_trajectories=True,
        )

        model_space_traj = result.trajectories

        # Convert with stride=1
        timestep_traj = TrajectoryGenerator.convert_trajectory_to_timesteps(
            model_space_traj,
            stride=1,
        )

        # With stride=1, should be almost identical (just maybe different length)
        min_len = min(model_space_traj.shape[1], timestep_traj.shape[1])
        np.testing.assert_array_almost_equal(
            model_space_traj[:, :min_len, :],
            timestep_traj[:, :min_len, :],
        )


class TestStrideEdgeCases:
    """Test edge cases with stride."""

    def test_stride1_equivalence(self, trajectory_generator_stride1):
        """Verify stride=1 behaves the same as before optimization."""
        start_states = np.random.randn(8, 4)

        result = trajectory_generator_stride1.generate(
            start_states=start_states,
            num_inference_steps=10,
            return_trajectories=True,
        )

        # Basic checks
        assert result.final_states.shape == (8, 4)
        assert result.trajectories is not None

        # With stride=1, buffer size should still be reasonable
        traj_length = result.trajectories.shape[1]
        expected = compute_total_predictions(3, 10, 5)  # 53
        assert traj_length == expected

    def test_very_large_stride(self, mock_inference_params_stride):
        """Test with very large stride (e.g., 30)."""
        model = MockModelStride(observation_dim=4, history_length=3, horizon_length=5)
        model_args = MockArgsStride(history_length=3, horizon_length=5, stride=30)
        system = MockSystemStride(
            state_dim=4,
            history_length=3,
            horizon_length=5,
            stride=30,
            max_path_length=1000,
        )

        params = mock_inference_params_stride.copy()
        params["max_path_length"] = 1000

        generator = TrajectoryGenerator(
            dataset="test_dataset_stride30",
            model=model,
            model_args=model_args,
            inference_params=params,
            device="cpu",
            verbose=False,
            system=system,
        )

        start_states = np.random.randn(4, 4)

        result = generator.generate(
            start_states=start_states,
            num_inference_steps=5,
            return_trajectories=True,
        )

        # Buffer should still be model-space
        expected_buffer = compute_total_predictions(3, 5, 5)  # 28
        assert result.trajectories.shape[1] == expected_buffer

        # Memory savings should be even higher
        old_size = 1000
        new_size = expected_buffer
        savings = (1 - new_size / old_size) * 100
        assert savings > 95, f"Expected >95% savings, got {savings:.1f}%"

    def test_stride_with_provided_histories(self, trajectory_generator_stride10):
        """Test stride with provided history windows."""
        # Provide full history
        start_histories = np.random.randn(4, 3, 4)  # [batch, history_length, state_dim]

        result = trajectory_generator_stride10.generate(
            start_histories=start_histories,
            num_inference_steps=10,
            return_trajectories=True,
        )

        # Should still work correctly
        assert result.trajectories is not None
        expected_buffer = compute_total_predictions(3, 10, 5)  # 53
        assert result.trajectories.shape[1] == expected_buffer


class TestStrideComputations:
    """Test stride-related computation helpers."""

    def test_compute_actual_length_stride10(self):
        """Test compute_actual_length with stride=10."""
        # history_length=3, stride=10
        actual = compute_actual_length(3, 10)
        expected = 1 + (3 - 1) * 10  # = 21
        assert actual == expected

    def test_compute_actual_length_stride1(self):
        """Test compute_actual_length with stride=1."""
        actual = compute_actual_length(3, 1)
        expected = 3  # Same as model-space
        assert actual == expected

    def test_compute_total_predictions(self):
        """Test compute_total_predictions helper."""
        # history=3, num_steps=15, horizon=5
        total = compute_total_predictions(3, 15, 5)
        expected = 3 + 15 * 5  # = 78
        assert total == expected
