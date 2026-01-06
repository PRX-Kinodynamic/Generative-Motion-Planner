import pytest
import numpy as np
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from collections import namedtuple

from genMoPlan.utils.trajectory_generator import TrajectoryGenerator


# Mock objects for testing
Sample = namedtuple("Sample", "trajectories values chains")


class MockModel:
    """Mock model for testing trajectory generation"""
    def __init__(self, observation_dim=2):
        self.observation_dim = observation_dim

    def __call__(self, cond, verbose=False, return_chain=False, **kwargs):
        """Simulate model forward pass returning trajectories"""
        batch_size = len(cond[0])
        # Return a mock Sample with trajectories
        trajectories = torch.randn(batch_size, 5, self.observation_dim)  # 5 = history + horizon
        return Sample(trajectories=trajectories, values=None, chains=None)

    def parameters(self):
        return [torch.tensor([1.0])]


class MockArgs:
    """Mock model arguments"""
    def __init__(self):
        self.history_length = 2
        self.horizon_length = 3
        self.observation_dim = 2
        self.stride = 1
        self.trajectory_normalizer = None
        self.method = "models.generative.FlowMatching"


class MockNormalizer:
    """Mock normalizer for testing normalization"""
    def __call__(self, data):
        return data * 0.1  # Scale down

    def unnormalize(self, data):
        return data * 10.0  # Scale up


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_inference_params():
    """Mock inference parameters"""
    return {
        "max_path_length": 10,
        "batch_size": 2,
        "n_runs": 3,
        "final_state_directory": "final_states",
        "post_process_fns": None,
        "post_process_fn_kwargs": {},
        "conditional_sample_kwargs": {},
        "load_ema": False,
    }


@pytest.fixture
def trajectory_generator(mock_inference_params):
    """Create a TrajectoryGenerator instance with mocked model"""
    model = MockModel()
    model_args = MockArgs()

    generator = TrajectoryGenerator(
        dataset="test_dataset",
        model=model,
        model_args=model_args,
        inference_params=mock_inference_params,
        device="cpu",
        verbose=False,
    )
    return generator


class TestTrajectoryGeneratorInitialization:
    """Tests for TrajectoryGenerator initialization"""

    def test_init_with_model_and_args(self, mock_inference_params):
        """Test initialization with pre-loaded model and args"""
        model = MockModel()
        model_args = MockArgs()

        generator = TrajectoryGenerator(
            dataset="test_dataset",
            model=model,
            model_args=model_args,
            inference_params=mock_inference_params,
            device="cpu",
        )

        assert generator.model is model
        assert generator.model_args is model_args
        assert generator.history_length == 2
        assert generator.horizon_length == 3
        assert generator.stride == 1
        assert generator.method_name == "flow_matching"

    def test_init_without_model_requires_model_path(self, mock_inference_params):
        """Test that initialization without model requires model_path"""
        with pytest.raises(ValueError, match="model_path.*is required"):
            TrajectoryGenerator(
                dataset="test_dataset",
                model=None,
                model_args=None,
                inference_params=mock_inference_params,
                device="cpu",
            )

    def test_init_with_model_requires_args(self, mock_inference_params):
        """Test that providing model requires model_args"""
        model = MockModel()

        with pytest.raises(ValueError, match="model_args.*must be provided"):
            TrajectoryGenerator(
                dataset="test_dataset",
                model=model,
                model_args=None,
                inference_params=mock_inference_params,
                device="cpu",
            )

    def test_init_without_inference_params_requires_dataset(self):
        """Test that initialization without inference_params requires dataset"""
        model = MockModel()
        model_args = MockArgs()

        with pytest.raises(ValueError, match="dataset.*must be provided"):
            TrajectoryGenerator(
                dataset=None,
                model=model,
                model_args=model_args,
                inference_params=None,
                device="cpu",
            )


class TestConfigurationMethods:
    """Tests for configuration setter methods"""

    def test_set_batch_size(self, trajectory_generator):
        """Test setting batch size updates both instance and inference params"""
        trajectory_generator.set_batch_size(128)

        assert trajectory_generator.batch_size == 128
        assert trajectory_generator.inference_params["batch_size"] == 128

    def test_set_horizon_length_only(self, trajectory_generator):
        """Test setting only horizon length"""
        trajectory_generator.set_horizon_and_max_path_lengths(horizon_length=5)

        assert trajectory_generator.horizon_length == 5
        assert trajectory_generator.inference_params["horizon_length"] == 5

    def test_set_max_path_length(self, trajectory_generator):
        """Test setting max path length directly"""
        trajectory_generator.set_horizon_and_max_path_lengths(max_path_length=20)

        assert trajectory_generator.max_path_length == 20
        assert trajectory_generator.inference_params["max_path_length"] == 20

    def test_set_num_inference_steps(self, trajectory_generator):
        """Test setting max path length via num_inference_steps"""
        # stride=1, history=2, horizon=3
        # actual_hist = 1 + (2-1)*1 = 2
        # actual_horz = 1 + (3-1)*1 = 3
        # max_path = 2 + (5 * 3) = 17
        trajectory_generator.set_horizon_and_max_path_lengths(num_inference_steps=5)

        assert trajectory_generator.max_path_length == 17

    def test_cannot_set_both_max_path_and_inference_steps(self, trajectory_generator):
        """Test that setting both max_path_length and num_inference_steps raises error"""
        with pytest.raises(ValueError, match="Cannot set both"):
            trajectory_generator.set_horizon_and_max_path_lengths(
                max_path_length=20,
                num_inference_steps=5
            )


class TestTimestampManagement:
    """Tests for timestamp and path management"""

    def test_timestamp_setter(self, trajectory_generator, temp_dir):
        """Test setting timestamp creates directory"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"
        trajectory_generator.timestamp = "2025-01-01_12-00-00"

        assert trajectory_generator.timestamp == "2025-01-01_12-00-00"
        assert os.path.exists(trajectory_generator.gen_traj_path)

    def test_timestamp_setter_requires_model_path(self, trajectory_generator):
        """Test setting timestamp without model_path raises error"""
        trajectory_generator.model_path = None

        with pytest.raises(ValueError, match="Cannot manage timestamps"):
            trajectory_generator.timestamp = "2025-01-01_12-00-00"

    def test_ensure_timestamp_creates_if_none(self, trajectory_generator, temp_dir):
        """Test ensure_timestamp creates timestamp if none exists"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        assert trajectory_generator.timestamp is None
        trajectory_generator.ensure_timestamp()
        assert trajectory_generator.timestamp is not None

    def test_ensure_timestamp_uses_provided_value(self, trajectory_generator, temp_dir):
        """Test ensure_timestamp uses provided value"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        trajectory_generator.ensure_timestamp("2025-01-01_12-00-00")
        assert trajectory_generator.timestamp == "2025-01-01_12-00-00"

    def test_ensure_timestamp_preserves_existing(self, trajectory_generator, temp_dir):
        """Test ensure_timestamp preserves existing timestamp when called without value"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        trajectory_generator.timestamp = "2025-01-01_12-00-00"
        trajectory_generator.ensure_timestamp()
        assert trajectory_generator.timestamp == "2025-01-01_12-00-00"


class TestTrajectoryGeneration:
    """Tests for trajectory generation methods"""

    def test_generate_trajectories_returns_correct_shape(self, trajectory_generator):
        """Test that generate_trajectories returns correct shape"""
        start_states = np.random.randn(10, 2)

        with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
            # Mock returns trajectories with shape (batch, path_length, dim)
            # Use side_effect to return correctly-sized arrays for each call
            def mock_generate(batch_states, **kwargs):
                batch_size = len(batch_states)
                return np.random.randn(batch_size, 10, 2)

            mock_gen.side_effect = mock_generate

            result = trajectory_generator.generate_trajectories(
                start_states,
                num_inference_steps=3,
            )

            assert result.shape[0] == 10  # batch size
            assert result.shape[2] == 2   # observation dim

    def test_generate_trajectories_only_final_states(self, trajectory_generator):
        """Test generating only final states using generate_final_states method"""
        start_states = np.random.randn(10, 2)

        with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
            # Mock returns only final states with shape (batch, dim)
            def mock_generate(batch_states, **kwargs):
                batch_size = len(batch_states)
                return np.random.randn(batch_size, 2)

            mock_gen.side_effect = mock_generate

            result = trajectory_generator.generate_final_states(
                start_states,
                num_inference_steps=3,
            )

            assert result.shape == (10, 2)

    def test_generate_trajectories_requires_one_of_max_path_or_steps(self, trajectory_generator):
        """Test that generate_trajectories requires exactly one of max_path_length or num_inference_steps"""
        start_states = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="exactly one of"):
            trajectory_generator.generate_trajectories(start_states)

    def test_generate_trajectories_batching(self, trajectory_generator):
        """Test that large batches are properly split"""
        start_states = np.random.randn(100, 2)

        with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
            def mock_generate(batch_states, **kwargs):
                batch_size = len(batch_states)
                return np.random.randn(batch_size, 2)

            mock_gen.side_effect = mock_generate

            result = trajectory_generator.generate_final_states(
                start_states,
                batch_size=10,
                num_inference_steps=3,
            )

            # Should be called 10 times (100 / 10)
            assert mock_gen.call_count == 10
            assert result.shape == (100, 2)

    def test_generate_trajectories_applies_normalization(self, trajectory_generator):
        """Test that normalization is applied during generation"""
        start_states = np.random.randn(5, 2)

        # Mock normalizer
        with patch.object(trajectory_generator, '_get_normalizer') as mock_norm_getter:
            mock_normalizer = MockNormalizer()
            mock_norm_getter.return_value = mock_normalizer

            with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
                def mock_generate(batch_states, **kwargs):
                    batch_size = len(batch_states)
                    return np.random.randn(batch_size, 10, 2)

                mock_gen.side_effect = mock_generate

                result = trajectory_generator.generate_trajectories(
                    start_states,
                    num_inference_steps=3,
                )

                # Check that normalizer was called
                mock_norm_getter.assert_called_once()

    def test_generate_final_states_returns_correct_shape(self, trajectory_generator):
        """Test that generate_final_states returns only final states"""
        start_states = np.random.randn(10, 2)

        with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
            def mock_generate(batch_states, **kwargs):
                batch_size = len(batch_states)
                return np.random.randn(batch_size, 2)

            mock_gen.side_effect = mock_generate

            result = trajectory_generator.generate_final_states(
                start_states,
                num_inference_steps=3,
            )

            assert result.shape == (10, 2)  # Only final states, not full trajectories

    def test_generate_final_states_vs_generate_trajectories(self, trajectory_generator):
        """Test that generate_final_states and generate_trajectories with only_return_final_states are equivalent"""
        start_states = np.random.randn(5, 2)

        with patch.object(trajectory_generator, '_generate_raw_trajectories') as mock_gen:
            def mock_generate(batch_states, **kwargs):
                batch_size = len(batch_states)
                if kwargs.get('only_return_final_states'):
                    return np.random.randn(batch_size, 2)
                else:
                    return np.random.randn(batch_size, 10, 2)

            mock_gen.side_effect = mock_generate

            # Both should call the internal method with only_return_final_states appropriately
            final_states = trajectory_generator.generate_final_states(
                start_states,
                num_inference_steps=3,
            )

            trajectories = trajectory_generator.generate_trajectories(
                start_states,
                num_inference_steps=3,
            )

            assert final_states.shape == (5, 2)
            assert trajectories.shape[0] == 5
            assert trajectories.ndim == 3  # Full trajectories


class TestMultipleRunGeneration:
    """Tests for generate_multiple_runs method"""

    def test_generate_multiple_runs_returns_correct_shape(self, trajectory_generator):
        """Test that generate_multiple_runs returns correct shaped arrays"""
        start_states = np.random.randn(10, 2)

        with patch.object(trajectory_generator, 'generate_trajectories') as mock_gen:
            # Return final states
            mock_gen.return_value = np.random.randn(10, 2)

            final_states, trajectories = trajectory_generator.generate_multiple_runs(
                start_states,
                n_runs=3,
                num_inference_steps=2,
                discard_trajectories=True,
            )

            # final_states should be (n_runs, num_states, dim)
            assert final_states.shape == (3, 10, 2)
            assert trajectories is None  # discarded

    def test_generate_multiple_runs_keeps_trajectories(self, trajectory_generator):
        """Test that generate_multiple_runs can keep full trajectories"""
        start_states = np.random.randn(10, 2)

        with patch.object(trajectory_generator, '_generate_trajectories_or_final_states') as mock_gen:
            # Return full trajectories (batch, path_length, dim)
            mock_gen.return_value = np.random.randn(10, 15, 2)

            final_states, trajectories = trajectory_generator.generate_multiple_runs(
                start_states,
                n_runs=3,
                num_inference_steps=2,
                discard_trajectories=False,
            )

            assert final_states.shape == (3, 10, 2)
            assert trajectories.shape == (3, 10, 15, 2)

    def test_generate_multiple_runs_requires_positive_runs(self, trajectory_generator):
        """Test that n_runs must be positive"""
        start_states = np.random.randn(10, 2)

        with pytest.raises(ValueError, match="must be positive"):
            trajectory_generator.generate_multiple_runs(
                start_states,
                n_runs=0,
                num_inference_steps=2,
            )


class TestSavingAndLoading:
    """Tests for saving and loading final states"""

    def test_save_final_states_creates_files(self, trajectory_generator, temp_dir):
        """Test that save_final_states creates files for each run"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        start_states = np.random.randn(5, 2)
        # final_states should be (n_runs, num_states, dim)
        final_states = np.random.randn(3, 5, 2)

        trajectory_generator.save_final_states(
            start_states,
            final_states,
            timestamp="2025-01-01_12-00-00",
        )

        # Check that files were created
        for run_idx in range(3):
            file_path = os.path.join(
                temp_dir,
                "final_states",
                "2025-01-01_12-00-00",
                f"final_states_run_{run_idx}.txt"
            )
            assert os.path.exists(file_path)

    def test_save_final_states_requires_3d_array(self, trajectory_generator, temp_dir):
        """Test that save_final_states requires 3D final_states array"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        start_states = np.random.randn(5, 2)
        final_states = np.random.randn(5, 2)  # Wrong shape (2D instead of 3D)

        with pytest.raises(ValueError, match="3D array"):
            trajectory_generator.save_final_states(
                start_states,
                final_states,
                timestamp="2025-01-01_12-00-00",
            )

    def test_load_saved_final_states_reads_files(self, trajectory_generator, temp_dir):
        """Test that load_saved_final_states reads saved files correctly"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        # First save some data
        start_states = np.random.randn(5, 2)
        final_states = np.random.randn(3, 5, 2)

        trajectory_generator.save_final_states(
            start_states,
            final_states,
            timestamp="2025-01-01_12-00-00",
        )

        # Reset generator state
        trajectory_generator._timestamp = None

        # Now load it back
        loaded_states, n_runs, timestamp = trajectory_generator.load_saved_final_states(
            expected_runs=3,
            timestamp="2025-01-01_12-00-00",
            parallel=False,
        )

        assert n_runs == 3
        assert timestamp == "2025-01-01_12-00-00"
        assert loaded_states.shape == (3, 5, 2)

    def test_load_saved_final_states_without_timestamp_uses_latest(self, trajectory_generator, temp_dir):
        """Test that load without timestamp uses latest available"""
        trajectory_generator.model_path = temp_dir
        trajectory_generator.final_state_directory = "final_states"

        # Save with two different timestamps
        start_states = np.random.randn(5, 2)
        final_states = np.random.randn(2, 5, 2)

        trajectory_generator.save_final_states(
            start_states,
            final_states,
            timestamp="2025-01-01_12-00-00",
        )

        trajectory_generator.save_final_states(
            start_states,
            final_states,
            timestamp="2025-01-02_12-00-00",  # Later timestamp
        )

        trajectory_generator._timestamp = None

        # Load without specifying timestamp
        loaded_states, n_runs, timestamp = trajectory_generator.load_saved_final_states(
            expected_runs=2,
            parallel=False,
        )

        # Should load the latest (2025-01-02)
        assert timestamp == "2025-01-02_12-00-00"


class TestInternalHelpers:
    """Tests for internal helper methods"""

    def test_compute_inference_steps_calculation(self, trajectory_generator):
        """Test that _compute_inference_steps calculates correctly"""
        # stride=1, history=2, horizon=3
        # actual_hist = 1 + (2-1)*1 = 2
        # actual_horz = 1 + (3-1)*1 = 3
        # remaining = 20 - 2 = 18
        # steps = ceil(18 / 3) = 6

        steps = trajectory_generator._compute_inference_steps(
            max_path_length=20,
            horizon_length=3
        )

        assert steps == 6

    def test_compute_inference_steps_requires_max_path_length(self, trajectory_generator):
        """Test that _compute_inference_steps requires max_path_length"""
        with pytest.raises(ValueError, match="max_path_length.*must be provided"):
            trajectory_generator._compute_inference_steps(
                max_path_length=None,
                horizon_length=3
            )

    def test_flatten_for_plotting_4d_array(self):
        """Test flattening 4D trajectory array for plotting"""
        # Shape: (num_states, n_runs, path_length, dim)
        trajectories = np.random.randn(5, 3, 10, 2)

        result = TrajectoryGenerator._flatten_for_plotting(trajectories)

        # Should flatten to (num_states * n_runs, path_length, dim)
        assert result.shape == (15, 10, 2)

    def test_flatten_for_plotting_3d_array(self):
        """Test flattening 3D trajectory array (already flat)"""
        trajectories = np.random.randn(15, 10, 2)

        result = TrajectoryGenerator._flatten_for_plotting(trajectories)

        assert result.shape == (15, 10, 2)

    def test_ensure_run_first_transposes_if_needed(self):
        """Test _ensure_run_first transposes when first dim matches start_states"""
        start_states = np.random.randn(10, 2)
        final_states = np.random.randn(10, 3, 2)  # (num_states, n_runs, dim)

        result = TrajectoryGenerator._ensure_run_first(final_states, start_states)

        # Should transpose to (n_runs, num_states, dim)
        assert result.shape == (3, 10, 2)

    def test_ensure_run_first_preserves_if_already_correct(self):
        """Test _ensure_run_first preserves shape if already run-first"""
        start_states = np.random.randn(10, 2)
        final_states = np.random.randn(3, 10, 2)  # Already (n_runs, num_states, dim)

        result = TrajectoryGenerator._ensure_run_first(final_states, start_states)

        assert result.shape == (3, 10, 2)
