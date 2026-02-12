import pytest
import numpy as np
import torch
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from collections import namedtuple

from genMoPlan.utils.trajectory_generator import TrajectoryGenerator, ResolvedParams
from genMoPlan.utils.generation_result import GenerationResult, TerminationResult
from genMoPlan.systems import Outcome


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


class MockSystem:
    """Minimal mock system required by TrajectoryGenerator."""
    def __init__(self, *, state_dim: int, history_length: int, horizon_length: int, stride: int, max_path_length: int):
        self.state_dim = state_dim
        self.state_names = [f"dim_{i}" for i in range(state_dim)]
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.stride = stride
        self.max_path_length = max_path_length
        # Provide a normalizer instance (single source of truth)
        self.normalizer = MockNormalizer()

    def evaluate_final_states(self, states):
        # Never terminate in these tests unless explicitly needed
        return np.full(states.shape[0], Outcome.INVALID.value, dtype=np.int32)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations"""
    import uuid
    base_tmp = "/common/home/st1122/Projects/genMoPlan/tmp"
    os.makedirs(base_tmp, exist_ok=True)
    temp_path = os.path.join(base_tmp, f"test_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_path, exist_ok=True)
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
    system = MockSystem(
        state_dim=model_args.observation_dim,
        history_length=model_args.history_length,
        horizon_length=model_args.horizon_length,
        stride=model_args.stride,
        max_path_length=mock_inference_params["max_path_length"],
    )

    generator = TrajectoryGenerator(
        dataset="test_dataset",
        model=model,
        model_args=model_args,
        inference_params=mock_inference_params,
        device="cpu",
        verbose=False,
        system=system,
    )
    return generator


class TestTrajectoryGeneratorInitialization:
    """Tests for TrajectoryGenerator initialization"""

    def test_init_with_model_and_args(self, mock_inference_params):
        """Test initialization with pre-loaded model and args"""
        model = MockModel()
        model_args = MockArgs()
        system = MockSystem(
            state_dim=model_args.observation_dim,
            history_length=model_args.history_length,
            horizon_length=model_args.horizon_length,
            stride=model_args.stride,
            max_path_length=mock_inference_params["max_path_length"],
        )

        generator = TrajectoryGenerator(
            dataset="test_dataset",
            model=model,
            model_args=model_args,
            inference_params=mock_inference_params,
            device="cpu",
            system=system,
        )

        assert generator.model is model
        assert generator.model_args is model_args
        assert generator.history_length == 2
        assert generator.horizon_length == 3
        assert generator.stride == 1
        assert generator.method_name == "flow_matching"

    def test_init_without_model_requires_model_path(self, mock_inference_params):
        """Test that initialization without model requires model_path"""
        system = MockSystem(state_dim=2, history_length=2, horizon_length=3, stride=1, max_path_length=10)
        with pytest.raises(ValueError, match="model_path.*is required"):
            TrajectoryGenerator(
                dataset="test_dataset",
                model=None,
                model_args=None,
                inference_params=mock_inference_params,
                device="cpu",
                system=system,
            )

    def test_init_with_model_requires_args(self, mock_inference_params):
        """Test that providing model requires model_args"""
        model = MockModel()
        system = MockSystem(state_dim=2, history_length=2, horizon_length=3, stride=1, max_path_length=10)

        with pytest.raises(ValueError, match="model_args.*must be provided"):
            TrajectoryGenerator(
                dataset="test_dataset",
                model=model,
                model_args=None,
                inference_params=mock_inference_params,
                device="cpu",
                system=system,
            )

    def test_init_without_inference_params_requires_dataset(self):
        """Test that initialization without inference_params requires dataset"""
        model = MockModel()
        model_args = MockArgs()
        system = MockSystem(state_dim=2, history_length=2, horizon_length=3, stride=1, max_path_length=10)

        with pytest.raises(ValueError, match="dataset.*must be provided"):
            TrajectoryGenerator(
                dataset=None,
                model=model,
                model_args=model_args,
                inference_params=None,
                device="cpu",
                system=system,
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


# =============================================================================
# Tests for generate() method and related refactoring
# =============================================================================


class MockSystemTerminating:
    """Mock system with termination checking based on state norm."""
    def __init__(self, termination_threshold=1.0):
        self.termination_threshold = termination_threshold
        self.state_dim = 2
        self.state_names = ["x", "y"]
        self.normalizer = MockNormalizer()
        self.history_length = 2
        self.horizon_length = 3
        self.stride = 1
        self.max_path_length = 50

    def evaluate_final_states(self, states):
        """Vectorized evaluation of final states."""
        norms = np.linalg.norm(states, axis=1)
        outcomes = np.where(
            norms < self.termination_threshold,
            Outcome.SUCCESS.value,
            Outcome.FAILURE.value
        )
        return outcomes.astype(np.int32)


@pytest.fixture
def trajectory_generator_terminating(mock_inference_params):
    """Create a TrajectoryGenerator with termination checking."""
    model = MockModel()
    model_args = MockArgs()
    system = MockSystemTerminating()

    generator = TrajectoryGenerator(
        dataset="test_dataset",
        model=model,
        model_args=model_args,
        inference_params=mock_inference_params,
        device="cpu",
        verbose=False,
        system=system,
    )
    return generator


class TestResolvedParams:
    """Tests for ResolvedParams dataclass."""

    def test_resolved_params_creation(self):
        """Test ResolvedParams can be created with all fields."""
        params = ResolvedParams(
            batch_size=32,
            max_path_length=100,
            num_inference_steps=10,
            horizon_length=8,
            history_length=4,
            stride=1,
            state_dim=6,
            model_sequence_length=12,
        )

        assert params.batch_size == 32
        assert params.max_path_length == 100
        assert params.num_inference_steps == 10
        assert params.horizon_length == 8
        assert params.history_length == 4
        assert params.stride == 1
        assert params.state_dim == 6
        assert params.model_sequence_length == 12


class TestGenerateMethodInputValidation:
    """Tests for generate() method input validation."""

    def test_generate_requires_start_states_or_histories(self, trajectory_generator):
        """Test generate() requires either start_states or start_histories."""
        with pytest.raises(ValueError, match="Either start_states or start_histories"):
            trajectory_generator.generate()

    def test_generate_rejects_both_start_states_and_histories(self, trajectory_generator):
        """Test generate() rejects both start_states and start_histories."""
        start_states = np.random.randn(5, 2)
        start_histories = np.random.randn(5, 2, 2)

        with pytest.raises(ValueError, match="Cannot provide both"):
            trajectory_generator.generate(
                start_states=start_states,
                start_histories=start_histories,
            )

    def test_generate_accepts_start_states(self, trajectory_generator):
        """Test generate() accepts start_states parameter."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
        )

        assert isinstance(result, GenerationResult)
        assert result.batch_size == 5

    def test_generate_accepts_start_histories(self, trajectory_generator):
        """Test generate() accepts start_histories parameter."""
        start_histories = np.random.randn(5, 2, 2)

        result = trajectory_generator.generate(
            start_histories=start_histories,
            num_inference_steps=2,
        )

        assert isinstance(result, GenerationResult)
        assert result.batch_size == 5


class TestGenerateMethodOutputFormat:
    """Tests for generate() method output format."""

    def test_generate_returns_generation_result(self, trajectory_generator):
        """Test generate() returns GenerationResult dataclass."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
        )

        assert isinstance(result, GenerationResult)

    def test_generate_final_states_shape(self, trajectory_generator):
        """Test generate() returns final_states with correct shape."""
        start_states = np.random.randn(10, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
        )

        assert result.final_states.shape == (10, 2)

    def test_generate_with_return_trajectories(self, trajectory_generator):
        """Test generate() returns trajectories when requested."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
            return_trajectories=True,
        )

        assert result.has_trajectories
        assert result.trajectories is not None
        assert result.trajectories.shape[0] == 5
        assert result.trajectories.shape[2] == 2

    def test_generate_without_return_trajectories(self, trajectory_generator):
        """Test generate() does not return trajectories by default."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
            return_trajectories=False,
        )

        assert not result.has_trajectories
        assert result.trajectories is None


class TestGenerateWithStartHistories:
    """Tests for generate() with start_histories parameter."""

    def test_generate_uses_provided_histories(self, trajectory_generator):
        """Test generate() uses exact history windows when provided."""
        start_histories = np.ones((5, 2, 2)) * 0.5

        result = trajectory_generator.generate(
            start_histories=start_histories,
            num_inference_steps=2,
        )

        assert result.batch_size == 5

    def test_generate_extracts_start_states_from_histories(self, trajectory_generator):
        """Test generate() extracts last state from histories for output sizing."""
        start_histories = np.zeros((5, 2, 2))
        start_histories[:, -1, :] = 1.0

        result = trajectory_generator.generate(
            start_histories=start_histories,
            num_inference_steps=1,
        )

        assert result.final_states.shape[0] == 5


class TestGenerateTerminationChecking:
    """Tests for termination checking in generate()."""

    def test_generate_with_system_returns_termination_info(self, trajectory_generator_terminating):
        """Test generate() returns termination info when system is provided."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator_terminating.generate(
            start_states=start_states,
            num_inference_steps=2,
        )

        assert result.has_termination_info
        assert result.termination_steps is not None
        assert result.termination_outcomes is not None
        assert result.termination_steps.shape == (5,)
        assert result.termination_outcomes.shape == (5,)

    def test_generate_with_system_returns_result(self, trajectory_generator):
        """Test generate() returns a valid result."""
        start_states = np.random.randn(5, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            num_inference_steps=2,
        )

        assert isinstance(result, GenerationResult)


class TestRequireSystemParameter:
    """Tests for mandatory system requirement."""

    def test_init_fails_without_system(self, mock_inference_params):
        """TrajectoryGenerator must always receive a system."""
        model = MockModel()
        model_args = MockArgs()

        with pytest.raises(ValueError, match="requires a `system` instance"):
            TrajectoryGenerator(
                dataset="test_dataset",
                model=model,
                model_args=model_args,
                inference_params=mock_inference_params,
                device="cpu",
                verbose=False,
                system=None,
            )

    def test_init_succeeds_with_system(self, mock_inference_params):
        """TrajectoryGenerator should initialize when system is provided."""
        model = MockModel()
        model_args = MockArgs()
        system = MockSystem(
            state_dim=2,
            history_length=2,
            horizon_length=3,
            stride=1,
            max_path_length=10,
        )

        generator = TrajectoryGenerator(
            dataset="test_dataset",
            model=model,
            model_args=model_args,
            inference_params=mock_inference_params,
            device="cpu",
            verbose=False,
            system=system,
        )

        assert generator.system is system


class TestResolveParams:
    """Tests for _resolve_params method."""

    def test_resolve_params_with_max_path_length(self, trajectory_generator):
        """Test _resolve_params with max_path_length calculates num_inference_steps."""
        params = trajectory_generator._resolve_params(
            batch_size=10,
            max_path_length=20,
            num_inference_steps=None,
            horizon_length=None,
        )

        assert params.max_path_length == 20
        assert params.num_inference_steps > 0

    def test_resolve_params_with_num_inference_steps(self, trajectory_generator):
        """Test _resolve_params with num_inference_steps calculates max_path_length."""
        params = trajectory_generator._resolve_params(
            batch_size=10,
            max_path_length=None,
            num_inference_steps=5,
            horizon_length=None,
        )

        assert params.num_inference_steps == 5
        assert params.max_path_length > 0

    def test_resolve_params_rejects_both(self, trajectory_generator):
        """Test _resolve_params rejects both max_path_length and num_inference_steps."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            trajectory_generator._resolve_params(
                batch_size=10,
                max_path_length=20,
                num_inference_steps=5,
                horizon_length=None,
            )

    def test_resolve_params_uses_defaults(self, trajectory_generator):
        """Test _resolve_params uses defaults from inference_params."""
        params = trajectory_generator._resolve_params(
            batch_size=10,
            max_path_length=None,
            num_inference_steps=None,
            horizon_length=None,
        )

        assert params.max_path_length == trajectory_generator.inference_params["max_path_length"]


class TestGenerateMultipleRuns:
    """Tests for generate_multiple_runs using new generate() method."""

    def test_generate_multiple_runs_returns_correct_shape(self, trajectory_generator):
        """Test generate_multiple_runs returns correctly shaped arrays."""
        start_states = np.random.randn(10, 2)

        final_states, trajectories = trajectory_generator.generate_multiple_runs(
            start_states=start_states,
            n_runs=3,
            num_inference_steps=2,
            return_trajectories=False,
        )

        assert final_states.shape == (3, 10, 2)
        assert trajectories is None

    def test_generate_multiple_runs_with_trajectories(self, trajectory_generator):
        """Test generate_multiple_runs with trajectory return."""
        start_states = np.random.randn(5, 2)

        final_states, trajectories = trajectory_generator.generate_multiple_runs(
            start_states=start_states,
            n_runs=2,
            num_inference_steps=2,
            return_trajectories=True,
        )

        assert final_states.shape == (2, 5, 2)
        assert trajectories is not None
        assert trajectories.shape[0] == 2
        assert trajectories.shape[1] == 5


class TestBatching:
    """Tests for batching in generate()."""

    def test_generate_respects_batch_size(self, trajectory_generator):
        """Test generate() respects batch_size parameter."""
        start_states = np.random.randn(100, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            batch_size=10,
            num_inference_steps=2,
        )

        assert result.final_states.shape[0] == 100

    def test_generate_handles_uneven_batches(self, trajectory_generator):
        """Test generate() handles when total doesn't divide evenly by batch_size."""
        start_states = np.random.randn(15, 2)

        result = trajectory_generator.generate(
            start_states=start_states,
            batch_size=4,
            num_inference_steps=2,
        )

        assert result.final_states.shape[0] == 15
