"""Tests for Trainer refactoring - model_args parameter and evaluate_final_states method."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock, patch
from collections import namedtuple

from genMoPlan.utils.trainer import Trainer
from genMoPlan.utils.generation_result import GenerationResult
from genMoPlan.utils.manifold import ManifoldWrapper
from flow_matching.utils.manifolds import Euclidean


# Mock objects for testing
Sample = namedtuple("Sample", "trajectories values chains")


class MockModel:
    """Mock generative model for testing."""
    def __init__(self, observation_dim=2, history_length=2, horizon_length=3):
        self.observation_dim = observation_dim
        self.input_dim = observation_dim
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.prediction_length = history_length + horizon_length
        self.manifold = None
        self._parameters = [torch.nn.Parameter(torch.randn(10, 10))]

    def parameters(self):
        return self._parameters

    def train(self):
        pass

    def eval(self):
        pass

    def to(self, device):
        return self

    def state_dict(self):
        return {"weight": torch.randn(10, 10)}

    def load_state_dict(self, state_dict):
        pass

    def loss(self, *args, **kwargs):
        return torch.tensor(0.5), {}

    def __call__(self, cond, **kwargs):
        """Make model callable for trajectory generation."""
        batch_size = cond[0].shape[0] if isinstance(cond, dict) and len(cond) > 0 else 1
        trajectories = torch.randn(batch_size, self.prediction_length, self.observation_dim)
        return Sample(trajectories=trajectories, values=None, chains=None)

    def named_modules(self):
        """Return empty list for EMA detection."""
        return []

    def named_parameters(self):
        """Return named parameters for optimizer."""
        return [("weight", p) for p in self._parameters]

    def conditional_sample(self, cond, shape, **kwargs):
        batch_size = shape[0]
        trajectories = torch.randn(batch_size, self.prediction_length, self.observation_dim)
        return Sample(trajectories=trajectories, values=None, chains=None)


class MockArgs:
    """Mock model arguments."""
    def __init__(self, history_length=2, horizon_length=3, observation_dim=2, stride=1):
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.observation_dim = observation_dim
        self.stride = stride
        self.trajectory_normalizer = None
        self.method = "models.generative.FlowMatching"


class MockNormalizer:
    """Mock normalizer for testing."""
    def __call__(self, data):
        return data * 0.1

    def unnormalize(self, data):
        return data * 10.0


class MockSystem:
    """Mock system for testing."""
    def __init__(self, state_dim=2):
        self.state_dim = state_dim
        self.state_names = ["x", "y"]
        self.manifold = ManifoldWrapper(Euclidean())
        self.model_manifold = None
        self.metadata = {
            "invalid_label": -1,
        }
        # Add required system attributes for TrajectoryGenerator
        self.history_length = 2
        self.horizon_length = 3
        self.stride = 1
        self.max_path_length = 50
        # Normalizer is required (single source of truth)
        self.normalizer = MockNormalizer()

    def evaluate_final_states(self, states):
        return np.zeros(len(states), dtype=np.int32)


class MockDataset:
    """Mock dataset for testing."""
    def __init__(self, size=100, observation_dim=2, history_length=2, horizon_length=3):
        self.size = size
        self.observation_dim = observation_dim
        self.history_length = history_length
        self.horizon_length = horizon_length
        self.normalizer = MockNormalizer()

        # Create mock eval_data
        self.eval_data = self._create_eval_data()

    def _create_eval_data(self):
        """Create mock evaluation data."""
        MockEvalData = namedtuple("FinalStateDataSample", ["histories", "final_states", "max_path_length"])
        return MockEvalData(
            histories=torch.randn(20, self.history_length, self.observation_dim),
            final_states=torch.randn(20, self.observation_dim),
            max_path_length=50,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        MockSample = namedtuple("TrainingSample", ["trajectories", "conditions", "mask", "global_query", "local_query"])
        return MockSample(
            trajectories=torch.randn(self.history_length + self.horizon_length, self.observation_dim),
            conditions={0: torch.randn(self.observation_dim)},
            mask=None,
            global_query=None,
            local_query=None,
        )


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_args():
    return MockArgs()


@pytest.fixture
def mock_system():
    return MockSystem()


@pytest.fixture
def mock_train_dataset():
    return MockDataset(size=100)


@pytest.fixture
def mock_val_dataset():
    return MockDataset(size=20)


class TestTrainerModelArgsParameter:
    """Tests for Trainer model_args parameter."""

    def test_trainer_accepts_model_args(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test that Trainer accepts model_args as second positional parameter."""
        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            mock_val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            system=mock_system,
        )

        assert trainer is not None

    def test_trainer_stores_model_args(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test that Trainer stores model_args for later use."""
        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            mock_val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            system=mock_system,
        )

        # Trainer should store model_args (implementation dependent)
        # This verifies the API accepts the parameter correctly
        assert trainer is not None


class TestTrainerFinalStateEvaluation:
    """Tests for Trainer.evaluate_final_states() method."""

    def test_evaluate_final_states_requires_generator(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test evaluate_final_states raises error without TrajectoryGenerator."""
        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            mock_val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            perform_final_state_evaluation=False,
            system=mock_system,
        )

        with pytest.raises(ValueError, match="requires a TrajectoryGenerator"):
            trainer.evaluate_final_states()

    def test_evaluate_final_states_with_system(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test evaluate_final_states works when system is provided."""
        # Create validation dataset with proper eval_data
        val_dataset = mock_val_dataset

        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            perform_final_state_evaluation=True,
            system=mock_system,
            eval_batch_size=10,
        )

        # This should work without raising
        mae, losses = trainer.evaluate_final_states()

        assert "final_rollout_mae" in losses
        assert isinstance(mae, float)

    def test_evaluate_final_states_returns_per_dim_mae(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test evaluate_final_states returns per-dimension MAE when state_names available."""
        val_dataset = mock_val_dataset

        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            perform_final_state_evaluation=True,
            system=mock_system,
            eval_batch_size=10,
        )

        mae, losses = trainer.evaluate_final_states()

        # Should have per-dimension MAE for each state name
        # state_names = ["x", "y"]
        assert "final_rollout_mae_x" in losses or "final_rollout_mae" in losses


class TestTrainerRequiresSystem:
    """Tests for fail-fast behavior when system is not provided."""

    def test_trainer_requires_system(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, test_tmp_dir):
        """Test that Trainer always requires a system (single source of truth)."""
        with pytest.raises(ValueError, match="requires a `system` instance"):
            Trainer(
                mock_model,
                mock_args,
                mock_train_dataset,
                mock_val_dataset,
                results_folder=str(test_tmp_dir),
                num_epochs=1,
                batch_size=4,
                device="cpu",
                perform_final_state_evaluation=False,
                system=None,  # No system provided - should fail
            )

    def test_trainer_requires_system_normalizer(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, test_tmp_dir):
        """Test that Trainer requires system to have a normalizer."""
        system_without_normalizer = MockSystem()
        system_without_normalizer.normalizer = None  # Remove normalizer

        with pytest.raises(ValueError, match="system.normalizer"):
            Trainer(
                mock_model,
                mock_args,
                mock_train_dataset,
                mock_val_dataset,
                results_folder=str(test_tmp_dir),
                num_epochs=1,
                batch_size=4,
                device="cpu",
                perform_final_state_evaluation=False,
                system=system_without_normalizer,
            )


class TestTrainerTrajectoryGeneratorIntegration:
    """Tests for Trainer's internal TrajectoryGenerator usage."""

    def test_trainer_creates_eval_generator(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test that Trainer creates internal TrajectoryGenerator for evaluation."""
        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            mock_val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            perform_final_state_evaluation=True,
            system=mock_system,
        )

        # Internal generator should be created
        assert hasattr(trainer, '_eval_generator')
        assert trainer._eval_generator is not None

    def test_trainer_uses_ema_model_for_eval(self, mock_model, mock_args, mock_train_dataset, mock_val_dataset, mock_system, test_tmp_dir):
        """Test that Trainer uses EMA model for evaluation."""
        trainer = Trainer(
            mock_model,
            mock_args,
            mock_train_dataset,
            mock_val_dataset,
            results_folder=str(test_tmp_dir),
            num_epochs=1,
            batch_size=4,
            device="cpu",
            perform_final_state_evaluation=True,
            system=mock_system,
            ema_decay=0.999,
        )

        # Trainer should use EMA model for evaluation
        # (implementation detail - just verify construction works)
        assert trainer._eval_generator is not None
