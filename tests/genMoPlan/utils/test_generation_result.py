"""Tests for GenerationResult and TerminationResult dataclasses."""

import pytest
import numpy as np

from genMoPlan.utils.generation_result import GenerationResult, TerminationResult


class TestTerminationResult:
    """Tests for TerminationResult dataclass."""

    def test_none_creates_no_termination(self):
        """Test TerminationResult.none() creates result with no terminations."""
        result = TerminationResult.none(5)

        assert result.step.shape == (5,)
        assert result.outcome.shape == (5,)
        assert np.all(result.step == -1)
        assert np.all(result.outcome == -1)
        assert result.any_terminated == False
        assert result.all_terminated == False

    def test_none_with_different_batch_sizes(self):
        """Test TerminationResult.none() with various batch sizes."""
        for batch_size in [1, 10, 100]:
            result = TerminationResult.none(batch_size)
            assert result.step.shape == (batch_size,)
            assert result.outcome.shape == (batch_size,)

    def test_update_with_new_terminations(self):
        """Test update() correctly updates only non-terminated trajectories."""
        # Start with no terminations
        original = TerminationResult.none(5)

        # Create new terminations for indices 0 and 2
        new_step = np.array([-1, -1, 10, -1, -1], dtype=np.int32)
        new_outcome = np.array([-1, -1, 0, -1, -1], dtype=np.int32)
        new_result = TerminationResult(
            step=new_step,
            outcome=new_outcome,
            any_terminated=True,
            all_terminated=False,
        )

        updated = original.update(new_result)

        assert updated.step[2] == 10
        assert updated.outcome[2] == 0
        assert updated.step[0] == -1  # Not terminated
        assert updated.any_terminated == True
        assert updated.all_terminated == False

    def test_update_does_not_overwrite_existing_terminations(self):
        """Test update() preserves already-terminated trajectories."""
        # Start with trajectory 0 already terminated
        original = TerminationResult(
            step=np.array([5, -1, -1], dtype=np.int32),
            outcome=np.array([1, -1, -1], dtype=np.int32),
            any_terminated=True,
            all_terminated=False,
        )

        # Try to terminate trajectory 0 again at a different step
        new_result = TerminationResult(
            step=np.array([20, 10, -1], dtype=np.int32),
            outcome=np.array([0, 0, -1], dtype=np.int32),
            any_terminated=True,
            all_terminated=False,
        )

        updated = original.update(new_result)

        # Trajectory 0 should keep original termination
        assert updated.step[0] == 5
        assert updated.outcome[0] == 1
        # Trajectory 1 should get new termination
        assert updated.step[1] == 10
        assert updated.outcome[1] == 0

    def test_update_all_terminated_flag(self):
        """Test that all_terminated flag is set correctly."""
        original = TerminationResult(
            step=np.array([5, -1], dtype=np.int32),
            outcome=np.array([0, -1], dtype=np.int32),
            any_terminated=True,
            all_terminated=False,
        )

        # Terminate the remaining trajectory
        new_result = TerminationResult(
            step=np.array([-1, 10], dtype=np.int32),
            outcome=np.array([-1, 1], dtype=np.int32),
            any_terminated=True,
            all_terminated=False,
        )

        updated = original.update(new_result)

        assert updated.all_terminated == True
        assert np.all(updated.step >= 0)


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_basic_creation(self):
        """Test basic GenerationResult creation."""
        final_states = np.random.randn(10, 4)
        result = GenerationResult(final_states=final_states)

        assert result.batch_size == 10
        assert result.state_dim == 4
        assert result.trajectories is None
        assert result.termination_steps is None
        assert result.termination_outcomes is None

    def test_with_trajectories(self):
        """Test GenerationResult with full trajectories."""
        final_states = np.random.randn(10, 4)
        trajectories = np.random.randn(10, 50, 4)

        result = GenerationResult(
            final_states=final_states,
            trajectories=trajectories,
        )

        assert result.has_trajectories == True
        assert result.trajectories.shape == (10, 50, 4)

    def test_with_termination_info(self):
        """Test GenerationResult with termination information."""
        final_states = np.random.randn(10, 4)
        termination_steps = np.array([5, -1, 10, -1, -1, 3, -1, -1, -1, 8], dtype=np.int32)
        termination_outcomes = np.array([0, -1, 1, -1, -1, 0, -1, -1, -1, 1], dtype=np.int32)

        result = GenerationResult(
            final_states=final_states,
            termination_steps=termination_steps,
            termination_outcomes=termination_outcomes,
        )

        assert result.has_termination_info == True

    def test_terminated_mask(self):
        """Test terminated_mask() returns correct boolean mask."""
        final_states = np.random.randn(5, 2)
        termination_steps = np.array([5, -1, 10, -1, 3], dtype=np.int32)

        result = GenerationResult(
            final_states=final_states,
            termination_steps=termination_steps,
        )

        mask = result.terminated_mask()

        expected_mask = np.array([True, False, True, False, True])
        np.testing.assert_array_equal(mask, expected_mask)

    def test_terminated_mask_no_termination_info(self):
        """Test terminated_mask() returns None when no termination info."""
        final_states = np.random.randn(5, 2)
        result = GenerationResult(final_states=final_states)

        assert result.terminated_mask() is None

    def test_num_terminated(self):
        """Test num_terminated() returns correct count."""
        final_states = np.random.randn(5, 2)
        termination_steps = np.array([5, -1, 10, -1, 3], dtype=np.int32)

        result = GenerationResult(
            final_states=final_states,
            termination_steps=termination_steps,
        )

        assert result.num_terminated() == 3

    def test_num_terminated_no_terminations(self):
        """Test num_terminated() returns 0 when no trajectories terminated."""
        final_states = np.random.randn(5, 2)
        termination_steps = np.full(5, -1, dtype=np.int32)

        result = GenerationResult(
            final_states=final_states,
            termination_steps=termination_steps,
        )

        assert result.num_terminated() == 0

    def test_num_terminated_no_info(self):
        """Test num_terminated() returns 0 when no termination info available."""
        final_states = np.random.randn(5, 2)
        result = GenerationResult(final_states=final_states)

        assert result.num_terminated() == 0

    def test_has_trajectories_false(self):
        """Test has_trajectories == False when trajectories not provided."""
        final_states = np.random.randn(5, 2)
        result = GenerationResult(final_states=final_states)

        assert result.has_trajectories == False

    def test_has_termination_info_false(self):
        """Test has_termination_info == False when termination info not provided."""
        final_states = np.random.randn(5, 2)
        result = GenerationResult(final_states=final_states)

        assert result.has_termination_info == False
