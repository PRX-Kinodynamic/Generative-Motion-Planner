"""Tests for genMoPlan/utils/data_processing.py stride computation functions."""

import pytest
import numpy as np

from genMoPlan.utils.data_processing import (
    compute_actual_length,
    compute_num_inference_steps,
    compute_max_path_length,
)


class TestComputeActualLength:
    """Tests for compute_actual_length function."""

    def test_stride_1(self):
        """With stride=1, actual length equals input length."""
        assert compute_actual_length(1, 1) == 1
        assert compute_actual_length(4, 1) == 4
        assert compute_actual_length(10, 1) == 10

    def test_stride_2(self):
        """With stride=2, actual length = 1 + (length-1)*2."""
        assert compute_actual_length(1, 2) == 1  # 1 + 0*2 = 1
        assert compute_actual_length(2, 2) == 3  # 1 + 1*2 = 3
        assert compute_actual_length(4, 2) == 7  # 1 + 3*2 = 7
        assert compute_actual_length(5, 2) == 9  # 1 + 4*2 = 9

    def test_stride_3(self):
        """With stride=3, actual length = 1 + (length-1)*3."""
        assert compute_actual_length(1, 3) == 1  # 1 + 0*3 = 1
        assert compute_actual_length(2, 3) == 4  # 1 + 1*3 = 4
        assert compute_actual_length(4, 3) == 10  # 1 + 3*3 = 10

    def test_length_1_any_stride(self):
        """With length=1, stride has no effect (always returns 1)."""
        assert compute_actual_length(1, 1) == 1
        assert compute_actual_length(1, 5) == 1
        assert compute_actual_length(1, 100) == 1


class TestComputeNumInferenceSteps:
    """Tests for compute_num_inference_steps function."""

    def test_basic_stride_1(self):
        """Basic case with stride=1."""
        # max_path=100, history=4, horizon=8, stride=1
        # remaining = 100 - 4 = 96, steps = ceil(96/8) = 12
        assert compute_num_inference_steps(100, 4, 8, 1) == 12

    def test_exact_fit(self):
        """When path length exactly fits history + n*horizon."""
        # max_path=20, history=4, horizon=8, stride=1
        # remaining = 20 - 4 = 16, steps = ceil(16/8) = 2
        assert compute_num_inference_steps(20, 4, 8, 1) == 2

    def test_needs_extra_step(self):
        """When there's a remainder, needs one more step."""
        # max_path=21, history=4, horizon=8, stride=1
        # remaining = 21 - 4 = 17, steps = ceil(17/8) = 3
        assert compute_num_inference_steps(21, 4, 8, 1) == 3

    def test_with_stride_2(self):
        """Test with stride=2."""
        # history_length=4, stride=2 -> actual_history = 1 + 3*2 = 7
        # horizon_length=8, stride=2 -> actual_horizon = 1 + 7*2 = 15
        # max_path=50 -> remaining = 50 - 7 = 43, steps = ceil(43/15) = 3
        assert compute_num_inference_steps(50, 4, 8, 2) == 3

    def test_minimum_one_step(self):
        """Should always return at least 1 step."""
        # Even if max_path <= history
        assert compute_num_inference_steps(4, 4, 8, 1) >= 1
        assert compute_num_inference_steps(2, 4, 8, 1) >= 1

    def test_small_horizon(self):
        """Test with small horizon values."""
        # max_path=10, history=2, horizon=1, stride=1
        # remaining = 10 - 2 = 8, steps = ceil(8/1) = 8
        assert compute_num_inference_steps(10, 2, 1, 1) == 8


class TestComputeMaxPathLength:
    """Tests for compute_max_path_length function."""

    def test_basic_stride_1(self):
        """Basic case with stride=1."""
        # n_steps=12, history=4, horizon=8, stride=1
        # max_path = 4 + 12*8 = 100
        assert compute_max_path_length(12, 4, 8, 1) == 100

    def test_with_stride_2(self):
        """Test with stride=2."""
        # history_length=4, stride=2 -> actual_history = 7
        # horizon_length=8, stride=2 -> actual_horizon = 15
        # max_path = 7 + 3*15 = 52
        assert compute_max_path_length(3, 4, 8, 2) == 52

    def test_single_step(self):
        """Test with single inference step."""
        # n_steps=1, history=4, horizon=8, stride=1
        # max_path = 4 + 1*8 = 12
        assert compute_max_path_length(1, 4, 8, 1) == 12


class TestRoundTrip:
    """Test that compute_num_inference_steps and compute_max_path_length are inverses."""

    @pytest.mark.parametrize("history,horizon,stride", [
        (4, 8, 1),
        (4, 8, 2),
        (2, 4, 1),
        (8, 16, 3),
        (1, 1, 1),
        (10, 5, 2),
    ])
    def test_max_path_to_steps_roundtrip(self, history, horizon, stride):
        """compute_max_path_length(compute_num_inference_steps(max_path)) >= max_path."""
        for max_path in [10, 50, 100, 200]:
            n_steps = compute_num_inference_steps(max_path, history, horizon, stride)
            recovered_path = compute_max_path_length(n_steps, history, horizon, stride)
            # Recovered path should be >= original (due to ceiling in compute_num_inference_steps)
            assert recovered_path >= max_path, (
                f"Failed for max_path={max_path}, history={history}, "
                f"horizon={horizon}, stride={stride}: got {recovered_path}"
            )

    @pytest.mark.parametrize("history,horizon,stride", [
        (4, 8, 1),
        (4, 8, 2),
        (2, 4, 1),
        (8, 16, 3),
    ])
    def test_steps_to_max_path_roundtrip(self, history, horizon, stride):
        """compute_num_inference_steps(compute_max_path_length(n_steps)) == n_steps."""
        for n_steps in [1, 5, 10, 20]:
            max_path = compute_max_path_length(n_steps, history, horizon, stride)
            recovered_steps = compute_num_inference_steps(max_path, history, horizon, stride)
            # This should be exact when starting from steps
            assert recovered_steps == n_steps, (
                f"Failed for n_steps={n_steps}, history={history}, "
                f"horizon={horizon}, stride={stride}: got {recovered_steps}"
            )
