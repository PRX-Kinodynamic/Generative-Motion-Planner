"""
Tests for history masking functionality in datasets.

These tests verify:
1. Mask constants have correct values
2. Padding strategies work correctly (zeros, first, last, mirror)
3. Mask creation follows correct convention (MASK_ON=0, MASK_OFF=1)
4. Mask and padding parameters work as expected
"""

import pytest
import torch
import numpy as np

from genMoPlan.datasets.constants import (
    MASK_ON,
    MASK_OFF,
    PADDING_STRATEGY_ZEROS,
    PADDING_STRATEGY_FIRST,
    PADDING_STRATEGY_LAST,
    PADDING_STRATEGY_MIRROR,
    VALID_PADDING_STRATEGIES,
    DEFAULT_HISTORY_MASK_PADDING_VALUE,
    DEFAULT_HISTORY_PADDING_STRATEGY,
    validate_mask,
    validate_padding_strategy,
)
from genMoPlan.datasets.utils import apply_padding, _create_mirror_padding


class TestMaskConstants:
    """Tests for mask constant values and conventions."""

    def test_mask_constant_values(self):
        """Verify MASK_ON and MASK_OFF have correct values."""
        assert MASK_ON == 0.0, "MASK_ON should be 0.0 (position is masked/missing)"
        assert MASK_OFF == 1.0, "MASK_OFF should be 1.0 (position is valid/present)"

    def test_mask_constants_are_different(self):
        """Verify MASK_ON and MASK_OFF are different values."""
        assert MASK_ON != MASK_OFF, "MASK_ON and MASK_OFF must be different"

    def test_mask_convention_for_model(self):
        """
        Test that mask convention works with model formula:
        x = mask * x + (1 - mask) * mask_token

        When mask=MASK_OFF (1.0): keeps original x
        When mask=MASK_ON (0.0): uses mask_token
        """
        x = torch.tensor([1.0, 2.0, 3.0])
        mask_token = torch.tensor([0.0, 0.0, 0.0])

        # Test MASK_OFF keeps original value
        mask = MASK_OFF
        result = mask * x + (1 - mask) * mask_token
        assert torch.allclose(result, x), "MASK_OFF should keep original value"

        # Test MASK_ON uses mask_token
        mask = MASK_ON
        result = mask * x + (1 - mask) * mask_token
        assert torch.allclose(result, mask_token), "MASK_ON should use mask_token"

    def test_valid_padding_strategies(self):
        """Verify all expected padding strategies are defined."""
        expected = ["zeros", "first", "last", "mirror"]
        for strategy in expected:
            assert strategy in VALID_PADDING_STRATEGIES, f"Strategy '{strategy}' should be valid"

    def test_default_values(self):
        """Verify default values are set correctly."""
        assert DEFAULT_HISTORY_MASK_PADDING_VALUE == "zeros", \
            "Default history_mask_padding_value should be 'zeros'"
        assert DEFAULT_HISTORY_PADDING_STRATEGY == "first", \
            "Default history_padding_strategy should be 'first'"


class TestValidateMask:
    """Tests for mask validation function."""

    def test_validate_valid_mask(self):
        """Test that valid masks pass validation."""
        mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert validate_mask(mask) is True

    def test_validate_mask_with_expected_length(self):
        """Test that masks with correct length pass validation."""
        mask = torch.tensor([0.0, 1.0, 1.0, 1.0, 1.0])
        assert validate_mask(mask, expected_len=5) is True

    def test_validate_mask_wrong_length(self):
        """Test that masks with wrong length fail validation."""
        mask = torch.tensor([0.0, 1.0, 1.0])
        with pytest.raises(ValueError, match="Mask length"):
            validate_mask(mask, expected_len=5)

    def test_validate_mask_out_of_range(self):
        """Test that masks with values outside [0,1] fail validation."""
        mask = torch.tensor([0.0, 1.0, 2.0])  # 2.0 is out of range
        with pytest.raises(ValueError, match="Mask values must be in"):
            validate_mask(mask)

    def test_validate_mask_none(self):
        """Test that None mask fails validation."""
        with pytest.raises(ValueError, match="Mask is None"):
            validate_mask(None)


class TestValidatePaddingStrategy:
    """Tests for padding strategy validation function."""

    def test_validate_valid_strategies(self):
        """Test that all valid strategies pass validation."""
        for strategy in VALID_PADDING_STRATEGIES:
            assert validate_padding_strategy(strategy) is True

    def test_validate_invalid_strategy(self):
        """Test that invalid strategies fail validation."""
        with pytest.raises(ValueError, match="Invalid padding strategy"):
            validate_padding_strategy("invalid_strategy")


class TestPaddingStrategies:
    """Tests for different padding strategies in apply_padding."""

    @pytest.fixture
    def sample_trajectory(self):
        """Create a sample trajectory for testing."""
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    def test_zeros_padding_left(self, sample_trajectory):
        """Test zeros padding strategy with pad_left=True."""
        padded = apply_padding(sample_trajectory, 5, pad_left=True, strategy="zeros")

        expected = torch.tensor([
            [0.0, 0.0],  # padded with zeros
            [0.0, 0.0],  # padded with zeros
            [0.0, 0.0],  # padded with zeros
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "Zeros padding should fill with zeros"

    def test_first_padding_left(self, sample_trajectory):
        """Test first padding strategy with pad_left=True."""
        padded = apply_padding(sample_trajectory, 5, pad_left=True, strategy="first")

        expected = torch.tensor([
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "First padding should fill with first element"

    def test_last_padding_left(self, sample_trajectory):
        """Test last padding strategy with pad_left=True."""
        padded = apply_padding(sample_trajectory, 5, pad_left=True, strategy="last")

        expected = torch.tensor([
            [3.0, 4.0],  # padded with last element
            [3.0, 4.0],  # padded with last element
            [3.0, 4.0],  # padded with last element
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "Last padding should fill with last element"

    def test_mirror_padding_left(self, sample_trajectory):
        """Test mirror padding strategy with pad_left=True."""
        padded = apply_padding(sample_trajectory, 5, pad_left=True, strategy="mirror")

        # Mirror reflects: [1,2], [3,4] -> reflected: [3,4], [1,2]
        # Need 3 padding positions, so tile and take first 3: [3,4], [1,2], [3,4]
        expected = torch.tensor([
            [3.0, 4.0],  # reflected[0]
            [1.0, 2.0],  # reflected[1]
            [3.0, 4.0],  # reflected[0] (tiled)
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "Mirror padding should use reflected sequence"

    def test_default_padding_left(self, sample_trajectory):
        """Test default padding behavior (backward compatible with 'first' for pad_left=True)."""
        padded = apply_padding(sample_trajectory, 5, pad_left=True)

        expected = torch.tensor([
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # padded with first element
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "Default left padding should use first element"

    def test_default_padding_right(self, sample_trajectory):
        """Test default padding behavior (backward compatible with 'last' for pad_left=False)."""
        padded = apply_padding(sample_trajectory, 5, pad_left=False)

        expected = torch.tensor([
            [1.0, 2.0],  # original
            [3.0, 4.0],  # original
            [3.0, 4.0],  # padded with last element
            [3.0, 4.0],  # padded with last element
            [3.0, 4.0],  # padded with last element
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "Default right padding should use last element"

    def test_pad_value_overrides_strategy(self, sample_trajectory):
        """Test that explicit pad_value overrides strategy."""
        custom_pad = torch.tensor([9.0, 9.0], dtype=torch.float32)
        padded = apply_padding(sample_trajectory, 5, pad_left=True, pad_value=custom_pad, strategy="zeros")

        expected = torch.tensor([
            [9.0, 9.0],  # custom pad value (overrides zeros strategy)
            [9.0, 9.0],
            [9.0, 9.0],
            [1.0, 2.0],
            [3.0, 4.0],
        ], dtype=torch.float32)

        assert torch.allclose(padded, expected), "pad_value should override strategy"

    def test_no_padding_needed(self, sample_trajectory):
        """Test that no padding is applied when trajectory is already correct length."""
        padded = apply_padding(sample_trajectory, 2, pad_left=True, strategy="zeros")
        assert torch.allclose(padded, sample_trajectory), "No padding should be applied when length matches"


class TestMirrorPadding:
    """Tests specifically for mirror padding behavior."""

    def test_mirror_padding_less_than_sequence_length(self):
        """Test mirror padding when pad_length < sequence length."""
        sequence = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
        padding = _create_mirror_padding(sequence, 2)

        # Reflected: [4, 3, 2, 1], take first 2: [4, 3]
        expected = torch.tensor([[4.0], [3.0]], dtype=torch.float32)
        assert torch.allclose(padding, expected)

    def test_mirror_padding_equal_to_sequence_length(self):
        """Test mirror padding when pad_length == sequence length."""
        sequence = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        padding = _create_mirror_padding(sequence, 3)

        # Reflected: [3, 2, 1]
        expected = torch.tensor([[3.0], [2.0], [1.0]], dtype=torch.float32)
        assert torch.allclose(padding, expected)

    def test_mirror_padding_more_than_sequence_length(self):
        """Test mirror padding when pad_length > sequence length (requires tiling)."""
        sequence = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        padding = _create_mirror_padding(sequence, 5)

        # Reflected: [2, 1], tiled: [2, 1, 2, 1, 2, 1, ...]
        # Take first 5: [2, 1, 2, 1, 2]
        expected = torch.tensor([[2.0], [1.0], [2.0], [1.0], [2.0]], dtype=torch.float32)
        assert torch.allclose(padding, expected)

    def test_mirror_padding_single_element(self):
        """Test mirror padding with single element sequence."""
        sequence = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        padding = _create_mirror_padding(sequence, 3)

        # Reflected: [5, 6], tiled to 3: [5, 6], [5, 6], [5, 6]
        expected = torch.tensor([[5.0, 6.0], [5.0, 6.0], [5.0, 6.0]], dtype=torch.float32)
        assert torch.allclose(padding, expected)

    def test_mirror_padding_zero_length(self):
        """Test mirror padding with zero pad length."""
        sequence = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        padding = _create_mirror_padding(sequence, 0)

        assert padding.shape[0] == 0, "Zero padding length should return empty tensor"
        assert padding.shape[1] == 1, "State dimension should be preserved"


class TestMaskShapeAndValues:
    """Tests for mask shape and value constraints."""

    def test_mask_is_binary(self):
        """Test that masks only contain 0 and 1 values."""
        mask = torch.tensor([MASK_ON, MASK_ON, MASK_OFF, MASK_OFF, MASK_OFF])
        unique_values = torch.unique(mask)

        assert len(unique_values) <= 2, "Mask should only contain two unique values"
        for val in unique_values:
            assert val in [MASK_ON, MASK_OFF], f"Mask value {val} is not MASK_ON or MASK_OFF"

    def test_mask_horizon_never_masked(self):
        """Test that horizon positions should always be MASK_OFF (valid)."""
        history_length = 5
        horizon_length = 3

        # Create a mask with some history masked
        hist_mask = torch.tensor([MASK_ON, MASK_ON, MASK_OFF, MASK_OFF, MASK_OFF])
        hor_mask = torch.full((horizon_length,), MASK_OFF)
        full_mask = torch.cat([hist_mask, hor_mask])

        # Verify horizon portion is all MASK_OFF
        horizon_portion = full_mask[history_length:]
        assert (horizon_portion == MASK_OFF).all(), "Horizon positions should never be masked"


class TestMaskWithMakeIndices:
    """Tests for mask creation with make_indices."""

    def test_history_mask_and_padding_mutual_exclusion(self):
        """Test that history_mask and history_padding cannot both be True."""
        from genMoPlan.datasets.utils import make_indices

        path_lengths = [10, 15]
        with pytest.raises(ValueError, match="Cannot use both history padding and history mask"):
            make_indices(
                path_lengths=path_lengths,
                history_length=5,
                use_history_padding=True,
                horizon_length=3,
                use_horizon_padding=False,
                stride=1,
                use_history_mask=True,  # Both True should raise error
            )

    def test_history_mask_generates_variable_length_indices(self):
        """Test that history_mask generates indices for variable-length histories."""
        from genMoPlan.datasets.utils import make_indices

        path_lengths = [10]
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=3,
            use_history_padding=False,
            horizon_length=2,
            use_horizon_padding=False,
            stride=1,
            use_history_mask=True,  # Enable masking
        )

        # Should have indices with different history lengths
        history_lengths = set()
        for idx in indices:
            hist_len = idx[2] - idx[1]  # history_end - history_start
            history_lengths.add(hist_len)

        # With use_history_mask=True, we should have variable history lengths
        assert len(history_lengths) > 1, \
            "use_history_mask should generate indices with variable history lengths"
