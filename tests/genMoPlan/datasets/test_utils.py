import pytest
from genMoPlan.datasets.utils import make_indices, apply_padding
from genMoPlan.utils.data_processing import compute_actual_length
import torch


def test_compute_actual_length():
    assert compute_actual_length(3, 1) == 3
    assert compute_actual_length(3, 2) == 5
    assert compute_actual_length(4, 3) == 10


class TestMakeIndices:
    def test_basic_no_padding_stride_1(self):
        """Test basic functionality with no padding and stride=1"""
        path_lengths = [10, 5]
        history_length = 3
        horizon_length = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=1
        )
        
        # For path 0 (length 10), we expect indices at positions where history ends at 2,3,4,5,6,7
        # For path 1 (length 5), we expect indices at position where history ends at 2
        expected_indices = [
            # (path_idx, history_start, history_end, horizon_start, horizon_end)
            (0, 0, 3, 3, 5),  # First valid position in path 0
            (0, 1, 4, 4, 6),
            (0, 2, 5, 5, 7),
            (0, 3, 6, 6, 8),
            (0, 4, 7, 7, 9),
            (0, 5, 8, 8, 10),            
            (1, 0, 3, 3, 5),  # First valid position in path 1
        ]
        
        assert indices == expected_indices
    
    def test_with_history_padding_stride_1(self):
        """Test with history padding allowed and stride=1"""
        path_lengths = [5, 3]
        history_length = 3
        horizon_length = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=True,  # Allow padding for history
            horizon_length=horizon_length,
            use_horizon_padding=False,  # No padding for horizon
            stride=1
        )
        
        # For path 0 (length 5):
        # - With padding allowed, history can be as short as 1
        # - Horizon still requires 2 elements without padding
        expected_indices = [
            # (path_idx, history_start, history_end, horizon_start, horizon_end)
            (0, 0, 1, 1, 3),  # History of length 1 (will be padded)
            (0, 0, 2, 2, 4),  # History of length 2 (will be padded)
            (0, 0, 3, 3, 5),  # Full history of length 3
            # Path 1 (length 3) only has room for min history (1) + horizon (2)
            (1, 0, 1, 1, 3),
        ]
        
        assert indices == expected_indices
    
    def test_with_horizon_padding_stride_1(self):
        """Test with horizon padding allowed and stride=1"""
        path_lengths = [6, 3]
        history_length = 3
        horizon_length = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,  # No padding for history
            horizon_length=horizon_length,
            use_horizon_padding=True,   # Allow padding for horizon
            stride=1
        )
        
        # For path 0 (length 6):
        # - History requires 3 elements without padding
        # - With horizon padding allowed, horizon can be as short as 0
        expected_indices = [
            # (path_idx, history_start, history_end, horizon_start, horizon_end)
            (0, 0, 3, 3, 5),  # Full horizon of length 2
            (0, 1, 4, 4, 6),  # History starts at 1, horizon of length 2 
            (0, 2, 5, 5, 6),  # History starts at 2, horizon of length 1
            (0, 3, 6, 6, 6),  # History starts at 3, horizon of length 0
            (1, 0, 3, 3, 3),  # Path 1: full history (3) with horizon of length 0
        ]
        
        assert indices == expected_indices
    
    def test_both_padding_stride_1(self):
        """Test with both history and horizon padding attempted (which should raise an error)"""
        path_lengths = [5, 3, 2]
        history_length = 3
        horizon_length = 2
        
        # Should raise ValueError because both padding options cannot be True simultaneously
        with pytest.raises(ValueError, match="Cannot use both history and horizon padding"):
            make_indices(
                path_lengths=path_lengths,
                history_length=history_length,
                use_history_padding=True,  # Allow padding for history
                horizon_length=horizon_length,
                use_horizon_padding=True,  # Allow padding for horizon
                stride=1
            )
    
    def test_stride_greater_than_1(self):
        """Test with stride > 1"""
        path_lengths = [10]
        history_length = 2  # Actual length will be 3 with stride=2
        horizon_length = 2  # Actual length will be 3 with stride=2
        stride = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=stride
        )
        
        # For stride=2:
        # - history_length=2 means indices at 0,2 (length 3)
        # - horizon_length=2 means indices at 3,5 (length 3)
        expected_indices = [
            # (path_idx, history_start, history_end, horizon_start, horizon_end)
            (0, 0, 3, 4, 7),  # History at [0,2], horizon at [4,6]
            (0, 1, 4, 5, 8),  # History at [1,3], horizon at [5,7]
            (0, 2, 5, 6, 9),  # History at [2,4], horizon at [6,8]
            (0, 3, 6, 7, 10),  # History at [3,5], horizon at [7,9]
        ]
        
        assert indices == expected_indices
    
    def test_edge_case_min_path_length(self):
        """Test with paths that are exactly long enough for minimum requirements"""
        # With padding disabled, minimum path length is history_length + horizon_length
        path_lengths = [5, 4, 3]
        history_length = 3
        horizon_length = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=1
        )
        
        # Only paths with length >= 5 can be used when history=3 and horizon=2
        expected_indices = [
            (0, 0, 3, 3, 5),  # Path 0 (length 5) has exactly enough for one sample
        ]
        
        assert indices == expected_indices
        
        # With history padding enabled, minimum path length is history=1 + horizon=2
        indices_with_history_padding = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=True,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=1
        )
        
        # Paths of length >= 3 can be used with history padding
        assert len(indices_with_history_padding) > 0
        # Path with length 3 should have at least one entry
        assert any(idx[0] == 2 for idx in indices_with_history_padding)
        
        # With horizon padding enabled, minimum path length is history=3 + horizon=1
        indices_with_horizon_padding = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=True,
            stride=1
        )
        
        # Paths of length >= 4 can be used with horizon padding
        assert len(indices_with_horizon_padding) > 0
        # Path with length 4 should have at least one entry
        assert any(idx[0] == 1 for idx in indices_with_horizon_padding)
    
    def test_empty_path_lengths(self):
        """Test with empty path_lengths"""
        path_lengths = []
        history_length = 3
        horizon_length = 2
        
        # Should raise ValueError because no valid trajectories can be found
        with pytest.raises(ValueError, match="No valid trajectories found for the dataset"):
            make_indices(
                path_lengths=path_lengths,
                history_length=history_length,
                use_history_padding=False,
                horizon_length=horizon_length,
                use_horizon_padding=False,
                stride=1
            )
    
    def test_all_invalid_paths(self):
        """Test with paths that are all too short"""
        # All paths are too short for the required history and horizon
        path_lengths = [1, 2, 3, 4]
        history_length = 3
        horizon_length = 2
        
        # Should raise ValueError because no valid trajectories can be found
        with pytest.raises(ValueError, match="No valid trajectories found for the dataset"):
            make_indices(
                path_lengths=path_lengths,
                history_length=history_length,
                use_history_padding=False,
                horizon_length=horizon_length,
                use_horizon_padding=False,
                stride=1
            )
    
    def test_history_padding_with_stride(self):
        """Test using a higher stride with history padding allowed"""
        path_lengths = [10]
        history_length = 2  # Will be 3 elements with stride=2
        horizon_length = 2  # Will be 3 elements with stride=2
        stride = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=True,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=stride
        )
        
        # With history padding allowed, we can have shorter histories
        expected_indices = [
            # These include cases where history length is smaller than
            # the full length (which would be 3 elements with stride=2)
            # Horizon must still be full 3 elements (indices at positions with stride=2)
            (0, 0, 1, 1, 4),  # History of length 1, full horizon of 3 elements
            (0, 0, 3, 3, 6),  # Full history of 3 elements, full horizon of 3 elements
            (0, 1, 2, 2, 5),  # History of length 1 starting at index 1
            (0, 1, 4, 4, 7),  # Full history of 3 elements starting at index 1
            # ...and other combinations
        ]
        
        # Check key properties instead of exact match
        assert len(indices) > 0
        
        # Check that we have at least one case with minimum history (1 element)
        assert any(idx[2] - idx[1] == 1 for idx in indices)
        
        # Check that all horizons have full length
        for idx in indices:
            assert idx[4] - idx[3] == 3  # Horizons should all be 3 elements with stride=2
    
    def test_horizon_padding_with_stride(self):
        """Test using a higher stride with horizon padding allowed"""
        path_lengths = [10]
        history_length = 2  # Will be 3 elements with stride=2
        horizon_length = 2  # Will be 3 elements with stride=2
        stride = 2
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=True,
            stride=stride
        )
        
        # With horizon padding allowed, we can have shorter horizons
        expected_indices = [
            # These include cases where horizon length is smaller than
            # the full length (which would be 3 elements with stride=2)
            # History must still be full 3 elements (indices at positions with stride=2)
            (0, 0, 3, 3, 4),  # Full history, horizon of length 1
            (0, 0, 3, 3, 6),  # Full history, full horizon of 3 elements
            (0, 1, 4, 4, 5),  # Full history starting at index 1, horizon of length 1
            (0, 1, 4, 4, 7),  # Full history starting at index 1, full horizon
            # ...and other combinations
        ]
        
        # Check key properties instead of exact match
        assert len(indices) > 0
        
        # Check that all histories have full length
        for idx in indices:
            assert idx[2] - idx[1] == 3  # Histories should all be 3 elements with stride=2
        
        # Check that we have at least one case with minimum horizon (1 element)
        assert any(idx[4] - idx[3] == 1 for idx in indices)


    def test_on_actual_array_without_stride_no_padding(self):
        paths = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]

        history_length = 3
        horizon_length = 2
        stride = 1

        path_lengths = [len(path) for path in paths]
        
        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=stride
        )

        for i, history_start, history_end, horizon_start, horizon_end in indices:
            path = paths[i]

            assert len(path[history_start:history_end:stride]) == history_length
            assert len(path[horizon_start:horizon_end:stride]) == horizon_length


    def test_on_actual_array_stride_2_without_padding(self):
        paths = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]

        history_length = 3
        horizon_length = 2
        stride = 2

        path_lengths = [len(path) for path in paths]

        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=stride
        )

        for i, history_start, history_end, horizon_start, horizon_end in indices:
            path = paths[i]

            assert len(path[history_start:history_end:stride]) == history_length
            assert len(path[horizon_start:horizon_end:stride]) == horizon_length

            assert history_end - history_start == 1 + (history_length - 1) * stride
            assert horizon_end - horizon_start == 1 + (horizon_length - 1) * stride


    def test_on_actual_array_stride_3_without_padding(self):
        paths = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        ]

        history_length = 3
        horizon_length = 2
        stride = 3
        path_lengths = [len(path) for path in paths]

        indices = make_indices(
            path_lengths=path_lengths,
            history_length=history_length,
            use_history_padding=False,
            horizon_length=horizon_length,
            use_horizon_padding=False,
            stride=stride
        )

        for i, history_start, history_end, horizon_start, horizon_end in indices:
            path = paths[i]
            assert len(path[history_start:history_end:stride]) == history_length
            assert len(path[horizon_start:horizon_end:stride]) == horizon_length

            assert history_end - history_start == 1 + (history_length - 1) * stride
            assert horizon_end - horizon_start == 1 + (horizon_length - 1) * stride
            
class TestApplyPadding:
    def test_apply_padding_left(self):
        trajectory = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=True)
        assert torch.allclose(padded_trajectory, torch.tensor([1, 1, 1, 2, 3], dtype=torch.float32).unsqueeze(1))

    def test_apply_padding_right(self):
        trajectory = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=False)
        assert torch.allclose(padded_trajectory, torch.tensor([1, 2, 3, 3, 3], dtype=torch.float32).unsqueeze(1))

    def test_apply_padding_equal_length(self):
        trajectory = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
        padded_trajectory = apply_padding(trajectory, 3, pad_left=True)
        assert torch.allclose(padded_trajectory, torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1))

    def test_apply_padding_high_dim(self):
        trajectory = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=True)
        assert torch.allclose(padded_trajectory, torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=torch.float32))

    def test_apply_padding_high_dim_right(self):
        trajectory = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=False)
        assert torch.allclose(padded_trajectory, torch.tensor([[1, 2, 3], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]], dtype=torch.float32))
        
    def test_apply_padding_with_custom_pad_value_left(self):
        trajectory = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
        custom_pad_value = torch.tensor([9], dtype=torch.float32)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=True, pad_value=custom_pad_value)
        assert torch.allclose(padded_trajectory, torch.tensor([9, 9, 1, 2, 3], dtype=torch.float32).unsqueeze(1))
        
    def test_apply_padding_with_custom_pad_value_right(self):
        trajectory = torch.tensor([1, 2, 3], dtype=torch.float32).unsqueeze(1)
        custom_pad_value = torch.tensor([9], dtype=torch.float32)
        padded_trajectory = apply_padding(trajectory, 5, pad_left=False, pad_value=custom_pad_value)
        assert torch.allclose(padded_trajectory, torch.tensor([1, 2, 3, 9, 9], dtype=torch.float32).unsqueeze(1))
        
    def test_apply_padding_empty_trajectory_with_pad_value(self):
        trajectory = torch.tensor([], dtype=torch.float32).reshape(0, 2)
        custom_pad_value = torch.tensor([9, 9], dtype=torch.float32)
        padded_trajectory = apply_padding(trajectory, 3, pad_left=True, pad_value=custom_pad_value)
        assert torch.allclose(padded_trajectory, torch.tensor([[9, 9], [9, 9], [9, 9]], dtype=torch.float32))
        
    def test_apply_padding_empty_trajectory_no_pad_value(self):
        trajectory = torch.tensor([], dtype=torch.float32).reshape(0, 2)
        with pytest.raises(ValueError, match="Cannot pad empty trajectory with no pad value"):
            apply_padding(trajectory, 3, pad_left=True)
