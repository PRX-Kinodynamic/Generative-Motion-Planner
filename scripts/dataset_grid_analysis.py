from os import path
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm
from collections import Counter

import genMoPlan.utils as utils


def compute_grid_granularity(points, tolerance=1e-6):
    """
    Compute grid granularity from 2D points.
    
    Args:
        points: numpy array of shape (N, 2) containing 2D coordinates
        tolerance: tolerance for considering points as being on the same grid line
    
    Returns:
        dict containing granularity information
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Method 1: Find unique coordinates (works for perfect grids)
    unique_x = np.unique(np.round(x_coords / tolerance) * tolerance)
    unique_y = np.unique(np.round(y_coords / tolerance) * tolerance)
    
    # Method 2: Compute minimum non-zero distances
    def min_nonzero_distance(coords):
        sorted_coords = np.sort(coords)
        diffs = np.diff(sorted_coords)
        nonzero_diffs = diffs[diffs > tolerance]
        return np.min(nonzero_diffs) if len(nonzero_diffs) > 0 else None
    
    min_x_spacing = min_nonzero_distance(x_coords)
    min_y_spacing = min_nonzero_distance(y_coords)
    
    # Method 3: Most common spacing (robust to noise)
    def most_common_spacing(coords, num_bins=1000):
        sorted_coords = np.sort(coords)
        diffs = np.diff(sorted_coords)
        nonzero_diffs = diffs[diffs > tolerance]
        
        if len(nonzero_diffs) == 0:
            return None
            
        # Bin the differences to find most common spacing
        hist, bin_edges = np.histogram(nonzero_diffs, bins=num_bins)
        max_bin_idx = np.argmax(hist)
        return (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    
    common_x_spacing = most_common_spacing(x_coords)
    common_y_spacing = most_common_spacing(y_coords)
    
    # Grid dimensions if points form a perfect grid
    grid_width = len(unique_x) if len(unique_x) > 1 else 1
    grid_height = len(unique_y) if len(unique_y) > 1 else 1
    
    # Check if points actually form a grid
    expected_grid_points = grid_width * grid_height
    actual_points = len(points)
    is_perfect_grid = (actual_points == expected_grid_points)
    
    return {
        'is_perfect_grid': is_perfect_grid,
        'grid_dimensions': (grid_width, grid_height),
        'unique_x_count': len(unique_x),
        'unique_y_count': len(unique_y),
        'min_x_spacing': min_x_spacing,
        'min_y_spacing': min_y_spacing,
        'common_x_spacing': common_x_spacing,
        'common_y_spacing': common_y_spacing,
        'x_range': (np.min(x_coords), np.max(x_coords)),
        'y_range': (np.min(y_coords), np.max(y_coords)),
        'total_points': actual_points,
        'expected_grid_points': expected_grid_points
    }


def print_grid_analysis(grid_info):
    """Print human-readable grid analysis"""
    print("\n=== Grid Analysis ===")
    print(f"Total points: {grid_info['total_points']}")
    print(f"Perfect grid: {grid_info['is_perfect_grid']}")
    print(f"Grid dimensions: {grid_info['grid_dimensions'][0]} x {grid_info['grid_dimensions'][1]}")
    
    if grid_info['is_perfect_grid']:
        print(f"Expected grid points: {grid_info['expected_grid_points']}")
    
    print(f"\nX-axis:")
    print(f"  Unique values: {grid_info['unique_x_count']}")
    print(f"  Range: {grid_info['x_range']}")
    print(f"  Min spacing: {grid_info['min_x_spacing']:.6f}")
    print(f"  Common spacing: {grid_info['common_x_spacing']:.6f}")
    
    print(f"\nY-axis:")
    print(f"  Unique values: {grid_info['unique_y_count']}")
    print(f"  Range: {grid_info['y_range']}")
    print(f"  Min spacing: {grid_info['min_y_spacing']:.6f}")
    print(f"  Common spacing: {grid_info['common_y_spacing']:.6f}")


def main(args):
    dataset_path = path.join("data_trajectories", args.dataset)
    trajectories_path = path.join(dataset_path, "trajectories")

    fnames = utils.get_fnames_to_load(dataset_path, trajectories_path)

    start_points = []

    for fname in tqdm(fnames):
        fpath = path.join(trajectories_path, fname)
        trajectory = utils.read_trajectory(fpath, args.observation_dim)
        start_points.append(trajectory[0])

    start_points = np.array(start_points)
    
    # Compute and display grid granularity
    grid_info = compute_grid_granularity(start_points, tolerance=args.tolerance)
    print_grid_analysis(grid_info)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(start_points[:, 0], start_points[:, 1], s=0.05)
    plt.title(f"Start Points - Grid: {grid_info['grid_dimensions'][0]}x{grid_info['grid_dimensions'][1]}")
    plt.savefig(path.join(dataset_path, "start_points.png"))


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize dataset trajectories")
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument(
        "--observation_dim", type=int, required=True, help="Observation dimension"
    )
    
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, 
        help="Tolerance for grid detection (default: 1e-6)"
    )

    main(parser.parse_args())
