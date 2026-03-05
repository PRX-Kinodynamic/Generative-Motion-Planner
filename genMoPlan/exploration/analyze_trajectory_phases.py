"""
Trajectory Phase Analysis for Cartpole Dataset

This script analyzes cartpole trajectories to understand:
1. How trajectories move from starting state to final state
2. What phases exist (swing-up, stabilizing, equilibrium)
3. How the dynamics distribution affects training

Usage:
    python -m genMoPlan.exploration.analyze_trajectory_phases
    
    Or from the project root:
    python genMoPlan/exploration/analyze_trajectory_phases.py
"""

import os
import sys
import numpy as np

# Set matplotlib backend before importing pyplot (for headless running)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.cartpole_pybullet import read_trajectory, max_path_length, state_names

# ============== CONFIGURATION ==============
OUTPUT_DIR = Path(__file__).parent / "cartpole_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase detection thresholds (will be computed adaptively)
N_SAMPLE_TRAJECTORIES = 100  # Number of trajectories to analyze in detail
RANDOM_SEED = 42

# ============== DATA LOADING ==============

def get_dataset_path():
    """Get the path to the cartpole dataset trajectories."""
    from genMoPlan.utils.paths import get_data_trajectories_path
    
    # Get base data trajectories path and append dataset name
    base_path = get_data_trajectories_path()
    dataset_path = Path(base_path) / "cartpole_pybullet" / "trajectories"
    return dataset_path


def load_trajectories_from_path(dataset_path, max_trajectories=None):
    """Load trajectories from the dataset directory."""
    trajectories = []
    
    # Find all trajectory files (could be .csv or .txt)
    traj_files = sorted(list(dataset_path.glob("*.csv")) + list(dataset_path.glob("*.txt")))
    
    # Filter out non-trajectory files
    traj_files = [f for f in traj_files if not f.name.startswith('shuffled') and not f.name.startswith('roa')]
    
    if max_trajectories:
        traj_files = traj_files[:max_trajectories]
    
    print(f"Loading {len(traj_files)} trajectories from {dataset_path}")
    
    for traj_file in traj_files:
        try:
            traj = read_trajectory(str(traj_file))
            trajectories.append(traj)
        except Exception as e:
            print(f"Error loading {traj_file}: {e}")
            continue
    
    print(f"Loaded {len(trajectories)} trajectories successfully")
    return trajectories


def load_trajectories_via_utils(dataset_name="cartpole_pybullet", max_trajectories=None):
    """Load trajectories using the project's utility functions."""
    from genMoPlan.utils.trajectory import load_trajectories as utils_load_trajectories
    
    print(f"Loading trajectories for dataset: {dataset_name}")
    trajectories = utils_load_trajectories(
        dataset=dataset_name,
        read_trajectory_fn=read_trajectory,
        dataset_size=max_trajectories,
        parallel=True
    )
    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


# ============== DYNAMICS COMPUTATION ==============

def compute_dynamics_magnitude(trajectory):
    """
    Compute per-timestep dynamics magnitude (state change rate).
    
    Args:
        trajectory: (T, 4) array of states [x, theta, x_dot, theta_dot]
    
    Returns:
        dynamics: (T-1,) array of dynamics magnitudes
    """
    # Compute state differences
    state_diffs = np.diff(trajectory, axis=0)
    
    # Compute magnitude of change (L2 norm)
    dynamics = np.linalg.norm(state_diffs, axis=1)
    
    return dynamics


def compute_per_component_dynamics(trajectory):
    """
    Compute per-component dynamics (absolute state changes).
    
    Returns:
        dict with keys: x, theta, x_dot, theta_dot
        Each value is (T-1,) array
    """
    state_diffs = np.abs(np.diff(trajectory, axis=0))
    
    return {
        'x': state_diffs[:, 0],
        'theta': state_diffs[:, 1],
        'x_dot': state_diffs[:, 2],
        'theta_dot': state_diffs[:, 3],
    }


# ============== PHASE DETECTION ==============

def detect_phases(trajectory, dynamics=None, equilibrium_threshold=0.01):
    """
    Segment trajectory into phases based on dynamics magnitude.
    
    Phases:
    - SWING_UP: High dynamics, theta far from 0
    - STABILIZING: Decreasing dynamics, theta approaching 0
    - EQUILIBRIUM: Low dynamics, theta near 0
    
    Args:
        trajectory: (T, 4) array
        dynamics: (T-1,) array of dynamics magnitudes (computed if None)
        equilibrium_threshold: threshold for equilibrium detection
    
    Returns:
        phases: list of (start_idx, end_idx, phase_label)
        equilibrium_start: timestep when equilibrium begins
    """
    if dynamics is None:
        dynamics = compute_dynamics_magnitude(trajectory)
    
    T = len(trajectory)
    
    # Compute rolling average of dynamics for smoother detection
    window = min(10, len(dynamics) // 10)
    if window > 0:
        dynamics_smooth = np.convolve(dynamics, np.ones(window)/window, mode='same')
    else:
        dynamics_smooth = dynamics
    
    # Detect equilibrium: low dynamics AND theta near 0
    theta = trajectory[:, 1]
    theta_near_zero = np.abs(theta) < 0.1  # ~6 degrees
    
    # Pad dynamics_smooth to match trajectory length
    dynamics_padded = np.concatenate([[dynamics_smooth[0]], dynamics_smooth])
    low_dynamics = dynamics_padded < equilibrium_threshold
    
    equilibrium_mask = theta_near_zero & low_dynamics
    
    # Find first sustained equilibrium (at least 20 consecutive timesteps)
    equilibrium_start = None
    consecutive = 0
    required_consecutive = 20
    
    for t in range(T):
        if equilibrium_mask[t]:
            consecutive += 1
            if consecutive >= required_consecutive and equilibrium_start is None:
                equilibrium_start = t - required_consecutive + 1
        else:
            consecutive = 0
    
    if equilibrium_start is None:
        equilibrium_start = T  # Never reached equilibrium
    
    # Define phases
    phases = []
    
    # Swing-up: from start until theta first approaches 0
    theta_first_near_zero = np.argmax(np.abs(theta) < 0.3) if np.any(np.abs(theta) < 0.3) else T
    
    if theta_first_near_zero > 0:
        phases.append((0, theta_first_near_zero, 'SWING_UP'))
    
    # Stabilizing: from first theta near zero until equilibrium
    if theta_first_near_zero < equilibrium_start:
        phases.append((theta_first_near_zero, equilibrium_start, 'STABILIZING'))
    
    # Equilibrium: from equilibrium_start to end
    if equilibrium_start < T:
        phases.append((equilibrium_start, T, 'EQUILIBRIUM'))
    
    return phases, equilibrium_start


def adaptive_slice_trajectory(trajectory, min_slice_length=25, variance_threshold=0.5):
    """
    Split trajectory into slices of similar consecutive movements.
    Uses variance of dynamics within sliding windows.
    
    Args:
        trajectory: (T, 4) array
        min_slice_length: minimum length of each slice
        variance_threshold: threshold for detecting slice boundaries
    
    Returns:
        slices: list of (start_idx, end_idx, avg_dynamics)
    """
    dynamics = compute_dynamics_magnitude(trajectory)
    T = len(dynamics)
    
    if T < min_slice_length * 2:
        return [(0, len(trajectory), np.mean(dynamics))]
    
    slices = []
    current_start = 0
    
    # Use change in dynamics variance to detect boundaries
    window = min_slice_length
    
    t = window
    while t < T - window:
        # Compare variance of left and right windows
        left_var = np.var(dynamics[t-window:t])
        right_var = np.var(dynamics[t:t+window])
        
        left_mean = np.mean(dynamics[t-window:t])
        right_mean = np.mean(dynamics[t:t+window])
        
        # Detect boundary if there's a significant change
        mean_change = abs(left_mean - right_mean) / (left_mean + 1e-6)
        
        if mean_change > variance_threshold and (t - current_start) >= min_slice_length:
            slices.append((current_start, t, np.mean(dynamics[current_start:t])))
            current_start = t
        
        t += 1
    
    # Add final slice
    slices.append((current_start, len(trajectory), np.mean(dynamics[current_start:])))
    
    return slices


# ============== ANALYSIS ==============

def analyze_single_trajectory(trajectory, trajectory_idx=0):
    """
    Analyze one trajectory and return statistics.
    """
    dynamics = compute_dynamics_magnitude(trajectory)
    phases, equilibrium_start = detect_phases(trajectory, dynamics)
    slices = adaptive_slice_trajectory(trajectory)
    
    return {
        'idx': trajectory_idx,
        'length': len(trajectory),
        'dynamics': dynamics,
        'phases': phases,
        'slices': slices,
        'equilibrium_start': equilibrium_start,
        'final_state': trajectory[-1],
        'initial_state': trajectory[0],
        'mean_dynamics': np.mean(dynamics),
        'trajectory': trajectory,
    }


def analyze_dataset(trajectories):
    """
    Aggregate analysis across all trajectories.
    """
    results = []
    
    for i, traj in enumerate(trajectories):
        result = analyze_single_trajectory(traj, i)
        results.append(result)
        
        if (i + 1) % 100 == 0:
            print(f"Analyzed {i + 1}/{len(trajectories)} trajectories")
    
    # Aggregate statistics
    equilibrium_starts = [r['equilibrium_start'] for r in results]
    trajectory_lengths = [r['length'] for r in results]
    
    # Phase duration statistics
    phase_durations = defaultdict(list)
    for r in results:
        for start, end, label in r['phases']:
            phase_durations[label].append(end - start)
    
    # Compute summary
    summary = {
        'n_trajectories': len(trajectories),
        'avg_length': np.mean(trajectory_lengths),
        'avg_equilibrium_start': np.mean(equilibrium_starts),
        'std_equilibrium_start': np.std(equilibrium_starts),
        'phase_durations': {k: (np.mean(v), np.std(v)) for k, v in phase_durations.items()},
        'results': results,
    }
    
    return summary


# ============== TRAINING DATA DISTRIBUTION ==============

def analyze_training_distribution(trajectories, stride=25, history_length=1, horizon_length=15):
    """
    Analyze how training windows are distributed across phases.
    """
    from genMoPlan.utils import compute_actual_length
    
    actual_history = compute_actual_length(history_length, stride)
    actual_horizon = compute_actual_length(horizon_length, stride)
    window_length = actual_history + actual_horizon
    
    print(f"\nTraining window analysis:")
    print(f"  stride={stride}, history_length={history_length}, horizon_length={horizon_length}")
    print(f"  actual_history={actual_history}, actual_horizon={actual_horizon}")
    print(f"  window covers {window_length} timesteps")
    
    # Count windows per phase
    phase_window_counts = defaultdict(int)
    total_windows = 0
    
    # For each trajectory, find what phase each possible window start falls into
    for traj in trajectories:
        T = len(traj)
        phases, _ = detect_phases(traj)
        
        # Create phase lookup
        phase_at_t = ['UNKNOWN'] * T
        for start, end, label in phases:
            for t in range(start, min(end, T)):
                phase_at_t[t] = label
        
        # Count windows
        for window_start in range(0, T - window_length + 1, stride):
            window_end = window_start + window_length
            
            # Determine phase of window (use midpoint)
            midpoint = (window_start + window_end) // 2
            if midpoint < T:
                phase = phase_at_t[midpoint]
                phase_window_counts[phase] += 1
                total_windows += 1
    
    print(f"\nTraining window distribution across phases:")
    for phase, count in sorted(phase_window_counts.items()):
        pct = 100 * count / total_windows if total_windows > 0 else 0
        print(f"  {phase}: {count} windows ({pct:.1f}%)")
    
    return phase_window_counts, total_windows


# ============== VISUALIZATION ==============

def plot_trajectory_with_phases(result, save_path=None):
    """
    Plot all 4 state dimensions over time with phase regions highlighted.
    """
    trajectory = result['trajectory']
    phases = result['phases']
    T = len(trajectory)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    colors = {'SWING_UP': 'red', 'STABILIZING': 'orange', 'EQUILIBRIUM': 'green', 'UNKNOWN': 'gray'}
    
    time = np.arange(T)
    
    for ax, (state_idx, state_name) in zip(axes, enumerate(state_names)):
        # Plot state
        ax.plot(time, trajectory[:, state_idx], 'b-', linewidth=1)
        ax.set_ylabel(state_name)
        ax.grid(True, alpha=0.3)
        
        # Highlight phases
        for start, end, label in phases:
            ax.axvspan(start, end, alpha=0.2, color=colors.get(label, 'gray'), label=label if ax == axes[0] else None)
    
    axes[-1].set_xlabel('Timestep')
    axes[0].set_title(f'Trajectory {result["idx"]} - Phase Analysis (Equilibrium starts at t={result["equilibrium_start"]})')
    
    # Add legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[phase], alpha=0.3) for phase in colors if phase != 'UNKNOWN']
    labels = [phase for phase in colors if phase != 'UNKNOWN']
    axes[0].legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_dynamics_over_time(result, save_path=None):
    """
    Plot dynamics magnitude over time.
    """
    dynamics = result['dynamics']
    phases = result['phases']
    
    fig, ax = plt.subplots(figsize=(14, 4))
    
    time = np.arange(len(dynamics))
    ax.plot(time, dynamics, 'b-', linewidth=0.5, alpha=0.7)
    
    # Add rolling average
    window = 20
    if len(dynamics) > window:
        rolling_avg = np.convolve(dynamics, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window//2, window//2 + len(rolling_avg)), rolling_avg, 'r-', linewidth=2, label='Rolling avg')
    
    # Highlight phases
    colors = {'SWING_UP': 'red', 'STABILIZING': 'orange', 'EQUILIBRIUM': 'green'}
    for start, end, label in phases:
        ax.axvspan(start, min(end, len(dynamics)), alpha=0.15, color=colors.get(label, 'gray'))
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Dynamics Magnitude')
    ax.set_title(f'Trajectory {result["idx"]} - Dynamics Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_adaptive_slices(result, save_path=None):
    """
    Visualize adaptive slices with different colors.
    """
    trajectory = result['trajectory']
    slices = result['slices']
    T = len(trajectory)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # Generate colors for slices
    n_slices = len(slices)
    # Handle both old and new matplotlib API for colormaps
    try:
        cmap = plt.colormaps.get_cmap('tab10')
    except AttributeError:
        cmap = plt.cm.get_cmap('tab10')
    
    time = np.arange(T)
    
    for ax, (state_idx, state_name) in zip(axes, enumerate(state_names)):
        ax.plot(time, trajectory[:, state_idx], 'k-', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(state_name)
        ax.grid(True, alpha=0.3)
        
        # Color each slice
        for i, (start, end, avg_dyn) in enumerate(slices):
            color = cmap(i % 10)
            ax.axvspan(start, end, alpha=0.3, color=color)
            ax.plot(time[start:end], trajectory[start:end, state_idx], color=color, linewidth=1.5)
    
    axes[-1].set_xlabel('Timestep')
    axes[0].set_title(f'Trajectory {result["idx"]} - Adaptive Slices ({n_slices} slices)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_phase_duration_distribution(summary, save_path=None):
    """
    Box plot of phase durations across all trajectories.
    """
    results = summary['results']
    
    phase_durations = defaultdict(list)
    for r in results:
        for start, end, label in r['phases']:
            phase_durations[label].append(end - start)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phases = ['SWING_UP', 'STABILIZING', 'EQUILIBRIUM']
    colors = ['red', 'orange', 'green']
    
    data = [phase_durations.get(p, [0]) for p in phases]
    bp = ax.boxplot(data, labels=phases, patch_artist=True)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    ax.set_ylabel('Duration (timesteps)')
    ax.set_title('Phase Duration Distribution Across Trajectories')
    ax.grid(True, alpha=0.3)
    
    # Add statistics text
    for i, (phase, durations) in enumerate(zip(phases, data)):
        if durations:
            mean = np.mean(durations)
            std = np.std(durations)
            ax.text(i + 1, ax.get_ylim()[1] * 0.95, f'μ={mean:.0f}\nσ={std:.0f}', 
                   ha='center', va='top', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_equilibrium_start_histogram(summary, save_path=None):
    """
    Histogram of when trajectories reach equilibrium.
    """
    results = summary['results']
    equilibrium_starts = [r['equilibrium_start'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(equilibrium_starts, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(equilibrium_starts), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(equilibrium_starts):.0f}')
    ax.axvline(np.median(equilibrium_starts), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(equilibrium_starts):.0f}')
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Equilibrium Start Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_dynamics_histogram(summary, save_path=None):
    """
    Histogram of dynamics magnitude across all trajectories.
    """
    results = summary['results']
    
    all_dynamics = np.concatenate([r['dynamics'] for r in results])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(all_dynamics, bins=100, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(all_dynamics), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(all_dynamics):.4f}')
    ax.axvline(np.median(all_dynamics), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(all_dynamics):.4f}')
    
    ax.set_xlabel('Dynamics Magnitude')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Dynamics Magnitude (State Change Rate)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_initial_vs_final_states(summary, save_path=None):
    """
    Scatter plot comparing initial and final states.
    """
    results = summary['results']
    
    initial_states = np.array([r['initial_state'] for r in results])
    final_states = np.array([r['final_state'] for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for ax, (idx, name) in zip(axes.flatten(), enumerate(state_names)):
        ax.scatter(initial_states[:, idx], final_states[:, idx], alpha=0.5, s=10)
        ax.set_xlabel(f'Initial {name}')
        ax.set_ylabel(f'Final {name}')
        ax.set_title(f'{name}: Initial vs Final')
        ax.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'r--', alpha=0.5, label='y=x')
        
        # Add horizontal/vertical lines at 0
        ax.axhline(0, color='green', linestyle=':', alpha=0.5)
        ax.axvline(0, color='green', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_full_trajectory_timeline(result, save_path=None):
    """
    Plot the full trajectory timeline showing theta evolution from start to finish.
    This visualization shows the complete picture of how the trajectory reaches equilibrium.
    """
    trajectory = result['trajectory']
    equilibrium_start = result['equilibrium_start']
    T = len(trajectory)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    time = np.arange(T)
    theta = trajectory[:, 1]  # theta is index 1
    
    # Top plot: Theta over full timeline
    ax1 = axes[0]
    ax1.plot(time, theta, 'b-', linewidth=1.5, label='θ (angle)')
    ax1.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Equilibrium (θ=0)')
    ax1.axhline(np.pi, color='red', linestyle=':', alpha=0.5, label='θ=π')
    ax1.axhline(-np.pi, color='red', linestyle=':', alpha=0.5, label='θ=-π')
    
    # Mark key timesteps
    ax1.axvline(equilibrium_start, color='green', linestyle='-', linewidth=2, alpha=0.5)
    ax1.annotate(f'Equilibrium reached\nt={equilibrium_start}', 
                 xy=(equilibrium_start, 0), xytext=(equilibrium_start + 20, 1.5),
                 fontsize=10, ha='left',
                 arrowprops=dict(arrowstyle='->', color='green'))
    
    # Mark t=351 (first rollout step end)
    if T > 351:
        ax1.axvline(351, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.annotate('t=351\n(Rollout Step 1)', xy=(351, ax1.get_ylim()[1]), 
                     xytext=(351, ax1.get_ylim()[1] * 0.9), fontsize=9, ha='center', color='orange')
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('θ (radians)', fontsize=12)
    ax1.set_title(f'Trajectory {result["idx"]} - Full Timeline (θ angle over time)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, T)
    
    # Bottom plot: All 4 state components
    ax2 = axes[1]
    colors = ['blue', 'red', 'green', 'purple']
    for i, (name, color) in enumerate(zip(state_names, colors)):
        ax2.plot(time, trajectory[:, i], color=color, linewidth=1, alpha=0.8, label=name)
    
    ax2.axvline(equilibrium_start, color='gray', linestyle='-', linewidth=2, alpha=0.5, label=f'Equilibrium (t={equilibrium_start})')
    ax2.axvline(351, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('State Value (normalized)', fontsize=12)
    ax2.set_title('All State Components Over Time', fontsize=12)
    ax2.legend(loc='upper right', ncol=5)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, T)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_trajectory_summary_comparison(results, save_path=None):
    """
    Plot multiple trajectories overlaid to show the common pattern.
    Shows how all trajectories converge to equilibrium.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Find max length
    max_T = max(len(r['trajectory']) for r in results)
    
    # Plot each trajectory's theta
    for i, result in enumerate(results):
        trajectory = result['trajectory']
        theta = trajectory[:, 1]
        time = np.arange(len(theta))
        alpha = 0.3 if len(results) > 10 else 0.7
        ax.plot(time, theta, 'b-', linewidth=0.5, alpha=alpha)
    
    # Compute and plot average trajectory
    # Pad trajectories to same length and compute mean
    all_thetas = []
    for result in results:
        theta = result['trajectory'][:, 1]
        # Pad with final value
        padded = np.pad(theta, (0, max_T - len(theta)), mode='edge')
        all_thetas.append(padded)
    
    mean_theta = np.mean(all_thetas, axis=0)
    std_theta = np.std(all_thetas, axis=0)
    time = np.arange(max_T)
    
    ax.plot(time, mean_theta, 'r-', linewidth=2, label='Mean trajectory')
    ax.fill_between(time, mean_theta - std_theta, mean_theta + std_theta, 
                    color='red', alpha=0.2, label='±1 std')
    
    # Reference lines
    ax.axhline(0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Equilibrium (θ=0)')
    ax.axvline(351, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='t=351 (Rollout Step 1)')
    
    # Compute average equilibrium start
    avg_eq_start = np.mean([r['equilibrium_start'] for r in results])
    ax.axvline(avg_eq_start, color='green', linestyle='-', linewidth=2, alpha=0.5)
    ax.annotate(f'Avg equilibrium\nt={avg_eq_start:.0f}', 
                xy=(avg_eq_start, 0), xytext=(avg_eq_start + 30, 1.5),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('θ (radians)', fontsize=12)
    ax.set_title(f'All {len(results)} Trajectories - θ Evolution Over Time\n(Shows convergence to equilibrium)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_T)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_training_window_distribution(phase_counts, total_windows, save_path=None):
    """
    Pie chart of training window distribution across phases.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    phases = list(phase_counts.keys())
    counts = [phase_counts[p] for p in phases]
    colors = {'SWING_UP': 'red', 'STABILIZING': 'orange', 'EQUILIBRIUM': 'green', 'UNKNOWN': 'gray'}
    pie_colors = [colors.get(p, 'gray') for p in phases]
    
    wedges, texts, autotexts = ax.pie(counts, labels=phases, autopct='%1.1f%%', 
                                       colors=pie_colors, startangle=90)
    
    ax.set_title(f'Training Window Distribution by Phase\n(Total: {total_windows} windows)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


# ============== MAIN ==============

def main():
    print("=" * 60)
    print("Cartpole Trajectory Phase Analysis")
    print("=" * 60)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # Load dataset using the project's utility functions
    try:
        trajectories = load_trajectories_via_utils(
            dataset_name="cartpole_pybullet",
            max_trajectories=None
        )
    except Exception as e:
        print(f"Error loading via utils: {e}")
        print("Trying direct path loading...")
        
        dataset_path = get_dataset_path()
        print(f"\nDataset path: {dataset_path}")
        
        if not dataset_path.exists():
            print(f"ERROR: Dataset path does not exist: {dataset_path}")
            print("Please check that the dataset is available.")
            return
        
        trajectories = load_trajectories_from_path(dataset_path, max_trajectories=None)
    
    if len(trajectories) == 0:
        print("ERROR: No trajectories loaded!")
        return
    
    # Sample for detailed analysis
    n_sample = min(N_SAMPLE_TRAJECTORIES, len(trajectories))
    sample_indices = np.random.choice(len(trajectories), n_sample, replace=False)
    sample_trajectories = [trajectories[i] for i in sample_indices]
    
    print(f"\nAnalyzing {n_sample} sampled trajectories...")
    
    # Run analysis
    summary = analyze_dataset(sample_trajectories)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total trajectories in dataset: {len(trajectories)}")
    print(f"Trajectories analyzed: {summary['n_trajectories']}")
    print(f"Average trajectory length: {summary['avg_length']:.1f} timesteps")
    print(f"Average time to equilibrium: {summary['avg_equilibrium_start']:.1f} ± {summary['std_equilibrium_start']:.1f} timesteps")
    
    print("\nPhase duration statistics:")
    for phase, (mean, std) in summary['phase_durations'].items():
        print(f"  {phase}: {mean:.1f} ± {std:.1f} timesteps")
    
    # Analyze training distribution
    print("\n" + "-" * 60)
    phase_counts, total_windows = analyze_training_distribution(sample_trajectories)
    
    # Generate visualizations
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    
    # Pick a few example trajectories for detailed plots
    example_indices = [0, len(sample_trajectories)//2, len(sample_trajectories)-1]
    
    for i in example_indices:
        result = summary['results'][i]
        
        # Trajectory with phases
        plot_trajectory_with_phases(
            result, 
            save_path=OUTPUT_DIR / f"trajectory_{i}_phases.png"
        )
        
        # Dynamics over time
        plot_dynamics_over_time(
            result,
            save_path=OUTPUT_DIR / f"trajectory_{i}_dynamics.png"
        )
        
        # Adaptive slices
        plot_adaptive_slices(
            result,
            save_path=OUTPUT_DIR / f"trajectory_{i}_slices.png"
        )
        
        # Full timeline visualization
        plot_full_trajectory_timeline(
            result,
            save_path=OUTPUT_DIR / f"trajectory_{i}_full_timeline.png"
        )
    
    # Summary plot showing all trajectories overlaid
    plot_trajectory_summary_comparison(
        summary['results'],
        save_path=OUTPUT_DIR / "all_trajectories_theta_comparison.png"
    )
    
    # Aggregate plots
    plot_phase_duration_distribution(
        summary,
        save_path=OUTPUT_DIR / "phase_duration_distribution.png"
    )
    
    plot_equilibrium_start_histogram(
        summary,
        save_path=OUTPUT_DIR / "equilibrium_start_distribution.png"
    )
    
    plot_dynamics_histogram(
        summary,
        save_path=OUTPUT_DIR / "dynamics_magnitude_distribution.png"
    )
    
    plot_initial_vs_final_states(
        summary,
        save_path=OUTPUT_DIR / "initial_vs_final_states.png"
    )
    
    plot_training_window_distribution(
        phase_counts, 
        total_windows,
        save_path=OUTPUT_DIR / "training_window_distribution.png"
    )
    
    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Return summary for potential further use
    return summary


if __name__ == "__main__":
    main()
