import os
from os import path
import shutil
from multiprocessing import Process
from typing import List, Optional, Sequence, Tuple

import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AnimationGenerator:
    """Handles the generation of animations from plot images."""

    def __init__(self, dataset_name: str, uncertainty_name: str):
        self.animation_fps = 1
        self.uncertainty_name = uncertainty_name
        self.roa_points, self.roa_labels = self._load_roa_data(dataset_name)
        # Use 'viridis' colormap for the background ROA labels
        self.roa_colormap = 'viridis'

    def _load_roa_data(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads the Region of Attraction (ROA) data from the specified dataset."""
        roa_file_path = path.join('data_trajectories', dataset_name, 'roa_labels.txt')
        if not path.exists(roa_file_path):
            raise FileNotFoundError(f"ROA data file not found at: {roa_file_path}")

        data = np.loadtxt(roa_file_path)

        # Expects format: <index> <state_1> <state_2> ... <state_n> <label>
        # For 2D, we expect 1 (index) + 2 (state dims) + 1 (label) = 4 columns
        if data.shape[1] != 4:
            raise ValueError(f"Expected 2D state data in roa_labels.txt (4 columns), but found {data.shape[1]} columns.")

        points = data[:, 1:-1]
        labels = data[:, -1].astype(int)
        return points, labels

    def _create_plot_frame(self, save_filepath: str, title: str,
                             foreground_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             foreground_type: Optional[str] = None,
                             foreground_alpha: float = 0.7,
                             plot_background: bool = True):
        """Creates and saves a single plot frame."""
        fig, ax = plt.subplots()

        # 1. Plot background ROA data
        if plot_background:
            ax.scatter(
                self.roa_points[:, 0], self.roa_points[:, 1], 
                c=self.roa_labels, 
                cmap=self.roa_colormap, 
                s=10, 
                marker='s',
                edgecolors='none',
            )

        # 2. Plot foreground data if provided
        if foreground_data and foreground_type:
            points, values = foreground_data
            if foreground_type == 'uncertainty':
                scatter_fg = ax.scatter(
                    points[:, 0], points[:, 1],
                    c=values, 
                    cmap='viridis',
                    s=10, 
                    edgecolors='none',
                    alpha=foreground_alpha,
                )
                fig.colorbar(scatter_fg, ax=ax)
            elif foreground_type == 'samples':
                # 'values' are the indices of the sampled points
                sampled_points = points[values]
                ax.scatter(
                    sampled_points[:, 0], sampled_points[:, 1],
                    s=10, 
                    c='r', 
                    marker='x', 
                    alpha=foreground_alpha,
                )

        ax.set_title(title)
        ax.set_xlim(self.roa_points[:, 0].min(), self.roa_points[:, 0].max())
        ax.set_ylim(self.roa_points[:, 1].min(), self.roa_points[:, 1].max())
        ax.grid(True)

        plt.savefig(save_filepath, bbox_inches='tight', dpi=150)
        plt.close(fig)

    @staticmethod
    def _compile_gif_and_cleanup(frame_paths: List[str], output_path: str, temp_dir_to_delete: str, fps: int):
        """Compiles a GIF from frames and cleans up the temporary directory."""
        try:
            with imageio.get_writer(output_path, mode='I', fps=fps) as writer:
                for filename in frame_paths:
                    image = imageio.imread(filename)
                    writer.append_data(image)
        finally:
            # Ensure cleanup happens even if imageio fails
            if path.exists(temp_dir_to_delete):
                shutil.rmtree(temp_dir_to_delete)

    def generate_animations(self, iteration: int, uncertainty: np.ndarray, dataset_start_points: np.ndarray, sampled_indices: Sequence[int], save_path: str):
        """
        Generates two animations for the given iteration in parallel:
        1. Uncertainty visualization.
        2. Newly sampled points visualization.
        """
        os.makedirs(save_path, exist_ok=True)

        # --- Uncertainty Animation ---
        tmp_uncertainty_path = path.join(save_path, f"tmp_uncertainty_{iteration}")
        os.makedirs(tmp_uncertainty_path, exist_ok=True)

        uncertainty_title = f"{self.uncertainty_name} - Iteration {iteration}"
        uncertainty_frames = []

        # Frame 1: Background + Transparent Uncertainty (for stable layout with colorbar)
        frame1_path = path.join(tmp_uncertainty_path, "frame_001.png")
        self._create_plot_frame(frame1_path, uncertainty_title,
                                foreground_data=(dataset_start_points, uncertainty),
                                foreground_type='uncertainty',
                                foreground_alpha=0.0)
        uncertainty_frames.append(frame1_path)

        # Frame 2: Uncertainty
        frame2_path = path.join(tmp_uncertainty_path, "frame_002.png")
        self._create_plot_frame(frame2_path, uncertainty_title,
                                foreground_data=(dataset_start_points, uncertainty),
                                foreground_type='uncertainty',
                                plot_background=False)
        uncertainty_frames.append(frame2_path)

        # --- Sampled Points Animation ---
        tmp_samples_path = path.join(save_path, f"tmp_samples_{iteration}")
        os.makedirs(tmp_samples_path, exist_ok=True)

        samples_title = f"New Samples - Iteration {iteration}"
        samples_frames = []

        # Frame 1: Background + Transparent Samples (for stable layout with legend)
        frame1_samples_path = path.join(tmp_samples_path, "frame_001.png")
        self._create_plot_frame(frame1_samples_path, samples_title,
                                foreground_data=(dataset_start_points, np.array(sampled_indices)),
                                foreground_type='samples',
                                foreground_alpha=0.0)
        samples_frames.append(frame1_samples_path)

        # Frame 2: Background + New Samples
        frame2_samples_path = path.join(tmp_samples_path, "frame_002.png")
        self._create_plot_frame(frame2_samples_path, samples_title,
                                foreground_data=(dataset_start_points, np.array(sampled_indices)),
                                foreground_type='samples')
        samples_frames.append(frame2_samples_path)

        # --- Spawn Parallel Processes to Compile GIFs ---
        uncertainty_gif_path = path.join(save_path, f"uncertainty_it{iteration}.gif")
        samples_gif_path = path.join(save_path, f"samples_it{iteration}.gif")

        uncertainty_process = Process(
            target=self._compile_gif_and_cleanup,
            args=(uncertainty_frames, uncertainty_gif_path, tmp_uncertainty_path, self.animation_fps)
        )

        samples_process = Process(
            target=self._compile_gif_and_cleanup,
            args=(samples_frames, samples_gif_path, tmp_samples_path, self.animation_fps)
        )

        uncertainty_process.start()
        samples_process.start() 
    