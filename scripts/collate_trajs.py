from scripts.paths import expand_model_paths
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob

base_path = "/common/home/st1122/Projects/genMoPlan/experiments/"
prefix = "flow_matching_manifold"
traj_file_name = "trajectories_1000_1.png"

models = [
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_5000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_3500",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_2000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_1000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_500",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_100",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_50",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_25",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_10",

]

models = [os.path.join(base_path, model) for model in models]

expanded_models = expand_model_paths(models)

# Extract data limits from model paths for sorting
def extract_data_limit(model_path):
    # Find the data_lim part of the path and extract the number
    if "data_lim_" in model_path:
        try:
            limit = int(model_path.split("data_lim_")[1])
            return limit
        except (ValueError, IndexError):
            return 0
    return 0

# Sort models by data limit (ascending)
sorted_models = sorted(expanded_models, key=extract_data_limit)

# Create output directory if it doesn't exist
os.makedirs(f"scripts/plots/{prefix}", exist_ok=True)

# Create a grid of trajectory plots
n_models = len(sorted_models)
n_cols = 3  # Number of columns in the grid
n_rows = (n_models + n_cols - 1) // n_cols  # Calculate number of rows needed

plt.figure(figsize=(15, 5 * n_rows))

for i, model_path in enumerate(sorted_models):
    plt.subplot(n_rows, n_cols, i + 1)
    
    # Find subdirectories in viz_trajs and look for the trajectory file
    viz_trajs_path = os.path.join(model_path, "viz_trajs")
    traj_image_path = None
    
    if os.path.exists(viz_trajs_path):
        # Look for subdirectories in viz_trajs
        subdirs = [d for d in glob.glob(os.path.join(viz_trajs_path, "*")) if os.path.isdir(d)]
        
        # Look for the trajectory file in each subdir
        for subdir in subdirs:
            potential_path = os.path.join(subdir, traj_file_name)
            if os.path.exists(potential_path):
                traj_image_path = potential_path
                break
    
    # Extract data limit for title
    data_limit = extract_data_limit(model_path)
    
    if traj_image_path and os.path.exists(traj_image_path):
        # Load and display image
        img = np.array(Image.open(traj_image_path))
        plt.imshow(img)
        plt.title(f"Data limit: {data_limit}")
    else:
        plt.text(0.5, 0.5, f"Image not found\nin any subdirectory of\n{viz_trajs_path}", 
                horizontalalignment='center', verticalalignment='center')
    
    # Remove axis ticks
    plt.xticks([])
    plt.yticks([])

plt.tight_layout()
plt.suptitle("Trajectory Visualizations Comparison", fontsize=16)
plt.subplots_adjust(top=0.95)
plt.savefig(f"scripts/plots/{prefix}/collated_trajectories.png", dpi=300)
print(f"Saved collated trajectories to scripts/plots/{prefix}/collated_trajectories.png")

# Print the model paths that were processed
print("\nProcessed models:")
for model in sorted_models:
    print(f"  - {model}")
