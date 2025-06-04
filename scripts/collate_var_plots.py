from scripts.paths import expand_model_paths
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
import re

# type = "std"
# type = "variance"
type = "sigmoid"
# type = "eight_root"

base_path = "/common/home/st1122/Projects/genMoPlan/experiments/"

variance_dir_name = f"final_state_{type}_10_runs"  # Directory containing variance plots
variance_file_pattern = f"final_state_{type}_*_*"  # Pattern for variance plot files

prefix = f"flow_matching_{variance_dir_name}"
# single_model = "pendulum_lqr_50k/flow_matching/25_04_05-20_05_55_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100"
single_model = "pendulum_lqr_50k/flow_matching/25_04_05-20_15_19_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_5000"

# Convert to full path and expand wildcards
model_paths = expand_model_paths([os.path.join(base_path, single_model)])
if not model_paths:
    print(f"No models found matching pattern: {single_model}")
    exit(1)

# Just use the first model that matches
model_path = model_paths[0]
print(f"Using model: {model_path}")

# Create output directory if it doesn't exist
os.makedirs(f"scripts/plots/{prefix}", exist_ok=True)

# Find all variance plots for the model
variance_dir = os.path.join(model_path, variance_dir_name)
if not os.path.exists(variance_dir):
    print(f"Variance directory not found: {variance_dir}")
    exit(1)

# Find all variance plot files in the directory
variance_plots = glob.glob(os.path.join(variance_dir, f"{variance_file_pattern}.png"))
if not variance_plots:
    print(f"No variance plots found in directory: {variance_dir}")
    exit(1)

# Extract steps and horizon info from filenames and organize plots
plot_grid = {}
steps_values = set()
horizon_values = set()

for plot_path in variance_plots:
    filename = os.path.basename(plot_path)
    # Expected format: final_state_variance_num_inference_steps_horizon_length.png
    match = re.search(f'final_state_{type}_(\d+)_(\d+)\.png', filename)
    if match:
        steps = int(match.group(1))
        horizon = int(match.group(2))
        
        steps_values.add(steps)
        horizon_values.add(horizon)
        
        plot_grid[(steps, horizon)] = plot_path
    else:
        print(f"Couldn't parse parameters from filename: {filename}")

# Sort the values for consistent grid layout
steps_values = sorted(steps_values)
horizon_values = sorted(horizon_values)

if not steps_values or not horizon_values:
    print("Could not extract valid steps and horizon values from filenames")
    exit(1)

# Create a grid of plots
n_rows = len(steps_values)
n_cols = len(horizon_values)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))

# Make axes a 2D array even if there's only one row or column
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes.reshape(1, -1)
elif n_cols == 1:
    axes = axes.reshape(-1, 1)

# Fill the grid with plots
for i, steps in enumerate(steps_values):
    for j, horizon in enumerate(horizon_values):
        ax = axes[i, j]
        
        if (steps, horizon) in plot_grid:
            plot_path = plot_grid[(steps, horizon)]
            img = np.array(Image.open(plot_path))
            ax.imshow(img)
            ax.set_title(f"Steps: {steps}, Horizon: {horizon}")
        else:
            ax.text(0.5, 0.5, f"No plot for\nSteps: {steps}\nHorizon: {horizon}", 
                   horizontalalignment='center', verticalalignment='center')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

# Add row and column labels
for i, steps in enumerate(steps_values):
    axes[i, 0].set_ylabel(f"Steps: {steps}", fontsize=12)

for j, horizon in enumerate(horizon_values):
    axes[0, j].set_title(f"Horizon: {horizon}", fontsize=12)

plt.tight_layout()

# Extract model info for the title
model_name = os.path.basename(model_path)
plt.suptitle(f"Final State {type} Plots - {model_name}", fontsize=16)
plt.subplots_adjust(top=0.95)

# Save the collated plot
output_filename = f"{model_name}_collated_{type}_grid.png"
plt.savefig(f"scripts/std_plots/{prefix}/{output_filename}", dpi=300)
print(f"Saved collated variance grid to scripts/plots/{prefix}/{output_filename}")
