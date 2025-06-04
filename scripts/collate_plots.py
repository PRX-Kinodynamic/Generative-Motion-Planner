#!/usr/bin/env python3
import glob
import os
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict
import re

def get_exact_paths(base_path,model_paths):
    expanded_paths = []
    
    for path in model_paths:
        path = os.path.join(base_path, path)
        if "*" in path:
            all_dirs = glob.glob(path)
            valid_dirs = [d for d in all_dirs if os.path.isdir(d)]
            expanded_paths.extend(valid_dirs)
        else:
            if os.path.exists(path):
                expanded_paths.append(path)
            else:
                print(f"Warning: Path {path} does not exist or does not contain best.pt")
    
    return expanded_paths

# base_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/diffusion"
# prefix = "diffusion/"
# experiments = [
#     "25_04_05-20_10_15_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_10",
#     "25_04_05-20_10_37_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_25",
#     "25_04_05-20_11_02_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_50",
#     "25_03_21-16_04_52_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100",
#     "25_03_21-16_05_21_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_500",
#     "25_03_21-16_05_54_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_1000",
#     "25_03_21-16_06_04_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_2000",
#     "25_03_21-16_06_07_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_3500",
#     "25_03_21-16_06_12_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_5000"
# ]

# base_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/flow_matching"
# prefix = "flow_matching/"
# experiments = [
# "25_04_05-20_16_38_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_10",
# "25_04_05-20_17_03_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_25",
# "25_04_05-20_17_13_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_50",
# "25_04_05-20_05_55_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100",
# "25_04_05-20_06_33_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_500",
# "25_04_05-20_07_37_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_1000",
# "25_04_05-20_08_15_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_2000",
# "25_04_05-20_15_01_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_3500",
# "25_04_05-20_15_19_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_5000",
# ]

base_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/flow_matching"
prefix = "flow_matching_manifold/"
experiments = [
"*manifold*data_lim_10",
"*manifold*data_lim_25",
"*manifold*data_lim_50",
"*manifold*data_lim_100",
"*manifold*data_lim_500",
"*manifold*data_lim_1000",
"*manifold*data_lim_2000",
"*manifold*data_lim_3500",
"*manifold*data_lim_5000",
]



experiments = get_exact_paths(base_path,experiments)

print(f"Found {len(experiments)} valid experiments:")
for experiment in experiments:
    print(f"  - {experiment}")

input("Press Enter to continue...")

# List of metrics to plot
metrics_to_plot = ["fp_rate", "fn_rate", "accuracy", "precision", "recall", "f1_score"]

image_details = [
    ("classification_results.png", "Classification Results"),
    ("roas.png", "ROA")
]

results_details = [
    ("classification_results.json", "Classification Metrics"),
]


all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str))))

n_datapoints_vs_experiment = {}

all_dist_thresholds = set()
all_prob_thresholds = set()

for experiment in experiments:
    n_datapoints = experiment.split("_data_lim_")[1]
    n_datapoints = int(n_datapoints)

    n_datapoints_vs_experiment[n_datapoints] = experiment

    exp_path = os.path.join(base_path, experiment)
    results_dir_path = os.path.join(exp_path, "results")

    for results_dir in os.listdir(results_dir_path):
        if not os.path.isdir(os.path.join(results_dir_path, results_dir)):
            continue

        results_path = os.path.join(results_dir_path, results_dir)

        if os.path.exists(os.path.join(results_path, "inference_params.json")):
            params_path = os.path.join(results_path, "inference_params.json")
        else:
            params_path = os.path.join(results_path, "roa_estimation_params.json")

        with open(params_path, "r") as f:
            params = json.load(f)

        dist_threshold = params["attractor_dist_threshold"]
        prob_threshold = params["attractor_prob_threshold"]

        all_dist_thresholds.add(dist_threshold)
        all_prob_thresholds.add(prob_threshold)

        all_results[n_datapoints][dist_threshold][prob_threshold] = results_path

all_dist_thresholds = sorted(list(all_dist_thresholds))
all_prob_thresholds = sorted(list(all_prob_thresholds))

# Get sorted data points and probability thresholds
    # Get sorted data points and probability thresholds
data_points = sorted(all_results.keys())
attractor_dist_thresholds = [0.075, 0.05, 0.025, 0.01]
attractor_prob_thresholds = [0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 1.0]

# Function to create a grid plot for a specific dist_threshold
def create_grid_plot_fixed_dist(image_name, image_title, fixed_dist_threshold):
    # Create a figure with a grid layout
    n_rows = len(data_points)
    n_cols = len(attractor_prob_thresholds)
    plt.figure(figsize=(18, 18))
    
    # Create the subplot grid
    for i, n_datapoint in enumerate(data_points):
        for j, prob_threshold in enumerate(attractor_prob_thresholds):
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            
            # Get the path for the specific configuration
            if fixed_dist_threshold in all_results[n_datapoint] and prob_threshold in all_results[n_datapoint][fixed_dist_threshold]:
                results_path = all_results[n_datapoint][fixed_dist_threshold][prob_threshold]
                img_path = os.path.join(results_path, image_name)
                
                # Check if the image exists
                if os.path.exists(img_path):
                    # Load and display the image
                    img = np.array(Image.open(img_path))
                    plt.imshow(img)
                    plt.title(f"n_data={n_datapoint}, prob={prob_threshold}")
                else:
                    plt.text(0.5, 0.5, f"Image not found\n{img_path}", 
                            horizontalalignment='center', verticalalignment='center')
            else:
                plt.text(0.5, 0.5, f"Configuration not found\nn_data={n_datapoint}, prob={prob_threshold}", 
                        horizontalalignment='center', verticalalignment='center')
            
            # Remove axis ticks
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"{image_title} Comparison (dist_threshold={fixed_dist_threshold})", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"scripts/plots/{prefix}{image_title}_comparison_dist_{fixed_dist_threshold}.png", dpi=300)
    print(f"Saved plot to {image_title}_comparison_dist_{fixed_dist_threshold}.png with {n_rows} data points x {n_cols} prob thresholds")

# Function to create a grid plot for a specific prob_threshold
def create_grid_plot_fixed_prob(image_name, image_title, fixed_prob_threshold):
    # Create a figure with a grid layout
    n_rows = len(data_points)
    n_cols = len(attractor_dist_thresholds)
    plt.figure(figsize=(18, 18))
    
    # Create the subplot grid
    for i, n_datapoint in enumerate(data_points):
        for j, dist_threshold in enumerate(attractor_dist_thresholds):
            plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
            
            # Get the path for the specific configuration
            if dist_threshold in all_results[n_datapoint] and fixed_prob_threshold in all_results[n_datapoint][dist_threshold]:
                results_path = all_results[n_datapoint][dist_threshold][fixed_prob_threshold]
                img_path = os.path.join(results_path, image_name)
                
                # Check if the image exists
                if os.path.exists(img_path):
                    # Load and display the image
                    img = np.array(Image.open(img_path))
                    plt.imshow(img)
                    plt.title(f"n_data={n_datapoint}, dist={dist_threshold}")
                else:
                    plt.text(0.5, 0.5, f"Image not found\n{img_path}", 
                            horizontalalignment='center', verticalalignment='center')
            else:
                plt.text(0.5, 0.5, f"Configuration not found\nn_data={n_datapoint}, dist={dist_threshold}", 
                        horizontalalignment='center', verticalalignment='center')
            
            # Remove axis ticks
            plt.xticks([])
            plt.yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"{image_title} Comparison (prob_threshold={fixed_prob_threshold})", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"scripts/plots/{prefix}{image_title}_comparison_prob_{fixed_prob_threshold}.png", dpi=300)
    print(f"Saved plot to {image_title}_comparison_prob_{fixed_prob_threshold}.png with {n_rows} data points x {n_cols} dist thresholds")

for image_name, image_title in image_details:
    # Create plots for each fixed dist_threshold
    for dist_threshold in attractor_dist_thresholds:
        create_grid_plot_fixed_dist(image_name,image_title, dist_threshold)
        
    # Create plots for each fixed prob_threshold
    for prob_threshold in attractor_prob_thresholds:
        create_grid_plot_fixed_prob(image_name, image_title, prob_threshold)


# Function to create a combined grid plot for all classification metrics with fixed dist threshold
def create_combined_metrics_grid_fixed_dist(fixed_dist_threshold):
    # Create a figure with a grid layout
    n_rows = len(data_points)
    n_cols = len(attractor_prob_thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 20))
    
    # Create the subplot grid
    for i, n_datapoint in enumerate(data_points):
        for j, prob_threshold in enumerate(attractor_prob_thresholds):
            ax = axes[i, j]
            
            # Get the path for the specific configuration
            if fixed_dist_threshold in all_results[n_datapoint] and prob_threshold in all_results[n_datapoint][fixed_dist_threshold]:
                results_path = all_results[n_datapoint][fixed_dist_threshold][prob_threshold]
                json_path = os.path.join(results_path, "classification_results.json")
                
                # Check if the json file exists
                if os.path.exists(json_path):
                    # Load the metrics from JSON
                    with open(json_path, "r") as f:
                        metrics = json.load(f)
                    
                    # Create a table with all metrics
                    metric_values = {}
                    for metric_name in metrics_to_plot:
                        metric_value = metrics.get(metric_name, "N/A")
                        if metric_value == "NaN" or metric_value == float('nan'):
                            metric_value = 0
                        metric_values[metric_name] = metric_value
                    
                    # Remove the axis elements
                    ax.axis('off')
                    
                    # Create a table with the metrics
                    cell_text = [[f"{name.replace('_', ' ').title()}:", f"{value:.4f}" if isinstance(value, (int, float)) else value] 
                                  for name, value in metric_values.items()]
                    table = ax.table(cellText=cell_text, loc='center', cellLoc='left', colWidths=[0.6, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    
                    # Add a title
                    ax.set_title(f"n_data={n_datapoint}, prob={prob_threshold}")
                else:
                    ax.text(0.5, 0.5, "Metrics not found", 
                           horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(0.5, 0.5, f"Configuration not found\nn_data={n_datapoint}, prob={prob_threshold}", 
                       horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.suptitle(f"Classification Metrics Comparison (dist_threshold={fixed_dist_threshold})", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"scripts/plots/{prefix}combined_metrics_dist_{fixed_dist_threshold}.png", dpi=300)
    print(f"Saved combined metrics plot to combined_metrics_dist_{fixed_dist_threshold}.png")

# Function to create a combined grid plot for all classification metrics with fixed prob threshold
def create_combined_metrics_grid_fixed_prob(fixed_prob_threshold):
    # Create a figure with a grid layout
    n_rows = len(data_points)
    n_cols = len(attractor_dist_thresholds)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 20))
    
    # Create the subplot grid
    for i, n_datapoint in enumerate(data_points):
        for j, dist_threshold in enumerate(attractor_dist_thresholds):
            ax = axes[i, j]
            
            # Get the path for the specific configuration
            if dist_threshold in all_results[n_datapoint] and fixed_prob_threshold in all_results[n_datapoint][dist_threshold]:
                results_path = all_results[n_datapoint][dist_threshold][fixed_prob_threshold]
                json_path = os.path.join(results_path, "classification_results.json")
                
                # Check if the json file exists
                if os.path.exists(json_path):
                    # Load the metrics from JSON
                    with open(json_path, "r") as f:
                        metrics = json.load(f)
                    
                    # Create a table with all metrics
                    metric_values = {}
                    for metric_name in metrics_to_plot:
                        metric_value = metrics.get(metric_name, "N/A")
                        if metric_value == "NaN" or metric_value == float('nan'):
                            metric_value = 0
                        metric_values[metric_name] = metric_value
                    
                    # Remove the axis elements
                    ax.axis('off')
                    
                    # Create a table with the metrics
                    cell_text = [[f"{name.replace('_', ' ').title()}:", f"{value:.4f}" if isinstance(value, (int, float)) else value] 
                                  for name, value in metric_values.items()]
                    table = ax.table(cellText=cell_text, loc='center', cellLoc='left', colWidths=[0.6, 0.4])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.5)
                    
                    # Add a title
                    ax.set_title(f"n_data={n_datapoint}, dist={dist_threshold}")
                else:
                    ax.text(0.5, 0.5, "Metrics not found", 
                           horizontalalignment='center', verticalalignment='center')
            else:
                ax.text(0.5, 0.5, f"Configuration not found\nn_data={n_datapoint}, dist={dist_threshold}", 
                       horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.suptitle(f"Classification Metrics Comparison (prob_threshold={fixed_prob_threshold})", fontsize=16)
    plt.subplots_adjust(top=0.95)
    plt.savefig(f"scripts/plots/{prefix}combined_metrics_prob_{fixed_prob_threshold}.png", dpi=300)
    print(f"Saved combined metrics plot to combined_metrics_prob_{fixed_prob_threshold}.png")

os.makedirs(f"scripts/plots/{prefix}", exist_ok=True)
# Create combined plots for each threshold
for dist_threshold in attractor_dist_thresholds:
    create_combined_metrics_grid_fixed_dist(dist_threshold)
    
for prob_threshold in attractor_prob_thresholds:
    create_combined_metrics_grid_fixed_prob(prob_threshold)


