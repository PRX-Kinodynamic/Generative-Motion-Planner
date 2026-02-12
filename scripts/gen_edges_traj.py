import importlib

from genMoPlan.eval.classifier import Classifier


dataset = "pendulum_lqr_50k"
model_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/diffusion/25_03_21-16_04_52_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100"
model_state_name = "best.pt"

# Build the latest system definition from the dataset config.
config_module = importlib.import_module(f"config.{dataset}")
get_system = getattr(config_module, "get_system", None)
if get_system is None:
    raise ValueError(
        f"Config module for dataset '{dataset}' does not define get_system()."
    )
system = get_system()

roa_estimator = Classifier(
    dataset=dataset,
    model_path=model_path,
    model_state_name=model_state_name,
    n_runs=1000,
    system=system,
)

roa_estimator.load_ground_truth()

start_states = roa_estimator.start_states

# Compute indices of start points with absolute values above (2.5, 3)
import numpy as np

# Get absolute values of start points
abs_start_states = np.abs(start_states)

# Find indices where absolute values exceed the thresholds (2.5, 3)
# Using AND condition instead of OR
edge_indices = np.where((abs_start_states[:, 0] > 2.5) & (abs_start_states[:, 1] > 3))[
    0
]

roa_estimator.start_states = start_states[edge_indices]
roa_estimator.expected_labels = roa_estimator.expected_labels[edge_indices]

roa_estimator.timestamp = "limited_trajectories"

roa_estimator.generate_trajectories(
    return_trajectories=True,
    save=True,
)
