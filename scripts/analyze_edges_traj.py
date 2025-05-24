from genMoPlan.utils import ROAEstimator
import numpy as np


dataset = "pendulum_lqr_50k"
model_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/diffusion/25_03_21-16_04_52_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100"
model_state_name = "best.pt"

roa_estimator = ROAEstimator(
    dataset=dataset,
    model_path=model_path,
    model_state_name=model_state_name,
    n_runs=1000,
)

roa_estimator.load_ground_truth()

start_points = roa_estimator.start_points


# Get absolute values of start points
abs_start_points = np.abs(start_points)

# Find indices where absolute values exceed the thresholds (2.5, 3)
# Using AND condition instead of OR
edge_indices = np.where(
    (abs_start_points[:, 0] > 2.5) & 
    (abs_start_points[:, 1] > 3)
)[0]

roa_estimator.start_points = start_points[edge_indices]
roa_estimator.expected_labels = roa_estimator.expected_labels[edge_indices]

roa_estimator.timestamp = "limited_trajectories"

roa_estimator.load_final_states("limited_trajectories")

no_img = False

roa_estimator.set_attractor_dist_threshold(0.025)
roa_estimator.set_attractor_prob_threshold(0.98)

roa_estimator.compute_attractor_labels()
roa_estimator.compute_attractor_probabilities(plot=not no_img)

roa_estimator.predict_attractor_labels(save=True, plot=not no_img)

if not no_img:
    roa_estimator.plot_roas(plot_separatrix=True)

roa_estimator.compute_classification_results(save=True)

roa_estimator.plot_classification_results()
