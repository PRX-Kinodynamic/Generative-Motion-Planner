from genMoPlan.utils import expand_model_paths
from scripts.viz_model import visualize_generated_trajectories


models = [
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_5000",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_3500",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_2000",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_1000",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_500",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_100",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_50",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_25",
    "experiments/pendulum_lqr_50k/flow_matching/*_manifold_data_lim_10",

]

expanded_models = expand_model_paths(models)


visualize_generated_trajectories(
    dataset="pendulum_lqr_50k",
    num_trajs=1000,
    model_paths=expanded_models,
    model_state_name="best.pt",
    observation_dim=2,
)