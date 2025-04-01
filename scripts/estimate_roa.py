import argparse

from mg_diffuse.utils import ROAEstimator

def estimate_roa(dataset, model_state_name, exp_path, n_runs, batch_size, timestamp, no_img, continue_gen, analyze):
    roa_estimator = ROAEstimator(dataset, model_state_name, exp_path, n_runs, batch_size)
    roa_estimator.init_ground_truth()

    if analyze or continue_gen:
        roa_estimator.load_final_states(timestamp=timestamp)
    
    if not analyze:
        roa_estimator.generate_trajectories(
            compute_labels=True,
            verbose=True, 
            save=True,
        )

    roa_estimator.compute_attractor_probabilities(plot=not no_img)

    roa_estimator.predict_attractor_labels(plot=not no_img)

    if not no_img:
        roa_estimator.plot_roas()

    roa_estimator.compute_prediction_metrics(save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")
    
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument("--exp_path", type=str, help="Experiment path")

    parser.add_argument(
        "--exp_paths", 
        type=str, 
        nargs="+", 
        help="Multiple experiment paths. If provided, will override --exp_path"
    )

    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )

    parser.add_argument(
        "--no_img",
        action="store_true",
        help="Do not generate an image with the attractor labels. Only possible for 2D datasets",
    )

    parser.add_argument(
        "--continue_gen",
        action="store_true",
        help="Continue generating trajectories from the last set",
    )

    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Only analyze data",
    )

    parser.add_argument(
        "--timestamp",
        type=str,
        help="Timestamp for the experiment to load the attractor labels. If not provided, the latest timestamp will be used.",
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        help="Number of runs to generate or load",
    )

    parser.add_argument(
        "--batch_size",
        type=float,
        help="Batch size for generating trajectories",
    )
    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Do not use parallel processing",
    )

    args = parser.parse_args()

    if args.exp_paths:
        for exp_path in args.exp_paths:
            print(f"\n\n[ scripts/estimate_roa ] Estimating ROA for {exp_path}\n\n")
            estimate_roa(args.dataset, args.model_state_name, exp_path, args.n_runs, args.batch_size, args.timestamp, args.no_img, args.continue_gen, args.analyze)
    else:
        estimate_roa(args.dataset, args.model_state_name, args.exp_path, args.n_runs, args.batch_size, args.timestamp, args.no_img, args.continue_gen, args.analyze)