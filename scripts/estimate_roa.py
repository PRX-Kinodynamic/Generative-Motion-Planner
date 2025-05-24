import argparse

from genMoPlan.utils import ROAEstimator

def estimate_roa(
    dataset, 
    model_state_name, 
    model_path,
    n_runs=None, 
    batch_size=None, 
    num_batches=None,
    timestamp=None, 
    no_img=False, 
    continue_gen=False, 
    analyze=False, 
    verbose=True, 
    attractor_dist_threshold=None,
    attractor_prob_threshold=None,
):
    if batch_size is not None:
        batch_size = int(batch_size)

    roa_estimator = ROAEstimator(
        dataset=dataset,
        model_state_name=model_state_name, 
        model_path=model_path, 
        n_runs=n_runs, 
        batch_size=batch_size, 
        num_batches=num_batches, 
        verbose=verbose
    )

    roa_estimator.load_ground_truth()

    if attractor_dist_threshold is not None:
        roa_estimator.set_attractor_dist_threshold(attractor_dist_threshold)
    if attractor_prob_threshold is not None:
        roa_estimator.set_attractor_prob_threshold(attractor_prob_threshold)

    if analyze or continue_gen:
        roa_estimator.load_final_states(timestamp=timestamp)
    
    if not analyze:
        roa_estimator.generate_trajectories(
            compute_labels=True,
            save=True,
        )

    roa_estimator.compute_attractor_labels() # In case, the attractor dist threshold has changed and the labels from old loaded runs and new runs are inconsistent

    roa_estimator.compute_attractor_probabilities(plot=not no_img)

    roa_estimator.predict_attractor_labels(save=True, plot=not no_img)

    if not no_img:
        roa_estimator.plot_roas(plot_separatrix=True)

    roa_estimator.compute_classification_results(save=True)

    roa_estimator.plot_classification_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")
    
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument("--model_path", type=str, help="Experiment path")

    parser.add_argument(
        "--model_paths", 
        type=str, 
        nargs="+", 
        help="Multiple experiment paths. If provided, will override --model_path"
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
        "--num_batches",
        type=int,
        help="Number of batches to generate or load",
    )

    parser.add_argument(
        "--no_parallel",
        action="store_true",
        help="Do not use parallel processing",
    )

    parser.add_argument(
        "--attractor_dist_threshold",
        type=float,
        help="Attractor distance threshold",
    )

    parser.add_argument(
        "--attractor_prob_threshold",
        type=float,
        help="Attractor probability threshold",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Do not print anything",
    )
    
    args = parser.parse_args()

    if args.model_paths:
        for model_path in args.model_paths:
            print(f"\n\n[ scripts/estimate_roa ] Estimating ROA for {model_path}\n\n")
            estimate_roa(
                dataset=args.dataset, 
                model_state_name=args.model_state_name, 
                model_path=model_path, 
                n_runs=args.n_runs, 
                batch_size=args.batch_size, 
                num_batches=args.num_batches, 
                timestamp=args.timestamp, 
                no_img=args.no_img, 
                continue_gen=args.continue_gen, 
                analyze=args.analyze, 
                attractor_dist_threshold=args.attractor_dist_threshold, 
                attractor_prob_threshold=args.attractor_prob_threshold,
                verbose=not args.silent
            )
    else:
        estimate_roa(
            dataset=args.dataset, 
            model_state_name=args.model_state_name, 
            model_path=args.model_path, 
            n_runs=args.n_runs, 
            batch_size=args.batch_size, 
            num_batches=args.num_batches, 
            timestamp=args.timestamp, 
            no_img=args.no_img, 
            continue_gen=args.continue_gen, 
            analyze=args.analyze, 
            attractor_dist_threshold=args.attractor_dist_threshold, 
            attractor_prob_threshold=args.attractor_prob_threshold,
            verbose=not args.silent
        )