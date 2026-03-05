import argparse
import importlib

from genMoPlan.eval.classifier import Classifier
from genMoPlan.utils import expand_model_paths
from genMoPlan.utils.model import load_model_args


def evaluate(
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
    outcome_prob_threshold=None,
    use_validation_data=False,
):

    dataset = None

    path_splits = model_path.split("/")

    for i, splits in enumerate(path_splits):
        if splits == "experiments":
            dataset = path_splits[i + 1]
            break

    if dataset is None:
        raise ValueError(f"Dataset not found in model path: {model_path}")

    if "#" in model_path:
        model_path = model_path.replace("#", "*")
        model_path = expand_model_paths(model_path)[0]

    if batch_size is not None:
        batch_size = int(batch_size)

    # Build the latest system definition from the dataset config.
    config_module = importlib.import_module(f"config.{dataset}")
    get_system = getattr(config_module, "get_system", None)
    if get_system is None:
        raise ValueError(
            f"Config module for dataset '{dataset}' does not define get_system()."
        )

    # Load model args to determine system configuration (e.g., use_manifold)
    model_args = load_model_args(model_path)
    system = get_system(
        config=getattr(config_module, "base"),
        use_manifold=getattr(model_args, "use_manifold", False),
        stride=getattr(model_args, "stride", 1),
        history_length=getattr(model_args, "history_length", 1),
        horizon_length=getattr(model_args, "horizon_length", 31),
    )


    classifier = Classifier(
        dataset=dataset,
        model_state_name=model_state_name,
        model_path=model_path,
        n_runs=n_runs,
        batch_size=batch_size,
        num_batches=num_batches,
        verbose=verbose,
        system=system,
    )

    if use_validation_data:
        classifier.load_validation_ground_truth()
    else:
        classifier.load_ground_truth()

    if outcome_prob_threshold is not None:
        classifier.set_outcome_prob_threshold(outcome_prob_threshold)

    if analyze or continue_gen:
        classifier.load_final_states(timestamp=timestamp)

    if not analyze:
        classifier.generate_trajectories(
            save=True,
        )

    # Outcome-based analysis driven purely by the system-defined outcomes.
    classifier.compute_outcome_labels()
    classifier.compute_outcome_probabilities()
    classifier.predict_outcomes(save=True)
    classifier.derive_labels_from_outcomes()

    # Compute final state errors if ground truth final states available
    if classifier.ground_truth_final_states is not None:
        classifier.compute_final_state_errors(save=True)

    classifier.compute_classification_results(save=True)

    if not no_img:
        try:
            classifier.plot_roas(plot_separatrix=True)
            classifier.plot_classification_results()
        except ValueError as e:
            if "more than 2D" in str(e):
                if verbose:
                    print(f"[ scripts/evaluate ] Skipping plots (state dim > 2)")
            else:
                raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model trajectories")

    parser.add_argument("--model_path", type=str, help="Experiment path")

    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        help="Multiple experiment paths. If provided, will override --model_path",
    )

    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )

    parser.add_argument(
        "--no_img",
        action="store_true",
        help="Do not generate any ROA or classification plots",
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
        help="Timestamp for the experiment to load precomputed final states. If not provided, the latest timestamp will be used.",
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
        "--outcome_prob_threshold",
        type=float,
        help="Outcome probability threshold for assigning a label/outcome",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Do not print anything",
    )

    parser.add_argument(
        "--use_validation_data",
        action="store_true",
        help="Use validation dataset as ground truth instead of test_set.txt. "
             "Reads val_dataset_size and shuffled_indices_fname from saved model args. "
             "Outputs are saved to results_val/ and final_states_val/ directories.",
    )

    args = parser.parse_args()

    if args.model_paths:
        for model_path in args.model_paths:
            print(f"\n\n[ scripts/evaluate ] Evaluating {model_path}\n\n")
            evaluate(
                model_state_name=args.model_state_name,
                model_path=model_path,
                n_runs=args.n_runs,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                timestamp=args.timestamp,
                no_img=args.no_img,
                continue_gen=args.continue_gen,
                analyze=args.analyze,
                outcome_prob_threshold=args.outcome_prob_threshold,
                verbose=not args.silent,
                use_validation_data=args.use_validation_data,
            )
    else:
        evaluate(
            model_state_name=args.model_state_name,
            model_path=args.model_path,
            n_runs=args.n_runs,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            timestamp=args.timestamp,
            no_img=args.no_img,
            continue_gen=args.continue_gen,
            analyze=args.analyze,
            outcome_prob_threshold=args.outcome_prob_threshold,
            verbose=not args.silent,
            use_validation_data=args.use_validation_data,
        )
