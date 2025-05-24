import argparse

from genMoPlan.eval.final_states import evaluate_final_state_variance
from genMoPlan.utils.roa import ROAEstimator

# num_inference_steps = [
#     1, 4, 8, 16
# ]

# horizon_lengths = [
#     1, 4, 8, 16, 32
# ]

num_inference_steps = [1]
horizon_lengths = [32]


def compute_final_state_variances(
    dataset: str,
    model_path: str,
    model_state_name: str,
    n_runs: int,
    num_batches: int,
):
    
    roa_estimator = ROAEstimator(
        dataset=dataset,
        model_path=model_path,
        model_state_name=model_state_name,
        n_runs=n_runs,
        num_batches=num_batches,
    )

    roa_estimator.load_ground_truth()

    for n in num_inference_steps:
        for horizon_length in horizon_lengths:
            final_state_variance = evaluate_final_state_variance(
                roa_estimator,
                horizon_length,
                n,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute final state variance")
    
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )

    parser.add_argument("--model_path", type=str, help="Experiment path")

    parser.add_argument(
        "--model_state_name", type=str, default="best.pt", help="Model state file name"
    )

    parser.add_argument(
        "--n_runs",
        type=int,
        help="Number of runs to generate or load",
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
    
    args = parser.parse_args()
    compute_final_state_variances(
        dataset=args.dataset,
        model_path=args.model_path,
        model_state_name=args.model_state_name,
        n_runs=args.n_runs,
        num_batches=args.num_batches,
    )