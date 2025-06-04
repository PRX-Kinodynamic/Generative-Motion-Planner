import argparse
import os
import numpy as np
from genMoPlan.eval.final_states import evaluate_final_state_std, plot_final_state_std, plot_final_state_std_sigmoid, plot_final_state_eight_root_std
from genMoPlan.utils.roa import ROAEstimator

TYPES_ENUM = {
    "std": 'std',
    "sigmoid": 'sigmoid',
    "eight_root": 'eight_root',
}


types = ['std', 'sigmoid', 'eight_root']

num_inference_steps = [
    1, 2, 4, 8, 16
]

horizon_lengths = [
    1, 4, 8, 16, 32
]

angle_indices = [0]

inference_normalization_params = {
    "mins": [None, -2*np.pi],
    "maxs": [None, 2*np.pi],
}

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
            print(f"\n\nComputing final states for n={n}, horizon_length={horizon_length}\n\n")

            final_state_variance = evaluate_final_state_std(
                roa_estimator,
                horizon_length,
                n,
                angle_indices,
                inference_normalization_params,
            )

            for type in types:
                plot_dir = f"{model_path}/final_state_{type}_{roa_estimator.n_runs}_runs"
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = f"{plot_dir}/final_state_{type}_{n}_{horizon_length}.png"
                if type == TYPES_ENUM["std"]:
                    plot_final_state_std(
                        final_state_variance,
                        roa_estimator.start_states,
                        plot_path,
                        f"Final State {type} (n={n}, horizon_length={horizon_length})"
                    )
                elif type == TYPES_ENUM["sigmoid"]:
                    plot_final_state_std_sigmoid(
                        final_state_variance,
                        roa_estimator.start_states,
                        plot_path,
                        f"Final State {type} (n={n}, horizon_length={horizon_length})"
                    )
                elif type == TYPES_ENUM["eight_root"]:
                    plot_final_state_eight_root_std(
                        final_state_variance,
                        roa_estimator.start_states,
                        plot_path,
                        f"Final State {type} (n={n}, horizon_length={horizon_length})"
                    )

                print(f"\n\nSaved plot to {plot_path}\n\n")

            roa_estimator.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Compute final state std")
    
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