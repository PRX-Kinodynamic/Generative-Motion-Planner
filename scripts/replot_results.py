from genMoPlan.utils.roa import ROAEstimator
import os
from tqdm import tqdm

if __name__ == "__main__":

    # Base path for all experiments
    base_path = "/common/home/st1122/Projects/genMoPlan/experiments/pendulum_lqr_50k/diffusion"

    # List of experiment folders with different data limits
    experiments = [
        "25_03_21-16_04_52_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_100",
        "25_03_21-16_05_21_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_500",
        "25_03_21-16_05_54_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_1000",
        "25_03_21-16_06_04_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_2000",
        "25_03_21-16_06_07_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_3500",
        "25_03_21-16_06_12_HILEN-1_HOLEN-31_HIPAD-F_HOPAD-T_STRD-1_data_lim_5000"
    ]

    total = 0

    for experiment in experiments:
        exp_path = os.path.join(base_path, experiment)
        results_dir_path = os.path.join(exp_path, "results")

        for results_dir in os.listdir(results_dir_path):
            if not os.path.isdir(os.path.join(results_dir_path, results_dir)):
                continue

            total += 1

    pbar = tqdm(total=total)

    for experiment in experiments:
        exp_path = os.path.join(base_path, experiment)
        results_dir_path = os.path.join(exp_path, "results")

        for results_dir in os.listdir(results_dir_path):
            if not os.path.isdir(os.path.join(results_dir_path, results_dir)):
                continue

            roa_estimator = ROAEstimator(
                dataset="pendulum_lqr_50k",
                model_path=exp_path,
                n_runs=100,
                batch_size=1000,
                verbose=False
            )
            roa_estimator.load_predicted_labels(os.path.join(results_dir_path, results_dir))
            roa_estimator.load_ground_truth()
            roa_estimator.plot_roas()
            # roa_estimator.compute_classification_results(save=False)
            # roa_estimator.plot_classification_results()

            pbar.update(1)

    pbar.close()