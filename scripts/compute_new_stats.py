import os
import glob
import shutil

from tqdm import tqdm
import gc

from genMoPlan.eval.roa import ROAEstimator
from scripts.estimate_roa import estimate_roa
from genMoPlan.utils import expand_model_paths

base_path = "/common/home/st1122/Projects/genMoPlan/experiments/"

models = [
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_5000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_3500",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_2000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_1000",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_500",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_100",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_50",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_25",
    "pendulum_lqr_50k/flow_matching/*_manifold_data_lim_10",

]

models = [os.path.join(base_path, model) for model in models]

expanded_models = expand_model_paths(models)

delete_old_results = input("Should delete old results? (y/n)")

if delete_old_results == "y":
    for model_path in expanded_models:
        results_path = os.path.join(model_path, "results")
        if os.path.exists(results_path):
            shutil.rmtree(results_path)

attractor_dist_thresholds = [0.075, 0.05, 0.025, 0.01]
attractor_prob_thresholds = [0.6, 0.75, 0.85, 0.9, 0.95, 0.98, 1.0]

for model_path in expanded_models:
    print(f'Estimating ROA for {model_path}')

    roa_estimator = ROAEstimator(
        dataset="pendulum_lqr_50k",
        model_state_name='best.pt',
        model_path=model_path,
        verbose=False,
    )

    roa_estimator.load_ground_truth()
    roa_estimator.load_final_states()

    total_runs = len(attractor_dist_thresholds) * len(attractor_prob_thresholds)

    with tqdm(total=total_runs) as pbar:
        for attractor_dist_threshold in attractor_dist_thresholds:
            print(f'  - Attractor dist threshold: {attractor_dist_threshold}')
            for attractor_prob_threshold in attractor_prob_thresholds:
                print(f'    - Attractor prob threshold: {attractor_prob_threshold}')

                roa_estimator.set_attractor_dist_threshold(attractor_dist_threshold)
                roa_estimator.set_attractor_prob_threshold(attractor_prob_threshold)

                roa_estimator.compute_attractor_labels()
                roa_estimator.compute_attractor_probabilities(plot=True)
                roa_estimator.predict_attractor_labels(save=True, plot=True)
                roa_estimator.plot_roas(plot_separatrix=True)
                roa_estimator.compute_classification_results(save=True)
                roa_estimator.plot_classification_results()
                
                gc.collect()
                pbar.update(1)

                roa_estimator.reset(for_analysis=True)



