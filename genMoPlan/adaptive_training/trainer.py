import os
from os import path
import random
from typing import Sequence, Set, Union, Optional

import torch
import matplotlib.pyplot as plt
import numpy as np

from genMoPlan.adaptive_training.data_sampler import *
from genMoPlan.adaptive_training.uncertainty import *
from genMoPlan.adaptive_training.dataset_combiner import *
from genMoPlan.adaptive_training.animation_generator import AnimationGenerator
from genMoPlan.adaptive_training.video_generator import (
    generate_iteration_evolution_videos,
)
from genMoPlan.eval.roa import Classifier
from genMoPlan.utils.trajectory_generator import TrajectoryGenerator
from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.datasets.trajectory import TrajectoryDataset
from genMoPlan import utils


def get_start_point(fname: str, dataset_path: str, read_trajectory_fn: Callable):
    fpath = path.join(dataset_path, "trajectories", fname)
    trajectory = read_trajectory_fn(fpath)

    if trajectory is None:
        return None

    return trajectory[0]


# def get_start_point_attractor(fname: str, dataset_path: str, read_trajectory_fn: Callable, attractors: np.ndarray):
#     fpath = path.join(dataset_path, "trajectories", fname)
#     trajectory = read_trajectory_fn(fpath)

#     if trajectory is None:
#         return None

#     final_point = trajectory[-1]

#     attractor_dist = np.linalg.norm(final_point - attractors, axis=1)

#     return attractors[np.argmin(attractor_dist)]


class AdaptiveTrainer:
    mean_uncertainties: list[float] = []
    logdir: Optional[str]
    animation_generator: Optional[AnimationGenerator] = None
    uncertainty: Uncertainty
    sampler: DiscreteSampler
    combiner: DatasetCombiner

    def __init__(
        self,
        model: GenerativeModel,
        args: object,
        *,
        combiner: Union[str, DatasetCombiner],
        uncertainty: Union[str, Uncertainty],
        sampler: Union[str, DiscreteSampler],
        n_runs: int,
        num_inference_steps: int,
        conditional_sample_kwargs: dict,
        post_process_fns: List[Callable],
        post_process_fn_kwargs: dict,
        sampling_batch_size: int,
        combiner_kwargs: dict = {},
        uncertainty_kwargs: dict = {},
        sampler_kwargs: dict = {},
        init_size: int = 100,
        val_size: int = 40,
        step_size: int = 70,
        stop_uncertainty: float = 0.01,
        max_iters: int = 30,
        filter_seen: bool = True,
        animate_plots: bool = False,
        **kwargs,
    ):
        self.model = model
        self.args = args
        self.init_size = init_size
        self.step_size = step_size
        self.stop_uncertainty = stop_uncertainty
        self.max_iters = max_iters
        self.filter_seen = filter_seen
        self.val_size = val_size
        self.animate_plots = animate_plots
        self.n_runs = n_runs
        self.num_inference_steps = num_inference_steps
        self.conditional_sample_kwargs = conditional_sample_kwargs
        self.post_process_fns = post_process_fns
        self.post_process_fn_kwargs = post_process_fn_kwargs
        self.sampling_batch_size = sampling_batch_size

        self._all_start_points, fnames = self._load_start_points()

        self.sampler = self._initialize_sampler(sampler, sampler_kwargs)
        self.combiner = self._initialize_combiner(combiner, combiner_kwargs)
        self.uncertainty = self._initialize_uncertainty(uncertainty, uncertainty_kwargs)

        self.roa_estimator, self.roa_labels = self._initialize_roa_estimator()
        self.trajectory_generator = TrajectoryGenerator(
            dataset=self.args.dataset,
            model=self.model,
            model_args=self.args,
            inference_params=utils.load_inference_params(self.args.dataset),
            device=self.args.device,
            verbose=True,
            default_batch_size=self.sampling_batch_size,
        )

        if self.animate_plots:
            self.animation_generator = AnimationGenerator(
                dataset_name=self.args.dataset,
                uncertainty_name=self.uncertainty.name,
            )

        # ------------------ validation split ------------------ #
        _val_ids = random.sample(list(range(len(fnames))), self.val_size)
        self.val_dataset = self._create_dataset(_val_ids, fnames, is_validation=True)

        # remove validation ids from train pool
        self._train_ids = [i for i in range(len(fnames)) if i not in _val_ids]
        self._train_fnames = [fnames[i] for i in self._train_ids]
        self._train_start_points = self._all_start_points[self._train_ids]

    def _initialize_sampler(self, sampler, sampler_kwargs):
        if isinstance(sampler, str):
            sampler_class = utils.import_class(sampler)
            return sampler_class(**sampler_kwargs)
        else:
            return sampler

    def _initialize_combiner(self, combiner, combiner_kwargs):
        if isinstance(combiner, str):
            combiner_class = utils.import_class(combiner)
            return combiner_class(**combiner_kwargs)
        else:
            return combiner

    def _initialize_uncertainty(self, uncertainty, uncertainty_kwargs):
        if isinstance(uncertainty, str):
            uncertainty_class = utils.import_class(uncertainty)
            return uncertainty_class(**uncertainty_kwargs)
        else:
            return uncertainty

    def _initialize_roa_estimator(self):
        from genMoPlan.utils.setup import get_dataset_config

        cfg = get_dataset_config(self.args.dataset)
        system = None
        get_system = getattr(cfg, "get_system", None)
        if callable(get_system):
            system = get_system()

        roa_estimator = Classifier(
            dataset=self.args.dataset,
            system=system,
        )

        roa_start_states, roa_expected_labels = utils.load_roa_labels(self.args.dataset)

        roa_labels = utils.query_roa_labels_for_start_points(
            self._all_start_points,
            roa_start_states,
            roa_expected_labels,
        )

        roa_estimator.start_states = self._all_start_points
        roa_estimator.expected_labels = roa_labels

        return roa_estimator, roa_labels

    def _load_start_points(self):
        dataset_path = path.join(utils.get_data_trajectories_path(), self.args.dataset)
        trajectories_path = path.join(dataset_path, "trajectories")

        fnames = utils.get_fnames_to_load(dataset_path, trajectories_path)

        args_list = [
            (fname, dataset_path, self.args.read_trajectory_fn) for fname in fnames
        ]

        start_points = utils.parallelize_toggle(
            get_start_point, args_list, parallel=True, desc="Loading start points"
        )

        start_points = np.array(
            [start_point for start_point in start_points if start_point is not None]
        )

        return start_points, fnames

    # ----------------------- private helpers ----------------------- #

    def _create_dataset(
        self,
        ids: Sequence[int],
        dataset_fnames: Sequence[str],
        is_validation: bool = False,
    ):
        fnames = [dataset_fnames[i] for i in ids]

        return TrajectoryDataset(
            dataset=self.args.dataset,
            horizon_length=self.args.horizon_length,
            history_length=self.args.history_length,
            stride=self.args.stride,
            observation_dim=self.args.observation_dim,
            trajectory_normalizer=getattr(self.args, "trajectory_normalizer", None),
            normalizer_params=getattr(self.args, "normalizer_params", {}),
            trajectory_preprocess_fns=getattr(
                self.args, "trajectory_preprocess_fns", ()
            ),
            preprocess_kwargs=getattr(
                self.args, "preprocess_kwargs", {"trajectory": {}}
            ),
            use_horizon_padding=self.args.use_horizon_padding,
            use_history_padding=self.args.use_history_padding,
            is_history_conditioned=self.args.is_history_conditioned,
            is_validation=is_validation,
            fnames=fnames,
            read_trajectory_fn=self.args.read_trajectory_fn,
        )

    def _build_trainer(self, train_dataset, val_dataset):
        return utils.Trainer(
            self.model,
            train_dataset,
            val_dataset,
            validation_kwargs=self.args.validation_kwargs,
            ema_decay=self.args.ema_decay,
            batch_size=self.args.batch_size,
            min_num_steps_per_epoch=self.args.min_num_steps_per_epoch,
            train_lr=self.args.learning_rate,
            gradient_accumulate_every=self.args.gradient_accumulate_every,
            val_batch_size=self.args.val_batch_size,
            num_epochs=self.args.num_epochs,
            patience=self.args.patience,
            min_delta=self.args.min_delta,
            warmup_epochs=self.args.warmup_epochs,
            early_stopping=self.args.early_stopping,
            save_freq=self.args.save_freq,
            log_freq=self.args.log_freq,
            save_parallel=self.args.save_parallel,
            results_folder=self.logdir,
            method=self.args.method,
            exp_name=f"{self.args.dataset}/{self.args.exp_name}",
            num_workers=self.args.num_workers,
            device=self.args.device,
            seed=self.args.seed,
            use_lr_scheduler=self.args.use_lr_scheduler,
            lr_scheduler_warmup_steps=self.args.lr_scheduler_warmup_steps,
            lr_scheduler_min_lr=self.args.lr_scheduler_min_lr,
            useAdamW=self.args.useAdamW,
            optimizer_kwargs=self.args.optimizer_kwargs,
            clip_grad_norm=getattr(self.args, "clip_grad_norm", None),
        )

    def _add_mean_uncertainty(self, mean_uncertainty: float):
        self.mean_uncertainties.append(mean_uncertainty)

        np.save(
            path.join(self.args.savepath, "mean_uncertainties.txt"),
            self.mean_uncertainties,
        )

        # Combined linear and log-scale plots in one figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Linear scale
        axes[0].plot(self.mean_uncertainties)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Mean uncertainty")
        axes[0].set_title("Linear Scale")

        # Log scale
        axes[1].plot(self.mean_uncertainties)
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Mean uncertainty")
        axes[1].set_title("Log Scale")

        fig.suptitle("Mean uncertainty over Iterations")
        fig.tight_layout()
        fig.savefig(path.join(self.args.savepath, "mean_uncertainties.png"))
        plt.close(fig)

    def _load_best_model(self):
        model_path = os.path.join(self.logdir, "best.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Best model not found at {model_path}")

        model_state_dict = torch.load(
            model_path, weights_only=False, map_location=torch.device(self.args.device)
        )
        self.model.load_state_dict(model_state_dict["model"])

    def _update_logdir(self, iteration: int):
        self.logdir = path.join(self.args.savepath, f"iteration_{iteration}")
        os.makedirs(self.logdir, exist_ok=True)

    def _train_iteration(self, iteration: int, train_ids: Sequence[int]):
        print(f"\n[ adaptive_training/trainer ] Iteration {iteration}")

        train_dataset = self._create_dataset(
            train_ids, self._train_fnames, is_validation=False
        )
        trainer = self._build_trainer(train_dataset, self.val_dataset)
        trainer.train(reraise_keyboard_interrupt=True)

        return trainer

    def _acheived_low_uncertainty(self, uncertainty: np.ndarray):

        mean_uncertainty = float(np.mean(uncertainty))

        self._add_mean_uncertainty(mean_uncertainty)

        print(
            f"[ adaptive_training/trainer ] Mean {self.uncertainty.name}: {mean_uncertainty:.6f}"
        )

        achieved_stop_uncertainty = mean_uncertainty < self.stop_uncertainty

        if achieved_stop_uncertainty:
            print(
                f"[ adaptive_training/trainer ] Reached target uncertainty {self.stop_uncertainty}. Stopping."
            )

        return achieved_stop_uncertainty

    def _get_sampling_uncertainty(self, uncertainty: np.ndarray, seen_set: Set[int]):
        """
        If filter_seen is True, set uncertainty of seen ids to 0.
        Otherwise, return the uncertainty of all ids.
        """

        sampling_uncertainty = uncertainty.copy()
        if not self.filter_seen:
            return sampling_uncertainty

        # Set uncertainty of seen ids to 0
        for i in seen_set:
            sampling_uncertainty[i] = 0

        return sampling_uncertainty

    def _update_animation(
        self, iteration: int, uncertainty: np.ndarray, new_ids: Sequence[int]
    ):
        if self.animate_plots:
            self.animation_generator.generate_animations(
                iteration, uncertainty, self._train_start_points, new_ids, self.logdir
            )

    def _create_videos(self, num_iterations: int):
        """
        Creates mp4 files for uncertainty and samples from the images generated in each iteration.
        """
        print("[ adaptive_training/trainer ] Creating evolution videos...")
        generate_iteration_evolution_videos(num_iterations, self.args.savepath)

    def _generate_final_states(self):
        """
        Generates final states for all start states and not only the
        ones in the current training dataset.
        """
        num_start_states = len(self._all_start_points)
        dim = self.args.observation_dim

        final_states = np.zeros((num_start_states, self.n_runs, dim))

        for i in tqdm(
            range(self.n_runs), desc=f"Generating final states {self.n_runs} runs"
        ):
            run_final_states = self.trajectory_generator.generate_trajectories(
                self._all_start_points,
                batch_size=self.sampling_batch_size,
                max_path_length=None,
                num_inference_steps=self.num_inference_steps,
                conditional_sample_kwargs=self.conditional_sample_kwargs,
                only_return_final_states=True,
                post_process_fns=self.post_process_fns,
                post_process_fn_kwargs=self.post_process_fn_kwargs,
                horizon_length=self.args.horizon_length,
                verbose=True,
            )
            final_states[:, i, :] = run_final_states

        np.save(
            path.join(self.logdir, f"final_states_{self.n_runs}_runs.npy"), final_states
        )

        return final_states

    def _estimate_roa(self, final_states: np.ndarray):
        """
        Estimates the ROA of the model using the final states.
        """
        s = 5

        self.roa_estimator.reset()
        self.roa_estimator.start_states = self._all_start_points
        self.roa_estimator.expected_labels = self.roa_labels

        self.roa_estimator.results_path = os.path.join(self.logdir, "roa_results")
        os.makedirs(self.roa_estimator.results_path, exist_ok=True)
        self.roa_estimator.set_final_states(final_states)

        # System-driven, outcome-based RoA estimation.
        self.roa_estimator.compute_outcome_labels()
        self.roa_estimator.compute_outcome_probabilities()
        self.roa_estimator.predict_outcomes(save=True)
        self.roa_estimator.derive_labels_from_outcomes()
        self.roa_estimator.plot_roas(plot_separatrix=True, s=s)
        self.roa_estimator.compute_classification_results(save=True)
        self.roa_estimator.plot_classification_results(s=s)

    # def _get_expected_final_points(self, ids: Sequence[int]):
    #     """
    #     Get the final points for the given ids.
    #     """

    #     attractors = np.array(list(self.roa_estimator.attractors.keys()))

    #     args_list = [(fname, self.args.dataset_path, self.args.read_trajectory_fn, attractors) for fname in ids]

    #     # expected_final = utils.parallelize_toggle(get_start_point_attractor, args_list, parallel=True, desc="Getting attractor labels")

    #     return attractor_labels

    # --------------------------- main loop ------------------------- #

    def run(self):
        num_train_states = len(self._train_start_points)
        all_train_ids = list(range(num_train_states))

        # List of ids that constitute the current training dataset
        current_train_ids = random.sample(
            all_train_ids, min(self.init_size, num_train_states)
        )

        # Set of ids that have appeared at least once (start points seen at least once)
        seen_set: Set[int] = set(current_train_ids)

        self.logdir = None

        last_iteration = -1
        trainer = None

        for iteration in range(self.max_iters):
            try:
                last_iteration = iteration
                if self.logdir is not None:  # Load best model from previous iteration
                    self._load_best_model()

                self._update_logdir(iteration)

                trainer = self._train_iteration(iteration, current_train_ids)

                it_final_states = self._generate_final_states()

                self._estimate_roa(it_final_states)

                it_train_final_states = it_final_states[self._train_ids]

                uncertainty = self.uncertainty.compute_normalized_uncertainty(
                    it_train_final_states,
                    self.args,
                    self._train_start_points,
                    save_path=self.logdir,
                    title_suffix=f"Iteration {iteration}",
                )

                if self._acheived_low_uncertainty(uncertainty):
                    break

                sampling_uncertainty = self._get_sampling_uncertainty(
                    uncertainty, seen_set
                )

                new_ids = self.sampler.sample(
                    sampling_uncertainty,
                    all_train_ids,
                    self._train_start_points,
                    self.step_size,
                    save_path=self.logdir,
                    hist_save_options={
                        "iteration_title": f"Iteration {iteration} - New Samples Histogram",
                        "overall_title": "New Samples Histogram until iteration {iteration}",
                    },
                    sample_plot_options={
                        "title": f"Iteration {iteration} - New Samples Plot",
                        "use_cmap": False,
                        "cmap": None,
                    },
                )

                if not new_ids:
                    print(
                        "[ adaptive_training/trainer ] Sampler returned no ids. Stopping."
                    )
                    break

                self._update_animation(iteration, uncertainty, new_ids)

                # Update dataset lists/sets
                current_train_ids = self.combiner.combine(
                    current_train_ids, new_ids, uncertainty
                )
                seen_set.update(new_ids)

                # Stop if we've now covered the entire pool (at least once)
                if len(seen_set) == num_train_states:
                    print(
                        "[ adaptive_training/trainer ] Exhausted all start states. Stopping."
                    )
                    break
            except KeyboardInterrupt:
                print("[ adaptive_training/trainer ] Keyboard interrupt. Stopping.")
                break

        print("[ adaptive_training/trainer ] Training complete.")

        if trainer is not None:
            trainer.save_model("final", save_path=self.args.savepath)

            self._load_best_model()

            trainer.save_model("best", save_path=self.args.savepath)

        if last_iteration > -1:
            self._create_videos(last_iteration + 1)
