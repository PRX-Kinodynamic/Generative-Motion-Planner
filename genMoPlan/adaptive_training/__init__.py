import os
from .data_sampler import *
from .uncertainty import *
from .dataset_combiner import *


from os import path
import random
from typing import Sequence, Set, Union, Optional

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.datasets.trajectory import TrajectoryDataset
from genMoPlan.eval.final_states import plot_final_state_std
from genMoPlan import utils

def load_start_point(fname, dataset_path, observation_dim):
    fpath = path.join(dataset_path, "trajectories", fname)
    trajectory = utils.read_trajectory(fpath, observation_dim)
    return trajectory[0]

class AdaptiveTrainer:
    combiner: DatasetCombiner
    uncertainty: Uncertainty
    sampler: Sampler
    mean_uncertainties: list[float] = []
    logdir: Optional[str]

    def __init__(
        self,
        model: GenerativeModel,
        args: object,
        *,
        combiner: Union[str, DatasetCombiner],
        uncertainty: Union[str, Uncertainty],
        sampler: Union[str, Sampler],
        combiner_kwargs: dict = {},
        uncertainty_kwargs: dict = {},
        sampler_kwargs: dict = {},
        init_size: int = 100,
        val_size: int = 40,
        step_size: int = 70,
        stop_std: float = 0.01,
        max_iters: int = 30,
        filter_seen: bool = True,
        **kwargs,
    ):
        self.model = model
        self.args = args

        self.init_size = init_size
        self.step_size = step_size
        self.stop_std = stop_std
        self.max_iters = max_iters
        self.filter_seen = filter_seen
        self.val_size = val_size

        if isinstance(combiner, str):
            combiner_class = utils.import_class(combiner)
            self.combiner = combiner_class(**combiner_kwargs)
        else:
            self.combiner = combiner

        if isinstance(uncertainty, str):
            uncertainty_class = utils.import_class(uncertainty)
            self.uncertainty = uncertainty_class(**uncertainty_kwargs)
        else:
            self.uncertainty = uncertainty

        if isinstance(sampler, str):
            sampler_class = utils.import_class(sampler)
            self.sampler = sampler_class(**sampler_kwargs)
        else:
            self.sampler = sampler

        start_points, fnames = self._load_start_points()

        # ------------------ validation split ------------------ #
        _val_ids = random.sample(list(range(len(fnames))), self.val_size)
        self.val_dataset = self._create_dataset(_val_ids, fnames, is_validation=True)

        # remove validation ids from train pool
        remaining_ids = [i for i in range(len(fnames)) if i not in _val_ids]
        self._dataset_fnames = [fnames[i] for i in remaining_ids]
        self._dataset_start_points = start_points[remaining_ids]
    
    def _load_start_points(self):
        dataset_path = path.join("data_trajectories", self.args.dataset)
        trajectories_path = path.join(dataset_path, "trajectories")

        fnames = utils.get_fnames_to_load(dataset_path, trajectories_path)

        args_list = [(fname, dataset_path, self.args.observation_dim) for fname in fnames]

        start_points = utils.parallelize_toggle(load_start_point, args_list, parallel=True, desc="Loading start points")

        start_points = np.array(start_points)

        return start_points, fnames

    # ----------------------- private helpers ----------------------- #

    def _create_dataset(self, ids: Sequence[int], dataset_fnames: Sequence[str], is_validation: bool = False):
        fnames = [dataset_fnames[i] for i in ids]

        return TrajectoryDataset(
            dataset=self.args.dataset,
            horizon_length=self.args.horizon_length,
            history_length=self.args.history_length,
            stride=self.args.stride,
            observation_dim=self.args.observation_dim,
            trajectory_normalizer=getattr(self.args, "trajectory_normalizer", None),
            normalizer_params=getattr(self.args, "normalizer_params", {}),
            trajectory_preprocess_fns=getattr(self.args, "trajectory_preprocess_fns", ()),
            preprocess_kwargs=getattr(self.args, "preprocess_kwargs", {"trajectory": {}}),
            use_horizon_padding=self.args.use_horizon_padding,
            use_history_padding=self.args.use_history_padding,
            is_history_conditioned=self.args.is_history_conditioned,
            is_validation=is_validation,
            fnames=fnames,
        )

    def _build_trainer(self, train_dataset, val_dataset):
        return utils.Trainer(
            self.model,
            train_dataset,
            val_dataset,
            validation_kwargs=self.args.validation_kwargs,
            ema_decay=self.args.ema_decay,
            batch_size=self.args.batch_size,
            min_num_batches_per_epoch=self.args.min_num_batches_per_epoch,
            train_lr=self.args.learning_rate,
            gradient_accumulate_every=self.args.gradient_accumulate_every,
            val_batch_size=self.args.val_batch_size,
            num_epochs=self.args.num_epochs,
            patience=self.args.patience,
            min_delta=self.args.min_delta,
            early_stopping=self.args.early_stopping,
            save_freq=self.args.save_freq,
            log_freq=self.args.log_freq,
            save_parallel=self.args.save_parallel,
            results_folder=self.logdir,
            bucket=self.args.bucket,
            n_reference=self.args.n_reference,
            method=self.args.method,
            exp_name=self.args.exp_name,
            num_workers=self.args.num_workers,
            device=self.args.device,
            seed=self.args.seed,
        )
    
    def _plot_new_samples(self, new_ids: Sequence[int], uncertainty: np.ndarray, titleSuffix: str):
        new_start_points = self._dataset_start_points[new_ids]
        new_uncertainty = uncertainty[new_ids]

        plot_final_state_std(new_uncertainty, new_start_points, save_path=os.path.join(self.logdir, "new_samples.png"), title=f"New Samples - {titleSuffix}", s=10)

    def _add_mean_uncertainty(self, mean_uncertainty: float):
        self.mean_uncertainties.append(mean_uncertainty)

        np.save(path.join(self.args.savepath, "mean_uncertainties.txt"), self.mean_uncertainties)

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

        model_state_dict = torch.load(model_path, weights_only=False, map_location=torch.device(self.args.device))
        self.model.load_state_dict(model_state_dict["model"])

    # --------------------------- main loop ------------------------- #

    def run(self):
        num_dataset_states = len(self._dataset_start_points)
        all_dataset_ids = list(range(num_dataset_states))

        # List of ids that constitute the current training dataset
        dataset_ids = random.sample(all_dataset_ids, min(self.init_size, num_dataset_states))

        # Set of ids that have appeared at least once
        seen_set: Set[int] = set(dataset_ids)

        self.logdir = None

        for iteration in range(self.max_iters):
            if self.logdir is not None: # Load best model from previous iteration
                self._load_best_model()

            self.logdir = path.join(self.args.savepath, f"iteration_{iteration}")
            os.makedirs(self.logdir, exist_ok=True)

            # ---------------- training ---------------- #
            print(f"\n[ adaptive_training/trainer ] Iteration {iteration}")

            train_dataset = self._create_dataset(dataset_ids, self._dataset_fnames, is_validation=False)

            trainer = self._build_trainer(train_dataset, self.val_dataset)
            trainer.train()

            # ---------------- uncertainty over all start states ---------------- #

            # If every start state has appeared at least once, stop the loop
            if len(seen_set) == num_dataset_states:
                print("[ adaptive_training/trainer ] All start states have been used. Stopping.")
                break

            # Compute uncertainty for the whole pool of start states
            uncertainty = self.uncertainty.compute(self.model, self.args, self._dataset_start_points, save_path=self.logdir, title_suffix=f"Iteration {iteration}")

            if self.filter_seen:
                unseen_mask = np.array([i not in seen_set for i in all_dataset_ids])
                mean_uncertainty = float(np.mean(uncertainty[unseen_mask])) if unseen_mask.any() else 0.0
            else:
                mean_uncertainty = float(np.mean(uncertainty))

            self._add_mean_uncertainty(mean_uncertainty)

            print(f"[ adaptive_training/trainer ] Mean std: {mean_uncertainty:.6f}")

            if mean_uncertainty < self.stop_std:
                print(f"[ adaptive_training/trainer ] Reached target std {self.stop_std}. Stopping.")
                break

            # Sample new candidate ids from the full pool
            if self.filter_seen:
                candidate_ids = []
                candidate_uncertainty = []
                for i in all_dataset_ids:
                    if i not in seen_set:
                        candidate_ids.append(i)
                        candidate_uncertainty.append(uncertainty[i])

                candidate_uncertainty = np.array(candidate_uncertainty)
            else:
                candidate_ids = all_dataset_ids
                candidate_uncertainty = uncertainty

            new_ids = self.sampler.sample(candidate_uncertainty, candidate_ids, self.step_size)

            if not new_ids:
                print("[ adaptive_training/trainer ] Sampler returned no ids. Stopping.")
                break

            self._plot_new_samples(new_ids, uncertainty, f"Iteration {iteration}")

            # Update dataset lists/sets
            dataset_ids = self.combiner.combine(dataset_ids, new_ids)
            seen_set.update(new_ids)

            # Stop if we've now covered the entire pool (at least once)
            if len(seen_set) == num_dataset_states:
                print("[ adaptive_training/trainer ] Exhausted all start states. Stopping.")
                break

        print("[ adaptive_training/trainer ] Training complete.")

        trainer.save_model("final", save_path=self.args.savepath)

        