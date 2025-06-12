import os
from .data_sampler import *
from .uncertainty import *
from .dataset_combiner import *



import math
from os import path
import random
from typing import Sequence, Set

import numpy as np
from tqdm import tqdm

from genMoPlan.models.generative.base import GenerativeModel
from genMoPlan.datasets.trajectory import TrajectoryDataset
from genMoPlan import utils


class AdaptiveTrainer:
    combiner: DatasetCombiner
    uncertainty: Uncertainty
    sampler: Sampler

    def __init__(
        self,
        model: GenerativeModel,
        dataset: str,
        args: object,
        *,
        combiner: str | DatasetCombiner,
        uncertainty: str | Uncertainty,
        sampler: str | Sampler,
        combiner_kwargs: dict = {},
        uncertainty_kwargs: dict = {},
        sampler_kwargs: dict = {},
        init_size: int = 100,
        val_size: int = 40,
        step_size: int = 70,
        stop_std: float = 0.01,
        max_iters: int = 30,
        filter_seen: bool = True,
    ):
        self.model = model
        self.dataset = dataset
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

        self._load_start_points()

        # ------------------ validation split ------------------ #
        self._val_ids = random.sample(list(range(len(self._dataset_start_points))), self.val_size)
        self._val_dataset = self._create_dataset(self._val_ids, is_validation=True)

        # remove validation ids from train pool
        self._dataset_fnames = [fname for i, fname in enumerate(self._dataset_fnames) if i not in self._val_ids]
        self._dataset_start_points = np.array([start_point for i, start_point in enumerate(self._dataset_start_points) if i not in self._val_ids])
    
    def _load_start_points(self):
        dataset_path = path.join("data_trajectories", self.dataset)
        trajectories_path = path.join(dataset_path, "trajectories")

        fnames = utils.get_fnames_to_load(dataset_path, trajectories_path)

        start_points = []

        for fname in tqdm(fnames, desc="Loading start points"):
            fpath = path.join(trajectories_path, fname)
            trajectory = utils.read_trajectory(fpath, self.args.observation_dim)
            start_points.append(trajectory[0])

        self._dataset_fnames = fnames
        self._dataset_start_points = np.array(start_points)

    # ----------------------- private helpers ----------------------- #

    def _create_dataset(self, ids: Sequence[int], dataset_fnames: Sequence[str], is_validation: bool = False):
        fnames = [dataset_fnames[i] for i in ids]

        return TrajectoryDataset(
            dataset=self.dataset,
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
    
    def _plot_new_samples(self, new_ids: Sequence[int], sigma: np.ndarray, titleSuffix: str):
        new_start_points = self._dataset_start_points[new_ids]
        new_sigma = sigma[new_ids]

        plot_final_state_std(new_sigma, new_start_points, savepath=self.logdir, title=f"New Samples - {titleSuffix}")

    # --------------------------- main loop ------------------------- #

    def run(self):
        num_dataset_states = len(self._dataset_start_points)
        all_dataset_ids = list(range(num_dataset_states))

        # List of ids that constitute the current training dataset
        dataset_ids = random.sample(all_dataset_ids, min(self.init_size, num_dataset_states))

        # Set of ids that have appeared at least once
        seen_set: Set[int] = set(dataset_ids)

        for iteration in range(self.max_iters):
            self.logdir = path.join(self.args.savepath, f"iteration_{iteration}")
            os.makedirs(self.logdir, exist_ok=True)

            print(f"\n[ adaptive_training/trainer ] Iteration {iteration}")

            # --------------------- train-val split ---------------------- #
            dataset_ids_list = list(dataset_ids)
            random.shuffle(dataset_ids_list)
            split = math.ceil(0.8 * len(dataset_ids_list)) if len(dataset_ids_list) > 1 else len(dataset_ids_list)
            train_ids = dataset_ids_list[:split]
            val_ids = (dataset_ids_list[split:] if split < len(dataset_ids_list) else dataset_ids_list[:1])

            train_dataset = self._create_dataset(train_ids, self._dataset_fnames, is_validation=False)

            trainer = self._build_trainer(train_dataset, val_dataset)
            trainer.train()

            # ---------------- uncertainty over all start states ---------------- #

            # If every start state has appeared at least once, stop the loop
            if len(seen_set) == num_dataset_states:
                print("[ adaptive_training/trainer ] All start states have been used. Stopping.")
                break

            # Compute uncertainty for the whole pool of start states
            sigma = self.uncertainty.compute(self.model, self._dataset_start_points, savepath=self.logdir, titleSuffix=f"Iteration {iteration}")

            if self.filter_seen:
                unseen_mask = np.array([i not in seen_set for i in all_dataset_ids])
                mean_sigma = float(np.mean(sigma[unseen_mask])) if unseen_mask.any() else 0.0
            else:
                mean_sigma = float(np.mean(sigma))
            print(f"[ adaptive_training/trainer ] Mean std: {mean_sigma:.6f}")

            if mean_sigma < self.stop_std:
                print(f"[ adaptive_training/trainer ] Reached target std {self.stop_std}. Stopping.")
                break

            # Sample new candidate ids from the full pool
            if self.filter_seen:
                candidate_ids = [i for i in all_dataset_ids if i not in seen_set]
            else:
                candidate_ids = all_dataset_ids

            new_ids = self.sampler.sample(sigma, candidate_ids, self.step_size)

            if not new_ids:
                print("[ adaptive_training/trainer ] Sampler returned no ids. Stopping.")
                break

            self._plot_new_samples(new_ids, sigma, f"Iteration {iteration}")

            # Update dataset lists/sets
            dataset_ids = self.combiner.combine(dataset_ids, new_ids)
            seen_set.update(new_ids)

            # Stop if we've now covered the entire pool (at least once)
            if len(seen_set) == num_dataset_states:
                print("[ adaptive_training/trainer ] Exhausted all start states. Stopping.")
                break

        print("[ adaptive_training/trainer ] Training complete.")

        