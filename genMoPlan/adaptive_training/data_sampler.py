from abc import ABC, abstractmethod
from typing import List, Optional, Sequence
import os

import numpy as np
import matplotlib.pyplot as plt

class DiscreteSampler(ABC):
    """
    Abstract discrete sampler interface.
    Any discrete sampler should inherit from this class and implement the following method:
    - _sample_indices
    
    The _sample_indices method should return the indices of the sampled candidates and the probabilities of the candidates.
    """

    @abstractmethod
    def _sample_indices(
        self, 
        uncertainty: np.ndarray, 
        num_candidates: int,
        num_samples: int,
    ) -> List[int]: ...

    def _plot_sample_hist(
        self, 
        probs: np.ndarray, 
        chosen_indices: np.ndarray, 
        save_path: str,
        hist_save_options: dict,
    ):
        import matplotlib.pyplot as plt

        prob_bins = np.linspace(0, probs.max(), 20)
        sample_probs = [probs[sample_id] for sample_id in chosen_indices]
        counts, bins, patches = plt.hist(sample_probs, bins=prob_bins, alpha=0.7, edgecolor='black')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title(hist_save_options.get('title', 'Histogram: Frequency vs Probability Buckets'))
        plt.savefig(os.path.join(save_path, 'sample_hist.png'))
        plt.close()

    def _plot_sample_plot(
        self, 
        uncertainty: np.ndarray,
        chosen_indices: np.ndarray, 
        dataset_start_points: np.ndarray, 
        save_path: str,
        sample_plot_options: dict,
    ):
        chosen_start_points = dataset_start_points[chosen_indices]

        plt.figure(figsize=(10.08, 8))
        plt.scatter(
            dataset_start_points[:, 0], dataset_start_points[:, 1], 
            c=uncertainty,
            cmap='viridis',
            alpha=1, 
            edgecolors='none',
            s=10,
        )
        plt.colorbar(label='Uncertainty')
        plt.scatter(
            chosen_start_points[:, 0], chosen_start_points[:, 1], 
            c='r',
            marker='x',
            alpha=1, 
            s=10,
        )
        plt.title(sample_plot_options.get('title', 'New Samples Plot'))
        plt.savefig(os.path.join(save_path, 'new_samples.png'))
        plt.close()
            
    def sample(
        self, 
        uncertainty: np.ndarray, 
        candidate_ids: Sequence[int], 
        dataset_start_points: np.ndarray,
        num_samples: int,
        save_path: str,
        hist_save_options: Optional[dict] = None,
        sample_plot_options: Optional[dict] = None,
    ) -> List[int]:

        if len(candidate_ids) == 0:
            return []

        num_samples = min(num_samples, len(candidate_ids))
        num_candidates = len(candidate_ids)

        chosen_indices, probs = self._sample_indices(
            uncertainty,
            num_candidates,
            num_samples,
        )

        np.save(os.path.join(save_path, 'sampled_indices.npy'), chosen_indices)


        if hist_save_options:
            self._plot_sample_hist(probs, chosen_indices, save_path, hist_save_options)

        if sample_plot_options:
            self._plot_sample_plot(uncertainty, chosen_indices, dataset_start_points, save_path, sample_plot_options)


        return [candidate_ids[i] for i in chosen_indices]


class UniformDiscreteSampler(DiscreteSampler):
    """
    Samples uniformly from the candidate ids.
    """
    def _sample_indices(
        self, 
        uncertainty: np.ndarray,
        num_candidates: int,
        num_samples: int = 1,
    ) -> List[int]:
        # Uniform probabilities over all candidates
        probs = np.full(num_candidates, 1.0 / num_candidates, dtype=float)

        chosen_indices = np.random.choice(
            num_candidates,
            size=num_samples,
            replace=False,
            p=probs,
        )

        return chosen_indices, probs



class WeightedDiscreteSampler(DiscreteSampler):
    """
    Constructs a distribution proportional to the provided uncertainty and samples from it.
    """
    def _sample_indices(
        self, 
        uncertainty: np.ndarray, 
        num_candidates: int, 
        num_samples: int = 1,
    ) -> List[int]:
        # Form probabilities proportional to the provided uncertainty.
        total_uncertainty = float(uncertainty.sum())
        if total_uncertainty <= 1e-8:
            print("Warning: all uncertainties are (close to) zero → falling back to uniform sampling.")
            # Degenerate case: all uncertainties are (close to) zero → fall back to uniform sampling.
            probs = np.full(num_candidates, 1.0 / num_candidates, dtype=float)
        else:
            probs = uncertainty / total_uncertainty

        chosen_indices = np.random.choice(
            num_candidates,
            size=num_samples,
            replace=False,
            p=probs,
        )

        return chosen_indices, probs

       

        