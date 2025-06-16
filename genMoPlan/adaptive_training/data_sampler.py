from typing import List, Sequence
import numpy as np


class Sampler:
    """Base sampler interface."""
    def sample(self, sigma: np.ndarray, candidate_ids: Sequence[int], k: int) -> List[int]:
        raise NotImplementedError


class WeightedSampler(Sampler):
    def sample(self, sigma: np.ndarray, candidate_ids: Sequence[int], k: int) -> List[int]:
        if len(candidate_ids) == 0:
            return []
        probs = sigma / (sigma.sum() + 1e-8)
        k = min(k, len(candidate_ids))
        chosen = np.random.choice(len(candidate_ids), size=k, replace=False, p=probs)
        return [candidate_ids[i] for i in chosen]