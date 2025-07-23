from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


class DatasetCombiner(ABC):
    """Base combiner interface."""

    @abstractmethod
    def combine(
        self,
        old_ids: Sequence[int],
        new_ids: Sequence[int],
        uncertainty: np.ndarray,
    ) -> List[int]:
        ...


class ConcatCombiner(DatasetCombiner):
    def combine(
        self,
        old_ids: Sequence[int],
        new_ids: Sequence[int],
        uncertainty: np.ndarray,
    ) -> List[int]:
        return [*old_ids, *new_ids]


class ForgettingCombiner(DatasetCombiner):
    def __init__(self, max_size: int):
        self.max_size = max_size

    def combine(
        self,
        old_ids: Sequence[int],
        new_ids: Sequence[int],
        uncertainty: np.ndarray,
    ) -> List[int]:
        combined_ids = list(set(old_ids) | set(new_ids))

        if len(combined_ids) <= self.max_size:
            return combined_ids

        num_to_remove = len(combined_ids) - self.max_size
        
        # We only consider removing points from the old dataset
        old_id_indices = np.array(old_ids)
        old_uncertainties = uncertainty[old_id_indices]
        
        # Find the indices of the easiest samples in the old dataset
        easiest_old_indices = np.argsort(old_uncertainties)[:num_to_remove]
        ids_to_remove = set(old_id_indices[easiest_old_indices])

        # Return the combined set minus the easiest old samples
        return [i for i in combined_ids if i not in ids_to_remove]
