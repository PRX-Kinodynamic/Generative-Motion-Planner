from typing import Set


class DatasetCombiner:
    """Base combiner interface."""
    def combine(self, old_ids: Set[int], new_ids: Set[int]) -> Set[int]:
        raise NotImplementedError


class ConcatCombiner(DatasetCombiner):
    def combine(self, old_ids: Set[int], new_ids: Set[int]) -> Set[int]:
        return [*old_ids, *new_ids]
