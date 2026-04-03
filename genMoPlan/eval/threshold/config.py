"""Configuration for threshold optimization."""

from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    """Configuration for threshold optimization.

    All four threshold modes (fixed, joint, lambda-only, delta-only) are always
    run during evaluation. This config controls the optimization objective and
    grid search parameters shared across modes.

    Attributes:
        optimize_objective: Objective function for optimization.
            - "loss": Weighted misclassification + unknown rate
            - "jstat": Youden's J statistic (sensitivity + specificity - 1)
            - "f1": F1 score (target_f1 used as minimum threshold)
        w: Weight for misclassification in loss objective.
            loss = w * misclass_rate + (1 - w) * unknown_rate
        target_f1: Minimum F1 target for "f1" objective.
        fixed_lambda_star: Fixed lambda used for fixed and delta-only modes.
        fixed_delta_star: Fixed delta used for fixed and lambda-only modes.
        lambda_grid_size: Number of grid points for lambda search.
        delta_grid_size: Number of grid points for delta search.
        delta_min: Minimum delta value in grid search.
        delta_max: Maximum delta value in grid search.
        use_p_invalid_veto: Whether to veto predictions with high p_invalid.
    """

    optimize_objective: str = "loss"
    w: float = 0.9
    target_f1: float = 0.90

    fixed_lambda_star: float = 0.5
    fixed_delta_star: float = 0.1

    lambda_grid_size: int = 100
    delta_grid_size: int = 100
    delta_min: float = 0.01
    delta_max: float = 0.49

    use_p_invalid_veto: bool = True

    def __post_init__(self):
        valid_objectives = {"loss", "jstat", "f1"}
        if self.optimize_objective not in valid_objectives:
            raise ValueError(
                f"optimize_objective must be one of {valid_objectives}, "
                f"got '{self.optimize_objective}'"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdConfig":
        """Create ThresholdConfig from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
