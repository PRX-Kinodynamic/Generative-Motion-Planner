"""Result types for threshold optimization."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ThresholdResult:
    """Result of threshold optimization.

    Attributes:
        lambda_star: Optimized (or fixed) probability threshold.
        delta_star: Optimized (or fixed) uncertainty band half-width.
        optimize_objective: The objective used for optimization (None if fixed).
        optimization_loss: Best objective value from grid search (None if fixed).
    """

    lambda_star: float = 0.5
    delta_star: float = 0.1
    optimize_objective: Optional[str] = None
    optimization_loss: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "lambda_star": self.lambda_star,
            "delta_star": self.delta_star,
            "optimize_objective": self.optimize_objective,
            "optimization_loss": self.optimization_loss,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdResult":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
