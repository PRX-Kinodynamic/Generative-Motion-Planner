"""Result types for conformal prediction."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConformalResult:
    """Result of conformal calibration.

    Attributes:
        q_hat: Single-class calibrated quantile threshold.
        q_hat_success: Per-class quantile for SUCCESS label.
        q_hat_failure: Per-class quantile for FAILURE label.
        lambda_star: Lambda from threshold optimization (used for scores).
        delta_star: Delta from threshold optimization (used for scores).
        alpha: Miscoverage level used for calibration.
        calibration_set_size: Number of calibration points used.
    """

    q_hat: Optional[float] = None
    q_hat_success: Optional[float] = None
    q_hat_failure: Optional[float] = None

    lambda_star: Optional[float] = None
    delta_star: Optional[float] = None

    alpha: Optional[float] = None
    calibration_set_size: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "q_hat": self.q_hat,
            "q_hat_success": self.q_hat_success,
            "q_hat_failure": self.q_hat_failure,
            "lambda_star": self.lambda_star,
            "delta_star": self.delta_star,
            "alpha": self.alpha,
            "calibration_set_size": self.calibration_set_size,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalResult":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
