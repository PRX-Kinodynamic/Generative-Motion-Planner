"""Configuration for conformal prediction."""

from dataclasses import dataclass


@dataclass
class ConformalConfig:
    """Configuration for conformal prediction.

    Attributes:
        alpha: Miscoverage level. Coverage guarantee is 1 - alpha.
            Default 0.1 gives 90% coverage.
    """

    alpha: float = 0.1

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
