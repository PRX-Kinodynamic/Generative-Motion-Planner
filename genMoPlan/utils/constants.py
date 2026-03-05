"""
Mask constants for history masking feature.

Convention:
- MASK_ON = 0.0: Position is MASKED (missing/padded/invalid) - replaced with learned mask_token
- MASK_OFF = 1.0: Position is UNMASKED (present/valid/real) - keeps original value

This convention follows the model formula:
    x = mask * x + (1 - mask) * mask_token

Where:
- mask=1 (MASK_OFF): keeps original x value
- mask=0 (MASK_ON): uses learned mask_token

Usage:
    from genMoPlan.utils.constants import MASK_ON, MASK_OFF

    # Create mask with all positions valid
    mask = torch.full((seq_len,), MASK_OFF)

    # Mark first 3 positions as masked/missing
    mask[:3] = MASK_ON
"""

# Mask value constants
MASK_ON = 0.0   # Position is MASKED (missing/padded/invalid) - replaced with mask_token
MASK_OFF = 1.0  # Position is UNMASKED (present/valid/real) - keeps original value

# Padding strategy constants
PADDING_STRATEGY_ZEROS = "zeros"
PADDING_STRATEGY_FIRST = "first"
PADDING_STRATEGY_LAST = "last"
PADDING_STRATEGY_MIRROR = "mirror"

VALID_PADDING_STRATEGIES = [
    PADDING_STRATEGY_ZEROS,
    PADDING_STRATEGY_FIRST,
    PADDING_STRATEGY_LAST,
    PADDING_STRATEGY_MIRROR,
]

# Default values for backward compatibility
DEFAULT_HISTORY_MASK_PADDING_VALUE = PADDING_STRATEGY_ZEROS
DEFAULT_HISTORY_PADDING_STRATEGY = PADDING_STRATEGY_FIRST


def validate_mask(mask, expected_len=None, context=""):
    """
    Validate that a mask has correct values and optionally correct length.

    Args:
        mask: Tensor to validate
        expected_len: Expected length of mask (optional)
        context: Context string for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If mask is invalid
    """
    import torch

    if mask is None:
        raise ValueError(f"{context}: Mask is None")

    if not isinstance(mask, torch.Tensor):
        raise ValueError(f"{context}: Mask must be a torch.Tensor, got {type(mask)}")

    if expected_len is not None and len(mask) != expected_len:
        raise ValueError(
            f"{context}: Mask length {len(mask)} != expected {expected_len}"
        )

    if mask.min() < MASK_ON or mask.max() > MASK_OFF:
        raise ValueError(
            f"{context}: Mask values must be in [{MASK_ON}, {MASK_OFF}], "
            f"got [{mask.min()}, {mask.max()}]"
        )

    return True


def validate_padding_strategy(strategy, valid_strategies=None, context=""):
    """
    Validate that a padding strategy is valid.

    Args:
        strategy: Strategy string to validate
        valid_strategies: List of valid strategies (defaults to VALID_PADDING_STRATEGIES)
        context: Context string for error messages

    Returns:
        True if valid

    Raises:
        ValueError: If strategy is invalid
    """
    if valid_strategies is None:
        valid_strategies = VALID_PADDING_STRATEGIES

    if strategy not in valid_strategies:
        raise ValueError(
            f"{context}: Invalid padding strategy '{strategy}'. "
            f"Valid options: {valid_strategies}"
        )

    return True
