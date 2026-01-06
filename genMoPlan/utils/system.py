"""
Backward compatibility module.

This module re-exports the system classes from their new location.
Please import from genMoPlan.utils.systems instead.
"""

from genMoPlan.utils.systems.base import BaseSystem, Outcome

__all__ = ["BaseSystem", "Outcome"]
