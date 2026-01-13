"""
Backward compatibility module.

This module re-exports the system classes from their new location.
Please import from genMoPlan.systems instead.
"""

from genMoPlan.systems.base import BaseSystem, Outcome

__all__ = ["BaseSystem", "Outcome"]
