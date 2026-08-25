"""
PhysCons: Toy PoC for patch routing with hard conservation constraints.

Combines ideas from:
- Shodh-MoE: patch-based routing
- PI-HC-MoE: hard conservation constraints via MoE
- PI-MFM: physics-informed residual correction

This is a toy 1D implementation, not a multimodal foundation model.
"""

__version__ = "0.1.0"

from .data import generate_conserved_field, generate_broken_field
from .router import PatchRouter
from .conservation import ConservationLayer

__all__ = [
    "generate_conserved_field",
    "generate_broken_field",
    "PatchRouter",
    "ConservationLayer",
]
