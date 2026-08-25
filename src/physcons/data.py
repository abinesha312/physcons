"""
Synthetic 1D conserved field generation.

For demonstration purposes only. Real physics would involve
PDEs, molecular dynamics, or other domain-specific simulations.
"""

import numpy as np
from typing import Tuple


def generate_conserved_field(
    n_points: int = 100,
    total_mass: float = 1000.0,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """
    Generate a synthetic 1D field where the scalar quantity is conserved.
    
    Args:
        n_points: Number of spatial points
        total_mass: Total conserved quantity (e.g., mass or energy)
        seed: Random seed for reproducibility
        
    Returns:
        field: 1D array of shape (n_points,)
        target_sum: The conserved quantity that should be maintained
    """
    rng = np.random.RandomState(seed)
    
    # Generate random positive values that sum to total_mass
    field = rng.exponential(scale=1.0, size=n_points)
    field = field / field.sum() * total_mass
    
    return field, total_mass


def generate_broken_field(
    n_points: int = 100,
    total_mass: float = 1000.0,
    violation_fraction: float = 0.1,
    seed: int = 43
) -> Tuple[np.ndarray, float]:
    """
    Generate a synthetic 1D field that violates conservation.
    
    This simulates what might happen if a model produces an output
    that doesn't respect physical constraints.
    
    Args:
        n_points: Number of spatial points
        total_mass: Original conserved quantity
        violation_fraction: Fraction by which to violate conservation
        seed: Random seed for reproducibility
        
    Returns:
        field: 1D array that does NOT sum to total_mass
        target_sum: What the sum SHOULD be
    """
    rng = np.random.RandomState(seed)
    
    # Start with a conserved field
    field = rng.exponential(scale=1.0, size=n_points)
    field = field / field.sum() * total_mass
    
    # Add random noise to break conservation
    noise = rng.normal(0, violation_fraction * total_mass / n_points, size=n_points)
    field = field + noise
    
    # Ensure no negative values (physical constraint)
    field = np.maximum(field, 1e-6)
    
    return field, total_mass
