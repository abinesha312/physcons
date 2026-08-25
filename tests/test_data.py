"""
Tests for synthetic data generation.
"""

import pytest
import numpy as np

from physcons.data import generate_conserved_field, generate_broken_field


def test_conserved_field_has_correct_sum():
    """Conserved field should sum to target."""
    field, target = generate_conserved_field(n_points=100, total_mass=1000.0)
    
    assert len(field) == 100
    assert np.allclose(field.sum(), target, rtol=1e-10)
    

def test_conserved_field_is_positive():
    """All values should be positive."""
    field, _ = generate_conserved_field(n_points=100, total_mass=1000.0)
    
    assert np.all(field > 0)


def test_conserved_field_is_reproducible():
    """Same seed should produce same field."""
    field1, _ = generate_conserved_field(seed=42)
    field2, _ = generate_conserved_field(seed=42)
    
    assert np.allclose(field1, field2)


def test_broken_field_violates_conservation():
    """Broken field should NOT sum to target."""
    field, target = generate_broken_field(
        n_points=100,
        total_mass=1000.0,
        violation_fraction=0.1
    )
    
    assert len(field) == 100
    # Should be different from target
    assert not np.allclose(field.sum(), target, rtol=1e-3)


def test_broken_field_is_positive():
    """All values should still be positive (physical constraint)."""
    field, _ = generate_broken_field(n_points=100, total_mass=1000.0)
    
    assert np.all(field > 0)


def test_different_seeds_produce_different_fields():
    """Different seeds should produce different fields."""
    field1, _ = generate_conserved_field(seed=1)
    field2, _ = generate_conserved_field(seed=2)
    
    assert not np.allclose(field1, field2)
