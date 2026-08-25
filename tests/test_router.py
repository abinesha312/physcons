"""
Tests for patch router.
"""

import pytest
import numpy as np

from physcons.router import PatchRouter, SimpleExpert


def test_simple_expert_forward():
    """Expert should transform input."""
    expert = SimpleExpert(input_dim=10, seed=42)
    x = np.ones(10)
    y = expert.forward(x)
    
    assert y.shape == (10,)
    # Should be different from input (has weights and bias)
    assert not np.allclose(y, x)


def test_router_initialization():
    """Router should initialize with correct parameters."""
    router = PatchRouter(patch_size=10, n_experts=3, seed=42)
    
    assert router.patch_size == 10
    assert router.n_experts == 3
    assert len(router.experts) == 3


def test_router_output_shape():
    """Router output should have same shape as input."""
    router = PatchRouter(patch_size=10, n_experts=3)
    field = np.ones(100)
    
    output = router.forward(field)
    
    assert output.shape == field.shape


def test_router_handles_non_divisible_length():
    """Router should handle fields not divisible by patch_size."""
    router = PatchRouter(patch_size=10, n_experts=3)
    field = np.ones(95)  # Not divisible by 10
    
    output = router.forward(field)
    
    assert output.shape == field.shape


def test_router_is_deterministic():
    """Same input should produce same output."""
    router = PatchRouter(patch_size=10, n_experts=3, seed=42)
    field = np.random.randn(100)
    
    output1 = router.forward(field)
    output2 = router.forward(field)
    
    assert np.allclose(output1, output2)


def test_router_routes_different_patches_differently():
    """Different patches should potentially route to different experts."""
    router = PatchRouter(patch_size=10, n_experts=3)
    
    # Create field with very different patch means
    field = np.concatenate([
        np.ones(10) * 1.0,
        np.ones(10) * 100.0,
        np.ones(10) * 10000.0,
    ])
    
    output = router.forward(field)
    
    # Output should exist and have correct shape
    assert output.shape == field.shape
