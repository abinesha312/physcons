"""
Tests for conservation layer and Rust FFI.
"""

import pytest
import numpy as np

from physcons.conservation import ConservationLayer


def test_conservation_layer_initialization():
    """Conservation layer should initialize."""
    layer = ConservationLayer(threshold=1e-3)
    
    assert layer.threshold == 1e-3


def test_compute_residual_zero_for_conserved():
    """Residual should be zero for perfectly conserved field."""
    layer = ConservationLayer()
    field = np.ones(100)
    target = 100.0
    
    residual = layer.compute_residual(field, target)
    
    assert residual < 1e-10


def test_compute_residual_nonzero_for_broken():
    """Residual should be nonzero for non-conserved field."""
    layer = ConservationLayer()
    field = np.ones(100)
    target = 200.0  # Field sums to 100, not 200
    
    residual = layer.compute_residual(field, target)
    
    assert residual > 0
    assert np.isclose(residual, 100.0)


def test_hard_correction_restores_conservation():
    """Hard correction should restore the target sum."""
    layer = ConservationLayer(threshold=1e-3)
    field = np.ones(100)  # Sums to 100
    target = 200.0
    
    corrected, diag = layer.apply_conservation(field, target)
    
    assert diag["corrected"]
    assert np.isclose(corrected.sum(), target, rtol=1e-10)


def test_no_correction_when_already_conserved():
    """Should not correct when field already conserves."""
    layer = ConservationLayer(threshold=1e-3)
    field = np.ones(100)
    target = 100.0
    
    corrected, diag = layer.apply_conservation(field, target)
    
    assert not diag["corrected"]
    assert np.allclose(corrected, field)


def test_diagnostics_contain_residual_info():
    """Diagnostics should contain residual information."""
    layer = ConservationLayer()
    field = np.ones(100)
    target = 200.0
    
    _, diag = layer.apply_conservation(field, target)
    
    assert "residual" in diag
    assert "relative_residual" in diag
    assert "corrected" in diag
    assert "rust_used" in diag


def test_empty_field_raises_error():
    """Empty field should raise ValueError."""
    layer = ConservationLayer()
    field = np.array([])
    target = 100.0
    
    with pytest.raises(ValueError, match="empty"):
        layer.compute_residual(field, target)


def test_nan_field_raises_error():
    """Field with NaN should raise ValueError."""
    layer = ConservationLayer()
    field = np.array([1.0, 2.0, np.nan, 4.0])
    target = 7.0
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        layer.compute_residual(field, target)


def test_inf_field_raises_error():
    """Field with Inf should raise ValueError."""
    layer = ConservationLayer()
    field = np.array([1.0, 2.0, np.inf, 4.0])
    target = 7.0
    
    with pytest.raises(ValueError, match="NaN or Inf"):
        layer.compute_residual(field, target)


def test_correction_with_near_zero_sum_raises_error():
    """Correction should fail safely when current sum is near zero."""
    layer = ConservationLayer(threshold=1e-10)
    field = np.array([1e-15, -1e-15, 1e-15])
    target = 100.0
    
    with pytest.raises(ValueError, match="too close to zero"):
        layer.apply_conservation(field, target)


def test_python_fallback_works():
    """Python fallback should work even if Rust is unavailable."""
    layer = ConservationLayer()
    
    # Force Python fallback by setting rust_available to False
    layer._rust_available = False
    
    field = np.ones(100)
    target = 200.0
    
    corrected, diag = layer.apply_conservation(field, target)
    
    assert not diag["rust_used"]
    assert diag["corrected"]
    assert np.isclose(corrected.sum(), target, rtol=1e-10)


def test_rust_extension_status():
    """Should report whether Rust extension is available."""
    layer = ConservationLayer()
    
    # Should have a boolean status
    status = layer.is_rust_available
    assert isinstance(status, bool)
    
    # If not available, should have error message
    if not status:
        error = layer.get_import_error()
        assert error is None or isinstance(error, str)
