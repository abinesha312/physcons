"""
Conservation layer that wraps the Rust kernel.

Implements the hard conservation constraint from PI-HC-MoE and
the residual-based correction from PI-MFM.
"""

import numpy as np
from typing import Tuple, Optional


class ConservationLayer:
    """
    Applies hard conservation constraints to model outputs.
    
    After the router processes the field, this layer:
    1. Computes the conservation residual (via Rust kernel)
    2. If residual exceeds threshold, applies hard correction
    3. Returns corrected field and diagnostic info
    """
    
    def __init__(self, threshold: float = 1e-3):
        """
        Args:
            threshold: Maximum allowed relative residual before correction
        """
        self.threshold = threshold
        self._rust_available = False
        
        # Try to import Rust extension with fail-safe
        try:
            import physcons_rust
            self._rust = physcons_rust
            self._rust_available = True
        except ImportError as e:
            self._import_error = str(e)
            # Fallback to pure Python implementation
            pass
    
    def _compute_residual_python(
        self,
        field: np.ndarray,
        target_sum: float
    ) -> float:
        """Pure Python fallback for residual computation."""
        if len(field) == 0:
            raise ValueError("Field cannot be empty")
        if not np.all(np.isfinite(field)):
            raise ValueError("Field contains NaN or Inf")
        
        current_sum = field.sum()
        residual = abs(current_sum - target_sum)
        return float(residual)
    
    def _hard_correct_python(
        self,
        field: np.ndarray,
        target_sum: float
    ) -> np.ndarray:
        """Pure Python fallback for hard correction."""
        if len(field) == 0:
            raise ValueError("Field cannot be empty")
        if not np.all(np.isfinite(field)):
            raise ValueError("Field contains NaN or Inf")
        
        current_sum = field.sum()
        if abs(current_sum) < 1e-12:
            raise ValueError("Cannot correct: current sum too close to zero")
        
        scale = target_sum / current_sum
        corrected = field * scale
        
        if not np.all(np.isfinite(corrected)):
            raise ValueError("Correction produced NaN or Inf")
        
        return corrected
    
    def compute_residual(
        self,
        field: np.ndarray,
        target_sum: float
    ) -> float:
        """
        Compute conservation residual.
        
        Args:
            field: Current field values
            target_sum: Target conserved quantity
            
        Returns:
            Absolute residual: |sum(field) - target_sum|
        """
        field_list = field.tolist()
        
        if self._rust_available:
            try:
                return self._rust.compute_residual(field_list, target_sum)
            except Exception as e:
                # Fail-safe: fall back to Python on Rust error
                return self._compute_residual_python(field, target_sum)
        else:
            return self._compute_residual_python(field, target_sum)
    
    def apply_conservation(
        self,
        field: np.ndarray,
        target_sum: float
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply hard conservation constraint if needed.
        
        Args:
            field: Field values (possibly violating conservation)
            target_sum: Target conserved quantity
            
        Returns:
            corrected_field: Field after correction (if needed)
            diagnostics: Dict with residual info and correction status
        """
        # Compute residual
        residual = self.compute_residual(field, target_sum)
        relative_residual = residual / abs(target_sum) if target_sum != 0 else residual
        
        diagnostics = {
            "residual": residual,
            "relative_residual": relative_residual,
            "corrected": False,
            "rust_used": self._rust_available,
        }
        
        # Check if correction is needed
        if relative_residual > self.threshold:
            # Apply hard correction
            field_list = field.tolist()
            
            if self._rust_available:
                try:
                    corrected_list = self._rust.hard_correct(field_list, target_sum)
                    corrected_field = np.array(corrected_list)
                except Exception as e:
                    # Fail-safe: fall back to Python
                    corrected_field = self._hard_correct_python(field, target_sum)
            else:
                corrected_field = self._hard_correct_python(field, target_sum)
            
            diagnostics["corrected"] = True
            
            # Compute residual after correction
            residual_after = self.compute_residual(corrected_field, target_sum)
            diagnostics["residual_after"] = residual_after
            diagnostics["relative_residual_after"] = (
                residual_after / abs(target_sum) if target_sum != 0 else residual_after
            )
            
            return corrected_field, diagnostics
        else:
            # No correction needed
            return field, diagnostics
    
    @property
    def is_rust_available(self) -> bool:
        """Check if Rust extension is available."""
        return self._rust_available
    
    def get_import_error(self) -> Optional[str]:
        """Get import error message if Rust extension failed to load."""
        return getattr(self, '_import_error', None)
