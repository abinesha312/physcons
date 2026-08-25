use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

/// Compute the global residual of a conserved scalar field.
/// 
/// For a conserved quantity, sum(field) should equal sum(initial_field).
/// This returns the absolute residual: |sum(field) - target_sum|
#[pyfunction]
fn compute_residual(field: Vec<f64>, target_sum: f64) -> PyResult<f64> {
    // Fail-safe: check for empty input
    if field.is_empty() {
        return Err(PyValueError::new_err("Field cannot be empty"));
    }
    
    // Fail-safe: check for NaN/Inf
    for &val in &field {
        if !val.is_finite() {
            return Err(PyValueError::new_err("Field contains NaN or Inf"));
        }
    }
    
    let current_sum: f64 = field.iter().sum();
    
    // Fail-safe: check result
    if !current_sum.is_finite() {
        return Err(PyValueError::new_err("Sum computation resulted in NaN or Inf"));
    }
    
    let residual = (current_sum - target_sum).abs();
    Ok(residual)
}

/// Hard conservation correction: project the field back to conserve the target sum.
/// 
/// This implements the hard constraint from PI-HC-MoE: if the residual exceeds
/// a threshold, we uniformly scale the field to restore conservation.
/// 
/// Returns the corrected field.
#[pyfunction]
fn hard_correct(field: Vec<f64>, target_sum: f64) -> PyResult<Vec<f64>> {
    // Fail-safe: check for empty input
    if field.is_empty() {
        return Err(PyValueError::new_err("Field cannot be empty"));
    }
    
    // Fail-safe: check for NaN/Inf
    for &val in &field {
        if !val.is_finite() {
            return Err(PyValueError::new_err("Field contains NaN or Inf"));
        }
    }
    
    let current_sum: f64 = field.iter().sum();
    
    // Fail-safe: avoid division by zero
    if current_sum.abs() < 1e-12 {
        return Err(PyValueError::new_err(
            "Cannot correct: current sum too close to zero"
        ));
    }
    
    // Scale factor to restore conservation
    let scale = target_sum / current_sum;
    
    // Fail-safe: check scale factor
    if !scale.is_finite() {
        return Err(PyValueError::new_err("Scale factor is NaN or Inf"));
    }
    
    // Apply uniform scaling
    let corrected: Vec<f64> = field.iter().map(|&x| x * scale).collect();
    
    // Fail-safe: verify corrected values
    for &val in &corrected {
        if !val.is_finite() {
            return Err(PyValueError::new_err(
                "Correction produced NaN or Inf"
            ));
        }
    }
    
    Ok(corrected)
}

#[pymodule]
fn physcons_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_residual, m)?)?;
    m.add_function(wrap_pyfunction!(hard_correct, m)?)?;
    Ok(())
}
