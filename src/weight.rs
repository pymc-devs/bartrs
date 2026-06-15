use std::ffi::c_double;

/// Safe trait for computing log-weights from predictions.
pub trait WeightFn {
    fn log_weight(&self, predictions: &[f64], indices: &[i32]) -> f64;
}

/// Weight function backed by a C function pointer from PyMC.
///
/// The unsafe FFI call is isolated behind this safe trait implementation.
pub struct PyMCWeightFn {
    func_ptr: unsafe extern "C" fn(*const f64, *const i32, usize) -> c_double,
}

impl PyMCWeightFn {
    /// Create a new PyMCWeightFn from a raw function pointer.
    ///
    /// # Safety
    /// The caller must ensure the function pointer remains valid for
    /// the lifetime of this struct and that it correctly interprets
    /// a (pointer, length) pair as a slice of f64 values.
    pub unsafe fn from_raw(ptr: unsafe extern "C" fn(*const f64, *const i32, usize) -> c_double) -> Self {
        Self { func_ptr: ptr }
    }
}

impl WeightFn for PyMCWeightFn {
    fn log_weight(&self, predictions: &[f64], indices: &[i32]) -> f64 {
        debug_assert_eq!(predictions.len(), indices.len());
        unsafe { (self.func_ptr)(predictions.as_ptr(), indices.as_ptr(), predictions.len()) }
    }
}
