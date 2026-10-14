//! Private module holding the compiled apbase native kernels.
//!
//! Not part of the public API. Kernel wrappers import the facade from here:
//! `use crate::_native::lib;`.

pub(crate) mod _lib;
mod _runtime;

pub(crate) use _lib as lib;

use std::ffi::c_int;

use crate::common::exceptions::{ApbaseError, Result};

/// Convert an integer argument to the Fortran default `integer` (C `int`).
///
/// Out-of-range values are a `NativeExecutionError` prefixed with `context`.
pub(crate) fn to_c_int(value: impl TryInto<c_int>, context: &str) -> Result<c_int> {
    value.try_into().map_err(|_| {
        ApbaseError::native_execution_error(format!("{context}: integer too large to convert to C int"))
    })
}

/// Length of a kernel array argument as a Fortran dimension.
pub(crate) fn len_c_int(len: usize, context: &str) -> Result<c_int> {
    to_c_int(len, context)
}

/// Fail before calling a kernel whose arrays do not share the expected length.
pub(crate) fn require_len(len: usize, expected: usize, argument: &str, context: &str) -> Result<()> {
    if len != expected {
        return Err(ApbaseError::native_execution_error(format!(
            "{context}: `{argument}` has {len} elements, expected {expected}"
        )));
    }
    Ok(())
}

/// Fixtures shared by the kernel wrapper tests in each `<package>/_native.rs`.
#[cfg(test)]
pub(crate) mod test_support {
    use crate::common::exceptions::{ApbaseError, ErrorKind, StatusErrors, raise_status_error};

    /// Spherical model values `[model_id, nugget, partial_sill, range, sse]`.
    pub(crate) const SPHERICAL: [f64; 5] = [1.0, 0.1, 1.0, 40.0, 0.0];

    /// 36 points on a 6 x 6 grid with spacing 10 and a smooth field.
    pub(crate) fn grid_points() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..6 {
            for j in 0..6 {
                let (xi, yj) = (10.0 * f64::from(i), 10.0 * f64::from(j));
                x.push(xi);
                y.push(yj);
                z.push(1.0 + 0.05 * xi + 0.02 * yj + 0.3 * (xi / 7.0).sin() * (yj / 9.0).cos());
            }
        }
        (x, y, z)
    }

    /// The kernel returned `expected`, and the table the public layer uses maps
    /// it to `kind`.
    pub(crate) fn assert_status(status: i32, expected: i32, table: StatusErrors, kind: ErrorKind) {
        assert_eq!(status, expected);
        assert_eq!(
            raise_status_error(status, table, "kernel failed").kind(),
            kind,
            "status {status}"
        );
    }

    /// A length mismatch is rejected before the kernel runs, naming the argument.
    pub(crate) fn assert_rejected(error: ApbaseError, argument: &str) {
        assert_eq!(error.kind(), ErrorKind::NativeExecutionError);
        assert!(
            error.message().contains(&format!("`{argument}`")),
            "{}",
            error.message()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::exceptions::ErrorKind;

    #[test]
    fn integer_arguments_must_fit_a_c_int() {
        assert_eq!(to_c_int(7_i64, "context").unwrap(), 7);
        let error = to_c_int(i64::from(c_int::MAX) + 1, "native IDW execution failed").unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NativeExecutionError);
        assert_eq!(
            error.message(),
            "native IDW execution failed: integer too large to convert to C int"
        );
        assert!(len_c_int(usize::MAX, "context").is_err());
    }
}
