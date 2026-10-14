//! Thin wrappers around the variogram native kernels.

use std::ffi::c_int;

use ndarray::Array1;

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// Fit candidate variogram models with the native implementation.
pub(crate) fn run_variogram(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    n_lags: i64,
    max_pairs: i64,
    max_distance: f64,
    n_threads: i64,
) -> Result<Array1<f64>> {
    const CONTEXT: &str = "error fitting variogram";
    require_len(y.len(), x.len(), "y", CONTEXT)?;
    require_len(z.len(), x.len(), "z", CONTEXT)?;
    let n = len_c_int(x.len(), CONTEXT)?;
    let n_lags = to_c_int(n_lags, CONTEXT)?;
    let max_pairs = to_c_int(max_pairs, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut model_info = [0.0f64; 5];

    lib::load();
    // SAFETY: x/y/z hold `n` values and model_info has the 5 slots the kernel writes.
    unsafe {
        lib::fit_variogram_models_(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            &n_lags,
            &max_pairs,
            &max_distance,
            &thread_count,
            model_info.as_mut_ptr(),
            &n,
        );
    }
    Ok(Array1::from(model_info.to_vec()))
}

/// Select the best variogram model from native model values.
pub(crate) fn run_select_variogram(model_info: &[f64]) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "error selecting variogram";
    require_len(model_info.len(), 5, "model_info", CONTEXT)?;
    let mut params = [0.0f64; 6];
    let mut status: c_int = 0;

    lib::load();
    // SAFETY: model_info holds 5 values and params has the 6 slots the kernel writes.
    unsafe {
        lib::select_variogram_(model_info.as_ptr(), params.as_mut_ptr(), &mut status);
    }
    Ok((Array1::from(params.to_vec()), status))
}

/// Evaluate a fitted variogram model at the flattened `distance` values.
pub(crate) fn run_evaluate_variogram(
    model_info: &[f64],
    distance: &[f64],
    n_threads: i64,
) -> Result<(Vec<f64>, i32)> {
    const CONTEXT: &str = "error evaluating variogram";
    require_len(model_info.len(), 5, "model_info", CONTEXT)?;
    let n = len_c_int(distance.len(), CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut gamma = vec![0.0f64; distance.len()];
    let mut status: c_int = 0;

    lib::load();
    // SAFETY: distance and gamma hold `n` values, model_info holds 5.
    unsafe {
        lib::evaluate_variogram_(
            model_info.as_ptr(),
            distance.as_ptr(),
            &thread_count,
            gamma.as_mut_ptr(),
            &mut status,
            &n,
        );
    }
    Ok((gamma, status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_native::test_support::{SPHERICAL, assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{ErrorKind, SELECT_STATUS_ERRORS};

    #[test]
    fn variogram_fit_returns_a_valid_model() {
        let (x, y, z) = grid_points();
        let model = run_variogram(&x, &y, &z, 10, 0, 0.0, 1).unwrap();
        assert!([1.0, 2.0, 3.0].contains(&model[0]), "{model}");
        assert!(model[1] >= 0.0 && model[2] >= 0.0, "{model}");
        assert!(model[3].is_finite() && model[3] > 0.0, "{model}");
        assert!(model[4].is_finite() && model[4] >= 0.0, "{model}");
    }

    #[test]
    fn variogram_select_unpacks_model_values() {
        let (params, status) = run_select_variogram(&[1.0, 0.1, 1.0, 40.0, 0.5]).unwrap();
        assert_eq!(status, 0);
        assert_eq!(params.to_vec(), vec![1.0, 0.1, 1.0, 0.1 + 1.0, 40.0, 0.5]);
    }

    #[test]
    fn variogram_evaluate_reaches_the_sill_at_the_range() {
        let (gamma, status) = run_evaluate_variogram(&SPHERICAL, &[20.0, 40.0, 1000.0], 1).unwrap();
        assert_eq!(status, 0);
        let sill = SPHERICAL[1] + SPHERICAL[2];
        assert!(gamma[0] > SPHERICAL[1] && gamma[0] < sill, "{gamma:?}");
        assert!(
            (gamma[1] - sill).abs() <= 1e-12 && (gamma[2] - sill).abs() <= 1e-12,
            "{gamma:?}"
        );
    }

    #[test]
    fn variogram_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        assert_rejected(run_variogram(&x, &y, &z[1..], 10, 0, 0.0, 1).unwrap_err(), "z");
        assert_rejected(run_variogram(&x, &y[1..], &z, 10, 0, 0.0, 1).unwrap_err(), "y");
        assert_rejected(run_select_variogram(&SPHERICAL[..4]).unwrap_err(), "model_info");
        assert_rejected(
            run_evaluate_variogram(&[0.0; 6], &[1.0], 1).unwrap_err(),
            "model_info",
        );
    }

    #[test]
    fn variogram_statuses_reach_the_documented_errors() {
        let table = SELECT_STATUS_ERRORS;
        let cases = [
            (
                [f64::NAN, 0.1, 1.0, 40.0, 0.0],
                1,
                ErrorKind::VariogramInvalidModelIdError,
            ),
            (
                [7.0, 0.1, 1.0, 40.0, 0.0],
                2,
                ErrorKind::VariogramInvalidModelError,
            ),
            (
                [1.0, 0.1, 1.0, 0.0, 0.0],
                3,
                ErrorKind::VariogramInvalidRangeError,
            ),
            (
                [1.0, 0.1, 1.0, f64::NAN, 0.0],
                3,
                ErrorKind::VariogramInvalidRangeError,
            ),
        ];
        for (model, expected, kind) in cases {
            assert_status(run_select_variogram(&model).unwrap().1, expected, table, kind);
            assert_status(
                run_evaluate_variogram(&model, &[1.0], 1).unwrap().1,
                expected,
                table,
                kind,
            );
        }
        let threads = run_evaluate_variogram(&SPHERICAL, &[1.0], 0).unwrap().1;
        assert_status(threads, 4, table, ErrorKind::VariogramInvalidThreadCountError);
    }
}
