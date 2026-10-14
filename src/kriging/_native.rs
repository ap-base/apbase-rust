//! Thin wrapper around the ordinary kriging native kernel.

use std::ffi::c_int;

use ndarray::Array1;

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// Run local ordinary kriging for the provided target coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_kriging(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    target_x: &[f64],
    target_y: &[f64],
    model_info: &[f64],
    radius: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "native kriging execution failed";
    require_len(y.len(), x.len(), "y", CONTEXT)?;
    require_len(z.len(), x.len(), "z", CONTEXT)?;
    require_len(target_y.len(), target_x.len(), "target_y", CONTEXT)?;
    require_len(model_info.len(), 5, "model_info", CONTEXT)?;
    let n = len_c_int(x.len(), CONTEXT)?;
    let ntargets = len_c_int(target_x.len(), CONTEXT)?;
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut estimates = vec![0.0f64; target_x.len()];
    let mut status: c_int = 0;

    lib::load();
    // SAFETY: source arrays hold `n` values, target and output arrays `ntargets`,
    // model_info holds 5.
    unsafe {
        lib::ordinary_kriging_local_(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            target_x.as_ptr(),
            target_y.as_ptr(),
            model_info.as_ptr(),
            &radius,
            &max_neighbors,
            &min_neighbors,
            &thread_count,
            estimates.as_mut_ptr(),
            &mut status,
            &n,
            &ntargets,
        );
    }
    Ok((Array1::from(estimates), status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_native::test_support::{SPHERICAL, assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{ErrorKind, KRIGING_STATUS_ERRORS};

    fn status_with(
        model: &[f64],
        radius: f64,
        max_neighbors: i64,
        min_neighbors: i64,
        n_threads: i64,
    ) -> i32 {
        let x = [0.0, 10.0, 20.0];
        run_kriging(
            &x,
            &x,
            &x,
            &[5.0],
            &[5.0],
            model,
            radius,
            max_neighbors,
            min_neighbors,
            n_threads,
        )
        .unwrap()
        .1
    }

    #[test]
    fn kriging_weights_sum_to_one_on_a_constant_field() {
        let (x, y, _) = grid_points();
        let z = vec![7.25; x.len()];
        let (estimates, status) = run_kriging(
            &x,
            &y,
            &z,
            &[3.0, 25.0, 47.0],
            &[4.0, 25.0, 12.0],
            &SPHERICAL,
            30.0,
            16,
            1,
            1,
        )
        .unwrap();
        assert_eq!(status, 0);
        for estimate in estimates {
            assert!((estimate - 7.25).abs() <= 1e-9, "{estimate}");
        }
    }

    #[test]
    fn kriging_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        let run = |z: &[f64], target_y: &[f64], model: &[f64]| {
            run_kriging(&x, &y, z, &[1.0], target_y, model, 30.0, 16, 1, 1).unwrap_err()
        };
        assert_rejected(run(&z[1..], &[1.0], &SPHERICAL), "z");
        assert_rejected(run(&z, &[1.0, 2.0], &SPHERICAL), "target_y");
        assert_rejected(run(&z, &[1.0], &SPHERICAL[..4]), "model_info");
    }

    #[test]
    fn kriging_statuses_reach_the_documented_errors() {
        let table = KRIGING_STATUS_ERRORS;
        let empty = run_kriging(&[], &[], &[], &[5.0], &[5.0], &SPHERICAL, 30.0, 16, 1, 1)
            .unwrap()
            .1;
        assert_status(empty, 1, table, ErrorKind::KrigingInvalidDataError);
        assert_status(
            status_with(&SPHERICAL, -1.0, 16, 1, 1),
            2,
            table,
            ErrorKind::KrigingInvalidRadiusError,
        );
        let config = ErrorKind::KrigingInvalidConfigurationError;
        assert_status(status_with(&SPHERICAL, 30.0, 2, 3, 1), 3, table, config);
        assert_status(status_with(&SPHERICAL, 30.0, 16, 1, 0), 3, table, config);
        for model in [
            [4.0, 0.1, 1.0, 40.0, 0.0],
            [1.0, 0.1, 1.0, 0.0, 0.0],
            [1.0, -0.1, 1.0, 40.0, 0.0],
            [1.0, 0.1, f64::NAN, 40.0, 0.0],
            [1.0, 0.0, 0.0, 40.0, 0.0],
        ] {
            assert_status(
                status_with(&model, 30.0, 16, 1, 1),
                6,
                table,
                ErrorKind::KrigingInvalidModelError,
            );
        }
    }
}
