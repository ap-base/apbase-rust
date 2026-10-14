//! Thin wrapper around the IDW native kernel.

use std::ffi::c_int;

use ndarray::Array1;

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// Run IDW for arbitrary target coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_idw(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    target_x: &[f64],
    target_y: &[f64],
    radius: f64,
    power: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "native IDW execution failed";
    require_len(y.len(), x.len(), "y", CONTEXT)?;
    require_len(z.len(), x.len(), "z", CONTEXT)?;
    require_len(target_y.len(), target_x.len(), "target_y", CONTEXT)?;
    let n = len_c_int(x.len(), CONTEXT)?;
    let ntargets = len_c_int(target_x.len(), CONTEXT)?;
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut estimates = vec![0.0f64; target_x.len()];
    let mut status: c_int = 0;

    lib::load();
    // SAFETY: source arrays hold `n` values, target and output arrays `ntargets`.
    unsafe {
        lib::idw_local_(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            target_x.as_ptr(),
            target_y.as_ptr(),
            &radius,
            &power,
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
    use crate::_native::test_support::{assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{ErrorKind, IDW_STATUS_ERRORS};

    fn status_with(
        x: &[f64],
        radius: f64,
        power: f64,
        max_neighbors: i64,
        min_neighbors: i64,
        n_threads: i64,
    ) -> i32 {
        run_idw(
            x,
            x,
            x,
            &[5.0],
            &[5.0],
            radius,
            power,
            max_neighbors,
            min_neighbors,
            n_threads,
        )
        .unwrap()
        .1
    }

    #[test]
    fn idw_is_exact_at_data_bounded_inside_and_uses_nearest_beyond_radius() {
        let (x, y, z) = grid_points();
        let targets_x = [x[7], 25.0, 1000.0];
        let targets_y = [y[7], 25.0, 1000.0];
        let (estimates, status) = run_idw(&x, &y, &z, &targets_x, &targets_y, 30.0, 2.0, 8, 1, 1).unwrap();
        assert_eq!(status, 0);
        assert_eq!(estimates[0].to_bits(), z[7].to_bits());
        let low = z.iter().copied().fold(f64::INFINITY, f64::min);
        let high = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(estimates[1] > low && estimates[1] < high, "{}", estimates[1]);

        // No source within the radius of (1000, 1000): the 8 nearest are used.
        let mut nearest: Vec<(f64, f64)> = x
            .iter()
            .zip(&y)
            .zip(&z)
            .map(|((&xi, &yi), &zi)| ((xi - 1000.0).powi(2) + (yi - 1000.0).powi(2), zi))
            .collect();
        nearest.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (numerator, denominator) = nearest[..8].iter().fold((0.0, 0.0), |(num, den), &(d2, zi)| {
            (num + zi / d2, den + 1.0 / d2)
        });
        let expected = numerator / denominator;
        assert!(
            (estimates[2] - expected).abs() <= 1e-12 * expected.abs(),
            "{} vs {expected}",
            estimates[2]
        );
    }

    #[test]
    fn idw_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        let run = |y: &[f64], z: &[f64], target_y: &[f64]| {
            run_idw(&x, y, z, &[1.0], target_y, 30.0, 2.0, 8, 1, 1).unwrap_err()
        };
        assert_rejected(run(&y[1..], &z, &[1.0]), "y");
        assert_rejected(run(&y, &z[1..], &[1.0]), "z");
        assert_rejected(run(&y, &z, &[]), "target_y");
    }

    #[test]
    fn idw_statuses_reach_the_documented_errors() {
        let x = [0.0, 10.0, 20.0];
        let table = IDW_STATUS_ERRORS;
        assert_status(
            status_with(&[], 30.0, 2.0, 8, 1, 1),
            1,
            table,
            ErrorKind::IDWInvalidDataError,
        );
        assert_status(
            status_with(&x, 0.0, 2.0, 8, 1, 1),
            2,
            table,
            ErrorKind::IDWInvalidRadiusError,
        );
        assert_status(
            status_with(&x, f64::NAN, 2.0, 8, 1, 1),
            2,
            table,
            ErrorKind::IDWInvalidRadiusError,
        );
        assert_status(
            status_with(&x, 30.0, 0.0, 8, 1, 1),
            3,
            table,
            ErrorKind::IDWInvalidPowerError,
        );
        assert_status(
            status_with(&x, 30.0, 2.0, 2, 3, 1),
            4,
            table,
            ErrorKind::IDWInvalidConfigurationError,
        );
        assert_status(
            status_with(&x, 30.0, 2.0, 8, 1, 0),
            4,
            table,
            ErrorKind::IDWInvalidConfigurationError,
        );
    }
}
