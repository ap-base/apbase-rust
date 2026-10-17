//! Thin wrappers around the spatial filtering native kernels.

use std::ffi::c_int;

use ndarray::Array1;

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// `(pct_diff, local_z, local_prob, status)` returned by [`run_local_zstats`].
pub(crate) type LocalZStats = (Array1<f64>, Array1<f64>, Array1<f64>, i32);

/// Run local z-statistics for source points.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_local_zstats(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    radius: f64,
    cell_factor: i64,
    max_neighbors: i64,
    candidate_fraction: f64,
    dedup_radius: f64,
    dedup_z_tol: f64,
    n_threads: i64,
) -> Result<LocalZStats> {
    const CONTEXT: &str = "native spatial filter execution failed";
    require_len(y.len(), x.len(), "y", CONTEXT)?;
    require_len(z.len(), x.len(), "z", CONTEXT)?;
    let n = len_c_int(x.len(), CONTEXT)?;
    let cell_factor = to_c_int(cell_factor, CONTEXT)?;
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut pct_diff = vec![0.0f64; x.len()];
    let mut local_z = vec![0.0f64; x.len()];
    let mut local_prob = vec![0.0f64; x.len()];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::viz_zstats_radius_gridagg_(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            &radius,
            &cell_factor,
            &max_neighbors,
            &candidate_fraction,
            &dedup_radius,
            &dedup_z_tol,
            &thread_count,
            pct_diff.as_mut_ptr(),
            local_z.as_mut_ptr(),
            local_prob.as_mut_ptr(),
            &mut status,
            &n,
        );
    }
    Ok((
        Array1::from(pct_diff),
        Array1::from(local_z),
        Array1::from(local_prob),
        status,
    ))
}

/// Apply the native threshold mask to local statistics.
pub(crate) fn run_filter_mask(
    pct_diff: &[f64],
    local_prob: &[f64],
    local_z: &[f64],
    filter_level: i64,
) -> Result<(Array1<bool>, i32)> {
    const CONTEXT: &str = "native spatial filter mask failed";
    require_len(local_prob.len(), pct_diff.len(), "local_prob", CONTEXT)?;
    require_len(local_z.len(), pct_diff.len(), "local_z", CONTEXT)?;
    let n = len_c_int(pct_diff.len(), CONTEXT)?;
    let filter_level = to_c_int(filter_level, CONTEXT)?;
    let mut keep_mask = vec![0i8; pct_diff.len()];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::viz_filter_mask_(
            pct_diff.as_ptr(),
            local_prob.as_ptr(),
            local_z.as_ptr(),
            &filter_level,
            keep_mask.as_mut_ptr(),
            &mut status,
            &n,
        );
    }
    Ok((keep_mask.iter().map(|keep| *keep != 0).collect(), status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_native::test_support::{assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{ErrorKind, FILTER_STATUS_ERRORS};
    use crate::filtering::{DEFAULT_CELL_FACTOR, DEFAULT_MAX_NEIGHBORS};

    struct Settings {
        radius: f64,
        cell_factor: i64,
        max_neighbors: i64,
        candidate_fraction: f64,
        dedup_radius: f64,
        dedup_z_tol: f64,
        n_threads: i64,
    }

    impl Default for Settings {
        fn default() -> Self {
            Settings {
                radius: 15.0,
                cell_factor: DEFAULT_CELL_FACTOR,
                max_neighbors: DEFAULT_MAX_NEIGHBORS,
                candidate_fraction: 1.0,
                dedup_radius: 0.0,
                dedup_z_tol: 0.0,
                n_threads: 1,
            }
        }
    }

    fn zstats(x: &[f64], y: &[f64], z: &[f64], s: Settings) -> Result<LocalZStats> {
        run_local_zstats(
            x,
            y,
            z,
            s.radius,
            s.cell_factor,
            s.max_neighbors,
            s.candidate_fraction,
            s.dedup_radius,
            s.dedup_z_tol,
            s.n_threads,
        )
    }

    #[test]
    fn filter_kernels_flag_a_lone_outlier() {
        let (x, y, mut z) = grid_points();
        let outlier = 14;
        z[outlier] *= 3.0;
        let (pct_diff, local_z, local_prob, status) = zstats(&x, &y, &z, Settings::default()).unwrap();
        assert_eq!(status, 0);
        assert_eq!(
            (pct_diff.len(), local_z.len(), local_prob.len()),
            (z.len(), z.len(), z.len())
        );
        let largest = pct_diff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(pct_diff[outlier], largest);
        for level in 1..=4 {
            let (keep, status) = run_filter_mask(
                pct_diff.as_slice().unwrap(),
                local_prob.as_slice().unwrap(),
                local_z.as_slice().unwrap(),
                level,
            )
            .unwrap();
            assert_eq!(status, 0);
            assert!(!keep[outlier], "level {level}");
        }
    }

    #[test]
    fn filter_mask_applies_the_level_thresholds() {
        let pct_diff = [0.0, 50.0, 0.0, f64::NAN, 17.0, 19.0];
        let local_prob = [0.5, 0.5, 0.05, 0.5, 0.5, 0.5];
        let local_z = [0.0, 0.0, 0.0, 0.0, 0.3, 0.3];
        let (light, status) = run_filter_mask(&pct_diff, &local_prob, &local_z, 1).unwrap();
        assert_eq!(status, 0);
        assert_eq!(light.to_vec(), vec![true, false, false, false, true, true]);
        let (aggressive, _) = run_filter_mask(&pct_diff, &local_prob, &local_z, 4).unwrap();
        assert_eq!(aggressive.to_vec(), vec![false, false, false, false, true, false]);
    }

    #[test]
    fn filter_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        assert_rejected(zstats(&x, &y[1..], &z, Settings::default()).unwrap_err(), "y");
        assert_rejected(zstats(&x, &y, &z[1..], Settings::default()).unwrap_err(), "z");
        assert_rejected(
            run_filter_mask(&[0.0, 0.0], &[0.5], &[0.0, 0.0], 1).unwrap_err(),
            "local_prob",
        );
        assert_rejected(
            run_filter_mask(&[0.0, 0.0], &[0.5, 0.5], &[0.0], 1).unwrap_err(),
            "local_z",
        );
    }

    #[test]
    fn filter_statuses_reach_the_documented_errors() {
        let (x, y, z) = grid_points();
        let table = FILTER_STATUS_ERRORS;
        let status = |s: Settings| zstats(&x, &y, &z, s).unwrap().3;
        let data = ErrorKind::SpatialFilterInvalidDataError;
        let config = ErrorKind::SpatialFilterInvalidConfigurationError;
        assert_status(
            zstats(&[], &[], &[], Settings::default()).unwrap().3,
            1,
            table,
            data,
        );
        let all_nan = vec![f64::NAN; z.len()];
        assert_status(
            zstats(&x, &y, &all_nan, Settings::default()).unwrap().3,
            1,
            table,
            data,
        );
        let radius = status(Settings {
            radius: 0.0,
            ..Default::default()
        });
        assert_status(radius, 2, table, ErrorKind::SpatialFilterInvalidRadiusError);
        for settings in [
            Settings {
                cell_factor: 0,
                ..Default::default()
            },
            Settings {
                max_neighbors: -1,
                ..Default::default()
            },
            Settings {
                n_threads: 0,
                ..Default::default()
            },
            Settings {
                candidate_fraction: 0.0,
                ..Default::default()
            },
            Settings {
                candidate_fraction: 1.5,
                ..Default::default()
            },
            Settings {
                dedup_radius: -1.0,
                ..Default::default()
            },
            Settings {
                dedup_z_tol: f64::NAN,
                ..Default::default()
            },
        ] {
            assert_status(status(settings), 3, table, config);
        }
        assert_status(run_filter_mask(&[], &[], &[], 1).unwrap().1, 1, table, data);
        assert_status(
            run_filter_mask(&[0.0], &[0.5], &[0.0], 0).unwrap().1,
            3,
            table,
            config,
        );
        assert_status(
            run_filter_mask(&[0.0], &[0.5], &[0.0], 5).unwrap().1,
            3,
            table,
            config,
        );
    }
}
