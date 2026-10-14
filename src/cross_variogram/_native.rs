//! Thin wrappers around the cross-variogram and covariate-screening kernels.

use std::ffi::c_int;

use ndarray::Array1;

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// Fit a cross-variogram model: `(model_values, rho, n_pairs)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fit_cross_variogram(
    xa: &[f64],
    ya: &[f64],
    za: &[f64],
    xb: &[f64],
    yb: &[f64],
    zb: &[f64],
    max_colocation_dist: f64,
    n_lags: i64,
    max_pairs: i64,
    max_distance: f64,
    n_threads: i64,
) -> Result<(Array1<f64>, f64, i64)> {
    const CONTEXT: &str = "error fitting cross-variogram";
    require_len(ya.len(), xa.len(), "ya", CONTEXT)?;
    require_len(za.len(), xa.len(), "za", CONTEXT)?;
    require_len(yb.len(), xb.len(), "yb", CONTEXT)?;
    require_len(zb.len(), xb.len(), "zb", CONTEXT)?;
    let na = len_c_int(xa.len(), CONTEXT)?;
    let nb = len_c_int(xb.len(), CONTEXT)?;
    let n_lags = to_c_int(n_lags, CONTEXT)?;
    let max_pairs = to_c_int(max_pairs, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut model_info = [0.0f64; 5];
    let mut rho = 0.0f64;
    let mut n_pairs: c_int = 0;

    lib::load();
    // SAFETY: side A arrays hold `na` values, side B arrays `nb`, model_info 5.
    unsafe {
        lib::fit_cross_variogram_(
            xa.as_ptr(),
            ya.as_ptr(),
            za.as_ptr(),
            xb.as_ptr(),
            yb.as_ptr(),
            zb.as_ptr(),
            &max_colocation_dist,
            &n_lags,
            &max_pairs,
            &max_distance,
            &thread_count,
            model_info.as_mut_ptr(),
            &mut rho,
            &mut n_pairs,
            &na,
            &nb,
        );
    }
    Ok((Array1::from(model_info.to_vec()), rho, i64::from(n_pairs)))
}

/// Screen one covariate: `(decision, p_value, admissible_fraction, status)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_screen_covariate(
    model_00: &[f64],
    model_uu: &[f64],
    model_0u: &[f64],
    rho: f64,
    n_pairs: i64,
    alpha: f64,
    min_abs_rho: f64,
    max_violation_fraction: f64,
) -> Result<(bool, f64, f64, i32)> {
    const CONTEXT: &str = "error screening covariate";
    require_len(model_00.len(), 5, "model_00", CONTEXT)?;
    require_len(model_uu.len(), 5, "model_uu", CONTEXT)?;
    require_len(model_0u.len(), 5, "model_0u", CONTEXT)?;
    let n_pairs = to_c_int(n_pairs, CONTEXT)?;
    let mut decision: c_int = 0;
    let mut p_value = 0.0f64;
    let mut admissible_fraction = 0.0f64;
    let mut status: c_int = 0;

    lib::load();
    // SAFETY: the three model vectors hold 5 values each; outputs are scalars.
    unsafe {
        lib::screen_covariate_(
            model_00.as_ptr(),
            model_uu.as_ptr(),
            model_0u.as_ptr(),
            &rho,
            &n_pairs,
            &alpha,
            &min_abs_rho,
            &max_violation_fraction,
            &mut decision,
            &mut p_value,
            &mut admissible_fraction,
            &mut status,
        );
    }
    Ok((decision != 0, p_value, admissible_fraction, status))
}

#[cfg(test)]
mod tests {
    //! Checks of the cross-variogram and covariate-screening kernel wrappers.

    use super::*;
    use crate::_native::test_support::{assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{CROSS_VARIOGRAM_STATUS_ERRORS, ErrorKind};

    const MODEL_SPHERICAL: f64 = 1.0;

    fn fisher_p_value_reference(rho: f64, n_pairs: i64) -> f64 {
        let z_fisher = 0.5 * ((1.0 + rho) / (1.0 - rho)).ln();
        let z_stat = z_fisher * ((n_pairs - 3) as f64).sqrt();
        // SAFETY: erf is a pure libm function.
        let normal_cdf = 0.5 * (1.0 + unsafe { lib::erf(z_stat.abs() / 2f64.sqrt()) });
        2.0 * (1.0 - normal_cdf)
    }

    fn models(cross_partial_sill: f64) -> ([f64; 5], [f64; 5], [f64; 5]) {
        (
            [MODEL_SPHERICAL, 0.0, 1.0, 10.0, 0.0],
            [MODEL_SPHERICAL, 0.0, 1.0, 10.0, 0.0],
            [MODEL_SPHERICAL, 0.0, cross_partial_sill, 10.0, 0.0],
        )
    }

    #[test]
    fn test_high_rho_small_n_is_rejected_by_significance() {
        let (model_00, model_uu, model_0u) = models(0.8);
        let (rho, n_pairs) = (0.8, 6);
        let (decision, p_value, admissible_fraction, status) =
            run_screen_covariate(&model_00, &model_uu, &model_0u, rho, n_pairs, 0.05, 0.3, 0.1).unwrap();
        let reference_p = fisher_p_value_reference(rho, n_pairs);
        assert_eq!(status, 0);
        assert!((p_value - reference_p).abs() <= 1e-9);
        assert!(reference_p > 0.05);
        assert!(!decision);
        assert_eq!(admissible_fraction, 1.0);
    }

    #[test]
    fn test_cauchy_schwarz_violation_is_rejected_even_when_significant() {
        let (model_00, model_uu, model_0u) = models(5.0);
        let (decision, p_value, admissible_fraction, status) =
            run_screen_covariate(&model_00, &model_uu, &model_0u, 0.99, 100, 0.05, 0.3, 0.1).unwrap();
        assert_eq!(status, 0);
        assert!(p_value < 0.05);
        assert!(admissible_fraction < 0.9);
        assert!(!decision);
    }

    #[test]
    fn test_insufficient_sample_reports_status_without_raising() {
        let (model_00, model_uu, model_0u) = models(0.8);
        let (decision, _, _, status) =
            run_screen_covariate(&model_00, &model_uu, &model_0u, 0.9, 3, 0.05, 0.3, 0.1).unwrap();
        assert_eq!(status, 1);
        assert!(!decision);
    }

    #[test]
    fn wrong_model_length_is_rejected_before_the_kernel() {
        let short = [1.0, 0.0, 1.0];
        let model = [MODEL_SPHERICAL, 0.0, 1.0, 10.0, 0.0];
        let error = run_screen_covariate(&short, &model, &model, 0.5, 10, 0.05, 0.3, 0.1).unwrap_err();
        assert_eq!(error.kind(), crate::common::ErrorKind::NativeExecutionError);
    }

    #[test]
    fn fit_cross_variogram_on_perfectly_correlated_colocated_samples() {
        let (x, y, za) = grid_points();
        let zb: Vec<f64> = za.iter().map(|v| 2.0 * v + 1.0).collect();
        let (model, rho, n_pairs) =
            run_fit_cross_variogram(&x, &y, &za, &x, &y, &zb, 0.0, 10, 0, 0.0, 1).unwrap();
        assert_eq!(n_pairs, 36);
        assert!((rho - 1.0).abs() <= 1e-12, "{rho}");
        assert!([1.0, 2.0, 3.0].contains(&model[0]), "{model}");
        assert!(model[3].is_finite() && model[3] > 0.0, "{model}");
    }

    #[test]
    fn fit_cross_variogram_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        let run =
            |ya: &[f64], zb: &[f64]| run_fit_cross_variogram(&x, ya, &z, &x, &y, zb, 0.0, 10, 0, 0.0, 1);
        assert_rejected(run(&y[1..], &z).unwrap_err(), "ya");
        assert_rejected(run(&y, &z[1..]).unwrap_err(), "zb");
    }

    #[test]
    fn screening_statuses_reach_the_documented_errors() {
        let table = CROSS_VARIOGRAM_STATUS_ERRORS;
        let (model_00, model_uu, model_0u) = models(0.8);
        let status = |m00: &[f64], muu: &[f64], m0u: &[f64], rho: f64, n_pairs: i64| {
            run_screen_covariate(m00, muu, m0u, rho, n_pairs, 0.05, 0.3, 0.1)
                .unwrap()
                .3
        };
        let sample = ErrorKind::CrossVariogramInsufficientSampleError;
        assert_status(status(&model_00, &model_uu, &model_0u, 0.9, 3), 1, table, sample);
        assert_status(
            status(&model_00, &model_uu, &model_0u, f64::NAN, 50),
            1,
            table,
            sample,
        );
        let bad_auto = [9.0, 0.0, 1.0, 10.0, 0.0];
        let auto = ErrorKind::CrossVariogramInvalidAutoModelError;
        assert_status(status(&bad_auto, &model_uu, &model_0u, 0.9, 50), 2, table, auto);
        assert_status(status(&model_00, &bad_auto, &model_0u, 0.9, 50), 2, table, auto);
        let cross = ErrorKind::CrossVariogramInvalidCrossModelError;
        assert_status(
            status(&model_00, &model_uu, &[0.0, 0.0, 0.8, 10.0, 0.0], 0.9, 50),
            3,
            table,
            cross,
        );
        assert_status(
            status(
                &model_00,
                &model_uu,
                &[MODEL_SPHERICAL, 0.0, 0.8, 0.0, 0.0],
                0.9,
                50,
            ),
            3,
            table,
            cross,
        );
    }
}
