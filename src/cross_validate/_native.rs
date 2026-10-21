//! Thin wrappers around the leave-one-out cross-validation kernels.

use std::ffi::c_int;

use ndarray::{Array1, Array2, Array3};

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::cokriging::fortran_order;
use crate::common::exceptions::Result;

/// Per-method kernel outputs: predictions plus MAE, RMSE, R2 and the number of
/// finite predictions.
#[derive(Clone, Debug)]
pub(crate) struct NativeMethodResult {
    pub(crate) predicted: Array1<f64>,
    pub(crate) mae: f64,
    pub(crate) rmse: f64,
    pub(crate) r2: f64,
    pub(crate) n_valid: i64,
}

impl NativeMethodResult {
    fn new(predicted: Vec<f64>, mae: f64, rmse: f64, r2: f64, n_valid: c_int) -> Self {
        NativeMethodResult {
            predicted: Array1::from(predicted),
            mae,
            rmse,
            r2,
            n_valid: i64::from(n_valid),
        }
    }
}

/// Run IDW and kriging leave-one-out: `(idw, kriging, status)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_cross_validate(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    sample_idx: &[i32],
    radius: f64,
    power: f64,
    model_info: &[f64],
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(NativeMethodResult, NativeMethodResult, i32)> {
    const CONTEXT: &str = "native cross_validate execution failed";
    require_len(y.len(), x.len(), "y", CONTEXT)?;
    require_len(z.len(), x.len(), "z", CONTEXT)?;
    require_len(model_info.len(), 5, "model_info", CONTEXT)?;
    let n = len_c_int(x.len(), CONTEXT)?;
    let nsamples = len_c_int(sample_idx.len(), CONTEXT)?;
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut idw_pred = vec![0.0f64; sample_idx.len()];
    let mut krig_pred = vec![0.0f64; sample_idx.len()];
    let (mut idw_mae, mut idw_rmse, mut idw_r2, mut idw_n_valid) = (0.0f64, 0.0f64, 0.0f64, 0 as c_int);
    let (mut krig_mae, mut krig_rmse, mut krig_r2, mut krig_n_valid) = (0.0f64, 0.0f64, 0.0f64, 0 as c_int);
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::cross_validate_local_(
            x.as_ptr(),
            y.as_ptr(),
            z.as_ptr(),
            sample_idx.as_ptr(),
            &radius,
            &power,
            model_info.as_ptr(),
            &max_neighbors,
            &min_neighbors,
            &thread_count,
            idw_pred.as_mut_ptr(),
            krig_pred.as_mut_ptr(),
            &mut idw_mae,
            &mut idw_rmse,
            &mut idw_r2,
            &mut idw_n_valid,
            &mut krig_mae,
            &mut krig_rmse,
            &mut krig_r2,
            &mut krig_n_valid,
            &mut status,
            &n,
            &nsamples,
        );
    }
    Ok((
        NativeMethodResult::new(idw_pred, idw_mae, idw_rmse, idw_r2, idw_n_valid),
        NativeMethodResult::new(krig_pred, krig_mae, krig_rmse, krig_r2, krig_n_valid),
        status,
    ))
}

/// Inputs of the cokriging leave-one-out kernel shared by the three methods.
pub(crate) struct CokrigingCrossValidateInputs<'a> {
    pub(crate) x0: &'a [f64],
    pub(crate) y0: &'a [f64],
    pub(crate) z0: &'a [f64],
    pub(crate) sample_idx: &'a [i32],
    pub(crate) secondary_x: &'a [f64],
    pub(crate) secondary_y: &'a [f64],
    pub(crate) secondary_z: &'a [f64],
    pub(crate) secondary_offset: &'a [i32],
    pub(crate) run_collocated: bool,
    pub(crate) collocated_model: &'a [f64],
    pub(crate) run_icm: bool,
    pub(crate) icm_model_id: f64,
    pub(crate) icm_range: f64,
    pub(crate) icm_nugget: &'a Array2<f64>,
    pub(crate) icm_sill: &'a Array2<f64>,
    pub(crate) run_lmc: bool,
    pub(crate) lmc_model_id: f64,
    pub(crate) lmc_ranges: &'a [f64],
    pub(crate) lmc_coefficients: &'a Array3<f64>,
    pub(crate) radius: f64,
    pub(crate) max_neighbors: i64,
    pub(crate) max_secondary_neighbors: i64,
    pub(crate) min_neighbors: i64,
    pub(crate) n_threads: i64,
}

/// Run collocated/ICM/LMC leave-one-out: `([collocated, icm, lmc], status)`.
pub(crate) fn run_cokriging_cross_validate(
    inputs: &CokrigingCrossValidateInputs<'_>,
) -> Result<([NativeMethodResult; 3], i32)> {
    const CONTEXT: &str = "native cokriging cross-validation execution failed";
    require_len(inputs.y0.len(), inputs.x0.len(), "y0", CONTEXT)?;
    require_len(inputs.z0.len(), inputs.x0.len(), "z0", CONTEXT)?;
    require_len(
        inputs.secondary_y.len(),
        inputs.secondary_x.len(),
        "secondary_y",
        CONTEXT,
    )?;
    require_len(
        inputs.secondary_z.len(),
        inputs.secondary_x.len(),
        "secondary_z",
        CONTEXT,
    )?;
    let size = inputs.secondary_offset.len();
    require_len(inputs.icm_nugget.len(), size * size, "icm_nugget", CONTEXT)?;
    require_len(inputs.icm_sill.len(), size * size, "icm_sill", CONTEXT)?;
    let coefficient_shape = inputs.lmc_coefficients.shape();
    require_len(coefficient_shape[1], size, "lmc_coefficients", CONTEXT)?;
    require_len(coefficient_shape[2], size, "lmc_coefficients", CONTEXT)?;

    let n = len_c_int(inputs.x0.len(), CONTEXT)?;
    let nsamples = len_c_int(inputs.sample_idx.len(), CONTEXT)?;
    let n_secondary_total = len_c_int(inputs.secondary_x.len(), CONTEXT)?;
    let nv = len_c_int(size, CONTEXT)?;
    let collocated_model_size = len_c_int(inputs.collocated_model.len(), CONTEXT)?;
    let n_lmc_structures = len_c_int(coefficient_shape[0], CONTEXT)?;
    let n_lmc_continuous = len_c_int(inputs.lmc_ranges.len(), CONTEXT)?;
    let max_neighbors = to_c_int(inputs.max_neighbors, CONTEXT)?;
    let max_secondary_neighbors = to_c_int(inputs.max_secondary_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(inputs.min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(inputs.n_threads, CONTEXT)?;
    let icm_nugget = fortran_order(inputs.icm_nugget);
    let icm_sill = fortran_order(inputs.icm_sill);
    let lmc_coefficients = fortran_order(inputs.lmc_coefficients);
    let run_collocated: c_int = c_int::from(inputs.run_collocated);
    let run_icm: c_int = c_int::from(inputs.run_icm);
    let run_lmc: c_int = c_int::from(inputs.run_lmc);

    let samples = inputs.sample_idx.len();
    let mut predictions = [
        vec![0.0f64; samples],
        vec![0.0f64; samples],
        vec![0.0f64; samples],
    ];
    let mut metrics = [[0.0f64; 3]; 3];
    let mut n_valid: [c_int; 3] = [0; 3];
    let mut status: c_int = 0;
    let [collocated_pred, icm_pred, lmc_pred] = &mut predictions;
    let [collocated_metrics, icm_metrics, lmc_metrics] = &mut metrics;
    let [collocated_n_valid, icm_n_valid, lmc_n_valid] = &mut n_valid;

    lib::load();
    unsafe {
        lib::cokriging_cross_validate_local_(
            inputs.x0.as_ptr(),
            inputs.y0.as_ptr(),
            inputs.z0.as_ptr(),
            inputs.sample_idx.as_ptr(),
            inputs.secondary_x.as_ptr(),
            inputs.secondary_y.as_ptr(),
            inputs.secondary_z.as_ptr(),
            inputs.secondary_offset.as_ptr(),
            &run_collocated,
            inputs.collocated_model.as_ptr(),
            &run_icm,
            &inputs.icm_model_id,
            &inputs.icm_range,
            icm_nugget.as_ptr(),
            icm_sill.as_ptr(),
            &run_lmc,
            &inputs.lmc_model_id,
            inputs.lmc_ranges.as_ptr(),
            lmc_coefficients.as_ptr(),
            &inputs.radius,
            &max_neighbors,
            &max_secondary_neighbors,
            &min_neighbors,
            &thread_count,
            collocated_pred.as_mut_ptr(),
            icm_pred.as_mut_ptr(),
            lmc_pred.as_mut_ptr(),
            &mut collocated_metrics[0],
            &mut collocated_metrics[1],
            &mut collocated_metrics[2],
            collocated_n_valid,
            &mut icm_metrics[0],
            &mut icm_metrics[1],
            &mut icm_metrics[2],
            icm_n_valid,
            &mut lmc_metrics[0],
            &mut lmc_metrics[1],
            &mut lmc_metrics[2],
            lmc_n_valid,
            &mut status,
            &n,
            &nsamples,
            &n_secondary_total,
            &nv,
            &collocated_model_size,
            &n_lmc_structures,
            &n_lmc_continuous,
        );
    }
    let [collocated_pred, icm_pred, lmc_pred] = predictions;
    let results = [
        NativeMethodResult::new(
            collocated_pred,
            metrics[0][0],
            metrics[0][1],
            metrics[0][2],
            n_valid[0],
        ),
        NativeMethodResult::new(icm_pred, metrics[1][0], metrics[1][1], metrics[1][2], n_valid[1]),
        NativeMethodResult::new(lmc_pred, metrics[2][0], metrics[2][1], metrics[2][2], n_valid[2]),
    ];
    Ok((results, status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_native::test_support::{SPHERICAL, assert_rejected, assert_status, grid_points};
    use crate::common::exceptions::{COKRIGING_STATUS_ERRORS, CROSS_VALIDATE_STATUS_ERRORS, ErrorKind};
    use ndarray::{Axis, array, stack};

    fn all_samples(n: usize) -> Vec<i32> {
        (1..=i32::try_from(n).unwrap()).collect()
    }

    /// The kernel's MAE, RMSE and R2 are the ones of the predictions it returned,
    /// which also proves the wrapper passes the scalar outputs in the right order.
    fn assert_metrics(result: &NativeMethodResult, observed: &[f64]) {
        assert_eq!(result.predicted.len(), observed.len());
        assert!(
            result.predicted.iter().all(|v| v.is_finite()),
            "{}",
            result.predicted
        );
        assert_eq!(result.n_valid, observed.len() as i64);
        let n = observed.len() as f64;
        let errors: Vec<f64> = result
            .predicted
            .iter()
            .zip(observed)
            .map(|(p, o)| p - o)
            .collect();
        let ss_res: f64 = errors.iter().map(|e| e * e).sum();
        let mean = observed.iter().sum::<f64>() / n;
        let ss_tot: f64 = observed.iter().map(|o| (o - mean).powi(2)).sum();
        let expected = [
            errors.iter().map(|e| e.abs()).sum::<f64>() / n,
            (ss_res / n).sqrt(),
            1.0 - ss_res / ss_tot,
        ];
        for (kernel, expected) in [result.mae, result.rmse, result.r2].into_iter().zip(expected) {
            assert!(
                (kernel - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                "{kernel} vs {expected}"
            );
        }
    }

    #[test]
    fn cross_validate_metrics_match_the_returned_predictions() {
        let (x, y, z) = grid_points();
        let samples = all_samples(x.len());
        let (idw, kriging, status) =
            run_cross_validate(&x, &y, &z, &samples, 30.0, 2.0, &SPHERICAL, 16, 1, 1).unwrap();
        assert_eq!(status, 0);
        assert_metrics(&idw, &z);
        assert_metrics(&kriging, &z);
        assert_ne!(idw.predicted, kriging.predicted);
    }

    #[test]
    fn cross_validate_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z) = grid_points();
        let samples = all_samples(x.len());
        let run = |y: &[f64], z: &[f64], model: &[f64]| {
            run_cross_validate(&x, y, z, &samples, 30.0, 2.0, model, 16, 1, 1).unwrap_err()
        };
        assert_rejected(run(&y[1..], &z, &SPHERICAL), "y");
        assert_rejected(run(&y, &z[1..], &SPHERICAL), "z");
        assert_rejected(run(&y, &z, &SPHERICAL[..4]), "model_info");
    }

    #[test]
    fn cross_validate_statuses_reach_the_documented_errors() {
        let (x, y, z) = grid_points();
        let samples = all_samples(x.len());
        let table = CROSS_VALIDATE_STATUS_ERRORS;
        let status = |sample_idx: &[i32], radius: f64, power: f64, model: &[f64], max_neighbors: i64| {
            run_cross_validate(&x, &y, &z, sample_idx, radius, power, model, max_neighbors, 1, 1)
                .unwrap()
                .2
        };
        let data = ErrorKind::CrossValidationInvalidDataError;
        assert_status(status(&[], 30.0, 2.0, &SPHERICAL, 16), 1, table, data);
        let index = ErrorKind::CrossValidationInvalidSampleIndexError;
        assert_status(status(&[0], 30.0, 2.0, &SPHERICAL, 16), 7, table, index);
        assert_status(status(&[37], 30.0, 2.0, &SPHERICAL, 16), 7, table, index);
        let radius = ErrorKind::CrossValidationInvalidRadiusError;
        assert_status(status(&samples, 0.0, 2.0, &SPHERICAL, 16), 2, table, radius);
        let power = ErrorKind::CrossValidationInvalidPowerError;
        assert_status(status(&samples, 30.0, -1.0, &SPHERICAL, 16), 3, table, power);
        let config = ErrorKind::CrossValidationInvalidConfigurationError;
        assert_status(status(&samples, 30.0, 2.0, &SPHERICAL, 0), 4, table, config);
        let model = ErrorKind::CrossValidationInvalidModelError;
        assert_status(
            status(&samples, 30.0, 2.0, &[0.0, 0.1, 1.0, 40.0, 0.0], 16),
            6,
            table,
            model,
        );
    }

    #[test]
    fn cokriging_cross_validate_wrapper_checks() {
        let (x, y, z) = grid_points();
        let secondary: Vec<f64> = z
            .iter()
            .zip(&x)
            .map(|(v, xi)| 0.8 * v + 0.1 * (xi / 13.0).cos())
            .collect();
        let samples = all_samples(x.len());
        let nugget = array![[0.1, 0.0], [0.0, 0.1]];
        let sill = array![[1.0, 0.5], [0.5, 1.0]];
        let coefficients = stack(Axis(0), &[nugget.view(), sill.view()]).unwrap();
        let inputs = CokrigingCrossValidateInputs {
            x0: &x,
            y0: &y,
            z0: &z,
            sample_idx: &samples,
            secondary_x: &x,
            secondary_y: &y,
            secondary_z: &secondary,
            secondary_offset: &[0, 36],
            run_collocated: true,
            collocated_model: &[1.0, 0.1, 1.0, 40.0, 0.0, 0.7],
            run_icm: true,
            icm_model_id: 1.0,
            icm_range: 40.0,
            icm_nugget: &nugget,
            icm_sill: &sill,
            run_lmc: true,
            lmc_model_id: 1.0,
            lmc_ranges: &[40.0],
            lmc_coefficients: &coefficients,
            radius: 30.0,
            max_neighbors: 16,
            max_secondary_neighbors: 16,
            min_neighbors: 1,
            n_threads: 1,
        };

        let (results, status) = run_cokriging_cross_validate(&inputs).unwrap();
        assert_eq!(status, 0);
        for result in &results {
            assert_metrics(result, &z);
        }

        let table = COKRIGING_STATUS_ERRORS;
        let status =
            |inputs: CokrigingCrossValidateInputs<'_>| run_cokriging_cross_validate(&inputs).unwrap().1;
        let data = ErrorKind::CoKrigingInvalidDataError;
        assert_status(
            status(CokrigingCrossValidateInputs {
                sample_idx: &[],
                ..inputs
            }),
            1,
            table,
            data,
        );
        assert_status(
            status(CokrigingCrossValidateInputs {
                secondary_offset: &[0, 30],
                ..inputs
            }),
            1,
            table,
            data,
        );
        let short_model = CokrigingCrossValidateInputs {
            collocated_model: &[1.0, 0.1, 1.0, 40.0, 0.0],
            ..inputs
        };
        assert_status(status(short_model), 1, table, data);
        let radius = ErrorKind::CoKrigingInvalidRadiusError;
        assert_status(
            status(CokrigingCrossValidateInputs {
                radius: 0.0,
                ..inputs
            }),
            2,
            table,
            radius,
        );
        let config = ErrorKind::CoKrigingInvalidConfigurationError;
        assert_status(
            status(CokrigingCrossValidateInputs {
                max_neighbors: 0,
                ..inputs
            }),
            3,
            table,
            config,
        );
        let model = ErrorKind::CoKrigingInvalidModelError;
        let rho = CokrigingCrossValidateInputs {
            collocated_model: &[1.0, 0.1, 1.0, 40.0, 0.0, 1.5],
            ..inputs
        };
        assert_status(status(rho), 6, table, model);
        assert_status(
            status(CokrigingCrossValidateInputs {
                icm_range: 0.0,
                ..inputs
            }),
            6,
            table,
            model,
        );
        assert_status(
            status(CokrigingCrossValidateInputs {
                lmc_ranges: &[40.0, 80.0],
                ..inputs
            }),
            6,
            table,
            model,
        );
        assert_eq!(
            status(CokrigingCrossValidateInputs {
                sample_idx: &[0],
                ..inputs
            }),
            7
        );

        let rejected =
            |inputs: CokrigingCrossValidateInputs<'_>| run_cokriging_cross_validate(&inputs).unwrap_err();
        assert_rejected(
            rejected(CokrigingCrossValidateInputs {
                z0: &z[1..],
                ..inputs
            }),
            "z0",
        );
        assert_rejected(
            rejected(CokrigingCrossValidateInputs {
                secondary_z: &secondary[1..],
                ..inputs
            }),
            "secondary_z",
        );
        let wide = Array2::zeros((3, 3));
        assert_rejected(
            rejected(CokrigingCrossValidateInputs {
                icm_nugget: &wide,
                ..inputs
            }),
            "icm_nugget",
        );
        let cube = Array3::zeros((2, 3, 3));
        assert_rejected(
            rejected(CokrigingCrossValidateInputs {
                lmc_coefficients: &cube,
                ..inputs
            }),
            "lmc_coefficients",
        );
    }
}
