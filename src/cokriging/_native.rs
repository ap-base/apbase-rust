//! Thin wrappers around the cokriging native kernels.
//!
//! Matrices cross this boundary in Fortran (column-major) order and come back
//! as `ndarray` arrays with logical indexing (`[i, j]`, `[s, i, j]`).

use std::ffi::c_int;

use ndarray::{Array1, Array2, Array3, ArrayBase, Data, Dimension, ShapeBuilder};

use crate::_native::{len_c_int, lib, require_len, to_c_int};
use crate::common::exceptions::Result;

/// Values of `array` in Fortran order (first index fastest).
pub(crate) fn fortran_order<S, D>(array: &ArrayBase<S, D>) -> Vec<f64>
where
    S: Data<Elem = f64>,
    D: Dimension,
{
    array.view().reversed_axes().iter().copied().collect()
}

struct Secondaries<'a> {
    x: &'a [f64],
    y: &'a [f64],
    z: &'a [f64],
    offset: &'a [i32],
}

fn check_secondaries(secondaries: &Secondaries<'_>, context: &str) -> Result<(c_int, c_int)> {
    require_len(secondaries.y.len(), secondaries.x.len(), "secondary_y", context)?;
    require_len(secondaries.z.len(), secondaries.x.len(), "secondary_z", context)?;
    Ok((
        len_c_int(secondaries.x.len(), context)?,
        len_c_int(secondaries.offset.len(), context)?,
    ))
}

fn check_primary(x0: &[f64], y0: &[f64], z0: &[f64], context: &str) -> Result<c_int> {
    require_len(y0.len(), x0.len(), "y0", context)?;
    require_len(z0.len(), x0.len(), "z0", context)?;
    len_c_int(x0.len(), context)
}

/// Run collocated cokriging for the provided target coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_collocated_cokriging(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    target_x: &[f64],
    target_y: &[f64],
    model_info: &[f64],
    secondary_x: &[f64],
    secondary_y: &[f64],
    secondary_z: &[f64],
    secondary_offset: &[i32],
    radius: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "native cokriging execution failed";
    let n = check_primary(x0, y0, z0, CONTEXT)?;
    require_len(target_y.len(), target_x.len(), "target_y", CONTEXT)?;
    let ntargets = len_c_int(target_x.len(), CONTEXT)?;
    let secondaries = Secondaries {
        x: secondary_x,
        y: secondary_y,
        z: secondary_z,
        offset: secondary_offset,
    };
    let (n_secondary_total, n_offsets) = check_secondaries(&secondaries, CONTEXT)?;
    let coreg_model_size = len_c_int(model_info.len(), CONTEXT)?;
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut estimates = vec![0.0f64; target_x.len()];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::collocated_cokriging_local_(
            x0.as_ptr(),
            y0.as_ptr(),
            z0.as_ptr(),
            target_x.as_ptr(),
            target_y.as_ptr(),
            model_info.as_ptr(),
            secondary_x.as_ptr(),
            secondary_y.as_ptr(),
            secondary_z.as_ptr(),
            secondary_offset.as_ptr(),
            &radius,
            &max_neighbors,
            &min_neighbors,
            &thread_count,
            estimates.as_mut_ptr(),
            &mut status,
            &n,
            &ntargets,
            &coreg_model_size,
            &n_secondary_total,
            &n_offsets,
        );
    }
    Ok((Array1::from(estimates), status))
}

/// Fit the ICM model: `(model_id, model_range, nugget_matrix, sill_matrix, status)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn run_fit_icm_model(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    secondary_x: &[f64],
    secondary_y: &[f64],
    secondary_z: &[f64],
    secondary_offset: &[i32],
    n_lags: i64,
    max_pairs: i64,
    max_distance: f64,
    max_colocation_dist: f64,
    n_threads: i64,
) -> Result<(f64, f64, Array2<f64>, Array2<f64>, i32)> {
    const CONTEXT: &str = "error fitting ICM model";
    let n = check_primary(x0, y0, z0, CONTEXT)?;
    let secondaries = Secondaries {
        x: secondary_x,
        y: secondary_y,
        z: secondary_z,
        offset: secondary_offset,
    };
    let (n_secondary_total, nv) = check_secondaries(&secondaries, CONTEXT)?;
    let n_lags = to_c_int(n_lags, CONTEXT)?;
    let max_pairs = to_c_int(max_pairs, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let size = secondary_offset.len();
    let mut model_id = 0.0f64;
    let mut model_range = 0.0f64;
    let mut nugget = vec![0.0f64; size * size];
    let mut sill = vec![0.0f64; size * size];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::fit_icm_model_(
            x0.as_ptr(),
            y0.as_ptr(),
            z0.as_ptr(),
            secondary_x.as_ptr(),
            secondary_y.as_ptr(),
            secondary_z.as_ptr(),
            secondary_offset.as_ptr(),
            &n_lags,
            &max_pairs,
            &max_distance,
            &max_colocation_dist,
            &thread_count,
            &mut model_id,
            &mut model_range,
            nugget.as_mut_ptr(),
            sill.as_mut_ptr(),
            &mut status,
            &n,
            &n_secondary_total,
            &nv,
        );
    }
    let nugget = Array2::from_shape_vec((size, size).f(), nugget)
        .expect("nv x nv")
        .as_standard_layout()
        .to_owned();
    let sill = Array2::from_shape_vec((size, size).f(), sill)
        .expect("nv x nv")
        .as_standard_layout()
        .to_owned();
    Ok((model_id, model_range, nugget, sill, status))
}

/// Run ICM cokriging for the provided target coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_co_kriging_icm(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    target_x: &[f64],
    target_y: &[f64],
    model_id: f64,
    model_range: f64,
    nugget_matrix: &Array2<f64>,
    sill_matrix: &Array2<f64>,
    secondary_x: &[f64],
    secondary_y: &[f64],
    secondary_z: &[f64],
    secondary_offset: &[i32],
    radius: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "native cokriging execution failed";
    let n = check_primary(x0, y0, z0, CONTEXT)?;
    require_len(target_y.len(), target_x.len(), "target_y", CONTEXT)?;
    let ntargets = len_c_int(target_x.len(), CONTEXT)?;
    let secondaries = Secondaries {
        x: secondary_x,
        y: secondary_y,
        z: secondary_z,
        offset: secondary_offset,
    };
    let (n_secondary_total, nv) = check_secondaries(&secondaries, CONTEXT)?;
    let size = secondary_offset.len();
    require_len(nugget_matrix.len(), size * size, "nugget_matrix", CONTEXT)?;
    require_len(sill_matrix.len(), size * size, "sill_matrix", CONTEXT)?;
    let nugget = fortran_order(nugget_matrix);
    let sill = fortran_order(sill_matrix);
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut estimates = vec![0.0f64; target_x.len()];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::co_kriging_icm_local_(
            x0.as_ptr(),
            y0.as_ptr(),
            z0.as_ptr(),
            target_x.as_ptr(),
            target_y.as_ptr(),
            &model_id,
            &model_range,
            nugget.as_ptr(),
            sill.as_ptr(),
            secondary_x.as_ptr(),
            secondary_y.as_ptr(),
            secondary_z.as_ptr(),
            secondary_offset.as_ptr(),
            &radius,
            &max_neighbors,
            &min_neighbors,
            &thread_count,
            estimates.as_mut_ptr(),
            &mut status,
            &n,
            &ntargets,
            &nv,
            &n_secondary_total,
        );
    }
    Ok((Array1::from(estimates), status))
}

/// Fit the LMC model: `(model_id, structure_ranges, coefficient_matrices, status)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn run_fit_lmc_model(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    secondary_x: &[f64],
    secondary_y: &[f64],
    secondary_z: &[f64],
    secondary_offset: &[i32],
    n_structures: i64,
    n_lags: i64,
    max_pairs: i64,
    max_distance: f64,
    max_colocation_dist: f64,
    n_threads: i64,
) -> Result<(f64, Array1<f64>, Array3<f64>, i32)> {
    const CONTEXT: &str = "error fitting LMC model";
    let n = check_primary(x0, y0, z0, CONTEXT)?;
    let secondaries = Secondaries {
        x: secondary_x,
        y: secondary_y,
        z: secondary_z,
        offset: secondary_offset,
    };
    let (n_secondary_total, nv) = check_secondaries(&secondaries, CONTEXT)?;
    let structures = usize::try_from(n_structures.max(1)).expect("positive");
    let n_structures = to_c_int(n_structures, CONTEXT)?;
    let n_continuous = n_structures - 1;
    let n_lags = to_c_int(n_lags, CONTEXT)?;
    let max_pairs = to_c_int(max_pairs, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let size = secondary_offset.len();
    let mut model_id = 0.0f64;
    let mut structure_ranges = vec![0.0f64; structures - 1];
    let mut coefficients = vec![0.0f64; structures * size * size];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::fit_lmc_model_(
            x0.as_ptr(),
            y0.as_ptr(),
            z0.as_ptr(),
            secondary_x.as_ptr(),
            secondary_y.as_ptr(),
            secondary_z.as_ptr(),
            secondary_offset.as_ptr(),
            &n_structures,
            &n_lags,
            &max_pairs,
            &max_distance,
            &max_colocation_dist,
            &thread_count,
            &mut model_id,
            structure_ranges.as_mut_ptr(),
            coefficients.as_mut_ptr(),
            &mut status,
            &n,
            &n_secondary_total,
            &nv,
            &n_continuous,
        );
    }
    let coefficients = Array3::from_shape_vec((structures, size, size).f(), coefficients)
        .expect("S x nv x nv")
        .as_standard_layout()
        .to_owned();
    Ok((model_id, Array1::from(structure_ranges), coefficients, status))
}

/// Run LMC cokriging for the provided target coordinates.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_co_kriging_lmc(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    target_x: &[f64],
    target_y: &[f64],
    model_id: f64,
    structure_ranges: &[f64],
    coefficient_matrices: &Array3<f64>,
    secondary_x: &[f64],
    secondary_y: &[f64],
    secondary_z: &[f64],
    secondary_offset: &[i32],
    radius: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    n_threads: i64,
) -> Result<(Array1<f64>, i32)> {
    const CONTEXT: &str = "native cokriging execution failed";
    let n = check_primary(x0, y0, z0, CONTEXT)?;
    require_len(target_y.len(), target_x.len(), "target_y", CONTEXT)?;
    let ntargets = len_c_int(target_x.len(), CONTEXT)?;
    let secondaries = Secondaries {
        x: secondary_x,
        y: secondary_y,
        z: secondary_z,
        offset: secondary_offset,
    };
    let (n_secondary_total, nv) = check_secondaries(&secondaries, CONTEXT)?;
    let shape = coefficient_matrices.shape();
    let n_structures = len_c_int(shape[0], CONTEXT)?;
    let n_continuous = len_c_int(structure_ranges.len(), CONTEXT)?;
    require_len(shape[1], secondary_offset.len(), "coefficient_matrices", CONTEXT)?;
    require_len(shape[2], secondary_offset.len(), "coefficient_matrices", CONTEXT)?;
    let coefficients = fortran_order(coefficient_matrices);
    let max_neighbors = to_c_int(max_neighbors, CONTEXT)?;
    let min_neighbors = to_c_int(min_neighbors, CONTEXT)?;
    let thread_count = to_c_int(n_threads, CONTEXT)?;
    let mut estimates = vec![0.0f64; target_x.len()];
    let mut status: c_int = 0;

    lib::load();
    unsafe {
        lib::co_kriging_lmc_local_(
            x0.as_ptr(),
            y0.as_ptr(),
            z0.as_ptr(),
            target_x.as_ptr(),
            target_y.as_ptr(),
            &model_id,
            structure_ranges.as_ptr(),
            coefficients.as_ptr(),
            secondary_x.as_ptr(),
            secondary_y.as_ptr(),
            secondary_z.as_ptr(),
            secondary_offset.as_ptr(),
            &radius,
            &max_neighbors,
            &min_neighbors,
            &thread_count,
            estimates.as_mut_ptr(),
            &mut status,
            &n,
            &ntargets,
            &nv,
            &n_secondary_total,
            &n_structures,
            &n_continuous,
        );
    }
    Ok((Array1::from(estimates), status))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::_native::test_support::{assert_rejected, assert_status};
    use crate::common::exceptions::{COKRIGING_STATUS_ERRORS, ErrorKind};
    use ndarray::array;

    const OFFSET: [i32; 2] = [0, 64];
    const TARGETS_X: [f64; 3] = [3.0, 25.0, 47.0];
    const TARGETS_Y: [f64; 3] = [4.0, 25.0, 12.0];
    /// Collocated model `[id, nugget, partial_sill, range, sse, rho]`.
    const COLLOCATED: [f64; 6] = [1.0, 0.1, 1.0, 40.0, 0.0, 0.7];

    /// `n` irregular but deterministic points in a `side` x `side` square (R2
    /// low-discrepancy sequence) with a stationary primary field and one
    /// correlated secondary at the same points. On small regular grids the joint
    /// ICM fit itself degenerates to zero matrices, so the cokriging fits use
    /// this layout.
    fn data_with(n: i32, side: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let plastic = 1.324_717_957_244_746_f64;
        let (mut x, mut y, mut z, mut secondary) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 1..=n {
            let i = f64::from(i);
            let xi = side * ((0.5 + i / plastic) % 1.0);
            let yi = side * ((0.5 + i / (plastic * plastic)) % 1.0);
            let zi = (xi / 9.0).sin() + (yi / 11.0).cos() + 0.3 * ((xi - yi) / 7.0).sin();
            x.push(xi);
            y.push(yi);
            z.push(zi);
            secondary.push(0.7 * zi + 0.3 * (xi / 5.0).cos() * (yi / 6.0).sin());
        }
        (x, y, z, secondary)
    }

    fn data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        data_with(64, 60.0)
    }

    fn icm_matrices() -> (Array2<f64>, Array2<f64>) {
        (array![[0.1, 0.0], [0.0, 0.1]], array![[1.0, 0.5], [0.5, 1.0]])
    }

    fn lmc_coefficients() -> Array3<f64> {
        let (nugget, sill) = icm_matrices();
        ndarray::stack(ndarray::Axis(0), &[nugget.view(), sill.view()]).unwrap()
    }

    #[test]
    fn fortran_order_puts_first_index_fastest() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(fortran_order(&matrix), vec![1.0, 3.0, 2.0, 4.0]);
        let cube = Array3::from_shape_fn((2, 2, 2), |(s, i, j)| (100 * s + 10 * i + j) as f64);
        let buffer = fortran_order(&cube);
        for s in 0..2 {
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(buffer[s + 2 * (i + 2 * j)], cube[[s, i, j]]);
                }
            }
        }
        let back = Array3::from_shape_vec((2, 2, 2).f(), buffer).unwrap();
        assert_eq!(back, cube);
    }

    #[allow(clippy::type_complexity)]
    fn collocated(
        model: &[f64],
        offset: &[i32],
        radius: f64,
        max_neighbors: i64,
        min_neighbors: i64,
    ) -> (Array1<f64>, i32) {
        let (x, y, z, s) = data();
        run_collocated_cokriging(
            &x,
            &y,
            &z,
            &TARGETS_X,
            &TARGETS_Y,
            model,
            &x,
            &y,
            &s,
            offset,
            radius,
            max_neighbors,
            min_neighbors,
            1,
        )
        .unwrap()
    }

    #[test]
    fn collocated_cokriging_estimates_finite_values() {
        let (estimates, status) = collocated(&COLLOCATED, &OFFSET, 30.0, 16, 1);
        assert_eq!(status, 0);
        assert!(estimates.iter().all(|v| v.is_finite()), "{estimates}");
    }

    #[test]
    fn collocated_cokriging_statuses_reach_the_documented_errors() {
        let table = COKRIGING_STATUS_ERRORS;
        let status = |model: &[f64], offset: &[i32], radius: f64, max_neighbors: i64, min_neighbors: i64| {
            collocated(model, offset, radius, max_neighbors, min_neighbors).1
        };
        let data_error = ErrorKind::CoKrigingInvalidDataError;
        assert_status(
            status(&COLLOCATED[..5], &OFFSET, 30.0, 16, 1),
            1,
            table,
            data_error,
        );
        assert_status(status(&COLLOCATED, &[0, 63], 30.0, 16, 1), 1, table, data_error);
        assert_status(status(&COLLOCATED, &[0], 30.0, 16, 1), 1, table, data_error);
        assert_status(
            status(&COLLOCATED, &OFFSET, 0.0, 16, 1),
            2,
            table,
            ErrorKind::CoKrigingInvalidRadiusError,
        );
        let config = ErrorKind::CoKrigingInvalidConfigurationError;
        assert_status(status(&COLLOCATED, &OFFSET, 30.0, 2, 3), 3, table, config);
        for model in [
            [0.0, 0.1, 1.0, 40.0, 0.0, 0.7],
            [1.0, 0.1, 1.0, -1.0, 0.0, 0.7],
            [1.0, 0.0, 0.0, 40.0, 0.0, 0.7],
            [1.0, 0.1, 1.0, 40.0, 0.0, 1.5],
        ] {
            assert_status(
                status(&model, &OFFSET, 30.0, 16, 1),
                6,
                table,
                ErrorKind::CoKrigingInvalidModelError,
            );
        }
    }

    #[test]
    fn icm_fit_feeds_icm_cokriging() {
        let (x, y, z, s) = data();
        let (model_id, model_range, nugget, sill, status) =
            run_fit_icm_model(&x, &y, &z, &x, &y, &s, &OFFSET, 10, 0, 0.0, 0.0, 1).unwrap();
        assert_eq!(status, 0);
        assert!([1.0, 2.0, 3.0].contains(&model_id), "{model_id}");
        assert!(model_range.is_finite() && model_range > 0.0, "{model_range}");
        assert_eq!((nugget.dim(), sill.dim()), ((2, 2), (2, 2)));
        assert_eq!((nugget[[0, 1]], sill[[0, 1]]), (nugget[[1, 0]], sill[[1, 0]]));
        assert!(sill[[0, 0]] > 0.0 && sill[[1, 1]] > 0.0, "{sill}");
        let (estimates, status) = run_co_kriging_icm(
            &x,
            &y,
            &z,
            &TARGETS_X,
            &TARGETS_Y,
            model_id,
            model_range,
            &nugget,
            &sill,
            &x,
            &y,
            &s,
            &OFFSET,
            30.0,
            16,
            1,
            1,
        )
        .unwrap();
        assert_eq!(status, 0);
        assert!(estimates.iter().all(|v| v.is_finite()), "{estimates}");
    }

    #[test]
    fn icm_statuses_reach_the_documented_errors() {
        let (x, y, z, s) = data();
        let table = COKRIGING_STATUS_ERRORS;
        let data_error = ErrorKind::CoKrigingInvalidDataError;
        let fit = |n_lags: i64, offset: &[i32], n_threads: i64| {
            run_fit_icm_model(&x, &y, &z, &x, &y, &s, offset, n_lags, 0, 0.0, 0.0, n_threads)
                .unwrap()
                .4
        };
        assert_status(fit(0, &OFFSET, 1), 1, table, data_error);
        assert_status(fit(10, &OFFSET, 0), 1, table, data_error);
        assert_status(fit(10, &[0, 30], 1), 1, table, data_error);

        let (nugget, sill) = icm_matrices();
        let krige = |model_id: f64,
                     model_range: f64,
                     nugget: &Array2<f64>,
                     radius: f64,
                     max_neighbors: i64,
                     min_neighbors: i64| {
            run_co_kriging_icm(
                &x,
                &y,
                &z,
                &TARGETS_X,
                &TARGETS_Y,
                model_id,
                model_range,
                nugget,
                &sill,
                &x,
                &y,
                &s,
                &OFFSET,
                radius,
                max_neighbors,
                min_neighbors,
                1,
            )
            .unwrap()
            .1
        };
        assert_status(
            krige(1.0, 40.0, &nugget, 0.0, 16, 1),
            2,
            table,
            ErrorKind::CoKrigingInvalidRadiusError,
        );
        let config = ErrorKind::CoKrigingInvalidConfigurationError;
        assert_status(krige(1.0, 40.0, &nugget, 30.0, 2, 3), 3, table, config);
        let model = ErrorKind::CoKrigingInvalidModelError;
        assert_status(krige(0.0, 40.0, &nugget, 30.0, 16, 1), 6, table, model);
        assert_status(krige(1.0, 0.0, &nugget, 30.0, 16, 1), 6, table, model);
        assert_status(
            krige(1.0, 40.0, &array![[-1.0, 0.0], [0.0, 0.1]], 30.0, 16, 1),
            6,
            table,
            model,
        );
    }

    #[test]
    fn icm_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z, s) = data();
        let (nugget, sill) = icm_matrices();
        let krige =
            |z0: &[f64], target_y: &[f64], nugget: &Array2<f64>, sill: &Array2<f64>, secondary_z: &[f64]| {
                run_co_kriging_icm(
                    &x,
                    &y,
                    z0,
                    &TARGETS_X,
                    target_y,
                    1.0,
                    40.0,
                    nugget,
                    sill,
                    &x,
                    &y,
                    secondary_z,
                    &OFFSET,
                    30.0,
                    16,
                    1,
                    1,
                )
                .unwrap_err()
            };
        assert_rejected(krige(&z[1..], &TARGETS_Y, &nugget, &sill, &s), "z0");
        assert_rejected(krige(&z, &TARGETS_Y[1..], &nugget, &sill, &s), "target_y");
        assert_rejected(
            krige(&z, &TARGETS_Y, &Array2::zeros((3, 3)), &sill, &s),
            "nugget_matrix",
        );
        assert_rejected(
            krige(&z, &TARGETS_Y, &nugget, &Array2::zeros((1, 1)), &s),
            "sill_matrix",
        );
        assert_rejected(krige(&z, &TARGETS_Y, &nugget, &sill, &s[1..]), "secondary_z");
        let fit = run_fit_icm_model(&x, &y[1..], &z, &x, &y, &s, &OFFSET, 10, 0, 0.0, 0.0, 1);
        assert_rejected(fit.unwrap_err(), "y0");
    }

    #[test]
    fn lmc_fit_feeds_lmc_cokriging() {
        let (x, y, z, s) = data();
        let (model_id, ranges, coefficients, status) =
            run_fit_lmc_model(&x, &y, &z, &x, &y, &s, &OFFSET, 2, 10, 0, 0.0, 0.0, 1).unwrap();
        assert_eq!(status, 0);
        assert!([1.0, 2.0, 3.0].contains(&model_id), "{model_id}");
        assert_eq!(ranges.len(), 1);
        assert!(ranges[0].is_finite() && ranges[0] > 0.0, "{ranges}");
        assert_eq!(coefficients.dim(), (2, 2, 2));
        assert!(coefficients.iter().all(|v| v.is_finite()), "{coefficients}");
        assert!(
            coefficients[[1, 0, 0]] > 0.0 && coefficients[[1, 1, 1]] > 0.0,
            "{coefficients}"
        );
        let (estimates, status) = run_co_kriging_lmc(
            &x,
            &y,
            &z,
            &TARGETS_X,
            &TARGETS_Y,
            model_id,
            ranges.as_slice().unwrap(),
            &coefficients,
            &x,
            &y,
            &s,
            &OFFSET,
            30.0,
            16,
            1,
            1,
        )
        .unwrap();
        assert_eq!(status, 0);
        assert!(estimates.iter().all(|v| v.is_finite()), "{estimates}");
    }

    #[test]
    fn lmc_statuses_reach_the_documented_errors() {
        let (x, y, z, s) = data();
        let table = COKRIGING_STATUS_ERRORS;
        let data_error = ErrorKind::CoKrigingInvalidDataError;
        let fit = |n_structures: i64, n_lags: i64| {
            run_fit_lmc_model(
                &x,
                &y,
                &z,
                &x,
                &y,
                &s,
                &OFFSET,
                n_structures,
                n_lags,
                0,
                0.0,
                0.0,
                1,
            )
            .unwrap()
            .3
        };
        assert_status(fit(1, 10), 1, table, data_error);
        assert_status(fit(2, 0), 1, table, data_error);

        let krige =
            |model_id: f64, ranges: &[f64], coefficients: &Array3<f64>, radius: f64, max_neighbors: i64| {
                run_co_kriging_lmc(
                    &x,
                    &y,
                    &z,
                    &TARGETS_X,
                    &TARGETS_Y,
                    model_id,
                    ranges,
                    coefficients,
                    &x,
                    &y,
                    &s,
                    &OFFSET,
                    radius,
                    max_neighbors,
                    1,
                    1,
                )
                .unwrap()
                .1
            };
        let coefficients = lmc_coefficients();
        assert_status(
            krige(1.0, &[40.0, 80.0], &coefficients, 30.0, 16),
            1,
            table,
            data_error,
        );
        assert_status(
            krige(1.0, &[40.0], &coefficients, 0.0, 16),
            2,
            table,
            ErrorKind::CoKrigingInvalidRadiusError,
        );
        let config = ErrorKind::CoKrigingInvalidConfigurationError;
        assert_status(krige(1.0, &[40.0], &coefficients, 30.0, 0), 3, table, config);
        let model = ErrorKind::CoKrigingInvalidModelError;
        assert_status(krige(9.0, &[40.0], &coefficients, 30.0, 16), 6, table, model);
        assert_status(krige(1.0, &[0.0], &coefficients, 30.0, 16), 6, table, model);
        let mut negative = coefficients.clone();
        negative[[1, 0, 0]] = -1.0;
        assert_status(krige(1.0, &[40.0], &negative, 30.0, 16), 6, table, model);
    }

    #[test]
    fn lmc_mismatched_lengths_are_rejected_before_the_kernel() {
        let (x, y, z, s) = data();
        let krige = run_co_kriging_lmc(
            &x,
            &y,
            &z,
            &TARGETS_X,
            &TARGETS_Y,
            1.0,
            &[40.0],
            &Array3::zeros((2, 3, 3)),
            &x,
            &y,
            &s,
            &OFFSET,
            30.0,
            16,
            1,
            1,
        );
        assert_rejected(krige.unwrap_err(), "coefficient_matrices");
        let fit = run_fit_lmc_model(&x, &y, &z, &x, &y[1..], &s, &OFFSET, 2, 10, 0, 0.0, 0.0, 1);
        assert_rejected(fit.unwrap_err(), "secondary_y");
    }
}
