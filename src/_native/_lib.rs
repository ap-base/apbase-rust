//! Facade for the compiled apbase Fortran kernels (`libapbase_native.a`).
//!
//! The single place that declares the native symbols. Kernel wrappers
//! (`<package>/_native.rs`) call [`load`] once and then the `extern "C"`
//! functions below. Every Fortran argument is passed by reference (gfortran
//! ABI); array dimensions are the trailing `n`-like arguments; multi-dimensional
//! arrays are column-major.

use std::ffi::{c_char, c_int, c_void};
use std::sync::Once;

use super::_runtime::configure_native_threading;

static LOADED: Once = Once::new();

/// Prepare the native runtime before the first kernel call (`lib.load()`).
pub(crate) fn load() {
    LOADED.call_once(|| {
        if let Some(threads) = configure_native_threading() {
            set_blas_threads(threads);
        }
    });
}

unsafe extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

// glibc's RTLD_DEFAULT: search the global symbol scope.
const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();

/// `OPENBLAS_NUM_THREADS=<threads>` equivalent, applied through OpenBLAS'
/// runtime API (the library is already initialized when Rust code runs).
/// Skipped silently when the linked BLAS is not OpenBLAS.
fn set_blas_threads(threads: c_int) {
    // SAFETY: dlsym is thread-safe; the symbol, when present, has the
    // documented OpenBLAS signature `void openblas_set_num_threads(int)`.
    unsafe {
        let symbol = dlsym(RTLD_DEFAULT, c"openblas_set_num_threads".as_ptr());
        if !symbol.is_null() {
            let set_num_threads: extern "C" fn(c_int) = std::mem::transmute(symbol);
            set_num_threads(threads);
        }
    }
}

unsafe extern "C" {
    // filtering/_native_src/filter_mask.f90
    pub(crate) fn viz_filter_mask_(
        pct_diff: *const f64,
        local_prob: *const f64,
        local_z: *const f64,
        filter_level: *const c_int,
        keep_mask: *mut i8,
        status: *mut c_int,
        n: *const c_int,
    );

    // filtering/_native_src/zstats_radius_gridagg.f90
    pub(crate) fn viz_zstats_radius_gridagg_(
        x: *const f64,
        y: *const f64,
        z: *const f64,
        radius: *const f64,
        cell_factor: *const c_int,
        max_neighbors: *const c_int,
        candidate_fraction: *const f64,
        dedup_radius: *const f64,
        dedup_z_tol: *const f64,
        thread_count: *const c_int,
        pct_diff: *mut f64,
        local_z: *mut f64,
        local_prob: *mut f64,
        status: *mut c_int,
        n: *const c_int,
    );

    // variogram/_native_src/variogram_fit.f90
    pub(crate) fn fit_variogram_models_(
        x: *const f64,
        y: *const f64,
        z: *const f64,
        n_lags: *const c_int,
        max_pairs: *const c_int,
        max_distance: *const f64,
        thread_count: *const c_int,
        model_info: *mut f64,
        n: *const c_int,
    );

    // variogram/_native_src/variogram_select.f90
    pub(crate) fn select_variogram_(model_info: *const f64, params: *mut f64, status: *mut c_int);

    pub(crate) fn evaluate_variogram_(
        model_info: *const f64,
        distance: *const f64,
        thread_count: *const c_int,
        gamma: *mut f64,
        status: *mut c_int,
        n: *const c_int,
    );

    // idw/_native_src/idw_local.f90
    pub(crate) fn idw_local_(
        x: *const f64,
        y: *const f64,
        z: *const f64,
        target_x: *const f64,
        target_y: *const f64,
        radius: *const f64,
        power: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        estimates: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        ntargets: *const c_int,
    );

    // kriging/_native_src/ordinary_kriging_local.f90
    pub(crate) fn ordinary_kriging_local_(
        x: *const f64,
        y: *const f64,
        z: *const f64,
        target_x: *const f64,
        target_y: *const f64,
        model_info: *const f64,
        radius: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        estimates: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        ntargets: *const c_int,
    );

    // cross_validate/_native_src/cross_validate_local.f90
    pub(crate) fn cross_validate_local_(
        x: *const f64,
        y: *const f64,
        z: *const f64,
        sample_idx: *const c_int,
        radius: *const f64,
        power: *const f64,
        model_info: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        idw_pred: *mut f64,
        krig_pred: *mut f64,
        idw_mae: *mut f64,
        idw_rmse: *mut f64,
        idw_r2: *mut f64,
        idw_n_valid: *mut c_int,
        krig_mae: *mut f64,
        krig_rmse: *mut f64,
        krig_r2: *mut f64,
        krig_n_valid: *mut c_int,
        status: *mut c_int,
        n: *const c_int,
        nsamples: *const c_int,
    );

    // cross_validate/_native_src/cokriging_cross_validate_local.f90
    pub(crate) fn cokriging_cross_validate_local_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        sample_idx: *const c_int,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        run_collocated: *const c_int,
        collocated_model: *const f64,
        run_icm: *const c_int,
        icm_model_id: *const f64,
        icm_range: *const f64,
        icm_nugget: *const f64,
        icm_sill: *const f64,
        run_lmc: *const c_int,
        lmc_model_id: *const f64,
        lmc_ranges: *const f64,
        lmc_coefficients: *const f64,
        radius: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        collocated_pred: *mut f64,
        icm_pred: *mut f64,
        lmc_pred: *mut f64,
        collocated_mae: *mut f64,
        collocated_rmse: *mut f64,
        collocated_r2: *mut f64,
        collocated_n_valid: *mut c_int,
        icm_mae: *mut f64,
        icm_rmse: *mut f64,
        icm_r2: *mut f64,
        icm_n_valid: *mut c_int,
        lmc_mae: *mut f64,
        lmc_rmse: *mut f64,
        lmc_r2: *mut f64,
        lmc_n_valid: *mut c_int,
        status: *mut c_int,
        n: *const c_int,
        nsamples: *const c_int,
        n_secondary_total: *const c_int,
        nv: *const c_int,
        collocated_model_size: *const c_int,
        n_lmc_structures: *const c_int,
        n_lmc_continuous: *const c_int,
    );

    // cross_variogram/_native_src/cross_variogram_fit.f90
    pub(crate) fn fit_cross_variogram_(
        xa: *const f64,
        ya: *const f64,
        za: *const f64,
        xb: *const f64,
        yb: *const f64,
        zb: *const f64,
        max_colocation_dist: *const f64,
        n_lags: *const c_int,
        max_pairs: *const c_int,
        max_distance: *const f64,
        thread_count: *const c_int,
        model_info: *mut f64,
        rho: *mut f64,
        n_pairs: *mut c_int,
        na: *const c_int,
        nb: *const c_int,
    );

    // cross_variogram/_native_src/covariate_screening.f90
    pub(crate) fn screen_covariate_(
        model_00: *const f64,
        model_uu: *const f64,
        model_0u: *const f64,
        rho: *const f64,
        n_pairs: *const c_int,
        alpha: *const f64,
        min_abs_rho: *const f64,
        max_violation_fraction: *const f64,
        decision: *mut c_int,
        p_value: *mut f64,
        admissible_fraction: *mut f64,
        status: *mut c_int,
    );

    // cokriging/_native_src/collocated_cokriging_local.f90
    pub(crate) fn collocated_cokriging_local_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        target_x: *const f64,
        target_y: *const f64,
        coreg_model: *const f64,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        radius: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        estimates: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        ntargets: *const c_int,
        coreg_model_size: *const c_int,
        n_secondary_total: *const c_int,
        n_offsets: *const c_int,
    );

    // cokriging/_native_src/icm_fit.f90
    pub(crate) fn fit_icm_model_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        n_lags: *const c_int,
        max_pairs: *const c_int,
        max_distance: *const f64,
        max_colocation_dist: *const f64,
        thread_count: *const c_int,
        model_id: *mut f64,
        model_range: *mut f64,
        nugget_matrix: *mut f64,
        sill_matrix: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        n_secondary_total: *const c_int,
        nv: *const c_int,
    );

    // cokriging/_native_src/co_kriging_icm_local.f90
    pub(crate) fn co_kriging_icm_local_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        target_x: *const f64,
        target_y: *const f64,
        model_id: *const f64,
        model_range: *const f64,
        nugget_matrix: *const f64,
        sill_matrix: *const f64,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        radius: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        estimates: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        ntargets: *const c_int,
        nv: *const c_int,
        n_secondary_total: *const c_int,
    );

    // cokriging/_native_src/lmc_fit.f90
    pub(crate) fn fit_lmc_model_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        n_structures: *const c_int,
        n_lags: *const c_int,
        max_pairs: *const c_int,
        max_distance: *const f64,
        max_colocation_dist: *const f64,
        thread_count: *const c_int,
        model_id: *mut f64,
        structure_ranges: *mut f64,
        coefficient_matrices: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        n_secondary_total: *const c_int,
        nv: *const c_int,
        n_continuous: *const c_int,
    );

    // cokriging/_native_src/co_kriging_lmc_local.f90
    pub(crate) fn co_kriging_lmc_local_(
        x0: *const f64,
        y0: *const f64,
        z0: *const f64,
        target_x: *const f64,
        target_y: *const f64,
        model_id: *const f64,
        structure_ranges: *const f64,
        coefficient_matrices: *const f64,
        secondary_x: *const f64,
        secondary_y: *const f64,
        secondary_z: *const f64,
        secondary_offset: *const c_int,
        radius: *const f64,
        max_neighbors: *const c_int,
        min_neighbors: *const c_int,
        thread_count: *const c_int,
        estimates: *mut f64,
        status: *mut c_int,
        n: *const c_int,
        ntargets: *const c_int,
        nv: *const c_int,
        n_secondary_total: *const c_int,
        n_structures: *const c_int,
        n_continuous: *const c_int,
    );
}

// Test-only reference math.
#[cfg(test)]
unsafe extern "C" {
    pub(crate) fn erf(x: f64) -> f64;
}
