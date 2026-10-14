//! High-level cokriging API.
//!
//! Estimates a primary variable using one or more spatially correlated
//! secondary variables. Run `apbase::cross_variogram::screen_secondary_variables`
//! first to select admissible secondaries.

mod _interpolation;
mod _native;

use ndarray::Array1;

pub use _interpolation::{
    CoKriging, CoKrigingMethod, CoKrigingModelParams, CoKrigingOptions, SecondaryInput,
};

pub(crate) use _interpolation::Fit;
pub(crate) use _native::fortran_order;
#[cfg(test)]
pub(crate) use _native::{run_co_kriging_icm, run_co_kriging_lmc, run_collocated_cokriging};

use crate::common::arrays::{ArrayLike, Targets};
use crate::common::exceptions::Result;

/// Fit cokriging and estimate `targets` values in one call.
///
/// One-shot wrapper around [`CoKriging`]; prefer [`CoKriging`] directly to reuse
/// the fitted model across several `interpolate` calls.
pub fn co_kriging(
    x0: &(impl ArrayLike + ?Sized),
    y0: &(impl ArrayLike + ?Sized),
    z0: &(impl ArrayLike + ?Sized),
    secondaries: &SecondaryInput,
    targets: &(impl Targets + ?Sized),
    options: CoKrigingOptions,
) -> Result<Array1<f64>> {
    CoKriging::new(options)?
        .fit(x0, y0, z0, secondaries)?
        .interpolate(targets)
}
