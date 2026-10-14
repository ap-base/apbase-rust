//! High-level inverse distance weighting API.

mod _interpolation;
mod _native;

use ndarray::Array1;

pub use _interpolation::{IDW, IDWOptions};

use crate::common::arrays::{ArrayLike, Targets};
use crate::common::exceptions::Result;

/// Fit IDW from source data and estimate target values.
///
/// One-shot wrapper around [`IDW`]: `IDW::new(options)?.fit(x, y, z)?.interpolate(targets)`.
pub fn idw(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    targets: &(impl Targets + ?Sized),
    options: IDWOptions,
) -> Result<Array1<f64>> {
    IDW::new(options)?.fit(x, y, z)?.interpolate(targets)
}
