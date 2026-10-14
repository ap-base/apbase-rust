//! High-level kriging API: local ordinary kriging. Target-grid generation lives in `apbase::grid`.

mod _interpolation;
mod _native;

use ndarray::Array1;

pub use crate::variogram::Variogram;
pub use _interpolation::{Kriging, KrigingOptions};

use crate::common::arrays::{ArrayLike, Targets};
use crate::common::exceptions::Result;

/// Interpolation diagnostics returned by [`ordinary_kriging_with_diagnostics`].
#[derive(Clone, Debug, PartialEq)]
pub struct KrigingDiagnostics {
    /// Local search radius actually used.
    pub radius: f64,
    /// Maximum neighbors per target.
    pub max_neighbors: i64,
    /// Minimum neighbors per target.
    pub min_neighbors: i64,
    /// OpenMP thread count.
    pub n_threads: i64,
}

/// Fit local ordinary kriging and estimate `targets` values.
///
/// One-shot wrapper around [`Kriging`]; prefer [`Kriging`] directly to reuse a
/// fitted variogram across several `interpolate` calls.
pub fn ordinary_kriging(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    targets: &(impl Targets + ?Sized),
    options: KrigingOptions,
) -> Result<Array1<f64>> {
    Kriging::new(options)?.fit(x, y, z)?.interpolate(targets)
}

/// [`ordinary_kriging`] with `return_diagnostics=True`: `(estimates, diagnostics)`.
pub fn ordinary_kriging_with_diagnostics(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    targets: &(impl Targets + ?Sized),
    options: KrigingOptions,
) -> Result<(Array1<f64>, KrigingDiagnostics)> {
    let max_neighbors = options.max_neighbors;
    let min_neighbors = options.min_neighbors;
    let mut kriging = Kriging::new(options)?;
    kriging.fit(x, y, z)?;
    let estimates = kriging.interpolate(targets)?;

    Ok((
        estimates,
        KrigingDiagnostics {
            radius: kriging.resolve_radius()?,
            max_neighbors,
            min_neighbors,
            n_threads: kriging.n_threads(),
        },
    ))
}
