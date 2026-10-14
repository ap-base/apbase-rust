//! Inverse distance weighting interpolation.

use ndarray::Array1;

use super::_native::run_idw;
use crate::common::arrays::{
    ArrayLike, Targets, as_float64_1d, extract_target_xy, filter_finite_xyz, validate_same_size_xyz,
};
use crate::common::exceptions::{ApbaseError, IDW_STATUS_ERRORS, Result, raise_status_error};
use crate::common::validation::require_neighbor_bounds;
use crate::config::{resolve_n_threads, resolve_variogram_profile};
use crate::variogram::{ModelParams, Variogram, VariogramOptions};

/// Options of [`IDW`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct IDWOptions {
    /// Local search radius; `None` uses [`ModelParams::search_radius`] of a fitted variogram.
    pub radius: Option<f64>,
    /// Exponent applied to distance in the IDW weights.
    pub power: f64,
    /// Maximum source points used per target coordinate.
    pub max_neighbors: i64,
    /// Minimum sources within the radius; below it the `max_neighbors` nearest
    /// sources are used regardless of distance.
    pub min_neighbors: i64,
}

impl Default for IDWOptions {
    fn default() -> Self {
        IDWOptions {
            radius: None,
            power: 2.0,
            max_neighbors: 40,
            min_neighbors: 3,
        }
    }
}

/// Interpolate projected coordinates with inverse distance weighting.
///
/// ```
/// use apbase::idw::{IDW, IDWOptions};
/// use ndarray::array;
///
/// let x = [0.0, 10.0, 0.0, 10.0];
/// let y = [0.0, 0.0, 10.0, 10.0];
/// let z = [5.0, 7.0, 8.0, 10.0];
/// let targets = array![[5.0, 5.0], [10.0, 10.0]];
/// let values = IDW::new(IDWOptions { radius: Some(20.0), min_neighbors: 1, ..Default::default() })?
///     .fit(&x, &y, &z)?
///     .interpolate(&targets)?;
/// assert_eq!(values[1], 10.0);
/// # Ok::<(), apbase::common::ApbaseError>(())
/// ```
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct IDW {
    radius: Option<f64>,
    power: f64,
    max_neighbors: i64,
    min_neighbors: i64,
    x: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    z: Option<Array1<f64>>,
    model_params: Option<ModelParams>,
    fitted: bool,
}

impl IDW {
    /// Create an unfitted IDW interpolator; invalid options are rejected here.
    pub fn new(options: IDWOptions) -> Result<Self> {
        IDW::validate_config(
            options.radius,
            options.power,
            options.max_neighbors,
            options.min_neighbors,
        )?;
        Ok(IDW {
            radius: options.radius,
            power: options.power,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
            x: None,
            y: None,
            z: None,
            model_params: None,
            fitted: false,
        })
    }

    /// Local search radius (`None` derives it from the variogram range).
    pub fn radius(&self) -> Option<f64> {
        self.radius
    }

    /// Exponent applied to distance in the IDW weights.
    pub fn power(&self) -> f64 {
        self.power
    }

    /// Maximum source points used per target coordinate.
    pub fn max_neighbors(&self) -> i64 {
        self.max_neighbors
    }

    /// Minimum sources within the radius; below it the `max_neighbors` nearest
    /// sources are used regardless of distance.
    pub fn min_neighbors(&self) -> i64 {
        self.min_neighbors
    }

    /// OpenMP thread count, resolved live from `apbase.config["n_threads"]`.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Create and fit an IDW interpolator.
    pub fn from_data(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        options: IDWOptions,
    ) -> Result<Self> {
        let mut idw = IDW::new(options)?;
        idw.fit(x, y, z)?;
        Ok(idw)
    }

    /// Whether finite source data are available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Parameters of the variogram used by the default radius.
    pub fn model_params(&self) -> Result<ModelParams> {
        self.require_fitted()?;
        self.model_params.clone().ok_or_else(|| {
            ApbaseError::value_error("IDW was fit with an explicit radius; no variogram model was fitted")
        })
    }

    /// Store finite source samples used by later interpolation.
    pub fn fit(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<&mut Self> {
        let x_array = as_float64_1d(x, "x")?;
        let y_array = as_float64_1d(y, "y")?;
        let z_array = as_float64_1d(z, "z")?;
        validate_same_size_xyz(&x_array, &y_array, &z_array)?;

        let (x_valid, y_valid, z_valid) = filter_finite_xyz(&x_array, &y_array, &z_array);
        if (x_valid.len() as i64) < self.min_neighbors {
            return Err(ApbaseError::value_error(
                "IDW requires at least min_neighbors finite points",
            ));
        }

        let mut model_params = None;
        if self.radius.is_none() {
            let (n_lags, max_pairs, max_distance) = resolve_variogram_profile();
            let variogram = Variogram::from_data(
                &x_valid,
                &y_valid,
                &z_valid,
                VariogramOptions {
                    n_lags,
                    max_pairs,
                    max_distance,
                },
            )?;
            model_params = Some(variogram.model_params()?);
        }

        self.x = Some(x_valid);
        self.y = Some(y_valid);
        self.z = Some(z_valid);
        self.model_params = model_params;
        self.fitted = true;
        Ok(self)
    }

    /// Estimate values at target coordinates (see `min_neighbors` for sparse targets).
    pub fn interpolate(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.require_fitted()?;
        let (x, y, z) = (self.x.as_ref(), self.y.as_ref(), self.z.as_ref());
        let (x, y, z) = (x.expect("fitted"), y.expect("fitted"), z.expect("fitted"));

        let radius_value = self.resolve_radius(self.model_params.as_ref(), x.len())?;
        if (x.len() as i64) < self.min_neighbors {
            return Err(ApbaseError::value_error(
                "IDW requires at least min_neighbors finite points",
            ));
        }

        let (target_x, target_y) = extract_target_xy(targets)?;
        if target_x.is_empty() {
            return Err(ApbaseError::value_error(
                "targets must contain at least one target point",
            ));
        }
        let (estimates, status) = run_idw(
            x.as_slice().expect("contiguous"),
            y.as_slice().expect("contiguous"),
            z.as_slice().expect("contiguous"),
            target_x.as_slice().expect("contiguous"),
            target_y.as_slice().expect("contiguous"),
            radius_value,
            self.power,
            self.max_neighbors,
            self.min_neighbors,
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                IDW_STATUS_ERRORS,
                "native IDW execution failed",
            ));
        }
        Ok(estimates)
    }

    /// Same as [`IDW::interpolate`].
    pub fn predict(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.interpolate(targets)
    }

    /// `idw(targets)`: same as [`IDW::interpolate`].
    pub fn call(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.interpolate(targets)
    }

    fn require_fitted(&self) -> Result<()> {
        if !self.fitted {
            return Err(ApbaseError::value_error(
                "call fit(x, y, z) before interpolate(targets)",
            ));
        }
        Ok(())
    }

    fn resolve_radius(&self, model_params: Option<&ModelParams>, n_points: usize) -> Result<f64> {
        if let Some(radius) = self.radius {
            return Ok(radius);
        }
        let Some(model_params) = model_params else {
            return Err(ApbaseError::value_error(
                "IDW requires a fitted variogram or explicit radius",
            ));
        };
        let radius_value = model_params.search_radius(n_points);
        if !radius_value.is_finite() || radius_value <= 0.0 {
            return Err(ApbaseError::value_error(
                "radius must be finite and greater than zero",
            ));
        }
        Ok(radius_value)
    }

    fn validate_config(
        radius: Option<f64>,
        power: f64,
        max_neighbors: i64,
        min_neighbors: i64,
    ) -> Result<()> {
        if let Some(radius) = radius {
            if !radius.is_finite() || radius <= 0.0 {
                return Err(ApbaseError::value_error(
                    "radius must be finite and greater than zero",
                ));
            }
        }
        if !power.is_finite() || power <= 0.0 {
            return Err(ApbaseError::value_error(
                "power must be finite and greater than zero",
            ));
        }
        require_neighbor_bounds(max_neighbors, min_neighbors, None)
    }
}
