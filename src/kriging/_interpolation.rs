//! Local ordinary kriging.

use ndarray::Array1;

use super::_native::run_kriging;
use crate::common::arrays::{
    ArrayLike, Targets, as_float64_1d, extract_target_xy, filter_finite_xyz, validate_same_size_xyz,
};
use crate::common::exceptions::{ApbaseError, KRIGING_STATUS_ERRORS, Result, raise_status_error};
use crate::common::validation::require_neighbor_bounds;
use crate::config::{resolve_max_neighbors_ceiling, resolve_n_threads, resolve_variogram_profile};
use crate::variogram::{ModelParams, Variogram, VariogramOptions};

/// Options of [`Kriging`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct KrigingOptions {
    /// Local search radius; `None` uses [`ModelParams::search_radius`] of the selected variogram.
    pub radius: Option<f64>,
    /// Maximum source points per target (capped at `apbase.config["max_neighbors_ceiling"]`).
    pub max_neighbors: i64,
    /// Minimum sources within the radius; below it the `max_neighbors` nearest
    /// sources are used regardless of distance.
    pub min_neighbors: i64,
    /// Optional native variogram model vector with five values.
    pub model_values: Option<Array1<f64>>,
}

impl Default for KrigingOptions {
    fn default() -> Self {
        KrigingOptions {
            radius: None,
            max_neighbors: 40,
            min_neighbors: 3,
            model_values: None,
        }
    }
}

/// Fit a variogram model and interpolate values with local ordinary kriging.
///
/// ```
/// use apbase::kriging::{Kriging, KrigingOptions};
/// use ndarray::array;
///
/// let model = array![1.0, 0.0, 1.0, 30.0, 0.0]; // spherical, nugget 0, sill 1, range 30
/// let x = [0.0, 10.0, 0.0, 10.0];
/// let y = [0.0, 0.0, 10.0, 10.0];
/// let z = [5.0, 7.0, 8.0, 10.0];
/// let targets = array![[5.0, 5.0]];
/// let values = Kriging::new(KrigingOptions { radius: Some(20.0), model_values: Some(model), ..Default::default() })?
///     .fit(&x, &y, &z)?
///     .interpolate(&targets)?;
/// assert!(values[0].is_finite());
/// # Ok::<(), apbase::common::ApbaseError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Kriging {
    radius: Option<f64>,
    max_neighbors: i64,
    min_neighbors: i64,
    initial_model_values: Option<Array1<f64>>,
    x: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    z: Option<Array1<f64>>,
    model_values: Option<Array1<f64>>,
    model_params: Option<ModelParams>,
    fitted: bool,
}

impl Kriging {
    /// Create an unfitted kriging interpolator; invalid options are rejected here.
    pub fn new(options: KrigingOptions) -> Result<Self> {
        let initial_model_values = match &options.model_values {
            Some(values) => Some(Variogram::normalize_model_values(values)?),
            None => None,
        };
        Kriging::validate_config(options.max_neighbors, options.min_neighbors)?;
        if let Some(radius) = options.radius {
            if !radius.is_finite() || radius <= 0.0 {
                return Err(ApbaseError::value_error(
                    "radius must be finite and greater than zero",
                ));
            }
        }
        Ok(Kriging {
            radius: options.radius,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
            initial_model_values,
            x: None,
            y: None,
            z: None,
            model_values: None,
            model_params: None,
            fitted: false,
        })
    }

    /// Local search radius (`None` derives it from the variogram range).
    pub fn radius(&self) -> Option<f64> {
        self.radius
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

    /// Create and fit a kriging interpolator.
    pub fn from_data(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        options: KrigingOptions,
    ) -> Result<Self> {
        let mut kriging = Kriging::new(options)?;
        kriging.fit(x, y, z)?;
        Ok(kriging)
    }

    /// Whether source data and a variogram model are available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Best fitted variogram model values.
    pub fn model_values(&self) -> Result<&Array1<f64>> {
        self.require_fitted()?;
        Ok(self.model_values.as_ref().expect("fitted"))
    }

    /// Readable parameters for the selected variogram model.
    pub fn model_params(&self) -> Result<ModelParams> {
        self.require_fitted()?;
        Ok(self.model_params.clone().expect("fitted"))
    }

    /// Store finite source data and fit or attach a variogram model.
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
                "kriging requires at least min_neighbors finite points",
            ));
        }

        let variogram = match &self.initial_model_values {
            None => {
                let (n_lags, max_pairs, max_distance) = resolve_variogram_profile();
                Variogram::from_data(
                    &x_valid,
                    &y_valid,
                    &z_valid,
                    VariogramOptions {
                        n_lags,
                        max_pairs,
                        max_distance,
                    },
                )?
            }
            Some(model_values) => Variogram::from_model(model_values)?,
        };

        self.x = Some(x_valid);
        self.y = Some(y_valid);
        self.z = Some(z_valid);
        self.model_values = Some(variogram.model_values()?.clone());
        self.model_params = Some(variogram.model_params()?);
        self.fitted = true;
        Ok(self)
    }

    /// Estimate values at target coordinates (see `min_neighbors` for sparse targets).
    pub fn interpolate(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.require_fitted()?;
        let x = self.x.as_ref().expect("fitted");
        let y = self.y.as_ref().expect("fitted");
        let z = self.z.as_ref().expect("fitted");
        let model_values = self.model_values.as_ref().expect("fitted");

        let (target_x, target_y) = extract_target_xy(targets)?;
        let radius_value = self.resolve_radius()?;
        let (estimates, status) = run_kriging(
            x.as_slice().expect("contiguous"),
            y.as_slice().expect("contiguous"),
            z.as_slice().expect("contiguous"),
            target_x.as_slice().expect("contiguous"),
            target_y.as_slice().expect("contiguous"),
            model_values.as_slice().expect("contiguous"),
            radius_value,
            self.max_neighbors,
            self.min_neighbors,
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                KRIGING_STATUS_ERRORS,
                "native kriging execution failed",
            ));
        }
        Ok(estimates)
    }

    /// Same as [`Kriging::interpolate`].
    pub fn predict(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.interpolate(targets)
    }

    /// Fit the interpolator and immediately estimate target values.
    pub fn fit_interpolate(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        targets: &(impl Targets + ?Sized),
    ) -> Result<Array1<f64>> {
        self.fit(x, y, z)?.interpolate(targets)
    }

    /// `kriging(targets)`: same as [`Kriging::interpolate`].
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

    /// Search radius used by `interpolate`: the explicit one or the variogram default.
    pub(crate) fn resolve_radius(&self) -> Result<f64> {
        self.require_fitted()?;
        if let Some(radius) = self.radius {
            return Ok(radius);
        }
        let n_points = self.x.as_ref().expect("fitted").len();
        let radius_value = self
            .model_params
            .as_ref()
            .expect("fitted")
            .search_radius(n_points);
        if !radius_value.is_finite() || radius_value <= 0.0 {
            return Err(ApbaseError::value_error(
                "radius must be finite and greater than zero",
            ));
        }
        Ok(radius_value)
    }

    fn validate_config(max_neighbors: i64, min_neighbors: i64) -> Result<()> {
        require_neighbor_bounds(
            max_neighbors,
            min_neighbors,
            Some(resolve_max_neighbors_ceiling()),
        )
    }
}
