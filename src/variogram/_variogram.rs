//! Fit, select, and evaluate a native variogram model.

use ndarray::{Array, Array1, ArrayBase, Data, Dimension};

use super::_native::{run_evaluate_variogram, run_select_variogram, run_variogram};
use crate::common::arrays::{ArrayLike, Element, as_float64_1d, filter_finite_xyz, validate_same_size_xyz};
use crate::common::exceptions::{ApbaseError, Result, SELECT_STATUS_ERRORS, raise_status_error};
use crate::config::{resolve_n_threads, resolve_variogram_max_pairs_ceiling};

/// Native model ids and their names.
pub const VARIOGRAM_NAMES: [(i64, &str); 3] = [(1, "spherical"), (2, "exponential"), (3, "gaussian")];

pub(crate) fn variogram_name(model_id: i64) -> Option<&'static str> {
    VARIOGRAM_NAMES
        .iter()
        .find(|(id, _)| *id == model_id)
        .map(|(_, name)| *name)
}

/// Readable parameters of a variogram model.
///
/// `sse` is `None` for the ICM/LMC primary parameters, which have no fit error
/// of their own.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelParams {
    /// Native model id (`1` spherical, `2` exponential, `3` gaussian).
    pub model_id: i64,
    /// Model name.
    pub model_name: &'static str,
    /// Nugget.
    pub nugget: f64,
    /// Partial sill.
    pub partial_sill: f64,
    /// Total sill (`nugget + partial_sill`).
    pub sill: f64,
    /// Practical range.
    pub range: f64,
    /// Weighted fit SSE.
    pub sse: Option<f64>,
}

/// Below this many source points the default search radius is the full range.
pub const FULL_RANGE_RADIUS_MAX_POINTS: usize = 10_000;

impl ModelParams {
    /// Default local search radius for `n_points` source points: `range` when
    /// there are fewer than [`FULL_RANGE_RADIUS_MAX_POINTS`], `range / 3` otherwise.
    pub fn search_radius(&self, n_points: usize) -> f64 {
        if n_points < FULL_RANGE_RADIUS_MAX_POINTS {
            self.range
        } else {
            self.range / 3.0
        }
    }
}

/// Options of [`Variogram`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct VariogramOptions {
    /// Number of lag bins used by the native fitter.
    pub n_lags: i64,
    /// Maximum sampled point pairs; `0` uses every eligible pair (capped at 1,000,000).
    pub max_pairs: i64,
    /// Maximum pair distance considered; `0` uses the native default extent.
    pub max_distance: f64,
}

impl Default for VariogramOptions {
    fn default() -> Self {
        VariogramOptions {
            n_lags: 50,
            max_pairs: 100_000,
            max_distance: 0.0,
        }
    }
}

/// Distances accepted by [`Variogram::evaluate`]: a scalar or any array shape.
pub trait DistanceLike {
    /// Same shape as the input, holding semivariances.
    type Output;

    /// Distances in logical order.
    fn distances(&self) -> Vec<f64>;

    /// Rebuild the input shape around `gamma`.
    fn reshape(&self, gamma: Vec<f64>) -> Self::Output;
}

impl DistanceLike for f64 {
    type Output = f64;

    fn distances(&self) -> Vec<f64> {
        vec![*self]
    }

    fn reshape(&self, gamma: Vec<f64>) -> f64 {
        gamma[0]
    }
}

impl<T: Element> DistanceLike for [T] {
    type Output = Array1<f64>;

    fn distances(&self) -> Vec<f64> {
        self.to_f64_vec()
    }

    fn reshape(&self, gamma: Vec<f64>) -> Array1<f64> {
        Array1::from(gamma)
    }
}

impl<T: Element> DistanceLike for Vec<T> {
    type Output = Array1<f64>;

    fn distances(&self) -> Vec<f64> {
        self.to_f64_vec()
    }

    fn reshape(&self, gamma: Vec<f64>) -> Array1<f64> {
        Array1::from(gamma)
    }
}

impl<T: Element, const N: usize> DistanceLike for [T; N] {
    type Output = Array1<f64>;

    fn distances(&self) -> Vec<f64> {
        self.to_f64_vec()
    }

    fn reshape(&self, gamma: Vec<f64>) -> Array1<f64> {
        Array1::from(gamma)
    }
}

impl<S, D> DistanceLike for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Element,
    D: Dimension,
{
    type Output = Array<f64, D>;

    fn distances(&self) -> Vec<f64> {
        ArrayLike::to_f64_vec(self)
    }

    fn reshape(&self, gamma: Vec<f64>) -> Array<f64, D> {
        Array::from_shape_vec(self.raw_dim(), gamma).expect("one semivariance per distance")
    }
}

/// Fit, select, and evaluate a native variogram model.
///
/// Fits candidate semivariogram models from finite source samples and selects
/// the best native model for evaluation or kriging. Coordinates are not
/// projected internally.
///
/// ```
/// use apbase::variogram::{Variogram, VariogramOptions};
///
/// let x = [0.0, 10.0, 0.0, 10.0];
/// let y = [0.0, 0.0, 10.0, 10.0];
/// let z = [5.0, 7.0, 8.0, 10.0];
/// let mut variogram = Variogram::new(VariogramOptions { n_lags: 4, max_pairs: 0, ..Default::default() })?;
/// let fitted = variogram.fit(&x, &y, &z);
/// // Four points are too few for the native fitter; it reports an invalid model.
/// assert!(fitted.is_err());
/// # Ok::<(), apbase::common::ApbaseError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Variogram {
    n_lags: i64,
    max_pairs: i64,
    max_distance: f64,
    x: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    z: Option<Array1<f64>>,
    model_values: Option<Array1<f64>>,
    model_params: Option<ModelParams>,
    fitted: bool,
}

impl Variogram {
    /// Create an unfitted variogram; invalid options are rejected here.
    pub fn new(options: VariogramOptions) -> Result<Self> {
        let variogram = Variogram {
            n_lags: options.n_lags,
            max_pairs: options.max_pairs,
            max_distance: options.max_distance,
            x: None,
            y: None,
            z: None,
            model_values: None,
            model_params: None,
            fitted: false,
        };
        variogram.validate_config()?;
        Ok(variogram)
    }

    /// Number of lag bins used by the native fitter.
    pub fn n_lags(&self) -> i64 {
        self.n_lags
    }

    /// Maximum sampled point pairs.
    pub fn max_pairs(&self) -> i64 {
        self.max_pairs
    }

    /// Maximum pair distance considered by the fitter.
    pub fn max_distance(&self) -> f64 {
        self.max_distance
    }

    /// OpenMP thread count, resolved live from `apbase.config["n_threads"]`.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Create and fit a variogram from source data.
    pub fn from_data(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        options: VariogramOptions,
    ) -> Result<Self> {
        let mut variogram = Variogram::new(options)?;
        variogram.fit(x, y, z)?;
        Ok(variogram)
    }

    /// Create a variogram from an existing native model vector
    /// `[model_id, nugget, partial_sill, range, sse]`.
    pub fn from_model(model_values: &(impl ArrayLike + ?Sized)) -> Result<Self> {
        let mut variogram = Variogram::new(VariogramOptions::default())?;
        variogram.set_model(model_values)?;
        Ok(variogram)
    }

    /// Whether a model is already available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Native model vector `[model_id, nugget, partial_sill, range, sse]`.
    pub fn model_values(&self) -> Result<&Array1<f64>> {
        self.require_fitted()?;
        Ok(self
            .model_values
            .as_ref()
            .expect("fitted variogram has model values"))
    }

    /// Readable parameters for the selected variogram model.
    pub fn model_params(&self) -> Result<ModelParams> {
        self.require_fitted()?;
        Ok(self
            .model_params
            .clone()
            .expect("fitted variogram has model params"))
    }

    /// Selected model range.
    pub fn range(&self) -> Result<f64> {
        Ok(self.model_params()?.range)
    }

    /// Fit candidate variogram models from source samples (non-finite rows are ignored).
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
        if x_valid.is_empty() {
            return Err(ApbaseError::value_error(
                "variogram requires at least one finite point",
            ));
        }

        let model_values = run_variogram(
            x_valid.as_slice().expect("contiguous"),
            y_valid.as_slice().expect("contiguous"),
            z_valid.as_slice().expect("contiguous"),
            self.n_lags,
            self.max_pairs,
            self.max_distance,
            self.n_threads(),
        )?;
        let model_params = Variogram::select_model_params(&model_values)?;

        self.x = Some(x_valid);
        self.y = Some(y_valid);
        self.z = Some(z_valid);
        self.model_values = Some(model_values);
        self.model_params = Some(model_params);
        self.fitted = true;
        Ok(self)
    }

    /// Attach an existing native model vector and select its parameters.
    pub fn set_model(&mut self, model_values: &(impl ArrayLike + ?Sized)) -> Result<&mut Self> {
        let model_array = Variogram::normalize_model_values(model_values)?;
        let model_params = Variogram::select_model_params(&model_array)?;
        self.x = None;
        self.y = None;
        self.z = None;
        self.model_values = Some(model_array);
        self.model_params = Some(model_params);
        self.fitted = true;
        Ok(self)
    }

    /// Evaluate semivariance for distances; the output keeps the input shape.
    pub fn evaluate<D: DistanceLike + ?Sized>(&self, distance: &D) -> Result<D::Output> {
        self.require_fitted()?;
        let (gamma, status) = run_evaluate_variogram(
            self.model_values()?.as_slice().expect("contiguous"),
            &distance.distances(),
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                SELECT_STATUS_ERRORS,
                "error evaluating variogram",
            ));
        }
        Ok(distance.reshape(gamma))
    }

    /// `variogram(distance)`: same as [`Variogram::evaluate`].
    pub fn call<D: DistanceLike + ?Sized>(&self, distance: &D) -> Result<D::Output> {
        self.evaluate(distance)
    }

    fn require_fitted(&self) -> Result<()> {
        if !self.fitted {
            return Err(ApbaseError::value_error(
                "call fit(x, y, z) or set_model(model_values) before using the variogram",
            ));
        }
        Ok(())
    }

    fn validate_config(&self) -> Result<()> {
        if self.n_lags <= 0 {
            return Err(ApbaseError::value_error("n_lags must be greater than zero"));
        }
        if self.max_pairs < 0 {
            return Err(ApbaseError::value_error("max_pairs must be >= 0"));
        }
        let max_pairs_ceiling = resolve_variogram_max_pairs_ceiling();
        if self.max_pairs > max_pairs_ceiling {
            return Err(ApbaseError::value_error(format!(
                "max_pairs must be <= {max_pairs_ceiling}"
            )));
        }
        if !self.max_distance.is_finite() || self.max_distance < 0.0 {
            return Err(ApbaseError::value_error("max_distance must be finite and >= 0"));
        }
        Ok(())
    }

    pub(crate) fn normalize_model_values(model_values: &(impl ArrayLike + ?Sized)) -> Result<Array1<f64>> {
        let model_array = as_float64_1d(model_values, "model_values")?;
        if model_array.len() != 5 {
            return Err(ApbaseError::value_error("model_values must contain 5 values"));
        }
        Ok(model_array)
    }

    pub(crate) fn select_model_params(model_values: &Array1<f64>) -> Result<ModelParams> {
        let (params, status) = run_select_variogram(model_values.as_slice().expect("contiguous"))?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                SELECT_STATUS_ERRORS,
                "error selecting variogram",
            ));
        }

        let model_id = crate::common::_numpy::rint(params[0]) as i64;
        let Some(model_name) = variogram_name(model_id) else {
            return Err(ApbaseError::value_error(format!(
                "invalid model returned by the native extension: {model_id}"
            )));
        };
        Ok(ModelParams {
            model_id,
            model_name,
            nugget: params[1],
            partial_sill: params[2],
            sill: params[3],
            range: params[4],
            sse: Some(params[5]),
        })
    }
}
