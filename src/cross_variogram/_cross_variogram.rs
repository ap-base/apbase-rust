//! Generic cross-variogram between two variables.

use ndarray::Array1;

use super::_native::run_fit_cross_variogram;
use crate::common::arrays::{ArrayLike, as_float64_1d, filter_finite_xyz, validate_same_size_xyz};
use crate::common::exceptions::{ApbaseError, Result};
use crate::config::{resolve_n_threads, resolve_variogram_max_pairs_ceiling};
use crate::variogram::{DistanceLike, ModelParams, Variogram};

/// Options of [`CrossVariogram`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossVariogramOptions {
    /// Number of lag bins.
    pub n_lags: i64,
    /// Maximum sampled point pairs (`0` uses every eligible pair).
    pub max_pairs: i64,
    /// Maximum pair distance considered (`0` uses the native default extent).
    pub max_distance: f64,
    /// Maximum distance to pair an A point with its nearest B point; `None`
    /// requires exact coincidence.
    pub max_colocation_dist: Option<f64>,
}

impl Default for CrossVariogramOptions {
    fn default() -> Self {
        CrossVariogramOptions {
            n_lags: 50,
            max_pairs: 100_000,
            max_distance: 0.0,
            max_colocation_dist: None,
        }
    }
}

/// Fit a cross-variogram model between two variables (A, B).
#[derive(Clone, Debug)]
pub struct CrossVariogram {
    options: CrossVariogramOptions,
    rho: Option<f64>,
    n_pairs: Option<i64>,
    model_values: Option<Array1<f64>>,
    fitted: bool,
}

impl CrossVariogram {
    /// Create an unfitted cross-variogram; invalid options are rejected here.
    pub fn new(options: CrossVariogramOptions) -> Result<Self> {
        let cross = CrossVariogram {
            options,
            rho: None,
            n_pairs: None,
            model_values: None,
            fitted: false,
        };
        cross.validate_config()?;
        Ok(cross)
    }

    /// Number of lag bins.
    pub fn n_lags(&self) -> i64 {
        self.options.n_lags
    }

    /// Maximum sampled point pairs.
    pub fn max_pairs(&self) -> i64 {
        self.options.max_pairs
    }

    /// Maximum pair distance considered.
    pub fn max_distance(&self) -> f64 {
        self.options.max_distance
    }

    /// Colocation distance (`None` means exact coincidence).
    pub fn max_colocation_dist(&self) -> Option<f64> {
        self.options.max_colocation_dist
    }

    /// OpenMP thread count, resolved live from `apbase.config["n_threads"]`.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Create and fit a cross-variogram.
    #[allow(clippy::too_many_arguments)]
    pub fn from_data(
        xa: &(impl ArrayLike + ?Sized),
        ya: &(impl ArrayLike + ?Sized),
        za: &(impl ArrayLike + ?Sized),
        xb: &(impl ArrayLike + ?Sized),
        yb: &(impl ArrayLike + ?Sized),
        zb: &(impl ArrayLike + ?Sized),
        options: CrossVariogramOptions,
    ) -> Result<Self> {
        let mut cross = CrossVariogram::new(options)?;
        cross.fit(xa, ya, za, xb, yb, zb)?;
        Ok(cross)
    }

    /// Whether a model is available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Native model vector `[model_id, nugget, partial_sill, range, sse]`.
    pub fn model_values(&self) -> Result<&Array1<f64>> {
        self.require_fitted()?;
        Ok(self.model_values.as_ref().expect("fitted"))
    }

    /// Readable parameters of the fitted cross-variogram model.
    pub fn model_params(&self) -> Result<ModelParams> {
        self.as_variogram()?.model_params()
    }

    /// Pearson correlation of the colocated pairs.
    pub fn rho(&self) -> Result<f64> {
        self.require_fitted()?;
        Ok(self.rho.expect("fitted"))
    }

    /// Number of colocated pairs.
    pub fn n_pairs(&self) -> Result<i64> {
        self.require_fitted()?;
        Ok(self.n_pairs.expect("fitted"))
    }

    /// Fit the cross-variogram (each side is filtered to finite rows separately).
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        &mut self,
        xa: &(impl ArrayLike + ?Sized),
        ya: &(impl ArrayLike + ?Sized),
        za: &(impl ArrayLike + ?Sized),
        xb: &(impl ArrayLike + ?Sized),
        yb: &(impl ArrayLike + ?Sized),
        zb: &(impl ArrayLike + ?Sized),
    ) -> Result<&mut Self> {
        let xa_array = as_float64_1d(xa, "xa")?;
        let ya_array = as_float64_1d(ya, "ya")?;
        let za_array = as_float64_1d(za, "za")?;
        validate_same_size_xyz(&xa_array, &ya_array, &za_array)?;
        let xb_array = as_float64_1d(xb, "xb")?;
        let yb_array = as_float64_1d(yb, "yb")?;
        let zb_array = as_float64_1d(zb, "zb")?;
        validate_same_size_xyz(&xb_array, &yb_array, &zb_array)?;

        let (xa_valid, ya_valid, za_valid) = filter_finite_xyz(&xa_array, &ya_array, &za_array);
        let (xb_valid, yb_valid, zb_valid) = filter_finite_xyz(&xb_array, &yb_array, &zb_array);
        if xa_valid.is_empty() || xb_valid.is_empty() {
            return Err(ApbaseError::value_error(
                "cross-variogram requires at least one finite point on each side",
            ));
        }

        let native_colocation_dist = self.options.max_colocation_dist.unwrap_or(0.0);
        let (model_values, rho, n_pairs) = run_fit_cross_variogram(
            xa_valid.as_slice().expect("contiguous"),
            ya_valid.as_slice().expect("contiguous"),
            za_valid.as_slice().expect("contiguous"),
            xb_valid.as_slice().expect("contiguous"),
            yb_valid.as_slice().expect("contiguous"),
            zb_valid.as_slice().expect("contiguous"),
            native_colocation_dist,
            self.options.n_lags,
            self.options.max_pairs,
            self.options.max_distance,
            self.n_threads(),
        )?;

        self.rho = Some(rho);
        self.n_pairs = Some(n_pairs);
        self.model_values = Some(model_values);
        self.fitted = true;
        Ok(self)
    }

    /// Evaluate the fitted cross-semivariance for distances (same shape as the input).
    pub fn evaluate<D: DistanceLike + ?Sized>(&self, distance: &D) -> Result<D::Output> {
        self.as_variogram()?.evaluate(distance)
    }

    /// `cross(distance)`: same as [`CrossVariogram::evaluate`].
    pub fn call<D: DistanceLike + ?Sized>(&self, distance: &D) -> Result<D::Output> {
        self.evaluate(distance)
    }

    fn as_variogram(&self) -> Result<Variogram> {
        self.require_fitted()?;
        Variogram::from_model(self.model_values.as_ref().expect("fitted"))
    }

    fn require_fitted(&self) -> Result<()> {
        if !self.fitted {
            return Err(ApbaseError::value_error(
                "call fit(xa, ya, za, xb, yb, zb) before using the cross-variogram",
            ));
        }
        Ok(())
    }

    fn validate_config(&self) -> Result<()> {
        if self.options.n_lags <= 0 {
            return Err(ApbaseError::value_error("n_lags must be greater than zero"));
        }
        if self.options.max_pairs < 0 {
            return Err(ApbaseError::value_error("max_pairs must be >= 0"));
        }
        let max_pairs_ceiling = resolve_variogram_max_pairs_ceiling();
        if self.options.max_pairs > max_pairs_ceiling {
            return Err(ApbaseError::value_error(format!(
                "max_pairs must be <= {max_pairs_ceiling}"
            )));
        }
        if !self.options.max_distance.is_finite() || self.options.max_distance < 0.0 {
            return Err(ApbaseError::value_error("max_distance must be finite and >= 0"));
        }
        if let Some(distance) = self.options.max_colocation_dist {
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ApbaseError::value_error(
                    "max_colocation_dist must be None or finite and greater than zero",
                ));
            }
        }
        Ok(())
    }
}
