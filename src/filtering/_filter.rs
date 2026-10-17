//! Filter projected spatial samples with global IQR and local z-statistics.

use std::fmt;
use std::str::FromStr;

use ndarray::Array1;

use super::_native::{run_filter_mask, run_local_zstats};
use crate::common::_numpy;
use crate::common::arrays::{ArrayLike, as_float64_1d, finite_mask_xyz, select, validate_same_size_xyz};
use crate::common::exceptions::{ApbaseError, FILTER_STATUS_ERRORS, Result, raise_status_error};
use crate::config::resolve_n_threads;
use crate::variogram::{ModelParams, Variogram, VariogramOptions};

/// Default maximum local neighbors per source point.
pub const DEFAULT_MAX_NEIGHBORS: i64 = 120;
/// Default spatial grid factor of the local kernel.
pub const DEFAULT_CELL_FACTOR: i64 = 3;
/// Default IQR whisker of the global filter.
pub const DEFAULT_GLOBAL_IQR_WHISKER: f64 = 3.0;

const VARIOGRAM_N_LAGS: i64 = 20;
const VARIOGRAM_MAX_PAIRS: i64 = 100_000;
const VARIOGRAM_MAX_DISTANCE: f64 = 0.0;

/// Filtering strictness, loosest to strictest (native threshold profile ids 1..4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FilterLevel {
    /// `"light"`.
    #[default]
    Light,
    /// `"moderate"`.
    Moderate,
    /// `"strict"`.
    Strict,
    /// `"aggressive"`.
    Aggressive,
}

/// `FILTER_LEVELS`: names and native threshold profile ids.
pub const FILTER_LEVELS: [(&str, i64); 4] = [("light", 1), ("moderate", 2), ("strict", 3), ("aggressive", 4)];

impl FilterLevel {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            FilterLevel::Light => "light",
            FilterLevel::Moderate => "moderate",
            FilterLevel::Strict => "strict",
            FilterLevel::Aggressive => "aggressive",
        }
    }

    /// Native threshold profile id consumed by `viz_filter_mask`.
    pub fn code(self) -> i64 {
        FILTER_LEVELS
            .iter()
            .find(|(name, _)| *name == self.as_str())
            .map(|(_, code)| *code)
            .expect("known level")
    }
}

impl fmt::Display for FilterLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for FilterLevel {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl FromStr for FilterLevel {
    type Err = ApbaseError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "light" => Ok(FilterLevel::Light),
            "moderate" => Ok(FilterLevel::Moderate),
            "strict" => Ok(FilterLevel::Strict),
            "aggressive" => Ok(FilterLevel::Aggressive),
            _ => {
                let valid: Vec<String> = FILTER_LEVELS
                    .iter()
                    .map(|(name, _)| format!("'{name}'"))
                    .collect();
                Err(ApbaseError::value_error(format!(
                    "filter_level must be one of {}, got {}",
                    valid.join(", "),
                    crate::config::quote_str(value)
                )))
            }
        }
    }
}

/// Per-point local filtering statistics, aligned with the original input rows.
#[derive(Clone, Debug, PartialEq)]
pub struct FilterStatistics {
    /// Local percent-difference statistic.
    pub pct_diff: Array1<f64>,
    /// Local z-score.
    pub local_z: Array1<f64>,
    /// Local probability statistic.
    pub local_prob: Array1<f64>,
}

/// Options of [`SpatialFilter`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialFilterOptions {
    /// Local search radius; `None` derives `range / 3` from a fitted variogram.
    pub radius: Option<f64>,
    /// Strictness level.
    pub filter_level: FilterLevel,
    /// Whether to remove global IQR outliers before local filtering.
    pub apply_global_iqr: bool,
    /// Maximum local neighbors per source point; `0` means uncapped.
    pub max_neighbors: i64,
    /// Fraction `(0, 1]` of in-radius candidates considered.
    pub candidate_fraction: f64,
    /// Spatial grid factor used by the local kernel.
    pub cell_factor: i64,
    /// IQR whisker of the optional global filter.
    pub global_iqr_whisker: f64,
    /// Distance below which points are collapsed as duplicates; `0.0` disables it.
    pub dedup_radius: f64,
    /// Value-closeness gate combined with `dedup_radius`.
    pub dedup_z_tol: f64,
}

impl Default for SpatialFilterOptions {
    fn default() -> Self {
        SpatialFilterOptions {
            radius: None,
            filter_level: FilterLevel::Light,
            apply_global_iqr: true,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            candidate_fraction: 1.0,
            cell_factor: DEFAULT_CELL_FACTOR,
            global_iqr_whisker: DEFAULT_GLOBAL_IQR_WHISKER,
            dedup_radius: 0.0,
            dedup_z_tol: 0.0,
        }
    }
}

/// Filter projected spatial samples with global IQR and local z-statistics.
///
/// ```
/// use apbase::filtering::{SpatialFilter, SpatialFilterOptions};
///
/// let x = [0.0, 10.0, 0.0, 10.0];
/// let y = [0.0, 0.0, 10.0, 10.0];
/// let z = [5.0, 7.0, 8.0, 10.0];
/// let mut filter = SpatialFilter::new(SpatialFilterOptions {
///     radius: Some(20.0),
///     apply_global_iqr: false,
///     ..Default::default()
/// })?;
/// let (x_keep, y_keep, z_keep) = filter.fit(&x, &y, &z)?.filter()?;
/// assert_eq!(x_keep.len(), y_keep.len());
/// assert_eq!(x_keep.len(), z_keep.len());
/// # Ok::<(), apbase::common::ApbaseError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SpatialFilter {
    options: SpatialFilterOptions,
    x: Option<Array1<f64>>,
    y: Option<Array1<f64>>,
    z: Option<Array1<f64>>,
    keep_mask: Option<Array1<bool>>,
    pct_diff: Option<Array1<f64>>,
    local_z: Option<Array1<f64>>,
    local_prob: Option<Array1<f64>>,
    model_params: Option<ModelParams>,
    fitted: bool,
}

impl SpatialFilter {
    /// Create an unfitted spatial filter; invalid options are rejected here.
    pub fn new(options: SpatialFilterOptions) -> Result<Self> {
        SpatialFilter::validate_config(&options)?;
        Ok(SpatialFilter {
            options,
            x: None,
            y: None,
            z: None,
            keep_mask: None,
            pct_diff: None,
            local_z: None,
            local_prob: None,
            model_params: None,
            fitted: false,
        })
    }

    /// Local search radius (`None` derives it from a fitted variogram).
    pub fn radius(&self) -> Option<f64> {
        self.options.radius
    }

    /// Filtering strictness.
    pub fn filter_level(&self) -> FilterLevel {
        self.options.filter_level
    }

    /// Whether global IQR outliers are removed before local filtering.
    pub fn apply_global_iqr(&self) -> bool {
        self.options.apply_global_iqr
    }

    /// Maximum local neighbors used per source point.
    pub fn max_neighbors(&self) -> i64 {
        self.options.max_neighbors
    }

    /// Fraction of in-radius candidates sampled locally.
    pub fn candidate_fraction(&self) -> f64 {
        self.options.candidate_fraction
    }

    /// Spatial grid factor used by the native local kernel.
    pub fn cell_factor(&self) -> i64 {
        self.options.cell_factor
    }

    /// IQR whisker used by the optional global filter.
    pub fn global_iqr_whisker(&self) -> f64 {
        self.options.global_iqr_whisker
    }

    /// Spatial duplicate-collapse radius.
    pub fn dedup_radius(&self) -> f64 {
        self.options.dedup_radius
    }

    /// Value-closeness gate paired with `dedup_radius`.
    pub fn dedup_z_tol(&self) -> f64 {
        self.options.dedup_z_tol
    }

    /// OpenMP thread count, resolved live from `apbase.config["n_threads"]`.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Create and fit a spatial filter.
    pub fn from_data(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        options: SpatialFilterOptions,
    ) -> Result<Self> {
        let mut filter = SpatialFilter::new(options)?;
        filter.fit(x, y, z)?;
        Ok(filter)
    }

    /// Whether source data and filtering outputs are available.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Parameters of the variogram used by the default radius.
    pub fn model_params(&self) -> Result<ModelParams> {
        self.require_fitted()?;
        self.model_params.clone().ok_or_else(|| {
            ApbaseError::value_error(
                "SpatialFilter was fit with an explicit radius; no variogram model was fitted",
            )
        })
    }

    /// Copies of the per-input-row local filtering diagnostics.
    pub fn statistics(&self) -> Result<FilterStatistics> {
        self.require_fitted()?;
        Ok(FilterStatistics {
            pct_diff: self.pct_diff.clone().expect("fitted"),
            local_z: self.local_z.clone().expect("fitted"),
            local_prob: self.local_prob.clone().expect("fitted"),
        })
    }

    /// Run global and local filtering for source samples.
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

        let finite_mask = finite_mask_xyz(&x_array, &y_array, &z_array);
        if !finite_mask.iter().any(|keep| *keep) {
            return Err(ApbaseError::value_error(
                "SpatialFilter requires at least one finite point",
            ));
        }

        let mut input_mask = finite_mask.clone();
        let mut x_valid = select(&x_array, &finite_mask);
        let mut y_valid = select(&y_array, &finite_mask);
        let mut z_valid = select(&z_array, &finite_mask);

        if self.options.apply_global_iqr {
            let global_mask = SpatialFilter::iqr_mask(&z_valid, self.options.global_iqr_whisker);
            let finite_indices = finite_mask
                .iter()
                .enumerate()
                .filter(|(_, keep)| **keep)
                .map(|(index, _)| index);
            for (index, keep) in finite_indices.zip(global_mask.iter()) {
                if !keep {
                    input_mask[index] = false;
                }
            }
            x_valid = select(&x_valid, &global_mask);
            y_valid = select(&y_valid, &global_mask);
            z_valid = select(&z_valid, &global_mask);
        }

        if x_valid.is_empty() {
            return Err(ApbaseError::value_error(
                "SpatialFilter requires at least one point after global filtering",
            ));
        }

        let mut model_params = None;
        if self.options.radius.is_none() {
            let variogram = Variogram::from_data(
                &x_valid,
                &y_valid,
                &z_valid,
                VariogramOptions {
                    n_lags: VARIOGRAM_N_LAGS,
                    max_pairs: VARIOGRAM_MAX_PAIRS,
                    max_distance: VARIOGRAM_MAX_DISTANCE,
                },
            )?;
            model_params = Some(variogram.model_params()?);
        }

        let radius_value = self.resolve_radius(model_params.as_ref())?;
        let (pct_diff, local_z, local_prob, status) = run_local_zstats(
            x_valid.as_slice().expect("contiguous"),
            y_valid.as_slice().expect("contiguous"),
            z_valid.as_slice().expect("contiguous"),
            radius_value,
            self.options.cell_factor,
            self.options.max_neighbors,
            self.options.candidate_fraction,
            self.options.dedup_radius,
            self.options.dedup_z_tol,
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                FILTER_STATUS_ERRORS,
                "native spatial filter execution failed",
            ));
        }

        let (local_mask, status) = run_filter_mask(
            pct_diff.as_slice().expect("contiguous"),
            local_prob.as_slice().expect("contiguous"),
            local_z.as_slice().expect("contiguous"),
            self.options.filter_level.code(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                FILTER_STATUS_ERRORS,
                "native spatial filter mask failed",
            ));
        }

        let size = x_array.len();
        let mut keep_mask = Array1::from_elem(size, false);
        let mut full_pct_diff = Array1::from_elem(size, f64::NAN);
        let mut full_local_z = Array1::from_elem(size, f64::NAN);
        let mut full_local_prob = Array1::from_elem(size, f64::NAN);
        let kept_rows = input_mask
            .iter()
            .enumerate()
            .filter(|(_, keep)| **keep)
            .map(|(index, _)| index);
        for (position, index) in kept_rows.enumerate() {
            keep_mask[index] = local_mask[position];
            full_pct_diff[index] = pct_diff[position];
            full_local_z[index] = local_z[position];
            full_local_prob[index] = local_prob[position];
        }

        self.x = Some(x_array);
        self.y = Some(y_array);
        self.z = Some(z_array);
        self.keep_mask = Some(keep_mask);
        self.pct_diff = Some(full_pct_diff);
        self.local_z = Some(full_local_z);
        self.local_prob = Some(full_local_prob);
        self.model_params = model_params;
        self.fitted = true;
        Ok(self)
    }

    /// Boolean keep mask aligned with the original input rows.
    pub fn mask(&self) -> Result<Array1<bool>> {
        self.require_fitted()?;
        Ok(self.keep_mask.clone().expect("fitted"))
    }

    /// Source coordinates and values where the keep mask is `true`.
    pub fn filter(&self) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        self.require_fitted()?;
        let keep_mask = self.keep_mask.as_ref().expect("fitted");
        Ok((
            select(self.x.as_ref().expect("fitted"), keep_mask),
            select(self.y.as_ref().expect("fitted"), keep_mask),
            select(self.z.as_ref().expect("fitted"), keep_mask),
        ))
    }

    /// Fit the filter and return the boolean keep mask.
    pub fn fit_mask(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<Array1<bool>> {
        self.fit(x, y, z)?.mask()
    }

    /// Fit the filter and return filtered source arrays.
    pub fn fit_filter(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        self.fit(x, y, z)?.filter()
    }

    /// `filter(x, y, z)`: same as [`SpatialFilter::fit_filter`].
    pub fn call(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        self.fit_filter(x, y, z)
    }

    fn resolve_radius(&self, model_params: Option<&ModelParams>) -> Result<f64> {
        if let Some(radius) = self.options.radius {
            return Ok(radius);
        }
        let Some(model_params) = model_params else {
            return Err(ApbaseError::value_error(
                "SpatialFilter requires a fitted variogram or explicit radius",
            ));
        };
        let radius_value = model_params.range / 3.0;
        if !radius_value.is_finite() || radius_value <= 0.0 {
            return Err(ApbaseError::value_error(
                "radius must be finite and greater than zero",
            ));
        }
        Ok(radius_value)
    }

    fn require_fitted(&self) -> Result<()> {
        if !self.fitted {
            return Err(ApbaseError::value_error(
                "call fit(x, y, z) before reading filtering results",
            ));
        }
        Ok(())
    }

    fn validate_config(options: &SpatialFilterOptions) -> Result<()> {
        if let Some(radius) = options.radius {
            if !radius.is_finite() || radius <= 0.0 {
                return Err(ApbaseError::value_error(
                    "radius must be finite and greater than zero",
                ));
            }
        }
        if options.max_neighbors < 0 {
            return Err(ApbaseError::value_error("max_neighbors must be >= 0"));
        }
        let fraction = options.candidate_fraction;
        if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
            return Err(ApbaseError::value_error("candidate_fraction must be in (0, 1]"));
        }
        if options.cell_factor <= 0 {
            return Err(ApbaseError::value_error("cell_factor must be greater than zero"));
        }
        if !options.global_iqr_whisker.is_finite() || options.global_iqr_whisker < 0.0 {
            return Err(ApbaseError::value_error(
                "global_iqr_whisker must be finite and >= 0",
            ));
        }
        if !options.dedup_radius.is_finite() || options.dedup_radius < 0.0 {
            return Err(ApbaseError::value_error("dedup_radius must be finite and >= 0"));
        }
        if !options.dedup_z_tol.is_finite() || options.dedup_z_tol < 0.0 {
            return Err(ApbaseError::value_error("dedup_z_tol must be finite and >= 0"));
        }
        Ok(())
    }

    fn iqr_mask(z: &Array1<f64>, whisker: f64) -> Array1<bool> {
        let quartiles = _numpy::quantile(z.as_slice().expect("contiguous"), &[0.25, 0.75]);
        let (q1, q3) = (quartiles[0], quartiles[1]);
        let iqr = q3 - q1;
        let lower = q1 - whisker * iqr;
        let upper = q3 + whisker * iqr;
        z.iter().map(|value| *value >= lower && *value <= upper).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_level_parsing_and_message() {
        assert_eq!("strict".parse::<FilterLevel>().unwrap(), FilterLevel::Strict);
        assert_eq!(FilterLevel::Aggressive.code(), 4);
        assert_eq!(
            "extreme".parse::<FilterLevel>().unwrap_err().message(),
            "filter_level must be one of 'light', 'moderate', 'strict', 'aggressive', got 'extreme'"
        );
    }

    #[test]
    fn invalid_options_are_rejected_with_their_messages() {
        let message = |options: SpatialFilterOptions| SpatialFilter::new(options).unwrap_err().to_string();
        assert_eq!(
            message(SpatialFilterOptions {
                radius: Some(0.0),
                ..Default::default()
            }),
            "radius must be finite and greater than zero"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                max_neighbors: -1,
                ..Default::default()
            }),
            "max_neighbors must be >= 0"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                candidate_fraction: 1.5,
                ..Default::default()
            }),
            "candidate_fraction must be in (0, 1]"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                cell_factor: 0,
                ..Default::default()
            }),
            "cell_factor must be greater than zero"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                global_iqr_whisker: f64::NAN,
                ..Default::default()
            }),
            "global_iqr_whisker must be finite and >= 0"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                dedup_radius: -1.0,
                ..Default::default()
            }),
            "dedup_radius must be finite and >= 0"
        );
        assert_eq!(
            message(SpatialFilterOptions {
                dedup_z_tol: -1.0,
                ..Default::default()
            }),
            "dedup_z_tol must be finite and >= 0"
        );
    }

    #[test]
    fn iqr_mask_uses_linear_quartiles() {
        let z = ndarray::array![1.0, 2.0, 3.0, 4.0, 100.0];
        let mask = SpatialFilter::iqr_mask(&z, 3.0);
        assert_eq!(mask, ndarray::array![true, true, true, true, false]);
    }
}
