//! Leave-one-out cross-validation for IDW and ordinary kriging, plus
//! [`select_best_model`] to pick the statistically better method.

use std::fmt;
use std::str::FromStr;

use indexmap::IndexMap;
use ndarray::{Array1, Array2, Array3};

use super::_native::{
    CokrigingCrossValidateInputs, NativeMethodResult, run_cokriging_cross_validate, run_cross_validate,
};
use crate::cokriging::{
    CoKriging, CoKrigingMethod, CoKrigingModelParams, CoKrigingOptions, Fit, SecondaryInput,
    is_no_collocated_samples_error,
};
use crate::common::_numpy;
use crate::common::_numpy::random::Generator;
use crate::common::arrays::{ArrayLike, as_float64_1d, filter_finite_xyz, validate_same_size_xyz};
use crate::common::exceptions::{
    ApbaseError, COKRIGING_STATUS_ERRORS, CROSS_VALIDATE_STATUS_ERRORS, ErrorKind, Result, raise_status_error,
};
use crate::common::validation::require_neighbor_bounds;
use crate::config::{resolve_max_neighbors_ceiling, resolve_n_threads, resolve_variogram_profile};
use crate::variogram::{ModelParams, Variogram, VariogramOptions};

/// Interpolation method selected between IDW and ordinary kriging.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// `"idw"`.
    Idw,
    /// `"kriging"`.
    Kriging,
}

impl Method {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Idw => "idw",
            Method::Kriging => "kriging",
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for Method {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Error metric used by [`select_best_model`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Root mean squared error.
    #[default]
    Rmse,
    /// Mean absolute error.
    Mae,
}

impl Metric {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            Metric::Rmse => "rmse",
            Metric::Mae => "mae",
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for Metric {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl FromStr for Metric {
    type Err = ApbaseError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "rmse" => Ok(Metric::Rmse),
            "mae" => Ok(Metric::Mae),
            _ => Err(ApbaseError::value_error("metric must be \"rmse\" or \"mae\"")),
        }
    }
}

/// Leave-one-out results for a single interpolation method.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodCrossValidation {
    /// Mean absolute error over finite predictions.
    pub mae: f64,
    /// Root mean squared error over finite predictions.
    pub rmse: f64,
    /// Coefficient of determination over finite predictions.
    pub r2: f64,
    /// Number of finite predictions contributing to the metrics.
    pub n_valid: i64,
    /// Per-sample predicted values.
    pub predicted: Array1<f64>,
    /// Held-out actual values.
    pub actual: Array1<f64>,
}

/// Leave-one-out cross-validation results for IDW and ordinary kriging.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossValidationResult {
    /// IDW results.
    pub idw: MethodCrossValidation,
    /// Ordinary kriging results.
    pub kriging: MethodCrossValidation,
    /// Number of held-out samples evaluated.
    pub n_samples: i64,
    /// Shared local search radius.
    pub radius: f64,
    /// Fitted variogram parameters.
    pub model_params: ModelParams,
}

/// Leave-one-out results for collocated/ICM/LMC cokriging.
#[derive(Clone, Debug, PartialEq)]
pub struct CokrigingCrossValidationResult {
    /// Results keyed by method name (`"collocated"`, `"icm"`, `"lmc"`).
    pub by_method: IndexMap<String, MethodCrossValidation>,
    /// Number of held-out primary samples evaluated.
    pub n_samples: i64,
    /// Shared local search radius.
    pub radius: f64,
    /// Fitted cokriging parameters keyed by method name.
    pub model_params: IndexMap<String, CoKrigingModelParams>,
}

/// Statistically-supported IDW-vs-kriging pick.
#[derive(Clone, Debug, PartialEq)]
pub struct BestModelResult {
    /// Selected method.
    pub method: Method,
    /// Metric used for comparison.
    pub metric: Metric,
    /// IDW paired score.
    pub idw_score: f64,
    /// Kriging paired score.
    pub kriging_score: f64,
    /// Absolute improvement of the selected method.
    pub margin: f64,
    /// Relative improvement of the selected method.
    pub relative_margin: f64,
    /// Whether the bootstrap confidence interval excludes zero.
    pub significant: bool,
    /// Confidence level used for the interval.
    pub confidence_level: f64,
    /// Lower bound of the idw-minus-kriging score difference.
    pub ci_low: f64,
    /// Upper bound of the idw-minus-kriging score difference.
    pub ci_high: f64,
    /// Number of paired finite samples used for comparison.
    pub n_common_valid: i64,
}

/// Statistically-supported best-of-N pick from a named method mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct MultiMethodBestModelResult {
    /// Selected method.
    pub method: String,
    /// Second-best method.
    pub runner_up: String,
    /// Metric used for comparison.
    pub metric: Metric,
    /// Paired scores over common finite held-out samples.
    pub scores: IndexMap<String, f64>,
    /// Absolute improvement of `method` over `runner_up`.
    pub margin: f64,
    /// Relative improvement of `method` over `runner_up`.
    pub relative_margin: f64,
    /// Whether the bootstrap confidence interval excludes zero.
    pub significant: bool,
    /// Confidence level used for the interval.
    pub confidence_level: f64,
    /// Lower bound of the runner-up-minus-winner score difference.
    pub ci_low: f64,
    /// Upper bound of the runner-up-minus-winner score difference.
    pub ci_high: f64,
    /// Number of paired finite samples used for comparison.
    pub n_common_valid: i64,
}

/// Options of [`cross_validate`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossValidateOptions {
    /// Maximum number of held-out samples; larger datasets are subsampled.
    pub max_points: i64,
    /// Seed of the subsample draw; `None` uses fresh entropy.
    pub seed: Option<u64>,
    /// Shared local search radius; `None` uses [`ModelParams::search_radius`].
    pub radius: Option<f64>,
    /// IDW distance exponent.
    pub power: f64,
    /// Maximum neighbors shared by both methods.
    pub max_neighbors: i64,
    /// Minimum neighbors shared by both methods.
    pub min_neighbors: i64,
    /// Optional native variogram model vector (5 values).
    pub model_values: Option<Array1<f64>>,
}

impl Default for CrossValidateOptions {
    fn default() -> Self {
        CrossValidateOptions {
            max_points: 500,
            seed: None,
            radius: None,
            power: 2.0,
            max_neighbors: 40,
            min_neighbors: 3,
            model_values: None,
        }
    }
}

/// Options of [`cross_validate_cokriging`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossValidateCokrigingOptions {
    /// Maximum number of held-out samples.
    pub max_points: i64,
    /// Seed of the subsample draw; `None` uses fresh entropy.
    pub seed: Option<u64>,
    /// Shared local search radius; `None` uses [`ModelParams::search_radius`] of the primary.
    pub radius: Option<f64>,
    /// Maximum neighbors shared by the three methods.
    pub max_neighbors: i64,
    /// Minimum neighbors shared by the three methods.
    pub min_neighbors: i64,
    /// Number of nested LMC structures.
    pub n_structures: i64,
    /// Pairing distance for the cross-covariances, see [`CoKrigingOptions::max_colocation_dist`].
    pub max_colocation_dist: Option<f64>,
    /// Secondary neighbor limit of ICM and LMC, see [`CoKrigingOptions::max_secondary_neighbors`].
    pub max_secondary_neighbors: Option<i64>,
}

impl Default for CrossValidateCokrigingOptions {
    fn default() -> Self {
        CrossValidateCokrigingOptions {
            max_points: 500,
            seed: None,
            radius: None,
            max_neighbors: 40,
            min_neighbors: 3,
            n_structures: 2,
            max_colocation_dist: None,
            max_secondary_neighbors: None,
        }
    }
}

/// Options of [`select_best_model`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectBestModelOptions {
    /// Error metric for the point estimate and every bootstrap replicate.
    pub metric: Metric,
    /// Number of paired bootstrap resamples.
    pub n_bootstrap: i64,
    /// Two-sided confidence level of the score-difference interval.
    pub confidence_level: f64,
    /// Seed of the bootstrap draw; `None` uses fresh entropy.
    pub seed: Option<u64>,
}

impl Default for SelectBestModelOptions {
    fn default() -> Self {
        SelectBestModelOptions {
            metric: Metric::Rmse,
            n_bootstrap: 2000,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Inputs accepted by [`select_best_model`]: one CV result or a set of named results.
pub trait SelectBestModelInput {
    /// Result type for this input.
    type Output;

    /// Run the selection.
    fn select_best_model(&self, options: &SelectBestModelOptions) -> Result<Self::Output>;
}

impl SelectBestModelInput for CrossValidationResult {
    type Output = BestModelResult;

    fn select_best_model(&self, options: &SelectBestModelOptions) -> Result<BestModelResult> {
        select_idw_vs_kriging(self, options)
    }
}

impl SelectBestModelInput for IndexMap<String, MethodCrossValidation> {
    type Output = MultiMethodBestModelResult;

    fn select_best_model(&self, options: &SelectBestModelOptions) -> Result<MultiMethodBestModelResult> {
        select_best_of_methods(self, options)
    }
}

/// Leave-one-out cross-validate local IDW and local ordinary kriging.
///
/// Each held-out point is re-estimated from the remaining source points,
/// reusing one variogram/radius and one spatial index for both methods.
pub fn cross_validate(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    options: CrossValidateOptions,
) -> Result<CrossValidationResult> {
    let x_array = as_float64_1d(x, "x")?;
    let y_array = as_float64_1d(y, "y")?;
    let z_array = as_float64_1d(z, "z")?;
    validate_same_size_xyz(&x_array, &y_array, &z_array)?;
    validate_config(
        options.max_points,
        options.power,
        options.max_neighbors,
        options.min_neighbors,
    )?;

    let (x_valid, y_valid, z_valid) = filter_finite_xyz(&x_array, &y_array, &z_array);
    if (x_valid.len() as i64) < options.min_neighbors + 1 {
        return Err(ApbaseError::value_error(
            "cross_validate requires at least min_neighbors + 1 finite points",
        ));
    }

    let variogram = match &options.model_values {
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
    let model_params = variogram.model_params()?;

    let radius_value = options
        .radius
        .unwrap_or_else(|| model_params.search_radius(x_valid.len()));
    if !radius_value.is_finite() || radius_value <= 0.0 {
        return Err(ApbaseError::value_error(
            "radius must be finite and greater than zero",
        ));
    }

    let sample_idx = select_sample_idx(x_valid.len(), options.max_points, options.seed);
    let (idw, kriging, status) = run_cross_validate(
        x_valid.as_slice().expect("contiguous"),
        y_valid.as_slice().expect("contiguous"),
        z_valid.as_slice().expect("contiguous"),
        &sample_idx,
        radius_value,
        options.power,
        variogram.model_values()?.as_slice().expect("contiguous"),
        options.max_neighbors,
        options.min_neighbors,
        resolve_n_threads(),
    )?;
    if status != 0 {
        return Err(raise_status_error(
            status,
            CROSS_VALIDATE_STATUS_ERRORS,
            "native cross-validation execution failed",
        ));
    }

    let actual = held_out(&z_valid, &sample_idx);
    Ok(CrossValidationResult {
        idw: method_result(idw, &actual),
        kriging: method_result(kriging, &actual),
        n_samples: sample_idx.len() as i64,
        radius: radius_value,
        model_params,
    })
}

/// Leave-one-out cross-validate collocated, ICM, and LMC cokriging together.
///
/// The three models are fit once from the full dataset; only the primary is
/// held out, sharing one primary spatial index and one radius. A method whose
/// model cannot be fitted is left out of `by_method` and `model_params`
/// instead of failing the call: any method when a secondary has too few
/// samples paired with the primary (see `max_colocation_dist`), and ICM or LMC
/// when the fit is rejected as non-positive-semidefinite or invalid. The first
/// fit error is returned only when no method can be fitted.
pub fn cross_validate_cokriging(
    x0: &(impl ArrayLike + ?Sized),
    y0: &(impl ArrayLike + ?Sized),
    z0: &(impl ArrayLike + ?Sized),
    secondaries: &SecondaryInput,
    options: CrossValidateCokrigingOptions,
) -> Result<CokrigingCrossValidationResult> {
    let x0_array = as_float64_1d(x0, "x0")?;
    let y0_array = as_float64_1d(y0, "y0")?;
    let z0_array = as_float64_1d(z0, "z0")?;
    validate_same_size_xyz(&x0_array, &y0_array, &z0_array)?;
    if options.max_points <= 0 {
        return Err(ApbaseError::value_error("max_points must be greater than zero"));
    }
    require_neighbor_bounds(
        options.max_neighbors,
        options.min_neighbors,
        Some(resolve_max_neighbors_ceiling()),
    )?;

    let fit = |method: CoKrigingMethod, n_structures: i64| {
        CoKriging::from_data(
            &x0_array,
            &y0_array,
            &z0_array,
            secondaries,
            CoKrigingOptions {
                method,
                radius: options.radius,
                max_neighbors: options.max_neighbors,
                min_neighbors: options.min_neighbors,
                n_structures,
                model_values: None,
                max_colocation_dist: options.max_colocation_dist,
                max_secondary_neighbors: options.max_secondary_neighbors,
            },
        )
    };
    let is_model_error = |error: &ApbaseError| {
        is_no_collocated_samples_error(error)
            || matches!(
                error.kind(),
                ErrorKind::CoKrigingNonPSDCoregionalizationError | ErrorKind::CoKrigingInvalidModelError
            )
    };
    let mut fitted: Vec<(&str, CoKriging)> = Vec::with_capacity(3);
    let mut first_error = None;
    for (name, method, n_structures) in [
        ("collocated", CoKrigingMethod::Collocated, 2),
        ("icm", CoKrigingMethod::Icm, 2),
        ("lmc", CoKrigingMethod::Lmc, options.n_structures),
    ] {
        match fit(method, n_structures) {
            Ok(model) => fitted.push((name, model)),
            Err(error) if is_model_error(&error) => {
                first_error.get_or_insert(error);
            }
            Err(error) => return Err(error),
        }
    }
    let Some((_, reference)) = fitted.first() else {
        return Err(first_error.expect("every skipped method recorded its error"));
    };
    let find = |name: &str| {
        fitted
            .iter()
            .find(|(fitted_name, _)| *fitted_name == name)
            .map(|(_, model)| model)
    };
    let (collocated, icm, lmc) = (find("collocated"), find("icm"), find("lmc"));

    let data = reference.fitted_data().expect("fitted cokriging has data");
    let radius_value = match options.radius {
        None => {
            let (n_lags, max_pairs, max_distance) = resolve_variogram_profile();
            let variogram = Variogram::from_data(
                &data.x0,
                &data.y0,
                &data.z0,
                VariogramOptions {
                    n_lags,
                    max_pairs,
                    max_distance,
                },
            )?;
            let radius_value = variogram.model_params()?.search_radius(data.x0.len());
            if !radius_value.is_finite() || radius_value <= 0.0 {
                return Err(ApbaseError::value_error(
                    "radius must be finite and greater than zero",
                ));
            }
            radius_value
        }
        Some(radius) => radius,
    };

    let sample_idx = select_sample_idx(data.x0.len(), options.max_points, options.seed);
    let n_variables = data.secondary_names.len() + 1;
    let zero_square = Array2::zeros((n_variables, n_variables));
    let lmc_structures = usize::try_from(options.n_structures).unwrap_or(2).max(2);
    let placeholder_ranges = Array1::ones(lmc_structures - 1);
    let placeholder_coefficients = Array3::zeros((lmc_structures, n_variables, n_variables));
    let (icm_model_id, model_range, nugget_matrix, sill_matrix) = match icm.and_then(CoKriging::fitted_model)
    {
        Some(Fit::Icm {
            model_id,
            model_range,
            nugget_matrix,
            sill_matrix,
        }) => (*model_id, *model_range, nugget_matrix, sill_matrix),
        _ => (1.0, 1.0, &zero_square, &zero_square),
    };
    let (lmc_model_id, structure_ranges, coefficient_matrices) = match lmc.and_then(CoKriging::fitted_model) {
        Some(Fit::Lmc {
            model_id,
            structure_ranges,
            coefficient_matrices,
        }) => (*model_id, structure_ranges, coefficient_matrices),
        _ => (1.0, &placeholder_ranges, &placeholder_coefficients),
    };
    let collocated_model = match collocated {
        Some(collocated) => collocated.model_values()?,
        None => Array1::zeros(4 + n_variables),
    };

    let (results, status) = run_cokriging_cross_validate(&CokrigingCrossValidateInputs {
        x0: data.x0.as_slice().expect("contiguous"),
        y0: data.y0.as_slice().expect("contiguous"),
        z0: data.z0.as_slice().expect("contiguous"),
        sample_idx: &sample_idx,
        secondary_x: data.secondary_x.as_slice().expect("contiguous"),
        secondary_y: data.secondary_y.as_slice().expect("contiguous"),
        secondary_z: data.secondary_z.as_slice().expect("contiguous"),
        secondary_offset: data.secondary_offset.as_slice().expect("contiguous"),
        run_collocated: collocated.is_some(),
        collocated_model: collocated_model.as_slice().expect("contiguous"),
        run_icm: icm.is_some(),
        icm_model_id,
        icm_range: model_range,
        icm_nugget: nugget_matrix,
        icm_sill: sill_matrix,
        run_lmc: lmc.is_some(),
        lmc_model_id,
        lmc_ranges: structure_ranges.as_slice().expect("contiguous"),
        lmc_coefficients: coefficient_matrices,
        radius: radius_value,
        max_neighbors: options.max_neighbors,
        max_secondary_neighbors: options.max_secondary_neighbors.unwrap_or(options.max_neighbors),
        min_neighbors: options.min_neighbors,
        n_threads: resolve_n_threads(),
    })?;
    if status != 0 {
        return Err(raise_status_error(
            status,
            COKRIGING_STATUS_ERRORS,
            "native cokriging cross-validation execution failed",
        ));
    }

    let actual = held_out(&data.z0, &sample_idx);
    let [collocated_result, icm_result, lmc_result] = results;
    let mut native_results = IndexMap::from([
        ("collocated", collocated_result),
        ("icm", icm_result),
        ("lmc", lmc_result),
    ]);
    let mut by_method = IndexMap::new();
    let mut model_params = IndexMap::new();
    for (name, model) in &fitted {
        let result = native_results
            .shift_remove(name)
            .expect("one native result per method");
        by_method.insert(name.to_string(), method_result(result, &actual));
        model_params.insert(name.to_string(), model.model_params()?);
    }
    Ok(CokrigingCrossValidationResult {
        by_method,
        n_samples: sample_idx.len() as i64,
        radius: radius_value,
        model_params,
    })
}

/// Pick the statistically better method from two or more evaluated methods.
///
/// A [`CrossValidationResult`] keeps the IDW-vs-kriging behavior and returns a
/// [`BestModelResult`]; any named mapping of [`MethodCrossValidation`] (e.g.
/// [`CokrigingCrossValidationResult::by_method`]) returns a
/// [`MultiMethodBestModelResult`]. Ties go to the first method (`"idw"`). A
/// paired bootstrap over the common finite samples estimates a confidence
/// interval of the score difference; `significant` means it excludes zero.
pub fn select_best_model<R: SelectBestModelInput + ?Sized>(
    result: &R,
    options: SelectBestModelOptions,
) -> Result<R::Output> {
    result.select_best_model(&options)
}

fn validate_selection(options: &SelectBestModelOptions) -> Result<()> {
    if !(0.0 < options.confidence_level && options.confidence_level < 1.0) {
        return Err(ApbaseError::value_error(
            "confidence_level must be strictly between 0 and 1",
        ));
    }
    if options.n_bootstrap <= 0 {
        return Err(ApbaseError::value_error("n_bootstrap must be greater than zero"));
    }
    Ok(())
}

fn select_idw_vs_kriging(
    result: &CrossValidationResult,
    options: &SelectBestModelOptions,
) -> Result<BestModelResult> {
    validate_selection(options)?;
    let (idw, krig) = (&result.idw, &result.kriging);
    if idw.predicted.len() != idw.actual.len() || krig.predicted.len() != krig.actual.len() {
        return Err(ApbaseError::value_error(
            "predicted and actual arrays must share one shape per method",
        ));
    }
    if !array_equal(&idw.actual, &krig.actual) {
        return Err(ApbaseError::value_error(
            "idw.actual and kriging.actual must be identical: cross_validate() always evaluates both methods on \
             the same held-out samples",
        ));
    }

    let actual = &idw.actual;
    let idw_valid = valid_mask(&idw.predicted, actual);
    let krig_valid = valid_mask(&krig.predicted, actual);
    let mask: Vec<bool> = idw_valid.iter().zip(&krig_valid).map(|(a, b)| *a && *b).collect();
    let n_common_valid = mask.iter().filter(|keep| **keep).count();

    if n_common_valid == 0 {
        return select_without_overlap(options, idw, krig, actual, &idw_valid, &krig_valid);
    }

    let idw_err = errors(&idw.predicted, actual, &mask);
    let krig_err = errors(&krig.predicted, actual, &mask);
    let idw_score = method_score(options.metric, &idw_err);
    let krig_score = method_score(options.metric, &krig_err);
    let method = if idw_score <= krig_score {
        Method::Idw
    } else {
        Method::Kriging
    };

    let loser_score = max_keep_first(idw_score, krig_score);
    let margin = (idw_score - krig_score).abs();
    let relative_margin = if loser_score == 0.0 {
        0.0
    } else {
        margin / loser_score
    };

    let (significant, ci_low, ci_high) = if n_common_valid < 2 {
        (false, f64::NAN, f64::NAN)
    } else {
        bootstrap_interval(options, &idw_err, &krig_err)
    };

    Ok(BestModelResult {
        method,
        metric: options.metric,
        idw_score,
        kriging_score: krig_score,
        margin,
        relative_margin,
        significant,
        confidence_level: options.confidence_level,
        ci_low,
        ci_high,
        n_common_valid: n_common_valid as i64,
    })
}

fn select_without_overlap(
    options: &SelectBestModelOptions,
    idw: &MethodCrossValidation,
    krig: &MethodCrossValidation,
    actual: &Array1<f64>,
    idw_valid: &[bool],
    krig_valid: &[bool],
) -> Result<BestModelResult> {
    let n_idw_valid = idw_valid.iter().filter(|keep| **keep).count();
    let n_krig_valid = krig_valid.iter().filter(|keep| **keep).count();
    if n_idw_valid == 0 && n_krig_valid == 0 {
        return Err(ApbaseError::value_error(
            "neither idw nor kriging produced a finite leave-one-out prediction; cannot select a best model",
        ));
    }
    if n_idw_valid > 0 && n_krig_valid > 0 {
        return Err(ApbaseError::value_error(
            "idw and kriging have no held-out sample with both a finite idw and a finite kriging prediction; \
             cannot form a paired comparison",
        ));
    }

    let (method, idw_score, kriging_score) = if n_idw_valid == 0 {
        let winner = method_score(options.metric, &errors(&krig.predicted, actual, krig_valid));
        (Method::Kriging, f64::NAN, winner)
    } else {
        let winner = method_score(options.metric, &errors(&idw.predicted, actual, idw_valid));
        (Method::Idw, winner, f64::NAN)
    };
    Ok(BestModelResult {
        method,
        metric: options.metric,
        idw_score,
        kriging_score,
        margin: f64::INFINITY,
        relative_margin: f64::INFINITY,
        significant: true,
        confidence_level: options.confidence_level,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        n_common_valid: 0,
    })
}

fn select_best_of_methods(
    methods: &IndexMap<String, MethodCrossValidation>,
    options: &SelectBestModelOptions,
) -> Result<MultiMethodBestModelResult> {
    validate_selection(options)?;
    if methods.len() < 2 {
        return Err(ApbaseError::value_error(
            "select_best_model requires at least two methods to compare",
        ));
    }

    let actual = &methods[0].actual;
    for method in methods.values() {
        if method.predicted.len() != method.actual.len() {
            return Err(ApbaseError::value_error(
                "predicted and actual arrays must share one shape per method",
            ));
        }
        if !array_equal(&method.actual, actual) {
            return Err(ApbaseError::value_error(
                "all methods must share identical held-out actual values -- cross_validate_cokriging() always \
                 evaluates every method on the same held-out samples",
            ));
        }
    }

    let valid_masks: IndexMap<&String, Vec<bool>> = methods
        .iter()
        .map(|(name, method)| (name, valid_mask(&method.predicted, actual)))
        .collect();
    let common_mask: Vec<bool> = (0..actual.len())
        .map(|i| actual[i].is_finite() && valid_masks.values().all(|mask| mask[i]))
        .collect();
    let n_common_valid = common_mask.iter().filter(|keep| **keep).count();

    if n_common_valid == 0 {
        return select_best_of_methods_without_overlap(options, methods, &valid_masks, actual);
    }

    let errors_by_name: IndexMap<&String, Vec<f64>> = methods
        .iter()
        .map(|(name, method)| (name, errors(&method.predicted, actual, &common_mask)))
        .collect();
    let scores: IndexMap<String, f64> = errors_by_name
        .iter()
        .map(|(name, err)| ((*name).clone(), method_score(options.metric, err)))
        .collect();
    let mut ranked: Vec<&String> = methods.keys().collect();
    ranked.sort_by(|a, b| {
        scores[*a]
            .partial_cmp(&scores[*b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (winner, runner_up) = (ranked[0], ranked[1]);
    let (winner_score, runner_up_score) = (scores[winner], scores[runner_up]);
    let margin = runner_up_score - winner_score;
    let relative_margin = if runner_up_score == 0.0 {
        0.0
    } else {
        margin / runner_up_score
    };

    let (significant, ci_low, ci_high) = if n_common_valid < 2 {
        (false, f64::NAN, f64::NAN)
    } else {
        bootstrap_interval(options, &errors_by_name[runner_up], &errors_by_name[winner])
    };

    Ok(MultiMethodBestModelResult {
        method: winner.clone(),
        runner_up: runner_up.clone(),
        metric: options.metric,
        scores,
        margin,
        relative_margin,
        significant,
        confidence_level: options.confidence_level,
        ci_low,
        ci_high,
        n_common_valid: n_common_valid as i64,
    })
}

fn select_best_of_methods_without_overlap(
    options: &SelectBestModelOptions,
    methods: &IndexMap<String, MethodCrossValidation>,
    valid_masks: &IndexMap<&String, Vec<bool>>,
    actual: &Array1<f64>,
) -> Result<MultiMethodBestModelResult> {
    let names_with_valid: Vec<&String> = valid_masks
        .iter()
        .filter(|(_, mask)| mask.iter().any(|keep| *keep))
        .map(|(name, _)| *name)
        .collect();
    if names_with_valid.is_empty() {
        return Err(ApbaseError::value_error(
            "no method produced a finite leave-one-out prediction; cannot select a best model",
        ));
    }
    if names_with_valid.len() > 1 {
        return Err(ApbaseError::value_error(
            "no held-out sample has a finite prediction from every method; cannot form a paired comparison",
        ));
    }

    let winner = names_with_valid[0];
    let winner_score = method_score(
        options.metric,
        &errors(&methods[winner].predicted, actual, &valid_masks[winner]),
    );
    let scores: IndexMap<String, f64> = methods
        .keys()
        .map(|name| (name.clone(), if name == winner { winner_score } else { f64::NAN }))
        .collect();
    let runner_up = methods
        .keys()
        .find(|name| *name != winner)
        .expect("at least two methods")
        .clone();

    Ok(MultiMethodBestModelResult {
        method: winner.clone(),
        runner_up,
        metric: options.metric,
        scores,
        margin: f64::INFINITY,
        relative_margin: f64::INFINITY,
        significant: true,
        confidence_level: options.confidence_level,
        ci_low: f64::NAN,
        ci_high: f64::NAN,
        n_common_valid: 0,
    })
}

/// Paired percentile bootstrap of `score(first) - score(second)`:
/// `(significant, ci_low, ci_high)`.
fn bootstrap_interval(options: &SelectBestModelOptions, first: &[f64], second: &[f64]) -> (bool, f64, f64) {
    let n = first.len();
    let n_bootstrap = options.n_bootstrap as usize;
    let mut rng = Generator::default_rng(options.seed);
    let resample_idx = rng.integers(0, n as i64, n_bootstrap * n);

    let mut first_sample = vec![0.0; n];
    let mut second_sample = vec![0.0; n];
    let diff: Vec<f64> = resample_idx
        .chunks_exact(n)
        .map(|row| {
            for (j, index) in row.iter().enumerate() {
                first_sample[j] = first[*index as usize];
                second_sample[j] = second[*index as usize];
            }
            method_score(options.metric, &first_sample) - method_score(options.metric, &second_sample)
        })
        .collect();

    let alpha = 1.0 - options.confidence_level;
    let interval = _numpy::quantile(&diff, &[alpha / 2.0, 1.0 - alpha / 2.0]);
    let (ci_low, ci_high) = (interval[0], interval[1]);
    (ci_low > 0.0 || ci_high < 0.0, ci_low, ci_high)
}

/// RMSE or MAE of an error vector (numpy mean, pairwise summation).
fn method_score(metric: Metric, error: &[f64]) -> f64 {
    match metric {
        Metric::Rmse => {
            let squared: Vec<f64> = error.iter().map(|value| value * value).collect();
            _numpy::mean(&squared).sqrt()
        }
        Metric::Mae => {
            let absolute: Vec<f64> = error.iter().map(|value| value.abs()).collect();
            _numpy::mean(&absolute)
        }
    }
}

fn valid_mask(predicted: &Array1<f64>, actual: &Array1<f64>) -> Vec<bool> {
    predicted
        .iter()
        .zip(actual)
        .map(|(p, a)| p.is_finite() && a.is_finite())
        .collect()
}

fn errors(predicted: &Array1<f64>, actual: &Array1<f64>, mask: &[bool]) -> Vec<f64> {
    predicted
        .iter()
        .zip(actual)
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|((p, a), _)| p - a)
        .collect()
}

/// `np.array_equal`: same length and every element `==` (NaN never equal).
fn array_equal(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x == y)
}

/// `max(a, b)` that keeps `a` on ties and whenever a comparison involves NaN
/// (`f64::max` would ignore the NaN instead).
fn max_keep_first(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// 1-based int32 sample indices; a seeded `choice` subsample above `max_points`.
fn select_sample_idx(n: usize, max_points: i64, seed: Option<u64>) -> Vec<i32> {
    if n as i64 <= max_points {
        return (1..=n as i32).collect();
    }
    let mut rng = Generator::default_rng(seed);
    rng.choice_without_replacement(n as i64, max_points)
        .into_iter()
        .map(|position| (position + 1) as i32)
        .collect()
}

fn held_out(z: &Array1<f64>, sample_idx: &[i32]) -> Array1<f64> {
    sample_idx.iter().map(|index| z[(*index - 1) as usize]).collect()
}

fn method_result(native: NativeMethodResult, actual: &Array1<f64>) -> MethodCrossValidation {
    MethodCrossValidation {
        mae: native.mae,
        rmse: native.rmse,
        r2: native.r2,
        n_valid: native.n_valid,
        predicted: native.predicted,
        actual: actual.clone(),
    }
}

fn validate_config(max_points: i64, power: f64, max_neighbors: i64, min_neighbors: i64) -> Result<()> {
    if max_points <= 0 {
        return Err(ApbaseError::value_error("max_points must be greater than zero"));
    }
    if !power.is_finite() || power <= 0.0 {
        return Err(ApbaseError::value_error(
            "power must be finite and greater than zero",
        ));
    }
    require_neighbor_bounds(
        max_neighbors,
        min_neighbors,
        Some(resolve_max_neighbors_ceiling()),
    )
}

#[cfg(test)]
#[path = "../../tests/common/numpy_rng.rs"]
mod numpy_rng;
#[cfg(test)]
#[path = "../../tests/common/ziggurat.rs"]
mod ziggurat;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn method(predicted: Array1<f64>, actual: Array1<f64>) -> MethodCrossValidation {
        MethodCrossValidation {
            mae: 0.0,
            rmse: 0.0,
            r2: 0.0,
            n_valid: 0,
            predicted,
            actual,
        }
    }

    use super::numpy_rng;

    fn smooth_field(x: &[f64], y: &[f64], seed: u64) -> Vec<f64> {
        let mut rng = numpy_rng::NumpyRng::new(seed);
        let centers = rng.uniform(0.0, 100.0, 16);
        let amps = rng.normal(0.0, 1.0, 8);
        let mut field = vec![0.0; x.len()];
        for (k, amp) in amps.iter().enumerate() {
            for (i, value) in field.iter_mut().enumerate() {
                let d2 = (x[i] - centers[2 * k]).powi(2) + (y[i] - centers[2 * k + 1]).powi(2);
                *value += amp * (-d2 / (2.0 * 25.0f64.powi(2))).exp();
            }
        }
        field
    }

    /// The batch LOO kernel equals holding each primary point out by hand and
    /// interpolating it with the fixed, fully fitted models.
    #[test]
    fn test_cross_validate_cokriging_matches_manual_loo() {
        use crate::cokriging::{run_co_kriging_icm, run_co_kriging_lmc, run_collocated_cokriging};

        let mut rng = numpy_rng::NumpyRng::new(6);
        let n = 60;
        let x0 = rng.uniform(0.0, 100.0, n);
        let y0 = rng.uniform(0.0, 100.0, n);
        let latent = smooth_field(&x0, &y0, 1006);
        let z0: Vec<f64> = latent
            .iter()
            .zip(rng.normal(0.0, 0.15, n))
            .map(|(l, e)| l + e)
            .collect();
        let zu: Vec<f64> = latent
            .iter()
            .zip(rng.normal(0.0, 0.25, n))
            .map(|(l, e)| 1.4 * l + e)
            .collect();
        let secondaries = SecondaryInput::from([(
            "secondary",
            (
                Array1::from(x0.clone()),
                Array1::from(y0.clone()),
                Array1::from(zu),
            ),
        )]);
        let (radius, max_neighbors, min_neighbors) = (40.0, 12, 1);
        let fit = |method| {
            CoKriging::from_data(
                &x0,
                &y0,
                &z0,
                &secondaries,
                CoKrigingOptions {
                    method,
                    radius: Some(radius),
                    max_neighbors,
                    min_neighbors,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (collocated, icm, lmc) = (
            fit(CoKrigingMethod::Collocated),
            fit(CoKrigingMethod::Icm),
            fit(CoKrigingMethod::Lmc),
        );

        let result = cross_validate_cokriging(
            &x0,
            &y0,
            &z0,
            &secondaries,
            CrossValidateCokrigingOptions {
                radius: Some(radius),
                max_neighbors,
                min_neighbors,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(result.n_samples, n as i64);

        let data = collocated.fitted_data().unwrap();
        let (sx, sy, sz) = (
            data.secondary_x.as_slice().unwrap(),
            data.secondary_y.as_slice().unwrap(),
            data.secondary_z.as_slice().unwrap(),
        );
        let offsets = [0i32, sx.len() as i32];
        let Some(Fit::Icm {
            model_id,
            model_range,
            nugget_matrix,
            sill_matrix,
        }) = icm.fitted_model()
        else {
            panic!()
        };
        let Some(Fit::Lmc {
            model_id: lmc_id,
            structure_ranges,
            coefficient_matrices,
        }) = lmc.fitted_model()
        else {
            panic!()
        };
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-6 + 1e-6 * b.abs();

        for pos in [0, 5, 17, 30, 45, n - 1] {
            let keep = |values: &[f64]| -> Vec<f64> {
                values
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != pos)
                    .map(|(_, v)| *v)
                    .collect()
            };
            let (hx, hy, hz) = (keep(&x0), keep(&y0), keep(&z0));
            let (tx, ty) = ([x0[pos]], [y0[pos]]);

            let (pred, status) = run_collocated_cokriging(
                &hx,
                &hy,
                &hz,
                &tx,
                &ty,
                collocated.model_values().unwrap().as_slice().unwrap(),
                sx,
                sy,
                sz,
                &offsets,
                radius,
                max_neighbors,
                min_neighbors,
                1,
            )
            .unwrap();
            assert_eq!(status, 0);
            assert!(
                close(result.by_method["collocated"].predicted[pos], pred[0]),
                "collocated {pos}"
            );

            let (pred, status) = run_co_kriging_icm(
                &hx,
                &hy,
                &hz,
                &tx,
                &ty,
                *model_id,
                *model_range,
                nugget_matrix,
                sill_matrix,
                sx,
                sy,
                sz,
                &offsets,
                radius,
                max_neighbors,
                max_neighbors,
                min_neighbors,
                1,
            )
            .unwrap();
            assert_eq!(status, 0);
            assert!(
                close(result.by_method["icm"].predicted[pos], pred[0]),
                "icm {pos}"
            );

            let (pred, status) = run_co_kriging_lmc(
                &hx,
                &hy,
                &hz,
                &tx,
                &ty,
                *lmc_id,
                structure_ranges.as_slice().unwrap(),
                coefficient_matrices,
                sx,
                sy,
                sz,
                &offsets,
                radius,
                max_neighbors,
                max_neighbors,
                min_neighbors,
                1,
            )
            .unwrap();
            assert_eq!(status, 0);
            assert!(
                close(result.by_method["lmc"].predicted[pos], pred[0]),
                "lmc {pos}"
            );
        }
    }

    #[test]
    fn sample_idx_is_one_based_and_complete_below_max_points() {
        assert_eq!(select_sample_idx(3, 500, Some(1)), vec![1, 2, 3]);
        let drawn = select_sample_idx(1000, 10, Some(42));
        assert_eq!(drawn.len(), 10);
        assert!(drawn.iter().all(|index| (1..=1000).contains(index)));
    }

    #[test]
    fn multi_method_requires_two_methods() {
        let methods = IndexMap::from([("a".to_string(), method(array![1.0], array![1.0]))]);
        let error = select_best_model(&methods, SelectBestModelOptions::default()).unwrap_err();
        assert_eq!(
            error.message(),
            "select_best_model requires at least two methods to compare"
        );
    }

    #[test]
    fn metric_parsing() {
        assert_eq!("mae".parse::<Metric>().unwrap(), Metric::Mae);
        assert_eq!(
            "r2".parse::<Metric>().unwrap_err().message(),
            "metric must be \"rmse\" or \"mae\""
        );
    }
}
