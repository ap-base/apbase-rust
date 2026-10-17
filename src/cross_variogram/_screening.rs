//! Secondary-variable screening for cokriging (Fisher z-test + Cauchy-Schwarz).

use std::fmt;
use std::str::FromStr;

use indexmap::IndexMap;
use ndarray::Array1;

use super::_cross_variogram::{CrossVariogram, CrossVariogramOptions};
use super::_native::run_screen_covariate;
use crate::common::arrays::{ArrayLike, as_float64_1d, filter_finite_xyz, validate_same_size_xyz};
use crate::common::exceptions::{ApbaseError, Result};
use crate::variogram::{ModelParams, Variogram, VariogramOptions};

fn status_message(status: i32) -> String {
    match status {
        0 => "ok".to_string(),
        1 => "insufficient sample".to_string(),
        2 => "invalid auto-variogram model".to_string(),
        3 => "invalid cross-variogram model".to_string(),
        other => format!("unknown status ({other})"),
    }
}

/// One secondary variable as `(x, y, z)` arrays.
pub type VariableXYZ = (Array1<f64>, Array1<f64>, Array1<f64>);

/// Multiple-testing correction applied to `alpha`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Correction {
    /// `"bonferroni"`: divide `alpha` by the number of candidates.
    #[default]
    Bonferroni,
    /// `"none"`: use `alpha` as given.
    None,
}

impl Correction {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            Correction::Bonferroni => "bonferroni",
            Correction::None => "none",
        }
    }
}

impl fmt::Display for Correction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Correction {
    type Err = ApbaseError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "bonferroni" => Ok(Correction::Bonferroni),
            "none" => Ok(Correction::None),
            _ => Err(ApbaseError::value_error(
                "correction must be \"bonferroni\" or \"none\"",
            )),
        }
    }
}

/// Per-candidate screening outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct CovariateReport {
    /// Correlation of the colocated pairs.
    pub rho: f64,
    /// Fisher z-test p-value.
    pub p_value: f64,
    /// Fraction of lags passing the Cauchy-Schwarz admissibility check.
    pub admissible_fraction: f64,
    /// Number of colocated pairs.
    pub n_pairs: i64,
    /// Whether the candidate was selected.
    pub decision: bool,
    /// Native status as a readable string (`"ok"`, `"insufficient sample"`, ...).
    pub status: String,
    /// Cross-variogram parameters, only when the native status is `0`.
    pub model_params: Option<ModelParams>,
}

/// Selected secondaries plus the per-candidate report.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningResult {
    /// Candidates kept, mapped to their original `(x, y, z)` arrays.
    pub selected: IndexMap<String, VariableXYZ>,
    /// Report for every candidate, in input order.
    pub report: IndexMap<String, CovariateReport>,
}

/// Options of [`screen_secondary_variables`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct ScreeningOptions {
    /// Significance level of the Fisher z-test.
    pub alpha: f64,
    /// Multiple-testing correction.
    pub correction: Correction,
    /// Minimum absolute correlation.
    pub min_abs_rho: f64,
    /// Maximum fraction of lags violating Cauchy-Schwarz.
    pub max_violation_fraction: f64,
    /// Colocation distance passed to [`CrossVariogram`].
    pub max_colocation_dist: Option<f64>,
    /// Number of lag bins of every variogram fit.
    pub n_lags: i64,
    /// Maximum sampled pairs of every variogram fit.
    pub max_pairs: i64,
    /// Maximum pair distance of every variogram fit.
    pub max_distance: f64,
}

impl Default for ScreeningOptions {
    fn default() -> Self {
        ScreeningOptions {
            alpha: 0.05,
            correction: Correction::Bonferroni,
            min_abs_rho: 0.3,
            max_violation_fraction: 0.1,
            max_colocation_dist: None,
            n_lags: 50,
            max_pairs: 100_000,
            max_distance: 0.0,
        }
    }
}

/// Screen candidate secondary variables against a primary.
///
/// A candidate is kept when its correlation with the primary is significant
/// (Fisher z-test, Bonferroni-corrected by default), strong enough
/// (`|rho| >= min_abs_rho`) and its cross-variogram is admissible
/// (Cauchy-Schwarz). Non-zero native statuses are reported, not raised.
pub fn screen_secondary_variables(
    x0: &(impl ArrayLike + ?Sized),
    y0: &(impl ArrayLike + ?Sized),
    z0: &(impl ArrayLike + ?Sized),
    candidates: &IndexMap<String, VariableXYZ>,
    options: ScreeningOptions,
) -> Result<ScreeningResult> {
    if candidates.is_empty() {
        return Err(ApbaseError::value_error(
            "candidates must contain at least one entry",
        ));
    }
    if !(0.0 < options.alpha && options.alpha < 1.0) {
        return Err(ApbaseError::value_error("alpha must be strictly between 0 and 1"));
    }
    if !options.min_abs_rho.is_finite() || !(0.0..=1.0).contains(&options.min_abs_rho) {
        return Err(ApbaseError::value_error("min_abs_rho must be between 0 and 1"));
    }
    if !options.max_violation_fraction.is_finite() || !(0.0..=1.0).contains(&options.max_violation_fraction) {
        return Err(ApbaseError::value_error(
            "max_violation_fraction must be between 0 and 1",
        ));
    }

    let x0_array = as_float64_1d(x0, "x0")?;
    let y0_array = as_float64_1d(y0, "y0")?;
    let z0_array = as_float64_1d(z0, "z0")?;
    validate_same_size_xyz(&x0_array, &y0_array, &z0_array)?;
    let (x0_valid, y0_valid, z0_valid) = filter_finite_xyz(&x0_array, &y0_array, &z0_array);
    if x0_valid.is_empty() {
        return Err(ApbaseError::value_error(
            "x0, y0, z0 require at least one finite point",
        ));
    }

    let variogram_options = VariogramOptions {
        n_lags: options.n_lags,
        max_pairs: options.max_pairs,
        max_distance: options.max_distance,
    };
    let primary_variogram = Variogram::from_data(&x0_valid, &y0_valid, &z0_valid, variogram_options.clone())?;

    let use_bonferroni = options.correction == Correction::Bonferroni && candidates.len() > 1;
    let effective_alpha = if use_bonferroni {
        options.alpha / candidates.len() as f64
    } else {
        options.alpha
    };

    let mut selected = IndexMap::new();
    let mut report = IndexMap::new();
    for (name, (xc, yc, zc)) in candidates {
        let quoted = crate::config::quote_str(name);
        let xc_array = as_float64_1d(xc, &format!("candidates[{quoted}][0]"))?;
        let yc_array = as_float64_1d(yc, &format!("candidates[{quoted}][1]"))?;
        let zc_array = as_float64_1d(zc, &format!("candidates[{quoted}][2]"))?;
        validate_same_size_xyz(&xc_array, &yc_array, &zc_array)?;
        let (xc_valid, yc_valid, zc_valid) = filter_finite_xyz(&xc_array, &yc_array, &zc_array);
        if xc_valid.is_empty() {
            report.insert(
                name.clone(),
                CovariateReport {
                    rho: f64::NAN,
                    p_value: f64::NAN,
                    admissible_fraction: f64::NAN,
                    n_pairs: 0,
                    decision: false,
                    status: status_message(1),
                    model_params: None,
                },
            );
            continue;
        }

        let candidate_variogram =
            Variogram::from_data(&xc_valid, &yc_valid, &zc_valid, variogram_options.clone())?;
        let cross = CrossVariogram::from_data(
            &x0_valid,
            &y0_valid,
            &z0_valid,
            &xc_valid,
            &yc_valid,
            &zc_valid,
            CrossVariogramOptions {
                n_lags: options.n_lags,
                max_pairs: options.max_pairs,
                max_distance: options.max_distance,
                max_colocation_dist: options.max_colocation_dist,
            },
        )?;
        let (decision, p_value, admissible_fraction, status) = run_screen_covariate(
            primary_variogram.model_values()?.as_slice().expect("contiguous"),
            candidate_variogram
                .model_values()?
                .as_slice()
                .expect("contiguous"),
            cross.model_values()?.as_slice().expect("contiguous"),
            cross.rho()?,
            cross.n_pairs()?,
            effective_alpha,
            options.min_abs_rho,
            options.max_violation_fraction,
        )?;

        let kept = status == 0 && decision;
        report.insert(
            name.clone(),
            CovariateReport {
                rho: cross.rho()?,
                p_value,
                admissible_fraction,
                n_pairs: cross.n_pairs()?,
                decision: kept,
                status: status_message(status),
                model_params: if status == 0 {
                    Some(cross.model_params()?)
                } else {
                    None
                },
            },
        );
        if kept {
            selected.insert(name.clone(), (xc.clone(), yc.clone(), zc.clone()));
        }
    }
    Ok(ScreeningResult { selected, report })
}
