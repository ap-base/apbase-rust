//! Collocated, ICM and LMC cokriging.

use std::fmt;
use std::str::FromStr;

use indexmap::IndexMap;
use ndarray::{Array1, Array2, Array3, Axis, concatenate, s};

use super::_native::{
    run_co_kriging_icm, run_co_kriging_lmc, run_collocated_cokriging, run_fit_icm_model, run_fit_lmc_model,
};
use crate::common::_numpy;
use crate::common::arrays::{
    ArrayLike, Targets, as_float64_1d, extract_target_xy, filter_finite_xyz, validate_same_size_xyz,
};
use crate::common::exceptions::{
    ApbaseError, COKRIGING_STATUS_ERRORS, ErrorKind, Result, raise_status_error,
};
use crate::common::validation::require_neighbor_bounds;
use crate::config::{
    quote_str, resolve_max_lmc_structures_ceiling, resolve_max_neighbors_ceiling,
    resolve_max_secondary_variables_ceiling, resolve_n_threads, resolve_variogram_profile,
};
use crate::cross_variogram::{CrossVariogram, CrossVariogramOptions, VariableXYZ};
use crate::variogram::{ModelParams, Variogram, VariogramOptions, variogram_name};

/// Cokriging method: `"collocated"`, `"icm"` or `"lmc"`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CoKrigingMethod {
    /// Intrinsic coregionalization model.
    Icm,
    /// Linear model of coregionalization.
    Lmc,
    /// Markov-1 collocated cokriging.
    #[default]
    Collocated,
}

impl CoKrigingMethod {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            CoKrigingMethod::Icm => "icm",
            CoKrigingMethod::Lmc => "lmc",
            CoKrigingMethod::Collocated => "collocated",
        }
    }
}

impl fmt::Display for CoKrigingMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for CoKrigingMethod {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl FromStr for CoKrigingMethod {
    type Err = ApbaseError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "icm" => Ok(CoKrigingMethod::Icm),
            "lmc" => Ok(CoKrigingMethod::Lmc),
            "collocated" => Ok(CoKrigingMethod::Collocated),
            _ => Err(ApbaseError::value_error(
                "method must be one of ('icm', 'lmc', 'collocated')",
            )),
        }
    }
}

/// Secondary variables: named, or positional (named `secondary_1`,
/// `secondary_2`, ...).
#[derive(Clone, Debug, PartialEq)]
pub enum SecondaryInput {
    /// Named secondaries, in insertion order.
    Mapping(IndexMap<String, VariableXYZ>),
    /// Unnamed secondaries.
    Sequence(Vec<VariableXYZ>),
}

impl From<IndexMap<String, VariableXYZ>> for SecondaryInput {
    fn from(value: IndexMap<String, VariableXYZ>) -> Self {
        SecondaryInput::Mapping(value)
    }
}

impl From<Vec<VariableXYZ>> for SecondaryInput {
    fn from(value: Vec<VariableXYZ>) -> Self {
        SecondaryInput::Sequence(value)
    }
}

impl From<Vec<(String, VariableXYZ)>> for SecondaryInput {
    fn from(value: Vec<(String, VariableXYZ)>) -> Self {
        SecondaryInput::Mapping(value.into_iter().collect())
    }
}

impl<const N: usize> From<[(&str, VariableXYZ); N]> for SecondaryInput {
    fn from(value: [(&str, VariableXYZ); N]) -> Self {
        SecondaryInput::Mapping(
            value
                .into_iter()
                .map(|(name, xyz)| (name.to_string(), xyz))
                .collect(),
        )
    }
}

impl SecondaryInput {
    fn named(&self) -> Vec<(String, &VariableXYZ)> {
        match self {
            SecondaryInput::Mapping(map) => map.iter().map(|(name, xyz)| (name.clone(), xyz)).collect(),
            SecondaryInput::Sequence(list) => list
                .iter()
                .enumerate()
                .map(|(i, xyz)| (format!("secondary_{}", i + 1), xyz))
                .collect(),
        }
    }
}

/// Options of [`CoKriging`]; `Default` gives the standard settings.
#[derive(Clone, Debug, PartialEq)]
pub struct CoKrigingOptions {
    /// Cokriging method.
    pub method: CoKrigingMethod,
    /// Local search radius; `None` uses [`ModelParams::search_radius`] of the primary variogram.
    pub radius: Option<f64>,
    /// Maximum primary neighbors (capped at `apbase.config["max_neighbors_ceiling"]`).
    pub max_neighbors: i64,
    /// Minimum primary sources within the radius; below it the `max_neighbors`
    /// nearest are used. A collocated secondary without a value within the
    /// radius of a target is left out of that target's system.
    pub min_neighbors: i64,
    /// Number of nested LMC structures (nugget + continuous), `>= 2`.
    pub n_structures: i64,
    /// Optional pre-fitted collocated model vector `[5 primary values, rho_1..rho_K]`.
    pub model_values: Option<Array1<f64>>,
    /// Distance within which a primary sample is paired with its nearest
    /// secondary point to estimate the cross-covariances; `None` pairs only
    /// exactly coincident locations. Heterotopic secondaries, such as
    /// rasters, need a distance (e.g. half the cell diagonal).
    pub max_colocation_dist: Option<f64>,
    /// ICM and LMC only: maximum neighbors of each secondary per target;
    /// `None` uses `max_neighbors`. Dense secondaries need far fewer than the
    /// primary, since the nearest ones screen the rest.
    pub max_secondary_neighbors: Option<i64>,
}

impl Default for CoKrigingOptions {
    fn default() -> Self {
        CoKrigingOptions {
            method: CoKrigingMethod::Collocated,
            radius: None,
            max_neighbors: 40,
            min_neighbors: 3,
            n_structures: 2,
            model_values: None,
            max_colocation_dist: None,
            max_secondary_neighbors: None,
        }
    }
}

/// Readable fitted parameters, one shape per method.
#[derive(Clone, Debug, PartialEq)]
pub enum CoKrigingModelParams {
    /// Collocated: primary variogram plus one correlation per secondary.
    Collocated {
        /// Primary variogram parameters.
        primary: ModelParams,
        /// Correlation per secondary name.
        rho: IndexMap<String, f64>,
    },
    /// ICM: one shared structure with nugget/sill coregionalization matrices.
    Icm {
        /// Primary parameters derived from the matrices.
        primary: ModelParams,
        /// Native model id.
        model_id: i64,
        /// Model name.
        model_name: &'static str,
        /// Shared range.
        range: f64,
        /// Secondary names, in order.
        secondary_names: Vec<String>,
        /// `(K+1) x (K+1)` nugget matrix.
        nugget_matrix: Array2<f64>,
        /// `(K+1) x (K+1)` sill matrix.
        sill_matrix: Array2<f64>,
    },
    /// LMC: nested structures with one coregionalization matrix each.
    Lmc {
        /// Primary parameters derived from the matrices.
        primary: ModelParams,
        /// Native model id.
        model_id: i64,
        /// Model name.
        model_name: &'static str,
        /// Ranges of the continuous structures.
        structure_ranges: Array1<f64>,
        /// Secondary names, in order.
        secondary_names: Vec<String>,
        /// `S x (K+1) x (K+1)` coefficient matrices.
        coefficient_matrices: Array3<f64>,
    },
}

impl CoKrigingModelParams {
    /// The `"primary"` entry.
    pub fn primary(&self) -> &ModelParams {
        match self {
            CoKrigingModelParams::Collocated { primary, .. }
            | CoKrigingModelParams::Icm { primary, .. }
            | CoKrigingModelParams::Lmc { primary, .. } => primary,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Fit {
    Collocated {
        model_values: Array1<f64>,
        rhos: Array1<f64>,
    },
    Icm {
        model_id: f64,
        model_range: f64,
        nugget_matrix: Array2<f64>,
        sill_matrix: Array2<f64>,
    },
    Lmc {
        model_id: f64,
        structure_ranges: Array1<f64>,
        coefficient_matrices: Array3<f64>,
    },
}

const NO_COLLOCATED_SAMPLES_MARKER: &str =
    " has too few samples paired with the primary samples to estimate its cross-covariance;";

fn no_collocated_samples_error(name: &str) -> ApbaseError {
    ApbaseError::value_error(format!(
        "secondary {}{NO_COLLOCATED_SAMPLES_MARKER} for heterotopic data such as a raster, set \
         max_colocation_dist (e.g. half the cell diagonal) to pair each primary sample with its nearest \
         secondary point",
        quote_str(name)
    ))
}

pub(crate) fn is_no_collocated_samples_error(error: &ApbaseError) -> bool {
    error.kind() == ErrorKind::ValueError && error.message().contains(NO_COLLOCATED_SAMPLES_MARKER)
}

/// Fitted standardized data shared with `cross_validate_cokriging`.
#[derive(Clone, Debug)]
pub(crate) struct FittedData {
    pub(crate) x0: Array1<f64>,
    pub(crate) y0: Array1<f64>,
    pub(crate) z0: Array1<f64>,
    pub(crate) secondary_names: Vec<String>,
    pub(crate) secondary_x: Array1<f64>,
    pub(crate) secondary_y: Array1<f64>,
    pub(crate) secondary_z: Array1<f64>,
    pub(crate) secondary_offset: Array1<i32>,
}

/// Estimate a primary variable using spatially correlated secondary variables.
///
/// Run `apbase::cross_variogram::screen_secondary_variables` first to select
/// admissible secondaries.
#[derive(Clone, Debug)]
pub struct CoKriging {
    method: CoKrigingMethod,
    radius: Option<f64>,
    max_neighbors: i64,
    min_neighbors: i64,
    n_structures: i64,
    initial_model_values: Option<Array1<f64>>,
    max_colocation_dist: Option<f64>,
    max_secondary_neighbors: Option<i64>,
    primary_model_params: Option<ModelParams>,
    data: Option<FittedData>,
    fit: Option<Fit>,
}

impl CoKriging {
    /// Create an unfitted cokriging interpolator; invalid options are rejected here.
    pub fn new(options: CoKrigingOptions) -> Result<Self> {
        if options.model_values.is_some() && options.method != CoKrigingMethod::Collocated {
            return Err(ApbaseError::new(
                ErrorKind::NotImplementedError,
                format!(
                    "model_values override is not supported yet for method=\"{}\"",
                    options.method
                ),
            ));
        }
        if options.n_structures < 2 {
            return Err(ApbaseError::value_error(
                "n_structures must be >= 2 (nugget + at least one continuous structure)",
            ));
        }
        let initial_model_values = match &options.model_values {
            Some(values) => Some(CoKriging::normalize_model_values(values)?),
            None => None,
        };
        require_neighbor_bounds(
            options.max_neighbors,
            options.min_neighbors,
            Some(resolve_max_neighbors_ceiling()),
        )?;
        if let Some(radius) = options.radius {
            if !radius.is_finite() || radius <= 0.0 {
                return Err(ApbaseError::value_error(
                    "radius must be finite and greater than zero",
                ));
            }
        }
        if let Some(distance) = options.max_colocation_dist {
            if !distance.is_finite() || distance <= 0.0 {
                return Err(ApbaseError::value_error(
                    "max_colocation_dist must be None or finite and greater than zero",
                ));
            }
        }
        if let Some(max_secondary_neighbors) = options.max_secondary_neighbors {
            if max_secondary_neighbors <= 0 {
                return Err(ApbaseError::validation_error(
                    "max_secondary_neighbors must be greater than zero",
                ));
            }
            let max_neighbors_ceiling = resolve_max_neighbors_ceiling();
            if max_secondary_neighbors > max_neighbors_ceiling {
                return Err(ApbaseError::validation_error(format!(
                    "max_secondary_neighbors must be <= {max_neighbors_ceiling}"
                )));
            }
        }
        Ok(CoKriging {
            method: options.method,
            radius: options.radius,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
            n_structures: options.n_structures,
            initial_model_values,
            max_colocation_dist: options.max_colocation_dist,
            max_secondary_neighbors: options.max_secondary_neighbors,
            primary_model_params: None,
            data: None,
            fit: None,
        })
    }

    /// Cokriging method.
    pub fn method(&self) -> CoKrigingMethod {
        self.method
    }

    /// Local search radius.
    pub fn radius(&self) -> Option<f64> {
        self.radius
    }

    /// Maximum primary neighbors.
    pub fn max_neighbors(&self) -> i64 {
        self.max_neighbors
    }

    /// Minimum primary sources within the radius before the nearest-sources fallback.
    pub fn min_neighbors(&self) -> i64 {
        self.min_neighbors
    }

    /// Number of nested LMC structures.
    pub fn n_structures(&self) -> i64 {
        self.n_structures
    }

    /// Pairing distance between primary samples and secondary points
    /// (`None`: exactly coincident locations only).
    pub fn max_colocation_dist(&self) -> Option<f64> {
        self.max_colocation_dist
    }

    /// Maximum neighbors of each secondary per target (ICM and LMC); `None`
    /// uses `max_neighbors`.
    pub fn max_secondary_neighbors(&self) -> Option<i64> {
        self.max_secondary_neighbors
    }

    /// OpenMP thread count, resolved live from `apbase.config["n_threads"]`.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Create and fit a cokriging interpolator.
    pub fn from_data(
        x0: &(impl ArrayLike + ?Sized),
        y0: &(impl ArrayLike + ?Sized),
        z0: &(impl ArrayLike + ?Sized),
        secondaries: &SecondaryInput,
        options: CoKrigingOptions,
    ) -> Result<Self> {
        let mut cokriging = CoKriging::new(options)?;
        cokriging.fit(x0, y0, z0, secondaries)?;
        Ok(cokriging)
    }

    /// Whether a model has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fit.is_some()
    }

    /// Flat model vector (collocated: `[primary 5, rhos]`; ICM: `[id, range,
    /// N.ravel(), S.ravel()]`; LMC: `[id, ranges, C.ravel()]`).
    pub fn model_values(&self) -> Result<Array1<f64>> {
        self.require_fitted()?;
        Ok(match self.fit.as_ref().expect("fitted") {
            Fit::Collocated { model_values, .. } => model_values.clone(),
            Fit::Icm {
                model_id,
                model_range,
                nugget_matrix,
                sill_matrix,
            } => [*model_id, *model_range]
                .into_iter()
                .chain(nugget_matrix.iter().copied())
                .chain(sill_matrix.iter().copied())
                .collect(),
            Fit::Lmc {
                model_id,
                structure_ranges,
                coefficient_matrices,
            } => std::iter::once(*model_id)
                .chain(structure_ranges.iter().copied())
                .chain(coefficient_matrices.iter().copied())
                .collect(),
        })
    }

    /// Readable fitted parameters.
    pub fn model_params(&self) -> Result<CoKrigingModelParams> {
        self.require_fitted()?;
        let primary = self.primary_model_params.clone().expect("fitted");
        let names = self.data.as_ref().expect("fitted").secondary_names.clone();
        Ok(match self.fit.as_ref().expect("fitted") {
            Fit::Collocated { rhos, .. } => CoKrigingModelParams::Collocated {
                primary,
                rho: names.into_iter().zip(rhos.iter().copied()).collect(),
            },
            Fit::Icm {
                model_id,
                model_range,
                nugget_matrix,
                sill_matrix,
            } => {
                let model_id = _numpy::rint(*model_id) as i64;
                CoKrigingModelParams::Icm {
                    primary,
                    model_id,
                    model_name: variogram_name(model_id).unwrap_or("unknown"),
                    range: *model_range,
                    secondary_names: names,
                    nugget_matrix: nugget_matrix.clone(),
                    sill_matrix: sill_matrix.clone(),
                }
            }
            Fit::Lmc {
                model_id,
                structure_ranges,
                coefficient_matrices,
            } => {
                let model_id = _numpy::rint(*model_id) as i64;
                CoKrigingModelParams::Lmc {
                    primary,
                    model_id,
                    model_name: variogram_name(model_id).unwrap_or("unknown"),
                    structure_ranges: structure_ranges.clone(),
                    secondary_names: names,
                    coefficient_matrices: coefficient_matrices.clone(),
                }
            }
        })
    }

    /// Fit the coregionalization model from the primary and secondary variables.
    pub fn fit(
        &mut self,
        x0: &(impl ArrayLike + ?Sized),
        y0: &(impl ArrayLike + ?Sized),
        z0: &(impl ArrayLike + ?Sized),
        secondaries: &SecondaryInput,
    ) -> Result<&mut Self> {
        let x0_array = as_float64_1d(x0, "x0")?;
        let y0_array = as_float64_1d(y0, "y0")?;
        let z0_array = as_float64_1d(z0, "z0")?;
        validate_same_size_xyz(&x0_array, &y0_array, &z0_array)?;

        let (x0_valid, y0_valid, z0_valid) = filter_finite_xyz(&x0_array, &y0_array, &z0_array);
        if (x0_valid.len() as i64) < self.min_neighbors {
            return Err(ApbaseError::value_error(
                "cokriging requires at least min_neighbors finite primary points",
            ));
        }

        let (names, secondary_arrays) = CoKriging::normalize_secondaries(secondaries)?;
        if secondary_arrays.is_empty() {
            return Err(ApbaseError::value_error(
                "cokriging requires at least one secondary variable",
            ));
        }
        let max_secondaries = resolve_max_secondary_variables_ceiling();
        if secondary_arrays.len() as i64 > max_secondaries {
            return Err(ApbaseError::value_error(format!(
                "number of secondaries must be <= {max_secondaries}"
            )));
        }

        let z0_slice = z0_valid.as_slice().expect("contiguous");
        let z0_mean = _numpy::mean(z0_slice);
        let z0_std = _numpy::std(z0_slice);
        if !z0_std.is_finite() || z0_std <= 0.0 {
            return Err(ApbaseError::value_error(
                "primary has zero variance; cannot standardize secondaries",
            ));
        }

        let mut standardized_z = Vec::new();
        let mut offsets = vec![0i32];
        for (name, (xu, _, zu)) in names.iter().zip(&secondary_arrays) {
            let zu_slice = zu.as_slice().expect("contiguous");
            let zu_mean = _numpy::mean(zu_slice);
            let zu_std = _numpy::std(zu_slice);
            if !zu_std.is_finite() || zu_std <= 0.0 {
                return Err(ApbaseError::value_error(format!(
                    "secondary {} has zero variance; cannot standardize",
                    quote_str(name)
                )));
            }
            let ratio = z0_std / zu_std;
            standardized_z.extend(zu.iter().map(|value| z0_mean + (value - zu_mean) * ratio));
            offsets.push(offsets[offsets.len() - 1] + xu.len() as i32);
        }
        let secondary_x = concatenate(
            Axis(0),
            &secondary_arrays
                .iter()
                .map(|(xu, _, _)| xu.view())
                .collect::<Vec<_>>(),
        )
        .expect("1D");
        let secondary_y = concatenate(
            Axis(0),
            &secondary_arrays
                .iter()
                .map(|(_, yu, _)| yu.view())
                .collect::<Vec<_>>(),
        )
        .expect("1D");
        let secondary_z = Array1::from(standardized_z);
        let secondary_offset = Array1::from(offsets);

        let (n_lags, max_pairs, max_distance) = resolve_variogram_profile();
        let variogram_options = VariogramOptions {
            n_lags,
            max_pairs,
            max_distance,
        };
        let data = FittedData {
            x0: x0_valid,
            y0: y0_valid,
            z0: z0_valid,
            secondary_names: names,
            secondary_x,
            secondary_y,
            secondary_z,
            secondary_offset,
        };

        match self.method {
            CoKrigingMethod::Collocated => {
                self.fit_collocated(&data, &secondary_arrays, &variogram_options)?
            }
            CoKrigingMethod::Icm => {
                self.secondary_rhos(&data, &secondary_arrays, &variogram_options)?;
                self.fit_icm(&data, &variogram_options)?
            }
            CoKrigingMethod::Lmc => {
                let max_structures = resolve_max_lmc_structures_ceiling();
                if self.n_structures > max_structures {
                    return Err(ApbaseError::value_error(format!(
                        "n_structures must be <= {max_structures}"
                    )));
                }
                self.secondary_rhos(&data, &secondary_arrays, &variogram_options)?;
                self.fit_lmc(&data, &variogram_options)?;
            }
        }
        self.data = Some(data);
        Ok(self)
    }

    fn fit_collocated(
        &mut self,
        data: &FittedData,
        secondary_arrays: &[VariableXYZ],
        variogram_options: &VariogramOptions,
    ) -> Result<()> {
        let (model_values, primary_model_params) = match &self.initial_model_values {
            None => {
                let primary_variogram =
                    Variogram::from_data(&data.x0, &data.y0, &data.z0, variogram_options.clone())?;
                let rhos = self.secondary_rhos(data, secondary_arrays, variogram_options)?;
                let model_values: Array1<f64> = primary_variogram
                    .model_values()?
                    .iter()
                    .copied()
                    .chain(rhos)
                    .collect();
                (model_values, primary_variogram.model_params()?)
            }
            Some(model_values) => {
                let k_expected = model_values.len() as i64 - 5;
                if k_expected != secondary_arrays.len() as i64 {
                    return Err(ApbaseError::value_error(format!(
                        "model_values encodes {k_expected} secondaries, but {} were given",
                        secondary_arrays.len()
                    )));
                }
                let primary = Variogram::from_model(&model_values.slice(s![..5]))?.model_params()?;
                (model_values.clone(), primary)
            }
        };
        let rhos = model_values.slice(s![5..]).to_owned();
        self.fit = Some(Fit::Collocated { model_values, rhos });
        self.primary_model_params = Some(primary_model_params);
        Ok(())
    }

    /// Correlation of each secondary with the primary over the paired samples.
    ///
    /// Fails with the "too few paired samples" error for a secondary that
    /// cannot be paired: every method needs its cross-covariance with the primary.
    fn secondary_rhos(
        &self,
        data: &FittedData,
        secondary_arrays: &[VariableXYZ],
        variogram_options: &VariogramOptions,
    ) -> Result<Vec<f64>> {
        let mut rhos = Vec::with_capacity(secondary_arrays.len());
        for (name, (xu, yu, zu)) in data.secondary_names.iter().zip(secondary_arrays) {
            let cross = CrossVariogram::from_data(
                &data.x0,
                &data.y0,
                &data.z0,
                xu,
                yu,
                zu,
                CrossVariogramOptions {
                    n_lags: variogram_options.n_lags,
                    max_pairs: variogram_options.max_pairs,
                    max_distance: variogram_options.max_distance,
                    max_colocation_dist: self.max_colocation_dist,
                },
            )?;
            let rho = cross.rho()?;
            if !rho.is_finite() {
                return Err(no_collocated_samples_error(name));
            }
            rhos.push(rho);
        }
        Ok(rhos)
    }

    /// The native pairing treats 0 as "exactly coincident only".
    fn colocation_dist_value(&self) -> f64 {
        self.max_colocation_dist.unwrap_or(0.0)
    }

    fn secondary_neighbor_limit(&self) -> i64 {
        self.max_secondary_neighbors.unwrap_or(self.max_neighbors)
    }

    fn fit_icm(&mut self, data: &FittedData, variogram_options: &VariogramOptions) -> Result<()> {
        let (model_id, model_range, nugget_matrix, sill_matrix, status) = run_fit_icm_model(
            data.x0.as_slice().expect("contiguous"),
            data.y0.as_slice().expect("contiguous"),
            data.z0.as_slice().expect("contiguous"),
            data.secondary_x.as_slice().expect("contiguous"),
            data.secondary_y.as_slice().expect("contiguous"),
            data.secondary_z.as_slice().expect("contiguous"),
            data.secondary_offset.as_slice().expect("contiguous"),
            variogram_options.n_lags,
            variogram_options.max_pairs,
            variogram_options.max_distance,
            self.colocation_dist_value(),
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                COKRIGING_STATUS_ERRORS,
                "native ICM fit failed",
            ));
        }
        let id = _numpy::rint(model_id) as i64;
        self.primary_model_params = Some(ModelParams {
            model_id: id,
            model_name: variogram_name(id).unwrap_or("unknown"),
            nugget: nugget_matrix[[0, 0]],
            partial_sill: sill_matrix[[0, 0]],
            sill: nugget_matrix[[0, 0]] + sill_matrix[[0, 0]],
            range: model_range,
            sse: None,
        });
        self.fit = Some(Fit::Icm {
            model_id,
            model_range,
            nugget_matrix,
            sill_matrix,
        });
        Ok(())
    }

    fn fit_lmc(&mut self, data: &FittedData, variogram_options: &VariogramOptions) -> Result<()> {
        let (model_id, structure_ranges, coefficient_matrices, status) = run_fit_lmc_model(
            data.x0.as_slice().expect("contiguous"),
            data.y0.as_slice().expect("contiguous"),
            data.z0.as_slice().expect("contiguous"),
            data.secondary_x.as_slice().expect("contiguous"),
            data.secondary_y.as_slice().expect("contiguous"),
            data.secondary_z.as_slice().expect("contiguous"),
            data.secondary_offset.as_slice().expect("contiguous"),
            self.n_structures,
            variogram_options.n_lags,
            variogram_options.max_pairs,
            variogram_options.max_distance,
            self.colocation_dist_value(),
            self.n_threads(),
        )?;
        if status != 0 {
            return Err(raise_status_error(
                status,
                COKRIGING_STATUS_ERRORS,
                "native LMC fit failed",
            ));
        }
        let nugget_diag = coefficient_matrices[[0, 0, 0]];
        let continuous: Vec<f64> = coefficient_matrices
            .slice(s![1.., 0, 0])
            .iter()
            .copied()
            .collect();
        let continuous_diag = _numpy::sum(&continuous);
        let id = _numpy::rint(model_id) as i64;
        self.primary_model_params = Some(ModelParams {
            model_id: id,
            model_name: variogram_name(id).unwrap_or("unknown"),
            nugget: nugget_diag,
            partial_sill: continuous_diag,
            sill: nugget_diag + continuous_diag,
            range: structure_ranges[structure_ranges.len() - 1],
            sse: None,
        });
        self.fit = Some(Fit::Lmc {
            model_id,
            structure_ranges,
            coefficient_matrices,
        });
        Ok(())
    }

    /// Estimate primary values at target coordinates.
    pub fn interpolate(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.require_fitted()?;
        let data = self.data.as_ref().expect("fitted");
        let (target_x, target_y) = extract_target_xy(targets)?;
        let radius_value =
            self.resolve_radius(self.primary_model_params.as_ref().expect("fitted"), data.x0.len())?;
        let x0 = data.x0.as_slice().expect("contiguous");
        let y0 = data.y0.as_slice().expect("contiguous");
        let z0 = data.z0.as_slice().expect("contiguous");
        let tx = target_x.as_slice().expect("contiguous");
        let ty = target_y.as_slice().expect("contiguous");
        let sx = data.secondary_x.as_slice().expect("contiguous");
        let sy = data.secondary_y.as_slice().expect("contiguous");
        let sz = data.secondary_z.as_slice().expect("contiguous");
        let offsets = data.secondary_offset.as_slice().expect("contiguous");

        let (estimates, status) = match self.fit.as_ref().expect("fitted") {
            Fit::Collocated { model_values, .. } => run_collocated_cokriging(
                x0,
                y0,
                z0,
                tx,
                ty,
                model_values.as_slice().expect("contiguous"),
                sx,
                sy,
                sz,
                offsets,
                radius_value,
                self.max_neighbors,
                self.min_neighbors,
                self.n_threads(),
            )?,
            Fit::Icm {
                model_id,
                model_range,
                nugget_matrix,
                sill_matrix,
            } => run_co_kriging_icm(
                x0,
                y0,
                z0,
                tx,
                ty,
                *model_id,
                *model_range,
                nugget_matrix,
                sill_matrix,
                sx,
                sy,
                sz,
                offsets,
                radius_value,
                self.max_neighbors,
                self.secondary_neighbor_limit(),
                self.min_neighbors,
                self.n_threads(),
            )?,
            Fit::Lmc {
                model_id,
                structure_ranges,
                coefficient_matrices,
            } => run_co_kriging_lmc(
                x0,
                y0,
                z0,
                tx,
                ty,
                *model_id,
                structure_ranges.as_slice().expect("contiguous"),
                coefficient_matrices,
                sx,
                sy,
                sz,
                offsets,
                radius_value,
                self.max_neighbors,
                self.secondary_neighbor_limit(),
                self.min_neighbors,
                self.n_threads(),
            )?,
        };
        if status != 0 {
            return Err(raise_status_error(
                status,
                COKRIGING_STATUS_ERRORS,
                "native cokriging execution failed",
            ));
        }
        Ok(estimates)
    }

    /// Same as [`CoKriging::interpolate`].
    pub fn predict(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.interpolate(targets)
    }

    /// Fit and immediately estimate target values.
    pub fn fit_interpolate(
        &mut self,
        x0: &(impl ArrayLike + ?Sized),
        y0: &(impl ArrayLike + ?Sized),
        z0: &(impl ArrayLike + ?Sized),
        secondaries: &SecondaryInput,
        targets: &(impl Targets + ?Sized),
    ) -> Result<Array1<f64>> {
        self.fit(x0, y0, z0, secondaries)?.interpolate(targets)
    }

    /// `cokriging(targets)`: same as [`CoKriging::interpolate`].
    pub fn call(&self, targets: &(impl Targets + ?Sized)) -> Result<Array1<f64>> {
        self.interpolate(targets)
    }

    pub(crate) fn fitted_data(&self) -> Option<&FittedData> {
        self.data.as_ref()
    }

    pub(crate) fn fitted_model(&self) -> Option<&Fit> {
        self.fit.as_ref()
    }

    fn require_fitted(&self) -> Result<()> {
        if self.fit.is_none() {
            return Err(ApbaseError::value_error(
                "call fit(x0, y0, z0, secondaries) before interpolate(targets)",
            ));
        }
        Ok(())
    }

    fn resolve_radius(&self, primary_model_params: &ModelParams, n_points: usize) -> Result<f64> {
        if let Some(radius) = self.radius {
            return Ok(radius);
        }
        let radius_value = primary_model_params.search_radius(n_points);
        if !radius_value.is_finite() || radius_value <= 0.0 {
            return Err(ApbaseError::value_error(
                "radius must be finite and greater than zero",
            ));
        }
        Ok(radius_value)
    }

    fn normalize_secondaries(secondaries: &SecondaryInput) -> Result<(Vec<String>, Vec<VariableXYZ>)> {
        let mut names = Vec::new();
        let mut arrays = Vec::new();
        for (name, (xu, yu, zu)) in secondaries.named() {
            let quoted = quote_str(&name);
            let xu_array = as_float64_1d(xu, &format!("secondaries[{quoted}][0]"))?;
            let yu_array = as_float64_1d(yu, &format!("secondaries[{quoted}][1]"))?;
            let zu_array = as_float64_1d(zu, &format!("secondaries[{quoted}][2]"))?;
            validate_same_size_xyz(&xu_array, &yu_array, &zu_array)?;
            let valid = filter_finite_xyz(&xu_array, &yu_array, &zu_array);
            if valid.0.is_empty() {
                return Err(ApbaseError::value_error(format!(
                    "secondary {quoted} has no finite points"
                )));
            }
            names.push(name);
            arrays.push(valid);
        }
        Ok((names, arrays))
    }

    fn normalize_model_values(model_values: &Array1<f64>) -> Result<Array1<f64>> {
        let model_array = as_float64_1d(model_values, "model_values")?;
        if model_array.len() < 6 {
            return Err(ApbaseError::value_error(
                "model_values must contain 5 primary values plus at least 1 rho",
            ));
        }
        Ok(model_array)
    }
}
