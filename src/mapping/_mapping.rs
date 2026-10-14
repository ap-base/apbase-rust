//! Automatic high-performance map generation.
//!
//! [`Map`] orchestrates `SpatialFilter`, `Grid`, `Variogram`, `cross_validate`,
//! `IDW` and `Kriging` into one pipeline. Geographic (lon/lat) input is detected
//! and converted internally, and the result coordinates are converted back to
//! the original input metric automatically.

use std::sync::Arc;

use geos::Geometry;
use ndarray::{Array1, Array2, Array3, Axis, stack};

use crate::common::arrays::{
    ArrayLike, as_float64_1d, filter_finite_xyz, normalize_masked_regular_grid_spec, validate_same_size_xyz,
};
use crate::common::coordinates::{
    CoordinateSystem, CoordinateTransform, GeographicMode, guess_xy_coordinate_system,
    prepare_metric_xy_with_transform, restore_original_xy,
};
use crate::common::exceptions::{ApbaseError, Result};
use crate::config::resolve_variogram_profile;
use crate::cross_validate::{
    CrossValidateOptions, CrossValidationResult, Method, SelectBestModelOptions, cross_validate,
    select_best_model,
};
use crate::filtering::{FilterStatistics, SpatialFilter, SpatialFilterOptions};
use crate::grid::{Grid, GridOptions};
use crate::idw::{IDW, IDWOptions};
use crate::kriging::{Kriging, KrigingOptions};
use crate::variogram::{Variogram, VariogramOptions};

/// Interpolation domain: a GEOS geometry, a WKT string, or WKB bytes.
#[derive(Clone)]
pub enum Bounds {
    /// Geometry.
    Geometry(Geometry),
    /// WKT string.
    Wkt(String),
    /// WKB bytes.
    Wkb(Vec<u8>),
}

impl From<Geometry> for Bounds {
    fn from(value: Geometry) -> Self {
        Bounds::Geometry(value)
    }
}

impl From<&str> for Bounds {
    fn from(value: &str) -> Self {
        Bounds::Wkt(value.to_string())
    }
}

impl From<String> for Bounds {
    fn from(value: String) -> Self {
        Bounds::Wkt(value)
    }
}

impl From<Vec<u8>> for Bounds {
    fn from(value: Vec<u8>) -> Self {
        Bounds::Wkb(value)
    }
}

impl From<&[u8]> for Bounds {
    fn from(value: &[u8]) -> Self {
        Bounds::Wkb(value.to_vec())
    }
}

impl std::fmt::Debug for Bounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Bounds::Geometry(_) => f.write_str("Bounds::Geometry(..)"),
            Bounds::Wkt(wkt) => f.debug_tuple("Bounds::Wkt").field(wkt).finish(),
            Bounds::Wkb(wkb) => write!(f, "Bounds::Wkb({} bytes)", wkb.len()),
        }
    }
}

/// Result of automatic map generation.
///
/// `x`, `y`, `z` are a compact point list -- one row per target point inside
/// the boundary. Call [`MapResult::to_raster`] to materialize a padded raster.
#[derive(Clone, Debug)]
pub struct MapResult {
    /// Target x coordinates, in the original input metric.
    pub x: Array1<f64>,
    /// Target y coordinates, in the original input metric.
    pub y: Array1<f64>,
    /// Interpolated values.
    pub z: Array1<f64>,
    /// Interpolation method selected by cross-validation.
    pub method: Method,
    /// Leave-one-out diagnostics used for method selection.
    pub cross_validation: CrossValidationResult,
    /// Local search radius used by the final interpolation.
    pub radius: f64,
    /// Output grid spacing.
    pub resolution: f64,
    /// Number of source points after finite and spatial filtering.
    pub n_source_points: i64,
    /// Local filtering diagnostics, `None` when filtering was disabled.
    pub filter_statistics: Option<FilterStatistics>,
    /// Transform used to recover output coordinates in the original system.
    pub coordinate_transform: Option<CoordinateTransform>,
}

impl MapResult {
    /// Reconstruct `(X, Y, Z)` as `(ny, nx)` arrays, NaN outside the boundary.
    ///
    /// Fails with `ValueError` when the compact points are not a regular masked
    /// grid (possible after the non-linear inverse of `geographic_mode="utm"`).
    pub fn to_raster(&self) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
        let points = stack(Axis(1), &[self.x.view(), self.y.view()]).expect("x and y share one length");
        let Some(spec) = normalize_masked_regular_grid_spec(&points) else {
            return Err(ApbaseError::value_error(
                "map points do not form a regular grid; cannot build a raster",
            ));
        };

        let shape = (spec.full_ny as usize, spec.full_nx as usize);
        let mut x_out = Array2::from_elem(shape, f64::NAN);
        let mut y_out = Array2::from_elem(shape, f64::NAN);
        let mut z_out = Array2::from_elem(shape, f64::NAN);

        let mut position = 0usize;
        for ((row, x_start), count) in spec
            .row_indices
            .iter()
            .zip(&spec.x_start_indices)
            .zip(&spec.counts)
        {
            let (row, x_start, count) = (*row as usize, *x_start as usize, *count as usize);
            for offset in 0..count {
                let column = x_start + offset;
                x_out[[row, column]] = spec.min_x + column as f64 * spec.dx;
                y_out[[row, column]] = spec.min_y + row as f64 * spec.dy;
                z_out[[row, column]] = self.z[position + offset];
            }
            position += count;
        }
        Ok((x_out, y_out, z_out))
    }

    /// Stack [`MapResult::to_raster`] into a single `(ny, nx, 3)` array.
    pub fn to_array3d(&self) -> Result<Array3<f64>> {
        let (x, y, z) = self.to_raster()?;
        Ok(stack(Axis(2), &[x.view(), y.view(), z.view()]).expect("same raster shape"))
    }
}

/// Options of [`Map`] and [`create_map`].
///
/// `resolution` has no default, so it is the argument of [`MapOptions::new`];
/// omitting it is a compile error:
///
/// ```compile_fail
/// let options = apbase::MapOptions { filter: false, ..Default::default() };
/// ```
#[derive(Clone, Debug)]
pub struct MapOptions {
    /// Output grid spacing (meters when geographic input is detected).
    pub resolution: f64,
    /// Whether to run `SpatialFilter` on the source data first.
    pub filter: bool,
    /// Optional interpolation domain in the same unit as `x`/`y`.
    pub bounds: Option<Bounds>,
    /// Projection for detected geographic input; `None` auto-picks.
    pub geographic_mode: Option<GeographicMode>,
    /// IDW distance exponent, used only if IDW is selected.
    pub power: f64,
    /// Maximum neighbors shared by cross-validation and final interpolation.
    pub max_neighbors: i64,
    /// Minimum neighbors shared by cross-validation and final interpolation.
    pub min_neighbors: i64,
    /// Maximum leave-one-out samples used by cross-validation.
    pub max_cv_points: i64,
    /// Seed of the cross-validation subsample and the bootstrap.
    pub seed: Option<u64>,
    /// Options forwarded to `SpatialFilter` (ignored when `filter` is false).
    pub filter_kwargs: Option<SpatialFilterOptions>,
}

impl MapOptions {
    /// Defaults for every argument except the required `resolution`.
    pub fn new(resolution: f64) -> Self {
        MapOptions {
            resolution,
            filter: true,
            bounds: None,
            geographic_mode: None,
            power: 2.0,
            max_neighbors: 40,
            min_neighbors: 3,
            max_cv_points: 500,
            seed: None,
            filter_kwargs: None,
        }
    }
}

/// Functional shortcut for the automatic [`Map`] pipeline.
///
/// Filters, cross-validates, and interpolates `x, y, z` into a map, picking IDW
/// or ordinary kriging via `select_best_model` on the leave-one-out results.
pub fn create_map(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    options: MapOptions,
) -> Result<MapResult> {
    build_map(x, y, z, &options)
}

/// Primary automatic map builder.
///
/// Configure it once and call it with source data:
///
/// ```no_run
/// use apbase::{Map, MapOptions};
///
/// # let (x, y, z) = (vec![0.0], vec![0.0], vec![0.0]);
/// let mut builder = Map::new(MapOptions::new(10.0));
/// let result = builder.call(&x, &y, &z)?;
/// println!("{} -> {} points", result.method, result.z.len());
/// # Ok::<(), apbase::common::ApbaseError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Map {
    /// Output grid spacing.
    pub resolution: f64,
    /// Whether to apply `SpatialFilter`.
    pub filter: bool,
    /// Optional interpolation domain.
    pub bounds: Option<Bounds>,
    /// Projection for detected geographic input.
    pub geographic_mode: Option<GeographicMode>,
    /// IDW distance exponent.
    pub power: f64,
    /// Maximum neighbors.
    pub max_neighbors: i64,
    /// Minimum neighbors.
    pub min_neighbors: i64,
    /// Maximum leave-one-out samples.
    pub max_cv_points: i64,
    /// Random seed for cross-validation sampling and model selection.
    pub seed: Option<u64>,
    /// Options forwarded to `SpatialFilter`.
    pub filter_kwargs: Option<SpatialFilterOptions>,
    /// Last generated map.
    pub result: Option<Arc<MapResult>>,
}

impl Map {
    /// Create a map pipeline without data; run it later with [`Map::call`].
    pub fn new(options: MapOptions) -> Self {
        Map {
            resolution: options.resolution,
            filter: options.filter,
            bounds: options.bounds,
            geographic_mode: options.geographic_mode,
            power: options.power,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
            max_cv_points: options.max_cv_points,
            seed: options.seed,
            filter_kwargs: options.filter_kwargs,
            result: None,
        }
    }

    /// Create a map pipeline and generate the map from `x`, `y`, `z` right away.
    pub fn from_data(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
        options: MapOptions,
    ) -> Result<Self> {
        let mut map = Map::new(options);
        map.fit_generate(x, y, z)?;
        Ok(map)
    }

    fn options(&self) -> MapOptions {
        MapOptions {
            resolution: self.resolution,
            filter: self.filter,
            bounds: self.bounds.clone(),
            geographic_mode: self.geographic_mode,
            power: self.power,
            max_neighbors: self.max_neighbors,
            min_neighbors: self.min_neighbors,
            max_cv_points: self.max_cv_points,
            seed: self.seed,
            filter_kwargs: self.filter_kwargs.clone(),
        }
    }

    /// Run the full pipeline and store the generated map in [`Map::result`].
    pub fn fit_generate(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<Arc<MapResult>> {
        let result = Arc::new(build_map(x, y, z, &self.options())?);
        self.result = Some(Arc::clone(&result));
        Ok(result)
    }

    /// `builder(x, y, z)`: same as [`Map::fit_generate`].
    pub fn call(
        &mut self,
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        z: &(impl ArrayLike + ?Sized),
    ) -> Result<Arc<MapResult>> {
        self.fit_generate(x, y, z)
    }
}

fn build_map(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    z: &(impl ArrayLike + ?Sized),
    options: &MapOptions,
) -> Result<MapResult> {
    let mut x_array = as_float64_1d(x, "x")?;
    let mut y_array = as_float64_1d(y, "y")?;
    let mut z_array = as_float64_1d(z, "z")?;
    validate_same_size_xyz(&x_array, &y_array, &z_array)?;

    let mut transform = None;
    if options.geographic_mode.is_some()
        || guess_xy_coordinate_system(&x_array, &y_array, false) == CoordinateSystem::Geographic
    {
        let (xm, ym, _, _, _, _, coordinate_transform) = prepare_metric_xy_with_transform(
            &x_array,
            &y_array,
            &x_array,
            &y_array,
            options.geographic_mode,
        )?;
        x_array = xm;
        y_array = ym;
        transform = Some(coordinate_transform);
    }

    (x_array, y_array, z_array) = filter_finite_xyz(&x_array, &y_array, &z_array);

    let mut filter_statistics = None;
    if options.filter {
        let mut spatial_filter = SpatialFilter::new(options.filter_kwargs.clone().unwrap_or_default())?;
        (x_array, y_array, z_array) = spatial_filter.fit_filter(&x_array, &y_array, &z_array)?;
        filter_statistics = Some(spatial_filter.statistics()?);
    }

    let (n_lags, max_pairs, max_distance) = resolve_variogram_profile();
    let variogram = Variogram::from_data(
        &x_array,
        &y_array,
        &z_array,
        VariogramOptions {
            n_lags,
            max_pairs,
            max_distance,
        },
    )?;
    let radius = variogram.model_params()?.search_radius(x_array.len());
    let model_values = variogram.model_values()?.clone();

    let cv_result = cross_validate(
        &x_array,
        &y_array,
        &z_array,
        CrossValidateOptions {
            max_points: options.max_cv_points,
            seed: options.seed,
            radius: Some(radius),
            power: options.power,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
            model_values: Some(model_values.clone()),
        },
    )?;
    let method = select_best_model(
        &cv_result,
        SelectBestModelOptions {
            seed: options.seed,
            ..Default::default()
        },
    )?
    .method;

    let boundary = normalize_bounds(options.bounds.as_ref())?;
    let mut grid = Grid::new(GridOptions {
        boundary,
        resolution: options.resolution,
        coordinate_transform: transform.clone(),
        ..GridOptions::default()
    })?;
    grid.fit(Some(&x_array), Some(&y_array))?;
    let points = grid.points()?;

    let estimates = match method {
        Method::Idw => IDW::new(IDWOptions {
            radius: Some(radius),
            power: options.power,
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
        })?
        .fit(&x_array, &y_array, &z_array)?
        .interpolate(points)?,
        Method::Kriging => Kriging::new(KrigingOptions {
            radius: Some(radius),
            model_values: Some(model_values),
            max_neighbors: options.max_neighbors,
            min_neighbors: options.min_neighbors,
        })?
        .fit(&x_array, &y_array, &z_array)?
        .interpolate(points)?,
    };

    let grid_x = points.column(0).to_owned();
    let grid_y = points.column(1).to_owned();
    let (result_x, result_y) = match &transform {
        Some(transform) => restore_original_xy(&grid_x, &grid_y, transform)?,
        None => (grid_x, grid_y),
    };

    Ok(MapResult {
        x: result_x,
        y: result_y,
        z: estimates,
        method,
        cross_validation: cv_result,
        radius,
        resolution: options.resolution,
        n_source_points: x_array.len() as i64,
        filter_statistics,
        coordinate_transform: transform,
    })
}

fn normalize_bounds(bounds: Option<&Bounds>) -> Result<Option<Geometry>> {
    Ok(match bounds {
        None => None,
        Some(Bounds::Geometry(geometry)) => Some(geometry.clone()),
        Some(Bounds::Wkt(wkt)) => Some(Geometry::new_from_wkt(wkt)?),
        Some(Bounds::Wkb(wkb)) => Some(Geometry::new_from_wkb(wkb)?),
    })
}
