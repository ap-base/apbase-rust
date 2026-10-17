//! Regular grid generation for interpolation targets.
//!
//! [`Grid`] builds a dense, regular point grid inside a supplied GEOS boundary
//! or inside a concave hull inferred from finite x/y source points. Candidate
//! rows are processed in chunks to avoid materializing very large meshes.

use std::cell::{Cell, RefCell};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use geos::{CoordSeq, Geom, Geometry};
use ndarray::{Array1, Array2, ShapeBuilder};

use crate::common::_numpy;
use crate::common::arrays::{
    ArrayLike, Targets, as_float64_1d, filter_finite_xy, format_thousands, validate_same_size_xy,
};
use crate::common::coordinates::{
    CoordinateSystem, CoordinateTransform, GeographicMode, apply_metric_transform_slices, guess_from_slices,
    reproject_geographic_xy_slices,
};
use crate::common::exceptions::{ApbaseError, Result};
use crate::config::{resolve_grid_max_points_ceiling, resolve_n_threads};

/// Boundary geometry type (a GEOS geometry).
pub type Boundary = Geometry;

/// Default maximum number of candidate points tested per chunk.
pub const DEFAULT_CHUNK_SIZE: i64 = 1_000_000;
/// Default concave-hull ratio used when the boundary is inferred.
pub const DEFAULT_HULL_RATIO: f64 = 0.12;

const HULL_DECIMATE_TARGET_POINTS: usize = 20_000;

/// Candidate x/y kept from one block of grid rows.
type BlockPoints = (Vec<f64>, Vec<f64>);

fn is_geographic_bucket(x: &[f64], y: &[f64]) -> bool {
    guess_from_slices(x, y, true) == CoordinateSystem::Geographic
}

/// `geometry.bounds` as `(minx, miny, maxx, maxy)`; NaN for an empty geometry.
fn bounds(geometry: &Geometry) -> Result<(f64, f64, f64, f64)> {
    if geometry.is_empty()? {
        return Ok((f64::NAN, f64::NAN, f64::NAN, f64::NAN));
    }
    let extent = geometry.get_extent()?;
    Ok((extent[0], extent[1], extent[2], extent[3]))
}

/// Apply a vectorized coordinate transform to every coordinate of `geometry`,
/// in GEOS traversal order.
fn transform_geometry(
    geometry: &Geometry,
    transformation: impl FnOnce(&[f64], &[f64]) -> Result<(Array1<f64>, Array1<f64>)>,
) -> Result<Geometry> {
    let collected = RefCell::new((Vec::new(), Vec::new()));
    geometry.transform_xy(|x, y| {
        let mut coords = collected.borrow_mut();
        coords.0.push(x);
        coords.1.push(y);
        Ok::<_, geos::Error>((x, y))
    })?;
    let (xs, ys) = collected.into_inner();
    let (new_x, new_y) = transformation(&xs, &ys)?;
    let position = Cell::new(0usize);
    let transformed = geometry.transform_xy(|_, _| {
        let i = position.get();
        position.set(i + 1);
        Ok::<_, geos::Error>((new_x[i], new_y[i]))
    })?;
    Ok(transformed)
}

/// Options of [`Grid`]; `Default` gives the standard settings.
#[derive(Clone)]
pub struct GridOptions {
    /// Optional interpolation domain; when `None`, a concave hull is inferred.
    pub boundary: Option<Boundary>,
    /// Grid spacing in the same units as the input coordinates.
    pub resolution: f64,
    /// Maximum number of candidate points tested per chunk.
    pub chunk_size: i64,
    /// Concave hull ratio used when the boundary is inferred.
    pub hull_ratio: f64,
    /// Transform applied to `boundary` so it lands in the frame of `x`/`y`
    /// already converted with `prepare_metric_xy`.
    pub coordinate_transform: Option<CoordinateTransform>,
    /// Reproject a geographic-looking `boundary` when `x`/`y` are metric.
    pub geographic_mode: Option<GeographicMode>,
}

impl Default for GridOptions {
    fn default() -> Self {
        GridOptions {
            boundary: None,
            resolution: 20.0,
            chunk_size: DEFAULT_CHUNK_SIZE,
            hull_ratio: DEFAULT_HULL_RATIO,
            coordinate_transform: None,
            geographic_mode: None,
        }
    }
}

/// Generate regular target points inside a boundary.
///
/// The boundary either comes from the caller (`boundary`, or the
/// [`Grid::from_wkt`]/[`Grid::from_wkb`] constructors) or is inferred as a
/// concave hull of `x`/`y` source points when none is given.
#[derive(Clone)]
pub struct Grid {
    /// Interpolation domain supplied by the caller.
    pub boundary: Option<Boundary>,
    /// Grid spacing.
    pub resolution: f64,
    /// Maximum number of candidate points tested per chunk.
    pub chunk_size: i64,
    /// Concave hull ratio.
    pub hull_ratio: f64,
    /// Transform applied to `boundary` before building the grid.
    pub coordinate_transform: Option<CoordinateTransform>,
    /// Projection used to reconcile a geographic `boundary` with metric `x`/`y`.
    pub geographic_mode: Option<GeographicMode>,
    points: Option<Array2<f64>>,
    fitted_boundary: Option<Boundary>,
    fitted: bool,
}

impl Grid {
    /// Create an unfitted grid; invalid options are rejected here.
    pub fn new(options: GridOptions) -> Result<Self> {
        let grid = Grid {
            boundary: options.boundary,
            resolution: options.resolution,
            chunk_size: options.chunk_size,
            hull_ratio: options.hull_ratio,
            coordinate_transform: options.coordinate_transform,
            geographic_mode: options.geographic_mode,
            points: None,
            fitted_boundary: None,
            fitted: false,
        };
        grid.validate_config()?;
        Ok(grid)
    }

    /// Create a grid and infer its boundary from x/y source points (concave hull).
    pub fn from_data_xy(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
        options: GridOptions,
    ) -> Result<Self> {
        let mut grid = Grid::new(options)?;
        let xy = Grid::convert_xy(x, y)?;
        grid.fit_xy(Some(xy))?;
        Ok(grid)
    }

    /// Create a grid whose boundary is parsed from a WKT string.
    pub fn from_wkt(wkt: &str, options: GridOptions) -> Result<Self> {
        let boundary = Geometry::new_from_wkt(wkt)?;
        let mut grid = Grid::new(GridOptions {
            boundary: Some(boundary),
            ..options
        })?;
        grid.fit(None, None)?;
        Ok(grid)
    }

    /// Create a grid whose boundary is parsed from WKB bytes.
    pub fn from_wkb(wkb: &[u8], options: GridOptions) -> Result<Self> {
        let boundary = Geometry::new_from_wkb(wkb)?;
        let mut grid = Grid::new(GridOptions {
            boundary: Some(boundary),
            ..options
        })?;
        grid.fit(None, None)?;
        Ok(grid)
    }

    /// Whether the grid points have already been generated.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Generated `(n_points, 2)` target coordinates, in Fortran (column-major) layout.
    pub fn points(&self) -> Result<&Array2<f64>> {
        Ok(self.fitted_state()?.0)
    }

    /// Buffered boundary used by the point-in-boundary tests.
    pub fn fitted_boundary(&self) -> Result<&Boundary> {
        Ok(self.fitted_state()?.1)
    }

    /// Thread count used to parallelize boundary containment tests.
    pub fn n_threads(&self) -> i64 {
        resolve_n_threads()
    }

    /// Generate grid points.
    ///
    /// With a `boundary`, `x`/`y` are optional and only reconcile the boundary
    /// with `coordinate_transform`/`geographic_mode`. Without one, `x`/`y` are
    /// required and a concave hull of their finite pairs becomes the boundary.
    pub fn fit(&mut self, x: Option<&dyn ArrayLike>, y: Option<&dyn ArrayLike>) -> Result<&mut Self> {
        let xy = match (x, y) {
            (Some(x), Some(y)) => Some(Grid::convert_xy(x, y)?),
            _ => None,
        };
        self.fit_xy(xy)
    }

    fn convert_xy(
        x: &(impl ArrayLike + ?Sized),
        y: &(impl ArrayLike + ?Sized),
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let x_array = as_float64_1d(x, "x")?;
        let y_array = as_float64_1d(y, "y")?;
        validate_same_size_xy(&x_array, &y_array)?;
        Ok((x_array, y_array))
    }

    pub(crate) fn fit_xy(&mut self, xy: Option<(Array1<f64>, Array1<f64>)>) -> Result<&mut Self> {
        let boundary = if let Some(raw) = &self.boundary {
            let mut raw_boundary = Clone::clone(raw);
            if let Some((x_array, y_array)) = xy {
                let (x_valid, y_valid) = filter_finite_xy(&x_array, &y_array);
                if !x_valid.is_empty() {
                    raw_boundary = self.reconcile_boundary_crs(
                        raw_boundary,
                        x_valid.as_slice().expect("contiguous"),
                        y_valid.as_slice().expect("contiguous"),
                    )?;
                }
            }
            self.prepare_boundary(&raw_boundary)?
        } else {
            let Some((x_array, y_array)) = xy else {
                return Err(ApbaseError::value_error(
                    "x and y are required to infer a boundary when none is provided",
                ));
            };
            let (x_valid, y_valid) = filter_finite_xy(&x_array, &y_array);
            if x_valid.is_empty() {
                return Err(ApbaseError::value_error(
                    "grid requires at least one finite point",
                ));
            }
            self.infer_boundary(
                x_valid.as_slice().expect("contiguous"),
                y_valid.as_slice().expect("contiguous"),
            )?
        };

        let points = self.generate_points(&boundary)?;
        self.points = Some(points);
        self.fitted_boundary = Some(boundary);
        self.fitted = true;
        Ok(self)
    }

    /// Fit the grid and return the generated target coordinates.
    pub fn generate(&mut self, x: Option<&dyn ArrayLike>, y: Option<&dyn ArrayLike>) -> Result<Array2<f64>> {
        Ok(self.fit(x, y)?.points()?.clone())
    }

    /// `grid(x, y)`: same as [`Grid::generate`].
    pub fn call(&mut self, x: Option<&dyn ArrayLike>, y: Option<&dyn ArrayLike>) -> Result<Array2<f64>> {
        self.generate(x, y)
    }

    fn infer_boundary(&self, x: &[f64], y: &[f64]) -> Result<Boundary> {
        let (hull_x, hull_y) = Grid::decimate_for_hull(x, y);
        let points = hull_x
            .iter()
            .zip(&hull_y)
            .map(|(px, py)| {
                let coords = CoordSeq::new_from_vec(&[[*px, *py]])?;
                Geometry::create_point(coords)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let multipoint = Geometry::create_multipoint(points)?;
        let boundary = multipoint.concave_hull(self.hull_ratio, false)?;
        self.prepare_boundary(&boundary)
    }

    fn prepare_boundary(&self, boundary: &Boundary) -> Result<Boundary> {
        if boundary.is_empty()? {
            return Err(ApbaseError::value_error("the resulting boundary is empty"));
        }
        let buffered = boundary.buffer(self.resolution * 0.5, 8)?;
        if buffered.is_empty()? {
            return Err(ApbaseError::value_error("the buffered boundary is empty"));
        }
        Ok(buffered)
    }

    fn reconcile_boundary_crs(&self, boundary: Boundary, x: &[f64], y: &[f64]) -> Result<Boundary> {
        if let Some(transform) = &self.coordinate_transform {
            return transform_geometry(&boundary, |xs, ys| {
                apply_metric_transform_slices(xs, ys, transform)
            });
        }

        if let Some(mode) = self.geographic_mode {
            let (minx, miny, maxx, maxy) = bounds(&boundary)?;
            let boundary_geographic = is_geographic_bucket(&[minx, maxx], &[miny, maxy]);
            if boundary_geographic && !is_geographic_bucket(x, y) {
                return transform_geometry(&boundary, |xs, ys| {
                    reproject_geographic_xy_slices(xs, ys, Some(mode))
                });
            }
        }

        Grid::warn_if_boundary_crs_mismatch(&boundary, x, y)?;
        Ok(boundary)
    }

    fn warn_if_boundary_crs_mismatch(boundary: &Boundary, x: &[f64], y: &[f64]) -> Result<()> {
        let (minx, miny, maxx, maxy) = bounds(boundary)?;
        if is_geographic_bucket(x, y) != is_geographic_bucket(&[minx, maxx], &[miny, maxy]) {
            log::warn!(
                "Grid boundary and input x/y coordinates appear to be in different coordinate systems (one looks \
                 geographic/lon-lat, the other looks projected/metric). Grid.resolution is interpreted in the same \
                 unit as x/y, so this mismatch likely means resolution and/or area are wrong. Make sure boundary and \
                 x/y share the same CRS (see apbase.common.coordinates.prepare_metric_xy)."
            );
        }
        Ok(())
    }

    fn axis_range(lo: f64, hi: f64, step: f64) -> Result<Vec<f64>> {
        let start = (lo / step).floor() * step;
        let stop = (hi / step).ceil() * step;
        let count_float = _numpy::rint((stop - start) / step);
        if count_float.is_nan() {
            return Err(ApbaseError::value_error("cannot convert float NaN to integer"));
        }
        let count = count_float as i64 + 1;
        if count <= 0 {
            return Ok(Vec::new());
        }
        Ok(_numpy::linspace(start, stop, count as usize))
    }

    fn generate_points(&self, boundary: &Boundary) -> Result<Array2<f64>> {
        let (min_x, min_y, max_x, max_y) = bounds(boundary)?;
        let step = self.resolution;
        let x_axis = Grid::axis_range(min_x, max_x, step)?;
        let y_axis = Grid::axis_range(min_y, max_y, step)?;

        let nx = x_axis.len();
        if nx == 0 {
            return Err(ApbaseError::value_error("empty grid on the x axis"));
        }
        let ny = y_axis.len();
        if ny == 0 {
            return Err(ApbaseError::value_error("empty grid on the y axis"));
        }

        let candidate_count = (nx as i64).saturating_mul(ny as i64);
        let ceiling = resolve_grid_max_points_ceiling();
        if candidate_count > ceiling {
            return Err(ApbaseError::value_error(format!(
                "grid would generate {} candidate points ({} x {}) before boundary filtering, above the {} ceiling \
                 (apbase.config[\"grid_max_points_ceiling\"]) -- check that `resolution` and `boundary` share the \
                 same unit/CRS; a common cause is geographic input auto-detected and reprojected to UTM meters while \
                 `resolution` was sized for a much coarser local unit",
                format_thousands(candidate_count),
                format_thousands(nx as i64),
                format_thousands(ny as i64),
                format_thousands(ceiling),
            )));
        }

        let rows_per_chunk = ((self.chunk_size / nx as i64).max(1)) as usize;
        let n_blocks = ny.div_ceil(rows_per_chunk);
        let n_threads = resolve_n_threads();

        let parts = if n_threads > 1 && n_blocks > 1 {
            Grid::intersects_blocks_parallel(boundary, &x_axis, &y_axis, rows_per_chunk, n_threads as usize)?
        } else {
            let prepared = boundary.to_prepared_geom()?;
            (0..n_blocks)
                .map(|block| Grid::filter_block(&prepared, &x_axis, &y_axis, rows_per_chunk, block))
                .collect::<Result<Vec<_>>>()?
        };

        let output_size: usize = parts.iter().map(|(px, _)| px.len()).sum();
        if output_size == 0 {
            return Err(ApbaseError::value_error(
                "no grid point falls inside the boundary",
            ));
        }
        let mut points = Array2::<f64>::zeros((output_size, 2).f());
        let mut row = 0;
        for (part_x, part_y) in parts {
            for (px, py) in part_x.into_iter().zip(part_y) {
                points[[row, 0]] = px;
                points[[row, 1]] = py;
                row += 1;
            }
        }
        Ok(points)
    }

    /// Candidates of one block (`np.tile(x_axis, rows)`, `np.repeat(y_block, nx)`)
    /// that intersect the boundary.
    fn filter_block(
        prepared: &geos::PreparedGeometry<'_>,
        x_axis: &[f64],
        y_axis: &[f64],
        rows_per_chunk: usize,
        block: usize,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let start = block * rows_per_chunk;
        let end = (start + rows_per_chunk).min(y_axis.len());
        let mut kept_x = Vec::new();
        let mut kept_y = Vec::new();
        for &cy in &y_axis[start..end] {
            for &cx in x_axis {
                if prepared.intersects_xy(cx, cy)? {
                    kept_x.push(cx);
                    kept_y.push(cy);
                }
            }
        }
        Ok((kept_x, kept_y))
    }

    /// Run the containment tests on `n_threads` threads, keeping block order
    /// (downstream `MapResult.to_raster` relies on row-major order). Each thread
    /// prepares its own copy of the boundary.
    fn intersects_blocks_parallel(
        boundary: &Boundary,
        x_axis: &[f64],
        y_axis: &[f64],
        rows_per_chunk: usize,
        n_threads: usize,
    ) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
        let n_blocks = y_axis.len().div_ceil(rows_per_chunk);
        let next_block = AtomicUsize::new(0);
        let results: Mutex<Vec<Option<Result<BlockPoints>>>> =
            Mutex::new((0..n_blocks).map(|_| None).collect());

        std::thread::scope(|scope| {
            for _ in 0..n_threads.min(n_blocks) {
                scope.spawn(|| {
                    let geometry = Clone::clone(boundary);
                    let prepared = match geometry.to_prepared_geom() {
                        Ok(prepared) => prepared,
                        Err(error) => {
                            let block = next_block.fetch_add(1, Ordering::SeqCst);
                            if block < n_blocks {
                                results.lock().expect("no poisoned lock")[block] = Some(Err(error.into()));
                            }
                            return;
                        }
                    };
                    loop {
                        let block = next_block.fetch_add(1, Ordering::SeqCst);
                        if block >= n_blocks {
                            break;
                        }
                        let outcome = Grid::filter_block(&prepared, x_axis, y_axis, rows_per_chunk, block);
                        results.lock().expect("no poisoned lock")[block] = Some(outcome);
                    }
                });
            }
        });

        results
            .into_inner()
            .expect("no poisoned lock")
            .into_iter()
            .map(|slot| slot.unwrap_or_else(|| Err(ApbaseError::value_error("grid block was not processed"))))
            .collect()
    }

    fn decimate_for_hull(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = x.len();
        if n <= HULL_DECIMATE_TARGET_POINTS {
            return (x.to_vec(), y.to_vec());
        }

        let area = (_numpy::max(x) - _numpy::min(x)) * (_numpy::max(y) - _numpy::min(y));
        if !area.is_finite() || area <= 0.0 {
            return (x.to_vec(), y.to_vec());
        }
        let cell = (area / HULL_DECIMATE_TARGET_POINTS as f64).sqrt();
        if !cell.is_finite() || cell <= 0.0 {
            return (x.to_vec(), y.to_vec());
        }

        let inv_cell = 1.0 / cell;
        let mut ix: Vec<i64> = x.iter().map(|value| (value * inv_cell).floor() as i64).collect();
        let mut iy: Vec<i64> = y.iter().map(|value| (value * inv_cell).floor() as i64).collect();
        let ix_min = *ix.iter().min().expect("non-empty");
        let iy_min = *iy.iter().min().expect("non-empty");
        ix.iter_mut().for_each(|value| *value -= ix_min);
        iy.iter_mut().for_each(|value| *value -= iy_min);
        let iy_max = *iy.iter().max().expect("non-empty");
        let keys: Vec<i64> = ix.iter().zip(&iy).map(|(a, b)| a * (iy_max + 1) + b).collect();

        let first_positions = _numpy::unique_first_positions(&keys);
        if first_positions.len() == n {
            return (x.to_vec(), y.to_vec());
        }
        (
            first_positions.iter().map(|p| x[*p]).collect(),
            first_positions.iter().map(|p| y[*p]).collect(),
        )
    }

    fn fitted_state(&self) -> Result<(&Array2<f64>, &Boundary)> {
        match (&self.points, &self.fitted_boundary) {
            (Some(points), Some(boundary)) if self.fitted => Ok((points, boundary)),
            _ => Err(ApbaseError::value_error("call fit() before accessing the grid")),
        }
    }

    fn validate_config(&self) -> Result<()> {
        if !self.resolution.is_finite() || self.resolution <= 0.0 {
            return Err(ApbaseError::value_error(
                "resolution must be finite and greater than zero",
            ));
        }
        if self.chunk_size <= 0 {
            return Err(ApbaseError::value_error("chunk_size must be greater than zero"));
        }
        if !self.hull_ratio.is_finite() || self.hull_ratio < 0.0 || self.hull_ratio > 1.0 {
            return Err(ApbaseError::value_error("hull_ratio must be between 0 and 1"));
        }
        Ok(())
    }
}

impl Targets for Grid {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        let points = self.points()?;
        Ok((points.column(0).to_owned(), points.column(1).to_owned()))
    }
}

impl std::fmt::Debug for Grid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Grid")
            .field("resolution", &self.resolution)
            .field("chunk_size", &self.chunk_size)
            .field("hull_ratio", &self.hull_ratio)
            .field("coordinate_transform", &self.coordinate_transform)
            .field("geographic_mode", &self.geographic_mode)
            .field("is_fitted", &self.fitted)
            .finish_non_exhaustive()
    }
}
