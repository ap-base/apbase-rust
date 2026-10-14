//! Regular target-grid generation, shared by IDW and kriging interpolation.
//!
//! Exports [`Grid`] and the functional wrapper [`create_point_grid`].

mod _grid;

use ndarray::Array2;

pub use _grid::{Boundary, DEFAULT_CHUNK_SIZE, DEFAULT_HULL_RATIO, Grid, GridOptions};

use crate::common::arrays::ArrayLike;
use crate::common::exceptions::Result;

/// Options of [`create_point_grid`].
///
/// `resolution` has no default, so it is the argument of [`CreatePointGridOptions::new`].
pub struct CreatePointGridOptions<'a> {
    /// Source x coordinates, required only when `boundary` is omitted.
    pub x: Option<&'a dyn ArrayLike>,
    /// Source y coordinates, required only when `boundary` is omitted.
    pub y: Option<&'a dyn ArrayLike>,
    /// Grid spacing in the same coordinate unit as `x` and `y`.
    pub resolution: f64,
    /// Optional interpolation boundary.
    pub boundary: Option<Boundary>,
    /// Maximum number of candidate points tested per chunk.
    pub chunk_size: i64,
    /// Concave hull ratio used when `boundary` is omitted.
    pub hull_ratio: f64,
}

impl CreatePointGridOptions<'_> {
    /// Defaults for every argument except the required `resolution`.
    pub fn new(resolution: f64) -> Self {
        CreatePointGridOptions {
            x: None,
            y: None,
            resolution,
            boundary: None,
            chunk_size: DEFAULT_CHUNK_SIZE,
            hull_ratio: DEFAULT_HULL_RATIO,
        }
    }
}

/// Create regular target coordinates, from a boundary or from `x`/`y`.
///
/// Returns the `(n_points, 2)` generated coordinates inside the boundary.
pub fn create_point_grid(options: CreatePointGridOptions<'_>) -> Result<Array2<f64>> {
    let mut grid = Grid::new(GridOptions {
        boundary: options.boundary,
        resolution: options.resolution,
        chunk_size: options.chunk_size,
        hull_ratio: options.hull_ratio,
        ..GridOptions::default()
    })?;
    grid.generate(options.x, options.y)
}
