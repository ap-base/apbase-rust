//! Shared building blocks: array coercion, coordinate handling, errors and validation.

pub(crate) mod _numpy;
pub mod arrays;
pub mod coordinates;
pub mod exceptions;
pub mod validation;

pub use arrays::{ArrayLike, Element, Targets, as_float64_1d, extract_target_xy};
pub use coordinates::{
    CoordinateSystem, CoordinateTransform, EARTH_RADIUS_M, GeographicMode, apply_metric_transform,
    guess_xy_coordinate_system, prepare_metric_xy, prepare_metric_xy_with_transform, reproject_geographic_xy,
    restore_original_xy,
};
pub use exceptions::{ApbaseError, ErrorKind, Result};
