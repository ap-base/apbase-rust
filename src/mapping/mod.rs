//! Automatic high-performance map generation.
//!
//! [`Map`] is the primary API orchestrating `SpatialFilter`, `Grid`,
//! `Variogram`, `cross_validate`, `IDW`, and `Kriging` into one map-building
//! pipeline. [`create_map`] is the functional shortcut around the same pipeline.

mod _mapping;

pub use _mapping::{Bounds, Map, MapOptions, MapResult, create_map};
