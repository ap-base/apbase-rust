//! Shared high-level variogram API.

mod _native;
mod _variogram;

pub use _variogram::{
    DistanceLike, FULL_RANGE_RADIUS_MAX_POINTS, ModelParams, VARIOGRAM_NAMES, Variogram, VariogramOptions,
};

pub(crate) use _variogram::variogram_name;
