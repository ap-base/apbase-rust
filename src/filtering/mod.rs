//! High-level spatial filtering API.

mod _filter;
mod _native;

pub use _filter::{
    DEFAULT_CELL_FACTOR, DEFAULT_GLOBAL_IQR_WHISKER, DEFAULT_MAX_NEIGHBORS, FILTER_LEVELS, FilterLevel,
    FilterStatistics, SpatialFilter, SpatialFilterOptions,
};
