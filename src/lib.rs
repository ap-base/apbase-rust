//! APbase: high-performance automatic map generation.
//!
//! The public top-level API is intentionally small: use [`Map`] or
//! [`create_map`] to create maps, and [`config`](static@config) for process-wide runtime
//! settings. Lower-level geostatistical building blocks remain available from
//! their dedicated modules.
//!
//! ```no_run
//! use apbase::{Map, MapOptions};
//!
//! # fn load() -> (Vec<f64>, Vec<f64>, Vec<f64>) { unimplemented!() }
//! let (x, y, z) = load();
//! apbase::config.set("n_threads", 4)?;
//!
//! let result = Map::new(MapOptions::new(10.0)).call(&x, &y, &z)?;
//! println!("{}", result.method); // "idw" or "kriging"
//! let (raster_x, raster_y, raster_z) = result.to_raster()?; // NaN outside the boundary
//! # let _ = (raster_x, raster_y, raster_z);
//!
//! let result = apbase::create_map(&x, &y, &z, MapOptions::new(10.0))?;
//! # let _ = result;
//! # Ok::<(), apbase::common::ApbaseError>(())
//! ```

mod _native;
pub mod cokriging;
pub mod common;
pub mod config;
pub mod cross_validate;
pub mod cross_variogram;
pub mod filtering;
pub mod grid;
pub mod idw;
pub mod kriging;
pub mod mapping;
pub mod variogram;

pub use config::config;
pub use mapping::{Map, MapOptions, MapResult, create_map};

/// Package version.
#[allow(non_upper_case_globals)]
pub const __version__: &str = env!("CARGO_PKG_VERSION");
