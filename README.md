# APbase (Rust)

[![Crates.io](https://img.shields.io/crates/v/apbase.svg)](https://crates.io/crates/apbase)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](Cargo.toml)
[![Kernel: Fortran](https://img.shields.io/badge/kernel-Fortran-734f96.svg)](#performance)
[![Parallel: OpenMP](https://img.shields.io/badge/parallel-OpenMP-orange.svg)](#performance)

<p align="center">
  <img src="assets/main_en.png" alt="APbase Rust" width="660"/>
</p>

APbase is an intelligent mapping pipeline for irregular field data, including
yield monitor, soil samples, and sensor grids. It automates the map workflow:
filtering source data, handling geographic coordinates when needed, building
the target grid, comparing interpolation methods, and selecting a ready-to-use
map without requiring users to choose a mathematical model by hand.

Full documentation: https://apbase.io

## Install

```bash
cargo add apbase
```

or in `Cargo.toml`:

```toml
[dependencies]
apbase = "0.1.2"
```

Each GitHub [release](https://github.com/ap-base/apbase-rust/releases) also carries the packaged
`apbase-<version>.crate`.

System libraries (Fortran/OpenMP/BLAS runtime, GEOS and PROJ):

```bash
sudo apt install libgfortran5 libgomp1 libopenblas0 libgeos-dev libproj-dev pkg-config
```

Supported target today: `x86_64-unknown-linux-gnu`.

## Quick Use

```rust
use apbase::{Map, MapOptions};

apbase::config.set("n_threads", 4)?;

let result = Map::new(MapOptions::new(10.0)).call(&x, &y, &z)?;

result.method;                          // "idw" or "kriging"
(&result.x, &result.y, &result.z);      // compact output points

let (rx, ry, rz) = result.to_raster()?; // dense raster, NaN outside the boundary
```

Functional shortcut:

```rust
let result = apbase::create_map(&x, &y, &z, MapOptions::new(10.0))?;
```

`x`, `y` and `z` can be slices, `Vec`s, arrays or `ndarray` arrays of any
numeric type. A runnable version lives in `examples/quick_use.rs`
(`cargo run --release --example quick_use`).

## Main API

- `apbase::Map`: automated map pipeline.
- `apbase::create_map`: one-shot map creation.
- `apbase::config`: runtime configuration, including OpenMP thread count.

Advanced modules are available under `apbase::filtering`, `apbase::grid`,
`apbase::variogram`, `apbase::idw`, `apbase::kriging`, `apbase::cokriging`,
`apbase::cross_variogram`, and `apbase::cross_validate`.

### Conventions

```rust
use apbase::common::ErrorKind;
use apbase::idw::{IDW, IDWOptions};
use apbase::{MapOptions, create_map};

// Optional settings are `<Name>Options` structs with `Default`;
// required ones are constructor arguments.
let result = create_map(&x, &y, &z, MapOptions { seed: Some(7), ..MapOptions::new(10.0) })?;

// Estimators: new(options) -> fit(&x, &y, &z) -> interpolate(&targets).
let estimates = IDW::new(IDWOptions { power: 3.0, ..Default::default() })?
    .fit(&x, &y, &z)?
    .interpolate(&targets)?;

// Errors carry a kind; `is` also matches the parent kinds.
match apbase::kriging::ordinary_kriging(&x, &y, &z, &targets, Default::default()) {
    Err(e) if e.is(ErrorKind::InterpolationError) => { /* ... */ }
    other => { /* ... */ }
}
```

- `Map` runs with `.call(&x, &y, &z)`; fitted values are read through getters
  (`kriging.model_params()?.range`).
- Results by method are `IndexMap`s in a stable order (`cv.by_method["icm"].rmse`).
- Every fallible call returns `Result<T, ApbaseError>`.

## Data Requirements

- `x`, `y`, and `z` must be one-dimensional arrays.
- Projected metric coordinates, such as UTM, are used directly.
- Geographic lon/lat input is converted internally before distance operations;
  distance settings such as `resolution` are meters.

## Performance

- Native Fortran kernels, linked statically (`native/<target>/libapbase_native.a`).
- OpenMP parallelism for critical numerical work.
- Local neighborhoods and neighbor limits instead of full distance matrices.
- Compact output by default; dense rasters are materialized only with
  `result.to_raster()`.

Set threads before creating maps:

```rust
apbase::config.set("n_threads", 4)?;
```

## Tests

```bash
cargo test --release            # unit, integration and doc tests
```

## Links

- Documentation: https://apbase.io

## License

Apache License 2.0. See [LICENSE](LICENSE).
