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
apbase = "0.1.4"
```

Each GitHub [release](https://github.com/ap-base/apbase-rust/releases) also carries the packaged
`apbase-<version>.crate`.

### Supported platforms

| Platform | Rust target | System libraries from |
|---|---|---|
| Linux x86_64 (glibc) | `x86_64-unknown-linux-gnu` | apt (or your distribution) |
| macOS, Apple Silicon | `aarch64-apple-darwin` | Homebrew |
| macOS, Intel | `x86_64-apple-darwin` | Homebrew |
| Windows x86_64 | `x86_64-pc-windows-gnu` | MSYS2 (MINGW64) |

The Fortran kernels are linked statically; the Fortran/OpenMP runtime,
OpenBLAS, GEOS and PROJ come from the system. PROJ must be 9.6.2 or newer; with
an older system PROJ, the `proj-sys` dependency builds PROJ from source, which
also needs `cmake`, `sqlite3` and `libtiff`.

**Linux** (Ubuntu/Debian):

```bash
sudo apt install gfortran libopenblas-dev libgeos-dev libproj-dev pkg-config
```

`gfortran` is not used to compile anything; it and `libopenblas-dev` provide the
`libgfortran.so` and `libopenblas.so` the link step needs. To run a built binary
(for example in a deployment image), only the shared runtime libraries are
needed: `libgfortran5`, `libgomp1`, `libopenblas0`, the GEOS C library
(`libgeos-c1t64` on Ubuntu 24.04), and the system PROJ when it was linked from
the system. A PROJ built from source is linked statically and needs only
`libsqlite3-0` at run time.

**macOS** (Homebrew):

```bash
brew install gcc openblas geos proj pkgconf
```

`gcc` provides libgfortran and libgomp. The build script finds them through
`gfortran`, and Homebrew's keg-only `openblas` under the Homebrew prefix. Built
binaries load these libraries from the Homebrew prefix at run time.

**Windows** (GNU toolchain): install [MSYS2](https://www.msys2.org) and, in its
MINGW64 shell:

```bash
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-openblas \
  mingw-w64-x86_64-geos mingw-w64-x86_64-proj mingw-w64-x86_64-pkgconf
```

Then build with the GNU Rust toolchain, with `C:\msys64\mingw64\bin` first on
`PATH` (it provides the linker, `pkgconf` and the runtime DLLs):

```powershell
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
$env:PKG_CONFIG = "pkgconf"
```

Built programs need the DLLs from `mingw64\bin` (libgfortran, libgomp, OpenBLAS,
GEOS, PROJ) on `PATH` or next to the executable. The MSYS2 MINGW64 shell already
puts them first. Git Bash puts Git's own `mingw64\bin` first instead, whose older
copies of the same DLLs make programs fail to start
(`STATUS_ENTRYPOINT_NOT_FOUND`). The MSVC toolchain (`x86_64-pc-windows-msvc`) is
not supported yet.

When the libraries live elsewhere, `APBASE_FORTRAN_LIB_DIR` (libgfortran and
libgomp) and `APBASE_BLAS_LIBS` (BLAS libraries, `-L<dir>` entries allowed)
point the build to them.

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

## Cokriging

When secondary variables correlated with the target one are available, such as
elevation, distance to a river or a sensor reading at the soil samples,
`apbase::cokriging` uses them in the estimate. Three methods are available:
collocated (Markov-1, the default), ICM (intrinsic coregionalization model) and
LMC (linear model of coregionalization). `apbase::cross_variogram` screens the
candidates first: a Fisher z-test for the correlation and a Cauchy-Schwarz check
that the cross-variogram is admissible.

```rust
use apbase::cokriging::{CoKrigingMethod, CoKrigingOptions, SecondaryInput, co_kriging};
use apbase::cross_variogram::{ScreeningOptions, screen_secondary_variables};

// candidates: IndexMap<String, (x, y, values)> of possible secondary variables.
let screening = screen_secondary_variables(&x, &y, &z, &candidates, ScreeningOptions::default())?;

let estimates = co_kriging(
    &x, &y, &z,
    &SecondaryInput::from(screening.selected),
    &targets,
    CoKrigingOptions { method: CoKrigingMethod::Collocated, ..Default::default() },
)?;
```

A secondary that does not share the primary's sample locations, such as a
raster (NDVI, elevation, apparent electrical conductivity), needs
`max_colocation_dist`: each primary sample is paired with its nearest secondary
point within that distance (half the cell diagonal for a grid) to estimate the
cross-covariances. For dense secondaries, `max_secondary_neighbors` (ICM and
LMC) keeps only the nearest ones; 8 to 16 is usually enough. Cokriging assumes
a roughly linear relation between the variables, so transform a secondary that
does not have one (for example, the square root of a distance).

```rust
let options = CoKrigingOptions {
    method: CoKrigingMethod::Icm,
    max_colocation_dist: Some(28.3), // 40 m raster
    max_secondary_neighbors: Some(8),
    ..Default::default()
};
```

`Map` chooses between IDW and ordinary kriging only; cokriging is called
directly. The test suite runs it on three published geostatistics datasets,
Jura, Meuse and Walker Lake, including the Meuse 40 m grid as a raster
secondary (see `tests/data/README.md`). A runnable version lives in
`examples/cokriging.rs` (`cargo run --release --example cokriging`).

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Questions and ideas go to
[Discussions](https://github.com/ap-base/apbase-rust/discussions); security
problems follow [SECURITY.md](SECURITY.md).

## Citation

If you use APbase in your work, please cite it with the metadata in
[CITATION.cff](CITATION.cff) (the "Cite this repository" button on GitHub).

## Links

- Documentation: https://apbase.io
- Source: https://github.com/ap-base/apbase-rust
- Issues: https://github.com/ap-base/apbase-rust/issues
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## License

Apache License 2.0. See [LICENSE](LICENSE).
