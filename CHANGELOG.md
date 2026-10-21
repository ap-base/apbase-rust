# Changelog

All notable changes to the `apbase` crate are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-10-21

### Added

- macOS support, Apple Silicon (`aarch64-apple-darwin`) and Intel
  (`x86_64-apple-darwin`), with the system libraries from Homebrew.
- Windows support with the GNU toolchain (`x86_64-pc-windows-gnu`), with the
  system libraries from MSYS2 (MINGW64). The MSVC toolchain is not supported yet.
- Build variable `APBASE_FORTRAN_LIB_DIR` for a libgfortran/libgomp directory
  the linker does not search; `APBASE_BLAS_LIBS` accepts `-L<dir>` entries.
- `examples/cokriging.rs`: screening and collocated cokriging on the Meuse
  dataset, then ICM with the 40 m grid as a raster secondary.
- Cokriging section in the README; `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
  `SECURITY.md`, `CITATION.cff` and issue templates.
- `CoKrigingOptions` and `CrossValidateCokrigingOptions` fields
  `max_colocation_dist` (pairing distance for the cross-covariances of
  heterotopic secondaries) and `max_secondary_neighbors` (secondary neighbor
  limit of ICM and LMC; `None` keeps `max_neighbors`).

### Fixed

- ICM and LMC cokriging estimated no cross-covariance for a secondary that
  does not share the primary's sample locations (for example, a raster): the
  fit silently set it to zero, and dense secondaries then left the block system
  singular (NaN estimates) or unstable. The new `max_colocation_dist` option
  pairs each primary sample with its nearest secondary point within that
  distance; without it, every method returns a `ValueError` naming the
  secondary instead of fitting a zero cross-covariance.
- With the gaussian model, ICM, LMC and collocated cokriging (and their
  cross-validation) give each variable a nugget of at least 0.01% of its sill.
  A gaussian model without nugget over close sources produced exploding
  estimates; models with a larger nugget, and the other models, are unchanged.

### Changed

- **Breaking:** `CoKrigingOptions` and `CrossValidateCokrigingOptions` have new
  fields; a struct literal that lists every field needs them too (or
  `..Default::default()`). ICM and LMC now refuse a secondary without samples
  paired with the primary instead of fitting a zero cross-covariance (see Fixed).
- CI runs on every branch, with tests on Linux, macOS (Apple Silicon and Intel)
  and Windows. The release workflow runs the same tests on the native libraries
  it packages and checks that `CITATION.cff` carries the released version.

## [0.1.4] - 2026-10-17

### Changed

- Results no longer depend on the unit of the values: variogram fits (now
  Cressie-weighted) and the kriging and cokriging systems are scale-invariant.
- Covariate screening tolerates a 10% excess over the Cauchy-Schwarz bound of
  the cross-variogram.
- ICM cokriging projects its coregionalization matrices onto the
  positive-semidefinite cone; LMC reports projection and allocation failures
  as errors instead of returning an invalid fit.
- `cross_validate::cross_validate_cokriging` leaves out of `by_method` and
  `model_params` a method whose model cannot be fitted, instead of failing the
  call; the first fit error is returned only when no method can be fitted.

### Fixed

- Collocated cokriging returns a `ValueError` naming the secondary when it has
  too few samples at the primary locations to estimate its correlation, instead
  of producing NaN estimates.

## [0.1.3] - 2026-10-16

### Changed

- First release on crates.io (`cargo add apbase`); the package metadata now
  links the repository.

## [0.1.2] - 2026-10-14

### Added

- First public release, for `x86_64-unknown-linux-gnu`.
- `Map` and `create_map`: automated map pipeline that filters the source data,
  converts geographic coordinates when needed, builds the target grid, fits a
  shared variogram, cross-validates IDW against ordinary kriging and keeps the
  better map.
- Advanced modules: `filtering`, `grid`, `variogram`, `idw`, `kriging`,
  `cokriging` (collocated, ICM and LMC), `cross_variogram` (cross-variogram fit
  and secondary-variable screening) and `cross_validate`.
- `config` for runtime settings, including the OpenMP thread count.
- Native Fortran kernels with OpenMP, linked statically. When fewer than
  `min_neighbors` sources lie within the search radius, the `max_neighbors`
  nearest are used, so every target inside the boundary gets a value.

[Unreleased]: https://github.com/ap-base/apbase-rust/compare/0.2.0...HEAD
[0.2.0]: https://github.com/ap-base/apbase-rust/compare/0.1.4...0.2.0
[0.1.4]: https://github.com/ap-base/apbase-rust/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/ap-base/apbase-rust/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/ap-base/apbase-rust/releases/tag/0.1.2
