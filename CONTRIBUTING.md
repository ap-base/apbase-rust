# Contributing to APbase (Rust)

Thanks for your interest in improving APbase. Bug reports, documentation fixes,
tests and code changes are all welcome.

- Questions and ideas: [Discussions](https://github.com/ap-base/apbase-rust/discussions).
- Bugs and feature requests: [Issues](https://github.com/ap-base/apbase-rust/issues)
  (use the templates).
- Security problems: never in a public issue; see [SECURITY.md](SECURITY.md).

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

1. Install Rust 1.85 or newer and the system libraries for your platform
   (Linux, macOS or Windows with the GNU toolchain), as listed under
   [Install](README.md#install) in the README.
2. Clone the repository and build:

   ```bash
   git clone https://github.com/ap-base/apbase-rust.git
   cd apbase-rust
   cargo build --release
   ```

The numerical kernels are a prebuilt static library under
`native/<target>/libapbase_native.a`. Pull requests do not modify these files;
if you find a problem in a kernel, open an issue with a minimal example.

## Checks

Run the same checks as CI before opening a pull request:

```bash
cargo fmt --all -- --check
cargo clippy --release --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --release --no-deps
cargo test --release
cargo run --release --example quick_use
cargo run --release --example cokriging
```

CI runs the tests on Linux, macOS (Apple Silicon and Intel) and Windows (GNU
toolchain), and checks the minimum Rust version (1.85).

## Code conventions

- Optional settings are `<Name>Options` structs with `Default`; required ones
  are constructor arguments.
- Estimators follow `new(options) -> fit(&x, &y, &z) -> interpolate(&targets)`.
- Every fallible call returns `Result<T, ApbaseError>`; errors carry an
  `ErrorKind`.
- Foreign declarations live only in `src/_native/_lib.rs`, and `unsafe` code
  only in the `_native` layer (`tests/test_architecture.rs` checks this).
- Keep public names, defaults and error messages stable; a change to any of
  them goes in [CHANGELOG.md](CHANGELOG.md).
- Add tests next to the behavior you change (`tests/test_<module>.rs`).

## Pull requests

1. Fork the repository (or create a branch, if you have access) from `main`.
2. Keep each pull request focused on one change, with tests and docs.
3. Add a line to the `Unreleased` section of [CHANGELOG.md](CHANGELOG.md).
4. Open the pull request against `main`. It needs a passing CI and an approval
   from a code owner before merging.

## Releases (maintainers)

1. Refresh the four native libraries from the kernels release and commit them:

   ```bash
   KERNELS_TOKEN="$(gh auth token)" .github/scripts/fetch_native_library.sh <kernels-tag> \
     x86_64-unknown-linux-gnu aarch64-apple-darwin x86_64-apple-darwin x86_64-pc-windows-gnu
   ```

2. Bump `version` in `Cargo.toml` and the version in the README install snippet.
3. Update `version` and `date-released` in `CITATION.cff`.
4. Move the `Unreleased` entries of `CHANGELOG.md` under the new version.
5. Run the "Release" workflow with the version and the same kernels tag. It
   checks that `Cargo.toml` and `CITATION.cff` agree with the version, runs CI
   on those native libraries, and creates the GitHub release with the packaged
   `.crate`.
6. Publish to crates.io with `cargo publish`.
