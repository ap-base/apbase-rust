//! Links the Fortran kernels shipped in `native/<target>/libapbase_native.a`
//! and the runtime libraries they need: libgfortran, libgomp (OpenMP), a
//! BLAS/LAPACK providing `dsyev_` (OpenBLAS by default), libm and libmvec.
//!
//! Environment overrides:
//! - `APBASE_NATIVE_LIB_DIR`: directory containing `libapbase_native.a`
//!   (development builds, unsupported targets).
//! - `APBASE_BLAS_LIBS`: space-separated BLAS/LAPACK libraries to link instead
//!   of `openblas`.

use std::env;
use std::path::PathBuf;

const SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu"];
const RUNTIME_LIBS: &[&str] = &["gfortran", "gomp"];
const MATH_LIBS: &[&str] = &["mvec", "m"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=APBASE_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=APBASE_BLAS_LIBS");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    // docs.rs only renders documentation: nothing is linked there.
    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let target = env::var("TARGET").expect("cargo always sets TARGET for build scripts");
    let lib_dir = match env::var_os("APBASE_NATIVE_LIB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            if !SUPPORTED_TARGETS.contains(&target.as_str()) {
                panic!(
                    "apbase ships its native kernels only for {SUPPORTED_TARGETS:?}; target `{target}` is \
                     not supported yet (set APBASE_NATIVE_LIB_DIR to a directory containing a \
                     libapbase_native.a built for it)"
                );
            }
            PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"))
                .join("native")
                .join(&target)
        }
    };

    let archive = lib_dir.join("libapbase_native.a");
    if !archive.is_file() {
        panic!("apbase native library not found: {}", archive.display());
    }
    println!("cargo:rerun-if-changed={}", archive.display());
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=apbase_native");

    for lib in RUNTIME_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    let blas_libs = env::var("APBASE_BLAS_LIBS").unwrap_or_else(|_| "openblas".to_string());
    for lib in blas_libs.split_whitespace() {
        println!("cargo:rustc-link-lib=dylib={}", lib.trim_start_matches("-l"));
    }
    for lib in MATH_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
