//! Links the Fortran kernels shipped in `native/<target>/libapbase_native.a`
//! and the runtime libraries they need: libgfortran, libgomp (OpenMP), a
//! BLAS/LAPACK providing `dsyev_` (OpenBLAS by default) and, on Linux, libm
//! and libmvec.
//!
//! The runtime comes from the system: the distribution packages on Linux,
//! Homebrew `gcc` and `openblas` on macOS, MSYS2 MINGW64 on Windows (GNU
//! toolchain, `x86_64-pc-windows-gnu`).
//!
//! Environment overrides:
//! - `APBASE_NATIVE_LIB_DIR`: directory containing `libapbase_native.a`
//!   (development builds, unsupported targets).
//! - `APBASE_FORTRAN_LIB_DIR`: directory containing libgfortran and libgomp,
//!   when they are not found automatically.
//! - `APBASE_BLAS_LIBS`: space-separated BLAS/LAPACK libraries to link instead
//!   of `openblas`; `-L<dir>` entries add search directories.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SUPPORTED_TARGETS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-gnu",
];
const RUNTIME_LIBS: &[&str] = &["gfortran", "gomp"];
const LINUX_MATH_LIBS: &[&str] = &["mvec", "m"];
const HOMEBREW_PREFIXES: &[&str] = &["/opt/homebrew", "/usr/local"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=APBASE_NATIVE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=APBASE_FORTRAN_LIB_DIR");
    println!("cargo:rerun-if-env-changed=APBASE_BLAS_LIBS");
    println!("cargo:rerun-if-env-changed=FC");
    println!("cargo:rerun-if-env-changed=HOMEBREW_PREFIX");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let target = env::var("TARGET").expect("cargo always sets TARGET for build scripts");
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("cargo sets CARGO_CFG_TARGET_OS");
    let lib_dir = match env::var_os("APBASE_NATIVE_LIB_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => {
            if !SUPPORTED_TARGETS.contains(&target.as_str()) {
                let hint = if target.ends_with("-windows-msvc") {
                    "; on Windows use the GNU toolchain (`rustup target add x86_64-pc-windows-gnu`)"
                } else {
                    ""
                };
                panic!(
                    "apbase ships its native kernels only for {SUPPORTED_TARGETS:?}; target `{target}` is \
                     not supported yet{hint} (or set APBASE_NATIVE_LIB_DIR to a directory containing a \
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

    for dir in fortran_lib_dirs(&target, &target_os) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in RUNTIME_LIBS {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }

    match env::var("APBASE_BLAS_LIBS") {
        Ok(blas_libs) => {
            for item in blas_libs.split_whitespace() {
                match item.strip_prefix("-L") {
                    Some(dir) => println!("cargo:rustc-link-search=native={dir}"),
                    None => println!("cargo:rustc-link-lib=dylib={}", item.trim_start_matches("-l")),
                }
            }
        }
        Err(_) => {
            if target_os == "macos" {
                // Homebrew's openblas is keg-only: not in the default linker paths.
                for dir in homebrew_dirs("opt/openblas/lib") {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
            }
            println!("cargo:rustc-link-lib=dylib=openblas");
        }
    }

    if target_os == "linux" {
        for lib in LINUX_MATH_LIBS {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

/// Directories holding libgfortran and libgomp that the linker may not search
/// by itself: the system linker on macOS does not know Homebrew GCC's paths.
fn fortran_lib_dirs(target: &str, target_os: &str) -> Vec<PathBuf> {
    if let Some(dir) = env::var_os("APBASE_FORTRAN_LIB_DIR") {
        return vec![PathBuf::from(dir)];
    }
    let extension = match target_os {
        "macos" => "dylib",
        "windows" => "dll.a",
        _ => return Vec::new(),
    };
    let mut dirs = Vec::new();
    // The installed gfortran knows where its runtime is; only meaningful when not cross-compiling.
    if env::var("HOST").as_deref() == Ok(target) {
        let fc = env::var("FC").unwrap_or_else(|_| "gfortran".to_string());
        for lib in RUNTIME_LIBS {
            if let Some(dir) = compiler_lib_dir(&fc, &format!("lib{lib}.{extension}")) {
                if !dirs.contains(&dir) {
                    dirs.push(dir);
                }
            }
        }
    }
    if dirs.is_empty() && target_os == "macos" {
        dirs = homebrew_dirs("opt/gcc/lib/gcc/current");
    }
    dirs
}

/// `<fc> -print-file-name=<file>` gives the full path when the compiler ships
/// that file, and the bare name otherwise.
fn compiler_lib_dir(fc: &str, file: &str) -> Option<PathBuf> {
    let output = Command::new(fc)
        .arg(format!("-print-file-name={file}"))
        .output()
        .ok()?;
    let path = PathBuf::from(String::from_utf8(output.stdout).ok()?.trim());
    if path.is_absolute() && path.is_file() {
        path.parent().map(Path::to_path_buf)
    } else {
        None
    }
}

/// Existing `<prefix>/<relative>` directories, `$HOMEBREW_PREFIX` first.
fn homebrew_dirs(relative: &str) -> Vec<PathBuf> {
    let mut prefixes: Vec<PathBuf> = env::var_os("HOMEBREW_PREFIX")
        .map(PathBuf::from)
        .into_iter()
        .collect();
    prefixes.extend(HOMEBREW_PREFIXES.iter().map(PathBuf::from));
    let mut dirs: Vec<PathBuf> = Vec::new();
    for prefix in prefixes {
        let dir = prefix.join(relative);
        if dir.is_dir() && !dirs.contains(&dir) {
            dirs.push(dir);
        }
    }
    dirs
}
