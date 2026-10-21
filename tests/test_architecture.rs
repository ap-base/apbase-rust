use std::fs;
use std::path::{Path, PathBuf};

fn all_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("readable dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            if !matches!(file_name(&path), "target" | ".git") {
                all_files(&path, out);
            }
        } else {
            out.push(path);
        }
    }
}

fn package_files() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    all_files(&root, &mut files);
    files.retain(|path| path.extension().is_some_and(|ext| ext == "rs"));
    files.sort();
    files
}

fn file_name(path: &Path) -> &str {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
}

fn code(path: &Path) -> String {
    fs::read_to_string(path)
        .expect("utf-8 source")
        .lines()
        .map(|line| match line.find("//") {
            Some(position) => &line[..position],
            None => line,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn test_only_lib_modules_reference_fortran_bindings() {
    let mut violations = Vec::new();
    for path in package_files() {
        if file_name(&path) == "_lib.rs" {
            continue;
        }
        let text = code(&path);
        if text.contains("extern \"C\"") || text.contains("#[link") || text.contains("libapbase_native") {
            violations.push(path.display().to_string());
        }
    }
    assert!(
        violations.is_empty(),
        "foreign declarations outside _native/_lib.rs: {violations:?}"
    );
}

#[test]
fn test_only_lib_modules_call_compiled_lib_symbols() {
    let mut violations = Vec::new();
    for path in package_files() {
        if matches!(file_name(&path), "_lib.rs" | "_native.rs") {
            continue;
        }
        for (number, line) in code(&path).lines().enumerate() {
            if line.contains("lib::") || line.contains("unsafe ") || line.contains("unsafe{") {
                violations.push(format!("{}:{}", path.display(), number + 1));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "native symbols or unsafe code outside the _native layer: {violations:?}"
    );
}

#[test]
fn every_layer_module_exists() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for module in [
        "lib.rs",
        "config.rs",
        "_native/mod.rs",
        "_native/_lib.rs",
        "_native/_runtime.rs",
        "common/mod.rs",
        "common/arrays.rs",
        "common/coordinates.rs",
        "common/exceptions.rs",
        "common/validation.rs",
        "filtering/mod.rs",
        "filtering/_filter.rs",
        "filtering/_native.rs",
        "grid/mod.rs",
        "grid/_grid.rs",
        "variogram/mod.rs",
        "variogram/_variogram.rs",
        "variogram/_native.rs",
        "idw/mod.rs",
        "idw/_interpolation.rs",
        "idw/_native.rs",
        "kriging/mod.rs",
        "kriging/_interpolation.rs",
        "kriging/_native.rs",
        "cokriging/mod.rs",
        "cokriging/_interpolation.rs",
        "cokriging/_native.rs",
        "cross_variogram/mod.rs",
        "cross_variogram/_cross_variogram.rs",
        "cross_variogram/_screening.rs",
        "cross_variogram/_native.rs",
        "cross_validate/mod.rs",
        "cross_validate/_cross_validation.rs",
        "cross_validate/_native.rs",
        "mapping/mod.rs",
        "mapping/_mapping.rs",
    ] {
        assert!(src.join(module).is_file(), "missing src/{module}");
    }
}

#[test]
fn no_fortran_sources_in_the_crate() {
    let mut files = Vec::new();
    all_files(Path::new(env!("CARGO_MANIFEST_DIR")), &mut files);
    let fortran: Vec<String> = files
        .iter()
        .filter(|path| {
            path.extension().and_then(|ext| ext.to_str()).is_some_and(|ext| {
                matches!(
                    ext.to_ascii_lowercase().as_str(),
                    "f" | "f90" | "f95" | "f03" | "f08"
                )
            })
        })
        .map(|path| path.display().to_string())
        .collect();
    assert!(fortran.is_empty(), "Fortran sources in the crate: {fortran:?}");
}
