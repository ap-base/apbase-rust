//! Shared helpers of the integration tests.

#![allow(dead_code)]

pub mod numpy_rng;
mod ziggurat;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

use ndarray::{Array, Array1, Array2, Dimension};

#[allow(unused_imports)]
pub use numpy_rng::NumpyRng;

/// Repository root.
pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `tests/data/<name>`.
pub fn data_path(name: &str) -> PathBuf {
    root().join("tests").join("data").join(name)
}

static CONFIG_LOCK: Mutex<()> = Mutex::new(());

/// Serialize tests touching the process-wide `apbase::config` (cargo runs tests
/// in parallel threads) and start every such test from an empty config.
pub fn config_guard() -> MutexGuard<'static, ()> {
    let guard = CONFIG_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    apbase::config._clear();
    guard
}

/// Bitwise equality, NaN equal to NaN.
pub fn same_f64(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

/// Assert `|a - b| <= atol + rtol * |b|` everywhere (NaN positions must match).
#[track_caller]
pub fn assert_allclose<D: Dimension>(
    actual: &Array<f64, D>,
    expected: &Array<f64, D>,
    rtol: f64,
    atol: f64,
    context: &str,
) {
    assert_eq!(actual.shape(), expected.shape(), "{context}: shape");
    for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || b.is_nan() {
            assert!(a.is_nan() && b.is_nan(), "{context}[{i}]: {a} vs {b}");
            continue;
        }
        assert!(
            (a - b).abs() <= atol + rtol * b.abs(),
            "{context}[{i}]: {a} vs {b} (rtol={rtol}, atol={atol})"
        );
    }
}

/// Columns of a CSV file in `tests/data` (empty cells become NaN).
pub fn read_csv(name: &str) -> HashMap<String, Vec<f64>> {
    let mut reader = csv::Reader::from_path(data_path(name)).expect("readable csv");
    let headers: Vec<String> = reader
        .headers()
        .expect("header")
        .iter()
        .map(str::to_string)
        .collect();
    let mut columns: HashMap<String, Vec<f64>> = headers.iter().map(|h| (h.clone(), Vec::new())).collect();
    for record in reader.records() {
        let record = record.expect("valid row");
        for (header, cell) in headers.iter().zip(record.iter()) {
            let value = if cell.is_empty() {
                f64::NAN
            } else {
                cell.parse().unwrap_or(f64::NAN)
            };
            columns.get_mut(header).expect("known column").push(value);
        }
    }
    columns
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting
/// (reference helper standing in for `np.linalg.solve`).
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut m = a.clone();
    let mut rhs = b.clone();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&i, &j| m[[i, col]].abs().total_cmp(&m[[j, col]].abs()))
            .expect("rows");
        if pivot != col {
            for k in 0..n {
                m.swap([col, k], [pivot, k]);
            }
            rhs.swap(col, pivot);
        }
        for row in col + 1..n {
            let factor = m[[row, col]] / m[[col, col]];
            for k in col..n {
                m[[row, k]] -= factor * m[[col, k]];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = Array1::zeros(n);
    for row in (0..n).rev() {
        let mut acc = rhs[row];
        for k in row + 1..n {
            acc -= m[[row, k]] * x[k];
        }
        x[row] = acc / m[[row, row]];
    }
    x
}

/// Eigenvalues of a symmetric matrix (cyclic Jacobi), ascending
/// (reference helper standing in for `np.linalg.eigvalsh`).
pub fn eigvalsh(matrix: &Array2<f64>) -> Vec<f64> {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |j| *j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[[i, j]].powi(2))
            .sum();
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[[p, q]].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * a[[p, q]]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[[k, p]];
                    let akq = a[[k, q]];
                    a[[k, p]] = c * akp - s * akq;
                    a[[k, q]] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[[p, k]];
                    let aqk = a[[q, k]];
                    a[[p, k]] = c * apk - s * aqk;
                    a[[q, k]] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut values: Vec<f64> = (0..n).map(|i| a[[i, i]]).collect();
    values.sort_by(f64::total_cmp);
    values
}

/// Path helper for messages.
pub fn display(path: &Path) -> String {
    path.display().to_string()
}

unsafe extern "C" {
    #[link_name = "erfc"]
    fn libm_erfc(x: f64) -> f64;
}

/// Complementary error function (glibc libm).
pub fn erfc(x: f64) -> f64 {
    // SAFETY: erfc is a pure libm function.
    unsafe { libm_erfc(x) }
}

/// `|actual - expected| <= max(rel * |expected|, abs)`.
pub fn approx(actual: f64, expected: f64, rel: f64, abs: f64) -> bool {
    (actual - expected).abs() <= (rel * expected.abs()).max(abs)
}

/// Lower Cholesky factor (reference helper standing in for `np.linalg.cholesky`).
pub fn cholesky(a: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = sum.max(0.0).sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    l
}

/// `np.std` (population) and `np.mean` used by reference helpers.
pub fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}
