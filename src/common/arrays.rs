//! Array coercion, finite filtering, target extraction and regular-grid detection.

use geos::{Geom, Geometry, GeometryTypes};
use ndarray::{Array1, ArrayBase, Data, Dimension};

use super::_numpy;
use super::exceptions::{ApbaseError, Result};

/// Numeric element types accepted by [`ArrayLike`] inputs.
pub trait Element: Copy + 'static {
    /// Convert to `f64`.
    fn to_f64(self) -> f64;

    /// Borrow a slice as `&[f64]` when the element type already is `f64`.
    fn as_f64_slice(values: &[Self]) -> Option<&[f64]> {
        let _ = values;
        None
    }
}

impl Element for f64 {
    fn to_f64(self) -> f64 {
        self
    }

    fn as_f64_slice(values: &[Self]) -> Option<&[f64]> {
        Some(values)
    }
}

macro_rules! lossy_element {
    ($($ty:ty),+) => {
        $(impl Element for $ty {
            #[allow(clippy::cast_lossless, clippy::cast_precision_loss)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        })+
    };
}

lossy_element!(f32, i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

/// Array input: anything convertible to an `f64` array.
///
/// Implemented for slices, `Vec`s, fixed-size arrays and `ndarray` arrays of any
/// [`Element`] type and dimension, so any of them can be passed directly:
/// `idw.fit(&x, &y, &z)`.
pub trait ArrayLike {
    /// Number of dimensions (`ndim`).
    fn ndim(&self) -> usize;

    /// Shape of the array.
    fn shape(&self) -> Vec<usize>;

    /// All values in logical (C) order, converted to `f64`.
    fn to_f64_vec(&self) -> Vec<f64>;

    /// Borrow the values when they already are contiguous `f64`.
    fn as_f64_slice(&self) -> Option<&[f64]> {
        None
    }
}

impl<T: Element> ArrayLike for [T] {
    fn ndim(&self) -> usize {
        1
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        self.iter().map(|value| value.to_f64()).collect()
    }

    fn as_f64_slice(&self) -> Option<&[f64]> {
        T::as_f64_slice(self)
    }
}

impl<T: Element> ArrayLike for Vec<T> {
    fn ndim(&self) -> usize {
        1
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.len()]
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        self.as_slice().to_f64_vec()
    }

    fn as_f64_slice(&self) -> Option<&[f64]> {
        T::as_f64_slice(self)
    }
}

impl<T: Element, const N: usize> ArrayLike for [T; N] {
    fn ndim(&self) -> usize {
        1
    }

    fn shape(&self) -> Vec<usize> {
        vec![N]
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        self.as_slice().to_f64_vec()
    }

    fn as_f64_slice(&self) -> Option<&[f64]> {
        T::as_f64_slice(self)
    }
}

impl<S, D> ArrayLike for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Element,
    D: Dimension,
{
    fn ndim(&self) -> usize {
        ArrayBase::ndim(self)
    }

    fn shape(&self) -> Vec<usize> {
        ArrayBase::shape(self).to_vec()
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        self.iter().map(|value| value.to_f64()).collect()
    }

    fn as_f64_slice(&self) -> Option<&[f64]> {
        self.as_slice().and_then(<S::Elem as Element>::as_f64_slice)
    }
}

impl<A: ArrayLike + ?Sized> ArrayLike for &A {
    fn ndim(&self) -> usize {
        (**self).ndim()
    }

    fn shape(&self) -> Vec<usize> {
        (**self).shape()
    }

    fn to_f64_vec(&self) -> Vec<f64> {
        (**self).to_f64_vec()
    }

    fn as_f64_slice(&self) -> Option<&[f64]> {
        (**self).as_f64_slice()
    }
}

/// Target coordinates accepted by `interpolate` and [`extract_target_xy`].
///
/// Implemented for `(n, 2)` arrays, slices/`Vec`s of `[x, y]` pairs, slices/`Vec`s
/// of GEOS points and a fitted [`Grid`](crate::grid::Grid).
pub trait Targets {
    /// Target x and y coordinates.
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)>;
}

fn invalid_targets_error() -> ApbaseError {
    ApbaseError::validation_error("targets must be a list of Points, [x, y] pairs, or an (n, 2) array")
}

impl<S, D> Targets for ArrayBase<S, D>
where
    S: Data,
    S::Elem: Element,
    D: Dimension,
{
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        let shape = ArrayBase::shape(self);
        if shape.len() != 2 || shape[1] != 2 {
            return Err(invalid_targets_error());
        }
        let values = ArrayLike::to_f64_vec(self);
        let x: Array1<f64> = values.iter().step_by(2).copied().collect();
        let y: Array1<f64> = values.iter().skip(1).step_by(2).copied().collect();
        Ok((x, y))
    }
}

impl<T: Element> Targets for [[T; 2]] {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        Ok((
            self.iter().map(|p| p[0].to_f64()).collect(),
            self.iter().map(|p| p[1].to_f64()).collect(),
        ))
    }
}

impl<T: Element> Targets for Vec<[T; 2]> {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        self.as_slice().target_xy()
    }
}

impl Targets for [Geometry] {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        if self.is_empty() {
            return Err(invalid_targets_error());
        }
        let mut x = Vec::with_capacity(self.len());
        let mut y = Vec::with_capacity(self.len());
        for geometry in self {
            if geometry.geometry_type()? != GeometryTypes::Point {
                return Err(invalid_targets_error());
            }
            if geometry.is_empty()? {
                x.push(f64::NAN);
                y.push(f64::NAN);
            } else {
                x.push(geometry.get_x()?);
                y.push(geometry.get_y()?);
            }
        }
        Ok((Array1::from(x), Array1::from(y)))
    }
}

impl Targets for Vec<Geometry> {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        self.as_slice().target_xy()
    }
}

impl<T: Targets + ?Sized> Targets for &T {
    fn target_xy(&self) -> Result<(Array1<f64>, Array1<f64>)> {
        (**self).target_xy()
    }
}

/// Run-length encoded representation of a regular masked point grid.
#[derive(Clone, Debug, PartialEq)]
pub struct MaskedRegularGridSpec {
    /// Lower-left x of the full regular grid.
    pub min_x: f64,
    /// Lower-left y of the full regular grid.
    pub min_y: f64,
    /// Grid spacing along x.
    pub dx: f64,
    /// Grid spacing along y.
    pub dy: f64,
    /// Number of columns of the full rectangular grid before masking.
    pub full_nx: i64,
    /// Number of rows of the full rectangular grid before masking.
    pub full_ny: i64,
    /// y-index of each run.
    pub row_indices: Array1<i32>,
    /// First x-index of each run.
    pub x_start_indices: Array1<i32>,
    /// Number of consecutive points of each run.
    pub counts: Array1<i32>,
    /// Number of points represented by all runs.
    pub point_count: i64,
}

/// Convert input values to a contiguous one-dimensional `float64` array.
pub fn as_float64_1d(values: &(impl ArrayLike + ?Sized), name: &str) -> Result<Array1<f64>> {
    if values.ndim() != 1 {
        return Err(ApbaseError::validation_error(format!(
            "{name} must be a 1D array"
        )));
    }
    Ok(Array1::from(values.to_f64_vec()))
}

/// Rows where both x and y are finite.
pub fn finite_mask_xy(x: &Array1<f64>, y: &Array1<f64>) -> Array1<bool> {
    x.iter()
        .zip(y)
        .map(|(a, b)| a.is_finite() && b.is_finite())
        .collect()
}

/// Rows where x, y, and z are all finite.
pub fn finite_mask_xyz(x: &Array1<f64>, y: &Array1<f64>, z: &Array1<f64>) -> Array1<bool> {
    x.iter()
        .zip(y)
        .zip(z)
        .map(|((a, b), c)| a.is_finite() && b.is_finite() && c.is_finite())
        .collect()
}

pub(crate) fn select(values: &Array1<f64>, mask: &Array1<bool>) -> Array1<f64> {
    values
        .iter()
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|(value, _)| *value)
        .collect()
}

/// Filter coordinate arrays to finite x/y pairs.
pub fn filter_finite_xy(x: &Array1<f64>, y: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {
    let mask = finite_mask_xy(x, y);
    if mask.iter().all(|keep| *keep) {
        return (x.clone(), y.clone());
    }
    (select(x, &mask), select(y, &mask))
}

/// Filter coordinate/value arrays to finite x/y/z rows.
pub fn filter_finite_xyz(
    x: &Array1<f64>,
    y: &Array1<f64>,
    z: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    let mask = finite_mask_xyz(x, y, z);
    if mask.iter().all(|keep| *keep) {
        return (x.clone(), y.clone(), z.clone());
    }
    (select(x, &mask), select(y, &mask), select(z, &mask))
}

/// Validate that x and y are non-empty arrays with identical length.
pub fn validate_same_size_xy(x: &Array1<f64>, y: &Array1<f64>) -> Result<()> {
    if x.is_empty() {
        return Err(ApbaseError::validation_error(
            "x and y must contain at least one point",
        ));
    }
    if x.len() != y.len() {
        return Err(ApbaseError::validation_error("x and y must have the same size"));
    }
    Ok(())
}

/// Validate that x, y, and z are non-empty arrays with identical length.
pub fn validate_same_size_xyz(x: &Array1<f64>, y: &Array1<f64>, z: &Array1<f64>) -> Result<()> {
    if x.is_empty() {
        return Err(ApbaseError::validation_error(
            "x, y, and z must contain at least one point",
        ));
    }
    if x.len() != y.len() || x.len() != z.len() {
        return Err(ApbaseError::validation_error(
            "x, y, and z must have the same size",
        ));
    }
    Ok(())
}

/// Extract target x/y coordinates from supported target containers.
pub fn extract_target_xy(targets: &(impl Targets + ?Sized)) -> Result<(Array1<f64>, Array1<f64>)> {
    targets.target_xy()
}

/// Detect and encode a regular masked grid from `(n, 2)` point coordinates.
///
/// Returns `None` when the points are empty, non-finite, irregularly spaced,
/// or not arranged as regular grid spans.
pub fn normalize_masked_regular_grid_spec<S, D>(grid: &ArrayBase<S, D>) -> Option<MaskedRegularGridSpec>
where
    S: Data,
    S::Elem: Element,
    D: Dimension,
{
    let shape = grid.shape();
    if shape.len() != 2 || shape[1] != 2 || shape[0] == 0 {
        return None;
    }
    let values = ArrayLike::to_f64_vec(grid);
    let x: Vec<f64> = values.iter().step_by(2).copied().collect();
    let y: Vec<f64> = values.iter().skip(1).step_by(2).copied().collect();
    if !(x.iter().all(|v| v.is_finite()) && y.iter().all(|v| v.is_finite())) {
        return None;
    }

    let unique_x = _numpy::unique(&x);
    let unique_y = _numpy::unique(&y);
    if unique_x.is_empty() || unique_y.is_empty() {
        return None;
    }
    let dx = axis_step(&unique_x)?;
    let dy = axis_step(&unique_y)?;

    let min_x = unique_x[0];
    let min_y = unique_y[0];
    let full_nx = _numpy::rint((unique_x[unique_x.len() - 1] - min_x) / dx) as i64 + 1;
    let full_ny = _numpy::rint((unique_y[unique_y.len() - 1] - min_y) / dy) as i64 + 1;
    if full_nx <= 0 || full_ny <= 0 {
        return None;
    }

    let x_index: Vec<i64> = x.iter().map(|v| _numpy::rint((v - min_x) / dx) as i64).collect();
    let y_index: Vec<i64> = y.iter().map(|v| _numpy::rint((v - min_y) / dy) as i64).collect();
    if x_index.iter().any(|i| *i < 0 || *i >= full_nx) || y_index.iter().any(|i| *i < 0 || *i >= full_ny) {
        return None;
    }

    let atol = dx.abs().max(dy.abs()).max(1.0) * 1.0e-8;
    let expected_x: Vec<f64> = x_index.iter().map(|i| min_x + *i as f64 * dx).collect();
    if !_numpy::allclose(&expected_x, &x, 1.0e-10, atol) {
        return None;
    }
    let expected_y: Vec<f64> = y_index.iter().map(|i| min_y + *i as f64 * dy).collect();
    if !_numpy::allclose(&expected_y, &y, 1.0e-10, atol) {
        return None;
    }

    let point_count = shape[0];
    let mut row_indices = Vec::new();
    let mut x_start_indices = Vec::new();
    let mut run_starts = Vec::new();
    for position in 0..point_count {
        let is_run_start = position == 0
            || y_index[position] != y_index[position - 1]
            || x_index[position] != x_index[position - 1] + 1;
        if is_run_start {
            run_starts.push(position);
            row_indices.push(y_index[position] as i32);
            x_start_indices.push(x_index[position] as i32);
        }
    }
    let counts: Vec<i32> = run_starts
        .iter()
        .enumerate()
        .map(|(k, start)| {
            let end = run_starts.get(k + 1).copied().unwrap_or(point_count);
            (end - start) as i32
        })
        .collect();

    Some(MaskedRegularGridSpec {
        min_x,
        min_y,
        dx,
        dy,
        full_nx,
        full_ny,
        row_indices: Array1::from(row_indices),
        x_start_indices: Array1::from(x_start_indices),
        counts: Array1::from(counts),
        point_count: point_count as i64,
    })
}

fn axis_step(axis: &[f64]) -> Option<f64> {
    if axis.len() == 1 {
        return Some(1.0);
    }
    let positive: Vec<f64> = axis
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .filter(|diff| *diff > 0.0)
        .collect();
    if positive.is_empty() {
        return None;
    }
    let step = _numpy::min(&positive);
    if !step.is_finite() || step <= 0.0 {
        return None;
    }
    let expected: Vec<f64> = axis
        .iter()
        .map(|value| axis[0] + _numpy::rint((value - axis[0]) / step) * step)
        .collect();
    if !_numpy::allclose(&expected, axis, 1.0e-10, step.abs().max(1.0) * 1.0e-8) {
        return None;
    }
    Some(step)
}

/// Float as shown in messages: shortest round-trip digits, `.0` for integral
/// values, scientific notation (`1e+16`, `1.5e-05`) outside `1e-4 <= |x| < 1e16`.
pub(crate) fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("scientific notation always has an exponent");
    let exponent: i32 = exponent.parse().expect("valid exponent");
    if (-4..16).contains(&exponent) {
        let positional = format!("{value}");
        if positional.contains('.') {
            positional
        } else {
            format!("{positional}.0")
        }
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    }
}

/// Integer with `,` thousands separators (`100,000,000`).
pub(crate) fn format_thousands(value: i64) -> String {
    let digits = value.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, ch) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    if value < 0 { format!("-{out}") } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::exceptions::ErrorKind;
    use ndarray::{Array2, array};

    #[test]
    fn as_float64_1d_rejects_other_dimensions() {
        let values: Vec<i32> = vec![1, 2, 3];
        assert_eq!(as_float64_1d(&values, "x").unwrap(), array![1.0, 2.0, 3.0]);
        assert_eq!(as_float64_1d(&[1.5f32, 2.5], "x").unwrap(), array![1.5, 2.5]);
        let matrix = Array2::<f64>::zeros((2, 2));
        let error = as_float64_1d(&matrix, "x").unwrap_err();
        assert_eq!(error.message(), "x must be a 1D array");
        assert_eq!(error.kind(), ErrorKind::ValidationError);
        let strided = array![1.0, 2.0, 3.0, 4.0];
        let view = strided.slice(ndarray::s![..;2]);
        assert_eq!(as_float64_1d(&view, "x").unwrap(), array![1.0, 3.0]);
    }

    #[test]
    fn size_validation_messages() {
        let empty = Array1::<f64>::zeros(0);
        let one = array![1.0];
        let two = array![1.0, 2.0];
        assert_eq!(
            validate_same_size_xyz(&empty, &empty, &empty)
                .unwrap_err()
                .message(),
            "x, y, and z must contain at least one point"
        );
        assert_eq!(
            validate_same_size_xyz(&one, &two, &one).unwrap_err().message(),
            "x, y, and z must have the same size"
        );
        assert_eq!(
            validate_same_size_xy(&empty, &empty).unwrap_err().message(),
            "x and y must contain at least one point"
        );
        assert_eq!(
            validate_same_size_xy(&one, &two).unwrap_err().message(),
            "x and y must have the same size"
        );
    }

    #[test]
    fn filter_finite_keeps_only_finite_rows() {
        let x = array![1.0, f64::NAN, 3.0];
        let y = array![1.0, 2.0, f64::INFINITY];
        let z = array![1.0, 2.0, 3.0];
        let (xf, yf, zf) = filter_finite_xyz(&x, &y, &z);
        assert_eq!((xf, yf, zf), (array![1.0], array![1.0], array![1.0]));
        assert_eq!(
            filter_finite_xy(&array![1.0, 2.0], &array![3.0, 4.0]).0,
            array![1.0, 2.0]
        );
    }

    #[test]
    fn targets_from_arrays_pairs_and_points() {
        let targets = array![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(
            extract_target_xy(&targets).unwrap(),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        let pairs = vec![[1.0, 2.0], [3.0, 4.0]];
        assert_eq!(
            extract_target_xy(&pairs).unwrap(),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        let points = vec![
            Geometry::new_from_wkt("POINT (1 2)").unwrap(),
            Geometry::new_from_wkt("POINT (3 4)").unwrap(),
        ];
        assert_eq!(
            extract_target_xy(&points).unwrap(),
            (array![1.0, 3.0], array![2.0, 4.0])
        );
        let wrong = array![[1.0, 2.0, 3.0]];
        assert!(
            extract_target_xy(&wrong)
                .unwrap_err()
                .is(ErrorKind::ValidationError)
        );
        let lines = vec![Geometry::new_from_wkt("LINESTRING (0 0, 1 1)").unwrap()];
        assert!(extract_target_xy(&lines).is_err());
        let none: Vec<Geometry> = Vec::new();
        assert!(extract_target_xy(&none).is_err());
    }

    #[test]
    fn grid_spec_single_point_and_irregular() {
        let single = array![[5.0, 7.0]];
        let spec = normalize_masked_regular_grid_spec(&single).unwrap();
        assert_eq!((spec.full_nx, spec.full_ny, spec.point_count), (1, 1, 1));
        assert_eq!((spec.dx, spec.dy), (1.0, 1.0));
        let irregular = array![[0.0, 0.0], [1.0, 0.0], [2.5, 0.0]];
        assert!(normalize_masked_regular_grid_spec(&irregular).is_none());
        let regular = array![[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [10.0, 10.0]];
        let spec = normalize_masked_regular_grid_spec(&regular).unwrap();
        assert_eq!(spec.row_indices, array![0, 1]);
        assert_eq!(spec.x_start_indices, array![0, 1]);
        assert_eq!(spec.counts, array![3, 1]);
    }

    #[test]
    fn message_formatting_helpers() {
        assert_eq!(float_repr(50.0), "50.0");
        assert_eq!(float_repr(0.1), "0.1");
        assert_eq!(float_repr(1e16), "1e+16");
        assert_eq!(float_repr(1.5e-5), "1.5e-05");
        assert_eq!(float_repr(0.0001), "0.0001");
        assert_eq!(format_thousands(100_000_000), "100,000,000");
        assert_eq!(format_thousands(999), "999");
        assert_eq!(format_thousands(1000), "1,000");
    }
}
