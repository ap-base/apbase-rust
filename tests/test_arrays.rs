mod common;

use apbase::common::ErrorKind;
use apbase::common::arrays::{extract_target_xy, normalize_masked_regular_grid_spec};
use common::NumpyRng;
use geos::{Geom, Geometry};
use ndarray::{Array1, Array2, array};

fn point(x: f64, y: f64) -> Geometry {
    Geometry::new_from_wkt(&format!("POINT ({x} {y})")).unwrap()
}

#[test]
fn test_extract_target_xy_from_ndarray() {
    let targets = array![[1.0, 2.0], [3.0, 4.0]];
    let (x, y) = extract_target_xy(&targets).unwrap();
    assert_eq!(x, array![1.0, 3.0]);
    assert_eq!(y, array![2.0, 4.0]);
}

#[test]
fn test_extract_target_xy_from_list_of_points() {
    let targets = vec![point(1.0, 2.0), point(3.0, 4.0), point(5.0, 6.0)];
    let (x, y): (Array1<f64>, Array1<f64>) = extract_target_xy(&targets).unwrap();
    assert_eq!(x, array![1.0, 3.0, 5.0]);
    assert_eq!(y, array![2.0, 4.0, 6.0]);
}

#[test]
fn test_extract_target_xy_from_wkb_points() {
    let wkb: Vec<Vec<u8>> = [point(1.0, 2.0), point(3.0, 4.0)]
        .iter()
        .map(|p| p.to_wkb().unwrap().to_vec())
        .collect();
    let targets: Vec<Geometry> = wkb
        .iter()
        .map(|bytes| Geometry::new_from_wkb(bytes).unwrap())
        .collect();
    let (x, y) = extract_target_xy(&targets).unwrap();
    assert_eq!(x, array![1.0, 3.0]);
    assert_eq!(y, array![2.0, 4.0]);
}

#[test]
fn test_extract_target_xy_rejects_non_point_geometry() {
    let targets = vec![
        point(1.0, 2.0),
        Geometry::new_from_wkt("LINESTRING (0 0, 1 1)").unwrap(),
    ];
    let error = extract_target_xy(&targets).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValidationError);
    assert_eq!(
        error.message(),
        "targets must be a list of Points, [x, y] pairs, or an (n, 2) array"
    );
}

#[test]
fn test_extract_target_xy_rejects_non_geometry_objects() {
    for error in [
        extract_target_xy(&Array2::<f64>::zeros((3, 3))).unwrap_err(),
        extract_target_xy(&array![1.0, 2.0, 3.0]).unwrap_err(),
        extract_target_xy(&Vec::<Geometry>::new()).unwrap_err(),
    ] {
        assert_eq!(error.kind(), ErrorKind::ValidationError);
        assert!(error.is(ErrorKind::ValueError));
    }
}

fn reference_rle(x_index: &[i64], y_index: &[i64]) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut rows, mut starts, mut counts) = (Vec::new(), Vec::new(), Vec::new());
    let mut pos = 0;
    while pos < x_index.len() {
        let row = y_index[pos];
        let start_x = x_index[pos];
        let mut count = 1;
        pos += 1;
        while pos < x_index.len() && y_index[pos] == row && x_index[pos] == start_x + count {
            count += 1;
            pos += 1;
        }
        rows.push(row as i32);
        starts.push(start_x as i32);
        counts.push(count as i32);
    }
    (rows, starts, counts)
}

#[test]
fn test_normalize_masked_regular_grid_spec_matches_reference_rle_fuzz() {
    let mut rng = NumpyRng::new(3);
    for _ in 0..200 {
        let ny = rng.integers(1, 6, 1)[0] as usize;
        let nx = rng.integers(1, 10, 1)[0] as usize;
        let dx = rng.uniform(0.5, 5.0, 1)[0];
        let dy = rng.uniform(0.5, 5.0, 1)[0];
        let (min_x, min_y) = (10.0, -20.0);

        let draws = rng.random(ny * nx);
        let mut keep = Array2::from_shape_fn((ny, nx), |(r, c)| draws[r * nx + c] > 0.35);
        let empty_cols: Vec<usize> = (0..nx).filter(|c| !keep.column(*c).iter().any(|k| *k)).collect();
        let rows_for_cols = rng.integers(0, ny as i64, empty_cols.len());
        for (col, row) in empty_cols.iter().zip(rows_for_cols) {
            keep[[row as usize, *col]] = true;
        }
        let empty_rows: Vec<usize> = (0..ny).filter(|r| !keep.row(*r).iter().any(|k| *k)).collect();
        let cols_for_rows = rng.integers(0, nx as i64, empty_rows.len());
        for (row, col) in empty_rows.iter().zip(cols_for_rows) {
            keep[[*row, col as usize]] = true;
        }

        let (mut rows, mut cols) = (Vec::new(), Vec::new());
        for ((r, c), kept) in keep.indexed_iter() {
            if *kept {
                rows.push(r as i64);
                cols.push(c as i64);
            }
        }
        let points = Array2::from_shape_fn((rows.len(), 2), |(i, j)| {
            if j == 0 {
                min_x + cols[i] as f64 * dx
            } else {
                min_y + rows[i] as f64 * dy
            }
        });

        let spec = normalize_masked_regular_grid_spec(&points).expect("regular grid");
        let (expected_rows, expected_starts, expected_counts) = reference_rle(&cols, &rows);
        assert_eq!(spec.row_indices.to_vec(), expected_rows);
        assert_eq!(spec.x_start_indices.to_vec(), expected_starts);
        assert_eq!(spec.counts.to_vec(), expected_counts);
        assert_eq!(spec.point_count, points.nrows() as i64);
        assert_eq!(
            spec.counts.iter().map(|c| i64::from(*c)).sum::<i64>(),
            points.nrows() as i64
        );
    }
}

#[test]
fn test_normalize_masked_regular_grid_spec_single_point() {
    let spec = normalize_masked_regular_grid_spec(&array![[5.0, 7.0]]).unwrap();
    assert_eq!(spec.point_count, 1);
    assert_eq!(spec.row_indices, array![0]);
    assert_eq!(spec.x_start_indices, array![0]);
    assert_eq!(spec.counts, array![1]);
}

#[test]
fn array_like_accepts_every_numeric_container() {
    use apbase::common::arrays::{as_float64_1d, filter_finite_xyz, validate_same_size_xyz};
    let from_i32 = as_float64_1d(&vec![1i32, 2, 3], "x").unwrap();
    let from_f32 = as_float64_1d(&[1.5f32, 2.5], "x").unwrap();
    let from_u64 = as_float64_1d(&[7u64], "x").unwrap();
    let base = array![1.0, 2.0, 3.0, 4.0];
    let from_view = as_float64_1d(&base.slice(ndarray::s![1..;2]), "x").unwrap();
    assert_eq!(from_i32, array![1.0, 2.0, 3.0]);
    assert_eq!(from_f32, array![1.5, 2.5]);
    assert_eq!(from_u64, array![7.0]);
    assert_eq!(from_view, array![2.0, 4.0]);
    assert_eq!(
        as_float64_1d(&Array2::<f64>::zeros((1, 1)), "z")
            .unwrap_err()
            .message(),
        "z must be a 1D array"
    );

    let (x, y, z) = (
        array![1.0, f64::NAN, 3.0],
        array![1.0, 2.0, 3.0],
        array![f64::INFINITY, 2.0, 3.0],
    );
    validate_same_size_xyz(&x, &y, &z).unwrap();
    let (xf, yf, zf) = filter_finite_xyz(&x, &y, &z);
    assert_eq!((xf, yf, zf), (array![3.0], array![3.0], array![3.0]));
}

#[test]
fn empty_point_geometry_becomes_nan() {
    let targets = vec![Geometry::new_from_wkt("POINT EMPTY").unwrap(), point(1.0, 1.0)];
    let (x, y) = extract_target_xy(&targets).unwrap();
    assert!(x[0].is_nan() && y[0].is_nan());
    assert_eq!((x[1], y[1]), (1.0, 1.0));
}
