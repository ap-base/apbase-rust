//! Inverse distance weighting (`apbase::idw`).

mod common;

use apbase::idw::{IDW, IDWOptions};
use common::assert_allclose;
use ndarray::{Array1, Array2, array};

fn reference_idw(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    grid: &Array2<f64>,
    radius: f64,
    power: f64,
) -> Array1<f64> {
    let mut out = Array1::from_elem(grid.nrows(), f64::NAN);
    for (i, target) in grid.rows().into_iter().enumerate() {
        let d2: Vec<f64> = x
            .iter()
            .zip(y)
            .map(|(px, py)| (px - target[0]).powi(2) + (py - target[1]).powi(2))
            .collect();
        if let Some(exact) = d2.iter().position(|d| *d <= 1.0e-24) {
            out[i] = z[exact];
            continue;
        }
        let in_range: Vec<usize> = (0..d2.len()).filter(|k| d2[*k] <= radius * radius).collect();
        if in_range.is_empty() {
            continue;
        }
        let weights: Vec<f64> = in_range.iter().map(|k| d2[*k].powf(-0.5 * power)).collect();
        let numerator: f64 = weights.iter().zip(&in_range).map(|(w, k)| w * z[*k]).sum();
        out[i] = numerator / weights.iter().sum::<f64>();
    }
    out
}

#[test]
fn test_idw_matches_reference() {
    let x = [500000.0, 500010.0, 500000.0, 500010.0];
    let y = [7000000.0, 7000000.0, 7000010.0, 7000010.0];
    let z = [0.0, 10.0, 20.0, 30.0];
    let grid = array![
        [500005.0, 7000005.0],
        [500000.0, 7000000.0],
        [500020.0, 7000020.0]
    ];

    let result = IDW::new(IDWOptions {
        radius: Some(30.0),
        power: 2.0,
        min_neighbors: 1,
        max_neighbors: 4,
    })
    .unwrap()
    .fit(&x, &y, &z)
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_idw(&x, &y, &z, &grid, 30.0, 2.0);
    assert_allclose(&result, &expected, 1e-5, 1e-8, "idw vs reference");
}

#[test]
fn test_idw_matches_reference_with_sparse_cell_hash_fallback() {
    let centers = [
        [0.0, 0.0],
        [50000.0, 0.0],
        [0.0, 50000.0],
        [50000.0, 50000.0],
        [25000.0, 25000.0],
    ];
    let offsets = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for (cluster_index, center) in centers.iter().enumerate() {
        for (offset_index, offset) in offsets.iter().enumerate() {
            x.push(center[0] + offset[0]);
            y.push(center[1] + offset[1]);
            z.push((cluster_index * 10 + offset_index) as f64);
        }
    }
    let grid = Array2::from_shape_fn((5, 2), |(i, j)| centers[i][j] + 1.0);
    let result = IDW::new(IDWOptions {
        radius: Some(10.0),
        power: 2.0,
        min_neighbors: 1,
        max_neighbors: 40,
    })
    .unwrap()
    .fit(&x, &y, &z)
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_idw(&x, &y, &z, &grid, 10.0, 2.0);
    assert!(expected.iter().all(|v| v.is_finite()));
    assert_allclose(&result, &expected, 1e-6, 1e-6, "sparse hash fallback");
}

#[test]
fn idw_validation_messages() {
    let message = |options: IDWOptions| IDW::new(options).unwrap_err().to_string();
    assert_eq!(
        message(IDWOptions {
            radius: Some(-1.0),
            ..Default::default()
        }),
        "radius must be finite and greater than zero"
    );
    assert_eq!(
        message(IDWOptions {
            power: 0.0,
            ..Default::default()
        }),
        "power must be finite and greater than zero"
    );
    assert_eq!(
        message(IDWOptions {
            max_neighbors: 0,
            ..Default::default()
        }),
        "max_neighbors must be greater than zero"
    );
    assert_eq!(
        message(IDWOptions {
            min_neighbors: 50,
            ..Default::default()
        }),
        "min_neighbors must be between 1 and max_neighbors"
    );

    let unfitted = IDW::new(IDWOptions::default()).unwrap();
    assert_eq!(
        unfitted.interpolate(&array![[0.0, 0.0]]).unwrap_err().message(),
        "call fit(x, y, z) before interpolate(targets)"
    );

    let mut idw = IDW::new(IDWOptions {
        radius: Some(5.0),
        min_neighbors: 1,
        ..Default::default()
    })
    .unwrap();
    idw.fit(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]).unwrap();
    assert_eq!(
        idw.interpolate(&Array2::<f64>::zeros((0, 2)))
            .unwrap_err()
            .message(),
        "targets must contain at least one target point"
    );
    assert_eq!(
        idw.model_params().unwrap_err().message(),
        "IDW was fit with an explicit radius; no variogram model was fitted"
    );
    let too_few = IDW::new(IDWOptions {
        min_neighbors: 3,
        ..Default::default()
    })
    .unwrap()
    .fit(&[0.0], &[0.0], &[1.0])
    .err();
    assert_eq!(
        too_few.unwrap().message(),
        "IDW requires at least min_neighbors finite points"
    );
}

#[test]
fn free_function_and_call_alias_match_class() {
    let x = [0.0, 10.0, 0.0, 10.0];
    let y = [0.0, 0.0, 10.0, 10.0];
    let z = [5.0, 7.0, 8.0, 10.0];
    let targets = vec![[5.0, 5.0], [10.0, 10.0]];
    let options = IDWOptions {
        radius: Some(20.0),
        min_neighbors: 1,
        ..Default::default()
    };
    let class = IDW::from_data(&x, &y, &z, options.clone()).unwrap();
    let function = apbase::idw::idw(&x, &y, &z, &targets, options).unwrap();
    assert_eq!(class.interpolate(&targets).unwrap(), function);
    assert_eq!(class.call(&targets).unwrap(), class.predict(&targets).unwrap());
    assert_eq!(function[1], 10.0, "exact hit returns the source value");
}
