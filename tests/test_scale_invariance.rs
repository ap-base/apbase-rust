//! Fits must not depend on the unit of the values (`apbase::variogram`,
//! `apbase::cross_variogram`, `apbase::cokriging`).

mod common;

use apbase::cokriging::{CoKriging, CoKrigingMethod, CoKrigingModelParams, CoKrigingOptions, SecondaryInput};
use apbase::cross_validate::{
    CrossValidateCokrigingOptions, CrossValidateOptions, cross_validate, cross_validate_cokriging,
};
use apbase::cross_variogram::{CrossVariogram, CrossVariogramOptions};
use apbase::kriging::{Kriging, KrigingOptions};
use apbase::variogram::{Variogram, VariogramOptions};
use apbase::{MapOptions, create_map};
use common::NumpyRng;
use indexmap::IndexMap;
use ndarray::{Array1, Array2};

const SCALES: [f64; 5] = [1e-9, 1e-3, 1.0, 1e4, 1e9];
const SOLVER_SCALES: [f64; 7] = [1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9];

struct Data {
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    u: Vec<f64>,
}

fn data() -> Data {
    let mut rng = NumpyRng::new(0);
    let (x, y) = (rng.uniform(0.0, 1000.0, 300), rng.uniform(0.0, 1000.0, 300));
    let noise_z = rng.normal(0.0, 1.0, 300);
    let noise_u = rng.normal(0.0, 3.0, 300);
    let field: Vec<f64> = (0..300)
        .map(|i| (x[i] / 150.0).sin() * (y[i] / 150.0).cos() * 10.0)
        .collect();
    let z = (0..300).map(|i| field[i] + noise_z[i]).collect();
    let u = (0..300).map(|i| 2.0 * field[i] + noise_u[i]).collect();
    Data { x, y, z, u }
}

fn scaled(values: &[f64], c: f64) -> Vec<f64> {
    values.iter().map(|v| c * v).collect()
}

fn assert_same_model(models: &[Vec<f64>]) {
    let reference: Vec<f64> = models[0].iter().map(|v| v / (SCALES[0] * SCALES[0])).collect();
    let largest = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    for (model, c) in models.iter().zip(SCALES).skip(1) {
        for (value, expected) in model.iter().zip(&reference) {
            let normalized = value / (c * c);
            assert!(
                (normalized - expected).abs() <= 1e-6 * expected.abs() + 1e-9 * largest,
                "scale {c}: {normalized} != {expected}"
            );
        }
    }
}

#[test]
fn test_variogram_fits_do_not_depend_on_the_unit_of_the_values() {
    let d = data();
    let mut auto = Vec::new();
    let mut cross = Vec::new();
    for c in SCALES {
        let (z, u) = (scaled(&d.z, c), scaled(&d.u, c));
        auto.push(
            Variogram::from_data(&d.x, &d.y, &z, VariogramOptions::default())
                .unwrap()
                .model_values()
                .unwrap()
                .to_vec(),
        );
        cross.push(
            CrossVariogram::from_data(&d.x, &d.y, &z, &d.x, &d.y, &u, CrossVariogramOptions::default())
                .unwrap()
                .model_values()
                .unwrap()
                .to_vec(),
        );
    }
    for models in [&auto, &cross] {
        assert!(models.iter().all(|m| m[0] == models[0][0]));
        assert!(
            models
                .iter()
                .all(|m| (m[3] - models[0][3]).abs() <= 1e-9 * models[0][3])
        );
        let sills: Vec<Vec<f64>> = models.iter().map(|m| m[1..3].to_vec()).collect();
        assert_same_model(&sills);
    }
}

#[test]
fn test_coregionalization_fits_do_not_depend_on_the_unit_of_the_values() {
    let d = data();
    for method in [CoKrigingMethod::Icm, CoKrigingMethod::Lmc] {
        let mut ranges = Vec::new();
        let mut matrices = Vec::new();
        for c in SCALES {
            let secondaries = SecondaryInput::from([(
                "u",
                (
                    Array1::from(d.x.clone()),
                    Array1::from(d.y.clone()),
                    Array1::from(scaled(&d.u, c)),
                ),
            )]);
            let model = CoKriging::from_data(
                &d.x,
                &d.y,
                &scaled(&d.z, c),
                &secondaries,
                CoKrigingOptions {
                    method,
                    ..Default::default()
                },
            )
            .unwrap();
            match model.model_params().unwrap() {
                CoKrigingModelParams::Icm {
                    range,
                    nugget_matrix,
                    sill_matrix,
                    ..
                } => {
                    ranges.push(vec![range]);
                    matrices.push(nugget_matrix.iter().chain(sill_matrix.iter()).copied().collect());
                }
                CoKrigingModelParams::Lmc {
                    structure_ranges,
                    coefficient_matrices,
                    ..
                } => {
                    ranges.push(structure_ranges.to_vec());
                    matrices.push(coefficient_matrices.iter().copied().collect());
                }
                other => panic!("unexpected params {other:?}"),
            }
        }
        for r in &ranges {
            for (value, expected) in r.iter().zip(&ranges[0]) {
                assert!(
                    (value - expected).abs() <= 1e-9 * expected.abs(),
                    "{method:?} range"
                );
            }
        }
        assert_same_model(&matrices);
    }
}

fn targets() -> Array2<f64> {
    let mut values = Vec::with_capacity(60);
    for i in 0..30 {
        let t = i as f64 / 29.0;
        values.push(50.0 + 900.0 * t);
        values.push(80.0 + 840.0 * t);
    }
    Array2::from_shape_vec((30, 2), values).unwrap()
}

fn predict(solver: &str, c: f64) -> IndexMap<String, Array1<f64>> {
    let d = data();
    let z = scaled(&d.z, c);
    let secondaries = SecondaryInput::from([(
        "u",
        (
            Array1::from(d.x.clone()),
            Array1::from(d.y.clone()),
            Array1::from(scaled(&d.u, c)),
        ),
    )]);
    let kriging_options = KrigingOptions {
        max_neighbors: 40,
        min_neighbors: 3,
        ..Default::default()
    };
    let one = |values: Array1<f64>| IndexMap::from([(solver.to_string(), values)]);
    match solver {
        "kriging" => one(Kriging::from_data(&d.x, &d.y, &z, kriging_options)
            .unwrap()
            .interpolate(&targets())
            .unwrap()),
        "cross_validate" => one(cross_validate(
            &d.x,
            &d.y,
            &z,
            CrossValidateOptions {
                max_points: 100,
                seed: Some(0),
                ..Default::default()
            },
        )
        .unwrap()
        .kriging
        .predicted),
        "cross_validate_cokriging" => cross_validate_cokriging(
            &d.x,
            &d.y,
            &z,
            &secondaries,
            CrossValidateCokrigingOptions {
                max_points: 100,
                seed: Some(0),
                ..Default::default()
            },
        )
        .unwrap()
        .by_method
        .into_iter()
        .map(|(method, values)| (method, values.predicted))
        .collect(),
        method => one(CoKriging::from_data(
            &d.x,
            &d.y,
            &z,
            &secondaries,
            CoKrigingOptions {
                method: method.parse().unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
        .interpolate(&targets())
        .unwrap()),
    }
}

#[test]
fn test_solvers_do_not_depend_on_the_unit_of_the_values() {
    for solver in [
        "kriging",
        "cross_validate",
        "collocated",
        "icm",
        "lmc",
        "cross_validate_cokriging",
    ] {
        let reference = predict(solver, 1.0);
        assert!(
            reference
                .values()
                .all(|values| values.iter().all(|v| v.is_finite())),
            "{solver}"
        );
        for c in SOLVER_SCALES {
            let scaled = predict(solver, c);
            assert_eq!(
                scaled.keys().collect::<Vec<_>>(),
                reference.keys().collect::<Vec<_>>(),
                "{solver} at {c}"
            );
            for (method, values) in &scaled {
                for (value, expected) in values.iter().zip(&reference[method]) {
                    assert!(value.is_finite(), "{method} at {c}");
                    let normalized = value / c;
                    assert!(
                        (normalized - expected).abs() <= 1e-6 * expected.abs() + 1e-9,
                        "{method} at {c}: {normalized} != {expected}"
                    );
                }
            }
        }
    }
}

#[test]
fn test_create_map_does_not_depend_on_the_unit_of_the_values() {
    let d = data();
    let options = || MapOptions {
        seed: Some(0),
        ..MapOptions::new(50.0)
    };
    let reference = create_map(&d.x, &d.y, &d.z, options()).unwrap();
    for c in SOLVER_SCALES {
        let result = create_map(&d.x, &d.y, &scaled(&d.z, c), options()).unwrap();
        assert_eq!(result.method, reference.method, "scale {c}");
        for (value, expected) in result.z.iter().zip(&reference.z) {
            let normalized = value / c;
            assert!(
                (normalized - expected).abs() <= 1e-6 * expected.abs() + 1e-9,
                "scale {c}: {normalized} != {expected}"
            );
        }
    }
}
