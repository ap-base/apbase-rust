//! Cokriging against published real-world benchmarks (Jura, Meuse, Walker
//! Lake). RMSE checks are regression freezes at `rel=0.05`.

mod common;

use apbase::cokriging::{CoKrigingMethod, CoKrigingOptions, SecondaryInput, co_kriging};
use apbase::common::ErrorKind;
use apbase::cross_variogram::{ScreeningOptions, VariableXYZ, screen_secondary_variables};
use apbase::kriging::{KrigingOptions, ordinary_kriging};
use common::{NumpyRng, approx, assert_allclose, read_csv};
use indexmap::IndexMap;
use ndarray::{Array1, Array2};

fn rmse(pred: &Array1<f64>, truth: &[f64]) -> f64 {
    let squared: Vec<f64> = pred
        .iter()
        .zip(truth)
        .map(|(p, t)| (p - t).powi(2))
        .filter(|v| !v.is_nan())
        .collect();
    (squared.iter().sum::<f64>() / squared.len() as f64).sqrt()
}

const JURA_RADIUS: f64 = 1.5;

struct Jura {
    x0: Vec<f64>,
    y0: Vec<f64>,
    z0: Vec<f64>,
    columns: std::collections::HashMap<String, Vec<f64>>,
    targets: Array2<f64>,
    truth: Vec<f64>,
}

fn jura() -> Jura {
    let pred = read_csv("jura_prediction.csv");
    let val = read_csv("jura_validation.csv");
    let targets = Array2::from_shape_fn((val["Xloc"].len(), 2), |(i, j)| {
        if j == 0 { val["Xloc"][i] } else { val["Yloc"][i] }
    });
    Jura {
        x0: pred["Xloc"].clone(),
        y0: pred["Yloc"].clone(),
        z0: pred["Cd"].clone(),
        truth: val["Cd"].clone(),
        columns: pred,
        targets,
    }
}

fn xyz(x: &[f64], y: &[f64], z: &[f64]) -> VariableXYZ {
    (
        Array1::from(x.to_vec()),
        Array1::from(y.to_vec()),
        Array1::from(z.to_vec()),
    )
}

fn options(method: CoKrigingMethod, radius: f64) -> CoKrigingOptions {
    CoKrigingOptions {
        method,
        radius: Some(radius),
        max_neighbors: 40,
        min_neighbors: 3,
        ..Default::default()
    }
}

fn kriging_options(radius: f64) -> KrigingOptions {
    KrigingOptions {
        radius: Some(radius),
        max_neighbors: 40,
        min_neighbors: 3,
        ..Default::default()
    }
}

#[test]
fn test_jura_lmc_cokriging_beats_ordinary_kriging_for_cadmium() {
    let data = jura();
    let pb = &data.columns["Pb"];
    let ok_pred = ordinary_kriging(
        &data.x0,
        &data.y0,
        &data.z0,
        &data.targets,
        kriging_options(JURA_RADIUS),
    )
    .unwrap();
    let secondaries = SecondaryInput::from([("Pb", xyz(&data.x0, &data.y0, pb))]);
    let ck_pred = co_kriging(
        &data.x0,
        &data.y0,
        &data.z0,
        &secondaries,
        &data.targets,
        options(CoKrigingMethod::Lmc, JURA_RADIUS),
    )
    .unwrap();
    assert!(!ok_pred.iter().any(|v| v.is_nan()));
    assert!(!ck_pred.iter().any(|v| v.is_nan()));
    let (ok_rmse, ck_rmse) = (rmse(&ok_pred, &data.truth), rmse(&ck_pred, &data.truth));
    assert!(ck_rmse < ok_rmse);
    assert!(
        approx(ok_rmse, 0.73658451360863, 0.05, 1e-12),
        "OK RMSE {ok_rmse}"
    );
    assert!(
        approx(ck_rmse, 0.718762064724412, 0.05, 1e-12),
        "LMC RMSE {ck_rmse}"
    );
}

#[test]
fn test_jura_collocated_and_icm_do_not_beat_ordinary_kriging() {
    let data = jura();
    let secondaries = SecondaryInput::from([
        ("Ni", xyz(&data.x0, &data.y0, &data.columns["Ni"])),
        ("Zn", xyz(&data.x0, &data.y0, &data.columns["Zn"])),
    ]);
    let collocated = co_kriging(
        &data.x0,
        &data.y0,
        &data.z0,
        &secondaries,
        &data.targets,
        options(CoKrigingMethod::Collocated, JURA_RADIUS),
    )
    .unwrap();
    let icm = co_kriging(
        &data.x0,
        &data.y0,
        &data.z0,
        &secondaries,
        &data.targets,
        options(CoKrigingMethod::Icm, JURA_RADIUS),
    )
    .unwrap();
    assert!(!collocated.iter().any(|v| v.is_nan()));
    assert!(!icm.iter().any(|v| v.is_nan()));
    assert!(approx(
        rmse(&collocated, &data.truth),
        0.9773576533793037,
        0.05,
        1e-12
    ));
    assert!(approx(rmse(&icm, &data.truth), 0.7435857058009598, 0.05, 1e-12));
}

#[test]
fn test_meuse_screening_and_cokriging_pipeline_reproduces_observed_zinc() {
    let meuse = read_csv("meuse.csv");
    let (x, y, zinc) = (&meuse["x"], &meuse["y"], &meuse["zinc"]);
    let candidates: IndexMap<String, VariableXYZ> = IndexMap::from([
        ("elev".to_string(), xyz(x, y, &meuse["elev"])),
        ("dist".to_string(), xyz(x, y, &meuse["dist"])),
    ]);
    let screening = screen_secondary_variables(x, y, zinc, &candidates, ScreeningOptions::default()).unwrap();
    let mut selected: Vec<&String> = screening.selected.keys().collect();
    selected.sort();
    assert_eq!(selected, vec!["dist", "elev"]);

    let targets: Vec<[f64; 2]> = x.iter().zip(y).map(|(a, b)| [*a, *b]).collect();
    let pred = co_kriging(
        x,
        y,
        zinc,
        &SecondaryInput::from(screening.selected.clone()),
        &targets,
        CoKrigingOptions {
            radius: Some(1000.0),
            max_neighbors: 40,
            min_neighbors: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!pred.iter().any(|v| v.is_nan()));
    assert_allclose(
        &pred,
        &Array1::from(zinc.clone()),
        1e-6,
        1e-6,
        "exact interpolator",
    );
}

const WALKER_RADIUS: f64 = 100.0;
const WALKER_N_TARGETS: i64 = 1500;

fn walker_sample() -> (Vec<f64>, Vec<f64>, Vec<f64>, VariableXYZ) {
    let walker = read_csv("walker_sample.csv");
    let (x0, y0, v0) = (walker["x"].clone(), walker["y"].clone(), walker["v"].clone());
    let keep: Vec<usize> = (0..walker["u"].len())
        .filter(|i| !walker["u"][*i].is_nan())
        .collect();
    let pick = |column: &str| -> Vec<f64> { keep.iter().map(|i| walker[column][*i]).collect() };
    let secondary = xyz(&pick("x"), &pick("y"), &pick("u"));
    (x0, y0, v0, secondary)
}

fn walker_exhaustive_targets() -> (Array2<f64>, Vec<f64>) {
    let exhaustive = read_csv("walker_exhaustive.csv");
    let idx = NumpyRng::new(0).choice(exhaustive["x"].len() as i64, WALKER_N_TARGETS);
    let targets = Array2::from_shape_fn((idx.len(), 2), |(i, j)| {
        let row = idx[i] as usize;
        if j == 0 {
            exhaustive["x"][row]
        } else {
            exhaustive["y"][row]
        }
    });
    let truth = idx.iter().map(|i| exhaustive["v"][*i as usize]).collect();
    (targets, truth)
}

#[test]
fn test_walker_lake_cokriging_regression_against_exhaustive_grid() {
    let (x0, y0, v0, secondary) = walker_sample();
    let (targets, truth) = walker_exhaustive_targets();
    let secondaries = SecondaryInput::from([("u", secondary)]);

    let ok_pred = ordinary_kriging(&x0, &y0, &v0, &targets, kriging_options(WALKER_RADIUS)).unwrap();
    assert!(!ok_pred.iter().any(|v| v.is_nan()));
    assert!(approx(rmse(&ok_pred, &truth), 146.24500236850068, 0.05, 1e-12));

    for (method, expected) in [
        (CoKrigingMethod::Collocated, 168.03058021141942),
        (CoKrigingMethod::Icm, 151.11274616648427),
        (CoKrigingMethod::Lmc, 163.93028459231954),
    ] {
        let pred = co_kriging(
            &x0,
            &y0,
            &v0,
            &secondaries,
            &targets,
            options(method, WALKER_RADIUS),
        )
        .unwrap();
        assert!(!pred.iter().any(|v| v.is_nan()), "{method}");
        let value = rmse(&pred, &truth);
        assert!(
            approx(value, expected, 0.05, 1e-12),
            "{method}: {value} vs {expected}"
        );
    }
}

#[test]
fn test_walker_lake_icm_rejects_ill_conditioned_sparse_subsample() {
    let (x0_full, y0_full, v0_full, secondary) = walker_sample();
    let (targets, _) = walker_exhaustive_targets();
    let sparse_idx = NumpyRng::new(0).choice(x0_full.len() as i64, 120);
    let pick = |values: &[f64]| -> Vec<f64> { sparse_idx.iter().map(|i| values[*i as usize]).collect() };
    let (x0, y0, v0) = (pick(&x0_full), pick(&y0_full), pick(&v0_full));

    let error = co_kriging(
        &x0,
        &y0,
        &v0,
        &SecondaryInput::from([("u", secondary)]),
        &targets,
        options(CoKrigingMethod::Icm, WALKER_RADIUS),
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::CoKrigingNonPSDCoregionalizationError);
}
