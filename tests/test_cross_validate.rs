//! Leave-one-out cross-validation and model selection (`apbase::cross_validate`).
//!
//! `test_cross_validate_cokriging_matches_manual_loo` needs the private native
//! wrappers and fitted matrices, so it lives as a unit test in
//! `src/cross_validate/_cross_validation.rs`.

mod common;

use std::time::Instant;

use apbase::cokriging::SecondaryInput;
use apbase::cross_validate::{
    BestModelResult, CrossValidateCokrigingOptions, CrossValidateOptions, CrossValidationResult,
    MethodCrossValidation, Metric, MultiMethodBestModelResult, SelectBestModelOptions, cross_validate,
    cross_validate_cokriging, select_best_model,
};
use apbase::variogram::ModelParams;
use common::{NumpyRng, approx, assert_allclose, solve};
use indexmap::IndexMap;
use ndarray::{Array1, Array2, array};

const MODEL_SPHERICAL: f64 = 1.0;

fn variogram_value(h: f64, nugget: f64, partial_sill: f64, model_range: f64) -> f64 {
    let ratio = (h / model_range).clamp(0.0, 1.0);
    let shape = if h / model_range >= 1.0 {
        1.0
    } else {
        1.5 * ratio - 0.5 * ratio.powi(3)
    };
    nugget + partial_sill * shape
}

#[allow(clippy::too_many_arguments)]
fn reference_kriging_point(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    tx: f64,
    ty: f64,
    nugget: f64,
    psill: f64,
    range: f64,
) -> f64 {
    let sill = nugget + psill;
    let n = x.len();
    let cov = |h: f64| {
        if h <= 0.0 {
            sill
        } else {
            sill - variogram_value(h, nugget, psill, range)
        }
    };
    let mut system = Array2::ones((n + 1, n + 1));
    for i in 0..n {
        for j in 0..n {
            system[[i, j]] = cov(((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt());
        }
    }
    system[[n, n]] = 0.0;
    let mut rhs = Array1::ones(n + 1);
    for k in 0..n {
        rhs[k] = cov(((x[k] - tx).powi(2) + (y[k] - ty).powi(2)).sqrt());
    }
    let weights = solve(&system, &rhs);
    (0..n).map(|k| weights[k] * z[k]).sum()
}

#[allow(clippy::too_many_arguments)]
fn reference_loocv(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    radius: f64,
    power: f64,
    nugget: f64,
    psill: f64,
    range: f64,
    min_neighbors: usize,
) -> (Array1<f64>, Array1<f64>) {
    let n = x.len();
    let mut idw_out = Array1::from_elem(n, f64::NAN);
    let mut krig_out = Array1::from_elem(n, f64::NAN);
    for i in 0..n {
        let others: Vec<usize> = (0..n).filter(|k| *k != i).collect();
        let d2: Vec<f64> = others
            .iter()
            .map(|k| (x[*k] - x[i]).powi(2) + (y[*k] - y[i]).powi(2))
            .collect();
        let within: Vec<usize> = (0..others.len()).filter(|k| d2[*k] <= radius * radius).collect();
        if within.len() < min_neighbors {
            continue;
        }
        if let Some(exact) = within.iter().find(|k| d2[**k] <= 1.0e-24) {
            idw_out[i] = z[others[*exact]];
            krig_out[i] = z[others[*exact]];
            continue;
        }
        let weights: Vec<f64> = within.iter().map(|k| d2[*k].powf(-0.5 * power)).collect();
        let numerator: f64 = weights.iter().zip(&within).map(|(w, k)| w * z[others[*k]]).sum();
        idw_out[i] = numerator / weights.iter().sum::<f64>();
        let (xs, ys, zs): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            within.iter().map(|k| x[others[*k]]).collect(),
            within.iter().map(|k| y[others[*k]]).collect(),
            within.iter().map(|k| z[others[*k]]).collect(),
        );
        krig_out[i] = reference_kriging_point(&xs, &ys, &zs, x[i], y[i], nugget, psill, range);
    }
    (idw_out, krig_out)
}

fn model_values(nugget: f64, partial_sill: f64, model_range: f64) -> Array1<f64> {
    array![MODEL_SPHERICAL, nugget, partial_sill, model_range, 0.0]
}

#[test]
fn test_cross_validate_matches_reference_loocv() {
    let mut rng = NumpyRng::new(0);
    let n = 30;
    let x = rng.uniform(0.0, 100.0, n);
    let y = rng.uniform(0.0, 100.0, n);
    let noise = rng.normal(0.0, 0.5, n);
    let z: Vec<f64> = (0..n).map(|i| 5.0 + 0.1 * x[i] + 0.2 * y[i] + noise[i]).collect();
    let (nugget, psill, range, radius) = (0.5, 4.0, 200.0, 200.0);

    let result = cross_validate(
        &x,
        &y,
        &z,
        CrossValidateOptions {
            radius: Some(radius),
            power: 2.0,
            max_neighbors: n as i64,
            min_neighbors: 1,
            model_values: Some(model_values(nugget, psill, range)),
            ..Default::default()
        },
    )
    .unwrap();
    let (expected_idw, expected_krig) = reference_loocv(&x, &y, &z, radius, 2.0, nugget, psill, range, 1);

    assert_eq!(result.n_samples, n as i64);
    assert_allclose(&result.idw.predicted, &expected_idw, 1e-6, 1e-6, "idw");
    assert_allclose(&result.kriging.predicted, &expected_krig, 1e-6, 1e-6, "kriging");

    let z = Array1::from(z);
    let mean = |v: &Array1<f64>| v.sum() / v.len() as f64;
    for (method, expected) in [(&result.idw, &expected_idw), (&result.kriging, &expected_krig)] {
        let err = expected - &z;
        assert!(approx(method.mae, mean(&err.mapv(f64::abs)), 1e-6, 0.0));
        assert!(approx(method.rmse, mean(&err.mapv(|e| e * e)).sqrt(), 1e-6, 0.0));
        let ss_tot = z.mapv(|v| (v - mean(&z)).powi(2)).sum();
        let expected_r2 = 1.0 - err.mapv(|e| e * e).sum() / ss_tot;
        assert!(approx(method.r2, expected_r2, 1e-6, 0.0));
    }
}

/// Samples 3 and 4 have one other source within the radius, below
/// `min_neighbors = 2`: both fall back to their nearest `max_neighbors` others
/// (here all four), the rest keep their radius neighborhood.
#[test]
fn test_cross_validate_uses_nearest_sources_below_min_neighbors() {
    let x = [0.0, 1.0, 2.0, 1000.0, 1001.0];
    let y = [0.0; 5];
    let z = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = cross_validate(
        &x,
        &y,
        &z,
        CrossValidateOptions {
            radius: Some(5.0),
            max_neighbors: 4,
            min_neighbors: 2,
            model_values: Some(model_values(0.5, 4.0, 30.0)),
            ..Default::default()
        },
    )
    .unwrap();
    for index in [3, 4] {
        let others: Vec<usize> = (0..5).filter(|k| *k != index).collect();
        let (xs, ys, zs): (Vec<f64>, Vec<f64>, Vec<f64>) = (
            others.iter().map(|k| x[*k]).collect(),
            others.iter().map(|k| y[*k]).collect(),
            others.iter().map(|k| z[*k]).collect(),
        );
        let weights: Vec<f64> = xs.iter().map(|xk| 1.0 / (xk - x[index]).powi(2)).collect();
        let idw: f64 =
            weights.iter().zip(&zs).map(|(w, zk)| w * zk).sum::<f64>() / weights.iter().sum::<f64>();
        let krig = reference_kriging_point(&xs, &ys, &zs, x[index], y[index], 0.5, 4.0, 30.0);
        assert!(approx(result.idw.predicted[index], idw, 1e-10, 0.0));
        assert!(approx(result.kriging.predicted[index], krig, 1e-8, 0.0));
    }
    assert_eq!(result.idw.n_valid, 5);
    assert_eq!(result.kriging.n_valid, 5);
}

#[test]
fn test_cross_validate_subsamples_to_max_points() {
    let mut rng = NumpyRng::new(1);
    let n = 800;
    let x = rng.uniform(0.0, 1000.0, n);
    let y = rng.uniform(0.0, 1000.0, n);
    let z = rng.normal(0.0, 1.0, n);
    let result = cross_validate(
        &x,
        &y,
        &z,
        CrossValidateOptions {
            max_points: 200,
            seed: Some(0),
            max_neighbors: 20,
            min_neighbors: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.n_samples, 200);
    assert_eq!(result.idw.predicted.len(), 200);
    assert_eq!(result.kriging.predicted.len(), 200);
    assert_eq!(result.idw.actual.len(), 200);
}

#[test]
fn test_cross_validate_uses_all_points_below_max_points() {
    let mut rng = NumpyRng::new(2);
    let n = 50;
    let x = rng.uniform(0.0, 100.0, n);
    let y = rng.uniform(0.0, 100.0, n);
    let z = rng.normal(0.0, 1.0, n);
    let result = cross_validate(
        &x,
        &y,
        &z,
        CrossValidateOptions {
            max_points: 500,
            max_neighbors: 20,
            min_neighbors: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.n_samples, n as i64);
    assert_eq!(result.idw.actual, Array1::from(z));
}

#[test]
fn test_cross_validate_is_fast_for_500_points() {
    let mut rng = NumpyRng::new(3);
    let n = 500;
    let x = rng.uniform(0.0, 5000.0, n);
    let y = rng.uniform(0.0, 5000.0, n);
    let z = rng.normal(50.0, 10.0, n);
    let options = CrossValidateOptions {
        max_points: 500,
        max_neighbors: 20,
        min_neighbors: 3,
        ..Default::default()
    };
    cross_validate(&x, &y, &z, options.clone()).unwrap();
    let start = Instant::now();
    cross_validate(&x, &y, &z, options).unwrap();
    let elapsed = start.elapsed().as_secs_f64();
    assert!(elapsed < 0.1, "cross_validate took {elapsed:.4}s for 500 points");
}

#[test]
fn test_cross_validate_rejects_non_positive_radius() {
    let error = cross_validate(
        &[0.0, 10.0, 0.0, 10.0],
        &[0.0, 0.0, 10.0, 10.0],
        &[5.0, 7.0, 8.0, 10.0],
        CrossValidateOptions {
            radius: Some(0.0),
            min_neighbors: 1,
            model_values: Some(model_values(1.0, 5.0, 30.0)),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(error.message().contains("radius"));
    assert!(error.is(apbase::common::ErrorKind::ValueError));
}

fn empty_model_params() -> ModelParams {
    ModelParams {
        model_id: 1,
        model_name: "spherical",
        nugget: 0.0,
        partial_sill: 0.0,
        sill: 0.0,
        range: 0.0,
        sse: None,
    }
}

fn method(predicted: Array1<f64>, actual: &Array1<f64>) -> MethodCrossValidation {
    let valid: Vec<usize> = (0..predicted.len())
        .filter(|i| predicted[*i].is_finite())
        .collect();
    let (mae, rmse) = if valid.is_empty() {
        (f64::NAN, f64::NAN)
    } else {
        let err: Vec<f64> = valid.iter().map(|i| predicted[*i] - actual[*i]).collect();
        let n = err.len() as f64;
        (
            err.iter().map(|e| e.abs()).sum::<f64>() / n,
            (err.iter().map(|e| e * e).sum::<f64>() / n).sqrt(),
        )
    };
    MethodCrossValidation {
        mae,
        rmse,
        r2: 0.0,
        n_valid: valid.len() as i64,
        predicted,
        actual: actual.clone(),
    }
}

fn make_result(idw_pred: Array1<f64>, krig_pred: Array1<f64>, actual: &Array1<f64>) -> CrossValidationResult {
    CrossValidationResult {
        idw: method(idw_pred, actual),
        kriging: method(krig_pred, actual),
        n_samples: actual.len() as i64,
        radius: 10.0,
        model_params: empty_model_params(),
    }
}

fn seeded(seed: u64) -> SelectBestModelOptions {
    SelectBestModelOptions {
        seed: Some(seed),
        ..Default::default()
    }
}

#[test]
fn test_select_best_model_picks_lower_rmse_method_and_flags_significance() {
    let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idw_pred = &actual + &array![2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0];
    let krig_pred = &actual + &array![0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1];
    let best: BestModelResult =
        select_best_model(&make_result(idw_pred, krig_pred, &actual), seeded(0)).unwrap();
    assert_eq!(best.method, "kriging");
    assert!(best.kriging_score < best.idw_score);
    assert!(best.significant);
    assert!(best.ci_low > 0.0);
}

#[test]
fn test_select_best_model_exact_tie_prefers_idw_and_is_not_significant() {
    let actual = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let pred = &actual + &array![0.5, -0.3, 0.2, -0.1, 0.4];
    let best = select_best_model(&make_result(pred.clone(), pred, &actual), seeded(1)).unwrap();
    assert_eq!(best.method, "idw");
    assert_eq!(best.margin, 0.0);
    assert!(!best.significant);
    assert_eq!(best.ci_low, 0.0);
    assert_eq!(best.ci_high, 0.0);
}

#[test]
fn test_select_best_model_trivial_winner_when_one_method_is_all_nan() {
    let actual = array![1.0, 2.0, 3.0, 4.0];
    let idw_pred = &actual + &array![0.5, -0.5, 0.2, -0.2];
    let best = select_best_model(
        &make_result(idw_pred, Array1::from_elem(4, f64::NAN), &actual),
        SelectBestModelOptions::default(),
    )
    .unwrap();
    assert_eq!(best.method, "idw");
    assert_eq!(best.n_common_valid, 0);
    assert!(best.significant);
    assert!(best.kriging_score.is_nan());
    assert!(best.ci_low.is_nan() && best.ci_high.is_nan());
}

#[test]
fn test_select_best_model_raises_when_both_methods_are_all_nan() {
    let actual = array![1.0, 2.0, 3.0];
    let nan = Array1::from_elem(3, f64::NAN);
    let error = select_best_model(
        &make_result(nan.clone(), nan, &actual),
        SelectBestModelOptions::default(),
    )
    .unwrap_err();
    assert!(error.message().contains("cannot select a best model"));
    assert!(error.is(apbase::common::ErrorKind::ValueError));
}

/// The metric `"r2"` is rejected when parsed, as a `ValueError`.
#[test]
fn test_select_best_model_rejects_invalid_parameters() {
    let actual = array![1.0, 2.0, 3.0, 4.0];
    let pred = &actual + 0.1;
    let result = make_result(pred.clone(), pred, &actual);
    assert_eq!(
        "r2".parse::<Metric>().unwrap_err().kind(),
        apbase::common::ErrorKind::ValueError
    );
    for options in [
        SelectBestModelOptions {
            confidence_level: 0.0,
            ..Default::default()
        },
        SelectBestModelOptions {
            confidence_level: 1.0,
            ..Default::default()
        },
        SelectBestModelOptions {
            n_bootstrap: 0,
            ..Default::default()
        },
    ] {
        let error = select_best_model(&result, options).unwrap_err();
        assert_eq!(error.kind(), apbase::common::ErrorKind::ValueError);
    }
}

#[test]
fn test_select_best_model_is_reproducible_with_seed() {
    let mut rng = NumpyRng::new(7);
    let actual = Array1::from(rng.normal(0.0, 1.0, 40));
    let idw_pred = &actual + &Array1::from(rng.normal(0.0, 0.3, 40));
    let krig_pred = &actual + &Array1::from(rng.normal(0.0, 0.5, 40));
    let result = make_result(idw_pred, krig_pred, &actual);
    let first = select_best_model(&result, seeded(42)).unwrap();
    let second = select_best_model(&result, seeded(42)).unwrap();
    assert_eq!(first.ci_low.to_bits(), second.ci_low.to_bits());
    assert_eq!(first.ci_high.to_bits(), second.ci_high.to_bits());
}

#[test]
fn test_select_best_model_rejects_mismatched_actual_arrays() {
    let actual_a = array![1.0, 2.0, 3.0];
    let actual_b = array![1.0, 2.0, 99.0];
    let result = CrossValidationResult {
        idw: MethodCrossValidation {
            mae: 0.0,
            rmse: 0.0,
            r2: 0.0,
            n_valid: 3,
            predicted: &actual_a + 0.1,
            actual: actual_a.clone(),
        },
        kriging: MethodCrossValidation {
            mae: 0.0,
            rmse: 0.0,
            r2: 0.0,
            n_valid: 3,
            predicted: &actual_b + 0.1,
            actual: actual_b,
        },
        n_samples: 3,
        radius: 10.0,
        model_params: empty_model_params(),
    };
    let error = select_best_model(&result, SelectBestModelOptions::default()).unwrap_err();
    assert!(error.message().contains("actual"));
}

fn smooth_field(x: &[f64], y: &[f64], seed: u64) -> Vec<f64> {
    let mut rng = NumpyRng::new(seed);
    let centers = rng.uniform(0.0, 100.0, 16);
    let amps = rng.normal(0.0, 1.0, 8);
    let mut field = vec![0.0; x.len()];
    for (k, amp) in amps.iter().enumerate() {
        for (i, value) in field.iter_mut().enumerate() {
            let d2 = (x[i] - centers[2 * k]).powi(2) + (y[i] - centers[2 * k + 1]).powi(2);
            *value += amp * (-d2 / (2.0 * 25.0f64.powi(2))).exp();
        }
    }
    field
}

fn cokriging_loo_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = NumpyRng::new(6);
    let n = 60;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 1006);
    let z0: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.15, n))
        .map(|(l, e)| l + e)
        .collect();
    let zu: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.25, n))
        .map(|(l, e)| 1.4 * l + e)
        .collect();
    (x0, y0, z0, zu)
}

#[test]
fn test_cross_validate_cokriging_metrics_match_predictions() {
    let (x0, y0, z0, zu) = cokriging_loo_fixture();
    let secondaries = SecondaryInput::from([(
        "secondary",
        (
            Array1::from(x0.clone()),
            Array1::from(y0.clone()),
            Array1::from(zu),
        ),
    )]);
    let result = cross_validate_cokriging(
        &x0,
        &y0,
        &z0,
        &secondaries,
        CrossValidateCokrigingOptions {
            max_neighbors: 12,
            min_neighbors: 1,
            ..Default::default()
        },
    )
    .unwrap();
    for (name, mcv) in &result.by_method {
        let valid: Vec<usize> = (0..mcv.predicted.len())
            .filter(|i| mcv.predicted[*i].is_finite())
            .collect();
        assert_eq!(mcv.n_valid, valid.len() as i64, "{name}");
        let err: Vec<f64> = valid.iter().map(|i| mcv.predicted[*i] - mcv.actual[*i]).collect();
        let n = err.len() as f64;
        assert!(
            approx(mcv.mae, err.iter().map(|e| e.abs()).sum::<f64>() / n, 1e-6, 0.0),
            "{name}"
        );
        assert!(
            approx(
                mcv.rmse,
                (err.iter().map(|e| e * e).sum::<f64>() / n).sqrt(),
                1e-6,
                0.0
            ),
            "{name}"
        );
        assert_eq!(mcv.actual, Array1::from(z0.clone()), "{name}");
    }
}

/// With a 5-unit radius most samples have fewer than `min_neighbors = 3` other
/// primaries in reach; every method still predicts all of them.
#[test]
fn test_cross_validate_cokriging_uses_nearest_sources_below_min_neighbors() {
    let (x0, y0, z0, zu) = cokriging_loo_fixture();
    let secondaries = SecondaryInput::from([(
        "secondary",
        (
            Array1::from(x0.clone()),
            Array1::from(y0.clone()),
            Array1::from(zu),
        ),
    )]);
    let result = cross_validate_cokriging(
        &x0,
        &y0,
        &z0,
        &secondaries,
        CrossValidateCokrigingOptions {
            radius: Some(5.0),
            max_neighbors: 12,
            min_neighbors: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.by_method.len(), 3);
    for (name, mcv) in &result.by_method {
        assert_eq!(mcv.n_valid, x0.len() as i64, "{name}");
        assert!(mcv.predicted.iter().all(|v| v.is_finite()), "{name}");
    }
}

fn cv_method(offsets: [f64; 8], actual: &Array1<f64>) -> MethodCrossValidation {
    MethodCrossValidation {
        mae: 0.0,
        rmse: 0.0,
        r2: 0.0,
        n_valid: 8,
        predicted: actual + &Array1::from(offsets.to_vec()),
        actual: actual.clone(),
    }
}

#[test]
fn test_select_best_model_generalized_picks_correct_winner() {
    let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let methods = IndexMap::from([
        (
            "collocated".to_string(),
            cv_method([2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0], &actual),
        ),
        (
            "icm".to_string(),
            cv_method([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1], &actual),
        ),
        (
            "lmc".to_string(),
            cv_method([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &actual),
        ),
    ]);
    let best: MultiMethodBestModelResult = select_best_model(&methods, seeded(0)).unwrap();
    assert_eq!(best.method, "icm");
    assert_eq!(best.runner_up, "lmc");
    assert!(best.scores["icm"] < best.scores["lmc"] && best.scores["lmc"] < best.scores["collocated"]);
    assert!(best.significant);
    assert_eq!(best.n_common_valid, 8);
}

#[test]
fn test_select_best_model_generalized_requires_at_least_two_methods() {
    let actual = array![1.0, 2.0, 3.0];
    let methods = IndexMap::from([(
        "only".to_string(),
        MethodCrossValidation {
            mae: 0.0,
            rmse: 0.0,
            r2: 0.0,
            n_valid: 3,
            predicted: actual.clone(),
            actual,
        },
    )]);
    let error = select_best_model(&methods, SelectBestModelOptions::default()).unwrap_err();
    assert!(error.message().contains("at least two methods"));
}

/// The `CrossValidationResult` call keeps returning `BestModelResult` (the
/// return type is fixed at compile time by the input type).
#[test]
fn test_select_best_model_legacy_two_method_call_unchanged() {
    let actual = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let idw_pred = &actual + &array![2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0];
    let krig_pred = &actual + &array![0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1];
    let best: BestModelResult =
        select_best_model(&make_result(idw_pred, krig_pred, &actual), seeded(0)).unwrap();
    assert_eq!(best.method, "kriging");
    assert!(best.kriging_score < best.idw_score);
}

#[test]
fn test_cross_validate_cokriging_skips_collocated_for_heterotopic_secondaries() {
    let mut rng = NumpyRng::new(0);
    let (x0, y0) = (rng.uniform(0.0, 1000.0, 200), rng.uniform(0.0, 1000.0, 200));
    let noise0 = rng.normal(0.0, 0.5, 200);
    let z0: Vec<f64> = (0..200)
        .map(|i| (x0[i] / 150.0).sin() * (y0[i] / 150.0).cos() * 10.0 + noise0[i])
        .collect();
    let (xu, yu) = (rng.uniform(0.0, 1000.0, 600), rng.uniform(0.0, 1000.0, 600));
    let noiseu = rng.normal(0.0, 1.0, 600);
    let zu: Vec<f64> = (0..600)
        .map(|i| -(xu[i] / 150.0).sin() * (yu[i] / 150.0).cos() * 10.0 + noiseu[i])
        .collect();
    let secondaries = SecondaryInput::from([("ec", (Array1::from(xu), Array1::from(yu), Array1::from(zu)))]);

    let result = cross_validate_cokriging(
        &x0,
        &y0,
        &z0,
        &secondaries,
        CrossValidateCokrigingOptions {
            seed: Some(0),
            ..Default::default()
        },
    )
    .unwrap();

    let methods: Vec<&str> = result.by_method.keys().map(String::as_str).collect();
    assert_eq!(methods, ["icm", "lmc"]);
    let params: Vec<&str> = result.model_params.keys().map(String::as_str).collect();
    assert_eq!(params, ["icm", "lmc"]);
    assert!(result.by_method.values().all(|method| method.rmse.is_finite()));
    let best = select_best_model(
        &result.by_method,
        SelectBestModelOptions {
            seed: Some(0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(["icm", "lmc"].contains(&best.method.as_str()));
}

fn mismatched_structure_field(x: &[f64], y: &[f64], seed: u64, n_centers: usize, scale: f64) -> Vec<f64> {
    let mut rng = NumpyRng::new(seed);
    let centers = rng.uniform(0.0, 100.0, n_centers * 2);
    let amps = rng.normal(0.0, 1.0, n_centers);
    let mut field = vec![0.0; x.len()];
    for (k, amp) in amps.iter().enumerate() {
        for (i, value) in field.iter_mut().enumerate() {
            let d2 = (x[i] - centers[2 * k]).powi(2) + (y[i] - centers[2 * k + 1]).powi(2);
            *value += amp * (-d2 / (2.0 * scale.powi(2))).exp();
        }
    }
    field
}

#[test]
fn test_cross_validate_cokriging_skips_a_method_whose_fit_is_rejected() {
    let mut rng = NumpyRng::new(0);
    let (x, y) = (rng.uniform(0.0, 100.0, 250), rng.uniform(0.0, 100.0, 250));
    let f_short = mismatched_structure_field(&x, &y, 1, 60, 3.0);
    let f_med = mismatched_structure_field(&x, &y, 2, 10, 15.0);
    let f_long = mismatched_structure_field(&x, &y, 3, 3, 70.0);
    let column = |values: Vec<f64>| {
        (
            Array1::from(x.clone()),
            Array1::from(y.clone()),
            Array1::from(values),
        )
    };
    let secondaries = SecondaryInput::from([
        (
            "short_medium",
            column((0..250).map(|i| f_short[i] + 3.0 * f_med[i]).collect()),
        ),
        (
            "short_long",
            column((0..250).map(|i| f_short[i] + 3.0 * f_long[i]).collect()),
        ),
        (
            "medium_long",
            column((0..250).map(|i| f_med[i] - f_long[i]).collect()),
        ),
    ]);

    let result = cross_validate_cokriging(
        &x,
        &y,
        &f_short,
        &secondaries,
        CrossValidateCokrigingOptions {
            seed: Some(0),
            ..Default::default()
        },
    )
    .unwrap();

    let methods: Vec<&str> = result.by_method.keys().map(String::as_str).collect();
    assert_eq!(methods, ["collocated", "lmc"]);
    let params: Vec<&str> = result.model_params.keys().map(String::as_str).collect();
    assert_eq!(params, ["collocated", "lmc"]);
    assert!(result.by_method.values().all(|method| method.rmse.is_finite()));
}
