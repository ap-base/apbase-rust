mod common;

use apbase::kriging::{Kriging, KrigingOptions, ordinary_kriging};
use common::{assert_allclose, solve};
use ndarray::{Array1, Array2, array};

const MODEL_SPHERICAL: i64 = 1;
const MODEL_EXPONENTIAL: i64 = 2;
const MODEL_GAUSSIAN: i64 = 3;

fn variogram_shape(h: f64, model_id: i64, model_range: f64) -> f64 {
    let ratio = h / model_range;
    match model_id {
        MODEL_SPHERICAL => {
            if ratio >= 1.0 {
                1.0
            } else {
                let clipped = ratio.clamp(0.0, 1.0);
                1.5 * clipped - 0.5 * clipped.powi(3)
            }
        }
        MODEL_EXPONENTIAL => 1.0 - (-3.0 * ratio).exp(),
        MODEL_GAUSSIAN => 1.0 - (-3.0 * ratio * ratio).exp(),
        _ => panic!("unsupported model_id: {model_id}"),
    }
}

fn covariance(h: f64, model_id: i64, nugget: f64, partial_sill: f64, model_range: f64) -> f64 {
    let sill = nugget + partial_sill;
    if h <= 0.0 {
        sill
    } else {
        sill - (nugget + partial_sill * variogram_shape(h, model_id, model_range))
    }
}

#[allow(clippy::too_many_arguments)]
fn reference_ordinary_kriging(
    x: &[f64],
    y: &[f64],
    z: &[f64],
    grid: &Array2<f64>,
    nugget: f64,
    partial_sill: f64,
    model_range: f64,
    model_id: i64,
) -> Array1<f64> {
    let n = x.len();
    let mut system = Array2::ones((n + 1, n + 1));
    for i in 0..n {
        for j in 0..n {
            let h = ((x[i] - x[j]).powi(2) + (y[i] - y[j]).powi(2)).sqrt();
            system[[i, j]] = covariance(h, model_id, nugget, partial_sill, model_range);
        }
    }
    system[[n, n]] = 0.0;
    grid.rows()
        .into_iter()
        .map(|target| {
            let mut rhs = Array1::ones(n + 1);
            for k in 0..n {
                let h0 = ((x[k] - target[0]).powi(2) + (y[k] - target[1]).powi(2)).sqrt();
                rhs[k] = covariance(h0, model_id, nugget, partial_sill, model_range);
            }
            let weights = solve(&system, &rhs);
            (0..n).map(|k| weights[k] * z[k]).sum()
        })
        .collect()
}

fn model_values(nugget: f64, partial_sill: f64, model_range: f64, model_id: i64) -> Array1<f64> {
    array![model_id as f64, nugget, partial_sill, model_range, 0.0]
}

fn kriging(model: Array1<f64>, radius: f64, max_neighbors: i64, min_neighbors: i64) -> Kriging {
    Kriging::new(KrigingOptions {
        model_values: Some(model),
        radius: Some(radius),
        max_neighbors,
        min_neighbors,
    })
    .unwrap()
}

#[test]
fn test_kriging_matches_reference_ordinary_kriging() {
    let x = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let grid = array![[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]];
    let (nugget, partial_sill, model_range) = (0.5, 4.0, 30.0);

    let result = kriging(
        model_values(nugget, partial_sill, model_range, MODEL_SPHERICAL),
        model_range,
        10,
        1,
    )
    .fit(&x, &y, &z)
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_ordinary_kriging(
        &x,
        &y,
        &z,
        &grid,
        nugget,
        partial_sill,
        model_range,
        MODEL_SPHERICAL,
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "spherical");
}

#[test]
fn test_kriging_matches_reference_for_exponential_and_gaussian_models() {
    let x = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let grid = array![[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]];
    let (nugget, partial_sill, model_range) = (0.5, 4.0, 30.0);
    for model_id in [MODEL_EXPONENTIAL, MODEL_GAUSSIAN] {
        let result = kriging(
            model_values(nugget, partial_sill, model_range, model_id),
            model_range,
            10,
            1,
        )
        .fit(&x, &y, &z)
        .unwrap()
        .interpolate(&grid)
        .unwrap();
        let expected =
            reference_ordinary_kriging(&x, &y, &z, &grid, nugget, partial_sill, model_range, model_id);
        assert_allclose(&result, &expected, 1e-6, 1e-6, &format!("model {model_id}"));
    }
}

#[test]
fn test_kriging_matches_reference_with_sparse_cell_hash_fallback() {
    let centers = [
        [0.0, 0.0],
        [50000.0, 0.0],
        [0.0, 50000.0],
        [50000.0, 50000.0],
        [25000.0, 25000.0],
    ];
    let offsets = [
        [0.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [2.0, 2.0],
        [1.0, 1.0],
        [1.0, 0.0],
    ];
    let (nugget, partial_sill, model_range) = (0.5, 4.0, 8.0);
    let (mut x, mut y, mut z, mut expected, mut targets) = (vec![], vec![], vec![], vec![], vec![]);
    for (cluster_index, center) in centers.iter().enumerate() {
        let cluster_x: Vec<f64> = offsets.iter().map(|o| center[0] + o[0]).collect();
        let cluster_y: Vec<f64> = offsets.iter().map(|o| center[1] + o[1]).collect();
        let cluster_z: Vec<f64> = (0..offsets.len())
            .map(|k| (cluster_index * 3 + k) as f64)
            .collect();
        let target = array![[center[0] + 1.0, center[1] + 1.0]];
        expected.extend(reference_ordinary_kriging(
            &cluster_x,
            &cluster_y,
            &cluster_z,
            &target,
            nugget,
            partial_sill,
            model_range,
            MODEL_SPHERICAL,
        ));
        x.extend(cluster_x);
        y.extend(cluster_y);
        z.extend(cluster_z);
        targets.push([center[0] + 1.0, center[1] + 1.0]);
    }
    let expected = Array1::from(expected);
    let result = kriging(
        model_values(nugget, partial_sill, model_range, MODEL_SPHERICAL),
        model_range,
        10,
        1,
    )
    .fit(&x, &y, &z)
    .unwrap()
    .interpolate(&targets)
    .unwrap();
    assert!(expected.iter().all(|v| v.is_finite()));
    assert_allclose(&result, &expected, 1e-6, 1e-6, "sparse hash fallback");
}

#[test]
fn test_kriging_exact_match_returns_source_value() {
    let x = [0.0, 10.0, 0.0, 10.0];
    let y = [0.0, 0.0, 10.0, 10.0];
    let z = [5.0, 7.0, 8.0, 10.0];
    let grid = array![[10.0, 0.0], [3.0, 3.0]];
    let result = kriging(model_values(1.0, 5.0, 30.0, MODEL_SPHERICAL), 30.0, 4, 1)
        .fit(&x, &y, &z)
        .unwrap()
        .interpolate(&grid)
        .unwrap();
    assert_eq!(result[0], 7.0);
}

#[test]
fn test_kriging_uses_nearest_sources_below_min_neighbors() {
    let result = kriging(model_values(1.0, 5.0, 30.0, MODEL_SPHERICAL), 5.0, 4, 2)
        .fit(&[0.0, 100.0], &[0.0, 100.0], &[1.0, 2.0])
        .unwrap()
        .interpolate(&array![[1.0, 1.0]])
        .unwrap();
    assert!(result[0] > 1.0 && result[0] < 1.5, "{}", result[0]);
}

#[test]
fn test_kriging_ignores_non_finite_source_rows() {
    let x = [0.0, 10.0, f64::NAN, 10.0];
    let y = [0.0, 0.0, 10.0, 10.0];
    let z = [5.0, 7.0, 8.0, f64::INFINITY];
    let mut two = kriging(model_values(1.0, 5.0, 30.0, MODEL_SPHERICAL), 30.0, 4, 2);
    two.fit(&x, &y, &z).unwrap();
    let mut three = kriging(model_values(1.0, 5.0, 30.0, MODEL_SPHERICAL), 30.0, 4, 3);
    assert_eq!(
        three.fit(&x, &y, &z).unwrap_err().message(),
        "kriging requires at least min_neighbors finite points"
    );
    let estimate = two.interpolate(&array![[5.0, 0.0]]).unwrap()[0];
    assert!(
        (estimate - 6.0).abs() < 1e-9,
        "midpoint of the two finite rows: {estimate}"
    );
}

#[test]
fn test_kriging_rejects_non_positive_radius() {
    let error = Kriging::new(KrigingOptions {
        model_values: Some(model_values(1.0, 5.0, 30.0, MODEL_SPHERICAL)),
        radius: Some(0.0),
        min_neighbors: 1,
        ..Default::default()
    })
    .unwrap_err();
    assert!(error.message().contains("radius"));
    assert_eq!(error.kind(), apbase::common::ErrorKind::ValueError);
}

#[test]
fn test_ordinary_kriging_free_function_matches_class() {
    let x = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let grid = array![[5.0, 5.0], [2.0, 8.0]];
    let options = KrigingOptions {
        model_values: Some(model_values(0.5, 4.0, 30.0, MODEL_SPHERICAL)),
        radius: Some(30.0),
        max_neighbors: 10,
        min_neighbors: 1,
    };
    let from_function = ordinary_kriging(&x, &y, &z, &grid, options.clone()).unwrap();
    let from_class = Kriging::from_data(&x, &y, &z, options)
        .unwrap()
        .interpolate(&grid)
        .unwrap();
    assert_eq!(from_function, from_class);
}

#[test]
fn kriging_validation_messages() {
    let _guard = common::config_guard();
    let message = |options: KrigingOptions| Kriging::new(options).unwrap_err().to_string();
    assert_eq!(
        message(KrigingOptions {
            model_values: Some(array![1.0, 2.0]),
            ..Default::default()
        }),
        "model_values must contain 5 values"
    );
    assert_eq!(
        message(KrigingOptions {
            max_neighbors: 501,
            ..Default::default()
        }),
        "max_neighbors must be <= 500"
    );
    apbase::config.set("max_neighbors_ceiling", 30).unwrap();
    let capped = message(KrigingOptions::default());
    apbase::config._clear();
    assert_eq!(capped, "max_neighbors must be <= 30");

    let unfitted = Kriging::new(KrigingOptions::default()).unwrap();
    assert_eq!(
        unfitted.model_values().unwrap_err().message(),
        "call fit(x, y, z) before interpolate(targets)"
    );
}

#[test]
fn empty_targets_report_the_native_status() {
    let fitted = Kriging::from_data(
        &[0.0, 10.0, 0.0],
        &[0.0, 0.0, 10.0],
        &[1.0, 2.0, 3.0],
        KrigingOptions {
            model_values: Some(model_values(0.0, 1.0, 30.0, 1)),
            radius: Some(20.0),
            min_neighbors: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let error = fitted.interpolate(&Array2::<f64>::zeros((0, 2))).unwrap_err();
    assert_eq!(error.kind(), apbase::common::ErrorKind::KrigingInvalidDataError);
    assert!(error.is(apbase::common::ErrorKind::InterpolationError));
    assert_eq!(error.message(), "invalid source or target data");
}
