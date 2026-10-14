//! Concurrent top-level calls into the native kernels (which reuse `SAVE`d workspaces under OMP critical sections)
//! must return exactly the sequential results.

mod common;

use apbase::cross_validate::{CrossValidateOptions, CrossValidationResult, cross_validate};
use apbase::idw::{IDW, IDWOptions};
use common::NumpyRng;
use ndarray::{Array1, Array2};

const ROUNDS: usize = 5;
const CALLS_PER_ROUND: usize = 8;
const MAX_WORKERS: usize = 4;

fn same(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| common::same_f64(*x, *y))
}

/// Run `CALLS_PER_ROUND` calls on `MAX_WORKERS` threads; call `i` runs task `i % 2`.
fn run_round<T: Send>(task: &(dyn Fn(usize) -> T + Sync)) -> Vec<T> {
    let mut results: Vec<Option<T>> = (0..CALLS_PER_ROUND).map(|_| None).collect();
    std::thread::scope(|scope| {
        let chunks: Vec<&mut [Option<T>]> = results.chunks_mut(CALLS_PER_ROUND / MAX_WORKERS).collect();
        for (worker, chunk) in chunks.into_iter().enumerate() {
            scope.spawn(move || {
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    *slot = Some(task(worker * (CALLS_PER_ROUND / MAX_WORKERS) + offset));
                }
            });
        }
    });
    results
        .into_iter()
        .map(|slot| slot.expect("every call ran"))
        .collect()
}

#[test]
fn test_idw_interpolate_is_safe_under_concurrent_top_level_calls() {
    let make_model = |seed: u64, z_offset: f64| -> (IDW, Array2<f64>) {
        let mut rng = NumpyRng::new(seed);
        let x = rng.uniform(0.0, 1000.0, 300);
        let y = rng.uniform(0.0, 1000.0, 300);
        let z: Vec<f64> = rng.uniform(0.0, 50.0, 300).iter().map(|v| v + z_offset).collect();
        let grid = Array2::from_shape_vec((200, 2), rng.uniform(0.0, 1000.0, 400)).unwrap();
        let model = IDW::from_data(
            &x,
            &y,
            &z,
            IDWOptions {
                radius: Some(80.0),
                power: 2.0,
                min_neighbors: 1,
                max_neighbors: 16,
            },
        )
        .unwrap();
        (model, grid)
    };
    let (model_a, grid_a) = make_model(1, 0.0);
    let (model_b, grid_b) = make_model(2, 1000.0);
    let expected_a = model_a.interpolate(&grid_a).unwrap();
    let expected_b = model_b.interpolate(&grid_b).unwrap();
    assert!(expected_a.iter().any(|v| v.is_finite()) && expected_b.iter().any(|v| v.is_finite()));

    for _ in 0..ROUNDS {
        let results = run_round(&|i| {
            if i % 2 == 0 {
                model_a.interpolate(&grid_a)
            } else {
                model_b.interpolate(&grid_b)
            }
        });
        for (i, result) in results.into_iter().enumerate() {
            let expected = if i % 2 == 0 { &expected_a } else { &expected_b };
            assert!(
                same(&result.unwrap(), expected),
                "concurrent idw_local call diverged from the sequential reference -- likely a data race"
            );
        }
    }
}

#[test]
fn test_cross_validate_is_safe_under_concurrent_top_level_calls() {
    let make_dataset = |seed: u64, z_offset: f64| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut rng = NumpyRng::new(seed);
        let x = rng.uniform(0.0, 1000.0, 200);
        let y = rng.uniform(0.0, 1000.0, 200);
        let z = rng.uniform(0.0, 50.0, 200).iter().map(|v| v + z_offset).collect();
        (x, y, z)
    };
    let (xa, ya, za) = make_dataset(1, 0.0);
    let (xb, yb, zb) = make_dataset(2, 1000.0);
    let options = CrossValidateOptions {
        radius: Some(80.0),
        power: 2.0,
        max_neighbors: 16,
        min_neighbors: 1,
        seed: Some(0),
        ..Default::default()
    };
    let run_a = || cross_validate(&xa, &ya, &za, options.clone()).unwrap();
    let run_b = || cross_validate(&xb, &yb, &zb, options.clone()).unwrap();
    let expected_a = run_a();
    let expected_b = run_b();

    for _ in 0..ROUNDS {
        let results: Vec<CrossValidationResult> = run_round(&|i| if i % 2 == 0 { run_a() } else { run_b() });
        for (i, result) in results.iter().enumerate() {
            let expected = if i % 2 == 0 { &expected_a } else { &expected_b };
            assert!(common::same_f64(result.idw.rmse, expected.idw.rmse));
            assert!(common::same_f64(result.kriging.rmse, expected.kriging.rmse));
            assert!(
                same(&result.idw.predicted, &expected.idw.predicted),
                "concurrent cross_validate_local diverged"
            );
        }
    }
}

// Every public interpolator can be shared across threads.

#[test]
fn public_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<apbase::Map>();
    assert_send_sync::<apbase::MapResult>();
    assert_send_sync::<apbase::idw::IDW>();
    assert_send_sync::<apbase::kriging::Kriging>();
    assert_send_sync::<apbase::cokriging::CoKriging>();
    assert_send_sync::<apbase::variogram::Variogram>();
    assert_send_sync::<apbase::filtering::SpatialFilter>();
    assert_send_sync::<apbase::grid::Grid>();
    assert_send_sync::<apbase::cross_variogram::CrossVariogram>();
    assert_send_sync::<apbase::common::ApbaseError>();
}

#[test]
fn concurrent_maps_match_the_sequential_result() {
    let meuse = common::read_csv("meuse.csv");
    let (x, y) = (&meuse["x"], &meuse["y"]);
    let z: Vec<f64> = meuse["zinc"].iter().map(|v| v.ln()).collect();
    let options = apbase::MapOptions {
        seed: Some(1),
        ..apbase::MapOptions::new(80.0)
    };
    let expected = apbase::create_map(x, y, &z, options.clone()).unwrap();
    let results = run_round(&|_| apbase::create_map(x, y, &z, options.clone()).unwrap());
    for result in results {
        assert!(same(&result.z, &expected.z));
        assert_eq!(result.method, expected.method);
    }
}
