mod common;

use apbase::filtering::{FilterLevel, SpatialFilter, SpatialFilterOptions};
use common::{NumpyRng, approx, erfc};
use ndarray::{Array1, array};

#[test]
fn test_spatial_filter_keeps_finite_constant_values_with_explicit_radius() {
    let x = [0.0, 10.0, 0.0, 10.0, f64::NAN];
    let y = [0.0, 0.0, 10.0, 10.0, 10.0];
    let z = [10.0, 10.0, 10.0, 10.0, 10.0];
    let filter = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            radius: Some(30.0),
            apply_global_iqr: false,
            max_neighbors: 0,
            filter_level: FilterLevel::Light,
            ..Default::default()
        },
    )
    .unwrap();
    let (rx, ry, rz) = filter.filter().unwrap();
    assert_eq!(filter.mask().unwrap(), array![true, true, true, true, false]);
    assert_eq!(rx.to_vec(), x[..4].to_vec());
    assert_eq!(ry.to_vec(), y[..4].to_vec());
    assert_eq!(rz.to_vec(), z[..4].to_vec());
}

#[test]
fn test_spatial_filter_dedup_radius_fixes_duplicate_clump_nan() {
    let mut rng = NumpyRng::new(0);
    let n_strip = 40;
    let x_strip: Vec<f64> = (0..n_strip).map(|i| i as f64 * 5.0).collect();
    let y_strip = vec![0.0; n_strip];
    let z_strip: Vec<f64> = rng.normal(0.0, 0.3, n_strip).iter().map(|v| 5.0 + v).collect();
    let n_clump = 25;
    let (cx, cy) = (x_strip[20], y_strip[20]);
    let x_clump: Vec<f64> = rng.uniform(-0.15, 0.15, n_clump).iter().map(|v| cx + v).collect();
    let y_clump: Vec<f64> = rng.uniform(-0.15, 0.15, n_clump).iter().map(|v| cy + v).collect();
    let z_clump = vec![5.4; n_clump];

    let x: Vec<f64> = x_strip.into_iter().chain(x_clump).collect();
    let y: Vec<f64> = y_strip.into_iter().chain(y_clump).collect();
    let z: Vec<f64> = z_strip.into_iter().chain(z_clump).collect();

    let base = SpatialFilterOptions {
        radius: Some(15.0),
        apply_global_iqr: false,
        max_neighbors: 20,
        ..Default::default()
    };
    let no_dedup = SpatialFilter::from_data(&x, &y, &z, base.clone()).unwrap();
    assert!(no_dedup.statistics().unwrap().local_z.iter().any(|v| v.is_nan()));
    let with_dedup = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            dedup_radius: 0.5,
            ..base
        },
    )
    .unwrap();
    assert!(
        !with_dedup
            .statistics()
            .unwrap()
            .local_z
            .iter()
            .any(|v| v.is_nan())
    );
}

#[test]
fn test_spatial_filter_dedup_radius_zero_matches_disabled_default() {
    let x = [0.0, 10.0, 0.0, 10.0, 5.0];
    let y = [0.0, 0.0, 10.0, 10.0, 5.0];
    let z = [5.0, 7.0, 8.0, 10.0, 6.0];
    let options = SpatialFilterOptions {
        radius: Some(20.0),
        apply_global_iqr: false,
        max_neighbors: 0,
        ..Default::default()
    };
    let baseline = SpatialFilter::from_data(&x, &y, &z, options.clone()).unwrap();
    let explicit_zero = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            dedup_radius: 0.0,
            ..options
        },
    )
    .unwrap();
    assert_eq!(baseline.mask().unwrap(), explicit_zero.mask().unwrap());
    let (a, b) = (
        baseline.statistics().unwrap(),
        explicit_zero.statistics().unwrap(),
    );
    for (left, right) in [
        (&a.pct_diff, &b.pct_diff),
        (&a.local_z, &b.local_z),
        (&a.local_prob, &b.local_prob),
    ] {
        assert!(
            left.iter()
                .zip(right.iter())
                .all(|(l, r)| common::same_f64(*l, *r))
        );
    }
}

#[test]
fn test_spatial_filter_dedup_radius_with_full_radius_neighbor_scan() {
    let x = [0.0, 0.05, 0.05, 20.0, 40.0];
    let y = [0.0, 0.0, 0.05, 0.0, 0.0];
    let z = [5.0, 5.0, 5.0, 9.0, 5.2];
    let filtered = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            radius: Some(100.0),
            apply_global_iqr: false,
            max_neighbors: 0,
            dedup_radius: 1.0,
            ..Default::default()
        },
    )
    .unwrap();
    let statistics = filtered.statistics().unwrap();
    assert!(statistics.pct_diff.iter().all(|v| v.is_finite()));
    assert!(statistics.local_z.iter().all(|v| v.is_finite()));
    assert!(statistics.local_prob.iter().all(|v| v.is_finite()));
}

#[test]
fn test_spatial_filter_uses_shared_variogram_radius() {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for gy in 0..8 {
        for gx in 0..8 {
            x.push(f64::from(gx));
            y.push(f64::from(gy));
        }
    }
    let z: Vec<f64> = x
        .iter()
        .zip(&y)
        .map(|(a, b)| (a * 0.3).sin() + (b * 0.2).cos())
        .collect();
    let filter = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            apply_global_iqr: false,
            max_neighbors: 16,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(filter.model_params().unwrap().range > 0.0);
    let mask: Array1<bool> = filter.mask().unwrap();
    assert_eq!(mask.len(), x.len());
}

#[test]
fn test_spatial_filter_knn_local_stats_stable_for_large_mean_small_variance() {
    let mut rng = NumpyRng::new(7);
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 100.0).collect();
    let y = vec![0.0; n];
    let z: Vec<f64> = rng.normal(0.0, 1.0e-4, n).iter().map(|v| 150.0 + v).collect();
    let radius = 250.0;
    let filter = SpatialFilter::from_data(
        &x,
        &y,
        &z,
        SpatialFilterOptions {
            radius: Some(radius),
            apply_global_iqr: false,
            max_neighbors: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let statistics = filter.statistics().unwrap();
    assert!(
        !statistics.local_z.iter().any(|v| v.is_nan()),
        "local_z collapsed to NaN under cancellation"
    );
    assert!(!statistics.local_prob.iter().any(|v| v.is_nan()));

    for i in 0..n {
        let neighbors: Vec<f64> = (0..n)
            .filter(|k| *k != i && (x[*k] - x[i]).abs() <= radius)
            .map(|k| z[k])
            .collect();
        assert!(!neighbors.is_empty() && neighbors.len() <= 4);
        let mean = neighbors.iter().sum::<f64>() / neighbors.len() as f64;
        let var = neighbors.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / neighbors.len() as f64;
        let expected_z = (z[i] - mean) / var.sqrt();
        let expected_prob = erfc(expected_z.abs() / 2f64.sqrt());
        assert!(
            approx(statistics.local_z[i], expected_z, 1e-6, 1e-9),
            "local_z[{i}]"
        );
        assert!(
            approx(statistics.local_prob[i], expected_prob, 1e-6, 1e-9),
            "local_prob[{i}]"
        );
    }
}

#[test]
fn filter_error_messages() {
    let unfitted = SpatialFilter::new(SpatialFilterOptions::default()).unwrap();
    assert_eq!(
        unfitted.mask().unwrap_err().message(),
        "call fit(x, y, z) before reading filtering results"
    );
    let mut filter = SpatialFilter::new(SpatialFilterOptions {
        radius: Some(5.0),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        filter.fit(&[f64::NAN], &[0.0], &[1.0]).unwrap_err().message(),
        "SpatialFilter requires at least one finite point"
    );
    filter
        .fit(&[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0])
        .unwrap();
    assert_eq!(
        filter.model_params().unwrap_err().message(),
        "SpatialFilter was fit with an explicit radius; no variogram model was fitted"
    );
    assert_eq!(
        filter
            .call(&[0.0, 1.0], &[0.0, 0.0], &[1.0, 2.0])
            .unwrap()
            .0
            .len(),
        filter.mask().unwrap().iter().filter(|k| **k).count()
    );
}
