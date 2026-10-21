mod common;

use apbase::cross_variogram::{
    Correction, CrossVariogram, CrossVariogramOptions, ScreeningOptions, VariableXYZ,
    screen_secondary_variables,
};
use apbase::variogram::{Variogram, VariogramOptions};
use common::{NumpyRng, assert_allclose};
use indexmap::IndexMap;
use ndarray::Array1;

fn smooth_field(x: &[f64], y: &[f64], seed: u64, n_centers: usize, length_scale: f64) -> Vec<f64> {
    let mut rng = NumpyRng::new(seed);
    let centers = rng.uniform(0.0, 100.0, n_centers * 2);
    let amps = rng.normal(0.0, 1.0, n_centers);
    let mut field = vec![0.0; x.len()];
    for (k, amp) in amps.iter().enumerate() {
        let (cx, cy) = (centers[2 * k], centers[2 * k + 1]);
        for (i, value) in field.iter_mut().enumerate() {
            let d2 = (x[i] - cx).powi(2) + (y[i] - cy).powi(2);
            *value += amp * (-d2 / (2.0 * length_scale.powi(2))).exp();
        }
    }
    field
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(p, q)| p + q).collect()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn pearson(u: &[f64], v: &[f64]) -> f64 {
    let (mu, mv) = (mean(u), mean(v));
    let cov: f64 = u.iter().zip(v).map(|(a, b)| (a - mu) * (b - mv)).sum();
    let su: f64 = u.iter().map(|a| (a - mu).powi(2)).sum();
    let sv: f64 = v.iter().map(|b| (b - mv).powi(2)).sum();
    cov / (su * sv).sqrt()
}

fn fisher_p_value_reference(rho: f64, n_pairs: i64) -> f64 {
    let z_fisher = 0.5 * ((1.0 + rho) / (1.0 - rho)).ln();
    let z_stat = z_fisher * ((n_pairs - 3) as f64).sqrt();
    let normal_cdf = 0.5 * (1.0 + (1.0 - common::erfc(z_stat.abs() / 2f64.sqrt())));
    2.0 * (1.0 - normal_cdf)
}

fn with_exact_correlation(z0: &[f64], rho: f64, seed: u64) -> Vec<f64> {
    let mut rng = NumpyRng::new(seed);
    let m0 = mean(z0);
    let centered: Vec<f64> = z0.iter().map(|v| v - m0).collect();
    let raw = rng.normal(0.0, 1.0, z0.len());
    let mr = mean(&raw);
    let raw_centered: Vec<f64> = raw.iter().map(|v| v - mr).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
    let scale = dot(&raw_centered, &centered) / dot(&centered, &centered);
    let mut orthogonal: Vec<f64> = raw_centered
        .iter()
        .zip(&centered)
        .map(|(r, c)| r - scale * c)
        .collect();
    let norm = dot(&orthogonal, &orthogonal).sqrt();
    orthogonal.iter_mut().for_each(|v| *v /= norm);
    let norm0 = dot(&centered, &centered).sqrt();
    centered
        .iter()
        .zip(&orthogonal)
        .map(|(c, o)| rho * (c / norm0) + (1.0 - rho * rho).sqrt() * o)
        .collect()
}

fn xyz(x: &[f64], y: &[f64], z: &[f64]) -> VariableXYZ {
    (
        Array1::from(x.to_vec()),
        Array1::from(y.to_vec()),
        Array1::from(z.to_vec()),
    )
}

#[test]
fn test_strongly_correlated_candidate_is_selected() {
    let x = NumpyRng::new(10).uniform(0.0, 100.0, 150);
    let y = NumpyRng::new(11).uniform(0.0, 100.0, 150);
    let latent = smooth_field(&x, &y, 12, 8, 25.0);
    let z0 = add(&latent, &NumpyRng::new(13).normal(0.0, 0.1, 150));
    let z_corr: Vec<f64> = latent
        .iter()
        .zip(NumpyRng::new(14).normal(0.0, 0.15, 150))
        .map(|(l, n)| 1.5 * l + n)
        .collect();

    let candidates = IndexMap::from([("corr".to_string(), xyz(&x, &y, &z_corr))]);
    let result = screen_secondary_variables(
        &x,
        &y,
        &z0,
        &candidates,
        ScreeningOptions {
            n_lags: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &result.report["corr"];
    assert_eq!(report.status, "ok");
    assert!(report.decision);
    assert!(result.selected.contains_key("corr"));
    assert!((report.rho - pearson(&z0, &z_corr)).abs() <= 1e-9);
    assert!((report.p_value - fisher_p_value_reference(report.rho, report.n_pairs)).abs() <= 1e-6);
}

#[test]
fn test_pure_noise_candidate_is_rejected_by_p_value() {
    let x = NumpyRng::new(20).uniform(0.0, 100.0, 150);
    let y = NumpyRng::new(21).uniform(0.0, 100.0, 150);
    let z0 = add(
        &smooth_field(&x, &y, 22, 8, 25.0),
        &NumpyRng::new(23).normal(0.0, 0.1, 150),
    );
    let z_noise = NumpyRng::new(24).normal(0.0, 1.0, 150);

    let candidates = IndexMap::from([("noise".to_string(), xyz(&x, &y, &z_noise))]);
    let result = screen_secondary_variables(
        &x,
        &y,
        &z0,
        &candidates,
        ScreeningOptions {
            n_lags: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &result.report["noise"];
    assert_eq!(report.status, "ok");
    assert!(!report.decision);
    assert!(!result.selected.contains_key("noise"));
    assert!((report.p_value - fisher_p_value_reference(report.rho, report.n_pairs)).abs() <= 1e-6);
    assert!(report.p_value > 0.05);
}

#[test]
fn test_insufficient_sample_reports_status_without_raising() {
    let shared_x = [0.0, 10.0, 0.0];
    let shared_y = [0.0, 0.0, 10.0];
    let primary_only = NumpyRng::new(50).uniform(20.0, 100.0, 54);
    let candidate_only = NumpyRng::new(51).uniform(20.0, 100.0, 54);
    let column =
        |values: &[f64], c: usize| -> Vec<f64> { values.iter().skip(c).step_by(2).copied().collect() };

    let x0: Vec<f64> = shared_x.iter().copied().chain(column(&primary_only, 0)).collect();
    let y0: Vec<f64> = shared_y.iter().copied().chain(column(&primary_only, 1)).collect();
    let z0 = add(
        &smooth_field(&x0, &y0, 52, 8, 25.0),
        &NumpyRng::new(53).normal(0.0, 0.05, x0.len()),
    );
    let xc: Vec<f64> = shared_x
        .iter()
        .copied()
        .chain(column(&candidate_only, 0))
        .collect();
    let yc: Vec<f64> = shared_y
        .iter()
        .copied()
        .chain(column(&candidate_only, 1))
        .collect();
    let zc = add(
        &smooth_field(&xc, &yc, 54, 8, 25.0),
        &NumpyRng::new(55).normal(0.0, 0.05, xc.len()),
    );

    let candidates = IndexMap::from([("barely_overlapping".to_string(), xyz(&xc, &yc, &zc))]);
    let result = screen_secondary_variables(
        &x0,
        &y0,
        &z0,
        &candidates,
        ScreeningOptions {
            n_lags: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let report = &result.report["barely_overlapping"];
    assert_eq!(report.n_pairs, 3);
    assert_eq!(report.status, "insufficient sample");
    assert!(!report.decision);
    assert!(!result.selected.contains_key("barely_overlapping"));
}

#[test]
fn test_bonferroni_correction_reduces_false_positive_vs_none() {
    let n = 40;
    let x = NumpyRng::new(30).uniform(0.0, 100.0, n);
    let y = NumpyRng::new(31).uniform(0.0, 100.0, n);
    let latent = smooth_field(&x, &y, 32, 8, 40.0);
    let z0 = add(&latent, &NumpyRng::new(33).normal(0.0, 0.05, n));
    let z_borderline = with_exact_correlation(&z0, 0.35, 34);

    let mut candidates = IndexMap::from([("borderline".to_string(), xyz(&x, &y, &z_borderline))]);
    for i in 0..9 {
        candidates.insert(
            format!("filler_noise_{i}"),
            xyz(&x, &y, &NumpyRng::new(100 + i).normal(0.0, 1.0, n)),
        );
    }
    let reference_p = fisher_p_value_reference(0.35, n as i64);
    let bonferroni_alpha = 0.05 / (candidates.len() as f64);
    assert!(
        bonferroni_alpha < reference_p && reference_p < 0.05,
        "scenario straddles both thresholds"
    );

    let none = screen_secondary_variables(
        &x,
        &y,
        &z0,
        &candidates,
        ScreeningOptions {
            correction: Correction::None,
            n_lags: 8,
            ..Default::default()
        },
    )
    .unwrap();
    let bonferroni = screen_secondary_variables(
        &x,
        &y,
        &z0,
        &candidates,
        ScreeningOptions {
            correction: Correction::Bonferroni,
            n_lags: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(none.report["borderline"].decision);
    assert!(none.selected.contains_key("borderline"));
    assert!(!bonferroni.report["borderline"].decision);
    assert!(!bonferroni.selected.contains_key("borderline"));
}

#[test]
fn test_cross_variogram_matches_reference_correlation_and_reduces_to_auto_variogram_when_equal() {
    let x = NumpyRng::new(40).uniform(0.0, 100.0, 150);
    let y = NumpyRng::new(41).uniform(0.0, 100.0, 150);
    let z0 = add(
        &smooth_field(&x, &y, 42, 8, 25.0),
        &NumpyRng::new(43).normal(0.0, 0.05, 150),
    );

    let auto = Variogram::from_data(
        &x,
        &y,
        &z0,
        VariogramOptions {
            n_lags: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let cross_self = CrossVariogram::from_data(
        &x,
        &y,
        &z0,
        &x,
        &y,
        &z0,
        CrossVariogramOptions {
            n_lags: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let h: Array1<f64> = Array1::linspace(1.0, 60.0, 30);
    assert_allclose(
        &auto.evaluate(&h).unwrap(),
        &cross_self.evaluate(&h).unwrap(),
        0.05,
        0.05,
        "auto vs cross",
    );
    assert!((cross_self.rho().unwrap() - 1.0).abs() <= 1e-9);
}

#[test]
fn test_cross_variogram_survives_repeated_fits_with_many_duplicate_points() {
    let mut rng = NumpyRng::new(77);
    let n_unique = 500;
    let xu = rng.uniform(0.0, 100.0, n_unique);
    let yu = rng.uniform(0.0, 100.0, n_unique);
    let field = smooth_field(&xu, &yu, 78, 8, 25.0);
    let zu = add(&field, &rng.normal(0.0, 0.1, n_unique));
    let dup_count = 400;
    let dup_idx = rng.integers(0, n_unique as i64, dup_count);
    let jitter = rng.normal(0.0, 0.01, dup_count);
    let x: Vec<f64> = xu
        .iter()
        .copied()
        .chain(dup_idx.iter().map(|i| xu[*i as usize]))
        .collect();
    let y: Vec<f64> = yu
        .iter()
        .copied()
        .chain(dup_idx.iter().map(|i| yu[*i as usize]))
        .collect();
    let z: Vec<f64> = zu
        .iter()
        .copied()
        .chain(dup_idx.iter().zip(&jitter).map(|(i, j)| zu[*i as usize] + j))
        .collect();
    let z_plus: Vec<f64> = z.iter().map(|v| v + 1.0).collect();

    for _ in 0..30 {
        let cross = CrossVariogram::from_data(
            &x,
            &y,
            &z,
            &x,
            &y,
            &z_plus,
            CrossVariogramOptions {
                n_lags: 20,
                max_pairs: 50_000,
                max_distance: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(cross.n_pairs().unwrap() > 0);
        assert!(cross.rho().unwrap().is_finite());
    }
}

#[test]
fn test_cross_variogram_rejects_size_mismatch() {
    let x = [0.0, 1.0, 2.0];
    let z = [1.0, 2.0, 3.0];
    let error = CrossVariogram::new(CrossVariogramOptions::default())
        .unwrap()
        .fit(&x, &x, &z, &x, &x, &[1.0, 2.0])
        .err()
        .unwrap();
    assert!(error.message().contains("size"));
    assert!(error.is(apbase::common::ErrorKind::ValueError));
}

#[test]
fn screening_validation_messages() {
    let x = [0.0, 1.0];
    let empty: IndexMap<String, VariableXYZ> = IndexMap::new();
    let error = screen_secondary_variables(&x, &x, &x, &empty, ScreeningOptions::default()).unwrap_err();
    assert_eq!(error.message(), "candidates must contain at least one entry");
    let candidates = IndexMap::from([("a".to_string(), xyz(&x, &x, &x))]);
    let message = |options: ScreeningOptions| {
        screen_secondary_variables(&x, &x, &x, &candidates, options)
            .unwrap_err()
            .to_string()
    };
    assert_eq!(
        message(ScreeningOptions {
            alpha: 1.0,
            ..Default::default()
        }),
        "alpha must be strictly between 0 and 1"
    );
    assert_eq!(
        message(ScreeningOptions {
            min_abs_rho: 1.5,
            ..Default::default()
        }),
        "min_abs_rho must be between 0 and 1"
    );
    assert_eq!(
        message(ScreeningOptions {
            max_violation_fraction: -0.1,
            ..Default::default()
        }),
        "max_violation_fraction must be between 0 and 1"
    );
    assert_eq!(
        "other".parse::<Correction>().unwrap_err().message(),
        "correction must be \"bonferroni\" or \"none\""
    );
    assert_eq!(
        CrossVariogram::new(CrossVariogramOptions {
            max_colocation_dist: Some(0.0),
            ..Default::default()
        })
        .unwrap_err()
        .message(),
        "max_colocation_dist must be None or finite and greater than zero"
    );
}

#[test]
fn test_screening_selects_nearly_collinear_secondary() {
    let mut rng = NumpyRng::new(1);
    let (x0, y0) = (rng.uniform(0.0, 1000.0, 200), rng.uniform(0.0, 1000.0, 200));
    let noise0 = rng.normal(0.0, 0.5, 200);
    let z0: Vec<f64> = (0..200)
        .map(|i| (x0[i] / 150.0).sin() * (y0[i] / 150.0).cos() * 10.0 + noise0[i])
        .collect();
    let (xu, yu) = (rng.uniform(0.0, 1000.0, 800), rng.uniform(0.0, 1000.0, 800));
    let noiseu = rng.normal(0.0, 1.0, 800);
    let zu: Vec<f64> = (0..800)
        .map(|i| 2.0 * (xu[i] / 150.0).sin() * (yu[i] / 150.0).cos() * 10.0 + noiseu[i])
        .collect();
    let candidates = IndexMap::from([(
        "ndvi".to_string(),
        (Array1::from(xu), Array1::from(yu), Array1::from(zu)),
    )]);

    let result = screen_secondary_variables(
        &x0,
        &y0,
        &z0,
        &candidates,
        ScreeningOptions {
            max_colocation_dist: Some(30.0),
            ..Default::default()
        },
    )
    .unwrap();

    let report = &result.report["ndvi"];
    assert!(report.rho > 0.98, "{}", report.rho);
    assert!(
        report.admissible_fraction >= 0.9,
        "{}",
        report.admissible_fraction
    );
    assert!(report.decision);
}
