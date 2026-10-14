//! Collocated, ICM and LMC cokriging (`apbase::cokriging`).
//!
//! Fitted matrices are read through the public `model_params()`.

mod common;

use apbase::cokriging::{
    CoKriging, CoKrigingMethod, CoKrigingModelParams, CoKrigingOptions, SecondaryInput, co_kriging,
};
use apbase::common::ErrorKind;
use apbase::kriging::{Kriging, KrigingOptions};
use common::{NumpyRng, assert_allclose, cholesky, eigvalsh, mean_std, solve};
use ndarray::{Array1, Array2, Array3, array};

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
                let c = ratio.clamp(0.0, 1.0);
                1.5 * c - 0.5 * c.powi(3)
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

fn standardize(z0: &[f64], zu: &[f64]) -> Vec<f64> {
    let (m0, s0) = mean_std(z0);
    let (mu, su) = mean_std(zu);
    zu.iter().map(|v| m0 + (v - mu) * (s0 / su)).collect()
}

type Xyz = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Independent solver of standardized collocated cokriging (MM1).
#[allow(clippy::too_many_arguments)]
fn reference_collocated(
    x0: &[f64],
    y0: &[f64],
    z0: &[f64],
    secondaries: &[Xyz],
    rhos: &[f64],
    targets: &Array2<f64>,
    nugget: f64,
    partial_sill: f64,
    model_range: f64,
    search: Option<(f64, usize)>,
) -> Array1<f64> {
    let k = secondaries.len();
    let sill = nugget + partial_sill;
    let standardized: Vec<Vec<f64>> = secondaries.iter().map(|(_, _, zu)| standardize(z0, zu)).collect();
    targets
        .rows()
        .into_iter()
        .map(|target| {
            let (tx, ty) = (target[0], target[1]);
            let d2: Vec<f64> = x0
                .iter()
                .zip(y0)
                .map(|(x, y)| (x - tx).powi(2) + (y - ty).powi(2))
                .collect();
            let order: Vec<usize> = match search {
                Some((radius, max_neighbors)) => {
                    let mut within: Vec<usize> =
                        (0..d2.len()).filter(|i| d2[*i] <= radius * radius).collect();
                    within.sort_by(|a, b| d2[*a].total_cmp(&d2[*b]));
                    within.truncate(max_neighbors);
                    within
                }
                None => (0..x0.len()).collect(),
            };
            let nn = order.len();
            let nsys = nn + k + 1;
            let mut a = Array2::<f64>::zeros((nsys, nsys));
            let mut b = Array1::<f64>::zeros(nsys);
            for (i, pi) in order.iter().enumerate() {
                for (j, pj) in order.iter().enumerate() {
                    let h = ((x0[*pi] - x0[*pj]).powi(2) + (y0[*pi] - y0[*pj]).powi(2)).sqrt();
                    a[[i, j]] = covariance(h, MODEL_SPHERICAL, nugget, partial_sill, model_range);
                }
            }
            let c0: Vec<f64> = order
                .iter()
                .map(|p| {
                    covariance(
                        ((x0[*p] - tx).powi(2) + (y0[*p] - ty).powi(2)).sqrt(),
                        MODEL_SPHERICAL,
                        nugget,
                        partial_sill,
                        model_range,
                    )
                })
                .collect();
            for i in 0..nn {
                b[i] = c0[i];
            }
            let mut collocated = vec![0.0; k];
            for u in 0..k {
                let (xu, yu, _) = &secondaries[u];
                let nearest = (0..xu.len())
                    .min_by(|p, q| {
                        let dp = (xu[*p] - tx).powi(2) + (yu[*p] - ty).powi(2);
                        let dq = (xu[*q] - tx).powi(2) + (yu[*q] - ty).powi(2);
                        dp.total_cmp(&dq)
                    })
                    .unwrap();
                collocated[u] = standardized[u][nearest];
                for i in 0..nn {
                    a[[i, nn + u]] = rhos[u] * c0[i];
                    a[[nn + u, i]] = rhos[u] * c0[i];
                }
                a[[nn + u, nn + u]] = sill;
                b[nn + u] = rhos[u] * sill;
                for v in u + 1..k {
                    a[[nn + u, nn + v]] = rhos[u] * rhos[v] * sill;
                    a[[nn + v, nn + u]] = rhos[u] * rhos[v] * sill;
                }
            }
            for i in 0..nn + k {
                a[[i, nsys - 1]] = 1.0;
                a[[nsys - 1, i]] = 1.0;
            }
            a[[nsys - 1, nsys - 1]] = 0.0;
            b[nsys - 1] = 1.0;
            let w = solve(&a, &b);
            (0..nn).map(|i| w[i] * z0[order[i]]).sum::<f64>()
                + (0..k).map(|u| w[nn + u] * collocated[u]).sum::<f64>()
        })
        .collect()
}

fn model_values(nugget: f64, partial_sill: f64, model_range: f64, rhos: &[f64]) -> Array1<f64> {
    [MODEL_SPHERICAL as f64, nugget, partial_sill, model_range, 0.0]
        .iter()
        .chain(rhos)
        .copied()
        .collect()
}

fn xyz(x: &[f64], y: &[f64], z: &[f64]) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    (
        Array1::from(x.to_vec()),
        Array1::from(y.to_vec()),
        Array1::from(z.to_vec()),
    )
}

fn collocated(model: Array1<f64>, radius: f64, max_neighbors: i64) -> CoKrigingOptions {
    CoKrigingOptions {
        model_values: Some(model),
        radius: Some(radius),
        max_neighbors,
        min_neighbors: 1,
        ..Default::default()
    }
}

#[test]
fn test_collocated_cokriging_k1_matches_reference() {
    let x0 = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y0 = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z0 = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let xu = [1.0, 9.0, 1.0, 9.0, 5.0, 6.0];
    let yu = [1.0, 1.0, 9.0, 9.0, 5.0, 3.0];
    let zu = [10.0, 14.0, 16.0, 20.0, 13.0, 12.0];
    let grid = array![[5.0, 5.0], [2.0, 8.0], [9.0, 1.0]];
    let (nugget, psill, range, rho) = (0.5, 4.0, 30.0, 0.9);

    let secondaries = SecondaryInput::from([("secondary", xyz(&xu, &yu, &zu))]);
    let result = CoKriging::from_data(
        &x0,
        &y0,
        &z0,
        &secondaries,
        collocated(model_values(nugget, psill, range, &[rho]), range, 10),
    )
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_collocated(
        &x0,
        &y0,
        &z0,
        &[(xu.to_vec(), yu.to_vec(), zu.to_vec())],
        &[rho],
        &grid,
        nugget,
        psill,
        range,
        None,
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "K1");
}

/// (5, 5) has 2 primaries within the radius, below `min_neighbors = 3`, so it
/// uses all 6 nearest and keeps its secondary; (100, 100) has nothing within
/// the radius, so the secondary is left out and it is ordinary kriging.
#[test]
fn test_collocated_cokriging_uses_nearest_primaries_and_drops_far_secondaries() {
    let x0 = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y0 = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z0 = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let xu = [1.0, 9.0, 1.0, 9.0, 5.0, 6.0];
    let yu = [1.0, 1.0, 9.0, 9.0, 5.0, 3.0];
    let zu = [10.0, 14.0, 16.0, 20.0, 13.0, 12.0];
    let (nugget, psill, range, rho) = (0.5, 4.0, 30.0, 0.9);

    let secondaries = SecondaryInput::from([("secondary", xyz(&xu, &yu, &zu))]);
    let options = CoKrigingOptions {
        min_neighbors: 3,
        ..collocated(model_values(nugget, psill, range, &[rho]), 5.0, 10)
    };
    let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, options).unwrap();

    let near = array![[5.0, 5.0]];
    let secondary = (xu.to_vec(), yu.to_vec(), zu.to_vec());
    let expected = reference_collocated(
        &x0,
        &y0,
        &z0,
        &[secondary],
        &[rho],
        &near,
        nugget,
        psill,
        range,
        None,
    );
    assert_allclose(&ck.interpolate(&near).unwrap(), &expected, 1e-6, 1e-6, "near");

    let far = array![[100.0, 100.0]];
    let expected = reference_collocated(&x0, &y0, &z0, &[], &[], &far, nugget, psill, range, None);
    assert_allclose(&ck.interpolate(&far).unwrap(), &expected, 1e-6, 1e-6, "far");
}

#[test]
fn test_collocated_cokriging_k2_matches_reference_with_correlated_secondaries() {
    let mut rng = NumpyRng::new(7);
    let n = 25;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let z0 = rng.uniform(1.0, 10.0, n);
    let xu1 = rng.uniform(0.0, 100.0, n);
    let yu1 = rng.uniform(0.0, 100.0, n);
    let zu1 = rng.uniform(1.0, 10.0, n);
    let xu2 = rng.uniform(0.0, 100.0, n);
    let yu2 = rng.uniform(0.0, 100.0, n);
    let zu2 = rng.uniform(1.0, 10.0, n);
    let grid = Array2::from_shape_vec((8, 2), rng.uniform(10.0, 90.0, 16)).unwrap();
    let (nugget, psill, range, rho1, rho2, search_radius) = (0.5, 4.0, 60.0, 0.8, 0.6, 1000.0);

    let secondaries = SecondaryInput::from(vec![xyz(&xu1, &yu1, &zu1), xyz(&xu2, &yu2, &zu2)]);
    let result = CoKriging::from_data(
        &x0,
        &y0,
        &z0,
        &secondaries,
        collocated(
            model_values(nugget, psill, range, &[rho1, rho2]),
            search_radius,
            n as i64,
        ),
    )
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_collocated(
        &x0,
        &y0,
        &z0,
        &[(xu1, yu1, zu1), (xu2, yu2, zu2)],
        &[rho1, rho2],
        &grid,
        nugget,
        psill,
        range,
        None,
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "K2");
}

#[test]
fn test_collocated_cokriging_truncates_primary_neighbors_correctly() {
    let mut rng = NumpyRng::new(11);
    let n = 50;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let z0 = rng.uniform(1.0, 10.0, n);
    let xu = rng.uniform(0.0, 100.0, n);
    let yu = rng.uniform(0.0, 100.0, n);
    let zu = rng.uniform(1.0, 10.0, n);
    let grid = Array2::from_shape_vec((10, 2), rng.uniform(10.0, 90.0, 20)).unwrap();
    let (nugget, psill, range, rho, radius, max_neighbors) = (0.5, 4.0, 30.0, 0.7, 40.0, 8);

    let secondaries = SecondaryInput::from([("secondary", xyz(&xu, &yu, &zu))]);
    let result = CoKriging::from_data(
        &x0,
        &y0,
        &z0,
        &secondaries,
        collocated(
            model_values(nugget, psill, range, &[rho]),
            radius,
            max_neighbors as i64,
        ),
    )
    .unwrap()
    .interpolate(&grid)
    .unwrap();
    let expected = reference_collocated(
        &x0,
        &y0,
        &z0,
        &[(xu, yu, zu)],
        &[rho],
        &grid,
        nugget,
        psill,
        range,
        Some((radius, max_neighbors)),
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "truncated");
}

#[test]
fn test_collocated_cokriging_fitted_model_beats_plain_kriging_when_primary_sparse() {
    let mut rng = NumpyRng::new(11);
    let n_dense = 150;
    let x = rng.uniform(0.0, 100.0, n_dense);
    let y = rng.uniform(0.0, 100.0, n_dense);
    let centers = rng.uniform(0.0, 100.0, 16);
    let amps = rng.normal(0.0, 1.0, 8);
    let mut latent = vec![0.0; n_dense];
    for (k, amp) in amps.iter().enumerate() {
        for i in 0..n_dense {
            let d2 = (x[i] - centers[2 * k]).powi(2) + (y[i] - centers[2 * k + 1]).powi(2);
            latent[i] += amp * (-d2 / (2.0 * 25.0f64.powi(2))).exp();
        }
    }
    let z_secondary: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.05, n_dense))
        .map(|(l, e)| 1.5 * l + e)
        .collect();
    let sparse_idx = rng.choice(n_dense as i64, 20);
    let x0: Vec<f64> = sparse_idx.iter().map(|i| x[*i as usize]).collect();
    let y0: Vec<f64> = sparse_idx.iter().map(|i| y[*i as usize]).collect();
    let z0: Vec<f64> = sparse_idx
        .iter()
        .zip(rng.normal(0.0, 0.05, 20))
        .map(|(i, e)| latent[*i as usize] + e)
        .collect();
    let remaining: Vec<i64> = (0..n_dense as i64).filter(|i| !sparse_idx.contains(i)).collect();
    let holdout: Vec<usize> = rng
        .choice(remaining.len() as i64, 20)
        .iter()
        .map(|p| remaining[*p as usize] as usize)
        .collect();
    let targets: Vec<[f64; 2]> = holdout.iter().map(|i| [x[*i], y[*i]]).collect();
    let actual: Vec<f64> = holdout.iter().map(|i| latent[*i]).collect();

    let secondaries = SecondaryInput::from([("secondary", xyz(&x, &y, &z_secondary))]);
    let cokrig = CoKriging::from_data(
        &x0,
        &y0,
        &z0,
        &secondaries,
        CoKrigingOptions {
            radius: Some(50.0),
            max_neighbors: 15,
            min_neighbors: 2,
            ..Default::default()
        },
    )
    .unwrap()
    .interpolate(&targets)
    .unwrap();
    let kriging = Kriging::from_data(
        &x0,
        &y0,
        &z0,
        KrigingOptions {
            radius: Some(50.0),
            max_neighbors: 15,
            min_neighbors: 2,
            ..Default::default()
        },
    )
    .unwrap()
    .interpolate(&targets)
    .unwrap();
    assert!(cokrig.iter().all(|v| v.is_finite()) && kriging.iter().all(|v| v.is_finite()));
    let rmse = |est: &Array1<f64>| {
        (est.iter().zip(&actual).map(|(e, a)| (e - a).powi(2)).sum::<f64>() / actual.len() as f64).sqrt()
    };
    assert!(
        rmse(&cokrig) < rmse(&kriging),
        "cokriging {} vs kriging {}",
        rmse(&cokrig),
        rmse(&kriging)
    );
}

#[test]
fn test_co_kriging_free_function_matches_class() {
    let x0 = [0.0, 10.0, 0.0, 10.0, 4.0, 7.0];
    let y0 = [0.0, 0.0, 10.0, 10.0, 6.0, 2.0];
    let z0 = [5.0, 7.0, 8.0, 10.0, 6.5, 6.0];
    let zu = [10.0, 14.0, 16.0, 20.0, 13.0, 12.0];
    let grid = array![[5.0, 5.0], [2.0, 8.0]];
    let secondaries = SecondaryInput::from([("secondary", xyz(&x0, &y0, &zu))]);
    let options = collocated(model_values(0.5, 4.0, 30.0, &[0.9]), 30.0, 10);
    let from_function = co_kriging(&x0, &y0, &z0, &secondaries, &grid, options.clone()).unwrap();
    let from_class = CoKriging::from_data(&x0, &y0, &z0, &secondaries, options)
        .unwrap()
        .interpolate(&grid)
        .unwrap();
    assert_eq!(from_function, from_class);
}

#[test]
fn test_cokriging_lmc_model_values_override_raises_not_implemented() {
    let error = CoKriging::new(CoKrigingOptions {
        method: CoKrigingMethod::Lmc,
        model_values: Some(array![1.0, 0.0, 1.0, 10.0, 0.0, 0.5]),
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::NotImplementedError);
    assert_eq!(
        error.message(),
        "model_values override is not supported yet for method=\"lmc\""
    );
}

fn nearest_within(x: &[f64], y: &[f64], tx: f64, ty: f64, radius: f64, max_neighbors: usize) -> Vec<usize> {
    let d2: Vec<f64> = x
        .iter()
        .zip(y)
        .map(|(px, py)| (px - tx).powi(2) + (py - ty).powi(2))
        .collect();
    let mut within: Vec<usize> = (0..d2.len()).filter(|i| d2[*i] <= radius * radius).collect();
    within.sort_by(|a, b| d2[*a].total_cmp(&d2[*b]));
    within.truncate(max_neighbors);
    within
}

/// General block-kriging reference shared by ICM and LMC (`cov(u, v, h)`).
fn reference_block_cokriging(
    variables: &[Xyz],
    targets: &Array2<f64>,
    radius: f64,
    max_neighbors: usize,
    cov: &dyn Fn(usize, usize, f64) -> f64,
) -> Array1<f64> {
    targets
        .rows()
        .into_iter()
        .map(|target| {
            let (tx, ty) = (target[0], target[1]);
            let neighbors: Vec<Vec<usize>> = variables
                .iter()
                .map(|(x, y, _)| nearest_within(x, y, tx, ty, radius, max_neighbors))
                .collect();
            let mut offsets = vec![0usize];
            for idx in &neighbors {
                offsets.push(offsets.last().unwrap() + idx.len());
            }
            let nsys = offsets.last().unwrap() + 1;
            let mut a = Array2::<f64>::zeros((nsys, nsys));
            let mut b = Array1::<f64>::zeros(nsys);
            for u in 0..variables.len() {
                for (i, pi) in neighbors[u].iter().enumerate() {
                    let gi = offsets[u] + i;
                    for v in 0..variables.len() {
                        for (j, pj) in neighbors[v].iter().enumerate() {
                            let gj = offsets[v] + j;
                            if gj <= gi {
                                continue;
                            }
                            let h = (variables[u].0[*pi] - variables[v].0[*pj])
                                .hypot(variables[u].1[*pi] - variables[v].1[*pj]);
                            let c = cov(u, v, h);
                            a[[gi, gj]] = c;
                            a[[gj, gi]] = c;
                        }
                    }
                    a[[gi, gi]] = cov(u, u, 0.0);
                    let h0 = (variables[u].0[*pi] - tx).hypot(variables[u].1[*pi] - ty);
                    b[gi] = cov(u, 0, h0);
                    a[[gi, nsys - 1]] = 1.0;
                    a[[nsys - 1, gi]] = 1.0;
                }
            }
            b[nsys - 1] = 1.0;
            let w = solve(&a, &b);
            let mut prediction = 0.0;
            for u in 0..variables.len() {
                for (i, pi) in neighbors[u].iter().enumerate() {
                    prediction += w[offsets[u] + i] * variables[u].2[*pi];
                }
            }
            prediction
        })
        .collect()
}

fn smooth_field(x: &[f64], y: &[f64], seed: u64, n_centers: usize, length_scale: f64) -> Vec<f64> {
    let mut rng = NumpyRng::new(seed);
    let centers = rng.uniform(0.0, 100.0, n_centers * 2);
    let amps = rng.normal(0.0, 1.0, n_centers);
    let mut field = vec![0.0; x.len()];
    for (k, amp) in amps.iter().enumerate() {
        for (i, value) in field.iter_mut().enumerate() {
            let d2 = (x[i] - centers[2 * k]).powi(2) + (y[i] - centers[2 * k + 1]).powi(2);
            *value += amp * (-d2 / (2.0 * length_scale.powi(2))).exp();
        }
    }
    field
}

fn icm_matrices(model: &CoKriging) -> (i64, f64, Array2<f64>, Array2<f64>) {
    match model.model_params().unwrap() {
        CoKrigingModelParams::Icm {
            model_id,
            range,
            nugget_matrix,
            sill_matrix,
            ..
        } => (model_id, range, nugget_matrix, sill_matrix),
        other => panic!("ICM params expected, got {other:?}"),
    }
}

fn lmc_matrices(model: &CoKriging) -> (i64, Array1<f64>, Array3<f64>) {
    match model.model_params().unwrap() {
        CoKrigingModelParams::Lmc {
            model_id,
            structure_ranges,
            coefficient_matrices,
            ..
        } => (model_id, structure_ranges, coefficient_matrices),
        other => panic!("LMC params expected, got {other:?}"),
    }
}

fn icm_cov(
    model_id: i64,
    range: f64,
    nugget: &Array2<f64>,
    sill: &Array2<f64>,
) -> impl Fn(usize, usize, f64) -> f64 {
    let (nugget, sill) = (nugget.clone(), sill.clone());
    move |u, v, h| {
        if h <= 1e-6 {
            nugget[[u, v]] + sill[[u, v]]
        } else {
            sill[[u, v]] * (1.0 - variogram_shape(h, model_id, range))
        }
    }
}

fn lmc_cov(
    model_id: i64,
    ranges: &Array1<f64>,
    coefficients: &Array3<f64>,
) -> impl Fn(usize, usize, f64) -> f64 {
    let (ranges, coefficients) = (ranges.clone(), coefficients.clone());
    move |u, v, h| {
        if h <= 1e-6 {
            return (0..coefficients.shape()[0])
                .map(|s| coefficients[[s, u, v]])
                .sum();
        }
        (1..coefficients.shape()[0])
            .map(|s| coefficients[[s, u, v]] * (1.0 - variogram_shape(h, model_id, ranges[s - 1])))
            .sum()
    }
}

fn block_options(method: CoKrigingMethod) -> CoKrigingOptions {
    CoKrigingOptions {
        method,
        radius: Some(40.0),
        max_neighbors: 12,
        min_neighbors: 1,
        ..Default::default()
    }
}

#[test]
fn test_icm_cokriging_k1_matches_reference() {
    let mut rng = NumpyRng::new(6);
    let n = 120;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 1006, 8, 25.0);
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
    let grid = Array2::from_shape_vec((6, 2), rng.uniform(10.0, 90.0, 12)).unwrap();

    let secondaries = SecondaryInput::from([("secondary", xyz(&x0, &y0, &zu))]);
    let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, block_options(CoKrigingMethod::Icm)).unwrap();
    let result = ck.interpolate(&grid).unwrap();
    let (model_id, range, nugget, sill) = icm_matrices(&ck);
    let variables = [
        (x0.clone(), y0.clone(), z0.clone()),
        (x0.clone(), y0.clone(), standardize(&z0, &zu)),
    ];
    let expected = reference_block_cokriging(
        &variables,
        &grid,
        40.0,
        12,
        &icm_cov(model_id, range, &nugget, &sill),
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "ICM K1");
}

#[test]
fn test_icm_cokriging_k2_matches_reference_with_correlated_secondaries() {
    let mut rng = NumpyRng::new(46);
    let n = 100;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 1046, 8, 25.0);
    let z0: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.15, n))
        .map(|(l, e)| l + e)
        .collect();
    let zu1: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.25, n))
        .map(|(l, e)| 1.2 * l + e)
        .collect();
    let zu2: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.25, n))
        .map(|(l, e)| 0.8 * l + e)
        .collect();
    let grid = Array2::from_shape_vec((5, 2), rng.uniform(10.0, 90.0, 10)).unwrap();

    let secondaries = SecondaryInput::from(vec![xyz(&x0, &y0, &zu1), xyz(&x0, &y0, &zu2)]);
    let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, block_options(CoKrigingMethod::Icm)).unwrap();
    let result = ck.interpolate(&grid).unwrap();
    let (model_id, range, nugget, sill) = icm_matrices(&ck);
    let variables = [
        (x0.clone(), y0.clone(), z0.clone()),
        (x0.clone(), y0.clone(), standardize(&z0, &zu1)),
        (x0.clone(), y0.clone(), standardize(&z0, &zu2)),
    ];
    let expected = reference_block_cokriging(
        &variables,
        &grid,
        40.0,
        12,
        &icm_cov(model_id, range, &nugget, &sill),
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "ICM K2");
    assert!(sill[[1, 2]] != 0.0 || nugget[[1, 2]] != 0.0);
}

#[test]
fn test_icm_cokriging_rejects_non_psd_coregionalization() {
    let trial_seed = 0u64;
    let mut rng = NumpyRng::new(trial_seed);
    let n = 250;
    let x = rng.uniform(0.0, 100.0, n);
    let y = rng.uniform(0.0, 100.0, n);
    let f_short = smooth_field(&x, &y, trial_seed * 3 + 1, 40, 4.0);
    let f_med = smooth_field(&x, &y, trial_seed * 3 + 2, 10, 15.0);
    let f_long = smooth_field(&x, &y, trial_seed * 3 + 3, 3, 70.0);
    let z0: Vec<f64> = f_short
        .iter()
        .zip(rng.normal(0.0, 0.01, n))
        .map(|(s, e)| s + e)
        .collect();
    let zu1: Vec<f64> = (0..n)
        .zip(rng.normal(0.0, 0.01, n))
        .map(|(i, e)| 0.6 * f_short[i] + f_med[i] + e)
        .collect();
    let zu2: Vec<f64> = (0..n)
        .zip(rng.normal(0.0, 0.01, n))
        .map(|(i, e)| 0.6 * f_short[i] + f_long[i] + e)
        .collect();

    let secondaries = SecondaryInput::from([
        ("medium_range", xyz(&x, &y, &zu1)),
        ("long_range", xyz(&x, &y, &zu2)),
    ]);
    let error = CoKriging::from_data(
        &x,
        &y,
        &z0,
        &secondaries,
        CoKrigingOptions {
            method: CoKrigingMethod::Icm,
            ..Default::default()
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::CoKrigingNonPSDCoregionalizationError);
    assert!(error.is(ErrorKind::InterpolationError) && error.is(ErrorKind::NativeExecutionError));
}

#[test]
fn test_lmc_cokriging_k1_matches_reference() {
    let mut rng = NumpyRng::new(70);
    let n = 120;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 71, 8, 25.0);
    let z0: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.05, n))
        .map(|(l, e)| l + e)
        .collect();
    let zu: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.1, n))
        .map(|(l, e)| 1.4 * l + e)
        .collect();
    let grid = Array2::from_shape_vec((6, 2), rng.uniform(10.0, 90.0, 12)).unwrap();

    let secondaries = SecondaryInput::from([("secondary", xyz(&x0, &y0, &zu))]);
    let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, block_options(CoKrigingMethod::Lmc)).unwrap();
    let result = ck.interpolate(&grid).unwrap();
    let (model_id, ranges, coefficients) = lmc_matrices(&ck);
    let variables = [
        (x0.clone(), y0.clone(), z0.clone()),
        (x0.clone(), y0.clone(), standardize(&z0, &zu)),
    ];
    let expected = reference_block_cokriging(
        &variables,
        &grid,
        40.0,
        12,
        &lmc_cov(model_id, &ranges, &coefficients),
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "LMC K1");
}

/// (300, 300) has no source within the radius: the primary falls back to its
/// `max_neighbors` nearest and no secondary is in reach.
#[test]
fn test_icm_and_lmc_cokriging_use_nearest_primaries_beyond_radius() {
    let mut rng = NumpyRng::new(6);
    let n = 120;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 1006, 8, 25.0);
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
    let secondaries = SecondaryInput::from([("secondary", xyz(&x0, &y0, &zu))]);
    let far = array![[300.0, 300.0]];
    let primary = [(x0.clone(), y0.clone(), z0.clone())];

    for method in [CoKrigingMethod::Icm, CoKrigingMethod::Lmc] {
        let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, block_options(method)).unwrap();
        let expected = if method == CoKrigingMethod::Icm {
            let (model_id, range, nugget, sill) = icm_matrices(&ck);
            reference_block_cokriging(
                &primary,
                &far,
                f64::INFINITY,
                12,
                &icm_cov(model_id, range, &nugget, &sill),
            )
        } else {
            let (model_id, ranges, coefficients) = lmc_matrices(&ck);
            reference_block_cokriging(
                &primary,
                &far,
                f64::INFINITY,
                12,
                &lmc_cov(model_id, &ranges, &coefficients),
            )
        };
        assert_allclose(
            &ck.interpolate(&far).unwrap(),
            &expected,
            1e-6,
            1e-6,
            method.as_str(),
        );
    }
}

#[test]
fn test_lmc_cokriging_k2_matches_reference_with_correlated_secondaries() {
    let mut rng = NumpyRng::new(72);
    let n = 100;
    let x0 = rng.uniform(0.0, 100.0, n);
    let y0 = rng.uniform(0.0, 100.0, n);
    let latent = smooth_field(&x0, &y0, 73, 8, 25.0);
    let z0: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.05, n))
        .map(|(l, e)| l + e)
        .collect();
    let zu1: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.1, n))
        .map(|(l, e)| 1.2 * l + e)
        .collect();
    let zu2: Vec<f64> = latent
        .iter()
        .zip(rng.normal(0.0, 0.1, n))
        .map(|(l, e)| 0.8 * l + e)
        .collect();
    let grid = Array2::from_shape_vec((5, 2), rng.uniform(10.0, 90.0, 10)).unwrap();

    let secondaries = SecondaryInput::from(vec![xyz(&x0, &y0, &zu1), xyz(&x0, &y0, &zu2)]);
    let ck = CoKriging::from_data(&x0, &y0, &z0, &secondaries, block_options(CoKrigingMethod::Lmc)).unwrap();
    let result = ck.interpolate(&grid).unwrap();
    let (model_id, ranges, coefficients) = lmc_matrices(&ck);
    let variables = [
        (x0.clone(), y0.clone(), z0.clone()),
        (x0.clone(), y0.clone(), standardize(&z0, &zu1)),
        (x0.clone(), y0.clone(), standardize(&z0, &zu2)),
    ];
    let expected = reference_block_cokriging(
        &variables,
        &grid,
        40.0,
        12,
        &lmc_cov(model_id, &ranges, &coefficients),
    );
    assert_allclose(&result, &expected, 1e-6, 1e-6, "LMC K2");
    assert!((0..coefficients.shape()[0]).any(|s| coefficients[[s, 1, 2]] != 0.0));
}

#[test]
fn test_lmc_goulard_voltz_recovers_known_coregionalization() {
    let mut rng = NumpyRng::new(80);
    let n = 220;
    let x = rng.uniform(0.0, 100.0, n);
    let y = rng.uniform(0.0, 100.0, n);
    let nv = 2;
    let true_range = 25.0;
    let true_nugget_b = array![[0.3, 0.05], [0.05, 0.3]];
    let true_sill_b = array![[1.0, 0.7], [0.7, 1.0]];

    let mut spatial_cov = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            spatial_cov[[i, j]] = covariance(
                (x[i] - x[j]).hypot(y[i] - y[j]),
                MODEL_SPHERICAL,
                0.0,
                1.0,
                true_range,
            );
        }
        spatial_cov[[i, i]] += 1e-9;
    }
    let l_spatial = cholesky(&spatial_cov);

    let mut structure_field = |b: &Array2<f64>, spatially_correlated: bool| -> Array2<f64> {
        let c = cholesky(&(b + &(Array2::<f64>::eye(nv) * 1e-12)));
        let mut w = Array2::from_shape_vec((nv, n), rng.standard_normal_vec(nv * n)).unwrap();
        if spatially_correlated {
            w = w.dot(&l_spatial.t());
        }
        c.dot(&w)
    };
    let z_nugget = structure_field(&true_nugget_b, false);
    let z_continuous = structure_field(&true_sill_b, true);
    let z = &z_nugget + &z_continuous;
    let (z0, zu) = (z.row(0).to_vec(), z.row(1).to_vec());

    let secondaries = SecondaryInput::from([("secondary", xyz(&x, &y, &zu))]);
    let ck = CoKriging::from_data(
        &x,
        &y,
        &z0,
        &secondaries,
        CoKrigingOptions {
            method: CoKrigingMethod::Lmc,
            n_structures: 2,
            max_neighbors: 30,
            min_neighbors: 3,
            ..Default::default()
        },
    )
    .unwrap();
    let (_, ranges, coefficients) = lmc_matrices(&ck);
    let fitted_nugget = coefficients.index_axis(ndarray::Axis(0), 0).to_owned();
    let fitted_sill = coefficients.index_axis(ndarray::Axis(0), 1).to_owned();
    assert_allclose(&fitted_sill, &true_sill_b, 0.35, 0.15, "fitted sill");
    assert!(
        (ranges[0] - true_range).abs() <= 0.5 * true_range,
        "range {}",
        ranges[0]
    );
    assert!(eigvalsh(&fitted_nugget).iter().all(|v| *v >= -1e-6));
    assert!(eigvalsh(&fitted_sill).iter().all(|v| *v >= -1e-6));
}

/// The method is an enum, so an unknown name fails when the string is parsed
/// (`ValueError`).
#[test]
fn test_cokriging_rejects_invalid_method() {
    let error = "bogus".parse::<CoKrigingMethod>().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValueError);
    assert!(error.message().contains("method"));
    assert_eq!(
        error.message(),
        "method must be one of ('icm', 'lmc', 'collocated')"
    );
}

#[test]
fn cokriging_validation_messages() {
    let _guard = common::config_guard();
    let message = |options: CoKrigingOptions| CoKriging::new(options).unwrap_err().to_string();
    assert_eq!(
        message(CoKrigingOptions {
            n_structures: 1,
            ..Default::default()
        }),
        "n_structures must be >= 2 (nugget + at least one continuous structure)"
    );
    assert_eq!(
        message(CoKrigingOptions {
            model_values: Some(array![1.0, 0.0, 1.0, 10.0, 0.0]),
            ..Default::default()
        }),
        "model_values must contain 5 primary values plus at least 1 rho"
    );
    assert_eq!(
        message(CoKrigingOptions {
            radius: Some(f64::NAN),
            ..Default::default()
        }),
        "radius must be finite and greater than zero"
    );

    let x = [0.0, 10.0, 0.0, 10.0];
    let z = [1.0, 2.0, 3.0, 4.0];
    let mut model = CoKriging::new(CoKrigingOptions::default()).unwrap();
    let constant = SecondaryInput::from([("flat", xyz(&x, &x, &[5.0, 5.0, 5.0, 5.0]))]);
    assert_eq!(
        model.fit(&x, &x, &z, &constant).unwrap_err().message(),
        "secondary 'flat' has zero variance; cannot standardize"
    );
    let empty = SecondaryInput::from(Vec::<(Array1<f64>, Array1<f64>, Array1<f64>)>::new());
    assert_eq!(
        model.fit(&x, &x, &z, &empty).unwrap_err().message(),
        "cokriging requires at least one secondary variable"
    );
    let no_finite = SecondaryInput::from([("nan", xyz(&[f64::NAN], &[0.0], &[1.0]))]);
    assert_eq!(
        model.fit(&x, &x, &z, &no_finite).unwrap_err().message(),
        "secondary 'nan' has no finite points"
    );
    let flat_primary = SecondaryInput::from([("s", xyz(&x, &x, &z))]);
    assert_eq!(
        model
            .fit(&x, &x, &[1.0, 1.0, 1.0, 1.0], &flat_primary)
            .unwrap_err()
            .message(),
        "primary has zero variance; cannot standardize secondaries"
    );
    apbase::config.set("max_secondary_variables_ceiling", 1).unwrap();
    let two = SecondaryInput::from(vec![xyz(&x, &x, &z), xyz(&x, &x, &z)]);
    let capped = model.fit(&x, &x, &z, &two).unwrap_err().to_string();
    apbase::config._clear();
    assert_eq!(capped, "number of secondaries must be <= 1");
    assert_eq!(
        model.interpolate(&array![[0.0, 0.0]]).unwrap_err().message(),
        "call fit(x0, y0, z0, secondaries) before interpolate(targets)"
    );
}
