mod common;

use std::collections::{HashMap, HashSet};

use apbase::cross_validate::{CrossValidateOptions, cross_validate};
use apbase::mapping::{Bounds, Map, MapOptions, MapResult, create_map};
use apbase::variogram::{Variogram, VariogramOptions};
use common::{NumpyRng, assert_allclose};
use geos::{Geom, Geometry};

fn synthetic_source(seed: u64, n: usize, extent: f64, offset: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = NumpyRng::new(seed);
    let x: Vec<f64> = rng.uniform(0.0, extent, n).iter().map(|v| offset + v).collect();
    let y: Vec<f64> = rng.uniform(0.0, extent, n).iter().map(|v| offset + v).collect();
    let noise = rng.normal(0.0, 0.05, n);
    let z = (0..n)
        .map(|i| ((x[i] - offset) / 12.0).sin() + ((y[i] - offset) / 12.0).cos() + noise[i])
        .collect();
    (x, y, z)
}

fn box_polygon(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Geometry {
    Geometry::new_from_wkt(&format!(
        "POLYGON (({maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}, {maxx} {miny}))"
    ))
    .unwrap()
}

#[test]
fn test_create_map_returns_compact_point_list_with_expected_shapes() {
    let (x, y, z) = synthetic_source(0, 300, 100.0, 0.0);
    let result: MapResult = create_map(&x, &y, &z, MapOptions::new(8.0)).unwrap();
    assert!(result.method == "idw" || result.method == "kriging");
    assert!(result.x.len() == result.y.len() && result.y.len() == result.z.len());
    assert!(!result.x.is_empty());
    assert!(result.n_source_points <= x.len() as i64);
    assert!(result.radius > 0.0);
    assert_eq!(result.resolution, 8.0);
}

#[test]
fn test_create_map_requires_resolution() {
    let options = MapOptions::new(8.0);
    assert_eq!(options.resolution, 8.0);
    assert!(options.filter && options.bounds.is_none() && options.seed.is_none());
    assert_eq!(
        (
            options.power,
            options.max_neighbors,
            options.min_neighbors,
            options.max_cv_points
        ),
        (2.0, 40, 3, 500)
    );
}

#[test]
fn test_map_requires_x_y_z_together() {
    let (x, y, z) = synthetic_source(1, 10, 100.0, 0.0);
    let lazy = Map::new(MapOptions::new(10.0));
    assert!(lazy.result.is_none());
    let error = Map::from_data(&x, &y, &z[..5], MapOptions::new(10.0)).unwrap_err();
    assert_eq!(error.message(), "x, y, and z must have the same size");
}

#[test]
fn test_map_filter_toggle_changes_source_points_and_statistics() {
    let (x, y, mut z) = synthetic_source(2, 200, 100.0, 0.0);
    z[0] = 1000.0;
    let filtered = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            filter: true,
            ..MapOptions::new(10.0)
        },
    )
    .unwrap();
    let unfiltered = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            filter: false,
            ..MapOptions::new(10.0)
        },
    )
    .unwrap();
    assert!(filtered.filter_statistics.is_some());
    assert!(unfiltered.filter_statistics.is_none());
    assert_eq!(unfiltered.n_source_points, x.len() as i64);
    assert!(filtered.n_source_points <= x.len() as i64);
}

#[test]
fn test_map_selects_method_matching_lower_cross_validation_rmse() {
    let (x, y, z) = synthetic_source(3, 250, 300.0, 0.0);
    let result = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            filter: false,
            ..MapOptions::new(15.0)
        },
    )
    .unwrap();

    let variogram = Variogram::from_data(
        &x,
        &y,
        &z,
        VariogramOptions {
            n_lags: 50,
            max_pairs: 100_000,
            max_distance: 0.0,
        },
    )
    .unwrap();
    let radius = variogram.model_params().unwrap().search_radius(x.len());
    let cv = cross_validate(
        &x,
        &y,
        &z,
        CrossValidateOptions {
            radius: Some(radius),
            power: 2.0,
            max_neighbors: 40,
            min_neighbors: 3,
            model_values: Some(variogram.model_values().unwrap().clone()),
            ..Default::default()
        },
    )
    .unwrap();
    let expected_method = if cv.idw.rmse <= cv.kriging.rmse {
        "idw"
    } else {
        "kriging"
    };
    assert_eq!(result.method, expected_method);
    assert_eq!(result.cross_validation.idw.rmse, cv.idw.rmse);
    assert_eq!(result.cross_validation.kriging.rmse, cv.kriging.rmse);
}

#[test]
fn test_map_bounds_accepts_geometry_wkt_and_wkb_equally() {
    let origin = 10_000.0;
    let (x, y, z) = synthetic_source(4, 150, 100.0, origin);
    let boundary = box_polygon(origin + 10.0, origin + 10.0, origin + 60.0, origin + 60.0);
    let run = |bounds: Bounds| {
        create_map(
            &x,
            &y,
            &z,
            MapOptions {
                bounds: Some(bounds),
                filter: false,
                ..MapOptions::new(8.0)
            },
        )
        .unwrap()
    };
    let from_geometry = run(Clone::clone(&boundary).into());
    let from_wkt = run(boundary.to_wkt().unwrap().into());
    let from_wkb = run(boundary.to_wkb().unwrap().to_vec().into());
    assert_eq!(from_geometry.x, from_wkt.x);
    assert_eq!(from_geometry.x, from_wkb.x);
    assert_allclose(&from_geometry.z, &from_wkt.z, 1e-5, 1e-8, "wkt");
    assert_allclose(&from_geometry.z, &from_wkb.z, 1e-5, 1e-8, "wkb");
}

#[test]
fn test_map_output_coordinates_preserve_input_metric_not_rebased() {
    let (offset, extent) = (500_000.0, 100.0);
    let boundary = box_polygon(offset, offset, offset + extent, offset + extent);
    let (x, y, z) = synthetic_source(5, 400, extent, offset);
    let result = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            bounds: Some(boundary.into()),
            filter: false,
            ..MapOptions::new(10.0)
        },
    )
    .unwrap();
    let min =
        |values: &mut dyn Iterator<Item = f64>| values.filter(|v| !v.is_nan()).fold(f64::INFINITY, f64::min);
    let max = |values: &mut dyn Iterator<Item = f64>| {
        values.filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, f64::max)
    };
    assert!(min(&mut result.x.iter().copied()) >= offset - 10.0);
    assert!(min(&mut result.y.iter().copied()) >= offset - 10.0);
    assert!(max(&mut result.x.iter().copied()) <= offset + extent + 10.0);
    assert!(max(&mut result.y.iter().copied()) <= offset + extent + 10.0);

    let (x_raster, y_raster, _) = result.to_raster().unwrap();
    assert!(min(&mut x_raster.iter().copied()) >= offset - 10.0);
    assert!(min(&mut y_raster.iter().copied()) >= offset - 10.0);
}

fn key(value: f64) -> i64 {
    (value * 1e6).round_ties_even() as i64
}

#[test]
fn test_map_to_raster_reconstructs_points_with_matching_coordinates_and_values() {
    let origin = 10_000.0;
    let boundary = box_polygon(origin, origin, origin + 20.0, origin + 20.0);
    let (x, y, z) = synthetic_source(6, 300, 30.0, origin - 5.0);
    let result = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            bounds: Some(boundary.into()),
            filter: false,
            ..MapOptions::new(5.0)
        },
    )
    .unwrap();
    let (x_raster, y_raster, z_raster) = result.to_raster().unwrap();
    assert_eq!(x_raster.shape(), y_raster.shape());
    assert_eq!(x_raster.shape(), z_raster.shape());

    let expected_points: HashSet<(i64, i64)> = result
        .x
        .iter()
        .zip(&result.y)
        .map(|(a, b)| (key(*a), key(*b)))
        .collect();
    let actual_points: HashSet<(i64, i64)> = x_raster
        .iter()
        .zip(y_raster.iter())
        .map(|(a, b)| (key(*a), key(*b)))
        .collect();
    assert_eq!(expected_points, actual_points);

    let point_to_z: HashMap<(i64, i64), f64> = result
        .x
        .iter()
        .zip(&result.y)
        .zip(&result.z)
        .map(|((a, b), c)| ((key(*a), key(*b)), *c))
        .collect();
    for ((xr, yr), zr) in x_raster.iter().zip(y_raster.iter()).zip(z_raster.iter()) {
        let expected = point_to_z[&(key(*xr), key(*yr))];
        if expected.is_nan() {
            assert!(zr.is_nan());
        } else {
            assert_eq!(*zr, expected);
        }
    }

    let cube = result.to_array3d().unwrap();
    assert_eq!(cube.shape(), &[x_raster.nrows(), x_raster.ncols(), 3]);
    assert_allclose(
        &cube.index_axis(ndarray::Axis(2), 0).to_owned(),
        &x_raster,
        1e-5,
        1e-8,
        "x layer",
    );
    assert_allclose(
        &cube.index_axis(ndarray::Axis(2), 1).to_owned(),
        &y_raster,
        1e-5,
        1e-8,
        "y layer",
    );
    assert_allclose(
        &cube.index_axis(ndarray::Axis(2), 2).to_owned(),
        &z_raster,
        1e-5,
        1e-8,
        "z layer",
    );
}

#[test]
fn test_map_eager_lazy_and_functional_forms_match() {
    let (x, y, z) = synthetic_source(7, 150, 100.0, 0.0);
    let options = MapOptions {
        filter: false,
        ..MapOptions::new(6.0)
    };
    let eager = Map::from_data(&x, &y, &z, options.clone()).unwrap();
    let lazy = Map::new(options.clone()).fit_generate(&x, &y, &z).unwrap();
    let via_call = Map::new(options.clone()).call(&x, &y, &z).unwrap();
    let via_function = create_map(&x, &y, &z, options).unwrap();

    let eager_result = eager.result.as_ref().expect("eager form stores the result");
    assert_eq!(eager_result.x, lazy.x);
    assert_eq!(eager_result.x, via_call.x);
    assert_eq!(eager_result.x, via_function.x);
    assert_allclose(&eager_result.z, &lazy.z, 1e-5, 1e-8, "lazy");
    assert_allclose(&eager_result.z, &via_call.z, 1e-5, 1e-8, "call");
    assert_allclose(&eager_result.z, &via_function.z, 1e-5, 1e-8, "function");
    assert!(
        eager_result.method == lazy.method
            && lazy.method == via_call.method
            && via_call.method == via_function.method
    );
}

#[test]
fn invalid_bounds_and_raster_errors() {
    let (x, y, z) = synthetic_source(8, 120, 100.0, 1000.0);
    let error = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            bounds: Some("NOT A WKT".into()),
            ..MapOptions::new(10.0)
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), apbase::common::ErrorKind::GEOSException);

    let mut result = create_map(
        &x,
        &y,
        &z,
        MapOptions {
            filter: false,
            ..MapOptions::new(10.0)
        },
    )
    .unwrap();
    result.x[0] += 0.123;
    assert_eq!(
        result.to_raster().unwrap_err().message(),
        "map points do not form a regular grid; cannot build a raster"
    );
}
