//! Real study-area boundaries of the reference datasets (`tests/data/*_area.wkt`)
//! used as `bounds`, compared with the concave hull inferred when none is given.

mod common;

use std::collections::HashSet;

use apbase::grid::{Grid, GridOptions};
use apbase::mapping::{Map, MapOptions, MapResult};
use common::{data_path, read_csv};
use geos::{CoordSeq, Geom, Geometry};

/// Jura coordinates are in km, inside the lon/lat range that `Map` auto-detects
/// as geographic; scaling to meters keeps them projected.
const JURA_KM_TO_M: f64 = 1000.0;

fn read_wkt(name: &str) -> String {
    std::fs::read_to_string(data_path(name)).expect("readable wkt")
}

fn boundary(name: &str) -> Geometry {
    Geometry::new_from_wkt(&read_wkt(name)).expect("valid wkt")
}

fn scaled(geometry: &Geometry, factor: f64) -> Geometry {
    geometry
        .transform_xy(|x, y| Ok::<_, geos::Error>((x * factor, y * factor)))
        .expect("scalable geometry")
}

fn point(x: f64, y: f64) -> Geometry {
    Geometry::create_point(CoordSeq::new_from_vec(&[[x, y]]).unwrap()).unwrap()
}

fn iou(a: &Geometry, b: &Geometry) -> f64 {
    a.intersection(b).unwrap().area().unwrap() / a.union(b).unwrap().area().unwrap()
}

fn meuse() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let meuse = read_csv("meuse.csv");
    let log_zinc = meuse["zinc"].iter().map(|v| v.ln()).collect();
    (meuse["x"].clone(), meuse["y"].clone(), log_zinc)
}

/// Training and validation points together, in meters.
fn jura_cadmium() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (mut x, mut y, mut z) = (Vec::new(), Vec::new(), Vec::new());
    for name in ["jura_prediction.csv", "jura_validation.csv"] {
        let table = read_csv(name);
        x.extend(table["Xloc"].iter().map(|v| v * JURA_KM_TO_M));
        y.extend(table["Yloc"].iter().map(|v| v * JURA_KM_TO_M));
        z.extend(&table["Cd"]);
    }
    (x, y, z)
}

fn walker_v() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let walker = read_csv("walker_sample.csv");
    (walker["x"].clone(), walker["y"].clone(), walker["v"].clone())
}

fn map(x: &[f64], y: &[f64], z: &[f64], resolution: f64, bounds: Option<String>) -> MapResult {
    let options = MapOptions {
        bounds: bounds.map(Into::into),
        seed: Some(0),
        ..MapOptions::new(resolution)
    };
    let result = Map::from_data(x, y, z, options).unwrap().result.unwrap();
    (*result).clone()
}

fn n_finite(result: &MapResult) -> usize {
    result.z.iter().filter(|v| v.is_finite()).count()
}

/// Every map point lies within `resolution / 2` of `real` (the grid buffer).
#[track_caller]
fn assert_points_hug_boundary(result: &MapResult, real: &Geometry, resolution: f64) {
    for (&x, &y) in result.x.iter().zip(&result.y) {
        let distance = real.distance(&point(x, y)).unwrap();
        assert!(
            distance <= resolution * 0.5 + 1e-9,
            "({x}, {y}) is {distance} away"
        );
    }
}

#[test]
fn test_reference_boundaries_are_valid_polygons_covering_every_sample() {
    let (mx, my, _) = meuse();
    let (jx, jy, _) = jura_cadmium();
    let (wx, wy, _) = walker_v();
    let cases = [
        ("meuse_area.wkt", 1.0, mx, my),
        ("jura_area.wkt", JURA_KM_TO_M, jx, jy),
        ("walker_area.wkt", 1.0, wx, wy),
    ];
    for (name, factor, x, y) in cases {
        let real = scaled(&boundary(name), factor);
        assert_eq!(real.get_type().unwrap(), "Polygon", "{name}");
        assert!(real.is_valid().unwrap(), "{name}");
        for (&px, &py) in x.iter().zip(&y) {
            assert!(
                real.covers(&point(px, py)).unwrap(),
                "{name}: ({px}, {py}) outside"
            );
        }
    }
}

/// `meuse.grid` (the official 40 m prediction grid, cell centers at 20 m
/// offsets) is contained in the apbase grid built at 20 m inside `meuse.area`.
#[test]
fn test_meuse_official_grid_is_subset_of_grid_inside_meuse_area() {
    let grid = Grid::from_wkt(
        &read_wkt("meuse_area.wkt"),
        GridOptions {
            resolution: 20.0,
            ..GridOptions::default()
        },
    )
    .unwrap();
    let points = grid.points().unwrap();
    let generated: HashSet<(i64, i64)> = points
        .rows()
        .into_iter()
        .map(|row| (row[0].round() as i64, row[1].round() as i64))
        .collect();
    let official = read_csv("meuse_grid.csv");
    assert_eq!(official["x"].len(), 3103);
    for (&x, &y) in official["x"].iter().zip(&official["y"]) {
        assert!(generated.contains(&(x as i64, y as i64)), "({x}, {y}) missing");
    }
}

/// How closely the inferred concave hull matches the real boundary: both are
/// compared as fitted (buffered by `resolution / 2`) boundaries.
#[test]
fn test_inferred_hull_overlaps_real_boundary() {
    let (mx, my, _) = meuse();
    let (jx, jy, _) = jura_cadmium();
    let (wx, wy, _) = walker_v();
    let cases = [
        ("meuse_area.wkt", 1.0, 40.0, mx, my, 0.6),
        ("jura_area.wkt", JURA_KM_TO_M, 50.0, jx, jy, 0.7),
        ("walker_area.wkt", 1.0, 1.0, wx, wy, 0.5),
    ];
    for (name, factor, resolution, x, y, min_iou) in cases {
        let options = || GridOptions {
            resolution,
            ..GridOptions::default()
        };
        let real_wkt = scaled(&boundary(name), factor).to_wkt().unwrap();
        let real = Grid::from_wkt(&real_wkt, options()).unwrap();
        let hull = Grid::from_data_xy(&x, &y, options()).unwrap();
        let (real_area, hull_area) = (
            real.fitted_boundary().unwrap().area().unwrap(),
            hull.fitted_boundary().unwrap().area().unwrap(),
        );
        let overlap = iou(real.fitted_boundary().unwrap(), hull.fitted_boundary().unwrap());
        eprintln!(
            "{name}: real area {real_area:.0}, hull area {hull_area:.0} ({:.3}), IoU {overlap:.4}, points real {} hull {}",
            hull_area / real_area,
            real.points().unwrap().nrows(),
            hull.points().unwrap().nrows(),
        );
        assert!(overlap >= min_iou, "{name}: IoU {overlap}");
    }
}

#[test]
fn test_meuse_map_with_real_boundary_stays_inside_it() {
    let (x, y, z) = meuse();
    let wkt = read_wkt("meuse_area.wkt");
    let real = map(&x, &y, &z, 40.0, Some(wkt.clone()));
    let hull = map(&x, &y, &z, 40.0, None);
    assert_points_hug_boundary(&real, &boundary("meuse_area.wkt"), 40.0);
    assert_eq!(real.method, hull.method);
    assert!(real.z.len() > hull.z.len());
    let (_, _, raster) = real.to_raster().unwrap();
    assert_eq!(n_finite(&real), real.z.len());
    assert_eq!(raster.iter().filter(|v| v.is_finite()).count(), real.z.len());
    eprintln!(
        "meuse map ({}): real {} points ({} finite), hull {} points ({} finite), radius {:.0}",
        real.method,
        real.z.len(),
        n_finite(&real),
        hull.z.len(),
        n_finite(&hull),
        real.radius
    );
}

#[test]
fn test_jura_map_with_real_boundary_stays_inside_it() {
    let (x, y, z) = jura_cadmium();
    let real_boundary = scaled(&boundary("jura_area.wkt"), JURA_KM_TO_M);
    let real = map(&x, &y, &z, 50.0, Some(real_boundary.to_wkt().unwrap()));
    let hull = map(&x, &y, &z, 50.0, None);
    assert_points_hug_boundary(&real, &real_boundary, 50.0);
    assert!(real.coordinate_transform.is_none());
    assert!(real.z.len() > hull.z.len());
    assert_eq!(n_finite(&real), real.z.len());
    eprintln!(
        "jura map ({}): real {} points ({} finite), hull {} points ({} finite), radius {:.0}",
        real.method,
        real.z.len(),
        n_finite(&real),
        hull.z.len(),
        n_finite(&hull),
        real.radius
    );
}

/// The real boundary places a target on every exhaustive-grid cell (the hull
/// only on part of them); finite estimates are scored against the true `v`.
#[test]
fn test_walker_map_with_real_boundary_covers_exhaustive_grid() {
    let (x, y, z) = walker_v();
    let real = map(&x, &y, &z, 1.0, Some(read_wkt("walker_area.wkt")));
    let hull = map(&x, &y, &z, 1.0, None);
    let exhaustive = read_csv("walker_exhaustive.csv");

    let score = |result: &MapResult| {
        let lookup: std::collections::HashMap<(i64, i64), f64> = result
            .x
            .iter()
            .zip(&result.y)
            .zip(&result.z)
            .map(|((&px, &py), &pz)| ((px.round() as i64, py.round() as i64), pz))
            .collect();
        let (mut covered, mut estimated, mut squared) = (0usize, 0usize, 0.0);
        for ((&ex, &ey), &ev) in exhaustive["x"].iter().zip(&exhaustive["y"]).zip(&exhaustive["v"]) {
            if let Some(estimate) = lookup.get(&(ex as i64, ey as i64)) {
                covered += 1;
                if estimate.is_finite() {
                    estimated += 1;
                    squared += (estimate - ev).powi(2);
                }
            }
        }
        (covered, estimated, (squared / estimated as f64).sqrt())
    };
    let (real_covered, real_estimated, real_rmse) = score(&real);
    let (hull_covered, hull_estimated, hull_rmse) = score(&hull);
    eprintln!(
        "walker map ({}, radius {:.1}): real covers {real_covered} ({real_estimated} finite, rmse {real_rmse:.2}); \
         hull covers {hull_covered} ({hull_estimated} finite, rmse {hull_rmse:.2})",
        real.method, real.radius
    );
    assert_eq!(real_covered, exhaustive["x"].len());
    assert!(hull_covered < real_covered);
    assert_eq!(real_estimated, real_covered);
}
