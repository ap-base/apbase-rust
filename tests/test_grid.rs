mod common;

use apbase::common::ErrorKind;
use apbase::grid::{CreatePointGridOptions, Grid, GridOptions, create_point_grid};
use common::NumpyRng;
use geos::{Geom, Geometry};
use ndarray::{Array1, Array2};

fn box_polygon(minx: f64, miny: f64, maxx: f64, maxy: f64) -> Geometry {
    Geometry::new_from_wkt(&format!(
        "POLYGON (({maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}, {maxx} {miny}))"
    ))
    .unwrap()
}

fn boundary_grid(boundary: Geometry, resolution: f64) -> Grid {
    let mut grid = Grid::new(GridOptions {
        boundary: Some(boundary),
        resolution,
        ..Default::default()
    })
    .unwrap();
    grid.fit(None, None).unwrap();
    grid
}

fn sorted_rows(points: &Array2<f64>) -> Vec<(f64, f64)> {
    let mut rows: Vec<(f64, f64)> = points.rows().into_iter().map(|row| (row[0], row[1])).collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

#[test]
fn test_grid_matches_regular_mesh_inside_box_boundary() {
    let _guard = common::config_guard();
    let grid = boundary_grid(box_polygon(0.0, 0.0, 10.0, 10.0), 5.0);
    let mut expected = vec![];
    for y in [0.0, 5.0, 10.0] {
        for x in [0.0, 5.0, 10.0] {
            expected.push((x, y));
        }
    }
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(sorted_rows(grid.points().unwrap()), expected);
    assert!(
        grid.points().unwrap().t().is_standard_layout(),
        "points are F_CONTIGUOUS"
    );
}

#[test]
fn test_grid_fit_ignores_xy_when_boundary_given() {
    let _guard = common::config_guard();
    let no_xy = boundary_grid(box_polygon(0.0, 0.0, 10.0, 10.0), 5.0)
        .points()
        .unwrap()
        .clone();
    let mut with_xy = Grid::new(GridOptions {
        boundary: Some(box_polygon(0.0, 0.0, 10.0, 10.0)),
        resolution: 5.0,
        ..Default::default()
    })
    .unwrap();
    with_xy.fit(Some(&[0.0]), Some(&[0.0])).unwrap();
    assert_eq!(&no_xy, with_xy.points().unwrap());
}

#[test]
fn test_grid_fit_requires_xy_when_boundary_missing() {
    let _guard = common::config_guard();
    let mut grid = Grid::new(GridOptions {
        resolution: 5.0,
        ..Default::default()
    })
    .unwrap();
    let error = grid.fit(None, None).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValueError);
    assert!(error.message().contains("x and y"));
}

#[test]
fn test_grid_point_count_is_independent_of_chunk_size() {
    let _guard = common::config_guard();
    let mut rng = NumpyRng::new(0);
    let x = rng.uniform(0.0, 1000.0, 200);
    let y = rng.uniform(0.0, 1000.0, 200);
    let small = Grid::from_data_xy(
        &x,
        &y,
        GridOptions {
            resolution: 10.0,
            chunk_size: 50,
            ..Default::default()
        },
    )
    .unwrap();
    let large = Grid::from_data_xy(
        &x,
        &y,
        GridOptions {
            resolution: 10.0,
            chunk_size: 1_000_000,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(small.points().unwrap().nrows(), large.points().unwrap().nrows());
    assert_eq!(small.points().unwrap(), large.points().unwrap());
}

#[test]
fn test_grid_infers_concave_hull_boundary_when_none_given() {
    let _guard = common::config_guard();
    let x = [0.0, 100.0, 0.0, 100.0];
    let y = [0.0, 0.0, 100.0, 100.0];
    let grid = Grid::from_data_xy(
        &x,
        &y,
        GridOptions {
            resolution: 10.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(grid.points().unwrap().nrows() > 0);
    let boundary = grid.fitted_boundary().unwrap();
    let prepared = boundary.to_prepared_geom().unwrap();
    assert!(
        x.iter()
            .zip(&y)
            .all(|(px, py)| prepared.intersects_xy(*px, *py).unwrap())
    );
}

#[test]
fn test_create_point_grid_matches_grid_class() {
    let _guard = common::config_guard();
    let mut rng = NumpyRng::new(1);
    let x = Array1::from(rng.uniform(0.0, 500.0, 100));
    let y = Array1::from(rng.uniform(0.0, 500.0, 100));
    let from_function = create_point_grid(CreatePointGridOptions {
        x: Some(&x),
        y: Some(&y),
        ..CreatePointGridOptions::new(25.0)
    })
    .unwrap();
    let from_class = Grid::from_data_xy(
        &x,
        &y,
        GridOptions {
            resolution: 25.0,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(&from_function, from_class.points().unwrap());
}

#[test]
fn test_create_point_grid_from_boundary_without_xy() {
    let _guard = common::config_guard();
    let from_function = create_point_grid(CreatePointGridOptions {
        boundary: Some(box_polygon(0.0, 0.0, 10.0, 10.0)),
        ..CreatePointGridOptions::new(5.0)
    })
    .unwrap();
    let from_class = boundary_grid(box_polygon(0.0, 0.0, 10.0, 10.0), 5.0);
    assert_eq!(&from_function, from_class.points().unwrap());
}

#[test]
fn test_grid_from_data_xy_matches_fit() {
    let _guard = common::config_guard();
    let mut rng = NumpyRng::new(2);
    let x = rng.uniform(0.0, 500.0, 100);
    let y = rng.uniform(0.0, 500.0, 100);
    let from_classmethod = Grid::from_data_xy(
        &x,
        &y,
        GridOptions {
            resolution: 25.0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut from_fit = Grid::new(GridOptions {
        resolution: 25.0,
        ..Default::default()
    })
    .unwrap();
    from_fit.fit(Some(&x), Some(&y)).unwrap();
    assert_eq!(from_classmethod.points().unwrap(), from_fit.points().unwrap());
}

#[test]
fn test_grid_from_wkt_matches_boundary_grid() {
    let _guard = common::config_guard();
    let boundary = box_polygon(0.0, 0.0, 10.0, 10.0);
    let from_wkt = Grid::from_wkt(
        &boundary.to_wkt().unwrap(),
        GridOptions {
            resolution: 5.0,
            ..Default::default()
        },
    )
    .unwrap();
    let from_boundary = boundary_grid(boundary, 5.0);
    assert_eq!(from_wkt.points().unwrap(), from_boundary.points().unwrap());
}

#[test]
fn test_grid_from_wkb_matches_boundary_grid() {
    let _guard = common::config_guard();
    let boundary = box_polygon(0.0, 0.0, 10.0, 10.0);
    let wkb = boundary.to_wkb().unwrap();
    let from_wkb = Grid::from_wkb(
        &wkb,
        GridOptions {
            resolution: 5.0,
            ..Default::default()
        },
    )
    .unwrap();
    let from_boundary = boundary_grid(boundary, 5.0);
    assert_eq!(from_wkb.points().unwrap(), from_boundary.points().unwrap());
}

#[test]
fn parallel_containment_matches_serial_order() {
    let _guard = common::config_guard();
    let mut rng = NumpyRng::new(5);
    let x = rng.uniform(0.0, 3000.0, 500);
    let y = rng.uniform(0.0, 3000.0, 500);
    let options = GridOptions {
        resolution: 15.0,
        chunk_size: 5000,
        ..Default::default()
    };
    let serial = Grid::from_data_xy(&x, &y, options.clone()).unwrap();
    apbase::config.set("n_threads", 4).unwrap();
    let parallel = Grid::from_data_xy(&x, &y, options).unwrap();
    apbase::config._clear();
    assert_eq!(serial.points().unwrap(), parallel.points().unwrap());
}

#[test]
fn candidate_ceiling_and_config_errors() {
    let _guard = common::config_guard();
    apbase::config.set("grid_max_points_ceiling", 10).unwrap();
    let mut grid = Grid::new(GridOptions {
        boundary: Some(box_polygon(0.0, 0.0, 10.0, 10.0)),
        resolution: 1.0,
        ..Default::default()
    })
    .unwrap();
    let error = grid.fit(None, None).unwrap_err();
    apbase::config._clear();
    assert_eq!(
        error.message(),
        "grid would generate 169 candidate points (13 x 13) before boundary filtering, above the 10 ceiling \
         (apbase.config[\"grid_max_points_ceiling\"]) -- check that `resolution` and `boundary` share the same \
         unit/CRS; a common cause is geographic input auto-detected and reprojected to UTM meters while \
         `resolution` was sized for a much coarser local unit"
    );

    let message = |options: GridOptions| Grid::new(options).unwrap_err().to_string();
    assert_eq!(
        message(GridOptions {
            resolution: 0.0,
            ..Default::default()
        }),
        "resolution must be finite and greater than zero"
    );
    assert_eq!(
        message(GridOptions {
            chunk_size: 0,
            ..Default::default()
        }),
        "chunk_size must be greater than zero"
    );
    assert_eq!(
        message(GridOptions {
            hull_ratio: 1.5,
            ..Default::default()
        }),
        "hull_ratio must be between 0 and 1"
    );
}

#[test]
fn empty_boundary_and_unfitted_access_fail() {
    let _guard = common::config_guard();
    let mut grid = Grid::new(GridOptions {
        boundary: Some(Geometry::new_from_wkt("POLYGON EMPTY").unwrap()),
        resolution: 5.0,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        grid.fit(None, None).unwrap_err().message(),
        "the resulting boundary is empty"
    );
    let unfitted = Grid::new(GridOptions::default()).unwrap();
    assert_eq!(
        unfitted.points().unwrap_err().message(),
        "call fit() before accessing the grid"
    );
    let error = Grid::from_wkt("NOT WKT", GridOptions::default()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::GEOSException);
}
