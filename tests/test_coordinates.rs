//! Coordinate detection and geographic/metric conversions (`apbase::common::coordinates`).

mod common;

use apbase::common::ErrorKind;
use apbase::common::coordinates::{
    CoordinateSystem, GeographicMode, prepare_metric_xy, reproject_geographic_xy,
};
use apbase::config::config;

fn field() -> ([f64; 4], [f64; 4]) {
    ([-47.0, -47.001, -47.0, -47.001], [-15.0, -15.0, -15.001, -15.001])
}

fn regional() -> ([f64; 2], [f64; 2]) {
    ([-46.6, -38.5], [-23.5, -12.9])
}

#[test]
fn test_prepare_metric_xy_auto_picks_local_below_ceiling() {
    let _guard = common::config_guard();
    let (lon, lat) = field();
    let (_, _, _, _, coordinate_system, epsg) = prepare_metric_xy(&lon, &lat, &lon, &lat, None).unwrap();
    assert_eq!(coordinate_system, "geographic");
    assert_eq!(epsg, None);
}

#[test]
fn test_prepare_metric_xy_auto_picks_utm_past_ceiling() {
    let _guard = common::config_guard();
    let (lon, lat) = regional();
    let (_, _, _, _, coordinate_system, epsg) = prepare_metric_xy(&lon, &lat, &lon, &lat, None).unwrap();
    assert_eq!(coordinate_system, CoordinateSystem::Geographic);
    assert!(epsg.is_some());
}

#[test]
fn test_reproject_geographic_xy_auto_detects_geographic_mode() {
    let _guard = common::config_guard();
    let lon = [-47.0, -47.001];
    let lat = [-15.0, -15.0];
    let (x, y) = reproject_geographic_xy(&lon, &lat, None).unwrap();
    assert_eq!(x.len(), lon.len());
    assert_eq!(y.len(), lat.len());
}

#[test]
fn test_prepare_metric_xy_local_mode_accepts_field_scale_extent() {
    let _guard = common::config_guard();
    let (lon, lat) = field();
    let (x, _, _, _, coordinate_system, epsg) =
        prepare_metric_xy(&lon, &lat, &lon, &lat, Some(GeographicMode::Local)).unwrap();
    assert_eq!(coordinate_system, "geographic");
    assert_eq!(epsg, None);
    assert_eq!(x.len(), lon.len());
}

#[test]
fn test_prepare_metric_xy_local_mode_rejects_regional_extent() {
    let _guard = common::config_guard();
    let (lon, lat) = regional();
    let error = prepare_metric_xy(&lon, &lat, &lon, &lat, Some(GeographicMode::Local)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValidationError);
    assert!(error.message().contains("local_mode_extent_km_ceiling"));
}

#[test]
fn test_prepare_metric_xy_utm_mode_unaffected_by_extent() {
    let _guard = common::config_guard();
    let (lon, lat) = regional();
    let (_, _, _, _, coordinate_system, epsg) =
        prepare_metric_xy(&lon, &lat, &lon, &lat, Some(GeographicMode::Utm)).unwrap();
    assert_eq!(coordinate_system, "geographic");
    assert!(epsg.is_some());
}

#[test]
fn test_local_mode_extent_ceiling_is_configurable() {
    let _guard = common::config_guard();
    let (lon, lat) = field();
    config.set("local_mode_extent_km_ceiling", 0.0).unwrap();
    let error = prepare_metric_xy(&lon, &lat, &lon, &lat, Some(GeographicMode::Local)).unwrap_err();
    config._clear();
    assert_eq!(error.kind(), ErrorKind::ValidationError);
    assert!(error.message().contains("local_mode_extent_km_ceiling"));
}

#[test]
fn test_reproject_geographic_xy_local_mode_rejects_regional_extent() {
    let _guard = common::config_guard();
    let (lon, lat) = regional();
    let error = reproject_geographic_xy(&lon, &lat, Some(GeographicMode::Local)).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValidationError);
    assert!(error.message().contains("local_mode_extent_km_ceiling"));
}

#[test]
fn non_geographic_input_is_only_rebased() {
    let _guard = common::config_guard();
    let x = [1000.0, 1010.0, f64::NAN];
    let y = [2000.0, 2050.0, 2010.0];
    let (xm, ym, _, _, system, epsg) = prepare_metric_xy(&x, &y, &x, &y, Some(GeographicMode::Utm)).unwrap();
    assert_eq!(system, CoordinateSystem::Projected);
    assert_eq!(epsg, None);
    assert_eq!(xm[0], 0.0);
    assert_eq!(xm[1], 10.0);
    assert!(xm[2].is_nan());
    assert_eq!(ym.to_vec()[..2], [0.0, 50.0]);
}

#[test]
fn no_finite_pair_is_rejected() {
    let _guard = common::config_guard();
    let x = [f64::NAN, 1.0];
    let y = [1.0, f64::NAN];
    let error = prepare_metric_xy(&x, &y, &x, &y, None).unwrap_err();
    assert_eq!(error.message(), "x and y must contain finite coordinates");
}
