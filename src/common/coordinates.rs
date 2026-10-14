//! Geographic/metric coordinate detection and conversion.
//!
//! UTM conversions use the PROJ pipeline `unitconvert(deg -> rad) + utm(zone,
//! ellps=WGS84)`, the operation PROJ selects for EPSG:4326 -> EPSG:326xx/327xx
//! with lon/lat axis order.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;
use std::str::FromStr;

use ndarray::Array1;
use proj::Proj;

use super::_numpy;
use super::arrays::ArrayLike;
use super::exceptions::{ApbaseError, Result};
use crate::config::resolve_local_mode_extent_km_ceiling;

/// Mean Earth radius (meters) used by the local equirectangular projection.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

const UTM_TRANSFORMER_CACHE_SIZE: usize = 64;

/// Result of [`guess_xy_coordinate_system`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CoordinateSystem {
    /// Lon/lat-like bounds.
    Geographic,
    /// UTM-like easting/northing bounds.
    Utm,
    /// Any other projected/metric coordinates.
    Projected,
    /// No finite coordinate pair.
    Invalid,
}

impl CoordinateSystem {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            CoordinateSystem::Geographic => "geographic",
            CoordinateSystem::Utm => "utm",
            CoordinateSystem::Projected => "projected",
            CoordinateSystem::Invalid => "invalid",
        }
    }
}

impl fmt::Display for CoordinateSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for CoordinateSystem {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

/// Projection used for geographic input: `"local"` or `"utm"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeographicMode {
    /// Flat equirectangular tangent plane anchored on the mean lon/lat.
    Local,
    /// Real UTM zone derived from the mean lon/lat.
    Utm,
}

impl GeographicMode {
    /// String value, as accepted by `parse`.
    pub fn as_str(self) -> &'static str {
        match self {
            GeographicMode::Local => "local",
            GeographicMode::Utm => "utm",
        }
    }
}

impl fmt::Display for GeographicMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<&str> for GeographicMode {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl FromStr for GeographicMode {
    type Err = ApbaseError;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "local" => Ok(GeographicMode::Local),
            "utm" => Ok(GeographicMode::Utm),
            _ => Err(ApbaseError::validation_error(
                "geographic_mode must be 'local', 'utm', or None (auto-detect)",
            )),
        }
    }
}

/// Metadata needed to reuse or invert a metric coordinate transform.
///
/// Obtained from [`prepare_metric_xy_with_transform`] and passed to
/// [`restore_original_xy`] to recover coordinates in the original input metric.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinateTransform {
    /// Detected input coordinate system.
    pub coordinate_system: CoordinateSystem,
    /// Projection mode used for geographic input, `None` when none was applied.
    pub geographic_mode: Option<GeographicMode>,
    /// UTM EPSG code used for geographic `"utm"` conversion.
    pub epsg: Option<i64>,
    /// Metric rebase origin subtracted from converted x coordinates.
    pub x0: f64,
    /// Metric rebase origin subtracted from converted y coordinates.
    pub y0: f64,
    /// Geographic anchor longitude used by `"local"`.
    pub lon0: Option<f64>,
    /// Geographic anchor latitude used by `"local"`.
    pub lat0: Option<f64>,
}

fn values_of<A: ArrayLike + ?Sized>(values: &A) -> Cow<'_, [f64]> {
    match values.as_f64_slice() {
        Some(slice) => Cow::Borrowed(slice),
        None => Cow::Owned(values.to_f64_vec()),
    }
}

fn finite_pairs(x: &[f64], y: &[f64]) -> Vec<bool> {
    x.iter()
        .zip(y)
        .map(|(a, b)| a.is_finite() && b.is_finite())
        .collect()
}

fn masked(values: &[f64], mask: &[bool]) -> Vec<f64> {
    values
        .iter()
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .map(|(value, _)| *value)
        .collect()
}

fn scatter(len: usize, mask: &[bool], values: &[f64]) -> Array1<f64> {
    let mut out = Array1::from_elem(len, f64::NAN);
    let mut source = values.iter();
    for (slot, keep) in out.iter_mut().zip(mask) {
        if *keep {
            *slot = *source.next().expect("one value per masked row");
        }
    }
    out
}

/// Guess whether x/y coordinates look geographic, UTM, or projected.
///
/// A fast bounds heuristic, not CRS detection from metadata. With
/// `assume_finite`, the finite filtering is skipped.
pub fn guess_xy_coordinate_system(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    assume_finite: bool,
) -> CoordinateSystem {
    guess_from_slices(&values_of(x), &values_of(y), assume_finite)
}

pub(crate) fn guess_from_slices(x: &[f64], y: &[f64], assume_finite: bool) -> CoordinateSystem {
    let (xmin, xmax, ymin, ymax) = if assume_finite {
        (_numpy::min(x), _numpy::max(x), _numpy::min(y), _numpy::max(y))
    } else {
        let mask = finite_pairs(x, y);
        if !mask.iter().any(|keep| *keep) {
            return CoordinateSystem::Invalid;
        }
        let xv = masked(x, &mask);
        let yv = masked(y, &mask);
        (
            _numpy::min(&xv),
            _numpy::max(&xv),
            _numpy::min(&yv),
            _numpy::max(&yv),
        )
    };

    let within = |value: f64, lo: f64, hi: f64| lo <= value && value <= hi;
    if within(xmin, -180.0, 180.0)
        && within(xmax, -180.0, 180.0)
        && within(ymin, -90.0, 90.0)
        && within(ymax, -90.0, 90.0)
    {
        return CoordinateSystem::Geographic;
    }
    if within(xmin, 100_000.0, 900_000.0)
        && within(xmax, 100_000.0, 900_000.0)
        && within(ymin, 0.0, 10_000_000.0)
        && within(ymax, 0.0, 10_000_000.0)
    {
        return CoordinateSystem::Utm;
    }
    CoordinateSystem::Projected
}

thread_local! {
    // `lru_cache(maxsize=64)` around `_utm_transformer`; PROJ objects are not Send.
    static UTM_TRANSFORMERS: RefCell<Vec<(i64, Rc<Proj>)>> = const { RefCell::new(Vec::new()) };
}

fn utm_pipeline(epsg: i64) -> Option<String> {
    let (zone, south) = match epsg {
        32601..=32660 => (epsg - 32600, ""),
        32701..=32760 => (epsg - 32700, " +south"),
        _ => return None,
    };
    Some(format!(
        "+proj=pipeline +step +proj=unitconvert +xy_in=deg +xy_out=rad +step +proj=utm +zone={zone}{south} +ellps=WGS84"
    ))
}

fn utm_transformer(epsg: i64) -> Result<Rc<Proj>> {
    UTM_TRANSFORMERS.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(position) = cache.iter().position(|(code, _)| *code == epsg) {
            let entry = cache.remove(position);
            let transformer = Rc::clone(&entry.1);
            cache.push(entry);
            return Ok(transformer);
        }
        let created = match utm_pipeline(epsg) {
            Some(pipeline) => Proj::new(&pipeline),
            None => Proj::new_known_crs("EPSG:4326", &format!("EPSG:{epsg}"), None),
        };
        let transformer =
            Rc::new(created.map_err(|_| ApbaseError::validation_error("failed to prepare UTM conversion"))?);
        if cache.len() == UTM_TRANSFORMER_CACHE_SIZE {
            cache.remove(0);
        }
        cache.push((epsg, Rc::clone(&transformer)));
        Ok(transformer)
    })
}

fn utm_epsg_from_lonlat(lon: &[f64], lat: &[f64]) -> Result<i64> {
    let mask = finite_pairs(lon, lat);
    if !mask.iter().any(|keep| *keep) {
        return Err(ApbaseError::validation_error(
            "lon/lat must contain at least one finite coordinate",
        ));
    }
    let lon0 = _numpy::mean(&masked(lon, &mask));
    let lat0 = _numpy::mean(&masked(lat, &mask));
    let zone = (((lon0 + 180.0) / 6.0).floor() as i64 + 1).clamp(1, 60);
    Ok(if lat0 >= 0.0 { 32600 } else { 32700 } + zone)
}

fn lonlat_to_local_xy(lon: &[f64], lat: &[f64], lon0: f64, lat0: f64) -> (Vec<f64>, Vec<f64>) {
    let lat0_rad = _numpy::deg2rad(lat0);
    let x_scale = EARTH_RADIUS_M * lat0_rad.cos();
    let x_local = lon
        .iter()
        .map(|value| x_scale * _numpy::deg2rad(value - lon0))
        .collect();
    let y_local = lat
        .iter()
        .map(|value| EARTH_RADIUS_M * _numpy::deg2rad(value - lat0))
        .collect();
    (x_local, y_local)
}

fn local_mode_extent_km(x_local: &[f64], y_local: &[f64], finite_mask: &[bool]) -> f64 {
    if !finite_mask.iter().any(|keep| *keep) {
        return 0.0;
    }
    let xv = masked(x_local, finite_mask);
    let yv = masked(y_local, finite_mask);
    (_numpy::max(&xv) - _numpy::min(&xv)).hypot(_numpy::max(&yv) - _numpy::min(&yv)) / 1000.0
}

fn check_local_mode_extent(x_local: &[f64], y_local: &[f64], finite_mask: &[bool]) -> Result<()> {
    if !finite_mask.iter().any(|keep| *keep) {
        return Ok(());
    }
    let ceiling_km = resolve_local_mode_extent_km_ceiling();
    let extent_km = local_mode_extent_km(x_local, y_local, finite_mask);
    if extent_km > ceiling_km {
        return Err(ApbaseError::validation_error(format!(
            "geographic_mode='local' approximates distance with a flat tangent plane anchored on the input's \
             mean lon/lat, which grows inaccurate over large extents; this input spans ~{extent_km:.0} km, above \
             the {ceiling_km:.0} km ceiling (apbase.config['local_mode_extent_km_ceiling']). Use \
             geographic_mode='utm' for regional or multi-zone extents."
        )));
    }
    Ok(())
}

fn auto_geographic_mode(x_local: &[f64], y_local: &[f64], finite_mask: &[bool]) -> GeographicMode {
    let extent_km = local_mode_extent_km(x_local, y_local, finite_mask);
    if extent_km <= resolve_local_mode_extent_km_ceiling() {
        GeographicMode::Local
    } else {
        GeographicMode::Utm
    }
}

fn lonlat_to_utm_xy(lon: &[f64], lat: &[f64], epsg: i64) -> Result<(Vec<f64>, Vec<f64>)> {
    let transformer = utm_transformer(epsg)?;
    let mut x_utm = vec![f64::NAN; lon.len()];
    let mut y_utm = vec![f64::NAN; lat.len()];
    for (i, (&lon_value, &lat_value)) in lon.iter().zip(lat).enumerate() {
        if lon_value.is_finite() && lat_value.is_finite() {
            // Points PROJ cannot transform become inf instead of an error.
            let (x, y) = transformer
                .project((lon_value, lat_value), false)
                .unwrap_or((f64::INFINITY, f64::INFINITY));
            x_utm[i] = x;
            y_utm[i] = y;
        }
    }
    Ok((x_utm, y_utm))
}

struct PreparedMetric {
    x: Array1<f64>,
    y: Array1<f64>,
    target_x: Array1<f64>,
    target_y: Array1<f64>,
    coordinate_system: CoordinateSystem,
    epsg: Option<i64>,
    transform: CoordinateTransform,
}

fn same_array<A: ArrayLike + ?Sized, B: ArrayLike + ?Sized>(a: &A, b: &B) -> bool {
    match (a.as_f64_slice(), b.as_f64_slice()) {
        (Some(a), Some(b)) => std::ptr::eq(a, b),
        _ => false,
    }
}

fn prepare_metric(
    x: &[f64],
    y: &[f64],
    targets: Option<(&[f64], &[f64])>,
    geographic_mode: Option<GeographicMode>,
) -> Result<PreparedMetric> {
    let coordinate_system = guess_from_slices(x, y, false);
    let mut epsg = None;
    let mut lon0 = None;
    let mut lat0 = None;
    let mut mode = None;

    let (mut x_metric, mut y_metric, mut target_metric) = if coordinate_system == CoordinateSystem::Geographic
    {
        let source_mask = finite_pairs(x, y);
        if !source_mask.iter().any(|keep| *keep) {
            return Err(ApbaseError::validation_error(
                "x and y must contain finite coordinates",
            ));
        }
        let mean_lon0 = _numpy::mean(&masked(x, &source_mask));
        let mean_lat0 = _numpy::mean(&masked(y, &source_mask));

        let mut local = None;
        let chosen = match geographic_mode {
            Some(chosen) => chosen,
            None => {
                let (x_local, y_local) = lonlat_to_local_xy(x, y, mean_lon0, mean_lat0);
                let chosen = auto_geographic_mode(&x_local, &y_local, &source_mask);
                local = Some((x_local, y_local));
                chosen
            }
        };
        mode = Some(chosen);

        if chosen == GeographicMode::Utm {
            let code = utm_epsg_from_lonlat(x, y)?;
            epsg = Some(code);
            let (xm, ym) = lonlat_to_utm_xy(x, y, code)?;
            let targets = match targets {
                Some((tx, ty)) => Some(lonlat_to_utm_xy(tx, ty, code)?),
                None => None,
            };
            (xm, ym, targets)
        } else {
            lon0 = Some(mean_lon0);
            lat0 = Some(mean_lat0);
            let (xm, ym) = local.unwrap_or_else(|| lonlat_to_local_xy(x, y, mean_lon0, mean_lat0));
            check_local_mode_extent(&xm, &ym, &source_mask)?;
            let targets = targets.map(|(tx, ty)| lonlat_to_local_xy(tx, ty, mean_lon0, mean_lat0));
            (xm, ym, targets)
        }
    } else {
        (
            x.to_vec(),
            y.to_vec(),
            targets.map(|(tx, ty)| (tx.to_vec(), ty.to_vec())),
        )
    };

    let finite_source = finite_pairs(&x_metric, &y_metric);
    if !finite_source.iter().any(|keep| *keep) {
        return Err(ApbaseError::validation_error(
            "x and y must contain finite coordinates",
        ));
    }
    let x0 = _numpy::min(&masked(&x_metric, &finite_source));
    let y0 = _numpy::min(&masked(&y_metric, &finite_source));
    for ((xv, yv), keep) in x_metric.iter_mut().zip(y_metric.iter_mut()).zip(&finite_source) {
        if *keep {
            *xv -= x0;
            *yv -= y0;
        }
    }

    let (target_x, target_y) = match target_metric.take() {
        Some((mut tx, mut ty)) => {
            let finite_target = finite_pairs(&tx, &ty);
            for ((xv, yv), keep) in tx.iter_mut().zip(ty.iter_mut()).zip(&finite_target) {
                if *keep {
                    *xv -= x0;
                    *yv -= y0;
                }
            }
            (Array1::from(tx), Array1::from(ty))
        }
        None => (Array1::from(x_metric.clone()), Array1::from(y_metric.clone())),
    };

    Ok(PreparedMetric {
        x: Array1::from(x_metric),
        y: Array1::from(y_metric),
        target_x,
        target_y,
        coordinate_system,
        epsg,
        transform: CoordinateTransform {
            coordinate_system,
            geographic_mode: mode,
            epsg,
            x0,
            y0,
            lon0,
            lat0,
        },
    })
}

/// Metric source/target coordinates plus detection metadata.
pub type MetricXY = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    CoordinateSystem,
    Option<i64>,
);

/// [`MetricXY`] plus the [`CoordinateTransform`].
pub type MetricXYWithTransform = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    CoordinateSystem,
    Option<i64>,
    CoordinateTransform,
);

fn prepare_dispatch<A, B, C, D>(
    x: &A,
    y: &B,
    target_x: &C,
    target_y: &D,
    geographic_mode: Option<GeographicMode>,
) -> Result<PreparedMetric>
where
    A: ArrayLike + ?Sized,
    B: ArrayLike + ?Sized,
    C: ArrayLike + ?Sized,
    D: ArrayLike + ?Sized,
{
    let x_values = values_of(x);
    let y_values = values_of(y);
    if same_array(x, target_x) && same_array(y, target_y) {
        prepare_metric(&x_values, &y_values, None, geographic_mode)
    } else {
        let tx = values_of(target_x);
        let ty = values_of(target_y);
        prepare_metric(&x_values, &y_values, Some((&tx, &ty)), geographic_mode)
    }
}

/// Convert source/target coordinates to a shared metric system, rebased to zero.
///
/// Pass the same arrays as `x`/`y` to reuse the source coordinates as targets.
/// `geographic_mode` is only used for input detected as geographic; `None`
/// auto-picks `"local"` below `apbase.config["local_mode_extent_km_ceiling"]`
/// and `"utm"` above it.
pub fn prepare_metric_xy(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    target_x: &(impl ArrayLike + ?Sized),
    target_y: &(impl ArrayLike + ?Sized),
    geographic_mode: Option<GeographicMode>,
) -> Result<MetricXY> {
    let prepared = prepare_dispatch(x, y, target_x, target_y, geographic_mode)?;
    Ok((
        prepared.x,
        prepared.y,
        prepared.target_x,
        prepared.target_y,
        prepared.coordinate_system,
        prepared.epsg,
    ))
}

/// [`prepare_metric_xy`] with `return_transform=True`.
pub fn prepare_metric_xy_with_transform(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    target_x: &(impl ArrayLike + ?Sized),
    target_y: &(impl ArrayLike + ?Sized),
    geographic_mode: Option<GeographicMode>,
) -> Result<MetricXYWithTransform> {
    let prepared = prepare_dispatch(x, y, target_x, target_y, geographic_mode)?;
    Ok((
        prepared.x,
        prepared.y,
        prepared.target_x,
        prepared.target_y,
        prepared.coordinate_system,
        prepared.epsg,
        prepared.transform,
    ))
}

/// Invert [`prepare_metric_xy`]: recover coordinates in the original input metric.
pub fn restore_original_xy(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    transform: &CoordinateTransform,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let x = values_of(x);
    let y = values_of(y);
    let mask = finite_pairs(&x, &y);
    if !mask.iter().any(|keep| *keep) {
        return Ok((
            Array1::from_elem(x.len(), f64::NAN),
            Array1::from_elem(y.len(), f64::NAN),
        ));
    }
    let x_shifted: Vec<f64> = masked(&x, &mask)
        .iter()
        .map(|value| value + transform.x0)
        .collect();
    let y_shifted: Vec<f64> = masked(&y, &mask)
        .iter()
        .map(|value| value + transform.y0)
        .collect();

    if transform.coordinate_system != CoordinateSystem::Geographic {
        return Ok((
            scatter(x.len(), &mask, &x_shifted),
            scatter(y.len(), &mask, &y_shifted),
        ));
    }

    if transform.geographic_mode == Some(GeographicMode::Utm) {
        let epsg = transform.epsg.ok_or_else(|| {
            ApbaseError::validation_error("error converting UTM coordinates back to geographic")
        })?;
        let transformer = utm_transformer(epsg)?;
        let mut lon = Vec::with_capacity(x_shifted.len());
        let mut lat = Vec::with_capacity(y_shifted.len());
        for (&xs, &ys) in x_shifted.iter().zip(&y_shifted) {
            let (lon_value, lat_value) = transformer
                .project((xs, ys), true)
                .unwrap_or((f64::INFINITY, f64::INFINITY));
            lon.push(lon_value);
            lat.push(lat_value);
        }
        return Ok((scatter(x.len(), &mask, &lon), scatter(y.len(), &mask, &lat)));
    }

    let (Some(lon0), Some(lat0)) = (transform.lon0, transform.lat0) else {
        return Err(ApbaseError::value_error(
            "local CoordinateTransform requires lon0 and lat0",
        ));
    };
    let denominator = EARTH_RADIUS_M * _numpy::deg2rad(lat0).cos();
    let lon: Vec<f64> = x_shifted
        .iter()
        .map(|value| lon0 + _numpy::rad2deg(value / denominator))
        .collect();
    let lat: Vec<f64> = y_shifted
        .iter()
        .map(|value| lat0 + _numpy::rad2deg(value / EARTH_RADIUS_M))
        .collect();
    Ok((scatter(x.len(), &mask, &lon), scatter(y.len(), &mask, &lat)))
}

/// Apply an existing [`CoordinateTransform`] to new x/y coordinates (forward
/// counterpart of [`restore_original_xy`]).
pub fn apply_metric_transform(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    transform: &CoordinateTransform,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let x = values_of(x);
    let y = values_of(y);
    apply_metric_transform_slices(&x, &y, transform)
}

pub(crate) fn apply_metric_transform_slices(
    x: &[f64],
    y: &[f64],
    transform: &CoordinateTransform,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let mask = finite_pairs(x, y);
    if !mask.iter().any(|keep| *keep) {
        return Ok((
            Array1::from_elem(x.len(), f64::NAN),
            Array1::from_elem(y.len(), f64::NAN),
        ));
    }
    let xv = masked(x, &mask);
    let yv = masked(y, &mask);
    let (x_metric, y_metric) = match transform.geographic_mode {
        Some(GeographicMode::Utm) => {
            let epsg = transform
                .epsg
                .ok_or_else(|| ApbaseError::value_error("utm CoordinateTransform requires an epsg code"))?;
            lonlat_to_utm_xy(&xv, &yv, epsg)?
        }
        Some(GeographicMode::Local) => {
            let (Some(lon0), Some(lat0)) = (transform.lon0, transform.lat0) else {
                return Err(ApbaseError::value_error(
                    "local CoordinateTransform requires lon0 and lat0",
                ));
            };
            lonlat_to_local_xy(&xv, &yv, lon0, lat0)
        }
        None => (xv, yv),
    };
    let x_out: Vec<f64> = x_metric.iter().map(|value| value - transform.x0).collect();
    let y_out: Vec<f64> = y_metric.iter().map(|value| value - transform.y0).collect();
    Ok((scatter(x.len(), &mask, &x_out), scatter(y.len(), &mask, &y_out)))
}

/// Project geographic (lon/lat) coordinates to metric, with no rebase.
pub fn reproject_geographic_xy(
    x: &(impl ArrayLike + ?Sized),
    y: &(impl ArrayLike + ?Sized),
    geographic_mode: Option<GeographicMode>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let x = values_of(x);
    let y = values_of(y);
    reproject_geographic_xy_slices(&x, &y, geographic_mode)
}

pub(crate) fn reproject_geographic_xy_slices(
    x: &[f64],
    y: &[f64],
    geographic_mode: Option<GeographicMode>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let mask = finite_pairs(x, y);
    if !mask.iter().any(|keep| *keep) {
        return Ok((
            Array1::from_elem(x.len(), f64::NAN),
            Array1::from_elem(y.len(), f64::NAN),
        ));
    }
    let xv = masked(x, &mask);
    let yv = masked(y, &mask);
    let all = vec![true; xv.len()];

    let mut local = None;
    let mode = match geographic_mode {
        Some(mode) => mode,
        None => {
            let lon0 = _numpy::mean(&xv);
            let lat0 = _numpy::mean(&yv);
            let (x_local, y_local) = lonlat_to_local_xy(&xv, &yv, lon0, lat0);
            let mode = auto_geographic_mode(&x_local, &y_local, &all);
            local = Some((x_local, y_local));
            mode
        }
    };

    let (x_metric, y_metric) = if mode == GeographicMode::Utm {
        let epsg = utm_epsg_from_lonlat(&xv, &yv)?;
        lonlat_to_utm_xy(&xv, &yv, epsg)?
    } else {
        let (x_metric, y_metric) = local.unwrap_or_else(|| {
            let lon0 = _numpy::mean(&xv);
            let lat0 = _numpy::mean(&yv);
            lonlat_to_local_xy(&xv, &yv, lon0, lat0)
        });
        check_local_mode_extent(&x_metric, &y_metric, &all)?;
        (x_metric, y_metric)
    };
    Ok((
        scatter(x.len(), &mask, &x_metric),
        scatter(y.len(), &mask, &y_metric),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guess_buckets() {
        assert_eq!(
            guess_xy_coordinate_system(&[-47.0, -46.9], &[-15.0, -14.9], false),
            CoordinateSystem::Geographic
        );
        assert_eq!(
            guess_xy_coordinate_system(&[500_000.0, 501_000.0], &[8_000_000.0, 8_001_000.0], false),
            "utm"
        );
        assert_eq!(
            guess_xy_coordinate_system(&[0.0, 1000.0], &[0.0, 1000.0], false),
            "projected"
        );
        assert_eq!(
            guess_xy_coordinate_system(&[f64::NAN], &[1.0], false),
            CoordinateSystem::Invalid
        );
        assert_eq!(
            guess_xy_coordinate_system(&[f64::NAN, 1.0], &[1.0, 1.0], true),
            CoordinateSystem::Projected
        );
    }

    #[test]
    fn utm_zone_edges() {
        assert_eq!(utm_epsg_from_lonlat(&[-180.0], &[10.0]).unwrap(), 32601);
        assert_eq!(utm_epsg_from_lonlat(&[180.0], &[10.0]).unwrap(), 32660);
        assert_eq!(utm_epsg_from_lonlat(&[-47.5], &[-15.0]).unwrap(), 32723);
        assert_eq!(utm_epsg_from_lonlat(&[3.0], &[0.0]).unwrap(), 32631);
        assert!(utm_epsg_from_lonlat(&[f64::NAN], &[0.0]).is_err());
    }

    #[test]
    fn utm_matches_reference_values() {
        // Reference EPSG:4326 -> EPSG:32722 / EPSG:32633 conversions.
        let (x, y) = lonlat_to_utm_xy(&[-47.5], &[-15.75], 32722).unwrap();
        assert_eq!((x[0], y[0]), (875132.8789027982, 8255605.003636604));
        let (x, y) = lonlat_to_utm_xy(&[15.2], &[45.1], 32633).unwrap();
        assert_eq!((x[0], y[0]), (515735.61572633445, 4994078.684134477));
    }

    #[test]
    fn local_round_trip_and_rebase() {
        let lon = [-47.50, -47.49, -47.48];
        let lat = [-15.70, -15.71, -15.72];
        let (xm, ym, _, _, system, epsg, transform) =
            prepare_metric_xy_with_transform(&lon, &lat, &lon, &lat, None).unwrap();
        assert_eq!(system, CoordinateSystem::Geographic);
        assert_eq!(epsg, None);
        assert_eq!(transform.geographic_mode, Some(GeographicMode::Local));
        assert_eq!(_numpy::min(xm.as_slice().unwrap()), 0.0);
        assert_eq!(_numpy::min(ym.as_slice().unwrap()), 0.0);
        let (lon_back, lat_back) = restore_original_xy(&xm, &ym, &transform).unwrap();
        for (a, b) in lon_back.iter().zip(lon) {
            assert!((a - b).abs() < 1e-12);
        }
        for (a, b) in lat_back.iter().zip(lat) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn geographic_mode_parsing() {
        assert_eq!("utm".parse::<GeographicMode>().unwrap(), GeographicMode::Utm);
        assert_eq!(
            "bad".parse::<GeographicMode>().unwrap_err().message(),
            "geographic_mode must be 'local', 'utm', or None (auto-detect)"
        );
    }

    #[test]
    fn format_rounds_half_to_even() {
        assert_eq!(format!("{:.0}", 0.5), "0");
        assert_eq!(format!("{:.0}", 1.5), "2");
        assert_eq!(format!("{:.0}", 2.5), "2");
    }
}
