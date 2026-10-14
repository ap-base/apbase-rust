//! Process-wide default overrides for apbase classes.
//!
//! [`config`] holds the explicitly configured keys and the `resolve_*` functions
//! read them live (with their defaults) every time a class needs a value.

use std::fmt;
use std::sync::{PoisonError, RwLock};

use crate::common::exceptions::{ApbaseError, ErrorKind, Result};

/// A configured value: an integer or a float.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConfigValue {
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
}

impl ConfigValue {
    /// Value as an integer; floats are truncated toward zero.
    pub fn as_i64(self) -> i64 {
        match self {
            ConfigValue::Int(value) => value,
            ConfigValue::Float(value) => value as i64,
        }
    }

    /// Value as a float.
    pub fn as_f64(self) -> f64 {
        match self {
            ConfigValue::Int(value) => value as f64,
            ConfigValue::Float(value) => value,
        }
    }
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Int(value) => write!(f, "{value}"),
            ConfigValue::Float(value) => f.write_str(&crate::common::arrays::float_repr(*value)),
        }
    }
}

macro_rules! config_value_from {
    ($variant:ident: $($ty:ty),+) => {
        $(impl From<$ty> for ConfigValue {
            fn from(value: $ty) -> Self {
                ConfigValue::$variant(value.into())
            }
        })+
    };
}

config_value_from!(Int: i8, i16, i32, i64, u8, u16, u32);
config_value_from!(Float: f32, f64);

impl From<bool> for ConfigValue {
    fn from(value: bool) -> Self {
        ConfigValue::Int(i64::from(value))
    }
}

impl From<usize> for ConfigValue {
    fn from(value: usize) -> Self {
        ConfigValue::Int(i64::try_from(value).unwrap_or(i64::MAX))
    }
}

type Validator = fn(&str, ConfigValue) -> Result<ConfigValue>;

/// Integer value of a setting: floats truncate toward zero, NaN and infinity are errors.
fn to_int(value: ConfigValue) -> Result<i64> {
    match value {
        ConfigValue::Int(value) => Ok(value),
        ConfigValue::Float(value) if value.is_nan() => Err(ApbaseError::new(
            ErrorKind::ValueError,
            "cannot convert float NaN to integer",
        )),
        ConfigValue::Float(value) if value.is_infinite() => Err(ApbaseError::new(
            ErrorKind::OverflowError,
            "cannot convert float infinity to integer",
        )),
        ConfigValue::Float(value) => Ok(value.trunc() as i64),
    }
}

fn positive_int(key: &str, value: ConfigValue) -> Result<ConfigValue> {
    let int_value = to_int(value)?;
    if int_value <= 0 {
        return Err(ApbaseError::value_error(format!(
            "{key} must be greater than zero"
        )));
    }
    Ok(ConfigValue::Int(int_value))
}

fn non_negative_int(key: &str, value: ConfigValue) -> Result<ConfigValue> {
    let int_value = to_int(value)?;
    if int_value < 0 {
        return Err(ApbaseError::value_error(format!("{key} must be >= 0")));
    }
    Ok(ConfigValue::Int(int_value))
}

fn non_negative_float(key: &str, value: ConfigValue) -> Result<ConfigValue> {
    let float_value = value.as_f64();
    if float_value < 0.0 {
        return Err(ApbaseError::value_error(format!("{key} must be >= 0")));
    }
    Ok(ConfigValue::Float(float_value))
}

const VALIDATORS: &[(&str, Validator)] = &[
    ("n_threads", positive_int),
    ("variogram_n_lags", positive_int),
    ("variogram_max_pairs", non_negative_int),
    ("variogram_max_distance", non_negative_float),
    ("max_neighbors_ceiling", positive_int),
    ("variogram_max_pairs_ceiling", positive_int),
    ("local_mode_extent_km_ceiling", non_negative_float),
    ("grid_max_points_ceiling", positive_int),
    ("max_secondary_variables_ceiling", positive_int),
    ("max_lmc_structures_ceiling", positive_int),
];

fn validator(key: &str) -> Option<(&'static str, Validator)> {
    VALIDATORS.iter().find(|(name, _)| *name == key).copied()
}

fn sorted_valid_keys() -> String {
    let mut keys: Vec<&str> = VALIDATORS.iter().map(|(name, _)| *name).collect();
    keys.sort_unstable();
    let quoted: Vec<String> = keys.iter().map(|key| format!("'{key}'")).collect();
    format!("[{}]", quoted.join(", "))
}

/// Global settings consulted when a class is constructed without an explicit value.
///
/// Use the process-wide instance [`config`]:
///
/// ```
/// use apbase::kriging::{Kriging, KrigingOptions};
///
/// apbase::config.set("n_threads", 4).unwrap();
/// assert_eq!(Kriging::new(KrigingOptions::default()).unwrap().n_threads(), 4);
/// # apbase::config._clear();
/// ```
pub struct Config {
    // Insertion order is kept; `Display` lists the keys in that order.
    values: RwLock<Vec<(&'static str, ConfigValue)>>,
}

impl Config {
    const fn new() -> Self {
        Config {
            values: RwLock::new(Vec::new()),
        }
    }

    /// Set a key.
    ///
    /// Unknown keys are a `KeyError`; out-of-range values are a `ValueError`.
    pub fn set(&self, key: &str, value: impl Into<ConfigValue>) -> Result<()> {
        let Some((name, validate)) = validator(key) else {
            return Err(ApbaseError::new(
                ErrorKind::KeyError,
                format!("unknown apbase.config key: '{key}'"),
            ));
        };
        let validated = validate(name, value.into())?;
        let mut values = self.values.write().unwrap_or_else(PoisonError::into_inner);
        match values.iter_mut().find(|(existing, _)| *existing == name) {
            Some(entry) => entry.1 = validated,
            None => values.push((name, validated)),
        }
        Ok(())
    }

    /// Read a key that must be set; an unset key is a `KeyError`.
    pub fn get_item(&self, key: &str) -> Result<ConfigValue> {
        self.get(key).ok_or_else(|| {
            ApbaseError::new(
                ErrorKind::KeyError,
                format!("'{key}' is not set; valid keys: {}", sorted_valid_keys()),
            )
        })
    }

    /// Return a configured value, or `None` when `key` is unset.
    pub fn get(&self, key: &str) -> Option<ConfigValue> {
        let values = self.values.read().unwrap_or_else(PoisonError::into_inner);
        values
            .iter()
            .find(|(name, _)| *name == key)
            .map(|(_, value)| *value)
    }

    /// Return a configured value or `default`.
    pub fn get_or(&self, key: &str, default: impl Into<ConfigValue>) -> ConfigValue {
        self.get(key).unwrap_or_else(|| default.into())
    }

    /// Whether `key` was explicitly set.
    pub fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    /// Remove every configured key (test helper).
    #[doc(hidden)]
    pub fn _clear(&self) {
        self.values
            .write()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let values = self.values.read().unwrap_or_else(PoisonError::into_inner);
        let items: Vec<String> = values
            .iter()
            .map(|(name, value)| format!("'{name}': {value}"))
            .collect();
        write!(f, "apbase.config({{{}}})", items.join(", "))
    }
}

/// Process-wide configuration instance.
#[allow(non_upper_case_globals)]
pub static config: Config = Config::new();

/// Resolve the native thread count.
///
/// Precedence is the configured `n_threads`, then the `APBASE_N_THREADS`
/// environment variable, then `1`.
pub fn resolve_n_threads() -> i64 {
    if let Some(configured) = config.get("n_threads") {
        return configured.as_i64();
    }

    if let Ok(env_value) = std::env::var("APBASE_N_THREADS") {
        match parse_int(&env_value) {
            Some(value) => return value.max(1),
            None => log::warn!(
                "APBASE_N_THREADS={} is not a valid integer; ignoring it and falling back to 1",
                quote_str(&env_value)
            ),
        }
    }

    1
}

/// Resolve the shared `(n_lags, max_pairs, max_distance)` variogram profile.
///
/// Used to derive the default search radius across `Kriging`, `IDW`,
/// `cross_validate` and `create_map`/`Map`; unset keys fall back to
/// `(50, 100_000, 0.0)`. Not used by `SpatialFilter`.
pub fn resolve_variogram_profile() -> (i64, i64, f64) {
    let n_lags = config.get_or("variogram_n_lags", 50);
    let max_pairs = config.get_or("variogram_max_pairs", 100_000);
    let max_distance = config.get_or("variogram_max_distance", 0.0);
    (n_lags.as_i64(), max_pairs.as_i64(), max_distance.as_f64())
}

/// Resolve the hard cap on `max_neighbors` for Kriging/cross_validate (default `500`).
pub fn resolve_max_neighbors_ceiling() -> i64 {
    config.get_or("max_neighbors_ceiling", 500).as_i64()
}

/// Resolve the hard cap on `Variogram.max_pairs` (default `200_000`).
pub fn resolve_variogram_max_pairs_ceiling() -> i64 {
    config.get_or("variogram_max_pairs_ceiling", 200_000).as_i64()
}

/// Resolve the extent ceiling (bounding-box diagonal, km) for `geographic_mode="local"` (default `50.0`).
pub fn resolve_local_mode_extent_km_ceiling() -> f64 {
    config.get_or("local_mode_extent_km_ceiling", 50.0).as_f64()
}

/// Resolve the hard cap on `Grid`'s candidate axis-grid size `nx * ny` (default `100_000_000`).
pub fn resolve_grid_max_points_ceiling() -> i64 {
    config.get_or("grid_max_points_ceiling", 100_000_000).as_i64()
}

/// Resolve the hard cap on the number of secondary variables for cokriging (default `20`).
pub fn resolve_max_secondary_variables_ceiling() -> i64 {
    config.get_or("max_secondary_variables_ceiling", 20).as_i64()
}

/// Resolve the hard cap on the number of nested LMC structures (default `5`).
pub fn resolve_max_lmc_structures_ceiling() -> i64 {
    config.get_or("max_lmc_structures_ceiling", 5).as_i64()
}

/// Parse a decimal integer: surrounding whitespace, an optional sign and single
/// underscores between digits are accepted.
fn parse_int(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    let (negative, digits) = match trimmed.as_bytes().first() {
        Some(b'-') => (true, &trimmed[1..]),
        Some(b'+') => (false, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') || digits.contains("__") {
        return None;
    }
    let cleaned: String = digits.chars().filter(|ch| *ch != '_').collect();
    if !cleaned.chars().all(|ch| ch.is_ascii_digit()) {
        return None;
    }
    let value: i64 = cleaned.parse().ok()?;
    Some(if negative { -value } else { value })
}

/// String quoted for error messages: single quotes unless the text contains a
/// single quote and no double quote; backslashes, quotes and control
/// characters are escaped.
pub(crate) fn quote_str(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_int_accepts_whitespace_signs_and_digit_separators() {
        assert_eq!(parse_int(" 4 "), Some(4));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-2"), Some(-2));
        assert_eq!(parse_int("1_000"), Some(1000));
        assert_eq!(parse_int("1__0"), None);
        assert_eq!(parse_int("_1"), None);
        assert_eq!(parse_int("4.0"), None);
        assert_eq!(parse_int("abc"), None);
        assert_eq!(parse_int(""), None);
    }

    #[test]
    fn quote_str_picks_the_quote_and_escapes() {
        assert_eq!(quote_str("abc"), "'abc'");
        assert_eq!(quote_str("it's"), "\"it's\"");
        assert_eq!(quote_str("a\\b"), "'a\\\\b'");
    }

    #[test]
    fn validators_truncate_floats_and_reject_non_finite_integers() {
        assert_eq!(
            positive_int("k", ConfigValue::Float(2.7)).unwrap(),
            ConfigValue::Int(2)
        );
        assert_eq!(
            positive_int("k", ConfigValue::Float(-0.5)).unwrap_err().message(),
            "k must be greater than zero"
        );
        assert_eq!(
            non_negative_int("k", ConfigValue::Int(-1)).unwrap_err().message(),
            "k must be >= 0"
        );
        assert_eq!(
            non_negative_float("k", ConfigValue::Int(3)).unwrap(),
            ConfigValue::Float(3.0)
        );
        // The float validator accepts NaN and infinity.
        assert!(non_negative_float("k", ConfigValue::Float(f64::NAN)).is_ok());
        assert!(non_negative_float("k", ConfigValue::Float(f64::INFINITY)).is_ok());
        assert_eq!(
            positive_int("k", ConfigValue::Float(f64::NAN))
                .unwrap_err()
                .kind(),
            ErrorKind::ValueError
        );
        assert_eq!(
            positive_int("k", ConfigValue::Float(f64::INFINITY))
                .unwrap_err()
                .kind(),
            ErrorKind::OverflowError
        );
    }

    #[test]
    fn sorted_valid_keys_lists_every_key() {
        assert_eq!(
            sorted_valid_keys(),
            "['grid_max_points_ceiling', 'local_mode_extent_km_ceiling', 'max_lmc_structures_ceiling', \
             'max_neighbors_ceiling', 'max_secondary_variables_ceiling', 'n_threads', \
             'variogram_max_distance', 'variogram_max_pairs', 'variogram_max_pairs_ceiling', \
             'variogram_n_lags']"
        );
    }
}
