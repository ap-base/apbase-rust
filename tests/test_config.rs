//! `apbase::config`: keys, validation, precedence and environment overrides.

mod common;

use std::sync::Mutex;

use apbase::common::ErrorKind;
use apbase::config::{config, resolve_n_threads};
use apbase::variogram::{Variogram, VariogramOptions};

const ENV_VAR: &str = "APBASE_N_THREADS";

/// Records `log::warn!` messages.
struct CapturingLogger {
    messages: Mutex<Vec<String>>,
}

impl log::Log for CapturingLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            self.messages.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static LOGGER: CapturingLogger = CapturingLogger {
    messages: Mutex::new(Vec::new()),
};

fn set_env(value: Option<&str>) {
    // SAFETY: every test of this binary holds `config_guard()`, so no other
    // thread reads or writes the environment concurrently.
    unsafe {
        match value {
            Some(value) => std::env::set_var(ENV_VAR, value),
            None => std::env::remove_var(ENV_VAR),
        }
    }
}

/// `_reset()` + `_restore()`: empty config and no env var around each test.
struct Reset {
    previous: Option<String>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

fn reset() -> Reset {
    let guard = common::config_guard();
    let previous = std::env::var(ENV_VAR).ok();
    set_env(None);
    Reset {
        previous,
        _guard: guard,
    }
}

impl Drop for Reset {
    fn drop(&mut self) {
        config._clear();
        set_env(self.previous.as_deref());
    }
}

#[test]
fn test_config_setitem_getitem_roundtrip() {
    let _reset = reset();
    config.set("n_threads", 4).unwrap();
    assert_eq!(config.get_item("n_threads").unwrap().as_i64(), 4);
    assert!(config.contains("n_threads"));
}

#[test]
fn test_config_rejects_unknown_key() {
    let _reset = reset();
    let error = config.set("not_a_real_key", 1).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::KeyError);
    assert_eq!(error.message(), "unknown apbase.config key: 'not_a_real_key'");
}

#[test]
fn test_config_rejects_non_positive_value() {
    let _reset = reset();
    let error = config.set("n_threads", 0).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ValueError);
    assert_eq!(error.message(), "n_threads must be greater than zero");
}

#[test]
fn test_resolve_n_threads_config_wins_over_env_var() {
    let _reset = reset();
    set_env(Some("8"));
    config.set("n_threads", 4).unwrap();
    assert_eq!(resolve_n_threads(), 4);
}

#[test]
fn test_resolve_n_threads_env_var_wins_over_default() {
    let _reset = reset();
    set_env(Some("8"));
    assert_eq!(resolve_n_threads(), 8);
}

#[test]
fn test_resolve_n_threads_falls_back_to_one() {
    let _reset = reset();
    assert_eq!(resolve_n_threads(), 1);
}

#[test]
fn test_variogram_picks_up_config_default() {
    let _reset = reset();
    config.set("n_threads", 3).unwrap();
    assert_eq!(
        Variogram::new(VariogramOptions::default()).unwrap().n_threads(),
        3
    );
}

#[test]
fn test_config_getitem_error_lists_valid_keys() {
    let _reset = reset();
    let error = config.get_item("n_threads").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::KeyError);
    assert!(error.message().contains("n_threads"));
    assert!(
        error
            .message()
            .starts_with("'n_threads' is not set; valid keys: ['grid_max_points_ceiling'")
    );
}

#[test]
fn test_resolve_n_threads_warns_and_falls_back_on_malformed_env_var() {
    let _reset = reset();
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
    LOGGER.messages.lock().unwrap().clear();

    set_env(Some("not-an-int"));
    assert_eq!(resolve_n_threads(), 1);
    let messages = LOGGER.messages.lock().unwrap();
    assert!(
        messages.iter().any(|message| message.contains(ENV_VAR)
            && message
                == "APBASE_N_THREADS='not-an-int' is not a valid integer; ignoring it and falling back to 1"),
        "warning not emitted: {messages:?}"
    );
}

// The remaining keys, the validators and the `Display` form.

#[test]
fn every_key_validates() {
    let _reset = reset();
    for key in [
        "n_threads",
        "variogram_n_lags",
        "max_neighbors_ceiling",
        "variogram_max_pairs_ceiling",
        "grid_max_points_ceiling",
        "max_secondary_variables_ceiling",
        "max_lmc_structures_ceiling",
    ] {
        assert_eq!(
            config.set(key, 0).unwrap_err().message(),
            format!("{key} must be greater than zero")
        );
        config.set(key, 2.9).unwrap();
        assert_eq!(
            config.get_item(key).unwrap().as_i64(),
            2,
            "{key}: int() truncates"
        );
    }
    assert_eq!(
        config.set("variogram_max_pairs", -1).unwrap_err().message(),
        "variogram_max_pairs must be >= 0"
    );
    config.set("variogram_max_pairs", 0).unwrap();
    for key in ["variogram_max_distance", "local_mode_extent_km_ceiling"] {
        assert_eq!(
            config.set(key, -0.5).unwrap_err().message(),
            format!("{key} must be >= 0")
        );
        config.set(key, 0).unwrap();
        assert_eq!(config.get_item(key).unwrap().as_f64(), 0.0);
    }
}

#[test]
fn resolvers_use_defaults_and_overrides() {
    let _reset = reset();
    use apbase::config::*;
    assert_eq!(resolve_variogram_profile(), (50, 100_000, 0.0));
    assert_eq!(resolve_max_neighbors_ceiling(), 500);
    assert_eq!(resolve_variogram_max_pairs_ceiling(), 200_000);
    assert_eq!(resolve_local_mode_extent_km_ceiling(), 50.0);
    assert_eq!(resolve_grid_max_points_ceiling(), 100_000_000);
    assert_eq!(resolve_max_secondary_variables_ceiling(), 20);
    assert_eq!(resolve_max_lmc_structures_ceiling(), 5);

    config.set("variogram_n_lags", 12).unwrap();
    config.set("variogram_max_distance", 250.5).unwrap();
    assert_eq!(resolve_variogram_profile(), (12, 100_000, 250.5));
    assert_eq!(config.get_or("max_neighbors_ceiling", 7).as_i64(), 7);
    assert_eq!(
        format!("{config}"),
        "apbase.config({'variogram_n_lags': 12, 'variogram_max_distance': 250.5})"
    );
}

#[test]
fn env_var_parsing_accepts_whitespace_signs_and_digit_separators() {
    let _reset = reset();
    for (value, expected) in [(" 6 ", 6), ("+3", 3), ("-2", 1), ("0", 1), ("1_0", 10)] {
        set_env(Some(value));
        assert_eq!(resolve_n_threads(), expected, "APBASE_N_THREADS={value:?}");
    }
}
