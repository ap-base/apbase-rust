//! Error types returned by apbase.
//!
//! Every fallible call returns [`Result`] with an [`ApbaseError`]: an
//! [`ErrorKind`] plus a message. Kinds form a hierarchy and [`ApbaseError::is`]
//! matches a kind or any of its ancestors, including both parents of the
//! status-mapped kinds (e.g. `IDWInvalidDataError` is both an
//! `InterpolationError` and a `NativeExecutionError`).

use std::fmt;

/// Every kind of error apbase can return.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::enum_variant_names)]
pub enum ErrorKind {
    // General-purpose kinds, outside the `ApbaseError` hierarchy.
    /// Invalid argument value.
    ValueError,
    /// Unknown or unset key.
    KeyError,
    /// Argument of an unsupported type.
    TypeError,
    /// Numeric conversion out of range.
    OverflowError,
    /// Option combination not supported yet.
    NotImplementedError,
    /// GEOS failure, e.g. invalid WKT/WKB.
    GEOSException,

    // Base hierarchy.
    /// Base class for all errors raised by apbase.
    ApbaseError,
    /// Input arrays or parameters fail validation (also a `ValueError`).
    ValidationError,
    /// The compiled native extension is missing or invalid.
    NativeExtensionError,
    /// A native extension call reports a non-zero status.
    NativeExecutionError,
    /// Variogram fitting or evaluation fails.
    VariogramError,
    /// Cross-variogram fitting or covariate screening fails.
    CrossVariogramError,
    /// Spatial filtering fails.
    SpatialFilterError,
    /// IDW or kriging interpolation fails.
    InterpolationError,
    /// Leave-one-out cross-validation fails.
    CrossValidationError,

    // IDW.
    /// The native IDW kernel receives invalid source or target data.
    IDWInvalidDataError,
    /// The IDW search radius is not finite and greater than zero.
    IDWInvalidRadiusError,
    /// The IDW distance power is not finite and greater than zero.
    IDWInvalidPowerError,
    /// The IDW neighbor, grid, or thread configuration is invalid.
    IDWInvalidConfigurationError,
    /// The native IDW kernel fails to allocate its workspace.
    IDWAllocationError,

    // Kriging.
    /// The native kriging kernel receives invalid source or target data.
    KrigingInvalidDataError,
    /// The kriging search radius is invalid.
    KrigingInvalidRadiusError,
    /// The kriging neighbor configuration is invalid.
    KrigingInvalidConfigurationError,
    /// The native kriging kernel fails to allocate its workspace.
    KrigingAllocationError,
    /// The kriging workspace capacity is exceeded.
    KrigingCapacityExceededError,
    /// Kriging receives an invalid variogram model.
    KrigingInvalidModelError,

    // Cokriging.
    /// The native cokriging kernel receives invalid source or target data.
    CoKrigingInvalidDataError,
    /// The cokriging search radius is invalid.
    CoKrigingInvalidRadiusError,
    /// The cokriging neighbor configuration is invalid.
    CoKrigingInvalidConfigurationError,
    /// The native cokriging kernel fails to allocate its workspace.
    CoKrigingAllocationError,
    /// The cokriging workspace capacity is exceeded.
    CoKrigingCapacityExceededError,
    /// Cokriging receives an invalid coregionalization model.
    CoKrigingInvalidModelError,
    /// The fitted coregionalization matrix is not positive semidefinite.
    CoKrigingNonPSDCoregionalizationError,

    // Spatial filtering.
    /// The native spatial filter kernel receives invalid source data.
    SpatialFilterInvalidDataError,
    /// The spatial filter search radius is not finite and greater than zero.
    SpatialFilterInvalidRadiusError,
    /// The spatial filter neighbor, cell, filter, or thread configuration is invalid.
    SpatialFilterInvalidConfigurationError,
    /// The native spatial filter kernel fails to allocate its workspace.
    SpatialFilterAllocationError,

    // Cross-validation.
    /// The native cross-validation kernel receives invalid source or sample data.
    CrossValidationInvalidDataError,
    /// The cross-validation search radius is not finite and greater than zero.
    CrossValidationInvalidRadiusError,
    /// The cross-validation IDW power is not finite and greater than zero.
    CrossValidationInvalidPowerError,
    /// The cross-validation neighbor, grid, or thread configuration is invalid.
    CrossValidationInvalidConfigurationError,
    /// The native cross-validation kernel fails to allocate its workspace.
    CrossValidationAllocationError,
    /// Cross-validation receives an invalid variogram model.
    CrossValidationInvalidModelError,
    /// A cross-validation sample index is out of range.
    CrossValidationInvalidSampleIndexError,

    // Variogram.
    /// The fitted model_info vector has a non-finite model id.
    VariogramInvalidModelIdError,
    /// The native extension returns an unrecognized model id.
    VariogramInvalidModelError,
    /// The fitted variogram range is invalid.
    VariogramInvalidRangeError,
    /// The requested thread count is invalid.
    VariogramInvalidThreadCountError,

    // Cross-variogram / covariate screening.
    /// Fewer than 4 colocated pairs are available for the Fisher z-test.
    CrossVariogramInsufficientSampleError,
    /// A primary or candidate auto-variogram model is invalid.
    CrossVariogramInvalidAutoModelError,
    /// The fitted cross-variogram model is invalid.
    CrossVariogramInvalidCrossModelError,
}

impl ErrorKind {
    /// Kind name, e.g. `"ValueError"`.
    pub fn name(self) -> String {
        format!("{self:?}")
    }

    /// Direct parent kinds, in declaration order.
    pub const fn bases(self) -> &'static [ErrorKind] {
        use ErrorKind::*;
        match self {
            ValueError | KeyError | TypeError | OverflowError | NotImplementedError | GEOSException
            | ApbaseError => &[],
            ValidationError => &[ApbaseError, ValueError],
            NativeExtensionError | NativeExecutionError | VariogramError | CrossVariogramError
            | SpatialFilterError | InterpolationError | CrossValidationError => &[ApbaseError],
            IDWInvalidDataError
            | IDWInvalidRadiusError
            | IDWInvalidPowerError
            | IDWInvalidConfigurationError
            | IDWAllocationError
            | KrigingInvalidDataError
            | KrigingInvalidRadiusError
            | KrigingInvalidConfigurationError
            | KrigingAllocationError
            | KrigingCapacityExceededError
            | KrigingInvalidModelError
            | CoKrigingInvalidDataError
            | CoKrigingInvalidRadiusError
            | CoKrigingInvalidConfigurationError
            | CoKrigingAllocationError
            | CoKrigingCapacityExceededError
            | CoKrigingInvalidModelError
            | CoKrigingNonPSDCoregionalizationError => &[InterpolationError, NativeExecutionError],
            SpatialFilterInvalidDataError
            | SpatialFilterInvalidRadiusError
            | SpatialFilterInvalidConfigurationError
            | SpatialFilterAllocationError => &[SpatialFilterError, NativeExecutionError],
            CrossValidationInvalidDataError
            | CrossValidationInvalidRadiusError
            | CrossValidationInvalidPowerError
            | CrossValidationInvalidConfigurationError
            | CrossValidationAllocationError
            | CrossValidationInvalidModelError
            | CrossValidationInvalidSampleIndexError => &[CrossValidationError, NativeExecutionError],
            VariogramInvalidModelIdError
            | VariogramInvalidModelError
            | VariogramInvalidRangeError
            | VariogramInvalidThreadCountError => &[VariogramError, NativeExecutionError],
            CrossVariogramInsufficientSampleError
            | CrossVariogramInvalidAutoModelError
            | CrossVariogramInvalidCrossModelError => &[CrossVariogramError, NativeExecutionError],
        }
    }

    /// `issubclass(self, other)`.
    pub fn is_subclass_of(self, other: ErrorKind) -> bool {
        self == other || self.bases().iter().any(|base| base.is_subclass_of(other))
    }

    /// Fixed message of the status-mapped kinds.
    pub const fn fixed_message(self) -> Option<&'static str> {
        use ErrorKind::*;
        Some(match self {
            IDWInvalidDataError => "invalid source or target data",
            IDWInvalidRadiusError => "radius must be finite and greater than zero",
            IDWInvalidPowerError => "power must be finite and greater than zero",
            IDWInvalidConfigurationError => "invalid neighbor, grid, or thread configuration",
            IDWAllocationError => "native IDW allocation failed",
            KrigingInvalidDataError => "invalid source or target data",
            KrigingInvalidRadiusError => "invalid radius",
            KrigingInvalidConfigurationError => "invalid neighbor configuration",
            KrigingAllocationError => "kriging workspace allocation failed",
            KrigingCapacityExceededError => "kriging workspace capacity exceeded",
            KrigingInvalidModelError => "invalid variogram model",
            CoKrigingInvalidDataError => "invalid source or target data",
            CoKrigingInvalidRadiusError => "invalid radius",
            CoKrigingInvalidConfigurationError => "invalid neighbor configuration",
            CoKrigingAllocationError => "cokriging workspace allocation failed",
            CoKrigingCapacityExceededError => "cokriging workspace capacity exceeded",
            CoKrigingInvalidModelError => "invalid coregionalization model",
            CoKrigingNonPSDCoregionalizationError => "coregionalization matrix is not positive semidefinite",
            SpatialFilterInvalidDataError => "invalid source data",
            SpatialFilterInvalidRadiusError => "radius must be finite and greater than zero",
            SpatialFilterInvalidConfigurationError => {
                "invalid neighbor, cell, filter, or thread configuration"
            }
            SpatialFilterAllocationError => "native spatial filter allocation failed",
            CrossValidationInvalidDataError => "invalid source or sample data",
            CrossValidationInvalidRadiusError => "radius must be finite and greater than zero",
            CrossValidationInvalidPowerError => "power must be finite and greater than zero",
            CrossValidationInvalidConfigurationError => "invalid neighbor, grid, or thread configuration",
            CrossValidationAllocationError => "native cross-validation allocation failed",
            CrossValidationInvalidModelError => "invalid variogram model",
            CrossValidationInvalidSampleIndexError => "sample_idx contains an out-of-range index",
            VariogramInvalidModelIdError => "model_info returned a non-finite model_id",
            VariogramInvalidModelError => "invalid model returned by the native extension",
            VariogramInvalidRangeError => "variogram returned an invalid range",
            VariogramInvalidThreadCountError => "invalid thread count",
            CrossVariogramInsufficientSampleError => "fewer than 4 colocated pairs available",
            CrossVariogramInvalidAutoModelError => "invalid primary or candidate auto-variogram model",
            CrossVariogramInvalidCrossModelError => "invalid cross-variogram model",
            _ => return None,
        })
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// An error returned by apbase: its [`ErrorKind`] plus a message.
#[derive(Clone, PartialEq, Eq)]
pub struct ApbaseError {
    kind: ErrorKind,
    message: String,
}

/// `Result` alias used by every fallible apbase call.
pub type Result<T, E = ApbaseError> = std::result::Result<T, E>;

impl ApbaseError {
    /// Create an error of `kind` with `message`.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        ApbaseError {
            kind,
            message: message.into(),
        }
    }

    /// Create a status-mapped error with its fixed message.
    pub fn from_kind(kind: ErrorKind) -> Self {
        ApbaseError::new(kind, kind.fixed_message().unwrap_or_default())
    }

    /// The error kind.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Whether the error's kind is `kind` or descends from it.
    pub fn is(&self, kind: ErrorKind) -> bool {
        self.kind.is_subclass_of(kind)
    }

    pub(crate) fn value_error(message: impl Into<String>) -> Self {
        ApbaseError::new(ErrorKind::ValueError, message)
    }

    pub(crate) fn validation_error(message: impl Into<String>) -> Self {
        ApbaseError::new(ErrorKind::ValidationError, message)
    }

    pub(crate) fn native_execution_error(message: impl Into<String>) -> Self {
        ApbaseError::new(ErrorKind::NativeExecutionError, message)
    }
}

impl fmt::Display for ApbaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl fmt::Debug for ApbaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({:?})", self.kind, self.message)
    }
}

impl std::error::Error for ApbaseError {}

impl From<geos::Error> for ApbaseError {
    /// `GEOSException` carrying GEOS' own message (e.g.
    /// `ParseException: Unknown type: 'X'`), without the caller prefix the geos
    /// crate adds.
    fn from(error: geos::Error) -> Self {
        let message = match &error {
            geos::Error::GeosError((_, Some(message))) => message.clone(),
            other => other.to_string(),
        };
        ApbaseError::new(ErrorKind::GEOSException, message)
    }
}

/// Mapping from native status codes to concrete error classes.
pub type StatusErrors = &'static [(i32, ErrorKind)];

/// Build the error mapped to a native status code.
///
/// Unknown codes become `NativeExecutionError("{fallback_message} (status {status})")`.
pub fn raise_status_error(status: i32, status_errors: StatusErrors, fallback_message: &str) -> ApbaseError {
    match status_errors.iter().find(|(code, _)| *code == status) {
        Some((_, kind)) => ApbaseError::from_kind(*kind),
        None => ApbaseError::native_execution_error(format!("{fallback_message} (status {status})")),
    }
}

/// IDW kernel status codes.
pub const IDW_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::IDWInvalidDataError),
    (2, ErrorKind::IDWInvalidRadiusError),
    (3, ErrorKind::IDWInvalidPowerError),
    (4, ErrorKind::IDWInvalidConfigurationError),
    (5, ErrorKind::IDWAllocationError),
];

/// Ordinary kriging kernel status codes.
pub const KRIGING_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::KrigingInvalidDataError),
    (2, ErrorKind::KrigingInvalidRadiusError),
    (3, ErrorKind::KrigingInvalidConfigurationError),
    (4, ErrorKind::KrigingAllocationError),
    (5, ErrorKind::KrigingCapacityExceededError),
    (6, ErrorKind::KrigingInvalidModelError),
];

/// Cokriging kernel status codes.
pub const COKRIGING_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::CoKrigingInvalidDataError),
    (2, ErrorKind::CoKrigingInvalidRadiusError),
    (3, ErrorKind::CoKrigingInvalidConfigurationError),
    (4, ErrorKind::CoKrigingAllocationError),
    (5, ErrorKind::CoKrigingCapacityExceededError),
    (6, ErrorKind::CoKrigingInvalidModelError),
    (7, ErrorKind::CoKrigingNonPSDCoregionalizationError),
];

/// Spatial filter kernel status codes.
pub const FILTER_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::SpatialFilterInvalidDataError),
    (2, ErrorKind::SpatialFilterInvalidRadiusError),
    (3, ErrorKind::SpatialFilterInvalidConfigurationError),
    (4, ErrorKind::SpatialFilterAllocationError),
];

/// Cross-validation kernel status codes.
pub const CROSS_VALIDATE_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::CrossValidationInvalidDataError),
    (2, ErrorKind::CrossValidationInvalidRadiusError),
    (3, ErrorKind::CrossValidationInvalidPowerError),
    (4, ErrorKind::CrossValidationInvalidConfigurationError),
    (5, ErrorKind::CrossValidationAllocationError),
    (6, ErrorKind::CrossValidationInvalidModelError),
    (7, ErrorKind::CrossValidationInvalidSampleIndexError),
];

/// Variogram selection/evaluation status codes.
pub const SELECT_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::VariogramInvalidModelIdError),
    (2, ErrorKind::VariogramInvalidModelError),
    (3, ErrorKind::VariogramInvalidRangeError),
    (4, ErrorKind::VariogramInvalidThreadCountError),
];

/// Covariate screening status codes. `screen_secondary_variables` records them
/// in `CovariateReport.status` instead of raising.
pub const CROSS_VARIOGRAM_STATUS_ERRORS: StatusErrors = &[
    (1, ErrorKind::CrossVariogramInsufficientSampleError),
    (2, ErrorKind::CrossVariogramInvalidAutoModelError),
    (3, ErrorKind::CrossVariogramInvalidCrossModelError),
];

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_TABLES: &[StatusErrors] = &[
        IDW_STATUS_ERRORS,
        KRIGING_STATUS_ERRORS,
        COKRIGING_STATUS_ERRORS,
        FILTER_STATUS_ERRORS,
        CROSS_VALIDATE_STATUS_ERRORS,
        SELECT_STATUS_ERRORS,
        CROSS_VARIOGRAM_STATUS_ERRORS,
    ];

    #[test]
    fn hierarchy_follows_the_declared_parents() {
        use ErrorKind as K;
        assert!(ApbaseError::validation_error("x").is(K::ValueError));
        assert!(ApbaseError::validation_error("x").is(K::ApbaseError));
        assert!(!ApbaseError::value_error("x").is(K::ApbaseError));
        let idw = ApbaseError::from_kind(K::IDWInvalidDataError);
        assert!(idw.is(K::InterpolationError) && idw.is(K::NativeExecutionError) && idw.is(K::ApbaseError));
        assert!(!idw.is(K::ValueError) && !idw.is(K::SpatialFilterError));
        let filter = ApbaseError::from_kind(K::SpatialFilterAllocationError);
        assert!(filter.is(K::SpatialFilterError) && filter.is(K::NativeExecutionError));
        let cv = ApbaseError::from_kind(K::CrossValidationInvalidSampleIndexError);
        assert!(cv.is(K::CrossValidationError) && !cv.is(K::InterpolationError));
        let vg = ApbaseError::from_kind(K::VariogramInvalidRangeError);
        assert!(vg.is(K::VariogramError) && vg.is(K::NativeExecutionError));
        let cross = ApbaseError::from_kind(K::CrossVariogramInvalidCrossModelError);
        assert!(cross.is(K::CrossVariogramError) && cross.is(K::NativeExecutionError));
        assert!(!ApbaseError::new(K::GEOSException, "x").is(K::ApbaseError));
    }

    #[test]
    fn every_status_class_has_its_fixed_message() {
        for table in ALL_TABLES {
            for (code, kind) in table.iter() {
                let error = raise_status_error(*code, table, "fallback");
                assert_eq!(error.kind(), *kind);
                assert_eq!(error.message(), kind.fixed_message().unwrap());
                assert!(error.is(ErrorKind::NativeExecutionError));
            }
        }
    }

    #[test]
    fn unknown_status_uses_fallback_message() {
        let error = raise_status_error(42, IDW_STATUS_ERRORS, "native IDW execution failed");
        assert_eq!(error.kind(), ErrorKind::NativeExecutionError);
        assert_eq!(error.message(), "native IDW execution failed (status 42)");
    }

    #[test]
    fn display_is_the_message_and_debug_names_the_class() {
        let error = ApbaseError::from_kind(ErrorKind::KrigingInvalidRadiusError);
        assert_eq!(error.to_string(), "invalid radius");
        assert_eq!(
            format!("{error:?}"),
            "KrigingInvalidRadiusError(\"invalid radius\")"
        );
        assert_eq!(ErrorKind::IDWAllocationError.name(), "IDWAllocationError");
    }
}
