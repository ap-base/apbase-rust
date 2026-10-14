//! Scalar validation helpers shared by the high-level classes.

use super::exceptions::{ApbaseError, Result};

/// Validate that a scalar value is finite and strictly positive.
pub fn require_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(ApbaseError::validation_error(format!(
            "{name} must be finite and greater than zero"
        )));
    }
    Ok(())
}

/// Validate that a scalar value is finite and non-negative.
pub fn require_non_negative_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ApbaseError::validation_error(format!(
            "{name} must be finite and >= 0"
        )));
    }
    Ok(())
}

/// Validate local-neighborhood bounds, optionally against a global ceiling.
pub fn require_neighbor_bounds(
    max_neighbors: i64,
    min_neighbors: i64,
    max_neighbors_ceiling: Option<i64>,
) -> Result<()> {
    if max_neighbors <= 0 {
        return Err(ApbaseError::validation_error(
            "max_neighbors must be greater than zero",
        ));
    }
    if let Some(ceiling) = max_neighbors_ceiling {
        if max_neighbors > ceiling {
            return Err(ApbaseError::validation_error(format!(
                "max_neighbors must be <= {ceiling}"
            )));
        }
    }
    if min_neighbors <= 0 || min_neighbors > max_neighbors {
        return Err(ApbaseError::validation_error(
            "min_neighbors must be between 1 and max_neighbors",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::exceptions::ErrorKind;

    #[test]
    fn neighbor_bound_messages() {
        let message = |max, min, ceiling| {
            require_neighbor_bounds(max, min, ceiling)
                .unwrap_err()
                .to_string()
        };
        assert_eq!(message(0, 1, None), "max_neighbors must be greater than zero");
        assert_eq!(message(600, 3, Some(500)), "max_neighbors must be <= 500");
        assert_eq!(
            message(10, 0, None),
            "min_neighbors must be between 1 and max_neighbors"
        );
        assert_eq!(
            message(10, 11, None),
            "min_neighbors must be between 1 and max_neighbors"
        );
        assert!(require_neighbor_bounds(500, 500, Some(500)).is_ok());
        let error = require_neighbor_bounds(0, 1, None).unwrap_err();
        assert!(error.is(ErrorKind::ValidationError) && error.is(ErrorKind::ValueError));
    }

    #[test]
    fn scalar_check_messages() {
        assert!(require_positive_finite(1.0, "radius").is_ok());
        assert_eq!(
            require_positive_finite(0.0, "radius").unwrap_err().to_string(),
            "radius must be finite and greater than zero"
        );
        assert!(require_positive_finite(f64::NAN, "radius").is_err());
        assert!(require_non_negative_finite(0.0, "tol").is_ok());
        assert_eq!(
            require_non_negative_finite(-1.0, "tol").unwrap_err().to_string(),
            "tol must be finite and >= 0"
        );
        assert!(require_non_negative_finite(f64::INFINITY, "tol").is_err());
    }
}
