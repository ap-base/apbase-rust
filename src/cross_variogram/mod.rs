//! Cross-variogram fitting and secondary-variable screening for cokriging.
//!
//! Runs *before* `apbase::cokriging`: fits a generic cross-variogram model
//! between any two variables, and screens candidate secondary variables against
//! a primary via a Fisher z-test (significance) and a Cauchy-Schwarz check
//! (admissibility).

mod _cross_variogram;
mod _native;
mod _screening;

pub use _cross_variogram::{CrossVariogram, CrossVariogramOptions};
pub use _screening::{
    Correction, CovariateReport, ScreeningOptions, ScreeningResult, VariableXYZ, screen_secondary_variables,
};
