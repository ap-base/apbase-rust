//! Leave-one-out cross-validation for IDW, ordinary kriging, and cokriging.
//!
//! Evaluates IDW/kriging and collocated/ICM/LMC cokriging accuracy (MAE, RMSE,
//! R2) on the same source dataset, plus [`select_best_model`] to pick the
//! statistically better method from either result.

mod _cross_validation;
mod _native;

pub use _cross_validation::{
    BestModelResult, CokrigingCrossValidationResult, CrossValidateCokrigingOptions, CrossValidateOptions,
    CrossValidationResult, Method, MethodCrossValidation, Metric, MultiMethodBestModelResult,
    SelectBestModelInput, SelectBestModelOptions, cross_validate, cross_validate_cokriging,
    select_best_model,
};
