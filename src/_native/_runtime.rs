//! Native runtime defaults applied before the first kernel call.

const BLAS_THREAD_ENV_VARS: &[&str] = &["OPENBLAS_NUM_THREADS", "GOTO_NUM_THREADS"];

/// Use one BLAS thread by default unless the caller configured the native libs.
///
/// The libraries are already loaded when this runs, so it returns the thread
/// count to apply through the BLAS API (`None` when the user set one of the
/// variables). OpenMP needs nothing: every kernel passes an explicit
/// `num_threads(thread_count)`.
pub(crate) fn configure_native_threading() -> Option<i32> {
    if BLAS_THREAD_ENV_VARS
        .iter()
        .any(|name| std::env::var_os(name).is_some())
    {
        None
    } else {
        Some(1)
    }
}
