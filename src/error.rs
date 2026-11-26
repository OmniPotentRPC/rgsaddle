//! Error type shared by the saddle sessions.

use thiserror::Error;

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
/// Failure returned by a saddle search, surface callback, or estimator.
pub enum SaddleError {
    #[error("shape mismatch: {0}")]
    /// An array or geometry has the wrong dimensions.
    Shape(String),
    #[error("surface evaluation failed: {0}")]
    /// The host potential or force callback failed.
    Surface(String),
    #[error("non-finite value in {0}")]
    /// A named input or computed quantity is NaN or infinite.
    NonFinite(&'static str),
    #[error("solver step failed: {0}")]
    /// The optimizer could not produce a valid step.
    Solver(String),
    #[error("invalid parameter: {0}")]
    /// A parameter is outside its documented domain.
    Invalid(String),
    /// ART lost the tracked index-1 direction and exhausted its
    /// configured recovery attempts while the rejected curvature was
    /// still negative or not finite. A finite non-negative one finishes
    /// the event as [`crate::ArtStatus::LostCurvature`] instead.
    #[error(
        "ART lost its index-1 mode after {attempts} recovery attempts (curvature {curvature:.3e}, overlap {overlap:.3e})"
    )]
    ArtIndexOneLost {
        /// Recovery attempts made before returning the error.
        attempts: usize,
        /// Curvature reported for the rejected mode.
        curvature: f64,
        /// Absolute overlap with the last accepted index-1 mode.
        overlap: f64,
    },
    /// Automated equilibration selected a tail whose two halves still drift.
    #[error(
        "constrained ensemble did not equilibrate after {equilibration_steps} steps: split-half drift {drift:.3e} exceeds {sigma_multiple:.1} standard errors ({standard_error:.3e})"
    )]
    NotEquilibrated {
        /// Engine steps discarded, including the configured minimum floor.
        equilibration_steps: usize,
        /// Absolute difference of the two half means.
        drift: f64,
        /// Root-sum-square block standard error of those means.
        standard_error: f64,
        /// Required multiple of `standard_error`.
        sigma_multiple: f64,
    },
    /// The retained tail is too correlated to support the requested blocks.
    #[error(
        "constrained ensemble has only {n_eff:.3} effective samples after {equilibration_steps} steps; at least {required:.3} are required"
    )]
    InsufficientEffectiveSamples {
        /// Engine steps discarded, including the configured minimum floor.
        equilibration_steps: usize,
        /// Effective sample count of the selected tail.
        n_eff: f64,
        /// Minimum effective sample count.
        required: f64,
    },
    /// The solver returned the state unchanged while the force is above
    /// its tolerance: the step can no longer make progress.
    #[error("solver step left the state unchanged: {0}")]
    Stalled(String),
    /// A dense factorization or spectrum hit a zero or underflowed pivot or mode.
    #[error("singular matrix at pivot or mode {0}")]
    Singular(usize),
}
