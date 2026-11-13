//! Error type shared by the saddle sessions.

use thiserror::Error;

#[derive(Debug, Error)]
/// The SaddleError.
pub enum SaddleError {
    #[error("shape mismatch: {0}")]
    /// The Shape.
    Shape(String),
    #[error("surface evaluation failed: {0}")]
    /// The Surface.
    Surface(String),
    #[error("non-finite value in {0}")]
    /// The str.
    NonFinite(&'static str),
    #[error("solver step failed: {0}")]
    /// The Solver.
    Solver(String),
    #[error("invalid parameter: {0}")]
    /// The Invalid.
    Invalid(String),
    /// The solver returned the state unchanged while the force is above
    /// its tolerance: the step can no longer make progress.
    #[error("solver step left the state unchanged: {0}")]
    Stalled(String),
}
