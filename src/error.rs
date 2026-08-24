//! Error type shared by the saddle sessions.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum SaddleError {
    #[error("shape mismatch: {0}")]
    Shape(String),
    #[error("surface evaluation failed: {0}")]
    Surface(String),
    #[error("non-finite value in {0}")]
    NonFinite(&'static str),
    #[error("solver step failed: {0}")]
    Solver(String),
}
