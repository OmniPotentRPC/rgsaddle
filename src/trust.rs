//! Dense quadratic steps restricted to a Euclidean trust ball.
//!
//! The returned step includes its boundary and norm information.
//! The restriction controller and eigensolver are provided by rgmin.

pub use rgmin::{RestrictedStep, TrustRegion};
