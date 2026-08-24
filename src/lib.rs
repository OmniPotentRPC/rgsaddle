//! Band and minimum-mode saddle mechanics over rgmin steppers.
//!
//! Two seams, stepping at both:
//!
//! - The **inner seam** is rgmin's [`rgmin::Solver`]: one optimizer
//!   step over an assembled band force. rgsaddle defines no optimizer
//!   of its own.
//! - The **outer seam** is [`band::BandSession::step`]: assemble NEB
//!   forces on the caller's surface, take one solver step, report.
//!   There is no run-to-completion contract; hosts own the loop and
//!   interleave their policy (trust, drift, acquisition, MMF)
//!   between steps. [`band::BandSession::run`] is a convenience loop
//!   over `step` and nothing more.
//!
//! The same shape carries the minimum-mode search
//! ([`minmode::MinModeSession`]): refresh the lowest curvature mode
//! (dimer rotation or Lanczos over finite-difference Hessian
//! actions), invert the force along it, take one solver step.
//!
//! Force assembly is pure: tangents (Mills–Jonsson–Schenter simple,
//! Henkelman–Jonsson improved), springs (uniform, energy-weighted,
//! Onsager–Machlup), projections (plain elastic band, NEB,
//! doubly-nudged), and the climbing-image force, ported from eOn's
//! NEBTangent / NEBSpringForce / NEBForceProjection with identical
//! branch structure.
//!
//! Positions are unwrapped Cartesian; a periodic host applies its
//! minimum-image convention before handing differences in.

pub mod band;
#[cfg(feature = "capi")]
pub mod capi;
pub mod error;
pub mod minmode;
pub mod projection;
pub mod spring;
pub mod tangent;

pub use band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface};
pub use error::SaddleError;
pub use minmode::{
    MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus, PointSurface,
};
pub use projection::ProjectionKind;
pub use spring::SpringKind;
pub use tangent::TangentKind;
