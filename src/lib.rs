//! Band, minimum-mode, and index-1 Newton saddle mechanics.
//!
//! The band and the minimum-mode search take one [`rgmin::Solver`]
//! step on a force this crate assembled. The host owns the loop.
//! [`band::BandSession::run`] and [`minmode::MinModeSession::run`]
//! are convenience loops over `step`.
//!
//! [`nichols::Index1Session`] is the index-1 search on a dense
//! Hessian: the Nichols displacement (climb the lowest mode, descend
//! the rest), a max-abs trust cap, and a Powell or Bofill update.
//! A host that already holds a spectrum, dense or banded, calls
//! [`nichols::nichols_step`] with that spectrum.
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
pub mod nichols;
pub mod projection;
pub mod spring;
pub mod tangent;

pub use band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface};
pub use error::SaddleError;
pub use minmode::{
    MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus, PointSurface,
};
pub use nichols::{
    HessianUpdate, Index1Config, Index1Report, Index1Session, Index1Status, NicholsMode,
    bofill_update, cap_max_abs, nichols_displacement, nichols_step, powell_update,
};
pub use projection::ProjectionKind;
pub use spring::SpringKind;
pub use tangent::TangentKind;
