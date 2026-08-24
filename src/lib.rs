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
//! ([`minmode::MinModeSession`]) and the IRC roll-down
//! ([`irc::IrcSession`]): kick along the imaginary mode, then take
//! Gonzalez--Schlegel / Sella steps on the mass-weighted sphere
//! (`rgmin::IrcTrust` over [`rgmin::ManifoldKind::MwRigid`]).
//!
//! Force assembly is pure: tangents (Mills–Jonsson–Schenter simple,
//! Henkelman–Jonsson improved), springs (uniform, energy-weighted,
//! Onsager–Machlup), projections (plain elastic band, NEB,
//! doubly-nudged), and the climbing-image force, ported from eOn's
//! NEBTangent / NEBSpringForce / NEBForceProjection with identical
//! branch structure.
//!
//! Positions are unwrapped Cartesian. Minimum-image differences go
//! through [linkcell](https://github.com/d-SEAMS/linkcell). Molecular
//! frames come from readcon-core (CON), readcon-db (corpus), and
//! readcon-chemfiles (foreign trajectories). Cutoff neighbour lists
//! are vesin, when a surface needs them.

pub mod band;
pub mod irc;
#[cfg(feature = "readcon")]
pub mod io;
#[cfg(feature = "readcon")]
pub use io::{frame_from_con, MolecularFrame};
#[cfg(feature = "capi")]
pub mod capi;
pub mod error;
pub mod mic;
pub mod minmode;
pub mod projection;
pub mod spring;
pub mod tangent;

pub use band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface};
pub use error::SaddleError;
pub use mic::{wrap_difference, Cell};
pub use irc::{IrcConfig, IrcDirection, IrcKind, IrcReport, IrcSession};
pub use minmode::{
    MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus, PointSurface,
};
pub use projection::ProjectionKind;
pub use spring::SpringKind;
pub use tangent::TangentKind;
