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
//! ([`minmode::MinModeSession`]), the IRC roll-down
//! ([`irc::IrcSession`]), and Sella order-0 / order-1 sessions
//! ([`sella_min::SellaMinSession`], [`sella_saddle::SellaSaddleSession`]):
//! kick along the imaginary mode, then take Gonzalez--Schlegel /
//! Sella steps on the mass-weighted sphere (`rgmin::IrcTrust` over
//! [`rgmin::ManifoldKind::MwRigid`]). SellaMin is QN + TrustRegion
//! on [`pes::CartesianPes`] (Cartesian BFGS, per-atom `||F||_2`) or
//! [`pes_internal::InternalPes`] (Sella `InternalPES`).
//! Sella `QuasiNewton` is [`qn::QuasiNewton`]: `rgmin::qn_get_s`
//! with proj / retr / transp. Sella `RationalFunctionOptimization` is
//! [`rfo::RationalFunctionOptimization`]: `rgmin::rfo_get_s` with
//! the Sella alpha / order contract, retracted through
//! [`rgmin::Manifold`] `project` / `retract` / `transport`.
//! Equality internals are [`constraints::Constraints`] on the same
//! manifold (`Translation` / `Rotation` / `Displacement`, plus
//! host-fixed bonds / angles / dihedrals). [`SellaMinSession`] and
//! [`SellaSaddleSession`] retract on [`geom::SellaGeom`]: the rigid
//! quotient by default, or a live `Constraints` chart. [`restricted::MaxInternalStep`]
//! is the per-coordinate clip Sella applies before the trust region.
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
#[cfg(feature = "readcon")]
pub mod io;
pub mod irc;
#[cfg(feature = "readcon")]
pub use io::{MolecularFrame, frame_from_con};
pub mod capi;
pub mod constraints;
pub mod eigensolve;
pub mod error;
pub mod force;
pub mod force_match;
pub mod geom;
pub mod internal;
pub mod linalg;
pub mod mic;
pub mod minmode;
#[cfg(feature = "python")]
pub mod python;
pub mod pes;
pub mod pes_internal;
pub mod projection;
pub mod qn;
pub mod restricted;
pub mod rfo;
pub mod samd;
pub mod sella_min;
pub mod sella_saddle;
pub mod spring;
pub mod tangent;

pub use band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface};
pub use constraints::{Constraints, Equality, InternalCounts};
pub use eigensolve::{
    exact_eigh, eigh_on, expand, lowest_on, rayleigh_ritz, rayleigh_ritz_iter, EigenDevice,
    ExpandKind,
};
pub use error::SaddleError;
pub use force::ForceGate;
pub use force_match::{covalent_pairs, force_match_hessian};
pub use geom::{SellaGeom, TrustSchedule};
pub use linalg::{modified_gram_schmidt, numerical_hvp};
pub use internal::{CartAxis, Displacement, InternalSlot, Rotation, Translation};
pub use irc::{IrcConfig, IrcDirection, IrcKind, IrcReport, IrcSession};
pub use mic::{Cell, wrap_difference};
pub use minmode::{
    MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus, PointSurface,
};
pub use pes::{CartesianPes, HessUpdate};
pub use pes_internal::{
    niggli_reduce_cell, niggli_reduce_vectors, CellCartesianPes, CellInternalPes, InternalPes,
    SellaPes,
};
pub use projection::ProjectionKind;
pub use qn::{get_stepper, retract_qn, QuasiNewton, StepperKind};
pub use restricted::{mis_clip, InternalWeights, MaxInternalStep, RestrictedKind};
pub use rfo::{RationalFunctionOptimization, rfo_stepper};
pub use samd::{SamdConfig, SamdReport, SamdSession};
pub use sella_min::{SellaMinConfig, SellaMinReport, SellaMinSession};
pub use sella_saddle::{SellaSaddleConfig, SellaSaddleReport, SellaSaddleSession};
pub use spring::SpringKind;
pub use tangent::TangentKind;
