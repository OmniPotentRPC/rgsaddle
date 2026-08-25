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
//! on [`pes::CartesianPes`] (Cartesian BFGS, per-atom `||F||_2`,
//! Sella `proj_trans` / `proj_rot` via [`CartesianPes::with_proj`])
//! or [`pes_internal::InternalPes`] (Sella `InternalPES`).
//! Sella `QuasiNewton` is [`qn::QuasiNewton`]: `rgmin::qn_get_s`
//! with proj / retr / transp. Sella `RationalFunctionOptimization` is
//! [`rfo::RationalFunctionOptimization`]: `rgmin::rfo_get_s` with
//! the Sella alpha / order contract, retracted through
//! [`rgmin::Manifold`] `project` / `retract` / `transport`.
//! Sella `PartitionedRationalFunctionOptimization` is
//! [`prfo::PartitionedRationalFunctionOptimization`]: RFO in the
//! `order` uphill modes plus RFO in the downhill complement, over
//! [`rgmin::rfo_get_s`] / [`rgmin::prfo_restricted`].
//! Equality internals are [`constraints::Constraints`] on the same
//! manifold (`Translation` / `Rotation` / `Displacement`, plus
//! host-fixed bonds / angles / dihedrals). [`SellaMinSession`] and
//! [`SellaSaddleSession`] retract on [`geom::SellaGeom`]: the rigid
//! quotient by default, or a live `Constraints` chart. [`restricted::TrustRegion`]
//! is Sella `cons(s) = ||s||` over [`rgmin::qn_restricted`].
//! [`restricted::RestrictedAtomicStep`] is Sella
//! `cons(s) = max_i ||s_i||` over [`rgmin::ras_clip`] (Cartesian
//! only; internals refuse it). [`restricted::MaxInternalStep`] is
//! the per-coordinate clip Sella applies before that Euclidean
//! radius.
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
pub mod cell_log;
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
pub mod pes;
pub mod pes_internal;
pub mod prfo;
pub mod projection;
#[cfg(feature = "python")]
pub mod python;
pub mod qn;
pub mod restricted;
pub mod rfo;
pub mod samd;
pub mod sella_min;
pub mod sella_saddle;
pub mod spring;
pub mod tangent;
pub mod vocn;

pub use band::{BandConfig, BandReport, BandSession, BandStatus, BandSurface};
pub use cell_log::{CellChart, expm_3x3, logm_3x3};
pub use constraints::{Constraints, Equality, InternalCounts};
pub use eigensolve::{
    EigenDevice, ExpandKind, eigh_on, exact_eigh, expand, lowest_on, rayleigh_ritz,
    rayleigh_ritz_iter,
};
pub use error::SaddleError;
pub use force::ForceGate;
pub use force_match::{covalent_pairs, force_match_hessian};
pub use geom::{SellaGeom, TrustSchedule};
pub use internal::{CartAxis, Displacement, InternalSlot, Rotation, Translation};
pub use irc::{IrcConfig, IrcDirection, IrcKind, IrcReport, IrcSession};
pub use linalg::{
    NumericalHessian, modified_gram_schmidt, numerical_hvp, numerical_hvp_on, numerical_hvp_proj,
    project_hvp, retract_hvp, transport_hvp,
};
pub use mic::{Cell, wrap_difference};
pub use minmode::{
    MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus, PointSurface,
};
pub use pes::CartesianPes;
pub use pes_internal::{
    CellCartesianPes, CellInternalPes, InternalPes, SellaPes, niggli_reduce_cell,
    niggli_reduce_vectors, place_perp_dummy,
};
pub use projection::ProjectionKind;
pub use qn::{
    HessUpdate, QuasiNewton, StepperKind, get_stepper, retract_qn, symmetrize_y, update_h,
    update_h_ms,
};

pub use prfo::{PartitionedRationalFunctionOptimization, prfo_stepper};
pub use restricted::{
    InternalWeights, MaxInternalStep, RAS_SYNONYMS, RestrictedAtomicStep, RestrictedKind,
    TRUST_SYNONYMS, TrustRegion, mis_clip, ras_cons, weights_for_equalities,
};
pub use rfo::{RationalFunctionOptimization, rfo_stepper};
pub use samd::{
    SamdConfig, SamdReport, SamdSession, project_velocity, retract_samd, transport_velocity,
};
pub use sella_min::{SellaMinConfig, SellaMinReport, SellaMinSession};
pub use sella_saddle::{SellaSaddleConfig, SellaSaddleReport, SellaSaddleSession};
pub use spring::SpringKind;
pub use tangent::TangentKind;
pub use vocn::{Found as VocnFound, Primitive as VocnPrimitive};
