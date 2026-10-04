//! Band, minimum-mode, and index-1 Newton saddle mechanics.
//!
//! The band and the minimum-mode search take one [`rgmin::Solver`]
//! step on a force this crate assembled. The host owns the loop.
//! [`band::BandSession::run`] and [`minmode::MinModeSession::run`]
//! are convenience loops over `step`.
//!
//! [`nichols::Index1Session`] is the index-1 search on a dense
//! Hessian: Baker's restricted-step partitioned RFO (maximize the
//! lowest mode, minimize the rest, with the step inside a Euclidean
//! trust sphere) and a Powell or Bofill update. The i-PI Nichols
//! shift remains [`nichols::nichols_step`] for a host that already
//! holds a spectrum.
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
pub mod prfo;
pub mod projection;
pub mod solid_state;
pub mod spring;
pub mod tangent;

pub use band::{
    BandConfig, BandEvaluation, BandReport, BandSession, BandStatus, BandSurface, Cell, CiConfig,
    SolidState,
};
pub use error::SaddleError;
pub use minmode::{
    FiniteDifference, MinModeConfig, MinModeKind, MinModeReport, MinModeSession, MinModeStatus,
    ModeEstimate, PointSurface,
};
pub use nichols::{
    HessianUpdate, Index1Config, Index1Report, Index1Session, Index1Status, NicholsMode,
    bofill_update, cap_max_abs, nichols_displacement, nichols_step, powell_update,
};
pub use prfo::{PrfoKind, partitioned_rfo_eigen, restricted_prfo_displacement};
pub use projection::ProjectionKind;
pub use spring::SpringKind;
pub use tangent::TangentKind;

pub mod cell_log;
pub mod constraints;
pub mod eigensolve;
pub mod force;
pub mod force_match;
pub mod geom;
pub mod internal;
pub mod kappa;
pub mod linalg;
pub mod mic;
pub mod pes;
pub mod pes_internal;
pub mod prfo_restricted;
pub mod qn;
pub mod restricted;
pub mod rfo;
pub mod samd;
pub mod sella_min;
pub mod sella_saddle;
pub mod vocn;

pub use cell_log::{CellChart, expm_3x3, logm_3x3};

pub use constraints::{Constraints, Equality, InternalCounts};

pub use eigensolve::{
    EigenDevice, ExpandKind, eigh_on, exact_eigh, expand, lowest_on, rayleigh_ritz,
    rayleigh_ritz_iter,
};

pub use force::ForceGate;

pub use force_match::{covalent_pairs, force_match_hessian};

pub use geom::{SellaGeom, TrustSchedule};

pub use internal::{CartAxis, Displacement, InternalSlot, Rotation, Translation};

pub use linalg::{
    NumericalHessian, modified_gram_schmidt, numerical_hvp, numerical_hvp_on, numerical_hvp_proj,
    project_hvp, retract_hvp, transport_hvp,
};

pub use pes::CartesianPes;

pub use pes_internal::{
    CellCartesianPes, CellInternalPes, InternalPes, SellaPes, niggli_reduce_cell,
    niggli_reduce_vectors, place_perp_dummy,
};

pub use qn::{
    HessUpdate, QuasiNewton, StepperKind, get_stepper, retract_qn, symmetrize_y, symmetrize_y_cols,
    symmetrize_y_pair, update_h, update_h_ms, update_hessian, update_hessian_cols,
};

pub use restricted::{
    InternalWeights, MaxInternalStep, RAS_SYNONYMS, RestrictedAtomicStep, RestrictedKind,
    TRUST_SYNONYMS, TrustRegion, mis_clip, ras_cons, weights_for_equalities,
};

pub use rfo::{RationalFunctionOptimization, rfo_stepper};

pub use samd::{
    SamdConfig, SamdReport, SamdSession, VelocitySofteningConfig, VelocitySofteningReport,
    project_velocity, retract_samd, soften_velocity_on, transport_velocity, verlet_increment,
};

pub use sella_min::{SellaMinConfig, SellaMinReport, SellaMinSession};

pub use sella_saddle::{SellaSaddleConfig, SellaSaddleReport, SellaSaddleSession};

pub use vocn::{Found as VocnFound, Primitive as VocnPrimitive};

pub use prfo::{
    PartitionedRationalFunctionOptimization, prfo_get_s, prfo_stepper, prfo_trust_region,
};

pub mod irc;
pub use irc::{IrcConfig, IrcDirection, IrcKind, IrcReport, IrcSession};

#[cfg(feature = "capi")]
pub mod capi_kappa;

pub mod afir;
pub use afir::{afir_force, try_afir_force};

pub mod rtr;
pub use rtr::{
    BandForces, BandRtr, RtrConfig, RtrRadius, RtrReport, TcgResult, TcgStop, band_forces,
    reparametrize_equal_arc, truncated_cg,
};

#[cfg(feature = "readcon")]
pub mod io;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "readcon")]
pub use io::{MolecularFrame, frame_from_con};

pub mod gpu;
pub use gpu::{
    DEFAULT_MIN_DIM, GPU_MIN_DIM, GpuPolicy, clear_oom_floor, cuda_available, cuda_device,
    fail_next_cuda_claim, gpu_eigh, gpu_eigh_env, gpu_eigh_t, gpu_eigh_with, gpu_ok, gpu_ok_env,
    gpu_project, gpu_project_env, gpu_project_with, gpu_qr, gpu_qr_env, gpu_qr_with,
    lock_oom_for_test, oom_floor, record_oom, to_gpu, to_gpu_matrix, to_gpu_view,
};

pub mod trust;
