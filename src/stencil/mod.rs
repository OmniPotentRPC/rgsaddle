//! Three-point stencil shared by a band image and a minimum-mode centre.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Eq. 9: the dimer translation and the
//! climbing image are one Householder reflection of the true force.
//! Goswami (2026), arXiv:2603.10992, Eqs. 2.14-2.16: the dimer is that
//! reflection on a three-image chain with free ends.
//!
//! Band presets run on [`Chain`] and [`Triad`]. Zoom-NEB packs with
//! fixed window ends unless the builder asks for perpendicular ends.
//! Energy-weighted springs follow Ásgeirsson et al., J. Chem. Theory
//! Comput. 17, 4929 (2021), doi:10.1021/acs.jctc.1c00462.

mod band_run;
mod batch;
mod builder;
mod chain;
mod control;
mod surface;
mod triad;

pub mod spring_ew;
pub mod zoom;

pub use batch::{IdBook, ids_for_callback};
pub(crate) use batch::{note_image_insert, publish_indices};
pub use builder::{
    BandBuilder, Climb, PointBuilder, Preset, Stencil, StencilReport, StencilStatus,
};
pub use chain::{Chain, EVAL_BLOCK, EvalPool, Mesh, MeshEvent};
pub use control::{Action, ArtControl, Controller, Fixed, Report, Role, StencilView};
pub use spring_ew::{K_MAX_EV_PER_A2, K_MIN_EV_PER_A2};
pub use surface::{FromBand, FromPoint, ImageId, Surface};
pub use triad::{
    Coupling, CouplingForces, Delta, Euclidean, Metric, Point, Sides, TangentRule, Triad, ascend,
    coupling_forces, curvature_along, perp, reflect, unit_tangent,
};
pub use zoom::{Interp, ZoomEnds, ZoomForce, ZoomParams, ZoomWatch, ZoomWindow};

pub const PHASE_BAND: i32 = 0;
pub const PHASE_ZOOM: i32 = 1;
pub const PHASE_INSERT: i32 = 2;
