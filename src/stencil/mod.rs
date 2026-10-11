//! Three-point stencil shared by a band image and a minimum-mode centre.
//!
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Eq. 9: the dimer translation and the
//! climbing image are one Householder reflection of the true force.
//! Goswami (2026), arXiv:2603.10992, Eqs. 2.14-2.16: the dimer is that
//! reflection on a three-image chain with free ends.

mod triad;

pub use triad::{
    Coupling, CouplingForces, Delta, Euclidean, Metric, Point, Sides, TangentRule, Triad, ascend,
    coupling_forces, curvature_along, perp, reflect, unit_tangent,
};
