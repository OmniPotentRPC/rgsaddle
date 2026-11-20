//! Error type shared by the saddle sessions.

use thiserror::Error;

use crate::minmode::CartesianLayout;

#[derive(Debug, Error)]
/// The SaddleError.
pub enum SaddleError {
    #[error("shape mismatch: {0}")]
    /// The Shape.
    Shape(String),
    #[error("surface evaluation failed: {0}")]
    /// The Surface.
    Surface(String),
    #[error("non-finite value in {0}")]
    /// The str.
    NonFinite(&'static str),
    #[error("solver step failed: {0}")]
    /// The Solver.
    Solver(String),
    #[error("invalid parameter: {0}")]
    /// The Invalid.
    Invalid(String),
    /// Sampling requested zero blocks.
    #[error("{context} needs at least one block")]
    MissingBlocks {
        /// Which sampler or integrator reported this.
        context: &'static str,
    },
    /// Sampling requested zero records.
    #[error("{context} needs at least one sample")]
    MissingSamples {
        /// Which sampler or integrator reported this.
        context: &'static str,
    },
    /// Ring excluded-mode assembly produced no bead basis.
    #[error("ring excluded-mode assembly produced no bead basis")]
    MissingExcludedBasis,
    /// PAFI was used before an overdamped time step was selected.
    #[error("PAFI needs an explicit positive time step")]
    MissingPafiTimeStep,
    /// Cartesian drift removal does not match the coordinate layout.
    #[error(
        "PAFI centre-of-geometry removal has {coordinates} coordinates for {atoms} configured atoms"
    )]
    PafiComLayoutMismatch {
        /// Coordinates supplied to the sampler.
        coordinates: usize,
        /// Configured real atom count.
        atoms: usize,
    },
    /// The configured atom count disagrees with the surface.
    #[error(
        "PAFI centre-of-geometry atom count {configured} disagrees with surface count {surface}"
    )]
    PafiComAtomCountMismatch {
        /// Atom count from `PafiConfig`.
        configured: usize,
        /// Atom count certified by the surface.
        surface: usize,
    },
    /// Cartesian drift removal was requested for an unsupported layout.
    #[error("PAFI centre-of-geometry removal is unsupported for {layout}")]
    UnsupportedPafiComLayout {
        /// Layout certified by the surface.
        layout: CartesianLayout,
    },
    /// A requested path coordinate lies outside the stored path.
    #[error("PAFI arc {arc:.16e} lies outside [{start:.16e}, {end:.16e}]")]
    PafiArcOutsidePath {
        /// Requested arc coordinate.
        arc: f64,
        /// First stored arc.
        start: f64,
        /// Last stored arc.
        end: f64,
    },
    /// Adaptive PAFI sampling exceeded the addressable sample count.
    #[error("adaptive PAFI sample count exceeds usize")]
    PafiSampleCountOverflow,
    /// A requested ring time step exceeded the stability bound.
    #[error(
        "ring time step {provided:.16e} exceeds stability bound {bound:.16e}; lower time_step or compute an explicit automatic step"
    )]
    TimeStepAboveBound {
        /// Caller-provided time step.
        provided: f64,
        /// Bound from the ring stiffness estimate.
        bound: f64,
    },
    /// The ring-polymer inverse temperature and the sampling temperature differ.
    #[error(
        "ring-polymer beta and PAFI thermal disagree: beta={beta:.16e}, thermal={thermal:.16e}, expected thermal={expected_thermal:.16e}"
    )]
    TemperatureMismatch {
        /// Inverse temperature in `RingSettings`.
        beta: f64,
        /// Thermal energy in `PafiConfig`.
        thermal: f64,
        /// `1/beta`.
        expected_thermal: f64,
    },
    /// A ring stiffness bound was requested without physical images.
    #[error("ring stiffness needs at least one physical image")]
    MissingCurvatureImages,
    /// The bead surface did not provide a Hessian-vector product.
    #[error("ring stiffness needs a Hessian-vector product at image {image}")]
    MissingHessianVector {
        /// Zero-based physical-image index.
        image: usize,
    },
    /// The supplied images and ring springs have no positive stiffness bound.
    #[error("ring stiffness bound must be positive, got {bound:.16e}")]
    NonPositiveStiffnessBound {
        /// Computed image-local Gershgorin-plus-spring bound.
        bound: f64,
    },
    /// Centre-of-geometry removal was requested for ring-polymer coordinates.
    #[error(
        "centre-of-geometry removal is unsupported for ring-polymer coordinates with {dof} coordinates per bead; set fix_com=false"
    )]
    UnsupportedRingCenterOfMass {
        /// Coordinates in one physical bead.
        dof: usize,
    },
    /// Automated equilibration selected a tail whose two halves still drift.
    #[error(
        "constrained ensemble did not equilibrate after {equilibration_steps} steps: split-half drift {drift:.3e} exceeds {sigma_multiple:.1} standard errors ({standard_error:.3e})"
    )]
    NotEquilibrated {
        /// Engine steps discarded, including the configured minimum floor.
        equilibration_steps: usize,
        /// Absolute difference of the two half means.
        drift: f64,
        /// Root-sum-square block standard error of those means.
        standard_error: f64,
        /// Required multiple of `standard_error`.
        sigma_multiple: f64,
    },
    /// The retained tail is too correlated to support the requested blocks.
    #[error(
        "constrained ensemble has only {n_eff:.3} effective samples after {equilibration_steps} steps; at least {required:.3} are required"
    )]
    InsufficientEffectiveSamples {
        /// Engine steps discarded, including the configured minimum floor.
        equilibration_steps: usize,
        /// Effective sample count of the selected tail.
        n_eff: f64,
        /// Minimum effective sample count.
        required: f64,
    },
    /// The solver returned the state unchanged while the force is above
    /// its tolerance: the step can no longer make progress.
    #[error("solver step left the state unchanged: {0}")]
    Stalled(String),
    /// A caller-supplied step or evaluation budget was used up.
    #[error("work budget exhausted: {0}")]
    Exhausted(String),
}
