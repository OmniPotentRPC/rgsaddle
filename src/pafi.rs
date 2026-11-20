//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. [`pafi_profile`]
//! takes the spline through the images of [`ArcPath`]. The hyperplane at
//! `s` is `(x - X) · t = 0`, with `t = X'` less its center-of-mass part
//! when the center of mass is held. With `ψ = (x - X) · X'' / |t|²`,
//!
//! `dF/ds = ⟨ (∇E · t) (1 - ψ) ⟩`.
//!
//! The paper's `ψ`, its Eq. (5), is `1 - ψ` here. [`PafiMeanForce`] selects
//! either the complete Eq. (8), including its analytic Jacobian derivative,
//! or the force term alone, which is what LAMMPS `fix pafi` and Swinburne's
//! public driver record. The force-only integral is the free energy of the
//! flat hyperplanes,
//! `-kT ln ∫ δ((x - X) · t / |t|) exp(-E/kT) dx`, which does not depend on
//! how the path is parametrized.
//!
//! [`pafi_frames`] samples the Euclidean surface measure on the flat
//! hyperplane `(x - X) · ν = 0`. With `τ = dX/dr` the path derivative and
//! `ν` the unit constraint normal,
//!
//! `dF/dr = ⟨ (∇E · ν) (τ · ν - (x - X) · dν/dr) ⟩`.
//!
//! When `ν` is the unit tangent and `r` is its arc length, `τ · ν = 1`
//! and this is the multiply form above. A normal that keeps only some of
//! the atoms is a different foliation, and [`pafi_frames`] therefore uses
//! the full tangent. Rigid translations are removed from it by
//! `HyperplaneConstraint::orthogonal_to`, and the estimator uses that same
//! unit normal.
//!
//! Production samples of the spline path arrive through
//! [`HyperplaneSampler`]. The built-in overdamped Langevin implementation,
//! the paper's Eq. (9), is the deterministic analytic-surface reference,
//! and the frame chains use the same step. [`PafiIntegrator`] selects the
//! direct Euler-Maruyama discretization or the step of Leimkuhler and
//! Matthews, Appl. Math. Res. Express 2013, 34 (2013),
//! doi:10.1093/amrx/abs010, Eq. (3). The latter sums two consecutive noise
//! draws, which makes stationary averages second order in the step and
//! exact on a harmonic plane. The walk projects drift and noise, then
//! restores the plane after every step. Holding the center of mass is
//! optional and is the `com yes` choice of `fix pafi`. The integral of
//! `dF/ds` from the reactant is the potential of mean force. One
//! trajectory uses block averages of the equilibrated records for its
//! standard error. An ensemble of trajectories uses the spread of
//! independent trajectory means, as Swinburne's reference code does. The
//! frame chain reports the larger of the block-mean error and the one from
//! the integrated autocorrelation time in a Sokal window, and continues
//! from where it stopped when a plane needs more samples.
//!
//! # What goes in
//!
//! * A [`PointSurface`] that returns the energy and the Cartesian gradient
//!   `∇E` (not the force) at a flat coordinate vector. Only
//!   [`PointSurface::eval`] is used. Energy and length are in whatever
//!   units the surface uses, and [`PafiConfig::thermal`] and the arc length
//!   follow them. With [`PafiConfig::fix_com`] the vector is read as
//!   `x₀, y₀, z₀, x₁, …`, three entries per atom.
//! * An [`ArcPath`]: ordered frames of the same length, such as the images
//!   of a converged nudged elastic band, from [`ArcPath::from_positions`],
//!   or frames with arc lengths of the caller's choosing, from
//!   [`ArcPath::new`]. The first frame is the reactant and the zero of the
//!   free energy, so it should be a relaxed minimum. Frames of a periodic
//!   cell must be unwrapped against the first, because the path is
//!   interpolated coordinate by coordinate.
//! * The arc lengths of the hyperplanes to sample, at least two, strictly
//!   increasing, inside [`ArcPath::bounds`]. [`ArcPath::arcs`] gives one
//!   plane per frame.
//! * A [`PafiConfig`].
//!
//! # What comes out
//!
//! [`pafi_profile`] returns a [`FreeEnergyProfile`], the potential of mean
//! force `F(s)` from the first plane with its barrier and dividing plane,
//! and one [`HyperplaneAverage`] per plane with the sampled gradient, its
//! standard error, `ψ`, the autocorrelation time and the retained
//! configurations. [`zero_temperature_barrier`] gives the `kT → 0` profile
//! from the same path without sampling, which is the check that the path
//! and the surface agree.
//!
//! # Example
//!
//! A path along the floor of `V = (1 - x²)² + ½ k(x) y²` with
//! `k = 1.5 + x/2`, from the minimum at `x = -1` to the saddle at `x = 0`.
//! The transverse stiffness grows along the path, so the free energy
//! barrier is the potential barrier of 1 plus the transverse entropy
//! `½ kT ln(k(0) / k(-1))`, here `½ kT ln 1.5`.
//!
//! ```
//! use ndarray::{Array1, ArrayView1};
//! use rgsaddle::{ArcPath, PafiConfig, PointSurface, SaddleError};
//! use rgsaddle::{pafi_profile, zero_temperature_barrier};
//!
//! struct Widening;
//!
//! impl PointSurface for Widening {
//!     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
//!         let (x, y) = (x[0], x[1]);
//!         let stiffness = 1.5 + 0.5 * x;
//!         let energy = (1.0 - x * x).powi(2) + 0.5 * stiffness * y * y;
//!         let gradient = Array1::from_vec(vec![
//!             -4.0 * x * (1.0 - x * x) + 0.25 * y * y,
//!             stiffness * y,
//!         ]);
//!         Ok((energy, gradient))
//!     }
//! }
//!
//! # fn main() -> Result<(), SaddleError> {
//! let images: Vec<Array1<f64>> = (0..11)
//!     .map(|i| Array1::from_vec(vec![-1.0 + i as f64 / 10.0, 0.0]))
//!     .collect();
//! let path = ArcPath::from_positions(&images)?;
//!
//! let thermal = 0.05;
//! let config = PafiConfig::default()
//!     .with_thermal(thermal)
//!     .with_time_step(0.004)
//!     .with_thermalize(400)
//!     .with_stride(8)
//!     .with_samples(600)
//!     .with_blocks(6)
//!     .with_seed(11)
//!     .with_fix_com(false)
//!     .with_retain(2);
//! let (profile, planes) = pafi_profile(&Widening, &path, &path.arcs(), &config)?;
//!
//! let truth = 1.0 + 0.5 * thermal * 1.5_f64.ln();
//! let tolerance = 0.02;
//! assert!((profile.barrier - truth).abs() < tolerance);
//! assert_eq!(profile.dividing, profile.arc.len() - 1);
//! assert_eq!(planes.len(), profile.arc.len());
//!
//! let cold = zero_temperature_barrier(&Widening, &path, &path.arcs())?;
//! assert!((cold.barrier - 1.0).abs() < tolerance);
//! # Ok(())
//! # }
//! ```
//!
//! `tests/pafi_analytic.rs` checks the same estimator against
//! quadrature on a bent valley, on a harmonic plane, and on a plane of two
//! wells, and `tests/pafi_reference.rs` against the reference code.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use rgmin::{Control, Error, Method, Oracle, Solver, Status, StopNorm};

use crate::Region;
use crate::error::SaddleError;
use crate::hyperplane_constraint::{HyperplaneConstraint, uniform_translations};
use crate::minmode::PointSurface;
#[cfg(feature = "ohtst")]
use crate::ohtst::{
    self, EndState, TransmissionCoefficient, TransmissionConfig, transmission_coefficient,
};
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler, ProjectedConstraint, TangentProjector,
};
use crate::relax::RelaxConfig;
use crate::reparam::Spline;
use crate::stats::{EquilibrationReport, Rng, block_mean, equilibrate, windowed_block_mean};

/// Time discretization of the projected overdamped Langevin equation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PafiIntegrator {
    /// Euler-Maruyama applied directly to Eq. (9) of Swinburne and
    /// Marinica (2018).
    EulerMaruyama,
    /// The correlated-noise step of Leimkuhler and Matthews (2013),
    /// second order for stationary averages.
    #[default]
    LeimkuhlerMatthews,
}

/// The built-in step. This is [`PafiIntegrator`].
pub type Integrator = PafiIntegrator;

/// Instantaneous mean-force estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum PafiMeanForce {
    /// The modified force recorded by LAMMPS `fix pafi` and the public
    /// PAFI driver: the first term of the paper's Eq. (8).
    #[default]
    ReferenceCode,
    /// Both terms of Swinburne and Marinica (2018), Eq. (8), including
    /// `kT d/ds ln(|ψ| / |X'|)`.
    PublishedEquation8,
}

/// Overdamped hyperplane sampling.
///
/// Non-exhaustive: start from [`PafiConfig::default`] and chain the
/// `with_*` setters, one per field. [`PafiConfig::thermal`],
/// [`PafiConfig::mobility`] and [`PafiConfig::time_step`] follow the units
/// of the surface: `μ dt ∇E` is a length, so `μ dt` is length² per energy.
/// The setters of the optional fields, [`PafiConfig::target_error`] and
/// [`PafiConfig::max_jump`], take a bare value or `None`. A run checks its
/// configuration on entry: `kT`, the mobility and the step must be finite
/// and positive, the stride and [`PafiConfig::trajectories`] positive,
/// [`PafiConfig::blocks`] at least 2, [`PafiConfig::samples`] at least
/// twice the blocks, and a given `target_error` or `max_jump` finite and
/// positive.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PafiConfig {
    /// `kT` in the energy unit of the surface. With energies in eV, it is
    /// `BOLTZMANN_EV_PER_K * T`.
    pub thermal: f64,
    /// Mobility `μ` in `dx = -μ ∇E dt + sqrt(2 μ kT dt) ξ`.
    pub mobility: f64,
    /// Overdamped time step `dt`.
    pub time_step: f64,
    /// Minimum engine warm-up floor before automated equilibration detection.
    pub thermalize: usize,
    /// Steps between recorded samples. The force is evaluated at every
    /// step, and the sample is taken from the first of each `stride`.
    pub stride: usize,
    /// Recorded samples. At least twice [`Self::blocks`].
    pub samples: usize,
    /// Contiguous blocks used for the standard error. At least 2.
    pub blocks: usize,
    /// Independent trajectories of equal length on each plane, sampled by
    /// [`average_hyperplane`]. With more than one, the reported error is
    /// the reference PAFI ensemble error
    /// `sqrt(sum((mean_i - mean)^2)) / trajectories`.
    pub trajectories: usize,
    /// Time discretization of the projected overdamped Langevin equation.
    pub integrator: PafiIntegrator,
    /// Instantaneous estimator to average.
    pub mean_force: PafiMeanForce,
    /// Seed of the noise.
    pub seed: u64,
    /// Hold the center of mass: remove the mean Cartesian drift from the
    /// normal, the force and the noise. It takes effect when the vector
    /// holds at least two atoms, three entries each; otherwise it is
    /// ignored.
    pub fix_com: bool,
    /// How many recorded configurations to keep, evenly spaced.
    pub retain: usize,
    /// Stop early when [`FreeEnergyProfile::barrier_error`] is at or below
    /// this. `None` runs one pass.
    pub target_error: Option<f64>,
    /// Most sampling passes, the first included, used to reach
    /// [`Self::target_error`]. Without a target there is one pass whatever
    /// this is, and `1` is also one pass.
    pub max_rounds: usize,
    /// Hard symmetry-unique region and optional additional predicates.
    pub region: Option<Arc<dyn Region>>,
    /// Largest displacement of one atom, in the length unit of the
    /// surface, between the plane minimum quenched from a sampled
    /// configuration and the one quenched from the path. Past it the
    /// configuration has left the basin of the path. `None` skips the test.
    pub max_jump: Option<f64>,
    /// The in-plane quench of [`Self::max_jump`]: its method, force
    /// tolerance, step cap and per-atom step, on the force with its part
    /// along the normal (and its center-of-mass part when held) removed.
    /// Reaching the step cap, or a stall of the method, ends the quench
    /// where it is.
    pub quench: RelaxConfig,
    /// Largest atom force left in the plane when [`pafi_frames`] relaxes a
    /// frame in its plane before sampling.
    pub relax_tol: f64,
    /// Largest atom force left by the in-plane relaxation of the windowed
    /// jump test. The test runs when [`Self::window`] is at least 1 and
    /// [`Self::max_jump`] is set; a window of 0 leaves the quench basin
    /// test of [`Self::max_jump`] in charge of [`average_hyperplane`].
    pub jump_tol: f64,
    /// Recorded samples between two tests of the windowed jump test.
    /// `0` leaves that test off and leaves the quench basin test in charge.
    /// A plane that keeps no window in its first pass halves this, up to
    /// four times, and thermalizes in windows of that length too.
    pub window: usize,
}

with_setters!(PafiConfig {
    with_thermal => thermal: val f64,
    with_mobility => mobility: val f64,
    with_time_step => time_step: val f64,
    with_thermalize => thermalize: val usize,
    with_stride => stride: val usize,
    with_samples => samples: val usize,
    with_blocks => blocks: val usize,
    with_trajectories => trajectories: val usize,
    with_integrator => integrator: val PafiIntegrator,
    with_mean_force => mean_force: val PafiMeanForce,
    with_seed => seed: val u64,
    with_fix_com => fix_com: val bool,
    with_retain => retain: val usize,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
    with_max_jump => max_jump: opt f64,
    with_quench => quench: val RelaxConfig,
    with_relax_tol => relax_tol: val f64,
    with_jump_tol => jump_tol: val f64,
    with_window => window: val usize,
});

impl PafiConfig {
    /// Restrict every fixed PAFI plane to a hard region.
    #[must_use]
    pub fn with_region(mut self, region: Arc<dyn Region>) -> Self {
        self.region = Some(region);
        self
    }
}

impl Default for PafiConfig {
    /// `kT` of 300 K in eV, unit mobility, time step 0.001, 200
    /// thermalization steps, one trajectory of 200 samples 5 steps apart
    /// in 5 blocks, the Leimkuhler-Matthews integrator, the reference-code
    /// mean-force estimator, seed 1, the center of mass held, 16 kept
    /// configurations, one pass, and a largest jump of 0.4 (Å), the
    /// `MaxJumpThresh` of Swinburne's reference code, quenched along the
    /// steepest-descent flow ([`Method::Ode12r`]) under the other
    /// [`RelaxConfig`] defaults.
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mobility: 1.0,
            time_step: 1.0e-3,
            thermalize: 200,
            stride: 5,
            samples: 200,
            blocks: 5,
            trajectories: 1,
            integrator: PafiIntegrator::LeimkuhlerMatthews,
            mean_force: PafiMeanForce::ReferenceCode,
            seed: 1,
            fix_com: true,
            retain: 16,
            target_error: None,
            max_rounds: 4,
            region: None,
            max_jump: Some(0.4),
            quench: RelaxConfig::default().with_method(Method::Ode12r),
            relax_tol: 1.0e-4,
            jump_tol: 1.0e-2,
            window: 0,
        }
    }
}

impl PafiConfig {
    fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI kT must be finite and positive".into(),
            ));
        }
        if !(self.mobility.is_finite() && self.mobility > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI mobility must be finite and positive".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI time step must be finite and positive".into(),
            ));
        }
        if self.stride == 0
            || self.blocks < 2
            || self.samples < self.blocks * 2
            || self.trajectories == 0
        {
            return Err(SaddleError::Invalid(
                "PAFI needs a positive stride and trajectory count, at least two blocks, and two samples per block".into(),
            ));
        }
        if let Some(target) = self.target_error
            && !(target.is_finite() && target > 0.0)
        {
            return Err(SaddleError::Invalid(
                "PAFI target error must be finite and positive".into(),
            ));
        }
        if let Some(jump) = self.max_jump {
            if !(jump.is_finite() && jump > 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI largest jump must be finite and positive".into(),
                ));
            }
            self.quench.check()?;
        }
        let positive = |value: f64| value.is_finite() && value > 0.0;
        if !(positive(self.relax_tol) && positive(self.jump_tol)) {
            return Err(SaddleError::Invalid(
                "PAFI relaxation and jump tolerances must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// One frame of a path, with its parameter.
///
/// The frames given to [`ArcPath::new`] must share one length, be finite,
/// and have strictly increasing [`PathFrame::arc`].
#[derive(Clone, Debug)]
pub struct PathFrame {
    /// Cartesian coordinates of the frame.
    pub positions: Array1<f64>,
    /// Path parameter of the frame. [`ArcPath::from_positions`] uses the
    /// summed chord lengths from the first frame.
    pub arc: f64,
}

/// Cubic-spline path `X(s)` through ordered frames.
///
/// [`ArcPath::geometry`] returns the point, tangent and curvature of the
/// path at any `s`, and [`pafi_profile`] and [`zero_temperature_barrier`]
/// take the path and the values of `s` to sample.
///
/// Every coordinate is the not-a-knot cubic spline of the frames over
/// their [`PathFrame::arc`] values: the spline interpolation of NEB images
/// of Swinburne and Marinica (2018) and the default end condition of
/// scipy's `CubicSpline`, which their reference code uses. Two frames give
/// the straight line and three the parabola through them. `X`, `X'` and
/// `X''` are continuous, so the hyperplanes turn smoothly along the path
/// and the plane through a frame is the same from either side of it.
#[derive(Clone, Debug)]
pub struct ArcPath {
    spline: Spline,
    frames: Vec<PathFrame>,
}

/// Position and the first three parameter derivatives of a path, from
/// [`ArcPath::geometry`]. [`average_hyperplane`] takes one of these as the
/// plane it samples.
#[derive(Clone, Debug)]
pub struct PathGeometry {
    /// `X(s)`.
    pub positions: Array1<f64>,
    /// `X' = dX/ds`, the hyperplane normal before any center-of-mass part
    /// is removed.
    pub tangent: Array1<f64>,
    /// `X'' = d²X/ds²`.
    pub curvature: Array1<f64>,
    /// `d³X/ds³`, the third derivative of the cubic spline, constant on
    /// each segment. At an interior knot it is the value of the segment
    /// the spline selects there.
    pub jerk: Array1<f64>,
    /// `s`.
    pub arc: f64,
}

impl ArcPath {
    /// Build the path from ordered frames and their parameters.
    ///
    /// The parameter is the caller's: chord lengths, as
    /// [`ArcPath::from_positions`] takes, or any strictly increasing scale
    /// such as the normalized distance from the first frame. `dF/ds` and the
    /// arcs passed to [`pafi_profile`] are in that parameter.
    ///
    /// Returns [`SaddleError::Invalid`] for fewer than two frames, for a
    /// parameter that does not strictly increase or is not finite, for
    /// neighboring frames that are equal, or for a singular spline system;
    /// and [`SaddleError::Shape`] for empty frames, frames of unequal length,
    /// or a coordinate that is not finite.
    pub fn new(frames: Vec<PathFrame>) -> Result<Self, SaddleError> {
        if frames.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let n = frames[0].positions.len();
        if n == 0 {
            return Err(SaddleError::Shape("PAFI path frame is empty".into()));
        }
        for (i, frame) in frames.iter().enumerate() {
            if frame.positions.len() != n || !frame.positions.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI path frames must share one finite length".into(),
                ));
            }
            if !frame.arc.is_finite() || (i > 0 && frame.arc <= frames[i - 1].arc) {
                return Err(SaddleError::Invalid(
                    "PAFI arc length must strictly increase".into(),
                ));
            }
            if i > 0 && frame.positions == frames[i - 1].positions {
                return Err(SaddleError::Invalid(
                    "neighboring PAFI frames must differ".into(),
                ));
            }
        }
        let arcs: Vec<f64> = frames.iter().map(|frame| frame.arc).collect();
        let mut values = Array2::zeros((frames.len(), n));
        for (mut row, frame) in values.outer_iter_mut().zip(&frames) {
            row.assign(&frame.positions);
        }
        let spline = Spline::fit(&arcs, &values)
            .ok_or_else(|| SaddleError::Invalid("the PAFI spline system is singular".into()))?;
        Ok(Self { spline, frames })
    }

    /// The same curve traversed from the last frame, with the parameter
    /// `first + last - s`. Integrating it gives the reversal test of a
    /// profile: its forward barrier starts from the other end.
    pub fn reversed(&self) -> Result<Self, SaddleError> {
        let (first, last) = self.bounds();
        let values = self.spline.values();
        Self::new(
            self.spline
                .knots()
                .iter()
                .zip(values.outer_iter())
                .rev()
                .map(|(arc, positions)| PathFrame {
                    positions: positions.to_owned(),
                    arc: first + last - arc,
                })
                .collect(),
        )
    }

    /// Build the path from frames in order, reactant first, such as the
    /// images of a nudged elastic band.
    ///
    /// The parameter of each frame is the summed length of the straight
    /// chords from the first, `|x₁ - x₀|` and so on in the whole coordinate
    /// vector, so `s` has the length unit of the positions and `dF/ds` is an
    /// energy per length. The curve through the frames is the spline of
    /// [`ArcPath`], not the chords. Neighboring frames must be finite and
    /// distinct; otherwise this returns [`SaddleError::Invalid`], and
    /// [`SaddleError::Shape`] for frames of unequal length.
    pub fn from_positions(positions: &[Array1<f64>]) -> Result<Self, SaddleError> {
        if positions.len() < 2 {
            return Err(SaddleError::Invalid(
                "a PAFI path needs at least two frames".into(),
            ));
        }
        let mut arc = 0.0;
        let mut frames = Vec::with_capacity(positions.len());
        frames.push(PathFrame {
            positions: positions[0].clone(),
            arc,
        });
        for pair in positions.windows(2) {
            let step = pair[1]
                .iter()
                .zip(pair[0].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            if !(step.is_finite() && step > 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI frames must be finite and distinct".into(),
                ));
            }
            arc += step;
            frames.push(PathFrame {
                positions: pair[1].clone(),
                arc,
            });
        }
        Self::new(frames)
    }

    /// Parameter of the first frame and of the last. [`ArcPath::geometry`]
    /// accepts this range.
    pub fn bounds(&self) -> (f64, f64) {
        let knots = self.spline.knots();
        (knots[0], knots[knots.len() - 1])
    }

    /// Parameter of every frame, in order. Passed to [`pafi_profile`] it
    /// samples one hyperplane through each frame.
    pub fn arcs(&self) -> Vec<f64> {
        self.spline.knots().to_vec()
    }

    /// Spline geometry at `arc`, which must lie in [`ArcPath::bounds`]; a
    /// parameter outside it is [`SaddleError::Invalid`], and a `NaN` or
    /// infinite `arc` is [`SaddleError::NonFinite`].
    pub fn geometry(&self, arc: f64) -> Result<PathGeometry, SaddleError> {
        if !arc.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        let (first, last) = self.bounds();
        if arc < first || arc > last {
            return Err(SaddleError::Invalid(format!(
                "PAFI coordinate {arc} is outside [{first}, {last}]"
            )));
        }
        let [positions, tangent, curvature, jerk] = self.spline.derivatives(arc);
        Ok(PathGeometry {
            positions,
            tangent,
            curvature,
            jerk,
            arc,
        })
    }

    /// The stored frames, reactant first.
    pub fn frames(&self) -> &[PathFrame] {
        &self.frames
    }

    /// The planes of [`pafi_frames`].
    ///
    /// The normal is the unit tangent of the parabola through the frame
    /// and its two neighbours, in the arc length of the chords, and the
    /// curvature is the same derivative of the normals. The end frames
    /// take the parabola through themselves and the next two. Both are of
    /// second order in the spacing, where a chord tilts the plane by half
    /// the angle the path turns over it, and a reversed band gives the
    /// same planes. Every atom carries the tangent: keeping only the atoms
    /// that hold some fraction of the summed squared steps replaces this
    /// plane by the one [`Self::truncated_planes`] builds, and the barrier
    /// of that plane depends on the fraction. [`ArcPath::geometry`] stays
    /// the spline through the same frames.
    pub fn frame_planes(&self) -> Result<FramePlanes, SaddleError> {
        self.planes_on(1.0)
    }

    /// Planes whose normals are zero off the fewest atoms, largest first,
    /// that carry `share` of the summed squared steps.
    ///
    /// This is not the plane of [`pafi_frames`]. The constraint
    /// `(x - X) · ν = 0` with `ν` the renormalized masked tangent is a
    /// different foliation from `(x - X) · τ = 0`. On a harmonic transverse
    /// well the two barriers differ by [`harmonic_mask_shift`]. `share`
    /// lies in `(0, 1]`; 1 is [`Self::frame_planes`].
    pub fn truncated_planes(&self, share: f64) -> Result<FramePlanes, SaddleError> {
        self.planes_on(share)
    }

    fn planes_on(&self, share: f64) -> Result<FramePlanes, SaddleError> {
        let count = self.frames.len();
        if !(share > 0.0 && share <= 1.0) {
            return Err(SaddleError::Invalid(
                "PAFI active share must lie in (0, 1]".into(),
            ));
        }
        let n = self.frames[0].positions.len();
        let width = atom_width(n);
        let steps: Vec<Array1<f64>> = self
            .frames
            .windows(2)
            .map(|pair| &pair[1].positions - &pair[0].positions)
            .collect();
        let active = active_atoms(&steps, width, share);
        let mut mask = Array1::<f64>::zeros(n);
        for atom in &active {
            for c in 0..width {
                mask[atom * width + c] = 1.0;
            }
        }
        let local: Vec<Array1<f64>> = steps.iter().map(|step| step * &mask).collect();
        let lengths: Vec<f64> = local.iter().map(norm).collect();
        if !lengths
            .iter()
            .all(|length| length.is_finite() && *length > 0.0)
        {
            return Err(SaddleError::Invalid(
                "a PAFI frame step moves none of the constrained atoms".into(),
            ));
        }
        let mut arcs = Vec::with_capacity(count);
        arcs.push(0.0);
        for length in &lengths {
            arcs.push(arcs[arcs.len() - 1] + length);
        }
        let normals = arc_derivative(&local, &lengths)
            .into_iter()
            .map(|tangent| {
                let length = norm(&tangent);
                if length.is_finite() && length > 0.0 {
                    Ok(tangent / length)
                } else {
                    Err(SaddleError::Invalid(
                        "the path turns back on itself at a PAFI frame".into(),
                    ))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let turns: Vec<Array1<f64>> = normals.windows(2).map(|pair| &pair[1] - &pair[0]).collect();
        let curvatures = arc_derivative(&turns, &lengths);
        Ok(FramePlanes {
            normals,
            curvatures,
            arcs,
            active,
        })
    }
}

/// The planes [`pafi_frames`] samples, one through every frame.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FramePlanes {
    /// Unit normal at each frame, zero off [`Self::active`].
    pub normals: Vec<Array1<f64>>,
    /// `dn/ds` at each frame, zero off [`Self::active`].
    pub curvatures: Vec<Array1<f64>>,
    /// Arc length of each frame from the first along the coordinates of
    /// [`Self::active`].
    pub arcs: Vec<f64>,
    /// Atoms the normals act on, ascending. An atom is a coordinate triple,
    /// or one coordinate when the length is not a multiple of three.
    pub active: Vec<usize>,
}

/// Mean free-energy gradient on one hyperplane.
///
/// `gradient` is the average of `(∇E · t)(1 - ψ)` over the recorded
/// samples, in energy per unit of the path parameter, and `error` is its
/// standard error from block means. With more than one pass
/// ([`PafiConfig::target_error`]) the passes are combined by inverse
/// variance, `samples` and `n_eff` are summed, and `tau` is the largest.
/// With more than one trajectory ([`PafiConfig::trajectories`]) the
/// gradient is the mean of the trajectories and `error` the spread of
/// their means; `equilibration` is that of the first trajectory.
#[derive(Clone, Debug)]
pub struct HyperplaneAverage {
    /// The constrained plane, with its normal toward increasing path arc.
    pub plane: Hyperplane,
    /// `dF/ds` in energy per arc length.
    pub gradient: f64,
    /// Standard error from the block means.
    pub error: f64,
    /// Mean of `ψ = (x - X) · X'' / |t|²`. The paper's locality condition,
    /// its Eq. (6), is `1 - ψ > 0`.
    pub psi: f64,
    /// Recorded samples in the production run.
    pub samples: usize,
    /// Integrated autocorrelation time of the recorded series, in samples.
    pub tau: f64,
    /// `samples / (2 τ)`.
    pub n_eff: f64,
    /// Force evaluations spent on this plane, the quenches of
    /// [`PafiConfig::max_jump`] included, summed over rounds.
    pub evaluations: usize,
    /// Configurations kept from the production run, on the plane.
    pub retained: Vec<Array1<f64>>,
    /// Automated equilibration, correlation and split-half drift report.
    pub equilibration: EquilibrationReport,
    /// Thermalizations and stretches of the production run sampled again
    /// because they left the basin of the path ([`PafiConfig::max_jump`]),
    /// summed over rounds.
    pub rejected: usize,
    /// Quenches of the basin test ([`PafiConfig::max_jump`]) that ended
    /// without the force falling to [`RelaxConfig::force_tol`], summed
    /// over rounds: the solver stalled, or spent the step cap. Such a
    /// quench ends where it is and the basin test compares that point, so
    /// a nonzero count means some of the tests ran on points that were
    /// not converged.
    pub unconverged_quenches: usize,
    /// Energy at the origin of a frame plane. The spline sampler leaves
    /// this non-finite.
    pub energy: f64,
    /// `∇E · n` at the origin of a frame plane. The spline sampler leaves
    /// this non-finite.
    pub zero: f64,
    /// Largest displacement of one atom from the origin over the recorded
    /// samples of a frame-plane chain. The spline sampler leaves this 0.
    pub widest: f64,
    /// Windows dropped by the windowed jump test. The spline sampler and
    /// an external engine report none.
    pub jumps: usize,
}

/// Potential of mean force from the first plane.
///
/// The vectors have one entry per plane, in the order of the arcs given to
/// [`pafi_profile`]. The integral is the trapezoid rule over the planes, so
/// planes spaced wider than the scale on which `dF/ds` changes bias it. The
/// errors add the errors of the planes in quadrature, weighted by the
/// trapezoid weights, as if the planes were independent. They do not
/// include the error of the quadrature over the planes.
#[derive(Clone, Debug)]
pub struct FreeEnergyProfile {
    /// Arc length of each plane.
    pub arc: Vec<f64>,
    /// `dF/ds` on each plane.
    pub gradient: Vec<f64>,
    /// Standard error of `gradient`.
    pub gradient_error: Vec<f64>,
    /// `F(s)`, trapezoid-integrated from the first plane.
    pub free_energy: Vec<f64>,
    /// Propagated standard error of `free_energy`.
    pub free_energy_error: Vec<f64>,
    /// Maximum of the potential of mean force. The reactant is zero.
    pub barrier: f64,
    /// Standard error of `barrier`.
    pub barrier_error: f64,
    /// Index of that maximum.
    pub dividing: usize,
    /// Recorded samples, summed over planes and rounds.
    pub samples: usize,
}

impl FreeEnergyProfile {
    /// Barrier from the last plane back over the dividing plane,
    /// `F(dividing) - F(last)`, with the standard error of the trapezoid
    /// sum over the planes between them. A path whose last plane is the
    /// product minimum gives the reverse hop from the same samples.
    pub fn reverse_barrier(&self) -> (f64, f64) {
        let last = self.free_energy.len().saturating_sub(1);
        let mut weights = vec![0.0; self.arc.len()];
        for k in self.dividing + 1..=last {
            let span = self.arc[k] - self.arc[k - 1];
            weights[k - 1] += 0.5 * span;
            weights[k] += 0.5 * span;
        }
        let variance: f64 = weights
            .iter()
            .zip(&self.gradient_error)
            .map(|(w, e)| w * w * e * e)
            .sum();
        (
            self.free_energy[self.dividing] - self.free_energy[last],
            variance.sqrt(),
        )
    }

    /// [`balanced_barriers`] of [`Self::barrier`] and
    /// [`Self::reverse_barrier`] against `end_gap`, the free energy of the
    /// last end less that of the first, known apart from this profile.
    pub fn balanced(&self, end_gap: f64) -> Result<BalancedBarriers, SaddleError> {
        balanced_barriers(
            (self.barrier, self.barrier_error),
            self.reverse_barrier(),
            end_gap,
        )
    }

    /// [`anchored_barriers`] of [`Self::barrier`] and
    /// [`Self::reverse_barrier`] against `end_gap`, the free energy of the
    /// last end less that of the first, known apart from this profile.
    pub fn anchored(&self, end_gap: f64) -> Result<BalancedBarriers, SaddleError> {
        anchored_barriers(
            (self.barrier, self.barrier_error),
            self.reverse_barrier(),
            end_gap,
        )
    }
}

/// Both barriers of one path, made to differ by the free-energy
/// difference of its ends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedBarriers {
    /// Barrier above the first end.
    pub forward: f64,
    /// Barrier above the last end, `end_gap` below [`Self::forward`].
    pub reverse: f64,
    /// Standard error of both: with half of [`Self::drift`] in quadrature
    /// from [`balanced_barriers`], that of the forward barrier from
    /// [`anchored_barriers`].
    pub error: f64,
    /// `forward - reverse - end_gap` of the inputs: what the integral adds
    /// to the free-energy difference of the ends.
    pub drift: f64,
}

/// Combine the forward and reverse barriers of one path given the
/// free-energy difference of its ends.
///
/// The barrier above the first plane and [`FreeEnergyProfile::reverse_barrier`]
/// above the last measure one dividing surface from the two ends, so their
/// difference is `end_gap`, the free energy of the last end less that of
/// the first, plus whatever the integral drifts along the path. With
/// `end_gap` known apart from the profile, `forward` and
/// `reverse + end_gap` are two estimates of one barrier. The result is
/// their inverse-variance mean, the reverse barrier `end_gap` below it, and
/// the drift. Half the drift is added to the error in quadrature: it is
/// what the mean moves when the whole drift lies on one side of the
/// dividing surface and the two estimates weigh alike; when it is known to
/// lie past it, [`anchored_barriers`] keeps the forward barrier instead.
/// `end_gap = 0` is a path between two equivalent states. Errors must be
/// positive.
pub fn balanced_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<BalancedBarriers, SaddleError> {
    check_barriers(forward, reverse, end_gap)?;
    let (weight_f, weight_r) = (forward.1.powi(-2), reverse.1.powi(-2));
    let mean = (weight_f * forward.0 + weight_r * (reverse.0 + end_gap)) / (weight_f + weight_r);
    let drift = forward.0 - reverse.0 - end_gap;
    Ok(BalancedBarriers {
        forward: mean,
        reverse: mean - end_gap,
        error: ((weight_f + weight_r).recip() + (0.5 * drift).powi(2)).sqrt(),
        drift,
    })
}

/// Both barriers of one path from its forward barrier and the free-energy
/// difference of its ends.
///
/// The forward barrier integrates the profile from the first plane to the
/// dividing surface and [`FreeEnergyProfile::reverse_barrier`] from there
/// to the last plane, so each carries the drift of its own stretch. With
/// `end_gap`, the free energy of the last end less that of the first,
/// known apart from the profile, the reverse barrier by detailed balance
/// is `forward - end_gap`, and only the forward stretch enters either. The
/// error is that of `forward`, and [`BalancedBarriers::drift`] still
/// reports `forward - reverse - end_gap`. It is the estimate to take over
/// [`balanced_barriers`] when the drift lies past the dividing surface, as
/// a path integrated again from its other end shows: the barriers from the
/// start of each integration then agree, and those from its end do not.
/// `end_gap = 0` is a path between two equivalent states. Errors must be
/// positive.
pub fn anchored_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<BalancedBarriers, SaddleError> {
    check_barriers(forward, reverse, end_gap)?;
    Ok(BalancedBarriers {
        forward: forward.0,
        reverse: forward.0 - end_gap,
        error: forward.1,
        drift: forward.0 - reverse.0 - end_gap,
    })
}

fn check_barriers(
    forward: (f64, f64),
    reverse: (f64, f64),
    end_gap: f64,
) -> Result<(), SaddleError> {
    if ![forward.0, forward.1, reverse.0, reverse.1, end_gap]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI barrier"));
    }
    if !(forward.1 > 0.0 && reverse.1 > 0.0) {
        return Err(SaddleError::Invalid(
            "PAFI barrier errors must be positive".into(),
        ));
    }
    Ok(())
}

/// Flat-hyperplane derivative at one configuration.
///
/// `dF/dr = (∇E · ν) (τ · ν - (x - X) · dν/dr)` on the Euclidean surface
/// measure of `(x - X) · ν = 0`. `normal` is the unit constraint normal
/// `ν`, `normal_rate` is `dν/dr`, `path_tangent` is `τ = dX/dr`, and
/// `displacement` is `x - X`. When `ν` is the unit path tangent and `r`
/// its arc length, `τ · ν = 1` and this is `(∇E · ν) (1 - ψ)`.
pub fn flat_hyperplane_gradient(
    gradient: ArrayView1<f64>,
    displacement: ArrayView1<f64>,
    normal: ArrayView1<f64>,
    normal_rate: ArrayView1<f64>,
    path_tangent: ArrayView1<f64>,
) -> Result<f64, SaddleError> {
    let n = gradient.len();
    if displacement.len() != n
        || normal.len() != n
        || normal_rate.len() != n
        || path_tangent.len() != n
    {
        return Err(SaddleError::Shape(
            "PAFI flat gradient needs one length for every vector".into(),
        ));
    }
    if !gradient
        .iter()
        .chain(displacement.iter())
        .chain(normal.iter())
        .chain(normal_rate.iter())
        .chain(path_tangent.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI flat gradient"));
    }
    let factor = path_tangent.dot(&normal) - displacement.dot(&normal_rate);
    Ok(gradient.dot(&normal) * factor)
}

/// Map a masked-plane barrier onto the barrier of another normal.
///
/// On a path whose transverse motion is harmonic, and whose planes pass
/// through the same stationary points, the flat-hyperplane barrier with
/// unit normal `n` and the one with unit normal `n*` differ by the
/// entropy of the two curvatures. With `c(n) = nᵀ H⁻¹ n` the compliance,
/// `det(H restricted to n⊥) = det(H) c(n)` for invertible `H`, so the
/// Gaussian integrals give
///
/// `ΔF(n*) = ΔF(n) + (kT/2) ln[ (c_s(n*)/c_s(n)) (c_m(n)/c_m(n*)) ]`.
///
/// `reactant_*` is `c` on the reactant plane and `dividing_*` is `c` on
/// the dividing plane. A negative curvature, as at a saddle, is carried
/// by a negative compliance; the ratio of the two dividing compliances
/// has to be positive, and likewise the reactant pair. The shift does not
/// include a slide of the plane's minimum off the path point: that energy
/// is `E(path) - E(minimum)` and is added by the caller.
pub fn harmonic_mask_shift(
    barrier: f64,
    reactant_full: f64,
    reactant_mask: f64,
    dividing_full: f64,
    dividing_mask: f64,
    thermal: f64,
) -> Result<f64, SaddleError> {
    let values = [
        barrier,
        reactant_full,
        reactant_mask,
        dividing_full,
        dividing_mask,
        thermal,
    ];
    if !values.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("PAFI mask shift"));
    }
    if thermal <= 0.0 {
        return Err(SaddleError::Invalid(
            "PAFI mask shift needs a positive kT".into(),
        ));
    }
    let ratio = (dividing_full / dividing_mask) * (reactant_mask / reactant_full);
    if !(ratio.is_finite() && ratio > 0.0) {
        return Err(SaddleError::Invalid(
            "PAFI mask compliances need a positive ratio at each end".into(),
        ));
    }
    Ok(barrier + 0.5 * thermal * ratio.ln())
}

/// Sample one hyperplane and return the PAFI gradient.
///
/// With [`PafiConfig::max_jump`] set, the production run is cut into
/// [`PafiConfig::blocks`] stretches. The thermalized configuration and the
/// end of every stretch are quenched in the plane and compared with the
/// plane minimum quenched from the path; one atom farther than `max_jump`
/// means the configuration has left the basin of the path. A stretch that
/// left it is dropped and sampled again from the end of the last kept one,
/// and a thermalization from the path. Swinburne's reference code drops
/// such a trajectory the same way. More dropped stretches than there are
/// blocks is an error: the plane does not hold the system in the basin.
///
/// With [`PafiConfig::trajectories`] above one, the plane is sampled by
/// that many independent trajectories, seeded `seed + i`, and the error is
/// the spread of their means.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    if config.window > 0 {
        return average_windowed(surface, geometry, config);
    }
    if config.trajectories == 1 {
        return average_trajectory(surface, geometry, config);
    }
    let mut trajectories = Vec::with_capacity(config.trajectories);
    for index in 0..config.trajectories {
        let local = config
            .clone()
            .with_trajectories(1)
            .with_seed(config.seed.wrapping_add(index as u64));
        trajectories.push(average_trajectory(surface, geometry, &local)?);
    }
    let count = trajectories.len() as f64;
    let gradient = trajectories
        .iter()
        .map(|trajectory| trajectory.gradient)
        .sum::<f64>()
        / count;
    let error = trajectories
        .iter()
        .map(|trajectory| {
            let residual = trajectory.gradient - gradient;
            residual * residual
        })
        .sum::<f64>()
        .sqrt()
        / count;
    let samples = trajectories
        .iter()
        .map(|trajectory| trajectory.samples)
        .sum();
    let psi = trajectories
        .iter()
        .map(|trajectory| trajectory.psi * trajectory.samples as f64)
        .sum::<f64>()
        / samples as f64;
    let mut retained = Vec::with_capacity(config.retain);
    for trajectory in &trajectories {
        let remaining = config.retain.saturating_sub(retained.len());
        retained.extend(trajectory.retained.iter().take(remaining).cloned());
    }
    Ok(HyperplaneAverage {
        plane: trajectories[0].plane.clone(),
        gradient,
        error,
        psi,
        samples,
        tau: trajectories
            .iter()
            .map(|trajectory| trajectory.tau)
            .fold(0.0, f64::max),
        n_eff: trajectories.iter().map(|trajectory| trajectory.n_eff).sum(),
        evaluations: trajectories
            .iter()
            .map(|trajectory| trajectory.evaluations)
            .sum(),
        retained,
        equilibration: trajectories[0].equilibration.clone(),
        rejected: trajectories
            .iter()
            .map(|trajectory| trajectory.rejected)
            .sum(),
        unconverged_quenches: trajectories
            .iter()
            .map(|trajectory| trajectory.unconverged_quenches)
            .sum(),
        energy: f64::NAN,
        zero: f64::NAN,
        widest: 0.0,
        jumps: 0,
    })
}

fn average_trajectory<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    let mut sampler = PafiAnalyticSampler {
        surface,
        geometry,
        config,
        rejected: 0,
        unconverged: 0,
    };
    let mut average = average_hyperplane_with_sampler(&mut sampler, geometry, config)?;
    average.rejected = sampler.rejected;
    average.unconverged_quenches = sampler.unconverged;
    Ok(average)
}

/// Shoot the maximum-free-energy plane of a sampled PAFI profile.
///
/// The starts are the constrained-equilibrium configurations retained by
/// [`pafi_profile`] or [`pafi_frames`] on the profile's dividing plane.
/// Shooting is unconstrained in both time directions and is delegated to
/// [`transmission_coefficient`]. The PAFI plane normal points toward
/// increasing path arc, the product side of a reactant-first band.
#[cfg(feature = "ohtst")]
pub fn pafi_transmission<S: PointSurface>(
    surface: &S,
    profile: &FreeEnergyProfile,
    planes: &[HyperplaneAverage],
    config: &TransmissionConfig,
) -> Result<TransmissionCoefficient, SaddleError> {
    if planes.len() != profile.free_energy.len() || profile.dividing >= planes.len() {
        return Err(SaddleError::Shape(
            "PAFI transmission needs one sampled plane per profile value".into(),
        ));
    }
    check_dividing_plane(profile.dividing, planes.len(), config)?;
    let dividing = &planes[profile.dividing];
    if dividing.retained.is_empty() {
        return Err(SaddleError::Invalid(
            "PAFI transmission needs retained constrained samples; raise PafiConfig::retain".into(),
        ));
    }
    let mut dynamics = ohtst::AnalyticSurfaceDynamics::new(surface);
    transmission_coefficient(&mut dynamics, &dividing.plane, &dividing.retained, config)
}

/// Refuse a dividing plane at either end of a band of `planes` planes when
/// the ends of a shot are assigned by side of the plane.
///
/// Side assignment has no reactant behind the first plane and no product
/// ahead of the last, so the shots there would not measure a crossing.
#[cfg(feature = "ohtst")]
pub(crate) fn check_dividing_plane(
    dividing: usize,
    planes: usize,
    config: &TransmissionConfig,
) -> Result<(), SaddleError> {
    if (dividing == 0 || dividing + 1 == planes) && matches!(&config.end_state, EndState::Side) {
        return Err(SaddleError::Invalid(
            "an endpoint PAFI dividing plane needs basin end-state assignment".into(),
        ));
    }
    Ok(())
}

/// Estimate the PAFI mean force from any constrained-ensemble engine.
pub fn average_hyperplane_with_sampler<E: HyperplaneSampler>(
    sampler: &mut E,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    let n = geometry.positions.len();
    if geometry.tangent.len() != n || geometry.curvature.len() != n || geometry.jerk.len() != n {
        return Err(SaddleError::Shape(
            "PAFI path derivatives must match the position".into(),
        ));
    }
    if !geometry
        .tangent
        .iter()
        .chain(geometry.curvature.iter())
        .chain(geometry.jerk.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI path geometry"));
    }
    let path_plane = PathPlane::new(geometry, holds_com(config, n))?;
    let plane = Hyperplane::from_path(geometry.positions.view(), path_plane.normal.view())?;
    let mut budget = HyperplaneSampleBudget::default()
        .with_thermal(config.thermal)
        .with_seed(config.seed)
        .with_thermalize(config.thermalize)
        .with_stride(config.stride)
        .with_samples(config.samples)
        .with_positions(true);
    if let Some(region) = &config.region {
        budget = budget.with_region(Arc::clone(region));
    }
    let ensemble = sampler.sample_hyperplane(&plane, &budget)?;
    estimate_pafi(ensemble, &path_plane, config, &plane)
}

/// The hyperplane `(x - X) · t = 0` through a point of the path.
struct PathPlane<'a> {
    geometry: &'a PathGeometry,
    /// `t`, less its center-of-mass part when that is held.
    normal: Array1<f64>,
    /// `dt/ds`, in the same center-of-mass quotient as `normal`.
    curvature: Array1<f64>,
    /// `d²t/ds²`, in the same center-of-mass quotient as `normal`.
    jerk: Array1<f64>,
    /// `|t|²`.
    nn: f64,
    hold_com: bool,
}

impl<'a> PathPlane<'a> {
    fn new(geometry: &'a PathGeometry, hold_com: bool) -> Result<Self, SaddleError> {
        let mut normal = geometry.tangent.clone();
        let mut curvature = geometry.curvature.clone();
        let mut jerk = geometry.jerk.clone();
        if hold_com {
            remove_com(&mut normal);
            remove_com(&mut curvature);
            remove_com(&mut jerk);
        }
        let nn = normal.dot(&normal);
        if !(nn.is_finite() && nn > 0.0) {
            return Err(SaddleError::Invalid("PAFI tangent vanishes".into()));
        }
        Ok(Self {
            geometry,
            normal,
            curvature,
            jerk,
            nn,
            hold_com,
        })
    }
}

fn holds_com(config: &PafiConfig, n: usize) -> bool {
    config.fix_com && n >= 6 && n.is_multiple_of(3)
}

fn estimate_pafi(
    ensemble: HyperplaneEnsemble,
    plane: &PathPlane<'_>,
    config: &PafiConfig,
    hyperplane: &Hyperplane,
) -> Result<HyperplaneAverage, SaddleError> {
    if ensemble.observations.len() != config.samples {
        return Err(SaddleError::Invalid(
            "PAFI engine did not return the requested samples".into(),
        ));
    }
    let psi_of = |sample: &HyperplaneObservation| {
        sample
            .position
            .as_ref()
            .map_or(sample.jacobian, |position| {
                (position - &plane.geometry.positions).dot(&plane.curvature) / plane.nn
            })
    };
    let observations = ensemble
        .observations
        .iter()
        .map(|sample| match (sample.pafi_gradient, &sample.position) {
            (Some(value), _) => Ok(value),
            (None, Some(position)) => {
                let slope = -sample.constraint_force * plane.nn.sqrt();
                mean_force(slope, position, plane, config).map(|(value, _)| value)
            }
            (None, None) if config.mean_force == PafiMeanForce::ReferenceCode => {
                Ok(-sample.constraint_force * plane.nn.sqrt() * (1.0 - sample.jacobian))
            }
            (None, None) => Err(SaddleError::Invalid(
                "the published PAFI estimator needs the sampled positions".into(),
            )),
        })
        .collect::<Result<Vec<f64>, SaddleError>>()?;
    if observations.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("PAFI projected gradient"));
    }
    let equilibration = equilibrate(
        &observations,
        config.thermalize,
        config.stride,
        config.blocks,
    )?;
    let kept_observations = &ensemble.observations[equilibration.discard..];
    let kept_values = &observations[equilibration.discard..];
    let psi = kept_observations.iter().map(psi_of).sum::<f64>() / kept_observations.len() as f64;
    let retain_gap = kept_observations
        .len()
        .div_ceil(config.retain.max(1))
        .max(1);
    let retained = if config.retain == 0 {
        Vec::new()
    } else {
        kept_observations
            .iter()
            .enumerate()
            .filter(|(index, _)| index.is_multiple_of(retain_gap))
            .filter_map(|(_, sample)| sample.position.clone())
            .take(config.retain)
            .collect()
    };
    let (gradient, error, _, _) = block_mean(kept_values, config.blocks);
    Ok(HyperplaneAverage {
        plane: hyperplane.clone(),
        gradient,
        error,
        psi,
        samples: config.samples,
        tau: equilibration.tau,
        n_eff: equilibration.n_eff,
        evaluations: ensemble.evaluations,
        retained,
        equilibration,
        rejected: 0,
        unconverged_quenches: 0,
        energy: f64::NAN,
        zero: f64::NAN,
        widest: 0.0,
        jumps: 0,
    })
}

struct PafiAnalyticSampler<'a, S> {
    surface: &'a S,
    geometry: &'a PathGeometry,
    config: &'a PafiConfig,
    /// Thermalizations and stretches the last request sampled again.
    rejected: usize,
    /// Quenches of the last request that ended short of the tolerance.
    unconverged: usize,
}

impl<S: PointSurface> HyperplaneSampler for PafiAnalyticSampler<'_, S> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let geometry = self.geometry;
        let config = self.config;
        let path_plane = PathPlane::new(geometry, holds_com(config, geometry.positions.len()))?;
        let mut rng = Rng::new(budget.seed);
        let walk = Walk {
            surface: self.surface,
            plane,
            region: budget.region.as_ref(),
            hold_com: path_plane.hold_com,
            drift_scale: config.mobility * config.time_step,
            thermal_scale: config.mobility * budget.thermal * config.time_step,
            integrator: config.integrator,
        };
        let mut evaluations = 0;
        let mut unconverged = 0;
        let basin = match config.max_jump {
            Some(_) => {
                let quench = walk.minimum(&geometry.positions, &config.quench)?;
                evaluations += quench.evaluations;
                unconverged += usize::from(quench.unconverged);
                Some(quench.position)
            }
            None => None,
        };
        let mut rejected = 0;
        let mut left = |x: &Array1<f64>, evaluations: &mut usize| -> Result<bool, SaddleError> {
            let (Some(basin), Some(limit)) = (&basin, config.max_jump) else {
                return Ok(false);
            };
            let quench = walk.minimum(x, &config.quench)?;
            *evaluations += quench.evaluations;
            unconverged += usize::from(quench.unconverged);
            if StopNorm::MaxAtom.of((&quench.position - basin).view()) <= limit {
                return Ok(false);
            }
            rejected += 1;
            if rejected > config.blocks {
                return Err(SaddleError::Solver(format!(
                    "PAFI plane at s = {} left the basin of the path {rejected} times",
                    geometry.arc
                )));
            }
            Ok(true)
        };
        let (mut x, mut noise) = loop {
            let mut x = geometry.positions.clone();
            let mut noise = walk.draw(x.len(), &mut rng);
            for _ in 0..budget.thermalize {
                let gradient = walk.gradient(&x)?;
                walk.step(&mut x, gradient, &mut noise, &mut rng)?;
            }
            evaluations += budget.thermalize;
            if !left(&x, &mut evaluations)? {
                break (x, noise);
            }
        };
        let stretch = budget.samples.div_ceil(config.blocks);
        let mut observations = Vec::with_capacity(budget.samples);
        let mut kept = (x.clone(), noise.clone());
        while observations.len() < budget.samples {
            let count = stretch.min(budget.samples - observations.len());
            let mut records = Vec::with_capacity(count);
            for _ in 0..count {
                for offset in 0..budget.stride {
                    let gradient = walk.gradient(&x)?;
                    if offset == 0 {
                        let (pafi_gradient, jacobian) =
                            projected_gradient(&gradient, &x, &path_plane, config)?;
                        if !pafi_gradient.is_finite() || !jacobian.is_finite() {
                            return Err(SaddleError::NonFinite("PAFI projected gradient"));
                        }
                        records.push(HyperplaneObservation {
                            constraint_force: -gradient.dot(&plane.normal()),
                            pafi_gradient: Some(pafi_gradient),
                            jacobian,
                            position: budget.positions.then(|| x.clone()),
                            projected_force: None,
                            projected_velocity: None,
                            observables: Default::default(),
                        });
                    }
                    walk.step(&mut x, gradient, &mut noise, &mut rng)?;
                }
            }
            evaluations += count * budget.stride;
            if left(&x, &mut evaluations)? {
                (x, noise) = kept.clone();
                continue;
            }
            observations.extend(records);
            kept = (x.clone(), noise.clone());
        }
        self.rejected = rejected;
        self.unconverged = unconverged;
        Ok(HyperplaneEnsemble {
            observations,
            evaluations,
            acceptance: 1.0,
        })
    }
}

/// Overdamped Langevin dynamics confined to a [`Hyperplane`].
struct Walk<'a, S> {
    surface: &'a S,
    plane: &'a Hyperplane,
    region: Option<&'a Arc<dyn Region>>,
    hold_com: bool,
    /// `μ dt`.
    drift_scale: f64,
    /// `μ kT dt`.
    thermal_scale: f64,
    integrator: PafiIntegrator,
}

impl<S: PointSurface> Walk<'_, S> {
    fn gradient(&self, x: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        let (_energy, gradient) = self.surface.eval(x.view())?;
        if gradient.len() != x.len() || !gradient.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        Ok(gradient)
    }

    /// Remove the part of `vector` along the normal, and its center-of-mass
    /// part when held.
    fn project(&self, vector: &mut Array1<f64>) {
        self.plane.project_in_place(vector);
        if self.hold_com {
            remove_com(vector);
            self.plane.project_in_place(vector);
        }
    }

    /// A standard normal vector in the plane.
    fn draw(&self, len: usize, rng: &mut Rng) -> Array1<f64> {
        let mut noise = Array1::from_shape_simple_fn(len, || rng.normal());
        self.project(&mut noise);
        noise
    }

    /// One step from `x`, where the gradient is `gradient`. With the
    /// Leimkuhler-Matthews integrator, Appl. Math. Res. Express 2013, 34
    /// (2013), doi:10.1093/amrx/abs010, Eq. (3), the noise is the sum of
    /// the draw of the last step, `noise`, and a new one, which replaces
    /// it; stationary averages are then second order in `dt`, and exact on
    /// a harmonic plane. Euler-Maruyama uses the new draw alone.
    fn step(
        &self,
        x: &mut Array1<f64>,
        gradient: Array1<f64>,
        noise: &mut Array1<f64>,
        rng: &mut Rng,
    ) -> Result<(), SaddleError> {
        let plane = self.plane;
        let mut drift = gradient;
        self.project(&mut drift);
        let fresh = self.draw(x.len(), rng);
        let random_step = match self.integrator {
            PafiIntegrator::EulerMaruyama => &fresh * (2.0 * self.thermal_scale).sqrt(),
            PafiIntegrator::LeimkuhlerMatthews => {
                (&*noise + &fresh) * (0.5 * self.thermal_scale).sqrt()
            }
        };
        let mut proposal = &*x - &(&drift * self.drift_scale) + &random_step;
        *noise = fresh;
        plane.restore_in_place(&mut proposal);
        if let Some(region) = self.region
            && let Some(violation) = region.violation(proposal.view())?
        {
            if let Some(penetration) = violation.penetration {
                let mut normal = TangentProjector::project(plane, violation.normal.view());
                let scale = normal.dot(&normal).sqrt();
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(SaddleError::Invalid(
                        "region boundary has no PAFI in-plane normal".into(),
                    ));
                }
                normal /= scale;
                proposal.scaled_add(-2.0 * penetration / scale, &normal);
                plane.restore_in_place(&mut proposal);
                if !region.contains(proposal.view())? {
                    proposal = x.clone();
                }
            } else {
                proposal = x.clone();
            }
        }
        *x = proposal;
        Ok(())
    }

    /// The minimum in the plane that `quench` reaches from `start`.
    ///
    /// The first failure of the surface, a gradient of the wrong length or
    /// a value that is not finite ends the quench with that error. A stall
    /// of the solver, or the step cap, ends it where it is, with
    /// [`Quench::unconverged`] set.
    fn minimum(&self, start: &Array1<f64>, quench: &RelaxConfig) -> Result<Quench, SaddleError> {
        let n = start.len();
        let mut solver = Solver::new(
            quench.method.clone(),
            Control::default()
                .with_maxiter(quench.max_steps)
                .with_gtol(quench.force_tol)
                .with_stop_norm(StopNorm::MaxAtom)
                .with_istep(quench.initial_step),
            n,
        );
        solver.set_atom_maxmove(quench.max_atom_move);
        let evaluations = AtomicUsize::new(0);
        let failure: Mutex<Option<SaddleError>> = Mutex::new(None);
        let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            let outcome = self.surface.eval(x).and_then(|(energy, mut gradient)| {
                if gradient.len() != n {
                    return Err(SaddleError::Shape(
                        "PAFI quench gradient must match the path".into(),
                    ));
                }
                if !energy.is_finite() || !gradient.iter().all(|value| value.is_finite()) {
                    return Err(SaddleError::NonFinite("PAFI quench energy or gradient"));
                }
                self.project(&mut gradient);
                Ok((energy, gradient))
            });
            match outcome {
                Ok(pair) => pair,
                Err(error) => {
                    let mut first = match failure.lock() {
                        Ok(guard) => guard,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    first.get_or_insert(error);
                    (f64::NAN, Array1::from_elem(n, f64::NAN))
                }
            }
        });
        let mut x = start.clone();
        let result = solver.run(&oracle, &mut x);
        let recorded = match failure.into_inner() {
            Ok(first) => first,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(error) = recorded {
            return Err(error);
        }
        let unconverged = match result {
            Ok(report) => report.status != Status::Tolerance,
            Err(Error::FireStalled { .. } | Error::AcceptStalled { .. }) => true,
            Err(error) => return Err(SaddleError::Solver(format!("PAFI plane quench: {error}"))),
        };
        self.plane.restore_in_place(&mut x);
        Ok(Quench {
            position: x,
            evaluations: evaluations.into_inner(),
            unconverged,
        })
    }
}

/// The end of a quench in a plane.
struct Quench {
    /// The point the quench ended at, on the plane.
    position: Array1<f64>,
    /// Force evaluations it took.
    evaluations: usize,
    /// Whether the solver stalled or spent the step cap instead of
    /// reaching the force tolerance.
    unconverged: bool,
}

#[cfg(all(test, feature = "ohtst"))]
mod transmission_tests {
    use ndarray::{Array1, ArrayView1, array};

    use super::*;

    struct Flat;

    impl PointSurface for Flat {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.0, Array1::zeros(x.len())))
        }
    }

    fn plane() -> HyperplaneAverage {
        HyperplaneAverage {
            plane: Hyperplane::new(array![0.0], array![1.0]).unwrap(),
            gradient: 0.0,
            error: 0.0,
            psi: 0.0,
            samples: 1,
            tau: 0.5,
            n_eff: 1.0,
            evaluations: 1,
            retained: vec![array![0.0]],
            equilibration: EquilibrationReport {
                discard: 0,
                equilibration_steps: 0,
                tau: 0.5,
                n_eff: 1.0,
                split_half_drift: 0.0,
                split_half_error: 0.0,
                sigma_multiple: 4.0,
                drift_passed: true,
            },
            rejected: 0,
            unconverged_quenches: 0,
            energy: 0.0,
            zero: 0.0,
            widest: 0.0,
            jumps: 0,
        }
    }

    #[test]
    fn endpoint_transmission_requires_basin_assignment() {
        let planes = vec![plane(), plane()];
        for dividing in [0, 1] {
            let profile = FreeEnergyProfile {
                arc: vec![0.0, 1.0],
                gradient: vec![0.0; 2],
                gradient_error: vec![0.0; 2],
                free_energy: vec![0.0; 2],
                free_energy_error: vec![0.0; 2],
                barrier: 0.0,
                barrier_error: 0.0,
                dividing,
                samples: 2,
            };
            let error = pafi_transmission(&Flat, &profile, &planes, &TransmissionConfig::default())
                .unwrap_err();
            assert!(matches!(
                error,
                SaddleError::Invalid(message)
                    if message.contains("endpoint PAFI dividing plane")
            ));
        }
    }
}

/// Integrate plane gradients with the trapezoid rule.
///
/// `gradients` and `errors` have one entry for each of `arcs`, which must
/// strictly increase. `F` is zero at the first arc, the dividing plane is
/// the first maximum of `F` (the first arc if it never rises), and
/// [`FreeEnergyProfile::barrier`] is its value there. The errors are the
/// standard errors of the gradients, finite and not negative. A mismatch in
/// length is [`SaddleError::Shape`] and the rest [`SaddleError::Invalid`] or
/// [`SaddleError::NonFinite`]. [`FreeEnergyProfile::samples`] is 0.
///
/// [`pafi_profile`] calls this with its plane averages. Call it directly
/// to integrate gradients that came from somewhere else, such as the
/// reference code.
pub fn integrate_profile(
    arcs: &[f64],
    gradients: &[f64],
    errors: &[f64],
) -> Result<FreeEnergyProfile, SaddleError> {
    if arcs.len() < 2 || arcs.len() != gradients.len() || arcs.len() != errors.len() {
        return Err(SaddleError::Shape(
            "PAFI integration needs matching arcs, gradients and errors".into(),
        ));
    }
    let mut free = vec![0.0; arcs.len()];
    for i in 1..arcs.len() {
        let ds = arcs[i] - arcs[i - 1];
        if !(ds.is_finite() && ds > 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI arcs must strictly increase".into(),
            ));
        }
        if !gradients[i].is_finite() || !gradients[i - 1].is_finite() {
            return Err(SaddleError::NonFinite("PAFI gradient"));
        }
        free[i] = free[i - 1] + 0.5 * (gradients[i - 1] + gradients[i]) * ds;
    }
    let free_error = prefix_errors(arcs, errors)?;
    let dividing = dividing_index(&free);
    Ok(FreeEnergyProfile {
        barrier: free[dividing],
        barrier_error: free_error[dividing],
        dividing,
        arc: arcs.to_vec(),
        gradient: gradients.to_vec(),
        gradient_error: errors.to_vec(),
        free_energy: free,
        free_energy_error: free_error,
        samples: 0,
    })
}

/// Sample every requested arc and integrate.
///
/// `arcs` are values of the path parameter, at least two and strictly
/// increasing, inside [`ArcPath::bounds`] (a value outside is
/// [`SaddleError::Invalid`]). The planes are sampled in parallel. Plane
/// `k` takes the seed [`PafiConfig::seed`] `+ k + 1`, so a profile is
/// reproducible and the planes do not share noise. The returned vector
/// holds one [`HyperplaneAverage`] per arc, in order.
///
/// An error from any plane ends the call: a failed [`PointSurface`]
/// evaluation, a vanishing tangent, a gradient of the wrong length or not
/// finite, or [`PafiConfig::max_jump`] rejecting more stretches than there
/// are blocks.
///
/// When [`PafiConfig::target_error`] is set, planes are resampled until the
/// barrier error is at or below that target or [`PafiConfig::max_rounds`]
/// is spent. Each later round takes between 2 and 8 times
/// [`PafiConfig::samples`], by the square of the ratio of the barrier error
/// to the target, with half the thermalization and fresh noise, and is
/// inverse-variance combined with the sum so far.
pub fn pafi_profile<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    if arcs.len() < 2 {
        return Err(SaddleError::Invalid(
            "PAFI needs at least two hyperplanes".into(),
        ));
    }
    let mut planes = sample_planes(surface, path, arcs, config, 0)?;
    let mut profile = profile_of(arcs, &planes)?;
    let rounds = config.max_rounds.max(1);
    let mut round = 1usize;
    while let Some(target) = config.target_error {
        if profile.barrier_error <= target || round >= rounds {
            break;
        }
        if profile.barrier_error <= 0.0 {
            break;
        }
        let requested = (profile.barrier_error / target).powi(2);
        let at_least_double = requested.max(2.0);
        let factor = at_least_double.min(8.0);
        let samples = ((config.samples as f64) * factor).ceil() as usize;
        let extra = config
            .clone()
            .with_samples(samples.max(config.blocks * 2))
            .with_thermalize(config.thermalize / 2)
            .with_target_error(None);
        let more = sample_planes(surface, path, arcs, &extra, round as u64 * 10_000)?;
        for (plane, added) in planes.iter_mut().zip(more) {
            let (gradient, error) =
                combine(plane.gradient, plane.error, added.gradient, added.error);
            plane.gradient = gradient;
            plane.error = error;
            plane.samples += added.samples;
            plane.evaluations += added.evaluations;
            plane.tau = plane.tau.max(added.tau);
            plane.n_eff += added.n_eff;
            plane.rejected += added.rejected;
            plane.unconverged_quenches += added.unconverged_quenches;
            if plane.retained.len() < config.retain {
                plane.retained.extend(added.retained);
            }
        }
        profile = profile_of(arcs, &planes)?;
        round += 1;
    }
    profile.samples = planes.iter().map(|plane| plane.samples).sum();
    Ok((profile, planes))
}

fn sample_planes<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
    config: &PafiConfig,
    salt: u64,
) -> Result<Vec<HyperplaneAverage>, SaddleError> {
    arcs.par_iter()
        .enumerate()
        .map(|(index, arc)| {
            let geometry = path.geometry(*arc)?;
            let local = config.clone().with_seed(
                config
                    .seed
                    .wrapping_add(salt)
                    .wrapping_add(index as u64)
                    .wrapping_add(1),
            );
            average_hyperplane(surface, &geometry, &local)
        })
        .collect()
}

fn profile_of(
    arcs: &[f64],
    planes: &[HyperplaneAverage],
) -> Result<FreeEnergyProfile, SaddleError> {
    let gradients: Vec<f64> = planes.iter().map(|plane| plane.gradient).collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    integrate_profile(arcs, &gradients, &errors)
}

fn combine(mean_a: f64, error_a: f64, mean_b: f64, error_b: f64) -> (f64, f64) {
    let va = error_a * error_a;
    let vb = error_b * error_b;
    if va == 0.0 && vb == 0.0 {
        return (0.5 * (mean_a + mean_b), 0.0);
    }
    if va == 0.0 {
        return (mean_a, 0.0);
    }
    if vb == 0.0 {
        return (mean_b, 0.0);
    }
    let weight_a = 1.0 / va;
    let weight_b = 1.0 / vb;
    let mean = (weight_a * mean_a + weight_b * mean_b) / (weight_a + weight_b);
    let error = (1.0 / (weight_a + weight_b)).sqrt();
    (mean, error)
}

/// Integral of `∇E · dX/ds` with the system held on the path.
///
/// Every span between neighboring `arcs` is cut into 64 parts and the
/// trapezoid rule applied, so the arcs need not be the frames. The returned
/// profile has 64 planes per span and the last arc, its errors are zero,
/// and [`FreeEnergyProfile::samples`] is 0. `arcs` must be at least two and
/// strictly increasing, and the surface needs only [`PointSurface::eval`].
///
/// The integral is the energy along the spline, `E(X(s)) - E(X(s0))` up to
/// the quadrature, and passes through the image energies at the frames.
/// On a converged minimum-energy path the force in each hyperplane
/// vanishes, so this is also the zero-temperature limit of the PAFI
/// integral and reproduces the potential barrier. Compare
/// [`FreeEnergyProfile::barrier`] of this profile with the highest image
/// energy less the first before sampling: a path that is not a
/// minimum-energy path, or a surface that is not the one the images were
/// relaxed on, shows there.
pub fn zero_temperature_barrier<S: PointSurface>(
    surface: &S,
    path: &ArcPath,
    arcs: &[f64],
) -> Result<FreeEnergyProfile, SaddleError> {
    if arcs.len() < 2 {
        return Err(SaddleError::Invalid(
            "the zero-temperature integral needs at least two points".into(),
        ));
    }
    let arcs = dense_arcs(arcs, 64);
    let mut gradients = Vec::with_capacity(arcs.len());
    let errors = vec![0.0; arcs.len()];
    for arc in &arcs {
        let geometry = path.geometry(*arc)?;
        let (_energy, gradient) = surface.eval(geometry.positions.view())?;
        if gradient.len() != geometry.tangent.len() {
            return Err(SaddleError::Shape(
                "zero-temperature gradient must match the path".into(),
            ));
        }
        gradients.push(gradient.dot(&geometry.tangent));
    }
    integrate_profile(&arcs, &gradients, &errors)
}

/// `parts` samples on every span. One trapezoid per image is not enough
/// when the directional derivative curves, and the integral then sits
/// below the barrier.
fn dense_arcs(arcs: &[f64], parts: usize) -> Vec<f64> {
    let parts = parts.max(1);
    let mut dense = Vec::with_capacity(arcs.len().saturating_mul(parts));
    for window in arcs.windows(2) {
        for k in 0..parts {
            dense.push(window[0] + (window[1] - window[0]) * (k as f64) / parts as f64);
        }
    }
    if let Some(last) = arcs.last() {
        dense.push(*last);
    }
    dense
}

fn prefix_errors(arcs: &[f64], errors: &[f64]) -> Result<Vec<f64>, SaddleError> {
    let mut out = vec![0.0; arcs.len()];
    for (i, slot) in out.iter_mut().enumerate().skip(1) {
        let mut used = vec![0.0; arcs.len()];
        for k in 1..=i {
            let span = arcs[k] - arcs[k - 1];
            used[k - 1] += 0.5 * span;
            used[k] += 0.5 * span;
        }
        let mut variance = 0.0;
        for j in 0..=i {
            let sigma = errors[j];
            if !(sigma.is_finite() && sigma >= 0.0) {
                return Err(SaddleError::Invalid(
                    "PAFI gradient error must be finite and non-negative".into(),
                ));
            }
            variance += used[j] * used[j] * sigma * sigma;
        }
        *slot = variance.sqrt();
    }
    Ok(out)
}

fn dividing_index(free: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..free.len() {
        if free[i] > free[best] {
            best = i;
        }
    }
    best
}

/// The selected Eq. (8) estimator and
/// `ψ = (x - X) · X'' / |t|²` for the plane normal `t`.
fn projected_gradient(
    gradient: &Array1<f64>,
    x: &Array1<f64>,
    plane: &PathPlane<'_>,
    config: &PafiConfig,
) -> Result<(f64, f64), SaddleError> {
    mean_force(gradient.dot(&plane.normal), x, plane, config)
}

/// [`projected_gradient`] from the slope `∇E · t`.
fn mean_force(
    slope: f64,
    x: &Array1<f64>,
    plane: &PathPlane<'_>,
    config: &PafiConfig,
) -> Result<(f64, f64), SaddleError> {
    let deviation = x - &plane.geometry.positions;
    let curvature_deviation = deviation.dot(&plane.curvature);
    let psi = curvature_deviation / plane.nn;
    let paper_psi = 1.0 - psi;
    let force = slope * paper_psi;
    match config.mean_force {
        PafiMeanForce::ReferenceCode => Ok((force, psi)),
        PafiMeanForce::PublishedEquation8 => {
            if !(paper_psi.is_finite() && paper_psi > 0.0) {
                return Err(SaddleError::Solver(format!(
                    "PAFI locality condition failed at s = {}: psi = {paper_psi}",
                    plane.geometry.arc
                )));
            }
            // At fixed x, q = (x-X)·X'', n = X'·X':
            // q' = (x-X)·X''' - X'·X'', n' = 2 X'·X''.
            let tangent_curvature = plane.normal.dot(&plane.curvature);
            let q_derivative = deviation.dot(&plane.jerk) - tangent_curvature;
            let norm_derivative = 2.0 * tangent_curvature;
            let paper_psi_derivative =
                -q_derivative / plane.nn + curvature_deviation * norm_derivative / plane.nn.powi(2);
            let jacobian =
                config.thermal * (paper_psi_derivative / paper_psi - tangent_curvature / plane.nn);
            Ok((force + jacobian, psi))
        }
    }
}

pub(crate) fn remove_com(vector: &mut Array1<f64>) {
    let n = vector.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for c in 0..3 {
            mean[c] += vector[3 * atom + c];
        }
    }
    for value in &mut mean {
        *value /= n as f64;
    }
    for atom in 0..n {
        for c in 0..3 {
            vector[3 * atom + c] -= mean[c];
        }
    }
}

/// Sample one plane with the windowed chain.
fn average_windowed<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    let directions = plane_directions(surface, geometry.positions.view(), config.fix_com)?;
    let plane = Plane::new(geometry, directions.view())?;
    let mut chain = Chain::new(&plane, config.seed, config);
    chain.run(surface, &plane, config, config.thermalize, config.samples)?;
    if chain.values.is_empty() {
        return Err(SaddleError::Stalled(
            "PAFI plane left the basin of the path in every window".into(),
        ));
    }
    Ok(chain.average(config, &plane))
}

/// One hyperplane on every frame of `path`, each sampled from its own
/// minimum, integrated against the zero-temperature energy of those minima.
///
/// The planes are [`ArcPath::frame_planes`]: the full tangent, not a
/// truncation of it. Each frame is relaxed in its plane to
/// [`PafiConfig::relax_tol`], giving `x*_k` and `E*_k`; its chain starts
/// there and `ψ` is measured from there. With
/// `g = (∇E · ν) (τ · ν - (x - X) · dν/ds)`,
///
/// `F_k = E*_k - E*_0 + Σ trapezoid(⟨g⟩ - g(x*))`
///
/// over the arc length of the path. At zero temperature the sum vanishes
/// and the barrier is the energy of the in-plane minima whatever the
/// spacing of the frames, so the trapezoid acts only on the thermal part.
/// The dividing surface is the maximum of `F`, and passes go on as in
/// [`pafi_profile`]. Each plane's [`HyperplaneAverage::evaluations`]
/// includes its relaxation. [`pafi_profile`] keeps the spline hyperplanes.
pub fn pafi_frames<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    let planes = path.frame_planes()?;
    sample_frame_planes(surface, path, &planes, config)
}

/// [`pafi_frames`] on the truncated planes of [`ArcPath::truncated_planes`].
///
/// The barrier is the free energy of that truncated foliation. It depends
/// on `share`. [`harmonic_mask_shift`] maps it onto the full-tangent
/// barrier when the transverse motion is harmonic and both planes pass
/// through the same points; a minimum that has slid off the path point
/// adds `E(path) - E(minimum)` at the dividing frame and subtracts it at
/// the reactant.
pub fn pafi_truncated<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    share: f64,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    let planes = path.truncated_planes(share)?;
    sample_frame_planes(surface, path, &planes, config)
}

fn sample_frame_planes<S: PointSurface + Sync>(
    surface: &S,
    path: &ArcPath,
    planes: &FramePlanes,
    config: &PafiConfig,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError> {
    config.check()?;
    let n = path.frames[0].positions.len();
    let relaxed = path
        .frames
        .par_iter()
        .zip(&planes.normals)
        .map(|(frame, normal)| {
            let geometry = PathGeometry {
                positions: frame.positions.clone(),
                tangent: normal.clone(),
                curvature: Array1::zeros(n),
                jerk: Array1::zeros(n),
                arc: 0.0,
            };
            let directions = plane_directions(surface, frame.positions.view(), config.fix_com)?;
            let plane = Plane::new(&geometry, directions.view())?;
            match relax_in_plane(surface, &plane, frame.positions.clone(), config.relax_tol)? {
                (minimum, evaluations, true) => Ok((minimum, evaluations)),
                _ => Err(SaddleError::Stalled(format!(
                    "PAFI in-plane relaxation did not reach {:e} in {RELAX_STEPS} steps",
                    config.relax_tol
                ))),
            }
        })
        .collect::<Result<Vec<_>, SaddleError>>()?;
    let geometries: Vec<PathGeometry> = relaxed
        .iter()
        .zip(planes.normals.iter().zip(&planes.curvatures))
        .zip(&planes.arcs)
        .map(
            |(((positions, _), (normal, curvature)), arc)| PathGeometry {
                positions: positions.clone(),
                tangent: normal.clone(),
                curvature: curvature.clone(),
                jerk: Array1::zeros(n),
                arc: *arc,
            },
        )
        .collect();
    let (profile, mut averages) = sample_rounds(surface, &geometries, config, |averages| {
        control_variate(&planes.arcs, averages)
    })?;
    for (average, (_, steps)) in averages.iter_mut().zip(&relaxed) {
        average.evaluations += steps;
    }
    Ok((profile, averages))
}

fn sample_rounds<S, F>(
    surface: &S,
    geometries: &[PathGeometry],
    config: &PafiConfig,
    integrate: F,
) -> Result<(FreeEnergyProfile, Vec<HyperplaneAverage>), SaddleError>
where
    S: PointSurface + Sync,
    F: Fn(&[HyperplaneAverage]) -> Result<FreeEnergyProfile, SaddleError>,
{
    let planes = geometries
        .iter()
        .map(|geometry| {
            let directions = plane_directions(surface, geometry.positions.view(), config.fix_com)?;
            Plane::new(geometry, directions.view())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut chains: Vec<Chain> = planes
        .iter()
        .enumerate()
        .map(|(index, plane)| {
            let seed = config.seed.wrapping_add(index as u64).wrapping_add(1);
            Chain::new(plane, seed, config)
        })
        .collect();
    chains
        .par_iter_mut()
        .zip(&planes)
        .try_for_each(|(chain, plane)| {
            chain.run(surface, plane, config, config.thermalize, config.samples)
        })?;
    if let Some(index) = chains.iter().position(|chain| chain.values.is_empty()) {
        return Err(SaddleError::Stalled(format!(
            "PAFI plane {index} left the basin of the path in every window"
        )));
    }
    let mut averages: Vec<HyperplaneAverage> = chains
        .iter()
        .zip(&planes)
        .map(|(chain, plane)| chain.average(config, plane))
        .collect();
    let mut profile = integrate(&averages)?;
    let mut round = 1usize;
    while let Some(target) = config.target_error {
        if profile.barrier_error <= target
            || round >= config.max_rounds.max(1)
            || profile.barrier_error <= 0.0
        {
            break;
        }
        let short = (profile.barrier_error / target).powi(2) - 1.0;
        let prefix = profile.dividing + 1;
        chains[..prefix]
            .par_iter_mut()
            .zip(&planes[..prefix])
            .try_for_each(|(chain, plane)| {
                let more = ((chain.values.len() as f64) * short).ceil() as usize;
                let more = more.clamp(config.samples, 8 * config.samples);
                chain.run(surface, plane, config, 0, more)
            })?;
        averages = chains
            .iter()
            .zip(&planes)
            .map(|(chain, plane)| chain.average(config, plane))
            .collect();
        profile = integrate(&averages)?;
        round += 1;
    }
    profile.samples = averages.iter().map(|plane| plane.samples).sum();
    Ok((profile, averages))
}

fn control_variate(
    arcs: &[f64],
    planes: &[HyperplaneAverage],
) -> Result<FreeEnergyProfile, SaddleError> {
    let excess: Vec<f64> = planes
        .iter()
        .map(|plane| plane.gradient - plane.zero)
        .collect();
    let errors: Vec<f64> = planes.iter().map(|plane| plane.error).collect();
    let mut profile = integrate_profile(arcs, &excess, &errors)?;
    let base = planes[0].energy;
    for (free, plane) in profile.free_energy.iter_mut().zip(planes) {
        *free += plane.energy - base;
    }
    profile.dividing = dividing_index(&profile.free_energy);
    profile.barrier = profile.free_energy[profile.dividing];
    profile.barrier_error = profile.free_energy_error[profile.dividing];
    profile.gradient = planes.iter().map(|plane| plane.gradient).collect();
    Ok(profile)
}

/// The plane `(x - X) · ν = 0` and the vectors the estimator dots with.
struct Plane<'a> {
    geometry: &'a PathGeometry,
    /// Constraint. Its normal is `ν`, rigid directions removed.
    constraint: HyperplaneConstraint,
    /// `ν (τ · ν)`, so `∇E · tangent` is `(∇E · ν)(τ · ν)`.
    tangent: Array1<f64>,
    /// `(dν/ds) / (τ · ν)`, so `(1 - (x - X) · curvature)` supplies the
    /// rest of `τ · ν - (x - X) · dν/ds`.
    curvature: Array1<f64>,
}

impl<'a> Plane<'a> {
    fn new(geometry: &'a PathGeometry, directions: ArrayView2<f64>) -> Result<Self, SaddleError> {
        let n = geometry.positions.len();
        if geometry.tangent.len() != n || geometry.curvature.len() != n {
            return Err(SaddleError::Shape(
                "PAFI tangent and curvature must match the position".into(),
            ));
        }
        if !geometry
            .positions
            .iter()
            .chain(geometry.tangent.iter())
            .chain(geometry.curvature.iter())
            .all(|v| v.is_finite())
        {
            return Err(SaddleError::NonFinite("PAFI path geometry"));
        }
        // ν = Pτ / |Pτ|, P dropping the rigid directions (none, when there
        // are none). The estimator wants (∇E · ν) (τ · ν - (x - X) · dν/ds),
        // so the stored τ and dτ/ds, in whatever parameter the path has,
        // enter only through ν, τ · ν and dν/ds.
        let (normal, rate) = HyperplaneConstraint::orthogonal_rate(
            geometry.tangent.view(),
            geometry.curvature.view(),
            directions,
        )?;
        let alpha = geometry.tangent.dot(&normal);
        if !(alpha.is_finite() && alpha != 0.0) {
            return Err(SaddleError::Invalid(
                "PAFI tangent is orthogonal to its rigid-mode-free normal".into(),
            ));
        }
        let (tangent, curvature) = (&normal * alpha, &rate / alpha);
        let constraint =
            HyperplaneConstraint::through(geometry.positions.clone(), tangent.view(), directions)?;
        Ok(Self {
            geometry,
            constraint,
            tangent,
            curvature,
        })
    }

    fn project(&self, vector: &mut Array1<f64>) {
        self.constraint.project(vector);
    }

    fn restore(&self, x: &mut Array1<f64>) {
        self.constraint.restore(x);
    }

    fn noise(&self, rng: &mut Rng) -> Array1<f64> {
        let mut noise = Array1::from_shape_fn(self.tangent.len(), |_| rng.normal());
        self.project(&mut noise);
        noise
    }
}

fn plane_directions<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    fix_com: bool,
) -> Result<Array2<f64>, SaddleError> {
    if !fix_com {
        return Ok(Array2::zeros((0, positions.len())));
    }
    let modes = surface.excluded_modes(positions)?;
    if modes.nrows() > 0 {
        if modes.ncols() != positions.len() {
            return Err(SaddleError::Shape(
                "PAFI rigid modes must match the path".into(),
            ));
        }
        return Ok(modes);
    }
    let n = positions.len();
    if n >= 6 && n.is_multiple_of(3) {
        return Ok(uniform_translations(n / 3));
    }
    Ok(Array2::zeros((0, n)))
}

/// Windows a pass may run for each one it asks for, the dropped included.
const WINDOW_TRIES: usize = 4;

/// Times a chain that keeps no window halves [`PafiConfig::window`].
const WINDOW_HALVINGS: u32 = 4;

/// One overdamped chain on one plane, kept between passes.
struct Chain {
    x: Array1<f64>,
    previous: Option<Array1<f64>>,
    rng: Rng,
    values: Vec<f64>,
    psi_sum: f64,
    retained: Vec<Array1<f64>>,
    evaluations: usize,
    widest: f64,
    origin: (f64, f64),
    minimum: Option<Array1<f64>>,
    kept: Option<Array1<f64>>,
    jumps: usize,
    /// Steps between two jump tests.
    window: usize,
}

impl Chain {
    fn new(plane: &Plane, seed: u64, config: &PafiConfig) -> Self {
        let mut rng = Rng::new(seed);
        let previous =
            (config.integrator == Integrator::LeimkuhlerMatthews).then(|| plane.noise(&mut rng));
        Self {
            x: plane.geometry.positions.clone(),
            previous,
            rng,
            values: Vec::new(),
            psi_sum: 0.0,
            retained: Vec::new(),
            evaluations: 0,
            widest: 0.0,
            origin: (f64::NAN, f64::NAN),
            minimum: None,
            kept: None,
            jumps: 0,
            window: config.window,
        }
    }

    /// `thermalize` steps, then `samples` more recorded `config.stride`
    /// apart in windows of `config.window`, each put to
    /// [`PafiConfig::max_jump`]. A dropped window is run again, from where
    /// the last kept window ended with fresh noise, or from the origin,
    /// thermalized again, when no window has been kept; the pass gives up
    /// after [`WINDOW_TRIES`] times the windows it asked for. A chain that
    /// has kept no window when its pass gives up leaves the basin of the
    /// path faster than one window. It halves its window and runs the
    /// pass again from the origin, thermalizing in windows as well, and
    /// goes on halving, the pass discarded and the chain thermalized again
    /// from the origin, while a pass drops more windows than it keeps, up
    /// to [`WINDOW_HALVINGS`] times: a long window that ends in the basin
    /// may still have spent time outside.
    /// A `window` of 0 records one pass and does not test jumps.
    fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        samples: usize,
    ) -> Result<(), SaddleError> {
        if config.window == 0 && self.window == 0 {
            self.window = samples.max(1);
        }
        let goal = self.values.len() + samples;
        let shortest = (config.window >> WINDOW_HALVINGS).max(1);
        let mut thermalize = thermalize;
        loop {
            let mark = (
                self.values.len(),
                self.retained.len(),
                self.psi_sum,
                self.widest,
                self.jumps,
            );
            let kept = self.pass(surface, plane, config, thermalize, goal)?;
            if config.window == 0 || config.max_jump.is_none() || self.window <= shortest {
                return Ok(());
            }
            if self.window == config.window {
                if !self.values.is_empty() {
                    return Ok(());
                }
            } else if kept > self.jumps - mark.4 {
                return Ok(());
            } else {
                self.values.truncate(mark.0);
                self.retained.truncate(mark.1);
                self.psi_sum = mark.2;
                self.widest = mark.3;
                self.kept = None;
                self.x.assign(&plane.geometry.positions);
                if let Some(previous) = self.previous.as_mut() {
                    *previous = plane.noise(&mut self.rng);
                }
                thermalize = config.thermalize;
            }
            self.window = self.window.div_ceil(2).max(shortest);
        }
    }

    /// One pass of [`Self::run`] in windows of the current length, giving
    /// the windows it kept.
    fn pass<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        goal: usize,
    ) -> Result<usize, SaddleError> {
        let window = self.window;
        let windowed = config.window > 0 && window < config.window;
        let mut warm = thermalize;
        let mut kept = 0;
        let mut tries = WINDOW_TRIES * (goal - self.values.len()).div_ceil(window);
        if windowed {
            tries += WINDOW_TRIES * thermalize.div_ceil(window);
        }
        while self.values.len() < goal && tries > 0 {
            tries -= 1;
            let (heat, take) = if windowed && warm > 0 {
                (warm.min(window), 0)
            } else {
                (warm, (goal - self.values.len()).min(window))
            };
            let mark = (
                self.values.len(),
                self.retained.len(),
                self.psi_sum,
                self.widest,
            );
            self.advance(surface, plane, config, heat, take)?;
            warm -= heat;
            let Some(limit) = config.max_jump.filter(|_| config.window > 0) else {
                continue;
            };
            if !self.jumped(surface, plane, config, limit)? {
                self.kept = Some(self.x.clone());
                kept += 1;
                continue;
            }
            self.values.truncate(mark.0);
            self.retained.truncate(mark.1);
            self.psi_sum = mark.2;
            self.widest = mark.3;
            self.jumps += 1;
            match &self.kept {
                Some(checkpoint) => {
                    self.x.assign(checkpoint);
                    warm += heat;
                }
                None => {
                    self.x.assign(&plane.geometry.positions);
                    warm = config.thermalize;
                }
            }
            if let Some(previous) = self.previous.as_mut() {
                *previous = plane.noise(&mut self.rng);
            }
        }
        Ok(kept)
    }

    fn jumped<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        limit: f64,
    ) -> Result<bool, SaddleError> {
        if self.minimum.is_none() {
            let (minimum, evaluations, _) = relax_in_plane(
                surface,
                plane,
                plane.geometry.positions.clone(),
                config.jump_tol,
            )?;
            self.evaluations += evaluations;
            self.minimum = Some(minimum);
        }
        let (quenched, evaluations, settled) =
            relax_in_plane(surface, plane, self.x.clone(), config.jump_tol)?;
        self.evaluations += evaluations;
        let minimum = self.minimum.as_ref().expect("minimum");
        Ok(!settled || largest_atom(&(&quenched - minimum)) > limit)
    }

    fn advance<S: PointSurface>(
        &mut self,
        surface: &S,
        plane: &Plane,
        config: &PafiConfig,
        thermalize: usize,
        samples: usize,
    ) -> Result<(), SaddleError> {
        let geometry = plane.geometry;
        let n = self.x.len();
        let h = config.mobility * config.time_step;
        let retain_gap = config.samples.div_ceil(config.retain.max(1)).max(1);
        for step in 0..thermalize + config.stride * samples {
            let (energy, gradient) = surface.eval(self.x.view())?;
            if gradient.len() != n || !gradient.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI gradient must match the path".into(),
                ));
            }
            if self.evaluations == 0 {
                self.origin = (energy, gradient.dot(&plane.tangent));
            }
            self.evaluations += 1;
            if step >= thermalize && (step - thermalize).is_multiple_of(config.stride) {
                let (value, psi) = chord_projected_gradient(
                    &gradient,
                    &self.x,
                    &geometry.positions,
                    &plane.tangent,
                    &plane.curvature,
                );
                if !value.is_finite() || !psi.is_finite() {
                    return Err(SaddleError::NonFinite("PAFI projected gradient"));
                }
                let recorded = self.values.len();
                self.values.push(value);
                self.psi_sum += psi;
                self.widest = self
                    .widest
                    .max(largest_atom(&(&self.x - &geometry.positions)));
                if config.retain > 0
                    && recorded.is_multiple_of(retain_gap)
                    && self.retained.len() < config.retain
                {
                    self.retained.push(self.x.clone());
                }
            }
            let saved = config.region.is_some().then(|| self.x.clone());
            let mut drift = gradient;
            plane.project(&mut drift);
            let fresh = plane.noise(&mut self.rng);
            self.x.scaled_add(-h, &drift);
            match self.previous.as_mut() {
                Some(previous) => {
                    let scale = (0.5 * h * config.thermal).sqrt();
                    self.x.scaled_add(scale, previous);
                    self.x.scaled_add(scale, &fresh);
                    *previous = fresh;
                }
                None => self.x.scaled_add((2.0 * h * config.thermal).sqrt(), &fresh),
            }
            plane.restore(&mut self.x);
            if let Some(region) = &config.region {
                hold_region(
                    plane,
                    region.as_ref(),
                    &mut self.x,
                    saved.as_ref().expect("region snapshot"),
                )?;
            }
        }
        Ok(())
    }

    fn average(&self, config: &PafiConfig, plane: &Plane) -> HyperplaneAverage {
        let (gradient, error, tau, n_eff) = windowed_block_mean(&self.values, config.blocks);
        let equilibration = equilibrate(
            &self.values,
            config.thermalize,
            config.stride,
            config.blocks,
        )
        .unwrap_or(EquilibrationReport {
            discard: 0,
            equilibration_steps: config.thermalize,
            tau,
            n_eff,
            split_half_drift: 0.0,
            split_half_error: error,
            sigma_multiple: 4.0,
            drift_passed: false,
        });
        HyperplaneAverage {
            plane: plane.constraint.plane().clone(),
            gradient,
            error,
            psi: self.psi_sum / self.values.len().max(1) as f64,
            samples: self.values.len(),
            tau,
            n_eff,
            evaluations: self.evaluations,
            retained: self.retained.clone(),
            equilibration,
            rejected: 0,
            unconverged_quenches: 0,
            energy: self.origin.0,
            zero: self.origin.1,
            widest: self.widest,
            jumps: self.jumps,
        }
    }
}

fn hold_region(
    plane: &Plane,
    region: &dyn Region,
    x: &mut Array1<f64>,
    saved: &Array1<f64>,
) -> Result<(), SaddleError> {
    let Some(violation) = region.violation(x.view())? else {
        return Ok(());
    };
    if let Some(penetration) = violation.penetration {
        let mut normal = violation.normal;
        plane.project(&mut normal);
        let scale = normal.dot(&normal).sqrt();
        if !(scale.is_finite() && scale > 0.0) {
            return Err(SaddleError::Invalid(
                "region boundary has no PAFI in-plane normal".into(),
            ));
        }
        normal /= scale;
        x.scaled_add(-2.0 * penetration / scale, &normal);
        plane.restore(x);
        if !region.contains(x.view())? {
            x.assign(saved);
        }
    } else {
        x.assign(saved);
    }
    Ok(())
}

const RELAX_STEPS: usize = 10_000;

fn relax_in_plane<S: PointSurface>(
    surface: &S,
    plane: &Plane,
    mut x: Array1<f64>,
    tol: f64,
) -> Result<(Array1<f64>, usize, bool), SaddleError> {
    const MAX_MOVE: f64 = 0.1;
    let mut velocity = Array1::<f64>::zeros(x.len());
    let (mut dt, mut alpha, mut positive) = (0.05, 0.1, 0usize);
    for step in 0..RELAX_STEPS {
        let (_, gradient) = surface.eval(x.view())?;
        if gradient.len() != x.len() || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        let mut force = -gradient;
        plane.project(&mut force);
        if largest_atom(&force) <= tol {
            return Ok((x, step + 1, true));
        }
        if force.dot(&velocity) > 0.0 {
            let mix = alpha * norm(&velocity) / norm(&force);
            velocity *= 1.0 - alpha;
            velocity.scaled_add(mix, &force);
            if positive > 5 {
                dt = (dt * 1.1_f64).min(0.5);
                alpha *= 0.99;
            }
            positive += 1;
        } else {
            velocity.fill(0.0);
            dt *= 0.5;
            alpha = 0.1;
            positive = 0;
        }
        velocity.scaled_add(dt, &force);
        plane.project(&mut velocity);
        let mut move_by = &velocity * dt;
        let widest = largest_atom(&move_by);
        if widest > MAX_MOVE {
            move_by *= MAX_MOVE / widest;
        }
        x += &move_by;
        plane.restore(&mut x);
    }
    Ok((x, RELAX_STEPS, false))
}

fn chord_projected_gradient(
    gradient: &Array1<f64>,
    x: &Array1<f64>,
    origin: &Array1<f64>,
    tangent: &Array1<f64>,
    curvature: &Array1<f64>,
) -> (f64, f64) {
    let psi = (x - origin).dot(curvature);
    let parallel = gradient.dot(tangent);
    (parallel * (1.0 - psi), psi)
}

fn atom_width(len: usize) -> usize {
    if len.is_multiple_of(3) { 3 } else { 1 }
}

fn active_atoms(steps: &[Array1<f64>], width: usize, share: f64) -> Vec<usize> {
    let atoms = steps.first().map_or(0, |step| step.len() / width);
    if share >= 1.0 {
        return (0..atoms).collect();
    }
    let mut weight = vec![0.0; atoms];
    for step in steps {
        for (atom, slot) in weight.iter_mut().enumerate() {
            *slot += (0..width)
                .map(|c| step[atom * width + c].powi(2))
                .sum::<f64>();
        }
    }
    let total: f64 = weight.iter().sum();
    let mut order: Vec<usize> = (0..atoms).collect();
    order.sort_by(|a, b| weight[*b].total_cmp(&weight[*a]));
    let mut carried = 0.0;
    let mut active = Vec::new();
    for atom in order {
        if carried >= share * total {
            break;
        }
        carried += weight[atom];
        active.push(atom);
    }
    active.sort_unstable();
    active
}

fn largest_atom(vector: &Array1<f64>) -> f64 {
    let width = atom_width(vector.len());
    vector
        .exact_chunks(width)
        .into_iter()
        .map(|atom| atom.dot(&atom).sqrt())
        .fold(0.0, f64::max)
}

fn norm(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}

fn arc_derivative(differences: &[Array1<f64>], steps: &[f64]) -> Vec<Array1<f64>> {
    let last = differences.len();
    if last == 1 {
        return vec![&differences[0] / steps[0]; 2];
    }
    let end = |near: &Array1<f64>, far: &Array1<f64>, h1: f64, h2: f64| {
        (near * (h2 * (2.0 * h1 + h2)) - far * (h1 * h1)) / (h1 * h2 * (h1 + h2))
    };
    let mut derivatives = Vec::with_capacity(last + 1);
    derivatives.push(end(&differences[0], &differences[1], steps[0], steps[1]));
    for k in 1..last {
        let (before, after) = (steps[k - 1], steps[k]);
        derivatives.push(
            (&differences[k] * (before * before) + &differences[k - 1] * (after * after))
                / (before * after * (before + after)),
        );
    }
    derivatives.push(end(
        &differences[last - 1],
        &differences[last - 2],
        steps[last - 1],
        steps[last - 2],
    ));
    derivatives
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[derive(Default)]
    struct RecordingSampler {
        planes: Vec<Hyperplane>,
    }

    impl HyperplaneSampler for RecordingSampler {
        fn sample_hyperplane(
            &mut self,
            plane: &Hyperplane,
            budget: &HyperplaneSampleBudget,
        ) -> Result<HyperplaneEnsemble, SaddleError> {
            self.planes.push(plane.clone());
            let observations = (0..budget.samples)
                .map(|_| HyperplaneObservation {
                    constraint_force: -1.0,
                    pafi_gradient: Some(1.0),
                    jacobian: 0.0,
                    position: Some(plane.point().to_owned()),
                    projected_force: None,
                    projected_velocity: None,
                    observables: Default::default(),
                })
                .collect();
            Ok(HyperplaneEnsemble {
                observations,
                evaluations: budget.samples,
                acceptance: 1.0,
            })
        }
    }

    #[test]
    fn pafi_hands_the_sampler_one_unchanged_path_plane() {
        let geometry = PathGeometry {
            positions: array![0.3, -0.4],
            tangent: array![2.0, 0.0],
            curvature: Array1::zeros(2),
            jerk: Array1::zeros(2),
            arc: 0.7,
        };
        let config = PafiConfig::default()
            .with_thermalize(0)
            .with_samples(40)
            .with_blocks(4);
        let mut sampler = RecordingSampler::default();
        average_hyperplane_with_sampler(&mut sampler, &geometry, &config).unwrap();
        assert_eq!(sampler.planes.len(), 1);
        assert_eq!(sampler.planes[0].point(), geometry.positions);
        assert_eq!(sampler.planes[0].normal(), array![1.0, 0.0]);
    }
}

#[cfg(test)]
mod plane_tests {
    use ndarray::{Array1, Array2, array};

    use super::{PathGeometry, Plane, uniform_translations};

    /// The same curve with its parameter scaled: the tangent by `k`, the
    /// curvature by `k²`.
    fn geometry(scale: f64) -> PathGeometry {
        PathGeometry {
            positions: array![0.1, -0.2, 0.3, 0.7, 0.4, -0.5],
            tangent: array![1.0, 0.5, -0.2, 0.3, 0.8, 0.1] * scale,
            curvature: array![0.2, -0.1, 0.4, 0.6, 0.0, -0.3] * (scale * scale),
            jerk: Array1::zeros(6),
            arc: 0.0,
        }
    }

    #[test]
    fn the_plane_does_not_depend_on_the_parameter_of_the_path() {
        let none = Array2::zeros((0, 6));
        let rigid = uniform_translations(2);
        for directions in [none.view(), rigid.view()] {
            let base_geometry = geometry(1.0);
            let base = Plane::new(&base_geometry, directions).expect("plane");
            for scale in [0.2, 5.0] {
                let scaled_geometry = geometry(scale);
                let scaled = Plane::new(&scaled_geometry, directions).expect("plane");
                let drift = (scaled.constraint.normal().to_owned() - base.constraint.normal())
                    .mapv(f64::abs)
                    .sum();
                assert!(drift < 1.0e-12, "normal moved by {drift}");
                let curvature = (&scaled.curvature - &base.curvature).mapv(f64::abs).sum();
                assert!(curvature < 1.0e-12, "curvature moved by {curvature}");
                let tangent = (&scaled.tangent - &(&base.tangent * scale))
                    .mapv(f64::abs)
                    .sum();
                assert!(tangent < 1.0e-12, "tangent moved by {tangent}");
            }
        }
    }
}

#[cfg(test)]
mod estimator_tests {
    use ndarray::Array1;

    use super::{PafiConfig, PafiMeanForce, PathGeometry, PathPlane, projected_gradient};

    #[test]
    fn published_estimator_includes_the_analytic_jacobian() {
        let geometry = PathGeometry {
            positions: Array1::zeros(2),
            tangent: Array1::from_vec(vec![1.0, 1.0]),
            curvature: Array1::from_vec(vec![0.0, 1.0]),
            jerk: Array1::zeros(2),
            arc: 1.0,
        };
        let plane = PathPlane::new(&geometry, false).expect("plane");
        let x = Array1::from_vec(vec![-0.2, 0.2]);
        let gradient = Array1::zeros(2);
        let reference = PafiConfig::default()
            .with_thermal(0.3)
            .with_mean_force(PafiMeanForce::ReferenceCode);
        let published = reference
            .clone()
            .with_mean_force(PafiMeanForce::PublishedEquation8);
        let (reference_force, reference_deviation) =
            projected_gradient(&gradient, &x, &plane, &reference).expect("reference estimator");
        let (published_force, published_deviation) =
            projected_gradient(&gradient, &x, &plane, &published).expect("published estimator");
        let roundoff = 128.0 * f64::EPSILON;
        assert!(reference_force.abs() <= roundoff);
        assert!((reference_deviation - 0.1).abs() <= roundoff);
        assert!((published_deviation - reference_deviation).abs() <= roundoff);
        assert!((published_force - 0.05).abs() <= roundoff);
    }

    #[test]
    fn published_estimator_enforces_the_paper_locality_condition() {
        let geometry = PathGeometry {
            positions: Array1::zeros(2),
            tangent: Array1::from_vec(vec![1.0, 1.0]),
            curvature: Array1::from_vec(vec![0.0, 1.0]),
            jerk: Array1::zeros(2),
            arc: 1.0,
        };
        let plane = PathPlane::new(&geometry, false).expect("plane");
        let config = PafiConfig::default().with_mean_force(PafiMeanForce::PublishedEquation8);
        let error = projected_gradient(
            &Array1::zeros(2),
            &Array1::from_vec(vec![-2.0, 2.0]),
            &plane,
            &config,
        )
        .expect_err("zero paper psi violates locality");
        assert!(error.to_string().contains("locality condition"));
    }
}
