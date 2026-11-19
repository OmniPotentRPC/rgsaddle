//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. The path `X(s)` is
//! the spline through the images of [`ArcPath`], and the hyperplane at `s`
//! is `(x - X) · t = 0`, with `t = X'` less its center-of-mass part when
//! the center of mass is held. With `ψ = (x - X) · X'' / |t|²`,
//!
//! `dF/ds = ⟨ (∇E · t) (1 - ψ) ⟩`.
//!
//! The paper's `ψ`, its Eq. (5), is `1 - ψ` here. [`PafiMeanForce`] selects
//! either the complete Eq. (8), including its analytic Jacobian derivative,
//! or the force term alone, which is what LAMMPS `fix pafi` and Swinburne's
//! public driver record. The force-only integral is the free energy of the
//! flat hyperplanes,
//! `-kT ln ∫ δ((x - X) · t / |t|) exp(-E/kT) dx`, which does not depend on
//! how the path is parametrized. Production samples arrive through
//! [`HyperplaneSampler`]. The built-in overdamped Langevin implementation,
//! the paper's Eq. (9), is retained only as the deterministic
//! analytic-surface reference. [`PafiIntegrator`] selects the direct
//! Euler-Maruyama discretization or the step of Leimkuhler and Matthews,
//! Appl. Math. Res. Express 2013, 34 (2013), doi:10.1093/amrx/abs010,
//! Eq. (3). The latter sums two consecutive noise draws, which makes
//! stationary averages second order in the step and exact on a harmonic
//! plane. The walk projects drift and noise, then restores the plane after
//! every step. Holding the center of mass is optional and is the `com yes`
//! choice of `fix pafi`. The integral of `dF/ds` from the reactant is the
//! potential of mean force. One trajectory uses block averages of the
//! equilibrated records for its standard error. An ensemble of
//! trajectories uses the spread of independent trajectory means, as
//! Swinburne's reference code does.
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

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use rgmin::{Control, Error, Method, Oracle, Solver, Status, StopNorm};

use crate::Region;
use crate::band::BandSurface;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler, ProjectedConstraint, TangentProjector,
};
use crate::relax::RelaxConfig;
use crate::reparam::{Spline, SplineEnds};
use crate::stats::{EquilibrationReport, Rng, block_mean, equilibrate};

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
        Ok(Self { spline })
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
}

impl PathGeometry {
    /// Position, tangent, and curvature of `spline` at `arc`.
    ///
    /// The polyline geometry of [`ArcPath`] stays a chord. This reads the
    /// cubic [`Spline`] instead, which is the path a fixed-energy
    /// trajectory integrates.
    ///
    /// # Errors
    ///
    /// The errors of [`Spline::sample`].
    pub fn on_spline(spline: &Spline, arc: f64) -> Result<Self, SaddleError> {
        let sample = spline.sample(arc)?;
        // The third derivative is the one-sided value from `derivatives`,
        // the same segment `ArcPath::geometry` reads.
        let jerk = spline.derivatives(arc)[3].clone();
        Ok(Self {
            positions: sample.value,
            tangent: sample.first,
            curvature: sample.second,
            jerk,
            arc,
        })
    }
}

/// One sample of inverted-potential motion on a normalized path.
#[derive(Clone, Copy, Debug)]
pub struct TrajectoryPoint {
    /// Time from the left turning point.
    pub time: f64,
    /// Normalized arc coordinate.
    pub arc: f64,
    /// `d(arc)/dt`.
    pub velocity: f64,
}

/// Mass-weighted path and the parallel force along it.
///
/// Knots are the normalized polyline arc length of the mass-weighted
/// images. Both splines use [`SplineEnds::Natural`]. [`Self::half`] integrates
/// the constrained dynamics of `-V` in that normalized coordinate.
#[derive(Clone, Debug)]
pub struct FixedEnergyTrajectory {
    coordinate: Spline,
    parallel_force: Spline,
}

impl FixedEnergyTrajectory {
    /// Spline images whose Cartesian gradients are already known.
    ///
    /// `positions` and `gradients` have one image per row.
    /// `masses` has one positive entry per coordinate.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Shape`] when the arrays disagree or there are fewer
    /// than two images. [`SaddleError::Invalid`] when the mass-weighted
    /// path has no arc length, and the errors of [`Spline::new`].
    pub fn from_gradients(
        positions: &Array2<f64>,
        gradients: &Array2<f64>,
        masses: ArrayView1<f64>,
    ) -> Result<Self, SaddleError> {
        let images = positions.nrows();
        let width = positions.ncols();
        if images < 2 || width == 0 {
            return Err(SaddleError::Shape(
                "a fixed-energy path needs at least two nonempty images".into(),
            ));
        }
        if gradients.nrows() != images || gradients.ncols() != width || masses.len() != width {
            return Err(SaddleError::Shape(
                "fixed-energy path gradients and masses must match the images".into(),
            ));
        }
        let sqrt_mass = masses.mapv(f64::sqrt);
        let mut coordinate = Array2::zeros((images, width));
        for image in 0..images {
            for coordinate_index in 0..width {
                coordinate[(image, coordinate_index)] =
                    positions[(image, coordinate_index)] * sqrt_mass[coordinate_index];
            }
        }
        let mut knots = vec![0.0; images];
        for image in 1..images {
            let mut length2 = 0.0;
            for coordinate_index in 0..width {
                let step = coordinate[(image, coordinate_index)]
                    - coordinate[(image - 1, coordinate_index)];
                length2 += step * step;
            }
            knots[image] = knots[image - 1] + length2.sqrt();
        }
        let total = knots[images - 1];
        if !(total.is_finite() && total > 0.0) {
            return Err(SaddleError::Invalid(
                "fixed-energy path has zero arc length".into(),
            ));
        }
        for value in &mut knots {
            *value /= total;
        }
        let coordinate_spline = Spline::new(&knots, &coordinate, SplineEnds::Natural)?;
        let mut parallel = Array2::zeros((images, 1));
        for image in 0..images {
            let geometry = PathGeometry::on_spline(&coordinate_spline, knots[image])?;
            let mut tangent_norm2 = 0.0;
            let mut force = 0.0;
            for coordinate_index in 0..width {
                let direction = geometry.tangent[coordinate_index];
                tangent_norm2 += direction * direction;
                force +=
                    -gradients[(image, coordinate_index)] * direction / sqrt_mass[coordinate_index];
            }
            parallel[(image, 0)] = force / tangent_norm2.sqrt();
        }
        Ok(Self {
            coordinate: coordinate_spline,
            parallel_force: Spline::new(&knots, &parallel, SplineEnds::Natural)?,
        })
    }

    /// Evaluate each image once, in order, and spline the path.
    ///
    /// # Errors
    ///
    /// The errors of [`BandSurface::eval`] and [`Self::from_gradients`].
    pub fn from_images<S: BandSurface>(
        surface: &S,
        positions: &Array2<f64>,
        masses: ArrayView1<f64>,
    ) -> Result<Self, SaddleError> {
        let images = positions.nrows();
        let width = positions.ncols();
        let mut gradients = Array2::zeros((images, width));
        for image in 0..images {
            let mut one = Array2::zeros((1, width));
            one.row_mut(0).assign(&positions.row(image));
            let mut energies = Array1::zeros(1);
            let mut gradient = Array2::zeros((1, width));
            surface.eval(one.view(), &mut energies, &mut gradient)?;
            gradients.row_mut(image).assign(&gradient.row(0));
        }
        Self::from_gradients(positions, &gradients, masses)
    }

    /// Mass-weighted coordinate spline. Knots lie in `[0, 1]`.
    pub fn coordinate(&self) -> &Spline {
        &self.coordinate
    }

    /// Parallel-force spline, one value per knot.
    pub fn parallel_force(&self) -> &Spline {
        &self.parallel_force
    }

    /// Acceleration of the normalized arc on the inverted potential.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when the coordinate tangent has no
    /// length, and the errors of [`Spline::sample`].
    pub fn acceleration(&self, arc: f64, velocity: f64) -> Result<f64, SaddleError> {
        let geometry = PathGeometry::on_spline(&self.coordinate, arc)?;
        let force = self.parallel_force.sample(arc)?.value[0];
        let metric = geometry.tangent.dot(&geometry.tangent);
        if !(metric.is_finite() && metric > 0.0) {
            return Err(SaddleError::Invalid(
                "path spline has a singular metric".into(),
            ));
        }
        let bend = geometry.tangent.dot(&geometry.curvature);
        Ok((-force * metric.sqrt() - bend * velocity * velocity) / metric)
    }

    /// Integrate from the left endpoint until the path ends or the
    /// velocity turns back. `time_step` is the fixed step of that
    /// integration. `max_steps` is how many of those steps may be taken.
    ///
    /// A proposal that reaches or passes the end knot is an integration
    /// event: the corrector acceleration is sampled at that knot, and the
    /// stored arc is the knot. No state past the knot is formed.
    ///
    /// # Errors
    ///
    /// [`SaddleError::Invalid`] when `time_step` is not positive and
    /// finite, or when a step stops advancing.
    /// [`SaddleError::Exhausted`] when `max_steps` is used up. The errors
    /// of [`Self::acceleration`].
    pub fn half(
        &self,
        time_step: f64,
        max_steps: usize,
    ) -> Result<Vec<TrajectoryPoint>, SaddleError> {
        if !(time_step.is_finite() && time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "fixed-energy trajectory step must be positive and finite".into(),
            ));
        }
        let mut trajectory = vec![TrajectoryPoint {
            time: 0.0,
            arc: 0.0,
            velocity: 0.0,
        }];
        let mut steps = 0usize;
        loop {
            if steps == max_steps {
                return Err(SaddleError::Exhausted(
                    "fixed-energy trajectory used its step budget".into(),
                ));
            }
            steps += 1;
            let current = *trajectory.last().ok_or_else(|| {
                SaddleError::Solver("fixed-energy trajectory lost its initial point".into())
            })?;
            let a0 = self.acceleration(current.arc, current.velocity)?;
            let proposed_arc =
                current.arc + time_step * current.velocity + 0.5 * time_step.powi(2) * a0;
            let predicted_velocity = current.velocity + time_step * a0;
            if !(proposed_arc.is_finite() && predicted_velocity.is_finite()) {
                return Err(SaddleError::Invalid(format!(
                    "inverted-potential trajectory stopped at s={:.17e}, v={:.17e}, a={:.17e}",
                    current.arc, current.velocity, a0
                )));
            }
            if proposed_arc >= 1.0 {
                let a1 = self.acceleration(1.0, predicted_velocity)?;
                let proposed_velocity = current.velocity + 0.5 * time_step * (a0 + a1);
                let ds = proposed_arc - current.arc;
                if !(ds.is_finite() && ds > 0.0 && proposed_velocity.is_finite()) {
                    return Err(SaddleError::Invalid(format!(
                        "inverted-potential trajectory stopped at s={:.17e}, v={:.17e}, a={:.17e}",
                        current.arc, current.velocity, a0
                    )));
                }
                let fraction = (1.0 - current.arc) / ds;
                trajectory.push(TrajectoryPoint {
                    time: current.time + fraction * time_step,
                    arc: 1.0,
                    velocity: current.velocity + fraction * (proposed_velocity - current.velocity),
                });
                return Ok(trajectory);
            }
            let a1 = self.acceleration(proposed_arc, predicted_velocity)?;
            let proposed_velocity = current.velocity + 0.5 * time_step * (a0 + a1);
            let next_time = current.time + time_step;
            if current.velocity > 0.0 && proposed_velocity <= 0.0 {
                let fraction = current.velocity / (current.velocity - proposed_velocity);
                trajectory.push(TrajectoryPoint {
                    time: current.time + fraction * time_step,
                    arc: current.arc + fraction * (proposed_arc - current.arc),
                    velocity: 0.0,
                });
                return Ok(trajectory);
            }
            if !(proposed_arc > current.arc
                && proposed_velocity.is_finite()
                && next_time.is_finite())
            {
                return Err(SaddleError::Invalid(format!(
                    "inverted-potential trajectory stopped at s={:.17e}, v={:.17e}, a={:.17e}",
                    current.arc, current.velocity, a0
                )));
            }
            trajectory.push(TrajectoryPoint {
                time: next_time,
                arc: proposed_arc,
                velocity: proposed_velocity,
            });
        }
    }

    /// Round trip: twice the time [`Self::half`] takes to cross or turn.
    ///
    /// # Errors
    ///
    /// The errors of [`Self::half`].
    pub fn period(&self, time_step: f64, max_steps: usize) -> Result<f64, SaddleError> {
        let trajectory = self.half(time_step, max_steps)?;
        let last = trajectory.last().ok_or_else(|| {
            SaddleError::Shape("fixed-energy trajectory produced no points".into())
        })?;
        Ok(2.0 * last.time)
    }
}

/// Cubic Hermite arc at `time` on a half-trajectory.
///
/// An exact node time returns that node's arc. Between nodes the
/// interpolant uses both endpoint velocities.
///
/// # Errors
///
/// [`SaddleError::Shape`] when `trajectory` has fewer than two points.
pub fn arc_at_time(trajectory: &[TrajectoryPoint], time: f64) -> Result<f64, SaddleError> {
    if trajectory.len() < 2 {
        return Err(SaddleError::Shape(
            "arc interpolation needs two trajectory points".into(),
        ));
    }
    let index = match trajectory.binary_search_by(|point| point.time.total_cmp(&time)) {
        Ok(index) => return Ok(trajectory[index].arc),
        Err(index) => index.saturating_sub(1).min(trajectory.len() - 2),
    };
    let left = trajectory[index];
    let right = trajectory[index + 1];
    let h = right.time - left.time;
    let u = (time - left.time) / h;
    let h00 = 2.0 * u.powi(3) - 3.0 * u.powi(2) + 1.0;
    let h10 = u.powi(3) - 2.0 * u.powi(2) + u;
    let h01 = -2.0 * u.powi(3) + 3.0 * u.powi(2);
    let h11 = u.powi(3) - u.powi(2);
    Ok(h00 * left.arc + h10 * h * left.velocity + h01 * right.arc + h11 * h * right.velocity)
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
    estimate_pafi(ensemble, &path_plane, config)
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

    #[test]
    fn a_straight_inverted_path_has_constant_acceleration() {
        let positions = array![[0.0], [1.0], [2.0]];
        let gradients = array![[2.0], [2.0], [2.0]];
        let path =
            FixedEnergyTrajectory::from_gradients(&positions, &gradients, array![1.0].view())
                .unwrap();
        let geometry = PathGeometry::on_spline(path.coordinate(), 0.5).unwrap();
        assert!((geometry.tangent[0] - 2.0).abs() < 1e-12);
        assert!(geometry.curvature[0].abs() < 1e-12);
        let acceleration = path.acceleration(0.25, 0.0).unwrap();
        assert!((acceleration - 1.0).abs() < 1e-12);
        let half = path.half(0.1, 10_000).unwrap();
        assert!((half.last().unwrap().arc - 1.0).abs() < 1e-12);
        assert!(matches!(path.half(0.1, 1), Err(SaddleError::Exhausted(_))));
        let period = path.period(0.1, 10_000).unwrap();
        assert!((period - 2.0 * half.last().unwrap().time).abs() < 1e-15);
        let mid = arc_at_time(&half, 0.0).unwrap();
        assert_eq!(mid, 0.0);
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
