//! Projected average force integration along a path.
//!
//! Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018),
//! doi:10.1103/PhysRevLett.120.135503, give a path-based free-energy
//! gradient that does not need a collective variable. The same estimator
//! is what LAMMPS `fix pafi` averages (Swinburne's implementation of that
//! paper): with `n = dX/ds` the path derivative, `dn/ds` its derivative,
//! and `ψ = (x - X) · dn/ds`,
//!
//! `dF/ds = ⟨ (∇E · n) (1 - ψ) ⟩`.
//!
//! The average is over the hyperplane `(x - X) · n = 0`. Sampling here is
//! overdamped Langevin (Brownian) dynamics in that plane. The parallel
//! piece of the drift and of the noise is removed, and the plane is
//! restored after every step. A center-of-mass projection is optional and
//! is the `com yes` choice of that fix. The integral of `dF/ds` from the
//! reactant is the potential of mean force. Block averages of the recorded
//! samples supply the standard error. The block length is at least twice
//! the integrated autocorrelation time, so a correlated series is not
//! treated as independent.

use ndarray::{Array1, ArrayView1};
use rayon::prelude::*;

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::stats::{Rng, block_mean};

/// Overdamped hyperplane sampling.
///
/// Non-exhaustive: start from [`PafiConfig::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct PafiConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Mobility `μ` in `dx = -μ ∇E dt + sqrt(2 μ kT dt) ξ`.
    pub mobility: f64,
    /// Overdamped time step.
    pub time_step: f64,
    /// Steps discarded before the first recorded sample.
    pub thermalize: usize,
    /// Steps between recorded samples.
    pub stride: usize,
    /// Recorded samples. At least twice [`Self::blocks`].
    pub samples: usize,
    /// Contiguous blocks used for the standard error. At least 2.
    pub blocks: usize,
    /// Seed of the noise.
    pub seed: u64,
    /// Remove the mean Cartesian drift when the system has at least two atoms.
    pub fix_com: bool,
    /// How many recorded configurations to keep, evenly spaced.
    pub retain: usize,
    /// Stop early when the integrated barrier error is at or below this.
    /// `None` runs one pass.
    pub target_error: Option<f64>,
    /// Extra sampling passes used to reach [`Self::target_error`].
    pub max_rounds: usize,
}

with_setters!(PafiConfig {
    with_thermal => thermal: val f64,
    with_mobility => mobility: val f64,
    with_time_step => time_step: val f64,
    with_thermalize => thermalize: val usize,
    with_stride => stride: val usize,
    with_samples => samples: val usize,
    with_blocks => blocks: val usize,
    with_seed => seed: val u64,
    with_fix_com => fix_com: val bool,
    with_retain => retain: val usize,
    with_target_error => target_error: opt f64,
    with_max_rounds => max_rounds: val usize,
});

impl Default for PafiConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            mobility: 1.0,
            time_step: 1.0e-3,
            thermalize: 200,
            stride: 5,
            samples: 200,
            blocks: 5,
            seed: 1,
            fix_com: true,
            retain: 16,
            target_error: None,
            max_rounds: 4,
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
        if self.stride == 0 || self.blocks < 2 || self.samples < self.blocks * 2 {
            return Err(SaddleError::Invalid(
                "PAFI needs a positive stride, at least two blocks, and two samples per block"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// One frame of a piecewise path, with its arc length from the reactant.
#[derive(Clone, Debug)]
pub struct PathFrame {
    /// Cartesian coordinates of the frame.
    pub positions: Array1<f64>,
    /// Arc length from the first frame.
    pub arc: f64,
}

/// Cubic Hermite path. The parameter is the polyline arc length.
#[derive(Clone, Debug)]
pub struct ArcPath {
    frames: Vec<PathFrame>,
    /// Endpoint derivatives kept so a caller can rebuild the Hermite data.
    /// The geometry used for the integral is the polyline through the frames:
    /// a cubic spline between NEB images cuts the corner off a curved saddle
    /// and the zero-temperature integral then falls short of the barrier.
    #[allow(dead_code)]
    tangents: Vec<Array1<f64>>,
}

/// Position, path derivative `n = dX/ds`, and `dn/ds` at one arc length.
#[derive(Clone, Debug)]
pub struct PathGeometry {
    /// `X(s)`.
    pub positions: Array1<f64>,
    /// `n = dX/ds`.
    pub tangent: Array1<f64>,
    /// `dn/ds`.
    pub curvature: Array1<f64>,
    /// `s`.
    pub arc: f64,
}

impl ArcPath {
    /// Build the path from ordered frames. Arc lengths must strictly increase.
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
        }
        let mut tangents = Vec::with_capacity(frames.len());
        for i in 0..frames.len() {
            let tangent = if i == 0 {
                difference(
                    &frames[1].positions,
                    &frames[0].positions,
                    frames[1].arc - frames[0].arc,
                )
            } else if i + 1 == frames.len() {
                difference(
                    &frames[i].positions,
                    &frames[i - 1].positions,
                    frames[i].arc - frames[i - 1].arc,
                )
            } else {
                difference(
                    &frames[i + 1].positions,
                    &frames[i - 1].positions,
                    frames[i + 1].arc - frames[i - 1].arc,
                )
            };
            if tangent.iter().all(|v| *v == 0.0) {
                return Err(SaddleError::Invalid(
                    "a PAFI frame has a zero tangent".into(),
                ));
            }
            tangents.push(tangent);
        }
        Ok(Self { frames, tangents })
    }

    /// Frames in order, reactant first.
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

    /// Arc length of the reactant and of the product.
    pub fn bounds(&self) -> (f64, f64) {
        (self.frames[0].arc, self.frames.last().expect("frames").arc)
    }

    /// Arc length of every stored frame.
    pub fn arcs(&self) -> Vec<f64> {
        self.frames.iter().map(|frame| frame.arc).collect()
    }

    /// Polyline geometry at `arc`. The tangent is the chord of the segment.
    pub fn geometry(&self, arc: f64) -> Result<PathGeometry, SaddleError> {
        if !arc.is_finite() {
            return Err(SaddleError::NonFinite("PAFI arc"));
        }
        let last = self.frames.len() - 1;
        let arc = arc.clamp(self.frames[0].arc, self.frames[last].arc);
        let mut seg = last - 1;
        for i in 0..last {
            if arc <= self.frames[i + 1].arc {
                seg = i;
                break;
            }
        }
        let s0 = self.frames[seg].arc;
        let s1 = self.frames[seg + 1].arc;
        let dt = s1 - s0;
        let t = ((arc - s0) / dt).clamp(0.0, 1.0);
        let x0 = &self.frames[seg].positions;
        let x1 = &self.frames[seg + 1].positions;
        let n = x0.len();
        let mut position = Array1::zeros(n);
        let mut tangent = Array1::zeros(n);
        for i in 0..n {
            let step = x1[i] - x0[i];
            position[i] = x0[i] + t * step;
            tangent[i] = step / dt;
        }
        Ok(PathGeometry {
            positions: position,
            tangent,
            curvature: Array1::zeros(n),
            arc,
        })
    }
}

/// Mean free-energy gradient on one hyperplane.
#[derive(Clone, Debug)]
pub struct HyperplaneAverage {
    /// `dF/ds` in energy per arc length.
    pub gradient: f64,
    /// Standard error from the block means.
    pub error: f64,
    /// Mean of `ψ = (x - X) · dn/ds`.
    pub psi: f64,
    /// Recorded samples in the production run.
    pub samples: usize,
    /// Integrated autocorrelation time of the recorded series, in samples.
    pub tau: f64,
    /// `samples / (2 τ)`.
    pub n_eff: f64,
    /// Force evaluations spent on this plane, summed over rounds.
    pub evaluations: usize,
    /// Configurations kept from the production run, on the plane.
    pub retained: Vec<Array1<f64>>,
}

/// Potential of mean force from the first plane.
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

/// Sample one hyperplane and return the PAFI gradient.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    config.check()?;
    let n = geometry.positions.len();
    if geometry.tangent.len() != n || geometry.curvature.len() != n {
        return Err(SaddleError::Shape(
            "PAFI tangent and curvature must match the position".into(),
        ));
    }
    if !geometry
        .tangent
        .iter()
        .chain(geometry.curvature.iter())
        .all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("PAFI path geometry"));
    }
    let tangent_norm = norm(&geometry.tangent);
    if !(tangent_norm.is_finite() && tangent_norm > 0.0) {
        return Err(SaddleError::Invalid("PAFI tangent vanishes".into()));
    }
    let mut x = geometry.positions.clone();
    let mut rng = Rng::new(config.seed);
    let sigma = (2.0 * config.mobility * config.thermal * config.time_step).sqrt();
    let drift_scale = config.mobility * config.time_step;
    let total = config.thermalize + config.stride * config.samples;
    let mut observations = Vec::with_capacity(config.samples);
    let mut psi_sum = 0.0;
    let mut retained = Vec::new();
    let retain_gap = config.samples.div_ceil(config.retain.max(1)).max(1);
    for step in 0..total {
        let (_energy, gradient) = surface.eval(x.view())?;
        if gradient.len() != n || !gradient.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::Shape(
                "PAFI gradient must match the path".into(),
            ));
        }
        if step >= config.thermalize && (step - config.thermalize).is_multiple_of(config.stride) {
            let (value, psi) = projected_gradient(&gradient, &x, geometry);
            if !value.is_finite() || !psi.is_finite() {
                return Err(SaddleError::NonFinite("PAFI projected gradient"));
            }
            let recorded = observations.len();
            observations.push(value);
            psi_sum += psi;
            if config.retain > 0
                && recorded.is_multiple_of(retain_gap)
                && retained.len() < config.retain
            {
                retained.push(x.clone());
            }
        }
        let mut drift = gradient;
        project_out(&mut drift, geometry.tangent.view());
        let mut noise = Array1::zeros(n);
        for value in noise.iter_mut() {
            *value = rng.normal();
        }
        project_out(&mut noise, geometry.tangent.view());
        if config.fix_com && n >= 6 && n.is_multiple_of(3) {
            remove_com(&mut drift);
            remove_com(&mut noise);
            project_out(&mut drift, geometry.tangent.view());
            project_out(&mut noise, geometry.tangent.view());
        }
        x = &x - &(&drift * drift_scale) + &(&noise * sigma);
        let nn = geometry.tangent.dot(&geometry.tangent);
        let slip = (&x - &geometry.positions).dot(&geometry.tangent) / nn;
        x = &x - &(&geometry.tangent * slip);
    }
    if observations.len() != config.samples {
        return Err(SaddleError::Invalid(
            "PAFI did not record the requested samples".into(),
        ));
    }
    let (gradient, error, tau, n_eff) = block_mean(&observations, config.blocks);
    Ok(HyperplaneAverage {
        gradient,
        error,
        psi: psi_sum / config.samples as f64,
        samples: config.samples,
        tau,
        n_eff,
        evaluations: total,
        retained,
    })
}

/// Integrate plane gradients with the trapezoid rule.
///
/// The dividing surface is the maximum of the potential of mean force.
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
/// When [`PafiConfig::target_error`] is set, planes are resampled until the
/// barrier error is at or below that target or [`PafiConfig::max_rounds`]
/// is spent. Later rounds are inverse-variance combined with the first.
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
        let factor = (profile.barrier_error / target).powi(2).clamp(2.0, 8.0);
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
/// This is the zero-temperature limit of the PAFI integral: `ψ = 0` and
/// there is no thermal average. On a converged minimum-energy path it
/// reproduces the potential barrier.
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
    let arcs = dense_arcs(arcs, 24);
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

/// Eight samples on every span. One trapezoid per image is not enough
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

fn projected_gradient(
    gradient: &Array1<f64>,
    x: &Array1<f64>,
    geometry: &PathGeometry,
) -> (f64, f64) {
    let psi = (x - &geometry.positions).dot(&geometry.curvature);
    let parallel = gradient.dot(&geometry.tangent);
    (parallel * (1.0 - psi), psi)
}

fn difference(end: &Array1<f64>, start: &Array1<f64>, ds: f64) -> Array1<f64> {
    (end - start) / ds
}

fn project_out(vector: &mut Array1<f64>, tangent: ArrayView1<f64>) {
    let nn = tangent.dot(&tangent);
    if nn == 0.0 {
        return;
    }
    let coef = vector.dot(&tangent) / nn;
    *vector -= &(tangent.to_owned() * coef);
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

fn norm(vector: &Array1<f64>) -> f64 {
    vector.dot(vector).sqrt()
}
