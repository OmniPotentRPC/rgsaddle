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
//! The average is over the hyperplane `(x - X) · n = 0`. Production samples
//! arrive through [`HyperplaneSampler`]. The built-in overdamped Langevin
//! implementation is retained only as the deterministic analytic-surface
//! reference: it projects drift and noise, then restores the plane after
//! every step. A center-of-mass projection is optional and is the `com yes`
//! choice of `fix pafi`. The integral of `dF/ds` from the reactant is the
//! potential of mean force. Block averages of the equilibrated records
//! supply the standard error.

use ndarray::Array1;
use rayon::prelude::*;
use std::sync::Arc;

use crate::Region;
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::projector::{
    Hyperplane, HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget,
    HyperplaneSampler, ProjectedConstraint, TangentProjector,
};
use crate::stats::{EquilibrationReport, Rng, block_mean, equilibrate};

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
    /// Minimum engine warm-up floor before automated equilibration detection.
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
    /// Hard symmetry-unique region and optional additional predicates.
    pub region: Option<Arc<dyn Region>>,
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

impl PafiConfig {
    /// Restrict every fixed PAFI plane to a hard region.
    #[must_use]
    pub fn with_region(mut self, region: Arc<dyn Region>) -> Self {
        self.region = Some(region);
        self
    }
}

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
            region: None,
        }
    }
}

impl PafiConfig {
    pub(crate) fn check(&self) -> Result<(), SaddleError> {
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
    /// Automated equilibration, correlation and split-half drift report.
    pub equilibration: EquilibrationReport,
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

/// Sample one hyperplane and return the PAFI gradient.
pub fn average_hyperplane<S: PointSurface>(
    surface: &S,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    let mut sampler = PafiAnalyticSampler {
        surface,
        geometry,
        config,
    };
    average_hyperplane_with_sampler(&mut sampler, geometry, config)
}

/// Estimate the PAFI mean force from any constrained-ensemble engine.
pub fn average_hyperplane_with_sampler<E: HyperplaneSampler>(
    sampler: &mut E,
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
    let plane = Hyperplane::from_path(geometry.positions.view(), geometry.tangent.view())?;
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
    estimate_pafi(ensemble, geometry, config)
}

fn estimate_pafi(
    ensemble: HyperplaneEnsemble,
    geometry: &PathGeometry,
    config: &PafiConfig,
) -> Result<HyperplaneAverage, SaddleError> {
    if ensemble.observations.len() != config.samples {
        return Err(SaddleError::Invalid(
            "PAFI engine did not return the requested samples".into(),
        ));
    }
    let observations: Vec<f64> = ensemble
        .observations
        .iter()
        .map(|sample| {
            sample.pafi_gradient.unwrap_or_else(|| {
                let psi = sample
                    .position
                    .as_ref()
                    .map_or(sample.jacobian, |position| {
                        (position - &geometry.positions).dot(&geometry.curvature)
                    });
                -sample.constraint_force * norm(&geometry.tangent) * (1.0 - psi)
            })
        })
        .collect();
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
    let psi = kept_observations
        .iter()
        .map(|sample| {
            sample
                .position
                .as_ref()
                .map_or(sample.jacobian, |position| {
                    (position - &geometry.positions).dot(&geometry.curvature)
                })
        })
        .sum::<f64>()
        / kept_observations.len() as f64;
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
    })
}

struct PafiAnalyticSampler<'a, S> {
    surface: &'a S,
    geometry: &'a PathGeometry,
    config: &'a PafiConfig,
}

impl<S: PointSurface> HyperplaneSampler for PafiAnalyticSampler<'_, S> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let geometry = self.geometry;
        let config = self.config;
        let n = geometry.positions.len();
        let mut x = geometry.positions.clone();
        let mut rng = Rng::new(budget.seed);
        let sigma = (2.0 * config.mobility * budget.thermal * config.time_step).sqrt();
        let drift_scale = config.mobility * config.time_step;
        let total = budget.thermalize + budget.stride * budget.samples;
        let mut observations = Vec::with_capacity(budget.samples);
        for step in 0..total {
            let (_energy, gradient) = self.surface.eval(x.view())?;
            if gradient.len() != n || !gradient.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::Shape(
                    "PAFI gradient must match the path".into(),
                ));
            }
            if step >= budget.thermalize && (step - budget.thermalize).is_multiple_of(budget.stride)
            {
                let (pafi_gradient, jacobian) = projected_gradient(&gradient, &x, geometry);
                observations.push(HyperplaneObservation {
                    constraint_force: -gradient.dot(&plane.normal()),
                    pafi_gradient: Some(pafi_gradient),
                    jacobian,
                    position: budget.positions.then(|| x.clone()),
                    projected_force: None,
                    projected_velocity: None,
                    observables: Default::default(),
                });
            }
            let mut drift = gradient;
            plane.project_in_place(&mut drift);
            let mut noise = Array1::zeros(n);
            for value in noise.iter_mut() {
                *value = rng.normal();
            }
            plane.project_in_place(&mut noise);
            if config.fix_com && n >= 6 && n.is_multiple_of(3) {
                remove_com(&mut drift);
                remove_com(&mut noise);
                plane.project_in_place(&mut drift);
                plane.project_in_place(&mut noise);
            }
            let mut proposal = &x - &(&drift * drift_scale) + &(&noise * sigma);
            plane.restore_in_place(&mut proposal);
            if let Some(region) = &budget.region
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
            x = proposal;
        }
        Ok(HyperplaneEnsemble {
            observations,
            evaluations: total,
            acceptance: 1.0,
        })
    }
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
