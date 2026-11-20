//! Constant-temperature sampling inside a hyperplane.
//!
//! The dynamics is Langevin at unit mass in mass-weighted coordinates,
//! restricted to the plane and, optionally, to fixed values of the
//! surface's excluded modes. The constraint is linear, so the restricted
//! Boltzmann density needs no Fixman term: the motion, the friction and the
//! noise are projected onto the tangent space of the plane, and positions
//! are projected back onto the plane after every drift.
//! [`PlaneIntegrator::Baoab`] is the unadjusted BAOAB splitting of
//! Leimkuhler and Matthews: it samples a harmonic plane exactly for any
//! stable step, `ω h < 2`, and its configurational error is `O(h²)`
//! otherwise. Walls reflect the drift specularly, which keeps that.
//! [`PlaneIntegrator::Ghmc`] refreshes the velocity, proposes one
//! velocity-Verlet step and accepts it on the change of the total energy,
//! reversing the velocity on a rejection or at a wall. That is generalized
//! hybrid Monte Carlo, exact for any step, and the check on the BAOAB bias.

use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, s};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::control::{HarmonicReference, PlaneControl};
use crate::ohtst::plane::{Hyperplane, orthonormal_rows};
use crate::projector::{
    HyperplaneEnsemble, HyperplaneObservation, HyperplaneSampleBudget, HyperplaneSampler,
};
use crate::region::Region;
use crate::stats::{EquilibrationReport, Rng, autocorr_time, batch_blocks, equilibrate};

/// Random projections of `(ν · ∇V) (q - p)` whose correlation times
/// join that of `ν · ∇V` in the block width.
const PROJECTIONS: usize = 3;

/// Upper bound on the stored partial sums, so the memory does not grow
/// with the number of records.
const CHUNKS: usize = 256;

/// Reflection passes per drift, for a step that crosses several walls.
const REFLECTIONS: usize = 8;

/// How [`sample_plane`] and [`reactant_density`](crate::ohtst::reactant_density)
/// integrate the restricted Langevin dynamics.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaneIntegrator {
    /// Unadjusted projected BAOAB, one force call per step.
    #[default]
    Baoab,
    /// Generalized hybrid Monte Carlo: exact, one force call per step and
    /// a rejection rate that grows with the step.
    Ghmc,
}

/// Langevin sampling at constant temperature.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LangevinConfig {
    /// `kT` in the energy unit of the surface.
    pub thermal: f64,
    /// Time step in the time unit of the mass-weighted surface (10.18 fs
    /// for eV, Å and amu).
    pub time_step: f64,
    /// Langevin friction `γ` in inverse time. Must be positive.
    pub friction: f64,
    /// The integrator. BAOAB needs `h` below `2/ω` for the stiffest
    /// in-plane frequency `ω`, and its bias falls as `h²`.
    pub integrator: PlaneIntegrator,
    /// Stein control variates from a fixed Hessian, applied by
    /// [`sample_plane`] to the normal force and the torques. `None` turns
    /// them off.
    pub control: Option<HarmonicReference>,
    /// Minimum engine warm-up floor before automated equilibration detection.
    pub burn_in: usize,
    /// Steps between records.
    pub stride: usize,
    /// Records.
    pub records: usize,
    /// Most blocks for the standard error. A block is at least twice the
    /// integrated autocorrelation time.
    pub blocks: usize,
    /// Seed of the noise.
    pub seed: u64,
    /// Configurations kept, evenly spaced over the records.
    pub retain: usize,
    /// Hold the surface's excluded modes, such as rigid translations, at
    /// their starting values.
    pub hold_excluded: bool,
    /// Hard symmetry-unique region and optional additional predicates.
    pub region: Option<Arc<dyn Region>>,
}

impl Default for LangevinConfig {
    fn default() -> Self {
        Self {
            thermal: 0.02585,
            time_step: 0.05,
            friction: 1.0,
            integrator: PlaneIntegrator::Baoab,
            control: None,
            burn_in: 500,
            stride: 5,
            records: 2000,
            blocks: 20,
            seed: 1,
            retain: 16,
            hold_excluded: true,
            region: None,
        }
    }
}

impl LangevinConfig {
    /// Set [`Self::thermal`].
    #[must_use]
    pub fn with_thermal(mut self, thermal: f64) -> Self {
        self.thermal = thermal;
        self
    }

    /// Set [`Self::time_step`].
    #[must_use]
    pub fn with_time_step(mut self, time_step: f64) -> Self {
        self.time_step = time_step;
        self
    }

    /// Set [`Self::friction`].
    #[must_use]
    pub fn with_friction(mut self, friction: f64) -> Self {
        self.friction = friction;
        self
    }

    /// Set [`Self::control`] to `reference`.
    #[must_use]
    pub fn with_control(mut self, reference: HarmonicReference) -> Self {
        self.control = Some(reference);
        self
    }

    /// Set [`Self::integrator`].
    #[must_use]
    pub fn with_integrator(mut self, integrator: PlaneIntegrator) -> Self {
        self.integrator = integrator;
        self
    }

    /// Set [`Self::burn_in`].
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Set [`Self::stride`].
    #[must_use]
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set [`Self::records`].
    #[must_use]
    pub fn with_records(mut self, records: usize) -> Self {
        self.records = records;
        self
    }

    /// Set [`Self::blocks`].
    #[must_use]
    pub fn with_blocks(mut self, blocks: usize) -> Self {
        self.blocks = blocks;
        self
    }

    /// Set [`Self::seed`].
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set [`Self::retain`].
    #[must_use]
    pub fn with_retain(mut self, retain: usize) -> Self {
        self.retain = retain;
        self
    }

    /// Set [`Self::hold_excluded`].
    #[must_use]
    pub fn with_hold_excluded(mut self, hold_excluded: bool) -> Self {
        self.hold_excluded = hold_excluded;
        self
    }

    /// Restrict constrained sampling to a hard region.
    #[must_use]
    pub fn with_region(mut self, region: Arc<dyn Region>) -> Self {
        self.region = Some(region);
        self
    }

    pub(crate) fn check(&self) -> Result<(), SaddleError> {
        if !(self.thermal.is_finite() && self.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler kT must be finite and positive".into(),
            ));
        }
        if !(self.time_step.is_finite() && self.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler time step must be finite and positive".into(),
            ));
        }
        if !(self.friction.is_finite() && self.friction > 0.0) {
            return Err(SaddleError::Invalid(
                "sampler friction must be finite and positive".into(),
            ));
        }
        if self.blocks == 0 {
            return Err(SaddleError::MissingBlocks { context: "sampler" });
        }
        if self.records == 0 {
            return Err(SaddleError::MissingSamples { context: "sampler" });
        }
        if self.stride == 0 || self.blocks < 2 || self.records < 2 * self.blocks {
            return Err(SaddleError::Invalid(
                "sampler needs a positive stride, two blocks, and two records per block".into(),
            ));
        }
        if let Some(reference) = &self.control {
            reference.check()?;
        }
        Ok(())
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

fn contiguous(a: Array1<f64>) -> Array1<f64> {
    if a.is_standard_layout() {
        a
    } else {
        a.as_standard_layout().into_owned()
    }
}

fn slice(a: &Array1<f64>) -> &[f64] {
    a.as_slice().expect("sampler buffers are contiguous")
}

fn slice_mut(a: &mut Array1<f64>) -> &mut [f64] {
    a.as_slice_mut().expect("sampler buffers are contiguous")
}

/// One step of constrained Langevin dynamics with an optional Metropolis
/// test. Positions keep `anchor`'s component along every `held` direction,
/// and a proposal on the positive side of a wall is rejected. The step
/// works in place on contiguous buffers.
/// A wall `normal · q ≤ offset`, and its unit normal within the plane.
struct Wall {
    normal: Array1<f64>,
    offset: f64,
    tangent: Array1<f64>,
    scale: f64,
}

pub(crate) struct Dynamics<'a, S> {
    surface: &'a S,
    held: Vec<Array1<f64>>,
    anchor: Vec<f64>,
    walls: Vec<Wall>,
    region: Option<&'a dyn Region>,
    kt: f64,
    h: f64,
    decay: f64,
    kick: f64,
    integrator: PlaneIntegrator,
    pub(crate) q: Array1<f64>,
    pub(crate) gradient: Array1<f64>,
    energy: f64,
    v: Array1<f64>,
    pub(crate) force: Array1<f64>,
    trial: Array1<f64>,
    trial_v: Array1<f64>,
    trial_force: Array1<f64>,
    rng: Rng,
    pub(crate) evaluations: usize,
    pub(crate) proposed: usize,
    pub(crate) accepted: usize,
    pub(crate) wall_hits: usize,
}

impl<'a, S: PointSurface> Dynamics<'a, S> {
    pub(crate) fn new(
        surface: &'a S,
        start: Array1<f64>,
        held: Vec<Array1<f64>>,
        walls: &'a [Hyperplane],
        region: Option<&'a dyn Region>,
        config: &LangevinConfig,
        rng: Rng,
    ) -> Result<Self, SaddleError> {
        if walls.iter().any(|wall| wall.distance(start.view()) > 0.0) {
            return Err(SaddleError::Invalid(
                "the starting configuration is past a wall".into(),
            ));
        }
        if let Some(region) = region
            && !region.contains(start.view())?
        {
            return Err(SaddleError::Invalid(
                "the starting configuration is outside the hard region".into(),
            ));
        }
        let start = contiguous(start);
        let held: Vec<Array1<f64>> = held.into_iter().map(contiguous).collect();
        let anchor = held.iter().map(|t| dot(slice(t), slice(&start))).collect();
        let n = start.len();
        let decay = (-config.friction * config.time_step).exp();
        let mut dynamics = Self {
            surface,
            held,
            anchor,
            walls: Vec::new(),
            region,
            kt: config.thermal,
            h: config.time_step,
            decay,
            kick: (config.thermal * (1.0 - decay * decay)).sqrt(),
            integrator: config.integrator,
            q: start,
            gradient: Array1::zeros(n),
            energy: 0.0,
            v: Array1::zeros(n),
            force: Array1::zeros(n),
            trial: Array1::zeros(n),
            trial_v: Array1::zeros(n),
            trial_force: Array1::zeros(n),
            rng,
            evaluations: 0,
            proposed: 0,
            accepted: 0,
            wall_hits: 0,
        };
        dynamics.walls = walls
            .iter()
            .map(|wall| {
                let normal = contiguous(wall.normal().to_owned());
                let offset = normal.dot(&wall.point());
                let mut inward = normal.clone();
                dynamics.tangent(slice_mut(&mut inward));
                let scale = inward.dot(&inward).sqrt();
                if scale > 0.0 {
                    inward /= scale;
                }
                Wall {
                    normal,
                    offset,
                    tangent: inward,
                    scale,
                }
            })
            .collect();
        let q0 = dynamics.q.clone();
        let (energy, gradient) = dynamics.evaluate(q0.view())?;
        dynamics.energy = energy;
        dynamics.set_force(gradient);
        let sigma = config.thermal.sqrt();
        for value in slice_mut(&mut dynamics.v) {
            *value = sigma * dynamics.rng.normal();
        }
        let mut v = std::mem::take(&mut dynamics.v);
        dynamics.tangent(slice_mut(&mut v));
        dynamics.v = v;
        Ok(dynamics)
    }

    fn evaluate(&mut self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = self.surface.eval(q)?;
        self.evaluations += 1;
        if gradient.len() != q.len() {
            return Err(SaddleError::Shape(
                "sampler gradient length differs from the configuration".into(),
            ));
        }
        if !energy.is_finite() || !gradient.iter().all(|g| g.is_finite()) {
            return Err(SaddleError::NonFinite("sampler energy or gradient"));
        }
        Ok((energy, contiguous(gradient)))
    }

    /// Store `gradient` and the projected force `-gradient`.
    fn set_force(&mut self, gradient: Array1<f64>) {
        self.gradient = gradient;
        let mut force = std::mem::take(&mut self.force);
        let f = slice_mut(&mut force);
        for (fi, gi) in f.iter_mut().zip(slice(&self.gradient)) {
            *fi = -gi;
        }
        self.tangent(f);
        self.force = force;
    }

    fn tangent(&self, v: &mut [f64]) {
        for direction in &self.held {
            let t = slice(direction);
            let c = dot(t, v);
            axpy(v, -c, t);
        }
    }

    fn constrain(&self, q: &mut [f64]) {
        for (direction, anchor) in self.held.iter().zip(&self.anchor) {
            let t = slice(direction);
            let c = dot(t, q) - anchor;
            axpy(q, -c, t);
        }
    }

    /// Advance one step. Returns whether the proposal was accepted.
    pub(crate) fn step(&mut self) -> Result<bool, SaddleError> {
        match self.integrator {
            PlaneIntegrator::Baoab => self.baoab(),
            PlaneIntegrator::Ghmc => self.ghmc(),
        }
    }

    /// Move `q` by `a v` in the plane, reflecting `q` and `v` at the walls.
    fn drift(&mut self, q: &mut [f64], v: &mut [f64], a: f64) -> Result<(), SaddleError> {
        let previous = q.to_vec();
        axpy(q, a, v);
        self.constrain(q);
        let mut hits = 0;
        for _ in 0..REFLECTIONS {
            let mut reflected = false;
            for wall in self.walls.iter().filter(|wall| wall.scale > 0.0) {
                let excess = (dot(slice(&wall.normal), q) - wall.offset) / wall.scale;
                if excess > 0.0 {
                    let u = slice(&wall.tangent);
                    axpy(q, -2.0 * excess, u);
                    let c = dot(u, v);
                    axpy(v, -2.0 * c, u);
                    reflected = true;
                    hits += 1;
                }
            }
            if !reflected {
                break;
            }
        }
        self.wall_hits += hits;
        if let Some(region) = self.region {
            for _ in 0..REFLECTIONS {
                let Some(violation) = region.violation(ArrayView1::from(&*q))? else {
                    return Ok(());
                };
                let mut tangent = contiguous(violation.normal);
                self.tangent(slice_mut(&mut tangent));
                let scale = tangent.dot(&tangent).sqrt();
                if !(scale.is_finite() && scale > 0.0) {
                    return Err(SaddleError::Invalid(
                        "region boundary has no in-plane normal".into(),
                    ));
                }
                tangent /= scale;
                let normal = slice(&tangent);
                match violation.penetration {
                    Some(penetration) => axpy(q, -2.0 * penetration / scale, normal),
                    None => q.copy_from_slice(&previous),
                }
                let component = dot(normal, v);
                axpy(v, -2.0 * component, normal);
                self.constrain(q);
                self.wall_hits += 1;
            }
            if !region.contains(ArrayView1::from(&*q))? {
                return Err(SaddleError::Solver(
                    "hard-region reflection did not return to the region".into(),
                ));
            }
        }
        Ok(())
    }

    fn baoab(&mut self) -> Result<bool, SaddleError> {
        let (decay, kick, h) = (self.decay, self.kick, self.h);
        let mut v = std::mem::take(&mut self.v);
        let mut q = std::mem::take(&mut self.q);
        axpy(slice_mut(&mut v), 0.5 * h, slice(&self.force));
        self.drift(slice_mut(&mut q), slice_mut(&mut v), 0.5 * h)?;
        for value in slice_mut(&mut v) {
            *value = decay * *value + kick * self.rng.normal();
        }
        self.tangent(slice_mut(&mut v));
        self.drift(slice_mut(&mut q), slice_mut(&mut v), 0.5 * h)?;
        self.proposed += 1;
        let evaluated = self.evaluate(q.view());
        self.q = q;
        match evaluated {
            Ok((energy, gradient)) => {
                self.energy = energy;
                self.set_force(gradient);
                axpy(slice_mut(&mut v), 0.5 * h, slice(&self.force));
                self.v = v;
                self.accepted += 1;
                Ok(true)
            }
            Err(error) => {
                self.v = v;
                Err(error)
            }
        }
    }

    fn ghmc(&mut self) -> Result<bool, SaddleError> {
        let (decay, kick, h) = (self.decay, self.kick, self.h);
        let mut v = std::mem::take(&mut self.v);
        for value in slice_mut(&mut v) {
            *value = decay * *value + kick * self.rng.normal();
        }
        self.tangent(slice_mut(&mut v));
        self.proposed += 1;
        let kinetic0 = 0.5 * dot(slice(&v), slice(&v));
        let mut trial_v = std::mem::take(&mut self.trial_v);
        let mut trial = std::mem::take(&mut self.trial);
        {
            let tv = slice_mut(&mut trial_v);
            tv.copy_from_slice(slice(&v));
            axpy(tv, 0.5 * h, slice(&self.force));
            let tq = slice_mut(&mut trial);
            tq.copy_from_slice(slice(&self.q));
            axpy(tq, h, slice(&trial_v));
            self.constrain(tq);
        }
        let mut blocked = self
            .walls
            .iter()
            .any(|wall| dot(slice(&wall.normal), slice(&trial)) > wall.offset);
        if let Some(region) = self.region {
            blocked |= !region.contains(trial.view())?;
        }
        if blocked {
            self.wall_hits += 1;
            v.mapv_inplace(|x| -x);
            self.v = v;
            self.trial = trial;
            self.trial_v = trial_v;
            return Ok(false);
        }
        let (energy1, gradient1) = match self.evaluate(trial.view()) {
            Ok(result) => result,
            Err(error) => {
                self.v = v;
                self.trial = trial;
                self.trial_v = trial_v;
                return Err(error);
            }
        };
        let mut force1 = std::mem::take(&mut self.trial_force);
        {
            let f1 = slice_mut(&mut force1);
            for (fi, gi) in f1.iter_mut().zip(slice(&gradient1)) {
                *fi = -gi;
            }
            self.tangent(f1);
            axpy(slice_mut(&mut trial_v), 0.5 * h, f1);
        }
        let kinetic1 = 0.5 * dot(slice(&trial_v), slice(&trial_v));
        let delta = energy1 + kinetic1 - self.energy - kinetic0;
        let accept = delta <= 0.0 || self.rng.uniform() < (-delta / self.kt).exp();
        if accept {
            std::mem::swap(&mut self.q, &mut trial);
            std::mem::swap(&mut v, &mut trial_v);
            std::mem::swap(&mut self.force, &mut force1);
            self.gradient = gradient1;
            self.energy = energy1;
            self.accepted += 1;
        } else {
            v.mapv_inplace(|x| -x);
        }
        self.trial_force = force1;
        self.v = v;
        self.trial = trial;
        self.trial_v = trial_v;
        Ok(accept)
    }

    pub(crate) fn acceptance(&self) -> f64 {
        if self.proposed == 0 {
            1.0
        } else {
            self.accepted as f64 / self.proposed as f64
        }
    }
}

/// Analytic-`PointSurface` reference backend for [`HyperplaneSampler`].
///
/// This projected BAOAB/GHMC implementation exists for analytic tests. A
/// production molecular-dynamics engine implements the same trait.
pub struct AnalyticSurfaceSampler<'a, S> {
    surface: &'a S,
    config: LangevinConfig,
    start: Option<Array1<f64>>,
    held: Option<Vec<Array1<f64>>>,
    rng: SamplerRng,
}

enum SamplerRng {
    BudgetSeed,
    Prepared(Rng),
}

impl<'a, S: PointSurface> AnalyticSurfaceSampler<'a, S> {
    /// Build the reference backend with no explicit starting position.
    pub fn new(surface: &'a S, config: LangevinConfig) -> Self {
        Self {
            surface,
            config,
            start: None,
            held: None,
            rng: SamplerRng::BudgetSeed,
        }
    }

    /// Start from `position`, projected onto the requested plane.
    pub fn with_start(mut self, position: Array1<f64>) -> Self {
        self.start = Some(position);
        self
    }
}

impl<S: PointSurface> HyperplaneSampler for AnalyticSurfaceSampler<'_, S> {
    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        let start = self
            .start
            .take()
            .map_or_else(|| plane.point().to_owned(), |q| plane.project(q.view()));
        let held = match self.held.take() {
            Some(held) => held,
            None => held_directions(
                self.surface,
                plane.point(),
                Some(plane.normal()),
                self.config.hold_excluded,
            )?,
        };
        let rng = match std::mem::replace(&mut self.rng, SamplerRng::BudgetSeed) {
            SamplerRng::BudgetSeed => Rng::new(budget.seed),
            SamplerRng::Prepared(rng) => rng,
        };
        let walls: [Hyperplane; 0] = [];
        let mut config = self.config.clone();
        config.thermal = budget.thermal;
        config.seed = budget.seed;
        config.burn_in = budget.thermalize;
        config.stride = budget.stride;
        config.records = budget.samples;
        let mut dynamics = Dynamics::new(
            self.surface,
            start,
            held,
            &walls,
            budget.region.as_deref(),
            &config,
            rng,
        )?;
        for _ in 0..budget.thermalize {
            dynamics.step()?;
        }
        let mut observations = Vec::with_capacity(budget.samples);
        for _ in 0..budget.samples {
            for _ in 0..budget.stride {
                dynamics.step()?;
            }
            observations.push(HyperplaneObservation {
                constraint_force: -plane.normal().dot(&dynamics.gradient),
                pafi_gradient: None,
                jacobian: 0.0,
                position: budget.positions.then(|| dynamics.q.clone()),
                projected_force: Some(dynamics.force.clone()),
                projected_velocity: None,
                observables: Default::default(),
            });
        }
        let evaluations = dynamics.evaluations;
        let acceptance = dynamics.acceptance();
        self.start = Some(dynamics.q.clone());
        Ok(HyperplaneEnsemble {
            observations,
            evaluations,
            acceptance,
        })
    }

    fn energy(&mut self, position: ArrayView1<f64>) -> Result<Option<f64>, SaddleError> {
        Ok(Some(self.surface.eval(position)?.0))
    }

    fn constrained_directions(
        &mut self,
        plane: &Hyperplane,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        held_directions(self.surface, plane.point(), None, self.config.hold_excluded)
    }
}

/// Averages over one hyperplane.
///
/// `grad_point` and `grad_normal` are the Euclidean gradient blocks of the
/// plane free energy `A(p, ν)`: `∂A/∂p = ⟨ν · ∇V⟩ ν` and
/// `∂A/∂ν = -⟨(ν · ∇V) (q - p)⟩`, the second taken orthogonal to `ν` and
/// to any held mode. A manifold turns them into a Riemannian gradient.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneSample {
    /// The sampled plane.
    pub plane: Hyperplane,
    /// Mean force along the normal, `⟨-ν · ∇V⟩`.
    pub normal_force: f64,
    /// Block standard error of [`Self::normal_force`].
    pub normal_force_error: f64,
    /// `∂A/∂p`.
    pub grad_point: Array1<f64>,
    /// `∂A/∂ν`.
    pub grad_normal: Array1<f64>,
    /// Block standard error of each entry of [`Self::grad_normal`].
    pub grad_normal_error: Array1<f64>,
    /// Mean configuration `⟨q⟩`.
    pub centroid: Array1<f64>,
    /// Block means of the packed gradient `[∂A/∂p; ∂A/∂ν]`, one row per
    /// block, for the standard error of any linear functional.
    pub block_gradients: Array2<f64>,
    /// `∂A/∂ν` with the plane turning about [`Self::centroid`] instead of
    /// its point: `-Cov(ν · ∇V, q)` off the normal and the held modes, which
    /// is `grad_normal + ⟨ν · ∇V⟩ (c - p)`.
    pub centroid_torque: Array1<f64>,
    /// Block standard error of each entry of [`Self::centroid_torque`].
    pub centroid_torque_error: Array1<f64>,
    /// [`Self::block_gradients`] with the torque about the centroid. Both
    /// factors of the covariance are centred in every block, so the rows
    /// leave out the noise of the lever arm `c - p`.
    pub centroid_block_gradients: Array2<f64>,
    /// Virial `⟨(P∇V) · (q - p)⟩`, `P` the projector onto the free
    /// directions of the plane. At equilibrium on the plane it is
    /// `free_dimensions · kT`, so a departure flags a run that has not
    /// equilibrated or a time step that biases the sampling.
    pub virial: f64,
    /// Block standard error of [`Self::virial`].
    pub virial_error: f64,
    /// Directions free in the plane: the dimension less the normal and the
    /// held modes.
    pub free_dimensions: usize,
    /// `kT` of the sampler.
    pub thermal: f64,
    /// Independent records behind [`Self::normal_force`]: the variance of
    /// `ν · ∇V` over the records divided by the squared block error.
    pub effective_samples: f64,
    /// Largest integrated autocorrelation time of the monitored series, in
    /// records.
    pub tau: f64,
    /// Records inside the blocks.
    pub records: usize,
    /// Surface evaluations.
    pub evaluations: usize,
    /// Accepted fraction of the proposals. One without the Metropolis test
    /// unless a wall rejected a step.
    pub acceptance: f64,
    /// Configuration after the last step.
    pub last: Array1<f64>,
    /// Configurations kept from the records, evenly spaced.
    pub retained: Vec<Array1<f64>>,
    /// The estimates before the control variates of
    /// [`LangevinConfig::control`], when they were applied. The fields
    /// above are then the controlled ones; [`Self::effective_samples`]
    /// stays uncontrolled.
    pub uncontrolled: Option<Uncontrolled>,
    /// Automated equilibration, correlation and split-half drift report.
    pub equilibration: EquilibrationReport,
}

/// Plane estimates without control variates.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Uncontrolled {
    /// Mean force along the normal.
    pub normal_force: f64,
    /// Block standard error of [`Self::normal_force`].
    pub normal_force_error: f64,
    /// `∂A/∂ν`.
    pub grad_normal: Array1<f64>,
    /// Block standard errors of [`Self::grad_normal`].
    pub grad_normal_error: Array1<f64>,
    /// The torque about the sample centroid.
    pub centroid_torque: Array1<f64>,
    /// Block standard errors of [`Self::centroid_torque`].
    pub centroid_torque_error: Array1<f64>,
}

impl PlaneSample {
    /// `[∂A/∂p; ∂A/∂ν]`, the layout of [`Hyperplane::packed`].
    pub fn packed_gradient(&self) -> Array1<f64> {
        let n = self.grad_point.len();
        let mut g = Array1::zeros(2 * n);
        g.slice_mut(s![..n]).assign(&self.grad_point);
        g.slice_mut(s![n..]).assign(&self.grad_normal);
        g
    }

    /// [`Self::packed_gradient`] with the plane anchored at
    /// [`Self::centroid`]: `[∂A/∂p; centroid torque]`.
    pub fn centroid_packed_gradient(&self) -> Array1<f64> {
        let n = self.grad_point.len();
        let mut g = Array1::zeros(2 * n);
        g.slice_mut(s![..n]).assign(&self.grad_point);
        g.slice_mut(s![n..]).assign(&self.centroid_torque);
        g
    }

    /// The derivative of `A` along a packed direction `[δp; δν]`, and its
    /// block standard error.
    pub fn directional(&self, direction: ArrayView1<f64>) -> Result<(f64, f64), SaddleError> {
        if direction.len() != self.block_gradients.ncols() {
            return Err(SaddleError::Shape(
                "a packed direction has twice the configuration length".into(),
            ));
        }
        let values: Vec<f64> = self
            .block_gradients
            .rows()
            .into_iter()
            .map(|row| row.dot(&direction))
            .collect();
        Ok(mean_and_error(&values))
    }
}

pub(crate) fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Orthonormal directions the motion keeps fixed: the normal, then the
/// surface's excluded modes when they are held.
pub(crate) fn held_directions<S: PointSurface>(
    surface: &S,
    at: ArrayView1<f64>,
    normal: Option<ArrayView1<f64>>,
    hold_excluded: bool,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let n = at.len();
    let excluded = if hold_excluded {
        let modes = surface.excluded_modes(at)?;
        if modes.nrows() > 0 && modes.ncols() != n {
            return Err(SaddleError::Shape(
                "excluded modes differ in length from the configuration".into(),
            ));
        }
        orthonormal_rows(modes.view(), n)
    } else {
        Vec::new()
    };
    let mut held = Vec::with_capacity(excluded.len() + 1);
    if let Some(normal) = normal {
        let overlap: f64 = excluded
            .iter()
            .map(|t| t.dot(&normal).powi(2))
            .sum::<f64>()
            .sqrt();
        let roundoff = f64::EPSILON * n as f64 * (excluded.len() + 1) as f64;
        if overlap > roundoff {
            return Err(SaddleError::Invalid(
                "the plane normal has a component along a held mode; take Hyperplane::orthogonal_to first"
                    .into(),
            ));
        }
        held.push(normal.to_owned());
    }
    held.extend(excluded);
    Ok(held)
}

/// The estimates that the control variates of [`LangevinConfig::control`]
/// correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Controlled {
    /// The normal force and the torques.
    All,
    /// The normal force only, which spares the linear solve per block that
    /// the torques need. The torques stay uncontrolled.
    NormalForce,
}

/// Sample the plane at `config.thermal` and average the gradient blocks
/// of its free energy.
///
/// `start` is projected onto the plane; `None` starts at the plane's
/// point. Held modes keep the values of the start.
pub fn sample_plane<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    start: Option<ArrayView1<f64>>,
    config: &LangevinConfig,
) -> Result<PlaneSample, SaddleError> {
    sample_plane_controlled(surface, plane, start, config, Controlled::All)
}

/// Estimate OH-TST plane statistics from an external constrained engine.
///
/// The backend supplies records through [`HyperplaneSampler`]; this function
/// performs only the free-energy-gradient and standard-error calculations.
pub fn sample_plane_with_sampler<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &LangevinConfig,
) -> Result<PlaneSample, SaddleError> {
    sample_plane_with_sampler_policy(sampler, plane, config, None)
}

pub(crate) fn sample_plane_pilot_with_sampler<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &LangevinConfig,
    false_rejection: f64,
) -> Result<PlaneSample, SaddleError> {
    if !(false_rejection.is_finite() && false_rejection > 0.0 && false_rejection < 1.0) {
        return Err(SaddleError::Invalid(
            "pilot false-rejection probability must lie inside (0, 1)".into(),
        ));
    }
    sample_plane_with_sampler_policy(sampler, plane, config, Some(false_rejection))
}

fn sample_plane_with_sampler_policy<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
    config: &LangevinConfig,
    pilot: Option<f64>,
) -> Result<PlaneSample, SaddleError> {
    config.check()?;
    if config.control.is_some() {
        return Err(SaddleError::Invalid(
            "external hyperplane samples do not use analytic control variates".into(),
        ));
    }
    let mut budget = HyperplaneSampleBudget::default()
        .with_thermal(config.thermal)
        .with_seed(config.seed)
        .with_thermalize(config.burn_in)
        .with_stride(config.stride)
        .with_samples(config.records)
        .with_positions(true);
    if let Some(region) = &config.region {
        budget = budget.with_region(Arc::clone(region));
    }
    let constrained = sampler.constrained_directions(plane)?;
    let ensemble = sampler.sample_hyperplane(plane, &budget)?;
    estimate_engine_plane(plane, config, &constrained, ensemble, pilot)
}

/// Directions held in the plane, with the normal component removed.
fn independent_constraints(
    plane: &Hyperplane,
    constrained: &[Array1<f64>],
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let n = plane.dim();
    let normal = plane.normal();
    let mut rows = Array2::zeros((constrained.len(), n));
    for (index, direction) in constrained.iter().enumerate() {
        if direction.len() != n {
            return Err(SaddleError::Shape(
                "constrained directions differ in length from the plane".into(),
            ));
        }
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("constrained direction"));
        }
        let mut row = direction.to_owned();
        let along = normal.dot(&row);
        row.scaled_add(-along, &normal);
        rows.row_mut(index).assign(&row);
    }
    Ok(orthonormal_rows(rows.view(), n))
}

fn remove_directions(vector: &mut Array1<f64>, directions: &[Array1<f64>]) {
    for direction in directions {
        let along = direction.dot(vector);
        vector.scaled_add(-along, direction);
    }
}

fn pilot_report(
    values: &[f64],
    config: &LangevinConfig,
    false_rejection: f64,
) -> Result<EquilibrationReport, SaddleError> {
    if values.len() < 8 {
        return Err(SaddleError::Invalid(
            "a pilot stationarity test needs four effective observations in each half".into(),
        ));
    }
    let tau = autocorr_time(values);
    let n_eff = values.len() as f64 / (2.0 * tau).max(1.0);
    // Use equal even-length halves so a stationary period-two integrator
    // signal has the same phase balance on both sides. For the prescribed
    // 14-record pilot this omits only the central pair from the diagnostic.
    let half = (values.len() / 4) * 2;
    let first = &values[..half];
    let second = &values[values.len() - half..];
    let (mean_a, iid_error_a) = mean_and_error(first);
    let (mean_b, iid_error_b) = mean_and_error(second);
    let error_a = iid_error_a * (2.0 * autocorr_time(first)).max(1.0).sqrt();
    let error_b = iid_error_b * (2.0 * autocorr_time(second)).max(1.0).sqrt();
    let drift = (mean_a - mean_b).abs();
    let error = error_a.hypot(error_b);
    let sigma_multiple = (1.0 / false_rejection).sqrt();
    // Each record enters one of the two half-means (one addition). The
    // comparison then takes two absolute values, one maximum, and the
    // product with γ. That is n + 4 floating-point operations.
    let operations = values.len() + 4;
    let gamma = operations as f64 * f64::EPSILON / (1.0 - operations as f64 * f64::EPSILON);
    let roundoff = gamma * mean_a.abs().max(mean_b.abs()).max(1.0);
    let report = EquilibrationReport {
        discard: 0,
        equilibration_steps: config.burn_in,
        tau,
        n_eff,
        split_half_drift: drift,
        split_half_error: error,
        sigma_multiple,
        drift_passed: drift <= (sigma_multiple * error).max(roundoff),
    };
    // Two effective observations in each half are the minimum needed to
    // estimate both half means and their sampling variances.
    let required = 4.0;
    if n_eff < required {
        return Err(SaddleError::InsufficientEffectiveSamples {
            equilibration_steps: report.equilibration_steps,
            n_eff,
            required,
        });
    }
    if !report.drift_passed {
        return Err(SaddleError::NotEquilibrated {
            equilibration_steps: report.equilibration_steps,
            drift,
            standard_error: error,
            sigma_multiple,
        });
    }
    Ok(report)
}

fn estimate_engine_plane(
    plane: &Hyperplane,
    config: &LangevinConfig,
    constrained: &[Array1<f64>],
    ensemble: HyperplaneEnsemble,
    pilot: Option<f64>,
) -> Result<PlaneSample, SaddleError> {
    if ensemble.observations.len() != config.records {
        return Err(SaddleError::Invalid(
            "OH-TST engine did not return the requested records".into(),
        ));
    }
    let force_series: Vec<f64> = ensemble
        .observations
        .iter()
        .map(|observation| observation.constraint_force)
        .collect();
    let equilibration = if let Some(false_rejection) = pilot {
        pilot_report(&force_series, config, false_rejection)?
    } else {
        equilibrate(&force_series, config.burn_in, config.stride, config.blocks)?
    };
    let observations = &ensemble.observations[equilibration.discard..];
    let n = plane.dim();
    let normal = plane.normal().to_owned();
    let point = plane.point().to_owned();
    let independent = independent_constraints(plane, constrained)?;
    // Width is floor(tail / requested blocks), and at least ceil(2 tau).
    // The remainder is dropped so every block has that width. Ceiling the
    // width instead leaves a single block when the tail is odd and two
    // blocks were requested: 15 records would ask for width 8.
    let correlation_width = ((2.0 * equilibration.tau).ceil() as usize).max(1);
    let Some((blocks, width)) = batch_blocks(observations.len(), config.blocks, correlation_width)
    else {
        return Err(SaddleError::InsufficientEffectiveSamples {
            equilibration_steps: equilibration.equilibration_steps,
            n_eff: equilibration.n_eff,
            required: 2.0 * correlation_width as f64,
        });
    };
    let records = blocks * width;
    let observations = &observations[..records];
    let mut block_gradients = Array2::zeros((blocks, 2 * n));
    let mut block_centroids = Array2::zeros((blocks, n));
    let mut normal_forces = Vec::with_capacity(blocks);
    let mut virials = Vec::with_capacity(blocks);
    let mut series = Vec::with_capacity(records);
    for block in 0..blocks {
        let mut force_sum = 0.0;
        let mut torque = Array1::zeros(n);
        let mut centroid = Array1::zeros(n);
        let mut virial = 0.0;
        for observation in &observations[block * width..(block + 1) * width] {
            let q = observation
                .position
                .as_ref()
                .ok_or_else(|| SaddleError::Invalid("OH-TST sampler omitted positions".into()))?;
            let force = observation.projected_force.as_ref().ok_or_else(|| {
                SaddleError::Invalid("OH-TST sampler omitted projected forces".into())
            })?;
            if q.len() != n || force.len() != n {
                return Err(SaddleError::Shape(
                    "OH-TST engine vectors differ from the plane".into(),
                ));
            }
            let d = -observation.constraint_force;
            let arm = q - &point;
            force_sum += observation.constraint_force;
            torque.scaled_add(-d, &arm);
            centroid += q;
            if independent.is_empty() {
                virial -= force.dot(&arm);
            } else {
                let mut tangent = force.clone();
                remove_directions(&mut tangent, &independent);
                virial -= tangent.dot(&arm);
            }
            series.push(d);
        }
        let scale = 1.0 / width as f64;
        force_sum *= scale;
        torque *= scale;
        centroid *= scale;
        virial *= scale;
        let along = normal.dot(&torque);
        torque.scaled_add(-along, &normal);
        remove_directions(&mut torque, &independent);
        let d = -force_sum;
        let mut row = block_gradients.row_mut(block);
        row.slice_mut(s![..n]).assign(&(&normal * d));
        row.slice_mut(s![n..]).assign(&torque);
        block_centroids.row_mut(block).assign(&centroid);
        normal_forces.push(force_sum);
        virials.push(virial);
    }
    let column_means = |rows: &Array2<f64>, offset: usize| {
        let mut means = Array1::zeros(n);
        let mut errors = Array1::zeros(n);
        for i in 0..n {
            let column = rows.column(offset + i).to_vec();
            (means[i], errors[i]) = mean_and_error(&column);
        }
        (means, errors)
    };
    let (normal_force, normal_force_error) = mean_and_error(&normal_forces);
    let (grad_point, _) = column_means(&block_gradients, 0);
    let (grad_normal, grad_normal_error) = column_means(&block_gradients, n);
    let centroid =
        block_centroids
            .mean_axis(ndarray::Axis(0))
            .ok_or(SaddleError::MissingSamples {
                context: "sampler block centroids",
            })?;
    let mut lever = &centroid - &point;
    remove_directions(&mut lever, &independent);
    let mut centroid_block_gradients = block_gradients.clone();
    for (block, mut row) in centroid_block_gradients.rows_mut().into_iter().enumerate() {
        let mut shift = (&block_centroids.row(block) - &centroid).to_owned();
        remove_directions(&mut shift, &independent);
        let mut torque = row.slice_mut(s![n..]);
        torque.scaled_add(-normal_forces[block], &lever);
        torque.scaled_add(-normal_force, &shift);
    }
    let (centroid_torque, centroid_torque_error) = column_means(&centroid_block_gradients, n);
    let (virial, virial_error) = mean_and_error(&virials);
    let tau = equilibration.tau;
    let series_mean = series.iter().sum::<f64>() / records as f64;
    let variance = series
        .iter()
        .map(|value| (value - series_mean).powi(2))
        .sum::<f64>()
        / (records - 1).max(1) as f64;
    let effective_samples = if normal_force_error > 0.0 {
        variance / normal_force_error.powi(2)
    } else {
        records as f64
    };
    let retain_every = config
        .records
        .checked_div(config.retain)
        .map_or(usize::MAX, |every| every.max(1));
    let retained: Vec<Array1<f64>> = observations
        .iter()
        .enumerate()
        .filter(|(index, _)| index % retain_every == retain_every - 1)
        .filter_map(|(_, observation)| observation.position.clone())
        .take(config.retain)
        .collect();
    let last = ensemble
        .observations
        .last()
        .and_then(|observation| observation.position.clone())
        .ok_or_else(|| SaddleError::Invalid("OH-TST sampler returned no position".into()))?;
    Ok(PlaneSample {
        plane: plane.clone(),
        normal_force,
        normal_force_error,
        grad_point,
        grad_normal,
        grad_normal_error,
        centroid,
        block_gradients,
        centroid_torque,
        centroid_torque_error,
        centroid_block_gradients,
        virial,
        virial_error,
        free_dimensions: n.saturating_sub(1 + independent.len()),
        thermal: config.thermal,
        effective_samples,
        tau,
        records,
        evaluations: ensemble.evaluations,
        acceptance: ensemble.acceptance,
        last,
        retained,
        uncontrolled: None,
        equilibration,
    })
}

/// [`sample_plane`] with the control variates restricted to `controlled`.
pub(crate) fn sample_plane_controlled<S: PointSurface>(
    surface: &S,
    plane: &Hyperplane,
    start: Option<ArrayView1<f64>>,
    config: &LangevinConfig,
    controlled: Controlled,
) -> Result<PlaneSample, SaddleError> {
    config.check()?;
    let n = plane.dim();
    let start = match start {
        Some(start) if start.len() != n => {
            return Err(SaddleError::Shape(
                "sampler start differs in length from the plane".into(),
            ));
        }
        Some(start) => plane.project(start),
        None => plane.point().to_owned(),
    };
    let normal = plane.normal().to_owned();
    let point = plane.point().to_owned();
    let held = held_directions(
        surface,
        point.view(),
        Some(normal.view()),
        config.hold_excluded,
    )?;
    let mut rng = Rng::new(config.seed);
    let mut probes = Vec::with_capacity(PROJECTIONS);
    let free_dimensions = n - held.len();
    for _ in 0..PROJECTIONS * usize::from(free_dimensions > 0) {
        let mut w: Array1<f64> = (0..n).map(|_| rng.normal()).collect();
        let raw_norm = w.dot(&w).sqrt();
        for _ in 0..2 {
            for direction in &held {
                let c = direction.dot(&w);
                w.scaled_add(-c, direction);
            }
        }
        let norm = w.dot(&w).sqrt();
        if !(raw_norm.is_finite() && norm.is_finite()) {
            return Err(SaddleError::NonFinite("sampler control probe"));
        }
        if norm > f64::EPSILON * n as f64 * raw_norm {
            probes.push(w / norm);
        }
    }
    let excluded = held[1..].to_vec();
    let mut control = match &config.control {
        Some(reference) => PlaneControl::new(surface, reference, &held)?,
        None => None,
    };
    let columns = 2 * n + 2 + if control.is_some() { 2 * n } else { 0 };
    let mut budget = HyperplaneSampleBudget::default()
        .with_thermal(config.thermal)
        .with_seed(config.seed)
        .with_thermalize(config.burn_in)
        .with_stride(config.stride)
        .with_samples(config.records)
        .with_positions(true);
    if let Some(region) = &config.region {
        budget = budget.with_region(Arc::clone(region));
    }
    let mut backend = AnalyticSurfaceSampler {
        surface,
        config: config.clone(),
        start: Some(start),
        held: Some(held),
        rng: SamplerRng::Prepared(rng),
    };
    let ensemble = backend.sample_hyperplane(plane, &budget)?;
    let force_series: Vec<f64> = ensemble
        .observations
        .iter()
        .map(|observation| observation.constraint_force)
        .collect();
    let equilibration = equilibrate(&force_series, config.burn_in, config.stride, config.blocks)?;
    let observations = &ensemble.observations[equilibration.discard..];
    let record_count = observations.len();
    let chunk = record_count.div_ceil(CHUNKS).max(1);
    let mut chunk_sums: Vec<Array1<f64>> = Vec::with_capacity(record_count.div_ceil(chunk));
    let mut sum = Array1::zeros(columns);
    let mut series = vec![Vec::with_capacity(record_count); 1 + probes.len()];
    let retain_every = record_count
        .checked_div(config.retain)
        .map_or(usize::MAX, |every| every.max(1));
    let mut retained = Vec::with_capacity(config.retain);
    let mut arm = vec![0.0; n];
    for (record, observation) in observations.iter().enumerate() {
        let q = slice(
            observation
                .position
                .as_ref()
                .ok_or_else(|| SaddleError::Invalid("OH-TST sampler omitted positions".into()))?,
        );
        let d = -observation.constraint_force;
        let projected_force = observation.projected_force.as_ref().ok_or_else(|| {
            SaddleError::Invalid("OH-TST sampler omitted projected forces".into())
        })?;
        let force = slice(projected_force);
        for ((a, qi), pi) in arm.iter_mut().zip(q).zip(slice(&point)) {
            *a = qi - pi;
        }
        let total = slice_mut(&mut sum);
        total[0] += d;
        axpy(&mut total[1..=n], d, &arm);
        axpy(&mut total[n + 1..=2 * n], 1.0, q);
        total[2 * n + 1] -= dot(force, &arm);
        if let Some(control) = &control {
            let lever = dot(slice(&control.coupling), &arm);
            let (gradient, weighted) = total[2 * n + 2..].split_at_mut(n);
            axpy(gradient, -1.0, force);
            axpy(weighted, -lever, force);
        }
        series[0].push(d);
        for (j, probe) in probes.iter().enumerate() {
            series[1 + j].push(d * dot(slice(probe), &arm));
        }
        if (record + 1) % chunk == 0 || record + 1 == record_count {
            chunk_sums.push(std::mem::replace(&mut sum, Array1::zeros(columns)));
        }
        if retained.len() < config.retain && record % retain_every == retain_every - 1 {
            retained.push(Array1::from(q.to_vec()));
        }
    }
    let tau = series
        .iter()
        .map(|values| autocorr_time(values))
        .fold(0.5, f64::max);
    let width = (record_count / config.blocks)
        .max((2.0 * tau).ceil() as usize)
        .div_ceil(chunk)
        .max(1);
    let n_blocks = (record_count / chunk) / width;
    if n_blocks < 2 {
        return Err(SaddleError::Invalid(format!(
            "{record_count} records hold fewer than two blocks of twice the correlation time {tau:.1}"
        )));
    }
    let per_block = (width * chunk) as f64;
    let project = |v: &mut Array1<f64>| {
        let c = normal.dot(v);
        v.scaled_add(-c, &normal);
        for t in &excluded {
            let c = t.dot(v);
            v.scaled_add(-c, t);
        }
    };
    let mut block_gradients = Array2::zeros((n_blocks, 2 * n));
    let mut block_centroids = Array2::zeros((n_blocks, n));
    let mut centroid = Array1::zeros(n);
    let mut normal_force = Vec::with_capacity(n_blocks);
    let mut virials = Vec::with_capacity(n_blocks);
    let mut block_controls = Vec::with_capacity(n_blocks);
    for block in 0..n_blocks {
        let mut total = Array1::zeros(columns);
        for part in &chunk_sums[block * width..(block + 1) * width] {
            total += part;
        }
        total /= per_block;
        let d = total[0];
        normal_force.push(-d);
        virials.push(total[2 * n + 1]);
        let mut torque = -total.slice(s![1..=n]).to_owned();
        project(&mut torque);
        let mut row = block_gradients.row_mut(block);
        row.slice_mut(s![..n]).assign(&(&normal * d));
        row.slice_mut(s![n..]).assign(&torque);
        let mean = total.slice(s![n + 1..=2 * n]);
        block_centroids.row_mut(block).assign(&mean);
        centroid.scaled_add(1.0 / n_blocks as f64, &mean);
        if columns > 2 * n + 2 {
            block_controls.push(total.slice(s![2 * n + 2..]).to_owned());
        }
    }
    let mut raw_blocks = None;
    if let Some(control) = control.as_mut() {
        let mut forces = Vec::with_capacity(n_blocks);
        let mut rows = block_gradients.clone();
        for (block, sums) in block_controls.iter().enumerate() {
            let gradient = sums.slice(s![..n]).to_owned();
            let correction = match controlled {
                Controlled::All => {
                    let weighted = sums.slice(s![n..]).to_owned();
                    let Some(correction) = control.solve(&weighted)? else {
                        break;
                    };
                    Some(correction)
                }
                Controlled::NormalForce => None,
            };
            let d = -normal_force[block] - control.coefficient.dot(&gradient);
            let mut row = rows.row_mut(block);
            row.slice_mut(s![..n]).assign(&(&normal * d));
            if let Some(correction) = correction {
                let mut torque = row.slice_mut(s![n..]);
                torque += &correction;
                torque.scaled_add(-config.thermal, &control.coefficient);
            }
            forces.push(-d);
        }
        if forces.len() == n_blocks {
            raw_blocks = Some((
                std::mem::replace(&mut normal_force, forces),
                std::mem::replace(&mut block_gradients, rows),
            ));
        }
    }
    let control_evaluations = control.as_ref().map_or(0, |control| control.evaluations);
    let mut lever = &centroid - &point;
    project(&mut lever);
    let centroid_rows = |gradients: &Array2<f64>, forces: &[f64]| {
        let (force, _) = mean_and_error(forces);
        let mut rows = gradients.clone();
        for (block, mut row) in rows.rows_mut().into_iter().enumerate() {
            let mut shift = &block_centroids.row(block) - &centroid;
            project(&mut shift);
            let mut torque = row.slice_mut(s![n..]);
            torque.scaled_add(-forces[block], &lever);
            torque.scaled_add(-force, &shift);
        }
        rows
    };
    let (force, force_error) = mean_and_error(&normal_force);
    let centroid_block_gradients = centroid_rows(&block_gradients, &normal_force);
    let column_means = |rows: &Array2<f64>, offset: usize| {
        let mut means = Array1::zeros(n);
        let mut errors = Array1::zeros(n);
        for i in 0..n {
            let column: Vec<f64> = rows.column(offset + i).to_vec();
            (means[i], errors[i]) = mean_and_error(&column);
        }
        (means, errors)
    };
    let (grad_point, _) = column_means(&block_gradients, 0);
    let (grad_normal, grad_normal_error) = column_means(&block_gradients, n);
    let (centroid_torque, centroid_torque_error) = column_means(&centroid_block_gradients, n);
    let (virial, virial_error) = mean_and_error(&virials);
    let uncontrolled = raw_blocks.map(|(forces, gradients)| {
        let (normal_force, normal_force_error) = mean_and_error(&forces);
        let (grad_normal, grad_normal_error) = column_means(&gradients, n);
        let (centroid_torque, centroid_torque_error) =
            column_means(&centroid_rows(&gradients, &forces), n);
        Uncontrolled {
            normal_force,
            normal_force_error,
            grad_normal,
            grad_normal_error,
            centroid_torque,
            centroid_torque_error,
        }
    });
    let raw_force_error = uncontrolled
        .as_ref()
        .map_or(force_error, |raw| raw.normal_force_error);
    let records = n_blocks * width * chunk;
    let forces = &series[0][..records];
    let mean = forces.iter().sum::<f64>() / records as f64;
    let variance =
        forces.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (records - 1).max(1) as f64;
    let effective_samples = if raw_force_error > 0.0 {
        variance / raw_force_error.powi(2)
    } else {
        records as f64
    };
    let last = ensemble
        .observations
        .last()
        .and_then(|observation| observation.position.clone())
        .ok_or_else(|| SaddleError::Invalid("OH-TST sampler returned no final position".into()))?;
    Ok(PlaneSample {
        plane: plane.clone(),
        normal_force: force,
        normal_force_error: force_error,
        grad_point,
        grad_normal,
        grad_normal_error,
        centroid,
        block_gradients,
        centroid_torque,
        centroid_torque_error,
        centroid_block_gradients,
        virial,
        virial_error,
        free_dimensions,
        thermal: config.thermal,
        effective_samples,
        tau,
        records,
        evaluations: ensemble.evaluations + control_evaluations,
        acceptance: ensemble.acceptance,
        last,
        retained,
        uncontrolled,
        equilibration,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PlanarBoundary, Region};
    use ndarray::array;

    /// `V = ½ qᵀ K q` with a translation-like excluded row.
    struct Bowl {
        k: Array2<f64>,
        excluded: Option<Array1<f64>>,
    }

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = self.k.dot(&q);
            Ok((0.5 * q.dot(&g), g))
        }

        fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
            match &self.excluded {
                Some(row) => Ok(row.clone().into_shape_with_order((1, q.len())).unwrap()),
                None => Ok(Array2::zeros((0, q.len()))),
            }
        }
    }

    fn bowl() -> Bowl {
        Bowl {
            k: array![[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 0.5]],
            excluded: None,
        }
    }

    #[test]
    fn samples_stay_in_the_plane_and_hold_modes() {
        let mut surface = bowl();
        surface.excluded = Some(array![0.0, 0.0, 1.0]);
        let plane = Hyperplane::new(array![0.2, 0.1, 0.4], array![1.0, 1.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.3)
            .with_time_step(0.2)
            .with_records(200)
            .with_blocks(4)
            .with_retain(5);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        assert_eq!(sample.retained.len(), 5);
        for q in sample.retained.iter().chain([&sample.last]) {
            assert!(plane.distance(q.view()).abs() < 1e-12);
            assert!((q[2] - 0.4).abs() < 1e-12);
        }
        assert!(sample.grad_normal[2].abs() < 1e-15);
        assert!(sample.grad_normal.dot(&plane.normal()).abs() < 1e-12);
        assert!(sample.centroid_torque[2].abs() < 1e-15);
        assert!(sample.centroid_torque.dot(&plane.normal()).abs() < 1e-12);
        assert_eq!(sample.free_dimensions, 1);
        assert!(sample.acceptance > 0.9);
        let tilted = Hyperplane::new(array![0.0, 0.0, 0.0], array![0.0, 0.0, 1.0]).unwrap();
        assert!(sample_plane(&surface, &tilted, None, &config).is_err());
    }

    #[test]
    fn engine_trait_feeds_the_ohtst_estimator() {
        let surface = bowl();
        let plane = Hyperplane::new(array![0.1, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.2)
            .with_records(200)
            .with_blocks(4);
        let mut backend = AnalyticSurfaceSampler::new(&surface, config.clone());
        let sample = sample_plane_with_sampler(&mut backend, &plane, &config).unwrap();
        assert_eq!(sample.records, 200);
        assert!(
            sample
                .retained
                .iter()
                .all(|q| plane.distance(q.view()).abs() < 1e-12)
        );
        assert!(sample.normal_force.is_finite());
    }

    struct DriftingBackend;

    impl HyperplaneSampler for DriftingBackend {
        fn sample_hyperplane(
            &mut self,
            _plane: &Hyperplane,
            budget: &HyperplaneSampleBudget,
        ) -> Result<HyperplaneEnsemble, SaddleError> {
            let observations = (0..budget.samples)
                .map(|index| HyperplaneObservation {
                    constraint_force: 0.01 * index as f64,
                    pafi_gradient: None,
                    jacobian: 0.0,
                    position: None,
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
    fn unequilibrated_engine_series_is_a_typed_error() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let config = LangevinConfig::default().with_records(400).with_blocks(8);
        let error = sample_plane_with_sampler(&mut DriftingBackend, &plane, &config)
            .expect_err("a drifting constraint force must not produce an estimate");
        assert!(matches!(
            error,
            SaddleError::NotEquilibrated { .. } | SaddleError::InsufficientEffectiveSamples { .. }
        ));
    }

    #[test]
    fn sampler_typed_errors_cover_empty_records_and_blocks() {
        let no_blocks = LangevinConfig::default().with_blocks(0);
        assert!(matches!(
            no_blocks.check(),
            Err(SaddleError::MissingBlocks { context: "sampler" })
        ));
        let no_records = LangevinConfig::default().with_records(0);
        assert!(matches!(
            no_records.check(),
            Err(SaddleError::MissingSamples { context: "sampler" })
        ));
    }

    #[test]
    fn a_seed_fixes_the_stream() {
        let surface = bowl();
        let plane = Hyperplane::new(array![0.1, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.3)
            .with_records(100)
            .with_blocks(4);
        let a = sample_plane(&surface, &plane, None, &config).unwrap();
        let b = sample_plane(&surface, &plane, None, &config).unwrap();
        assert_eq!(a.last, b.last);
        assert_eq!(a.block_gradients, b.block_gradients);
        let c = sample_plane(&surface, &plane, None, &config.clone().with_seed(2)).unwrap();
        assert_ne!(a.last, c.last);
    }

    #[test]
    fn the_restricted_gaussian_is_sampled() {
        // On ν = e_x through p, q = (p_x, y, z) with (y, z) Gaussian of
        // precision K_⊥ and mean -K_⊥⁻¹ k_⊥x p_x, and ν · ∇V = (K q)_x.
        let surface = bowl();
        let px = 0.6;
        let plane = Hyperplane::new(array![px, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let kt = 0.4;
        let config = LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.4)
            .with_friction(0.5)
            .with_records(4000)
            .with_burn_in(200);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        let k_perp = array![[1.0, 0.2], [0.2, 0.5]];
        let det = k_perp[(0, 0)] * k_perp[(1, 1)] - k_perp[(0, 1)] * k_perp[(1, 0)];
        let inv = array![[0.5 / det, -0.2 / det], [-0.2 / det, 1.0 / det]];
        let mean = -inv.dot(&array![0.3 * px, 0.0]);
        let d_mean = 2.0 * px + 0.3 * mean[0];
        assert!((-sample.normal_force - d_mean).abs() < 4.0 * sample.normal_force_error);
        assert!(sample.normal_force_error < 0.02);
        assert!((sample.centroid[1] - mean[0]).abs() < 0.05);
        assert!((sample.centroid[2] - mean[1]).abs() < 0.05);
        // On the plane ⟨(P∇V) ⊗ (q - p)⟩ = kT P, whose trace is 2 kT here.
        assert_eq!(sample.free_dimensions, 2);
        assert_eq!(sample.thermal, kt);
        assert!(
            (sample.virial - 2.0 * kt).abs() < 4.0 * sample.virial_error,
            "virial {} ± {}",
            sample.virial,
            sample.virial_error
        );
        assert!(sample.virial_error < 0.03);
        assert!(sample.effective_samples > 100.0);
        assert!(sample.effective_samples < 2.0 * sample.records as f64);
        let mut lever = &sample.centroid - &plane.point();
        lever[0] = 0.0;
        let shifted = &sample.grad_normal - &(&lever * sample.normal_force);
        for (a, b) in sample.centroid_torque.iter().zip(&shifted) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!(sample.centroid_torque[0].abs() < 1e-15);
    }

    /// `K = diag(1, 100, 4)` cut by `q₀ = 0` leaves in-plane frequencies 10
    /// and 2. At `h = 0.15`, `ω h = 1.5`, and both integrators still give
    /// the exact virial `2 kT`. An OBABO splitting would give
    /// `kT Σ (1 - ω² h² / 4)⁻¹ ≈ 3.3 kT`.
    #[test]
    fn both_integrators_sample_a_stiff_harmonic_plane_exactly() {
        let surface = Bowl {
            k: Array2::from_diag(&array![1.0, 100.0, 4.0]),
            excluded: None,
        };
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        for integrator in [PlaneIntegrator::Baoab, PlaneIntegrator::Ghmc] {
            let config = LangevinConfig::default()
                .with_thermal(0.5)
                .with_time_step(0.15)
                .with_friction(2.0)
                .with_records(4000)
                .with_integrator(integrator);
            let sample = sample_plane(&surface, &plane, None, &config).unwrap();
            assert_eq!(sample.free_dimensions, 2);
            assert!(
                (sample.virial - 1.0).abs() < 4.0 * sample.virial_error,
                "{integrator:?}: virial {} ± {}",
                sample.virial,
                sample.virial_error
            );
            assert!(sample.virial_error < 0.05, "{}", sample.virial_error);
            match integrator {
                PlaneIntegrator::Baoab => assert_eq!(sample.acceptance, 1.0),
                _ => assert!(sample.acceptance < 1.0),
            }
        }
    }

    #[test]
    fn hard_half_plane_preserves_the_restricted_distribution() {
        let surface = Bowl {
            k: Array2::eye(2),
            excluded: None,
        };
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        // Interior y >= 0; the outward normal points toward negative y.
        let region = Arc::new(PlanarBoundary::new(array![0.0, 0.0], array![0.0, -1.0]).unwrap());
        let kt = 0.2;
        for integrator in [PlaneIntegrator::Baoab, PlaneIntegrator::Ghmc] {
            let config = LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.15)
                .with_friction(1.5)
                .with_burn_in(500)
                .with_records(4000)
                .with_blocks(20)
                .with_retain(64)
                .with_integrator(integrator)
                .with_region(region.clone());
            let sample = sample_plane(&surface, &plane, None, &config).unwrap();
            let exact_mean = (2.0 * kt / std::f64::consts::PI).sqrt();
            assert!((sample.centroid[1] - exact_mean).abs() < 0.04);
            assert!(
                sample
                    .retained
                    .iter()
                    .chain(std::iter::once(&sample.last))
                    .all(|point| region.contains(point.view()).unwrap())
            );
            if integrator == PlaneIntegrator::Ghmc {
                assert!(sample.acceptance < 1.0);
            }
        }
    }

    #[test]
    fn walker_near_a_neighbouring_plane_never_crosses_it() {
        let surface = Bowl {
            k: Array2::eye(2),
            excluded: None,
        };
        let plane = Hyperplane::new(array![0.0, 0.09], array![1.0, 0.0]).unwrap();
        let wall = Arc::new(PlanarBoundary::new(array![0.0, 0.1], array![0.0, 1.0]).unwrap());
        let config = LangevinConfig::default()
            .with_time_step(0.3)
            .with_records(500)
            .with_blocks(5)
            .with_retain(500)
            .with_region(wall);
        let sample = sample_plane(&surface, &plane, None, &config).unwrap();
        assert!(
            sample
                .retained
                .iter()
                .chain(std::iter::once(&sample.last))
                .all(|point| point[1] <= 0.1)
        );
    }

    struct HoldsEz;

    impl HyperplaneSampler for HoldsEz {
        fn sample_hyperplane(
            &mut self,
            _plane: &Hyperplane,
            budget: &HyperplaneSampleBudget,
        ) -> Result<HyperplaneEnsemble, SaddleError> {
            let observations = (0..budget.samples)
                .map(|_| HyperplaneObservation {
                    constraint_force: 1.0,
                    pafi_gradient: None,
                    jacobian: 0.0,
                    position: Some(array![0.0, 0.2, 0.4]),
                    projected_force: Some(array![0.0, 1.0, 2.0]),
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

        fn constrained_directions(
            &mut self,
            _plane: &Hyperplane,
        ) -> Result<Vec<Array1<f64>>, SaddleError> {
            Ok(vec![array![0.0, 0.0, 1.0]])
        }
    }

    #[test]
    fn engine_estimates_honour_constrained_directions() {
        let plane = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let config = LangevinConfig::default()
            .with_records(40)
            .with_blocks(4)
            .with_burn_in(0)
            .with_stride(1);
        let sample = sample_plane_with_sampler(&mut HoldsEz, &plane, &config).unwrap();
        assert_eq!(sample.free_dimensions, 1);
        assert!(
            sample.grad_normal[2].abs() < 1.0e-12,
            "{:?}",
            sample.grad_normal
        );
    }

    /// Engine-path pass counts for the same seeds as
    /// `iid_batch_pass_rates_match_the_equal_width_layout`. A `div_ceil`
    /// width at two blocks drops the odd lengths from about 180 of 200
    /// passes to about 10. Cells with fewer records than two per block are
    /// rejected by the sampler configuration before any batch is built;
    /// record count 13 with 7 blocks is that one cell, and the equilibration
    /// test covers it.
    #[test]
    fn engine_iid_batches_match_the_equal_width_pass_rates() {
        const TRIALS: usize = 200;
        const BASE: [[usize; 6]; 18] = [
            [188, 198, 200, 200, 198, 189],
            [190, 198, 200, 200, 200, 195],
            [184, 199, 199, 200, 199, 194],
            [183, 198, 198, 200, 200, 198],
            [185, 198, 198, 200, 200, 199],
            [191, 198, 199, 200, 200, 200],
            [189, 198, 199, 199, 200, 200],
            [194, 196, 199, 199, 200, 200],
            [190, 196, 199, 199, 200, 200],
            [189, 198, 199, 199, 200, 200],
            [194, 196, 200, 200, 200, 200],
            [190, 198, 199, 200, 200, 200],
            [190, 197, 199, 200, 200, 200],
            [193, 197, 199, 200, 200, 200],
            [194, 197, 199, 200, 200, 200],
            [179, 198, 200, 200, 200, 200],
            [182, 194, 199, 199, 200, 200],
            [190, 195, 199, 199, 200, 200],
        ];
        struct Iid {
            seed: u64,
        }
        impl HyperplaneSampler for Iid {
            fn sample_hyperplane(
                &mut self,
                plane: &Hyperplane,
                budget: &HyperplaneSampleBudget,
            ) -> Result<HyperplaneEnsemble, SaddleError> {
                let mut rng = Rng::new(self.seed);
                let origin = plane.point().to_owned();
                let zeros = Array1::zeros(origin.len());
                let observations = (0..budget.samples)
                    .map(|_| {
                        HyperplaneObservation::default()
                            .with_constraint_force(rng.normal())
                            .with_position(origin.clone())
                            .with_projected_force(zeros.clone())
                    })
                    .collect();
                Ok(HyperplaneEnsemble {
                    observations,
                    evaluations: budget.samples,
                    acceptance: 1.0,
                })
            }
        }
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        for (row, records) in (13..=30).enumerate() {
            for (col, blocks) in (2..=7).enumerate() {
                if records < 2 * blocks {
                    continue;
                }
                let mut passes = 0usize;
                for seed in 1..=TRIALS as u64 {
                    let config = LangevinConfig::default()
                        .with_records(records)
                        .with_blocks(blocks)
                        .with_stride(1)
                        .with_burn_in(0)
                        .with_seed(seed);
                    match sample_plane_with_sampler(&mut Iid { seed }, &plane, &config) {
                        Ok(sample) => {
                            let nrows = sample.block_gradients.nrows();
                            assert!(nrows >= 2);
                            assert_eq!(sample.records % nrows, 0);
                            assert!(sample.records / nrows >= 1);
                            assert!(sample.records <= records);
                            passes += 1;
                        }
                        Err(
                            SaddleError::InsufficientEffectiveSamples { .. }
                            | SaddleError::NotEquilibrated { .. },
                        ) => {}
                        Err(error) => {
                            panic!("records {records} blocks {blocks} seed {seed}: {error}")
                        }
                    }
                }
                let variance =
                    BASE[row][col] as f64 * (TRIALS - BASE[row][col]) as f64 / TRIALS as f64;
                let allowance = (3.0 * variance.sqrt()).ceil() as usize;
                let floor = BASE[row][col].saturating_sub(allowance);
                assert!(
                    passes >= floor,
                    "records {records} blocks {blocks}: {passes} passes, base {}, allowance {floor}",
                    BASE[row][col]
                );
            }
        }
    }
}
