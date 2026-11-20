//! The maximum of the plane free energy, by Riemannian trust region on
//! rgmin's product manifold.
//!
//! A plane is a pivot `c` and a point `(ν, d)` of `S(U) × R`, the plane
//! `{q : ν · (q - c) = d}`. `S(U)` is the unit sphere of the subspace `U`
//! of normals that the sampler leaves free: the complement of its held
//! modes, rgmin's `ManifoldKind::sphere_excluding`, or the span of
//! [`PlaneOptimizerConfig::rotation_basis`] off them,
//! `ManifoldKind::sphere_within`. The offset `d` is a Euclidean factor
//! of weight one, so a trust radius is a length along the normal. The
//! weight of the sphere factor, or its `TangentMetric`, comes from
//! [`PlaneMetric`]. A plane that only turns or only moves keeps one
//! factor.
//!
//! Moving the pivot within the plane leaves the plane where it is. With
//! [`PlaneOptimizerConfig::recenter`] the gauge hook of rgmin's
//! `minimize_rtr_gauged` puts the pivot of every accepted plane at the
//! centroid of its sample with `d = 0`, so the plane turns about that
//! centroid, where turning and moving decouple in the harmonic limit.
//!
//! The trust region minimizes `-A`. Its gradient is the sampled one,
//! `∂A/∂d = ⟨ν · ∇V⟩` and `∂A/∂ν` the torque about the pivot, which is
//! [`PlaneSample::grad_normal`] of the plane. Every plane is sampled from
//! the start plane's point with the sampler's seed and burn-in, so the
//! samples of nearby planes share their noise, and the change of `A` from
//! the current plane to a trial is the trapezoid of their gradients along
//! the chord between them. The trust region compares that change with its
//! model to accept the trial and to size the next radius. The model
//! Hessian is the harmonic one of [`PlaneMetric::Harmonic`], or one
//! curvature per factor measured along every trial step.
//!
//! Sample sizing and convergence share the root-sum-square norm of the
//! translational and orientational gradient blocks. A gradient within
//! [`PlaneOptimizerConfig::confidence`] standard errors of zero is tested
//! for convergence after the norm test of Byrd, Chin, Nocedal and Wu,
//! Math. Program. 134, 127 (2012), doi:10.1007/s10107-012-0572-5, grows the
//! sample when its error exceeds `θ |g|`. The accepted step uses the same
//! norm and confidence multiplier after its error is propagated through the
//! current model curvature. A sample that still needs growth at
//! [`PlaneOptimizerConfig::max_records`] returns a typed sample-limit status.
//! Iteration and evaluation caps likewise return a finite [`PlaneOptimum`]
//! with a typed [`PlaneOptimizerStatus`].

use std::cell::RefCell;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::manifold::{ManifoldKind, ProductFactor, ProductManifold, TangentMetric};
use rgmin::rtr::{RtrOptions, minimize_rtr_gauged};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::nichols::sym_eigh;
use crate::ohtst::anharmonic::hessian_action;
use crate::ohtst::control::HarmonicReference;
use crate::ohtst::plane::{Hyperplane, orthonormal_rows};
use crate::ohtst::sample::{
    AnalyticSurfaceSampler, LangevinConfig, PlaneSample, held_directions, mean_and_error,
    sample_plane_with_sampler,
};
use crate::projector::HyperplaneSampler;

/// Geometry of the normal in [`PlaneOptimizer`]. The offset along the
/// normal always has weight one.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub enum PlaneMetric {
    /// Weight `ℓ²` on the turn of the normal, with
    /// `ℓ = max_translation / max_rotation` of the
    /// [`PlaneOptimizerConfig`]: turning by the largest rotation is as far
    /// as moving by the largest translation. The model Hessian of each
    /// factor is a curvature measured from the change of the gradient
    /// along each trial step.
    #[default]
    Blocks,
    /// The harmonic Hessian at the reference point, usually the saddle,
    /// as a `TangentMetric`: `G = (kT/|λ₁|)(I + |λ₁| H∥⁺)` on the tangent
    /// space of the normal, with `λ₁ = νᵀH₀ν` at the start normal and
    /// `H∥` the Hessian within the start plane, off the held modes. `|λ₁|`
    /// times this metric, with `|λ₁|` on the offset, is the second
    /// derivative of `-A` for a quadratic saddle about its centroid, so the
    /// trust-region step is a harmonic Newton step. `H∥` comes from
    /// [`PlaneOptimizerConfig::metric_rank`] Lanczos steps of Hessian
    /// actions; directions they do not reach take `H∥⁺ = 0`.
    Harmonic(HarmonicReference),
}

/// Settings of [`PlaneOptimizer`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneOptimizerConfig {
    /// Sampling of each plane. Its records are the first sample size.
    /// Every plane starts from the start plane's point with this seed and
    /// burn-in.
    pub sampler: LangevinConfig,
    /// Largest sample, in records.
    pub max_records: usize,
    /// `θ` of the norm test. Zero keeps the sample size fixed.
    pub norm_test: f64,
    /// Largest trust radius, a mass-weighted length: the longest move of
    /// the plane along its normal in one step. The first radius is an
    /// eighth of it.
    pub max_translation: f64,
    /// Turn of the normal, in radians, that [`PlaneMetric::Blocks`] counts
    /// as long as [`Self::max_translation`].
    pub max_rotation: f64,
    /// Metric of the normal.
    pub metric: PlaneMetric,
    /// Lanczos steps of the in-plane Hessian for [`PlaneMetric::Harmonic`],
    /// one Hessian action each. At least the free dimension of the plane
    /// gives `H∥` exactly.
    pub metric_rank: usize,
    /// Rows spanning the directions the normal may take, such as the
    /// unstable mode and the softest modes in the plane. Their part along
    /// the held modes is dropped, and the start normal must lie in their
    /// span. `None` lets the normal take any direction off the held modes.
    pub rotation_basis: Option<Array2<f64>>,
    /// Move the plane along its normal.
    pub translate: bool,
    /// Turn the normal. Without translation the plane turns about its
    /// point.
    pub rotate: bool,
    /// Turn each accepted plane about the centroid of its sample: move the
    /// pivot there and the offset to zero. Applies only when
    /// [`Self::translate`] is set, because a plane that cannot move turns
    /// about its point.
    pub recenter: bool,
    /// `k` in the accepted-step convergence test: a translation or
    /// orientation step is converged when it is below `k` propagated
    /// standard errors.
    pub confidence: f64,
    /// Absolute accepted-step length below which a block is converged.
    pub tolerance: f64,
    /// Converged accepted steps in a row that end [`PlaneOptimizer::run`].
    pub patience: usize,
    /// Outer trust-region iterations of [`PlaneOptimizer::run`], one trial
    /// plane each.
    pub max_iterations: usize,
    /// Hard cap on reported force/energy evaluations.
    pub max_evaluations: usize,
}

impl Default for PlaneOptimizerConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            max_records: 16_000,
            norm_test: 0.5,
            max_translation: 0.1,
            max_rotation: 0.05,
            metric: PlaneMetric::Blocks,
            metric_rank: 32,
            rotation_basis: None,
            translate: true,
            rotate: true,
            recenter: true,
            confidence: 2.0,
            tolerance: 0.0,
            patience: 3,
            max_iterations: 100,
            max_evaluations: 10_000_000,
        }
    }
}

impl PlaneOptimizerConfig {
    /// Set [`Self::sampler`].
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set [`Self::max_records`].
    #[must_use]
    pub fn with_max_records(mut self, max_records: usize) -> Self {
        self.max_records = max_records;
        self
    }

    /// Set [`Self::norm_test`].
    #[must_use]
    pub fn with_norm_test(mut self, norm_test: f64) -> Self {
        self.norm_test = norm_test;
        self
    }

    /// Set [`Self::max_translation`].
    #[must_use]
    pub fn with_max_translation(mut self, max_translation: f64) -> Self {
        self.max_translation = max_translation;
        self
    }

    /// Set [`Self::max_rotation`].
    #[must_use]
    pub fn with_max_rotation(mut self, max_rotation: f64) -> Self {
        self.max_rotation = max_rotation;
        self
    }

    /// Set [`Self::metric`].
    #[must_use]
    pub fn with_metric(mut self, metric: PlaneMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set [`Self::metric_rank`].
    #[must_use]
    pub fn with_metric_rank(mut self, metric_rank: usize) -> Self {
        self.metric_rank = metric_rank;
        self
    }

    /// Set [`Self::rotation_basis`].
    #[must_use]
    pub fn with_rotation_basis(mut self, rotation_basis: Array2<f64>) -> Self {
        self.rotation_basis = Some(rotation_basis);
        self
    }

    /// Set [`Self::translate`].
    #[must_use]
    pub fn with_translate(mut self, translate: bool) -> Self {
        self.translate = translate;
        self
    }

    /// Set [`Self::rotate`].
    #[must_use]
    pub fn with_rotate(mut self, rotate: bool) -> Self {
        self.rotate = rotate;
        self
    }

    /// Set [`Self::recenter`].
    #[must_use]
    pub fn with_recenter(mut self, recenter: bool) -> Self {
        self.recenter = recenter;
        self
    }

    /// Set [`Self::confidence`].
    #[must_use]
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence;
        self
    }

    /// Set [`Self::tolerance`].
    #[must_use]
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set [`Self::patience`].
    #[must_use]
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set [`Self::max_iterations`].
    #[must_use]
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set [`Self::max_evaluations`].
    #[must_use]
    pub fn with_max_evaluations(mut self, max_evaluations: usize) -> Self {
        self.max_evaluations = max_evaluations;
        self
    }

    fn check(&self, dim: usize) -> Result<(), SaddleError> {
        self.sampler.check()?;
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(positive(self.max_translation)
            && positive(self.max_rotation)
            && positive(self.confidence))
        {
            return Err(SaddleError::Invalid(
                "plane optimizer radii and confidence must be finite and positive".into(),
            ));
        }
        if !(self.norm_test.is_finite()
            && self.norm_test >= 0.0
            && self.tolerance.is_finite()
            && self.tolerance >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "norm-test θ and tolerance must be finite and non-negative".into(),
            ));
        }
        if !(self.translate || self.rotate) {
            return Err(SaddleError::Invalid(
                "the plane optimizer has to translate, rotate, or both".into(),
            ));
        }
        let first_sample = self
            .sampler
            .burn_in
            .saturating_add(self.sampler.stride.saturating_mul(self.sampler.records))
            .saturating_add(1);
        if self.patience == 0
            || self.max_records < self.sampler.records
            || self.max_iterations == 0
            || self.max_evaluations < first_sample
        {
            return Err(SaddleError::Invalid(
                "optimizer caps must admit one iteration and the first sample".into(),
            ));
        }
        if let PlaneMetric::Harmonic(reference) = &self.metric {
            reference.check()?;
            if reference.point.len() != dim {
                return Err(SaddleError::Shape(
                    "the harmonic reference differs in length from the plane".into(),
                ));
            }
            if self.rotate && self.metric_rank == 0 {
                return Err(SaddleError::Invalid(
                    "the harmonic metric needs at least one Lanczos step".into(),
                ));
            }
        }
        if let Some(basis) = &self.rotation_basis
            && (basis.ncols() != dim || basis.iter().any(|v| !v.is_finite()))
        {
            return Err(SaddleError::Shape(
                "rotation basis rows must be finite and as long as the plane".into(),
            ));
        }
        Ok(())
    }
}

/// One accepted plane of [`PlaneOptimizer::run`], after any resampling.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneTrace {
    /// The plane.
    pub plane: Hyperplane,
    /// Mean force along the normal.
    pub normal_force: f64,
    /// Its block error.
    pub normal_force_error: f64,
    /// Length of the torque about the pivot, `∂A/∂ν` on the tangent space
    /// of the normal. Zero when the plane does not turn.
    pub rotation: f64,
    /// Root-sum-square block error of that torque.
    pub rotation_error: f64,
    /// Records in the sample.
    pub records: usize,
}

/// The converged plane of [`PlaneOptimizer::run`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneOptimizerStatus {
    /// The accepted-step criterion passed for the configured patience.
    Converged,
    /// The gradient was not resolved at the configured record ceiling.
    MaxRecords,
    /// The outer iteration cap was reached with a finite last estimate.
    MaxIterations,
    /// Another sample would exceed the force/energy evaluation cap.
    MaxEvaluations,
}

/// The last plane estimate returned by [`PlaneOptimizer::run`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct PlaneOptimum {
    /// Why the optimizer stopped.
    pub status: PlaneOptimizerStatus,
    /// The last accepted plane.
    pub plane: Hyperplane,
    /// Its sample.
    pub sample: PlaneSample,
    /// Outer trust-region iterations, one trial plane each.
    pub iterations: usize,
    /// Conjugate-gradient iterations of the trust-region subproblems.
    pub inner_iterations: usize,
    /// Trial planes the trust region accepted.
    pub accepted: usize,
    /// Planes sampled: the start, the trials and the resamples.
    pub samples: usize,
    /// Surface evaluations over the run, Hessian actions of the metric
    /// included.
    pub evaluations: usize,
    /// Every accepted plane, the start first.
    pub trace: Vec<PlaneTrace>,
}

/// Stochastic ascent of the plane free energy by rgmin's Riemannian trust
/// region on the product of the sphere of normals and the offset.
#[derive(Clone, Debug)]
pub struct PlaneOptimizer {
    plane: Hyperplane,
    config: PlaneOptimizerConfig,
}

impl PlaneOptimizer {
    /// Start at `plane`.
    pub fn new(plane: &Hyperplane, config: PlaneOptimizerConfig) -> Result<Self, SaddleError> {
        config.check(plane.dim())?;
        Ok(Self {
            plane: plane.clone(),
            config,
        })
    }

    /// The start plane.
    pub fn plane(&self) -> &Hyperplane {
        &self.plane
    }

    /// The settings.
    pub fn config(&self) -> &PlaneOptimizerConfig {
        &self.config
    }

    /// Climb `A` on `surface` until [`PlaneOptimizerConfig::patience`]
    /// accepted planes in a row are within the confidence. Running out of
    /// iterations is an error.
    pub fn run<S: PointSurface>(&self, surface: &S) -> Result<PlaneOptimum, SaddleError> {
        let config = &self.config;
        let dim = self.plane.dim();
        let held = held_directions(
            surface,
            self.plane.point(),
            None,
            config.sampler.hold_excluded,
        )?;
        let subspace = Subspace::new(dim, held, config.rotation_basis.as_ref(), config.rotate)?;
        let normal = subspace.start(self.plane.normal())?;
        let mut evaluations = 0;
        let (model, metric) = match &config.metric {
            PlaneMetric::Blocks => (
                Model::Secant {
                    rotation: 0.0,
                    offset: 0.0,
                },
                None,
            ),
            PlaneMetric::Harmonic(reference) => {
                let rank = if config.rotate { config.metric_rank } else { 0 };
                let metric = Arc::new(HarmonicRotation::new(
                    surface,
                    reference,
                    &subspace,
                    normal.view(),
                    config.sampler.thermal,
                    rank,
                    &mut evaluations,
                )?);
                let model = Model::Harmonic {
                    curvature: metric.curvature,
                    metric: Arc::clone(&metric),
                };
                (model, Some(metric))
            }
        };
        let layout = Layout {
            dim,
            rotate: config.rotate,
            translate: config.translate,
            normal,
            subspace,
        };
        let mut sampler = AnalyticSurfaceSampler::new(surface, config.sampler.clone())
            .with_start(self.plane.point().to_owned());
        self.run_engine(&mut sampler, layout, model, metric, evaluations)
    }

    /// Climb `A` using an engine-neutral constrained sampler.
    ///
    /// The generic engine path uses the block metric. A harmonic metric
    /// additionally needs Hessian actions and remains available through
    /// [`Self::run`] for analytic [`PointSurface`] references.
    pub fn run_with_sampler<E: HyperplaneSampler>(
        &self,
        sampler: &mut E,
    ) -> Result<PlaneOptimum, SaddleError> {
        let config = &self.config;
        if matches!(config.metric, PlaneMetric::Harmonic(_)) {
            return Err(SaddleError::Invalid(
                "a generic plane sampler cannot construct the harmonic optimizer metric".into(),
            ));
        }
        let dim = self.plane.dim();
        let held = constrained_basis(sampler, &self.plane)?;
        let subspace = Subspace::new(dim, held, config.rotation_basis.as_ref(), config.rotate)?;
        let normal = subspace.start(self.plane.normal())?;
        let layout = Layout {
            dim,
            rotate: config.rotate,
            translate: config.translate,
            normal,
            subspace,
        };
        self.run_engine(
            sampler,
            layout,
            Model::Secant {
                rotation: 0.0,
                offset: 0.0,
            },
            None,
            0,
        )
    }

    fn run_engine<E: HyperplaneSampler>(
        &self,
        sampler: &mut E,
        layout: Layout,
        model: Model,
        metric: Option<Arc<HarmonicRotation>>,
        evaluations: usize,
    ) -> Result<PlaneOptimum, SaddleError> {
        let config = &self.config;
        let manifold = layout.manifold(config, metric)?;
        let x0 = layout.pack(layout.normal.view(), 0.0);
        let climb = RefCell::new(Climb {
            sampler,
            config,
            pivot: self.plane.point().to_owned(),
            layout,
            records: config.sampler.records,
            model,
            current: None,
            trial: None,
            tested: false,
            streak: 0,
            converged: false,
            status: None,
            accepted_step: None,
            samples: 0,
            evaluations,
            trace: Vec::new(),
            blocks: Blocks::default(),
            error: None,
        });
        let options = RtrOptions::default()
            .with_radius_max(config.max_translation)
            .with_grad_tol(0.0)
            .with_max_iter(config.max_iterations);
        let result = minimize_rtr_gauged(
            &manifold,
            x0,
            |x| climb.borrow_mut().value(x),
            |x| climb.borrow_mut().gradient(x),
            |_, v| climb.borrow().hessian(v),
            |x| climb.borrow_mut().gauge(x),
            options,
        );
        let climb = climb.into_inner();
        if let Some(error) = climb.error {
            return Err(error);
        }
        let Some(current) = climb.current else {
            return Err(SaddleError::Solver("no plane was sampled".into()));
        };
        let status = match climb.status {
            Some(status) => status,
            None if climb.converged => PlaneOptimizerStatus::Converged,
            None => PlaneOptimizerStatus::MaxIterations,
        };
        Ok(PlaneOptimum {
            status,
            plane: current.plane,
            sample: current.sample,
            iterations: result.iterations,
            inner_iterations: result.inner_iterations,
            accepted: result.accepted,
            samples: climb.samples,
            evaluations: climb.evaluations,
            trace: climb.trace,
        })
    }
}

/// Optimize a dividing plane through an engine-neutral constrained sampler.
pub fn optimize_plane<E: HyperplaneSampler>(
    sampler: &mut E,
    initial: &Hyperplane,
    config: &PlaneOptimizerConfig,
) -> Result<PlaneOptimum, SaddleError> {
    PlaneOptimizer::new(initial, config.clone())?.run_with_sampler(sampler)
}

/// Orthonormal held modes reported by the sampler, including any component
/// along the plane normal so [`Subspace::start`] can reject that overlap.
fn constrained_basis<E: HyperplaneSampler>(
    sampler: &mut E,
    plane: &Hyperplane,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let dim = plane.dim();
    let directions = sampler.constrained_directions(plane)?;
    let mut rows = Array2::zeros((directions.len(), dim));
    for (index, direction) in directions.iter().enumerate() {
        if direction.len() != dim {
            return Err(SaddleError::Shape(
                "constrained directions differ in length from the plane".into(),
            ));
        }
        if direction.iter().any(|value| !value.is_finite()) {
            return Err(SaddleError::NonFinite("constrained direction"));
        }
        rows.row_mut(index).assign(direction);
    }
    Ok(orthonormal_rows(rows.view(), dim))
}

/// The directions the normal may take.
#[derive(Clone, Debug)]
struct Subspace {
    /// Orthonormal held modes of the sampler.
    held: Vec<Array1<f64>>,
    /// Orthonormal basis of the rotation subspace off the held modes.
    within: Option<Vec<Array1<f64>>>,
}

impl Subspace {
    fn new(
        dim: usize,
        held: Vec<Array1<f64>>,
        rotation_basis: Option<&Array2<f64>>,
        rotate: bool,
    ) -> Result<Self, SaddleError> {
        if rotate && held.len() + 1 >= dim {
            return Err(SaddleError::Invalid(
                "the held modes leave the normal no room to turn".into(),
            ));
        }
        let within = match rotation_basis {
            None => None,
            Some(basis) => {
                let mut rows = basis.clone();
                for mut row in rows.rows_mut() {
                    for mode in &held {
                        let c = mode.dot(&row);
                        row.scaled_add(-c, mode);
                    }
                }
                let basis = orthonormal_rows(rows.view(), dim);
                if basis.is_empty() {
                    return Err(SaddleError::Invalid(
                        "the rotation basis lies in the held modes".into(),
                    ));
                }
                Some(basis)
            }
        };
        Ok(Self { held, within })
    }

    /// `P_U v`.
    fn project(&self, v: ArrayView1<f64>) -> Array1<f64> {
        match &self.within {
            Some(basis) => {
                let mut out = Array1::zeros(v.len());
                for b in basis {
                    out.scaled_add(b.dot(&v), b);
                }
                out
            }
            None => {
                let mut out = v.to_owned();
                for mode in &self.held {
                    let c = mode.dot(&out);
                    out.scaled_add(-c, mode);
                }
                out
            }
        }
    }

    /// `v` on the tangent space of `S(U)` at `normal`.
    fn tangent(&self, normal: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        let mut out = self.project(v);
        let c = normal.dot(&out);
        out.scaled_add(-c, &normal);
        out
    }

    /// The start normal on `S(U)`.
    fn start(&self, normal: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let inside = self.project(normal);
        let outside = &normal - &inside;
        if outside.dot(&outside).sqrt() > 1.0e-6 {
            return Err(SaddleError::Invalid(match self.within {
                Some(_) => "the start normal lies outside the rotation basis".into(),
                None => "the plane normal has a component along a held mode; take Hyperplane::orthogonal_to first".into(),
            }));
        }
        let norm = inside.dot(&inside).sqrt();
        Ok(inside / norm)
    }

    fn kind(&self, dim: usize) -> Result<ManifoldKind, SaddleError> {
        let rows = self.within.as_ref().unwrap_or(&self.held);
        let mut basis = Array2::zeros((rows.len(), dim));
        for (mut row, b) in basis.rows_mut().into_iter().zip(rows) {
            row.assign(b);
        }
        let kind = if self.within.is_some() {
            ManifoldKind::sphere_within(basis)
        } else {
            ManifoldKind::sphere_excluding(basis)
        };
        kind.map_err(|e| SaddleError::Invalid(format!("the sphere of normals: {e}")))
    }
}

/// What the packed point of the product holds: the normal when the plane
/// turns, then the offset when it moves.
struct Layout {
    dim: usize,
    rotate: bool,
    translate: bool,
    /// The start normal, which a plane that does not turn keeps.
    normal: Array1<f64>,
    subspace: Subspace,
}

impl Layout {
    fn len(&self) -> usize {
        usize::from(self.translate) + if self.rotate { self.dim } else { 0 }
    }

    fn normal(&self, x: &Array1<f64>) -> Array1<f64> {
        if self.rotate {
            x.slice(s![..self.dim]).to_owned()
        } else {
            self.normal.clone()
        }
    }

    fn pack(&self, normal: ArrayView1<f64>, offset: f64) -> Array1<f64> {
        let mut x = Array1::zeros(self.len());
        if self.rotate {
            x.slice_mut(s![..self.dim]).assign(&normal);
        }
        if self.translate {
            x[self.len() - 1] = offset;
        }
        x
    }

    fn plane(&self, pivot: &Array1<f64>, x: &Array1<f64>) -> Result<Hyperplane, SaddleError> {
        let normal = self.normal(x);
        let offset = if self.translate { x[x.len() - 1] } else { 0.0 };
        Hyperplane::new(pivot + &(&normal * offset), normal)
    }

    /// `∇A` in the packing, from the torque about the pivot and the mean
    /// normal force.
    fn gradient(
        &self,
        normal: ArrayView1<f64>,
        torque: ArrayView1<f64>,
        normal_force: f64,
    ) -> Array1<f64> {
        let mut g = Array1::zeros(self.len());
        if self.rotate {
            g.slice_mut(s![..self.dim])
                .assign(&self.subspace.tangent(normal, torque));
        }
        if self.translate {
            g[self.len() - 1] = -normal_force;
        }
        g
    }

    fn manifold(
        &self,
        config: &PlaneOptimizerConfig,
        metric: Option<Arc<HarmonicRotation>>,
    ) -> Result<ProductManifold, SaddleError> {
        let mut factors = Vec::with_capacity(2);
        if self.rotate {
            let sphere = ProductFactor::new(self.subspace.kind(self.dim)?, self.dim);
            factors.push(match metric {
                Some(metric) => sphere.with_metric(metric),
                None => {
                    let arm = config.max_translation / config.max_rotation;
                    sphere.with_weight(arm * arm)
                }
            });
        }
        if self.translate {
            factors.push(ProductFactor::new(ManifoldKind::Euclidean, 1));
        }
        ProductManifold::new(factors)
            .map_err(|e| SaddleError::Invalid(format!("the plane manifold: {e}")))
    }
}

/// The model Hessian of `-A`.
enum Model {
    /// A curvature per factor, from the last trial step that measured a
    /// positive one; zero until then, which sends the step to the edge of
    /// the trust region.
    Secant { rotation: f64, offset: f64 },
    /// `|λ₁|` times the harmonic metric.
    Harmonic {
        curvature: f64,
        metric: Arc<HarmonicRotation>,
    },
}

impl Model {
    /// Fit the curvatures to the change of `∇A` along a trial step.
    fn update(&mut self, layout: &Layout, step: &Array1<f64>, change: &Array1<f64>) {
        let Self::Secant { rotation, offset } = self else {
            return;
        };
        if layout.rotate {
            let s = step.slice(s![..layout.dim]);
            let length = s.dot(&s);
            let k = -change.slice(s![..layout.dim]).dot(&s) / length;
            if length > 0.0 && k.is_finite() && k > 0.0 {
                *rotation = k;
            }
        }
        if layout.translate {
            let last = step.len() - 1;
            let k = -change[last] / step[last];
            if step[last] != 0.0 && k.is_finite() && k > 0.0 {
                *offset = k;
            }
        }
    }

    fn curvatures(&self) -> (f64, f64) {
        match self {
            Self::Secant { rotation, offset } => (*rotation, *offset),
            Self::Harmonic { curvature, .. } => (*curvature, *curvature),
        }
    }
}

/// One sampled plane.
struct Iterate {
    /// Its packed point.
    x: Array1<f64>,
    plane: Hyperplane,
    sample: PlaneSample,
    /// `∇A` in the packing.
    gradient: Array1<f64>,
    /// `-A` from the start, by trapezoids along the accepted steps.
    value: f64,
}

/// Gradient blocks of an accepted plane and their errors.
#[derive(Clone, Copy, Default)]
struct Blocks {
    translation: f64,
    translation_error: f64,
    rotation: f64,
    rotation_error: f64,
}

impl Blocks {
    fn norm(self) -> f64 {
        self.translation.hypot(self.rotation)
    }

    fn error_norm(self) -> f64 {
        self.translation_error.hypot(self.rotation_error)
    }
}

#[derive(Clone, Copy, Default)]
struct AcceptedStep {
    translation: f64,
    rotation: f64,
}

impl AcceptedStep {
    fn norm(self) -> f64 {
        self.translation.hypot(self.rotation)
    }
}

/// The state the trust region's callbacks share.
struct Climb<'a, E> {
    sampler: &'a mut E,
    config: &'a PlaneOptimizerConfig,
    layout: Layout,
    pivot: Array1<f64>,
    records: usize,
    model: Model,
    current: Option<Iterate>,
    trial: Option<Iterate>,
    /// The current plane has had its convergence test.
    tested: bool,
    streak: usize,
    converged: bool,
    status: Option<PlaneOptimizerStatus>,
    accepted_step: Option<AcceptedStep>,
    samples: usize,
    evaluations: usize,
    trace: Vec<PlaneTrace>,
    blocks: Blocks,
    error: Option<SaddleError>,
}

impl<E: HyperplaneSampler> Climb<'_, E> {
    fn sample_cost(&self, records: usize) -> usize {
        self.config
            .sampler
            .burn_in
            .saturating_add(self.config.sampler.stride.saturating_mul(records))
            .saturating_add(1)
    }

    fn reserve_sample(&mut self, records: usize) -> bool {
        if self.evaluations.saturating_add(self.sample_cost(records)) > self.config.max_evaluations
        {
            self.status = Some(PlaneOptimizerStatus::MaxEvaluations);
            false
        } else {
            true
        }
    }

    fn sample(&mut self, x: &Array1<f64>) -> Result<Iterate, SaddleError> {
        let plane = self.layout.plane(&self.pivot, x)?;
        let sampler = self.config.sampler.clone().with_records(self.records);
        let sample = sample_plane_with_sampler(self.sampler, &plane, &sampler)?;
        self.samples += 1;
        self.evaluations += sample.evaluations;
        let gradient = self.layout.gradient(
            plane.normal(),
            sample.grad_normal.view(),
            sample.normal_force,
        );
        Ok(Iterate {
            x: x.clone(),
            plane,
            sample,
            gradient,
            value: 0.0,
        })
    }

    /// Put the pivot at the centroid of the plane's sample, the offset to
    /// zero, and the torque about the centroid.
    fn recenter(&mut self, iterate: &mut Iterate) -> Result<(), SaddleError> {
        let normal = iterate.plane.normal().to_owned();
        let centroid = iterate.plane.project(iterate.sample.centroid.view());
        let lever = &centroid - &iterate.plane.point();
        let torque = &iterate.sample.grad_normal - &(lever * iterate.sample.normal_force);
        iterate.plane = Hyperplane::new(centroid.clone(), normal.clone())?;
        if self.layout.translate {
            let last = iterate.x.len() - 1;
            iterate.x[last] = 0.0;
        }
        iterate.gradient =
            self.layout
                .gradient(normal.view(), torque.view(), iterate.sample.normal_force);
        self.pivot = centroid;
        Ok(())
    }

    fn value(&mut self, x: &Array1<f64>) -> f64 {
        if self.error.is_some() {
            return f64::NAN;
        }
        self.try_value(x).unwrap_or_else(|error| {
            self.error = Some(error);
            f64::NAN
        })
    }

    fn try_value(&mut self, x: &Array1<f64>) -> Result<f64, SaddleError> {
        match &self.current {
            None => {
                let mut start = self.sample(x)?;
                if self.config.recenter && self.config.translate {
                    self.recenter(&mut start)?;
                }
                self.current = Some(start);
                self.tested = false;
                return Ok(0.0);
            }
            Some(current) if current.x == *x => return Ok(current.value),
            Some(_) => {}
        }
        if !self.reserve_sample(self.records) {
            return Ok(f64::INFINITY);
        }
        let mut trial = self.sample(x)?;
        let current = self.current.as_ref().expect("a current plane");
        let step = x - &current.x;
        let rise = 0.5 * (&current.gradient + &trial.gradient).dot(&step);
        trial.value = current.value - rise;
        let change = &trial.gradient - &current.gradient;
        self.model.update(&self.layout, &step, &change);
        let value = trial.value;
        self.trial = Some(trial);
        Ok(value)
    }

    fn gauge(&mut self, x: &mut Array1<f64>) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.try_gauge(x) {
            self.error = Some(error);
        }
    }

    fn try_gauge(&mut self, x: &mut Array1<f64>) -> Result<(), SaddleError> {
        let mut trial = match self.trial.take() {
            Some(trial) if trial.x == *x => trial,
            None if self.current.as_ref().is_some_and(|current| current.x == *x) => {
                self.accepted_step = Some(AcceptedStep {
                    translation: 0.0,
                    rotation: 0.0,
                });
                self.tested = false;
                return Ok(());
            }
            _ => {
                return Err(SaddleError::Solver(
                    "the trust region accepted a plane that was not sampled".into(),
                ));
            }
        };
        let current = self.current.as_ref().expect("an accepted plane");
        let step = &trial.x - &current.x;
        let rotation = if self.layout.rotate {
            step.slice(s![..self.layout.dim])
                .dot(&step.slice(s![..self.layout.dim]))
                .sqrt()
        } else {
            0.0
        };
        let translation = if self.layout.translate {
            step[step.len() - 1].abs()
        } else {
            0.0
        };
        self.accepted_step = Some(AcceptedStep {
            translation,
            rotation,
        });
        if self.config.recenter && self.config.translate {
            self.recenter(&mut trial)?;
            x.assign(&trial.x);
        }
        self.current = Some(trial);
        self.tested = false;
        Ok(())
    }

    fn gradient(&mut self, x: &Array1<f64>) -> Array1<f64> {
        if self.error.is_some() {
            return Array1::from_elem(x.len(), f64::NAN);
        }
        self.try_gradient(x).unwrap_or_else(|error| {
            self.error = Some(error);
            Array1::from_elem(x.len(), f64::NAN)
        })
    }

    fn try_gradient(&mut self, x: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        if self.current.as_ref().is_none_or(|current| current.x != *x) {
            return Err(SaddleError::Solver(
                "the trust region asked for the gradient off the current plane".into(),
            ));
        }
        if self.status.is_some() {
            return Ok(Array1::zeros(x.len()));
        }
        if !self.tested {
            self.tested = true;
            self.test()?;
        }
        if self.converged {
            return Ok(Array1::zeros(x.len()));
        }
        let current = self.current.as_ref().expect("checked above");
        Ok(-&current.gradient)
    }

    /// Grow the sample by the norm test, test the blocks, and record the
    /// plane.
    fn test(&mut self) -> Result<(), SaddleError> {
        let mut blocks = self.blocks();
        if let Some(records) = self.grown(&blocks) {
            if !self.reserve_sample(records) {
                return Ok(());
            }
            self.records = records;
            let current = self.current.take().expect("an accepted plane");
            let mut again = self.sample(&current.x)?;
            again.value = current.value;
            if self.config.recenter && self.config.translate {
                self.recenter(&mut again)?;
            }
            self.current = Some(again);
            blocks = self.blocks();
        }
        let sample_limited = self.records >= self.config.max_records && self.needs_growth(&blocks);
        let (rotation_curvature, translation_curvature) = self.model.curvatures();
        let propagated = |error: f64, curvature: f64| {
            if curvature.is_finite() && curvature > 0.0 {
                error / curvature
            } else {
                0.0
            }
        };
        let value = blocks.norm();
        let value_error = blocks.error_norm();
        let value_zero = self.statistically_zero(value, value_error);
        let within = match self.accepted_step {
            Some(step) => {
                let step_value = step.norm();
                let step_error = propagated(blocks.translation_error, translation_curvature)
                    .hypot(propagated(blocks.rotation_error, rotation_curvature));
                value_zero && self.statistically_zero(step_value, step_error)
            }
            None => value_zero,
        };
        self.streak = if within { self.streak + 1 } else { 0 };
        self.converged =
            self.streak >= self.config.patience || (self.accepted_step.is_none() && within);
        if self.converged {
            self.status = Some(PlaneOptimizerStatus::Converged);
        } else if sample_limited {
            self.status = Some(PlaneOptimizerStatus::MaxRecords);
        }
        self.blocks = blocks;
        let current = self.current.as_ref().expect("an accepted plane");
        self.trace.push(PlaneTrace {
            plane: current.plane.clone(),
            normal_force: current.sample.normal_force,
            normal_force_error: current.sample.normal_force_error,
            rotation: blocks.rotation,
            rotation_error: blocks.rotation_error,
            records: current.sample.records,
        });
        Ok(())
    }

    /// Whether the norm test requires a larger sample.
    fn needs_growth(&self, blocks: &Blocks) -> bool {
        if self.config.norm_test == 0.0 {
            return false;
        }
        blocks.error_norm() > self.config.norm_test * blocks.norm()
    }

    /// Sample size the norm test asks for, when it is larger.
    fn grown(&self, blocks: &Blocks) -> Option<usize> {
        if !self.needs_growth(blocks) || self.records >= self.config.max_records {
            return None;
        }
        let variance = blocks.error_norm().powi(2);
        let signal = (self.config.norm_test * blocks.norm()).powi(2);
        let grown = if signal > 0.0 {
            (self.records as f64 * variance / signal).ceil()
        } else {
            f64::INFINITY
        };
        let records = (grown.min(self.config.max_records as f64) as usize).max(self.records);
        (records > self.records).then_some(records)
    }

    fn statistically_zero(&self, value: f64, error: f64) -> bool {
        value <= self.config.confidence * error || value <= self.config.tolerance
    }

    fn blocks(&self) -> Blocks {
        let current = self.current.as_ref().expect("an accepted plane");
        let sample = &current.sample;
        let mut blocks = Blocks::default();
        if self.layout.translate {
            blocks.translation = sample.normal_force.abs();
            blocks.translation_error = sample.normal_force_error;
        }
        if self.layout.rotate {
            let dim = self.layout.dim;
            let torque = current.gradient.slice(s![..dim]);
            blocks.rotation = torque.dot(&torque).sqrt();
            let rows = if self.config.recenter && self.config.translate {
                &sample.centroid_block_gradients
            } else {
                &sample.block_gradients
            };
            let normal = current.plane.normal();
            let rows: Vec<Array1<f64>> = rows
                .rows()
                .into_iter()
                .map(|row| self.layout.subspace.tangent(normal, row.slice(s![dim..])))
                .collect();
            let variance: f64 = (0..dim)
                .map(|i| {
                    let column: Vec<f64> = rows.iter().map(|row| row[i]).collect();
                    mean_and_error(&column).1.powi(2)
                })
                .sum();
            blocks.rotation_error = variance.sqrt();
        }
        blocks
    }

    fn hessian(&self, v: &Array1<f64>) -> Array1<f64> {
        let dim = self.layout.dim;
        let mut out = Array1::zeros(v.len());
        let (rotation, offset) = match &self.model {
            Model::Secant { rotation, offset } => (*rotation, *offset),
            Model::Harmonic { curvature, .. } => (*curvature, *curvature),
        };
        if self.layout.rotate {
            let turn = v.slice(s![..dim]).to_owned();
            let image = match &self.model {
                Model::Harmonic { metric, .. } => metric.full(&turn),
                Model::Secant { .. } => turn,
            };
            out.slice_mut(s![..dim]).assign(&(image * rotation));
        }
        if self.layout.translate {
            let last = v.len() - 1;
            out[last] = offset * v[last];
        }
        out
    }
}

/// The harmonic metric of [`PlaneMetric::Harmonic`]:
/// `M = s I + Σ_k w_k y_k y_kᵀ` with `s = kT/|λ₁|` and Ritz pairs
/// `(θ_k, y_k)` of `H∥`, `w_k = kT/θ_k`, compressed to the tangent space
/// of `S(U)` at each normal.
struct HarmonicRotation {
    /// `|λ₁|`.
    curvature: f64,
    scale: f64,
    modes: Vec<Array1<f64>>,
    weights: Vec<f64>,
    subspace: Subspace,
    /// For a rotation basis `B`: the coordinates `Z = [B y_k]` and
    /// `K = (s W⁻¹ + Z Zᵀ)⁻¹`, so that `(B M Bᵀ)⁻¹ = (I - Zᵀ K Z) / s`.
    within: Option<(Array2<f64>, Array2<f64>)>,
}

impl HarmonicRotation {
    fn new<S: PointSurface>(
        surface: &S,
        reference: &HarmonicReference,
        subspace: &Subspace,
        normal: ArrayView1<f64>,
        thermal: f64,
        rank: usize,
        evaluations: &mut usize,
    ) -> Result<Self, SaddleError> {
        let dim = normal.len();
        let at = reference.point.view();
        let step = reference.hvp_step;
        let normal = normal.to_owned();
        let lambda = normal.dot(&hessian_action(surface, at, &normal, step, evaluations)?);
        if lambda.is_nan() || lambda >= 0.0 {
            return Err(SaddleError::Invalid(format!(
                "the reference Hessian is not negative along the start normal: {lambda:.3e}"
            )));
        }
        let free = |v: &Array1<f64>| {
            let mut out = v.clone();
            for mode in subspace.held.iter().chain(std::iter::once(&normal)) {
                let c = mode.dot(&out);
                out.scaled_add(-c, mode);
            }
            out
        };
        let steps = rank.min(dim.saturating_sub(1 + subspace.held.len()));
        let mut basis: Vec<Array1<f64>> = Vec::with_capacity(steps);
        let mut diagonal = Vec::with_capacity(steps);
        let mut off = Vec::with_capacity(steps);
        let mut q = free(&probe(dim));
        let norm = q.dot(&q).sqrt();
        if steps > 0 && norm > 0.0 {
            q /= norm;
            basis.push(q);
            loop {
                let last = basis.last().expect("pushed above");
                let mut w = free(&hessian_action(surface, at, last, step, evaluations)?);
                diagonal.push(last.dot(&w));
                for _ in 0..2 {
                    for b in &basis {
                        let c = b.dot(&w);
                        w.scaled_add(-c, b);
                    }
                }
                let beta = w.dot(&w).sqrt();
                let scale = diagonal.iter().fold(lambda.abs(), |m, d| m.max(d.abs()));
                if basis.len() == steps || beta.is_nan() || beta <= 1.0e-10 * scale {
                    break;
                }
                off.push(beta);
                basis.push(w / beta);
            }
        }
        let mut modes = Vec::new();
        let mut weights = Vec::new();
        if !diagonal.is_empty() {
            let m = diagonal.len();
            let mut tridiagonal = Array2::zeros((m, m));
            for (i, d) in diagonal.iter().enumerate() {
                tridiagonal[(i, i)] = *d;
            }
            for (i, b) in off.iter().enumerate() {
                tridiagonal[(i, i + 1)] = *b;
                tridiagonal[(i + 1, i)] = *b;
            }
            let (theta, vectors) = sym_eigh(tridiagonal.view())?;
            let top = theta.iter().fold(0.0_f64, |m, t| m.max(t.abs()));
            for (k, &value) in theta.iter().enumerate() {
                if value < -1.0e-8 * top {
                    return Err(SaddleError::Invalid(format!(
                        "the reference Hessian has a negative direction in the start plane: {value:.3e}"
                    )));
                }
                if value <= 1.0e-8 * top {
                    continue;
                }
                let mut y = Array1::zeros(dim);
                for (j, b) in basis.iter().enumerate() {
                    y.scaled_add(vectors[(j, k)], b);
                }
                let length = y.dot(&y).sqrt();
                modes.push(y / length);
                weights.push(thermal / value);
            }
        }
        let scale = thermal / lambda.abs();
        let within = subspace.within.as_ref().map(|rows| {
            let mut coords = Array2::zeros((modes.len(), rows.len()));
            for (k, y) in modes.iter().enumerate() {
                for (i, b) in rows.iter().enumerate() {
                    coords[(k, i)] = b.dot(y);
                }
            }
            let mut kernel = coords.dot(&coords.t());
            for (k, w) in weights.iter().enumerate() {
                kernel[(k, k)] += scale / w;
            }
            (coords, kernel)
        });
        let within = match within {
            None => None,
            Some((coords, kernel)) => Some((coords, spd_inverse(&kernel)?)),
        };
        Ok(Self {
            curvature: lambda.abs(),
            scale,
            modes,
            weights,
            subspace: subspace.clone(),
            within,
        })
    }

    /// `M v` in the ambient space.
    fn full(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = v * self.scale;
        for (y, w) in self.modes.iter().zip(&self.weights) {
            out.scaled_add(w * y.dot(v), y);
        }
        out
    }

    /// `M⁻¹ v` in the ambient space.
    fn inverse(&self, v: &Array1<f64>) -> Array1<f64> {
        let mut out = v.clone();
        for (y, w) in self.modes.iter().zip(&self.weights) {
            out.scaled_add(-w / (self.scale + w) * y.dot(v), y);
        }
        out / self.scale
    }
}

impl TangentMetric for HarmonicRotation {
    fn apply(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        let v = self.subspace.tangent(x.view(), v.view());
        self.subspace.tangent(x.view(), self.full(&v).view())
    }

    /// The tangent `y` with `P M y = r`: `y = M⁻¹(r - C μ)` with `C` the
    /// normal and the held modes, `μ` from `Cᵀ y = 0`. With a rotation
    /// basis the same holds in its coordinates with `C` the normal alone.
    fn solve(&self, x: &Array1<f64>, r: &Array1<f64>) -> Array1<f64> {
        let r = self.subspace.tangent(x.view(), r.view());
        match (&self.within, &self.subspace.within) {
            (Some((coords, kernel)), Some(rows)) => {
                let to =
                    |v: &Array1<f64>| -> Array1<f64> { rows.iter().map(|b| b.dot(v)).collect() };
                let inverse = |v: &Array1<f64>| -> Array1<f64> {
                    let z = coords.dot(v);
                    (v - &coords.t().dot(&kernel.dot(&z))) / self.scale
                };
                let (normal, r) = (to(x), to(&r));
                let z = inverse(&r);
                let u = inverse(&normal);
                let mu = normal.dot(&z) / normal.dot(&u);
                let y = z - u * mu;
                let mut out = Array1::zeros(x.len());
                for (b, c) in rows.iter().zip(&y) {
                    out.scaled_add(*c, b);
                }
                out
            }
            _ => {
                let constraints: Vec<&Array1<f64>> =
                    std::iter::once(x).chain(&self.subspace.held).collect();
                let z = self.inverse(&r);
                let u: Vec<Array1<f64>> = constraints.iter().map(|c| self.inverse(c)).collect();
                let k = constraints.len();
                let mut schur = Array2::zeros((k, k));
                let mut rhs = Array1::zeros(k);
                for (i, c) in constraints.iter().enumerate() {
                    rhs[i] = c.dot(&z);
                    for (j, uj) in u.iter().enumerate() {
                        schur[(i, j)] = c.dot(uj);
                    }
                }
                let mu = spd_solve(&schur, &rhs).unwrap_or_else(|| Array1::zeros(k));
                let mut y = z;
                for (m, uj) in mu.iter().zip(&u) {
                    y.scaled_add(-m, uj);
                }
                y
            }
        }
    }
}

/// A fixed start for Lanczos, SplitMix64 uniforms on `[-1, 1)`.
fn probe(dim: usize) -> Array1<f64> {
    let mut state: u64 = 0x2545_f491_4f6c_dd1d;
    (0..dim)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            2.0 * ((z >> 11) as f64 / (1u64 << 53) as f64) - 1.0
        })
        .collect()
}

/// Lower Cholesky factor of a symmetric positive definite matrix.
fn cholesky(a: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[(i, j)];
            for k in 0..j {
                sum -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 {
                    return None;
                }
                l[(i, i)] = sum.sqrt();
            } else {
                l[(i, j)] = sum / l[(j, j)];
            }
        }
    }
    Some(l)
}

fn cholesky_solve(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut y = b.clone();
    for i in 0..n {
        for k in 0..i {
            y[i] -= l[(i, k)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    for i in (0..n).rev() {
        for k in i + 1..n {
            y[i] -= l[(k, i)] * y[k];
        }
        y[i] /= l[(i, i)];
    }
    y
}

fn spd_solve(a: &Array2<f64>, b: &Array1<f64>) -> Option<Array1<f64>> {
    cholesky(a).map(|l| cholesky_solve(&l, b))
}

fn spd_inverse(a: &Array2<f64>) -> Result<Array2<f64>, SaddleError> {
    let n = a.nrows();
    let l = cholesky(a).ok_or_else(|| {
        SaddleError::Solver("the harmonic metric is not positive definite".into())
    })?;
    let mut inverse = Array2::zeros((n, n));
    for i in 0..n {
        let mut e = Array1::zeros(n);
        e[i] = 1.0;
        inverse.column_mut(i).assign(&cholesky_solve(&l, &e));
    }
    Ok(inverse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ohtst::{ExactGaussianPlaneSampler, QuadraticModel};
    use crate::projector::{HyperplaneEnsemble, HyperplaneSampleBudget, HyperplaneSampler};
    use ndarray::array;

    /// `V = -½ x² + y² + 1.5 z²`: the free energy of a plane is
    /// `(kT/2) ln|νᵀH⁻¹ν| + (ν · p)² / (2 νᵀH⁻¹ν)`, a maximum at `ν = e_x`
    /// through the origin.
    struct HarmonicSaddle;

    impl PointSurface for HarmonicSaddle {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = array![-q[0], 2.0 * q[1], 3.0 * q[2]];
            Ok((
                0.5 * (-q[0] * q[0] + 2.0 * q[1] * q[1] + 3.0 * q[2] * q[2]),
                g,
            ))
        }
    }

    /// `V = -x²/2 + y²`. For a unit normal `ν` and signed offset `d`,
    /// the constrained Gaussian has
    /// `A = d²/(2 νᵀH⁻¹ν) + (kT/2) ln(-νᵀH⁻¹ν) + constant`.
    /// Its unique unoriented optimum is `d = 0`, `[ν] = [e_x]`.
    struct HarmonicSaddle2d;

    impl PointSurface for HarmonicSaddle2d {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((-0.5 * q[0] * q[0] + q[1] * q[1], array![-q[0], 2.0 * q[1]]))
        }
    }

    /// At `x = 0`, `kT = 0.2`, the constrained `y` variance is `0.1`.
    /// The true normal mean is zero while finite samples remain noisy.
    struct NoisyZeroMean;

    impl PointSurface for NoisyZeroMean {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y) = (q[0], q[1]);
            Ok((
                -0.5 * x * x + y * y + 0.2 * x * (y * y - 0.1),
                array![-x + 0.2 * (y * y - 0.1), 2.0 * y + 0.4 * x * y],
            ))
        }
    }

    fn config() -> PlaneOptimizerConfig {
        PlaneOptimizerConfig::default()
            .with_sampler(
                LangevinConfig::default()
                    .with_thermal(0.2)
                    .with_time_step(0.3)
                    .with_burn_in(200)
                    .with_records(1000),
            )
            .with_max_translation(0.1)
            .with_max_rotation(0.1)
            .with_tolerance(1.0e-6)
            .with_max_iterations(200)
    }

    fn assert_at_the_saddle_plane(optimum: &PlaneOptimum) {
        assert_eq!(optimum.status, PlaneOptimizerStatus::Converged);
        let nu = optimum.plane.normal();
        let tilt = (nu[1] * nu[1] + nu[2] * nu[2]).sqrt();
        let offset = optimum.plane.distance(array![0.0, 0.0, 0.0].view());
        assert!(tilt < 1.0e-4, "tilt {tilt}");
        assert!(offset.abs() < 1.0e-4, "offset {offset}");
        assert_eq!(optimum.trace.len(), optimum.accepted + 1);
        assert!(optimum.samples > optimum.iterations);
    }

    #[test]
    fn harmonic_saddle_plane_is_found() {
        let start = Hyperplane::new(array![0.3, 0.1, -0.2], array![1.0, 0.3, -0.25]).unwrap();
        let optimum = PlaneOptimizer::new(&start, config())
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap();
        assert_at_the_saddle_plane(&optimum);
        let first = optimum.trace[0].plane.normal();
        assert!((first.dot(&start.normal()) - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn trait_level_optimizer_uses_exact_plane_samples() {
        let start = Hyperplane::new(array![0.05, 0.0, 0.0], array![1.0, 0.04, -0.03]).unwrap();
        let model = QuadraticModel::new(
            array![0.0, 0.0, 0.0],
            Array2::from_diag(&array![-1.0, 2.0, 3.0]),
        );
        let mut sampler = ExactGaussianPlaneSampler::new(model).unwrap();
        let config = config()
            .with_sampler(
                LangevinConfig::default()
                    .with_thermal(0.2)
                    .with_records(600)
                    .with_blocks(12),
            )
            .with_max_records(600)
            .with_tolerance(2.0e-6);
        let optimum = optimize_plane(&mut sampler, &start, &config).unwrap();
        assert_at_the_saddle_plane(&optimum);
    }

    #[test]
    fn unresolved_noisy_optimum_reports_the_record_cap() {
        let start = Hyperplane::new(array![0.02, 0.0], array![1.0, 0.03]).unwrap();
        let sampling = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.1)
            .with_burn_in(200)
            .with_stride(4)
            .with_records(600)
            .with_blocks(12)
            .with_seed(73);
        let mut sampler = AnalyticSurfaceSampler::new(&NoisyZeroMean, sampling.clone());
        let config = PlaneOptimizerConfig::default()
            .with_sampler(sampling)
            .with_max_records(2_400)
            .with_max_iterations(120)
            .with_max_evaluations(500_000)
            .with_tolerance(2.0e-4);
        let optimum = optimize_plane(&mut sampler, &start, &config).unwrap();
        assert_eq!(optimum.status, PlaneOptimizerStatus::MaxRecords);
        assert!(optimum.evaluations <= config.max_evaluations);
        assert!(optimum.plane.distance(array![0.0, 0.0].view()).abs() < 0.01);
        assert!(optimum.plane.normal()[1].abs() < 0.01);
    }

    #[test]
    fn iteration_cap_returns_the_last_finite_estimate() {
        let start = Hyperplane::new(array![0.3, 0.1, -0.2], array![1.0, 0.3, -0.25]).unwrap();
        let optimum = PlaneOptimizer::new(&start, config().with_max_iterations(1))
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap();
        assert_eq!(optimum.status, PlaneOptimizerStatus::MaxIterations);
        assert!(optimum.sample.normal_force.is_finite());
    }

    #[test]
    fn fixed_noisy_sample_can_pass_the_statistical_test() {
        let start = Hyperplane::new(array![0.25, 0.0], array![1.0, 0.2]).unwrap();
        let sampling = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.1)
            .with_burn_in(10)
            .with_stride(1)
            .with_records(8)
            .with_blocks(2)
            .with_seed(7);
        let config = PlaneOptimizerConfig::default()
            .with_sampler(sampling)
            .with_max_records(8)
            .with_patience(1)
            .with_max_iterations(10);
        let optimum = PlaneOptimizer::new(&start, config)
            .unwrap()
            .run(&NoisyZeroMean)
            .unwrap();
        assert_eq!(optimum.status, PlaneOptimizerStatus::Converged);
    }

    #[test]
    fn grown_sample_at_the_cap_does_not_report_converged() {
        let start = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let sampling = LangevinConfig::default()
            .with_thermal(0.2)
            .with_time_step(0.1)
            .with_burn_in(0)
            .with_stride(1)
            .with_records(8_000)
            .with_blocks(2)
            .with_seed(7);
        let config = PlaneOptimizerConfig::default()
            .with_sampler(sampling)
            .with_max_records(16_000)
            .with_confidence(0.01)
            .with_rotate(false)
            .with_patience(1)
            .with_max_iterations(3);
        let optimum = PlaneOptimizer::new(&start, config)
            .unwrap()
            .run(&NoisyZeroMean)
            .unwrap();
        assert_eq!(optimum.sample.records, 16_000);
        assert_eq!(optimum.status, PlaneOptimizerStatus::MaxRecords);
        assert!(optimum.iterations < 3);
    }

    #[test]
    fn evaluation_cap_returns_the_last_finite_estimate() {
        let start = Hyperplane::new(array![0.05, 0.0, 0.0], array![1.0, 0.04, -0.03]).unwrap();
        let model = QuadraticModel::new(
            array![0.0, 0.0, 0.0],
            Array2::from_diag(&array![-1.0, 2.0, 3.0]),
        );
        let mut sampler = ExactGaussianPlaneSampler::new(model).unwrap();
        let sampling = LangevinConfig::default().with_records(600).with_blocks(12);
        let first_sample_cap = sampling.burn_in + sampling.stride * sampling.records + 1;
        let config = PlaneOptimizerConfig::default()
            .with_sampler(sampling)
            .with_max_records(600)
            .with_max_evaluations(first_sample_cap);
        let optimum = optimize_plane(&mut sampler, &start, &config).unwrap();
        assert_eq!(optimum.status, PlaneOptimizerStatus::MaxEvaluations);
        assert!(optimum.evaluations <= config.max_evaluations);
        assert!(optimum.sample.normal_force.is_finite());
    }

    #[test]
    fn analytic_two_dimensional_plane_is_found() {
        let start = Hyperplane::new(array![0.2, -0.1], array![1.0, 0.15]).unwrap();
        let optimum = PlaneOptimizer::new(&start, config())
            .unwrap()
            .run(&HarmonicSaddle2d)
            .unwrap();
        let normal = optimum.plane.normal();
        assert!(normal[1].abs() < 1.0e-4, "normal = {normal:?}");
        assert!(
            optimum.plane.distance(array![0.0, 0.0].view()).abs() < 1.0e-4,
            "point = {:?}",
            optimum.plane.point()
        );
    }

    #[test]
    fn the_harmonic_metric_finds_the_saddle_plane() {
        let start = Hyperplane::new(array![0.3, 0.1, -0.2], array![1.0, 0.3, -0.25]).unwrap();
        let reference = HarmonicReference::new(array![0.0, 0.0, 0.0]);
        let config = config().with_metric(PlaneMetric::Harmonic(reference));
        let optimum = PlaneOptimizer::new(&start, config)
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap();
        assert_at_the_saddle_plane(&optimum);
    }

    /// `H∥ = diag(2, 3)` in the plane `x = 0` and `λ₁ = -1`, so
    /// `G = (kT/|λ₁|)(I + |λ₁| H∥⁻¹)` is diagonal with `kT (1, 1.5, 4/3)`.
    /// It inverts on every tangent space, inside a rotation basis too.
    #[test]
    fn the_harmonic_metric_inverts_on_the_tangent_space() {
        let normal = array![1.0, 0.0, 0.0];
        let reference = HarmonicReference::new(array![0.0, 0.0, 0.0]);
        let kt = 0.2;
        for basis in [None, Some(array![[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])] {
            let subspace = Subspace::new(3, Vec::new(), basis.as_ref(), true).unwrap();
            let mut evaluations = 0;
            let metric = HarmonicRotation::new(
                &HarmonicSaddle,
                &reference,
                &subspace,
                normal.view(),
                kt,
                8,
                &mut evaluations,
            )
            .unwrap();
            assert!((metric.curvature - 1.0).abs() < 1.0e-6);
            assert_eq!(evaluations, 2 * 3);
            let image = metric.full(&array![1.0, 1.0, 1.0]);
            let exact = array![kt, 1.5 * kt, 4.0 / 3.0 * kt];
            assert!(
                (&image - &exact).iter().all(|e| e.abs() < 1.0e-6),
                "{image}"
            );
            let at = subspace.project(array![0.8, 0.36, 0.48].view());
            let at = &at / at.dot(&at).sqrt();
            let v = subspace.tangent(at.view(), array![0.1, -0.7, 0.4].view());
            let gv = metric.apply(&at, &v);
            let back = metric.solve(&at, &gv);
            assert!(
                (&back - &v).iter().all(|e| e.abs() < 1.0e-10),
                "{back} against {v}"
            );
            assert!(gv.dot(&v) > 0.0);
        }
    }

    #[test]
    fn rotation_alone_turns_about_the_point() {
        let start = Hyperplane::new(array![0.0, 0.4, 0.0], array![1.0, 0.2, 0.0]).unwrap();
        let optimum = PlaneOptimizer::new(&start, config().with_translate(false))
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap();
        assert_eq!(optimum.plane.point(), start.point());
        assert!(optimum.plane.normal()[1].abs() < 1.0e-4);
    }

    #[test]
    fn a_rotation_basis_keeps_the_normal_in_its_span() {
        let start = Hyperplane::new(array![0.1, 0.0, 0.2], array![1.0, 0.3, 0.0]).unwrap();
        let config = config().with_rotation_basis(array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let optimum = PlaneOptimizer::new(&start, config.clone())
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap();
        assert_eq!(optimum.plane.normal()[2], 0.0);
        assert!(optimum.plane.normal()[1].abs() < 1.0e-4);
        let outside = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.3]).unwrap();
        let error = PlaneOptimizer::new(&outside, config)
            .unwrap()
            .run(&HarmonicSaddle)
            .unwrap_err();
        assert!(error.to_string().contains("rotation basis"), "{error}");
    }

    #[test]
    fn settings_are_checked() {
        let start = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        for config in [
            config().with_translate(false).with_rotate(false),
            config().with_max_rotation(0.0),
            config().with_patience(0),
            config().with_rotation_basis(array![[1.0, 0.0]]),
            config().with_metric(PlaneMetric::Harmonic(HarmonicReference::new(array![0.0]))),
        ] {
            assert!(PlaneOptimizer::new(&start, config).is_err());
        }
    }

    /// One held direction in two dimensions fills the sphere of normals.
    struct HoldsTheOnlyTurn {
        sampled: bool,
    }

    impl HyperplaneSampler for HoldsTheOnlyTurn {
        fn sample_hyperplane(
            &mut self,
            _plane: &Hyperplane,
            _budget: &HyperplaneSampleBudget,
        ) -> Result<HyperplaneEnsemble, SaddleError> {
            self.sampled = true;
            Err(SaddleError::Invalid(
                "the optimizer sampled despite a full constraint".into(),
            ))
        }

        fn constrained_directions(
            &mut self,
            _plane: &Hyperplane,
        ) -> Result<Vec<Array1<f64>>, SaddleError> {
            Ok(vec![array![0.0, 1.0]])
        }
    }

    #[test]
    fn a_full_constraint_stops_the_optimizer_before_sampling() {
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
        let optimizer = PlaneOptimizer::new(&plane, PlaneOptimizerConfig::default()).unwrap();
        let mut sampler = HoldsTheOnlyTurn { sampled: false };
        let error = optimizer
            .run_with_sampler(&mut sampler)
            .expect_err("the held direction fills the sphere");
        assert!(error.to_string().contains("no room to turn"), "{error}");
        assert!(!sampler.sampled);
    }
}
