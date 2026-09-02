//! The band session: assemble NEB forces on a caller surface, take
//! one rgmin solver step, report. Hosts own the loop.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use rgmin::{Control, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::mic::{Cell, wrap_difference};
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::spring::SpringKind;
use crate::tangent::TangentKind;
use crate::tangent::compute_tangent;

/// Climbing-image activation, the eOn trigger rule: CI arms when the
/// convergence force falls under `factor * baseline` or under the
/// absolute trigger.
#[derive(Clone, Copy, Debug)]
pub struct CiConfig {
    pub trigger_factor: f64,
    pub trigger_force: f64,
}

/// Band configuration. `force_tol` is compared to
/// [`crate::ForceGate::value`] of the projected interior force.
#[derive(Clone, Debug)]
pub struct BandConfig {
    pub tangent: TangentKind,
    pub spring: SpringKind,
    pub projection: ProjectionKind,
    pub climbing: Option<CiConfig>,
    pub cell: Option<Cell>,
    pub force_tol: f64,
    pub force_gate: crate::ForceGate,
    pub max_move: f64,
    /// Band stepper. FIRE by default, matching eOn's velocity NEB
    /// stepper. The projected band force is non-conservative;
    /// `Accept::None` on L-BFGS / BFGS / steepest takes the
    /// maxmove-clipped step (rgmin-65z1).
    pub method: Method,
}

impl Default for BandConfig {
    fn default() -> Self {
        Self {
            tangent: TangentKind::Improved,
            spring: SpringKind::Uniform { k: 5.0 },
            projection: ProjectionKind::Neb,
            climbing: Some(CiConfig {
                trigger_factor: 0.5,
                trigger_force: 0.0,
            }),
            cell: None,
            force_tol: 1e-3,
            force_gate: crate::ForceGate::LinfNorm,
            max_move: 0.2,
            method: Method::Fire {
                kind: rgmin::FireKind::V2,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandStatus {
    Running,
    Converged,
}

/// One step's report.
#[derive(Clone, Debug)]
pub struct BandReport {
    pub status: BandStatus,
    pub max_force: f64,
    pub ci_index: Option<usize>,
    pub iteration: usize,
}

/// The caller's surface: fused energies and gradients for every image
/// in one call (the batched channel is the measured win; a per-image
/// surface implements this with a loop).
pub trait BandSurface: Sync {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError>;
}

const BASELINE_UNSET: u64 = u64::MAX;
const CI_NONE: i64 = -1;

/// Climbing/baseline state shared with the evaluation closure.
/// Atomics because the rgmin oracle demands `Sync`; every store
/// happens at the single evaluation point of an `Accept::None` step.
struct ClimbState {
    baseline_bits: AtomicU64,
    ci_index: AtomicI64,
}

impl ClimbState {
    fn new() -> Self {
        Self {
            baseline_bits: AtomicU64::new(BASELINE_UNSET),
            ci_index: AtomicI64::new(CI_NONE),
        }
    }
    fn baseline(&self) -> Option<f64> {
        let bits = self.baseline_bits.load(Ordering::Relaxed);
        (bits != BASELINE_UNSET).then(|| f64::from_bits(bits))
    }
    fn ci(&self) -> Option<usize> {
        let i = self.ci_index.load(Ordering::Relaxed);
        (i >= 0).then_some(i as usize)
    }
    fn reset(&self) {
        self.baseline_bits.store(BASELINE_UNSET, Ordering::Relaxed);
        self.ci_index.store(CI_NONE, Ordering::Relaxed);
    }
}

/// Assemble the projected band force. Returns (pseudo-energy,
/// projected interior forces, max abs component).
fn assemble_band(
    config: &BandConfig,
    climb: &ClimbState,
    surface: &dyn BandSurface,
    positions: &Array2<f64>,
) -> Result<(f64, Array1<f64>, f64), SaddleError> {
    let n_images = positions.nrows();
    let dof = positions.ncols();
    let mut energies = Array1::zeros(n_images);
    let mut gradients = Array2::zeros((n_images, dof));
    surface.eval(positions.view(), &mut energies, &mut gradients)?;
    if !energies.iter().all(|e| e.is_finite()) {
        return Err(SaddleError::NonFinite("band energies"));
    }

    let mut max_e = f64::NEG_INFINITY;
    let mut max_i = 1;
    for i in 1..n_images - 1 {
        if energies[i] > max_e {
            max_e = energies[i];
            max_i = i;
        }
    }

    let ci_at = climb.ci();
    let mut projected = Array1::zeros((n_images - 2) * dof);
    let mut max_component: f64 = 0.0;
    for i in 1..n_images - 1 {
        let mut pos_diff_next = (&positions.row(i + 1) - &positions.row(i)).to_owned();
        let mut pos_diff_prev = (&positions.row(i) - &positions.row(i - 1)).to_owned();
        if let Some(cell) = &config.cell {
            wrap_difference(cell, &mut pos_diff_next);
            wrap_difference(cell, &mut pos_diff_prev);
        }
        let dist_next = pos_diff_next.dot(&pos_diff_next).sqrt();
        let dist_prev = pos_diff_prev.dot(&pos_diff_prev).sqrt();
        let tangent = compute_tangent(
            config.tangent,
            pos_diff_next.view(),
            pos_diff_prev.view(),
            energies[i],
            energies[i - 1],
            energies[i + 1],
        );
        let force = -&gradients.row(i);
        let spring = config.spring.compute(
            i,
            tangent.view(),
            dist_next,
            dist_prev,
            pos_diff_next.view(),
            pos_diff_prev.view(),
            positions.row(i),
            positions.row(i - 1),
            positions.row(i + 1),
        );

        let image_force = if ci_at == Some(i) {
            let dneb = if config.projection == ProjectionKind::DoublyNudged {
                let fp = force_perp(force.view(), tangent.view());
                dneb_component(spring.full.view(), tangent.view(), fp.view())
            } else {
                Array1::zeros(dof)
            };
            climbing_image_force(force.view(), tangent.view(), dneb.view())
        } else {
            config
                .projection
                .project(force.view(), tangent.view(), &spring)
        };
        for c in 0..dof {
            let v = image_force[c];
            if v.abs() > max_component {
                max_component = v.abs();
            }
            projected[(i - 1) * dof + c] = v;
        }
    }

    // Baseline capture and CI arming happen on assembled forces, once
    // per evaluation point.
    if climb.baseline().is_none() {
        climb
            .baseline_bits
            .store(max_component.to_bits(), Ordering::Relaxed);
    }
    if let Some(ci) = &config.climbing {
        let base = climb.baseline().unwrap_or(max_component);
        if max_component < base * ci.trigger_factor || max_component < ci.trigger_force {
            climb.ci_index.store(max_i as i64, Ordering::Relaxed);
        }
    }

    let pseudo_energy: f64 = (1..n_images - 1).map(|i| energies[i]).sum();
    let max_force = config.force_gate.value(projected.view());
    Ok((pseudo_energy, projected, max_force))
}

/// Stepping band relaxation over an rgmin solver. Endpoints (rows 0
/// and `n_images - 1`) never move.
pub struct BandSession {
    config: BandConfig,
    positions: Array2<f64>,
    solver: Solver,
    climb: ClimbState,
    iteration: usize,
}

impl BandSession {
    pub fn new(config: BandConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        let n_images = initial.nrows();
        let dof = initial.ncols();
        if n_images < 3 || dof == 0 || !dof.is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "band needs >= 3 images of 3N dof; got {n_images} x {dof}"
            )));
        }
        if let SpringKind::Weighted { ks } = &config.spring
            && ks.len() != n_images - 1
        {
            return Err(SaddleError::Shape(format!(
                "weighted springs need n_images - 1 = {} constants; got {}",
                n_images - 1,
                ks.len()
            )));
        }
        let interior_dof = (n_images - 2) * dof;
        let control = Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 1.0,
            maxmove: Some(config.max_move),
        };
        let mut solver = Solver::new(config.method.clone(), control, interior_dof);
        solver.set_highs(true);
        Ok(Self {
            config,
            positions: initial,
            solver,
            climb: ClimbState::new(),
            iteration: 0,
        })
    }

    pub fn positions(&self) -> ArrayView2<'_, f64> {
        self.positions.view()
    }

    pub fn climbing_image(&self) -> Option<usize> {
        self.climb.ci()
    }

    /// Replace the band (host-side move between steps: acquisition,
    /// reparameterization). Optimizer history survives; call
    /// [`BandSession::reset`] as well when the surface changed.
    pub fn set_positions(&mut self, positions: Array2<f64>) -> Result<(), SaddleError> {
        if positions.dim() != self.positions.dim() {
            return Err(SaddleError::Shape("set_positions shape".into()));
        }
        self.positions = positions;
        Ok(())
    }

    /// The model-update boundary: drop quasi-Newton history, the
    /// climbing baseline, and the armed climbing image. Mirrors
    /// eon_relax_reset.
    pub fn reset(&mut self) {
        self.solver.forget();
        self.climb.reset();
    }

    fn interior_flat(&self) -> Array1<f64> {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let mut flat = Array1::zeros((n_images - 2) * dof);
        for i in 1..n_images - 1 {
            flat.slice_mut(s![(i - 1) * dof..i * dof])
                .assign(&self.positions.row(i));
        }
        flat
    }

    fn scatter_interior(&mut self, flat: ArrayView1<f64>) {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        for i in 1..n_images - 1 {
            self.positions
                .row_mut(i)
                .assign(&flat.slice(s![(i - 1) * dof..i * dof]));
        }
    }

    /// One solver step over the assembled band force. The host
    /// interleaves its policy between calls.
    pub fn step<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        let n_images = self.positions.nrows();
        let dof = self.positions.ncols();
        let interior_dof = (n_images - 2) * dof;

        let endpoint_first = self.positions.row(0).to_owned();
        let endpoint_last = self.positions.row(n_images - 1).to_owned();
        let config = &self.config;
        let climb = &self.climb;
        let oracle = Oracle::unbounded(interior_dof, move |x: ArrayView1<f64>| {
            let mut full = Array2::zeros((n_images, dof));
            full.row_mut(0).assign(&endpoint_first);
            full.row_mut(n_images - 1).assign(&endpoint_last);
            for i in 1..n_images - 1 {
                full.row_mut(i).assign(&x.slice(s![(i - 1) * dof..i * dof]));
            }
            match assemble_band(config, climb, surface, &full) {
                Ok((e, f, _)) => (e, -f),
                Err(_) => (f64::INFINITY, Array1::zeros(interior_dof)),
            }
        });

        let mut x = self.interior_flat();
        let solver = &mut self.solver;
        solver
            .step(&oracle, &mut x)
            .map_err(|e| SaddleError::Solver(e.to_string()))?;
        drop(oracle);
        self.scatter_interior(x.view());
        self.iteration += 1;

        let (_, _, max_force) = assemble_band(&self.config, &self.climb, surface, &self.positions)?;
        let status = if max_force <= self.config.force_tol {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        Ok(BandReport {
            status,
            max_force,
            ci_index: self.climb.ci(),
            iteration: self.iteration,
        })
    }

    /// Convenience loop over [`BandSession::step`]; nothing more.
    pub fn run<S: BandSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<BandReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == BandStatus::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}
