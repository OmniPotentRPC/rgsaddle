//! Band presets on [`Chain`](super::Chain) and the triad operations.
//!
//! NEB, doubly nudged, climbing-image, energy-weighted, and zoom share
//! one force assembly. AutoNEB keeps its insertion loop and reports
//! the chain's stable image ids to the callback.

use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::autoneb::{AutoNeb, AutoNebTuning};
use crate::band::{BandConfig, BandStatus, CiConfig};
use crate::error::SaddleError;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component};
use crate::spring::SpringForces;
use crate::stencil::spring_ew::{K_MAX_EV_PER_A2, K_MIN_EV_PER_A2, fill_constants};

use super::batch::{IdBook, bind_for_step};
use super::chain::{Chain, EvalPool};
use super::idpp::IdppRun;
use super::triad::{Coupling, TangentRule, coupling_forces, perp, unit_tangent};
use super::zoom::{ZoomEnds, ZoomParams, ZoomWatch, pack_window, select_window};
use super::{PHASE_BAND, PHASE_INSERT, PHASE_ZOOM, Preset, StencilReport, StencilStatus, Surface};

const UNSUPPORTED_OCI: &str = "OCI-NEB is not implemented yet";
const UNSUPPORTED_SIDPP: &str = "S-IDPP initial paths are not implemented yet";

#[derive(Clone, Copy)]
struct ForceLaw {
    projection: ProjectionKind,
    climb: bool,
    tangent: TangentRule,
    k: f64,
    energy_weighted: Option<(f64, f64)>,
    trigger_factor: f64,
    trigger_force: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Neb,
    Dneb,
    Ci,
    Weighted,
    Zoom,
    Auto,
    Idpp,
}

struct Inner {
    chain: Chain,
    free_ends: bool,
    armed: bool,
    baseline: Option<f64>,
    ci: Option<usize>,
    total_rows: usize,
    last_projected: Option<Array1<f64>>,
    last_max: f64,
    error: Option<SaddleError>,
}

pub struct BandEngine {
    inner: Arc<Mutex<Inner>>,
    flat_cache: Array1<f64>,
    pool: Arc<EvalPool>,
    projection: ProjectionKind,
    climb: bool,
    tangent: TangentRule,
    k: f64,
    energy_weighted: Option<(f64, f64)>,
    force_tol: f64,
    max_move: f64,
    method: Method,
    trigger_factor: f64,
    trigger_force: f64,
    band_slack: f64,
    image_scale: Option<f64>,
    solver: Solver,
    solver_n: usize,
    iteration: usize,
    zoom: Option<ZoomWatch>,
    zoom_due: bool,
    zoomed: bool,
    zoom_window: Option<(usize, usize)>,
    post_steps: usize,
    auto: Option<AutoNeb>,
    book: Option<Arc<IdBook>>,
    idpp: Option<IdppRun>,
}

pub struct BandSpec {
    pub preset: Preset,
    pub x: Array2<f64>,
    pub threads: usize,
    pub k: f64,
    pub energy_weighted: Option<(f64, f64)>,
    pub force_tol: f64,
    pub max_move: f64,
    pub method: Method,
    pub tangent: TangentRule,
    pub trigger_factor: f64,
    pub trigger_force: f64,
    pub band_slack: f64,
    pub image_scale: Option<f64>,
    pub zoom: Option<ZoomParams>,
    pub n_simul: usize,
    pub n_max: usize,
}

impl BandEngine {
    pub fn new(spec: BandSpec) -> Result<Self, SaddleError> {
        let kind = kind_of(spec.preset)?;
        if spec.threads == 0 {
            return Err(SaddleError::Invalid(
                "chain evaluation needs at least one thread".into(),
            ));
        }
        if let Some((k_min, k_max)) = spec.energy_weighted {
            check_bounds(k_min, k_max)?;
        }
        if !(spec.k.is_finite() && spec.k > 0.0) {
            return Err(SaddleError::Invalid(
                "spring constant must be finite and positive".into(),
            ));
        }
        if !(spec.force_tol.is_finite() && spec.force_tol > 0.0) {
            return Err(SaddleError::Invalid(
                "force tolerance must be finite and positive".into(),
            ));
        }
        if !(spec.max_move.is_finite() && spec.max_move > 0.0) {
            return Err(SaddleError::Invalid(
                "max move must be finite and positive".into(),
            ));
        }
        let chain = Chain::new(spec.x.clone())?;
        let n = spec.x.nrows();
        let dof = spec.x.ncols();
        if n < 2 {
            return Err(SaddleError::Shape(
                "a band needs at least two images".into(),
            ));
        }
        let projection = if kind == Kind::Dneb {
            ProjectionKind::DoublyNudged
        } else {
            ProjectionKind::Neb
        };
        let climb = matches!(kind, Kind::Ci | Kind::Weighted | Kind::Zoom | Kind::Auto);
        let zoom = match kind {
            Kind::Zoom => Some(ZoomWatch::new(spec.zoom.unwrap_or_default())),
            _ => None,
        };
        if kind == Kind::Zoom && !climb {
            return Err(SaddleError::Invalid(
                "zoom-NEB needs a climbing image".into(),
            ));
        }
        let free = (n.saturating_sub(2)) * dof;
        let solver_n = free.max(dof);
        let solver = force_solver(spec.method.clone(), solver_n, spec.max_move)?;
        let flat_cache = spec.x.iter().copied().collect();
        let idpp = if kind == Kind::Idpp {
            Some(IdppRun::new(
                spec.x.clone(),
                spec.k,
                spec.force_tol,
                spec.max_move,
                spec.method.clone(),
            )?)
        } else {
            None
        };
        let (auto, book) = if kind == Kind::Auto {
            let mut band = band_config(&spec, projection, climb);
            band.autoneb = Some(AutoNebTuning {
                n_simul: spec.n_simul,
                n_max: spec.n_max,
                threads: spec.threads,
                ..AutoNebTuning::default()
            });
            if let Some((k_min, k_max)) = spec.energy_weighted {
                band.spring = crate::spring::SpringKind::EnergyWeighted { k_min, k_max };
            }
            let book = IdBook::new(n);
            let search = {
                let _guard = bind_for_step(book.clone());
                AutoNeb::new(band, spec.x)?
            };
            (Some(search), Some(book))
        } else {
            (None, None)
        };
        Ok(Self {
            inner: Arc::new(Mutex::new(Inner {
                chain,
                free_ends: false,
                armed: false,
                baseline: None,
                ci: None,
                total_rows: 0,
                last_projected: None,
                last_max: f64::NAN,
                error: None,
            })),
            flat_cache,
            pool: Arc::new(EvalPool::new(spec.threads)?),
            projection,
            climb,
            tangent: spec.tangent,
            k: spec.k,
            energy_weighted: spec.energy_weighted,
            force_tol: spec.force_tol,
            max_move: spec.max_move,
            method: spec.method,
            trigger_factor: spec.trigger_factor,
            trigger_force: spec.trigger_force,
            band_slack: spec.band_slack,
            image_scale: spec.image_scale,
            solver,
            solver_n,
            iteration: 0,
            zoom,
            zoom_due: false,
            zoomed: false,
            zoom_window: None,
            post_steps: 0,
            auto,
            book,
            idpp,
        })
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|err| err.into_inner())
    }

    pub fn coordinate_slice(&self) -> &[f64] {
        self.flat_cache.as_slice().unwrap_or(&[])
    }

    pub fn positions(&self) -> Array2<f64> {
        self.lock().chain.positions().to_owned()
    }

    pub fn energies(&self) -> Option<Vec<f64>> {
        let inner = self.lock();
        if inner.chain.filled() {
            return Some(inner.chain.energies().to_vec());
        }
        drop(inner);
        let search = self.auto.as_ref()?;
        let known: Vec<f64> = search
            .energies()
            .iter()
            .filter_map(|value| *value)
            .collect();
        if known.len() == search.energies().len() && !known.is_empty() {
            Some(known)
        } else {
            None
        }
    }

    pub fn image_ids(&self) -> Vec<u64> {
        if let Some(book) = &self.book {
            return book
                .ids
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .clone();
        }
        self.lock().chain.ids().to_vec()
    }

    pub fn climbing_index(&self) -> Option<usize> {
        self.lock().ci
    }

    pub fn zoom_window(&self) -> Option<(usize, usize)> {
        self.zoom_window
    }

    pub fn step<S: Surface>(&mut self, surface: &S) -> Result<StencilReport, SaddleError> {
        if self.idpp.is_some() {
            let _ = surface;
            return self.step_idpp();
        }
        if self.auto.is_some() {
            return self.step_auto(surface);
        }
        if self.zoom_due {
            return self.pack();
        }
        if self.zoomed
            && self
                .zoom
                .as_ref()
                .is_some_and(|watch| watch.params().max_iterations > 0)
            && self.post_steps
                >= self
                    .zoom
                    .as_ref()
                    .map(|watch| watch.params().max_iterations)
                    .unwrap_or(0)
        {
            return Ok(self.report(StencilStatus::Running, f64::INFINITY, 0, PHASE_BAND));
        }
        let report = self.optimizer_step(surface)?;
        if self.zoomed {
            self.post_steps += 1;
        }
        let (n_images, ci_now) = {
            let inner = self.lock();
            (inner.chain.positions().nrows(), inner.ci)
        };
        if let Some(watch) = self.zoom.as_mut()
            && !self.zoomed
            && watch.observe(
                n_images,
                self.climb && ci_now.is_some(),
                ci_now,
                report.max_force,
                self.force_tol,
            )
        {
            self.zoom_due = true;
        }
        Ok(report)
    }

    fn step_idpp(&mut self) -> Result<StencilReport, SaddleError> {
        let report = self.idpp.as_mut().expect("idpp").step()?;
        let positions = self.idpp.as_ref().expect("idpp").positions();
        {
            let mut inner = self.lock();
            let ids = inner.chain.ids().to_vec();
            let next = inner.chain.next_id();
            inner.chain.replace(positions, ids, next)?;
        }
        self.remember_flat();
        self.iteration = report.iteration;
        Ok(report)
    }

    fn step_auto<S: Surface>(&mut self, surface: &S) -> Result<StencilReport, SaddleError> {
        let book = self
            .book
            .clone()
            .ok_or_else(|| SaddleError::Invalid("AutoNEB has no image ids".into()))?;
        let _guard = bind_for_step(book);
        let before = self
            .auto
            .as_ref()
            .map(|search| search.insertions().len())
            .unwrap_or(0);
        let before_rows = self
            .auto
            .as_ref()
            .map(|search| search.surface_rows())
            .unwrap_or(0);
        let search = self.auto.as_mut().expect("autoneb");
        let report = search.step(surface)?;
        let inserted = search.insertions().len() > before;
        let positions = search.positions().to_owned();
        let ids = self
            .book
            .as_ref()
            .expect("book")
            .ids
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clone();
        let next = self
            .book
            .as_ref()
            .expect("book")
            .next
            .load(std::sync::atomic::Ordering::Relaxed);
        {
            let mut inner = self.lock();
            inner.chain.replace(positions, ids, next)?;
            inner.ci = report.ci_index;
        }
        self.remember_flat();
        self.iteration = report.iteration;
        let delta = report.surface_rows.saturating_sub(before_rows);
        {
            let mut inner = self.lock();
            inner.total_rows = report.surface_rows;
        }
        let phase = if inserted { PHASE_INSERT } else { PHASE_BAND };
        let ci = self.lock().ci;
        Ok(StencilReport {
            status: match report.status {
                BandStatus::Converged => StencilStatus::Converged,
                BandStatus::Running => StencilStatus::Running,
            },
            max_force: report.max_force,
            curvature: f64::NAN,
            iteration: self.iteration,
            evaluations: delta,
            total_evaluations: report.surface_rows,
            phase,
            ci_index: ci,
            zoom_window: None,
        })
    }

    fn remember_flat(&mut self) {
        let coords = self.lock().chain.coordinate_slice().to_vec();
        self.flat_cache = Array1::from_vec(coords);
    }

    fn optimizer_step<S: Surface>(&mut self, surface: &S) -> Result<StencilReport, SaddleError> {
        let (n, dof, free_ends) = {
            let inner = self.lock();
            (
                inner.chain.positions().nrows(),
                inner.chain.positions().ncols(),
                inner.free_ends,
            )
        };
        let width = if free_ends { n * dof } else { (n - 2) * dof };
        if self.solver_n != width {
            self.solver = force_solver(self.method.clone(), width, self.max_move)?;
            self.solver_n = width;
        }
        let shared = Arc::clone(&self.inner);
        let pool = Arc::clone(&self.pool);
        let law = ForceLaw {
            projection: self.projection,
            climb: self.climb,
            tangent: self.tangent,
            k: self.k,
            energy_weighted: self.energy_weighted,
            trigger_factor: self.trigger_factor,
            trigger_force: self.trigger_force,
        };
        let before = self.lock().total_rows;
        let oracle = Oracle::unbounded(width, move |flat: ArrayView1<f64>| {
            let mut inner = shared.lock().unwrap_or_else(|err| err.into_inner());
            match Self::assemble(&mut inner, pool.as_ref(), surface, flat, law) {
                Ok((energy, force, _)) => (energy, -force),
                Err(err) => {
                    inner.error = Some(err);
                    (f64::INFINITY, Array1::zeros(width))
                }
            }
        });
        let mut flat = {
            let inner = self.lock();
            if inner.free_ends {
                inner.chain.all_flat()
            } else {
                inner.chain.interior_flat()
            }
        };
        let stepped = self.solver.step(&oracle, &mut flat);
        drop(oracle);
        if let Some(err) = self.lock().error.take() {
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        {
            let shared = Arc::clone(&self.inner);
            let pool = Arc::clone(&self.pool);
            let mut inner = shared.lock().unwrap_or_else(|err| err.into_inner());
            Self::assemble(&mut inner, pool.as_ref(), surface, flat.view(), law)?;
        }
        self.remember_flat();
        self.iteration += 1;
        let judged = self.judge();
        let delta = self.lock().total_rows.saturating_sub(before);
        Ok(self.report(judged.0, judged.1, delta, PHASE_BAND))
    }
    fn assemble<S: Surface>(
        inner: &mut Inner,
        pool: &EvalPool,
        surface: &S,
        flat: ArrayView1<f64>,
        law: ForceLaw,
    ) -> Result<(f64, Array1<f64>, f64), SaddleError> {
        if inner.free_ends {
            inner.chain.write_all(flat)?;
        } else {
            inner.chain.write_interior(flat)?;
        }
        inner.chain.evaluate(surface, pool)?;
        let rows = inner.chain.positions().nrows();
        inner.total_rows += rows;
        let positions = inner.chain.positions().to_owned();
        let energies = inner.chain.energies().to_vec();
        let gradients = inner.chain.gradients().to_owned();
        if energies.iter().any(|value| !value.is_finite())
            || gradients.iter().any(|value| !value.is_finite())
        {
            return Err(SaddleError::NonFinite("band"));
        }
        let n = positions.nrows();
        let dof = positions.ncols();
        if n < 3 {
            return Err(SaddleError::Shape("a band step needs three images".into()));
        }
        let mut max_i = 1usize;
        for i in 2..n - 1 {
            if energies[i] > energies[max_i] {
                max_i = i;
            }
        }
        let ceiling = energies[0].max(energies[n - 1]);
        let climb_target = (energies[max_i] > ceiling).then_some(max_i);
        let ci_at = if inner.armed { climb_target } else { None };
        if inner.armed {
            inner.ci = climb_target;
        }
        let mut ks = vec![law.k; n - 1];
        if let Some((k_min, k_max)) = law.energy_weighted {
            fill_constants(&energies, k_min, k_max, &mut ks);
        }
        let moving = if inner.free_ends { n } else { n - 2 };
        let mut projected = Array1::zeros(moving * dof);
        if inner.free_ends {
            let ends = [
                free_end_force(positions.row(0), positions.row(1), gradients.row(0)),
                free_end_force(
                    positions.row(n - 1),
                    positions.row(n - 2),
                    gradients.row(n - 1),
                ),
            ];
            projected.slice_mut(s![0..dof]).assign(&ends[0]);
            projected
                .slice_mut(s![(n - 1) * dof..n * dof])
                .assign(&ends[1]);
        }
        for i in 1..n - 1 {
            let force = image_force(
                &positions,
                &energies,
                &gradients,
                &ks,
                i,
                ci_at == Some(i),
                law,
            )?;
            let row = if inner.free_ends { i } else { i - 1 };
            projected
                .slice_mut(s![row * dof..(row + 1) * dof])
                .assign(&force);
        }
        let mut max_component = max_abs(&projected);
        if inner.baseline.is_none() {
            inner.baseline = Some(max_component);
        }
        let baseline = inner.baseline.unwrap_or(max_component);
        if law.climb
            && !inner.armed
            && (max_component < baseline * law.trigger_factor || max_component < law.trigger_force)
        {
            inner.armed = true;
            if let Some(ci_i) = climb_target {
                inner.ci = Some(ci_i);
                let force = image_force(&positions, &energies, &gradients, &ks, ci_i, true, law)?;
                let row = if inner.free_ends { ci_i } else { ci_i - 1 };
                projected
                    .slice_mut(s![row * dof..(row + 1) * dof])
                    .assign(&force);
                max_component = max_abs(&projected);
            }
        }
        inner.last_projected = Some(projected.clone());
        inner.last_max = max_component;
        let pseudo = if inner.free_ends {
            energies.iter().sum()
        } else {
            energies[1..n - 1].iter().sum()
        };
        Ok((pseudo, projected, max_component))
    }

    fn judge(&self) -> (StencilStatus, f64) {
        let inner = self.lock();
        let Some(projected) = inner.last_projected.clone() else {
            return (StencilStatus::Running, inner.last_max);
        };
        let dof = inner.chain.positions().ncols();
        let free_ends = inner.free_ends;
        let ci = inner.ci;
        let last_max = inner.last_max;
        drop(inner);
        if dof == 0 || !projected.len().is_multiple_of(dof) {
            return (StencilStatus::Running, last_max);
        }
        let rows = projected.len() / dof;
        let ci_row = ci.and_then(|ci| {
            if free_ends {
                (ci < rows).then_some(ci)
            } else {
                ci.checked_sub(1).filter(|row| *row < rows)
            }
        });
        let mut worst = 0.0;
        let mut ci_force = last_max;
        for row in 0..rows {
            let force = projected.slice(s![row * dof..(row + 1) * dof]);
            let linf = max_abs_view(force);
            if linf > worst {
                worst = linf;
            }
            if Some(row) == ci_row {
                ci_force = linf;
            }
        }
        let climbing = self.climb && ci.is_some();
        let converged = if !climbing {
            worst <= self.force_tol
        } else {
            let mut ok = ci_force <= self.force_tol;
            if let Some(scale) = self.image_scale {
                ok &= (0..rows).all(|row| {
                    Some(row) == ci_row
                        || max_abs_view(projected.slice(s![row * dof..(row + 1) * dof]))
                            <= scale * self.force_tol
                });
            } else {
                ok &= worst <= self.band_slack * self.force_tol;
            }
            ok
        };
        let status = if converged {
            StencilStatus::Converged
        } else {
            StencilStatus::Running
        };
        let reported = if climbing { ci_force } else { worst };
        (status, reported)
    }

    fn pack(&mut self) -> Result<StencilReport, SaddleError> {
        let params = self
            .zoom
            .as_ref()
            .map(ZoomWatch::params)
            .ok_or_else(|| SaddleError::Invalid("zoom pack without zoom parameters".into()))?;
        let (filled, ci, energies, count, old, next, positions) = {
            let inner = self.lock();
            (
                inner.chain.filled(),
                inner.ci,
                inner.chain.energies().to_vec(),
                inner.chain.positions().nrows(),
                inner.chain.ids().to_vec(),
                inner.chain.next_id(),
                inner.chain.positions().to_owned(),
            )
        };
        if !filled {
            return Err(SaddleError::Invalid(
                "zoom pack needs the energies of the last evaluation".into(),
            ));
        }
        let ci =
            ci.ok_or_else(|| SaddleError::Invalid("zoom pack needs a climbing image".into()))?;
        let Some((lo, hi)) = select_window(&energies, ci, params.window) else {
            self.zoom_due = false;
            self.zoomed = true;
            self.iteration += 1;
            return Ok(self.report(StencilStatus::Running, 0.0, 0, PHASE_ZOOM));
        };
        let packed = pack_window(positions.view(), lo, hi, count, params.interpolation, None);
        let mut ids = Vec::with_capacity(count);
        let mut next = next;
        ids.push(old[lo]);
        for _ in 1..count.saturating_sub(1) {
            ids.push(next);
            next += 1;
        }
        if count > 1 {
            ids.push(old[hi]);
        }
        {
            let mut inner = self.lock();
            inner.chain.replace(packed, ids, next)?;
            inner.free_ends = params.ends == ZoomEnds::FreePerp;
            inner.armed = false;
            inner.baseline = None;
            inner.ci = None;
            inner.last_projected = None;
        }
        let free = params.ends == ZoomEnds::FreePerp;
        let dof = positions.ncols();
        let width = if free { count * dof } else { (count - 2) * dof };
        self.solver = force_solver(self.method.clone(), width, self.max_move)?;
        self.solver_n = width;
        self.remember_flat();
        self.zoom_due = false;
        self.zoomed = true;
        self.zoom_window = Some((lo, hi));
        self.post_steps = 0;
        self.iteration += 1;
        Ok(self.report(StencilStatus::Running, 0.0, 0, PHASE_ZOOM))
    }

    fn report(
        &self,
        status: StencilStatus,
        max_force: f64,
        evaluations: usize,
        phase: i32,
    ) -> StencilReport {
        StencilReport {
            status,
            max_force,
            curvature: f64::NAN,
            iteration: self.iteration,
            evaluations,
            total_evaluations: {
                let inner = self.lock();
                inner.total_rows
            },
            phase,
            ci_index: {
                let inner = self.lock();
                inner.ci
            },
            zoom_window: if phase == PHASE_ZOOM {
                self.zoom_window
            } else {
                None
            },
        }
    }
}

fn kind_of(preset: Preset) -> Result<Kind, SaddleError> {
    Ok(match preset {
        Preset::Neb => Kind::Neb,
        Preset::Dneb => Kind::Dneb,
        Preset::CiNeb => Kind::Ci,
        Preset::EnergyWeightedNeb => Kind::Weighted,
        Preset::ZoomNeb => Kind::Zoom,
        Preset::AutoNeb => Kind::Auto,
        Preset::Idpp => Kind::Idpp,
        Preset::OciNeb => return Err(SaddleError::Invalid(UNSUPPORTED_OCI.into())),
        Preset::SIdpp => return Err(SaddleError::Invalid(UNSUPPORTED_SIDPP.into())),
        other => {
            return Err(SaddleError::Invalid(format!(
                "{} is a point preset",
                other.name()
            )));
        }
    })
}

fn check_bounds(k_min: f64, k_max: f64) -> Result<(), SaddleError> {
    if !(k_min.is_finite() && k_max.is_finite() && k_min > 0.0 && k_max > 0.0) {
        return Err(SaddleError::Invalid(
            "energy-weighted spring bounds must be finite and positive".into(),
        ));
    }
    if k_min > k_max {
        return Err(SaddleError::Invalid(
            "energy-weighted springs need k_min at most k_max".into(),
        ));
    }
    Ok(())
}

fn band_config(spec: &BandSpec, projection: ProjectionKind, climb: bool) -> BandConfig {
    let tangent = match spec.tangent {
        TangentRule::Simple => crate::tangent::TangentKind::Simple,
        TangentRule::Improved => crate::tangent::TangentKind::Improved,
    };
    BandConfig {
        tangent,
        spring: crate::spring::SpringKind::Uniform { k: spec.k },
        projection,
        climbing: climb.then_some(CiConfig {
            trigger_factor: spec.trigger_factor,
            trigger_force: spec.trigger_force,
        }),
        force_tol: spec.force_tol,
        max_move: spec.max_move,
        method: spec.method.clone(),
        remove_translation: true,
        autoneb: None,
        ..BandConfig::default()
    }
}

fn force_solver(method: Method, n: usize, max_move: f64) -> Result<Solver, SaddleError> {
    crate::band::check_force_driven(&method)?;
    if n == 0 {
        return Err(SaddleError::Shape(
            "a band solver needs a degree of freedom".into(),
        ));
    }
    let control = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: 1.0,
        maxmove: None,
        ftol_rel: None,
    };
    let lbfgs = matches!(method, Method::Lbfgs { .. });
    let mut solver = Solver::new(method, control, n);
    solver.set_atom_maxmove(max_move);
    if lbfgs {
        solver.set_accept(Accept::Step);
    }
    Ok(solver)
}

fn image_force(
    positions: &Array2<f64>,
    energies: &[f64],
    gradients: &Array2<f64>,
    ks: &[f64],
    i: usize,
    climb: bool,
    law: ForceLaw,
) -> Result<Array1<f64>, SaddleError> {
    let dof = positions.ncols();
    let pos_prev = positions.row(i - 1);
    let pos = positions.row(i);
    let pos_next = positions.row(i + 1);
    let pos_diff_next = &pos_next - &pos;
    let pos_diff_prev = &pos - &pos_prev;
    let dist_next = pos_diff_next.dot(&pos_diff_next).sqrt();
    let dist_prev = pos_diff_prev.dot(&pos_diff_prev).sqrt();
    let tangent = unit_tangent(
        law.tangent,
        pos_diff_next.view(),
        pos_diff_prev.view(),
        energies[i],
        energies[i - 1],
        energies[i + 1],
    )
    .map_err(|err| SaddleError::Invalid(format!("image {i}: {err}")))?;
    let force = -&gradients.row(i);
    let coupling = if law.energy_weighted.is_some() {
        Coupling::Weighted {
            k_next: ks[i],
            k_prev: ks[i - 1],
        }
    } else {
        Coupling::Uniform { k: law.k }
    };
    let coupled = coupling_forces(
        &coupling,
        tangent.view(),
        dist_next,
        dist_prev,
        pos_diff_next.view(),
        pos_diff_prev.view(),
    );
    let spring = SpringForces {
        parallel: coupled.parallel,
        full: coupled.full,
    };
    let mut projected = if climb {
        let dneb = if law.projection == ProjectionKind::DoublyNudged {
            let fp = perp(force.view(), tangent.view());
            dneb_component(spring.full.view(), tangent.view(), fp.view())
        } else {
            Array1::zeros(dof)
        };
        climbing_image_force(force.view(), tangent.view(), dneb.view())
    } else {
        law.projection
            .project(force.view(), tangent.view(), &spring)
    };
    crate::band::remove_net_translation(&mut projected);
    Ok(projected)
}

fn free_end_force(
    pos: ArrayView1<f64>,
    neighbour: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
) -> Array1<f64> {
    let diff = &neighbour - &pos;
    let norm = diff.dot(&diff).sqrt();
    let tangent = if norm > 1e-15 {
        &diff / norm
    } else {
        Array1::zeros(diff.len())
    };
    let force = -&gradient.to_owned();
    let mut projected = perp(force.view(), tangent.view());
    crate::band::remove_net_translation(&mut projected);
    projected
}

fn max_abs(values: &Array1<f64>) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()))
}

fn max_abs_view(values: ArrayView1<f64>) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |max, value| max.max(value.abs()))
}

pub fn default_k_bounds() -> (f64, f64) {
    (K_MIN_EV_PER_A2, K_MAX_EV_PER_A2)
}
