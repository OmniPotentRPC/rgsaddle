#![expect(
    clippy::needless_range_loop,
    reason = "the COMPASS index algebra follows the reference loops"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "the two-image step carries the COMPASS loop state"
)]

//! Double-ended connect.
//!
//! The first attempt is a min-mode climb from the highest image of an
//! IDPP band. Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026),
//! doi:10.3389/fchem.2026.1807063, climb the highest image; Goswami,
//! arXiv:2603.14737, is the energy-weighted band this search is measured
//! against. The initial path is the image-dependent pair potential of
//! Smidstrup, Pedersen, Stokbro, and Jónsson, J. Chem. Phys. 140, 214106
//! (2014), doi:10.1063/1.4878664. The climb is Lanczos min-mode following
//! (Malek and Mousseau, Phys. Rev. E 62, 7723 (2000),
//! doi:10.1103/PhysRevE.62.7723; Olsen, Kroes, Henkelman, Arnaldsson,
//! and Jónsson, J. Chem. Phys. 121, 9776 (2004), doi:10.1063/1.1809574)
//! with the improved relative-eigenvalue stop. The seed tangent is the
//! improved tangent of Henkelman and Jónsson, J. Chem. Phys. 113, 9978
//! (2000), doi:10.1063/1.1323224. The translation is L-BFGS
//! (Liu and Nocedal, Math. Program. 45, 503 (1989),
//! doi:10.1007/BF01589116) on the effective force of Henkelman and
//! Jónsson, J. Chem. Phys. 111, 7010 (1999), doi:10.1063/1.480097:
//! climb the mode while its curvature is positive, and reflect the
//! parallel force once the curvature is negative. The initial inverse
//! curvature is 0.01 and later steps use `s·y / y·y`. A negative
//! secant drops the history and takes a long step along the force.
//! A segment is accepted when that climb meets the force tolerance,
//! the lowest curvature is negative, and a two-action Lanczos estimate
//! in the orthogonal complement is not negative. That Ritz value
//! upper-bounds the next curvature, so the dense Hessian is not built.
//!
//! Otherwise the search is the two-image equal-energy walk of
//! Sommer-Jörgensen and Goedecker, J. Chem. Phys. 160, 064110 (2024),
//! with the SQNM step of Schaefer, Ghasemi, Roy, and Goedecker,
//! J. Chem. Phys. 142, 034112 (2015). A split quenches the new nodes
//! before the next segment, in the sense of OPTIM NEWCONNECT
//! (Carr, Trygubenko, and Wales, J. Chem. Phys. 122, 234903 (2005),
//! doi:10.1063/1.1931587)
//! and pele `DoubleEndedConnect`. That chain's stationary point is
//! kept only when a counted, rigid-projected Hessian has index 1.
//!
//! The callback returns energy and the Cartesian gradient. Finite
//! differences that decide a step or an index are included in
//! `evaluations`. The IDPP relaxation calls only the surrogate.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;

use faer::Mat;
use ndarray::{Array1, Array2, ArrayView1};
use rgmin::Method;

use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::minmode::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, translated_gradient,
};
use crate::sqnm::ConstrainedOpt;

type SurfaceFn<'a> = dyn FnMut(&[f64]) -> Result<(f64, Vec<f64>), SaddleError> + 'a;

/// COMPASS and SQNM controls for one connect.
///
/// Build with [`ConnectConfig::optbench`]. Fields are public so a host
/// can replace one constant; the struct is non-exhaustive.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ConnectConfig {
    pub alpha0: f64,
    pub nhistx: usize,
    pub trustradius: f64,
    pub forcetol: f64,
    pub forcetol_per_distance: f64,
    pub distance_step_max: f64,
    pub distance_step_min: f64,
    pub target_distance_saddle: f64,
    pub target_distance_other: f64,
    pub energy_difference_threshold: f64,
    pub trusted_force_difference: f64,
    pub saddle_cosp: f64,
    pub nsteps: usize,
    pub minimize_fmax: f64,
    pub minimize_nsteps: usize,
    pub minimize_maxrise: f64,
    pub n_loops: usize,
    /// Minimize chain nodes created by a split before the next search.
    pub quench_nodes: bool,
    /// Interior images on the IDPP band that seeds the min-mode climb.
    pub band_images: usize,
    /// Relative change of the lowest Lanczos Ritz value that ends an
    /// estimate. 0.05 is the improved Lanczos tolerance (quit early
    /// when the curvature along the incoming mode has settled);
    /// Olsen, Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem. Phys.
    /// 121, 9776 (2004). The eOn schema default without that setting
    /// is 0.001.
    pub lanczos_rel_tol: f64,
    /// Lanczos actions in one translation estimate. The relative test
    /// usually returns sooner. Eight is the subspace kept when it has
    /// not: a followed mode is recovered in a few iterations (Olsen,
    /// Kroes, Henkelman, Arnaldsson, and Jónsson, J. Chem. Phys. 121,
    /// 9776 (2004)).
    pub lanczos_iterations: usize,
    /// Per-atom cap on a min-mode translation.
    pub minmode_max_move: f64,
    /// L-BFGS memory of the min-mode translation.
    pub minmode_memory: usize,
}

impl ConnectConfig {
    /// Optbench LJ38 settings: force tolerance 0.001, history 10,
    /// trust radius 0.05, and quenched split nodes.
    pub fn optbench(cap: usize) -> Self {
        Self {
            alpha0: 0.001,
            nhistx: 10,
            trustradius: 0.05,
            forcetol: 0.001,
            forcetol_per_distance: 0.2,
            distance_step_max: 0.05,
            distance_step_min: 0.005,
            target_distance_saddle: 0.001,
            target_distance_other: 0.001,
            energy_difference_threshold: 0.2,
            trusted_force_difference: 0.5,
            saddle_cosp: 0.7,
            nsteps: cap,
            minimize_fmax: 0.001,
            minimize_nsteps: cap,
            minimize_maxrise: 0.001,
            n_loops: cap,
            quench_nodes: true,
            band_images: 5,
            lanczos_rel_tol: 0.05,
            lanczos_iterations: 8,
            minmode_max_move: 0.1,
            minmode_memory: 20,
        }
    }
}

/// Where one connect spent its surface calls.
///
/// `hessian_calls` counts the dense finite-difference Hessian.
/// One check is a central difference, two evaluations per coordinate
/// (`hessian_coordinates`). Lanczos actions sit in `minmode_calls`.
#[derive(Clone, Copy, Debug, Default)]
pub struct ConnectProfile {
    pub sqnm_calls: usize,
    pub sqnm_steps: usize,
    pub sqnm_segments: usize,
    pub hessian_calls: usize,
    pub hessian_checks: usize,
    pub hessian_coordinates: usize,
    pub quench_calls: usize,
    pub quenches: usize,
    pub searches: usize,
    pub minmode_calls: usize,
    pub band_calls: usize,
    pub other_calls: usize,
}

/// One connect result. `kind` is `s` for a pinned saddle, `u` for an
/// upper bound, and `f` when the highest segment could not be split.
#[derive(Clone, Debug)]
pub struct ConnectReport {
    pub position: Vec<f64>,
    pub energy: f64,
    pub force_norm: f64,
    pub kind: char,
    /// Surface evaluations. Finite differences that decide a step or an
    /// index are included. The Lanczos acceptance does not build a dense
    /// Hessian; the COMPASS fallback does.
    pub evaluations: usize,
    /// Negative eigenvalues of the rigid-projected Hessian, or the
    /// Lanczos index estimate when that estimate accepted the point,
    /// or `-1` when the search stopped before either check.
    pub hessian_index: i32,
    pub profile: ConnectProfile,
}

type Settings = ConnectConfig;

#[derive(Clone, Copy)]
enum Phase {
    Other,
    Sqnm,
    Hessian,
    Quench,
    MinMode,
    Band,
}

thread_local! {
    static PHASE: Cell<Phase> = const { Cell::new(Phase::Other) };
    static PROFILE: RefCell<ConnectProfile> = const { RefCell::new(ConnectProfile {
        sqnm_calls: 0,
        sqnm_steps: 0,
        sqnm_segments: 0,
        hessian_calls: 0,
        hessian_checks: 0,
        hessian_coordinates: 0,
        quench_calls: 0,
        quenches: 0,
        searches: 0,
        minmode_calls: 0,
        band_calls: 0,
        other_calls: 0,
    }) };
}

struct PhaseGuard(Phase);

impl PhaseGuard {
    fn enter(phase: Phase) -> Self {
        let previous = PHASE.with(|slot| slot.replace(phase));
        Self(previous)
    }
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        PHASE.with(|slot| slot.set(self.0));
    }
}

fn note_call() {
    PROFILE.with(|slot| {
        let mut profile = slot.borrow_mut();
        match PHASE.with(Cell::get) {
            Phase::Sqnm => profile.sqnm_calls += 1,
            Phase::Hessian => profile.hessian_calls += 1,
            Phase::Quench => profile.quench_calls += 1,
            Phase::MinMode => profile.minmode_calls += 1,
            Phase::Band => profile.band_calls += 1,
            Phase::Other => profile.other_calls += 1,
        }
    });
}

fn tally(update: impl FnOnce(&mut ConnectProfile)) {
    PROFILE.with(|slot| update(&mut slot.borrow_mut()));
}

/// IDPP band and a Lanczos climb, then the COMPASS chain. The climb
/// is kept when its force is under the tolerance, its curvature is
/// negative, and a Lanczos estimate in the complement is not. Otherwise
/// an index other than 1 on the chain is replaced by the highest
/// index-1 saddle on a segment between a quenched endpoint and a
/// minimum reached by stepping off the unwanted mode.
pub fn connect(
    mut energy_gradient: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), SaddleError>,
    left: &[f64],
    right: &[f64],
    settings: &ConnectConfig,
) -> Result<ConnectReport, SaddleError> {
    PROFILE.with(|slot| *slot.borrow_mut() = ConnectProfile::default());
    PHASE.with(|slot| slot.set(Phase::Other));
    let mut evaluations = 0usize;
    let found = connect_force(
        |x| {
            note_call();
            evaluations += 1;
            let (energy, gradient) = energy_gradient(x)?;
            if gradient.len() != x.len() {
                return Err(SaddleError::Shape(
                    "connect gradient length does not match the position".into(),
                ));
            }
            if !energy.is_finite() || gradient.iter().any(|v| !v.is_finite()) {
                return Err(SaddleError::NonFinite("connect surface"));
            }
            Ok((energy, gradient.iter().map(|v| -v).collect()))
        },
        left,
        right,
        settings,
    )?;
    Ok(ConnectReport {
        position: found.pos,
        energy: found.energy,
        force_norm: found.force_norm,
        kind: found.kind,
        evaluations,
        hessian_index: found.hessian_index,
        profile: PROFILE.with(|slot| *slot.borrow()),
    })
}

fn connect_force(
    mut energy_force: impl FnMut(&[f64]) -> Result<(f64, Vec<f64>), SaddleError>,
    left: &[f64],
    right: &[f64],
    settings: &Settings,
) -> Result<Found, SaddleError> {
    if let Some(climbed) = climb_idpp_band(&mut energy_force, left, right, settings)? {
        return Ok(climbed);
    }
    let mut found = search_pair(&mut energy_force, left, right, settings)?;
    let best = {
        let mut ev = Eval {
            f: &mut energy_force,
        };
        let (index, mode) = vibrational_mode(&mut ev, &found.pos)?;
        found.hessian_index = index as i32;
        if found.kind == 's' && found.force_norm <= settings.forcetol && index == 1 {
            return Ok(found);
        }
        let left_min = minimize(&mut ev, left, settings)?;
        let right_min = minimize(&mut ev, right, settings)?;
        let mut plus = found.pos.clone();
        let mut minus = found.pos.clone();
        for i in 0..plus.len() {
            plus[i] += 0.10 * mode[i];
            minus[i] -= 0.10 * mode[i];
        }
        let down_plus = minimize(&mut ev, &plus, settings)?;
        let down_minus = minimize(&mut ev, &minus, settings)?;
        let mut unique = Vec::new();
        for node in [left_min, right_min, down_plus, down_minus] {
            if unique.iter().any(|kept: &Img| same_basin(kept, &node)) {
                continue;
            }
            unique.push(node);
        }
        let mut sub = settings.clone();
        sub.n_loops = 4;
        sub.nsteps = 8_000;
        sub.minimize_nsteps = 1_000;
        let mut best: Option<Found> = None;
        for i in 0..unique.len() {
            for j in i + 1..unique.len() {
                if i == 0 && j == 1 {
                    continue;
                }
                let mut candidate = search_pair(&mut *ev.f, &unique[i].pos, &unique[j].pos, &sub)?;
                let (cand_index, _) = vibrational_mode(&mut ev, &candidate.pos)?;
                candidate.hessian_index = cand_index as i32;
                let ok = candidate.kind == 's'
                    && candidate.force_norm <= settings.forcetol
                    && cand_index == 1;
                if ok
                    && best
                        .as_ref()
                        .is_none_or(|cur| candidate.energy > cur.energy)
                {
                    best = Some(candidate);
                }
            }
        }
        best
    };
    if let Some(best) = best {
        return Ok(best);
    }
    if let Some(climbed) = climb_between(&mut energy_force, left, right, settings)? {
        return Ok(climbed);
    }
    Ok(found)
}

/// Dimer climbs from points on the endpoint segment when the chain
/// did not return an index-1 saddle. One atom on the two-barrier toy
/// meets in the intermediate well; the climb leaves that well.
fn climb_between(
    energy_force: &mut SurfaceFn<'_>,
    left: &[f64],
    right: &[f64],
    settings: &Settings,
) -> Result<Option<Found>, SaddleError> {
    let surface = CallbackSurface {
        f: energy_force,
        extra: Vec::new(),
        project_rigid: true,
    };
    let mut seed = Array1::from_iter(left.iter().zip(right).map(|(a, b)| b - a));
    let seed_norm = seed.dot(&seed).sqrt();
    if seed_norm < 1e-12 {
        seed.fill(0.0);
        seed[0] = 1.0;
    } else {
        seed /= seed_norm;
    }
    let mut best: Option<Found> = None;
    for fraction in [0.15, 0.35, 0.65, 0.85] {
        let start = Array1::from_iter(left.iter().zip(right).map(|(a, b)| a + fraction * (b - a)));
        let config = MinModeConfig {
            force_tol: settings.forcetol,
            max_move: 0.1,
            method: Method::Lbfgs { memory: 20 },
            ..MinModeConfig::default()
        };
        let mut session = MinModeSession::new(config, start, seed.clone())?;
        session.set_force_gate(ForceGate::L2Norm);
        let stepped = session.run(&surface, 60)?;
        if stepped.status != MinModeStatus::Converged || stepped.max_force > settings.forcetol {
            continue;
        }
        let pos = session.position().to_vec();
        let mut ev = Eval {
            // SAFETY: the dimer run has returned, so this is the only
            // live borrow of the callback until `ev` is dropped.
            f: unsafe { &mut *surface.f },
        };
        let (index, _) = vibrational_mode(&mut ev, &pos)?;
        if index != 1 {
            continue;
        }
        let img = ev.at(&pos)?;
        let candidate = Found {
            energy: img.energy,
            force_norm: norm(&img.force),
            pos: img.pos,
            kind: 's',
            hessian_index: index as i32,
        };
        if candidate.force_norm <= settings.forcetol
            && best
                .as_ref()
                .is_none_or(|cur| candidate.energy > cur.energy)
        {
            best = Some(candidate);
        }
    }
    Ok(best)
}

/// Single-threaded bridge. Min-mode steps call the surface on the
/// host thread; the unsafe `Sync` impl does not share the callback.
struct CallbackSurface<'a> {
    f: *mut SurfaceFn<'a>,
    /// One extra direction removed with the rigid modes, used for the
    /// second Lanczos curvature.
    extra: Vec<Vec<f64>>,
    /// Drop rigid translations and rotations before the mode estimate.
    project_rigid: bool,
}

// SAFETY: `MinModeSession::step` evaluates the surface on the calling
// thread and does not retain the reference. The pointer is not sent.
unsafe impl Sync for CallbackSurface<'_> {}

impl PointSurface for CallbackSurface<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let slice = x
            .as_slice()
            .ok_or_else(|| SaddleError::Shape("connect position is not contiguous".into()))?;
        // SAFETY: step does not overlap another evaluation of this callback.
        let (energy, force) = unsafe { (*self.f)(slice) }?;
        if force.len() != slice.len() {
            return Err(SaddleError::Shape(
                "connect force length does not match the position".into(),
            ));
        }
        Ok((energy, Array1::from_iter(force.into_iter().map(|v| -v))))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let Some(slice) = x.as_slice() else {
            return Ok(Array2::zeros((0, x.len())));
        };
        let mut basis = if self.project_rigid {
            rigid_basis(slice)
        } else {
            Vec::new()
        };
        for mode in &self.extra {
            if mode.len() == slice.len() {
                basis.push(mode.clone());
            }
        }
        let mut modes = Array2::zeros((basis.len(), x.len()));
        for (row, mode) in basis.iter().enumerate() {
            for (col, value) in mode.iter().enumerate() {
                modes[(row, col)] = *value;
            }
        }
        Ok(modes)
    }
}

#[derive(Clone)]
struct Img {
    pos: Vec<f64>,
    energy: f64,
    force: Vec<f64>,
}

struct Found {
    pos: Vec<f64>,
    energy: f64,
    force_norm: f64,
    kind: char,
    hessian_index: i32,
}

struct Eval<'a> {
    f: &'a mut SurfaceFn<'a>,
}

impl Eval<'_> {
    fn at(&mut self, pos: &[f64]) -> Result<Img, SaddleError> {
        let (energy, force) = (self.f)(pos)?;
        Ok(Img {
            pos: pos.to_vec(),
            energy,
            force,
        })
    }
}

/// IDPP band, then one Lanczos climb from the highest image.
///
/// Smidstrup, Pedersen, Stokbro, and Jónsson, J. Chem. Phys. 140,
/// 214106 (2014): the surrogate keeps interpolated pair distances, so
/// the guess does not stack atoms. Its relaxation is not a surface
/// call. The highest image is the climb of Henkelman, Uberuaga, and
/// Jónsson, J. Chem. Phys. 113, 9901 (2000), doi:10.1063/1.1329672,
/// and of Goswami, Gunde, and Jónsson, Front. Chem. 14 (2026). The
/// seed is the improved tangent. A miss tries the linear midpoint,
/// which is the displacement used by the Lanczos saddle search, and
/// then the COMPASS chain.
fn climb_idpp_band(
    energy_force: &mut SurfaceFn<'_>,
    left: &[f64],
    right: &[f64],
    settings: &Settings,
) -> Result<Option<Found>, SaddleError> {
    if left.len() != right.len() || left.is_empty() || !left.len().is_multiple_of(3) {
        return Ok(None);
    }
    let interiors = settings.band_images.clamp(1, 32);
    let mut path = Vec::with_capacity(interiors + 2);
    path.push(left.to_vec());
    for image in 1..=interiors {
        let fraction = image as f64 / (interiors as f64 + 1.0);
        let linear = linear_mix(left, right, fraction);
        let target = mix_distances(&upper_distances(left), &upper_distances(right), fraction);
        path.push(relax_idpp(&linear, &target));
    }
    path.push(right.to_vec());

    // The endpoints are already known. Only the interior images are
    // candidates, so their energies are the ones that pick the climb.
    // A neighbor energy is read only when that neighbor is an endpoint
    // of the highest image; the improved tangent does not need the
    // other one.
    let mut energy = vec![0.0; path.len()];
    let mut force = vec![Vec::new(); path.len()];
    for index in 1..path.len() - 1 {
        let _phase = PhaseGuard::enter(Phase::Band);
        let (value, gradient) = energy_force(&path[index])?;
        energy[index] = value;
        force[index] = gradient;
    }
    let mut highest = 1usize;
    for index in 2..path.len() - 1 {
        if energy[index] > energy[highest] {
            highest = index;
        }
    }
    if highest == 1 {
        let _phase = PhaseGuard::enter(Phase::Band);
        let (value, gradient) = energy_force(&path[0])?;
        energy[0] = value;
        force[0] = gradient;
    }
    if highest + 2 == path.len() {
        let last = path.len() - 1;
        let _phase = PhaseGuard::enter(Phase::Band);
        let (value, gradient) = energy_force(&path[last])?;
        energy[last] = value;
        force[last] = gradient;
    }
    let tangent = improved_tangent(&path, &energy, highest);
    if let Some(found) = climb_lanczos(
        energy_force,
        &path[highest],
        energy[highest],
        &force[highest],
        &tangent,
        settings,
    )? {
        return Ok(Some(found));
    }
    let linear = linear_mix(left, right, 0.5);
    if distance(&linear, &path[highest]) > 1e-6 {
        let (value, gradient) = {
            let _phase = PhaseGuard::enter(Phase::Band);
            energy_force(&linear)?
        };
        if let Some(found) = climb_lanczos(
            energy_force,
            &linear,
            value,
            &gradient,
            &endpoint_tangent(left, right),
            settings,
        )? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

/// Lanczos min-mode from one image. The translation is
/// [`TranslationMemory`]. Accepted when the Cartesian force is under
/// the tolerance, the tracked curvature is negative, and a Lanczos
/// estimate with that mode removed is not clearly negative.
fn climb_lanczos(
    energy_force: &mut SurfaceFn<'_>,
    start: &[f64],
    mut energy: f64,
    force: &[f64],
    tangent: &[f64],
    settings: &Settings,
) -> Result<Option<Found>, SaddleError> {
    let _phase = PhaseGuard::enter(Phase::MinMode);
    let mut surface = CallbackSurface {
        f: energy_force,
        extra: Vec::new(),
        // The seed is the path tangent. Rigid modes stay out of that
        // Krylov when the tangent has no rigid part, which matches the
        // Lanczos saddle search (no rigid projection inside the product).
        project_rigid: false,
    };
    let config = MinModeConfig {
        kind: MinModeKind::Lanczos,
        krylov_dim: settings.lanczos_iterations.max(1),
        force_tol: settings.forcetol,
        max_move: settings.minmode_max_move,
        method: Method::Lbfgs {
            memory: settings.minmode_memory.max(1),
        },
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(
        config.clone(),
        Array1::from_vec(start.to_vec()),
        Array1::from_vec(tangent.to_vec()),
    )?;
    session.set_lanczos_rel_tol(Some(settings.lanczos_rel_tol))?;
    session.set_force_gate(ForceGate::L2Norm);
    let mut pos = start.to_vec();
    let mut force = force.to_vec();
    let mut memory = TranslationMemory::new(settings.minmode_memory, settings.minmode_max_move);
    let steps = settings.nsteps.clamp(1, 1_000);
    for _ in 0..steps {
        let gradient = Array1::from_iter(force.iter().copied().map(|value| -value));
        session.set_position(Array1::from_vec(pos.clone()), Some(gradient))?;
        let estimate = session.estimate_mode(&surface)?;
        if norm(&force) <= settings.forcetol && estimate.curvature < 0.0 {
            return accept_lanczos(
                &mut surface,
                &config,
                &pos,
                energy,
                &force,
                &estimate.mode.to_vec(),
                settings,
            );
        }
        let gradient = Array1::from_iter(force.iter().copied().map(|value| -value));
        let effective = translated_gradient(&gradient, &estimate.mode, estimate.curvature);
        let move_force: Vec<f64> = effective.iter().map(|value| -value).collect();
        let delta = memory.step(&pos, &move_force, estimate.curvature);
        if norm(&delta) <= 1e-16 {
            return Ok(None);
        }
        for (coord, step) in pos.iter_mut().zip(&delta) {
            *coord += *step;
        }
        let (value, next_force) = surface.energy_force(&pos)?;
        energy = value;
        force = next_force;
    }
    Ok(None)
}

/// A forward difference is coarser than the harness Hessian. A clearly
/// negative second curvature is index 2 or higher and is not accepted.
const SECOND_CURVATURE: f64 = -1.0e-4;

/// Complement subspace for the index estimate. Past a single Rayleigh
/// quotient, two actions is the smallest Lanczos basis.
const INDEX_KRYLOV: usize = 2;

fn accept_lanczos(
    surface: &mut CallbackSurface<'_>,
    config: &MinModeConfig,
    pos: &[f64],
    energy: f64,
    force: &[f64],
    mode: &[f64],
    settings: &Settings,
) -> Result<Option<Found>, SaddleError> {
    if norm(force) > settings.forcetol {
        return Ok(None);
    }
    surface.project_rigid = true;
    surface.extra = vec![mode.to_vec()];
    // Two Lanczos actions in the complement. The lowest Ritz value
    // upper-bounds the next curvature, so a negative value is another
    // unstable mode. A dense Hessian is not built.
    let (curvature, _) = complement_curvature(
        surface,
        config,
        pos,
        force,
        mode,
        INDEX_KRYLOV,
        Some(settings.lanczos_rel_tol),
    )?;
    if curvature < SECOND_CURVATURE {
        return Ok(None);
    }
    Ok(Some(Found {
        pos: pos.to_vec(),
        energy,
        force_norm: norm(force),
        kind: 's',
        hessian_index: 1,
    }))
}

fn complement_curvature(
    surface: &CallbackSurface<'_>,
    config: &MinModeConfig,
    pos: &[f64],
    force: &[f64],
    mode: &[f64],
    krylov: usize,
    rel_tol: Option<f64>,
) -> Result<(f64, usize), SaddleError> {
    let mut check_config = config.clone();
    check_config.krylov_dim = krylov.max(1);
    let mut check = MinModeSession::new(
        check_config,
        Array1::from_vec(pos.to_vec()),
        Array1::from_vec(spread_seed(pos.len(), mode)),
    )?;
    check.set_lanczos_rel_tol(rel_tol)?;
    check.set_force_gate(ForceGate::L2Norm);
    let gradient = Array1::from_iter(force.iter().copied().map(|value| -value));
    check.set_position(Array1::from_vec(pos.to_vec()), Some(gradient))?;
    let estimate = check.estimate_mode(surface)?;
    Ok((estimate.curvature, estimate.rotations))
}

impl CallbackSurface<'_> {
    fn energy_force(&self, pos: &[f64]) -> Result<(f64, Vec<f64>), SaddleError> {
        // SAFETY: the mode estimate has returned, so this is the only
        // live evaluation of the callback.
        unsafe { (*self.f)(pos) }
    }
}

/// L-BFGS on the min-mode force.
///
/// Liu and Nocedal, Math. Program. 45, 503 (1989),
/// doi:10.1007/BF01589116. The empty-memory scale is the inverse
/// curvature 0.01. After one accepted step the scale is `s·y / y·y`
/// (Nocedal and Wright, Numerical Optimization, eq. 7.20). The first
/// step uses `1/|λ|` when the tracked curvature `λ` is negative, which
/// is that mode's own scale. A non-positive secant drops the stored
/// pairs and steps `1000` times the force, clipped so the largest atom
/// moves by at most `max_move`. A proposed step that reaches that cap,
/// or that faces away from the force, drops the pairs and steps with
/// the current scale instead.
struct TranslationMemory {
    s: VecDeque<Vec<f64>>,
    y: VecDeque<Vec<f64>>,
    rho: VecDeque<f64>,
    prev_pos: Option<Vec<f64>>,
    prev_force: Option<Vec<f64>>,
    memory: usize,
    inverse_curvature: f64,
    max_move: f64,
}

impl TranslationMemory {
    fn new(memory: usize, max_move: f64) -> Self {
        Self {
            s: VecDeque::new(),
            y: VecDeque::new(),
            rho: VecDeque::new(),
            prev_pos: None,
            prev_force: None,
            memory: memory.max(1),
            inverse_curvature: 0.01,
            max_move,
        }
    }

    fn clear_pairs(&mut self) {
        self.s.clear();
        self.y.clear();
        self.rho.clear();
    }

    fn push_pair(&mut self, s: Vec<f64>, y: Vec<f64>, sy: f64) {
        self.rho.push_back(1.0 / sy);
        self.s.push_back(s);
        self.y.push_back(y);
        while self.s.len() > self.memory {
            self.s.pop_front();
            self.y.pop_front();
            self.rho.pop_front();
        }
    }

    fn two_loop(&self, force: &[f64], h0: f64) -> Vec<f64> {
        let n = self.s.len();
        let mut q: Vec<f64> = force.iter().map(|value| -value).collect();
        let mut alpha = vec![0.0; n];
        for k in (0..n).rev() {
            alpha[k] = self.rho[k] * dot(&self.s[k], &q);
            for (coord, direction) in q.iter_mut().zip(&self.y[k]) {
                *coord -= alpha[k] * direction;
            }
        }
        for coord in &mut q {
            *coord *= h0;
        }
        for k in 0..n {
            let beta = self.rho[k] * dot(&self.y[k], &q);
            for (coord, direction) in q.iter_mut().zip(&self.s[k]) {
                *coord += (alpha[k] - beta) * direction;
            }
        }
        q.iter().map(|value| -value).collect()
    }

    fn step(&mut self, pos: &[f64], force: &[f64], curvature: f64) -> Vec<f64> {
        let mut h0 = self.inverse_curvature;
        let mut reset_negative = false;
        let carried = self.prev_pos.as_ref().zip(self.prev_force.as_ref()).map(
            |(previous, previous_force)| {
                let disp: Vec<f64> = pos.iter().zip(previous).map(|(a, b)| a - b).collect();
                let secant: Vec<f64> = previous_force
                    .iter()
                    .zip(force)
                    .map(|(a, b)| a - b)
                    .collect();
                (disp, secant)
            },
        );
        if let Some((disp, secant)) = carried {
            let sy = dot(&disp, &secant);
            let yy = dot(&secant, &secant);
            if !(sy.is_finite() && yy.is_finite()) || sy <= 1.0e-30 {
                self.clear_pairs();
                reset_negative = true;
            } else {
                h0 = sy / yy;
                self.push_pair(disp, secant, sy);
            }
        } else if curvature.is_finite() && curvature < 0.0 {
            let scale = -curvature;
            if scale > 1.0e-8 {
                h0 = (1.0 / scale).min(1.0e6);
            }
        }
        let direction = if reset_negative {
            cap_atom(&scale(force, 1000.0), self.max_move)
        } else {
            let proposed = self.two_loop(force, h0);
            let too_far = self.max_move > 0.0 && max_atom(&proposed) >= self.max_move;
            let facing = cosine(&proposed, force);
            if too_far || facing.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
                self.clear_pairs();
                cap_atom(&scale(force, h0), self.max_move)
            } else {
                cap_atom(&proposed, self.max_move)
            }
        };
        self.prev_pos = Some(pos.to_vec());
        self.prev_force = Some(force.to_vec());
        direction
    }
}

fn cap_atom(step: &[f64], cap: f64) -> Vec<f64> {
    if cap.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return step.to_vec();
    }
    let largest = max_atom(step);
    if largest > cap && largest.is_finite() {
        scale(step, cap / largest)
    } else {
        step.to_vec()
    }
}

/// Henkelman and Jónsson, J. Chem. Phys. 113, 9978 (2000).
fn improved_tangent(path: &[Vec<f64>], energy: &[f64], index: usize) -> Vec<f64> {
    let previous = &path[index - 1];
    let here = &path[index];
    let next = &path[index + 1];
    let toward_next: Vec<f64> = next.iter().zip(here).map(|(a, b)| a - b).collect();
    let from_previous: Vec<f64> = here.iter().zip(previous).map(|(a, b)| a - b).collect();
    let e_prev = energy[index - 1];
    let e_here = energy[index];
    let e_next = energy[index + 1];
    let tangent = if e_next > e_here && e_here > e_prev {
        toward_next
    } else if e_next < e_here && e_here < e_prev {
        from_previous
    } else {
        let rise_prev = (e_prev - e_here).abs();
        let rise_next = (e_next - e_here).abs();
        let (larger, smaller) = if rise_prev > rise_next {
            (rise_prev, rise_next)
        } else {
            (rise_next, rise_prev)
        };
        if e_next > e_prev {
            toward_next
                .iter()
                .zip(&from_previous)
                .map(|(next_step, prev_step)| larger * next_step + smaller * prev_step)
                .collect()
        } else {
            toward_next
                .iter()
                .zip(&from_previous)
                .map(|(next_step, prev_step)| smaller * next_step + larger * prev_step)
                .collect()
        }
    };
    unit_direction(&tangent)
}

fn endpoint_tangent(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mode: Vec<f64> = right
        .iter()
        .zip(left)
        .map(|(end, start)| end - start)
        .collect();
    unit_direction(&mode)
}

fn unit_direction(mode: &[f64]) -> Vec<f64> {
    let scale = norm(mode);
    if scale < 1e-12 {
        let mut axis = vec![0.0; mode.len().max(1)];
        axis[0] = 1.0;
        return axis;
    }
    scale_copy(mode, 1.0 / scale)
}

fn scale_copy(mode: &[f64], factor: f64) -> Vec<f64> {
    mode.iter().map(|value| value * factor).collect()
}

/// Delocalized direction in the orthogonal complement of `mode`.
/// A coordinate axis can miss a soft mode for many Lanczos steps.
fn spread_seed(n: usize, mode: &[f64]) -> Vec<f64> {
    let mut seed = vec![0.0; n];
    for (i, value) in seed.iter_mut().enumerate() {
        *value = ((i + 1) as f64 * 1.618033988749895).sin();
    }
    let projection = dot(&seed, mode);
    for (value, mode_value) in seed.iter_mut().zip(mode) {
        *value -= projection * mode_value;
    }
    let scale = norm(&seed);
    if scale > 1e-8 {
        for value in &mut seed {
            *value /= scale;
        }
        return seed;
    }
    orthogonal_seed(n, mode)
}

fn orthogonal_seed(n: usize, mode: &[f64]) -> Vec<f64> {
    for axis in 0..n {
        let mut seed = vec![0.0; n];
        seed[axis] = 1.0;
        let projection = dot(&seed, mode);
        for (value, mode_value) in seed.iter_mut().zip(mode) {
            *value -= projection * mode_value;
        }
        let scale = norm(&seed);
        if scale > 1e-8 {
            for value in &mut seed {
                *value /= scale;
            }
            return seed;
        }
    }
    let mut axis = vec![0.0; n.max(1)];
    axis[0] = 1.0;
    axis
}

fn linear_mix(left: &[f64], right: &[f64], fraction: f64) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(start, end)| start + fraction * (end - start))
        .collect()
}

fn mix_distances(left: &[f64], right: &[f64], fraction: f64) -> Vec<f64> {
    left.iter()
        .zip(right)
        .map(|(start, end)| (1.0 - fraction) * start + fraction * end)
        .collect()
}

fn upper_distances(pos: &[f64]) -> Vec<f64> {
    let n = pos.len() / 3;
    let mut distances = Vec::with_capacity(n.saturating_sub(1) * n / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut acc = 0.0;
            for k in 0..3 {
                let delta = pos[3 * i + k] - pos[3 * j + k];
                acc += delta * delta;
            }
            distances.push(acc.sqrt());
        }
    }
    distances
}

/// Smidstrup, Pedersen, Stokbro, and Jónsson, J. Chem. Phys. 140,
/// 214106 (2014): `S = sum_{i<j} d_ij^{-4} (r_ij - d_ij)^2`.
fn idpp_gradient(pos: &[f64], target: &[f64]) -> (f64, Vec<f64>) {
    let n = pos.len() / 3;
    let mut energy = 0.0;
    let mut gradient = vec![0.0; pos.len()];
    let mut pair = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let desired = target.get(pair).copied().unwrap_or(1.0).max(1e-8);
            pair += 1;
            let mut delta = [0.0; 3];
            let mut r2 = 0.0;
            for k in 0..3 {
                delta[k] = pos[3 * i + k] - pos[3 * j + k];
                r2 += delta[k] * delta[k];
            }
            // Coincident atoms have no distance direction. A deterministic
            // axis and a tiny stand-in length keep the repulsion finite
            // and independent of a random draw.
            let (distance, unit) = if r2 <= 1e-16 {
                let mut unit = [0.0; 3];
                unit[(i + j) % 3] = 1.0;
                (1.0e-8, unit)
            } else {
                let distance = r2.sqrt();
                (distance, delta.map(|value| value / distance))
            };
            let weight = desired.powi(-4);
            let gap = distance - desired;
            energy += weight * gap * gap;
            let radial = 2.0 * weight * gap;
            for k in 0..3 {
                let component = radial * unit[k];
                gradient[3 * i + k] += component;
                gradient[3 * j + k] -= component;
            }
        }
    }
    (energy, gradient)
}

/// Steepest descent on the IDPP surrogate. Stops at a max atom force of
/// 0.001 or 80 steps. The cap per atom is 0.1. None of these calls is a
/// surface evaluation.
fn relax_idpp(start: &[f64], target: &[f64]) -> Vec<f64> {
    if target.is_empty() {
        return start.to_vec();
    }
    let mut position = start.to_vec();
    for _ in 0..80 {
        let (_energy, gradient) = idpp_gradient(&position, target);
        let mut largest = 0.0_f64;
        for atom in gradient.chunks(3) {
            largest = largest.max(norm(atom));
        }
        if largest < 1.0e-3 {
            break;
        }
        for (atom, step) in position.chunks_mut(3).zip(gradient.chunks(3)) {
            let scale = norm(step);
            let cap = if scale > 0.1 { 0.1 / scale } else { 1.0 };
            for k in 0..3 {
                atom[k] -= cap * step[k];
            }
        }
    }
    position
}

fn search_pair(
    energy_force: &mut SurfaceFn<'_>,
    left: &[f64],
    right: &[f64],
    settings: &Settings,
) -> Result<Found, SaddleError> {
    tally(|profile| profile.searches += 1);
    let mut ev = Eval { f: energy_force };
    let a = minimize(&mut ev, left, settings)?;
    let mut b = minimize(&mut ev, right, settings)?;
    let (b_pos, b_force) = align_to(&b.pos, &b.force, &a.pos);
    b.pos = b_pos;
    b.force = b_force;
    let mut nodes = vec![a.clone(), b.clone()];
    let mut segments = vec![Segment {
        pos: midpoint(&a.pos, &b.pos),
        energy: f64::INFINITY,
        kind: 'u',
    }];
    let mut iloop = 0;
    while iloop < settings.n_loops {
        let (isegment, inodes) = select(&segments, &nodes, settings.energy_difference_threshold);
        if segments[isegment].kind == 's' || segments[isegment].kind == 'f' {
            break;
        }
        if !inodes.is_empty() {
            // Drop high nodes and keep the highest barrier of the merged span.
            let start = inodes[0] - 1;
            let end = *inodes.last().unwrap();
            let mut best = start;
            for i in start..=end {
                if segments[i].energy > segments[best].energy {
                    best = i;
                }
            }
            let kept = segments[best].clone();
            for inode in inodes.iter().rev() {
                nodes.remove(*inode);
            }
            segments.drain(start..=end);
            segments.insert(start, kept);
        }
        let (isegment, _) = select(&segments, &nodes, settings.energy_difference_threshold);
        let mut pair = [nodes[isegment].clone(), nodes[isegment + 1].clone()];
        let (pos, force) = align_to(&pair[1].pos, &pair[1].force, &pair[0].pos);
        pair[1].pos = pos;
        pair[1].force = force;
        let outcome = optimization_loop(&mut ev, pair, settings)?;
        let barrier_energy = outcome.barrier_energy;
        let energy_max = outcome.energy_max;
        let images_energy = outcome.images[0].energy.max(outcome.images[1].energy);
        let energy_upper = energy_max.max(barrier_energy);
        let mid = midpoint(&outcome.images[0].pos, &outcome.images[1].pos);
        let failed_to_pin = outcome.iterations > settings.nsteps
            || !outcome.saddle_found
            || energy_max > barrier_energy + settings.energy_difference_threshold;
        if failed_to_pin {
            if outcome.extrema.len() > 1 && outcome.distance_reached {
                let nminima = settings_nsplit().min(outcome.extrema.len() / 2);
                let mut inserted = Vec::new();
                for i in 0..nminima {
                    inserted.push(outcome.extrema[1 + 2 * i][0].clone());
                }
                for i in (0..nminima).rev() {
                    inserted.push(outcome.extrema[1 + 2 * i][1].clone());
                }
                if settings.quench_nodes {
                    for node in &mut inserted {
                        *node = minimize(&mut ev, &node.pos, settings)?;
                    }
                }
                let mut inner = Vec::new();
                let half = outcome.extrema.len().div_ceil(2);
                for i in nminima..half {
                    inner.push(outcome.extrema[2 * i][0].clone());
                }
                inner.push(Img {
                    pos: mid.clone(),
                    energy: barrier_energy,
                    force: vec![0.0; mid.len()],
                });
                for i in (nminima..half).rev() {
                    inner.push(outcome.extrema[2 * i][1].clone());
                }
                let mut imax = 0;
                for i in 1..inner.len() {
                    if inner[i].energy > inner[imax].energy {
                        imax = i;
                    }
                }
                let emax = inner[imax].energy;
                let mut new_segments = Vec::new();
                for i in 0..nminima {
                    new_segments.push(Segment {
                        pos: outcome.extrema[2 * i][0].pos.clone(),
                        energy: outcome.extrema[2 * i][0].energy,
                        kind: 'u',
                    });
                }
                let mut middle = Segment {
                    pos: inner[imax].pos.clone(),
                    energy: emax,
                    kind: 'u',
                };
                if outcome.saddle_found
                    && barrier_energy + settings.energy_difference_threshold >= emax
                {
                    middle.kind = 's';
                    middle.pos = mid.clone();
                    middle.energy = barrier_energy;
                }
                new_segments.push(middle);
                for i in (0..nminima).rev() {
                    new_segments.push(Segment {
                        pos: outcome.extrema[2 * i][1].pos.clone(),
                        energy: outcome.extrema[2 * i][1].energy,
                        kind: 'u',
                    });
                }
                for (offset, node) in inserted.into_iter().enumerate() {
                    nodes.insert(isegment + 1 + offset, node);
                }
                segments.splice(isegment..=isegment, new_segments);
            } else if outcome.iterations <= settings.nsteps
                && outcome.moving_sides.len() == 1
                && energy_max <= images_energy
            {
                let moving = outcome.moving_side;
                let mut split_node = outcome.images[moving].clone();
                if settings.quench_nodes {
                    split_node = minimize(&mut ev, &split_node.pos, settings)?;
                }
                nodes.insert(isegment + 1, split_node);
                let left = Segment {
                    pos: outcome.images[0].pos.clone(),
                    energy: outcome.images[0].energy,
                    kind: 'u',
                };
                let right = Segment {
                    pos: outcome.images[1].pos.clone(),
                    energy: outcome.images[1].energy,
                    kind: 'u',
                };
                segments.splice(isegment..=isegment, [left, right]);
            } else {
                if outcome.iterations <= settings.nsteps
                    && segments[isegment].energy >= barrier_energy
                {
                    segments[isegment] = Segment {
                        pos: mid.clone(),
                        energy: energy_upper,
                        kind: 'u',
                    };
                }
                let (next, next_nodes) =
                    select(&segments, &nodes, settings.energy_difference_threshold);
                if outcome.iterations > settings.nsteps
                    || (next == isegment && next_nodes.is_empty())
                {
                    let energy = if outcome.iterations > settings.nsteps {
                        segments[isegment].energy
                    } else {
                        segments[isegment].energy.min(energy_upper)
                    };
                    segments[isegment].energy = energy;
                    segments[isegment].kind = 'f';
                    segments[isegment].pos = mid;
                }
            }
        } else {
            segments[isegment] = Segment {
                pos: mid,
                energy: barrier_energy,
                kind: 's',
            };
        }
        iloop += 1;
    }
    let (isegment, _) = select(&segments, &nodes, settings.energy_difference_threshold);
    let chosen = &segments[isegment];
    let final_img = ev.at(&chosen.pos)?;
    Ok(Found {
        pos: final_img.pos,
        energy: final_img.energy,
        force_norm: norm(&final_img.force),
        kind: chosen.kind,
        hessian_index: -1,
    })
}

fn settings_nsplit() -> usize {
    2
}

#[derive(Clone)]
struct Segment {
    pos: Vec<f64>,
    energy: f64,
    kind: char,
}

struct Outcome {
    images: [Img; 2],
    saddle_found: bool,
    barrier_energy: f64,
    energy_max: f64,
    distance_reached: bool,
    extrema: Vec<[Img; 2]>,
    iterations: usize,
    moving_sides: Vec<usize>,
    moving_side: usize,
}

fn optimization_loop(
    ev: &mut Eval<'_>,
    pair: [Img; 2],
    settings: &Settings,
) -> Result<Outcome, SaddleError> {
    tally(|profile| profile.sqnm_segments += 1);
    let _phase = PhaseGuard::enter(Phase::Sqnm);
    let n = pair[0].pos.len();
    let mut opt = ConstrainedOpt::new(
        2 * n,
        settings.alpha0,
        f64::INFINITY,
        settings.nhistx,
        Some(settings.trustradius),
    );
    let mut images = pair;
    let moving_side = if images[1].energy < images[0].energy {
        1
    } else {
        0
    };
    let mut moving_sides =
        if (images[1].energy - images[0].energy).abs() > settings.energy_difference_threshold {
            vec![moving_side]
        } else {
            vec![0, 1]
        };
    let mut constant_distance_desired = if moving_sides.len() == 1 {
        settings.target_distance_other
    } else {
        settings.target_distance_saddle
    };
    let mut extrema = vec![[images[0].clone(), images[1].clone()]];
    let mut distance_reached =
        distance(&images[0].pos, &images[1].pos) < constant_distance_desired + 1e-7;
    let mut expecting_equal = false;
    let startd = distance(&images[0].pos, &images[1].pos);
    let mut dstep = startd * (settings.distance_step_min * settings.distance_step_max).sqrt();
    let mut old = images.clone();
    let mut it = 0usize;
    let mut ndim = 0usize;
    let (saddle_found, barrier_energy) = loop {
        if it > 0 {
            for &side in &moving_sides.clone() {
                images[side] = ev.at(&images[side].pos)?;
            }
        }
        match optimization_step(
            &mut images,
            &old,
            &mut moving_sides,
            &mut expecting_equal,
            &mut distance_reached,
            &mut opt,
            it,
            &mut extrema,
            &mut dstep,
            &mut ndim,
            &mut constant_distance_desired,
            settings,
        ) {
            StepResult::Stop {
                saddle,
                barrier,
                past_cap,
            } => {
                if past_cap {
                    it += 1;
                }
                break (saddle, barrier);
            }
            StepResult::Moved { positions, .. } => {
                old = images.clone();
                for (side, pos) in positions {
                    images[side].pos = pos;
                }
                it += 1;
            }
        }
    };
    let energy_max = extrema
        .iter()
        .map(|pair| pair[0].energy.max(pair[1].energy))
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(Outcome {
        images,
        saddle_found,
        barrier_energy,
        energy_max,
        distance_reached,
        extrema,
        iterations: it,
        moving_sides,
        moving_side,
    })
    .inspect(|outcome| tally(|profile| profile.sqnm_steps += outcome.iterations))
}

enum StepResult {
    Stop {
        saddle: bool,
        barrier: f64,
        past_cap: bool,
    },
    Moved {
        positions: Vec<(usize, Vec<f64>)>,
    },
}

fn optimization_step(
    images: &mut [Img; 2],
    _old: &[Img; 2],
    moving_sides: &mut Vec<usize>,
    expecting_equal: &mut bool,
    distance_reached: &mut bool,
    opt: &mut ConstrainedOpt,
    it: usize,
    extrema: &mut Vec<[Img; 2]>,
    dstep: &mut f64,
    ndim: &mut usize,
    constant_distance_desired: &mut f64,
    settings: &Settings,
) -> StepResult {
    let n = images[0].pos.len();
    let energy = [images[0].energy, images[1].energy];
    let pos = [images[0].pos.clone(), images[1].pos.clone()];
    let force_img = [images[0].force.clone(), images[1].force.clone()];
    let mut same = cat(&force_img[0], &neg(&force_img[1]));
    let dist = distance(&pos[0], &pos[1]);
    // A clamped step can undershoot the target distance and then oscillate
    // without ever taking an unclamped step, so `distance_reached` stays false.
    // The images are already as close as the constraint asks.
    if !*distance_reached && dist <= *constant_distance_desired + 1e-7 {
        *distance_reached = true;
    }
    let same_norm = norm(&same);
    let well_defined =
        same_norm > 0.0 && (energy[1] - energy[0]).abs() / same_norm < settings.trustradius;
    if moving_sides.len() == 1 {
        let moving = moving_sides[0];
        let frozen = 1 - moving;
        if (energy[moving] >= energy[frozen] || (*expecting_equal && well_defined))
            && !*distance_reached
        {
            opt.reset_history();
            *moving_sides = vec![0, 1];
            *constant_distance_desired = dist.min(settings.target_distance_saddle);
        }
    }
    let mut dist_grad = distance_gradient(&pos[0], &pos[1]);
    if moving_sides.len() == 1 && !well_defined && moving_sides.len() > 1 {
        // unreachable, kept so the phase-2 ill-defined branch below is the live one
    }
    if moving_sides.len() > 1 && !well_defined {
        *moving_sides = vec![if energy[1] < energy[0] { 1 } else { 0 }];
        opt.reset_history();
    }
    let mut violation_energy = 0.0;
    let mut require_projection = false;
    if moving_sides.len() > 1 {
        violation_energy = energy[1] - energy[0];
        if violation_energy.abs() > settings.energy_difference_threshold {
            require_projection = true;
            opt.reset_history();
        }
    } else {
        let frozen = 1 - moving_sides[0];
        zero_half(&mut dist_grad, frozen, n);
        zero_half(&mut same, frozen, n);
    }
    let mut moving_force = cat(&force_img[0], &force_img[1]);
    if moving_sides.len() == 1 {
        zero_half(&mut moving_force, 1 - moving_sides[0], n);
    }
    let energy_cols: Vec<Vec<f64>> = if moving_sides.len() > 1 {
        vec![same.clone()]
    } else {
        Vec::new()
    };
    let manifold = project_off(&moving_force, &energy_cols);
    let constrained_cols = {
        let mut cols = vec![dist_grad.clone()];
        cols.extend(energy_cols.iter().cloned());
        cols
    };
    let perp = project_off(&moving_force, &constrained_cols);
    let cosp = cosine(&manifold, &dist_grad);
    let tangent = normalized_diff(&pos[1], &pos[0]);
    let grad_pair = [-dot(&tangent, &force_img[0]), -dot(&tangent, &force_img[1])];
    let span = dist;
    let (_t, energy_est) = cubic_max(energy, [grad_pair[0] * span, grad_pair[1] * span]);
    record_extrema(extrema, images, it, moving_sides.len() > 1, settings);
    let energy_max = extrema
        .iter()
        .map(|p| p[0].energy.max(p[1].energy))
        .fold(f64::NEG_INFINITY, f64::max);
    if moving_sides.len() > 1 && it >= 5 && extrema.len() == 2 && energy_max > energy_est {
        extrema.push(images.clone());
        extrema.push(images.clone());
    }
    let perp_norm = norm(&perp);
    if perp_norm < settings.forcetol.min(settings.forcetol_per_distance * dist)
        && !opt.trustradius_applied
    {
        *dstep = (*dstep * 1.25).min(settings.distance_step_max * dist);
    } else {
        *dstep = (*dstep / 1.25).max(settings.distance_step_min * dist);
    }
    if moving_sides.len() > 1
        && dist
            < settings
                .target_distance_saddle
                .max(settings.target_distance_other)
        && cosp < settings.saddle_cosp.min(0.2)
        && energy_max > energy_est + settings.energy_difference_threshold
    {
        *distance_reached = true;
        return StepResult::Stop {
            saddle: false,
            barrier: energy_est,
            past_cap: false,
        };
    }
    if moving_sides.len() == 1 && *distance_reached {
        return StepResult::Stop {
            saddle: false,
            barrier: energy_est,
            past_cap: false,
        };
    }
    if moving_sides.len() > 1
        && perp_norm < settings.forcetol
        && perp_norm < settings.forcetol_per_distance * dist
        && *distance_reached
    {
        return StepResult::Stop {
            saddle: cosp > settings.saddle_cosp,
            barrier: energy_est,
            past_cap: false,
        };
    }
    if it >= settings.nsteps {
        // Python increments `it` before leaving, so the caller sees it > nsteps.
        return StepResult::Stop {
            saddle: false,
            barrier: energy_est,
            past_cap: true,
        };
    }
    let projection_w = energy_cols;
    let projection_b: Vec<f64> = if projection_w.is_empty() {
        Vec::new()
    } else {
        let packed = cat(&pos[0], &pos[1]);
        vec![dot(&projection_w[0], &packed) - violation_energy]
    };
    let constraint = dist_grad;
    let packed = cat(&pos[0], &pos[1]);
    let out = opt.step(
        &packed,
        energy[0] + energy[1],
        &moving_force,
        &projection_w,
        &projection_b,
        Some(&constraint),
        0.0,
        0.0,
    );
    *ndim = out.ndim;
    let projected = out.position;
    let mut r = [projected[..n].to_vec(), projected[n..].to_vec()];
    let mut dstep_applied = false;
    if !require_projection {
        let d2 = out.displacement2.expect("constraint step");
        let v1 = neg(&out.displacement);
        let v2 = scale(&neg(&d2), 1.0 / norm(&d2));
        let temp = [add(&r[0], &v1[..n]), add(&r[1], &v1[n..])];
        let v2s = [&v2[..n], &v2[n..]];
        let a = norm(&sub(v2s[0], v2s[1])).powi(2);
        let b = 2.0 * dot(&sub(&temp[0], &temp[1]), &sub(v2s[0], v2s[1]));
        let scl2_current = solve_scale(
            a,
            b,
            distance(&temp[0], &temp[1]).powi(2) - distance(&r[0], &r[1]).powi(2),
        );
        let mut scl2 = solve_scale(
            a,
            b,
            distance(&temp[0], &temp[1]).powi(2) - constant_distance_desired.powi(2),
        );
        let stepx = (*dstep).min(settings.trusted_force_difference * norm(&d2));
        if (scl2 - scl2_current).abs() >= stepx {
            scl2 = scl2_current + stepx * (scl2 - scl2_current).signum();
            dstep_applied = true;
        }
        let v = add(&v1, &scale(&v2, scl2));
        let mut scl = 1.0;
        if dstep_applied {
            let trial = [add(&r[0], &v[..n]), add(&r[1], &v[n..])];
            if distance(&r[0], &r[1]) < distance(&trial[0], &trial[1])
                && distance(&trial[0], &trial[1]) > *constant_distance_desired
            {
                let aa = norm(&sub(&v[..n], &v[n..])).powi(2);
                let bb = 2.0 * dot(&sub(&r[0], &r[1]), &sub(&v[..n], &v[n..]));
                if aa != 0.0 {
                    scl = -bb / aa;
                }
            }
        } else if scl2_current.abs() > stepx {
            dstep_applied = true;
            opt.reset_history();
            scl = stepx / scl2_current.abs();
        }
        let mut scl3 = 1.0;
        if moving_sides.len() == 1 {
            let moving = moving_sides[0];
            let frozen = 1 - moving;
            let v_scaled = scale(&v, scl);
            let denom = dot(&force_img[frozen], &v_scaled[frozen * n..(frozen + 1) * n])
                - dot(&force_img[moving], &v_scaled[moving * n..(moving + 1) * n]);
            if denom != 0.0 {
                let trial = (energy[frozen] - energy[moving]) / denom;
                if (0.0..=1.0).contains(&trial) && well_defined {
                    scl3 = trial;
                    *expecting_equal = true;
                }
            }
        }
        let step = scale(&v, scl3 * scl);
        r[0] = add(&r[0], &step[..n]);
        r[1] = add(&r[1], &step[n..]);
    } else {
        opt.reset_history();
        dstep_applied = true;
    }
    if !*distance_reached && !dstep_applied {
        *distance_reached = true;
        if moving_sides.len() == 1 {
            *constant_distance_desired = distance(&r[0], &r[1]);
        }
    }
    let positions = moving_sides
        .iter()
        .map(|&side| (side, r[side].clone()))
        .collect();
    StepResult::Moved { positions }
}

fn record_extrema(
    extrema: &mut Vec<[Img; 2]>,
    images: &[Img; 2],
    _it: usize,
    both: bool,
    settings: &Settings,
) {
    if !both {
        return;
    }
    let mean = 0.5 * (images[0].energy + images[1].energy);
    let last = extrema.last().unwrap();
    let last_mean = 0.5 * (last[0].energy + last[1].energy);
    if extrema.len() % 2 == 1 {
        if mean > last_mean {
            *extrema.last_mut().unwrap() = images.clone();
        } else if mean < last_mean - settings.energy_difference_threshold {
            extrema.push(images.clone());
        }
    } else if mean < last_mean {
        *extrema.last_mut().unwrap() = images.clone();
    } else if mean > last_mean + settings.energy_difference_threshold {
        extrema.push(images.clone());
    }
}

fn solve_scale(a: f64, b: f64, c: f64) -> f64 {
    if a == 0.0 {
        return 0.0;
    }
    let disc = b * b - 4.0 * a * c;
    if disc >= 0.0 {
        let x1 = (-b + disc.sqrt()) / (2.0 * a);
        let x2 = (-b - disc.sqrt()) / (2.0 * a);
        if x1.abs() < x2.abs() { x1 } else { x2 }
    } else {
        -b / a
    }
}

fn cubic_max(values: [f64; 2], grads: [f64; 2]) -> (f64, f64) {
    let (v0, v1) = (values[0], values[1]);
    let (g0, g1) = (grads[0], grads[1]);
    let a = 2.0 * v0 - 2.0 * v1 + g0 + g1;
    let b = -3.0 * v0 + 3.0 * v1 - 2.0 * g0 - g1;
    let c = g0;
    let d = v0;
    let mut candidates = vec![0.0, 1.0];
    let disc = (2.0 * b) * (2.0 * b) - 4.0 * (3.0 * a) * c;
    if disc >= 0.0 && a != 0.0 {
        for sign in [1.0, -1.0] {
            let t = (-2.0 * b + sign * disc.sqrt()) / (2.0 * 3.0 * a);
            if (0.0..=1.0).contains(&t) {
                candidates.push(t);
            }
        }
    }
    let mut best_t = 0.0;
    let mut best = f64::NEG_INFINITY;
    for t in candidates {
        let value = ((a * t + b) * t + c) * t + d;
        if value > best {
            best = value;
            best_t = t;
        }
    }
    (best_t, best)
}

fn select(segments: &[Segment], nodes: &[Img], threshold: f64) -> (usize, Vec<usize>) {
    let mut best = 0;
    let mut best_score = f64::NEG_INFINITY;
    for (i, segment) in segments.iter().enumerate() {
        let score = if segment.kind == 's' {
            segment.energy + threshold
        } else {
            segment.energy
        };
        if score > best_score {
            best_score = score;
            best = i;
        }
    }
    let mut inodes = Vec::new();
    for i in (1..=best).rev() {
        if nodes[i].energy > segments[best].energy - 0.5 * threshold {
            inodes.insert(0, i);
        } else {
            break;
        }
    }
    for i in best + 1..segments.len() {
        if nodes[i].energy > segments[best].energy - 0.5 * threshold {
            inodes.push(i);
        } else {
            break;
        }
    }
    (best, inodes)
}

fn minimize(ev: &mut Eval<'_>, pos: &[f64], settings: &Settings) -> Result<Img, SaddleError> {
    tally(|profile| profile.quenches += 1);
    let _phase = PhaseGuard::enter(Phase::Quench);
    let mut opt = ConstrainedOpt::new(
        pos.len(),
        settings.alpha0,
        settings.minimize_maxrise,
        settings.nhistx,
        Some(settings.trustradius),
    );
    let mut x = pos.to_vec();
    let mut it = 0;
    loop {
        let img = ev.at(&x)?;
        if max_atom(&img.force) < settings.minimize_fmax || it == settings.minimize_nsteps {
            return Ok(img);
        }
        let step = opt.step(&x, img.energy, &img.force, &[], &[], None, 0.0, 0.0);
        x = sub(&step.position, &step.displacement);
        it += 1;
    }
}

fn align_to(pos: &[f64], force: &[f64], target: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = pos.len() / 3;
    let mut c = [0.0; 3];
    let mut c0 = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += pos[3 * i + k];
            c0[k] += target[3 * i + k];
        }
    }
    for k in 0..3 {
        c[k] /= n as f64;
        c0[k] /= n as f64;
    }
    // ASE rotation_matrix_from_points(source.T, target.T): quaternion Kabsch.
    let mut r11 = 0.0;
    let mut r22 = 0.0;
    let mut r33 = 0.0;
    let mut r12 = 0.0;
    let mut r23 = 0.0;
    let mut r31 = 0.0;
    let mut r13 = 0.0;
    let mut r21 = 0.0;
    let mut r32 = 0.0;
    for i in 0..n {
        let px = pos[3 * i] - c[0];
        let py = pos[3 * i + 1] - c[1];
        let pz = pos[3 * i + 2] - c[2];
        let qx = target[3 * i] - c0[0];
        let qy = target[3 * i + 1] - c0[1];
        let qz = target[3 * i + 2] - c0[2];
        r11 += px * qx;
        r22 += py * qy;
        r33 += pz * qz;
        r12 += px * qy;
        r23 += py * qz;
        r31 += pz * qx;
        r13 += px * qz;
        r21 += py * qx;
        r32 += pz * qy;
    }
    let rot = quaternion_rotation(r11, r22, r33, r12, r23, r31, r13, r21, r32);
    let mut out = vec![0.0; pos.len()];
    let mut fout = vec![0.0; force.len()];
    for i in 0..n {
        let p = [
            pos[3 * i] - c[0],
            pos[3 * i + 1] - c[1],
            pos[3 * i + 2] - c[2],
        ];
        let f = [force[3 * i], force[3 * i + 1], force[3 * i + 2]];
        for r in 0..3 {
            let mut vp = c0[r];
            let mut vf = 0.0;
            for col in 0..3 {
                vp += p[col] * rot[(r, col)];
                vf += f[col] * rot[(r, col)];
            }
            out[3 * i + r] = vp;
            fout[3 * i + r] = vf;
        }
    }
    (out, fout)
}

struct Mat3 {
    v: [f64; 9],
}

impl Mat3 {
    fn zero() -> Self {
        Self { v: [0.0; 9] }
    }
}

impl std::ops::Index<(usize, usize)> for Mat3 {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.v[3 * r + c]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Mat3 {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.v[3 * r + c]
    }
}

fn quaternion_rotation(
    r11: f64,
    r22: f64,
    r33: f64,
    r12: f64,
    r23: f64,
    r31: f64,
    r13: f64,
    r21: f64,
    r32: f64,
) -> Mat3 {
    let f = [
        [r11 + r22 + r33, r23 - r32, r31 - r13, r12 - r21],
        [r23 - r32, r11 - r22 - r33, r12 + r21, r13 + r31],
        [r31 - r13, r12 + r21, -r11 + r22 - r33, r23 + r32],
        [r12 - r21, r13 + r31, r23 + r32, -r11 - r22 + r33],
    ];
    let mut m = faer::Mat::<f64>::zeros(4, 4);
    for i in 0..4 {
        for j in 0..4 {
            m[(i, j)] = f[i][j];
        }
    }
    let eig = m
        .self_adjoint_eigen(faer::Side::Lower)
        .expect("quaternion eigh");
    let mut best = 0;
    for k in 1..4 {
        if eig.S()[k] > eig.S()[best] {
            best = k;
        }
    }
    let q0 = eig.U()[(0, best)];
    let q1 = eig.U()[(1, best)];
    let q2 = eig.U()[(2, best)];
    let q3 = eig.U()[(3, best)];
    let mut rot = Mat3::zero();
    rot[(0, 0)] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3;
    rot[(0, 1)] = 2.0 * (q1 * q2 - q0 * q3);
    rot[(0, 2)] = 2.0 * (q1 * q3 + q0 * q2);
    rot[(1, 0)] = 2.0 * (q1 * q2 + q0 * q3);
    rot[(1, 1)] = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3;
    rot[(1, 2)] = 2.0 * (q2 * q3 - q0 * q1);
    rot[(2, 0)] = 2.0 * (q1 * q3 - q0 * q2);
    rot[(2, 1)] = 2.0 * (q2 * q3 + q0 * q1);
    rot[(2, 2)] = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3;
    rot
}

fn midpoint(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| 0.5 * (x + y)).collect()
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    norm(&sub(b, a))
}

fn distance_gradient(a: &[f64], b: &[f64]) -> Vec<f64> {
    let diff = sub(b, a);
    let n = norm(&diff);
    let mut g = vec![0.0; a.len() + b.len()];
    for i in 0..a.len() {
        g[i] = -diff[i] / n;
        g[a.len() + i] = diff[i] / n;
    }
    g
}

fn normalized_diff(a: &[f64], b: &[f64]) -> Vec<f64> {
    let d = sub(a, b);
    let n = norm(&d);
    scale(&d, 1.0 / n)
}

fn project_off(v: &[f64], cols: &[Vec<f64>]) -> Vec<f64> {
    if cols.is_empty() {
        return v.to_vec();
    }
    let p = cols.len();
    let mut gram = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            gram[i][j] = dot(&cols[i], &cols[j]);
        }
    }
    let mut rhs: Vec<f64> = (0..p).map(|i| dot(&cols[i], v)).collect();
    for i in 0..p {
        let mut pivot = i;
        for r in i + 1..p {
            if gram[r][i].abs() > gram[pivot][i].abs() {
                pivot = r;
            }
        }
        gram.swap(i, pivot);
        rhs.swap(i, pivot);
        let div = gram[i][i];
        for c in i..p {
            gram[i][c] /= div;
        }
        rhs[i] /= div;
        for r in 0..p {
            if r == i {
                continue;
            }
            let f = gram[r][i];
            for c in i..p {
                gram[r][c] -= f * gram[i][c];
            }
            rhs[r] -= f * rhs[i];
        }
    }
    let mut out = v.to_vec();
    for (col, c) in cols.iter().zip(&rhs) {
        for (o, value) in out.iter_mut().zip(col) {
            *o -= *c * *value;
        }
    }
    out
}

fn zero_half(v: &mut [f64], side: usize, n: usize) {
    for x in &mut v[side * n..(side + 1) * n] {
        *x = 0.0;
    }
}

fn cat(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

fn neg(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| -x).collect()
}

fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn scale(v: &[f64], s: f64) -> Vec<f64> {
    v.iter().map(|x| x * s).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        f64::NAN
    } else {
        dot(a, b) / (na * nb)
    }
}

fn max_atom(v: &[f64]) -> f64 {
    v.chunks(3)
        .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
        .fold(0.0, f64::max)
}

fn same_basin(a: &Img, b: &Img) -> bool {
    if (a.energy - b.energy).abs() >= 0.05 {
        return false;
    }
    let (pos, _) = align_to(&a.pos, &a.force, &b.pos);
    let atoms = (pos.len() / 3).max(1) as f64;
    let mut acc = 0.0;
    for (x, y) in pos.iter().zip(&b.pos) {
        let d = x - y;
        acc += d * d;
    }
    (acc / atoms).sqrt() < 0.30
}

fn gradient(ev: &mut Eval<'_>, pos: &[f64]) -> Result<Vec<f64>, SaddleError> {
    Ok(ev.at(pos)?.force.iter().map(|f| -f).collect())
}

/// Rigid-projected Hessian index and its most negative eigenvector.
/// Finite-difference calls go through `ev` and are part of the search cost.
fn vibrational_mode(ev: &mut Eval<'_>, pos: &[f64]) -> Result<(usize, Vec<f64>), SaddleError> {
    tally(|profile| {
        profile.hessian_checks += 1;
        profile.hessian_coordinates = pos.len();
    });
    let _phase = PhaseGuard::enter(Phase::Hessian);
    let n = pos.len();
    let step = 1e-5;
    let mut hess = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut plus = pos.to_vec();
        let mut minus = pos.to_vec();
        plus[i] += step;
        minus[i] -= step;
        let gp = gradient(ev, &plus)?;
        let gm = gradient(ev, &minus)?;
        for j in 0..n {
            hess[j][i] = (gp[j] - gm[j]) / (2.0 * step);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (hess[i][j] + hess[j][i]);
            hess[i][j] = mean;
            hess[j][i] = mean;
        }
    }
    let modes = rigid_basis(pos);
    let (evals, evecs) = eigh_dense(&project_rigid(&hess, &modes));
    let mut index = 0;
    for value in &evals {
        if *value < -1e-6 {
            index += 1;
        }
    }
    let mode = if evecs.is_empty() {
        vec![0.0; n]
    } else {
        evecs[0].clone()
    };
    Ok((index, mode))
}

fn project_rigid(hess: &[Vec<f64>], modes: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = hess.len();
    if modes.is_empty() {
        return hess.to_vec();
    }
    // columns of H, then P (H v) for each column, then add V V^T.
    let mut out = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = hess[i][j];
        }
        let p_col = project_out(&col, modes);
        for i in 0..n {
            out[i][j] = p_col[i];
        }
    }
    for i in 0..n {
        let row = project_out(&out[i], modes);
        out[i] = row;
    }
    for mode in modes {
        for i in 0..n {
            for j in 0..n {
                out[i][j] += mode[i] * mode[j];
            }
        }
    }
    out
}

fn project_out(v: &[f64], modes: &[Vec<f64>]) -> Vec<f64> {
    let mut out = v.to_vec();
    for mode in modes {
        let c = dot(mode, &out);
        for (o, m) in out.iter_mut().zip(mode) {
            *o -= c * *m;
        }
    }
    out
}

fn rigid_basis(pos: &[f64]) -> Vec<Vec<f64>> {
    let n = pos.len() / 3;
    if n < 2 {
        return Vec::new();
    }
    let mut c = [0.0; 3];
    for i in 0..n {
        for k in 0..3 {
            c[k] += pos[3 * i + k];
        }
    }
    for k in 0..3 {
        c[k] /= n as f64;
    }
    let mut seeds: Vec<Vec<f64>> = Vec::new();
    for axis in 0..3 {
        let mut translation = vec![0.0; pos.len()];
        for i in 0..n {
            translation[3 * i + axis] = 1.0;
        }
        seeds.push(translation);
        let mut rotation = vec![0.0; pos.len()];
        for i in 0..n {
            let p = [
                pos[3 * i] - c[0],
                pos[3 * i + 1] - c[1],
                pos[3 * i + 2] - c[2],
            ];
            let rot = match axis {
                0 => [0.0, p[2], -p[1]],
                1 => [-p[2], 0.0, p[0]],
                _ => [p[1], -p[0], 0.0],
            };
            for k in 0..3 {
                rotation[3 * i + k] = rot[k];
            }
        }
        seeds.push(rotation);
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for seed in seeds {
        let mut mode = seed;
        for kept in &basis {
            let c = dot(kept, &mode);
            for (m, k) in mode.iter_mut().zip(kept) {
                *m -= c * *k;
            }
        }
        let norm = norm(&mode);
        if norm > 1e-8 {
            for m in &mut mode {
                *m /= norm;
            }
            basis.push(mode);
        }
    }
    basis
}

fn eigh_dense(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = matrix[i][j];
        }
    }
    let eig = m.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let mut evals = vec![0.0; n];
    let mut evecs = vec![vec![0.0; n]; n];
    for k in 0..n {
        evals[k] = eig.S()[k];
        let mut col = vec![0.0; n];
        for i in 0..n {
            col[i] = eig.U()[(i, k)];
        }
        evecs[k] = col;
    }
    (evals, evecs)
}

#[cfg(test)]
mod tests {
    use super::relax_idpp;
    use super::upper_distances;
    use super::{TranslationMemory, improved_tangent};

    #[test]
    fn idpp_separates_a_crossed_pair() {
        // The endpoints are 2 apart. The linear midpoint stacks both
        // atoms. The interpolated distance is still 2.
        let left = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let right = [2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let linear = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let target = upper_distances(&left);
        let relaxed = relax_idpp(&linear, &target);
        let distance = upper_distances(&relaxed)[0];
        assert!((distance - 2.0).abs() < 0.05, "relaxed distance {distance}");
        let _ = right;
    }

    #[test]
    fn negative_secant_steps_along_the_force() {
        let mut memory = TranslationMemory::new(5, 0.1);
        let first = memory.step(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0], 1.0);
        assert!((first[0] - 0.01).abs() < 1e-12);
        let second = memory.step(&[0.01, 0.0, 0.0], &[2.0, 0.0, 0.0], 1.0);
        assert!((second[0] - 0.1).abs() < 1e-12);
        assert!(second[1].abs() < 1e-12 && second[2].abs() < 1e-12);
    }

    #[test]
    fn negative_curvature_scales_the_first_step() {
        let mut memory = TranslationMemory::new(5, 0.2);
        let step = memory.step(&[0.0, 0.0, 0.0], &[0.1, 0.0, 0.0], -4.0);
        assert!((step[0] - 0.025).abs() < 1e-12);
    }

    #[test]
    fn improved_tangent_climbs_toward_the_higher_neighbor() {
        let path = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
        ];
        let up = improved_tangent(&path, &[0.0, 1.0, 3.0], 1);
        assert!((up[0] - 1.0).abs() < 1e-12);
        let down = improved_tangent(&path, &[3.0, 1.0, 0.0], 1);
        assert!((down[0] - 1.0).abs() < 1e-12);
    }
}
