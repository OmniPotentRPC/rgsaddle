//! A direct saddle for one hop of a defect next to another.
//!
//! The search is the fixed-cell minimum-mode dimer. The seed is the
//! isolated hop, placed at the pair separation. Both ends of the
//! unstable mode are quenched. A result is the hop only when those
//! quenches are the two intended minima and the index is 1. A
//! recombination or a capture is an outcome, not a barrier of the hop.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, FireKind, Method, Oracle, Solver};

use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeStatus, PointSurface};
use crate::solid_dimer::{CellMask, IndexCheck, SolidDimer, SolidDimerConfig};

type M3 = [[f64; 3]; 3];

/// Fixed-cell dimer controls used for a published climb.
#[derive(Clone, Debug, PartialEq)]
pub struct PairOptions {
    pub force_tol: f64,
    pub max_move: f64,
    pub flat: f64,
    pub rotation_tol: f64,
    pub max_rotations: usize,
    pub max_steps: usize,
    /// Displacement from the minimum along the connecting mode.
    pub nudge: f64,
    pub relax_force: f64,
    pub relax_steps: usize,
    /// Last atoms whose order is matched between the two minima.
    pub tail: usize,
    /// Old whole-cell RMS threshold. Classification does not read it.
    /// A few atoms moving by an angstrom in a cell of thousands still
    /// have a small RMS, so the match is [`Self::match_atom`].
    pub match_rms: f64,
    /// Largest per-atom minimum-image move still counted as that minimum.
    pub match_atom: f64,
    /// A quench this far below both minima, matching neither, has recombined.
    pub recombine_drop: f64,
    pub method: Method,
}

impl Default for PairOptions {
    fn default() -> Self {
        Self {
            force_tol: 2.0e-3,
            max_move: 0.008,
            flat: 0.005,
            rotation_tol: 1.0e-3,
            max_rotations: 12,
            max_steps: 800,
            nudge: 0.02,
            relax_force: 1.0e-3,
            relax_steps: 400,
            tail: 0,
            match_rms: 0.2,
            match_atom: 0.8,
            recombine_drop: 0.5,
            method: Method::Fire { kind: FireKind::V2 },
        }
    }
}

impl PairOptions {
    /// The fixed-cell climb: 2e-3 force, 0.008 Å cap, flat 0.005.
    pub fn published() -> Self {
        Self::default()
    }
}

/// Isolated hop: the two minima and the index-1 saddle between them.
///
/// `converged` is true only when the saddle is index 1 and the two
/// quenches of its unstable mode are these two minima.
#[derive(Clone, Debug, PartialEq)]
pub struct HopGeometry {
    pub barrier: f64,
    pub cell: M3,
    pub minimum: Array1<f64>,
    pub saddle: Array1<f64>,
    pub final_min: Array1<f64>,
    pub mode: Array1<f64>,
    pub negative_modes: Option<usize>,
    pub force_calls: usize,
    pub converged: bool,
}

/// The isolated geometry moved onto a pair, with one atom removed.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedPair {
    pub initial: Array1<f64>,
    pub saddle: Array1<f64>,
    pub final_min: Array1<f64>,
    pub mode: Array1<f64>,
    pub dropped_distance: f64,
}

/// Which basin a quench of one side of the mode reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairEnd {
    InitialMinimum,
    FinalMinimum,
    Recombined,
    Captured,
    OffPathway,
}

/// Why the climb is or is not the requested hop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairOutcome {
    /// Index 1, and the two quenches are the two intended minima.
    Hop,
    /// An end annihilated the pair.
    Recombined,
    /// An end fell into a supplied capture basin.
    Captured,
    /// The climb returned to one minimum.
    ReturnedToMinimum,
    /// The stationary point is not index 1.
    NotIndexOne,
    /// The dimer was still running at the step cap, or was rejected.
    DidNotConverge,
    /// An end is neither intended minimum nor a supplied capture.
    OffPathway,
}

/// Direct saddle of one hop in a fixed cell.
#[derive(Clone, Debug, PartialEq)]
pub struct PairSaddle {
    pub barrier: f64,
    pub saddle_geometry: Array1<f64>,
    pub initial_min: Array1<f64>,
    pub final_min: Array1<f64>,
    pub lowest_mode: Array1<f64>,
    pub force_calls: usize,
    pub converged: bool,
    pub index: Option<usize>,
    pub outcome: PairOutcome,
    pub end_forward: PairEnd,
    pub end_backward: PairEnd,
    /// Largest per-atom minimum-image distance of each quench from each minimum.
    /// This, compared with [`PairOptions::match_atom`], is the match.
    pub forward_atom_initial: f64,
    pub forward_atom_final: f64,
    pub backward_atom_initial: f64,
    pub backward_atom_final: f64,
    /// RMS of the minimum-image displacement over the atoms that moved,
    /// for the same four comparisons. Not used to classify the end.
    pub forward_moved_rms_initial: f64,
    pub forward_moved_rms_final: f64,
    pub backward_moved_rms_initial: f64,
    pub backward_moved_rms_final: f64,
}

struct Counted<'a, S> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval_in_cell(x, cell)
    }
}

/// Climb from a nudged minimum to the saddle that connects two minima.
pub fn isolated_saddle<S: PointSurface>(
    surface: &S,
    cell: M3,
    mut initial: Array1<f64>,
    mut final_pos: Array1<f64>,
    options: &PairOptions,
) -> Result<HopGeometry, SaddleError> {
    require_match_options(options)?;
    if initial.len() != final_pos.len() || !initial.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "the two minima need the same Cartesian length".into(),
        ));
    }
    let counted = Counted {
        inner: surface,
        calls: AtomicUsize::new(0),
    };
    if options.relax_steps > 0 {
        relax(
            &counted,
            &mut initial,
            options.relax_force,
            options.relax_steps,
        )?;
        relax(
            &counted,
            &mut final_pos,
            options.relax_force,
            options.relax_steps,
        )?;
    }
    if options.tail > 0 {
        align_tail(&initial, &mut final_pos, cell, options.tail)?;
    }
    let (e_min, _) = counted.eval(initial.view())?;
    let connect = mic_difference(&final_pos, &initial, cell);
    let norm = connect.dot(&connect).sqrt();
    if !(norm.is_finite() && norm > 1.0e-14) {
        return Err(SaddleError::Invalid(
            "the two minima are the same point".into(),
        ));
    }
    let mut start = initial.clone();
    if options.nudge > 0.0 {
        let step = options.nudge / norm;
        for (slot, component) in start.iter_mut().zip(connect.iter()) {
            *slot += step * component;
        }
    }
    let found = climb(&counted, cell, start, connect, e_min, options)?;
    let ends = classify_saddle(
        &counted,
        cell,
        &found.positions,
        &found.mode,
        &initial,
        &final_pos,
        options,
    )?;
    let connected = found.index_one
        && matches!(
            (ends.forward, ends.backward),
            (PairEnd::InitialMinimum, PairEnd::FinalMinimum)
                | (PairEnd::FinalMinimum, PairEnd::InitialMinimum)
        );
    let calls = counted.calls.load(Ordering::Relaxed);
    Ok(HopGeometry {
        barrier: found.barrier,
        cell,
        minimum: initial,
        saddle: found.positions,
        final_min: final_pos,
        mode: found.mode,
        negative_modes: found.negative_modes,
        force_calls: calls,
        converged: connected,
    })
}

impl HopGeometry {
    /// JSON document a consumer can load instead of climbing again.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"format\": \"rgsaddle.hop_geometry\",\n");
        out.push_str("  \"version\": 1,\n");
        out.push_str(&format!("  \"barrier_eV\": {},\n", self.barrier));
        out.push_str(&format!(
            "  \"converged\": {},\n",
            if self.converged { "true" } else { "false" }
        ));
        out.push_str("  \"negative_modes\": ");
        match self.negative_modes {
            Some(value) => out.push_str(&value.to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\n");
        out.push_str(&format!("  \"force_calls\": {},\n", self.force_calls));
        out.push_str("  \"cell\": ");
        push_cell(&mut out, self.cell);
        out.push_str(",\n  \"minimum\": ");
        push_array(&mut out, &self.minimum);
        out.push_str(",\n  \"saddle\": ");
        push_array(&mut out, &self.saddle);
        out.push_str(",\n  \"final\": ");
        push_array(&mut out, &self.final_min);
        out.push_str(",\n  \"mode\": ");
        push_array(&mut out, &self.mode);
        out.push_str("\n}\n");
        out
    }

    /// The document written by [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        if !text.contains("\"format\": \"rgsaddle.hop_geometry\"") {
            return Err(SaddleError::Invalid(
                "not an rgsaddle hop geometry document".into(),
            ));
        }
        let barrier = field_f64(text, "barrier_eV")?;
        let force_calls = field_f64(text, "force_calls")? as usize;
        let converged = text.contains("\"converged\": true");
        let negative_modes = if text.contains("\"negative_modes\": null") {
            None
        } else {
            Some(field_f64(text, "negative_modes")? as usize)
        };
        let cell = field_cell(text)?;
        let minimum = field_array(text, "minimum")?;
        let saddle = field_array(text, "saddle")?;
        let final_min = field_array(text, "final")?;
        let mode = field_array(text, "mode")?;
        if minimum.len() != saddle.len() || minimum.len() != final_min.len() {
            return Err(SaddleError::Shape(
                "the stored minima and saddle differ in length".into(),
            ));
        }
        Ok(Self {
            barrier,
            cell,
            minimum,
            saddle,
            final_min,
            mode,
            negative_modes,
            force_calls,
            converged,
        })
    }

    /// Reflect the hop so the saddle axis points to the same side of
    /// `axis × jump`. A negative return means the geometry was mirrored.
    /// A zero jump, as in an on-site rotation, is left alone.
    pub fn canonicalize_branch(&mut self, axis: [f64; 3], jump: [f64; 3]) -> i32 {
        let axis = unit3(axis);
        let jump_u = unit3(jump);
        let normal = cross(axis, jump_u);
        let normal_len = length3(normal);
        if normal_len < 1.0e-8 {
            return 0;
        }
        let normal = [
            normal[0] / normal_len,
            normal[1] / normal_len,
            normal[2] / normal_len,
        ];
        let mut sad = closest_axis(&self.saddle, self.cell);
        if dot3(sad, axis) < 0.0 {
            sad = [-sad[0], -sad[1], -sad[2]];
        }
        if dot3(sad, normal) >= 0.0 {
            return 1;
        }
        reflect_positions(&mut self.minimum, normal, self.cell);
        reflect_positions(&mut self.saddle, normal, self.cell);
        reflect_positions(&mut self.final_min, normal, self.cell);
        self.mode = mic_difference(&self.saddle, &self.minimum, self.cell);
        -1
    }

    /// Rotate about the origin, translate, and delete the atom nearest `drop_near`.
    pub fn place(
        &self,
        rotation: M3,
        shift: [f64; 3],
        cell: M3,
        length: f64,
        drop_near: [f64; 3],
        max_drop: f64,
    ) -> Result<PlacedPair, SaddleError> {
        let moved_min = rigid_place(&self.minimum, rotation, shift, length);
        let moved_sad = rigid_place(&self.saddle, rotation, shift, length);
        let moved_fin = rigid_place(&self.final_min, rotation, shift, length);
        let index = pbc_nearest(&moved_min, drop_near, cell);
        let here = atom_at(&moved_min, index);
        let dropped_distance = mic_length(here, drop_near, cell);
        if !(dropped_distance.is_finite() && dropped_distance <= max_drop) {
            return Err(SaddleError::Invalid(format!(
                "the deleted atom is {dropped_distance:.3} A from the neighbour defect"
            )));
        }
        let initial = drop_atom(&moved_min, index);
        let saddle = drop_atom(&moved_sad, index);
        let final_min = drop_atom(&moved_fin, index);
        let mode = mic_difference(&saddle, &initial, cell);
        Ok(PlacedPair {
            initial,
            saddle,
            final_min,
            mode,
            dropped_distance,
        })
    }
}

/// Quench both sides of an existing saddle and name the minima.
pub fn classify_saddle<S: PointSurface>(
    surface: &S,
    cell: M3,
    saddle: &Array1<f64>,
    mode: &Array1<f64>,
    initial: &Array1<f64>,
    final_min: &Array1<f64>,
    options: &PairOptions,
) -> Result<EndClassification, SaddleError> {
    require_match_options(options)?;
    let counted = Counted {
        inner: surface,
        calls: AtomicUsize::new(0),
    };
    let (e_min, _) = counted.eval(initial.view())?;
    let (e_final, _) = counted.eval(final_min.view())?;
    let (forward_pos, forward_energy) = quench_side(&counted, saddle, mode, 1.0)?;
    let (backward_pos, backward_energy) = quench_side(&counted, saddle, mode, -1.0)?;
    let forward_initial = geometry_separation(&forward_pos, initial, cell)?;
    let forward_final = geometry_separation(&forward_pos, final_min, cell)?;
    let backward_initial = geometry_separation(&backward_pos, initial, cell)?;
    let backward_final = geometry_separation(&backward_pos, final_min, cell)?;
    Ok(EndClassification {
        forward: classify_end(
            forward_initial.max_atom,
            forward_final.max_atom,
            forward_energy,
            e_min,
            e_final,
            options,
        ),
        backward: classify_end(
            backward_initial.max_atom,
            backward_final.max_atom,
            backward_energy,
            e_min,
            e_final,
            options,
        ),
        forward_initial: forward_initial.max_atom,
        forward_final: forward_final.max_atom,
        backward_initial: backward_initial.max_atom,
        backward_final: backward_final.max_atom,
        forward_moved_rms_initial: forward_initial.moved_rms,
        forward_moved_rms_final: forward_final.moved_rms,
        backward_moved_rms_initial: backward_initial.moved_rms,
        backward_moved_rms_final: backward_final.moved_rms,
        forward_frame: closest_frame(&forward_pos, cell),
        backward_frame: closest_frame(&backward_pos, cell),
    })
}

/// Which minimum each side of a saddle fell into.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndClassification {
    pub forward: PairEnd,
    pub backward: PairEnd,
    pub forward_initial: f64,
    pub forward_final: f64,
    pub backward_initial: f64,
    pub backward_final: f64,
    /// RMS over the atoms that moved. Not the match.
    pub forward_moved_rms_initial: f64,
    pub forward_moved_rms_final: f64,
    pub backward_moved_rms_initial: f64,
    pub backward_moved_rms_final: f64,
    /// Closest-pair centre, axis and split of each quench.
    pub forward_frame: ([f64; 3], [f64; 3], f64),
    pub backward_frame: ([f64; 3], [f64; 3], f64),
}

/// Climb from `placed.saddle` and quench both sides of the mode.
pub fn pair_saddle_search<S: PointSurface>(
    surface: &S,
    cell: M3,
    placed: &PlacedPair,
    options: &PairOptions,
) -> Result<PairSaddle, SaddleError> {
    require_match_options(options)?;
    let counted = Counted {
        inner: surface,
        calls: AtomicUsize::new(0),
    };
    let mut initial = placed.initial.clone();
    let mut final_min = placed.final_min.clone();
    if options.relax_steps > 0 {
        relax(
            &counted,
            &mut initial,
            options.relax_force,
            options.relax_steps,
        )?;
        relax(
            &counted,
            &mut final_min,
            options.relax_force,
            options.relax_steps,
        )?;
    }
    let (e_min, _) = counted.eval(initial.view())?;
    let (e_final, _) = counted.eval(final_min.view())?;
    let found = climb(
        &counted,
        cell,
        placed.saddle.clone(),
        placed.mode.clone(),
        e_min,
        options,
    )?;
    let (forward_pos, forward_energy) = quench_side(&counted, &found.positions, &found.mode, 1.0)?;
    let (backward_pos, backward_energy) =
        quench_side(&counted, &found.positions, &found.mode, -1.0)?;
    let forward_initial = geometry_separation(&forward_pos, &initial, cell)?;
    let forward_final = geometry_separation(&forward_pos, &final_min, cell)?;
    let backward_initial = geometry_separation(&backward_pos, &initial, cell)?;
    let backward_final = geometry_separation(&backward_pos, &final_min, cell)?;
    let end_forward = classify_end(
        forward_initial.max_atom,
        forward_final.max_atom,
        forward_energy,
        e_min,
        e_final,
        options,
    );
    let end_backward = classify_end(
        backward_initial.max_atom,
        backward_final.max_atom,
        backward_energy,
        e_min,
        e_final,
        options,
    );
    let outcome = outcome_of(
        found.finished,
        found.index_one,
        end_forward,
        end_backward,
        found.barrier,
        &found.positions,
        &initial,
        cell,
        options,
    );
    Ok(PairSaddle {
        barrier: found.barrier,
        saddle_geometry: found.positions,
        initial_min: initial,
        final_min,
        lowest_mode: found.mode,
        force_calls: counted.calls.load(Ordering::Relaxed),
        converged: outcome == PairOutcome::Hop,
        index: found.negative_modes,
        outcome,
        end_forward,
        end_backward,
        forward_atom_initial: forward_initial.max_atom,
        forward_atom_final: forward_final.max_atom,
        backward_atom_initial: backward_initial.max_atom,
        backward_atom_final: backward_final.max_atom,
        forward_moved_rms_initial: forward_initial.moved_rms,
        forward_moved_rms_final: forward_final.moved_rms,
        backward_moved_rms_initial: backward_initial.moved_rms,
        backward_moved_rms_final: backward_final.moved_rms,
    })
}

struct Climbed {
    positions: Array1<f64>,
    mode: Array1<f64>,
    barrier: f64,
    negative_modes: Option<usize>,
    finished: bool,
    index_one: bool,
}

fn climb<S: PointSurface>(
    surface: &S,
    cell: M3,
    start: Array1<f64>,
    mode: Array1<f64>,
    minimum_energy: f64,
    options: &PairOptions,
) -> Result<Climbed, SaddleError> {
    let search = MinModeConfig {
        method: options.method.clone(),
        force_tol: options.force_tol,
        max_move: options.max_move,
        dr: 1.0e-3,
        rotation_tol: options.rotation_tol,
        max_rotations: options.max_rotations,
        seed_curvature: false,
        ..MinModeConfig::default()
    };
    let mut dimer = SolidDimer::new(
        SolidDimerConfig::new(search)
            .with_mask(CellMask::frozen())
            .with_external_stress([[0.0; 3]; 3])
            .with_index_check(IndexCheck::first_order())
            .with_flat(options.flat),
        start,
        cell,
        mode,
    )?;
    let mut report = dimer.step(surface)?;
    let mut steps = 1usize;
    while report.status == MinModeStatus::Running && steps < options.max_steps {
        report = dimer.step(surface)?;
        steps += 1;
    }
    let (energy, _) = surface.eval(dimer.positions())?;
    let barrier = energy - minimum_energy;
    if !barrier.is_finite() {
        return Err(SaddleError::NonFinite("pair barrier"));
    }
    let finished = report.status == MinModeStatus::Converged && !report.rejected;
    let index_one = finished && report.curvature < 0.0 && report.negative_modes == Some(1);
    Ok(Climbed {
        positions: dimer.positions().to_owned(),
        mode: dimer.mode().to_owned(),
        barrier,
        negative_modes: report.negative_modes,
        finished,
        index_one,
    })
}

fn relax<S: PointSurface>(
    surface: &S,
    positions: &mut Array1<f64>,
    force_tol: f64,
    max_steps: usize,
) -> Result<(), SaddleError> {
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..Control::default()
        },
        positions.len(),
    );
    solver.set_atom_maxmove(0.06);
    for _ in 0..max_steps {
        let oracle = Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
            surface
                .eval(x)
                .unwrap_or_else(|_| (f64::INFINITY, Array1::zeros(x.len())))
        });
        solver
            .step(&oracle, positions)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        let (_, gradient) = surface.eval(positions.view())?;
        let max_component = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
        if max_component < force_tol {
            break;
        }
    }
    Ok(())
}

fn quench_side<S: PointSurface>(
    surface: &S,
    saddle: &Array1<f64>,
    mode: &Array1<f64>,
    sign: f64,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let norm = mode.dot(mode).sqrt();
    if !(norm.is_finite() && norm > 1.0e-14) {
        return Err(SaddleError::Invalid(
            "the saddle has no unstable mode to quench".into(),
        ));
    }
    let mut trial = saddle + &(mode * (sign * 0.25 / norm));
    relax(surface, &mut trial, 1.0e-2, 80)?;
    let (energy, _) = surface.eval(trial.view())?;
    Ok((trial, energy))
}

fn require_match_options(options: &PairOptions) -> Result<(), SaddleError> {
    if !(options.match_atom.is_finite() && options.match_atom > 0.0) {
        return Err(SaddleError::Invalid(
            "match_atom must be a positive length".into(),
        ));
    }
    if !(options.match_rms.is_finite() && options.match_rms > 0.0) {
        return Err(SaddleError::Invalid(
            "match_rms must be a positive length; the match is match_atom".into(),
        ));
    }
    Ok(())
}

fn classify_end(
    to_initial: f64,
    to_final: f64,
    energy: f64,
    e_initial: f64,
    e_final: f64,
    options: &PairOptions,
) -> PairEnd {
    let initial_hit = to_initial <= to_final && to_initial < options.match_atom;
    let final_hit = to_final < to_initial && to_final < options.match_atom;
    if initial_hit && (energy - e_initial).abs() < 0.05 {
        return PairEnd::InitialMinimum;
    }
    if final_hit && (energy - e_final).abs() < 0.05 {
        return PairEnd::FinalMinimum;
    }
    let floor = e_initial.min(e_final);
    if energy < floor - options.recombine_drop {
        return PairEnd::Recombined;
    }
    if initial_hit || final_hit {
        return PairEnd::Captured;
    }
    PairEnd::OffPathway
}

#[allow(clippy::too_many_arguments)]
fn outcome_of(
    finished: bool,
    index_one: bool,
    forward: PairEnd,
    backward: PairEnd,
    barrier: f64,
    saddle: &Array1<f64>,
    initial: &Array1<f64>,
    cell: M3,
    options: &PairOptions,
) -> PairOutcome {
    if !finished {
        return PairOutcome::DidNotConverge;
    }
    if !index_one {
        return PairOutcome::NotIndexOne;
    }
    let ends = [forward, backward];
    if ends.contains(&PairEnd::Recombined) {
        return PairOutcome::Recombined;
    }
    if ends.contains(&PairEnd::Captured) {
        return PairOutcome::Captured;
    }
    let has_initial = ends.contains(&PairEnd::InitialMinimum);
    let has_final = ends.contains(&PairEnd::FinalMinimum);
    if has_initial && has_final {
        return PairOutcome::Hop;
    }
    if barrier.abs() < 0.02 && mic_max_atom(saddle, initial, cell) < options.match_atom {
        return PairOutcome::ReturnedToMinimum;
    }
    if has_initial || has_final {
        return PairOutcome::ReturnedToMinimum;
    }
    PairOutcome::OffPathway
}

/// Match the order of the last `tail` atoms so the mode is the hop.
pub fn align_tail(
    initial: &Array1<f64>,
    final_pos: &mut Array1<f64>,
    cell: M3,
    tail: usize,
) -> Result<(), SaddleError> {
    let n = initial.len() / 3;
    if tail == 0 || tail > n || final_pos.len() / 3 != n {
        return Err(SaddleError::Shape(
            "the tail is longer than the configuration".into(),
        ));
    }
    let start = n - tail;
    let mut atoms = Vec::new();
    for atom in start..n {
        atoms.push(atom_at(final_pos, atom));
    }
    let mut order: Vec<usize> = (0..tail).collect();
    let mut best_order = order.clone();
    let mut best = f64::MAX;
    permute(
        &mut order,
        0,
        initial,
        &atoms,
        cell,
        start,
        &mut best,
        &mut best_order,
    );
    for (slot, index) in best_order.iter().enumerate() {
        let point = atoms[*index];
        for c in 0..3 {
            final_pos[3 * (start + slot) + c] = point[c];
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn permute(
    order: &mut Vec<usize>,
    start: usize,
    initial: &Array1<f64>,
    atoms: &[[f64; 3]],
    cell: M3,
    body: usize,
    best: &mut f64,
    best_order: &mut Vec<usize>,
) {
    if start == order.len() {
        let mut cost = 0.0;
        for (slot, index) in order.iter().enumerate() {
            cost += mic_length(atom_at(initial, body + slot), atoms[*index], cell);
        }
        if cost < *best {
            *best = cost;
            *best_order = order.clone();
        }
        return;
    }
    for i in start..order.len() {
        order.swap(start, i);
        permute(
            order,
            start + 1,
            initial,
            atoms,
            cell,
            body,
            best,
            best_order,
        );
        order.swap(start, i);
    }
}

fn rigid_place(positions: &Array1<f64>, rotation: M3, shift: [f64; 3], length: f64) -> Array1<f64> {
    let mut out = positions.clone();
    for atom in 0..positions.len() / 3 {
        let v = atom_at(positions, atom);
        let turned = [
            rotation[0][0] * v[0] + rotation[0][1] * v[1] + rotation[0][2] * v[2],
            rotation[1][0] * v[0] + rotation[1][1] * v[1] + rotation[1][2] * v[2],
            rotation[2][0] * v[0] + rotation[2][1] * v[1] + rotation[2][2] * v[2],
        ];
        let shifted = [
            turned[0] + shift[0],
            turned[1] + shift[1],
            turned[2] + shift[2],
        ];
        let wrapped = wrap_cubic(shifted, length);
        for c in 0..3 {
            out[3 * atom + c] = wrapped[c];
        }
    }
    out
}

fn wrap_cubic(point: [f64; 3], length: f64) -> [f64; 3] {
    point.map(|value| {
        let mut wrapped = value % length;
        if wrapped < 0.0 {
            wrapped += length;
        }
        wrapped
    })
}

fn pbc_nearest(positions: &Array1<f64>, point: [f64; 3], cell: M3) -> usize {
    let mut best = 0usize;
    let mut best_d = f64::MAX;
    for atom in 0..positions.len() / 3 {
        let distance = mic_length(atom_at(positions, atom), point, cell);
        if distance < best_d {
            best_d = distance;
            best = atom;
        }
    }
    best
}

fn drop_atom(positions: &Array1<f64>, index: usize) -> Array1<f64> {
    let mut kept = Vec::with_capacity(positions.len() - 3);
    for atom in 0..positions.len() / 3 {
        if atom == index {
            continue;
        }
        let point = atom_at(positions, atom);
        kept.extend_from_slice(&point);
    }
    Array1::from(kept)
}

fn atom_at(positions: &Array1<f64>, atom: usize) -> [f64; 3] {
    [
        positions[3 * atom],
        positions[3 * atom + 1],
        positions[3 * atom + 2],
    ]
}

fn mic_difference(target: &Array1<f64>, origin: &Array1<f64>, cell: M3) -> Array1<f64> {
    let mut mode = target - origin;
    let n = mode.len() / 3;
    for atom in 0..n {
        let delta = mic_delta(atom_at(target, atom), atom_at(origin, atom), cell);
        for c in 0..3 {
            mode[3 * atom + c] = delta[c];
        }
    }
    mode
}

/// An atom this far from its image, in angstroms, counts as moved for the
/// diagnostic RMS. The match itself is the largest displacement.
const MOVED_FLOOR_A: f64 = 1.0e-4;

/// Per-atom minimum-image separation of two geometries in the same order.
///
/// `max_atom` is the match. `moved_rms` is the RMS over atoms that moved
/// by more than [`MOVED_FLOOR_A`], so a handful of displaced atoms in a
/// large cell are not diluted by the atoms that did not move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometrySeparation {
    pub max_atom: f64,
    pub moved_rms: f64,
    pub moved_atoms: usize,
}

pub fn geometry_separation(
    left: &Array1<f64>,
    right: &Array1<f64>,
    cell: M3,
) -> Result<GeometrySeparation, SaddleError> {
    let n = left.len() / 3;
    if n == 0 || right.len() / 3 != n {
        return Err(SaddleError::Shape(
            "the two geometries need the same number of atoms".into(),
        ));
    }
    let mut max_atom = 0.0;
    let mut sum = 0.0;
    let mut moved_atoms = 0usize;
    for atom in 0..n {
        let delta = mic_delta(atom_at(left, atom), atom_at(right, atom), cell);
        let length = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if length > max_atom {
            max_atom = length;
        }
        if length > MOVED_FLOOR_A {
            sum += length * length;
            moved_atoms += 1;
        }
    }
    let moved_rms = if moved_atoms == 0 {
        0.0
    } else {
        (sum / moved_atoms as f64).sqrt()
    };
    Ok(GeometrySeparation {
        max_atom,
        moved_rms,
        moved_atoms,
    })
}

fn mic_max_atom(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> f64 {
    let n = left.len() / 3;
    if n == 0 || right.len() / 3 != n {
        return f64::MAX;
    }
    let mut best = 0.0;
    for atom in 0..n {
        let d = mic_delta(atom_at(left, atom), atom_at(right, atom), cell);
        let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if length > best {
            best = length;
        }
    }
    best
}

#[cfg(test)]
fn mic_rms(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> f64 {
    let n = left.len() / 3;
    if n == 0 || right.len() / 3 != n {
        return f64::MAX;
    }
    let mut sum = 0.0;
    for atom in 0..n {
        let d = mic_delta(atom_at(left, atom), atom_at(right, atom), cell);
        sum += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    }
    (sum / n as f64).sqrt()
}

fn mic_length(point: [f64; 3], origin: [f64; 3], cell: M3) -> f64 {
    let d = mic_delta(point, origin, cell);
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}

fn closest_frame(positions: &Array1<f64>, cell: M3) -> ([f64; 3], [f64; 3], f64) {
    let n = positions.len() / 3;
    let mut best = f64::MAX;
    let mut pair = (0usize, 1usize);
    let mut delta = [1.0, 0.0, 0.0];
    for i in 0..n {
        for j in i + 1..n {
            let step = mic_delta(atom_at(positions, j), atom_at(positions, i), cell);
            let length = length3(step);
            if length < best {
                best = length;
                pair = (i, j);
                delta = step;
            }
        }
    }
    let origin = atom_at(positions, pair.0);
    let center = [
        origin[0] + 0.5 * delta[0],
        origin[1] + 0.5 * delta[1],
        origin[2] + 0.5 * delta[2],
    ];
    (center, unit3(delta), best)
}

fn closest_axis(positions: &Array1<f64>, cell: M3) -> [f64; 3] {
    let n = positions.len() / 3;
    let mut best = f64::MAX;
    let mut axis = [1.0, 0.0, 0.0];
    for i in 0..n {
        for j in i + 1..n {
            let delta = mic_delta(atom_at(positions, j), atom_at(positions, i), cell);
            let length = length3(delta);
            if length < best {
                best = length;
                axis = delta;
            }
        }
    }
    unit3(axis)
}

fn reflect_positions(positions: &mut Array1<f64>, normal: [f64; 3], cell: M3) {
    for atom in 0..positions.len() / 3 {
        let delta = mic_delta(atom_at(positions, atom), [0.0, 0.0, 0.0], cell);
        let along = dot3(delta, normal);
        let reflected = [
            delta[0] - 2.0 * along * normal[0],
            delta[1] - 2.0 * along * normal[1],
            delta[2] - 2.0 * along * normal[2],
        ];
        let wrapped = wrap_into(reflected, cell);
        for c in 0..3 {
            positions[3 * atom + c] = wrapped[c];
        }
    }
}

fn wrap_into(point: [f64; 3], cell: M3) -> [f64; 3] {
    let inv = invert(cell);
    let s = [
        point[0] * inv[0][0] + point[1] * inv[1][0] + point[2] * inv[2][0],
        point[0] * inv[0][1] + point[1] * inv[1][1] + point[2] * inv[2][1],
        point[0] * inv[0][2] + point[1] * inv[1][2] + point[2] * inv[2][2],
    ];
    let w = [
        s[0] - s[0].floor(),
        s[1] - s[1].floor(),
        s[2] - s[2].floor(),
    ];
    [
        w[0] * cell[0][0] + w[1] * cell[1][0] + w[2] * cell[2][0],
        w[0] * cell[0][1] + w[1] * cell[1][1] + w[2] * cell[2][1],
        w[0] * cell[0][2] + w[1] * cell[1][2] + w[2] * cell[2][2],
    ]
}

fn unit3(v: [f64; 3]) -> [f64; 3] {
    let n = length3(v).max(1.0e-30);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn length3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn mic_delta(point: [f64; 3], origin: [f64; 3], cell: M3) -> [f64; 3] {
    let inv = invert(cell);
    let d = [
        point[0] - origin[0],
        point[1] - origin[1],
        point[2] - origin[2],
    ];
    let s = [
        d[0] * inv[0][0] + d[1] * inv[1][0] + d[2] * inv[2][0],
        d[0] * inv[0][1] + d[1] * inv[1][1] + d[2] * inv[2][1],
        d[0] * inv[0][2] + d[1] * inv[1][2] + d[2] * inv[2][2],
    ];
    let w = [
        s[0] - s[0].round(),
        s[1] - s[1].round(),
        s[2] - s[2].round(),
    ];
    [
        w[0] * cell[0][0] + w[1] * cell[1][0] + w[2] * cell[2][0],
        w[0] * cell[0][1] + w[1] * cell[1][1] + w[2] * cell[2][1],
        w[0] * cell[0][2] + w[1] * cell[1][2] + w[2] * cell[2][2],
    ]
}

fn invert(h: M3) -> M3 {
    let d = det(h);
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (h[1][1] * h[2][2] - h[1][2] * h[2][1]) / d;
    c[0][1] = (h[0][2] * h[2][1] - h[0][1] * h[2][2]) / d;
    c[0][2] = (h[0][1] * h[1][2] - h[0][2] * h[1][1]) / d;
    c[1][0] = (h[1][2] * h[2][0] - h[1][0] * h[2][2]) / d;
    c[1][1] = (h[0][0] * h[2][2] - h[0][2] * h[2][0]) / d;
    c[1][2] = (h[0][2] * h[1][0] - h[0][0] * h[1][2]) / d;
    c[2][0] = (h[1][0] * h[2][1] - h[1][1] * h[2][0]) / d;
    c[2][1] = (h[0][1] * h[2][0] - h[0][0] * h[2][1]) / d;
    c[2][2] = (h[0][0] * h[1][1] - h[0][1] * h[1][0]) / d;
    c
}

fn det(h: M3) -> f64 {
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

fn push_array(out: &mut String, values: &Array1<f64>) {
    out.push('[');
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!("{value}"));
    }
    out.push(']');
}

fn push_cell(out: &mut String, cell: M3) {
    out.push('[');
    for (i, row) in cell.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        for (j, value) in row.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            out.push_str(&format!("{value}"));
        }
        out.push(']');
    }
    out.push(']');
}

fn field_f64(text: &str, key: &str) -> Result<f64, SaddleError> {
    let marker = format!("\"{key}\":");
    let Some(start) = text.find(&marker) else {
        return Err(SaddleError::Invalid(format!("hop geometry lacks {key}")));
    };
    let rest = text[start + marker.len()..].trim_start();
    let end = rest
        .find([',', '\n', '}'])
        .ok_or_else(|| SaddleError::Invalid(format!("hop geometry {key} is truncated")))?;
    rest[..end]
        .trim()
        .parse()
        .map_err(|_| SaddleError::Invalid(format!("hop geometry {key} is not a number")))
}

fn field_array(text: &str, key: &str) -> Result<Array1<f64>, SaddleError> {
    let marker = format!("\"{key}\":");
    let Some(start) = text.find(&marker) else {
        return Err(SaddleError::Invalid(format!("hop geometry lacks {key}")));
    };
    let rest = text[start + marker.len()..].trim_start();
    if !rest.starts_with('[') {
        return Err(SaddleError::Invalid(format!(
            "hop geometry {key} is not an array"
        )));
    }
    let Some(end) = rest.find(']') else {
        return Err(SaddleError::Invalid(format!(
            "hop geometry {key} is truncated"
        )));
    };
    let mut values = Vec::new();
    for piece in rest[1..end].split(',') {
        let piece = piece.trim();
        if piece.is_empty() {
            continue;
        }
        values.push(
            piece
                .parse()
                .map_err(|_| SaddleError::Invalid(format!("hop geometry {key} has a bad entry")))?,
        );
    }
    Ok(Array1::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hop_geometry_json_round_trips() {
        let geometry = HopGeometry {
            barrier: 0.1,
            cell: [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
            minimum: Array1::from(vec![0.0, 0.0, 0.0]),
            saddle: Array1::from(vec![0.1, 0.0, 0.0]),
            final_min: Array1::from(vec![0.2, 0.0, 0.0]),
            mode: Array1::from(vec![0.1, 0.0, 0.0]),
            negative_modes: Some(1),
            force_calls: 4,
            converged: true,
        };
        let loaded = HopGeometry::from_json(&geometry.to_json()).unwrap();
        assert!((loaded.barrier - 0.1).abs() < 1.0e-12);
        assert_eq!(loaded.force_calls, 4);
        assert_eq!(loaded.negative_modes, Some(1));
        assert!(loaded.converged);
        assert!((loaded.saddle[0] - 0.1).abs() < 1.0e-12);
    }

    #[test]
    fn a_few_moved_atoms_in_a_large_cell_do_not_match() {
        let n = 2048usize;
        let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
        let reference = Array1::zeros(n * 3);
        let mut product = reference.clone();
        for atom in 0..3 {
            product[3 * atom] += 2.0;
        }
        let whole = mic_rms(&product, &reference, cell);
        assert!(
            whole < 0.2,
            "whole-cell RMS {whole} would have been called the same minimum"
        );
        let separation = geometry_separation(&product, &reference, cell).unwrap();
        assert_eq!(separation.moved_atoms, 3);
        assert!((separation.max_atom - 2.0).abs() < 1.0e-12);
        assert!((separation.moved_rms - 2.0).abs() < 1.0e-12);
        let mut options = PairOptions::published();
        options.match_atom = 0.2;
        let end = classify_end(separation.max_atom, 10.0, 1.0, 1.0, 1.0, &options);
        assert_ne!(end, PairEnd::FinalMinimum);
        assert_ne!(end, PairEnd::InitialMinimum);
        assert_eq!(end, PairEnd::OffPathway);
    }
}

fn field_cell(text: &str) -> Result<M3, SaddleError> {
    let marker = "\"cell\":";
    let Some(start) = text.find(marker) else {
        return Err(SaddleError::Invalid("hop geometry lacks a cell".into()));
    };
    let rest = text[start + marker.len()..].trim_start();
    let mut rows = [[0.0; 3]; 3];
    let open = rest
        .find('[')
        .ok_or_else(|| SaddleError::Invalid("hop geometry cell is truncated".into()))?;
    let mut cursor = &rest[open + 1..];
    for row in &mut rows {
        let open = cursor
            .find('[')
            .ok_or_else(|| SaddleError::Invalid("hop geometry cell is truncated".into()))?;
        cursor = &cursor[open + 1..];
        let close = cursor
            .find(']')
            .ok_or_else(|| SaddleError::Invalid("hop geometry cell is truncated".into()))?;
        let mut numbers = cursor[..close].split(',');
        for slot in row.iter_mut() {
            let piece = numbers
                .next()
                .ok_or_else(|| SaddleError::Invalid("hop geometry cell is short".into()))?;
            *slot = piece
                .trim()
                .parse()
                .map_err(|_| SaddleError::Invalid("hop geometry cell has a bad entry".into()))?;
        }
        cursor = &cursor[close + 1..];
    }
    Ok(rows)
}
