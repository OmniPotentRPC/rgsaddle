//! Weights for the convex min-mode climb.
//!
//! The weights multiply the climb step and nothing else. The mode, its
//! curvature, the dimer rotation, and the rotation memory are the
//! unweighted ones, so a weight does not clear them. The concave step
//! is the reflected force on the whole space.
//!
//! Two constructions are here. eOn's `confine_positive` replaces the
//! positive-curvature force by a masked copy of the parallel part
//! (`client/MinModeSaddleSearch.cpp` lines 199–262 on TheochemUI/eOn
//! `develop`): `bowl_breakout` keeps the atoms with the largest forces,
//! and the threshold variant lowers its cutoff until enough components
//! survive. The dynamic active volume of Liang and Xu, Comput. Mater.
//! Sci. 228, 112354 (2023), doi:10.1016/j.commatsci.2023.112354, is a
//! sigmoid of a per-atom input. The reference is
//! `generate_sigmoid_scale`, `compute_RescaleRAS`, and
//! `compute_xRescale_logdisps` in
//! `seakmc_p/spsearch/SPSearch.py` of SEAKMC/SEAKMC_py_parallel. The
//! sigmoid is applied to the climb direction, not to the mode, and the
//! ranked atoms are the whole system. There is no active-volume shell.

use ndarray::{Array1, ArrayView1};

use crate::error::SaddleError;

/// Geometry the weights may read. The mode and the gradient are the
/// ones the unweighted rotation just produced.
pub struct ClimbFrame<'a> {
    pub position: ArrayView1<'a, f64>,
    /// Geometry the session was given, the minimum a host climbed from.
    pub origin: ArrayView1<'a, f64>,
    pub mode: ArrayView1<'a, f64>,
    /// Energy gradient, not the reflected force.
    pub gradient: ArrayView1<'a, f64>,
    /// Cartesian atoms in front of the vector. A longer vector is a
    /// solid-state chart; the tail is the cell.
    pub n_atoms: usize,
    /// Atoms the host froze. The session forces their weights to zero
    /// after [`ClimbWeights::weights`] returns, so a frozen atom does
    /// not move in the confined step.
    pub frozen: &'a [usize],
}

/// Per-coordinate weights of one convex climb step.
///
/// The session forms the step direction as the unit vector of
/// `weights * mode`, with the sign taken from the unweighted mode so
/// the step still climbs it. A weight of one everywhere is the
/// unweighted step.
pub trait ClimbWeights {
    fn weights(&mut self, frame: &ClimbFrame<'_>) -> Array1<f64>;

    /// Per-atom length of a trial displacement.
    ///
    /// `None` keeps the Euclidean length of each xyz triple, which is
    /// the stock per-atom `max_move` cap. A length is a norm: scaling
    /// the displacement scales the reported lengths by the same factor.
    /// The cap then makes the largest length equal `max_move`.
    ///
    /// This is the slot a Gaussian-process dimer uses for a
    /// force-variance length (gpridge `AtomicDimer.h` line 1789). No
    /// surrogate is wired here.
    fn step_lengths(
        &mut self,
        _origin: ArrayView1<f64>,
        _trial: ArrayView1<f64>,
    ) -> Option<Array1<f64>> {
        None
    }
}

/// Unit vector whose per-atom length supplies participation weights.
///
/// `None` means the source is not ready. [`Confine::Participation`]
/// then uses its DAV fallback. A later Gaussian-process dimer returns
/// the surrogate's projected lowest Hessian mode (gpridge
/// `gpr/AtomicDimer.cpp` lines 5497 and 5519).
pub trait ModeSource: Send {
    fn vector(&mut self, frame: &ClimbFrame<'_>) -> Option<Array1<f64>>;
}

/// The dimer's own mode. This is the source used until a surrogate exists.
#[derive(Clone, Copy, Debug, Default)]
pub struct DimerMode;

impl ModeSource for DimerMode {
    fn vector(&mut self, frame: &ClimbFrame<'_>) -> Option<Array1<f64>> {
        Some(frame.mode.to_owned())
    }
}

/// Sigmoid map of one DAV input.
///
/// `xrange` is SEAKMC's `range` and `r0` is `r4zero`: the zero of the
/// logit sits at `min + r0 * (max - min)`. `min_span` is `MinSpan`. A
/// sample whose max and min differ by less than that is left at one,
/// which is the reference's refusal to rescale a flat input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidMap {
    pub xrange: f64,
    pub r0: f64,
    pub min_span: f64,
}

impl SigmoidMap {
    /// LOGN map. `XRange4LOGV` 20, `Ratio4Zero4LOGV` 0.2, `MinSpan4LOGV` 4
    /// in SEAKMC `run_script/input.yaml`.
    pub fn log_n() -> Self {
        Self {
            xrange: 20.0,
            r0: 0.2,
            min_span: 4.0,
        }
    }

    /// RAS map. `XRange4RAS` 40, `Ratio4Zero4RAS` 0.3, `MinSpan4RAS` 40.
    /// Liang and Xu use 0.5 in the iron-vacancy figure, which puts the
    /// logit zero at the median rank. [`SigmoidMap::r0`] is that knob.
    pub fn ras() -> Self {
        Self {
            xrange: 40.0,
            r0: 0.3,
            min_span: 40.0,
        }
    }

    fn check(self, name: &str) -> Result<(), SaddleError> {
        if !(self.xrange.is_finite() && self.xrange > 0.0) {
            return Err(SaddleError::Invalid(format!(
                "{name} xrange must be positive and finite"
            )));
        }
        if !self.r0.is_finite() {
            return Err(SaddleError::Invalid(format!("{name} r0 must be finite")));
        }
        if !(self.min_span.is_finite() && self.min_span >= 0.0) {
            return Err(SaddleError::Invalid(format!(
                "{name} min span must be non-negative and finite"
            )));
        }
        Ok(())
    }
}

/// What the sigmoid reads.
///
/// `Ras` ranks atoms by distance to `centre`, nearest first, and reverses
/// that rank so the centre stays heavy. The rank uses the current
/// positions. `LogN` is the log of the per-atom mode length, the
/// reference `compute_xRescale_logdisps` at its shipped moving-average
/// period of one (the average is the current value). `LogNRas` is the
/// component-wise minimum of the two, which is SEAKMC's default mix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DavInput {
    Ras { centre: [f64; 3] },
    LogN,
    LogNRas { centre: [f64; 3] },
}

impl DavInput {
    fn centre(self) -> Option<[f64; 3]> {
        match self {
            Self::Ras { centre } | Self::LogNRas { centre } => Some(centre),
            Self::LogN => None,
        }
    }

    fn uses_log(self) -> bool {
        matches!(self, Self::LogN | Self::LogNRas { .. })
    }

    fn uses_ras(self) -> bool {
        matches!(self, Self::Ras { .. } | Self::LogNRas { .. })
    }
}

/// Which convex steps carry a weight. [`Confine::None`] is the stock climb.
pub enum Confine {
    None,
    /// eOn `bowl_breakout`: the `n` free atoms with the largest force.
    /// Ties keep the earlier atom, as the scan in
    /// `MinModeSaddleSearch.cpp` lines 215–227 does (`f_max < norm`).
    /// `n == 0` leaves a zero step, matching `nBowlActive <= 0`.
    TopForce {
        n: usize,
    },
    /// eOn's other `confine_positive` branch, lines 234–262.
    ///
    /// A component whose absolute force is below `min_force` is dropped.
    /// `min_force` is multiplied by `scale` until at least `min_active`
    /// components remain or `max(3 N, min_active)` passes have run.
    /// `boost` is eOn's factor on the surviving parallel part. It is a
    /// common scale when it is positive, so the unit climb direction
    /// does not depend on its magnitude. A negative boost reverses that
    /// direction. Zero leaves a zero step.
    Threshold {
        min_force: f64,
        scale: f64,
        min_active: usize,
        boost: f64,
    },
    /// Liang and Xu sigmoid on the whole system.
    Sigmoid {
        input: DavInput,
        log: SigmoidMap,
        ras: SigmoidMap,
        /// SEAKMC `PowerOnV`, 4 in the shipped settings.
        power: f64,
        /// SEAKMC `MinValue4LOGV`. The floor inside the log is `exp` of this.
        log_floor: f64,
    },
    /// Per-atom participation in an external mode.
    ///
    /// `w_i = max(floor, (|v_i| / max_j |v_j|)^power)`, with `|v_i|` the
    /// Euclidean length of atom `i` in the vector `mode_source` returns.
    /// The same weight is written on x, y, and z. A source that returns
    /// `None` uses `fallback`, which is the LOGN sigmoid unless the host
    /// passed another one. That is the DAV stand-in until a surrogate
    /// is trained.
    Participation {
        mode_source: Box<dyn ModeSource>,
        floor: f64,
        power: f64,
        fallback: Box<Confine>,
    },
}

impl std::fmt::Debug for Confine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("None"),
            Self::TopForce { n } => write!(f, "TopForce {{ n: {n} }}"),
            Self::Threshold {
                min_force,
                scale,
                min_active,
                boost,
            } => write!(
                f,
                "Threshold {{ min_force: {min_force}, scale: {scale}, min_active: {min_active}, boost: {boost} }}"
            ),
            Self::Sigmoid { input, .. } => write!(f, "Sigmoid {{ input: {input:?} }}"),
            Self::Participation {
                floor,
                power,
                fallback,
                ..
            } => write!(
                f,
                "Participation {{ floor: {floor}, power: {power}, fallback: {fallback:?} }}"
            ),
        }
    }
}

impl Default for Confine {
    fn default() -> Self {
        Self::None
    }
}

impl Confine {
    pub fn top_force(n: usize) -> Self {
        Self::TopForce { n }
    }

    /// eOn's shipped threshold: min force 0.5, scale 0.9, 30 components, boost 10.
    pub fn eon_threshold() -> Self {
        Self::Threshold {
            min_force: 0.5,
            scale: 0.9,
            min_active: 30,
            boost: 10.0,
        }
    }

    pub fn threshold(min_force: f64, scale: f64, min_active: usize, boost: f64) -> Self {
        Self::Threshold {
            min_force,
            scale,
            min_active,
            boost,
        }
    }

    /// LOGN sigmoid. This is the participation fallback: it needs no centre.
    pub fn log_n() -> Self {
        Self::Sigmoid {
            input: DavInput::LogN,
            log: SigmoidMap::log_n(),
            ras: SigmoidMap::ras(),
            power: 4.0,
            log_floor: -20.0,
        }
    }

    /// LOGN+RAS with the SEAKMC settings. `centre` is the defect, in the
    /// same Cartesian frame as the positions. Every atom is ranked.
    pub fn log_n_ras(centre: [f64; 3]) -> Self {
        Self::Sigmoid {
            input: DavInput::LogNRas { centre },
            log: SigmoidMap::log_n(),
            ras: SigmoidMap::ras(),
            power: 4.0,
            log_floor: -20.0,
        }
    }

    /// Participation in `source`, falling back to [`Confine::log_n`].
    ///
    /// [`Confine::own_mode`] passes the dimer's own mode. A surrogate
    /// passes its projected lowest mode and returns `None` until it is trained.
    pub fn participation(source: impl ModeSource + 'static, floor: f64, power: f64) -> Self {
        Self::participation_or(source, floor, power, Self::log_n())
    }

    /// [`Confine::participation`] with an explicit fallback.
    pub fn participation_or(
        source: impl ModeSource + 'static,
        floor: f64,
        power: f64,
        fallback: Self,
    ) -> Self {
        Self::Participation {
            mode_source: Box::new(source),
            floor,
            power,
            fallback: Box::new(fallback),
        }
    }

    /// Participation in the dimer's own mode. `floor` 0 and `power` 1
    /// keep the peak atom at weight one and scale the others by their
    /// share of that peak.
    pub fn own_mode(floor: f64, power: f64) -> Self {
        Self::participation(DimerMode, floor, power)
    }

    pub fn check(&self) -> Result<(), SaddleError> {
        match self {
            Self::None | Self::TopForce { .. } => Ok(()),
            Self::Threshold {
                min_force,
                scale,
                boost,
                ..
            } => {
                if !min_force.is_finite() {
                    return Err(SaddleError::Invalid(
                        "confine threshold min force must be finite".into(),
                    ));
                }
                if !(scale.is_finite() && *scale > 0.0) {
                    return Err(SaddleError::Invalid(
                        "confine threshold scale must be positive and finite".into(),
                    ));
                }
                if !boost.is_finite() {
                    return Err(SaddleError::Invalid(
                        "confine threshold boost must be finite".into(),
                    ));
                }
                Ok(())
            }
            Self::Sigmoid {
                input,
                log,
                ras,
                power,
                log_floor,
            } => {
                if let Some(centre) = input.centre()
                    && !centre.iter().all(|v| v.is_finite())
                {
                    return Err(SaddleError::NonFinite("confine centre"));
                }
                if input.uses_log() {
                    log.check("log sigmoid")?;
                }
                if input.uses_ras() {
                    ras.check("ras sigmoid")?;
                }
                if !(power.is_finite() && *power > 0.0) {
                    return Err(SaddleError::Invalid(
                        "confine sigmoid power must be positive and finite".into(),
                    ));
                }
                if !log_floor.is_finite() {
                    return Err(SaddleError::Invalid(
                        "confine log floor must be finite".into(),
                    ));
                }
                Ok(())
            }
            Self::Participation {
                floor,
                power,
                fallback,
                ..
            } => {
                if !(floor.is_finite() && *floor >= 0.0) {
                    return Err(SaddleError::Invalid(
                        "participation floor must be non-negative and finite".into(),
                    ));
                }
                if !(power.is_finite() && *power > 0.0) {
                    return Err(SaddleError::Invalid(
                        "participation power must be positive and finite".into(),
                    ));
                }
                fallback.check()
            }
        }
    }
}

impl ClimbWeights for Confine {
    fn weights(&mut self, frame: &ClimbFrame<'_>) -> Array1<f64> {
        let n = frame.position.len();
        let n_atoms = frame.n_atoms.min(n / 3);
        let mut w = match self {
            Self::None => Array1::ones(n),
            Self::TopForce { n: keep } => top_force_weights(frame, n_atoms, *keep),
            Self::Threshold {
                min_force,
                scale,
                min_active,
                boost,
            } => threshold_weights(frame, n_atoms, *min_force, *scale, *min_active, *boost),
            Self::Sigmoid {
                input,
                log,
                ras,
                power,
                log_floor,
            } => sigmoid_weights(frame, n_atoms, *input, *log, *ras, *power, *log_floor),
            Self::Participation {
                mode_source,
                floor,
                power,
                fallback,
            } => {
                let floor = *floor;
                let power = *power;
                let ready = mode_source.vector(frame).filter(|vector| {
                    vector.len() == frame.position.len() && vector.iter().all(|c| c.is_finite())
                });
                if let Some(vector) = ready {
                    participation_weights(frame, vector.view(), n_atoms, floor, power)
                } else {
                    fallback.weights(frame)
                }
            }
        };
        if w.len() != n {
            w = Array1::ones(n);
        }
        for value in w.iter_mut().skip(3 * n_atoms) {
            *value = 1.0;
        }
        w
    }
}

/// `w_i = max(floor, (|v_i| / max_j |v_j|)^power)` on free atoms.
#[allow(clippy::needless_range_loop)]
fn participation_weights(
    frame: &ClimbFrame<'_>,
    vector: ArrayView1<f64>,
    n_atoms: usize,
    floor: f64,
    power: f64,
) -> Array1<f64> {
    let mut norms = vec![0.0; n_atoms];
    let mut peak = 0.0;
    for atom in 0..n_atoms {
        if frozen_atom(frame, atom) {
            continue;
        }
        let i = 3 * atom;
        if i + 2 >= vector.len() {
            break;
        }
        let len =
            (vector[i] * vector[i] + vector[i + 1] * vector[i + 1] + vector[i + 2] * vector[i + 2])
                .sqrt();
        norms[atom] = len;
        if len > peak {
            peak = len;
        }
    }
    let mut w = Array1::zeros(frame.position.len());
    for atom in 0..n_atoms {
        if frozen_atom(frame, atom) {
            continue;
        }
        let ratio = if peak > 0.0 { norms[atom] / peak } else { 0.0 };
        let weight = floor.max(ratio.powf(power));
        let i = 3 * atom;
        w[i] = weight;
        w[i + 1] = weight;
        w[i + 2] = weight;
    }
    w
}

fn frozen_atom(frame: &ClimbFrame<'_>, atom: usize) -> bool {
    frame.frozen.contains(&atom)
}

fn atom_force_norm(gradient: ArrayView1<f64>, atom: usize) -> f64 {
    let i = 3 * atom;
    let n = gradient[i] * gradient[i]
        + gradient[i + 1] * gradient[i + 1]
        + gradient[i + 2] * gradient[i + 2];
    n.sqrt()
}

/// Greedy selection from `MinModeSaddleSearch.cpp` lines 215–231.
fn top_force_weights(frame: &ClimbFrame<'_>, n_atoms: usize, keep: usize) -> Array1<f64> {
    let mut w = Array1::zeros(frame.position.len());
    if keep == 0 || n_atoms == 0 {
        return w;
    }
    let mut norm: Vec<f64> = (0..n_atoms)
        .map(|atom| {
            if frozen_atom(frame, atom) {
                f64::NEG_INFINITY
            } else {
                atom_force_norm(frame.gradient, atom)
            }
        })
        .collect();
    let free = norm.iter().filter(|n| n.is_finite()).count();
    let keep = keep.min(free);
    for _ in 0..keep {
        let mut chosen = None;
        let mut best = f64::NEG_INFINITY;
        for (atom, &value) in norm.iter().enumerate() {
            if value.is_finite() && (chosen.is_none() || best < value) {
                best = value;
                chosen = Some(atom);
            }
        }
        let Some(atom) = chosen else {
            break;
        };
        // A picked row is removed, as `forceTemp.row(i_max).setZero()`.
        norm[atom] = f64::NEG_INFINITY;
        let i = 3 * atom;
        w[i] = 1.0;
        w[i + 1] = 1.0;
        w[i + 2] = 1.0;
    }
    w
}

/// Threshold loop from `MinModeSaddleSearch.cpp` lines 234–262.
///
/// The comparison is against the original force on every pass. Frozen
/// atoms are not counted and stay at weight zero.
fn threshold_weights(
    frame: &ClimbFrame<'_>,
    n_atoms: usize,
    mut min_force: f64,
    scale: f64,
    min_active: usize,
    boost: f64,
) -> Array1<f64> {
    let mut w = Array1::zeros(frame.position.len());
    if boost == 0.0 || n_atoms == 0 {
        return w;
    }
    let alive = boost.signum();
    let free_atoms = (0..n_atoms)
        .filter(|&atom| !frozen_atom(frame, atom))
        .count();
    let max_tries = (3 * free_atoms).max(min_active).max(1);
    for _ in 0..max_tries {
        let mut sufficient = 0usize;
        for atom in 0..n_atoms {
            if frozen_atom(frame, atom) {
                continue;
            }
            for k in 0..3 {
                let slot = 3 * atom + k;
                if frame.gradient[slot].abs() < min_force {
                    w[slot] = 0.0;
                } else {
                    sufficient += 1;
                    w[slot] = alive;
                }
            }
        }
        if sufficient >= min_active {
            break;
        }
        min_force *= scale;
    }
    w
}

fn sigmoid_weights(
    frame: &ClimbFrame<'_>,
    n_atoms: usize,
    input: DavInput,
    log: SigmoidMap,
    ras: SigmoidMap,
    power: f64,
    log_floor: f64,
) -> Array1<f64> {
    let free: Vec<usize> = (0..n_atoms)
        .filter(|&atom| !frozen_atom(frame, atom))
        .collect();
    let mut per_atom = vec![1.0; n_atoms];
    if free.is_empty() {
        return broadcast(&per_atom, frame.position.len());
    }
    let log_w = if input.uses_log() {
        let sample = log_sample(frame, &free, power, log_floor);
        sigmoid_scale(&sample, log)
    } else {
        vec![1.0; free.len()]
    };
    let ras_w = if input.uses_ras() {
        let centre = input.centre().unwrap_or([0.0; 3]);
        let sample = ras_sample(frame, &free, centre);
        sigmoid_scale(&sample, ras)
    } else {
        vec![1.0; free.len()]
    };
    for (k, &atom) in free.iter().enumerate() {
        per_atom[atom] = match input {
            DavInput::LogNRas { .. } => log_w[k].min(ras_w[k]),
            DavInput::LogN => log_w[k],
            DavInput::Ras { .. } => ras_w[k],
        };
    }
    broadcast(&per_atom, frame.position.len())
}

fn broadcast(per_atom: &[f64], len: usize) -> Array1<f64> {
    let mut w = Array1::ones(len);
    for (atom, &weight) in per_atom.iter().enumerate() {
        let i = 3 * atom;
        if i + 2 >= len {
            break;
        }
        w[i] = weight;
        w[i + 1] = weight;
        w[i + 2] = weight;
    }
    w
}

/// `ln(d^power + exp(log_floor))`, `compute_xRescale_logdisps`.
fn log_sample(frame: &ClimbFrame<'_>, free: &[usize], power: f64, log_floor: f64) -> Vec<f64> {
    let floor = log_floor.exp();
    free.iter()
        .map(|&atom| {
            let i = 3 * atom;
            let d2 = frame.mode[i] * frame.mode[i]
                + frame.mode[i + 1] * frame.mode[i + 1]
                + frame.mode[i + 2] * frame.mode[i + 2];
            (d2.sqrt().powf(power) + floor).ln()
        })
        .collect()
}

/// Reverse distance rank. Nearest to `centre` gets `n - 1`, farthest gets 0.
/// An exact tie keeps the earlier atom nearer, so the rank is stable.
fn ras_sample(frame: &ClimbFrame<'_>, free: &[usize], centre: [f64; 3]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..free.len()).collect();
    let dist = |atom: usize| {
        let i = 3 * atom;
        let dx = frame.position[i] - centre[0];
        let dy = frame.position[i + 1] - centre[1];
        let dz = frame.position[i + 2] - centre[2];
        dx * dx + dy * dy + dz * dz
    };
    order.sort_by(|&a, &b| {
        dist(free[a])
            .partial_cmp(&dist(free[b]))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let n = free.len();
    let mut sample = vec![0.0; n];
    for (rank, slot) in order.into_iter().enumerate() {
        sample[slot] = (n - 1 - rank) as f64;
    }
    sample
}

/// `generate_sigmoid_scale`. A span below `min_span` returns ones.
fn sigmoid_scale(sample: &[f64], map: SigmoidMap) -> Vec<f64> {
    if sample.is_empty() {
        return Vec::new();
    }
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    for &value in sample {
        if value < xmin {
            xmin = value;
        }
        if value > xmax {
            xmax = value;
        }
    }
    if !(xmin.is_finite() && xmax.is_finite()) || xmax - xmin < map.min_span {
        return vec![1.0; sample.len()];
    }
    let span = xmax - xmin;
    let s = span / map.xrange;
    let s0 = xmin + span * map.r0;
    sample
        .iter()
        .map(|&value| {
            let z = (value - s0) / s;
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

/// Unit vector of `weights * mode`. All-ones weights return `mode`
/// unchanged, so the caller can keep the unweighted step bit for bit.
/// A vanishing product is a zero vector: the step does not move.
pub(crate) fn weighted_mode(mode: ArrayView1<f64>, weights: ArrayView1<f64>) -> Array1<f64> {
    if weights.len() != mode.len() {
        return mode.to_owned();
    }
    if weights.iter().all(|&weight| weight == 1.0) {
        return mode.to_owned();
    }
    let mut dir = mode.to_owned();
    for (component, &weight) in dir.iter_mut().zip(weights.iter()) {
        *component *= weight;
    }
    let norm = dir.dot(&dir).sqrt();
    if norm > 0.0 && norm.is_finite() {
        dir /= norm;
    } else {
        dir.fill(0.0);
    }
    dir
}

/// Frozen atoms stay put. The cell tail of a solid-state chart stays at
/// weight one, so confinement does not mask the strain.
pub(crate) fn apply_chart(weights: &mut Array1<f64>, n_atoms: usize, frozen: &[usize]) {
    let atoms = 3 * n_atoms;
    for value in weights.iter_mut().skip(atoms) {
        *value = 1.0;
    }
    for &atom in frozen {
        let start = 3 * atom;
        if start + 2 < weights.len().min(atoms) {
            weights[start] = 0.0;
            weights[start + 1] = 0.0;
            weights[start + 2] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ClimbFrame, ClimbWeights, Confine, DavInput, SigmoidMap};
    use ndarray::{Array1, array};

    fn frame<'a>(
        position: &'a Array1<f64>,
        mode: &'a Array1<f64>,
        gradient: &'a Array1<f64>,
        frozen: &'a [usize],
    ) -> ClimbFrame<'a> {
        ClimbFrame {
            position: position.view(),
            origin: position.view(),
            mode: mode.view(),
            gradient: gradient.view(),
            n_atoms: position.len() / 3,
            frozen,
        }
    }

    #[test]
    fn sigmoid_matches_the_reference_map() {
        let sample = [0.0, 1.0, 2.0, 3.0];
        let weights = super::sigmoid_scale(
            &sample,
            SigmoidMap {
                xrange: 2.0,
                r0: 0.5,
                min_span: 0.1,
            },
        );
        let expect = [
            2.6894142136999510e-1,
            4.1742979353768528e-1,
            5.8257020646231472e-1,
            7.3105857863000490e-1,
        ];
        for (got, want) in weights.iter().zip(expect) {
            assert!((got - want).abs() < 1e-15, "{got} vs {want}");
        }
    }

    #[test]
    fn a_flat_sample_stays_at_one() {
        let weights = super::sigmoid_scale(
            &[1.0, 1.2],
            SigmoidMap {
                xrange: 20.0,
                r0: 0.2,
                min_span: 4.0,
            },
        );
        assert_eq!(weights, vec![1.0, 1.0]);
    }

    #[test]
    fn ras_puts_the_nearest_atom_highest() {
        let position = array![0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let mode = Array1::ones(9);
        let gradient = Array1::ones(9);
        let view = frame(&position, &mode, &gradient, &[]);
        let sample = super::ras_sample(&view, &[0, 1, 2], [0.0, 0.0, 0.0]);
        assert_eq!(sample, vec![2.0, 0.0, 1.0]);
    }

    #[test]
    fn log_n_ras_takes_the_smaller_weight() {
        let position = array![0.0, 0.0, 0.0, 4.0, 0.0, 0.0];
        let mode = array![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let gradient = Array1::zeros(6);
        let view = frame(&position, &mode, &gradient, &[]);
        let mut confine = Confine::Sigmoid {
            input: DavInput::LogNRas {
                centre: [0.0, 0.0, 0.0],
            },
            log: SigmoidMap {
                xrange: 20.0,
                r0: 0.2,
                min_span: 0.0,
            },
            ras: SigmoidMap {
                xrange: 40.0,
                r0: 0.5,
                min_span: 0.0,
            },
            power: 4.0,
            log_floor: -20.0,
        };
        let w = confine.weights(&view);
        assert!((w[0] - w[1]).abs() < 1e-15);
        assert!((w[0] - w[2]).abs() < 1e-15);
        let log_only = {
            let mut kind = Confine::Sigmoid {
                input: DavInput::LogN,
                log: SigmoidMap {
                    xrange: 20.0,
                    r0: 0.2,
                    min_span: 0.0,
                },
                ras: SigmoidMap::ras(),
                power: 4.0,
                log_floor: -20.0,
            };
            kind.weights(&view)
        };
        let ras_only = {
            let mut kind = Confine::Sigmoid {
                input: DavInput::Ras {
                    centre: [0.0, 0.0, 0.0],
                },
                log: SigmoidMap::log_n(),
                ras: SigmoidMap {
                    xrange: 40.0,
                    r0: 0.5,
                    min_span: 0.0,
                },
                power: 4.0,
                log_floor: -20.0,
            };
            kind.weights(&view)
        };
        for i in 0..6 {
            let smaller = log_only[i].min(ras_only[i]);
            assert!((w[i] - smaller).abs() < 1e-12, "{i}");
        }
    }

    #[test]
    fn participation_scales_each_atom_by_its_share_of_the_peak() {
        let position = Array1::zeros(9);
        let mode = array![0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 3.0];
        let gradient = Array1::zeros(9);
        let view = frame(&position, &mode, &gradient, &[]);
        let mut confine = Confine::own_mode(0.25, 2.0);
        let w = confine.weights(&view);
        // Atom norms 0, 5, 3. Peak 5. Ratios 0, 1, 0.6. Squares 0, 1, 0.36.
        assert!((w[0] - 0.25).abs() < 1e-12);
        assert!((w[1] - 0.25).abs() < 1e-12);
        assert!((w[2] - 0.25).abs() < 1e-12);
        assert!((w[3] - 1.0).abs() < 1e-12);
        assert!((w[6] - 0.36).abs() < 1e-12);
    }

    struct Silent;
    impl super::ModeSource for Silent {
        fn vector(&mut self, _frame: &ClimbFrame<'_>) -> Option<Array1<f64>> {
            None
        }
    }

    #[test]
    fn an_untrained_source_uses_the_fallback() {
        let position = array![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.5];
        let mode = Array1::ones(9);
        let gradient = array![0.1, 0.0, 0.0, 5.0, 0.0, 0.0, 0.2, 0.0, 0.0];
        let view = frame(&position, &mode, &gradient, &[]);
        let mut fallback = Confine::top_force(1);
        let expect = fallback.weights(&view);
        let mut confine = Confine::participation_or(Silent, 0.0, 1.0, Confine::top_force(1));
        let w = confine.weights(&view);
        for (got, want) in w.iter().zip(expect.iter()) {
            assert!((got - want).abs() < 1e-15);
        }
    }

    #[test]
    fn ones_leave_the_mode_untouched() {
        let mode = array![0.0, 0.6, 0.8];
        let weights = Array1::ones(3);
        let dir = super::weighted_mode(mode.view(), weights.view());
        assert_eq!(dir, mode);
    }
}
