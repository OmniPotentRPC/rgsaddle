//! Convex-climb weights. The mode is estimated without them.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    ClimbFrame, ClimbWeights, Confine, DimerMode, MinModeConfig, MinModeKind, MinModeSession,
    MinModeStatus, ModeSource, PointSurface, SaddleError, exact_eigh,
};

struct Counted<S> {
    inner: S,
    calls: std::sync::atomic::AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counted<S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.eval(x)
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.inner.excluded_modes(x)
    }
}

/// V = (x² − 1)² + 2 y² + 2 z². The saddle is the origin.
struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y, z) = (r[0], r[1], r[2]);
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
        Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
    }
}

struct Ones;

impl ClimbWeights for Ones {
    fn weights(&mut self, frame: &ClimbFrame<'_>) -> Array1<f64> {
        Array1::ones(frame.position.len())
    }
}

#[test]
fn unit_weights_match_the_unweighted_dimer() {
    let config = MinModeConfig {
        max_rotations: 8,
        force_tol: 1e-8,
        max_move: 0.2,
        ..MinModeConfig::default()
    };
    let start = array![0.85, 0.15, -0.05];
    let mode = array![0.7, 0.2, 0.1];
    let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut weighted = MinModeSession::new(config, start, mode).unwrap();
    weighted.set_climb_weights(Ones);
    let surface = DoubleWell;
    for _ in 0..6 {
        let left = plain.step(&surface).unwrap();
        let right = weighted.step(&surface).unwrap();
        assert_eq!(left.curvature.to_bits(), right.curvature.to_bits());
        assert_eq!(left.rotations, right.rotations);
        assert_eq!(plain.position(), weighted.position());
        assert_eq!(plain.mode(), weighted.mode());
    }
}

/// Two atoms. Atom 0 sits on a convex well. Atom 1 is a softer spring,
/// so the lowest mode is atom 1. The force is on atom 0.
struct SplitWell;

impl PointSurface for SplitWell {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let x = r[0];
        let mut energy = (x * x - 1.0).powi(2);
        let mut g = Array1::zeros(r.len());
        g[0] = 4.0 * x * (x * x - 1.0);
        for i in 1..r.len() {
            let k = if i < 3 { 6.0 } else { 0.4 };
            energy += 0.5 * k * r[i] * r[i];
            g[i] = k * r[i];
        }
        Ok((energy, g))
    }
}

#[test]
fn weights_do_not_change_the_mode_or_the_curvature() {
    let config = MinModeConfig {
        max_rotations: 0,
        force_tol: 1e-9,
        max_move: 0.1,
        rotation_angle_tol: 0.0,
        ..MinModeConfig::default()
    };
    let start = array![0.9, 0.0, 0.0, 0.2, 0.0, 0.0];
    let mode = array![1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut weighted = MinModeSession::new(config, start.clone(), mode).unwrap();
    weighted.set_confine(Confine::top_force(1)).unwrap();
    let left = plain.step(&SplitWell).unwrap();
    let right = weighted.step(&SplitWell).unwrap();
    assert!((left.curvature - right.curvature).abs() < 1e-12);
    for (a, b) in plain.mode().iter().zip(weighted.mode().iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    assert_ne!(plain.position(), weighted.position());
    // Top-force keeps the loaded atom and drops the second atom's share of the mode.
    let plain_second = (plain.position()[3] - start[3]).abs();
    let weighted_second = (weighted.position()[3] - 0.2).abs();
    assert!(plain_second > weighted_second);
}

#[test]
fn a_negative_curvature_step_ignores_the_weights() {
    let config = MinModeConfig {
        max_rotations: 4,
        force_tol: 1e-12,
        max_move: 0.05,
        ..MinModeConfig::default()
    };
    let start = array![0.15, 0.2, -0.1];
    let mode = array![1.0, 0.2, 0.0];
    let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut weighted = MinModeSession::new(config, start, mode).unwrap();
    weighted.set_confine(Confine::top_force(1)).unwrap();
    plain.step(&DoubleWell).unwrap();
    weighted.step(&DoubleWell).unwrap();
    assert!(plain.curvature() < 0.0);
    assert_eq!(plain.position(), weighted.position());
    assert_eq!(plain.mode(), weighted.mode());
}

/// Diagonal springs, so every direction is convex and the initial mode
/// is already an eigenvector. `max_rotations` still runs the rotation;
/// the rotational force is zero.
struct Springs {
    k: f64,
}

impl PointSurface for Springs {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = 0.5 * self.k * r.dot(&r);
        Ok((energy, r.to_owned() * self.k))
    }
}

fn eon_top_atoms(force_norm: &[f64], n: usize) -> Vec<usize> {
    // MinModeSaddleSearch.cpp:215-227. The scan keeps the earlier atom
    // when the norms are equal (`f_max < norm`).
    let mut left: Vec<f64> = force_norm.to_vec();
    let mut chosen = Vec::new();
    let n = n.min(left.len());
    for _ in 0..n {
        let mut best = left[0];
        let mut index = 0usize;
        for (i, &value) in left.iter().enumerate() {
            if best < value {
                best = value;
                index = i;
            }
        }
        left[index] = f64::NEG_INFINITY;
        chosen.push(index);
    }
    chosen
}

fn eon_threshold_mask(
    gradient: &[f64],
    min_force: f64,
    scale: f64,
    min_active: usize,
    boost: f64,
) -> Vec<f64> {
    // MinModeSaddleSearch.cpp:234-262. Each pass reads the original force.
    let n_atoms = gradient.len() / 3;
    let max_tries = (3 * n_atoms).max(min_active).max(1);
    let mut min_force = min_force;
    let mut mask = vec![0.0; gradient.len()];
    let alive = if boost == 0.0 { 0.0 } else { boost.signum() };
    for _ in 0..max_tries {
        let mut sufficient = 0usize;
        for (slot, &component) in gradient.iter().enumerate() {
            if component.abs() < min_force {
                mask[slot] = 0.0;
            } else {
                sufficient += 1;
                mask[slot] = alive;
            }
        }
        if sufficient >= min_active {
            break;
        }
        min_force *= scale;
    }
    mask
}

#[test]
fn top_force_matches_eon_bowl_breakout() {
    let gradient = array![0.1, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let norms = [0.1, 3.0, 1.0, 2.0];
    let chosen = eon_top_atoms(&norms, 2);
    assert_eq!(chosen, vec![1, 3]);
    let config = MinModeConfig {
        max_rotations: 4,
        force_tol: 1e-15,
        max_move: 0.2,
        ..MinModeConfig::default()
    };
    let mode = Array1::ones(12);
    let mut session = MinModeSession::new(config, gradient.clone(), mode).unwrap();
    session.set_confine(Confine::top_force(2)).unwrap();
    let before = session.position().to_owned();
    session.step(&Springs { k: 1.0 }).unwrap();
    let step = &session.position() - &before;
    for atom in 0..4 {
        let i = 3 * atom;
        let moved = step.slice(ndarray::s![i..i + 3]).map(|v| v.abs()).sum() > 1e-8;
        assert_eq!(moved, chosen.contains(&atom), "atom {atom}");
    }
    let mut largest = 0.0_f64;
    for atom in 0..4 {
        let i = 3 * atom;
        let r = (step[i].powi(2) + step[i + 1].powi(2) + step[i + 2].powi(2)).sqrt();
        largest = largest.max(r);
    }
    assert!((largest - 0.2).abs() < 1e-10);
}

#[test]
fn threshold_matches_eon_confine_positive() {
    let gradient = array![0.1, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0];
    let mask = eon_threshold_mask(gradient.as_slice().unwrap(), 2.5, 0.5, 2, 10.0);
    // 3 and 2 survive; 1 does not until the cutoff would fall further.
    assert!(mask[3] > 0.0 && mask[11] > 0.0);
    assert!(mask.iter().filter(|v| **v > 0.0).count() == 2);
    let config = MinModeConfig {
        max_rotations: 4,
        force_tol: 1e-15,
        max_move: 0.2,
        ..MinModeConfig::default()
    };
    let mut session = MinModeSession::new(config, gradient.clone(), Array1::ones(12)).unwrap();
    session
        .set_confine(Confine::threshold(2.5, 0.5, 2, 10.0))
        .unwrap();
    let before = session.position().to_owned();
    session.step(&Springs { k: 1.0 }).unwrap();
    let step = &session.position() - &before;
    for (i, (&bit, &delta)) in mask.iter().zip(step.iter()).enumerate() {
        let moved = delta.abs() > 1e-8;
        assert_eq!(moved, bit > 0.0, "component {i}");
    }
    // A negative boost reverses the surviving parallel part.
    let config = MinModeConfig {
        max_rotations: 4,
        force_tol: 1e-15,
        max_move: 0.2,
        ..MinModeConfig::default()
    };
    let mut negative = MinModeSession::new(config, gradient, Array1::ones(12)).unwrap();
    negative
        .set_confine(Confine::threshold(2.5, 0.5, 2, -10.0))
        .unwrap();
    let before = negative.position().to_owned();
    negative.step(&Springs { k: 1.0 }).unwrap();
    let flipped = &negative.position() - &before;
    for (a, b) in step.iter().zip(flipped.iter()) {
        assert!((a + b).abs() < 1e-10);
    }
}

#[test]
fn frozen_atoms_and_the_cell_tail_stay_out_of_the_mask() {
    let config = MinModeConfig {
        max_rotations: 0,
        force_tol: 1e-15,
        max_move: 0.2,
        rotation_angle_tol: 0.0,
        ..MinModeConfig::default()
    };
    // Two atoms plus nine cell coordinates. The cell is stiff so it is
    // not the lowest mode, and the mode we hand in is kept: the rotation
    // cap is zero.
    let mut position = Array1::zeros(15);
    position[0] = 0.2;
    position[3] = 4.0;
    for i in 6..15 {
        position[i] = 0.05;
    }
    let mut mode = Array1::zeros(15);
    mode[0] = 0.2;
    mode[3] = 0.4;
    for i in 6..15 {
        mode[i] = 0.3;
    }
    let mut session = MinModeSession::new(config, position.clone(), mode.clone()).unwrap();
    session.set_cell_tail(9).unwrap();
    session.set_frozen(&[1]).unwrap();
    session.set_confine(Confine::top_force(1)).unwrap();
    let surface = Springs { k: 1.0 };
    let before_mode = session.mode().to_owned();
    session.step(&surface).unwrap();
    assert_eq!(session.mode(), before_mode.view());
    let step = &session.position() - &position;
    // Frozen atom 1 does not move.
    assert!(step[3].abs() + step[4].abs() + step[5].abs() < 1e-12);
    // The cell tail is weight 1, so it moves with the mode.
    let cell = step.slice(ndarray::s![6..]).map(|v| v.abs()).sum();
    assert!(cell > 1e-3, "cell displacement {cell}");
}

struct Twice;

impl ClimbWeights for Twice {
    fn weights(&mut self, frame: &ClimbFrame<'_>) -> Array1<f64> {
        Array1::ones(frame.position.len())
    }

    fn step_lengths(
        &mut self,
        origin: ArrayView1<f64>,
        trial: ArrayView1<f64>,
    ) -> Option<Array1<f64>> {
        let n = origin.len() / 3;
        let mut lengths = Array1::zeros(n);
        for atom in 0..n {
            let i = 3 * atom;
            let dx = trial[i] - origin[i];
            let dy = trial[i + 1] - origin[i + 1];
            let dz = trial[i + 2] - origin[i + 2];
            lengths[atom] = 2.0 * (dx * dx + dy * dy + dz * dz).sqrt();
        }
        Some(lengths)
    }
}

#[test]
fn a_per_atom_metric_rescales_the_capped_step() {
    let config = MinModeConfig {
        max_rotations: 0,
        force_tol: 1e-15,
        max_move: 0.2,
        rotation_angle_tol: 0.0,
        ..MinModeConfig::default()
    };
    let start = array![0.4, 0.1, 0.0];
    let mode = array![1.0, 0.0, 0.0];
    let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut metric = MinModeSession::new(config, start.clone(), mode).unwrap();
    metric.set_climb_weights(Twice);
    plain.step(&Springs { k: 1.0 }).unwrap();
    metric.step(&Springs { k: 1.0 }).unwrap();
    let plain_step = &plain.position() - &start;
    let metric_step = &metric.position() - &start;
    for (a, b) in plain_step.iter().zip(metric_step.iter()) {
        assert!((b - 0.5 * a).abs() < 1e-10, "{b} vs {a}");
    }
}

#[test]
fn participation_uses_the_dimers_own_mode() {
    let config = MinModeConfig {
        max_rotations: 0,
        force_tol: 1e-15,
        max_move: 0.2,
        rotation_angle_tol: 0.0,
        ..MinModeConfig::default()
    };
    let start = array![0.3, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0];
    let mode = array![0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 3.0];
    let mut session = MinModeSession::new(config, start.clone(), mode).unwrap();
    session.set_confine(Confine::own_mode(0.25, 2.0)).unwrap();
    let mode_before = session.mode().to_owned();
    session.step(&Springs { k: 1.0 }).unwrap();
    for (a, b) in session.mode().iter().zip(mode_before.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    let step = &session.position() - &start;
    // Atom 0's participation is the floor, so it moves less than atom 1.
    let a0 = (step[0].powi(2) + step[1].powi(2) + step[2].powi(2)).sqrt();
    let a1 = (step[3].powi(2) + step[4].powi(2) + step[5].powi(2)).sqrt();
    let a2 = (step[6].powi(2) + step[7].powi(2) + step[8].powi(2)).sqrt();
    assert!(a1 > a0 && a1 > a2);
    assert!((a1 - 0.2).abs() < 1e-8);
    // DimerMode is the documented own-mode source.
    let mut source = DimerMode;
    let frame_mode = mode_before.clone();
    let dummy = Array1::zeros(9);
    let seen = source
        .vector(&ClimbFrame {
            position: dummy.view(),
            origin: dummy.view(),
            mode: frame_mode.view(),
            gradient: dummy.view(),
            n_atoms: 3,
            frozen: &[],
        })
        .unwrap();
    assert_eq!(seen, frame_mode);
}

#[test]
fn relatch_is_off_until_the_host_asks() {
    let config = MinModeConfig {
        max_rotations: 4,
        force_tol: 1e-9,
        max_move: 0.2,
        ..MinModeConfig::default()
    };
    let start = array![0.9, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mode = array![1.0, 0.0, 0.0, 0.2, 0.0, 0.0];
    let mut held = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut again = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    held.set_confine(Confine::top_force(1)).unwrap();
    again.set_confine(Confine::top_force(1)).unwrap();
    again.set_relatch(true);
    held.step(&SplitWell).unwrap();
    again.step(&SplitWell).unwrap();
    // Inside the barrier the curvature along x is negative and lower
    // than the soft spring, so the weight lifts.
    let inside = array![0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
    held.set_position(inside.clone(), None).unwrap();
    again.set_position(inside, None).unwrap();
    let report = held.step(&SplitWell).unwrap();
    again.step(&SplitWell).unwrap();
    assert!(report.curvature < 0.0);
    held.set_position(start.clone(), None).unwrap();
    again.set_position(start.clone(), None).unwrap();
    held.step(&SplitWell).unwrap();
    again.step(&SplitWell).unwrap();
    let held_soft = (held.position()[3]).abs();
    let again_soft = (again.position()[3]).abs();
    // Without relatch the soft atom takes the convex step. With it, the
    // top-force mask still holds that atom.
    assert!(held_soft > again_soft);
}

/// Bond double well plus soft spectators tied to atom 0.
struct Bond {
    n_spectators: usize,
}

impl Bond {
    fn n_atoms(&self) -> usize {
        2 + self.n_spectators
    }
}

fn bond_energy(r: f64) -> (f64, f64) {
    let a = r - 1.2;
    let b = r - 2.4;
    let v = a * a * b * b;
    let dv = 2.0 * a * b * (a + b);
    (v, dv)
}

impl PointSurface for Bond {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut g = Array1::zeros(x.len());
        let d = [x[3] - x[0], x[4] - x[1], x[5] - x[2]];
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if r > 1e-8 {
            let (v, dv) = bond_energy(r);
            energy += v;
            for k in 0..3 {
                let comp = dv * d[k] / r;
                g[3 + k] += comp;
                g[k] -= comp;
            }
        }
        // Soft springs along x, tied to atom 0, so the low mode still
        // moves the bond. y and z stay stiff.
        let k = 0.5;
        for s in 0..self.n_spectators {
            let base = 3 * (2 + s);
            let stretch = (x[base] - x[0]) - (1.6 + 0.2 * s as f64);
            energy += 0.5 * k * stretch * stretch;
            g[base] += k * stretch;
            g[0] -= k * stretch;
            for axis in 1..3 {
                energy += 4.0 * x[base + axis] * x[base + axis];
                g[base + axis] += 8.0 * x[base + axis];
            }
        }
        energy += 8.0 * x[1] * x[1] + 8.0 * x[2] * x[2];
        energy += 8.0 * x[4] * x[4] + 8.0 * x[5] * x[5];
        g[1] += 16.0 * x[1];
        g[2] += 16.0 * x[2];
        g[4] += 16.0 * x[4];
        g[5] += 16.0 * x[5];
        Ok((energy, g))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let n = x.len();
        let n_atoms = n / 3;
        let mut modes = Array2::zeros((3, n));
        let scale = (n_atoms as f64).sqrt();
        for axis in 0..3 {
            for atom in 0..n_atoms {
                modes[(axis, 3 * atom + axis)] = 1.0 / scale;
            }
        }
        Ok(modes)
    }
}

fn bond_start(n_spectators: usize, separation: f64) -> Array1<f64> {
    let mut x = Array1::zeros(3 * (2 + n_spectators));
    x[3] = separation;
    for s in 0..n_spectators {
        x[3 * (2 + s)] = 1.6 + 0.2 * s as f64;
    }
    x
}

fn max_component(gradient: ArrayView1<f64>) -> f64 {
    gradient.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn projected_eigenvalues(surface: &Bond, x: ArrayView1<f64>) -> Vec<f64> {
    let n = x.len();
    let dr = 1e-5;
    let (_, g0) = surface.eval(x).unwrap();
    let mut h = Array2::zeros((n, n));
    for i in 0..n {
        let mut shifted = x.to_owned();
        shifted[i] += dr;
        let (_, gp) = surface.eval(shifted.view()).unwrap();
        for j in 0..n {
            h[(j, i)] = (gp[j] - g0[j]) / dr;
        }
    }
    for i in 0..n {
        for j in 0..i {
            let s = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = s;
            h[(j, i)] = s;
        }
    }
    let n_atoms = surface.n_atoms();
    let scale = (n_atoms as f64).sqrt();
    for axis in 0..3 {
        let mut t = Array1::zeros(n);
        for atom in 0..n_atoms {
            t[3 * atom + axis] = 1.0 / scale;
        }
        for i in 0..n {
            let hi = (0..n).map(|k| h[(i, k)] * t[k]).sum::<f64>();
            for k in 0..n {
                h[(i, k)] -= hi * t[k];
            }
        }
        for j in 0..n {
            let col = (0..n).map(|i| h[(i, j)] * t[i]).sum::<f64>();
            for i in 0..n {
                h[(i, j)] -= t[i] * col;
            }
        }
    }
    let (vals, _) = exact_eigh(h.view()).unwrap();
    vals.to_vec()
}

#[test]
fn confined_climb_recovers_the_bond_saddle_sooner() {
    let spectators = 8;
    let surface = Bond {
        n_spectators: spectators,
    };
    let start = bond_start(spectators, 2.28);
    let mut mode = Array1::zeros(start.len());
    mode[3] = 1.0;
    for s in 0..spectators {
        mode[3 * (2 + s)] = 0.35;
        mode[3 * (2 + s) + 1] = -0.15;
    }
    let config = MinModeConfig {
        kind: MinModeKind::Dimer,
        max_rotations: 8,
        force_tol: 5e-3,
        max_move: 0.15,
        ..MinModeConfig::default()
    };
    let plain_surface = Counted {
        inner: Bond {
            n_spectators: spectators,
        },
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let weighted_surface = Counted {
        inner: Bond {
            n_spectators: spectators,
        },
        calls: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut plain = MinModeSession::new(config.clone(), start.clone(), mode.clone()).unwrap();
    let mut weighted = MinModeSession::new(config, start.clone(), mode).unwrap();
    weighted.set_confine(Confine::top_force(2)).unwrap();
    let mut plain_convex = 0usize;
    let mut weighted_convex = 0usize;
    let mut trace = String::new();
    for _step in 0..80 {
        let before = plain_surface
            .calls
            .load(std::sync::atomic::Ordering::Relaxed);
        let report = plain.step(&plain_surface).unwrap();
        if report.curvature >= 0.0 {
            plain_convex += plain_surface
                .calls
                .load(std::sync::atomic::Ordering::Relaxed)
                - before;
        }
        if report.status == MinModeStatus::Converged {
            break;
        }
    }
    for step in 0..80 {
        let before = weighted_surface
            .calls
            .load(std::sync::atomic::Ordering::Relaxed);
        let report = weighted.step(&weighted_surface).unwrap();
        if report.curvature >= 0.0 {
            weighted_convex += weighted_surface
                .calls
                .load(std::sync::atomic::Ordering::Relaxed)
                - before;
        }
        let sep = {
            let p = weighted.position();
            let dx = p[3] - p[0];
            let dy = p[4] - p[1];
            let dz = p[5] - p[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        trace.push_str(&format!(
            "{step} curv {:.4} sep {:.4} calls {}\n",
            report.curvature, sep, report.evaluations
        ));
        if report.status == MinModeStatus::Converged {
            break;
        }
    }
    assert!(
        weighted.curvature() < 0.0,
        "weighted curvature {}\n{trace}",
        weighted.curvature()
    );
    let (_, g) = surface.eval(weighted.position()).unwrap();
    assert!(max_component(g.view()) <= 5e-3);
    let bond = (weighted.position()[3] - weighted.position()[0]
        + (weighted.position()[4] - weighted.position()[1])
            .powi(2)
            .sqrt())
    .abs();
    let _ = bond;
    let dx = weighted.position()[3] - weighted.position()[0];
    let dy = weighted.position()[4] - weighted.position()[1];
    let dz = weighted.position()[5] - weighted.position()[2];
    let separation = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!((separation - 1.8).abs() < 0.15, "separation {separation}");
    let evals = projected_eigenvalues(&surface, weighted.position());
    let negative = evals.iter().filter(|v| **v < -1e-3).count();
    assert_eq!(negative, 1, "eigenvalues {evals:?}");
    let weighted_calls = weighted_surface
        .calls
        .load(std::sync::atomic::Ordering::Relaxed);
    let plain_calls = plain_surface
        .calls
        .load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        weighted_convex < plain_convex.max(plain_calls),
        "convex calls weighted {weighted_convex} plain {plain_convex} total {weighted_calls} {plain_calls}"
    );
    assert!(weighted_calls <= plain_calls);
}
