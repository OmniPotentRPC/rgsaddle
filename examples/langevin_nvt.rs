//! NVT Langevin velocity Verlet for one defect pair.
//!
//! [`Langevin`] and [`EnergyGradient`] are not copper-specific. The
//! executable's test is a vacancy plus a <100> split interstitial on the
//! Mishin EAM1 that `tests/mishin_cu.rs` evaluates (Mishin, Mehl,
//! Papaconstantopoulos, Voter, and Kress, Phys. Rev. B 63, 224106 (2001),
//! Table I). Recombination follows the first method of Mwungeli, Liu, and
//! Demkowicz, Phys. Rev. Materials 9, 023601 (2025): an atom is
//! recombined-crystal when its coordination, inside 3.1 Å, is 12, sampled
//! every 0.25 ps. The volume is the shell sum of those capture
//! probabilities, not the survival-curve slope in that paper.
//!
//! Time is the same code unit as `npt_lattice` in the copper test.
//! `angular_scale_rad_s` converts a physical timestep so that
//! `mass_amu * acceleration = gradient` with the gradient in eV/Å.
//!
//! ```text
//! MD_TEMPS=500 MD_SEEDS=8 MD_PS=8 MD_DT_FS=2 MD_GAMMA_PER_PS=1 MD_RMAX=10.8
//! cargo run --release --example langevin_nvt
//! ```
//!
//! `MD_CHECK=1` checks the cohesive energy and times one force call.
//! `MD_CAPTURE=ws` replaces the 12-fold test by Wigner-Seitz occupancy:
//! recombined when every lattice site holds exactly one atom. Above about
//! 600 K thermal vibration alone keeps 100 or more atoms off 12-fold
//! coordination, so the 12-fold count never reaches zero.
//! A CSV row is skipped when the same temperature, shell, orientation,
//! seed, and window are already in the output.
//!
//! `MD_SIA_TRACER=1` runs the free-dumbbell replicas of `MD_MSD=1`, same
//! seeds, for `MD_SIA_BLOCKS` blocks of `MD_SIA_TAU` ps and writes, for
//! every lag between block ends, the squared displacements of the doubly
//! occupied site, of the dumbbell centre, and of its farther atom, and the
//! squared displacement summed over all atoms, which is N times the
//! per-atom G_MSD of Zhang et al., Phys. Rev. Materials 6, 053605 (2022),
//! Eq. (5). `MD_SIA_TRACER_NOTE=1` rewrites the tracer notes and
//! `validation/direct_md_dynamics.md` from their CSVs.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::Instant;

use rgsaddle::{BOLTZMANN_EV_PER_K, angular_scale_rad_s};

const RC: f64 = 5.50679;
const H_CUT: f64 = 0.50037;
const E1: f64 = 2.01458e2;
const E2: f64 = 6.59288e-3;
const R01: f64 = 0.83591;
const R02: f64 = 4.46867;
const ALPHA1: f64 = 2.97758;
const ALPHA2: f64 = 1.54927;
const DELTA: f64 = 0.86225e-2;
const RS: [f64; 3] = [2.24000, 1.80000, 1.20000];
const S_REP: [f64; 3] = [4.0, 40.0, 1.15000e3];
const DENSITY_A: f64 = 3.80362;
const R03: f64 = -2.19885;
const R04: f64 = -2.61984e2;
const BETA1: f64 = 0.17394;
const BETA2: f64 = 5.35661e2;
const F0: f64 = -2.28235;
const F2: f64 = 1.35535;
const Q1: f64 = -1.27775;
const Q2: f64 = -0.86074;
const Q3: f64 = 1.78804;
const Q4: f64 = 2.97571;
const EMB_Q1: f64 = 0.40000;
const EMB_Q2: f64 = 0.30000;

const CU_MASS_AMU: f64 = 63.546;
const COORD_CUTOFF_A: f64 = 3.1;
const SAMPLE_PS: f64 = 0.25;
const DUMBBELL_OFFSET_A: f64 = 1.025;
const TABLE_H: f64 = 2.0e-3;

/// Energy and Cartesian gradient of any potential the driver can step.
pub trait EnergyGradient {
    fn energy_gradient(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
        gradient: &mut [f64],
    ) -> f64;
}

/// One NVT Langevin velocity-Verlet step in the branch's code time.
///
/// The half kicks, the exact Ornstein-Uhlenbeck update, and the two half
/// drifts are the same split as `npt_lattice` in `tests/mishin_cu.rs`.
#[derive(Clone, Copy)]
pub struct Langevin {
    pub dt: f64,
    pub gamma: f64,
    pub mass: f64,
    pub thermal: f64,
}

impl Langevin {
    pub fn from_physical(dt_fs: f64, gamma_per_ps: f64, temperature: f64, mass: f64) -> Self {
        let omega = angular_scale_rad_s();
        Self {
            dt: dt_fs * 1.0e-15 * omega,
            gamma: gamma_per_ps * 1.0e12 / omega,
            mass,
            thermal: BOLTZMANN_EV_PER_K * temperature,
        }
    }

    pub fn step<P: EnergyGradient>(
        &self,
        potential: &mut P,
        positions: &mut [f64],
        velocities: &mut [f64],
        gradient: &mut [f64],
        cell: [[f64; 3]; 3],
        rng: &mut SplitRng,
    ) -> f64 {
        let half = 0.5 * self.dt / self.mass;
        for (velocity, slope) in velocities.iter_mut().zip(gradient.iter()) {
            *velocity -= half * slope;
        }
        for (position, velocity) in positions.iter_mut().zip(velocities.iter()) {
            *position += 0.5 * self.dt * velocity;
        }
        let damp = (-self.gamma * self.dt).exp();
        let noise = (self.thermal / self.mass * (1.0 - damp * damp)).sqrt();
        for velocity in velocities.iter_mut() {
            *velocity = damp * *velocity + noise * rng.normal();
        }
        for (position, velocity) in positions.iter_mut().zip(velocities.iter()) {
            *position += 0.5 * self.dt * velocity;
        }
        let energy = potential.energy_gradient(positions, cell, gradient);
        for (velocity, slope) in velocities.iter_mut().zip(gradient.iter()) {
            *velocity -= half * slope;
        }
        energy
    }
}

pub struct SplitRng {
    state: u64,
}

impl SplitRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

pub struct TrajectoryOutcome {
    pub captured: bool,
    pub time_ps: f64,
    pub steps: usize,
    pub initial_score: usize,
    pub final_score: usize,
}

/// Maxwellian velocities, then velocity Verlet until `recombined` or the window.
///
/// `score` is recorded at the start and at the end. Capture time is the
/// first sample at which `recombined` is true. Sampling starts after the
/// first interval, so a geometry that is already recombined is time zero.
#[allow(clippy::too_many_arguments)]
pub fn langevin_until<P, F, S>(
    potential: &mut P,
    positions: &mut [f64],
    cell: [[f64; 3]; 3],
    thermostat: Langevin,
    equil: Langevin,
    equil_ps: f64,
    window_ps: f64,
    sample_ps: f64,
    dt_fs: f64,
    seed: u64,
    mut recombined: F,
    mut score: S,
) -> TrajectoryOutcome
where
    P: EnergyGradient,
    F: FnMut(&[f64], [[f64; 3]; 3]) -> bool,
    S: FnMut(&[f64], [[f64; 3]; 3]) -> usize,
{
    let mut velocities = vec![0.0; positions.len()];
    let mut rng = SplitRng::new(seed);
    let sigma = (thermostat.thermal / thermostat.mass).sqrt();
    for velocity in &mut velocities {
        *velocity = rng.normal() * sigma;
    }
    zero_com(&mut velocities);
    let mut gradient = vec![0.0; positions.len()];
    potential.energy_gradient(positions, cell, &mut gradient);
    let initial_score = score(positions, cell);
    if recombined(positions, cell) {
        return TrajectoryOutcome {
            captured: true,
            time_ps: 0.0,
            steps: 0,
            initial_score,
            final_score: initial_score,
        };
    }
    let sample_steps = ((sample_ps / (dt_fs * 1.0e-3)).round() as usize).max(1);
    let total_steps = ((window_ps / (dt_fs * 1.0e-3)).round() as usize).max(sample_steps);
    let equil_steps = ((equil_ps / (dt_fs * 1.0e-3)).round() as usize).min(total_steps);
    let mut captured = false;
    let mut when = total_steps;
    for step in 1..=total_steps {
        let phase = if step <= equil_steps {
            equil
        } else {
            thermostat
        };
        phase.step(
            potential,
            positions,
            &mut velocities,
            &mut gradient,
            cell,
            &mut rng,
        );
        if step % sample_steps == 0 && recombined(positions, cell) {
            captured = true;
            when = step;
            break;
        }
    }
    let final_score = score(positions, cell);
    TrajectoryOutcome {
        captured,
        time_ps: when as f64 * dt_fs * 1.0e-3,
        steps: when,
        initial_score,
        final_score,
    }
}

fn zero_com(velocities: &mut [f64]) {
    let n = velocities.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            mean[axis] += velocities[3 * atom + axis];
        }
    }
    let inv = 1.0 / n as f64;
    for atom in 0..n {
        for axis in 0..3 {
            velocities[3 * atom + axis] -= mean[axis] * inv;
        }
    }
}

fn neighbor_bins(home: [f64; 3], nside: [i32; 3]) -> ([usize; 27], usize) {
    let mut bins = [0usize; 27];
    let mut count = 0usize;
    for ox in -1..=1 {
        for oy in -1..=1 {
            for oz in -1..=1 {
                let bx = (home[0] as i32 + ox).rem_euclid(nside[0]);
                let by = (home[1] as i32 + oy).rem_euclid(nside[1]);
                let bz = (home[2] as i32 + oz).rem_euclid(nside[2]);
                let bin = ((bx * nside[1] + by) * nside[2] + bz) as usize;
                if bins[..count].contains(&bin) {
                    continue;
                }
                bins[count] = bin;
                count += 1;
            }
        }
    }
    (bins, count)
}

struct Pair {
    i: u32,
    j: u32,
    d: [f64; 3],
    inv_r: f64,
    pv: f64,
    pd: f64,
    rv: f64,
    rd: f64,
}

/// Half of the 3×3×3 bin stencil. Each unordered pair of bins is seen once
/// when the grid has at least three bins on a side.
const HALF_STENCIL: [(i32, i32, i32); 14] = [
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];

fn morse(r: f64, r0: f64, alpha: f64) -> (f64, f64) {
    let z = -alpha * (r - r0);
    let e1 = z.exp();
    let e2 = (2.0 * z).exp();
    (e2 - 2.0 * e1, 2.0 * alpha * (e1 - e2))
}

fn cutoff_weight(r: f64) -> (f64, f64) {
    let x = (r - RC) / H_CUT;
    if x >= 0.0 {
        return (0.0, 0.0);
    }
    let x2 = x * x;
    let x4 = x2 * x2;
    let den = 1.0 + x4;
    let psi = x4 / den;
    let dpsi = 4.0 * x * x2 / (den * den) / H_CUT;
    (psi, dpsi)
}

fn pair_analytic(r: f64) -> (f64, f64) {
    if r >= RC {
        return (0.0, 0.0);
    }
    let (m1, dm1) = morse(r, R01, ALPHA1);
    let (m2, dm2) = morse(r, R02, ALPHA2);
    let (psi, dpsi) = cutoff_weight(r);
    let core = E1 * m1 + E2 * m2 + DELTA;
    let mut value = core * psi;
    let mut slope = (E1 * dm1 + E2 * dm2) * psi + core * dpsi;
    for (&rs, &strength) in RS.iter().zip(S_REP.iter()) {
        if r < rs {
            let gap = rs - r;
            let gap3 = gap * gap * gap;
            value -= strength * gap * gap3;
            slope += 4.0 * strength * gap3;
        }
    }
    (value, slope)
}

fn density_analytic(r: f64) -> (f64, f64) {
    if r >= RC {
        return (0.0, 0.0);
    }
    let (psi, dpsi) = cutoff_weight(r);
    let d3 = r - R03;
    let g1 = (-BETA1 * d3 * d3).exp();
    let dg1 = g1 * (-2.0 * BETA1 * d3);
    let arg = -BETA2 * (r - R04);
    let (g2, dg2) = if arg < -700.0 {
        (0.0, 0.0)
    } else {
        let g2 = arg.exp();
        (g2, -BETA2 * g2)
    };
    let inner = DENSITY_A * g1 + g2;
    let dinner = DENSITY_A * dg1 + dg2;
    (inner * psi, dinner * psi + inner * dpsi)
}

fn embed_analytic(rho: f64) -> (f64, f64) {
    let x = rho - 1.0;
    let x2 = x * x;
    let x3 = x2 * x;
    if rho < 1.0 {
        let value = F0 + 0.5 * F2 * x2 + Q1 * x3 + Q2 * x2 * x2 + Q3 * x2 * x3 + Q4 * x3 * x3;
        let slope =
            F2 * x + 3.0 * Q1 * x2 + 4.0 * Q2 * x3 + 5.0 * Q3 * x2 * x2 + 6.0 * Q4 * x2 * x3;
        (value, slope)
    } else {
        let num = F0 + 0.5 * F2 * x2 + Q1 * x3 + EMB_Q1 * x2 * x2;
        let den = 1.0 + EMB_Q2 * x3;
        let dnum = F2 * x + 3.0 * Q1 * x2 + 4.0 * EMB_Q1 * x3;
        let dden = 3.0 * EMB_Q2 * x2;
        (num / den, (dnum * den - num * dden) / (den * den))
    }
}

fn positive_image(sx: i32, sy: i32, sz: i32) -> bool {
    sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0)))
}

/// Mishin Cu EAM1. Orthogonal cells with side longer than `2 RC` use a
/// cell list and a cubic Hermite table of the analytic pair and density.
struct MishinCu {
    table: Vec<[f64; 4]>,
    knots: usize,
    pairs: Vec<Pair>,
    rho: Vec<f64>,
    fp: Vec<f64>,
    counts: Vec<u32>,
    offsets: Vec<u32>,
    order: Vec<u32>,
    packed: Vec<[f64; 3]>,
    neigh: Vec<(u32, u32)>,
    anchor: Vec<f64>,
}

impl MishinCu {
    fn new() -> Self {
        let knots = (RC / TABLE_H).ceil() as usize + 1;
        let mut table = vec![[0.0; 4]; knots];
        for (index, knot) in table.iter_mut().enumerate() {
            let r = index as f64 * TABLE_H;
            let (pair_value, pair_slope) = pair_analytic(r);
            let (density, density_slope) = density_analytic(r);
            *knot = [pair_value, pair_slope, density, density_slope];
        }
        Self {
            table,
            knots,
            pairs: Vec::new(),
            rho: Vec::new(),
            fp: Vec::new(),
            counts: Vec::new(),
            offsets: Vec::new(),
            order: Vec::new(),
            packed: Vec::new(),
            neigh: Vec::new(),
            anchor: Vec::new(),
        }
    }

    #[inline(always)]
    fn sample(&self, r: f64) -> [f64; 4] {
        if r >= RC {
            return [0.0; 4];
        }
        let t_grid = (r / TABLE_H).max(0.0);
        let mut index = t_grid.floor() as usize;
        if index + 1 >= self.knots {
            index = self.knots - 2;
        }
        let t = t_grid - index as f64;
        let one_t = 1.0 - t;
        let h00 = (1.0 + 2.0 * t) * one_t * one_t;
        let h10 = t * one_t * one_t;
        let h01 = t * t * (3.0 - 2.0 * t);
        let h11 = t * t * (t - 1.0);
        let dh00 = -6.0 * t * one_t;
        let dh10 = one_t * (1.0 - 3.0 * t);
        let dh01 = 6.0 * t * one_t;
        let dh11 = 3.0 * t * t - 2.0 * t;
        let left = self.table[index];
        let right = self.table[index + 1];
        let mut out = [0.0; 4];
        for slot in 0..2 {
            let f0 = left[2 * slot];
            let d0 = left[2 * slot + 1];
            let f1 = right[2 * slot];
            let d1 = right[2 * slot + 1];
            out[2 * slot] = h00 * f0 + h10 * TABLE_H * d0 + h01 * f1 + h11 * TABLE_H * d1;
            out[2 * slot + 1] =
                (dh00 * f0 + dh10 * TABLE_H * d0 + dh01 * f1 + dh11 * TABLE_H * d1) / TABLE_H;
        }
        out
    }

    fn push_pair(&mut self, i: usize, j: usize, d: [f64; 3], r: f64) {
        let sampled = self.sample(r);
        self.pairs.push(Pair {
            i: i as u32,
            j: j as u32,
            d,
            inv_r: 1.0 / r,
            pv: sampled[0],
            pd: sampled[1],
            rv: sampled[2],
            rd: sampled[3],
        });
    }

    fn accumulate(&mut self, n: usize, gradient: &mut [f64]) -> f64 {
        self.rho.clear();
        self.rho.resize(n, 0.0);
        for bond in &self.pairs {
            let i = bond.i as usize;
            self.rho[i] += bond.rv;
            if bond.i != bond.j {
                self.rho[bond.j as usize] += bond.rv;
            } else {
                self.rho[i] += bond.rv;
            }
        }
        let mut energy = 0.0;
        self.fp.clear();
        self.fp.resize(n, 0.0);
        for i in 0..n {
            let (value, slope) = embed_analytic(self.rho[i]);
            energy += value;
            self.fp[i] = slope;
        }
        gradient.fill(0.0);
        for bond in &self.pairs {
            let i = bond.i as usize;
            if bond.i != bond.j {
                let j = bond.j as usize;
                energy += bond.pv;
                let deriv = bond.pd + (self.fp[i] + self.fp[j]) * bond.rd;
                let scale = deriv * bond.inv_r;
                for axis in 0..3 {
                    let force = scale * bond.d[axis];
                    gradient[3 * i + axis] -= force;
                    gradient[3 * j + axis] += force;
                }
            } else {
                energy += bond.pv;
            }
        }
        energy
    }
}

impl EnergyGradient for MishinCu {
    fn energy_gradient(
        &mut self,
        positions: &[f64],
        cell: [[f64; 3]; 3],
        gradient: &mut [f64],
    ) -> f64 {
        let n = positions.len() / 3;
        let lengths = [cell[0][0].abs(), cell[1][1].abs(), cell[2][2].abs()];
        let orthogonal = cell[0][1].abs()
            + cell[0][2].abs()
            + cell[1][0].abs()
            + cell[1][2].abs()
            + cell[2][0].abs()
            + cell[2][1].abs()
            < 1.0e-8;
        let skin = 1.0;
        if orthogonal && lengths.iter().all(|length| *length > 2.0 * (RC + skin)) {
            if self.list_stale(positions, lengths, skin) {
                self.pairs.clear();
                self.pairs_cell(positions, lengths, RC + skin);
                self.neigh.clear();
                self.neigh
                    .extend(self.pairs.iter().map(|bond| (bond.i, bond.j)));
                self.anchor.clear();
                self.anchor.extend_from_slice(positions);
                self.pairs.clear();
            }
            self.pairs_from_list(positions, lengths);
        } else {
            self.pairs.clear();
            self.anchor.clear();
            self.neigh.clear();
            if orthogonal {
                self.pairs_cell(positions, lengths, RC);
            } else {
                self.pairs_images(positions, cell);
            }
        }
        self.accumulate(n, gradient)
    }
}

impl MishinCu {
    fn pairs_images(&mut self, positions: &[f64], cell: [[f64; 3]; 3]) {
        let n = positions.len() / 3;
        let lengths = [0, 1, 2].map(|row| {
            (cell[row][0] * cell[row][0]
                + cell[row][1] * cell[row][1]
                + cell[row][2] * cell[row][2])
                .sqrt()
        });
        let min_len = lengths.into_iter().fold(f64::MAX, f64::min);
        let nimg = ((RC / min_len).ceil() as i32).max(1);
        let cutoff2 = RC * RC;
        for i in 0..n {
            let ri = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            for j in i..n {
                let rj = [positions[3 * j], positions[3 * j + 1], positions[3 * j + 2]];
                for sx in -nimg..=nimg {
                    for sy in -nimg..=nimg {
                        for sz in -nimg..=nimg {
                            if i == j && !positive_image(sx, sy, sz) {
                                continue;
                            }
                            if i == j && sx == 0 && sy == 0 && sz == 0 {
                                continue;
                            }
                            let shift = [
                                sx as f64 * cell[0][0]
                                    + sy as f64 * cell[1][0]
                                    + sz as f64 * cell[2][0],
                                sx as f64 * cell[0][1]
                                    + sy as f64 * cell[1][1]
                                    + sz as f64 * cell[2][1],
                                sx as f64 * cell[0][2]
                                    + sy as f64 * cell[1][2]
                                    + sz as f64 * cell[2][2],
                            ];
                            let d = [
                                rj[0] + shift[0] - ri[0],
                                rj[1] + shift[1] - ri[1],
                                rj[2] + shift[2] - ri[2],
                            ];
                            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r2 > 1.0e-16 && r2 < cutoff2 {
                                self.push_pair(i, j, d, r2.sqrt());
                            }
                        }
                    }
                }
            }
        }
    }

    fn list_stale(&self, positions: &[f64], lengths: [f64; 3], skin: f64) -> bool {
        if self.anchor.len() != positions.len() || self.neigh.is_empty() {
            return true;
        }
        let limit = 0.25 * skin * skin;
        for atom in 0..positions.len() / 3 {
            let mut distance = 0.0;
            for axis in 0..3 {
                let mut delta = positions[3 * atom + axis] - self.anchor[3 * atom + axis];
                delta -= lengths[axis] * (delta / lengths[axis]).round();
                distance += delta * delta;
            }
            if distance > limit {
                return true;
            }
        }
        false
    }

    fn pairs_from_list(&mut self, positions: &[f64], lengths: [f64; 3]) {
        self.pairs.clear();
        let cutoff2 = RC * RC;
        let neigh = self.neigh.clone();
        for &(left, right) in &neigh {
            let i = left as usize;
            let j = right as usize;
            let mut d = [0.0; 3];
            for axis in 0..3 {
                d[axis] = positions[3 * j + axis] - positions[3 * i + axis];
                d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 > 1.0e-16 && r2 < cutoff2 {
                self.push_pair(i, j, d, r2.sqrt());
            }
        }
    }

    fn pairs_cell(&mut self, positions: &[f64], lengths: [f64; 3], cutoff: f64) {
        let n = positions.len() / 3;
        let nside = lengths.map(|length| ((length / cutoff).floor() as i32).max(1));
        let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
        self.counts.clear();
        self.counts.resize(bins_n, 0);
        let mut bins = vec![0usize; n];
        for i in 0..n {
            let bin = bin_of(
                [
                    positions[3 * i] / lengths[0],
                    positions[3 * i + 1] / lengths[1],
                    positions[3 * i + 2] / lengths[2],
                ],
                nside,
            );
            bins[i] = bin;
            self.counts[bin] += 1;
        }
        self.offsets.clear();
        self.offsets.resize(bins_n + 1, 0);
        for bin in 0..bins_n {
            self.offsets[bin + 1] = self.offsets[bin] + self.counts[bin];
        }
        let mut cursor = self.offsets.clone();
        self.order.clear();
        self.order.resize(n, 0);
        self.packed.clear();
        self.packed.resize(n, [0.0; 3]);
        for i in 0..n {
            let bin = bins[i];
            let slot = cursor[bin] as usize;
            self.order[slot] = i as u32;
            self.packed[slot] = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            cursor[bin] += 1;
        }
        let cutoff2 = cutoff * cutoff;
        for slot_i in 0..n {
            let i = self.order[slot_i] as usize;
            let ri = self.packed[slot_i];
            let bin_i = bins[i] as i32;
            let ix = bin_i / (nside[1] * nside[2]);
            let rem = bin_i % (nside[1] * nside[2]);
            let iy = rem / nside[2];
            let iz = rem % nside[2];
            let mut seen = [usize::MAX; 14];
            let mut nseen = 0usize;
            for (ox, oy, oz) in HALF_STENCIL {
                let bx = (ix + ox).rem_euclid(nside[0]);
                let by = (iy + oy).rem_euclid(nside[1]);
                let bz = (iz + oz).rem_euclid(nside[2]);
                let bin = ((bx * nside[1] + by) * nside[2] + bz) as usize;
                if seen[..nseen].contains(&bin) {
                    continue;
                }
                seen[nseen] = bin;
                nseen += 1;
                let start = self.offsets[bin] as usize;
                let end = self.offsets[bin + 1] as usize;
                for slot in start..end {
                    let j = self.order[slot] as usize;
                    if ox == 0 && oy == 0 && oz == 0 && j <= i {
                        continue;
                    }
                    let rj = self.packed[slot];
                    let mut d = [rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]];
                    for axis in 0..3 {
                        d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
                    }
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 > 1.0e-16 && r2 < cutoff2 {
                        self.push_pair(i, j, d, r2.sqrt());
                    }
                }
            }
        }
    }
}

fn bin_of(frac: [f64; 3], nside: [i32; 3]) -> usize {
    let ix = (frac[0].rem_euclid(1.0) * nside[0] as f64) as i32 % nside[0];
    let iy = (frac[1].rem_euclid(1.0) * nside[1] as f64) as i32 % nside[1];
    let iz = (frac[2].rem_euclid(1.0) * nside[2] as f64) as i32 % nside[2];
    ((ix * nside[1] + iy) * nside[2] + iz) as usize
}

fn analytic_gradient(positions: &[f64], cell: [[f64; 3]; 3], gradient: &mut [f64]) -> f64 {
    let mut pot = MishinCu::new();
    pot.pairs_images(positions, cell);
    let n = positions.len() / 3;
    pot.rho.resize(n, 0.0);
    for bond in &pot.pairs {
        let r = 1.0 / bond.inv_r;
        let weight = density_analytic(r).0;
        let i = bond.i as usize;
        pot.rho[i] += weight;
        if bond.i != bond.j {
            pot.rho[bond.j as usize] += weight;
        } else {
            pot.rho[i] += weight;
        }
    }
    let mut energy = 0.0;
    pot.fp.resize(n, 0.0);
    for i in 0..n {
        let (value, slope) = embed_analytic(pot.rho[i]);
        energy += value;
        pot.fp[i] = slope;
    }
    gradient.fill(0.0);
    for bond in &pot.pairs {
        let r = 1.0 / bond.inv_r;
        let (value, dv) = pair_analytic(r);
        let drho = density_analytic(r).1;
        let i = bond.i as usize;
        if bond.i != bond.j {
            let j = bond.j as usize;
            energy += value;
            let deriv = dv + (pot.fp[i] + pot.fp[j]) * drho;
            for axis in 0..3 {
                let force = deriv * bond.d[axis] * bond.inv_r;
                gradient[3 * i + axis] -= force;
                gradient[3 * j + axis] += force;
            }
        } else {
            energy += value;
        }
    }
    energy
}

/// Atoms whose neighbor count inside 3.1 Å is not 12.
pub fn coordination_deficit(positions: &[f64], cell: [[f64; 3]; 3]) -> usize {
    let n = positions.len() / 3;
    let lengths = [cell[0][0], cell[1][1], cell[2][2]];
    let cutoff2 = COORD_CUTOFF_A * COORD_CUTOFF_A;
    let mut coord = vec![0u16; n];
    if lengths.iter().all(|length| *length > 2.0 * COORD_CUTOFF_A) {
        let nside = lengths.map(|length| ((length / COORD_CUTOFF_A).floor() as i32).max(1));
        let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
        let mut members = vec![Vec::new(); bins_n];
        for i in 0..n {
            let bin = bin_of(
                [
                    positions[3 * i] / lengths[0],
                    positions[3 * i + 1] / lengths[1],
                    positions[3 * i + 2] / lengths[2],
                ],
                nside,
            );
            members[bin].push(i);
        }
        for i in 0..n {
            let ri = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
            let home = [
                (ri[0] / lengths[0]).rem_euclid(1.0) * nside[0] as f64,
                (ri[1] / lengths[1]).rem_euclid(1.0) * nside[1] as f64,
                (ri[2] / lengths[2]).rem_euclid(1.0) * nside[2] as f64,
            ];
            let (bins, bin_count) = neighbor_bins(home, nside);
            for bin in bins.into_iter().take(bin_count) {
                for &j in &members[bin] {
                    if j <= i {
                        continue;
                    }
                    let mut d = [
                        positions[3 * j] - ri[0],
                        positions[3 * j + 1] - ri[1],
                        positions[3 * j + 2] - ri[2],
                    ];
                    for axis in 0..3 {
                        d[axis] -= lengths[axis] * (d[axis] / lengths[axis]).round();
                    }
                    let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    if r2 < cutoff2 {
                        coord[i] += 1;
                        coord[j] += 1;
                    }
                }
            }
        }
    }
    coord.into_iter().filter(|count| *count != 12).count()
}

pub fn crystal_recombined(positions: &[f64], cell: [[f64; 3]; 3]) -> bool {
    coordination_deficit(positions, cell) == 0
}

fn fcc(n: usize, a: f64) -> (Vec<f64>, [[f64; 3]; 3]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut values = Vec::with_capacity(n * n * n * 12);
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for site in basis {
                    values.push((ix as f64 + site[0]) * a);
                    values.push((iy as f64 + site[1]) * a);
                    values.push((iz as f64 + site[2]) * a);
                }
            }
        }
    }
    let length = n as f64 * a;
    (
        values,
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    )
}

fn shell_table(a: f64, r_max: f64) -> Vec<(f64, [f64; 3], usize)> {
    let limit = ((2.0 * r_max / a).ceil() as i32) + 1;
    let mut raw = Vec::new();
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                let r = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                if r < 0.5 * a || r > r_max {
                    continue;
                }
                raw.push((r, cart));
            }
        }
    }
    raw.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut found: Vec<(f64, [f64; 3], usize)> = Vec::new();
    for (radius, cart) in raw {
        if let Some(last) = found.last_mut()
            && (radius - last.0).abs() < 1.0e-3
        {
            last.2 += 1;
            continue;
        }
        found.push((radius, cart, 1));
    }
    found
}

/// Vacancy plus a <100> dumbbell on one shell site. Atom count is unchanged.
pub fn place_vacancy_dumbbell(
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
) -> (Vec<f64>, [[f64; 3]; 3]) {
    let (flat, cell) = fcc(repeats, a);
    let n = flat.len() / 3;
    let mut points = Vec::with_capacity(n);
    for atom in 0..n {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
    let vacancy = nearest(&points, center, None);
    let sia_guess = [
        points[vacancy][0] + shell[0],
        points[vacancy][1] + shell[1],
        points[vacancy][2] + shell[2],
    ];
    let sia = nearest(&points, sia_guess, Some(vacancy));
    let sia_pos = points[sia];
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1.0e-8);
    let unit = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let mut keep = Vec::with_capacity(n);
    for (index, point) in points.iter().enumerate() {
        if index == sia || index == vacancy {
            continue;
        }
        keep.push(*point);
    }
    keep.push([
        sia_pos[0] - DUMBBELL_OFFSET_A * unit[0],
        sia_pos[1] - DUMBBELL_OFFSET_A * unit[1],
        sia_pos[2] - DUMBBELL_OFFSET_A * unit[2],
    ]);
    keep.push([
        sia_pos[0] + DUMBBELL_OFFSET_A * unit[0],
        sia_pos[1] + DUMBBELL_OFFSET_A * unit[1],
        sia_pos[2] + DUMBBELL_OFFSET_A * unit[2],
    ]);
    let mut values = Vec::with_capacity(keep.len() * 3);
    for point in keep {
        values.extend(point);
    }
    (values, cell)
}

fn nearest(points: &[[f64; 3]], target: [f64; 3], skip: Option<usize>) -> usize {
    let mut best = 0usize;
    let mut best_r = f64::MAX;
    for (index, point) in points.iter().enumerate() {
        if skip == Some(index) {
            continue;
        }
        let d = [
            point[0] - target[0],
            point[1] - target[1],
            point[2] - target[2],
        ];
        let r = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r < best_r {
            best_r = r;
            best = index;
        }
    }
    best
}

fn measured_lattice(temperature: f64) -> f64 {
    match temperature as i32 {
        300 => 3.628311,
        400 => 3.633713,
        500 => 3.638666,
        600 => 3.644160,
        700 => 3.649788,
        _ => 3.638666,
    }
}

const ORIENTATIONS: [(&str, [f64; 3]); 3] = [
    ("100", [1.0, 0.0, 0.0]),
    ("010", [0.0, 1.0, 0.0]),
    ("001", [0.0, 0.0, 1.0]),
];

struct Job {
    temperature: f64,
    lattice: f64,
    shell_r: f64,
    shell: [f64; 3],
    shell_sites: usize,
    orientation: &'static str,
    axis: [f64; 3],
    seed_index: u64,
    seed: u64,
}

struct RunSpec {
    temps: Vec<f64>,
    seeds: u64,
    window_ps: f64,
    dt_fs: f64,
    gamma_per_ps: f64,
    r_max_500: f64,
    repeats: usize,
    capture_ws: bool,
}

fn env_f64(key: &str, default: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

fn run_spec() -> RunSpec {
    let temps = env::var("MD_TEMPS").unwrap_or_else(|_| "500".to_string());
    let temps = temps
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| item.trim().parse::<f64>().unwrap_or(500.0))
        .collect();
    RunSpec {
        temps,
        seeds: env_u64("MD_SEEDS", 8),
        window_ps: env_f64("MD_PS", 8.0),
        dt_fs: env_f64("MD_DT_FS", 2.0),
        gamma_per_ps: env_f64("MD_GAMMA_PER_PS", 1.0),
        r_max_500: env_f64("MD_RMAX", 10.8),
        repeats: env_u64("MD_REPEATS", 13) as usize,
        capture_ws: env::var("MD_CAPTURE").ok().as_deref() == Some("ws"),
    }
}

fn jobs_for(spec: &RunSpec) -> Vec<Job> {
    let mut jobs = Vec::new();
    let a500 = measured_lattice(500.0);
    for &temperature in &spec.temps {
        let lattice = measured_lattice(temperature);
        let r_max = spec.r_max_500 * lattice / a500;
        for (radius, vector, count) in shell_table(lattice, r_max) {
            for (label, axis) in ORIENTATIONS {
                for seed_index in 0..spec.seeds {
                    let seed = mix_seed(&[
                        temperature.to_bits(),
                        radius.to_bits(),
                        label.as_bytes()[0] as u64,
                        seed_index,
                        0xC0FFEE,
                    ]);
                    jobs.push(Job {
                        temperature,
                        lattice,
                        shell_r: radius,
                        shell: vector,
                        shell_sites: count,
                        orientation: label,
                        axis,
                        seed_index,
                        seed,
                    });
                }
            }
        }
    }
    jobs
}

fn mix_seed(parts: &[u64]) -> u64 {
    let mut hash = 0x9E37_79B9_7F4A_7C15u64;
    for part in parts {
        hash ^= part.wrapping_add(0x9E37_79B9_7F4A_7C15);
        hash = hash.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        hash ^= hash >> 30;
    }
    hash | 1
}

fn output_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("validation");
    path.push("direct_md_viv.csv");
    if let Ok(over) = env::var("MD_OUT") {
        return PathBuf::from(over);
    }
    path
}

fn note_path(csv: &std::path::Path) -> PathBuf {
    csv.with_extension("md")
}

const HEADER: &str = "temperature_K,lattice_A,repeats,n_atoms,shell_A,shell_x,shell_y,shell_z,orientation,shell_sites,seed,dt_fs,gamma_per_ps,window_ps,captured,time_ps,steps,initial_non12,final_non12";

fn row_key(temperature: f64, shell: f64, orientation: &str, seed: u64, window: f64) -> String {
    format!("{temperature:.0},{shell:.6},{orientation},{seed},{window:.4}")
}

fn existing_keys(path: &std::path::Path, window: f64) -> BTreeMap<String, ()> {
    let mut keys = BTreeMap::new();
    let Ok(file) = File::open(path) else {
        return keys;
    };
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 16 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(seed) = fields[10].parse::<u64>() else {
            continue;
        };
        let Ok(stored_window) = fields[13].parse::<f64>() else {
            continue;
        };
        if (stored_window - window).abs() > 1.0e-6 {
            continue;
        }
        keys.insert(row_key(temperature, shell, fields[8], seed, window), ());
    }
    keys
}

fn run_job(potential: &mut MishinCu, spec: &RunSpec, job: &Job) -> (usize, TrajectoryOutcome) {
    let (mut positions, cell) =
        place_vacancy_dumbbell(spec.repeats, job.lattice, job.shell, job.axis);
    let n_atoms = positions.len() / 3;
    let thermostat =
        Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, job.temperature, CU_MASS_AMU);
    let equil = Langevin::from_physical(spec.dt_fs, 1.0, job.temperature, CU_MASS_AMU);
    let outcome = if spec.capture_ws {
        let reference = mean_position(&positions);
        let defects = |positions: &[f64], _: [[f64; 3]; 3]| {
            let mean = mean_position(positions);
            let shift = [0, 1, 2].map(|axis| mean[axis] - reference[axis]);
            occupancy_defects(positions, job.lattice, spec.repeats, shift)
        };
        langevin_until(
            potential,
            &mut positions,
            cell,
            thermostat,
            equil,
            2.0,
            spec.window_ps,
            SAMPLE_PS,
            spec.dt_fs,
            job.seed,
            |positions, cell| defects(positions, cell) == 0,
            defects,
        )
    } else {
        langevin_until(
            potential,
            &mut positions,
            cell,
            thermostat,
            equil,
            2.0,
            spec.window_ps,
            SAMPLE_PS,
            spec.dt_fs,
            job.seed,
            crystal_recombined,
            coordination_deficit,
        )
    };
    (n_atoms, outcome)
}

fn format_row(spec: &RunSpec, job: &Job, n_atoms: usize, outcome: &TrajectoryOutcome) -> String {
    format!(
        "{:.0},{:.6},{},{n_atoms},{:.6},{:.8},{:.8},{:.8},{},{},{},{:.4},{:.4},{:.4},{},{:.4},{},{},{}",
        job.temperature,
        job.lattice,
        spec.repeats,
        job.shell_r,
        job.shell[0],
        job.shell[1],
        job.shell[2],
        job.orientation,
        job.shell_sites,
        job.seed_index,
        spec.dt_fs,
        spec.gamma_per_ps,
        spec.window_ps,
        u8::from(outcome.captured),
        outcome.time_ps,
        outcome.steps,
        outcome.initial_score,
        outcome.final_score,
    )
}

struct Group {
    temperature: f64,
    window: f64,
    shell: f64,
    orientation: String,
    sites: f64,
    captured: f64,
    trials: f64,
}

const DUMBBELL_ALPHA: f64 = 2.0 / 3.0;

/// Rate of one of the eight translation channels of a `<100>` dumbbell
/// with defect diffusivity `diffusivity`, `ν = D / (α a²)`: the SIA
/// migration frequency of `K_iv = V_iv D_i / (α a²) = V_iv ν` in section 1
/// of Mwungeli, Liu, and Demkowicz.
fn dumbbell_channel_rate(lattice: f64, diffusivity: f64) -> f64 {
    diffusivity / (DUMBBELL_ALPHA * lattice * lattice)
}

/// `V_iv / Ω = k / (ν Ω) = 4 α k / (a D_i)` for a capture rate `k` in Å³/ps.
fn rate_to_volume(rate: f64, lattice: f64, d_i: f64) -> f64 {
    rate / dumbbell_channel_rate(lattice, d_i) / (lattice.powi(3) / 4.0)
}

/// Mean and standard error of `D_msd` over the `MD_MSD` replicas of
/// `validation/direct_md_sia_d.csv` with this temperature, lattice constant,
/// timestep, and friction, Å²/ps, and the number of replicas.
fn replica_diffusivity(
    temperature: f64,
    lattice: f64,
    dt_fs: f64,
    gamma_per_ps: f64,
) -> Option<(f64, f64, usize)> {
    let text = fs::read_to_string(validation_file("direct_md_sia_d.csv")).ok()?;
    let values: Vec<f64> = text
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let value = |index: usize| fields.get(index)?.parse::<f64>().ok();
            let matches = (value(0)? - temperature).abs() < 0.5
                && (value(1)? - lattice).abs() < 1.0e-5
                && (value(3)? - dt_fs).abs() < 1.0e-9
                && (value(4)? - gamma_per_ps).abs() < 1.0e-9;
            if matches { value(12) } else { None }
        })
        .collect();
    (values.len() > 1).then(|| {
        let (mean, error) = mean_and_error(&values);
        (mean, error, values.len())
    })
}

fn first_method_report(csv: &std::path::Path) -> String {
    #[derive(Clone)]
    struct Trial {
        temperature: f64,
        lattice: f64,
        repeats: f64,
        shell: f64,
        orientation: String,
        sites: f64,
        dt_fs: f64,
        gamma: f64,
        window: f64,
        captured: bool,
        time_ps: f64,
    }
    let Ok(file) = File::open(csv) else {
        return String::new();
    };
    let mut trials = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 19 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(lattice) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(repeats) = fields[2].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(sites) = fields[9].parse::<f64>() else {
            continue;
        };
        let Ok(dt_fs) = fields[11].parse::<f64>() else {
            continue;
        };
        let Ok(gamma) = fields[12].parse::<f64>() else {
            continue;
        };
        let Ok(window) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(captured) = fields[14].parse::<u8>() else {
            continue;
        };
        let Ok(time_ps) = fields[15].parse::<f64>() else {
            continue;
        };
        trials.push(Trial {
            temperature,
            lattice,
            repeats,
            shell,
            orientation: fields[8].to_string(),
            sites,
            dt_fs,
            gamma,
            window,
            captured: captured == 1,
            time_ps,
        });
    }
    if trials.is_empty() {
        return String::new();
    }
    let mut text = String::new();
    text.push_str("## Capture fraction against time\n\n");
    text.push_str("The fraction is the number of trajectories of that shell and orientation that had recombined by that time, divided by the number of trajectories. `V_shell` is the shell sum, sites/3 times the fraction, over every shell that has all three orientations. Sites outside 10.8 Å are not in `V_shell`.\n\n");
    let mut temperatures: BTreeMap<u64, f64> = BTreeMap::new();
    for trial in &trials {
        temperatures.insert(trial.temperature.to_bits(), trial.temperature);
    }
    let marks = [
        0.25_f64, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 150.0, 200.0, 300.0,
    ];
    for temperature in temperatures.values() {
        let members: Vec<&Trial> = trials
            .iter()
            .filter(|trial| trial.temperature == *temperature)
            .collect();
        let window = members[0].window;
        let lattice = members[0].lattice;
        let repeats = members[0].repeats;
        text.push_str(&format!("### {temperature:.0} K\n\n"));
        text.push_str("| shell (Å) | orientation |");
        for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            text.push_str(&format!(" {mark:.0} ps |"));
        }
        text.push_str(" last new capture (ps) |\n|---:|:---:|");
        for _ in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            text.push_str("---:|");
        }
        text.push_str("---:|\n");
        let mut groups: BTreeMap<(u64, String), Vec<&Trial>> = BTreeMap::new();
        for trial in &members {
            groups
                .entry((trial.shell.to_bits(), trial.orientation.clone()))
                .or_default()
                .push(trial);
        }
        for ((_, _), group) in &groups {
            let trial = group[0];
            text.push_str(&format!(
                "| {shell:.4} | {orient} |",
                shell = trial.shell,
                orient = trial.orientation
            ));
            for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
                let hit = group
                    .iter()
                    .filter(|item| item.captured && item.time_ps <= *mark)
                    .count();
                text.push_str(&format!(" {:.3} |", hit as f64 / group.len() as f64));
            }
            let last = group
                .iter()
                .filter(|item| item.captured)
                .map(|item| item.time_ps)
                .fold(0.0, f64::max);
            text.push_str(&format!(" {last:.2} |\n"));
        }
        text.push_str("\n| time (ps) | V_shell/Ω |\n|---:|---:|\n");
        let mut shells: BTreeMap<u64, Vec<&Trial>> = BTreeMap::new();
        for trial in &members {
            shells.entry(trial.shell.to_bits()).or_default().push(trial);
        }
        for mark in marks.iter().filter(|mark| **mark <= window + 1.0e-9) {
            let mut volume = 0.0;
            for group in shells.values() {
                let mut orients: BTreeMap<&str, Vec<&Trial>> = BTreeMap::new();
                for trial in group {
                    orients
                        .entry(trial.orientation.as_str())
                        .or_default()
                        .push(trial);
                }
                if orients.len() < 3 {
                    continue;
                }
                for orient in orients.values() {
                    let hit = orient
                        .iter()
                        .filter(|item| item.captured && item.time_ps <= *mark)
                        .count();
                    volume += orient[0].sites / 3.0 * hit as f64 / orient.len() as f64;
                }
            }
            text.push_str(&format!("| {mark:.2} | {volume:.2} |\n"));
        }
        text.push('\n');
        let diffusivity = table_diffusivity(*temperature);
        let total_sites = repeats * repeats * repeats * 4.0;
        let model_volume = (repeats * lattice).powi(3);
        let mut fit_times = Vec::new();
        for mark in marks
            .iter()
            .filter(|mark| **mark <= window + 1.0e-9 && **mark > 0.0)
        {
            let mut captured_sites = 0.0;
            for group in shells.values() {
                let mut orients: BTreeMap<&str, Vec<&Trial>> = BTreeMap::new();
                for trial in group {
                    orients
                        .entry(trial.orientation.as_str())
                        .or_default()
                        .push(trial);
                }
                if orients.len() < 3 {
                    continue;
                }
                for orient in orients.values() {
                    let hit = orient
                        .iter()
                        .filter(|item| item.captured && item.time_ps <= *mark)
                        .count();
                    captured_sites += orient[0].sites / 3.0 * hit as f64 / orient.len() as f64;
                }
            }
            let survival = (1.0 - captured_sites / total_sites).max(1.0e-12);
            fit_times.push((*mark, survival.ln()));
        }
        text.push_str("The first method of Mwungeli, Liu, and Demkowicz fits ln(n(t)/N) against time through the origin. The slope is -k/V_model, with V_model the 13³ cell volume. n(t)/N here counts every lattice site: a simulated shell contributes its capture fraction, and a site outside 10.8 Å is counted as not yet recombined. The recombination volume is V_iv = k α a² / D_i with α = 2/3, from K_iv = V_iv D_i / (α a²) = V_iv ν in section 1 of that paper, divided by Ω = a³/4.\n\n");
        if fit_times.is_empty() {
            continue;
        }
        let denominator: f64 = fit_times.iter().map(|(time, _)| time * time).sum();
        let slope = fit_times
            .iter()
            .map(|(time, log_s)| time * log_s)
            .sum::<f64>()
            / denominator;
        let residual = fit_times
            .iter()
            .map(|(time, log_s)| {
                let miss = log_s - slope * time;
                miss * miss
            })
            .sum::<f64>();
        let dof = (fit_times.len() as f64 - 1.0).max(1.0);
        let slope_sigma = (residual / dof / denominator).sqrt();
        let rate = -slope * model_volume;
        let rate_sigma = slope_sigma * model_volume;
        text.push_str(&format!(
            "Fit on {} samples up to {window:.0} ps: slope {slope:.6e} ± {slope_sigma:.6e} /ps, k {rate:.6e} ± {rate_sigma:.6e} Å³/ps.\n\n",
            fit_times.len()
        ));
        match diffusivity {
            Some((d_i, d_sigma)) if d_i > 0.0 => {
                let per_atom = rate_to_volume(rate, lattice, d_i);
                let rel = ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                text.push_str(&format!(
                    "Table 1 lists D_i = {d_i:.3} ± {d_sigma:.3} Å²/ps at this temperature, 4951 D* from Ref. [12] (see `direct_md_dynamics.md`). With it, first-method V_iv = {per_atom:.2} ± {:.2} Ω.\n\n",
                    per_atom * rel
                ));
            }
            _ => {
                text.push_str("No Table 1 diffusivity is recorded for this temperature, so k is not converted with it.\n\n");
            }
        }
        let dynamics = (members[0].dt_fs, members[0].gamma);
        match replica_diffusivity(*temperature, lattice, dynamics.0, dynamics.1) {
            Some((d_i, d_sigma, replicas)) if d_i > 0.0 => {
                let per_atom = rate_to_volume(rate, lattice, d_i);
                let rel = ((rate_sigma / rate).powi(2) + (d_sigma / d_i).powi(2)).sqrt();
                text.push_str(&format!(
                    "The defect diffusivity with these dynamics, `D_msd` of `direct_md_sia_d.md` over {replicas} replicas, is {d_i:.3} ± {d_sigma:.3} Å²/ps. With it, first-method V_iv = {per_atom:.2} ± {:.2} Ω.\n\n",
                    per_atom * rel
                ));
            }
            _ => {
                text.push_str("No free-dumbbell replica with these dynamics is recorded for this temperature, so k is not converted with a defect diffusivity.\n\n");
            }
        }
    }
    text
}

/// Table 1 D_i of Mwungeli, Liu, and Demkowicz and its error, Å²/ps: 4951
/// D* from their Ref. [12], the per-atom tracer diffusivity times the atoms
/// of its cell (`validation/direct_md_dynamics.md`).
fn table_diffusivity(temperature: f64) -> Option<(f64, f64)> {
    // Å²/ps from Table 1 of Mwungeli, Liu, and Demkowicz, quoted there from Ref. [12].
    match temperature as i32 {
        300 => Some((0.143, 0.006)),
        400 => Some((0.327, 0.005)),
        500 => Some((0.526, 0.006)),
        600 => Some((0.728, 0.010)),
        700 => Some((0.910, 0.012)),
        _ => None,
    }
}

fn summarize(csv: &std::path::Path) -> String {
    let file = File::open(csv).expect("csv");
    let mut groups: BTreeMap<(u64, u64, String, u64), Group> = BTreeMap::new();
    let mut occupancy = false;
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        occupancy |= line.contains("ws_defects");
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 19 {
            continue;
        }
        let Ok(temperature) = fields[0].parse::<f64>() else {
            continue;
        };
        let Ok(shell) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(sites) = fields[9].parse::<f64>() else {
            continue;
        };
        let Ok(window) = fields[13].parse::<f64>() else {
            continue;
        };
        let Ok(captured) = fields[14].parse::<f64>() else {
            continue;
        };
        let key = (
            temperature.to_bits(),
            shell.to_bits(),
            fields[8].to_string(),
            window.to_bits(),
        );
        let group = groups.entry(key).or_insert_with(|| Group {
            temperature,
            window,
            shell,
            orientation: fields[8].to_string(),
            sites,
            captured: 0.0,
            trials: 0.0,
        });
        group.captured += captured;
        group.trials += 1.0;
    }
    let mut by_temp: BTreeMap<(u64, u64), Vec<&Group>> = BTreeMap::new();
    for group in groups.values() {
        by_temp
            .entry((group.temperature.to_bits(), group.window.to_bits()))
            .or_default()
            .push(group);
    }
    let mut text = String::new();
    text.push_str("# Direct-MD vacancy-SIA volume, Mishin Cu\n\n");
    let test = if occupancy {
        "Capture is Wigner-Seitz occupancy (`MD_CAPTURE=ws`): every atom is assigned to its nearest lattice site after the drift of the cell's centre of mass is removed, and the pair has recombined at the first sample, every 0.25 ps, where every site holds exactly one atom. The columns `initial_ws_defects` and `final_ws_defects` count sites that do not. The 12-fold coordination test of Mwungeli, Liu, and Demkowicz (Phys. Rev. Materials 9, 023601 (2025)) cannot be used at 700 K, where thermal vibration alone keeps 100 to 200 atoms of the 13³ cell off 12-fold coordination, so its count never reaches zero."
    } else {
        "Capture is the paper's 12-fold test: coordination cutoff 3.1 Å, sampled every 0.25 ps (Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025))."
    };
    text.push_str(&format!("Independent NVT trajectories of one vacancy and one <100> dumbbell on the Mishin EAM1 evaluated by `tests/mishin_cu.rs`. The driver is `examples/langevin_nvt.rs`. The cell is 13×13×13 conventional cells (8788 atoms) at the branch's NPT lattice constant. {test} The shell sum counts recombined sites, with each orientation owning one third of its shell. The survival-curve volume defined in section 2 of that paper is computed from the same trajectories.\n\n"));
    text.push_str("The clock starts at the ideal dumbbell (offset 1.025 Å) with Maxwellian velocities. There is no separate NPH equilibration. The thermostat is the Langevin velocity Verlet in `npt_lattice`, without the barostat. Friction and timestep are the columns `gamma_per_ps` and `dt_fs`.\n\n");
    if !occupancy {
        text.push_str("A 3 ps control at 500 K, 2 fs, and gamma = 1/ps, sampled every 0.25 ps, fixes what a zero in the 12-fold count means. The perfect 13³ crystal started at 0 atoms off 12-fold coordination and then fluctuated: minimum 0, mean 6.2, maximum 14, with 4 of 12 samples exactly 0. The nearest shell, a <100> dumbbell, started at 13 and fell into that same cloud (minimum 0, mean 5.7, maximum 16, 2 of 12 samples exactly 0). The outermost shell inside 10.8 Å started at 20 and stayed separated (minimum 22, mean 32.0, maximum 47, 0 of 12 samples exactly 0). Capture in the table is the first sample whose off-12 count is 0.\n\n");
    }
    text.push_str("Binomial standard error of a capture probability is sqrt(p (1-p) / N). The volume error adds the orientation contributions in quadrature. A shell whose trajectories all agree has a binomial error of zero.\n\n");
    if groups.is_empty() {
        text.push_str("No trajectory rows were found.\n");
        return text;
    }
    for ((_, _), members) in &by_temp {
        let temperature = members[0].temperature;
        let window = members[0].window;
        text.push_str(&format!("## {temperature:.0} K, window {window:.2} ps\n\n"));
        text.push_str("| shell (Å) | orientation | sites | captured | N | p | binomial σ |\n");
        text.push_str("|---:|:---:|---:|---:|---:|---:|---:|\n");
        let mut shells: BTreeMap<u64, Vec<&Group>> = BTreeMap::new();
        for group in members {
            shells.entry(group.shell.to_bits()).or_default().push(group);
        }
        let mut volume = 0.0;
        let mut variance = 0.0;
        let mut used = 0usize;
        let mut skipped = Vec::new();
        for group in members {
            let p = group.captured / group.trials;
            let sigma = (p * (1.0 - p) / group.trials).sqrt();
            text.push_str(&format!(
                "| {shell:.4} | {orient} | {sites:.0} | {cap:.0} | {n:.0} | {p:.4} | {sigma:.4} |\n",
                shell = group.shell,
                orient = group.orientation,
                sites = group.sites,
                cap = group.captured,
                n = group.trials,
                p = p,
                sigma = sigma,
            ));
        }
        text.push('\n');
        for orient_groups in shells.values() {
            let labels: BTreeMap<&str, &Group> = orient_groups
                .iter()
                .map(|group| (group.orientation.as_str(), *group))
                .collect();
            if !labels.contains_key("100")
                || !labels.contains_key("010")
                || !labels.contains_key("001")
            {
                skipped.push(orient_groups[0].shell);
                continue;
            }
            used += 1;
            for group in labels.values() {
                let p = group.captured / group.trials;
                let share = group.sites / 3.0;
                volume += share * p;
                variance += share * share * p * (1.0 - p) / group.trials;
            }
        }
        text.push_str(&format!(
            "Shell sum = {volume:.2} ± {sigma:.2} Ω over {used} shells with all three orientations. This counts sites that recombined inside the window. It is not the survival-curve volume.\n\n",
            sigma = variance.sqrt(),
        ));
        if !skipped.is_empty() {
            text.push_str("Shells left out because an orientation is missing:");
            for shell in skipped {
                text.push_str(&format!(" {shell:.4}"));
            }
            text.push_str(".\n\n");
        }
        if (temperature - 500.0).abs() < 1.0 {
            text.push_str("Commit 0a5b8661 counts eight translation channels of a <100> dumbbell and turns the axis on a hop. benchmarks/defect_pair_run12_volume.csv gives the 13³ first-passage volume as 202.4328 ± 5.1390 Ω at 300 K, 187.6200 ± 4.6448 Ω at 500 K, and 204.5416 ± 3.6445 Ω at 700 K. Table 1 of Mwungeli, Liu and Demkowicz, 8788 atoms at 500 K, is 185 ± 23 Ω (k-method) and 161 ± 15 Ω (mean-time method).\n\n");
        }
        if (temperature - 300.0).abs() < 1.0 {
            text.push_str("Table 1 at 300 K, 8788 atoms, is 254 ± 21 Ω (k-method) and 286 ± 28 Ω (mean-time method).\n\n");
        }
        if (temperature - 700.0).abs() < 1.0 {
            text.push_str("Table 1 at 700 K, 8788 atoms, is 203 ± 10 Ω (k-method) and 210 ± 21 Ω (mean-time method).\n\n");
        }
    }
    text.push_str(&first_method_report(csv));
    text.push_str("Raw per-trajectory outcomes are the CSV this note was written from.\n");
    text
}

/// The doubly occupied Wigner-Seitz site of an fcc cell holding one extra atom.
///
/// A site is an integer triple in units of a/2 with an even sum, wrapped
/// into the `repeats`³ cell. Each atom goes to its nearest site (round every
/// coordinate, then re-round the worst one if the sum is odd). `shift` is the
/// lattice translation to remove first. Returns `None` unless exactly one
/// site holds two atoms and none holds more.
fn nearest_site(atom: &[f64], half: f64, period: i64, shift: [f64; 3]) -> (usize, [i64; 3]) {
    let u = [
        (atom[0] - shift[0]) / half,
        (atom[1] - shift[1]) / half,
        (atom[2] - shift[2]) / half,
    ];
    let mut site = u.map(|value| value.round() as i64);
    if (site[0] + site[1] + site[2]).rem_euclid(2) != 0 {
        let mut worst = 0;
        for axis in 1..3 {
            if (u[axis] - site[axis] as f64).abs() > (u[worst] - site[worst] as f64).abs() {
                worst = axis;
            }
        }
        site[worst] += if u[worst] > site[worst] as f64 { 1 } else { -1 };
    }
    let wrapped = site.map(|value| value.rem_euclid(period));
    let index = ((wrapped[0] * period + wrapped[1]) * period + wrapped[2]) as usize;
    (index, wrapped)
}

fn interstitial_site(
    positions: &[f64],
    lattice: f64,
    repeats: usize,
    shift: [f64; 3],
) -> Option<[i64; 3]> {
    let period = 2 * repeats as i64;
    doubled_site(&atom_sites(positions, lattice, period, shift), period)
}

/// The wrapped nearest site of every atom, as [`nearest_site`] assigns it.
fn atom_sites(positions: &[f64], lattice: f64, period: i64, shift: [f64; 3]) -> Vec<[i64; 3]> {
    positions
        .as_chunks::<3>()
        .0
        .iter()
        .map(|atom| nearest_site(atom, 0.5 * lattice, period, shift).1)
        .collect()
}

/// The site that two entries of `sites` share, or `None` unless exactly one
/// site holds two and none holds more.
fn doubled_site(sites: &[[i64; 3]], period: i64) -> Option<[i64; 3]> {
    let mut count = vec![0u8; (period * period * period) as usize];
    let mut doubled = None;
    for site in sites {
        let index = ((site[0] * period + site[1]) * period + site[2]) as usize;
        count[index] = count[index].saturating_add(1);
        match count[index] {
            2 if doubled.is_none() => doubled = Some(*site),
            2.. => return None,
            _ => {}
        }
    }
    doubled
}

/// Lattice sites of the `repeats`³ fcc cell that do not hold exactly one atom.
///
/// The assignment is the one in [`interstitial_site`]. A vacancy with an
/// interstitial counts two and the recombined crystal zero, independent of
/// temperature while atoms stay inside their Wigner-Seitz cells.
pub fn occupancy_defects(
    positions: &[f64],
    lattice: f64,
    repeats: usize,
    shift: [f64; 3],
) -> usize {
    let period = 2 * repeats as i64;
    let sites = (period * period * period) as usize;
    let mut count = vec![0u8; sites];
    for atom in positions.as_chunks::<3>().0 {
        let (index, _) = nearest_site(atom, 0.5 * lattice, period, shift);
        count[index] = count[index].saturating_add(1);
    }
    let mut defects = 0;
    for i in 0..period {
        for j in 0..period {
            for k in 0..period {
                if (i + j + k) % 2 == 0 && count[((i * period + j) * period + k) as usize] != 1 {
                    defects += 1;
                }
            }
        }
    }
    defects
}

fn mean_position(positions: &[f64]) -> [f64; 3] {
    let n = (positions.len() / 3).max(1) as f64;
    let mut mean = [0.0; 3];
    for atom in positions.as_chunks::<3>().0 {
        for axis in 0..3 {
            mean[axis] += atom[axis] / n;
        }
    }
    mean
}

fn site_step(from: [i64; 3], to: [i64; 3], period: i64) -> [i64; 3] {
    let mut step = [0; 3];
    for axis in 0..3 {
        let mut delta = (to[axis] - from[axis]).rem_euclid(period);
        if delta > period / 2 {
            delta -= period;
        }
        step[axis] = delta;
    }
    step
}

/// Per-replica statistics of one tracked interstitial.
struct DiffusionReplica {
    samples: usize,
    rejected: usize,
    site_changes: usize,
    jumps: usize,
    jump_sq: f64,
    msd_end: f64,
    elapsed_ps: f64,
    d_msd: f64,
}

/// Samples a site must hold before a move to it counts as a jump.
const DWELL_SAMPLES: usize = 3;
const TRACK_SAMPLE_PS: f64 = 0.1;

/// Multi-origin MSD slope 6 D over lags from `lag_min` to `lag_max` samples.
fn msd_slope(track: &[[f64; 3]], lag_min: usize, lag_max: usize, sample_ps: f64) -> f64 {
    let mut lags = Vec::new();
    let mut values = Vec::new();
    for lag in lag_min..=lag_max.min(track.len().saturating_sub(1)) {
        let mut sum = 0.0;
        let origins = track.len() - lag;
        for origin in 0..origins {
            let a = track[origin];
            let b = track[origin + lag];
            sum += (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2);
        }
        lags.push(lag as f64 * sample_ps);
        values.push(sum / origins as f64);
    }
    let n = lags.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_t = lags.iter().sum::<f64>() / n;
    let mean_m = values.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (t, m) in lags.iter().zip(values.iter()) {
        cov += (t - mean_t) * (m - mean_m);
        var += (t - mean_t) * (t - mean_t);
    }
    cov / var / 6.0
}

fn track_interstitial(
    temperature: f64,
    seed: u64,
    window_ps: f64,
    spec: &RunSpec,
) -> DiffusionReplica {
    let lattice = measured_lattice(temperature);
    let (mut positions, cell) = isolated_dumbbell(spec.repeats, lattice);
    let mut potential = MishinCu::new();
    let production =
        Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, temperature, CU_MASS_AMU);
    let equil = Langevin::from_physical(spec.dt_fs, 1.0, temperature, CU_MASS_AMU);
    let mut rng = SplitRng::new(mix_seed(&[temperature.to_bits(), seed, 0xD1FF]));
    let sigma = (production.thermal / production.mass).sqrt();
    let mut velocities: Vec<f64> = (0..positions.len()).map(|_| rng.normal() * sigma).collect();
    zero_com(&mut velocities);
    let mut gradient = vec![0.0; positions.len()];
    potential.energy_gradient(&positions, cell, &mut gradient);
    let reference = mean_position(&positions);
    let period = 2 * spec.repeats as i64;
    let step_ps = spec.dt_fs * 1.0e-3;
    let equil_steps = (2.0 / step_ps).round() as usize;
    let sample_steps = ((TRACK_SAMPLE_PS / step_ps).round() as usize).max(1);
    let production_steps = (window_ps / step_ps).round() as usize;
    let mut site = None;
    let mut unwrapped = [0i64; 3];
    let mut committed = [0i64; 3];
    let mut candidate = [0i64; 3];
    let mut dwell = 0usize;
    let mut out = DiffusionReplica {
        samples: 0,
        rejected: 0,
        site_changes: 0,
        jumps: 0,
        jump_sq: 0.0,
        msd_end: 0.0,
        elapsed_ps: production_steps as f64 * step_ps,
        d_msd: 0.0,
    };
    let mut track = Vec::with_capacity(production_steps / sample_steps + 1);
    for step in 1..=equil_steps + production_steps {
        let phase = if step <= equil_steps {
            equil
        } else {
            production
        };
        phase.step(
            &mut potential,
            &mut positions,
            &mut velocities,
            &mut gradient,
            cell,
            &mut rng,
        );
        if step < equil_steps || !(step - equil_steps).is_multiple_of(sample_steps) {
            continue;
        }
        let mean = mean_position(&positions);
        let shift = [
            mean[0] - reference[0],
            mean[1] - reference[1],
            mean[2] - reference[2],
        ];
        let Some(here) = interstitial_site(&positions, lattice, spec.repeats, shift) else {
            out.rejected += 1;
            continue;
        };
        if let Some(previous) = site {
            let delta = site_step(previous, here, period);
            if delta != [0, 0, 0] {
                out.site_changes += 1;
                for axis in 0..3 {
                    unwrapped[axis] += delta[axis];
                }
            }
        }
        site = Some(here);
        if step == equil_steps {
            committed = unwrapped;
            candidate = unwrapped;
            out.site_changes = 0;
            out.rejected = 0;
        }
        if unwrapped == candidate {
            dwell += 1;
        } else {
            candidate = unwrapped;
            dwell = 1;
        }
        if dwell == DWELL_SAMPLES && candidate != committed {
            let d2: i64 = (0..3)
                .map(|axis| (candidate[axis] - committed[axis]).pow(2))
                .sum();
            out.jumps += 1;
            out.jump_sq += d2 as f64 * 0.25 * lattice * lattice;
            committed = candidate;
        }
        track.push(unwrapped.map(|value| value as f64 * 0.5 * lattice));
    }
    out.samples = track.len();
    if let (Some(first), Some(last)) = (track.first(), track.last()) {
        out.msd_end = (0..3).map(|axis| (last[axis] - first[axis]).powi(2)).sum();
    }
    let lag_max = ((track.len() / 8).min((20.0 / TRACK_SAMPLE_PS) as usize)).max(2);
    out.d_msd = msd_slope(&track, (lag_max / 4).max(1), lag_max, TRACK_SAMPLE_PS);
    out
}

const DIFFUSION_HEADER: &str = "temperature_K,lattice_A,seed,dt_fs,gamma_per_ps,window_ps,samples,rejected,site_changes,committed_jumps,jump_sq_A2,msd_end_A2,D_msd_A2_per_ps,D_end_A2_per_ps,D_jump_A2_per_ps";

/// Free-interstitial defect diffusivity from independent replicas: the MSD
/// of the doubly occupied site, which `MD_SIA_TRACER` sets against the
/// tracer sum N D*.
///
/// `MD_PS_BY_T=300:250,500:150` sets the production window per temperature;
/// otherwise every temperature uses `MD_PS`. `MD_SEED_START` offsets the
/// replica seeds so a later run adds replicas. Replicas run in parallel and
/// append to `MD_OUT` (default validation/direct_md_sia_d.csv).
fn diffusion_run(spec: &RunSpec) {
    let windows: BTreeMap<i64, f64> = env::var("MD_PS_BY_T")
        .unwrap_or_default()
        .split(',')
        .filter_map(|item| {
            let (t, ps) = item.split_once(':')?;
            Some((t.trim().parse().ok()?, ps.trim().parse().ok()?))
        })
        .collect();
    let path = env::var("MD_OUT").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_sia_d.csv")
    });
    if !path.exists() {
        let mut header = File::create(&path).expect("diffusivity csv");
        writeln!(header, "{DIFFUSION_HEADER}").unwrap();
    }
    let first_seed = env_u64("MD_SEED_START", 0);
    let mut jobs = Vec::new();
    for &temperature in &spec.temps {
        let window = windows
            .get(&(temperature as i64))
            .copied()
            .unwrap_or(spec.window_ps);
        for seed in first_seed..first_seed + spec.seeds {
            jobs.push((temperature, seed, window));
        }
    }
    let file = Mutex::new(OpenOptions::new().append(true).open(&path).expect("append"));
    let cursor = AtomicUsize::new(0);
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len().max(1));
    thread::scope(|scope| {
        for _ in 0..workers {
            let (jobs, cursor, file) = (&jobs, &cursor, &file);
            scope.spawn(move || {
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&(temperature, seed, window)) = jobs.get(index) else {
                        break;
                    };
                    let r = track_interstitial(temperature, seed, window, spec);
                    let d_end = r.msd_end / (6.0 * r.elapsed_ps);
                    let d_jump = r.jump_sq / (6.0 * r.elapsed_ps);
                    let line = format!(
                        "{temperature:.0},{:.6},{seed},{:.4},{:.4},{:.2},{},{},{},{},{:.6},{:.6},{:.8e},{:.8e},{:.8e}",
                        measured_lattice(temperature),
                        spec.dt_fs,
                        spec.gamma_per_ps,
                        r.elapsed_ps,
                        r.samples,
                        r.rejected,
                        r.site_changes,
                        r.jumps,
                        r.jump_sq,
                        r.msd_end,
                        r.d_msd,
                        d_end,
                        d_jump,
                    );
                    writeln!(file.lock().expect("csv lock"), "{line}").unwrap();
                    println!(
                        "msd T {temperature:.0} seed {seed} D {:.4} D_end {d_end:.4} D_jump {d_jump:.4} jumps {} changes {} rejected {}",
                        r.d_msd, r.jumps, r.site_changes, r.rejected
                    );
                }
            });
        }
    });
}

/// Production block of a tracer replica, ps: `MD_SIA_TAU`, or 40, 20 and
/// 10 ps up to 300 K, up to 500 K and above.
fn sia_block_ps(temperature: f64) -> f64 {
    if let Ok(text) = env::var("MD_SIA_TAU")
        && let Ok(value) = text.parse::<f64>()
    {
        return value;
    }
    match temperature as i32 {
        t if t <= 300 => 40.0,
        t if t <= 500 => 20.0,
        _ => 10.0,
    }
}

/// Snapshots of one free dumbbell at the start of production and at the
/// end of each block, on the replica of [`track_interstitial`] with the
/// same seed.
struct TracerRun {
    lattice: f64,
    tau_ps: f64,
    unresolved: usize,
    /// Resolved snapshots at which the sum of every atom's unwrapped site,
    /// less the dumbbell site, differs from its value at the start.
    sum_rule_breaks: usize,
    /// Unwrapped doubly occupied site, half-lattice units.
    ws: Vec<[i64; 3]>,
    /// Midpoint of the two atoms in that site, Å.
    centre: Vec<[f64; 3]>,
    /// The one of those two atoms farther from the site, Å.
    atom: Vec<[f64; 3]>,
    /// Σ |Δr|² of the site from sample to sample since production began, Å².
    jump_r2: Vec<f64>,
    /// Samples since production began at which the site moved.
    jumps: Vec<usize>,
    /// Unwrapped nearest site of every atom, half-lattice units.
    sites: Vec<Vec<[i64; 3]>>,
    /// Every atom less the centre-of-mass drift, Å.
    positions: Vec<Vec<f64>>,
}

/// Offsets from the doubly occupied `site` of the two atoms in it, Å.
fn pair_offsets(
    positions: &[f64],
    shift: [f64; 3],
    sites: &[[i64; 3]],
    site: [i64; 3],
    lattice: f64,
    period: i64,
) -> [[f64; 3]; 2] {
    let length = period as f64 * 0.5 * lattice;
    let mut offsets = [[0.0; 3]; 2];
    let members = sites.iter().enumerate().filter(|(_, here)| **here == site);
    for (slot, (atom, _)) in offsets.iter_mut().zip(members) {
        *slot = [0, 1, 2].map(|axis| {
            let delta =
                positions[3 * atom + axis] - shift[axis] - site[axis] as f64 * 0.5 * lattice;
            delta - length * (delta / length).round()
        });
    }
    offsets
}

fn sia_tracer_trajectory(
    temperature: f64,
    seed: u64,
    tau_ps: f64,
    blocks: usize,
    spec: &RunSpec,
) -> TracerRun {
    let lattice = measured_lattice(temperature);
    let (mut positions, cell) = isolated_dumbbell(spec.repeats, lattice);
    let mut potential = MishinCu::new();
    let production =
        Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, temperature, CU_MASS_AMU);
    let equil = Langevin::from_physical(spec.dt_fs, 1.0, temperature, CU_MASS_AMU);
    let mut rng = SplitRng::new(mix_seed(&[temperature.to_bits(), seed, 0xD1FF]));
    let sigma = (production.thermal / production.mass).sqrt();
    let mut velocities: Vec<f64> = (0..positions.len()).map(|_| rng.normal() * sigma).collect();
    zero_com(&mut velocities);
    let mut gradient = vec![0.0; positions.len()];
    potential.energy_gradient(&positions, cell, &mut gradient);
    let reference = mean_position(&positions);
    let period = 2 * spec.repeats as i64;
    let step_ps = spec.dt_fs * 1.0e-3;
    let equil_steps = (2.0 / step_ps).round() as usize;
    let sample_steps = ((TRACK_SAMPLE_PS / step_ps).round() as usize).max(1);
    let block_steps = ((tau_ps / step_ps).round() as usize).div_ceil(sample_steps) * sample_steps;
    let mut previous = atom_sites(&positions, lattice, period, [0.0; 3]);
    let mut site = doubled_site(&previous, period).expect("the starting dumbbell holds one site");
    let mut pair = pair_offsets(&positions, [0.0; 3], &previous, site, lattice, period);
    let mut sites = previous.clone();
    let mut unwrapped = [0i64; 3];
    let mut jump_r2 = 0.0;
    let mut jumps = 0usize;
    let mut unresolved = 0usize;
    let mut balance = None;
    let mut run = TracerRun {
        lattice,
        tau_ps: block_steps as f64 * step_ps,
        unresolved: 0,
        sum_rule_breaks: 0,
        ws: Vec::with_capacity(blocks + 1),
        centre: Vec::with_capacity(blocks + 1),
        atom: Vec::with_capacity(blocks + 1),
        jump_r2: Vec::with_capacity(blocks + 1),
        jumps: Vec::with_capacity(blocks + 1),
        sites: Vec::with_capacity(blocks + 1),
        positions: Vec::with_capacity(blocks + 1),
    };
    for step in 1..=equil_steps + blocks * block_steps {
        let phase = if step <= equil_steps {
            equil
        } else {
            production
        };
        phase.step(
            &mut potential,
            &mut positions,
            &mut velocities,
            &mut gradient,
            cell,
            &mut rng,
        );
        // Production samples fall where `track_interstitial` takes them.
        let sampled = if step < equil_steps {
            step.is_multiple_of(sample_steps)
        } else {
            (step - equil_steps).is_multiple_of(sample_steps)
        };
        if !sampled {
            continue;
        }
        let mean = mean_position(&positions);
        let shift = [0, 1, 2].map(|axis| mean[axis] - reference[axis]);
        let wrapped = atom_sites(&positions, lattice, period, shift);
        for ((now, before), total) in wrapped.iter().zip(&previous).zip(&mut sites) {
            let delta = site_step(*before, *now, period);
            for (value, delta) in total.iter_mut().zip(delta) {
                *value += delta;
            }
        }
        let resolved = doubled_site(&wrapped, period);
        match resolved {
            Some(here) => {
                let delta = site_step(site, here, period);
                let mut hop = 0.0;
                for (total, delta) in unwrapped.iter_mut().zip(delta) {
                    *total += delta;
                    hop += (delta as f64 * 0.5 * lattice).powi(2);
                }
                if step > equil_steps {
                    jumps += usize::from(here != site);
                    jump_r2 += hop;
                }
                site = here;
                pair = pair_offsets(&positions, shift, &wrapped, here, lattice, period);
            }
            None => unresolved += usize::from(step > equil_steps),
        }
        previous = wrapped;
        if step == equil_steps {
            jumps = 0;
            jump_r2 = 0.0;
            unresolved = 0;
        } else if step < equil_steps || !(step - equil_steps).is_multiple_of(block_steps) {
            continue;
        }
        if resolved.is_some() {
            let total = sites.iter().fold([0i64; 3], |sum, here| {
                [sum[0] + here[0], sum[1] + here[1], sum[2] + here[2]]
            });
            let gap = [0, 1, 2].map(|axis| total[axis] - unwrapped[axis]);
            match balance {
                None => balance = Some(gap),
                Some(start) => run.sum_rule_breaks += usize::from(gap != start),
            }
        }
        let origin = unwrapped.map(|value| value as f64 * 0.5 * lattice);
        let squared = |offset: [f64; 3]| offset.iter().map(|value| value * value).sum::<f64>();
        let farther = if squared(pair[0]) >= squared(pair[1]) {
            pair[0]
        } else {
            pair[1]
        };
        run.ws.push(unwrapped);
        run.centre
            .push([0, 1, 2].map(|axis| origin[axis] + 0.5 * (pair[0][axis] + pair[1][axis])));
        run.atom
            .push([0, 1, 2].map(|axis| origin[axis] + farther[axis]));
        run.jump_r2.push(jump_r2);
        run.jumps.push(jumps);
        run.sites.push(sites.clone());
        run.positions.push(
            positions
                .iter()
                .enumerate()
                .map(|(index, value)| value - shift[index % 3])
                .collect(),
        );
    }
    run.unresolved = unresolved;
    run
}

/// Squared displacements between two snapshots of a [`TracerRun`], Å².
#[derive(Clone, Copy, Default)]
struct TracerSums {
    ws: f64,
    centre: f64,
    atom: f64,
    /// Σ |Δr|² of the site over the samples in between.
    jump: f64,
    /// Samples in between at which the site moved.
    jumps: f64,
    /// Σ over atoms of the squared change of each atom's site.
    site: f64,
    /// Σ over atoms of the squared displacement along each axis.
    tracer: [f64; 3],
}

impl TracerSums {
    fn between(run: &TracerRun, from: usize, to: usize) -> Self {
        let half = 0.5 * run.lattice;
        let squared = |to: [f64; 3], from: [f64; 3]| {
            to.iter()
                .zip(from)
                .map(|(to, from)| (to - from).powi(2))
                .sum::<f64>()
        };
        let lattice_squared = |to: &[i64; 3], from: &[i64; 3]| {
            to.iter()
                .zip(from)
                .map(|(to, from)| (to - from).pow(2))
                .sum::<i64>()
        };
        let site = run.sites[to]
            .iter()
            .zip(&run.sites[from])
            .map(|(to, from)| lattice_squared(to, from))
            .sum::<i64>();
        let mut tracer = [0.0; 3];
        for (index, (to, from)) in run.positions[to]
            .iter()
            .zip(&run.positions[from])
            .enumerate()
        {
            tracer[index % 3] += (to - from).powi(2);
        }
        Self {
            ws: lattice_squared(&run.ws[to], &run.ws[from]) as f64 * half * half,
            centre: squared(run.centre[to], run.centre[from]),
            atom: squared(run.atom[to], run.atom[from]),
            jump: run.jump_r2[to] - run.jump_r2[from],
            jumps: (run.jumps[to] - run.jumps[from]) as f64,
            site: site as f64 * half * half,
            tracer,
        }
    }

    fn values(&self) -> [f64; 9] {
        [
            self.ws,
            self.centre,
            self.atom,
            self.jump,
            self.jumps,
            self.site,
            self.tracer[0],
            self.tracer[1],
            self.tracer[2],
        ]
    }

    fn from_values(values: [f64; 9]) -> Self {
        let [ws, centre, atom, jump, jumps, site, x, y, z] = values;
        Self {
            ws,
            centre,
            atom,
            jump,
            jumps,
            site,
            tracer: [x, y, z],
        }
    }

    fn tracer_total(&self) -> f64 {
        self.tracer.iter().sum()
    }
}

/// Every lag between snapshots of a run: the mean over all origins, and the
/// value from the first snapshot alone.
fn tracer_lags(run: &TracerRun) -> Vec<(usize, usize, TracerSums, TracerSums)> {
    let count = run.ws.len();
    (1..count)
        .map(|lag| {
            let origins = count - lag;
            let mut sum = [0.0; 9];
            for from in 0..origins {
                let values = TracerSums::between(run, from, from + lag).values();
                for (total, value) in sum.iter_mut().zip(values) {
                    *total += value;
                }
            }
            let mean = TracerSums::from_values(sum.map(|total| total / origins as f64));
            (lag, origins, mean, TracerSums::between(run, 0, lag))
        })
        .collect()
}

const SIA_TRACER_HEADER: &str = "temperature_K,lattice_A,repeats,n_atoms,dt_fs,gamma_per_ps,seed,tau_ps,lag,origins,ws_A2,centre_A2,atom_A2,jump_A2,jumps,site_tracer_A2,tracer_x_A2,tracer_y_A2,tracer_z_A2,first_ws_A2,first_centre_A2,first_atom_A2,first_jump_A2,first_jumps,first_site_tracer_A2,first_tracer_x_A2,first_tracer_y_A2,first_tracer_z_A2,unresolved_samples,sum_rule_breaks";

fn tracer_rows(spec: &RunSpec, temperature: f64, seed: u64, run: &TracerRun) -> Vec<String> {
    let n_atoms = run
        .positions
        .first()
        .map_or(0, |positions| positions.len() / 3);
    let join = |sums: &TracerSums| {
        sums.values()
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    tracer_lags(run)
        .iter()
        .map(|(lag, origins, mean, first)| {
            format!(
                "{temperature:.0},{:.6},{},{n_atoms},{:.4},{:.4},{seed},{:.4},{lag},{origins},{},{},{},{}",
                run.lattice,
                spec.repeats,
                spec.dt_fs,
                spec.gamma_per_ps,
                run.tau_ps,
                join(mean),
                join(first),
                run.unresolved,
                run.sum_rule_breaks,
            )
        })
        .collect()
}

/// One walk of a <100> dumbbell on an ideal fcc lattice by Johnson jumps
/// alone. A jump takes one atom of the pair to one of the four nearest
/// sites on its side of the axis; there it and that site's atom form a
/// dumbbell along the jump's perpendicular component, with the mover on the
/// side it came from. Returns Σ over atoms of |Δs|² and the dumbbell
/// displacement, in half-lattice units, and Σ over atoms of Δs.
fn johnson_walk(jumps: usize, rng: &mut SplitRng) -> (i64, [i64; 3], [i64; 3]) {
    let mut occupant: std::collections::HashMap<[i64; 3], usize> = Default::default();
    let mut homes = vec![[0i64; 3]; 2];
    let mut places = vec![[0i64; 3]; 2];
    let mut members = [0usize, 1];
    let mut site = [0i64; 3];
    let mut axis = 2usize;
    for _ in 0..jumps {
        let bits = rng.u64() >> 61;
        let side = (bits & 1) as usize;
        let across = [(axis + 1) % 3, (axis + 2) % 3][((bits >> 1) & 1) as usize];
        let sign = if bits >> 2 == 0 { -1 } else { 1 };
        let mut target = site;
        target[axis] += if side == 1 { 1 } else { -1 };
        target[across] += sign;
        let mover = members[side];
        let host = occupant.remove(&target).unwrap_or_else(|| {
            homes.push(target);
            places.push(target);
            homes.len() - 1
        });
        occupant.insert(site, members[1 - side]);
        places[mover] = target;
        members = if sign > 0 {
            [mover, host]
        } else {
            [host, mover]
        };
        site = target;
        axis = across;
    }
    let mut squared = 0;
    let mut net = [0i64; 3];
    for (home, place) in homes.iter().zip(&places) {
        for ((total, home), place) in net.iter_mut().zip(home).zip(place) {
            *total += place - home;
            squared += (place - home).pow(2);
        }
    }
    (squared, site, net)
}

/// Tracer correlation factor of [`johnson_walk`]: Σ|Δs|² over n λ² after
/// `jumps` jumps, mean and standard error over `walks` walks.
fn johnson_correlation(walks: usize, jumps: usize, seed: u64) -> (f64, f64) {
    let mut rng = SplitRng::new(seed);
    let values: Vec<f64> = (0..walks)
        .map(|_| johnson_walk(jumps, &mut rng).0 as f64 / (2.0 * jumps as f64))
        .collect();
    mean_and_error(&values)
}

/// Mean and standard error of the mean; the error is NaN for one value.
fn mean_and_error(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, f64::NAN);
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    (mean, (variance / n).sqrt())
}

/// Least-squares slope of `(x, y)` points with a free intercept.
fn fitted_slope(points: &[(f64, f64)]) -> f64 {
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let covariance = points
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let spread = points
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    covariance / spread
}

struct TracerRow {
    temperature: f64,
    lattice: f64,
    repeats: usize,
    atoms: usize,
    dt_fs: f64,
    gamma: f64,
    seed: u64,
    tau: f64,
    lag: usize,
    mean: TracerSums,
    first: TracerSums,
    unresolved: f64,
    breaks: f64,
    /// Label, equilibration ps, rescale interval ps, rescale window K and
    /// piston period ps of a campaign that did not run Langevin.
    dynamics: Option<(String, [f64; 4])>,
}

fn read_tracer_rows(csv: &std::path::Path) -> Vec<TracerRow> {
    let Ok(file) = File::open(csv) else {
        return Vec::new();
    };
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines().map_while(Result::ok) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 30 {
            continue;
        }
        let sums = |start: usize| -> Option<TracerSums> {
            let mut values = [0.0; 9];
            for (slot, text) in values.iter_mut().zip(&fields[start..start + 9]) {
                *slot = text.parse().ok()?;
            }
            Some(TracerSums::from_values(values))
        };
        let parsed: Option<TracerRow> = (|| {
            Some(TracerRow {
                temperature: fields[0].parse().ok()?,
                lattice: fields[1].parse().ok()?,
                repeats: fields[2].parse().ok()?,
                atoms: fields[3].parse().ok()?,
                dt_fs: fields[4].parse().ok()?,
                gamma: fields[5].parse().ok()?,
                seed: fields[6].parse().ok()?,
                tau: fields[7].parse().ok()?,
                lag: fields[8].parse().ok()?,
                mean: sums(10)?,
                first: sums(19)?,
                unresolved: fields[28].parse().ok()?,
                breaks: fields[29].parse().ok()?,
                dynamics: dynamics_columns(&fields, 30),
            })
        })();
        if let Some(row) = parsed {
            rows.push(row);
        }
    }
    rows
}

fn dynamics_columns(fields: &[&str], first: usize) -> Option<(String, [f64; 4])> {
    if fields.len() < first + 5 {
        return None;
    }
    let mut values = [0.0; 4];
    for (slot, text) in values.iter_mut().zip(&fields[first + 1..first + 5]) {
        *slot = text.parse().ok()?;
    }
    Some((fields[first].to_string(), values))
}

/// Diffusivities of one trajectory, Å²/ps. `block` values are the lag-one
/// mean over 6τ; `slope` values fit the multi-origin mean from lag L/8 to
/// L/2 of L. The Ref. [12] values fit 2 D t to the per-axis sum from the
/// first snapshot over every lag and average the three axes.
struct TracerSeed {
    lattice: f64,
    tau: f64,
    lags: usize,
    jumps_per_ps: f64,
    ws: f64,
    centre: f64,
    atom: f64,
    jump: f64,
    ws_slope: f64,
    site: f64,
    site_slope: f64,
    tracer_slope: f64,
    reference: f64,
    unresolved: f64,
    breaks: f64,
}

fn tracer_seeds(rows: &[TracerRow], temperature: f64) -> Vec<TracerSeed> {
    let mut by_seed: BTreeMap<u64, Vec<&TracerRow>> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.temperature == temperature) {
        by_seed.entry(row.seed).or_default().push(row);
    }
    let mut seeds = Vec::new();
    for group in by_seed.values_mut() {
        group.sort_by_key(|row| row.lag);
        let lags = group.len();
        let Some(one) = group.iter().find(|row| row.lag == 1) else {
            continue;
        };
        let tau = one.tau;
        let low = (lags / 8).max(1);
        let high = (lags / 2).max(low + 1);
        let window: Vec<&&TracerRow> = group
            .iter()
            .filter(|row| (low..=high).contains(&row.lag))
            .collect();
        let slope = |value: &dyn Fn(&TracerSums) -> f64, rows: &[&&TracerRow]| {
            let points: Vec<(f64, f64)> = rows
                .iter()
                .map(|row| (row.lag as f64 * tau, value(&row.mean)))
                .collect();
            fitted_slope(&points)
        };
        let first = |value: &dyn Fn(&TracerSums) -> f64| {
            let points: Vec<(f64, f64)> = group
                .iter()
                .map(|row| (row.lag as f64 * tau, value(&row.first)))
                .collect();
            fitted_slope(&points)
        };
        let axes = [0, 1, 2].map(|axis| first(&|sums: &TracerSums| sums.tracer[axis]) / 2.0);
        let block = 6.0 * tau;
        seeds.push(TracerSeed {
            lattice: one.lattice,
            tau,
            lags,
            jumps_per_ps: one.mean.jumps / tau,
            ws: one.mean.ws / block,
            centre: one.mean.centre / block,
            atom: one.mean.atom / block,
            jump: one.mean.jump / block,
            ws_slope: slope(&|sums: &TracerSums| sums.ws, &window) / 6.0,
            site: one.mean.site / block,
            site_slope: slope(&|sums: &TracerSums| sums.site, &window) / 6.0,
            tracer_slope: slope(&|sums: &TracerSums| sums.tracer_total(), &window) / 6.0,
            reference: axes.iter().sum::<f64>() / 3.0,
            unresolved: one.unresolved,
            breaks: one.breaks,
        });
    }
    seeds
}

/// Mean and standard error across trajectories of a [`TracerSeed`] value.
fn across_seeds(seeds: &[TracerSeed], value: impl Fn(&TracerSeed) -> f64) -> (f64, f64) {
    mean_and_error(&seeds.iter().map(value).collect::<Vec<_>>())
}

/// Tracer correlation factor: the mean `tracer` N D* over the mean
/// Wigner-Seitz block D, with both standard errors in quadrature. A ratio
/// per trajectory would put one trajectory's noisy slope in the denominator.
fn correlation_factor(seeds: &[TracerSeed], tracer: impl Fn(&TracerSeed) -> f64) -> (f64, f64) {
    let (ratio, error, _) = ratio_with_error(
        across_seeds(seeds, tracer),
        across_seeds(seeds, |seed| seed.ws),
    );
    (ratio, error)
}

/// `top / bottom` with both standard errors in quadrature, and the
/// difference over those errors.
fn ratio_with_error(top: (f64, f64), bottom: (f64, f64)) -> (f64, f64, f64) {
    let ratio = top.0 / bottom.0;
    let error = ratio.abs() * ((top.1 / top.0).powi(2) + (bottom.1 / bottom.0).powi(2)).sqrt();
    let z = (top.0 - bottom.0) / (top.1.powi(2) + bottom.1.powi(2)).sqrt();
    (ratio, error, z)
}

fn sia_tracer_note(csv: &std::path::Path) -> String {
    let rows = read_tracer_rows(csv);
    let mut text = String::new();
    let Some(head) = rows.first() else {
        text.push_str("# Free dumbbell tracer diffusivity, Mishin Cu\n\nNo trajectories.\n");
        return text;
    };
    let dynamics = match &head.dynamics {
        Some((label, [equil_ps, rescale_ps, _, piston_ps])) if label == "nph_nve" => format!(
            "{equil_ps:.0} ps of isotropic zero-pressure NPH (Andersen piston, period {piston_ps:.1} ps), then NVE in the mean box of the second half of that stage, both rescaled to the target temperature every {rescale_ps:.2} ps"
        ),
        _ => format!(
            "2 ps of Langevin at γ = 1/ps, then Langevin at γ = {}/ps",
            head.gamma
        ),
    };
    text.push_str("# Free dumbbell tracer diffusivity, Mishin Cu\n\n");
    text.push_str(&format!("One <100> dumbbell in the perfect {0}×{0}×{0} cell ({1} atoms) on the Mishin EAM1 of `tests/mishin_cu.rs`: Maxwellian start, {dynamics}, timestep {2} fs. The a column is the production lattice constant. Every 0.1 ps the centre-of-mass drift is removed and every atom is assigned its nearest ideal-lattice site. The defect is located three ways: the one doubly occupied Wigner-Seitz cell (as in `direct_md_sia_d.md`), the midpoint of the two atoms in it, and the one of those two farther from the site. Each atom's site is unwrapped from sample to sample, and every atom's position less the drift is stored at the start of production and at the end of each block of τ.\n\n", head.repeats, head.atoms, head.dt_fs));
    text.push_str("Two diffusivities are measured from the same trajectories. The defect diffusivity is the mean-square displacement of the defect over 6t. The tracer quantity is Σ_i |Δr_i|² over 6t, summed over every atom i; this is N D*, with D* the per-atom self-diffusivity that Eq. (5) of Ref. [12] (B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022)) computes, G_MSD = (1/N) Σ_i Δx_i Δx_j with 2Dt = G_MSD at long times. Its erratum (Phys. Rev. Materials 9, 069901(E) (2025)) multiplies that D* by N = 4951 to report an SIA diffusivity, and Table 1 of Mwungeli, Liu, and Demkowicz lists those values. N D* is measured from atom sites (one atom changes site by λ = a/√2 per dumbbell jump, so the vibrational term is absent) and from the drift-removed positions. Every displacement is in the frame of the centre of mass.\n\n");
    text.push_str("Block values are the lag-τ mean over every block. Slope values fit the mean over all origins against lag, with an intercept, from lag L/8 to L/2 of the L blocks. The Ref. [12] column follows that paper: one origin, the start of production; each axis fitted as 2Dt with an intercept over every lag; the three axes averaged. f is N D* over the block D of the Wigner-Seitz site, each the mean over trajectories, with both standard errors in quadrature: f takes the sites slope and f (Ref. [12]) the Ref. [12] fit. The block D of the defect equals its slope when successive defect jumps are uncorrelated beyond τ, which the slope column tests; the block N D* lies above the slope because a block of n jumps has the lattice-walk f(n) of `direct_md_dynamics.md`, about 0.8/n above its limit. The jump column sums |Δr|² of the Wigner-Seitz site from sample to sample over 6t: the jump rate times λ²/6 when every move is one nearest-neighbour jump. Every value is the mean over trajectories ± the standard error across them.\n\n");
    let mut temperatures: Vec<f64> = rows.iter().map(|row| row.temperature).collect();
    temperatures.sort_by(f64::total_cmp);
    temperatures.dedup();
    let cell = |value: (f64, f64)| format!("{:.4} ± {:.4}", value.0, value.1);
    text.push_str("## Defect position\n\n| T (K) | a (Å) | seeds | blocks | τ (ps) | site changes/ps | D, Wigner-Seitz (Å²/ps) | D, pair midpoint | D, farther atom | D, jump sum | D, Wigner-Seitz slope |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for temperature in &temperatures {
        let seeds = tracer_seeds(&rows, *temperature);
        if seeds.is_empty() {
            continue;
        }
        text.push_str(&format!(
            "| {temperature:.0} | {:.6} | {} | {} | {:.1} | {:.3} | {} | {} | {} | {} | {} |\n",
            seeds[0].lattice,
            seeds.len(),
            seeds[0].lags,
            seeds[0].tau,
            across_seeds(&seeds, |seed| seed.jumps_per_ps).0,
            cell(across_seeds(&seeds, |seed| seed.ws)),
            cell(across_seeds(&seeds, |seed| seed.centre)),
            cell(across_seeds(&seeds, |seed| seed.atom)),
            cell(across_seeds(&seeds, |seed| seed.jump)),
            cell(across_seeds(&seeds, |seed| seed.ws_slope)),
        ));
    }
    text.push_str("\n## Tracer\n\n| T (K) | N D*, sites, block (Å²/ps) | N D*, sites, slope | N D*, positions, slope | N D*, Ref. [12] fit | Table 1 D_i [12] | Ref. [12] fit / Table 1 | z | f | f (Ref. [12]) | sum-rule breaks | unresolved samples |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for temperature in &temperatures {
        let seeds = tracer_seeds(&rows, *temperature);
        if seeds.is_empty() {
            continue;
        }
        let reference = across_seeds(&seeds, |seed| seed.reference);
        let table = table_diffusivity(*temperature);
        let versus = table.map_or("- | - | -".to_string(), |paper| {
            let (ratio, error, z) = ratio_with_error(reference, paper);
            format!(
                "{:.3} ± {:.3} | {ratio:.3} ± {error:.3} | {z:.2}",
                paper.0, paper.1
            )
        });
        text.push_str(&format!(
            "| {temperature:.0} | {} | {} | {} | {} | {versus} | {} | {} | {:.0} | {:.0} |\n",
            cell(across_seeds(&seeds, |seed| seed.site)),
            cell(across_seeds(&seeds, |seed| seed.site_slope)),
            cell(across_seeds(&seeds, |seed| seed.tracer_slope)),
            cell(reference),
            cell(correlation_factor(&seeds, |seed| seed.site_slope)),
            cell(correlation_factor(&seeds, |seed| seed.reference)),
            seeds.iter().map(|seed| seed.breaks).sum::<f64>(),
            seeds.iter().map(|seed| seed.unresolved).sum::<f64>(),
        ));
    }
    text.push_str("\n## Lag dependence\n\nMeans over trajectories of the multi-origin mean at each lag t, over 6t.\n\n| T (K) | t (ps) | origins | site changes | D, Wigner-Seitz (Å²/ps) | N D*, sites | ratio |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    for temperature in &temperatures {
        let mut by_lag: BTreeMap<usize, Vec<&TracerRow>> = BTreeMap::new();
        for row in rows.iter().filter(|row| row.temperature == *temperature) {
            by_lag.entry(row.lag).or_default().push(row);
        }
        let count = by_lag.len();
        for (lag, group) in &by_lag {
            if !(lag.is_power_of_two() || *lag == count / 2 || *lag == count) {
                continue;
            }
            let time = *lag as f64 * group[0].tau;
            let mean = |value: &dyn Fn(&TracerSums) -> f64| {
                group.iter().map(|row| value(&row.mean)).sum::<f64>() / group.len() as f64
            };
            let ws = mean(&|sums: &TracerSums| sums.ws) / (6.0 * time);
            let site = mean(&|sums: &TracerSums| sums.site) / (6.0 * time);
            text.push_str(&format!(
                "| {temperature:.0} | {time:.0} | {} | {:.1} | {ws:.4} | {site:.4} | {:.3} |\n",
                count + 1 - lag,
                mean(&|sums: &TracerSums| sums.jumps),
                site / ws,
            ));
        }
    }
    text.push_str("\nRaw lag statistics per trajectory are the CSV this note was written from.\n");
    text
}

fn validation_file(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation")
        .join(name)
}

/// A tracer CSV in `validation/` and its dynamics.
struct TracerCampaign {
    file: &'static str,
    label: &'static str,
}

/// The tracer campaigns of `direct_md_dynamics.md`, at zero total pressure;
/// the last repeats the first at a 2 fs step.
const SIA_TRACER_CAMPAIGNS: [TracerCampaign; 3] = [
    TracerCampaign {
        file: "direct_md_sia_tracer_p0_gamma0.1_dt10fs.csv",
        label: "Langevin, γ = 0.1/ps, zero total pressure",
    },
    TracerCampaign {
        file: "direct_md_sia_tracer_nph_nve_dt10fs.csv",
        label: "NPH then NVE, rescaled",
    },
    TracerCampaign {
        file: "direct_md_sia_tracer_p0_gamma0.1_dt2fs.csv",
        label: "Langevin, γ = 0.1/ps, zero total pressure, 2 fs",
    },
];

/// Defect and tracer diffusivities of one temperature, mean ± standard error
/// across trajectories, Å²/ps.
struct TracerEstimate {
    dt_fs: f64,
    seeds: usize,
    /// Wigner-Seitz site changes per ps.
    jumps_per_ps: (f64, f64),
    /// Wigner-Seitz site, summed from sample to sample.
    jump: (f64, f64),
    /// Wigner-Seitz site, block.
    defect: (f64, f64),
    /// N D*, the Ref. [12] fit.
    tracer: (f64, f64),
    /// `tracer` over `defect`.
    correlation: (f64, f64),
}

fn tracer_estimate(rows: &[TracerRow], temperature: f64) -> Option<TracerEstimate> {
    let seeds = tracer_seeds(rows, temperature);
    let head = rows.iter().find(|row| row.temperature == temperature)?;
    (!seeds.is_empty()).then(|| TracerEstimate {
        dt_fs: head.dt_fs,
        seeds: seeds.len(),
        jumps_per_ps: across_seeds(&seeds, |seed| seed.jumps_per_ps),
        jump: across_seeds(&seeds, |seed| seed.jump),
        defect: across_seeds(&seeds, |seed| seed.ws),
        tracer: across_seeds(&seeds, |seed| seed.reference),
        correlation: correlation_factor(&seeds, |seed| seed.reference),
    })
}

/// Every tracer note, then `direct_md_dynamics.md`.
fn write_tracer_notes() {
    for campaign in &SIA_TRACER_CAMPAIGNS {
        let path = validation_file(campaign.file);
        if path.exists() {
            fs::write(note_path(&path), sia_tracer_note(&path)).expect("note");
        }
    }
    fs::write(validation_file("direct_md_dynamics.md"), dynamics_note()).expect("note");
}

/// Jump counts and walk counts of the lattice correlation factor in the notes.
const JOHNSON_WALKS: [(usize, usize); 5] = [
    (10, 40_000),
    (100, 10_000),
    (1_000, 2_000),
    (10_000, 400),
    (100_000, 100),
];

/// What the D_i of Table 1 is, from the papers and from the tracer campaigns.
fn table_d_section(temperatures: &[f64]) -> String {
    let mut text = String::new();
    text.push_str("## What the Table 1 D_i is\n\n");
    text.push_str("Page numbers are those of the accepted manuscripts on OSTI: Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), OSTI 2513909; B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022), their Ref. [12], OSTI 1870695.\n\n");
    text.push_str("- Section 1 (p. 2): \"The recombination rate coefficient K_iv may be written as K_iv = V_iv D_i/(αa²), where V_iv is the vacancy-SIA recombination volume, D_i is SIA diffusivity, a is the cubic lattice parameter, and α is a numerical factor that depends on the SIA migration mechanism (α = 2/3 for [100] split dumbbell SIAs in Cu [10]). The quotient D_i/(αa²) is just the SIA migration frequency, ν.\"\n");
    text.push_str("- Section 2 (p. 5): \"we calculate the corresponding recombination volume as V_iv = K_iv αa²/D_i. The values of SIA diffusivity D_i for the temperatures of interest are taken from Ref. [12], which obtained D_i directly from mean squared displacements of SIAs in MD simulations\". With Ω = a³/4 this is V_iv/Ω = 4αK_iv/(aD_i); α multiplies K_iv.\n");
    text.push_str("- Section 5 (p. 10), after Eq. (4), K_iv = (1/3)Ω Σ_j ν_j over the Johnson jump paths that lead to recombination: \"If we further assume that all the migration rates ν_j equal ν—the rate of SIA migration in the interior of a perfect crystal, far from any other defects—then, from the relation K_iv = V_iv D_i/(αa²) = V_iv ν (see section 1), we get V_iv = (1/3)Ωs\". ν is the rate of one jump path of the defect, and D_i = (2/3)a²ν is the diffusivity of the defect for eight uncorrelated jumps of length a/√2; section 6 (p. 15) writes the same D_i = (2/3)a²ν₀^i e^(−ΔE_i/k_BT) for TST.\n");
    text.push_str("- Ref. [12], Eq. (5) (p. 4): the \"generalized mean square displacement (MSD) per atom matrix\" G_MSD = (1/N) Σ_i [Δx_i², Δx_iΔy_i, …], with \"displacements of individual atoms from their initial positions\", \"2Dt = G_MSD in the limit of a long time\", and \"linear least-squares-fits to the corresponding elements of G_MSD over time\" after the first 1 ps. The cell has 4951 atoms with the SIA (p. 3); the volume follows thermal expansion and the runs are 2 ns of NVE (p. 4).\n");
    text.push_str("- Erratum, Phys. Rev. Materials 9, 069901(E) (2025), doi:10.1103/8hb2-86lm, replaces the p. 4 sentence that this D \"is the SIA diffusivity\" with \"The diffusivity thereby obtained is the self-diffusivity of Cu, not the SIA diffusivity, because G_MSD is computed on a per-atom basis\", and states that the diffusivities of Figs. 4 and 5(b) \"may be converted to SIA diffusivities by multiplying them by the number of atoms in the model, i.e., 4951.\" Fig. 4 of Ref. [12] (p. 9) plots those diffusivities at 700 K in µm²/s; at zero strain the components read 1.84 µm²/s = 1.84×10⁻⁴ Å²/ps, and 4951 times that is 0.911 Å²/ps, the 700 K D_i of Table 1 (0.910 ± 0.012).\n");
    text.push_str("- Potential: section 3 (p. 6), \"All atomistic simulations are carried out using the embedded atom method (EAM) potential for Cu by Mishin et al. [32]\"; Ref. [12] (p. 3), \"an embedded atom method (EAM) [24] potential for Cu developed by Mishin et al. [25]\". Both references are Y. Mishin, M. J. Mehl, D. A. Papaconstantopoulos, A. F. Voter, and J. D. Kress, Phys. Rev. B 63, 224106 (2001), the EAM1 of `tests/mishin_cu.rs` that every campaign here uses.\n\n");
    text.push_str("Table 1 D_i is therefore 4951 D*, with D* the per-atom diffusivity of Eq. (5): N D* = Σ_i ⟨Δx_i²⟩/2t, a sum of tracer displacements. One Johnson jump moves one atom by the defect's own jump vector, so N D* would equal the defect diffusivity only if successive jumps moved uncorrelated atoms. They do not: the atom that has just moved is one of the two atoms of the new dumbbell, so it moves again at the next jump with probability 1/2, and that move points back along its last one on average. N D* = f D_defect defines the tracer correlation factor f of the mechanism, while the ν of section 1 is the defect's own jump rate, with (2/3)a²ν = D_defect when successive defect jumps are uncorrelated. Both are measured below on the same free-dumbbell trajectories (`direct_md_sia_tracer_*.md`). N D* is the Ref. [12] fit of drift-removed positions; defect D is the Wigner-Seitz block value of the same trajectories; f is N D* over defect D, the ratio of the two means. Ratios carry both errors in quadrature, and z is the difference over the two errors in quadrature.\n\n");
    text.push_str("| T (K) | dynamics | CSV | seeds | N D*, Ref. [12] fit (Å²/ps) | Table 1 D_i [12] | N D* / Table 1 | z | defect D (Å²/ps) | defect D / Table 1 | f |\n|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut tracer_ratios = Vec::new();
    let mut defect_ratios = Vec::new();
    let mut factors = Vec::new();
    for temperature in temperatures {
        let Some(table) = table_diffusivity(*temperature) else {
            continue;
        };
        for campaign in SIA_TRACER_CAMPAIGNS.iter().take(2) {
            let rows = read_tracer_rows(&validation_file(campaign.file));
            let Some(estimate) = tracer_estimate(&rows, *temperature) else {
                continue;
            };
            let (tracer_ratio, tracer_error, z) = ratio_with_error(estimate.tracer, table);
            let (defect_ratio, defect_error, _) = ratio_with_error(estimate.defect, table);
            text.push_str(&format!(
                "| {temperature:.0} | {} | `{}` | {} | {:.4} ± {:.4} | {:.3} ± {:.3} | {tracer_ratio:.3} ± {tracer_error:.3} | {z:.2} | {:.4} ± {:.4} | {defect_ratio:.3} ± {defect_error:.3} | {:.4} ± {:.4} |\n",
                campaign.label,
                campaign.file,
                estimate.seeds,
                estimate.tracer.0,
                estimate.tracer.1,
                table.0,
                table.1,
                estimate.defect.0,
                estimate.defect.1,
                estimate.correlation.0,
                estimate.correlation.1,
            ));
            tracer_ratios.push(tracer_ratio);
            defect_ratios.push(defect_ratio);
            factors.push(estimate.correlation.0);
        }
    }
    let span = |values: &[f64]| {
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let high = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        format!("{low:.3} to {high:.3}")
    };
    if !tracer_ratios.is_empty() {
        text.push_str(&format!(
            "\nAcross these rows N D* / Table 1 runs {}, defect D / Table 1 runs {}, and f runs {}.\n",
            span(&tracer_ratios),
            span(&defect_ratios),
            span(&factors)
        ));
    }
    text.push_str("\nThe same correlation on an ideal lattice: a <100> dumbbell that moves only by Johnson jumps, the moving atom ending on the near side of the new dumbbell (`johnson_walk`). f(n) is Σ_i |Δs_i|² over n λ² after n jumps, mean ± standard error over the walks, seed 0x5EED. The lattice value is not an input to any number above.\n\n| jumps n | walks | f(n) |\n|---:|---:|---:|\n");
    for (jumps, walks) in JOHNSON_WALKS {
        let (factor, error) = johnson_correlation(walks, jumps, 0x5EED);
        text.push_str(&format!(
            "| {jumps} | {walks} | {factor:.4} ± {error:.4} |\n"
        ));
    }
    text.push('\n');
    text
}

/// The first tracer campaign against its repeat at a shorter timestep, or
/// `None` before the repeat has a temperature in common with it.
fn timestep_section(temperatures: &[f64]) -> Option<String> {
    let coarse = read_tracer_rows(&validation_file(SIA_TRACER_CAMPAIGNS[0].file));
    let fine = read_tracer_rows(&validation_file(SIA_TRACER_CAMPAIGNS[2].file));
    let mut steps = String::new();
    let mut remarks = Vec::new();
    for temperature in temperatures {
        let (Some(wide), Some(narrow)) = (
            tracer_estimate(&coarse, *temperature),
            tracer_estimate(&fine, *temperature),
        ) else {
            continue;
        };
        for (label, top, bottom) in [
            ("site changes/ps", narrow.jumps_per_ps, wide.jumps_per_ps),
            ("D, jump sum", narrow.jump, wide.jump),
            ("defect D", narrow.defect, wide.defect),
            ("N D*, Ref. [12] fit", narrow.tracer, wide.tracer),
            ("f", narrow.correlation, wide.correlation),
        ] {
            let (ratio, error, z) = ratio_with_error(top, bottom);
            steps.push_str(&format!(
                "| {temperature:.0} | {label} | {:.4} ± {:.4} ({} seeds) | {:.4} ± {:.4} ({} seeds) | {ratio:.3} ± {error:.3} | {z:.2} |\n",
                top.0, top.1, narrow.seeds, bottom.0, bottom.1, wide.seeds
            ));
        }
        let ((fine_walk, fine_walk_error, _), (coarse_walk, coarse_walk_error, _)) = (
            ratio_with_error(narrow.defect, narrow.jump),
            ratio_with_error(wide.defect, wide.jump),
        );
        remarks.push(format!(
            "At {temperature:.0} K, defect D over the jump-sum D, which is 1 when successive defect jumps are uncorrelated, is {fine_walk:.3} ± {fine_walk_error:.3} at {:.0} fs and {coarse_walk:.3} ± {coarse_walk_error:.3} at {:.0} fs.",
            narrow.dt_fs, wide.dt_fs
        ));
        if let Some(table) = table_diffusivity(*temperature) {
            let ((fine_ratio, fine_error, fine_z), (coarse_ratio, coarse_error, coarse_z)) = (
                ratio_with_error(narrow.tracer, table),
                ratio_with_error(wide.tracer, table),
            );
            remarks.push(format!(
                "N D* over the Table 1 D_i is {fine_ratio:.3} ± {fine_error:.3} (z = {fine_z:.2}) at {:.0} fs and {coarse_ratio:.3} ± {coarse_error:.3} (z = {coarse_z:.2}) at {:.0} fs.",
                narrow.dt_fs, wide.dt_fs
            ));
        }
    }
    if steps.is_empty() {
        return None;
    }
    let mut text = format!(
        "## Timestep\n\nEvery other campaign in this note integrates at {:.0} fs. `{}` repeats `{}` at {:.0} fs with the same seeds and τ and half the blocks. Neither Mwungeli, Liu, and Demkowicz nor Ref. [12] states an MD timestep.\n\n| T (K) | quantity | {:.0} fs | {:.0} fs | ratio | z |\n|---:|:---|---:|---:|---:|---:|\n{steps}",
        coarse[0].dt_fs,
        SIA_TRACER_CAMPAIGNS[2].file,
        SIA_TRACER_CAMPAIGNS[0].file,
        fine[0].dt_fs,
        fine[0].dt_fs,
        coarse[0].dt_fs
    );
    text.push_str(&format!("\n{}\n\n", remarks.join(" ")));
    Some(text)
}

fn dynamics_note() -> String {
    let temperatures = [300.0_f64, 500.0, 700.0];
    let mut text = String::new();
    let timestep = timestep_section(&temperatures);
    text.push_str("# Table 1 D_i and the free-dumbbell tracer, Mishin Cu\n\n");
    text.push_str("Our free-dumbbell defect D (`direct_md_sia_d.md`) is about twice the Table 1 D_i of Mwungeli, Liu, and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), which they quote from Ref. [12], B. Zhang et al., Phys. Rev. Materials 6, 053605 (2022). The first section shows from both papers and from tracer trajectories that the two are different quantities");
    text.push_str(if timestep.is_some() {
        "; the second repeats one campaign at a shorter timestep"
    } else {
        ""
    });
    text.push_str(". The tracer campaigns ran at zero total pressure, at the lattice constants of their notes rather than those of `measured_lattice`, under Langevin with γ = 0.1/ps and under the paper's protocol (NPH for 30 ps, then NVE with velocity rescaling). Every campaign uses the same potential, 13³ cell, timestep, seeds, and blocks");
    text.push_str(if timestep.is_some() {
        ", except the shorter-timestep repeat of the Timestep section, which has half the blocks"
    } else {
        ""
    });
    text.push_str(". `MD_SIA_TRACER=1` in `examples/langevin_nvt.rs` takes the same measurements on the `MD_MSD` replicas, and `MD_SIA_TRACER_NOTE=1` rewrites this note and the tracer notes from their CSVs. Each number below is read from the CSV named in its row.\n\n");
    text.push_str(&table_d_section(&temperatures));
    if let Some(section) = &timestep {
        text.push_str(section);
    }
    let end = text.trim_end().len();
    text.truncate(end);
    text.push('\n');
    text
}

/// Output of a tracer campaign: `MD_SIA_OUT`, or one CSV per friction and
/// timestep at the lattice constants of [`measured_lattice`].
fn sia_tracer_path(spec: &RunSpec) -> PathBuf {
    if let Ok(over) = env::var("MD_SIA_OUT") {
        return PathBuf::from(over);
    }
    validation_file(&format!(
        "direct_md_sia_tracer_gamma{}_dt{}fs.csv",
        spec.gamma_per_ps, spec.dt_fs
    ))
}

/// Runs [`sia_tracer_trajectory`] for every temperature and seed of the
/// spec that the output does not hold yet, longest blocks first, one per
/// core, and appends the lag rows of each trajectory together.
fn run_sia_tracer() {
    let spec = run_spec();
    let blocks = env_u64("MD_SIA_BLOCKS", 40) as usize;
    let path = sia_tracer_path(&spec);
    let done: std::collections::BTreeSet<(i64, u64)> = read_tracer_rows(&path)
        .iter()
        .map(|row| (row.temperature as i64, row.seed))
        .collect();
    let mut jobs: Vec<(f64, u64, f64)> = Vec::new();
    for &temperature in &spec.temps {
        for seed in 0..spec.seeds {
            if !done.contains(&(temperature as i64, seed)) {
                jobs.push((temperature, seed, sia_block_ps(temperature)));
            }
        }
    }
    jobs.sort_by(|left, right| right.2.total_cmp(&left.2));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("validation dir");
    }
    if !path.exists() {
        let mut header = File::create(&path).expect("tracer csv");
        writeln!(header, "{SIA_TRACER_HEADER}").unwrap();
    }
    println!(
        "tracer jobs {} temps {:?} seeds {} blocks {blocks} dt_fs {} gamma {}",
        jobs.len(),
        spec.temps,
        spec.seeds,
        spec.dt_fs,
        spec.gamma_per_ps
    );
    let file = Mutex::new(OpenOptions::new().append(true).open(&path).expect("append"));
    let cursor = AtomicUsize::new(0);
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len().max(1));
    thread::scope(|scope| {
        for _ in 0..workers {
            let (jobs, cursor, file, spec) = (&jobs, &cursor, &file, &spec);
            scope.spawn(move || {
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&(temperature, seed, tau)) = jobs.get(index) else {
                        break;
                    };
                    let run = sia_tracer_trajectory(temperature, seed, tau, blocks, spec);
                    let rows = tracer_rows(spec, temperature, seed, &run);
                    {
                        let mut handle = file.lock().expect("csv lock");
                        for row in &rows {
                            writeln!(handle, "{row}").unwrap();
                        }
                    }
                    let first = TracerSums::between(&run, 0, 1);
                    println!(
                        "tracer T {temperature:.0} seed {seed} D_ws {:.4} N D*_site {:.4} breaks {} unresolved {}",
                        first.ws / (6.0 * run.tau_ps),
                        first.site / (6.0 * run.tau_ps),
                        run.sum_rule_breaks,
                        run.unresolved
                    );
                }
            });
        }
    });
    fs::write(note_path(&path), sia_tracer_note(&path)).expect("note");
}

fn isolated_dumbbell(repeats: usize, a: f64) -> (Vec<f64>, [[f64; 3]; 3]) {
    let (flat, cell) = fcc(repeats, a);
    let n = flat.len() / 3;
    let mut points = Vec::with_capacity(n);
    for atom in 0..n {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
    let host = nearest(&points, center, None);
    let host_pos = points[host];
    let mut values = Vec::with_capacity((n + 1) * 3);
    for (index, point) in points.iter().enumerate() {
        if index != host {
            values.extend(point);
        }
    }
    values.extend([host_pos[0] - DUMBBELL_OFFSET_A, host_pos[1], host_pos[2]]);
    values.extend([host_pos[0] + DUMBBELL_OFFSET_A, host_pos[1], host_pos[2]]);
    (values, cell)
}

fn main() {
    let (perfect, cell) = fcc(1, 3.615);
    let mut gradient = vec![0.0; perfect.len()];
    let energy = analytic_gradient(&perfect, cell, &mut gradient);
    let per_atom = energy / 4.0;
    let max_force = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    println!("cohesive_eV {per_atom:.6} max_force {max_force:.3e}");
    if (per_atom + 3.54).abs() > 5.0e-4 || max_force > 1.0e-6 {
        eprintln!("Mishin crystal check failed");
        std::process::exit(1);
    }
    if env::var("MD_STABILITY").ok().as_deref() == Some("1") {
        let spec = run_spec();
        let a = measured_lattice(spec.temps[0]);
        let thermostat =
            Langevin::from_physical(spec.dt_fs, spec.gamma_per_ps, spec.temps[0], CU_MASS_AMU);
        let mut pot = MishinCu::new();
        let shells = shell_table(a, 10.8);
        let cases = [
            ("perfect", None),
            ("near", Some(shells[0].1)),
            ("far", Some(shells[shells.len() - 1].1)),
        ];
        for (label, shell) in cases {
            let (mut positions, cell) = if let Some(shell) = shell {
                place_vacancy_dumbbell(spec.repeats, a, shell, [1.0, 0.0, 0.0])
            } else {
                fcc(spec.repeats, a)
            };
            let mut velocities = vec![0.0; positions.len()];
            let mut rng = SplitRng::new(7);
            let sigma = (thermostat.thermal / thermostat.mass).sqrt();
            for velocity in &mut velocities {
                *velocity = rng.normal() * sigma;
            }
            zero_com(&mut velocities);
            let mut gradient = vec![0.0; positions.len()];
            pot.energy_gradient(&positions, cell, &mut gradient);
            let initial = coordination_deficit(&positions, cell);
            let sample = ((SAMPLE_PS / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
            let steps =
                ((spec.window_ps.min(20.0) / (spec.dt_fs * 1.0e-3)).round() as usize).max(1);
            let mut min_non12 = usize::MAX;
            let mut max_non12 = 0usize;
            let mut sum_non12 = 0.0;
            let mut zeros = 0usize;
            let mut samples = 0usize;
            for step in 1..=steps {
                thermostat.step(
                    &mut pot,
                    &mut positions,
                    &mut velocities,
                    &mut gradient,
                    cell,
                    &mut rng,
                );
                if step % sample == 0 {
                    let deficit = coordination_deficit(&positions, cell);
                    min_non12 = min_non12.min(deficit);
                    max_non12 = max_non12.max(deficit);
                    sum_non12 += deficit as f64;
                    zeros += usize::from(deficit == 0);
                    samples += 1;
                }
            }
            let kinetic =
                velocities.iter().map(|value| value * value).sum::<f64>() * 0.5 * CU_MASS_AMU;
            let target = 1.5 * (positions.len() / 3) as f64 * BOLTZMANN_EV_PER_K * spec.temps[0];
            println!(
                "coord {label} initial {initial} min {min_non12} mean {:.1} max {max_non12} zeros {zeros}/{samples} kinetic {kinetic:.1} target {target:.1}",
                sum_non12 / samples as f64
            );
        }
        return;
    }
    if env::var("MD_CHECK").ok().as_deref() == Some("1") {
        let mut check = MishinCu::new();
        let a = measured_lattice(500.0);
        let (big, big_cell) = fcc(13, a);
        let mut big_gradient = vec![0.0; big.len()];
        for _ in 0..2 {
            check.energy_gradient(&big, big_cell, &mut big_gradient);
        }
        let started = Instant::now();
        let calls = 10usize;
        for _ in 0..calls {
            check.energy_gradient(&big, big_cell, &mut big_gradient);
        }
        let each = started.elapsed().as_secs_f64() / calls as f64;
        println!(
            "force_call_s {each:.4} atoms {} pairs {}",
            big.len() / 3,
            check.pairs.len()
        );
        return;
    }

    if env::var("MD_MSD").ok().as_deref() == Some("1") {
        diffusion_run(&run_spec());
        return;
    }
    if env::var("MD_SIA_TRACER").ok().as_deref() == Some("1") {
        run_sia_tracer();
        return;
    }
    if env::var("MD_SIA_TRACER_NOTE").ok().as_deref() == Some("1") {
        write_tracer_notes();
        return;
    }

    let spec = run_spec();
    let jobs = jobs_for(&spec);
    let path = output_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("validation dir");
    }
    if !path.exists() {
        let mut header = File::create(&path).expect("csv");
        let names = if spec.capture_ws {
            HEADER.replace(
                "initial_non12,final_non12",
                "initial_ws_defects,final_ws_defects",
            )
        } else {
            HEADER.to_string()
        };
        writeln!(header, "{names}").unwrap();
    }
    let done = existing_keys(&path, spec.window_ps);
    let pending: Vec<&Job> = jobs
        .iter()
        .filter(|job| {
            !done.contains_key(&row_key(
                job.temperature,
                job.shell_r,
                job.orientation,
                job.seed_index,
                spec.window_ps,
            ))
        })
        .collect();
    println!(
        "jobs {} pending {} temps {:?} seeds {} window_ps {} dt_fs {} rmax {}",
        jobs.len(),
        pending.len(),
        spec.temps,
        spec.seeds,
        spec.window_ps,
        spec.dt_fs,
        spec.r_max_500
    );
    let file = Mutex::new(OpenOptions::new().append(true).open(&path).expect("append"));
    let cursor = AtomicUsize::new(0);
    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(pending.len().max(1));
    let started = Instant::now();
    thread::scope(|scope| {
        for _ in 0..workers {
            let pending = &pending;
            let cursor = &cursor;
            let file = &file;
            let spec = &spec;
            scope.spawn(move || {
                let mut potential = MishinCu::new();
                loop {
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    if index >= pending.len() {
                        break;
                    }
                    let job = pending[index];
                    let (n_atoms, outcome) = run_job(&mut potential, spec, job);
                    let line = format_row(spec, job, n_atoms, &outcome);
                    let mut handle: MutexGuard<'_, File> = file.lock().expect("csv lock");
                    writeln!(handle, "{line}").unwrap();
                    println!(
                        "{}/{} T {:.0} shell {:.3} {} seed {} captured {} t {:.3} ps non12 {} -> {}",
                        index + 1,
                        pending.len(),
                        job.temperature,
                        job.shell_r,
                        job.orientation,
                        job.seed_index,
                        u8::from(outcome.captured),
                        outcome.time_ps,
                        outcome.initial_score,
                        outcome.final_score
                    );
                }
            });
        }
    });
    let note = summarize(&path);
    fs::write(note_path(&path), note).expect("note");
    println!(
        "wrote {} in {:.1}s",
        path.display(),
        started.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Harmonic {
        k: f64,
    }

    impl EnergyGradient for Harmonic {
        fn energy_gradient(
            &mut self,
            positions: &[f64],
            _cell: [[f64; 3]; 3],
            gradient: &mut [f64],
        ) -> f64 {
            let mut energy = 0.0;
            for (slope, position) in gradient.iter_mut().zip(positions.iter()) {
                *slope = self.k * position;
                energy += 0.5 * self.k * position * position;
            }
            energy
        }
    }

    #[test]
    fn mishin_cohesive_energy_matches_the_copper_test() {
        let (positions, cell) = fcc(1, 3.615);
        let mut gradient = vec![0.0; positions.len()];
        let energy = analytic_gradient(&positions, cell, &mut gradient);
        assert!((energy / 4.0 + 3.54).abs() < 5.0e-4, "{}", energy / 4.0);
        assert!(gradient.iter().all(|value| value.abs() < 1.0e-6));
    }

    #[test]
    fn tabulated_cell_list_matches_the_analytic_gradient() {
        let (mut positions, cell) = fcc(6, 3.638666);
        positions[0] += 0.07;
        positions[4] -= 0.04;
        let mut analytic = vec![0.0; positions.len()];
        let mut table = vec![0.0; positions.len()];
        let energy_a = analytic_gradient(&positions, cell, &mut analytic);
        let mut pot = MishinCu::new();
        let energy_t = pot.energy_gradient(&positions, cell, &mut table);
        assert!(
            (energy_a - energy_t).abs() < 1.0e-6,
            "{energy_a} {energy_t}"
        );
        for (left, right) in analytic.iter().zip(table.iter()) {
            assert!((left - right).abs() < 1.0e-6, "{left} {right}");
        }
    }

    #[test]
    fn velocity_verlet_holds_the_harmonic_energy() {
        let mut potential = Harmonic { k: 1.0 };
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let mut positions = vec![1.0, 0.0, 0.0];
        let mut velocities = vec![0.0, 0.0, 0.0];
        let mut gradient = vec![0.0; 3];
        potential.energy_gradient(&positions, cell, &mut gradient);
        let thermostat = Langevin {
            dt: 0.01,
            gamma: 0.0,
            mass: 1.0,
            thermal: 0.0,
        };
        let mut rng = SplitRng::new(1);
        let mut worst = 0.0_f64;
        for _ in 0..2_000 {
            let energy = thermostat.step(
                &mut potential,
                &mut positions,
                &mut velocities,
                &mut gradient,
                cell,
                &mut rng,
            );
            let kinetic = 0.5 * velocities.iter().map(|v| v * v).sum::<f64>();
            worst = worst.max((energy + kinetic - 0.5).abs());
        }
        assert!(worst < 1.0e-4, "{worst}");
    }

    #[test]
    fn the_doubly_occupied_site_is_the_dumbbell_host_under_any_translation() {
        let a = measured_lattice(500.0);
        let (mut positions, _) = isolated_dumbbell(5, a);
        let host = interstitial_site(&positions, a, 5, [0.0; 3]).expect("one double site");
        let pair = &positions[positions.len() - 6..];
        let middle = [0, 1, 2].map(|axis| ((pair[axis] + pair[3 + axis]) / a).round() as i64);
        assert_eq!(host, middle);
        let shift = [0.31, -0.22, 0.17];
        for atom in positions.chunks_exact_mut(3) {
            for axis in 0..3 {
                atom[axis] += shift[axis] + 5.0 * a;
            }
        }
        assert_eq!(interstitial_site(&positions, a, 5, shift), Some(host));
        assert_eq!(site_step([9, 0, 1], [0, 9, 1], 10), [1, -1, 0]);
    }

    #[test]
    fn occupancy_counts_the_pair_and_not_a_perfect_crystal() {
        let a = measured_lattice(700.0);
        let (perfect, _) = fcc(4, a);
        assert_eq!(occupancy_defects(&perfect, a, 4, [0.0; 3]), 0);
        let shells = shell_table(a, 6.0);
        let (pair, _) = place_vacancy_dumbbell(4, a, shells[3].1, [0.0, 0.0, 1.0]);
        assert_eq!(occupancy_defects(&pair, a, 4, [0.0; 3]), 2);
    }

    #[test]
    fn placed_pair_keeps_8788_atoms_and_is_not_already_a_crystal() {
        let a = measured_lattice(500.0);
        let shells = shell_table(a, 10.8 * a / measured_lattice(500.0));
        assert!(shells.len() > 4);
        let (positions, cell) = place_vacancy_dumbbell(13, a, shells[0].1, [0.0, 0.0, 1.0]);
        assert_eq!(positions.len() / 3, 8788);
        assert!(coordination_deficit(&positions, cell) > 0);
    }

    /// The k that [`rate_to_volume`] turns into `volume` Ω.
    fn volume_to_rate(volume: f64, lattice: f64, d_i: f64) -> f64 {
        volume * lattice.powi(3) / 4.0 * dumbbell_channel_rate(lattice, d_i)
    }

    #[test]
    fn rate_to_volume_reproduces_the_published_600_k_volume() {
        // Fig. 1 of Mwungeli, Liu, and Demkowicz fits log(n/N) = -0.00173 t
        // (t in ps) for 8788 atoms at 600 K; Table 1 lists a = 3.649 Å,
        // D_i = 0.728 Å²/ps, and V_iv = 186 ± 14 Ω for that model.
        let lattice = 3.649_f64;
        let (d_i, _) = table_diffusivity(600.0).expect("Table 1 D_i");
        let rate = 0.00173 * (13.0 * lattice).powi(3);
        let volume = rate_to_volume(rate, lattice, d_i);
        assert!((volume - 185.4).abs() < 0.05, "{volume}");
        assert!((volume - 186.0).abs() < 14.0, "{volume}");
        let back = volume_to_rate(volume, lattice, d_i);
        assert!((back - rate).abs() / rate < 1.0e-12, "{back} {rate}");
        // α inverted, k a² / (α D_i Ω), gives 1/α² = 9/4 of the volume.
        let inverted = rate * lattice * lattice / (DUMBBELL_ALPHA * d_i) / (lattice.powi(3) / 4.0);
        assert!(
            (inverted / volume - 2.25).abs() < 1.0e-12,
            "{inverted} {volume}"
        );
        // Table 1, k-method, 8788 atoms, at its lattice parameters.
        for (temperature, k_method, a) in [
            (300.0, 254.0, 3.631),
            (500.0, 185.0, 3.643),
            (700.0, 203.0, 3.655),
        ] {
            let (d_i, _) = table_diffusivity(temperature).expect("Table 1 D_i");
            let rate = volume_to_rate(k_method, a, d_i);
            assert!((rate_to_volume(rate, a, d_i) - k_method).abs() < 1.0e-9);
        }
    }

    #[test]
    fn the_replica_diffusivity_is_the_free_dumbbell_note() {
        // direct_md_sia_d.md, 10 fs and 0.1/ps: 0.278 ± 0.033, 0.860 ± 0.071
        // and 1.886 ± 0.188 Å²/ps over 8 replicas.
        for (temperature, d, error) in [
            (300.0, 0.278, 0.033),
            (500.0, 0.860, 0.071),
            (700.0, 1.886, 0.188),
        ] {
            let lattice = measured_lattice(temperature);
            let (mean, sigma, replicas) =
                replica_diffusivity(temperature, lattice, 10.0, 0.1).expect("replicas");
            assert_eq!(replicas, 8);
            assert!((mean - d).abs() < 5.0e-4, "{mean}");
            assert!((sigma - error).abs() < 5.0e-4, "{sigma}");
        }
        assert!(replica_diffusivity(500.0, 3.643, 10.0, 0.1).is_none());
    }

    #[test]
    fn site_deltas_take_the_minimum_image() {
        for (delta, wrapped) in [
            (0, 0),
            (1, 1),
            (13, 13),
            (14, -12),
            (25, -1),
            (-25, 1),
            (-13, 13),
        ] {
            assert_eq!(
                site_step([0, 0, 0], [delta, 0, 0], 26),
                [wrapped, 0, 0],
                "{delta}"
            );
            assert_eq!(
                site_step([3, 5, 8], [3, 5, 8 + delta], 26),
                [0, 0, wrapped],
                "{delta}"
            );
        }
    }

    #[test]
    fn tracer_lags_average_over_every_origin() {
        let lattice = 2.0;
        let run = TracerRun {
            lattice,
            tau_ps: 5.0,
            unresolved: 0,
            sum_rule_breaks: 0,
            ws: vec![[0, 0, 0], [1, 1, 0], [1, 1, 2]],
            centre: vec![[0.0; 3], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            atom: vec![[0.0; 3], [0.0, 3.0, 0.0], [0.0, 3.0, 4.0]],
            jump_r2: vec![0.0, 2.0, 6.0],
            jumps: vec![0, 1, 3],
            sites: vec![
                vec![[0, 0, 0], [2, 0, 0]],
                vec![[1, 1, 0], [2, 0, 0]],
                vec![[1, 1, 0], [2, 0, 2]],
            ],
            positions: vec![
                vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0, 2.0, 0.0, 0.0],
                vec![1.0, 1.0, 0.0, 2.0, 0.0, 2.0],
            ],
        };
        let lags = tracer_lags(&run);
        assert_eq!(lags.len(), 2);
        let (lag, origins, mean, first) = &lags[0];
        assert_eq!((*lag, *origins), (1, 2));
        // Half-lattice units are a/2 = 1 Å here.
        assert!((mean.ws - (2.0 + 4.0) / 2.0).abs() < 1.0e-12);
        assert!((first.ws - 2.0).abs() < 1.0e-12);
        assert!((mean.centre - (1.0 + 4.0) / 2.0).abs() < 1.0e-12);
        assert!((mean.atom - (9.0 + 16.0) / 2.0).abs() < 1.0e-12);
        assert!((mean.jump - (2.0 + 4.0) / 2.0).abs() < 1.0e-12);
        assert!((mean.jumps - 1.5).abs() < 1.0e-12);
        assert!((mean.site - (2.0 + 4.0) / 2.0).abs() < 1.0e-12);
        assert!((mean.tracer_total() - (2.0 + 4.0) / 2.0).abs() < 1.0e-12);
        assert!((first.tracer[0] - 1.0).abs() < 1.0e-12);
        let (lag, origins, mean, first) = &lags[1];
        assert_eq!((*lag, *origins), (2, 1));
        assert!((mean.ws - 6.0).abs() < 1.0e-12 && (first.ws - 6.0).abs() < 1.0e-12);
        assert!((mean.site - 6.0).abs() < 1.0e-12);
        assert!((mean.tracer[2] - 4.0).abs() < 1.0e-12);
    }

    #[test]
    fn a_short_tracer_run_keeps_one_snapshot_per_block() {
        let spec = RunSpec {
            temps: vec![700.0],
            seeds: 1,
            window_ps: 0.0,
            dt_fs: 10.0,
            gamma_per_ps: 0.1,
            r_max_500: 0.0,
            repeats: 4,
            capture_ws: false,
        };
        let run = sia_tracer_trajectory(700.0, 0, 1.0, 3, &spec);
        assert_eq!(run.ws.len(), 4);
        assert_eq!(run.positions[0].len(), 3 * 257);
        assert_eq!(run.sum_rule_breaks, 0);
        assert!((run.tau_ps - 1.0).abs() < 1.0e-12);
        let lags = tracer_lags(&run);
        assert_eq!(lags.len(), 3);
        let (_, _, _, whole) = lags[2];
        // No two fcc sites are closer than λ = a/√2.
        let lambda_squared = 0.5 * run.lattice * run.lattice;
        assert!(whole.jumps > 0.0);
        assert!(whole.jump >= whole.jumps * lambda_squared * (1.0 - 1.0e-12));
    }

    #[test]
    fn a_johnson_walk_moves_the_atoms_by_the_dumbbell_displacement() {
        let mut rng = SplitRng::new(7);
        let jumps = 200;
        let walks = 400;
        let mut defect = 0.0;
        for _ in 0..walks {
            let (squared, site, net) = johnson_walk(jumps, &mut rng);
            assert_eq!(site, net);
            assert_eq!(squared % 2, 0);
            assert_eq!(site.iter().sum::<i64>().rem_euclid(2), 0);
            defect += site.iter().map(|value| (value * value) as f64).sum::<f64>();
        }
        // Uncorrelated jumps of λ² = 2 half-lattice units: ⟨|R|²⟩ = 2n.
        let ratio = defect / walks as f64 / (2.0 * jumps as f64);
        assert!((ratio - 1.0).abs() < 0.15, "{ratio}");
    }

    #[test]
    fn the_johnson_tracer_correlation_lies_below_one_half() {
        let (factor, error) = johnson_correlation(400, 1_000, 3);
        assert!(error < 0.01, "{error}");
        assert!(factor > 0.35 && factor < 0.5, "{factor} ± {error}");
    }
}
