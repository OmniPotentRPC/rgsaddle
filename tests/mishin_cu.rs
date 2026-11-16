//! Mishin copper EAM1 and the shear-sensitive interstitial saddle.
//!
//! The functions are the EAM1 form of Mishin, Mehl, Papaconstantopoulos,
//! Voter, and Kress, Phys. Rev. B 63, 224106 (2001),
//! doi:10.1103/PhysRevB.63.224106, Table I. Zhang, Wheatley, Chen, Qian,
//! and Demkowicz, Phys. Rev. Materials 6, 053605 (2022),
//! doi:10.1103/PhysRevMaterials.6.053605, find that a few percent of shear
//! switches a <100> dumbbell toward <110>. The dumbbell is the split
//! interstitial on one lattice site: that site is vacant and the extra
//! atom is the interstitial. Each search starts at the relaxed <100>
//! minimum, not at a precomputed saddle.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{
    BOLTZMANN_EV_PER_K, CellMask, DefectSource, ElasticDipole, ElasticMultipole, FieldSolver,
    FmmSolver, IndexCheck, Kelvin, LINEAR_FOLD_EV, MinModeConfig, MinModeStatus, PointSurface,
    SaddleError, SaddleSeed, SolidDimer, SolidDimerConfig, Strain, StrainResponse, StressCheck,
    Trust, TrustPolicy, cubic_rotations, index_at,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/fcc_dumbbell.rs"
));

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

struct CachedEval {
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
    energy: f64,
    gradient: Array1<f64>,
    stress: [[f64; 3]; 3],
}

struct Copper {
    calls: AtomicUsize,
    last: Mutex<Option<CachedEval>>,
}

impl Copper {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            last: Mutex::new(None),
        }
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Relaxed)
    }

    fn evaluate(
        &self,
        positions: ArrayView1<f64>,
        cell: [[f64; 3]; 3],
    ) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
        if let Ok(slot) = self.last.lock()
            && let Some(prev) = slot.as_ref()
            && prev.cell == cell
            && prev.positions.len() == positions.len()
            && prev
                .positions
                .iter()
                .zip(positions.iter())
                .all(|(left, right)| left == right)
        {
            return (prev.energy, prev.gradient.clone(), prev.stress);
        }
        self.calls.fetch_add(1, Ordering::Relaxed);
        let (energy, gradient, stress) = eam_eval(positions, cell);
        if let Ok(mut slot) = self.last.lock() {
            *slot = Some(CachedEval {
                positions: positions.to_owned(),
                cell,
                energy,
                gradient: gradient.clone(),
                stress,
            });
        }
        (energy, gradient, stress)
    }
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

fn morse(r: f64, r0: f64, alpha: f64) -> (f64, f64) {
    let z = -alpha * (r - r0);
    let e1 = z.exp();
    let e2 = (2.0 * z).exp();
    (e2 - 2.0 * e1, 2.0 * alpha * (e1 - e2))
}

fn pair(r: f64) -> (f64, f64) {
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

fn density(r: f64) -> (f64, f64) {
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

fn embed(rho: f64) -> (f64, f64) {
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

struct Pair {
    i: usize,
    j: usize,
    d: [f64; 3],
    r: f64,
}

#[allow(clippy::needless_range_loop)]
fn eam_eval(positions: ArrayView1<f64>, cell: [[f64; 3]; 3]) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
    let n = positions.len() / 3;
    let lengths = [0, 1, 2].map(|row| {
        (cell[row][0] * cell[row][0] + cell[row][1] * cell[row][1] + cell[row][2] * cell[row][2])
            .sqrt()
    });
    let min_len = lengths.into_iter().fold(f64::MAX, f64::min);
    let nimg = ((RC / min_len).ceil() as i32).max(1);
    let cutoff2 = RC * RC;
    let mut pairs = Vec::new();
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
                        if r2 > 1e-16 && r2 < cutoff2 {
                            pairs.push(Pair {
                                i,
                                j,
                                d,
                                r: r2.sqrt(),
                            });
                        }
                    }
                }
            }
        }
    }
    let mut rho = vec![0.0; n];
    for bond in &pairs {
        let weight = density(bond.r).0;
        rho[bond.i] += weight;
        if bond.i != bond.j {
            rho[bond.j] += weight;
        } else {
            // The opposite image is the other neighbor of this atom.
            rho[bond.i] += weight;
        }
    }
    let mut energy = 0.0;
    let mut fp = vec![0.0; n];
    for i in 0..n {
        let (value, slope) = embed(rho[i]);
        energy += value;
        fp[i] = slope;
    }
    let mut gradient = Array1::zeros(positions.len());
    let mut virial = [[0.0; 3]; 3];
    for bond in &pairs {
        let (v, dv) = pair(bond.r);
        let (_, drho) = density(bond.r);
        let inv_r = 1.0 / bond.r;
        if bond.i != bond.j {
            energy += v;
            let deriv = dv + (fp[bond.i] + fp[bond.j]) * drho;
            for c in 0..3 {
                let force = deriv * bond.d[c] * inv_r;
                gradient[3 * bond.i + c] -= force;
                gradient[3 * bond.j + c] += force;
            }
            for (a, virial_row) in virial.iter_mut().enumerate() {
                for (b, slot) in virial_row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        } else {
            // +s and -s are both neighbors. Their forces cancel; energy and virial add.
            energy += v;
            let deriv = dv + 2.0 * fp[bond.i] * drho;
            for (a, virial_row) in virial.iter_mut().enumerate() {
                for (b, slot) in virial_row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        }
    }
    let volume = det(cell).abs();
    let mut stress = [[0.0; 3]; 3];
    for (a, virial_row) in virial.iter().enumerate() {
        for (b, virial_ab) in virial_row.iter().enumerate() {
            stress[a][b] = virial_ab / volume;
        }
    }
    (energy, gradient, stress)
}

fn positive_image(sx: i32, sy: i32, sz: i32) -> bool {
    sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0)))
}

fn det(h: [[f64; 3]; 3]) -> f64 {
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

fn flat(cell: [[f64; 3]; 3]) -> [f64; 9] {
    [
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    ]
}

fn unflat(cell: &[f64; 9]) -> [[f64; 3]; 3] {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

struct Held<'a> {
    copper: &'a Copper,
    cell: [[f64; 3]; 3],
}

impl PointSurface for Held<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, self.cell);
        Ok((energy, gradient))
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, unflat(cell));
        Ok((energy, gradient))
    }

    fn cauchy_stress(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let (_, _, stress) = self.copper.evaluate(x, unflat(cell));
        Ok(Some(flat(stress)))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let n = x.len() / 3;
        let mut modes = Array2::zeros((3, x.len()));
        for atom in 0..n {
            for c in 0..3 {
                modes[(c, 3 * atom + c)] = 1.0;
            }
        }
        Ok(modes)
    }
}

fn fcc(n: usize, a: f64) -> (Array1<f64>, [[f64; 3]; 3]) {
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut values = Vec::new();
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                for b in basis {
                    values.push((ix as f64 + b[0]) * a);
                    values.push((iy as f64 + b[1]) * a);
                    values.push((iz as f64 + b[2]) * a);
                }
            }
        }
    }
    let length = n as f64 * a;
    (
        Array1::from(values),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    )
}

fn apply_shear(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    gamma: f64,
) -> (Array1<f64>, [[f64; 3]; 3]) {
    let deform = [[1.0, 0.0, 0.0], [gamma, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] =
                cell[i][0] * deform[0][j] + cell[i][1] * deform[1][j] + cell[i][2] * deform[2][j];
        }
    }
    let mut out = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            out[3 * atom + c] =
                row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
        }
    }
    (out, h)
}

fn dumbbell(perfect: &Array1<f64>, direction: [f64; 3], separation: f64) -> Array1<f64> {
    let mut values = Vec::with_capacity(perfect.len() + 3);
    values.extend_from_slice(&perfect.as_slice().unwrap()[3..]);
    for sign in [-0.5, 0.5] {
        for component in direction {
            values.push(sign * separation * component);
        }
    }
    Array1::from(values)
}

/// FIRE to `tol` eV/Å on the largest atom force under a 0.06 Å cap, on
/// the direct pair sum. Returns the steps taken.
/// FIRE 2.0 under a 0.06 Å cap until the largest gradient component is
/// below `tol`; the steps taken. The shear rows are compared with eOn's on
/// this component test, so it is not the per-atom norm of
/// `relax_to_tolerance`.
fn relax_direct(
    copper: &Copper,
    positions: &mut Array1<f64>,
    cell: [[f64; 3]; 3],
    tol: f64,
) -> usize {
    let mut solver = Solver::new(
        Method::fire(FireKind::V2),
        Control::default().with_maxiter(usize::MAX).with_gtol(0.0),
        positions.len(),
    );
    solver.set_atom_maxmove(0.06);
    let oracle = Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
        let (energy, gradient, _) = copper.evaluate(x, cell);
        (energy, gradient)
    });
    for done in 1..=rgsaddle::DEFAULT_MAX_STEPS {
        solver.step(&oracle, positions).expect("relax");
        let (_, gradient, _) = copper.evaluate(positions.view(), cell);
        if gradient.iter().fold(0.0_f64, |m, g| m.max(g.abs())) < tol {
            return done;
        }
    }
    panic!(
        "direct-sum relaxation above {tol:e} after {} steps",
        rgsaddle::DEFAULT_MAX_STEPS
    );
}

fn max_abs(gradient: &Array1<f64>) -> f64 {
    gradient.iter().map(|v| v.abs()).fold(0.0, f64::max)
}

fn invert(h: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
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

fn closest_pair(positions: &Array1<f64>, cell: [[f64; 3]; 3]) -> (usize, usize, [f64; 3]) {
    let inv = invert(cell);
    let n = positions.len() / 3;
    let mut best = f64::MAX;
    let mut which = (0, 1, [0.0; 3]);
    for i in 0..n {
        for j in i + 1..n {
            let d = [
                positions[3 * j] - positions[3 * i],
                positions[3 * j + 1] - positions[3 * i + 1],
                positions[3 * j + 2] - positions[3 * i + 2],
            ];
            let s = [
                d[0] * inv[0][0] + d[1] * inv[1][0] + d[2] * inv[2][0],
                d[0] * inv[0][1] + d[1] * inv[1][1] + d[2] * inv[2][1],
                d[0] * inv[0][2] + d[1] * inv[1][2] + d[2] * inv[2][2],
            ];
            let wrapped = [
                s[0] - s[0].round(),
                s[1] - s[1].round(),
                s[2] - s[2].round(),
            ];
            let cart = [
                wrapped[0] * cell[0][0] + wrapped[1] * cell[1][0] + wrapped[2] * cell[2][0],
                wrapped[0] * cell[0][1] + wrapped[1] * cell[1][1] + wrapped[2] * cell[2][1],
                wrapped[0] * cell[0][2] + wrapped[1] * cell[1][2] + wrapped[2] * cell[2][2],
            ];
            let r2 = cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2];
            if r2 < best {
                best = r2;
                which = (i, j, cart);
            }
        }
    }
    which
}

fn family_angle(u: [f64; 3], dirs: &[[f64; 3]]) -> f64 {
    dirs.iter()
        .map(|d| {
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            (u[0] * d[0] + u[1] * d[1] + u[2] * d[2]).abs() / n
        })
        .fold(0.0_f64, f64::max)
        .clamp(0.0, 1.0)
        .acos()
        .to_degrees()
}

fn axis_report(dir: [f64; 3]) -> (f64, f64, [f64; 3], f64) {
    let n = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    let u = [dir[0] / n, dir[1] / n, dir[2] / n];
    let to_100 = family_angle(u, &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let to_110 = family_angle(
        u,
        &[
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ],
    );
    (to_100, to_110, u, n)
}

fn shear_yx(cube: [[f64; 3]; 3], h: [[f64; 3]; 3]) -> f64 {
    let inv = invert(cube);
    inv[1][0] * h[0][0] + inv[1][1] * h[1][0] + inv[1][2] * h[2][0]
}

fn minimum_image_difference(
    target: &Array1<f64>,
    origin: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Array1<f64> {
    let inv = invert(cell);
    let mut mode = target - origin;
    for atom in 0..mode.len() / 3 {
        let d = [mode[3 * atom], mode[3 * atom + 1], mode[3 * atom + 2]];
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
        for c in 0..3 {
            mode[3 * atom + c] = w[0] * cell[0][c] + w[1] * cell[1][c] + w[2] * cell[2][c];
        }
    }
    mode
}

struct SearchRow {
    calls: usize,
    barrier: f64,
    enthalpy_barrier: f64,
    curvature: f64,
    status: MinModeStatus,
    max_force: f64,
    steps: usize,
    stress: [[f64; 3]; 3],
    angle_100: f64,
    angle_110: f64,
    axis: [f64; 3],
    separation: f64,
    final_shear: f64,
    volume: f64,
    switched: bool,
    negative_modes: Option<usize>,
    rejected: bool,
    calls_at_loose: usize,
    index_evaluations: usize,
    eigenvalues: Vec<f64>,
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
}

#[allow(clippy::too_many_arguments)]
fn saddle_from_minimum(
    copper: &Copper,
    start: &Array1<f64>,
    cell: [[f64; 3]; 3],
    mode: &Array1<f64>,
    mask: CellMask,
    external: [[f64; 3]; 3],
    cube: [[f64; 3]; 3],
    e_min: f64,
    min_angle_100: f64,
    _min_angle_110: f64,
    max_steps: usize,
) -> SearchRow {
    let before = copper.calls();
    let search = MinModeConfig::default()
        .with_method(Method::fire(FireKind::V2))
        .with_force_tol(2e-3)
        // At 0.008 Å the 5% fixed-cell search steps past the switch, which
        // sits 7.8 meV above the minimum, onto an unrelated saddle.
        .with_max_move(0.004)
        .with_dr(1e-3)
        .with_rotation_tol(1e-3)
        .with_max_rotations(12)
        .with_seed_curvature(false);
    let mut dimer = SolidDimer::new(
        SolidDimerConfig::new(search)
            .with_mask(mask)
            .with_external_stress(external)
            .with_index_check(IndexCheck::first_order())
            .with_flat(0.005),
        start.clone(),
        cell,
        mode.clone(),
    )
    .unwrap();
    let surface = Held { copper, cell };
    let mut report = dimer.step(&surface).expect("dimer");
    let mut calls_at_loose = None;
    let mut steps_taken = 1usize;
    loop {
        if calls_at_loose.is_none() && report.curvature < 0.0 && report.max_force <= 2e-3 {
            calls_at_loose = Some(copper.calls() - before);
        }
        if report.status != MinModeStatus::Running || steps_taken >= max_steps {
            break;
        }
        report = dimer.step(&surface).expect("dimer");
        steps_taken += 1;
    }
    let calls = copper.calls() - before;
    let (e_sad, _, stress_now) = copper.evaluate(dimer.positions(), dimer.cell());
    let stress = dimer.stress().unwrap_or(stress_now);
    let enthalpy = dimer.energy().unwrap_or(e_sad);
    let (_, _, dir) = closest_pair(&dimer.positions().to_owned(), dimer.cell());
    let (angle_100, angle_110, axis, separation) = axis_report(dir);
    // [110] has the same sign on x and y. [-110] is the other saddle.
    let switched = report.status == MinModeStatus::Converged
        && !report.rejected
        && report.curvature < 0.0
        && min_angle_100 < 25.0
        && angle_110 < 15.0
        && axis[0] * axis[1] > 0.0
        && angle_100 > min_angle_100 + 10.0;
    SearchRow {
        calls,
        barrier: e_sad - e_min,
        enthalpy_barrier: enthalpy - e_min,
        curvature: report.curvature,
        status: report.status,
        max_force: report.max_force,
        steps: report.iteration,
        stress,
        angle_100,
        angle_110,
        axis,
        separation,
        final_shear: shear_yx(cube, dimer.cell()),
        volume: dimer.volume(),
        switched,
        negative_modes: report.negative_modes,
        rejected: report.rejected,
        calls_at_loose: calls_at_loose.unwrap_or(calls),
        index_evaluations: report.index_evaluations,
        eigenvalues: report.eigenvalues,
        positions: dimer.positions().to_owned(),
        cell: dimer.cell(),
    }
}

fn scan_fixed_soft_mode(copper: &Copper, row: &SearchRow) {
    let surface = Held {
        copper,
        cell: row.cell,
    };
    for step in [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6] {
        let report = match index_at(
            &surface,
            row.positions.view(),
            IndexCheck::first_order().with_step(step),
        ) {
            Ok(report) => report,
            Err(err) => {
                eprintln!("FD_STEP step={step:.0e} failed {err}");
                continue;
            }
        };
        let second = report.values.get(1).copied().unwrap_or(f64::NAN);
        let overlap = translation_overlap(report.vectors.column(1), row.positions.len() / 3);
        eprintln!(
            "FD_STEP step={step:.0e} negative={} lowest={} second={second:.8e} translation_overlap={overlap:.6e} spectrum={}",
            report.negative,
            report.values.first().copied().unwrap_or(f64::NAN),
            report
                .values
                .iter()
                .take(6)
                .map(|value| format!("{value:.8e}"))
                .collect::<Vec<_>>()
                .join(";")
        );
    }
}

fn trace_soft_mode(copper: &Copper, row: &SearchRow) {
    let surface = Held {
        copper,
        cell: row.cell,
    };
    let origin = index_at(
        &surface,
        row.positions.view(),
        IndexCheck::first_order().with_step(1e-4),
    )
    .expect("soft mode at the rejected point");
    let e0 = copper.evaluate(row.positions.view(), row.cell).0;
    eprintln!(
        "SOFT_ORIGIN negative={} spectrum={}",
        origin.negative,
        origin
            .values
            .iter()
            .take(4)
            .map(|value| format!("{value:.8e}"))
            .collect::<Vec<_>>()
            .join(";")
    );
    if origin.negative < 2 {
        return;
    }
    let hard = origin.vectors.column(0).to_owned();
    for sign in [1.0, -1.0] {
        let mut q = row.positions.clone();
        let soft0 = origin.vectors.column(origin.negative - 1).to_owned();
        for step in 1..=12 {
            let (_, gradient) = surface.eval(q.view()).expect("soft walk gradient");
            let slope = gradient.dot(&soft0);
            let dir = if step == 1 || slope.abs() < 1e-8 {
                sign
            } else {
                -slope.signum()
            };
            q = &q + &(&soft0 * (dir * 0.05));
            let along_hard = (&q - &row.positions).dot(&hard);
            q = &q - &(&hard * along_hard);
            let report = match index_at(
                &surface,
                q.view(),
                IndexCheck::first_order().with_step(1e-4),
            ) {
                Ok(report) => report,
                Err(err) => {
                    eprintln!("SOFT_WALK sign={sign} step={step} failed {err}");
                    break;
                }
            };
            let energy = copper.evaluate(q.view(), row.cell).0;
            eprintln!(
                "SOFT_WALK sign={sign} step={step} dE={} negative={} lowest={} second={} hard_overlap={}",
                energy - e0,
                report.negative,
                report.values.first().copied().unwrap_or(f64::NAN),
                report.values.get(1).copied().unwrap_or(f64::NAN),
                along_hard
            );
            if report.negative <= 1 {
                break;
            }
        }
    }
}

fn translation_overlap(mode: ndarray::ArrayView1<f64>, n_atoms: usize) -> f64 {
    let mut mode_norm = 0.0;
    for value in mode.iter() {
        mode_norm += value * value;
    }
    mode_norm = mode_norm.sqrt();
    if mode_norm == 0.0 || n_atoms == 0 {
        return 0.0;
    }
    let mut largest: f64 = 0.0;
    for axis in 0..3 {
        let mut dot = 0.0;
        for atom in 0..n_atoms.min(mode.len() / 3) {
            dot += mode[3 * atom + axis];
        }
        let overlap = dot.abs() / (mode_norm * (n_atoms as f64).sqrt());
        largest = largest.max(overlap);
    }
    largest
}

struct ReactantIndex {
    negative: usize,
    lowest: f64,
    spectrum: String,
    calls: usize,
}

fn reactant_index(
    copper: &Copper,
    minimum: &Array1<f64>,
    cell: [[f64; 3]; 3],
    mask: CellMask,
    external: [[f64; 3]; 3],
) -> ReactantIndex {
    let before = copper.calls();
    let search = MinModeConfig::default().with_force_tol(2e-3);
    let dimer = SolidDimer::new(
        SolidDimerConfig::new(search)
            .with_mask(mask)
            .with_external_stress(external)
            .with_index_check(IndexCheck::first_order()),
        minimum.clone(),
        cell,
        Array1::ones(minimum.len()),
    )
    .unwrap();
    let surface = Held { copper, cell };
    let report = dimer.index(&surface).expect("reactant index");
    let spectrum = report
        .values
        .iter()
        .take(6)
        .map(|value| format!("{value:.8e}"))
        .collect::<Vec<_>>()
        .join(";");
    ReactantIndex {
        negative: report.negative,
        lowest: report.values.first().copied().unwrap_or(0.0),
        spectrum,
        calls: copper.calls() - before,
    }
}

fn print_row(shear: f64, method: &str, row: &SearchRow) {
    eprintln!(
        "RESULT shear={shear:.2} method={method} calls={} calls_at_loose={} index_evals={} barrier_eV={} enthalpy_barrier_eV={} curvature={} status={:?} rejected={} negative={} spectrum={} max_force={} steps={} stress_xx={} stress_yy={} stress_zz={} stress_yz={} stress_xz={} stress_xy={} axis_x={} axis_y={} axis_z={} separation={} angle_100={} angle_110={} switch={} final_shear={} volume={}",
        row.calls,
        row.calls_at_loose,
        row.index_evaluations,
        row.barrier,
        row.enthalpy_barrier,
        row.curvature,
        row.status,
        row.rejected as u8,
        row.negative_modes
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".into()),
        row.eigenvalues
            .iter()
            .take(6)
            .map(|value| format!("{value:.8e}"))
            .collect::<Vec<_>>()
            .join(";"),
        row.max_force,
        row.steps,
        row.stress[0][0],
        row.stress[1][1],
        row.stress[2][2],
        row.stress[2][1],
        row.stress[2][0],
        row.stress[1][0],
        row.axis[0],
        row.axis[1],
        row.axis[2],
        row.separation,
        row.angle_100,
        row.angle_110,
        row.switched as u8,
        row.final_shear,
        row.volume,
    );
}

fn result_phrase(unstable: bool, row: Option<&SearchRow>) -> String {
    if unstable {
        return "no barrier / reactant unstable".to_string();
    }
    let row = row.expect("a stable reactant has a search");
    if row.rejected {
        format!("rejected index {}", row.negative_modes.unwrap_or(0))
    } else if row.negative_modes == Some(1) && row.status == MinModeStatus::Converged {
        "index 1".to_string()
    } else {
        format!(
            "unresolved status {:?} negative {:?}",
            row.status, row.negative_modes
        )
    }
}

#[test]
fn mishin_crystal_matches_the_cohesive_energy_and_the_stress() {
    let copper = Copper::new();
    let a = 3.615;
    let (positions, cell) = fcc(1, a);
    let (energy, gradient, stress) = copper.evaluate(positions.view(), cell);
    let per_atom = energy / 4.0;
    eprintln!("cohesive_eV {per_atom} stress_xx {}", stress[0][0]);
    assert!((per_atom + 3.54).abs() < 5e-4, "cohesive {per_atom}");
    assert!(max_abs(&gradient) < 1e-6, "force {}", max_abs(&gradient));
    assert!(
        stress.iter().flatten().all(|v| v.abs() < 1e-4),
        "stress {stress:?}"
    );

    let mut moved = positions.clone();
    moved[0] += 0.04;
    let (_, analytic, _) = copper.evaluate(moved.view(), cell);
    let step = 1e-6;
    for c in 0..3 {
        let mut plus = moved.clone();
        let mut minus = moved.clone();
        plus[c] += step;
        minus[c] -= step;
        let e_plus = copper.evaluate(plus.view(), cell).0;
        let e_minus = copper.evaluate(minus.view(), cell).0;
        let fd = (e_plus - e_minus) / (2.0 * step);
        assert!(
            (fd - analytic[c]).abs() < 1e-5,
            "force {c}: {fd} vs {}",
            analytic[c]
        );
    }

    let gamma = 0.02;
    let (sheared_pos, sheared_cell) = apply_shear(&positions, cell, gamma);
    let (_, _, analytic_stress) = copper.evaluate(sheared_pos.view(), sheared_cell);
    let volume = det(sheared_cell).abs();
    let eps = 1e-6;
    let deform_plus = [[1.0 + eps, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let deform_minus = [[1.0 - eps, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let strain_of = |base: [[f64; 3]; 3], deform: [[f64; 3]; 3]| {
        let mut h = [[0.0; 3]; 3];
        let mut r = Array1::zeros(sheared_pos.len());
        for i in 0..3 {
            for j in 0..3 {
                h[i][j] = base[i][0] * deform[0][j]
                    + base[i][1] * deform[1][j]
                    + base[i][2] * deform[2][j];
            }
        }
        for atom in 0..sheared_pos.len() / 3 {
            let row = [
                sheared_pos[3 * atom],
                sheared_pos[3 * atom + 1],
                sheared_pos[3 * atom + 2],
            ];
            for c in 0..3 {
                r[3 * atom + c] =
                    row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
            }
        }
        copper.evaluate(r.view(), h).0
    };
    let fd_stress = (strain_of(sheared_cell, deform_plus) - strain_of(sheared_cell, deform_minus))
        / (2.0 * eps)
        / volume;
    assert!(
        (fd_stress - analytic_stress[0][0]).abs() < 1e-5,
        "stress fd {fd_stress} analytic {}",
        analytic_stress[0][0]
    );
}

/// [`Held`] with its stress handed back as a pressure tensor, `-σ`: a
/// LAMMPS `pxx ... pyz` with only its unit converted.
struct AsPressure<'a>(Held<'a>);

impl PointSurface for AsPressure<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.0.eval(x)
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        self.0.eval_in_cell(x, cell)
    }

    fn cauchy_stress(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        Ok(self
            .0
            .cauchy_stress(x, cell)?
            .map(|sigma| sigma.map(|v| -v)))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.0.excluded_modes(x)
    }
}

/// [`Held`] with its box refreshed only when an extent lx, ly or lz of
/// the cell changes, the way a driver that tracks the extents would.
struct ByExtents<'a> {
    copper: &'a Copper,
    cell: [[f64; 3]; 3],
    held: Mutex<Option<[f64; 9]>>,
}

impl ByExtents<'_> {
    fn held_box(&self, cell: &[f64; 9]) -> [[f64; 3]; 3] {
        let mut held = self.held.lock().unwrap();
        if held.is_none_or(|old| [0, 4, 8].iter().any(|&k| old[k] != cell[k])) {
            *held = Some(*cell);
        }
        unflat(&held.expect("refreshed"))
    }
}

impl PointSurface for ByExtents<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, self.cell);
        Ok((energy, gradient))
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, self.held_box(cell));
        Ok((energy, gradient))
    }

    fn cauchy_stress(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let (_, _, stress) = self.copper.evaluate(x, self.held_box(cell));
        Ok(Some(flat(stress)))
    }
}

/// A 4×4×4 copper cell stretched 1.2%, its rows `shape` in units of a
/// side, with the first site vacant and its [110] neighbour at the bond
/// centre. Returns the cell, the positions, and the hop direction.
fn vacancy_hop(shape: [[f64; 3]; 3]) -> ([[f64; 3]; 3], Array1<f64>, Array1<f64>) {
    let repeats = 4;
    let (perfect, cube) = fcc(repeats, 3.615 * 1.012);
    let side = cube[0][0];
    let cell = shape.map(|row| row.map(|v| v * side));
    let mut sites: Vec<[f64; 3]> = perfect
        .as_slice()
        .unwrap()
        .chunks(3)
        .skip(1)
        .map(|r| [r[0] / side, r[1] / side, r[2] / side])
        .collect();
    let neighbour = 0.5 / repeats as f64;
    let hopper = sites
        .iter()
        .position(|s| s == &[neighbour, neighbour, 0.0])
        .expect("a nearest neighbour of the vacancy");
    sites[hopper] = [0.5 * neighbour, 0.5 * neighbour, 0.0];
    let start: Array1<f64> = sites
        .iter()
        .flat_map(|s| {
            (0..3).map(move |c| s[0] * cell[0][c] + s[1] * cell[1][c] + s[2] * cell[2][c])
        })
        .collect();
    let mut hop = Array1::zeros(start.len());
    hop[3 * hopper] = std::f64::consts::FRAC_1_SQRT_2;
    hop[3 * hopper + 1] = std::f64::consts::FRAC_1_SQRT_2;
    (cell, start, hop)
}

/// A vacancy hop at its fixed-cell saddle, in a cell stretched and
/// tilted the way a 700 K snapshot leaves it. Diagonal strains held
/// with the shears free keep the volume exactly, and the search
/// converges once the shear stress of the snapshot has relaxed. The
/// opposite mask contracts the cell and converges as well. The same
/// surface with its stress reported as a pressure is stopped before
/// the first step.
#[test]
fn vacancy_hop_relaxes_free_shears_at_fixed_volume() {
    let copper = Copper::new();
    let (cell, start, hop) = vacancy_hop([
        [1.0021, 0.0, 0.0],
        [0.004, 0.9987, 0.0],
        [-0.003, 0.002, 1.0008],
    ]);
    let held = Held {
        copper: &copper,
        cell,
    };
    let mut fixed = SolidDimer::new(
        SolidDimerConfig::frozen(MinModeConfig::default()),
        start,
        cell,
        hop,
    )
    .unwrap();
    let fixed_report = fixed.run(&held, 200).unwrap();
    assert_eq!(fixed_report.status, MinModeStatus::Converged);
    let saddle_energy = copper.evaluate(fixed.positions(), cell).0;

    let restart = |mask: CellMask| {
        let config = SolidDimerConfig::new(MinModeConfig::default())
            .with_pressure(0.0)
            .with_weight(1.0)
            .with_index_check(IndexCheck::first_order())
            .with_mask(mask)
            .with_stress_check(StressCheck::new());
        SolidDimer::new(
            config,
            fixed.positions().to_owned(),
            cell,
            fixed.mode().to_owned(),
        )
        .unwrap()
    };
    let shears = CellMask::voigt([false, false, false, true, true, true]);
    let diagonals = CellMask::voigt([true, true, true, false, false, false]);

    let pressure = AsPressure(Held {
        copper: &copper,
        cell,
    });
    match restart(shears).step(&pressure) {
        Err(SaddleError::Surface(message)) => {
            assert!(
                message.contains("-1.0000 times the derivative"),
                "{message}"
            );
        }
        other => panic!("a pressure tensor passed the stress check: {other:?}"),
    }

    for mask in [shears, diagonals] {
        let mut dimer = restart(mask);
        let volume = dimer.volume();
        let report = dimer.run(&held, 80).unwrap();
        let relaxed = dimer.cell();
        let energy = copper.evaluate(dimer.positions(), relaxed).0;
        eprintln!(
            "VACANCY_HOP free={:?} fixed_steps={} steps={} max_force={:.2e} curvature={:.4} energy_drop_eV={:.5} volume_change={:.3e} tilt_change={:.4} {:.4} {:.4}",
            mask.flags(),
            fixed_report.iteration,
            report.iteration,
            report.max_force,
            report.curvature,
            saddle_energy - energy,
            dimer.volume() / volume - 1.0,
            relaxed[1][0] - cell[1][0],
            relaxed[2][0] - cell[2][0],
            relaxed[2][1] - cell[2][1],
        );
        assert_eq!(
            report.status,
            MinModeStatus::Converged,
            "{:?}: max force {} after {} steps",
            mask.flags(),
            report.max_force,
            report.iteration
        );
        assert!(!report.rejected);
        assert_eq!(report.negative_modes, Some(1));
        if mask == shears {
            for axis in 0..3 {
                assert_eq!(relaxed[axis][axis], cell[axis][axis]);
            }
            assert_eq!(dimer.volume(), volume);
            assert!((relaxed[1][0] - cell[1][0]).abs() > 1e-2, "{relaxed:?}");
            assert!(energy < saddle_energy, "{energy} {saddle_energy}");
        } else {
            // A held shear leaves each tilt in proportion to its column.
            for (row, col) in [(1, 0), (2, 0), (2, 1)] {
                let kept = relaxed[row][col] / relaxed[col][col];
                assert!((kept - cell[row][col] / cell[col][col]).abs() < 1e-12);
            }
            assert!(dimer.volume() < volume);
        }
    }
}

/// The hop at its fixed-cell saddle in an untilted cell under the
/// tension of a hot snapshot, where the only shear stress is the
/// defect's, under one percent of the tension. Held diagonals with
/// free shears change the cell at a fixed lx, ly, lz and volume, so a
/// box refreshed only when those change never sees the search. The
/// stress check stops it before the first step and passes the surface
/// that follows every entry.
#[test]
fn stress_check_stops_a_box_keyed_on_its_extents() {
    let copper = Copper::new();
    let (cell, start, hop) =
        vacancy_hop([[1.0021, 0.0, 0.0], [0.0, 0.9987, 0.0], [0.0, 0.0, 1.0008]]);
    let held = Held {
        copper: &copper,
        cell,
    };
    let mut fixed = SolidDimer::new(
        SolidDimerConfig::frozen(MinModeConfig::default()),
        start,
        cell,
        hop,
    )
    .unwrap();
    let fixed_report = fixed.run(&held, 200).unwrap();
    assert_eq!(fixed_report.status, MinModeStatus::Converged);
    let (_, _, stress) = copper.evaluate(fixed.positions(), cell);
    let tension = (0..3).map(|k| stress[k][k]).fold(f64::INFINITY, f64::min);
    eprintln!(
        "KEYED_BOX fixed_steps={} tension={:.4e} shear_yx={:.4e}",
        fixed_report.iteration, tension, stress[1][0]
    );
    assert!(
        stress[1][0].abs() > 1e-5 && stress[1][0].abs() < 0.01 * tension,
        "{stress:?}"
    );

    let config = SolidDimerConfig::new(MinModeConfig::default())
        .with_pressure(0.0)
        .with_mask(CellMask::voigt([false, false, false, true, true, true]))
        .with_stress_check(StressCheck::new());
    let restart = || {
        SolidDimer::new(
            config.clone(),
            fixed.positions().to_owned(),
            cell,
            fixed.mode().to_owned(),
        )
        .unwrap()
    };
    let keyed = ByExtents {
        copper: &copper,
        cell,
        held: Mutex::new(None),
    };
    let mut dimer = restart();
    let before = dimer.generalized().to_owned();
    match dimer.step(&keyed) {
        Err(SaddleError::Surface(message)) => {
            assert!(message.contains("every diagonal entry agrees"), "{message}");
        }
        other => panic!("a box keyed on its extents passed the stress check: {other:?}"),
    }
    assert_eq!(dimer.generalized(), before.view());
    restart().step(&held).unwrap();
}

#[test]
fn dumbbell_searches_start_at_the_sheared_minimum() {
    let copper = Copper::new();
    let n = 3;
    let (perfect, cube) = fcc(n, 3.615);
    let along_100 = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let along_110 = dumbbell(
        &perfect,
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ],
        2.05,
    );
    eprintln!(
        "atoms {} cell_side {}",
        along_100.len() / 3,
        n as f64 * 3.615
    );
    let mut barriers = Vec::new();
    let mut lines = vec![
        "shear,method,calls,steps,barrier_eV,enthalpy_barrier_eV,curvature,max_force,stress_xx,stress_yy,stress_zz,stress_yz,stress_xz,stress_xy,dipole_xy_eV,angle_100_deg,angle_110_deg,switch,final_shear,volume_A3,negative_modes,reactant_negative,reactant_lowest,calls_at_loose,index_evaluations,result".to_string(),
    ];
    for gamma in [0.0, 0.02, 0.03, 0.04, 0.05] {
        let (sheared, cell) = apply_shear(&along_100, cube, gamma);
        let (sheared_110, _) = apply_shear(&along_110, cube, gamma);
        let mut other = sheared_110;
        let other_steps = relax_direct(&copper, &mut other, cell, 1e-3);
        let (e_other, g_other, _) = copper.evaluate(other.view(), cell);
        let (_, _, dir_other) = closest_pair(&other, cell);
        let (other_100, other_110, _, other_sep) = axis_report(dir_other);
        let mut minimum = sheared;
        let relax_steps = relax_direct(&copper, &mut minimum, cell, 1e-3);
        let (e_min, g_min, sigma_min) = copper.evaluate(minimum.view(), cell);
        let g = max_abs(&g_min);
        let (_, _, dir) = closest_pair(&minimum, cell);
        let (min_100, min_110, axis, separation) = axis_report(dir);
        eprintln!(
            "MINIMUM shear={gamma:.2} relax_steps={relax_steps} max_force={g} energy={e_min} angle_100={min_100} angle_110={min_110} axis={} {} {} stress_xy={}",
            axis[0], axis[1], axis[2], sigma_min[1][0]
        );
        eprintln!("minimum_pair_distance {separation}");
        eprintln!(
            "RELAXED_110 shear={gamma:.2} steps={other_steps} max_force={} gap_vs_100_eV={} angle_100={other_100} angle_110={other_110} separation={other_sep}",
            max_abs(&g_other),
            e_other - e_min
        );
        let connect = minimum_image_difference(&other, &minimum, cell);
        assert!(g < 2e-3, "minimum force {g} at shear {gamma}");
        assert!((shear_yx(cube, cell) - gamma).abs() < 1e-12);
        if gamma == 0.0 {
            assert!(min_100 < 15.0, "cubic minimum left <100>: {min_100}");
        }
        // The lowest mode is two-sided. A 0.02 Å nudge toward the <110>
        // stationary point makes the uphill direction the soft side.
        let norm = connect.dot(&connect).sqrt();
        let mut nudged = minimum.clone();
        for (slot, component) in nudged.iter_mut().zip(connect.iter()) {
            *slot += 0.02 * component / norm;
        }
        let mode = connect.clone();
        let minimum_for_search = nudged;
        let e_nudge = copper.evaluate(minimum_for_search.view(), cell).0;
        eprintln!("nudge_dE_eV {}", e_nudge - e_min);
        let fixed_reactant =
            reactant_index(&copper, &minimum, cell, CellMask::frozen(), [[0.0; 3]; 3]);
        eprintln!(
            "REACTANT shear={gamma:.2} cell=fixed negative={} lowest={} calls={} spectrum={}",
            fixed_reactant.negative,
            fixed_reactant.lowest,
            fixed_reactant.calls,
            fixed_reactant.spectrum
        );
        let fixed = if fixed_reactant.negative > 0 {
            None
        } else {
            let fixed = saddle_from_minimum(
                &copper,
                &minimum_for_search,
                cell,
                &mode,
                CellMask::frozen(),
                [[0.0; 3]; 3],
                cube,
                e_min,
                min_100,
                min_110,
                2500,
            );
            print_row(gamma, "fixed", &fixed);
            if gamma == 0.0 {
                scan_fixed_soft_mode(&copper, &fixed);
            }
            Some(fixed)
        };
        let mut external = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                external[i][j] = -sigma_min[i][j];
            }
        }
        let variable_reactant = reactant_index(&copper, &minimum, cell, CellMask::all(), external);
        eprintln!(
            "REACTANT shear={gamma:.2} cell=variable negative={} lowest={} calls={} spectrum={}",
            variable_reactant.negative,
            variable_reactant.lowest,
            variable_reactant.calls,
            variable_reactant.spectrum
        );
        let variable = if variable_reactant.negative > 0 {
            None
        } else {
            let variable = saddle_from_minimum(
                &copper,
                &minimum_for_search,
                cell,
                &mode,
                CellMask::all(),
                external,
                cube,
                e_min,
                min_100,
                min_110,
                2500,
            );
            print_row(gamma, "variable", &variable);
            Some(variable)
        };
        let fixed = fixed.expect("fixed-cell reactant is a minimum");
        assert_eq!(fixed.status, MinModeStatus::Converged);
        assert!(fixed.curvature < 0.0);
        if gamma == 0.0 && fixed.negative_modes == Some(2) {
            trace_soft_mode(&copper, &fixed);
        } else {
            assert!(!fixed.rejected, "fixed index {:?}", fixed.negative_modes);
            assert_eq!(fixed.negative_modes, Some(1));
            assert!(
                fixed.switched,
                "fixed saddle left the <110> switch at {gamma}"
            );
        }
        assert!(fixed.barrier.is_finite() || fixed.rejected);
        if gamma == 0.0 && fixed.negative_modes == Some(1) {
            assert!(fixed.barrier > 0.4, "{}", fixed.barrier);
        }
        if gamma == 0.05 && fixed.negative_modes == Some(1) {
            assert!(fixed.barrier < 0.02, "{}", fixed.barrier);
        }
        if let Some(variable) = &variable {
            assert_eq!(variable.status, MinModeStatus::Converged);
            assert!(
                !variable.rejected,
                "variable index {:?}",
                variable.negative_modes
            );
            assert_eq!(variable.negative_modes, Some(1));
            assert!(variable.curvature < 0.0 && variable.barrier.is_finite());
        }
        lines.push(csv_row(gamma, "fixed", &fixed_reactant, Some(&fixed)));
        lines.push(csv_row(
            gamma,
            "variable",
            &variable_reactant,
            variable.as_ref(),
        ));
        barriers.push(fixed.barrier);
    }
    for pair in barriers.windows(2) {
        assert!(pair[1] < pair[0], "fixed barriers {barriers:?}");
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/solid_dimer_cu.csv");
    let mut file = File::create(&path).expect("create copper csv");
    for line in &lines {
        writeln!(file, "{line}").expect("write copper csv");
    }
    eprintln!("wrote {} rows to {}", lines.len() - 1, path.display());
}

fn csv_row(shear: f64, method: &str, reactant: &ReactantIndex, row: Option<&SearchRow>) -> String {
    let phrase = result_phrase(reactant.negative > 0, row);
    let blank = String::new();
    let num = |value: f64| value.to_string();
    let (
        calls,
        steps,
        barrier,
        enthalpy,
        curvature,
        max_force,
        stress,
        angle_100,
        angle_110,
        switched,
        shear_out,
        volume,
        negative,
        loose,
        index_evals,
    ) = if let Some(row) = row {
        (
            row.calls.to_string(),
            row.steps.to_string(),
            if phrase.starts_with("index") {
                num(row.barrier)
            } else {
                blank.clone()
            },
            if phrase.starts_with("index") {
                num(row.enthalpy_barrier)
            } else {
                blank.clone()
            },
            num(row.curvature),
            num(row.max_force),
            row.stress,
            num(row.angle_100),
            num(row.angle_110),
            (row.switched as u8).to_string(),
            num(row.final_shear),
            num(row.volume),
            row.negative_modes
                .map(|n| n.to_string())
                .unwrap_or_else(|| blank.clone()),
            row.calls_at_loose.to_string(),
            row.index_evaluations.to_string(),
        )
    } else {
        (
            reactant.calls.to_string(),
            "0".to_string(),
            blank.clone(),
            blank.clone(),
            blank.clone(),
            blank.clone(),
            [[0.0; 3]; 3],
            blank.clone(),
            blank.clone(),
            blank.clone(),
            blank.clone(),
            blank.clone(),
            blank.clone(),
            reactant.calls.to_string(),
            reactant.calls.to_string(),
        )
    };
    let dipole = if row.is_some() {
        (volume.parse::<f64>().unwrap_or(0.0) * stress[1][0]).to_string()
    } else {
        blank.clone()
    };
    [
        shear.to_string(),
        method.to_string(),
        calls,
        steps,
        barrier,
        enthalpy,
        curvature,
        max_force,
        if row.is_some() {
            num(stress[0][0])
        } else {
            blank.clone()
        },
        if row.is_some() {
            num(stress[1][1])
        } else {
            blank.clone()
        },
        if row.is_some() {
            num(stress[2][2])
        } else {
            blank.clone()
        },
        if row.is_some() {
            num(stress[2][1])
        } else {
            blank.clone()
        },
        if row.is_some() {
            num(stress[2][0])
        } else {
            blank.clone()
        },
        if row.is_some() {
            num(stress[1][0])
        } else {
            blank.clone()
        },
        dipole,
        angle_100,
        angle_110,
        switched,
        shear_out,
        volume,
        negative,
        reactant.negative.to_string(),
        reactant.lowest.to_string(),
        loose,
        index_evals,
        phrase,
    ]
    .join(",")
}

fn identity() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn rotate_positions(positions: &Array1<f64>, rotation: [[f64; 3]; 3]) -> Array1<f64> {
    let mut out = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let v = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for i in 0..3 {
            out[3 * atom + i] =
                rotation[i][0] * v[0] + rotation[i][1] * v[1] + rotation[i][2] * v[2];
        }
    }
    out
}

#[allow(clippy::needless_range_loop)]
fn deform(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
) -> (Array1<f64>, [[f64; 3]; 3]) {
    let mut grad = eps;
    for i in 0..3 {
        grad[i][i] += 1.0;
    }
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] = cell[i][0] * grad[0][j] + cell[i][1] * grad[1][j] + cell[i][2] * grad[2][j];
        }
    }
    let mut out = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            out[3 * atom + c] = row[0] * grad[0][c] + row[1] * grad[1][c] + row[2] * grad[2][c];
        }
    }
    (out, h)
}

fn csv_num(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.16}")
    } else {
        String::new()
    }
}

fn csv_opt(value: Option<f64>) -> String {
    value.map(csv_num).unwrap_or_default()
}

fn rel_rate(method_barrier: f64, explicit_barrier: f64, temperature: f64) -> f64 {
    let delta = method_barrier - explicit_barrier;
    (-delta / (BOLTZMANN_EV_PER_K * temperature)).exp() - 1.0
}

fn eon_barrier(strain: f64, variable: bool) -> Option<f64> {
    let key = (strain * 1000.0).round() as i32;
    if variable {
        return match key {
            0 => Some(0.3470069354772818),
            20 => Some(0.16255571725275786),
            30 => Some(0.08805194212288825),
            40 => Some(0.027924583692424676),
            _ => None,
        };
    }
    match key {
        0 => Some(0.4738130973207717),
        20 => Some(0.2435460895545134),
        30 => Some(0.14829201811522807),
        40 => Some(0.06786216555531155),
        50 => Some(0.007793737414942825),
        _ => None,
    }
}

struct ExplicitOut {
    calls: usize,
    relax_min_calls: usize,
    relax_110_calls: usize,
    reactant_index_calls: usize,
    search_calls: usize,
    barrier: f64,
    enthalpy: f64,
    accepted: bool,
    xy_switch: bool,
    n_neg: Option<usize>,
    reactant_negative: usize,
    reactant_lowest: f64,
    angle_110: f64,
    axis: [f64; 3],
    e_min: f64,
    min_force: f64,
    status: String,
    curvature: f64,
    saddle: Array1<f64>,
    minimum: Array1<f64>,
    cell: [[f64; 3]; 3],
    sigma_min: [[f64; 3]; 3],
    sigma_sad: [[f64; 3]; 3],
    mode: Array1<f64>,
    min_gradient: Array1<f64>,
    sad_gradient: Array1<f64>,
}

#[allow(clippy::too_many_arguments)]
fn explicit_barrier(
    copper: &Copper,
    cube: [[f64; 3]; 3],
    along_100: &Array1<f64>,
    along_110: &Array1<f64>,
    rotation: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
    variable: bool,
    xy_path: bool,
) -> ExplicitOut {
    let spun_100 = rotate_positions(along_100, rotation);
    let spun_110 = rotate_positions(along_110, rotation);
    let (placed_100, cell) = deform(&spun_100, cube, eps);
    let (placed_110, _) = deform(&spun_110, cube, eps);
    let mut other = placed_110;
    let before_110 = copper.calls();
    relax_direct(copper, &mut other, cell, 1e-3);
    let relax_110_calls = copper.calls() - before_110;
    let mut minimum = placed_100;
    let before_min = copper.calls();
    relax_direct(copper, &mut minimum, cell, 1e-3);
    let relax_min_calls = copper.calls() - before_min;
    let (e_min, g_min, sigma_min) = copper.evaluate(minimum.view(), cell);
    let min_gradient = g_min.clone();
    let min_force = max_abs(&g_min);
    let (_, _, dir) = closest_pair(&minimum, cell);
    let (min_100, _, _, _) = axis_report(dir);
    let connect = minimum_image_difference(&other, &minimum, cell);
    let norm = connect.dot(&connect).sqrt();
    let mut nudged = minimum.clone();
    if norm > 0.0 {
        for (slot, component) in nudged.iter_mut().zip(connect.iter()) {
            *slot += 0.02 * component / norm;
        }
    }
    let mut external = [[0.0; 3]; 3];
    let mask = if variable {
        for i in 0..3 {
            for j in 0..3 {
                external[i][j] = -sigma_min[i][j];
            }
        }
        CellMask::all()
    } else {
        CellMask::frozen()
    };
    let before_index = copper.calls();
    let reactant = reactant_index(copper, &minimum, cell, mask, external);
    let reactant_index_calls = copper.calls() - before_index;
    let before_search = copper.calls();
    let row = saddle_from_minimum(
        copper, &nudged, cell, &connect, mask, external, cube, e_min, min_100, 0.0, 2500,
    );
    let search_calls = copper.calls() - before_search;
    let (_, sad_gradient, sigma_sad) = copper.evaluate(row.positions.view(), row.cell);
    let mode = minimum_image_difference(&row.positions, &minimum, cell);
    let index1 = row.status == MinModeStatus::Converged
        && !row.rejected
        && row.negative_modes == Some(1)
        && row.curvature < 0.0
        && row.angle_110 < 15.0;
    let accepted = if xy_path {
        index1 && row.switched
    } else {
        index1
    };
    ExplicitOut {
        calls: relax_min_calls + relax_110_calls + reactant_index_calls + search_calls,
        relax_min_calls,
        relax_110_calls,
        reactant_index_calls,
        search_calls,
        barrier: row.barrier,
        enthalpy: row.enthalpy_barrier,
        accepted,
        xy_switch: row.switched,
        n_neg: row.negative_modes,
        reactant_negative: reactant.negative,
        reactant_lowest: reactant.lowest,
        angle_110: row.angle_110,
        axis: row.axis,
        e_min,
        min_force,
        status: format!("{:?}", row.status),
        curvature: row.curvature,
        saddle: row.positions.clone(),
        minimum: minimum.clone(),
        cell,
        sigma_min,
        sigma_sad,
        mode,
        min_gradient,
        sad_gradient,
    }
}

struct Tally {
    fixed: bool,
    calibration: bool,
    batch: bool,
    explicit_calls: usize,
    method_calls: usize,
    barrier_error: Option<f64>,
    rate_300: Option<f64>,
    rate_600: Option<f64>,
    refined: bool,
    linear: bool,
}

#[test]
#[ignore = "release copper dimer head-to-head; writes benchmarks/strain_response_head_to_head.csv"]
#[allow(clippy::needless_range_loop)]
fn strain_response_head_to_head() {
    let copper = Copper::new();
    let (perfect, cube) = fcc(3, 3.615);
    let along_100 = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let along_110 = dumbbell(
        &perfect,
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ],
        2.05,
    );
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benchmarks/strain_response_head_to_head.csv");
    let mut file = File::create(&path).expect("create head-to-head csv");
    writeln!(
        file,
        "kind,path,strain,cell,variant,events,calibration,explicit_barrier_eV,explicit_enthalpy_eV,explicit_is_barrier,xy_switch,explicit_n_neg,reactant_negative,reactant_lowest,explicit_angle_110_deg,explicit_axis_x,explicit_axis_y,explicit_axis_z,explicit_calls,relax_min_calls,relax_110_calls,reactant_index_calls,search_calls,linear_barrier_eV,method_barrier_eV,method_trust,method_refine_calls,method_refine_ok,barrier_error_eV,rate_rel_error_300K,rate_rel_error_600K,eon_eV,explicit_minus_eon_eV,wall_s,note"
    )
    .unwrap();
    file.flush().unwrap();
    let mut tallies = Vec::new();

    let before_bulk = copper.calls();
    let mut bulk = perfect.clone();
    relax_direct(&copper, &mut bulk, cube, 1e-3);
    let (_, _, sigma_bulk) = copper.evaluate(bulk.view(), cube);
    let bulk_calls = copper.calls() - before_bulk;

    let started = Instant::now();
    let zero = explicit_barrier(
        &copper,
        cube,
        &along_100,
        &along_110,
        identity(),
        Strain::zero().tensor(),
        false,
        true,
    );
    let zero_wall = started.elapsed().as_secs_f64();
    assert!(
        zero.accepted,
        "zero-strain saddle was not the index-1 <110> switch: status {} n_neg {:?} angle {} axis {:?}",
        zero.status, zero.n_neg, zero.angle_110, zero.axis
    );
    let volume = det(zero.cell).abs();
    let p_min = ElasticDipole::from_stress(volume, zero.sigma_min, sigma_bulk).unwrap();
    let p_sad = ElasticDipole::from_stress(volume, zero.sigma_sad, sigma_bulk).unwrap();
    let multipole = match force_multipole(&zero.minimum, &zero.min_gradient)
        .and_then(|_| force_multipole(&zero.saddle, &zero.sad_gradient))
    {
        Ok(()) => "relaxed residual forces accepted".to_string(),
        Err(err) => format!("relaxed residual forces refused: {err}"),
    };
    let captured = Reference {
        barrier0: zero.barrier,
        p_min,
        p_sad,
        p_sad_xy: p_sad.tensor()[0][1],
        saddle: zero.saddle.clone(),
        mode: zero.mode.clone(),
        multipole,
    };
    let calls_after_zero = copper.calls();
    eprintln!(
        "ZERO fixed barrier {} calls {} Psd_xy {} multipole {} wall {:.1}s",
        zero.barrier, zero.calls, captured.p_sad_xy, captured.multipole, zero_wall
    );

    let mut response = StrainResponse::new(captured.barrier0, captured.p_min, captured.p_sad)
        .unwrap()
        .with_prefactor(1.0, None)
        .unwrap()
        .with_point_group(&cubic_rotations())
        .unwrap();

    let calib_strain = Strain::shear_yx(0.005);
    let calib_started = Instant::now();
    let calib_estimate = response.barrier(&calib_strain);
    let calib = explicit_barrier(
        &copper,
        cube,
        &along_100,
        &along_110,
        variant_rotation(&response, &calib_estimate.variant),
        calib_strain.tensor(),
        false,
        true,
    );
    assert!(
        calib.accepted,
        "0.5% xy calibration search was not the shear-aligned switch: {}",
        calib.status
    );
    let residual = (calib.barrier - calib_estimate.barrier).abs();
    let coeff = residual / calib_strain.norm().powi(2);
    response = response
        .with_policy(
            TrustPolicy::default()
                .with_quadratic(coeff)
                .with_error_tolerance(LINEAR_FOLD_EV)
                .with_floor(0.0),
        )
        .unwrap();
    let build_calls = copper.calls();
    let build_note = format!(
        "one-time dipole build. barrier0_eV={} bulk_calls={bulk_calls} calls_after_zero={calls_after_zero} zero_explicit_calls={} calib_explicit_calls={} build_calls={build_calls} quadratic_eV={coeff:.16} residual_at_0.5pct_eV={residual:.16} tolerance_eV={LINEAR_FOLD_EV} prefactor_Hz=1 relative rate error does not depend on the shared prefactor multipole={} variant={}",
        csv_num(zero.barrier),
        zero.calls,
        calib.calls,
        captured.multipole.replace(',', ";"),
        calib_estimate.variant
    );
    write_sparse(
        &mut file,
        "build",
        "xy",
        "0",
        "fixed",
        "1",
        &build_calls.to_string(),
        &build_calls.to_string(),
        "",
        "",
        "",
        started.elapsed().as_secs_f64(),
        &build_note,
    );
    write_point(
        &mut file,
        &mut tallies,
        "grid",
        "xy",
        0.0,
        false,
        &calib_estimate.variant,
        1,
        false,
        &zero,
        Some(captured.barrier0),
        Some(captured.barrier0),
        "explicit_reference",
        0,
        false,
        zero_wall,
        "zero-strain fixed saddle; method barrier is barrier0; these calls also sit inside the build",
    );
    let variable_zero_started = Instant::now();
    let variable_zero = explicit_barrier(
        &copper,
        cube,
        &along_100,
        &along_110,
        identity(),
        Strain::zero().tensor(),
        true,
        true,
    );
    eprintln!(
        "ZERO variable enthalpy {} accepted {} calls {}",
        variable_zero.enthalpy, variable_zero.accepted, variable_zero.calls
    );
    write_point(
        &mut file,
        &mut tallies,
        "grid",
        "xy",
        0.0,
        true,
        &calib_estimate.variant,
        1,
        false,
        &variable_zero,
        Some(captured.barrier0),
        None,
        "",
        0,
        false,
        variable_zero_started.elapsed().as_secs_f64(),
        "unstrained cell, then free; dipole line is not this enthalpy",
    );
    write_point(
        &mut file,
        &mut tallies,
        "grid",
        "xy",
        0.005,
        false,
        &calib_estimate.variant,
        1,
        true,
        &calib,
        Some(calib_estimate.barrier),
        Some(calib_estimate.barrier),
        "linear",
        0,
        false,
        calib_started.elapsed().as_secs_f64(),
        "calibration point for the quadratic coefficient; the method answer is the dipole line",
    );
    file.flush().unwrap();

    let paths = ["xy", "xz", "uniaxial_x", "hydrostatic"];
    let strains = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05];
    for path_name in paths {
        for strain_value in strains {
            for variable in [false, true] {
                if path_name == "xy" && !variable && (strain_value - 0.005_f64).abs() < 1e-15 {
                    continue;
                }
                if variable && (path_name == "uniaxial_x" || path_name == "hydrostatic") {
                    continue;
                }
                let strain = strain_of(path_name, strain_value);
                let point_started = Instant::now();
                let estimate = response.barrier(&strain);
                let rotation = variant_rotation(&response, &estimate.variant);
                let found = explicit_barrier(
                    &copper,
                    cube,
                    &along_100,
                    &along_110,
                    rotation,
                    strain.tensor(),
                    variable,
                    path_name == "xy",
                );
                let (method_barrier, trust, refine_calls, refine_ok, note) = if variable {
                    (
                        None,
                        String::new(),
                        0,
                        false,
                        "variable-cell enthalpy is a different ensemble from the dipole line"
                            .to_string(),
                    )
                } else if matches!(estimate.trust, Trust::Linear | Trust::Explicit) {
                    (
                        Some(estimate.barrier),
                        trust_label(&estimate.trust),
                        0,
                        false,
                        "dipole line trusted; no refine".to_string(),
                    )
                } else {
                    let before_refine = copper.calls();
                    let surface = Held {
                        copper: &copper,
                        cell: deform(&along_100, cube, strain.tensor()).1,
                    };
                    let seed = SaddleSeed {
                        positions: rotate_positions(&captured.saddle, rotation),
                        cell: cube,
                        mode: rotate_positions(&captured.mode, rotation),
                        minimum_energy: found.e_min,
                    };
                    match response.refine(&strain, &surface, &seed, 2500) {
                        Ok(refined) => {
                            let calls = copper.calls() - before_refine + found.relax_min_calls;
                            (
                                Some(refined.barrier),
                                trust_label(&refined.trust),
                                calls,
                                true,
                                format!(
                                    "refined from the affine-strained zero-strain saddle; dimer_calls={}",
                                    copper.calls() - before_refine
                                ),
                            )
                        }
                        Err(err) => (
                            Some(estimate.barrier),
                            trust_label(&estimate.trust),
                            copper.calls() - before_refine + found.relax_min_calls,
                            false,
                            format!("refine failed: {}", err.to_string().replace(',', ";")),
                        ),
                    }
                };
                eprintln!(
                    "{path_name} {strain_value} variable={variable} explicit={} accepted={} method={} calls={} refine_calls={refine_calls} {:.1}s",
                    found.barrier,
                    found.accepted,
                    method_barrier.unwrap_or(f64::NAN),
                    found.calls,
                    point_started.elapsed().as_secs_f64()
                );
                write_point(
                    &mut file,
                    &mut tallies,
                    "grid",
                    path_name,
                    strain_value,
                    variable,
                    &estimate.variant,
                    1,
                    false,
                    &found,
                    Some(estimate.barrier),
                    method_barrier,
                    &trust,
                    refine_calls,
                    refine_ok,
                    point_started.elapsed().as_secs_f64(),
                    &note,
                );
                file.flush().unwrap();
            }
        }
    }

    let mut batch_explicit_once = Vec::new();
    let mut batch_seen = Vec::<(i64, usize)>::new();
    let sources: Vec<_> = (0..4)
        .map(|site| {
            let mut saddle = [[0.0; 3]; 3];
            saddle[0][1] = 8.0;
            saddle[1][0] = 8.0;
            DefectSource {
                position: [site as f64 * 6.0, 0.0, 0.0],
                dipole: ElasticDipole::from_tensor(saddle).unwrap(),
            }
        })
        .collect();
    let fields = FmmSolver::new(Kelvin::new(40.0, 0.3).unwrap())
        .fields(&sources)
        .unwrap();
    for site in 0..4 {
        let applied = Strain::shear_yx(0.01).tensor();
        let mut tensor = applied;
        for i in 0..3 {
            for j in 0..3 {
                tensor[i][j] += fields[site].strain.tensor()[i][j];
            }
        }
        let strain = Strain::from_tensor(tensor).unwrap();
        let point_started = Instant::now();
        let before = copper.calls();
        let estimate = response.barrier(&strain);
        let query_calls = copper.calls() - before;
        let rotation = variant_rotation(&response, &estimate.variant);
        let found = explicit_barrier(
            &copper,
            cube,
            &along_100,
            &along_110,
            rotation,
            strain.tensor(),
            false,
            true,
        );
        let (method_barrier, trust, refine_calls, refine_ok, note) = if matches!(
            estimate.trust,
            Trust::Linear | Trust::Explicit
        ) {
            (
                Some(estimate.barrier),
                trust_label(&estimate.trust),
                query_calls,
                false,
                format!(
                    "example field site {site}; events=50; one search measured, batch multiplies it"
                ),
            )
        } else {
            let before_refine = copper.calls();
            let surface = Held {
                copper: &copper,
                cell: deform(&along_100, cube, strain.tensor()).1,
            };
            let seed = SaddleSeed {
                positions: rotate_positions(&captured.saddle, rotation),
                cell: cube,
                mode: rotate_positions(&captured.mode, rotation),
                minimum_energy: found.e_min,
            };
            match response.refine(&strain, &surface, &seed, 2500) {
                Ok(refined) => (
                    Some(refined.barrier),
                    trust_label(&refined.trust),
                    query_calls + copper.calls() - before_refine + found.relax_min_calls,
                    true,
                    format!("example field site {site}; refined"),
                ),
                Err(err) => (
                    Some(estimate.barrier),
                    trust_label(&estimate.trust),
                    query_calls + copper.calls() - before_refine + found.relax_min_calls,
                    false,
                    format!(
                        "example field site {site}; refine failed: {}",
                        err.to_string().replace(',', ";")
                    ),
                ),
            }
        };
        batch_explicit_once.push(found.calls);
        let strain_key = (strain.tensor()[1][0] * 1.0e12).round() as i64;
        if !batch_seen.iter().any(|(key, _)| *key == strain_key) {
            batch_seen.push((strain_key, found.calls));
        }
        write_point(
            &mut file,
            &mut tallies,
            "batch",
            "akmc_strain",
            strain.tensor()[1][0],
            false,
            &estimate.variant,
            50,
            false,
            &found,
            Some(estimate.barrier),
            method_barrier,
            &trust,
            refine_calls,
            refine_ok,
            point_started.elapsed().as_secs_f64(),
            &format!("{note} norm={}", csv_num(strain.norm())),
        );
        file.flush().unwrap();
    }

    let fixed: Vec<_> = tallies
        .iter()
        .filter(|row| row.fixed && !row.batch)
        .collect();
    let variable: Vec<_> = tallies
        .iter()
        .filter(|row| !row.fixed && !row.batch)
        .collect();
    let explicit_fixed: usize = fixed.iter().map(|row| row.explicit_calls).sum::<usize>();
    let method_fixed: usize = build_calls + fixed.iter().map(|row| row.method_calls).sum::<usize>();
    let explicit_variable: usize = variable.iter().map(|row| row.explicit_calls).sum::<usize>();
    let batch_explicit: usize = batch_explicit_once
        .iter()
        .map(|calls| calls * 50)
        .sum::<usize>();
    let batch_method: usize = tallies
        .iter()
        .filter(|row| row.batch)
        .map(|row| row.method_calls)
        .sum::<usize>();
    let max_barrier = csv_opt(largest_abs(
        fixed
            .iter()
            .filter(|row| !row.calibration)
            .filter_map(|row| row.barrier_error),
    ));
    let max_linear_barrier = csv_opt(largest_abs(
        fixed
            .iter()
            .filter(|row| row.linear && !row.calibration)
            .filter_map(|row| row.barrier_error),
    ));
    let max_linear_300 = csv_opt(largest_abs(
        fixed
            .iter()
            .filter(|row| row.linear && !row.calibration)
            .filter_map(|row| row.rate_300),
    ));
    let max_linear_600 = csv_opt(largest_abs(
        fixed
            .iter()
            .filter(|row| row.linear && !row.calibration)
            .filter_map(|row| row.rate_600),
    ));
    let max_refined = csv_opt(largest_abs(
        fixed
            .iter()
            .filter(|row| row.refined)
            .filter_map(|row| row.barrier_error),
    ));
    let fixed_note = format!(
        "rows={} explicit_calls sums the fixed grid searches; method_refine_calls column is build_calls={build_calls} plus fixed refine attributions; barrier_error is max abs error excluding the 0.5% calibration; rate columns are max abs relative rate error on trusted linear points excluding calibration; max_abs_barrier_linear={max_linear_barrier} max_abs_barrier_refined={max_refined}",
        fixed.len()
    );
    write_sparse(
        &mut file,
        "total",
        "fixed_grid",
        "",
        "fixed",
        &fixed.len().to_string(),
        &explicit_fixed.to_string(),
        &method_fixed.to_string(),
        &max_barrier,
        &max_linear_300,
        &max_linear_600,
        started.elapsed().as_secs_f64(),
        &fixed_note,
    );
    write_sparse(
        &mut file,
        "total",
        "variable_grid",
        "",
        "variable",
        &variable.len().to_string(),
        &explicit_variable.to_string(),
        "",
        "",
        "",
        "",
        started.elapsed().as_secs_f64(),
        "explicit_calls sums the variable-cell searches. The dipole line is the imposed-strain barrier and does not replace these enthalpies.",
    );
    let batch_method_with_build = batch_method + build_calls;
    let batch_barrier = csv_opt(largest_abs(
        tallies
            .iter()
            .filter(|row| row.batch)
            .filter_map(|row| row.barrier_error),
    ));
    let batch_300 = csv_opt(largest_abs(
        tallies
            .iter()
            .filter(|row| row.batch)
            .filter_map(|row| row.rate_300),
    ));
    let batch_600 = csv_opt(largest_abs(
        tallies
            .iter()
            .filter(|row| row.batch)
            .filter_map(|row| row.rate_600),
    ));
    let unique_explicit: usize = batch_seen.iter().map(|(_, calls)| calls).sum::<usize>();
    let batch_note = format!(
        "200 events = 50 at each of 4 examples/akmc_strain.rs sites. explicit_calls multiplies each measured search by 50. unique strains={}; one search at each sums to {unique_explicit}. method_refine_calls includes the one-time build ({build_calls}) plus one query or refine per site (marginal {batch_method}). max_abs_rate_600={batch_600}",
        batch_seen.len()
    );
    write_sparse(
        &mut file,
        "total",
        "akmc_batch",
        "",
        "fixed",
        "200",
        &batch_explicit.to_string(),
        &batch_method_with_build.to_string(),
        &batch_barrier,
        &batch_300,
        &batch_600,
        started.elapsed().as_secs_f64(),
        &batch_note,
    );
    file.flush().unwrap();
}

struct Reference {
    barrier0: f64,
    p_min: ElasticDipole,
    p_sad: ElasticDipole,
    p_sad_xy: f64,
    saddle: Array1<f64>,
    mode: Array1<f64>,
    multipole: String,
}

#[allow(clippy::too_many_arguments)]
fn write_sparse(
    file: &mut File,
    kind: &str,
    path: &str,
    strain: &str,
    cell: &str,
    events: &str,
    explicit_calls: &str,
    method_calls: &str,
    barrier_error: &str,
    rate_300: &str,
    rate_600: &str,
    wall_s: f64,
    note: &str,
) {
    let mut fields = vec![String::new(); 35];
    fields[0] = kind.to_string();
    fields[1] = path.to_string();
    fields[2] = strain.to_string();
    fields[3] = cell.to_string();
    fields[5] = events.to_string();
    fields[18] = explicit_calls.to_string();
    fields[26] = method_calls.to_string();
    fields[28] = barrier_error.to_string();
    fields[29] = rate_300.to_string();
    fields[30] = rate_600.to_string();
    fields[33] = format!("{wall_s:.3}");
    fields[34] = format!("\"{}\"", note.replace('"', "'"));
    writeln!(file, "{}", fields.join(",")).unwrap();
}

fn force_multipole(positions: &Array1<f64>, gradient: &Array1<f64>) -> Result<(), SaddleError> {
    let n = positions.len() / 3;
    let mut pos = Vec::with_capacity(n);
    let mut force = Vec::with_capacity(n);
    for i in 0..n {
        pos.push([positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]]);
        force.push([-gradient[3 * i], -gradient[3 * i + 1], -gradient[3 * i + 2]]);
    }
    ElasticMultipole::about([0.0, 0.0, 0.0], &pos, &force, 3).map(|_| ())
}

fn strain_of(path: &str, value: f64) -> Strain {
    match path {
        "xy" => Strain::shear_yx(value),
        "xz" => Strain::shear_xz(value),
        "uniaxial_x" => Strain::uniaxial_x(value),
        "hydrostatic" => Strain::hydrostatic(value),
        _ => unreachable!("{path}"),
    }
}

fn variant_rotation(response: &StrainResponse, name: &str) -> [[f64; 3]; 3] {
    response
        .variants()
        .iter()
        .find(|variant| variant.name == name)
        .unwrap_or_else(|| panic!("missing variant {name}"))
        .rotation
}

fn trust_label(trust: &Trust) -> String {
    match trust {
        Trust::Linear => "linear".to_string(),
        Trust::Explicit => "explicit".to_string(),
        Trust::NeedsSearch(reason) => reason.to_string(),
    }
}

fn largest_abs(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut best = None;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        best = Some(best.map_or(value.abs(), |prior: f64| prior.max(value.abs())));
    }
    best
}

#[allow(clippy::too_many_arguments)]
fn write_point(
    file: &mut File,
    tallies: &mut Vec<Tally>,
    kind: &str,
    path: &str,
    strain: f64,
    variable: bool,
    variant: &str,
    events: usize,
    calibration: bool,
    found: &ExplicitOut,
    linear: Option<f64>,
    method_barrier: Option<f64>,
    trust: &str,
    refine_calls: usize,
    refine_ok: bool,
    wall_s: f64,
    note: &str,
) {
    let compare = found.accepted && !variable;
    let error = match (compare, method_barrier) {
        (true, Some(method)) => Some(method - found.barrier),
        _ => None,
    };
    let rate_300 = error.map(|_| rel_rate(method_barrier.unwrap(), found.barrier, 300.0));
    let rate_600 = error.map(|_| rel_rate(method_barrier.unwrap(), found.barrier, 600.0));
    let eon = if path == "xy" {
        eon_barrier(strain, variable)
    } else {
        None
    };
    let versus_eon = match (eon, variable) {
        (Some(reference), false) if found.accepted => Some(found.barrier - reference),
        (Some(reference), true) if found.accepted => Some(found.enthalpy - reference),
        _ => None,
    };
    writeln!(
        file,
        "{kind},{path},{strain},{},{variant},{events},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},\"{trust}\",{refine_calls},{},{},{},{},{},{},{:.3},\"{}\"",
        if variable { "variable" } else { "fixed" },
        calibration as u8,
        csv_num(found.barrier),
        csv_num(found.enthalpy),
        found.accepted as u8,
        found.xy_switch as u8,
        found.n_neg.map(|n| n.to_string()).unwrap_or_default(),
        found.reactant_negative,
        csv_num(found.reactant_lowest),
        csv_num(found.angle_110),
        csv_num(found.axis[0]),
        csv_num(found.axis[1]),
        csv_num(found.axis[2]),
        found.calls,
        found.relax_min_calls,
        found.relax_110_calls,
        found.reactant_index_calls,
        found.search_calls,
        csv_opt(linear),
        csv_opt(method_barrier),
        refine_ok as u8,
        csv_opt(error),
        csv_opt(rate_300),
        csv_opt(rate_600),
        csv_opt(eon),
        csv_opt(versus_eon),
        wall_s,
        format_args!(
            "{} status={} curvature={} min_force={}",
            note.replace('"', "'"),
            found.status,
            csv_num(found.curvature),
            csv_num(found.min_force)
        )
    )
    .unwrap();
    tallies.push(Tally {
        fixed: !variable,
        calibration,
        batch: kind == "batch",
        explicit_calls: found.calls,
        method_calls: refine_calls,
        barrier_error: error,
        rate_300,
        rate_600,
        refined: refine_ok && trust == "explicit",
        linear: trust == "linear" || trust == "explicit_reference",
    });
}

fn eam_eval_fast(
    positions: ArrayView1<f64>,
    cell: [[f64; 3]; 3],
) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
    if cell[0][1].abs()
        + cell[0][2].abs()
        + cell[1][0].abs()
        + cell[1][2].abs()
        + cell[2][0].abs()
        + cell[2][1].abs()
        > 1.0e-8
    {
        return eam_eval(positions, cell);
    }
    let n = positions.len() / 3;
    let pairs = image_pairs(positions, cell, false);
    accumulate_eam(n, positions.len(), &pairs, det(cell).abs())
}

/// Pairs within [`RC`] in an orthorhombic `cell`, in the order of a scan
/// that tries every image shift of an atom against the bins around it.
/// Unless `every_shift` is set, a neighbor is tried only at the shifts
/// that bring each of its coordinate differences inside the cutoff: a
/// rounded sum of squares is no smaller than any of its terms, so no other
/// shift passes. A stable sort on the shift then restores the scan order,
/// and the pairs are those of the full scan, bit for bit and in order. An
/// atom whose shifts land in different bins, and a cell that needs more
/// than one image, take the full scan.
fn image_pairs(positions: ArrayView1<f64>, cell: [[f64; 3]; 3], every_shift: bool) -> Vec<Pair> {
    let n = positions.len() / 3;
    let lengths = [cell[0][0].abs(), cell[1][1].abs(), cell[2][2].abs()];
    let nimg = ((RC / lengths.into_iter().fold(f64::MAX, f64::min)).ceil() as i32).max(1);
    let nside = lengths.map(|length| ((length / RC).floor() as i32).max(1));
    let bins_n = (nside[0] * nside[1] * nside[2]) as usize;
    let mut bins = vec![Vec::new(); bins_n];
    let inv = invert(cell);
    let mut frac = vec![[0.0; 3]; n];
    for i in 0..n {
        let r = [positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]];
        frac[i] = [
            r[0] * inv[0][0] + r[1] * inv[1][0] + r[2] * inv[2][0],
            r[0] * inv[0][1] + r[1] * inv[1][1] + r[2] * inv[2][1],
            r[0] * inv[0][2] + r[1] * inv[1][2] + r[2] * inv[2][2],
        ];
        bins[frac_bin(frac[i], nside)].push(i);
    }
    let cutoff2 = RC * RC;
    let mut pairs = Vec::new();
    let mut found = Vec::new();
    for i in 0..n {
        let origin = |shift: [i32; 3]| {
            frac_bin_ijk(
                [0, 1, 2].map(|c| (frac[i][c] - shift[c] as f64).rem_euclid(1.0)),
                nside,
            )
        };
        let centre = origin([0; 3]);
        if every_shift || nimg != 1 || origin([-1; 3]) != centre || origin([1; 3]) != centre {
            for sx in -nimg..=nimg {
                for sy in -nimg..=nimg {
                    for sz in -nimg..=nimg {
                        for slot in neighbor_slots(origin([sx, sy, sz]), nside) {
                            for &j in &bins[slot] {
                                if j < i || (j == i && !positive_image(sx, sy, sz)) {
                                    continue;
                                }
                                let shift = [
                                    sx as f64 * cell[0][0],
                                    sy as f64 * cell[1][1],
                                    sz as f64 * cell[2][2],
                                ];
                                let d = [
                                    positions[3 * j] + shift[0] - positions[3 * i],
                                    positions[3 * j + 1] + shift[1] - positions[3 * i + 1],
                                    positions[3 * j + 2] + shift[2] - positions[3 * i + 2],
                                ];
                                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                                if r2 > 1e-16 && r2 < cutoff2 {
                                    pairs.push(Pair {
                                        i,
                                        j,
                                        d,
                                        r: r2.sqrt(),
                                    });
                                }
                            }
                        }
                    }
                }
            }
            continue;
        }
        for slot in neighbor_slots(centre, nside) {
            'neighbor: for &j in &bins[slot] {
                if j < i {
                    continue;
                }
                let mut near = [[(0, 0.0); 3]; 3];
                let mut count = [0; 3];
                for (c, axis) in near.iter_mut().enumerate() {
                    for shift in -1..=1 {
                        let d =
                            positions[3 * j + c] + shift as f64 * cell[c][c] - positions[3 * i + c];
                        if d.abs() < RC {
                            axis[count[c]] = (shift, d);
                            count[c] += 1;
                        }
                    }
                    if count[c] == 0 {
                        continue 'neighbor;
                    }
                }
                for &(sx, dx) in &near[0][..count[0]] {
                    for &(sy, dy) in &near[1][..count[1]] {
                        for &(sz, dz) in &near[2][..count[2]] {
                            if j == i && !positive_image(sx, sy, sz) {
                                continue;
                            }
                            let r2 = dx * dx + dy * dy + dz * dz;
                            if r2 > 1e-16 && r2 < cutoff2 {
                                let pair = Pair {
                                    i,
                                    j,
                                    d: [dx, dy, dz],
                                    r: r2.sqrt(),
                                };
                                found.push((9 * sx + 3 * sy + sz, pair));
                            }
                        }
                    }
                }
            }
        }
        found.sort_by_key(|&(shift, _)| shift);
        pairs.extend(found.drain(..).map(|(_, pair)| pair));
    }
    pairs
}

/// The bins at offsets `-1..=1` from `origin` along each axis, each once,
/// in the order the offsets first reach them.
fn neighbor_slots(origin: [i32; 3], nside: [i32; 3]) -> Vec<usize> {
    let mut slots = Vec::with_capacity(27);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let ix = (origin[0] + dx).rem_euclid(nside[0]);
                let iy = (origin[1] + dy).rem_euclid(nside[1]);
                let iz = (origin[2] + dz).rem_euclid(nside[2]);
                let slot = ((ix * nside[1] + iy) * nside[2] + iz) as usize;
                if !slots.contains(&slot) {
                    slots.push(slot);
                }
            }
        }
    }
    slots
}

fn frac_bin(s: [f64; 3], nside: [i32; 3]) -> usize {
    let ijk = frac_bin_ijk(s, nside);
    ((ijk[0] * nside[1] + ijk[1]) * nside[2] + ijk[2]) as usize
}

fn frac_bin_ijk(s: [f64; 3], nside: [i32; 3]) -> [i32; 3] {
    [0, 1, 2].map(|c| {
        let scaled = s[c].rem_euclid(1.0) * nside[c] as f64;
        let mut bin = scaled.floor() as i32;
        if bin >= nside[c] {
            bin = nside[c] - 1;
        }
        if bin < 0 {
            bin = 0;
        }
        bin
    })
}

fn accumulate_eam(
    n: usize,
    n_coord: usize,
    pairs: &[Pair],
    volume: f64,
) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
    let mut rho = vec![0.0; n];
    for bond in pairs {
        let weight = density(bond.r).0;
        rho[bond.i] += weight;
        if bond.i != bond.j {
            rho[bond.j] += weight;
        } else {
            rho[bond.i] += weight;
        }
    }
    let mut energy = 0.0;
    let mut fp = vec![0.0; n];
    for i in 0..n {
        let (value, slope) = embed(rho[i]);
        energy += value;
        fp[i] = slope;
    }
    let mut gradient = Array1::zeros(n_coord);
    let mut virial = [[0.0; 3]; 3];
    for bond in pairs {
        let (v, dv) = pair(bond.r);
        let (_, drho) = density(bond.r);
        let inv_r = 1.0 / bond.r;
        if bond.i != bond.j {
            energy += v;
            let deriv = dv + (fp[bond.i] + fp[bond.j]) * drho;
            for c in 0..3 {
                let force = deriv * bond.d[c] * inv_r;
                gradient[3 * bond.i + c] -= force;
                gradient[3 * bond.j + c] += force;
            }
            for (a, virial_row) in virial.iter_mut().enumerate() {
                for (b, slot) in virial_row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        } else {
            energy += v;
            let deriv = dv + 2.0 * fp[bond.i] * drho;
            for (a, virial_row) in virial.iter_mut().enumerate() {
                for (b, slot) in virial_row.iter_mut().enumerate() {
                    *slot += deriv * bond.d[a] * bond.d[b] * inv_r;
                }
            }
        }
    }
    let mut stress = [[0.0; 3]; 3];
    for (a, virial_row) in virial.iter().enumerate() {
        for (b, virial_ab) in virial_row.iter().enumerate() {
            stress[a][b] = virial_ab / volume;
        }
    }
    (energy, gradient, stress)
}

#[test]
fn fast_mishin_matches_the_direct_pair_sum() {
    let (perfect, cell) = fcc(3, 3.615);
    let (energy, gradient, stress) = eam_eval(perfect.view(), cell);
    let (fast_e, fast_g, fast_s) = eam_eval_fast(perfect.view(), cell);
    assert!((energy - fast_e).abs() < 1e-8, "{energy} {fast_e}");
    for (left, right) in gradient.iter().zip(fast_g.iter()) {
        assert!((left - right).abs() < 1e-7, "{left} {right}");
    }
    for a in 0..3 {
        for b in 0..3 {
            assert!((stress[a][b] - fast_s[a][b]).abs() < 1e-8);
        }
    }
    let defect = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let (energy, gradient, stress) = eam_eval(defect.view(), cell);
    let (fast_e, fast_g, fast_s) = eam_eval_fast(defect.view(), cell);
    assert!((energy - fast_e).abs() < 1e-6, "{energy} {fast_e}");
    for (left, right) in gradient.iter().zip(fast_g.iter()) {
        assert!((left - right).abs() < 1e-6, "{left} {right}");
    }
    for a in 0..3 {
        for b in 0..3 {
            assert!((stress[a][b] - fast_s[a][b]).abs() < 1e-6);
        }
    }
}

#[test]
fn image_pairs_match_the_full_scan_bit_for_bit() {
    let mut rng = SplitRng::new(29);
    for n in [2, 3, 4, 6, 7] {
        let (mut positions, cell) = fcc(n, 3.615);
        for value in positions.iter_mut() {
            *value += 0.3 * rng.normal();
        }
        let atoms = positions.len() / 3;
        for atom in (0..atoms).step_by(7) {
            positions[3 * atom] += cell[0][0];
        }
        for atom in (2..atoms).step_by(11) {
            positions[3 * atom + 2] -= cell[2][2];
        }
        // A few ulps from a bin edge, an atom's shifts can land in other bins.
        let nside = ((cell[1][1] / RC).floor() as usize).max(1);
        for (k, atom) in (5..atoms).step_by(5).enumerate().take(48) {
            let edge = cell[1][1] * (k % (nside + 1)) as f64 / nside as f64;
            let ulps = (k / (nside + 1)) as f64 - 4.0;
            positions[3 * atom + 1] = edge + ulps * edge.abs().max(1.0) * f64::EPSILON;
        }
        let fast = image_pairs(positions.view(), cell, false);
        let scan = image_pairs(positions.view(), cell, true);
        assert!(!fast.is_empty(), "{n}");
        assert_eq!(fast.len(), scan.len(), "{n}");
        for (left, right) in fast.iter().zip(&scan) {
            assert_eq!((left.i, left.j), (right.i, right.j), "{n}");
            assert_eq!(left.d.map(f64::to_bits), right.d.map(f64::to_bits), "{n}");
            assert_eq!(left.r.to_bits(), right.r.to_bits(), "{n}");
        }
    }
}

struct FastCopper {
    calls: AtomicUsize,
}

impl FastCopper {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
        }
    }

    fn evaluate(
        &self,
        positions: ArrayView1<f64>,
        cell: [[f64; 3]; 3],
    ) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        eam_eval_fast(positions, cell)
    }
}

struct FastHeld<'a> {
    copper: &'a FastCopper,
    cell: [[f64; 3]; 3],
}

impl PointSurface for FastHeld<'_> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, self.cell);
        Ok((energy, gradient))
    }

    fn eval_in_cell(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, unflat(cell));
        Ok((energy, gradient))
    }

    fn cauchy_stress(
        &self,
        x: ArrayView1<f64>,
        cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        let (_, _, stress) = self.copper.evaluate(x, unflat(cell));
        Ok(Some(flat(stress)))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let n = x.len() / 3;
        let mut modes = Array2::zeros((3, x.len()));
        for atom in 0..n {
            for c in 0..3 {
                modes[(c, 3 * atom + c)] = 1.0;
            }
        }
        Ok(modes)
    }
}

/// FIRE to `tol` eV/Å on the largest atom force, for a start that is
/// already in the basin it should end in.
fn settle(
    copper: &FastCopper,
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
    tol: f64,
) -> Result<Array1<f64>, SaddleError> {
    rgsaddle::relax_to_tolerance(
        &FastHeld { copper, cell },
        positions,
        cell,
        &pair_fire().with_force_tol(tol),
    )
    .map(|done| done.positions)
}

/// The steepest-descent flow to `tol` eV/Å, for a start whose basin is the
/// result: a push off a saddle or a placed pair.
fn descend(
    copper: &FastCopper,
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
    tol: f64,
) -> Result<Array1<f64>, SaddleError> {
    rgsaddle::relax_to_tolerance(
        &FastHeld { copper, cell },
        positions,
        cell,
        &pair_relax().with_force_tol(tol),
    )
    .map(|done| done.positions)
}

fn shell_table(a: f64, repeats: usize) -> Vec<(f64, [f64; 3], usize)> {
    shell_table_within(a, repeats, 10.8)
}

/// Lattice vectors out to the scan cutoff, grouped into orbits of a
/// `<100>` dumbbell along x: `x -> -x` and the eight symmetries of the
/// square in `(y, z)`. A shell of one radius can hold several orbits, and
/// so can one radius with two crystallographic families. Each orbit is
/// `(radius, representative, sites)` with the representative
/// `(|i|, max(|j|, |k|), min(|j|, |k|)) a / 2`, so the choice does not
/// depend on how equal radii happen to round.
fn axis_orbits(a: f64, repeats: usize) -> Vec<(f64, [f64; 3], usize)> {
    let max_r = 10.8_f64.min(repeats as f64 * a * 0.48);
    let limit = ((2.0 * max_r / a).ceil() as i32) + 1;
    let mut orbits: std::collections::BTreeMap<(i32, i32, i32, i32), usize> = Default::default();
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 {
                    continue;
                }
                let n2 = i * i + j * j + k * k;
                let r = 0.5 * a * (n2 as f64).sqrt();
                if n2 < 1 || r < 0.5 * a || r > max_r {
                    continue;
                }
                let (high, low) = (j.abs().max(k.abs()), j.abs().min(k.abs()));
                *orbits.entry((n2, i.abs(), high, low)).or_default() += 1;
            }
        }
    }
    orbits
        .into_iter()
        .map(|((n2, i, j, k), sites)| {
            (
                0.5 * a * (n2 as f64).sqrt(),
                [0.5 * a * i as f64, 0.5 * a * j as f64, 0.5 * a * k as f64],
                sites,
            )
        })
        .collect()
}

fn shell_table_within(a: f64, repeats: usize, max_r: f64) -> Vec<(f64, [f64; 3], usize)> {
    let max_r = max_r.min(repeats as f64 * a * 0.48);
    let limit = ((2.0 * max_r / a).ceil() as i32) + 1;
    let mut raw = Vec::new();
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                let r = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                if r < 0.5 * a || r > max_r {
                    continue;
                }
                raw.push((r, cart));
            }
        }
    }
    raw.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap());
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
        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        if r2 < best_r {
            best_r = r2;
            best = index;
        }
    }
    best
}

struct BuiltPair {
    positions: Array1<f64>,
    vacancy: [f64; 3],
    dumbbell: [usize; 2],
}

fn build_pair(repeats: usize, a: f64, shell: [f64; 3], vacancy: bool) -> BuiltPair {
    build_oriented(repeats, a, shell, [1.0, 0.0, 0.0], vacancy)
}

fn build_oriented(
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
    vacancy: bool,
) -> BuiltPair {
    let (flat, cell) = fcc(repeats, a);
    let mut points = Vec::new();
    for atom in 0..flat.len() / 3 {
        points.push([flat[3 * atom], flat[3 * atom + 1], flat[3 * atom + 2]]);
    }
    let center = [cell[0][0] * 0.5, cell[1][1] * 0.5, cell[2][2] * 0.5];
    let vac = nearest(&points, center, None);
    let sia_guess = [
        points[vac][0] + shell[0],
        points[vac][1] + shell[1],
        points[vac][2] + shell[2],
    ];
    let sia = nearest(&points, sia_guess, Some(vac));
    let vac_pos = points[vac];
    let sia_pos = points[sia];
    let mut keep = Vec::new();
    for (index, point) in points.iter().enumerate() {
        if index == sia || (vacancy && index == vac) {
            continue;
        }
        keep.push(*point);
    }
    let offset = 1.025;
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1.0e-8);
    let unit = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    keep.push([
        sia_pos[0] - offset * unit[0],
        sia_pos[1] - offset * unit[1],
        sia_pos[2] - offset * unit[2],
    ]);
    keep.push([
        sia_pos[0] + offset * unit[0],
        sia_pos[1] + offset * unit[1],
        sia_pos[2] + offset * unit[2],
    ]);
    let n = keep.len();
    let mut values = Vec::with_capacity(n * 3);
    for point in keep {
        values.extend(point);
    }
    BuiltPair {
        positions: Array1::from(values),
        vacancy: vac_pos,
        dumbbell: [n - 2, n - 1],
    }
}

fn centroid(positions: &Array1<f64>, atoms: [usize; 2]) -> [f64; 3] {
    let mut c = [0.0; 3];
    for atom in atoms {
        for k in 0..3 {
            c[k] += positions[3 * atom + k];
        }
    }
    [c[0] * 0.5, c[1] * 0.5, c[2] * 0.5]
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
}
fn env_csv(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .unwrap_or_else(|_| default.to_string())
        .split(',')
        .filter(|item| !item.is_empty())
        .map(|item| item.trim().to_string())
        .collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(default)
}

struct SplitRng {
    state: u64,
}

impl SplitRng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }

    fn u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

fn zero_mean(values: &mut Array1<f64>) {
    let n = values.len() / 3;
    if n == 0 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for c in 0..3 {
            mean[c] += values[3 * atom + c];
        }
    }
    for atom in 0..n {
        for c in 0..3 {
            values[3 * atom + c] -= mean[c] / n as f64;
        }
    }
}

fn mic_distance(a: [f64; 3], b: [f64; 3], cell: [[f64; 3]; 3]) -> f64 {
    let mut acc = 0.0;
    for c in 0..3 {
        let length = cell[c][c];
        let mut delta = a[c] - b[c];
        delta -= length * (delta / length).round();
        acc += delta * delta;
    }
    acc.sqrt()
}

fn point_of(positions: &Array1<f64>, atom: usize) -> [f64; 3] {
    [
        positions[3 * atom],
        positions[3 * atom + 1],
        positions[3 * atom + 2],
    ]
}

fn hausdorff(left: &Array1<f64>, right: &Array1<f64>, cell: [[f64; 3]; 3]) -> f64 {
    directed_gap(left, right, cell).max(directed_gap(right, left, cell))
}

fn directed_gap(left: &Array1<f64>, right: &Array1<f64>, cell: [[f64; 3]; 3]) -> f64 {
    let n = left.len() / 3;
    let m = right.len() / 3;
    let mut worst = 0.0_f64;
    for i in 0..n {
        let p = point_of(left, i);
        let mut best = f64::MAX;
        for j in 0..m {
            best = best.min(mic_distance(p, point_of(right, j), cell));
        }
        worst = worst.max(best);
    }
    worst
}

#[allow(clippy::needless_range_loop)]
fn axis_of(positions: &Array1<f64>, dumbbell: [usize; 2]) -> [f64; 3] {
    let mut vector = [0.0; 3];
    for c in 0..3 {
        vector[c] = positions[3 * dumbbell[1] + c] - positions[3 * dumbbell[0] + c];
    }
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if norm > 0.0 {
        for c in 0..3 {
            vector[c] /= norm;
        }
    }
    vector
}

fn axis_dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    (left[0] * right[0] + left[1] * right[1] + left[2] * right[2]).abs()
}

fn miller(cart: [f64; 3], a: f64) -> (i32, i32, i32) {
    let scale = 2.0 / a;
    (
        (cart[0] * scale).round() as i32,
        (cart[1] * scale).round() as i32,
        (cart[2] * scale).round() as i32,
    )
}

fn cart_of(key: (i32, i32, i32), a: f64) -> [f64; 3] {
    [
        key.0 as f64 * 0.5 * a,
        key.1 as f64 * 0.5 * a,
        key.2 as f64 * 0.5 * a,
    ]
}

/// Zero-pressure lattice parameter of the Mishin crystal.
///
/// Langevin dynamics plus a Berendsen barostat. The stress this EAM
/// returns is tension-positive, so a compressed cell (negative trace)
/// is expanded. The average is the second half of the run.
fn npt_lattice(copper: &FastCopper, temperature: f64) -> f64 {
    let repeats = 6usize;
    let (mut positions, mut cell) = fcc(repeats, 3.615);
    let thermal = rgsaddle::BOLTZMANN_EV_PER_K * temperature;
    let omega = rgsaddle::angular_scale_rad_s();
    let dt = 2.0e-15 * omega;
    let gamma = 1.0e12 / omega;
    let mass = 63.546;
    let mut rng = SplitRng::new((temperature as u64).wrapping_mul(17).wrapping_add(11));
    let mut velocity = Array1::zeros(positions.len());
    let sigma = (thermal / mass).sqrt();
    for value in velocity.iter_mut() {
        *value = rng.normal() * sigma;
    }
    zero_mean(&mut velocity);
    let beta = 1.0 / 0.87;
    let tau = 0.5e-12 * omega;
    let steps = 4000usize;
    let mut sum = 0.0;
    let mut count = 0.0;
    for step in 0..steps {
        let (_, gradient, _) = copper.evaluate(positions.view(), cell);
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass));
        positions = &positions + &(&velocity * (0.5 * dt));
        let damp = (-gamma * dt).exp();
        let noise = (thermal / mass * (1.0 - damp * damp)).sqrt();
        for value in velocity.iter_mut() {
            *value = damp * *value + noise * rng.normal();
        }
        zero_mean(&mut velocity);
        positions = &positions + &(&velocity * (0.5 * dt));
        let (_, gradient, stress) = copper.evaluate(positions.view(), cell);
        velocity = &velocity + &(&gradient * (-0.5 * dt / mass));
        let hydro = (stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        let mu = (1.0 - beta * dt / tau * hydro).clamp(0.995, 1.005);
        let scale = mu.cbrt();
        positions *= scale;
        cell[0][0] *= scale;
        cell[1][1] *= scale;
        cell[2][2] *= scale;
        if step + 1 == steps / 2 || step + 1 == steps {
            eprintln!(
                "npt {temperature} K step {} a {:.5} stress {:.4e}",
                step + 1,
                cell[0][0] / repeats as f64,
                hydro
            );
        }
        if step >= steps / 2 {
            sum += cell[0][0] / repeats as f64;
            count += 1.0;
        }
    }
    sum / count
}

#[derive(Clone)]
struct Escape {
    product: Array1<f64>,
    barrier: f64,
    direction: String,
    axis_dot: f64,
}

fn enumerate_escapes(
    copper: &FastCopper,
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    fixed: [f64; 3],
    dumbbell: [usize; 2],
    tries: usize,
    seed: u64,
) -> Vec<Escape> {
    let held = FastHeld { copper, cell };
    let before = rgsaddle::pair_separation(reactant.view(), cell, fixed, &dumbbell).unwrap_or(0.0);
    let axis = axis_of(reactant, dumbbell);
    let found: Vec<Option<Escape>> = (0..tries)
        .into_par_iter()
        .map(|trial| {
            let mut rng = SplitRng::new(seed.wrapping_add(trial as u64).wrapping_mul(0x9E37));
            let mut kick = Array1::zeros(reactant.len());
            for atom in dumbbell {
                for c in 0..3 {
                    kick[3 * atom + c] = rng.normal() * 0.2;
                }
            }
            zero_mean(&mut kick);
            let norm = kick.dot(&kick).sqrt();
            if norm < 1.0e-6 {
                return None;
            }
            let mode = &kick / norm;
            let seed_pos = reactant + &kick;
            let search = MinModeConfig::default()
                .with_force_tol(0.04)
                .with_max_move(0.1)
                .with_dr(1.0e-3)
                .with_max_rotations(8);
            let dimer_config = SolidDimerConfig::frozen(search)
                .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1));
            let mut dimer = SolidDimer::new(dimer_config, seed_pos, cell, mode).ok()?;
            let report = dimer.run(&held, 80).ok()?;
            if report.status != MinModeStatus::Converged
                || report.rejected
                || report.negative_modes != Some(1)
            {
                eprintln!(
                    "trial {trial} status {:?} force {:.3} steps {} negative {:?} rejected {}",
                    report.status,
                    report.max_force,
                    report.iteration,
                    report.negative_modes,
                    report.rejected
                );
                let _ = std::io::stderr().flush();
                return None;
            }
            let saddle = dimer.positions().to_owned();
            let mode = dimer.mode().to_owned();
            let length = mode.dot(&mode).sqrt();
            if length < 1.0e-8 {
                return None;
            }
            let mut peak = 0.0_f64;
            let n_mode = mode.len() / 3;
            for atom in 0..n_mode {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += mode[3 * atom + c] * mode[3 * atom + c];
                }
                peak = peak.max(acc.sqrt());
            }
            if peak < 1.0e-8 {
                return None;
            }
            let step = &mode * (0.5 / peak);
            let plus = &saddle + &step;
            let minus = &saddle - &step;
            let plus = descend(copper, plus, cell, 1.5e-3).ok()?;
            let minus = descend(copper, minus, cell, 1.5e-3).ok()?;
            let gap_plus = hausdorff(&plus, reactant, cell);
            let gap_minus = hausdorff(&minus, reactant, cell);
            let (reactant_energy, _) = held.eval(reactant.view()).ok()?;
            let (energy_plus, _) = held.eval(plus.view()).ok()?;
            let (energy_minus, _) = held.eval(minus.view()).ok()?;
            let (product, back, back_energy, product_energy) = if gap_plus < gap_minus {
                (minus, gap_plus, energy_plus, energy_minus)
            } else {
                (plus, gap_minus, energy_minus, energy_plus)
            };
            let turned = axis_dot(axis, axis_of(&product, dumbbell));
            let other = hausdorff(&product, reactant, cell);
            let (saddle_now, _) = held
                .eval(saddle.view())
                .unwrap_or((f64::NAN, Array1::zeros(0)));
            eprintln!(
                "trial {trial} back {back:.3} dE {:.3} product {other:.3} dE {:.3} saddle {:.3}",
                back_energy - reactant_energy,
                product_energy - reactant_energy,
                saddle_now - reactant_energy
            );
            let _ = std::io::stderr().flush();
            if back > 1.6
                || (back_energy - reactant_energy).abs() > 0.08
                || product_energy < reactant_energy - 1.0
                || other < 0.45
            {
                return None;
            }
            let after = rgsaddle::pair_separation(product.view(), cell, fixed, &dumbbell).ok()?;
            let direction = rgsaddle::classify_separation(before, after, 0.08).ok()?;
            let (saddle_energy, _) = held.eval(saddle.view()).ok()?;
            if !saddle_energy.is_finite() || !reactant_energy.is_finite() {
                return None;
            }
            Some(Escape {
                product,
                barrier: saddle_energy - reactant_energy,
                direction: direction.as_str().to_string(),
                axis_dot: turned,
            })
        })
        .collect();
    let mut ranked: Vec<Escape> = found.into_iter().flatten().collect();
    ranked.sort_by(|left, right| left.barrier.partial_cmp(&right.barrier).unwrap());
    let mut kept = Vec::new();
    for direction in ["toward", "away", "tangential"] {
        let mut taken = 0usize;
        for escape in &ranked {
            if escape.direction != direction {
                continue;
            }
            if kept
                .iter()
                .any(|prior: &Escape| hausdorff(&prior.product, &escape.product, cell) < 0.4)
            {
                continue;
            }
            kept.push(escape.clone());
            taken += 1;
            if taken == 1 {
                break;
            }
        }
    }
    kept
}

struct ShellRow {
    repeats: usize,
    temperature: f64,
    radius: f64,
    count: usize,
    converged: bool,
    recombined: bool,
    gap: f64,
    separation: f64,
    key: (i32, i32, i32),
    steps: usize,
    max_force: f64,
    vacancies: usize,
    interstitials: usize,
}

impl ShellRow {
    fn scanned(
        repeats: usize,
        temperature: f64,
        a: f64,
        (radius, vector, count): (f64, [f64; 3], usize),
        relaxed: &Result<PairState, SaddleError>,
    ) -> Self {
        let mut row = Self {
            repeats,
            temperature,
            radius,
            count,
            converged: false,
            recombined: false,
            gap: f64::NAN,
            separation: f64::NAN,
            key: miller(vector, a),
            steps: 0,
            max_force: f64::NAN,
            vacancies: 0,
            interstitials: 0,
        };
        if let Ok(pair) = relaxed {
            row.converged = true;
            row.recombined = pair.recombined;
            row.gap = pair.gap;
            row.separation = pair.separation;
            row.steps = pair.steps;
            row.max_force = pair.max_force;
            row.vacancies = pair.vacancies;
            row.interstitials = pair.interstitials;
        }
        row
    }
}

const SHELL_HEADER: &str = "cell_repeats,temperature_K,radius_A,sites,recombined,gap_A,separation_A,miller,converged,relax_steps,max_force_eV_per_A,vacancies,interstitials";

fn shell_line(row: &ShellRow) -> String {
    format!(
        "{},{},{:.4},{},{},{:.4},{:.4},{} {} {},{},{},{:.3e},{},{}",
        row.repeats,
        row.temperature,
        row.radius,
        row.count,
        row.recombined as u8,
        row.gap,
        row.separation,
        row.key.0,
        row.key.1,
        row.key.2,
        row.converged as u8,
        row.steps,
        row.max_force,
        row.vacancies,
        row.interstitials,
    )
}

struct Prepared {
    key: (i32, i32, i32),
    radius: f64,
    escapes: Vec<Escape>,
}

struct RateJob {
    repeats: usize,
    temperature: f64,
    a: f64,
    key: (i32, i32, i32),
    radius: f64,
    escapes: Vec<Escape>,
    a_ref: f64,
}

/// FIRE 2.0 to the default 1e-3 eV/Å largest atom force, with the 0.08 Å
/// per-atom cap and time step the pair searches use.
fn pair_fire() -> rgsaddle::RelaxConfig {
    rgsaddle::RelaxConfig::default()
        .with_max_atom_move(0.08)
        .with_initial_step(0.08)
}

/// ODE12r, the steepest-descent flow, to 1e-3 eV/Å under the same cap.
///
/// A placement's fate is the basin the flow carries it into. From the
/// `<200>` placements perpendicular to the dumbbell, FIRE and L-BFGS each
/// end in recombination on some lattices and in a bound pair on others,
/// both converged, so neither defines the basin.
fn pair_relax() -> rgsaddle::RelaxConfig {
    pair_fire().with_method(Method::Ode12r)
}

/// Steepest descent under a 0.02 Å per-atom cap: a second, finer
/// integration of the same flow, to check [`pair_relax`].
fn pair_flow_check() -> rgsaddle::RelaxConfig {
    pair_fire()
        .with_method(Method::Steepest)
        .with_max_atom_move(0.02)
        .with_initial_step(0.02)
}

/// A placement relaxed to its force tolerance.
///
/// `recombined` is the site occupancy of the relaxed configuration: with a
/// vacancy, every lattice site singly occupied. `gap` and `separation`
/// are the dumbbell's atoms and its centroid to the vacated site, kept as
/// descriptors only.
struct PairState {
    positions: Array1<f64>,
    vacancy: [f64; 3],
    dumbbell: [usize; 2],
    cell: [[f64; 3]; 3],
    gap: f64,
    separation: f64,
    recombined: bool,
    steps: usize,
    max_force: f64,
    vacancies: usize,
    interstitials: usize,
}

fn relax_pair(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    vacancy: bool,
    relax: &rgsaddle::RelaxConfig,
) -> Result<PairState, SaddleError> {
    relax_oriented(copper, repeats, a, shell, [1.0, 0.0, 0.0], vacancy, relax)
}

/// [`relax_pair`] with the dumbbell along `axis`.
fn relax_oriented(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
    vacancy: bool,
    relax: &rgsaddle::RelaxConfig,
) -> Result<PairState, SaddleError> {
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let built = build_oriented(repeats, a, shell, axis, vacancy);
    let done =
        rgsaddle::relax_to_tolerance(&FastHeld { copper, cell }, built.positions, cell, relax)?;
    let positions = done.positions;
    let (sites, _) = fcc(repeats, a);
    let occupancy = rgsaddle::site_occupancy(sites.view(), positions.view(), cell)?;
    let gap = dist(
        point_of(&positions, built.dumbbell[0]),
        point_of(&positions, built.dumbbell[1]),
    );
    let separation = dist(centroid(&positions, built.dumbbell), built.vacancy);
    Ok(PairState {
        positions,
        vacancy: built.vacancy,
        dumbbell: built.dumbbell,
        cell,
        gap,
        separation,
        recombined: vacancy && occupancy.is_perfect(),
        steps: done.steps,
        max_force: done.max_force,
        vacancies: occupancy.vacancy_count(),
        interstitials: occupancy.interstitial_count(),
    })
}

/// Whether a relaxed vacancy–interstitial cell has recombined: every site of
/// the perfect lattice the cell holds is singly occupied. The lattice is
/// the 4 n³ fcc crystal with this atom count, mapped onto `cell`.
fn recombined_in(positions: &Array1<f64>, cell: [[f64; 3]; 3]) -> bool {
    let atoms = positions.len() / 3;
    let repeats = (atoms as f64 / 4.0).cbrt().round() as usize;
    assert_eq!(4 * repeats.pow(3), atoms, "not a 4 n^3 fcc cell");
    let (cubic, _) = fcc(repeats, 1.0);
    let mut sites = Array1::zeros(cubic.len());
    for site in 0..atoms {
        for c in 0..3 {
            sites[3 * site + c] = (0..3)
                .map(|k| cubic[3 * site + k] / repeats as f64 * cell[k][c])
                .sum::<f64>();
        }
    }
    rgsaddle::site_occupancy(sites.view(), positions.view(), cell)
        .expect("site occupancy")
        .is_perfect()
}

/// A placement followed down the flow, for a campaign that cannot go on
/// without its state; a relaxation that does not converge stops the run.
fn placed(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
    vacancy: bool,
) -> PairState {
    relax_oriented(copper, repeats, a, shell, axis, vacancy, &pair_relax()).unwrap_or_else(
        |error| panic!("placement {shell:?} axis {axis:?} on {repeats}^3 at a = {a}: {error}"),
    )
}

fn scale_positions(positions: &Array1<f64>, factor: f64) -> Array1<f64> {
    positions.mapv(|value| value * factor)
}

fn local_atoms(positions: &Array1<f64>, center: [f64; 3], cell: [[f64; 3]; 3]) -> Vec<usize> {
    (0..positions.len() / 3)
        .filter(|atom| mic_distance(point_of(positions, *atom), center, cell) < 4.8)
        .collect()
}

/// Table 1 of Mwungeli, Liu and Demkowicz, Phys. Rev. Materials 9, 023601
/// (2025), read from the accepted manuscript. Volumes are Ω, α-method
/// then β-bar method. The manuscript prose says 5312 and 6924 atoms
/// where this table says 5324 and 6912; 4 n³ is 5324 and 6912.
fn table_i(temperature: f64) -> &'static str {
    match temperature as i32 {
        300 => "8788 atoms: 254±21 and 286±28",
        400 => "8788 atoms: 212±12 and 210±20",
        500 => {
            "4000: 309±18 and 287±34; 5324: 169±17 and 180±20; 6912: 188±11 and 191±19; 8788: 185±23 and 161±15; 16384: 202±13 and 206±22"
        }
        600 => {
            "4000: 218±13 and 238±22; 5324: 170±8 and 177±15; 6912: 198±10 and 203±20; 8788: 186±14 and 191±18; 16384: 183±9 and 194±18"
        }
        700 => "8788 atoms: 203±10 and 210±21",
        _ => "this temperature is not in Table 1",
    }
}

fn cu_config(
    temperatures: &[f64],
    samples: usize,
    target: f64,
    bulk: [[f64; 3]; 3],
    harmonic: Vec<usize>,
    seed: u64,
) -> rgsaddle::DefectPairConfig {
    let search = MinModeConfig::default()
        .with_force_tol(2.0e-2)
        .with_max_move(0.1)
        .with_dr(1.0e-3)
        .with_max_rotations(8);
    rgsaddle::DefectPairConfig::default()
        .with_dimer(
            SolidDimerConfig::frozen(search)
                .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1)),
        )
        .with_band(
            rgsaddle::BandConfig::default()
                .with_force_tol(0.01)
                .with_max_move(0.06)
                .with_time_step(0.08)
                .with_method(Method::fire(FireKind::V2))
                .with_spring(rgsaddle::SpringKind::Uniform { k: 1.0 })
                .with_climbing(rgsaddle::CiConfig::always())
                .with_min_mode(None)
                .with_autoneb(None)
                .with_zoom(None)
                .with_quickmin(false),
        )
        .with_images(9)
        .with_max_dimer_steps(100)
        .with_max_band_steps(120)
        .with_relax(pair_relax())
        .with_seed_fraction(0.45)
        .with_push(0.3)
        .with_connect_rmsd(1.5)
        .with_direction_tol(0.08)
        .with_pafi(
            rgsaddle::PafiConfig::default()
                .with_thermalize(40)
                .with_stride(16)
                .with_samples(samples)
                .with_blocks(6)
                .with_time_step(0.01)
                .with_mobility(0.2)
                .with_seed(seed)
                .with_fix_com(true)
                .with_retain(4)
                .with_target_error(target)
                .with_max_rounds(4),
        )
        .with_shooting(
            rgsaddle::ShootingConfig::default()
                .with_friction(2.0)
                .with_time_step(0.03)
                .with_steps(80)
                .with_record_every(20)
                .with_shots(32)
                .with_blocks(4)
                .with_seed(seed.wrapping_add(9))
                .with_plateau_record(2)
                .with_mass(63.546),
        )
        .with_temperatures(temperatures.to_vec())
        .with_boltzmann(rgsaddle::BOLTZMANN_EV_PER_K)
        .with_mass(63.546)
        .with_angular_scale(rgsaddle::angular_scale_rad_s())
        .with_harmonic_step(1.0e-3)
        .with_harmonic_atoms(harmonic)
        .with_planes(11)
        .with_zero_t_tol(0.02)
        .with_bulk_stress(bulk)
}

struct Campaign {
    catalogue: rgsaddle::RateCatalogue,
    shells: Vec<ShellRow>,
    lattice: Vec<(f64, f64)>,
    notes: Vec<String>,
}

fn write_campaign(campaign: &Campaign) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut csv = File::create(root.join("defect_pair_cu.csv")).expect("csv");
    writeln!(
        csv,
        "shell,direction,hop,temperature_K,cell_repeats,n_atoms,separation_before_A,separation_after_A,outcome,connected,index1,band_converged,band_force_eV_per_A,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,dF_harmonic_eV,nu_Hz,vineyard_Hz,kappa,kappa_error,harmonic_rate_Hz,rate_Hz,pafi_samples,pafi_evaluations"
    )
    .unwrap();
    for entry in &campaign.catalogue.entries {
        writeln!(
            csv,
            "{},{},{},{},{},{},{:.6},{:.6},{},{},{},{},{:.6e},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            entry.shell,
            entry.direction.as_str(),
            entry.hop,
            entry.temperature,
            entry.cell_repeats,
            entry.n_atoms,
            entry.separation_before,
            entry.separation_after,
            entry.outcome.as_str(),
            entry.connected as u8,
            entry.index1 as u8,
            entry.band_converged as u8,
            entry.band_force,
            num_or(&Some(entry.d_e0)),
            num_or(&entry.zero_t),
            num_or(&entry.d_f),
            num_or(&entry.d_f_error),
            num_or(&entry.d_f_harmonic),
            num_or(&entry.nu),
            num_or(&entry.vineyard),
            num_or(&entry.kappa),
            num_or(&entry.kappa_error),
            num_or(&entry.harmonic_rate),
            num_or(&entry.rate),
            entry.pafi_samples,
            entry.pafi_evaluations,
        )
        .unwrap();
    }
    std::fs::write(
        root.join("defect_pair_cu.json"),
        campaign.catalogue.to_json(),
    )
    .unwrap();
    let mut shells = File::create(root.join("defect_pair_shells.csv")).expect("shells");
    writeln!(shells, "{SHELL_HEADER}").unwrap();
    for row in &campaign.shells {
        writeln!(shells, "{}", shell_line(row)).unwrap();
    }
    let mut lattice = File::create(root.join("defect_pair_lattice.csv")).expect("lattice");
    writeln!(lattice, "temperature_K,a_A,npt_repeats").unwrap();
    for (temperature, a) in &campaign.lattice {
        writeln!(lattice, "{temperature},{a:.6},6").unwrap();
    }
    std::fs::write(
        root.join("defect_pair_cu_notes.txt"),
        campaign.notes.join("\n"),
    )
    .unwrap();
}

fn rate_or_harmonic(entry: &rgsaddle::RateEntry) -> Option<f64> {
    if let Some(rate) = entry.rate.filter(|value| value.is_finite() && *value > 0.0) {
        return Some(rate);
    }
    let barrier = entry.d_f?;
    let prefactor = entry.nu.or(entry.vineyard)?;
    let thermal = entry.temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    if !(thermal.is_finite() && thermal > 0.0 && prefactor.is_finite() && prefactor > 0.0) {
        return entry
            .harmonic_rate
            .filter(|value| value.is_finite() && *value > 0.0);
    }
    let kappa = entry
        .kappa
        .filter(|value| (0.0..=1.0).contains(value) && entry.kappa_error.unwrap_or(1.0) < 0.5)
        .unwrap_or(1.0);
    let exponent = -barrier / thermal;
    if exponent < -700.0 {
        return Some(0.0);
    }
    Some(kappa * prefactor * exponent.exp())
}

fn viv_lines(campaign: &Campaign) -> Vec<String> {
    let mut lines = vec![
        "Table 1 of Mwungeli, Liu and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), from the accepted manuscript. α then β-bar, in Ω.".to_string(),
    ];
    for temperature in [300.0, 400.0, 500.0, 600.0, 700.0] {
        lines.push(format!("  {temperature:.0} K {}", table_i(temperature)));
    }
    lines.push(
        "V_iv estimate: spontaneous sites, plus each stable shell's site count times k_toward / (k_toward + k_away + k_tangential). A second column weights each toward event by (sites / n_toward) * k / k_mig, with k_mig the isolated-dumbbell translation rate.".to_string(),
    );
    let mut sizes: Vec<u32> = campaign
        .catalogue
        .entries
        .iter()
        .map(|entry| entry.cell_repeats)
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    let mut temps: Vec<i64> = campaign
        .catalogue
        .entries
        .iter()
        .map(|entry| entry.temperature.round() as i64)
        .collect();
    temps.sort_unstable();
    temps.dedup();
    let mut viv = String::from(
        "cell_repeats,temperature_K,spontaneous_sites,capture_volume_Omega,rate_volume_Omega,k_mig_Hz,table_1\n",
    );
    for repeats in sizes {
        for temperature in &temps {
            let shells: Vec<&ShellRow> = campaign
                .shells
                .iter()
                .filter(|row| {
                    row.repeats == repeats as usize
                        && row.temperature.round() as i64 == *temperature
                })
                .collect();
            if shells.is_empty() {
                continue;
            }
            let spontaneous: f64 = shells
                .iter()
                .filter(|row| row.recombined)
                .map(|row| row.count as f64)
                .sum();
            let entries: Vec<&rgsaddle::RateEntry> = campaign
                .catalogue
                .entries
                .iter()
                .filter(|entry| {
                    entry.cell_repeats == repeats
                        && entry.temperature.round() as i64 == *temperature
                        && entry.shell != "isolated"
                })
                .collect();
            let isolated: Vec<&rgsaddle::RateEntry> = campaign
                .catalogue
                .entries
                .iter()
                .filter(|entry| {
                    entry.cell_repeats == repeats
                        && entry.temperature.round() as i64 == *temperature
                        && entry.shell == "isolated"
                        && entry.hop.starts_with("translate")
                })
                .collect();
            let k_mig = {
                let values: Vec<f64> = isolated
                    .iter()
                    .filter_map(|entry| rate_or_harmonic(entry))
                    .collect();
                if values.is_empty() {
                    None
                } else {
                    Some(values.iter().sum::<f64>() / values.len() as f64)
                }
            };
            let mut capture = spontaneous;
            let mut weighted = spontaneous;
            let mut shells_used = 0usize;
            let radii: Vec<String> = entries.iter().map(|entry| entry.shell.clone()).collect();
            let mut unique = radii.clone();
            unique.sort();
            unique.dedup();
            for shell in unique {
                let group: Vec<&&rgsaddle::RateEntry> = entries
                    .iter()
                    .filter(|entry| entry.shell == shell)
                    .collect();
                let sum_dir = |name: &str| -> f64 {
                    group
                        .iter()
                        .filter(|entry| entry.direction.as_str() == name)
                        .filter_map(|entry| rate_or_harmonic(entry))
                        .sum()
                };
                let toward = sum_dir("toward");
                let away = sum_dir("away");
                let tangential = sum_dir("tangential");
                let denom = toward + away + tangential;
                let count = shells
                    .iter()
                    .find(|row| format!("{:.3}", row.radius) == shell)
                    .map(|row| row.count as f64)
                    .unwrap_or(0.0);
                if denom > 0.0 && count > 0.0 {
                    capture += count * toward / denom;
                    shells_used += 1;
                }
                let n_toward = group
                    .iter()
                    .filter(|entry| {
                        entry.direction.as_str() == "toward" && rate_or_harmonic(entry).is_some()
                    })
                    .count();
                if let Some(k_mig) = k_mig
                    && n_toward > 0
                    && count > 0.0
                {
                    for entry in group
                        .iter()
                        .filter(|entry| entry.direction.as_str() == "toward")
                    {
                        if let Some(rate) = rate_or_harmonic(entry) {
                            weighted += count / n_toward as f64 * rate / k_mig;
                        }
                    }
                }
            }
            let line = format!(
                "{repeats},{temperature},{spontaneous:.1},{capture:.1},{weighted:.1},{},{} (shells with rates {shells_used})",
                k_mig
                    .map(|value| format!("{value:.6e}"))
                    .unwrap_or_default(),
                table_i(*temperature as f64)
            );
            lines.push(line.clone());
            viv.push_str(&line);
            viv.push('\n');
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    std::fs::write(root.join("defect_pair_viv.csv"), viv).unwrap();
    lines
}

fn asymmetry_lines(entries: &[rgsaddle::RateEntry]) -> Vec<String> {
    let mut lines = Vec::new();
    let mut keys: Vec<(u32, String, i64)> = entries
        .iter()
        .filter(|entry| entry.shell != "isolated")
        .map(|entry| {
            (
                entry.cell_repeats,
                entry.shell.clone(),
                entry.temperature.round() as i64,
            )
        })
        .collect();
    keys.sort();
    keys.dedup();
    let mut resolved = 0usize;
    let mut compared = 0usize;
    for (repeats, shell, temperature) in keys {
        let mean = |direction: &str| -> Option<(f64, f64, f64)> {
            let rows: Vec<&rgsaddle::RateEntry> = entries
                .iter()
                .filter(|entry| {
                    entry.cell_repeats == repeats
                        && entry.shell == shell
                        && entry.temperature.round() as i64 == temperature
                        && entry.direction.as_str() == direction
                        && entry.d_f.is_some()
                })
                .collect();
            if rows.is_empty() {
                return None;
            }
            let barrier =
                rows.iter().filter_map(|entry| entry.d_f).sum::<f64>() / rows.len() as f64;
            let error = rows
                .iter()
                .filter_map(|entry| entry.d_f_error)
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                / rows.len() as f64;
            let static_barrier =
                rows.iter().map(|entry| entry.d_e0).sum::<f64>() / rows.len() as f64;
            Some((barrier, error, static_barrier))
        };
        let (Some((toward, toward_err, toward_e)), Some((away, away_err, away_e))) =
            (mean("toward"), mean("away"))
        else {
            continue;
        };
        compared += 1;
        let difference = toward - away;
        let error = (toward_err * toward_err + away_err * away_err).sqrt();
        let significant = difference.abs() > 2.0 * error && difference.abs() > 0.005;
        if significant {
            resolved += 1;
        }
        lines.push(format!(
            "repeats {repeats} shell {shell} A T {temperature} K toward-away dF {difference:.4} ± {error:.4} eV (dE0 {toward_e:.4} vs {away_e:.4})"
        ));
    }
    if compared == 0 {
        lines.push(
            "No shell has a converged PAFI barrier in both the toward and the away direction, so the asymmetry is not measured."
                .into(),
        );
    } else if resolved == 0 {
        lines.push(format!(
            "The toward/away free-energy difference is not resolved in any of the {compared} shells that have both barriers. The asymmetry is not there at the 5 meV error bar."
        ));
    } else {
        lines.push(format!(
            "A toward/away free-energy difference larger than twice its error, and larger than 5 meV, appears in {resolved} of {compared} shells."
        ));
    }
    let mut isolated: Vec<(u32, i64, f64, f64)> = Vec::new();
    for entry in entries
        .iter()
        .filter(|entry| entry.shell == "isolated" && entry.d_f.is_some())
    {
        isolated.push((
            entry.cell_repeats,
            entry.temperature.round() as i64,
            entry.d_e0,
            entry.d_f.unwrap_or(0.0),
        ));
    }
    if isolated.is_empty() {
        lines.push("The isolated dumbbell produced no PAFI barrier.".into());
    }
    lines
}

#[test]
#[ignore = "release Mishin Cu defect-pair catalogue; writes benchmarks/defect_pair_cu.csv"]
fn defect_pair_cu_catalogue() {
    let started = Instant::now();
    let explicit = std::env::var("RGSADDLE_REPEATS").ok();
    let mut sizes: Vec<usize> = if explicit.is_some() {
        env_csv("RGSADDLE_REPEATS", "8,10")
            .iter()
            .map(|item| item.parse().expect("repeats"))
            .collect()
    } else {
        vec![8, 10]
    };
    let temperatures: Vec<f64> = env_csv("RGSADDLE_TEMPS", "300,400,500,600,700")
        .iter()
        .map(|item| item.parse().expect("temperature"))
        .collect();
    let max_shells = env_usize("RGSADDLE_MAX_SHELLS", 0);
    let tries = env_usize("RGSADDLE_TRIES", 8);
    let samples = env_usize("RGSADDLE_SAMPLES", 32);
    let target = env_usize("RGSADDLE_TARGET_MEV", 5) as f64 * 1.0e-3;
    let copper = FastCopper::new();
    let relax = pair_relax();
    let mut lattice = Vec::new();
    for temperature in &temperatures {
        let a = npt_lattice(&copper, *temperature);
        eprintln!("a({temperature} K) = {a:.5} A");
        let _ = std::io::stderr().flush();
        assert!(
            (3.50..3.85).contains(&a),
            "NPT lattice parameter {a} at {temperature} K is not a copper lattice"
        );
        lattice.push((*temperature, a));
    }
    let reference = lattice
        .iter()
        .min_by(|left, right| {
            (left.0 - 500.0)
                .abs()
                .partial_cmp(&(right.0 - 500.0).abs())
                .unwrap()
        })
        .copied()
        .unwrap();
    let mut campaign = Campaign {
        catalogue: rgsaddle::RateCatalogue {
            entries: Vec::new(),
        },
        shells: Vec::new(),
        lattice: lattice.clone(),
        notes: vec![format!(
            "NPT a(T) on a 6x6x6 cell, Berendsen zero pressure, second half of 8 ps. Reference lattice for hop discovery is {:.5} A at {} K. Band force tolerance 0.01 eV/A, FIRE, climbing armed only after a plain relaxation. Finite-T PAFI runs only when the zero-temperature integral matches dE0 within 0.02 eV. Target sampling error {target:.4} eV.",
            reference.1, reference.0
        )],
    };
    let rate_sizes = sizes.clone();
    let mut jobs: Vec<RateJob> = Vec::new();
    if explicit.is_none() {
        sizes.extend([12, 14]);
    }
    for &repeats in &sizes {
        let rate_this =
            rate_sizes.contains(&repeats) && std::env::var("RGSADDLE_SCAN_ONLY").is_err();
        let shell_cap = if !rate_this {
            0
        } else if max_shells > 0 {
            max_shells
        } else if repeats >= 10 {
            1
        } else {
            2
        };
        campaign.notes.push(format!(
            "repeats {repeats}: recombination scan{}, hop search {}",
            if rate_this { " and rates" } else { " only" },
            if shell_cap == 0 && !rate_this {
                "skipped".to_string()
            } else if shell_cap == 0 {
                "every stable shell".to_string()
            } else {
                format!("first {shell_cap} stable shells")
            }
        ));
        let mut prepared: Vec<Prepared> = Vec::new();
        if rate_this {
            let shells = shell_table(reference.1, repeats);
            let mut stable = 0usize;
            for (radius, vector, _count) in shells {
                let pair = match relax_pair(&copper, repeats, reference.1, vector, true, &relax) {
                    Ok(pair) => pair,
                    Err(error) => {
                        campaign.notes.push(format!(
                            "repeats {repeats} shell {radius:.3} A skipped for hop search: {error}"
                        ));
                        continue;
                    }
                };
                if pair.recombined {
                    continue;
                }
                let PairState {
                    positions,
                    vacancy,
                    dumbbell,
                    cell,
                    ..
                } = pair;
                if shell_cap != 0 && stable >= shell_cap {
                    break;
                }
                stable += 1;
                let escapes = enumerate_escapes(
                    &copper,
                    &positions,
                    cell,
                    vacancy,
                    dumbbell,
                    tries,
                    (repeats as u64) * 1000 + stable as u64,
                );
                eprintln!(
                    "discovered repeats {repeats} shell {radius:.3} A escapes {} in {:.0}s",
                    escapes.len(),
                    started.elapsed().as_secs_f64()
                );
                let _ = std::io::stderr().flush();
                for escape in &escapes {
                    eprintln!(
                        "  {} barrier {:.4} eV axis {:.2}",
                        escape.direction, escape.barrier, escape.axis_dot
                    );
                }
                prepared.push(Prepared {
                    key: miller(vector, reference.1),
                    radius,
                    escapes,
                });
            }
            if repeats == 8 || rate_sizes.len() == 1 {
                let isolated = build_pair(repeats, reference.1, [reference.1, 0.0, 0.0], false);
                let length = repeats as f64 * reference.1;
                let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
                let positions = rgsaddle::relax_to_tolerance(
                    &FastHeld {
                        copper: &copper,
                        cell,
                    },
                    isolated.positions,
                    cell,
                    &relax,
                )
                .expect("isolated dumbbell relaxation")
                .positions;
                let center = centroid(&positions, isolated.dumbbell);
                let spectator = [center[0] + 6.0, center[1], center[2]];
                let escapes = enumerate_escapes(
                    &copper,
                    &positions,
                    cell,
                    spectator,
                    isolated.dumbbell,
                    tries,
                    9_000 + repeats as u64,
                );
                eprintln!(
                    "isolated repeats {repeats} escapes {} in {:.0}s",
                    escapes.len(),
                    started.elapsed().as_secs_f64()
                );
                prepared.push(Prepared {
                    key: (0, 0, 0),
                    radius: 0.0,
                    escapes,
                });
            }
        }
        for &(temperature, a) in &lattice {
            let shells = axis_orbits(a, repeats);
            for (radius, vector, count) in &shells {
                let relaxed = relax_pair(&copper, repeats, a, *vector, true, &relax);
                let row = ShellRow::scanned(
                    repeats,
                    temperature,
                    a,
                    (*radius, *vector, *count),
                    &relaxed,
                );
                if let Err(error) = &relaxed {
                    campaign.notes.push(format!(
                        "repeats {repeats} T {temperature} shell {radius:.3} A not counted: {error}"
                    ));
                } else if row.recombined {
                    campaign.notes.push(format!(
                        "repeats {repeats} T {temperature} shell {radius:.4} A recombined, {count} sites, gap {:.4} separation {:.4}",
                        row.gap, row.separation
                    ));
                }
                campaign.shells.push(row);
            }
            if !rate_this {
                continue;
            }
            for item in &prepared {
                if item.escapes.is_empty() {
                    continue;
                }
                if item.radius == 0.0 && ![300, 500, 700].contains(&(temperature as i32)) {
                    continue;
                }
                let shell_recombines = item.radius != 0.0
                    && campaign.shells.iter().any(|row| {
                        row.repeats == repeats
                            && (row.temperature - temperature).abs() < 1.0
                            && row.key == item.key
                            && row.recombined
                    });
                if shell_recombines {
                    continue;
                }
                jobs.push(RateJob {
                    repeats,
                    temperature,
                    a,
                    key: item.key,
                    radius: item.radius,
                    escapes: item.escapes.clone(),
                    a_ref: reference.1,
                });
            }
        }
        write_campaign(&campaign);
    }
    eprintln!("{} rate jobs after discovery", jobs.len());
    let _ = std::io::stderr().flush();
    let store = Mutex::new(Vec::new());
    jobs.par_iter().for_each(|job| {
        let outcome = run_rate_job(&copper, job, samples, target);
        match outcome {
            Ok(entries) => {
                for entry in &entries {
                    eprintln!(
                        "  repeats {} {} {} {} outcome {} band {:.4} dE0 {:.4} zeroT {} dF {} err {} calls {}",
                        job.repeats,
                        entry.shell,
                        entry.hop,
                        entry.direction.as_str(),
                        entry.outcome.as_str(),
                        entry.band_force,
                        entry.d_e0,
                        num_or(&entry.zero_t),
                        num_or(&entry.d_f),
                        num_or(&entry.d_f_error),
                        entry.pafi_evaluations
                    );
                }
                let _ = std::io::stderr().flush();
                store.lock().expect("rates").extend(entries);
            }
            Err(error) => eprintln!("rate job failed: {error}"),
        }
    });
    campaign
        .catalogue
        .entries
        .extend(store.into_inner().expect("rates"));
    campaign
        .notes
        .extend(asymmetry_lines(&campaign.catalogue.entries));
    campaign.notes.extend(viv_lines(&campaign));
    let evaluations: usize = campaign
        .catalogue
        .entries
        .iter()
        .map(|entry| entry.pafi_evaluations)
        .sum();
    campaign.notes.push(format!(
        "PAFI force evaluations {evaluations}. Wall time {:.1} min. Sizes requested for rates: {}. Sizes scanned: {}.",
        started.elapsed().as_secs_f64() / 60.0,
        rate_sizes.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(","),
        sizes.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(",")
    ));
    write_campaign(&campaign);
    eprintln!("{}", campaign.notes.join("\n"));
}

/// Largest gap or separation difference, in Å, allowed between the end
/// states of one placement under two optimizers at 1e-3 eV/Å.
const OPTIMIZER_AGREEMENT_A: f64 = 0.02;

/// End states of one placement under two optimizers, compared.
fn optimizer_disagreement(first: &PairState, second: &PairState) -> Option<String> {
    let gap = (first.gap - second.gap).abs();
    let separation = (first.separation - second.separation).abs();
    if first.recombined != second.recombined
        || !(gap <= OPTIMIZER_AGREEMENT_A && separation <= OPTIMIZER_AGREEMENT_A)
    {
        Some(format!(
            "recombined {} gap {:.4} separation {:.4} against recombined {} gap {:.4} separation {:.4}",
            first.recombined,
            first.gap,
            first.separation,
            second.recombined,
            second.gap,
            second.separation
        ))
    } else {
        None
    }
}

/// The 3.63 Å `<200>` and the 4.44 Å `<211>` placements on the 300 K
/// lattice, in a 4x4x4 cell, with the dumbbell along x. Sixty FIRE steps
/// stop the first at a 2.36 Å gap under a 1 eV/Å force, which a gap window
/// reads as bound while sixty L-BFGS steps reach recombination. Followed
/// to 1e-3 eV/Å by ODE12r and by steepest descent, the first recombines
/// and the second stays a bound pair.
#[test]
fn pair_recombination_does_not_depend_on_the_optimizer() {
    let copper = FastCopper::new();
    let (repeats, a) = (4usize, measured_lattice(300.0));
    let optimizers = [("ODE12r", pair_relax()), ("steepest", pair_flow_check())];
    for (shell, recombines) in [([a, 0.0, 0.0], true), ([a, 0.5 * a, 0.5 * a], false)] {
        let truncated = relax_pair(
            &copper,
            repeats,
            a,
            shell,
            true,
            &pair_fire().with_max_steps(60),
        );
        assert!(
            truncated.is_err(),
            "60 FIRE steps accepted as converged for {shell:?}"
        );
        let states: Vec<(&str, PairState)> = optimizers
            .iter()
            .map(|(name, relax)| {
                let state = relax_pair(&copper, repeats, a, shell, true, relax)
                    .unwrap_or_else(|error| panic!("{name} {shell:?}: {error}"));
                assert!(state.max_force < rgsaddle::DEFAULT_FORCE_TOL);
                (*name, state)
            })
            .collect();
        for (name, state) in &states[1..] {
            if let Some(text) = optimizer_disagreement(&states[0].1, state) {
                panic!("{} and {name} disagree at {shell:?}: {text}", states[0].0);
            }
        }
        assert_eq!(states[0].1.recombined, recombines, "{shell:?}");
    }
}

#[test]
#[ignore = "recombination scan followed to 1e-3 eV/A by ODE12r and steepest descent; writes benchmarks/defect_pair_shells.csv"]
fn defect_pair_recombination_scan() {
    let started = Instant::now();
    let sizes: Vec<usize> = env_csv("RGSADDLE_REPEATS", "8,10")
        .iter()
        .map(|item| item.parse().expect("repeats"))
        .collect();
    let temperatures: Vec<f64> = env_csv("RGSADDLE_TEMPS", "300,400,500,600,700")
        .iter()
        .map(|item| item.parse().expect("temperature"))
        .collect();
    let copper = FastCopper::new();
    let flow = pair_relax();
    let check = pair_flow_check();
    let deviation = Mutex::new(Vec::new());
    let mut jobs = Vec::new();
    for &repeats in &sizes {
        for &temperature in &temperatures {
            let a = measured_lattice(temperature);
            for shell in axis_orbits(a, repeats) {
                jobs.push((repeats, temperature, a, shell));
            }
        }
    }
    let results: Vec<(ShellRow, Option<String>, f64)> = jobs
        .par_iter()
        .map(|&(repeats, temperature, a, shell)| {
            let clock = Instant::now();
            let by_flow = relax_pair(&copper, repeats, a, shell.1, true, &flow);
            let by_check = relax_pair(&copper, repeats, a, shell.1, true, &check);
            let disagreement = match (&by_flow, &by_check) {
                (Ok(left), Ok(right)) => {
                    deviation.lock().expect("deviation").push((
                        (left.gap - right.gap).abs(),
                        (left.separation - right.separation).abs(),
                        left.steps,
                        right.steps,
                    ));
                    optimizer_disagreement(left, right)
                }
                (left, right) => Some(format!(
                    "ODE12r {:?}; steepest {:?}",
                    left.as_ref().err().map(ToString::to_string),
                    right.as_ref().err().map(ToString::to_string)
                )),
            };
            let row = ShellRow::scanned(repeats, temperature, a, shell, &by_flow);
            (row, disagreement, clock.elapsed().as_secs_f64())
        })
        .collect();
    let deviation = deviation.into_inner().expect("deviation");
    let largest =
        |pick: fn(&(f64, f64, usize, usize)) -> f64| deviation.iter().map(pick).fold(0.0, f64::max);
    eprintln!(
        "ODE12r against steepest descent over {} placements: largest gap difference {:.2e} A, separation {:.2e} A; ODE12r steps at most {}, steepest at most {}",
        deviation.len(),
        largest(|d| d.0),
        largest(|d| d.1),
        deviation.iter().map(|d| d.2).max().unwrap_or(0),
        deviation.iter().map(|d| d.3).max().unwrap_or(0)
    );
    let mut csv = String::from(SHELL_HEADER);
    csv.push('\n');
    let mut problems = Vec::new();
    for (row, disagreement, seconds) in &results {
        csv.push_str(&shell_line(row));
        csv.push('\n');
        eprintln!(
            "repeats {} T {} shell {:.4} sites {} recombined {} gap {:.4} separation {:.4} steps {} force {:.2e} vacancies {} interstitials {} ({seconds:.1}s)",
            row.repeats,
            row.temperature,
            row.radius,
            row.count,
            row.recombined as u8,
            row.gap,
            row.separation,
            row.steps,
            row.max_force,
            row.vacancies,
            row.interstitials
        );
        if let Some(text) = disagreement {
            problems.push(format!(
                "repeats {} T {} shell {:.4}: {text}",
                row.repeats, row.temperature, row.radius
            ));
        }
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    std::fs::write(root.join("defect_pair_shells.csv"), csv).expect("shells");
    let viv_path = root.join("defect_pair_viv.csv");
    let previous = std::fs::read_to_string(&viv_path).unwrap_or_default();
    let is_site_row = |line: &&str| {
        line.split(',')
            .next()
            .is_some_and(|f| f.parse::<usize>().is_ok())
    };
    let recorded: std::collections::BTreeMap<(usize, i64), String> = previous
        .lines()
        .skip(1)
        .take_while(is_site_row)
        .filter_map(|line| {
            let mut fields = line.splitn(4, ',');
            let repeats = fields.next()?.parse().ok()?;
            let temperature = fields.next()?.parse::<f64>().ok()?.round() as i64;
            Some(((repeats, temperature), fields.nth(1)?.to_string()))
        })
        .collect();
    let mut table = String::from("cell_repeats,temperature_K,spontaneous_sites,table_1_Omega\n");
    for &repeats in &sizes {
        for &temperature in &temperatures {
            let sites: usize = results
                .iter()
                .map(|(row, _, _)| row)
                .filter(|row| {
                    row.repeats == repeats && row.temperature == temperature && row.recombined
                })
                .map(|row| row.count)
                .sum();
            let reference = recorded
                .get(&(repeats, temperature.round() as i64))
                .cloned()
                .unwrap_or_else(|| table_i(temperature).to_string());
            table.push_str(&format!("{repeats},{temperature},{sites},{reference}\n"));
        }
    }
    // The capture rows that defect_pair_viv appends follow the site table.
    for line in previous.lines().skip(1).skip_while(is_site_row) {
        table.push_str(line);
        table.push('\n');
    }
    std::fs::write(&viv_path, table).expect("viv");
    let unconverged = results.iter().filter(|(row, _, _)| !row.converged).count();
    let worst = results
        .iter()
        .map(|(_, _, seconds)| *seconds)
        .fold(0.0, f64::max);
    eprintln!(
        "{} placements, {unconverged} not converged, slowest {worst:.1}s, wall {:.1} min",
        results.len(),
        started.elapsed().as_secs_f64() / 60.0
    );
    assert!(
        problems.is_empty(),
        "ODE12r and steepest descent disagree:\n{}",
        problems.join("\n")
    );
}

const CENSUS_HEADER: &str = "cell_repeats,temperature_K,radius_A,sites,recombined,gap_A,separation_A,miller,converged,relax_steps,max_force_eV_per_A,vacancies,interstitials,lattice_A,check_recombined,check_steps,seconds";

/// Rows of `defect_pair_census.csv` already written, by temperature and
/// orbit representative.
fn census_done(text: &str) -> std::collections::HashSet<(i64, String)> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let temperature = fields.get(1)?.parse::<f64>().ok()?.round() as i64;
            Some((temperature, fields.get(7)?.to_string()))
        })
        .collect()
}

#[test]
#[ignore = "flow census of every axis orbit in the 8788-atom cell on the Table 1 lattices"]
fn defect_pair_census() {
    let started = Instant::now();
    let repeats = 13usize;
    let temperatures: Vec<f64> = env_csv("RGSADDLE_TEMPS", "300,500,700")
        .iter()
        .map(|item| item.parse().expect("temperature"))
        .collect();
    let copper = FastCopper::new();
    let flow = pair_relax();
    let check = pair_flow_check();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_census.csv");
    if !path.exists() {
        std::fs::write(&path, format!("{CENSUS_HEADER}\n")).expect("census");
    }
    let done = census_done(&std::fs::read_to_string(&path).expect("census"));
    let mut jobs = Vec::new();
    for &temperature in &temperatures {
        let a = demkowicz_lattice(temperature);
        for shell in axis_orbits(a, repeats) {
            let (i, j, k) = miller(shell.1, a);
            if !done.contains(&(temperature.round() as i64, format!("{i} {j} {k}"))) {
                jobs.push((temperature, a, shell));
            }
        }
    }
    // The far orbits relax slowest; starting them first keeps the threads busy.
    jobs.sort_by(|left, right| right.2.0.total_cmp(&left.2.0));
    eprintln!(
        "census: {} placements on {} threads",
        jobs.len(),
        rayon::current_num_threads()
    );
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let problems = Mutex::new(Vec::new());
    jobs.par_iter().for_each(|&(temperature, a, shell)| {
        let clock = Instant::now();
        let by_flow = relax_pair(&copper, repeats, a, shell.1, true, &flow);
        let by_check = relax_pair(&copper, repeats, a, shell.1, true, &check);
        let row = ShellRow::scanned(repeats, temperature, a, shell, &by_flow);
        let seconds = clock.elapsed().as_secs_f64();
        let disagreement = match (&by_flow, &by_check) {
            (Ok(left), Ok(right)) => optimizer_disagreement(left, right),
            (left, right) => Some(format!(
                "ODE12r {:?}; steepest {:?}",
                left.as_ref().err().map(ToString::to_string),
                right.as_ref().err().map(ToString::to_string)
            )),
        };
        let (check_recombined, check_steps) = by_check
            .as_ref()
            .map(|pair| (pair.recombined as u8, pair.steps))
            .unwrap_or((0, 0));
        let line = format!(
            "{},{a:.3},{check_recombined},{check_steps},{seconds:.1}",
            shell_line(&row)
        );
        eprintln!(
            "census {line} at {:.1} min",
            started.elapsed().as_secs_f64() / 60.0
        );
        append_checkpoint(&checkpoint, &line);
        if let Some(text) = disagreement {
            problems.lock().expect("problems").push(format!(
                "T {temperature} shell {:.4} {:?}: {text}",
                shell.0, row.key
            ));
        }
    });
    eprintln!(
        "census wall {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
    let problems = problems.into_inner().expect("problems");
    assert!(
        problems.is_empty(),
        "ODE12r and steepest descent disagree:\n{}",
        problems.join("\n")
    );
}

fn run_rate_job(
    copper: &FastCopper,
    job: &RateJob,
    samples: usize,
    target: f64,
) -> Result<Vec<rgsaddle::RateEntry>, String> {
    let factor = job.a / job.a_ref;
    let vector = if job.radius == 0.0 {
        [job.a, 0.0, 0.0]
    } else {
        cart_of(job.key, job.a)
    };
    let vacancy_flag = job.radius != 0.0;
    let pair = relax_pair(
        copper,
        job.repeats,
        job.a,
        vector,
        vacancy_flag,
        &pair_relax(),
    )
    .map_err(|error| error.to_string())?;
    if pair.recombined {
        return Ok(Vec::new());
    }
    let PairState {
        positions,
        vacancy,
        dumbbell,
        cell,
        ..
    } = pair;
    let fixed = if vacancy_flag {
        vacancy
    } else {
        let center = centroid(&positions, dumbbell);
        [center[0] + 6.0, center[1], center[2]]
    };
    let mut hops = Vec::new();
    for (index, escape) in job.escapes.iter().enumerate() {
        let kind = if escape.axis_dot < 0.6 {
            "rotate"
        } else {
            "translate"
        };
        hops.push(rgsaddle::HopTemplate {
            name: format!("{kind}-{}-{index}", escape.direction),
            product: scale_positions(&escape.product, factor),
            saddle: None,
            mode: None,
            orientation: String::new(),
            final_state: String::new(),
        });
    }
    let (perfect, _) = fcc(job.repeats, job.a);
    let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
    let harmonic = local_atoms(&positions, centroid(&positions, dumbbell), cell);
    let shell = if vacancy_flag {
        format!(
            "{:.3}",
            (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
        )
    } else {
        "isolated".to_string()
    };
    eprintln!(
        "rates repeats {} shell {shell} T {} hops {}",
        job.repeats,
        job.temperature,
        hops.len()
    );
    let _ = std::io::stderr().flush();
    let setup = rgsaddle::PairSetup {
        positions,
        cell,
        fixed,
        mobile: vec![dumbbell[0], dumbbell[1]],
        shell,
        cell_repeats: job.repeats as u32,
        hops,
    };
    let config = cu_config(
        &[job.temperature],
        samples,
        target,
        bulk,
        harmonic,
        17 + job.repeats as u64 + job.temperature as u64,
    );
    rgsaddle::defect_pair_rates(&FastHeld { copper, cell }, &setup, &config)
        .map(|found| found.entries)
        .map_err(|error| error.to_string())
}

fn num_or(value: &Option<f64>) -> String {
    match value {
        Some(value) if value.is_finite() => format!("{value:.8e}"),
        _ => String::new(),
    }
}

struct LocalAtom {
    offset: [f64; 3],
    to_saddle: [f64; 3],
    to_product: [f64; 3],
    mode: [f64; 3],
}

struct Mechanism {
    kind: String,
    barrier: f64,
    axis: [f64; 3],
    atoms: Vec<LocalAtom>,
}

fn mat_vec(rotation: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * vector[0] + rotation[0][1] * vector[1] + rotation[0][2] * vector[2],
        rotation[1][0] * vector[0] + rotation[1][1] * vector[1] + rotation[1][2] * vector[2],
        rotation[2][0] * vector[0] + rotation[2][1] * vector[1] + rotation[2][2] * vector[2],
    ]
}

fn unit3(vector: [f64; 3]) -> [f64; 3] {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if norm < 1.0e-12 {
        return vector;
    }
    [vector[0] / norm, vector[1] / norm, vector[2] / norm]
}

fn axis_label(axis: [f64; 3]) -> String {
    let axis = unit3(axis);
    format!(
        "{}{}{}",
        if axis[0].abs() > 0.7 { 1 } else { 0 },
        if axis[1].abs() > 0.7 { 1 } else { 0 },
        if axis[2].abs() > 0.7 { 1 } else { 0 }
    )
}

fn rotations_onto(from: [f64; 3], to: [f64; 3]) -> Vec<[[f64; 3]; 3]> {
    let from = unit3(from);
    let to = unit3(to);
    cubic_rotations()
        .into_iter()
        .filter(|rotation| {
            let image = mat_vec(*rotation, from);
            image[0] * to[0] + image[1] * to[1] + image[2] * to[2] > 0.9
        })
        .collect()
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

#[allow(clippy::too_many_arguments)]
fn capture_mechanism(
    _copper: &FastCopper,
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    dumbbell: [usize; 2],
    saddle: &Array1<f64>,
    product: &Array1<f64>,
    mode: &Array1<f64>,
    barrier: f64,
) -> Mechanism {
    let center = centroid(reactant, dumbbell);
    let axis = axis_of(reactant, dumbbell);
    let mut atoms = Vec::new();
    let n = reactant.len() / 3;
    for atom in 0..n {
        let origin = point_of(reactant, atom);
        if mic_distance(origin, center, cell) > 4.5 && atom != dumbbell[0] && atom != dumbbell[1] {
            continue;
        }
        let delta_s = [
            saddle[3 * atom] - origin[0],
            saddle[3 * atom + 1] - origin[1],
            saddle[3 * atom + 2] - origin[2],
        ];
        let delta_p = [
            product[3 * atom] - origin[0],
            product[3 * atom + 1] - origin[1],
            product[3 * atom + 2] - origin[2],
        ];
        let moved = delta_p[0] * delta_p[0] + delta_p[1] * delta_p[1] + delta_p[2] * delta_p[2];
        let moved_s = delta_s[0] * delta_s[0] + delta_s[1] * delta_s[1] + delta_s[2] * delta_s[2];
        if moved < 0.04 * 0.04
            && moved_s < 0.04 * 0.04
            && atom != dumbbell[0]
            && atom != dumbbell[1]
        {
            continue;
        }
        let offset = [
            origin[0] - center[0],
            origin[1] - center[1],
            origin[2] - center[2],
        ];
        atoms.push(LocalAtom {
            offset,
            to_saddle: delta_s,
            to_product: delta_p,
            mode: [mode[3 * atom], mode[3 * atom + 1], mode[3 * atom + 2]],
        });
    }
    let shift = dist(centroid(product, dumbbell), center);
    Mechanism {
        kind: if shift > 0.6 {
            "translate".into()
        } else {
            "rotate".into()
        },
        barrier,
        axis,
        atoms,
    }
}

fn search_seed(
    copper: &FastCopper,
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    dumbbell: [usize; 2],
    kick: &Array1<f64>,
) -> Option<Mechanism> {
    let mode_norm = kick.dot(kick).sqrt();
    if mode_norm < 1.0e-6 {
        return None;
    }
    let mode = kick / mode_norm;
    let seed = reactant + kick;
    let search = MinModeConfig::default()
        .with_force_tol(0.04)
        .with_max_move(0.1)
        .with_dr(1.0e-3)
        .with_max_rotations(8);
    let dimer_config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1));
    let mut dimer = SolidDimer::new(dimer_config, seed, cell, mode).ok()?;
    let held = FastHeld { copper, cell };
    let report = dimer.run(&held, 80).ok()?;
    if report.status != MinModeStatus::Converged
        || report.rejected
        || report.negative_modes != Some(1)
    {
        return None;
    }
    let saddle = dimer.positions().to_owned();
    let unstable = dimer.mode().to_owned();
    let peak = {
        let mut best = 0.0_f64;
        for atom in 0..saddle.len() / 3 {
            let mut acc = 0.0;
            for c in 0..3 {
                acc += unstable[3 * atom + c] * unstable[3 * atom + c];
            }
            best = best.max(acc.sqrt());
        }
        best
    };
    if peak < 1.0e-8 {
        return None;
    }
    let step = &unstable * (0.45 / peak);
    let plus = &saddle + &step;
    let minus = &saddle - &step;
    let plus = descend(copper, plus, cell, 1.5e-3).ok()?;
    let minus = descend(copper, minus, cell, 1.5e-3).ok()?;
    let gap_plus = hausdorff(&plus, reactant, cell);
    let gap_minus = hausdorff(&minus, reactant, cell);
    let (reactant_energy, _) = held.eval(reactant.view()).ok()?;
    let (product, back, back_pos) = if gap_plus < gap_minus {
        (minus, gap_plus, plus)
    } else {
        (plus, gap_minus, minus)
    };
    let (back_energy, _) = held.eval(back_pos.view()).ok()?;
    let (product_energy, _) = held.eval(product.view()).ok()?;
    let (saddle_energy, _) = held.eval(saddle.view()).ok()?;
    if back > 1.0
        || (back_energy - reactant_energy).abs() > 0.08
        || product_energy < reactant_energy - 0.5
    {
        return None;
    }
    Some(capture_mechanism(
        copper,
        reactant,
        cell,
        dumbbell,
        &saddle,
        &product,
        &unstable,
        saddle_energy - reactant_energy,
    ))
}

fn find_mechanisms(copper: &FastCopper, repeats: usize, a: f64) -> Vec<Mechanism> {
    let built = build_oriented(repeats, a, [a, 0.0, 0.0], [1.0, 0.0, 0.0], false);
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut reactant = built.positions;
    reactant = settle(copper, reactant, cell, 1.0e-3).expect("reactant relaxation");
    let center = centroid(&reactant, built.dumbbell);
    let mut kicks = Vec::new();
    let mut perp = Array1::zeros(reactant.len());
    for (atom, sign) in built.dumbbell.into_iter().zip([-1.0, 1.0]) {
        perp[3 * atom + 1] = sign * 0.5;
    }
    kicks.push(perp);
    let mut gentle = Array1::zeros(reactant.len());
    for (atom, sign) in built.dumbbell.into_iter().zip([-1.0, 1.0]) {
        gentle[3 * atom + 1] = sign * 0.3;
    }
    kicks.push(gentle);
    let n = reactant.len() / 3;
    for atom in 0..n {
        if atom == built.dumbbell[0] || atom == built.dumbbell[1] {
            continue;
        }
        let place = point_of(&reactant, atom);
        if !(2.2..3.4).contains(&mic_distance(place, center, cell)) {
            continue;
        }
        let mut dir = [0.0; 3];
        for c in 0..3 {
            let mut delta = place[c] - center[c];
            delta -= length * (delta / length).round();
            dir[c] = delta;
        }
        let dir = unit3(dir);
        let mut kick = Array1::zeros(reactant.len());
        for site in built.dumbbell {
            for c in 0..3 {
                kick[3 * site + c] = 0.55 * dir[c];
            }
        }
        kicks.push(kick);
        if kicks.len() > 8 {
            break;
        }
    }
    let mut found = Vec::new();
    for kick in kicks {
        if let Some(mechanism) = search_seed(copper, &reactant, cell, built.dumbbell, &kick) {
            eprintln!(
                "isolated {} barrier {:.4} eV atoms {}",
                mechanism.kind,
                mechanism.barrier,
                mechanism.atoms.len()
            );
            let _ = std::io::stderr().flush();
            found.push(mechanism);
        }
    }
    let mut kept = Vec::new();
    for kind in ["translate", "rotate"] {
        if let Some(best) = found
            .iter()
            .filter(|item| item.kind == kind)
            .min_by(|left, right| left.barrier.partial_cmp(&right.barrier).unwrap())
        {
            kept.push(best.clone());
        }
    }
    kept
}

impl Clone for Mechanism {
    fn clone(&self) -> Self {
        Mechanism {
            kind: self.kind.clone(),
            barrier: self.barrier,
            axis: self.axis,
            atoms: self.atoms.clone(),
        }
    }
}

impl Clone for LocalAtom {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for LocalAtom {}

#[allow(clippy::needless_range_loop)]
fn assign_mapped(
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    center: [f64; 3],
    rotation: [[f64; 3]; 3],
    mechanism: &Mechanism,
) -> Option<rgsaddle::HopTemplate> {
    let n = reactant.len() / 3;
    let mut saddle = reactant.clone();
    let mut product = reactant.clone();
    let mut mode = Array1::zeros(reactant.len());
    let mut used = vec![false; n];
    for atom in &mechanism.atoms {
        let target = [
            center[0] + mat_vec(rotation, atom.offset)[0],
            center[1] + mat_vec(rotation, atom.offset)[1],
            center[2] + mat_vec(rotation, atom.offset)[2],
        ];
        let mut best = None;
        let mut best_d = 0.75_f64;
        for index in 0..n {
            if used[index] {
                continue;
            }
            let distance = mic_distance(point_of(reactant, index), target, cell);
            if distance < best_d {
                best_d = distance;
                best = Some(index);
            }
        }
        let Some(index) = best else {
            continue;
        };
        used[index] = true;
        let to_saddle = mat_vec(rotation, atom.to_saddle);
        let to_product = mat_vec(rotation, atom.to_product);
        let mode_atom = mat_vec(rotation, atom.mode);
        for c in 0..3 {
            saddle[3 * index + c] = reactant[3 * index + c] + to_saddle[c];
            product[3 * index + c] = reactant[3 * index + c] + to_product[c];
            mode[3 * index + c] = mode_atom[c];
        }
    }
    if mode.dot(&mode) < 1.0e-8 {
        return None;
    }
    Some(rgsaddle::HopTemplate {
        name: mechanism.kind.clone(),
        product,
        saddle: Some(saddle),
        mode: Some(mode),
        orientation: axis_label(mat_vec(rotation, mechanism.axis)),
        final_state: String::new(),
    })
}

fn mapped_hops(
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    center: [f64; 3],
    axis: [f64; 3],
    mechanisms: &[Mechanism],
) -> Vec<rgsaddle::HopTemplate> {
    let mut hops = Vec::new();
    for mechanism in mechanisms {
        for rotation in rotations_onto(mechanism.axis, axis) {
            if let Some(hop) = assign_mapped(reactant, cell, center, rotation, mechanism) {
                if hops.iter().any(|prior: &rgsaddle::HopTemplate| {
                    hausdorff(&prior.product, &hop.product, cell) < 0.45
                }) {
                    continue;
                }
                hops.push(hop);
            }
        }
    }
    hops
}

#[derive(Clone)]
struct SciJob {
    repeats: usize,
    shell: String,
    radius: f64,
    orientation: String,
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
    fixed: [f64; 3],
    mobile: Vec<usize>,
    hop: rgsaddle::HopTemplate,
    a: f64,
}

fn run_science_job(
    copper: &FastCopper,
    job: &SciJob,
    temperatures: &[f64],
    finite: bool,
) -> Result<Vec<rgsaddle::RateEntry>, String> {
    let (perfect, _) = fcc(job.repeats, job.a);
    let (_, _, bulk) = copper.evaluate(perfect.view(), job.cell);
    let center = centroid(&job.positions, [job.mobile[0], job.mobile[1]]);
    let harmonic = local_atoms(&job.positions, center, job.cell);
    let config = cu_config(temperatures, 16, 0.01, bulk, harmonic, 41);
    let pafi = config
        .pafi
        .clone()
        .with_max_rounds(3)
        .with_samples(16)
        .with_thermalize(20);
    let config = config
        .with_finite_temperature(finite)
        .with_zero_t_tol(0.01)
        .with_pafi(pafi)
        .with_max_dimer_steps(60)
        .with_max_band_steps(80)
        .with_images(7)
        .with_connect_rmsd(0.9);
    let setup = rgsaddle::PairSetup {
        positions: job.positions.clone(),
        cell: job.cell,
        fixed: job.fixed,
        mobile: job.mobile.clone(),
        shell: job.shell.clone(),
        cell_repeats: job.repeats as u32,
        hops: vec![job.hop.clone()],
    };
    rgsaddle::defect_pair_rates(
        &FastHeld {
            copper,
            cell: job.cell,
        },
        &setup,
        &config,
    )
    .map(|found| found.entries)
    .map_err(|error| error.to_string())
}

fn scale_job(job: &SciJob, a: f64) -> SciJob {
    let factor = a / job.a;
    let length = job.repeats as f64 * a;
    let mut hop = job.hop.clone();
    hop.product = scale_positions(&hop.product, factor);
    if let Some(saddle) = &hop.saddle {
        hop.saddle = Some(scale_positions(saddle, factor));
    }
    if let Some(mode) = &hop.mode {
        hop.mode = Some(scale_positions(mode, factor));
    }
    SciJob {
        repeats: job.repeats,
        shell: job.shell.clone(),
        radius: job.radius * factor,
        orientation: job.orientation.clone(),
        positions: scale_positions(&job.positions, factor),
        cell: [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        fixed: [
            job.fixed[0] * factor,
            job.fixed[1] * factor,
            job.fixed[2] * factor,
        ],
        mobile: job.mobile.clone(),
        hop,
        a,
    }
}

#[test]
#[ignore = "Mishin Cu recombination-volume catalogue seeded from the free dumbbell"]
fn defect_pair_science() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let a500 = measured_lattice(500.0);
    let repeats = 8usize;
    let mechanisms = find_mechanisms(&copper, repeats, a500);
    eprintln!(
        "mechanisms {} in {:.0}s",
        mechanisms.len(),
        started.elapsed().as_secs_f64()
    );
    let _ = std::io::stderr().flush();
    assert!(
        !mechanisms.is_empty(),
        "the isolated dumbbell produced no hop"
    );
    let shells = shell_table(a500, repeats);
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut seen_orientation = Vec::new();
    let mut jobs = Vec::new();
    let mut recombined_notes = Vec::new();
    for (radius, vector, count) in &shells {
        if *radius > 3.0 * a500 {
            continue;
        }
        let max_radius = std::env::var("RGSADDLE_MAX_RADIUS")
            .ok()
            .and_then(|text| text.parse::<f64>().ok())
            .unwrap_or(1.0e9);
        if *radius > max_radius {
            continue;
        }
        let shell_hat = unit3(*vector);
        for axis in axes {
            let key = ((axis[0] * shell_hat[0] + axis[1] * shell_hat[1] + axis[2] * shell_hat[2])
                .abs()
                * 1000.0)
                .round() as i32;
            let shell_key = miller(*vector, a500);
            let mark = (shell_key, key);
            if seen_orientation.contains(&mark) {
                continue;
            }
            seen_orientation.push(mark);
            let PairState {
                mut positions,
                vacancy,
                dumbbell,
                cell,
                gap,
                separation,
                recombined,
                ..
            } = placed(&copper, repeats, a500, *vector, axis, true);
            if recombined {
                recombined_notes.push(format!(
                    "500 K repeats {repeats} shell {radius:.3} orientation {} recombined sites {count} gap {gap:.3} separation {separation:.3}",
                    axis_label(axis)
                ));
                continue;
            }
            let center = centroid(&positions, dumbbell);
            let hops = mapped_hops(&positions, cell, center, axis, &mechanisms);
            eprintln!(
                "shell {radius:.3} orientation {} hops {} in {:.0}s",
                axis_label(axis),
                hops.len(),
                started.elapsed().as_secs_f64()
            );
            let _ = std::io::stderr().flush();
            for (index, mut hop) in hops.into_iter().enumerate() {
                hop.final_state = format!("{index}");
                hop.orientation = axis_label(axis);
                jobs.push(SciJob {
                    repeats,
                    shell: format!("{radius:.3}"),
                    radius: *radius,
                    orientation: axis_label(axis),
                    positions: positions.clone(),
                    cell,
                    fixed: vacancy,
                    mobile: vec![dumbbell[0], dumbbell[1]],
                    hop,
                    a: a500,
                });
            }
            let _ = &mut positions;
        }
    }
    eprintln!("{} static jobs", jobs.len());
    let static_rows: Vec<rgsaddle::RateEntry> = jobs
        .par_iter()
        .flat_map(|job| run_science_job(&copper, job, &[500.0], false).unwrap_or_default())
        .collect();
    for entry in &static_rows {
        eprintln!(
            "dE0 {} {} {} {} {:.4} zero {} outcome {} band {:.4}",
            entry.shell,
            entry.orientation,
            entry.hop,
            entry.direction.as_str(),
            entry.d_e0,
            num_or(&entry.zero_t),
            entry.outcome.as_str(),
            entry.band_force
        );
    }
    let _ = std::io::stderr().flush();
    let mut thermal_jobs = Vec::new();
    for entry in &static_rows {
        if entry.outcome.as_str() != "rated" {
            continue;
        }
        if entry.direction.as_str() == "tangential" && !entry.hop.starts_with("rotate") {
            continue;
        }
        if entry.d_e0 > 0.5 {
            continue;
        }
        let Some(job) = jobs.iter().find(|job| {
            job.shell == entry.shell
                && job.orientation == entry.orientation
                && job.hop.name == entry.hop
        }) else {
            continue;
        };
        thermal_jobs.push(job.clone());
    }
    eprintln!("{} thermal jobs", thermal_jobs.len());
    let mut thermal_rows = Vec::new();
    for temperature in [300.0, 500.0] {
        let a = measured_lattice(temperature);
        let batch: Vec<rgsaddle::RateEntry> = thermal_jobs
            .par_iter()
            .map(|job| scale_job(job, a))
            .flat_map(|job| {
                run_science_job(&copper, &job, &[temperature], true).unwrap_or_default()
            })
            .collect();
        thermal_rows.extend(batch);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut csv = File::create(root.join("defect_pair_hops.csv")).expect("hops");
    writeln!(csv, "shell_A,orientation,hop,final_state,direction,temperature_K,cell_repeats,outcome,connected,index1,band_force_eV_per_A,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,vineyard_Hz,kappa,rate_Hz,separation_before_A,separation_after_A").unwrap();
    let mut catalogue = rgsaddle::RateCatalogue {
        entries: Vec::new(),
    };
    for entry in static_rows.iter().chain(thermal_rows.iter()) {
        catalogue.entries.push(entry.clone());
        writeln!(
            csv,
            "{},{},{},{},{},{},{},{},{},{},{:.5e},{},{},{},{},{},{},{},{},{:.4},{:.4}",
            entry.shell,
            entry.orientation,
            entry.hop,
            entry.final_state,
            entry.direction.as_str(),
            entry.temperature,
            entry.cell_repeats,
            entry.outcome.as_str(),
            entry.connected as u8,
            entry.index1 as u8,
            entry.band_force,
            num_or(&Some(entry.d_e0)),
            num_or(&entry.zero_t),
            num_or(&entry.d_f),
            num_or(&entry.d_f_error),
            num_or(&entry.nu),
            num_or(&entry.vineyard),
            num_or(&entry.kappa),
            num_or(&entry.rate),
            entry.separation_before,
            entry.separation_after
        )
        .unwrap();
    }
    std::fs::write(root.join("defect_pair_hops.json"), catalogue.to_json()).unwrap();
    let mut notes = recombined_notes;
    notes.extend(orientation_scan(&copper));
    notes.push(format!(
        "Isolated mechanisms: {}. Static jobs {}. Thermal jobs {}. Wall {:.1} min.",
        mechanisms
            .iter()
            .map(|item| format!("{} {:.4}", item.kind, item.barrier))
            .collect::<Vec<_>>()
            .join(", "),
        jobs.len(),
        thermal_jobs.len(),
        started.elapsed().as_secs_f64() / 60.0
    ));
    notes.push("NPT lattice parameters were measured earlier on this potential: 3.628311, 3.633713, 3.638666, 3.644160, 3.649788 A at 300 to 700 K.".into());
    notes.push("An AKMC edge is (shell, orientation, final_state, temperature). rate_Hz is the exit rate and nu_Hz is the attempt frequency along the path.".into());
    std::fs::write(root.join("defect_pair_hops_notes.txt"), notes.join("\n")).unwrap();
    eprintln!("{}", notes.join("\n"));
}

fn orientation_scan(copper: &FastCopper) -> Vec<String> {
    let mut tasks = Vec::new();
    let scan_sizes: Vec<usize> = std::env::var("RGSADDLE_SCAN_SIZES")
        .unwrap_or_else(|_| "8,10,12".into())
        .split(',')
        .filter_map(|item| item.trim().parse().ok())
        .collect();
    for repeats in scan_sizes {
        for temperature in [300.0, 400.0, 500.0, 600.0, 700.0] {
            let a = measured_lattice(temperature);
            for (radius, vector, count) in shell_table(a, repeats) {
                if radius > 3.0 * a {
                    continue;
                }
                for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                    tasks.push((repeats, temperature, a, radius, vector, count, axis));
                }
            }
        }
    }
    eprintln!("recombination tasks {}", tasks.len());
    let _ = std::io::stderr().flush();
    let rows: Vec<String> = tasks
        .par_iter()
        .filter_map(|(repeats, temperature, a, radius, vector, count, axis)| {
            let PairState {
        gap,
        separation,
        recombined,
        ..
    } = placed(copper, *repeats, *a, *vector, *axis, true);
            if !recombined {
                return None;
            }
            Some(format!(
                "recombined repeats {repeats} T {temperature} shell {radius:.3} orientation {} sites {count} gap {gap:.3} separation {separation:.3}",
                axis_label(*axis)
            ))
        })
        .collect();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut file = File::create(root.join("defect_pair_recombination.csv")).expect("recombination");
    writeln!(file, "note").unwrap();
    for row in &rows {
        writeln!(file, "{row}").unwrap();
    }
    rows
}

fn wrap_delta(from: [f64; 3], to: [f64; 3], length: f64) -> [f64; 3] {
    let mut delta = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    for item in &mut delta {
        *item -= length * (*item / length).round();
    }
    delta
}

fn crystal_image(reactant: &Array1<f64>, repeats: usize, a: f64) -> Array1<f64> {
    let (perfect, _) = fcc(repeats, a);
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let n = reactant.len() / 3;
    let mut sites = Vec::with_capacity(perfect.len() / 3);
    for atom in 0..perfect.len() / 3 {
        sites.push(point_of(&perfect, atom));
    }
    let mut pairs = Vec::with_capacity(n * sites.len());
    for atom in 0..n {
        let here = point_of(reactant, atom);
        for (index, site) in sites.iter().enumerate() {
            pairs.push((mic_distance(here, *site, cell), atom, index));
        }
    }
    pairs.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut atom_used = vec![false; n];
    let mut site_used = vec![false; sites.len()];
    let mut product = reactant.clone();
    let mut assigned = 0usize;
    for (_, atom, site) in pairs {
        if atom_used[atom] || site_used[site] {
            continue;
        }
        atom_used[atom] = true;
        site_used[site] = true;
        let here = point_of(reactant, atom);
        let delta = wrap_delta(here, sites[site], length);
        for axis in 0..3 {
            product[3 * atom + axis] = here[axis] + delta[axis];
        }
        assigned += 1;
        if assigned == n {
            break;
        }
    }
    product
}

fn capture_band(climbing: bool) -> rgsaddle::BandConfig {
    rgsaddle::BandConfig::default()
        .with_force_tol(0.02)
        .with_max_move(0.08)
        .with_time_step(0.08)
        .with_method(Method::fire(FireKind::V2))
        .with_spring(rgsaddle::SpringKind::Uniform { k: 1.0 })
        .with_climbing(climbing.then(rgsaddle::CiConfig::always))
        .with_min_mode(None)
        .with_autoneb(None)
        .with_zoom(None)
        .with_quickmin(false)
}

fn relax_neb(
    copper: &FastCopper,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Result<(Array2<f64>, f64), String> {
    let wrapped = rgsaddle::mic::Cell::from_ase(cell).map_err(|error| error.to_string())?;
    let guess = rgsaddle::linear_path(reactant.view(), product.view(), 9, Some(&wrapped))
        .map_err(|error| error.to_string())?;
    let held = FastHeld { copper, cell };
    let mut neb = rgsaddle::SolidNeb::frozen(capture_band(false), guess)
        .map_err(|error| error.to_string())?;
    let mut force = f64::MAX;
    for _ in 0..50 {
        let report = neb.step(&held).map_err(|error| error.to_string())?;
        force = report.max_force;
        if force < 0.05 {
            break;
        }
    }
    let relaxed = neb.cartesian().map_err(|error| error.to_string())?;
    let mut neb = rgsaddle::SolidNeb::frozen(capture_band(true), relaxed)
        .map_err(|error| error.to_string())?;
    for _ in 0..40 {
        let report = neb.step(&held).map_err(|error| error.to_string())?;
        force = report.max_force;
        if force < 0.02 {
            break;
        }
    }
    Ok((neb.cartesian().map_err(|error| error.to_string())?, force))
}

fn polish_index(
    copper: &FastCopper,
    saddle: &Array1<f64>,
    mode: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Option<(f64, bool)> {
    let scale = mode.dot(mode).sqrt();
    if scale < 1.0e-8 {
        return None;
    }
    let search = MinModeConfig::default()
        .with_force_tol(0.03)
        .with_max_move(0.08)
        .with_dr(1.0e-3)
        .with_max_rotations(6);
    let config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1));
    let mut dimer = SolidDimer::new(config, saddle.clone(), cell, mode / scale).ok()?;
    let held = FastHeld { copper, cell };
    let report = dimer.run(&held, 40).ok()?;
    if report.status != MinModeStatus::Converged {
        return None;
    }
    let (energy, _) = held.eval(dimer.positions()).ok()?;
    let index1 = !report.rejected && report.negative_modes == Some(1);
    if hausdorff(&dimer.positions().to_owned(), saddle, cell) > 1.2 {
        return None;
    }
    Some((energy, index1))
}

#[derive(Clone)]
struct CaptureRow {
    shell: String,
    orientation: String,
    d_e0: f64,
    zero_t: f64,
    band_force: f64,
    index1: bool,
    note: String,
    saddle: Array1<f64>,
    product: Array1<f64>,
    reactant: Array1<f64>,
    cell: [[f64; 3]; 3],
    fixed: [f64; 3],
    mobile: Vec<usize>,
}

fn capture_one(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    radius: f64,
    axis: [f64; 3],
    mechanisms: &[Mechanism],
) -> Result<CaptureRow, String> {
    let vector = shell_table(a, repeats)
        .into_iter()
        .find(|(value, _, _)| (*value - radius).abs() < 0.03)
        .map(|(_, vector, _)| vector)
        .ok_or_else(|| format!("no shell at {radius}"))?;
    let PairState {
        positions: reactant,
        vacancy,
        dumbbell,
        cell,
        recombined,
        ..
    } = placed(copper, repeats, a, vector, axis, true);
    if recombined {
        return Err("reactant recombined".into());
    }
    let center = centroid(&reactant, dumbbell);
    let hops = mapped_hops(&reactant, cell, center, axis, mechanisms);
    let held = FastHeld { copper, cell };
    let (reactant_energy, _) = held
        .eval(reactant.view())
        .map_err(|error| error.to_string())?;
    let mobile = [dumbbell[0], dumbbell[1]];
    let before = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
    let mut note = String::from("no toward channel");
    let mut chosen: Option<CaptureRow> = None;
    for hop in hops {
        let after =
            rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile).unwrap_or(before);
        if after > before - 0.05 {
            continue;
        }
        let mut product = hop.product.clone();
        product = settle(copper, product, cell, 1.0e-3).expect("product relaxation");
        let (product_energy, _) = held
            .eval(product.view())
            .map_err(|error| error.to_string())?;
        let drop = reactant_energy - product_energy;
        let Some(hint) = hop.saddle.clone() else {
            continue;
        };
        let Some(mode) = hop.mode.clone() else {
            continue;
        };
        let Some((saddle_energy, index1)) = polish_index(copper, &hint, &mode, cell) else {
            note = format!("toward drop {drop:.3} eV; dimer did not stay on the seed");
            continue;
        };
        if !index1 {
            note = format!("toward drop {drop:.3} eV; saddle is not index-1");
            continue;
        }
        let d_e0 = saddle_energy - reactant_energy;
        let (band, band_force) =
            relax_neb(copper, &reactant, &hint, cell).unwrap_or((Array2::zeros((0, 0)), f64::MAX));
        let zero_t = if band.nrows() >= 2 {
            let mut frames = Vec::new();
            for row in band.rows() {
                frames.push(row.to_owned());
            }
            rgsaddle::ArcPath::from_positions(&frames)
                .and_then(|path| rgsaddle::zero_temperature_barrier(&held, &path, &path.arcs()))
                .map(|profile| profile.barrier)
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let barrierless = d_e0 < BOLTZMANN_EV_PER_K * 500.0;
        let row = CaptureRow {
            shell: format!("{radius:.3}"),
            orientation: axis_label(axis),
            d_e0,
            zero_t,
            band_force,
            index1: true,
            note: format!(
                "drop {drop:.3} eV; {}",
                if drop < 2.0 {
                    "product stayed bound"
                } else if barrierless {
                    "effectively barrierless (below kT at 500 K)"
                } else {
                    "index-1 capture saddle"
                }
            ),
            saddle: hint,
            product,
            reactant: reactant.clone(),
            cell,
            fixed: vacancy,
            mobile: vec![dumbbell[0], dumbbell[1]],
        };
        let replace = chosen.as_ref().is_none_or(|prior| row.d_e0 < prior.d_e0);
        if replace && drop >= 2.0 {
            chosen = Some(row);
        } else if chosen.is_none() {
            note = row.note.clone();
        }
    }
    chosen.ok_or(note)
}

fn bound_orientations() -> Vec<(f64, String)> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_hops_static.csv");
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 || fields[6] != "rated" {
            continue;
        }
        let Ok(radius) = fields[0].parse::<f64>() else {
            continue;
        };
        let key = format!("{radius:.3}");
        if pairs
            .iter()
            .any(|(shell, orientation)| shell == &key && orientation == fields[1])
        {
            continue;
        }
        pairs.push((key, fields[1].to_string()));
    }
    pairs.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    pairs
        .into_iter()
        .filter_map(|(shell, orientation)| {
            shell
                .parse::<f64>()
                .ok()
                .map(|radius| (radius, orientation))
        })
        .collect()
}

fn axis_from_label(label: &str) -> [f64; 3] {
    match label {
        "010" => [0.0, 1.0, 0.0],
        "001" => [0.0, 0.0, 1.0],
        _ => [1.0, 0.0, 0.0],
    }
}

fn checkpoint_has(text: &str, key: &str) -> bool {
    text.lines().any(|line| line.starts_with(key))
}

fn append_checkpoint(file: &Mutex<File>, line: &str) {
    let mut handle = file.lock().unwrap_or_else(|poison| poison.into_inner());
    writeln!(handle, "{line}").unwrap();
    let _ = handle.flush();
}

fn pafi_stored(text: &str, shell: &str, orientation: &str, hop: &str, temperature: f64) -> bool {
    text.lines().any(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13 || fields[0] != "pafi" {
            return false;
        }
        fields[1] == shell
            && fields[2] == orientation
            && fields[3] == hop
            && fields[5].parse::<f64>().ok() == Some(temperature)
            && fields[12] == "rated"
            && fields[9]
                .parse::<f64>()
                .ok()
                .is_some_and(|error| error <= 0.01)
    })
}

fn rotation_job(copper: &FastCopper, repeats: usize, a: f64) -> SciJob {
    let PairState {
        positions,
        vacancy,
        dumbbell,
        cell,
        ..
    } = placed(copper, repeats, a, [a, 0.0, 0.0], [1.0, 0.0, 0.0], false);
    let center = centroid(&positions, dumbbell);
    let half = dist(point_of(&positions, dumbbell[0]), center).max(0.9);
    let mut product = positions.clone();
    let mut saddle = positions.clone();
    let mut mode = Array1::zeros(positions.len());
    let diagonal = unit3([1.0, 1.0, 0.0]);
    for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
        for axis in 0..3 {
            let along_y = [0.0, 1.0, 0.0][axis];
            product[3 * atom + axis] = center[axis] + sign * half * along_y;
            saddle[3 * atom + axis] = center[axis] + sign * half * diagonal[axis];
            mode[3 * atom + axis] = product[3 * atom + axis] - positions[3 * atom + axis];
        }
    }
    SciJob {
        repeats,
        shell: "isolated".into(),
        radius: 0.0,
        orientation: "100".into(),
        positions,
        cell,
        fixed: vacancy,
        mobile: vec![dumbbell[0], dumbbell[1]],
        hop: rgsaddle::HopTemplate {
            name: "rotate".into(),
            product,
            saddle: Some(saddle),
            mode: Some(mode),
            orientation: "100".into(),
            final_state: "010".into(),
        },
        a,
    }
}

fn record_entries(file: &Mutex<File>, kind: &str, entries: &[rgsaddle::RateEntry]) {
    for entry in entries {
        let line = format!(
            "{kind},{},{},{},{},{},{:.6},{},{},{},{},{},{},{:.5},{}",
            entry.shell,
            entry.orientation,
            entry.hop,
            entry.direction.as_str(),
            entry.temperature,
            entry.d_e0,
            num_or(&entry.zero_t),
            num_or(&entry.d_f),
            num_or(&entry.d_f_error),
            num_or(&entry.nu),
            num_or(&entry.rate),
            entry.outcome.as_str(),
            entry.band_force,
            entry.final_state.replace(',', ";")
        );
        eprintln!("{line}");
        let _ = std::io::stderr().flush();
        append_checkpoint(file, &line);
    }
}

#[test]
#[ignore = "capture saddles and PAFI for the Cu vacancy-dumbbell pair"]
fn defect_pair_capture() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let a500 = measured_lattice(500.0);
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_capture.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("checkpoint");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let mechanisms = find_mechanisms(&copper, repeats, a500);
    eprintln!(
        "mechanisms {} in {:.0}s",
        mechanisms.len(),
        started.elapsed().as_secs_f64()
    );
    let pairs = bound_orientations();
    eprintln!("{} bound shell/orientation pairs", pairs.len());
    let rows: Vec<CaptureRow> = pairs
        .par_iter()
        .filter_map(|(radius, orientation)| {
            let key = format!("capture,{radius:.3},{orientation},");
            if checkpoint_has(&existing, &key) {
                eprintln!("skip capture {radius:.3} {orientation}");
                return None;
            }
            match capture_one(&copper, repeats, a500, *radius, axis_from_label(orientation), &mechanisms) {
                Ok(row) => {
                    let line = format!(
                        "capture,{},{},capture,toward,500,{:.6},{:.6},,,,,{},{:.5},{}",
                        row.shell,
                        row.orientation,
                        row.d_e0,
                        row.zero_t,
                        if row.band_force > 0.05 {
                            "not_converged"
                        } else if row.index1 {
                            "index1"
                        } else {
                            "neb"
                        },
                        row.band_force,
                        row.note.replace(',', ";")
                    );
                    eprintln!("{line}");
                    let _ = std::io::stderr().flush();
                    append_checkpoint(&checkpoint, &line);
                    Some(row)
                }
                Err(error) => {
                    let line = format!(
                        "capture,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,{error}"
                    );
                    eprintln!("{line}");
                    append_checkpoint(&checkpoint, &line);
                    None
                }
            }
        })
        .collect();
    let mut jobs = Vec::new();
    if mechanisms.iter().any(|item| item.kind == "translate") {
        let translate: Vec<Mechanism> = mechanisms
            .iter()
            .filter(|item| item.kind == "translate")
            .cloned()
            .collect();
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            ..
        } = placed(
            &copper,
            repeats,
            a500,
            [a500, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            false,
        );
        let center = centroid(&positions, dumbbell);
        if let Some(hop) = mapped_hops(&positions, cell, center, [1.0, 0.0, 0.0], &translate)
            .into_iter()
            .next()
        {
            jobs.push(SciJob {
                repeats,
                shell: "isolated".into(),
                radius: 0.0,
                orientation: "100".into(),
                positions,
                cell,
                fixed: vacancy,
                mobile: vec![dumbbell[0], dumbbell[1]],
                hop,
                a: a500,
            });
        }
    }
    jobs.push(rotation_job(&copper, repeats, a500));
    for radius in [3.639_f64, 4.456, 5.146] {
        for (have, orientation) in &pairs {
            if (*have - radius).abs() > 0.03 {
                continue;
            }
            let Some((_, vector, _)) = shell_table(a500, repeats)
                .into_iter()
                .find(|(value, _, _)| (*value - radius).abs() < 0.03)
            else {
                continue;
            };
            let axis = axis_from_label(orientation);
            let PairState {
                positions,
                vacancy,
                dumbbell,
                cell,
                recombined,
                ..
            } = placed(&copper, repeats, a500, vector, axis, true);
            if recombined {
                continue;
            }
            let center = centroid(&positions, dumbbell);
            for hop in mapped_hops(&positions, cell, center, axis, &mechanisms) {
                let mobile = [dumbbell[0], dumbbell[1]];
                let before = rgsaddle::pair_separation(positions.view(), cell, vacancy, &mobile)
                    .unwrap_or(0.0);
                let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile)
                    .unwrap_or(0.0);
                if after <= before + 0.08 {
                    continue;
                }
                jobs.push(SciJob {
                    repeats,
                    shell: format!("{radius:.3}"),
                    radius,
                    orientation: orientation.clone(),
                    positions: positions.clone(),
                    cell,
                    fixed: vacancy,
                    mobile: vec![dumbbell[0], dumbbell[1]],
                    hop,
                    a: a500,
                });
            }
            let saved = rows.iter().find(|row| {
                (row.shell.parse::<f64>().unwrap_or(0.0) - radius).abs() < 0.03
                    && row.orientation == *orientation
            });
            let prior = existing
                .lines()
                .find(|line| line.starts_with(&format!("capture,{radius:.3},{orientation},")));
            let recomputed =
                if saved.is_none() && !prior.is_some_and(|line| line.contains("barrierless")) {
                    capture_one(&copper, repeats, a500, radius, axis, &mechanisms).ok()
                } else {
                    None
                };
            if saved
                .or(recomputed.as_ref())
                .is_some_and(|row| row.d_e0 >= BOLTZMANN_EV_PER_K * 500.0 && row.band_force <= 0.05)
            {
                let source = saved.cloned().or(recomputed);
                let Some(found) = source else { continue };
                jobs.push(SciJob {
                    repeats,
                    shell: found.shell.clone(),
                    radius,
                    orientation: orientation.clone(),
                    positions: found.reactant.clone(),
                    cell: found.cell,
                    fixed: found.fixed,
                    mobile: found.mobile.clone(),
                    hop: rgsaddle::HopTemplate {
                        name: "capture".into(),
                        product: found.product.clone(),
                        saddle: Some(found.saddle.clone()),
                        mode: None,
                        orientation: orientation.clone(),
                        final_state: "crystal".into(),
                    },
                    a: a500,
                });
            }
        }
    }
    eprintln!(
        "{} pafi jobs in {:.0}s",
        jobs.len(),
        started.elapsed().as_secs_f64()
    );
    for temperature in [500.0, 300.0] {
        let a = measured_lattice(temperature);
        for job in &jobs {
            let scaled = scale_job(job, a);
            let fresh = std::fs::read_to_string(&path).unwrap_or_default();
            if pafi_stored(
                &fresh,
                &scaled.shell,
                &scaled.orientation,
                &scaled.hop.name,
                temperature,
            ) {
                eprintln!(
                    "skip pafi {} {} {} {temperature}",
                    scaled.shell, scaled.orientation, scaled.hop.name
                );
                continue;
            }
            let entries =
                run_science_job(&copper, &scaled, &[temperature], true).unwrap_or_default();
            if entries.is_empty() {
                append_checkpoint(
                    &checkpoint,
                    &format!(
                        "pafi,{},{},{},unknown,{temperature},,,,,,,not_converged,0,search failed",
                        scaled.shell, scaled.orientation, scaled.hop.name
                    ),
                );
                continue;
            }
            record_entries(&checkpoint, "pafi", &entries);
        }
    }
    eprintln!(
        "capture run finished in {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

fn drive_phase(
    copper: &FastCopper,
    cell: [[f64; 3]; 3],
    positions: Array2<f64>,
    climbing: bool,
    method: Method,
    steps: usize,
) -> Result<(Array2<f64>, f64), String> {
    let time_step = if matches!(method, Method::Ode12r) {
        0.05
    } else {
        0.08
    };
    let config = capture_band(climbing)
        .with_force_tol(0.01)
        .with_time_step(time_step)
        .with_method(method);
    let held = FastHeld { copper, cell };
    let mut neb =
        rgsaddle::SolidNeb::frozen(config, positions).map_err(|error| error.to_string())?;
    let mut force = f64::MAX;
    let mut best = neb.cartesian().map_err(|error| error.to_string())?;
    let mut best_force = f64::MAX;
    for _ in 0..steps {
        let report = neb.step(&held).map_err(|error| error.to_string())?;
        force = report.max_force;
        if force < best_force {
            best_force = force;
            best = neb.cartesian().map_err(|error| error.to_string())?;
        }
        if force < 0.01 {
            break;
        }
    }
    if force <= best_force {
        Ok((neb.cartesian().map_err(|error| error.to_string())?, force))
    } else {
        Ok((best, best_force))
    }
}

fn capture_neb_band(
    copper: &FastCopper,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Result<(Array2<f64>, f64), String> {
    let wrapped = rgsaddle::mic::Cell::from_ase(cell).map_err(|error| error.to_string())?;
    let idpp = rgsaddle::IdppConfig::default()
        .with_max_iterations(40)
        .with_grad_tol(5.0e-2)
        .with_max_move(0.15)
        .with_memory(8);
    let guess = rgsaddle::idpp_path(reactant.view(), product.view(), 17, Some(&wrapped), &idpp)
        .or_else(|_| rgsaddle::linear_path(reactant.view(), product.view(), 17, Some(&wrapped)))
        .map_err(|error| error.to_string())?;
    let (relaxed, relaxed_force) =
        drive_phase(copper, cell, guess, false, Method::fire(FireKind::V2), 70)?;
    let (climbed, climbed_force) = drive_phase(
        copper,
        cell,
        relaxed.clone(),
        true,
        Method::fire(FireKind::V2),
        50,
    )?;
    if climbed_force <= 0.02 || climbed_force <= relaxed_force {
        return Ok((climbed, climbed_force));
    }
    drive_phase(copper, cell, relaxed, true, Method::Ode12r, 40)
}

struct NebCapture {
    d_e0: f64,
    zero_t: f64,
    band_force: f64,
    index1: bool,
    barrierless: bool,
    note: String,
    saddle: Array1<f64>,
    product: Array1<f64>,
    reactant: Array1<f64>,
    cell: [[f64; 3]; 3],
    fixed: [f64; 3],
    mobile: Vec<usize>,
}

fn neb_capture(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    radius: f64,
    axis: [f64; 3],
) -> Result<NebCapture, String> {
    let vector = shell_table(a, repeats)
        .into_iter()
        .find(|(value, _, _)| (*value - radius).abs() < 0.03)
        .map(|(_, vector, _)| vector)
        .ok_or_else(|| format!("no shell near {radius}"))?;
    let PairState {
        positions: reactant,
        vacancy,
        dumbbell,
        cell,
        recombined,
        ..
    } = placed(copper, repeats, a, vector, axis, true);
    if recombined {
        return Err("reactant recombined on relaxation".into());
    }
    let mut product = crystal_image(&reactant, repeats, a);
    product = settle(copper, product, cell, 1.0e-3).expect("product relaxation");
    let held = FastHeld { copper, cell };
    let (reactant_energy, _) = held
        .eval(reactant.view())
        .map_err(|error| error.to_string())?;
    let (product_energy, _) = held
        .eval(product.view())
        .map_err(|error| error.to_string())?;
    let drop = reactant_energy - product_energy;
    let (band, band_force) = capture_neb_band(copper, &reactant, &product, cell)?;
    let mut energies = Vec::with_capacity(band.nrows());
    for row in band.rows() {
        let (energy, _) = held.eval(row).map_err(|error| error.to_string())?;
        energies.push(energy);
    }
    let reactant_band = energies[0];
    let mut best = 0usize;
    for (index, energy) in energies.iter().enumerate() {
        if *energy > energies[best] {
            best = index;
        }
    }
    let rise = energies[best] - reactant_band;
    let mut frames = Vec::new();
    for row in band.rows() {
        frames.push(row.to_owned());
    }
    let path = rgsaddle::ArcPath::from_positions(&frames).map_err(|error| error.to_string())?;
    let zero = rgsaddle::zero_temperature_barrier(&held, &path, &path.arcs())
        .map_err(|error| error.to_string())?;
    let downhill = rise <= 0.001 && zero.barrier <= 0.001 && band_force <= 0.05;
    let mut d_e0 = rise;
    let mut index1 = false;
    let mut saddle = band.row(best).to_owned();
    if !downhill && best > 0 && best + 1 < band.nrows() && band_force <= 0.05 {
        let mode = &band.row(best + 1).to_owned() - &band.row(best - 1).to_owned();
        if let Some((energy, found)) = polish_index(copper, &saddle, &mode, cell) {
            index1 = found;
            if found {
                d_e0 = energy - reactant_energy;
                saddle = band.row(best).to_owned();
            }
        }
    }
    let note = if downhill {
        format!(
            "barrierless; rise {rise:.4} eV; integral {:.4} eV; drop {drop:.3} eV",
            zero.barrier
        )
    } else if band_force > 0.05 {
        format!(
            "band not converged; rise {rise:.4} eV; integral {:.4} eV; drop {drop:.3} eV",
            zero.barrier
        )
    } else if index1 {
        format!("index-1; rise {rise:.4} eV; drop {drop:.3} eV",)
    } else {
        format!("neb maximum, index not confirmed; rise {rise:.4} eV; drop {drop:.3} eV")
    };
    Ok(NebCapture {
        d_e0,
        zero_t: zero.barrier,
        band_force,
        index1,
        barrierless: downhill,
        note,
        saddle,
        product,
        reactant,
        cell,
        fixed: vacancy,
        mobile: vec![dumbbell[0], dumbbell[1]],
    })
}

fn run_dense_job(
    copper: &FastCopper,
    job: &SciJob,
    temperature: f64,
) -> Result<Vec<rgsaddle::RateEntry>, String> {
    let (perfect, _) = fcc(job.repeats, job.a);
    let (_, _, bulk) = copper.evaluate(perfect.view(), job.cell);
    let center = centroid(&job.positions, [job.mobile[0], job.mobile[1]]);
    let harmonic = local_atoms(&job.positions, center, job.cell);
    let config = cu_config(&[temperature], 24, 0.01, bulk, harmonic, 41);
    let pafi = config
        .pafi
        .clone()
        .with_max_rounds(4)
        .with_samples(24)
        .with_thermalize(20);
    let band = config
        .band
        .clone()
        .with_force_tol(8.0e-3)
        .with_time_step(0.04);
    let config = config
        .with_finite_temperature(true)
        .with_zero_t_tol(0.01)
        .with_pafi(pafi)
        .with_max_dimer_steps(70)
        .with_max_band_steps(300)
        .with_band(band)
        .with_images(9)
        .with_connect_rmsd(1.0);
    let setup = rgsaddle::PairSetup {
        positions: job.positions.clone(),
        cell: job.cell,
        fixed: job.fixed,
        mobile: job.mobile.clone(),
        shell: job.shell.clone(),
        cell_repeats: job.repeats as u32,
        hops: vec![job.hop.clone()],
    };
    rgsaddle::defect_pair_rates(
        &FastHeld {
            copper,
            cell: job.cell,
        },
        &setup,
        &config,
    )
    .map(|found| found.entries)
    .map_err(|error| error.to_string())
}

fn finish_line(kind: &str, entry: &rgsaddle::RateEntry) -> String {
    let kappa = match num_or(&entry.kappa) {
        value if value.is_empty() => "no kappa".to_string(),
        value => format!("kappa {value}"),
    };
    format!(
        "{kind},{},{},{},{},{},{:.6},{},{},{},{},{},{},{:.5},{kappa}",
        entry.shell,
        entry.orientation,
        entry.hop,
        entry.direction.as_str(),
        entry.temperature,
        entry.d_e0,
        num_or(&entry.zero_t),
        num_or(&entry.d_f),
        num_or(&entry.d_f_error),
        num_or(&entry.nu),
        num_or(&entry.rate),
        entry.outcome.as_str(),
        entry.band_force,
    )
}

#[test]
#[ignore = "capture NEB, remaining PAFI, and the away asymmetry"]
fn defect_pair_finish() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let a500 = measured_lattice(500.0);
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_finish.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("finish");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let mut captures = Vec::new();
    for (radius, orientation) in [(3.639_f64, "010"), (4.456, "100")] {
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let key = format!("capture,{radius:.3},{orientation},");
        if checkpoint_has(&existing, &key) {
            eprintln!("skip capture {radius} {orientation}");
            continue;
        }
        match neb_capture(&copper, repeats, a500, radius, axis_from_label(orientation)) {
            Ok(row) => {
                let outcome = if row.barrierless {
                    "barrierless"
                } else if row.band_force > 0.05 {
                    "not_converged"
                } else if row.index1 {
                    "index1"
                } else {
                    "neb"
                };
                let line = format!(
                    "capture,{radius:.3},{orientation},capture,toward,500,{:.6},{:.6},,,,,{outcome},{:.5},{}",
                    row.d_e0,
                    row.zero_t,
                    row.band_force,
                    row.note.replace(',', ";")
                );
                eprintln!("{line}");
                let _ = std::io::stderr().flush();
                append_checkpoint(&checkpoint, &line);
                captures.push((radius, orientation.to_string(), row));
            }
            Err(error) => {
                let line = format!(
                    "capture,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,{error}"
                );
                eprintln!("{line}");
                append_checkpoint(&checkpoint, &line);
            }
        }
    }
    let mut jobs = Vec::new();
    for (radius, _orientation, row) in &captures {
        if row.barrierless || row.band_force > 0.05 || row.d_e0 <= 0.001 {
            continue;
        }
        jobs.push(SciJob {
            repeats,
            shell: format!("{radius:.3}"),
            radius: *radius,
            orientation: row.note.clone(),
            positions: row.reactant.clone(),
            cell: row.cell,
            fixed: row.fixed,
            mobile: row.mobile.clone(),
            hop: rgsaddle::HopTemplate {
                name: "capture".into(),
                product: row.product.clone(),
                saddle: Some(row.saddle.clone()),
                mode: None,
                orientation: if (*radius - 3.639).abs() < 0.02 {
                    "010".into()
                } else {
                    "100".into()
                },
                final_state: "crystal".into(),
            },
            a: a500,
        });
    }
    for job in jobs.iter_mut() {
        let orientation = if (job.radius - 3.639).abs() < 0.02 {
            "010"
        } else {
            "100"
        };
        job.orientation = orientation.into();
        job.hop.orientation = orientation.into();
    }
    let mechanisms = find_mechanisms(&copper, repeats, a500);
    jobs.push(rotation_job(&copper, repeats, a500));
    let shells = [(3.639_f64, "010", "away"), (4.456, "100", "both")];
    for (radius, orientation, which) in shells {
        let vector = shell_table(a500, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.03)
            .map(|(_, vector, _)| vector)
            .unwrap();
        let axis = axis_from_label(orientation);
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            ..
        } = placed(&copper, repeats, a500, vector, axis, true);
        let center = centroid(&positions, dumbbell);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before =
            rgsaddle::pair_separation(positions.view(), cell, vacancy, &mobile).unwrap_or(0.0);
        let mut index = 0usize;
        for mut hop in mapped_hops(&positions, cell, center, axis, &mechanisms) {
            let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile)
                .unwrap_or(before);
            if which == "away" && after <= before + 0.08 {
                continue;
            }
            if which == "both" && after <= before + 0.08 {
                continue;
            }
            hop.name = format!("translate-{index}");
            hop.final_state = format!("away-{index}");
            hop.orientation = orientation.into();
            index += 1;
            jobs.push(SciJob {
                repeats,
                shell: format!("{radius:.3}"),
                radius,
                orientation: orientation.into(),
                positions: positions.clone(),
                cell,
                fixed: vacancy,
                mobile: vec![dumbbell[0], dumbbell[1]],
                hop,
                a: a500,
            });
        }
    }
    eprintln!(
        "{} finish jobs in {:.0}s",
        jobs.len(),
        started.elapsed().as_secs_f64()
    );
    for temperature in [500.0_f64, 300.0] {
        let a = measured_lattice(temperature);
        for job in &jobs {
            if job.hop.name == "rotate" && temperature < 400.0 {
                continue;
            }
            if job.hop.name == "translate"
                && (job.radius - 3.639).abs() < 0.02
                && temperature > 400.0
            {
                continue;
            }
            let scaled = scale_job(job, a);
            let fresh = std::fs::read_to_string(&path).unwrap_or_default();
            let marker = format!(
                "pafi,{},{},{},",
                scaled.shell, scaled.orientation, scaled.hop.name
            );
            if fresh.lines().any(|line| {
                let fields: Vec<&str> = line.split(',').collect();
                fields.len() > 12
                    && line.starts_with(&marker)
                    && fields[5].parse::<f64>().ok() == Some(temperature)
                    && fields[12] == "rated"
                    && fields[9]
                        .parse::<f64>()
                        .ok()
                        .is_some_and(|error| error <= 0.01)
            }) {
                eprintln!("skip {marker}");
                continue;
            }
            let entries = run_dense_job(&copper, &scaled, temperature).unwrap_or_default();
            if entries.is_empty() {
                append_checkpoint(&checkpoint, &format!("{marker}not_converged"));
                continue;
            }
            for entry in &entries {
                let line = finish_line("pafi", entry);
                eprintln!("{line}");
                let _ = std::io::stderr().flush();
                append_checkpoint(&checkpoint, &line);
            }
        }
    }
    eprintln!(
        "finish run {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

struct TinyRng(u64);

impl TinyRng {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn gauss(&mut self) -> f64 {
        let u = self.unit().clamp(1.0e-12, 1.0);
        let v = self.unit();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn local_seeds(
    reactant: &Array1<f64>,
    cell: [[f64; 3]; 3],
    dumbbell: [usize; 2],
    vacancy: [f64; 3],
    salt: u64,
    n_seeds: usize,
) -> Vec<rgsaddle::DimerSeed> {
    let center = centroid(reactant, dumbbell);
    let length = cell[0][0];
    let mut rng = TinyRng(salt);
    let mut seeds = Vec::new();
    let n = reactant.len() / 3;
    let nearby: Vec<usize> = (0..n)
        .filter(|atom| mic_distance(point_of(reactant, *atom), center, cell) < 4.2)
        .collect();
    let to_vac = wrap_delta(center, vacancy, length);
    let to_vac = unit3(to_vac);
    let axis = unit3(wrap_delta(
        point_of(reactant, dumbbell[0]),
        point_of(reactant, dumbbell[1]),
        length,
    ));
    let mut push_atoms = |scale: f64, direction: [f64; 3], which: &[usize]| {
        let mut kick = Array1::zeros(reactant.len());
        for atom in which {
            for c in 0..3 {
                kick[3 * atom + c] = scale * direction[c];
            }
        }
        seeds.push(rgsaddle::DimerSeed {
            positions: reactant + &kick,
            mode: kick,
        });
    };
    push_atoms(0.40, to_vac, &dumbbell);
    push_atoms(0.65, to_vac, &dumbbell);
    let closer = if mic_distance(point_of(reactant, dumbbell[0]), vacancy, cell)
        < mic_distance(point_of(reactant, dumbbell[1]), vacancy, cell)
    {
        dumbbell[0]
    } else {
        dumbbell[1]
    };
    push_atoms(0.50, to_vac, &[closer]);
    push_atoms(0.80, to_vac, &[closer]);
    let perp = unit3([
        axis[1] * to_vac[2] - axis[2] * to_vac[1],
        axis[2] * to_vac[0] - axis[0] * to_vac[2],
        axis[0] * to_vac[1] - axis[1] * to_vac[0],
    ]);
    let mut rotations = Vec::new();
    for angle in [0.35_f64, 0.7] {
        let mut kick = Array1::zeros(reactant.len());
        for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
            let offset = wrap_delta(center, point_of(reactant, atom), length);
            let turned = [
                offset[0] * angle.cos() + perp[0] * sign * offset_len(&offset) * angle.sin(),
                offset[1] * angle.cos() + perp[1] * sign * offset_len(&offset) * angle.sin(),
                offset[2] * angle.cos() + perp[2] * sign * offset_len(&offset) * angle.sin(),
            ];
            for c in 0..3 {
                kick[3 * atom + c] = turned[c] - offset[c];
            }
        }
        rotations.push(kick);
    }
    push_atoms(0.45, perp, &dumbbell);
    push_atoms(-0.45, perp, &dumbbell);
    for kick in rotations {
        seeds.push(rgsaddle::DimerSeed {
            positions: reactant + &kick,
            mode: kick,
        });
    }
    while seeds.len() < n_seeds {
        let mut kick = Array1::zeros(reactant.len());
        for atom in &nearby {
            for c in 0..3 {
                kick[3 * atom + c] = 0.35 * rng.gauss();
            }
        }
        seeds.push(rgsaddle::DimerSeed {
            positions: reactant + &kick,
            mode: kick,
        });
    }
    seeds
}

fn offset_len(offset: &[f64; 3]) -> f64 {
    (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2])
        .sqrt()
        .max(1.0e-8)
}

fn survey_config(bulk: [[f64; 3]; 3]) -> rgsaddle::DefectPairConfig {
    let search = MinModeConfig::default()
        .with_force_tol(0.04)
        .with_max_move(0.1)
        .with_dr(1.0e-3)
        .with_max_rotations(6);
    cu_config(&[500.0], 8, 0.01, bulk, Vec::new(), 7)
        .with_finite_temperature(false)
        .with_max_dimer_steps(40)
        .with_relax(pair_relax().with_force_tol(2.0e-3))
        .with_push(0.35)
        .with_dimer(
            SolidDimerConfig::frozen(search)
                .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(0)),
        )
}

fn curvature_along(held: &FastHeld, positions: &Array1<f64>, tangent: &Array1<f64>) -> f64 {
    let scale = tangent.dot(tangent).sqrt().max(1.0e-12);
    let step = tangent / scale * 1.0e-3;
    let (e0, _) = held
        .eval(positions.view())
        .unwrap_or((0.0, Array1::zeros(1)));
    let (ep, _) = held
        .eval((positions + &step).view())
        .unwrap_or((e0, Array1::zeros(1)));
    let (em, _) = held
        .eval((positions - &step).view())
        .unwrap_or((e0, Array1::zeros(1)));
    (ep + em - 2.0 * e0) / (1.0e-3 * 1.0e-3)
}

#[test]
#[ignore = "local capture saddles from seeded dimers, then a short climbing band"]
fn defect_pair_local_capture() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let a500 = measured_lattice(500.0);
    let repeats = 8usize;
    let length = repeats as f64 * a500;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let (perfect, _) = fcc(repeats, a500);
    let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_local.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("local");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let jobs = [
        (2.573_f64, "100"),
        (2.573, "010"),
        (2.573, "001"),
        (3.639, "010"),
        (3.639, "100"),
        (3.639, "001"),
        (4.456, "100"),
        (4.456, "010"),
        (5.146, "001"),
        (5.146, "100"),
    ];
    for (radius, orientation) in jobs {
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        if checkpoint_has(&existing, &format!("survey,{radius:.3},{orientation},")) {
            eprintln!("skip survey {radius} {orientation}");
            continue;
        }
        let vector = shell_table(a500, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.04)
            .map(|(_, vector, _)| vector);
        let Some(vector) = vector else {
            append_checkpoint(
                &checkpoint,
                &format!(
                    "survey,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,no shell"
                ),
            );
            continue;
        };
        let PairState {
            positions: reactant,
            vacancy,
            dumbbell,
            cell,
            gap,
            separation,
            recombined,
            ..
        } = placed(
            &copper,
            repeats,
            a500,
            vector,
            axis_from_label(orientation),
            true,
        );
        if recombined {
            let line = format!(
                "survey,{radius:.3},{orientation},capture,toward,500,0.000000,0.000000,,,,,barrierless,0,recombines on relaxation; gap {gap:.3} separation {separation:.3}"
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let held = FastHeld {
            copper: &copper,
            cell,
        };
        let (reactant_energy, _) = held.eval(reactant.view()).expect("reactant");
        let config = survey_config(bulk);
        let seeds = local_seeds(&reactant, cell, dumbbell, vacancy, 0x5eed, 32);
        eprintln!(
            "survey {radius:.3} {orientation} seeds {} at {:.0}s",
            seeds.len(),
            started.elapsed().as_secs_f64()
        );
        let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
        let mut kept = Vec::new();
        for saddle in &found {
            let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0
                || (saddle.plus_energy - reactant_energy).abs() < 0.2;
            let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0
                || (saddle.minus_energy - reactant_energy).abs() < 0.2;
            let plus_re = recombined_in(&saddle.plus, cell);
            let minus_re = recombined_in(&saddle.minus, cell);
            if (plus_back && minus_re) || (minus_back && plus_re) {
                kept.push(saddle);
            }
        }
        kept.sort_by(|left, right| {
            (left.saddle_energy - reactant_energy)
                .partial_cmp(&(right.saddle_energy - reactant_energy))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if kept.is_empty() {
            let line = format!(
                "survey,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,no recombination saddle in {} index-1 hits of {} seeds",
                found.len(),
                seeds.len()
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        for (index, saddle) in kept.iter().enumerate() {
            let d_e0 = saddle.saddle_energy - reactant_energy;
            let line = format!(
                "survey,{radius:.3},{orientation},capture-{index},toward,500,{d_e0:.6},,,,,index1,0,eigenvalue {}",
                saddle.negative_eigenvalue.unwrap_or(f64::NAN)
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
        }
        let best = kept
            .iter()
            .find(|saddle| saddle.saddle_energy - reactant_energy > 0.001)
            .unwrap_or(&kept[0]);
        let d_e0 = best.saddle_energy - reactant_energy;
        if d_e0 <= 0.001 {
            let line = format!(
                "capture,{radius:.3},{orientation},capture,toward,500,{d_e0:.6},{d_e0:.6},,,,,barrierless,0,lowest connecting index-1 is {d_e0:.4} eV; no positive barrier above 1 meV"
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let sign = if best.plus_energy < best.minus_energy {
            1.0
        } else {
            -1.0
        };
        let mut peak = 0.0_f64;
        for atom in 0..best.mode.len() / 3 {
            let mut acc = 0.0;
            for c in 0..3 {
                acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
            }
            peak = peak.max(acc.sqrt());
        }
        let mut step = 0.04_f64;
        let mut start = best.saddle.clone();
        if peak > 1.0e-8 {
            while step >= 0.01 {
                let trial = &best.saddle + &(&best.mode * (sign * step / peak));
                let trial_energy = held
                    .eval(trial.view())
                    .map(|(energy, _)| energy)
                    .unwrap_or(f64::NAN);
                if trial_energy.is_finite() && trial_energy > best.saddle_energy - 0.3 {
                    start = trial;
                    break;
                }
                step *= 0.5;
            }
        }
        let floor = best.saddle_energy - 0.6;
        let (downhill, down_energy) =
            rgsaddle::descend_to_floor(&held, start, cell, &config, floor).expect("descend");
        let band_config = config
            .clone()
            .with_band(
                config
                    .band
                    .clone()
                    .with_force_tol(2.0e-3)
                    .with_time_step(0.04),
            )
            .with_max_band_steps(320)
            .with_images(13)
            .with_zero_t_tol(0.02);
        let (band, converged, force) = rgsaddle::segment_band(
            &held,
            &band_config,
            &reactant,
            &best.saddle,
            &downhill,
            cell,
            0.1,
            5.0,
        )
        .unwrap_or_else(|error| {
            eprintln!("band {radius} {orientation}: {error}");
            (
                rgsaddle::ArcPath::from_positions(&[reactant.clone(), best.saddle.clone()])
                    .expect("path"),
                false,
                f64::MAX,
            )
        });
        let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
        let zero_v = zero
            .as_ref()
            .map(|profile| profile.barrier)
            .unwrap_or(f64::NAN);
        let integral_ok = zero
            .as_ref()
            .is_some_and(|profile| (profile.barrier - d_e0).abs() < 0.05);
        let outcome =
            if converged && force <= 2.0e-3 && integral_ok && down_energy > reactant_energy - 1.5 {
                "index1"
            } else {
                "not_converged"
            };
        let line = format!(
            "capture,{radius:.3},{orientation},capture,toward,500,{d_e0:.6},{zero_v:.6},,,,,{outcome},{force:.5},downhill_rel {:.3}; seeds {} kept {}",
            down_energy - reactant_energy,
            seeds.len(),
            kept.len()
        );
        eprintln!("{line}");
        let _ = std::io::stderr().flush();
        append_checkpoint(&checkpoint, &line);
        if outcome != "index1" {
            continue;
        }
        for temperature in [500.0_f64, 300.0] {
            let factor = measured_lattice(temperature) / a500;
            let scaled_cell = [
                [cell[0][0] * factor, 0.0, 0.0],
                [0.0, cell[1][1] * factor, 0.0],
                [0.0, 0.0, cell[2][2] * factor],
            ];
            let frames: Vec<Array1<f64>> = band
                .arcs()
                .iter()
                .map(|arc| band.geometry(*arc).expect("frame").positions * factor)
                .collect();
            let scaled = rgsaddle::ArcPath::from_positions(&frames).expect("scaled path");
            let held_t = FastHeld {
                copper: &copper,
                cell: scaled_cell,
            };
            let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
            let pafi = band_config
                .pafi
                .clone()
                .with_thermal(thermal)
                .with_samples(20)
                .with_max_rounds(4)
                .with_target_error(0.01)
                .with_thermalize(20);
            let arcs = scaled.arcs();
            let Ok((profile, planes)) = rgsaddle::pafi_profile(&held_t, &scaled, &arcs, &pafi)
            else {
                append_checkpoint(
                    &checkpoint,
                    &format!(
                        "pafi,{radius:.3},{orientation},capture,toward,{temperature},,,,,,,not_converged,0,pafi failed"
                    ),
                );
                continue;
            };
            let dividing = profile.dividing;
            let geometry = scaled.geometry(profile.arc[dividing]).expect("geometry");
            let shooting = band_config
                .shooting
                .clone()
                .with_thermal(thermal)
                .with_mass(63.546);
            let starts = if planes[dividing].retained.is_empty() {
                vec![geometry.positions.clone()]
            } else {
                planes[dividing].retained.clone()
            };
            let transmission = rgsaddle::bennett::bennett_chandler(
                &held_t,
                &starts,
                geometry.positions.view(),
                geometry.tangent.view(),
                &shooting,
            );
            let kappa = transmission
                .as_ref()
                .map(|value| value.kappa)
                .unwrap_or(f64::NAN);
            let reactant_t = &reactant * factor;
            let reactant_tangent = scaled
                .geometry(scaled.bounds().0)
                .expect("reactant tangent")
                .tangent;
            let lambda = curvature_along(&held_t, &reactant_t, &reactant_tangent);
            let nu =
                rgsaddle::attempt_frequency(lambda, 63.546, rgsaddle::angular_scale_rad_s()).ok();
            let rate = nu.and_then(|prefactor| {
                if (0.0..=1.0).contains(&kappa) {
                    Some(prefactor * kappa * (-profile.barrier / thermal).exp())
                } else {
                    None
                }
            });
            let line = format!(
                "pafi,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},{zero_v:.6},{:.6},{:.6},{},{},rated,{force:.5},kappa {kappa:.6}",
                profile.barrier,
                profile.barrier_error,
                num_or(&nu),
                num_or(&rate)
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
        }
    }
    let mechanisms = find_mechanisms(&copper, repeats, a500);
    let translate: Vec<Mechanism> = mechanisms
        .into_iter()
        .filter(|item| item.kind == "translate")
        .collect();
    if !translate.is_empty() {
        let vector = shell_table(a500, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - 3.639).abs() < 0.04)
            .map(|(_, vector, _)| vector)
            .expect("3.639");
        let a300 = measured_lattice(300.0);
        let factor = a300 / a500;
        let vector = [vector[0] * factor, vector[1] * factor, vector[2] * factor];
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell: cell300,
            ..
        } = placed(&copper, repeats, a300, vector, axis_from_label("010"), true);
        let center = centroid(&positions, dumbbell);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before =
            rgsaddle::pair_separation(positions.view(), cell300, vacancy, &mobile).unwrap_or(0.0);
        for mut hop in mapped_hops(&positions, cell300, center, [0.0, 1.0, 0.0], &translate) {
            let after = rgsaddle::pair_separation(hop.product.view(), cell300, vacancy, &mobile)
                .unwrap_or(before);
            if after <= before + 0.08 {
                continue;
            }
            hop.name = "translate-away".into();
            hop.orientation = "010".into();
            let job = SciJob {
                repeats,
                shell: "3.639".into(),
                radius: 3.639 * a300 / a500,
                orientation: "010".into(),
                positions: positions.clone(),
                cell: cell300,
                fixed: vacancy,
                mobile: vec![dumbbell[0], dumbbell[1]],
                hop,
                a: a300,
            };
            let entries = run_dense_job(&copper, &job, 300.0).unwrap_or_default();
            for entry in &entries {
                let line = finish_line("pafi", entry);
                eprintln!("{line}");
                append_checkpoint(&checkpoint, &line);
            }
            break;
        }
    }
    eprintln!(
        "local capture {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

/// The energy of a converged basin.
fn basin_energy(copper: &FastCopper, positions: &Array1<f64>, cell: [[f64; 3]; 3]) -> f64 {
    FastHeld { copper, cell }
        .eval(positions.view())
        .expect("basin energy")
        .0
}

#[test]
#[ignore = "re-referenced capture barriers and the kinetic recombination volume"]
fn defect_pair_viv() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_viv.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("viv");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let mut jobs: Vec<(f64, &str, f64)> = Vec::new();
    for temperature in [500.0_f64, 300.0, 700.0] {
        let a = measured_lattice(temperature);
        for (radius, orientation) in [
            (2.573_f64, "001"),
            (3.639, "010"),
            (3.639, "100"),
            (3.639, "001"),
            (4.456, "100"),
            (4.456, "010"),
            (5.146, "001"),
            (6.302, "100"),
            (6.302, "010"),
            (6.807, "100"),
            (7.277, "100"),
            (7.719, "100"),
        ] {
            if temperature != 500.0 && radius > 5.2 {
                continue;
            }
            jobs.push((
                radius * a / measured_lattice(500.0),
                orientation,
                temperature,
            ));
        }
    }
    let (perfect, _) = fcc(repeats, measured_lattice(500.0));
    let length = repeats as f64 * measured_lattice(500.0);
    let (_, _, bulk) = copper.evaluate(
        perfect.view(),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    );
    for (radius, orientation, temperature) in jobs {
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        let key = format!("ref,{radius:.3},{orientation},{temperature},");
        if checkpoint_has(&existing, &key) {
            continue;
        }
        let a = measured_lattice(temperature);
        let vector = shell_table(a, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.05)
            .map(|(_, vector, _)| vector);
        let Some(vector) = vector else { continue };
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            recombined,
            ..
        } = placed(
            &copper,
            repeats,
            a,
            vector,
            axis_from_label(orientation),
            true,
        );
        if recombined {
            let line = format!(
                "ref,{radius:.3},{orientation},capture,toward,{temperature},0.000000,,,,,spontaneous,0,recombines on relaxation; counted in V_iv"
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let reactant = positions;
        let reactant_energy = basin_energy(&copper, &reactant, cell);
        let held = FastHeld {
            copper: &copper,
            cell,
        };
        let n_seeds = if matches!(
            (radius as i32, orientation),
            (3, "100") | (3, "001") | (4, "010")
        ) {
            48
        } else {
            32
        };
        let seeds = local_seeds(
            &reactant,
            cell,
            dumbbell,
            vacancy,
            temperature as u64,
            n_seeds,
        );
        eprintln!(
            "ref {radius:.3} {orientation} {temperature} seeds {} E {reactant_energy:.3} at {:.0}s",
            seeds.len(),
            started.elapsed().as_secs_f64()
        );
        let config = survey_config(bulk);
        let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
        let mut kept: Vec<&rgsaddle::BasinSaddle> = found
            .iter()
            .filter(|saddle| {
                let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
                let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
                let plus_re = recombined_in(&saddle.plus, cell);
                let minus_re = recombined_in(&saddle.minus, cell);
                (plus_back && minus_re) || (minus_back && plus_re)
            })
            .collect();
        kept.sort_by(|left, right| {
            (left.saddle_energy - reactant_energy)
                .partial_cmp(&(right.saddle_energy - reactant_energy))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if kept.is_empty() {
            let line = format!(
                "ref,{radius:.3},{orientation},capture,toward,{temperature},,,,,,not_converged,0,no recombination saddle in {} index-1 of {} seeds; reference {reactant_energy:.4}",
                found.len(),
                seeds.len()
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let best = kept
            .iter()
            .find(|saddle| saddle.saddle_energy >= reactant_energy - 1.0e-4)
            .unwrap_or(&kept[0]);
        let d_e0 = best.saddle_energy - reactant_energy;
        let line = format!(
            "ref,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},,,,,index1,0,n={} lowest_relative {:.4} reference {reactant_energy:.4}",
            kept.len(),
            kept[0].saddle_energy - reactant_energy
        );
        eprintln!("{line}");
        append_checkpoint(&checkpoint, &line);
        if rgsaddle::capture_is_spontaneous(false, Some(d_e0)) {
            let line = format!(
                "ref,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},,,,,spontaneous,0,re-referenced barrier {d_e0:.4} eV is at most 1 meV; counted in V_iv"
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let sign = if best.plus_energy < best.minus_energy {
            1.0
        } else {
            -1.0
        };
        let mut peak = 0.0_f64;
        for atom in 0..best.mode.len() / 3 {
            let mut acc = 0.0;
            for c in 0..3 {
                acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
            }
            peak = peak.max(acc.sqrt());
        }
        let mut step = 0.04_f64;
        let mut start = best.saddle.clone();
        if peak > 1.0e-8 {
            while step >= 0.01 {
                let trial = &best.saddle + &(&best.mode * (sign * step / peak));
                let trial_energy = held
                    .eval(trial.view())
                    .map(|(energy, _)| energy)
                    .unwrap_or(f64::NAN);
                if trial_energy.is_finite() && trial_energy > best.saddle_energy - 0.3 {
                    start = trial;
                    break;
                }
                step *= 0.5;
            }
        }
        let (downhill, down_energy) = rgsaddle::descend_to_floor(
            &held,
            start.clone(),
            cell,
            &config,
            best.saddle_energy - 0.6,
        )
        .expect("descent from the saddle to the floor");
        let band_config = config
            .clone()
            .with_band(
                config
                    .band
                    .clone()
                    .with_force_tol(2.0e-3)
                    .with_time_step(0.04),
            )
            .with_max_band_steps(320)
            .with_images(13);
        let (band, converged, force) = match rgsaddle::segment_band(
            &held,
            &band_config,
            &reactant,
            &best.saddle,
            &downhill,
            cell,
            0.1,
            5.0,
        ) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("band {error}");
                continue;
            }
        };
        let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
        let zero_v = zero
            .as_ref()
            .map(|profile| profile.barrier)
            .unwrap_or(f64::NAN);
        let ok = converged
            && force <= 2.0e-3
            && (zero_v - d_e0).abs() < 0.05
            && down_energy > reactant_energy - 1.5;
        let outcome = if ok { "index1" } else { "not_converged" };
        let line = format!(
            "capture,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},{zero_v:.6},,,,,{outcome},{force:.5},downhill_rel {:.3}",
            down_energy - reactant_energy
        );
        eprintln!("{line}");
        let _ = std::io::stderr().flush();
        append_checkpoint(&checkpoint, &line);
        if !ok {
            continue;
        }
        for thermal_t in [500.0_f64, 300.0] {
            if (temperature - 500.0).abs() > 1.0 && (thermal_t - temperature).abs() > 1.0 {
                continue;
            }
            let thermal = thermal_t * rgsaddle::BOLTZMANN_EV_PER_K;
            let pafi = band_config
                .pafi
                .clone()
                .with_thermal(thermal)
                .with_samples(20)
                .with_max_rounds(4)
                .with_target_error(0.01)
                .with_thermalize(20);
            let Ok((profile, planes)) = rgsaddle::pafi_profile(&held, &band, &band.arcs(), &pafi)
            else {
                continue;
            };
            let geometry = band
                .geometry(profile.arc[profile.dividing])
                .expect("dividing");
            let reactant_tangent = band.geometry(band.bounds().0).expect("end").tangent;
            let shooting = band_config
                .shooting
                .clone()
                .with_thermal(thermal)
                .with_mass(63.546);
            let starts = if planes[profile.dividing].retained.is_empty() {
                vec![geometry.positions.clone()]
            } else {
                planes[profile.dividing].retained.clone()
            };
            let transmission = rgsaddle::bennett::bennett_chandler(
                &held,
                &starts,
                geometry.positions.view(),
                geometry.tangent.view(),
                &shooting,
            );
            let kappa = transmission
                .as_ref()
                .map(|value| value.kappa)
                .unwrap_or(f64::NAN);
            let lambda = curvature_along(&held, &reactant, &reactant_tangent);
            let nu =
                rgsaddle::attempt_frequency(lambda, 63.546, rgsaddle::angular_scale_rad_s()).ok();
            let rate = nu
                .filter(|_| (0.0..=1.0).contains(&kappa))
                .map(|prefactor| prefactor * kappa * (-profile.barrier / thermal).exp());
            let line = format!(
                "pafi,{radius:.3},{orientation},capture,toward,{thermal_t},{d_e0:.6},{zero_v:.6},{:.6},{:.6},{},{},rated,{force:.5},kappa {kappa:.6}",
                profile.barrier,
                profile.barrier_error,
                num_or(&nu),
                num_or(&rate)
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
        }
    }
    eprintln!("viv {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

fn distinct_orientations(vector: [f64; 3]) -> Vec<&'static str> {
    let hat = unit3(vector);
    let mut seen = Vec::new();
    let mut out = Vec::new();
    for (axis, label) in [
        ([1.0, 0.0, 0.0], "100"),
        ([0.0, 1.0, 0.0], "010"),
        ([0.0, 0.0, 1.0], "001"),
    ] {
        let key = ((axis[0] * hat[0] + axis[1] * hat[1] + axis[2] * hat[2]).abs() * 1000.0).round()
            as i32;
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        out.push(label);
    }
    out
}

#[test]
#[ignore = "finer 4.456 PAFI and the recombination-volume sum"]
fn defect_pair_run9() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_run9.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("run9");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let a500 = measured_lattice(500.0);
    let length = repeats as f64 * a500;
    let (perfect, _) = fcc(repeats, a500);
    let (_, _, bulk) = copper.evaluate(
        perfect.view(),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    );
    // The strain-response cutoff API is not on this branch, so every shell is searched directly.
    let vector = shell_table(a500, repeats)
        .into_iter()
        .find(|(value, _, _)| (*value - 4.456).abs() < 0.05)
        .map(|(_, v, _)| v)
        .expect("4.456");
    let PairState {
        positions,
        vacancy,
        dumbbell,
        cell,
        recombined,
        ..
    } = placed(&copper, repeats, a500, vector, axis_from_label("100"), true);
    let reactant = positions;
    let reactant_energy = basin_energy(&copper, &reactant, cell);
    if !recombined {
        let held = FastHeld {
            copper: &copper,
            cell,
        };
        let seeds = local_seeds(&reactant, cell, dumbbell, vacancy, 99, 32);
        let config = survey_config(bulk);
        let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
        let mut kept: Vec<&rgsaddle::BasinSaddle> = found
            .iter()
            .filter(|saddle| {
                let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
                let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
                let plus_re = recombined_in(&saddle.plus, cell);
                let minus_re = recombined_in(&saddle.minus, cell);
                (plus_back && minus_re) || (minus_back && plus_re)
            })
            .collect();
        kept.sort_by(|left, right| {
            (left.saddle_energy - reactant_energy)
                .partial_cmp(&(right.saddle_energy - reactant_energy))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(best) = kept
            .iter()
            .find(|saddle| saddle.saddle_energy >= reactant_energy - 1.0e-4)
            .or(kept.first())
        {
            let d_e0 = best.saddle_energy - reactant_energy;
            let sign = if best.plus_energy < best.minus_energy {
                1.0
            } else {
                -1.0
            };
            let mut peak = 0.0_f64;
            for atom in 0..best.mode.len() / 3 {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
                }
                peak = peak.max(acc.sqrt());
            }
            let start = if peak > 1.0e-8 {
                &best.saddle + &(&best.mode * (sign * 0.04 / peak))
            } else {
                best.saddle.clone()
            };
            let (downhill, _) =
                rgsaddle::descend_to_floor(&held, start, cell, &config, best.saddle_energy - 0.4)
                    .expect("descent from the saddle to the floor");
            let band_config = config
                .clone()
                .with_band(
                    config
                        .band
                        .clone()
                        .with_force_tol(2.0e-3)
                        .with_time_step(0.03),
                )
                .with_max_band_steps(240)
                .with_images(21);
            if let Ok((band, force_ok, force)) = rgsaddle::segment_band(
                &held,
                &band_config,
                &reactant,
                &best.saddle,
                &downhill,
                cell,
                0.1,
                8.0,
            ) {
                let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
                let zero_v = zero.as_ref().map(|p| p.barrier).unwrap_or(f64::NAN);
                for thermal_t in [500.0_f64, 300.0] {
                    let thermal = thermal_t * rgsaddle::BOLTZMANN_EV_PER_K;
                    let pafi = band_config
                        .pafi
                        .clone()
                        .with_thermal(thermal)
                        .with_samples(48)
                        .with_max_rounds(3)
                        .with_target_error(0.01)
                        .with_thermalize(30);
                    let dense = {
                        let nodes = band.arcs();
                        let mut arcs = Vec::new();
                        for window in nodes.windows(2) {
                            arcs.push(window[0]);
                            arcs.push(0.5 * (window[0] + window[1]));
                        }
                        arcs.push(*nodes.last().unwrap_or(&0.0));
                        arcs
                    };
                    if let Ok((profile, planes)) =
                        rgsaddle::pafi_profile(&held, &band, &dense, &pafi)
                    {
                        let below = profile.dividing == 0
                            || (profile.barrier.abs() < 1.0e-4 && profile.barrier_error < 1.0e-4);
                        let outcome = if below { "below_resolution" } else { "rated" };
                        let geometry = band.geometry(profile.arc[profile.dividing]).expect("plane");
                        let reactant_tangent =
                            band.geometry(band.bounds().0).expect("reactant").tangent;
                        let shooting = band_config
                            .shooting
                            .clone()
                            .with_thermal(thermal)
                            .with_mass(63.546);
                        let starts = if planes[profile.dividing].retained.is_empty() {
                            vec![geometry.positions.clone()]
                        } else {
                            planes[profile.dividing].retained.clone()
                        };
                        let kappa = rgsaddle::bennett::bennett_chandler(
                            &held,
                            &starts,
                            geometry.positions.view(),
                            geometry.tangent.view(),
                            &shooting,
                        )
                        .map(|t| t.kappa)
                        .unwrap_or(f64::NAN);
                        let nu = rgsaddle::attempt_frequency(
                            curvature_along(&held, &reactant, &reactant_tangent),
                            63.546,
                            rgsaddle::angular_scale_rad_s(),
                        )
                        .ok();
                        let rate = nu.filter(|_| (0.0..=1.0).contains(&kappa) && !below).map(
                            |prefactor| prefactor * kappa * (-profile.barrier / thermal).exp(),
                        );
                        let line = format!(
                            "pafi,4.456,100,capture,toward,{thermal_t},{d_e0:.6},{zero_v:.6},{:.6},{:.6},{},{},{outcome},{force:.5},dividing {} images 21 force_ok {force_ok}",
                            profile.barrier,
                            profile.barrier_error,
                            num_or(&nu),
                            num_or(&rate),
                            profile.dividing
                        );
                        eprintln!("{line}");
                        append_checkpoint(&checkpoint, &line);
                    }
                }
            }
        }
    }
    // The strain-response cutoff is not on this branch. Every shell out to the
    // table limit is searched directly, per distinct dumbbell orientation.
    // Shells the live `defect_pair_viv` job already owns are left to it.
    let mut jobs = Vec::new();
    for temperature in [500.0_f64, 300.0, 700.0] {
        let a = measured_lattice(temperature);
        for (radius, vector, _) in shell_table(a, repeats) {
            if radius < 5.70 {
                continue;
            }
            for orientation in distinct_orientations(vector) {
                if scheduled_by_viv(radius, orientation, temperature) {
                    continue;
                }
                jobs.push((radius, orientation, temperature));
            }
        }
    }
    for (radius, orientation, temperature) in jobs {
        if (temperature - 500.0).abs() > 1.0
            && !capture_still_competes(&root, radius, orientation, temperature)
        {
            continue;
        }
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        let viv = std::fs::read_to_string(root.join("defect_pair_viv.csv")).unwrap_or_default();
        let key = format!("ref,{radius:.3},{orientation},{temperature},");
        if checkpoint_has(&prior, &key) || checkpoint_has(&viv, &key) {
            continue;
        }
        survey_shell(
            &copper,
            &checkpoint,
            bulk,
            repeats,
            radius,
            orientation,
            temperature,
            started,
        );
    }
    write_viv_summary(&copper);
}

fn scheduled_by_viv(radius: f64, orientation: &str, temperature: f64) -> bool {
    let r500 = radius * measured_lattice(500.0) / measured_lattice(temperature);
    let key = ((r500 * 1000.0).round() as i32, orientation);
    let inner = matches!(
        key,
        (2573, "001")
            | (3639, "010")
            | (3639, "100")
            | (3639, "001")
            | (4456, "100")
            | (4456, "010")
            | (5146, "001")
    );
    if inner {
        return true;
    }
    (temperature - 500.0).abs() < 1.0
        && matches!(
            key,
            (6302, "100") | (6302, "010") | (6807, "100") | (7277, "100") | (7719, "100")
        )
}

fn capture_still_competes(
    root: &std::path::Path,
    radius: f64,
    orientation: &str,
    temperature: f64,
) -> bool {
    let a500 = measured_lattice(500.0);
    let r500 = radius * a500 / measured_lattice(temperature);
    let text = std::fs::read_to_string(root.join("defect_pair_run9.csv")).unwrap_or_default()
        + &std::fs::read_to_string(root.join("defect_pair_viv.csv")).unwrap_or_default();
    let mut saw = false;
    let mut barrier = f64::INFINITY;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        if (shell - r500).abs() > 0.05 || fields[2] != orientation {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - 500.0).abs() > 1.0 {
            continue;
        }
        saw = true;
        if line.contains(",spontaneous,")
            || line.contains(",barrierless,")
            || line.contains("recombines on relaxation")
        {
            return true;
        }
        if let Ok(value) = fields[6].parse::<f64>()
            && value.is_finite()
        {
            barrier = barrier.min(value);
        }
    }
    saw && barrier <= 0.15
}

#[allow(clippy::too_many_arguments)]
fn survey_shell(
    copper: &FastCopper,
    checkpoint: &Mutex<File>,
    bulk: [[f64; 3]; 3],
    repeats: usize,
    radius: f64,
    orientation: &str,
    temperature: f64,
    started: Instant,
) {
    let a = measured_lattice(temperature);
    let Some(vector) = shell_table(a, repeats)
        .into_iter()
        .find(|(value, _, _)| (*value - radius).abs() < 0.05)
        .map(|(_, vector, _)| vector)
    else {
        return;
    };
    let PairState {
        positions,
        vacancy,
        dumbbell,
        cell,
        recombined,
        ..
    } = placed(
        copper,
        repeats,
        a,
        vector,
        axis_from_label(orientation),
        true,
    );
    if recombined {
        let line = format!(
            "ref,{radius:.3},{orientation},capture,toward,{temperature},0.000000,,,,,spontaneous,0,recombines on relaxation; counted in V_iv"
        );
        eprintln!("{line}");
        append_checkpoint(checkpoint, &line);
        return;
    }
    let reactant = positions;
    let reactant_energy = basin_energy(copper, &reactant, cell);
    let held = FastHeld { copper, cell };
    let seeds = local_seeds(
        &reactant,
        cell,
        dumbbell,
        vacancy,
        (temperature as u64)
            .wrapping_mul(17)
            .wrapping_add((radius * 10.0) as u64),
        32,
    );
    eprintln!(
        "run9 {radius:.3} {orientation} {temperature} seeds {} E {reactant_energy:.3} at {:.0}s",
        seeds.len(),
        started.elapsed().as_secs_f64()
    );
    let config = survey_config(bulk);
    let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
    let mut kept: Vec<&rgsaddle::BasinSaddle> = found
        .iter()
        .filter(|saddle| {
            let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
            let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
            let plus_re = recombined_in(&saddle.plus, cell);
            let minus_re = recombined_in(&saddle.minus, cell);
            (plus_back && minus_re) || (minus_back && plus_re)
        })
        .collect();
    kept.sort_by(|left, right| {
        (left.saddle_energy - reactant_energy)
            .partial_cmp(&(right.saddle_energy - reactant_energy))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if kept.is_empty() {
        let line = format!(
            "ref,{radius:.3},{orientation},capture,toward,{temperature},,,,,,not_converged,0,no recombination saddle in {} index-1 of {} seeds; reference {reactant_energy:.4}",
            found.len(),
            seeds.len()
        );
        eprintln!("{line}");
        append_checkpoint(checkpoint, &line);
        return;
    }
    let best = kept
        .iter()
        .find(|saddle| saddle.saddle_energy >= reactant_energy - 1.0e-4)
        .unwrap_or(&kept[0]);
    let d_e0 = best.saddle_energy - reactant_energy;
    let lowest = kept[0].saddle_energy - reactant_energy;
    let line = format!(
        "ref,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},,,,,index1,0,n={} lowest_relative {lowest:.4} reference {reactant_energy:.4}",
        kept.len()
    );
    eprintln!("{line}");
    append_checkpoint(checkpoint, &line);
    if rgsaddle::capture_is_spontaneous(false, Some(d_e0.min(lowest))) {
        let line = format!(
            "ref,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},,,,,spontaneous,0,re-referenced barrier {lowest:.4} eV is at most 1 meV; counted in V_iv"
        );
        eprintln!("{line}");
        append_checkpoint(checkpoint, &line);
        return;
    }
    // The isolated [100] translation barrier is 0.091 eV. A capture barrier
    // above 0.15 eV is slower than that hop at 300–700 K for a prefactor of
    // 2e12 Hz, so the band is not required to decide the comparison.
    if d_e0 > 0.15 {
        let line = format!(
            "capture,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},,,,,barrier_loses,0,lowest connecting barrier {d_e0:.4} eV is above the 0.091 eV translation"
        );
        eprintln!("{line}");
        append_checkpoint(checkpoint, &line);
        return;
    }
    let sign = if best.plus_energy < best.minus_energy {
        1.0
    } else {
        -1.0
    };
    let mut peak = 0.0_f64;
    for atom in 0..best.mode.len() / 3 {
        let mut acc = 0.0;
        for c in 0..3 {
            acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
        }
        peak = peak.max(acc.sqrt());
    }
    let mut step = 0.04_f64;
    let mut start = best.saddle.clone();
    if peak > 1.0e-8 {
        while step >= 0.01 {
            let trial = &best.saddle + &(&best.mode * (sign * step / peak));
            let trial_energy = held
                .eval(trial.view())
                .map(|(energy, _)| energy)
                .unwrap_or(f64::NAN);
            if trial_energy.is_finite() && trial_energy > best.saddle_energy - 0.3 {
                start = trial;
                break;
            }
            step *= 0.5;
        }
    }
    let (downhill, down_energy) = rgsaddle::descend_to_floor(
        &held,
        start.clone(),
        cell,
        &config,
        best.saddle_energy - 0.6,
    )
    .expect("descent from the saddle to the floor");
    let band_config = config
        .clone()
        .with_band(
            config
                .band
                .clone()
                .with_force_tol(2.0e-3)
                .with_time_step(0.04),
        )
        .with_max_band_steps(240)
        .with_images(13);
    let (band, converged, force) = match rgsaddle::segment_band(
        &held,
        &band_config,
        &reactant,
        &best.saddle,
        &downhill,
        cell,
        0.1,
        5.0,
    ) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("band {error}");
            return;
        }
    };
    let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
    let zero_v = zero
        .as_ref()
        .map(|profile| profile.barrier)
        .unwrap_or(f64::NAN);
    let ok = converged
        && force <= 2.0e-3
        && (zero_v - d_e0).abs() < 0.05
        && down_energy > reactant_energy - 1.5;
    let outcome = if ok { "index1" } else { "not_converged" };
    let line = format!(
        "capture,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},{zero_v:.6},,,,,{outcome},{force:.5},downhill_rel {:.3}",
        down_energy - reactant_energy
    );
    eprintln!("{line}");
    append_checkpoint(checkpoint, &line);
    if !ok {
        return;
    }
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let pafi = band_config
        .pafi
        .clone()
        .with_thermal(thermal)
        .with_samples(24)
        .with_max_rounds(3)
        .with_target_error(0.01)
        .with_thermalize(20);
    let Ok((profile, planes)) = rgsaddle::pafi_profile(&held, &band, &band.arcs(), &pafi) else {
        return;
    };
    let below =
        profile.dividing == 0 || (profile.barrier.abs() < 1.0e-4 && profile.barrier_error < 1.0e-4);
    let rated = if below { "below_resolution" } else { "rated" };
    let geometry = band
        .geometry(profile.arc[profile.dividing])
        .expect("dividing");
    let reactant_tangent = band.geometry(band.bounds().0).expect("end").tangent;
    let shooting = band_config
        .shooting
        .clone()
        .with_thermal(thermal)
        .with_mass(63.546);
    let starts = if planes[profile.dividing].retained.is_empty() {
        vec![geometry.positions.clone()]
    } else {
        planes[profile.dividing].retained.clone()
    };
    let kappa = rgsaddle::bennett::bennett_chandler(
        &held,
        &starts,
        geometry.positions.view(),
        geometry.tangent.view(),
        &shooting,
    )
    .map(|value| value.kappa)
    .unwrap_or(f64::NAN);
    let nu = rgsaddle::attempt_frequency(
        curvature_along(&held, &reactant, &reactant_tangent),
        63.546,
        rgsaddle::angular_scale_rad_s(),
    )
    .ok();
    let rate = nu
        .filter(|_| (0.0..=1.0).contains(&kappa) && !below)
        .map(|prefactor| prefactor * kappa * (-profile.barrier / thermal).exp());
    let line = format!(
        "pafi,{radius:.3},{orientation},capture,toward,{temperature},{d_e0:.6},{zero_v:.6},{:.6},{:.6},{},{},{rated},{force:.5},dividing {} kappa {kappa:.6}",
        profile.barrier,
        profile.barrier_error,
        num_or(&nu),
        num_or(&rate),
        profile.dividing
    );
    eprintln!("{line}");
    append_checkpoint(checkpoint, &line);
}

fn write_viv_summary(copper: &FastCopper) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut text = String::new();
    for name in [
        "defect_pair_viv.csv",
        "defect_pair_local.csv",
        "defect_pair_run9.csv",
        "defect_pair_tight.csv",
    ] {
        if let Ok(extra) = std::fs::read_to_string(root.join(name)) {
            text.push_str(&extra);
            text.push('\n');
        }
    }
    let mut out = File::create(root.join("defect_pair_volume.csv")).expect("volume");
    writeln!(out, "temperature_K,threshold_meV,volume_over_Omega,spontaneous_sites,kinetic_sites,demkowicz_table_I,unresolved").unwrap();
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = measured_lattice(temperature);
        let omega = a * a * a / 4.0;
        for threshold_mev in [0.5_f64, 1.0, 2.0] {
            let threshold = threshold_mev * 1.0e-3;
            let mut sites = Vec::new();
            let mut unresolved = Vec::new();
            for (radius, vector, count) in shell_table(a, repeats_of()) {
                let share = count as f64 / 3.0;
                for orientation in ["100", "010", "001"] {
                    let Some(source) =
                        orientation_with_data(&text, radius, vector, orientation, temperature)
                    else {
                        continue;
                    };
                    let barrier = lowest_barrier(&text, radius, source, temperature);
                    let recombines = text.lines().any(|line| {
                        let fields: Vec<&str> = line.split(',').collect();
                        if fields.len() < 13 {
                            return false;
                        }
                        let Ok(shell) = fields[1].parse::<f64>() else {
                            return false;
                        };
                        let Ok(temp) = fields[5].parse::<f64>() else {
                            return false;
                        };
                        (shell - radius).abs() < 0.05
                            && fields[2] == source
                            && (temp - temperature).abs() < 1.0
                            && (fields[12] == "spontaneous" || fields[12] == "barrierless")
                    });
                    let spontaneous =
                        rgsaddle::capture_is_spontaneous_within(recombines, barrier, threshold);
                    let rate = pafi_rate(&text, radius, source, temperature);
                    if spontaneous {
                        sites.push(rgsaddle::VolumeSite {
                            id: format!("{radius:.3}-{orientation}"),
                            count: share,
                            outcome: rgsaddle::SiteOutcome::Spontaneous,
                            rates: Vec::new(),
                        });
                    } else if let Some(capture) = rate {
                        let Some(away) = isolated_translation_rate(temperature) else {
                            unresolved.push(format!("{radius:.3}-{orientation}"));
                            continue;
                        };
                        sites.push(rgsaddle::VolumeSite {
                            id: format!("{radius:.3}-{orientation}"),
                            count: share,
                            outcome: rgsaddle::SiteOutcome::Kinetic,
                            rates: vec![("recombined".into(), capture), ("escaped".into(), away)],
                        });
                    } else if barrier.is_some() {
                        unresolved.push(format!("{radius:.3}-{orientation}"));
                    }
                }
            }
            let volume = rgsaddle::kinetic_recombination_volume(&sites, omega);
            match volume {
                Ok(value) => {
                    let pending = unresolved.join(" ");
                    let demkowicz = match temperature as i32 {
                        300 => "8788 atoms, Table 1: 254 ± 21 and 286 ± 28",
                        500 => "8788 atoms, Table 1: 185 ± 23 and 161 ± 15",
                        700 => "8788 atoms, Table 1: 203 ± 10 and 210 ± 21",
                        _ => "",
                    };
                    writeln!(
                        out,
                        "{temperature},{threshold_mev},{:.6},{},{},{demkowicz},{pending}",
                        value.volume / omega,
                        value.spontaneous_sites,
                        value.kinetic_sites
                    )
                    .unwrap();
                    eprintln!(
                        "V/Omega {temperature} K threshold {threshold_mev} meV = {:.4} spontaneous {} kinetic {} unresolved {}",
                        value.volume / omega,
                        value.spontaneous_sites,
                        value.kinetic_sites,
                        unresolved.len()
                    );
                }
                Err(error) => {
                    writeln!(out, "{temperature},{threshold_mev},,,,{error}").unwrap();
                }
            }
        }
    }
    writeln!(out, "note,,,,,Table 1 of the accepted manuscript of Mwungeli, Liu and Demkowicz, Phys. Rev. Materials 9, 023601 (2025), doi:10.1103/PhysRevMaterials.9.023601, section 4. The published PDF page number is not in that HTML. 8788-atom rows, first method then mean-time method, in atomic volumes: 300 K 254 ± 21 and 286 ± 28; 400 K 212 ± 12 and 210 ± 20; 500 K 185 ± 23 and 161 ± 15; 600 K 186 ± 14 and 191 ± 18; 700 K 203 ± 10 and 210 ± 21. Size series at 500 K: 4000 atoms 309 ± 18 and 287 ± 34; 5324 atoms 169 ± 17 and 180 ± 20; 6912 atoms 188 ± 11 and 191 ± 19; 16384 atoms 202 ± 13 and 206 ± 22. At 600 K: 4000 atoms 218 ± 13 and 238 ± 22; 5324 atoms 170 ± 8 and 177 ± 15; 6912 atoms 198 ± 10 and 203 ± 20; 16384 atoms 183 ± 9 and 194 ± 18. Escape for a kinetic site is the isolated [100] translation rate at that temperature; 700 K has no such rate in defect_pair_capture.csv. The strain-response cutoff is not on this branch, so shells were searched directly. A row is partial while unresolved shells remain.").unwrap();
    let _ = copper;
}

fn repeats_of() -> usize {
    8
}

#[test]
#[ignore = "sum the recombination volume from the catalogues on disk"]
fn defect_pair_volume_only() {
    write_viv_summary(&FastCopper::new());
}

fn orientation_with_data(
    text: &str,
    radius: f64,
    vector: [f64; 3],
    orientation: &str,
    temperature: f64,
) -> Option<&'static str> {
    let class = |label: &str| -> i32 {
        let axis = axis_from_label(label);
        let hat = unit3(vector);
        ((axis[0] * hat[0] + axis[1] * hat[1] + axis[2] * hat[2]).abs() * 1000.0).round() as i32
    };
    let want = class(orientation);
    let mut best: Option<&'static str> = None;
    let mut best_force = f64::INFINITY;
    let mut any: Option<&'static str> = None;
    for label in ["100", "010", "001"] {
        if class(label) != want {
            continue;
        }
        let mut has_ref = false;
        let mut force = None;
        for line in text.lines() {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 6 {
                continue;
            }
            let Ok(shell) = fields[1].parse::<f64>() else {
                continue;
            };
            let Ok(temp) = fields[5].parse::<f64>() else {
                continue;
            };
            if (shell - radius).abs() > 0.05
                || fields[2] != label
                || (temp - temperature).abs() > 1.0
            {
                continue;
            }
            if fields[0] == "ref" || fields[0] == "survey" {
                has_ref = true;
            }
            if fields[0] == "pafi"
                && fields.len() > 13
                && fields[12] == "rated"
                && let Ok(value) = fields[13].parse::<f64>()
                && value.is_finite()
            {
                force = Some(force.map_or(value, |prior: f64| prior.min(value)));
            }
        }
        if has_ref {
            any = Some(label);
        }
        if let Some(value) = force
            && value < best_force
        {
            best_force = value;
            best = Some(label);
        }
    }
    best.or(any)
}

fn lowest_barrier(text: &str, radius: f64, orientation: &str, temperature: f64) -> Option<f64> {
    let mut best = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        if (shell - radius).abs() > 0.05 || fields[2] != orientation {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 {
            continue;
        }
        if let Ok(value) = fields[6].parse::<f64>()
            && value.is_finite()
        {
            best = Some(best.map_or(value, |prior: f64| prior.min(value)));
        }
        if let Some(pos) = line.find("lowest_relative ")
            && let Ok(value) = line[pos + "lowest_relative ".len()..]
                .split_whitespace()
                .next()
                .unwrap_or("")
                .parse::<f64>()
            && value.is_finite()
        {
            best = Some(best.map_or(value, |prior: f64| prior.min(value)));
        }
    }
    best
}

fn isolated_translation_rate(temperature: f64) -> Option<f64> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_capture.csv");
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 12 || fields[0] != "pafi" || fields[1] != "isolated" {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 {
            continue;
        }
        if let Ok(rate) = fields[11].parse::<f64>()
            && rate.is_finite()
            && rate > 0.0
        {
            return Some(rate);
        }
    }
    None
}

fn descended(text: &str, radius: f64, orientation: &str, temperature: f64) -> bool {
    let mut saw = false;
    let mut dropped = false;
    for line in text.lines() {
        if !line.starts_with("capture,") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (shell - radius).abs() > 0.05
            || fields[2] != orientation
            || (temp - temperature).abs() > 1.0
        {
            continue;
        }
        saw = true;
        if let Some(pos) = line.find("downhill_rel ")
            && let Ok(value) = line[pos + "downhill_rel ".len()..]
                .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
                .next()
                .unwrap_or("")
                .parse::<f64>()
            && value < -0.02
        {
            dropped = true;
        }
    }
    !saw || dropped
}

fn pafi_rate(text: &str, radius: f64, orientation: &str, temperature: f64) -> Option<f64> {
    if !descended(text, radius, orientation, temperature) {
        return None;
    }
    for line in text.lines() {
        if !line.starts_with("pafi,") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13 {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        if (shell - radius).abs() > 0.05 || fields[2] != orientation {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 || fields[12] == "below_resolution" {
            continue;
        }
        if let (Ok(barrier), Ok(error)) = (fields[8].parse::<f64>(), fields[9].parse::<f64>())
            && barrier.abs() < 1.0e-4
            && error < 1.0e-4
        {
            continue;
        }
        if let Ok(rate) = fields[11].parse::<f64>()
            && rate.is_finite()
            && rate > 0.0
        {
            return Some(rate);
        }
    }
    None
}

const TIGHT_FORCE_EV_PER_A: f64 = 1.5e-3;

struct TightPoint {
    energy: f64,
    force: f64,
    index1: bool,
    positions: Array1<f64>,
    mode: Array1<f64>,
}

fn tight_dimer(
    copper: &FastCopper,
    cell: [[f64; 3]; 3],
    seed: Array1<f64>,
    mode: Array1<f64>,
) -> Option<TightPoint> {
    let norm = mode.dot(&mode).sqrt();
    if norm < 1.0e-8 {
        return None;
    }
    let search = MinModeConfig::default()
        .with_force_tol(TIGHT_FORCE_EV_PER_A)
        .with_max_move(0.08)
        .with_dr(1.0e-3)
        .with_max_rotations(12);
    let dimer_config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(0));
    let held = FastHeld { copper, cell };
    let mut dimer = SolidDimer::new(dimer_config, seed, cell, &mode / norm).ok()?;
    let report = dimer.run(&held, 200).ok()?;
    let positions = dimer.positions().to_owned();
    let (energy, _) = held.eval(positions.view()).ok()?;
    let index1 = report.status == MinModeStatus::Converged
        && !report.rejected
        && report.negative_modes == Some(1)
        && report.max_force <= TIGHT_FORCE_EV_PER_A * 1.01;
    Some(TightPoint {
        energy,
        force: report.max_force,
        index1,
        positions,
        mode: dimer.mode().to_owned(),
    })
}

fn saddle_assignment(
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    dumbbell: [usize; 2],
) -> &'static str {
    if axis_dot(axis_of(reactant, dumbbell), axis_of(saddle, dumbbell)) < 0.85 {
        "rotation"
    } else {
        "translation"
    }
}

fn write_tight(file: &Mutex<File>, line: &str) {
    eprintln!("{line}");
    let _ = std::io::stderr().flush();
    append_checkpoint(file, line);
}

#[test]
#[ignore = "1.5e-3 eV/A rotation, capture, and channel assignment on each lattice"]
fn defect_pair_tight() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_tight.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("tight");
        writeln!(header, "kind,shell,orientation,seed,temperature_K,dE0_eV,max_force_eV_A,force_tol_eV_A,index1,assignment,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let mut isolated: Vec<(f64, f64, f64)> = Vec::new();
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = measured_lattice(temperature);
        let length = repeats as f64 * a;
        let cell_box = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
        let (perfect, _) = fcc(repeats, a);
        let (_, _, _bulk) = copper.evaluate(perfect.view(), cell_box);
        let job = rotation_job(&copper, repeats, a);
        let dumbbell = [job.mobile[0], job.mobile[1]];
        let mut reactant = job.positions.clone();
        reactant = settle(&copper, reactant, job.cell, 2.0e-4).expect("reactant relaxation");
        let held = FastHeld {
            copper: &copper,
            cell: job.cell,
        };
        let (reactant_energy, _) = held
            .eval(reactant.view())
            .unwrap_or((f64::NAN, Array1::zeros(1)));
        let gap = dist(
            point_of(&reactant, dumbbell[0]),
            point_of(&reactant, dumbbell[1]),
        );
        if !(1.2..2.5).contains(&gap) || !reactant_energy.is_finite() {
            write_tight(
                &checkpoint,
                &format!(
                    "rotation,isolated,100,rotate,{temperature},,,,,false,,dumbbell gap {gap:.3} is not bound"
                ),
            );
            continue;
        }
        let center = centroid(&reactant, dumbbell);
        let half = dist(point_of(&reactant, dumbbell[0]), center).max(0.9);
        let diagonal = unit3([1.0, 1.0, 0.0]);
        let mut hint = reactant.clone();
        let mut mode = Array1::zeros(reactant.len());
        for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
            for axis in 0..3 {
                hint[3 * atom + axis] = center[axis] + sign * half * diagonal[axis];
                mode[3 * atom + axis] = sign * ([0.0, 1.0, 0.0][axis] - [1.0, 0.0, 0.0][axis]);
            }
        }
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        if !checkpoint_has(
            &prior,
            &format!("rotation,isolated,100,rotate,{temperature},"),
        ) && let Some(hit) = tight_dimer(&copper, job.cell, hint, mode)
        {
            let barrier = hit.energy - reactant_energy;
            let assignment =
                saddle_assignment(&reactant, &hit.positions, [job.mobile[0], job.mobile[1]]);
            write_tight(
                &checkpoint,
                &format!(
                    "rotation,isolated,100,rotate,{temperature},{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},own lattice a={a:.6}; dimer force tolerance",
                    hit.force, hit.index1
                ),
            );
            if hit.index1 {
                isolated.push((temperature, barrier, f64::NAN));
            }
        }
        if !checkpoint_has(
            &prior,
            &format!("translation,isolated,100,translate,{temperature},"),
        ) {
            let mut kick = Array1::zeros(reactant.len());
            for atom in [job.mobile[0], job.mobile[1]] {
                kick[3 * atom] = 0.45;
            }
            if let Some(hit) = tight_dimer(&copper, job.cell, &reactant + &kick, kick) {
                let barrier = hit.energy - reactant_energy;
                let assignment =
                    saddle_assignment(&reactant, &hit.positions, [job.mobile[0], job.mobile[1]]);
                write_tight(
                    &checkpoint,
                    &format!(
                        "translation,isolated,100,translate,{temperature},{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},own lattice a={a:.6}; dimer force tolerance",
                        hit.force, hit.index1
                    ),
                );
                if let Some(row) = isolated
                    .iter_mut()
                    .find(|row| (row.0 - temperature).abs() < 1.0)
                {
                    row.2 = barrier;
                }
            }
        }
        let _ = reactant;
    }
    for temperature in [300.0_f64, 500.0, 700.0] {
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        let key = format!("translation,isolated,100,neighbor,{temperature},");
        if checkpoint_has(&prior, &key) {
            continue;
        }
        let a = measured_lattice(temperature);
        let mechanisms = find_mechanisms(&copper, repeats, a);
        let Some(translate) = mechanisms.iter().find(|item| item.kind == "translate") else {
            write_tight(
                &checkpoint,
                &format!(
                    "translation,isolated,100,neighbor,{temperature},,,,,false,,no translation mechanism at 0.04 eV/A"
                ),
            );
            continue;
        };
        let job = rotation_job(&copper, repeats, a);
        let dumbbell = [job.mobile[0], job.mobile[1]];
        let mut reactant = job.positions.clone();
        reactant = settle(&copper, reactant, job.cell, 2.0e-4).expect("reactant relaxation");
        let held = FastHeld {
            copper: &copper,
            cell: job.cell,
        };
        let (reactant_energy, _) = held
            .eval(reactant.view())
            .unwrap_or((f64::NAN, Array1::zeros(1)));
        let center = centroid(&reactant, dumbbell);
        let Some(hop) = mapped_hops(
            &reactant,
            job.cell,
            center,
            axis_of(&reactant, dumbbell),
            std::slice::from_ref(translate),
        )
        .into_iter()
        .next() else {
            write_tight(
                &checkpoint,
                &format!(
                    "translation,isolated,100,neighbor,{temperature},,,,,false,,translation mechanism did not map"
                ),
            );
            continue;
        };
        let (Some(hint), Some(mode)) = (hop.saddle, hop.mode) else {
            continue;
        };
        if let Some(hit) = tight_dimer(&copper, job.cell, hint, mode) {
            let barrier = hit.energy - reactant_energy;
            let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
            write_tight(
                &checkpoint,
                &format!(
                    "translation,isolated,100,neighbor,{temperature},{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},neighbor kick polished from 0.04 eV/A; own lattice a={a:.6}",
                    hit.force, hit.index1
                ),
            );
        }
    }
    let captures = [
        (5.146_f64, "001"),
        (4.456, "100"),
        (6.302, "100"),
        (6.807, "100"),
        (5.753, "100"),
        (8.136, "100"),
    ];
    for temperature in [500.0_f64, 300.0, 700.0] {
        let a = measured_lattice(temperature);
        let length = repeats as f64 * a;
        let (perfect, _) = fcc(repeats, a);
        let (_, _, bulk) = copper.evaluate(
            perfect.view(),
            [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        );
        for (radius_500, orientation) in captures {
            let radius = radius_500 * a / measured_lattice(500.0);
            let prior = std::fs::read_to_string(&path).unwrap_or_default();
            let key = format!("capture,{radius:.3},{orientation},capture,{temperature},");
            if checkpoint_has(&prior, &key) {
                continue;
            }
            let Some(vector) = shell_table(a, repeats)
                .into_iter()
                .find(|(value, _, _)| (*value - radius).abs() < 0.05)
                .map(|(_, vector, _)| vector)
            else {
                continue;
            };
            let PairState {
                positions,
                vacancy,
                dumbbell,
                cell,
                recombined,
                ..
            } = placed(
                &copper,
                repeats,
                a,
                vector,
                axis_from_label(orientation),
                true,
            );
            if recombined {
                write_tight(
                    &checkpoint,
                    &format!(
                        "capture,{radius:.3},{orientation},capture,{temperature},0.000000,0,{TIGHT_FORCE_EV_PER_A:.6e},false,spontaneous,recombines on relaxation"
                    ),
                );
                continue;
            }
            let reactant = positions;
            let reactant_energy = basin_energy(&copper, &reactant, cell);
            let held = FastHeld {
                copper: &copper,
                cell,
            };
            let seeds = local_seeds(&reactant, cell, dumbbell, vacancy, temperature as u64, 24);
            eprintln!(
                "tight capture {radius:.3} {orientation} {temperature} at {:.0}s",
                started.elapsed().as_secs_f64()
            );
            let found = rgsaddle::seeded_saddles(&held, &survey_config(bulk), cell, &seeds);
            let mut kept: Vec<&rgsaddle::BasinSaddle> = found
                .iter()
                .filter(|saddle| {
                    let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
                    let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
                    let plus_re = recombined_in(&saddle.plus, cell);
                    let minus_re = recombined_in(&saddle.minus, cell);
                    (plus_back && minus_re) || (minus_back && plus_re)
                })
                .collect();
            kept.sort_by(|left, right| {
                (left.saddle_energy - reactant_energy)
                    .partial_cmp(&(right.saddle_energy - reactant_energy))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let Some(best) = kept
                .iter()
                .find(|saddle| saddle.saddle_energy >= reactant_energy - 1.0e-4)
                .or(kept.first())
            else {
                write_tight(
                    &checkpoint,
                    &format!(
                        "capture,{radius:.3},{orientation},capture,{temperature},,,,,false,,no connecting saddle in {} index-1 of {} seeds at 0.04 eV/A before the 1.5e-3 polish",
                        found.len(),
                        seeds.len()
                    ),
                );
                continue;
            };
            let loose = best.saddle_energy - reactant_energy;
            let Some(hit) = tight_dimer(&copper, cell, best.saddle.clone(), best.mode.clone())
            else {
                write_tight(
                    &checkpoint,
                    &format!(
                        "capture,{radius:.3},{orientation},capture,{temperature},{loose:.6},,,{TIGHT_FORCE_EV_PER_A:.6e},false,,polish failed; loose barrier {loose:.6} at 0.04 eV/A"
                    ),
                );
                continue;
            };
            let barrier = hit.energy - reactant_energy;
            let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
            write_tight(
                &checkpoint,
                &format!(
                    "capture,{radius:.3},{orientation},capture,{temperature},{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},loose 0.04 eV/A barrier {loose:.6}; polished at 1.5e-3",
                    hit.force, hit.index1
                ),
            );
        }
    }
    let a500 = measured_lattice(500.0);
    let mechanisms = find_mechanisms(&copper, repeats, a500);
    eprintln!(
        "tight mechanisms {} at {:.0}s",
        mechanisms.len(),
        started.elapsed().as_secs_f64()
    );
    let length = repeats as f64 * a500;
    let (perfect, _) = fcc(repeats, a500);
    let (_, _, _bulk) = copper.evaluate(
        perfect.view(),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    );
    for (radius, orientation) in bound_orientations() {
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        if checkpoint_has(&prior, &format!("channel,{radius:.3},{orientation},")) {
            continue;
        }
        let Some(vector) = shell_table(a500, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.05)
            .map(|(_, vector, _)| vector)
        else {
            continue;
        };
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            recombined,
            ..
        } = placed(
            &copper,
            repeats,
            a500,
            vector,
            axis_from_label(&orientation),
            true,
        );
        if recombined {
            write_tight(
                &checkpoint,
                &format!(
                    "channel,{radius:.3},{orientation},none,500,,,,,false,spontaneous,recombines on relaxation"
                ),
            );
            continue;
        }
        let reactant = positions;
        let reactant_energy = basin_energy(&copper, &reactant, cell);
        let center = centroid(&reactant, dumbbell);
        let axis = axis_of(&reactant, dumbbell);
        let hops = mapped_hops(&reactant, cell, center, axis, &mechanisms);
        if hops.is_empty() {
            write_tight(
                &checkpoint,
                &format!("channel,{radius:.3},{orientation},none,500,,,,,false,,no mapped seed"),
            );
            continue;
        }
        for hop in hops {
            let Some(hint) = hop.saddle.clone() else {
                continue;
            };
            let Some(mode) = hop.mode.clone() else {
                continue;
            };
            let before = dist(centroid(&reactant, dumbbell), vacancy);
            let after = dist(centroid(&hop.product, dumbbell), vacancy);
            let direction = if after + 0.3 < before {
                "toward"
            } else if after > before + 0.3 {
                "away"
            } else {
                "tangential"
            };
            let Some(hit) = tight_dimer(&copper, cell, hint, mode) else {
                continue;
            };
            let barrier = hit.energy - reactant_energy;
            let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
            let mut note = format!(
                "seed {} landed on {assignment}; direction {direction}",
                hop.name
            );
            if (radius - 3.639).abs() < 0.05
                && orientation == "010"
                && direction == "away"
                && (barrier - 0.135).abs() > 0.01
            {
                note.push_str("; did not reach the stored 0.135 eV saddle");
            }
            if assignment == "rotation" && hop.name == "translate" {
                note.push_str("; tangential or away guess is the rotation saddle");
            }
            write_tight(
                &checkpoint,
                &format!(
                    "channel,{radius:.3},{orientation},{},500,{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},{note}",
                    hop.name, hit.force, hit.index1
                ),
            );
        }
    }
    eprintln!("tight {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

fn cubic_perp(axis: [f64; 3]) -> [f64; 3] {
    let ax = axis[0].abs();
    let ay = axis[1].abs();
    let az = axis[2].abs();
    if ax <= ay && ax <= az {
        [1.0, 0.0, 0.0]
    } else if ay <= az {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn rotation_hint(reactant: &Array1<f64>, dumbbell: [usize; 2]) -> (Array1<f64>, Array1<f64>) {
    let center = centroid(reactant, dumbbell);
    let axis = axis_of(reactant, dumbbell);
    let perp = cubic_perp(axis);
    let diagonal = unit3([axis[0] + perp[0], axis[1] + perp[1], axis[2] + perp[2]]);
    let half = dist(point_of(reactant, dumbbell[0]), center).max(0.9);
    let mut hint = reactant.clone();
    let mut mode = Array1::zeros(reactant.len());
    for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
        for c in 0..3 {
            hint[3 * atom + c] = center[c] + sign * half * diagonal[c];
            mode[3 * atom + c] = sign * (perp[c] - axis[c]);
        }
    }
    (hint, mode)
}

#[test]
#[ignore = "rotation-seeded channels at 1.5e-3 eV/A"]
fn defect_pair_rotate_channels() {
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_rotate_channels.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("rotate channels");
        writeln!(header, "kind,shell,orientation,seed,temperature_K,dE0_eV,max_force_eV_A,force_tol_eV_A,index1,assignment,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let a = measured_lattice(500.0);
    let shells = [
        (3.639_f64, "010"),
        (3.639, "100"),
        (4.456, "100"),
        (5.146, "001"),
        (6.807, "001"),
        (7.277, "010"),
        (7.719, "100"),
        (7.719, "001"),
        (8.136, "001"),
    ];
    for (radius, orientation) in shells {
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        if checkpoint_has(
            &prior,
            &format!("channel,{radius:.3},{orientation},rotate,500,"),
        ) {
            continue;
        }
        let Some(vector) = shell_table(a, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.05)
            .map(|(_, vector, _)| vector)
        else {
            continue;
        };
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            recombined,
            ..
        } = placed(
            &copper,
            repeats,
            a,
            vector,
            axis_from_label(orientation),
            true,
        );
        if recombined {
            write_tight(
                &checkpoint,
                &format!(
                    "channel,{radius:.3},{orientation},rotate,500,,,,,false,spontaneous,recombines on relaxation"
                ),
            );
            continue;
        }
        let reactant = positions;
        let reactant_energy = basin_energy(&copper, &reactant, cell);
        let (hint, mode) = rotation_hint(&reactant, dumbbell);
        let Some(hit) = tight_dimer(&copper, cell, hint, mode) else {
            continue;
        };
        let barrier = hit.energy - reactant_energy;
        let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
        let center = centroid(&reactant, dumbbell);
        let saddle_center = centroid(&hit.positions, dumbbell);
        let shift = dist(saddle_center, center);
        let away = {
            let before = dist(center, vacancy);
            let after = dist(saddle_center, vacancy);
            if after > before + 0.15 {
                "away"
            } else if after + 0.15 < before {
                "toward"
            } else {
                "tangential"
            }
        };
        let mut note = format!(
            "<110> rotation seed; center shift {shift:.3} A; saddle sits {away} of the vacancy"
        );
        if (radius - 3.639).abs() < 0.05 && orientation == "010" && (barrier - 0.135).abs() > 0.01 {
            note.push_str("; this seed did not reach the stored 0.135 eV saddle");
        }
        write_tight(
            &checkpoint,
            &format!(
                "channel,{radius:.3},{orientation},rotate,500,{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},{note}",
                hit.force, hit.index1
            ),
        );
        if (radius - 3.639).abs() < 0.05 && orientation == "010" {
            let from_vac = {
                let raw = [
                    center[0] - vacancy[0],
                    center[1] - vacancy[1],
                    center[2] - vacancy[2],
                ];
                unit3(raw)
            };
            let mut kick = Array1::zeros(reactant.len());
            for atom in dumbbell {
                for c in 0..3 {
                    kick[3 * atom + c] = 0.45 * from_vac[c];
                }
            }
            if let Some(hit) = tight_dimer(&copper, cell, &reactant + &kick, kick.clone()) {
                let barrier = hit.energy - reactant_energy;
                let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
                let mut note = "away translation guess".to_string();
                if (barrier - 0.135).abs() > 0.01 {
                    note.push_str("; did not reach the stored 0.135 eV saddle");
                } else {
                    note.push_str("; reached the stored 0.135 eV saddle");
                }
                write_tight(
                    &checkpoint,
                    &format!(
                        "channel,{radius:.3},{orientation},translate-away,500,{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},{note}",
                        hit.force, hit.index1
                    ),
                );
            }
        }
    }
}

#[test]
#[ignore = "polish the isolated translation from the neighbor mechanism at 1.5e-3 eV/A"]
fn defect_pair_translation() {
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_translation.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("translation");
        writeln!(header, "kind,shell,orientation,seed,temperature_K,dE0_eV,max_force_eV_A,force_tol_eV_A,index1,assignment,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = measured_lattice(temperature);
        let mechanisms = find_mechanisms(&copper, repeats, a);
        let Some(translate) = mechanisms.iter().find(|item| item.kind == "translate") else {
            write_tight(
                &checkpoint,
                &format!(
                    "translation,isolated,100,neighbor,{temperature},,,,,false,,no translation mechanism"
                ),
            );
            continue;
        };
        let job = rotation_job(&copper, repeats, a);
        let dumbbell = [job.mobile[0], job.mobile[1]];
        let mut reactant = job.positions.clone();
        reactant = settle(&copper, reactant, job.cell, 2.0e-4).expect("reactant relaxation");
        let held = FastHeld {
            copper: &copper,
            cell: job.cell,
        };
        let (reactant_energy, _) = held
            .eval(reactant.view())
            .unwrap_or((f64::NAN, Array1::zeros(1)));
        let center = centroid(&reactant, dumbbell);
        let Some(hop) = mapped_hops(
            &reactant,
            job.cell,
            center,
            axis_of(&reactant, dumbbell),
            std::slice::from_ref(translate),
        )
        .into_iter()
        .next() else {
            continue;
        };
        let (Some(hint), Some(mode)) = (hop.saddle, hop.mode) else {
            continue;
        };
        let Some(hit) = tight_dimer(&copper, job.cell, hint, mode) else {
            continue;
        };
        let barrier = hit.energy - reactant_energy;
        let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
        let stored = if (temperature - 300.0).abs() < 1.0 {
            0.094614
        } else if (temperature - 500.0).abs() < 1.0 {
            0.090966
        } else {
            f64::NAN
        };
        let delta = if stored.is_finite() {
            format!(
                "; stored {:.6} differs by {:.6} eV",
                stored,
                (barrier - stored).abs()
            )
        } else {
            String::new()
        };
        write_tight(
            &checkpoint,
            &format!(
                "translation,isolated,100,neighbor,{temperature},{barrier:.6},{:.6e},{TIGHT_FORCE_EV_PER_A:.6e},{},{assignment},neighbor mechanism on lattice a={a:.6}{delta}",
                hit.force, hit.index1
            ),
        );
    }
}

fn shoot_product(
    held: &FastHeld,
    starts: &[Array1<f64>],
    positions: ArrayView1<f64>,
    normal: ArrayView1<f64>,
    shooting: &rgsaddle::ShootingConfig,
) -> (f64, f64, &'static str) {
    let forward = rgsaddle::bennett::bennett_chandler(held, starts, positions, normal, shooting);
    let kappa = forward
        .as_ref()
        .map(|value| value.kappa)
        .unwrap_or(f64::NAN);
    let error = forward
        .as_ref()
        .map(|value| value.error)
        .unwrap_or(f64::NAN);
    (kappa, error, "normal from reactant to product")
}

fn catalogue_text() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut text = String::new();
    for name in [
        "defect_pair_capture.csv",
        "defect_pair_viv.csv",
        "defect_pair_local.csv",
        "defect_pair_run9.csv",
        "defect_pair_tight.csv",
        "defect_pair_run11.csv",
    ] {
        if let Ok(extra) = std::fs::read_to_string(root.join(name)) {
            text.push_str(&extra);
            text.push('\n');
        }
    }
    text
}

fn pafi_kappa_unresolved(text: &str, temperature: f64) -> bool {
    text.lines().any(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13
            || fields[0] != "pafi"
            || fields[1] != "isolated"
            || fields[3] != "translate"
        {
            return false;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            return false;
        };
        if (temp - temperature).abs() > 1.0 {
            return false;
        }
        fields.iter().any(|field| {
            field
                .split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .any(|pair| {
                    pair[0] == "error"
                        && pair[1]
                            .trim_end_matches(|c: char| !c.is_ascii_digit() && c != '.')
                            .parse::<f64>()
                            .ok()
                            .is_some_and(|error| error > 0.1)
                })
        })
    })
}

fn md_channel_rate(temperature: f64) -> Option<f64> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_diffusivity.csv");
    let text = std::fs::read_to_string(path).ok()?;
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 {
            continue;
        }
        let Ok(temp) = fields[0].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 {
            continue;
        }
        let jumps = fields[5].parse::<f64>().ok()?;
        if jumps < 20.0 {
            return None;
        }
        let rate = fields[6].parse::<f64>().ok()?;
        return (rate.is_finite() && rate > 0.0).then_some(rate);
    }
    None
}

fn migration_rate(text: &str, temperature: f64) -> Option<f64> {
    let mut best = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13
            || fields[0] != "pafi"
            || fields[1] != "isolated"
            || fields[3] != "translate"
        {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 || fields[12] != "rated" {
            continue;
        }
        if let Ok(rate) = fields[11].parse::<f64>()
            && rate.is_finite()
            && rate > 0.0
        {
            best = Some(rate);
        }
    }
    best
}

fn cubic_moduli(copper: &FastCopper, a: f64) -> (f64, f64, f64) {
    let (perfect, cell) = fcc(2, a);
    let g = 1.0e-3;
    let stressed = |factor: f64, shear: bool| {
        let mut positions = perfect.clone();
        let mut cell = cell;
        if shear {
            for atom in 0..positions.len() / 3 {
                positions[3 * atom] += factor * positions[3 * atom + 1];
            }
            cell[0][1] += factor * cell[1][1];
        } else {
            for atom in 0..positions.len() / 3 {
                positions[3 * atom] *= 1.0 + factor;
            }
            cell[0][0] *= 1.0 + factor;
        }
        copper.evaluate(positions.view(), cell).2
    };
    let plus = stressed(g, false);
    let minus = stressed(-g, false);
    let c11 = (plus[0][0] - minus[0][0]) / (2.0 * g);
    let shear_plus = stressed(g, true);
    let shear_minus = stressed(-g, true);
    let c44 = (shear_plus[0][1] - shear_minus[0][1]) / (2.0 * g);
    let c12 = (plus[1][1] - minus[1][1]) / (2.0 * g);
    (c11, c44, c12)
}

fn strain_kick(strain: [[f64; 3]; 3], axis: [f64; 3]) -> [f64; 3] {
    [
        strain[0][0] * axis[0] + strain[0][1] * axis[1] + strain[0][2] * axis[2],
        strain[1][0] * axis[0] + strain[1][1] * axis[1] + strain[1][2] * axis[2],
        strain[2][0] * axis[0] + strain[2][1] * axis[1] + strain[2][2] * axis[2],
    ]
}

#[test]
#[ignore = "700 K translation, unresolved shells, and the first-passage volume"]
fn defect_pair_run11() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let repeats = 8usize;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_run11.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("run11");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let a700 = measured_lattice(700.0);
    let prior = std::fs::read_to_string(&path).unwrap_or_default();
    if !checkpoint_has(&prior, "pafi,isolated,100,translate,toward,700,") {
        let length = repeats as f64 * a700;
        let (perfect, cell) = fcc(repeats, a700);
        let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
        let mechanisms = find_mechanisms(&copper, repeats, a700);
        if let Some(translate) = mechanisms.iter().find(|item| item.kind == "translate") {
            let job = rotation_job(&copper, repeats, a700);
            let dumbbell = [job.mobile[0], job.mobile[1]];
            let mut reactant = job.positions.clone();
            reactant = settle(&copper, reactant, job.cell, 2.0e-4).expect("reactant relaxation");
            let held = FastHeld {
                copper: &copper,
                cell: job.cell,
            };
            let (reactant_energy, _) = held
                .eval(reactant.view())
                .unwrap_or((f64::NAN, Array1::zeros(1)));
            let center = centroid(&reactant, dumbbell);
            if let Some(hop) = mapped_hops(
                &reactant,
                job.cell,
                center,
                axis_of(&reactant, dumbbell),
                std::slice::from_ref(translate),
            )
            .into_iter()
            .next()
                && let (Some(hint), Some(mode)) = (hop.saddle, hop.mode)
                && let Some(hit) = tight_dimer(&copper, job.cell, hint, mode)
            {
                let mut product = hop.product;
                product = settle(&copper, product, job.cell, 2.0e-4).expect("product relaxation");
                record_pafi(
                    &copper,
                    &checkpoint,
                    bulk,
                    "isolated",
                    "100",
                    "translate",
                    "toward",
                    700.0,
                    &reactant,
                    reactant_energy,
                    &hit.positions,
                    &hit.mode,
                    &product,
                    job.cell,
                    true,
                );
                let _ = length;
            }
        }
    }
    let jobs = [
        (4.456_f64, "100", 700.0_f64),
        (5.146, "001", 700.0),
        (6.807, "001", 500.0),
        (6.807, "001", 700.0),
        (5.146, "001", 300.0),
    ];
    for (radius_500, orientation, temperature) in jobs {
        let a = measured_lattice(temperature);
        let radius = radius_500 * a / measured_lattice(500.0);
        let prior = std::fs::read_to_string(&path).unwrap_or_default();
        if checkpoint_has(
            &prior,
            &format!("pafi,{radius:.3},{orientation},capture,toward,{temperature},"),
        ) || checkpoint_has(
            &prior,
            &format!("capture,{radius:.3},{orientation},capture,toward,{temperature},"),
        ) {
            continue;
        }
        let length = repeats as f64 * a;
        let (perfect, _) = fcc(repeats, a);
        let (_, _, bulk) = copper.evaluate(
            perfect.view(),
            [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
        );
        let Some(vector) = shell_table(a, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.05)
            .map(|(_, vector, _)| vector)
        else {
            continue;
        };
        let PairState {
            positions,
            vacancy,
            dumbbell,
            cell,
            recombined,
            ..
        } = placed(
            &copper,
            repeats,
            a,
            vector,
            axis_from_label(orientation),
            true,
        );
        if recombined {
            write_tight(
                &checkpoint,
                &format!(
                    "capture,{radius:.3},{orientation},capture,toward,{temperature},0.000000,,,,,spontaneous,0,recombines on relaxation"
                ),
            );
            continue;
        }
        let reactant = positions;
        let reactant_energy = basin_energy(&copper, &reactant, cell);
        let held = FastHeld {
            copper: &copper,
            cell,
        };
        let seeds = local_seeds(
            &reactant,
            cell,
            dumbbell,
            vacancy,
            temperature as u64 + 11,
            32,
        );
        eprintln!(
            "run11 {radius:.3} {orientation} {temperature} at {:.0}s",
            started.elapsed().as_secs_f64()
        );
        let found = rgsaddle::seeded_saddles(&held, &survey_config(bulk), cell, &seeds);
        let mut kept: Vec<&rgsaddle::BasinSaddle> = found
            .iter()
            .filter(|saddle| {
                let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
                let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
                let plus_re = recombined_in(&saddle.plus, cell);
                let minus_re = recombined_in(&saddle.minus, cell);
                (plus_back && minus_re) || (minus_back && plus_re)
            })
            .collect();
        kept.sort_by(|left, right| {
            (left.saddle_energy - reactant_energy)
                .partial_cmp(&(right.saddle_energy - reactant_energy))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let Some(best) = kept
            .iter()
            .find(|saddle| saddle.saddle_energy >= reactant_energy - 1.0e-4)
            .or(kept.first())
        else {
            write_tight(
                &checkpoint,
                &format!(
                    "capture,{radius:.3},{orientation},capture,toward,{temperature},,,,,,not_converged,0,no connecting saddle in {} index-1 of 32 seeds",
                    found.len()
                ),
            );
            continue;
        };
        let Some(hit) = tight_dimer(&copper, cell, best.saddle.clone(), best.mode.clone()) else {
            continue;
        };
        let sign = if best.plus_energy < best.minus_energy {
            1.0
        } else {
            -1.0
        };
        let mut peak = 0.0_f64;
        for atom in 0..hit.mode.len() / 3 {
            let mut acc = 0.0;
            for c in 0..3 {
                acc += hit.mode[3 * atom + c] * hit.mode[3 * atom + c];
            }
            peak = peak.max(acc.sqrt());
        }
        let start = if peak > 1.0e-8 {
            &hit.positions + &(&hit.mode * (sign * 0.04 / peak))
        } else {
            hit.positions.clone()
        };
        let config = survey_config(bulk);
        let (downhill, down_energy) =
            rgsaddle::descend_to_floor(&held, start, cell, &config, hit.energy - 0.6)
                .expect("descent from the saddle to the floor");
        if down_energy > reactant_energy - 0.2 {
            write_tight(
                &checkpoint,
                &format!(
                    "capture,{radius:.3},{orientation},capture,toward,{temperature},{:.6},,,,,not_converged,0,descent did not leave the saddle; downhill_rel {:.3}",
                    hit.energy - reactant_energy,
                    down_energy - reactant_energy
                ),
            );
            continue;
        }
        record_pafi(
            &copper,
            &checkpoint,
            bulk,
            &format!("{radius:.3}"),
            orientation,
            "capture",
            "toward",
            temperature,
            &reactant,
            reactant_energy,
            &hit.positions,
            &hit.mode,
            &downhill,
            cell,
            false,
        );
    }
    strain_shells(&copper, &checkpoint, repeats, started);
    assemble_passage();
    eprintln!("run11 {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

#[allow(clippy::too_many_arguments)]
fn record_pafi(
    copper: &FastCopper,
    checkpoint: &Mutex<File>,
    bulk: [[f64; 3]; 3],
    shell: &str,
    orientation: &str,
    hop: &str,
    direction: &str,
    temperature: f64,
    reactant: &Array1<f64>,
    reactant_energy: f64,
    saddle: &Array1<f64>,
    _mode: &Array1<f64>,
    downhill: &Array1<f64>,
    cell: [[f64; 3]; 3],
    append_capture_catalogue: bool,
) {
    let held = FastHeld { copper, cell };
    let config = survey_config(bulk);
    let segment = config
        .band
        .clone()
        .with_force_tol(2.0e-3)
        .with_time_step(0.04)
        .with_spring(rgsaddle::SpringKind::EnergyWeighted {
            k_min: 0.1,
            k_max: 5.0,
        });
    let config = config
        .with_band(segment)
        .with_max_band_steps(240)
        .with_images(13);
    let (band, converged, force) = match rgsaddle::segment_band(
        &held, &config, reactant, saddle, downhill, cell, 0.1, 5.0,
    ) {
        Ok(value) => value,
        Err(error) => {
            write_tight(
                checkpoint,
                &format!(
                    "capture,{shell},{orientation},{hop},{direction},{temperature},,,,,,not_converged,0,band {error}"
                ),
            );
            return;
        }
    };
    let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
    let zero_v = zero
        .as_ref()
        .map(|profile| profile.barrier)
        .unwrap_or(f64::NAN);
    let d_e0 = {
        let geometry = band.geometry(band.arcs()[band.arcs().len() / 2]).ok();
        let _ = geometry;
        held.eval(saddle.view())
            .map(|(energy, _)| energy - reactant_energy)
            .unwrap_or(f64::NAN)
    };
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let pafi = config
        .pafi
        .clone()
        .with_thermal(thermal)
        .with_samples(16)
        .with_max_rounds(3)
        .with_target_error(0.01)
        .with_thermalize(20);
    let Ok((profile, planes)) = rgsaddle::pafi_profile(&held, &band, &band.arcs(), &pafi) else {
        return;
    };
    let mut dividing = profile.dividing;
    let mut dividing_note = "free-energy maximum";
    if dividing == 0 {
        let nodes = band.arcs();
        let mut best_i = 0usize;
        let mut best_e = f64::MIN;
        for (index, arc) in nodes.iter().enumerate() {
            let Ok(geometry) = band.geometry(*arc) else {
                continue;
            };
            let energy = held
                .eval(geometry.positions.view())
                .map(|(energy, _)| energy)
                .unwrap_or(f64::MIN);
            if energy > best_e {
                best_e = energy;
                best_i = index;
            }
        }
        dividing = best_i;
        dividing_note = "free-energy maximum sat on the reactant; shot from the highest band image";
    }
    let Some(arc) = band.arcs().get(dividing).copied() else {
        return;
    };
    let Ok(geometry) = band.geometry(arc) else {
        return;
    };
    let shooting = config
        .shooting
        .clone()
        .with_thermal(thermal)
        .with_mass(63.546);
    let starts = if dividing < planes.len() && !planes[dividing].retained.is_empty() {
        planes[dividing].retained.clone()
    } else {
        vec![geometry.positions.clone()]
    };
    let normal = rgsaddle::product_tangent(&band, arc).unwrap_or_else(|_| geometry.tangent.clone());
    let (kappa, kappa_error, kappa_note) = shoot_product(
        &held,
        &starts,
        geometry.positions.view(),
        normal.view(),
        &shooting,
    );
    let reactant_tangent = band
        .geometry(band.bounds().0)
        .map(|frame| frame.tangent)
        .unwrap_or(geometry.tangent.clone());
    let nu = rgsaddle::attempt_frequency(
        curvature_along(&held, reactant, &reactant_tangent),
        63.546,
        rgsaddle::angular_scale_rad_s(),
    )
    .ok();
    let rate = nu
        .filter(|_| (0.0..=1.0).contains(&kappa))
        .map(|prefactor| prefactor * kappa * (-profile.barrier / thermal).exp());
    let outcome = if (0.0..=1.0).contains(&kappa) && converged {
        "rated"
    } else {
        "not_converged"
    };
    let line = format!(
        "pafi,{shell},{orientation},{hop},{direction},{temperature},{d_e0:.6},{zero_v:.6},{:.6},{:.6},{},{},{outcome},{force:.5},kappa {kappa:.6} error {kappa_error:.6}; {kappa_note}; {dividing_note}",
        profile.barrier,
        profile.barrier_error,
        num_or(&nu),
        num_or(&rate)
    );
    write_tight(checkpoint, &line);
    if append_capture_catalogue {
        let capture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_capture.csv");
        if let Ok(mut file) = File::options().append(true).open(capture) {
            let _ = writeln!(file, "{line}");
        }
    }
}

fn strain_shells(copper: &FastCopper, checkpoint: &Mutex<File>, repeats: usize, started: Instant) {
    let temperature = 500.0_f64;
    let a = measured_lattice(temperature);
    let (c11, c44, c12) = cubic_moduli(copper, a);
    let poisson = if (c11 + c12).abs() > 1.0e-8 {
        c12 / (c11 + c12)
    } else {
        f64::NAN
    };
    eprintln!(
        "moduli C11 {c11:.4} C44 {c44:.4} C12 {c12:.4} nu {poisson:.4} at {:.0}s",
        started.elapsed().as_secs_f64()
    );
    let Ok(green) = rgsaddle::Kelvin::new(c44.max(1.0e-6), poisson) else {
        write_tight(
            checkpoint,
            &format!(
                "strain,moduli,100,none,none,{temperature},{c11:.6},{c44:.6},,,,not_converged,0,Kelvin rejected nu {poisson:.4}"
            ),
        );
        return;
    };
    let (perfect, cell) = fcc(repeats, a);
    let (_, _, bulk_stress) = copper.evaluate(perfect.view(), cell);
    let mut vacancy = Array1::zeros(perfect.len() - 3);
    for src in 3..perfect.len() {
        vacancy[src - 3] = perfect[src];
    }
    let (_, _, vac_stress) = copper.evaluate(vacancy.view(), cell);
    let volume = cell[0][0] * cell[1][1] * cell[2][2];
    let Ok(dipole) = rgsaddle::ElasticDipole::from_stress(volume, vac_stress, bulk_stress) else {
        return;
    };
    for radius in [8.533_f64, 8.913, 9.277, 9.965, 10.292, 10.608] {
        let prior = std::fs::read_to_string(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run11.csv"),
        )
        .unwrap_or_default();
        if checkpoint_has(&prior, &format!("strain,{radius:.3},100,")) {
            continue;
        }
        let Some(vector) = shell_table(a, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - radius).abs() < 0.05)
            .map(|(_, vector, _)| vector)
        else {
            continue;
        };
        let Ok(strain) = green.strain_from_dipole(vector, dipole) else {
            continue;
        };
        let PairState {
            positions,
            vacancy: vacancy_at,
            dumbbell,
            cell: pair_cell,
            recombined,
            ..
        } = placed(copper, repeats, a, vector, axis_from_label("100"), true);
        if recombined {
            write_tight(
                checkpoint,
                &format!(
                    "strain,{radius:.3},100,translate,away,{temperature},0.000000,,,,,spontaneous,0,recombines on relaxation"
                ),
            );
            continue;
        }
        let reactant = positions;
        let reactant_energy = basin_energy(copper, &reactant, pair_cell);
        let kick = strain_kick(strain, axis_of(&reactant, dumbbell));
        let mut seed = reactant.clone();
        let mut mode = Array1::zeros(reactant.len());
        let away = unit3([
            centroid(&reactant, dumbbell)[0] - vacancy_at[0],
            centroid(&reactant, dumbbell)[1] - vacancy_at[1],
            centroid(&reactant, dumbbell)[2] - vacancy_at[2],
        ]);
        for atom in dumbbell {
            for c in 0..3 {
                seed[3 * atom + c] += 0.35 * away[c] + kick[c];
                mode[3 * atom + c] = away[c];
            }
        }
        let Some(hit) = tight_dimer(copper, pair_cell, seed, mode) else {
            write_tight(
                checkpoint,
                &format!(
                    "strain,{radius:.3},100,translate,away,{temperature},,,,,,not_converged,0,strain-mapped dimer did not converge"
                ),
            );
            continue;
        };
        let barrier = hit.energy - reactant_energy;
        let free = 0.089786_f64;
        let recovered = (barrier - free).abs() < 0.005;
        let note = if recovered {
            format!(
                "strain-free translation barrier recovered; |{barrier:.4} - {free:.4}| < 0.005 eV; force tolerance 1.5e-3"
            )
        } else {
            format!(
                "strain-mapped barrier {barrier:.4} eV vs free translation {free:.4} eV; force tolerance 1.5e-3"
            )
        };
        write_tight(
            checkpoint,
            &format!(
                "strain,{radius:.3},100,translate,away,{temperature},{barrier:.6},{:.6e},1.500000e-3,{},translation,{note}",
                hit.force, hit.index1
            ),
        );
    }
}

fn capture_rate_row(
    text: &str,
    radius: f64,
    orientation: &str,
    temperature: f64,
) -> Option<(f64, f64)> {
    if !descended(text, radius, orientation, temperature) {
        return None;
    }
    for line in text.lines() {
        if !line.starts_with("pafi,") {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13 {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        if (shell - radius).abs() > 0.05 || fields[2] != orientation {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 || fields[3] != "capture" {
            continue;
        }
        let Ok(rate) = fields[11].parse::<f64>() else {
            continue;
        };
        if !(rate.is_finite() && rate > 0.0) {
            continue;
        }
        let error = fields
            .get(9)
            .and_then(|value| value.parse::<f64>().ok())
            .unwrap_or(0.0);
        let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
        let relative = if thermal > 0.0 { error / thermal } else { 0.0 };
        return Some((rate, relative));
    }
    None
}

#[allow(clippy::needless_range_loop)]
fn assemble_passage() {
    let text = catalogue_text();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut out = File::create(root.join("defect_pair_passage.csv")).expect("passage");
    writeln!(out, "temperature_K,threshold_meV,volume_over_Omega,sigma,spontaneous_sites,kinetic_sites,demkowicz_8788,cell").unwrap();
    let repeats = 8usize;
    for temperature in [300.0_f64, 500.0, 700.0] {
        let Some(k_mig) = migration_rate(&text, temperature) else {
            writeln!(out, "{temperature},,,, ,,,no isolated translation rate").unwrap();
            eprintln!("no migration rate at {temperature}");
            continue;
        };
        let a = measured_lattice(temperature);
        let omega = a * a * a / 4.0;
        let shells = shell_table(a, repeats);
        let nn = a / 2.0_f64.sqrt();
        let mut sites_of: Vec<Vec<[f64; 3]>> = shells.iter().map(|_| Vec::new()).collect();
        let limit = ((2.0 * 11.0 / a).ceil() as i32) + 1;
        for i in -limit..=limit {
            for j in -limit..=limit {
                for k in -limit..=limit {
                    if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                    let radius = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                    if let Some(index) = shells
                        .iter()
                        .position(|(value, _, _)| (value - radius).abs() < 1.0e-3)
                    {
                        sites_of[index].push(cart);
                    }
                }
            }
        }
        let nn_vecs = [
            [0.5, 0.5, 0.0],
            [0.5, -0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.5, 0.0, -0.5],
            [-0.5, 0.0, 0.5],
            [-0.5, 0.0, -0.5],
            [0.0, 0.5, 0.5],
            [0.0, 0.5, -0.5],
            [0.0, -0.5, 0.5],
            [0.0, -0.5, -0.5],
        ];
        let mut fraction = vec![vec![0.0_f64; shells.len()]; shells.len()];
        for (index, members) in sites_of.iter().enumerate() {
            if members.is_empty() {
                continue;
            }
            let mut hits = vec![0.0_f64; shells.len()];
            for site in members {
                for step in nn_vecs {
                    let next = [
                        site[0] + step[0] * a,
                        site[1] + step[1] * a,
                        site[2] + step[2] * a,
                    ];
                    let radius = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                    if let Some(other) = shells
                        .iter()
                        .position(|(value, _, _)| (value - radius).abs() < 1.0e-3)
                    {
                        hits[other] += 1.0;
                    }
                }
            }
            let norm = members.len() as f64;
            for (other, count) in hits.into_iter().enumerate() {
                fraction[index][other] = count / norm;
            }
        }
        let _ = nn;
        let demkowicz = match temperature as i32 {
            300 => "254-286",
            500 => "161-185",
            700 => "203-210",
            _ => "",
        };
        for threshold_mev in [0.5_f64, 1.0, 2.0] {
            let mut total = 0.0;
            let mut spontaneous_total = 0.0;
            let mut kinetic_total = 0.0;
            let mut samples = [0.0; 16];
            for orientation in ["100", "010", "001"] {
                let mut sites = Vec::new();
                let mut capture_index = Vec::new();
                for (index, (radius, _vector, count)) in shells.iter().enumerate() {
                    let id = format!("{radius:.3}-{orientation}");
                    let share = *count as f64 / 3.0;
                    let source = orientation_with_data(
                        &text,
                        *radius,
                        shells[index].1,
                        orientation,
                        temperature,
                    )
                    .unwrap_or(orientation);
                    let barrier = lowest_barrier(&text, *radius, source, temperature);
                    let recombines = text.lines().any(|line| {
                        let fields: Vec<&str> = line.split(',').collect();
                        if fields.len() < 13 {
                            return false;
                        }
                        let Ok(shell) = fields[1].parse::<f64>() else {
                            return false;
                        };
                        let Ok(temp) = fields[5].parse::<f64>() else {
                            return false;
                        };
                        (shell - radius).abs() < 0.05
                            && fields[2] == source
                            && (temp - temperature).abs() < 1.0
                            && (fields[12] == "spontaneous" || fields[12] == "barrierless")
                    });
                    let spontaneous = rgsaddle::capture_is_spontaneous_within(
                        recombines,
                        barrier,
                        threshold_mev * 1.0e-3,
                    );
                    if spontaneous {
                        sites.push(rgsaddle::VolumeSite {
                            id,
                            count: share,
                            outcome: rgsaddle::SiteOutcome::Spontaneous,
                            rates: Vec::new(),
                        });
                        continue;
                    }
                    let capture = capture_rate_row(&text, *radius, source, temperature);
                    let mut rates = Vec::new();
                    let mut capture_neighbors = 0.0_f64;
                    let mut outside = 0.0_f64;
                    if let Some(members) = sites_of.get(index) {
                        for site in members {
                            for step in nn_vecs {
                                let next = [
                                    site[0] + step[0] * a,
                                    site[1] + step[1] * a,
                                    site[2] + step[2] * a,
                                ];
                                let next_r =
                                    (next[0] * next[0] + next[1] * next[1] + next[2] * next[2])
                                        .sqrt();
                                // The vacancy sits at the origin. A nearest-neighbor
                                // hop onto it is capture, including when this shell
                                // is kinetic.
                                if next_r <= 0.5 * a {
                                    capture_neighbors += 1.0;
                                } else if shells
                                    .iter()
                                    .all(|(value, _, _)| (value - next_r).abs() > 1.0e-3)
                                {
                                    outside += 1.0;
                                }
                            }
                        }
                        if !members.is_empty() {
                            capture_neighbors /= members.len() as f64;
                            outside /= members.len() as f64;
                        }
                    }
                    for other in 0..shells.len() {
                        let weight = fraction[index][other];
                        if weight <= 0.0 {
                            continue;
                        }
                        let other_radius = shells[other].0;
                        let other_id = format!("{other_radius:.3}-{orientation}");
                        let other_source = orientation_with_data(
                            &text,
                            other_radius,
                            shells[other].1,
                            orientation,
                            temperature,
                        )
                        .unwrap_or(orientation);
                        let other_barrier =
                            lowest_barrier(&text, other_radius, other_source, temperature);
                        let other_recombines = text.lines().any(|line| {
                            let fields: Vec<&str> = line.split(',').collect();
                            if fields.len() < 13 {
                                return false;
                            }
                            let Ok(shell) = fields[1].parse::<f64>() else {
                                return false;
                            };
                            let Ok(temp) = fields[5].parse::<f64>() else {
                                return false;
                            };
                            (shell - other_radius).abs() < 0.05
                                && fields[2] == other_source
                                && (temp - temperature).abs() < 1.0
                                && (fields[12] == "spontaneous" || fields[12] == "barrierless")
                        });
                        if rgsaddle::capture_is_spontaneous_within(
                            other_recombines,
                            other_barrier,
                            threshold_mev * 1.0e-3,
                        ) {
                            // `weight` is the mean number of neighbor hops into
                            // that shell. Those hops are capture, counted once
                            // at the measured per-hop rate.
                            capture_neighbors += weight;
                        } else {
                            rates.push((other_id, weight * k_mig));
                        }
                    }
                    if capture_neighbors > 0.0 {
                        let (per_hop, relative) = capture.unwrap_or((k_mig, 0.0));
                        let total = capture_neighbors * per_hop;
                        capture_index.push((sites.len(), total, relative));
                        rates.push(("recombined".into(), total));
                    } else if let Some((rate, relative)) = capture {
                        capture_index.push((sites.len(), rate, relative));
                        rates.push(("recombined".into(), rate));
                    }
                    if outside > 0.0 {
                        rates.push(("escaped".into(), outside * k_mig));
                    }
                    if rates.is_empty() {
                        continue;
                    }
                    sites.push(rgsaddle::VolumeSite {
                        id,
                        count: share,
                        outcome: rgsaddle::SiteOutcome::Kinetic,
                        rates,
                    });
                }
                let max_r = shells.last().map(|(radius, _, _)| *radius).unwrap_or(0.0);
                let boundary = format!("{max_r:.3}-{orientation}");
                let jump = a / 2.0_f64.sqrt();
                let return_probability =
                    rgsaddle::spherical_return_probability(max_r, jump).unwrap_or(0.0);
                // Sites one nearest-neighbor hop past the outer shell. The
                // outer shell's own multiplicity is the boundary, not this layer.
                // Three orientations share the same sites, as with count/3.
                let mut outward_sites = std::collections::BTreeSet::new();
                if let Some(members) = sites_of.last() {
                    for site in members {
                        for step in nn_vecs {
                            let next = [
                                site[0] + step[0] * a,
                                site[1] + step[1] * a,
                                site[2] + step[2] * a,
                            ];
                            let next_r =
                                (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                            if next_r <= 0.5 * a {
                                continue;
                            }
                            if shells
                                .iter()
                                .any(|(value, _, _)| (value - next_r).abs() <= 1.0e-3)
                            {
                                continue;
                            }
                            outward_sites.insert((
                                (next[0] * 1.0e6).round() as i64,
                                (next[1] * 1.0e6).round() as i64,
                                (next[2] * 1.0e6).round() as i64,
                            ));
                        }
                    }
                }
                let return_count = outward_sites.len() as f64 / 3.0;
                let embedded = if let Some(boundary_site) = sites.iter().find(|site| {
                    site.id == boundary && site.outcome == rgsaddle::SiteOutcome::Kinetic
                }) {
                    let outward_rate = boundary_site
                        .rates
                        .iter()
                        .find(|(destination, _)| destination == "escaped")
                        .map(|(_, rate)| *rate)
                        .unwrap_or(k_mig);
                    rgsaddle::embed_free_migration(
                        &sites,
                        &rgsaddle::FreeMigration {
                            boundary,
                            outward_rate,
                            return_probability,
                            return_count,
                        },
                    )
                    .unwrap_or(sites.clone())
                } else {
                    sites.clone()
                };
                let Ok(volume) = rgsaddle::kinetic_recombination_volume(&embedded, omega) else {
                    continue;
                };
                total += volume.total_sites;
                spontaneous_total += volume.spontaneous_sites;
                kinetic_total += volume.kinetic_sites;
                for sample in 0..samples.len() {
                    let mut varied = embedded.clone();
                    for (place, rate, relative) in &capture_index {
                        if let Some(site) = varied.get_mut(*place) {
                            let sign = if sample % 2 == 0 { 1.0 } else { -1.0 };
                            let factor = (sign * relative).exp();
                            for edge in &mut site.rates {
                                if edge.0 == "recombined" {
                                    edge.1 = rate * factor;
                                }
                            }
                        }
                    }
                    if let Ok(value) = rgsaddle::kinetic_recombination_volume(&varied, omega) {
                        samples[sample] += value.total_sites;
                    }
                }
            }
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let variance = samples
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f64>()
                / samples.len() as f64;
            let sigma = variance.sqrt();
            writeln!(out, "{temperature},{threshold_mev},{total:.4},{sigma:.4},{spontaneous_total:.4},{kinetic_total:.4},{demkowicz},8x8x8=2048 atoms; Demkowicz size-converged at 8788").unwrap();
            eprintln!(
                "passage {temperature} K {threshold_mev} meV V/Omega {total:.2} ± {sigma:.2} spontaneous {spontaneous_total:.1} kinetic {kinetic_total:.1}"
            );
        }
    }
    writeln!(out, "note,The return layer is one nearest-neighbor hop past the outer shell. Its return probability is R/(R+lambda), the spherical first-passage value. Cell is 8x8x8 (2048 atoms); Table 1 of the accepted manuscript, Phys. Rev. Materials 9, 023601 (2025), doi:10.1103/PhysRevMaterials.9.023601, section 4, is size-independent only at 8788 atoms and above.").unwrap();
}

#[test]
#[ignore = "assemble the first-passage recombination volume from the catalogues"]
fn defect_pair_passage() {
    assemble_passage();
}

fn demkowicz_lattice(temperature: f64) -> f64 {
    // Table 1 of the accepted manuscript, Phys. Rev. Materials 9, 023601,
    // section 4. These are their zero-pressure lattice parameters, not ours.
    match temperature as i32 {
        300 => 3.631,
        500 => 3.643,
        700 => 3.655,
        _ => 3.643,
    }
}

fn shell_772(a: f64, repeats: usize) -> Option<[f64; 3]> {
    let target = a * 4.5_f64.sqrt();
    shell_table(a, repeats)
        .into_iter()
        .find(|(radius, _, _)| (*radius - target).abs() < 0.05)
        .map(|(_, vector, _)| vector)
}

fn classify_shell(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    vector: [f64; 3],
    orientation: &str,
) -> (bool, f64, f64) {
    let PairState {
        gap,
        separation,
        recombined,
        ..
    } = placed(
        copper,
        repeats,
        a,
        vector,
        axis_from_label(orientation),
        true,
    );
    (recombined, gap, separation)
}

#[test]
#[ignore = "8788-atom recombination volume and the 7.72 A size check"]
fn defect_pair_run12() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_run12.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("run12");
        writeln!(header, "kind,repeats,temperature_K,a_A,radius_A,orientation,outcome,gap_A,separation_A,barrier_eV,force_eV_A,note").unwrap();
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    for temperature in [500.0_f64, 300.0, 700.0] {
        let a = demkowicz_lattice(temperature);
        for repeats in [8usize, 10, 13] {
            let prior = std::fs::read_to_string(&path).unwrap_or_default();
            let key = format!("size,{repeats},{temperature},");
            if checkpoint_has(&prior, &key) {
                continue;
            }
            let Some(vector) = shell_772(a, repeats) else {
                write_tight(
                    &checkpoint,
                    &format!("size,{repeats},{temperature},{a:.3},,,,missing,,,,no 7.72 shell"),
                );
                continue;
            };
            let radius =
                (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
            let t0 = started.elapsed().as_secs_f64();
            let (spontaneous, gap, separation) = classify_shell(&copper, repeats, a, vector, "100");
            let outcome = if spontaneous { "spontaneous" } else { "bound" };
            let atoms = 4 * repeats * repeats * repeats;
            write_tight(
                &checkpoint,
                &format!(
                    "size,{repeats},{temperature},{a:.3},{radius:.3},100,{outcome},{gap:.3},{separation:.3},,,{atoms} atoms; wall {:.1}s",
                    started.elapsed().as_secs_f64() - t0
                ),
            );
        }
    }
    for repeats in [8usize, 13] {
        for temperature in [300.0_f64, 500.0, 700.0] {
            let a = demkowicz_lattice(temperature);
            for (radius, vector, count) in shell_table(a, repeats) {
                if radius > 9.0 {
                    continue;
                }
                for orientation in distinct_orientations(vector) {
                    let prior = std::fs::read_to_string(&path).unwrap_or_default();
                    let key =
                        format!("census,{repeats},{temperature},{a:.3},{radius:.3},{orientation},");
                    if checkpoint_has(&prior, &key) {
                        continue;
                    }
                    let PairState {
                        gap,
                        separation,
                        recombined,
                        ..
                    } = placed(
                        &copper,
                        repeats,
                        a,
                        vector,
                        axis_from_label(orientation),
                        true,
                    );
                    let spontaneous = recombined;
                    let outcome = if spontaneous { "spontaneous" } else { "bound" };
                    write_tight(
                        &checkpoint,
                        &format!(
                            "census,{repeats},{temperature},{a:.3},{radius:.3},{orientation},{outcome},{gap:.3},{separation:.3},,,sites {count}; relax only"
                        ),
                    );
                }
            }
        }
    }
    for repeats in [8usize, 13] {
        for temperature in [300.0_f64, 500.0, 700.0] {
            let a = demkowicz_lattice(temperature);
            for (radius, vector, count) in shell_table(a, repeats) {
                let reduced = radius / a;
                let interesting = (reduced - 1.0).abs() < 0.04
                    || (reduced - 1.5_f64.sqrt()).abs() < 0.04
                    || (reduced - 2.0_f64.sqrt()).abs() < 0.04;
                if !interesting {
                    continue;
                }
                for orientation in distinct_orientations(vector) {
                    let prior = std::fs::read_to_string(&path).unwrap_or_default();
                    if checkpoint_has(
                        &prior,
                        &format!(
                            "quench,{repeats},{temperature},{a:.3},{radius:.3},{orientation},"
                        ),
                    ) {
                        continue;
                    }
                    let (spontaneous, gap, separation) =
                        classify_shell(&copper, repeats, a, vector, orientation);
                    let outcome = if spontaneous { "spontaneous" } else { "bound" };
                    write_tight(
                        &checkpoint,
                        &format!(
                            "quench,{repeats},{temperature},{a:.3},{radius:.3},{orientation},{outcome},{gap:.3},{separation:.3},,,sites {count}; full quench"
                        ),
                    );
                }
            }
        }
    }
    rotation_channels_300(&copper, &checkpoint, started);
    assemble_run12();
    eprintln!("run12 {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

fn rotation_channels_300(copper: &FastCopper, checkpoint: &Mutex<File>, started: Instant) {
    let temperature = 300.0_f64;
    let a = demkowicz_lattice(temperature);
    let repeats = 8usize;
    let length = repeats as f64 * a;
    let (perfect, _) = fcc(repeats, a);
    let (_, _, bulk) = copper.evaluate(
        perfect.view(),
        [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]],
    );
    let _ = bulk;
    for (radius, vector, _) in shell_table(a, repeats) {
        if !(4.0..8.5).contains(&radius) {
            continue;
        }
        for orientation in ["100", "001"] {
            let prior = std::fs::read_to_string(
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run12.csv"),
            )
            .unwrap_or_default();
            if checkpoint_has(
                &prior,
                &format!("rotation,{repeats},{temperature},{a:.3},{radius:.3},{orientation},"),
            ) {
                continue;
            }
            let PairState {
                positions,
                dumbbell,
                cell,
                recombined,
                ..
            } = placed(
                copper,
                repeats,
                a,
                vector,
                axis_from_label(orientation),
                true,
            );
            if recombined {
                write_tight(
                    checkpoint,
                    &format!(
                        "rotation,{repeats},{temperature},{a:.3},{radius:.3},{orientation},spontaneous,,,,,quench not needed"
                    ),
                );
                continue;
            }
            let reactant = positions;
            let reactant_energy = basin_energy(copper, &reactant, cell);
            eprintln!(
                "rotation {radius:.3} {orientation} at {:.0}s",
                started.elapsed().as_secs_f64()
            );
            let (hint, mode) = rotation_hint(&reactant, dumbbell);
            let Some(hit) = tight_dimer(copper, cell, hint, mode) else {
                write_tight(
                    checkpoint,
                    &format!(
                        "rotation,{repeats},{temperature},{a:.3},{radius:.3},{orientation},not_converged,,,,,dimer failed"
                    ),
                );
                continue;
            };
            let barrier = hit.energy - reactant_energy;
            let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
            write_tight(
                checkpoint,
                &format!(
                    "rotation,{repeats},{temperature},{a:.3},{radius:.3},{orientation},{assignment},{barrier:.6},{:.6e},{},,{assignment} seed; force tolerance 1.5e-3; free rotation at 300 K was 0.229987 eV",
                    hit.force, hit.index1
                ),
            );
        }
    }
}

fn assemble_run12() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let census = std::fs::read_to_string(root.join("defect_pair_run12.csv")).unwrap_or_default();
    let rates = catalogue_text();
    let mut out = File::create(root.join("defect_pair_run12_volume.csv")).expect("volume");
    writeln!(out, "temperature_K,method,volume_over_Omega,sigma,spontaneous_sites,kinetic_sites,demkowicz_8788,note").unwrap();
    let diffusivity = |temperature: f64| match temperature as i32 {
        300 => 0.143e12,
        500 => 0.526e12,
        700 => 0.910e12,
        _ => f64::NAN,
    };
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = demkowicz_lattice(temperature);
        let repeats = 13usize;
        let n_atoms = 4.0 * (repeats * repeats * repeats) as f64;
        let omega = a * a * a / 4.0;
        let k_mig = migration_rate(&rates, temperature);
        let Some(k_mig) = k_mig else {
            writeln!(out, "{temperature},first-passage,,,,,,no migration rate").unwrap();
            continue;
        };
        let shells = shell_table(a, repeats);
        let mut spontaneous = std::collections::HashSet::new();
        let mut changed = Vec::new();
        for (radius, vector, count) in &shells {
            if *radius > 11.0 {
                continue;
            }
            for orientation in ["100", "010", "001"] {
                let hat = unit3(*vector);
                let class_of = |label: &str| {
                    let axis = axis_from_label(label);
                    ((axis[0] * hat[0] + axis[1] * hat[1] + axis[2] * hat[2]).abs() * 1000.0)
                        .round() as i32
                };
                let want = class_of(orientation);
                let matches = |kind: &str, label: &str, line: &str| {
                    let fields: Vec<&str> = line.split(',').collect();
                    fields.len() > 7
                        && fields[0] == kind
                        && fields[1] == "13"
                        && fields[2]
                            .parse::<f64>()
                            .ok()
                            .is_some_and(|t| (t - temperature).abs() < 1.0)
                        && fields[4]
                            .parse::<f64>()
                            .ok()
                            .is_some_and(|r| (r - radius).abs() < 0.02)
                        && fields[5] == label
                };
                let mut line = None;
                for kind in ["quench", "census"] {
                    if let Some(row) = census.lines().find(|row| matches(kind, orientation, row)) {
                        line = Some(row);
                        break;
                    }
                    if let Some(row) = census.lines().find(|row| {
                        let fields: Vec<&str> = row.split(',').collect();
                        fields.len() > 5
                            && class_of(fields[5]) == want
                            && matches(kind, fields[5], row)
                    }) {
                        line = Some(row);
                        break;
                    }
                }
                let is_spontaneous =
                    line.is_some_and(|row| row.split(',').nth(6) == Some("spontaneous"));
                let old = text_spontaneous_at_8(
                    &rates,
                    *radius * measured_lattice(temperature) / a,
                    orientation,
                    temperature,
                );
                if is_spontaneous != old {
                    changed.push(format!(
                        "{radius:.3}-{orientation}:{}->{}",
                        if old { "spontaneous" } else { "bound" },
                        if is_spontaneous {
                            "spontaneous"
                        } else {
                            "bound"
                        }
                    ));
                }
                if is_spontaneous {
                    spontaneous.insert(format!("{radius:.3}-{orientation}"));
                    let _ = (vector, count);
                }
            }
        }
        let max_r = shells
            .iter()
            .filter(|(radius, _, _)| *radius <= 11.0)
            .map(|(radius, _, _)| *radius)
            .fold(0.0, f64::max);
        let jump = a / 2.0_f64.sqrt();
        let return_sites = return_layer_count(a, &shells, max_r);
        let (
            total,
            sigma,
            spont_sites,
            kinetic_sites,
            weight,
            mean_graph,
            radial,
            glide,
            rotated,
            drifted,
            _,
            _,
        ) = channel_volumes(
            &shells,
            &|id: &str| f64::from(u8::from(spontaneous.contains(id))),
            RateScale::RECORDED,
            &rates,
            temperature,
            k_mig,
            a,
            omega,
            max_r,
            jump,
            return_sites,
            11.0,
            false,
            &|_, _| 0.0,
        );
        let demkowicz = match temperature as i32 {
            300 => "254-286",
            500 => "161-185",
            700 => "203-210",
            _ => "",
        };
        let changed_note = if changed.is_empty() {
            "no shell changed versus 8^3".to_string()
        } else {
            changed.join(" ")
        };
        writeln!(out, "{temperature},first-passage,{total:.4},{sigma:.4},{spont_sites:.4},{kinetic_sites:.4},{demkowicz},13x13x13=8788; a={a}; return layer {return_sites} sites; radial+glide+rotation; shells changed: {changed_note}").unwrap();
        writeln!(out, "{temperature},radial,{radial:.4},,{spont_sites:.4},{kinetic_sites:.4},{demkowicz},toward and away translation only").unwrap();
        writeln!(out, "{temperature},glide,{glide:.4},,,,{demkowicz},radial plus non-radial translation (|dr| < half a nearest-neighbor hop); delta {:.4}", glide - radial).unwrap();
        writeln!(out, "{temperature},rotation,{rotated:.4},,,,{demkowicz},glide plus two <110> rotation saddles per target axis; delta {:.4}", rotated - glide).unwrap();
        writeln!(out, "{temperature},drift,{drifted:.4},,,,{demkowicz},rotation graph with measured strain barriers on away hops; delta {:.4}", drifted - rotated).unwrap();
        eprintln!(
            "13^3 {temperature} K radial {radial:.2} glide {glide:.2} (d {:.2}) rotation {rotated:.2} (d {:.2}) drift {drifted:.2} (d {:.2}) full {total:.2} ± {sigma:.2}",
            glide - radial,
            rotated - glide,
            drifted - rotated
        );
        let tau_near = if weight > 0.0 {
            mean_graph / weight
        } else {
            f64::NAN
        };
        let tau_cell = cell_averaged_time(repeats, a, k_mig, max_r, tau_near);
        let nu_tst = (2.0 / 3.0) * k_mig;
        let nu_table = (2.0 / 3.0) * diffusivity(temperature) / (a * a);
        if tau_cell.is_finite() && tau_cell > 0.0 {
            let volume_tst = n_atoms / (tau_cell * nu_tst);
            let volume_table = n_atoms / (tau_cell * nu_table);
            writeln!(out, "{temperature},mean-time-tst,{volume_tst:.4},,{spont_sites:.4},,{demkowicz},tau {tau_cell:.6e} s; nu=(2/3)*k_translation from this run").unwrap();
            writeln!(out, "{temperature},mean-time-table,{volume_table:.4},,{spont_sites:.4},,{demkowicz},same tau; nu=(2/3)*D/a^2 with D from their Table 1").unwrap();
            eprintln!(
                "13^3 mean-time {temperature} K tst {volume_tst:.2} table-D {volume_table:.2} tau {tau_cell:.3e}"
            );
        }
    }
}

fn return_layer_count(a: f64, shells: &[(f64, [f64; 3], usize)], max_r: f64) -> usize {
    let nn_vecs = [
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, -0.5],
        [0.0, -0.5, 0.5],
        [0.0, -0.5, -0.5],
    ];
    let mut outward = std::collections::BTreeSet::new();
    let limit = ((2.0 * (max_r + a) / a).ceil() as i32) + 2;
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                let radius = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                if (radius - max_r).abs() > 1.0e-3 {
                    continue;
                }
                for step in nn_vecs {
                    let next = [
                        cart[0] + step[0] * a,
                        cart[1] + step[1] * a,
                        cart[2] + step[2] * a,
                    ];
                    let next_r = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                    if next_r <= 0.5 * a {
                        continue;
                    }
                    if shells
                        .iter()
                        .any(|(value, _, _)| (value - next_r).abs() <= 1.0e-3)
                    {
                        continue;
                    }
                    outward.insert((
                        (next[0] * 1.0e6).round() as i64,
                        (next[1] * 1.0e6).round() as i64,
                        (next[2] * 1.0e6).round() as i64,
                    ));
                }
            }
        }
    }
    outward.len()
}

fn cell_averaged_time(repeats: usize, a: f64, k_mig: f64, radius_cut: f64, tau_near: f64) -> f64 {
    let (points, cell) = fcc(repeats, a);
    let length = cell[0][0];
    let center = length * 0.5;
    let n = points.len() / 3;
    let mut n_in = 0usize;
    let mut outside = Vec::new();
    for atom in 0..n {
        let mut d = [0.0; 3];
        for c in 0..3 {
            let delta = points[3 * atom + c] - center;
            d[c] = delta - length * (delta / length).round();
        }
        let radius = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        if radius <= radius_cut {
            n_in += 1;
        } else if outside.len() < 64 {
            outside.push(atom);
        }
    }
    let steps = [
        [0.5, 0.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, -0.5],
        [0.0, -0.5, 0.5],
        [0.0, -0.5, -0.5],
    ];
    // A <100> dumbbell has eight translation channels, not twelve.
    let hop_rate = 8.0 * k_mig;
    let mut sample = 0.0;
    let mut seed = 0x1234_5678_u64;
    for start in &outside {
        let start = *start;
        let mut place = [
            points[3 * start],
            points[3 * start + 1],
            points[3 * start + 2],
        ];
        let mut hops = 0u32;
        loop {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let step = steps[(seed >> 33) as usize % 8];
            for c in 0..3 {
                place[c] += step[c] * a;
                place[c] -= length * (place[c] / length).floor();
            }
            hops += 1;
            let mut d = [0.0; 3];
            for c in 0..3 {
                let delta = place[c] - center;
                d[c] = delta - length * (delta / length).round();
            }
            let radius = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if radius <= radius_cut || hops > 200_000 {
                break;
            }
        }
        sample += hops as f64 / hop_rate + tau_near;
    }
    let n_out = n - n_in;
    let mean_out = if outside.is_empty() {
        tau_near
    } else {
        sample / outside.len() as f64
    };
    (n_in as f64 * tau_near + n_out as f64 * mean_out) / n as f64
}

fn rotation_channel(text: &str, temperature: f64) -> (f64, String) {
    let mut pafi: Option<(f64, f64)> = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13
            || fields[0] != "pafi"
            || fields[1] != "isolated"
            || fields[3] != "rotate"
            || fields[12] != "rated"
        {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        let Ok(nu) = fields[10].parse::<f64>() else {
            continue;
        };
        let Ok(rate) = fields[11].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() < 1.0 && rate.is_finite() && rate > 0.0 {
            return (rate, format!("PAFI rate {rate:.6e} Hz of one <110> saddle"));
        }
        if nu.is_finite()
            && nu > 0.0
            && pafi
                .is_none_or(|(prior, _)| (temp - temperature).abs() < (prior - temperature).abs())
        {
            pafi = Some((temp, nu));
        }
    }
    let mut barrier = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6
            || fields[0] != "rotation"
            || fields[1] != "isolated"
            || fields[3] != "rotate"
        {
            continue;
        }
        let Ok(temp) = fields[4].parse::<f64>() else {
            continue;
        };
        let Ok(value) = fields[5].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() < 1.0 && value.is_finite() && value > 0.0 {
            barrier = Some(value);
        }
    }
    let Some(barrier) = barrier else {
        return (0.0, "no rotation barrier".into());
    };
    let Some((nu_t, nu)) = pafi else {
        return (
            0.0,
            format!("barrier {barrier:.6} eV and no attempt frequency"),
        );
    };
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let rate = nu * (-barrier / thermal).exp();
    (
        rate,
        format!(
            "TST of the measured {barrier:.6} eV barrier with the {nu_t:.0} K attempt frequency {nu:.6e} Hz; no {temperature:.0} K rotation PAFI"
        ),
    )
}

fn strain_away_factor(text: &str, temperature: f64, radius_ours: f64, free: f64) -> f64 {
    let mut best: Option<(f64, f64)> = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 10 || fields[0] != "strain" || fields[4] != "away" || fields[9] != "true"
        {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        let Ok(barrier) = fields[6].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 || (shell - radius_ours).abs() > 0.05 || barrier <= 0.0
        {
            continue;
        }
        let gap = (shell - radius_ours).abs();
        if best.is_none_or(|(prior, _)| gap < prior) {
            best = Some((gap, barrier));
        }
    }
    let Some((_, barrier)) = best else { return 1.0 };
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    (-(barrier - free) / thermal).exp()
}

fn isolated_attempt_frequency(text: &str, temperature: f64) -> Option<f64> {
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13
            || fields[0] != "pafi"
            || fields[1] != "isolated"
            || fields[3] != "translate"
            || fields[12] != "rated"
        {
            continue;
        }
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        let Ok(nu) = fields[10].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() < 1.0 && nu.is_finite() && nu > 0.0 {
            return Some(nu);
        }
    }
    None
}

/// Away barriers from the cell-size study, as 1/L^3 intercepts.
///
/// The 4.456 Å shell has two <100> away minima. The nearer geometric
/// landing carries the minimum whose separation is 4.98-4.99 Å. The
/// farther landing carries the minimum at 5.15-5.17 Å. The 3.639 Å away
/// hop is the converged intercept, one barrier for every away landing.
fn sized_away_rate(text: &str, temperature: f64, source: f64, dest: f64, a: f64) -> Option<f64> {
    let source_n = source / a;
    let dest_n = dest / a;
    let barrier = if (source_n - 1.0).abs() < 0.04 && dest_n > source_n + 0.05 {
        match temperature as i32 {
            300 => Some(0.4052),
            500 => Some(0.1343),
            _ => None,
        }
    } else if (source_n - 1.5_f64.sqrt()).abs() < 0.04 && dest_n > source_n + 0.05 {
        let nearer = dest_n < 1.72;
        match (temperature as i32, nearer) {
            (300, true) => Some(0.1006),
            (300, false) => Some(0.0840),
            (500, true) => Some(0.0968),
            (500, false) => Some(0.0817),
            _ => None,
        }
    } else {
        None
    }?;
    let nu = isolated_attempt_frequency(text, temperature)?;
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let rate = nu * (-barrier / thermal).exp();
    rate.is_finite().then_some(rate)
}

fn shell_rotation_rate(
    text: &str,
    radius: f64,
    orientation: &str,
    temperature: f64,
) -> Option<f64> {
    let mut best: Option<(f64, f64)> = None;
    for line in text.lines() {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13
            || fields[0] != "pafi"
            || fields[3] != "rotate"
            || fields[12] != "rated"
        {
            continue;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            continue;
        };
        let Ok(temp) = fields[5].parse::<f64>() else {
            continue;
        };
        let Ok(rate) = fields[11].parse::<f64>() else {
            continue;
        };
        if (temp - temperature).abs() > 1.0
            || fields[2] != orientation
            || !rate.is_finite()
            || rate <= 0.0
        {
            continue;
        }
        let gap = (shell - radius).abs();
        if gap > 0.08 {
            continue;
        }
        if best.is_none_or(|(prior, _)| gap < prior) {
            best = Some((gap, rate));
        }
    }
    best.map(|(_, rate)| rate)
}

fn strain_bias(
    binding: &dyn Fn(f64, &str) -> f64,
    source_r: f64,
    source_o: &str,
    dest_r: f64,
    dest_o: &str,
    temperature: f64,
) -> f64 {
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    if !(thermal.is_finite() && thermal > 0.0) {
        return 1.0;
    }
    let delta = binding(dest_r, dest_o) - binding(source_r, source_o);
    let factor = (-0.5 * delta / thermal).exp();
    if factor.is_finite() && factor > 0.0 {
        factor
    } else {
        1.0
    }
}

/// Shifts of the rotation and capture rates of [`channel_volumes`], for
/// propagating their uncertainty: `rotation` multiplies every rotation
/// rate, and each capture rate is multiplied by `exp(capture * relative)`
/// with `relative` its own free-energy error over kT.
#[derive(Clone, Copy)]
struct RateScale {
    rotation: f64,
    capture: f64,
}

impl RateScale {
    const RECORDED: Self = Self {
        rotation: 1.0,
        capture: 0.0,
    };
}

#[allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
fn channel_volumes(
    shells: &[(f64, [f64; 3], usize)],
    spontaneous: &dyn Fn(&str) -> f64,
    scale: RateScale,
    rates: &str,
    temperature: f64,
    k_mig: f64,
    a: f64,
    omega: f64,
    max_r: f64,
    _jump: f64,
    _return_sites: usize,
    cutoff: f64,
    size_barriers: bool,
    binding: &dyn Fn(f64, &str) -> f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let kept: Vec<_> = shells
        .iter()
        .filter(|(radius, _, _)| *radius <= cutoff)
        .cloned()
        .collect();
    let nn_vecs = [
        [0.5, 0.5, 0.0],
        [0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [-0.5, 0.0, -0.5],
        [0.0, 0.5, 0.5],
        [0.0, 0.5, -0.5],
        [0.0, -0.5, 0.5],
        [0.0, -0.5, -0.5],
    ];
    let mut members: Vec<Vec<[f64; 3]>> = kept.iter().map(|_| Vec::new()).collect();
    let limit = ((2.0 * (cutoff + 0.4) / a).ceil() as i32) + 1;
    for i in -limit..=limit {
        for j in -limit..=limit {
            for k in -limit..=limit {
                if (i + j + k) % 2 != 0 || (i == 0 && j == 0 && k == 0) {
                    continue;
                }
                let cart = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
                let radius = (cart[0] * cart[0] + cart[1] * cart[1] + cart[2] * cart[2]).sqrt();
                if let Some(index) = kept
                    .iter()
                    .position(|(value, _, _)| (value - radius).abs() < 1.0e-3)
                {
                    members[index].push(cart);
                }
            }
        }
    }
    let hop = a / 2.0_f64.sqrt();
    // Diagnostic for one axis. A <100> dumbbell translates on eight neighbor
    // directions, and the hop rotates the axis. Summing the twelve FCC
    // neighbors, shells weighted equally, gave along-line 78 and oblique
    // 74.83; that count includes the four in-plane directions and keeps the
    // axis. The sums below are the eight translation channels of axis 001.
    let mut along_sum = 0.0_f64;
    let mut oblique_sum = 0.0_f64;
    for (index, group) in members.iter().enumerate() {
        if group.is_empty() {
            continue;
        }
        let mut along = 0.0_f64;
        let mut oblique = 0.0_f64;
        for site in group {
            for step in nn_vecs {
                let step_a = [step[0] * a, step[1] * a, step[2] * a];
                if dumbbell_translation_axis([0.0, 0.0, 1.0], step_a).is_none() {
                    continue;
                }
                let next = [
                    site[0] + step_a[0],
                    site[1] + step_a[1],
                    site[2] + step_a[2],
                ];
                let next_r = (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                if next_r <= 0.5 * a {
                    continue;
                }
                let Some(other) = kept
                    .iter()
                    .position(|(value, _, _)| (value - next_r).abs() < 1.0e-3)
                else {
                    continue;
                };
                if other == index {
                    continue;
                }
                if (next_r - kept[index].0).abs() < 0.5 * hop {
                    oblique += 1.0;
                } else {
                    along += 1.0;
                }
            }
        }
        let norm = group.len() as f64;
        along_sum += along / norm;
        oblique_sum += oblique / norm;
    }
    eprintln!(
        "13^3 {temperature} K <100> translation weights, axis 001: along-line {along_sum:.2} oblique {oblique_sum:.2}"
    );
    let (k_rot, rotation_note) = rotation_channel(rates, temperature);
    eprintln!("13^3 {temperature} K rotation {k_rot:.6e} Hz; {rotation_note}");
    let translation_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_translation.csv"),
    )
    .unwrap_or_default();
    let free_barrier = translation_text
        .lines()
        .find_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 6
                || fields[0] != "translation"
                || fields[1] != "isolated"
                || fields[3] != "neighbor"
            {
                return None;
            }
            let temp = fields[4].parse::<f64>().ok()?;
            let barrier = fields[5].parse::<f64>().ok()?;
            ((temp - temperature).abs() < 1.0 && barrier > 0.0).then_some(barrier)
        })
        .unwrap_or(0.0);
    let a_ours = measured_lattice(temperature);
    let solve = |glide: bool,
                 rotation: bool,
                 drift: bool|
     -> (
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<(usize, f64, f64)>,
        Vec<rgsaddle::VolumeSite>,
        f64,
    ) {
        let mut sites = Vec::new();
        let mut capture_index = Vec::new();
        let mut landings = Vec::new();
        for orientation in ["100", "010", "001"] {
            for (index, (radius, _, count)) in kept.iter().enumerate() {
                let id = format!("{radius:.3}-{orientation}");
                let share = *count as f64 / 3.0;
                let fraction = spontaneous(&id).clamp(0.0, 1.0);
                if fraction > 0.0 {
                    sites.push(rgsaddle::VolumeSite {
                        id: if fraction < 1.0 {
                            format!("{id}~spontaneous")
                        } else {
                            id.clone()
                        },
                        count: share * fraction,
                        outcome: rgsaddle::SiteOutcome::Spontaneous,
                        rates: Vec::new(),
                    });
                }
                if fraction >= 1.0 {
                    continue;
                }
                let share = share * (1.0 - fraction);
                let mut capture_neighbors = 0.0_f64;
                let mut edges = Vec::new();
                let mut rotation_capture = 0.0_f64;
                let mut translations: std::collections::BTreeMap<
                    (usize, &'static str),
                    (f64, rgsaddle::DirectionClass),
                > = std::collections::BTreeMap::new();
                let axis = axis_from_label(orientation);
                if let Some(group) = members.get(index) {
                    for site in group {
                        for step in nn_vecs {
                            let step_a = [step[0] * a, step[1] * a, step[2] * a];
                            let Some(new_axis) = dumbbell_translation_axis(axis, step_a) else {
                                continue;
                            };
                            let landed = if new_axis[0] > 0.5 {
                                "100"
                            } else if new_axis[1] > 0.5 {
                                "010"
                            } else {
                                "001"
                            };
                            let next = [
                                site[0] + step_a[0],
                                site[1] + step_a[1],
                                site[2] + step_a[2],
                            ];
                            let next_r =
                                (next[0] * next[0] + next[1] * next[1] + next[2] * next[2]).sqrt();
                            if next_r <= 0.5 * a {
                                capture_neighbors += 1.0;
                                continue;
                            }
                            let class = if (next_r - radius).abs() < 0.5 * hop {
                                rgsaddle::DirectionClass::Tangential
                            } else if next_r < *radius {
                                rgsaddle::DirectionClass::Toward
                            } else {
                                rgsaddle::DirectionClass::Away
                            };
                            if let Some(other) = kept
                                .iter()
                                .position(|(value, _, _)| (value - next_r).abs() < 1.0e-3)
                            {
                                let entry =
                                    translations.entry((other, landed)).or_insert((0.0, class));
                                entry.0 += 1.0;
                            } else if next_r <= max_r + 1.0e-6
                                && let Some(other) = kept
                                    .iter()
                                    .position(|(value, _, _)| (value - next_r).abs() < 0.05)
                            {
                                let entry =
                                    translations.entry((other, landed)).or_insert((0.0, class));
                                entry.0 += 1.0;
                            } else {
                                let bias = strain_bias(
                                    binding,
                                    *radius,
                                    orientation,
                                    next_r,
                                    landed,
                                    temperature,
                                );
                                landings.push(rgsaddle::SphereLanding {
                                    source: id.clone(),
                                    boundary: format!("{max_r:.3}-{landed}"),
                                    rate: k_mig / group.len() as f64 * bias,
                                    distance: next_r,
                                });
                            }
                        }
                    }
                    if !group.is_empty() {
                        let norm = group.len() as f64;
                        capture_neighbors /= norm;
                        for value in translations.values_mut() {
                            value.0 /= norm;
                        }
                    }
                }
                for ((other, landed), (weight, class)) in translations {
                    let use_edge = class != rgsaddle::DirectionClass::Tangential || glide;
                    if !use_edge || weight <= 0.0 {
                        continue;
                    }
                    let source_factor = if drift {
                        strain_away_factor(rates, temperature, radius * a_ours / a, free_barrier)
                    } else {
                        1.0
                    };
                    let destination_factor = if drift {
                        strain_away_factor(
                            rates,
                            temperature,
                            kept[other].0 * a_ours / a,
                            free_barrier,
                        )
                    } else {
                        1.0
                    };
                    let factor =
                        rgsaddle::shell_pair_factor(class, source_factor, destination_factor);
                    let other_id = format!("{:.3}-{landed}", kept[other].0);
                    let caught = spontaneous(&other_id).clamp(0.0, 1.0);
                    capture_neighbors += weight * caught;
                    if caught < 1.0 {
                        let hop_rate = if size_barriers && class == rgsaddle::DirectionClass::Away {
                            sized_away_rate(rates, temperature, *radius, kept[other].0, a)
                                .unwrap_or(k_mig)
                        } else {
                            k_mig
                        };
                        let bias = strain_bias(
                            binding,
                            *radius,
                            orientation,
                            kept[other].0,
                            landed,
                            temperature,
                        );
                        edges.push((other_id, weight * (1.0 - caught) * hop_rate * factor * bias));
                    }
                }
                if rotation {
                    for other_orient in ["100", "010", "001"] {
                        if other_orient == orientation {
                            continue;
                        }
                        let dest = format!("{radius:.3}-{other_orient}");
                        let per_saddle =
                            shell_rotation_rate(rates, *radius, orientation, temperature)
                                .unwrap_or(k_rot);
                        let rate = 2.0 * per_saddle * scale.rotation;
                        let caught = spontaneous(&dest).clamp(0.0, 1.0);
                        rotation_capture += rate * caught;
                        if caught < 1.0 && rate > 0.0 {
                            edges.push((dest, rate * (1.0 - caught)));
                        }
                    }
                }
                let radius_ours = radius * a_ours / a;
                let capture = capture_rate_row(rates, radius_ours, orientation, temperature)
                    .map(|(rate, relative)| (rate * (scale.capture * relative).exp(), relative));
                if capture_neighbors > 0.0 {
                    let (per_hop, relative) = capture.unwrap_or((k_mig, 0.0));
                    let total = capture_neighbors * per_hop;
                    capture_index.push((sites.len(), total, relative));
                    edges.push(("recombined".into(), total));
                } else if let Some((rate, relative)) = capture {
                    capture_index.push((sites.len(), rate, relative));
                    edges.push(("recombined".into(), rate));
                }
                if rotation_capture > 0.0 {
                    edges.push(("recombined".into(), rotation_capture));
                }
                if edges.is_empty() {
                    edges.push(("escaped".into(), k_mig));
                }
                sites.push(rgsaddle::VolumeSite {
                    id,
                    count: share,
                    outcome: rgsaddle::SiteOutcome::Kinetic,
                    rates: edges,
                });
            }
        }
        sites = rgsaddle::embed_sphere_return(&sites, max_r, &landings).unwrap_or(sites);
        let solved = rgsaddle::kinetic_recombination_volume(&sites, omega).ok();
        let (total, spont, kinetic) = match &solved {
            Some(volume) => (
                volume.total_sites,
                volume.spontaneous_sites,
                volume.kinetic_sites,
            ),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
        let mut estimates = Vec::new();
        if let Some(volume) = &solved {
            for place in &volume.placements {
                let Some((radius_text, _)) = place.id.rsplit_once('-') else {
                    continue;
                };
                let Ok(radius) = radius_text.parse::<f64>() else {
                    continue;
                };
                if radius < 8.0 || radius < 0.55 * max_r {
                    continue;
                }
                estimates.push(place.probability * radius);
            }
        }
        estimates
            .sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        let r_eff = if estimates.is_empty() {
            f64::NAN
        } else {
            estimates[estimates.len() / 2]
        };
        let time = rgsaddle::mean_absorption_time(&sites).unwrap_or(f64::NAN);
        let mut sealed = sites.clone();
        for site in &mut sealed {
            if site.id.starts_with("free-") {
                site.rates
                    .retain(|(destination, _)| destination != "escaped");
            } else if let Some(orient) = site.id.rsplit('-').next() {
                let layer = format!("free-{orient}");
                for edge in &mut site.rates {
                    if edge.0 == "escaped" {
                        edge.0 = layer.clone();
                    }
                }
            }
        }
        let tau_capture = rgsaddle::mean_absorption_time(&sealed).unwrap_or(f64::NAN);
        (
            total,
            spont,
            kinetic,
            time,
            tau_capture,
            capture_index,
            sites,
            r_eff,
        )
    };
    let (radial, _, _, _, _, _, _, _) = solve(false, false, false);
    let (glide, _, _, _, _, _, _, _) = solve(true, false, false);
    let (rotated, spont, kinetic, time, tau_capture, capture_index, embedded, r_eff) =
        solve(true, true, false);
    let (drifted, _, _, _, _, _, _, _) = solve(true, true, true);
    let mut samples = [0.0; 16];
    for sample in 0..samples.len() {
        let mut varied = embedded.clone();
        for (place, rate, relative) in &capture_index {
            if let Some(site) = varied.get_mut(*place) {
                let sign = if sample % 2 == 0 { 1.0 } else { -1.0 };
                let factor = (sign * relative).exp();
                if let Some(edge) = site.rates.iter_mut().find(|(destination, value)| {
                    destination == "recombined"
                        && (*value - rate).abs() <= 1.0e-6 * rate.abs().max(1.0)
                }) {
                    edge.1 = rate * factor;
                }
            }
        }
        if let Ok(value) = rgsaddle::kinetic_recombination_volume(&varied, omega) {
            samples[sample] = value.total_sites;
        }
    }
    let sample_mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let sigma = (samples
        .iter()
        .map(|value| (value - sample_mean) * (value - sample_mean))
        .sum::<f64>()
        / samples.len() as f64)
        .sqrt();
    let weight = embedded.iter().map(|site| site.count).sum::<f64>();
    (
        rotated,
        sigma,
        spont,
        kinetic,
        weight,
        time * weight,
        radial,
        glide,
        rotated,
        drifted,
        tau_capture,
        r_eff,
    )
}

fn text_spontaneous_at_8(
    text: &str,
    radius_ours: f64,
    orientation: &str,
    temperature: f64,
) -> bool {
    let barrier = lowest_barrier(text, radius_ours, orientation, temperature);
    let recombines = text.lines().any(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 13 {
            return false;
        }
        let Ok(shell) = fields[1].parse::<f64>() else {
            return false;
        };
        let Ok(temp) = fields[5].parse::<f64>() else {
            return false;
        };
        (shell - radius_ours).abs() < 0.08
            && fields[2] == orientation
            && (temp - temperature).abs() < 1.0
            && (fields[12] == "spontaneous" || fields[12] == "barrierless")
    });
    rgsaddle::capture_is_spontaneous_within(recombines, barrier, 0.001)
}

#[test]
#[ignore = "assemble the 8788-atom volume from the size census"]
fn defect_pair_run12_volume() {
    assemble_run12();
}

struct PairField {
    kelvin: rgsaddle::Kelvin,
    vacancy: rgsaddle::ElasticDipole,
    sia: [rgsaddle::ElasticDipole; 3],
}

impl PairField {
    fn interaction(&self, radius: f64, direction: [f64; 3], which: usize) -> f64 {
        let separation = [
            direction[0] * radius,
            direction[1] * radius,
            direction[2] * radius,
        ];
        self.kelvin
            .dipole_interaction(separation, self.vacancy, self.sia[which])
            .unwrap_or(0.0)
    }

    fn binding(&self, radius: f64, orientation: &str) -> f64 {
        let which = match orientation {
            "010" => 1,
            "001" => 2,
            _ => 0,
        };
        let mut sum = 0.0;
        let mut n = 0.0;
        for direction in fibonacci_directions(24) {
            sum += self.interaction(radius, direction, which);
            n += 1.0;
        }
        if n > 0.0 { sum / n } else { 0.0 }
    }

    fn effective(&self, radius: f64, thermal: f64) -> f64 {
        let mut sum = 0.0;
        let mut n = 0.0;
        for direction in fibonacci_directions(24) {
            for which in 0..3 {
                sum += (self.interaction(radius, direction, which) / thermal).exp();
                n += 1.0;
            }
        }
        if n > 0.0 && sum > 0.0 {
            thermal * (sum / n).ln()
        } else {
            0.0
        }
    }
}

fn fibonacci_directions(count: usize) -> Vec<[f64; 3]> {
    let mut out = Vec::with_capacity(count);
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..count {
        let y = 1.0 - (2.0 * i as f64 + 1.0) / count as f64;
        let ring = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden * i as f64;
        out.push([ring * theta.cos(), y, ring * theta.sin()]);
    }
    out
}

fn rotate_dipole(
    dipole: rgsaddle::ElasticDipole,
    rotation: [[f64; 3]; 3],
) -> rgsaddle::ElasticDipole {
    dipole.transform(rotation)
}

fn probe_binding(copper: &FastCopper, a: f64) {
    let (perfect, cell) = fcc(8, a);
    let (bulk, _, _) = copper.evaluate(perfect.view(), cell);
    let mut vacancy = perfect.clone();
    let n = vacancy.len();
    vacancy = vacancy.slice(ndarray::s![3..n]).to_owned();
    let (e_vac, _, _) = copper.evaluate(vacancy.view(), cell);
    let sia = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let (e_sia, _, _) = copper.evaluate(sia.view(), cell);
    for (i, j, k) in [
        (4, 0, 0),
        (3, 3, 0),
        (4, 4, 0),
        (6, 0, 0),
        (4, 4, 4),
        (8, 0, 0),
        (6, 6, 0),
    ] {
        let site = [i as f64 * 0.5 * a, j as f64 * 0.5 * a, k as f64 * 0.5 * a];
        let radius = (site[0] * site[0] + site[1] * site[1] + site[2] * site[2]).sqrt();
        let mut pair = Vec::new();
        let atoms = perfect.len() / 3;
        for atom in 1..atoms {
            let point = [
                perfect[3 * atom],
                perfect[3 * atom + 1],
                perfect[3 * atom + 2],
            ];
            let d = [point[0] - site[0], point[1] - site[1], point[2] - site[2]];
            if d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < 1.0e-8 {
                continue;
            }
            pair.extend_from_slice(&perfect.as_slice().unwrap()[3 * atom..3 * atom + 3]);
        }
        for sign in [-0.5, 0.5] {
            pair.push(site[0] + sign * 2.05);
            pair.push(site[1]);
            pair.push(site[2]);
        }
        let positions = Array1::from(pair);
        let (energy, _, _) = copper.evaluate(positions.view(), cell);
        let binding = energy - e_vac - e_sia + bulk;
        eprintln!(
            "binding r {radius:.2} A E {binding:.4} eV r^3 E {:.1}",
            binding * radius.powi(3)
        );
    }
}

/// FIRE 2.0 to 1e-4 eV/Å for the stress of one isolated defect. The
/// dipole is read from that stress, so the tolerance is ten times below
/// that of `pair_relax`.
fn field_relax() -> rgsaddle::RelaxConfig {
    pair_fire().with_force_tol(1.0e-4)
}

fn pair_field(copper: &FastCopper, a: f64) -> Result<PairField, SaddleError> {
    pair_field_with(copper, a, &field_relax())
}

fn pair_field_with(
    copper: &FastCopper,
    a: f64,
    relax: &rgsaddle::RelaxConfig,
) -> Result<PairField, SaddleError> {
    let repeats = 8usize;
    let (perfect, cell) = fcc(repeats, a);
    let held = FastHeld { copper, cell };
    let (_, _, bulk_stress) = copper.evaluate(perfect.view(), cell);
    let drop = perfect.len() - 3;
    let vacancy = perfect.slice(ndarray::s![..drop]).to_owned();
    let vacancy = rgsaddle::relax_to_tolerance(&held, vacancy, cell, relax)?.positions;
    let (_, _, vacancy_stress) = copper.evaluate(vacancy.view(), cell);
    let sia = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let sia = rgsaddle::relax_to_tolerance(&held, sia, cell, relax)?.positions;
    let (_, _, sia_stress) = copper.evaluate(sia.view(), cell);
    let volume = cell[0][0] * cell[1][1] * cell[2][2];
    let vacancy_dipole = rgsaddle::ElasticDipole::from_stress(volume, vacancy_stress, bulk_stress)?;
    let sia = rgsaddle::ElasticDipole::from_stress(volume, sia_stress, bulk_stress)?;
    let to_010 = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let to_001 = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];
    let (c11, c12, c44) = cubic_moduli(copper, a);
    let shear = (c11 - c12 + 3.0 * c44) / 5.0;
    let bulk = (c11 + 2.0 * c12) / 3.0;
    let poisson = if bulk + shear / 3.0 > 0.0 {
        (3.0 * bulk - 2.0 * shear) / (6.0 * bulk + 2.0 * shear)
    } else {
        0.3
    };
    let kelvin = rgsaddle::Kelvin::new(shear.abs().max(1.0e-6), poisson.clamp(-0.9, 0.49))?;
    eprintln!("dipoles vacancy {vacancy_dipole:?} sia {sia:?} mu {shear:.4} nu {poisson:.3}");
    Ok(PairField {
        kelvin,
        vacancy: vacancy_dipole,
        sia: [sia, rotate_dipole(sia, to_010), rotate_dipole(sia, to_001)],
    })
}

/// The rate-model volumes of [`channel_volumes`] at one cutoff: the site
/// sum, the free sphere of the Debye-matched radius, and that radius as a
/// sink in the 13³ cell where Table 1 measures k.
struct RateVolumes {
    total: f64,
    sigma: f64,
    spont_sites: f64,
    chain_radius: f64,
    d_graph: f64,
    r_eff: f64,
    sphere: f64,
    decay_volume: f64,
    mean_volume: f64,
    tau_cell: f64,
}

#[allow(clippy::too_many_arguments)]
fn rate_volumes(
    shells: &[(f64, [f64; 3], usize)],
    spontaneous: &dyn Fn(&str) -> f64,
    scale: RateScale,
    rates: &str,
    temperature: f64,
    k_mig: f64,
    a: f64,
    field: Option<&PairField>,
    cutoff: f64,
) -> RateVolumes {
    let omega = a * a * a / 4.0;
    let max_r = shells.last().map(|(radius, _, _)| *radius).unwrap_or(0.0);
    let jump = a / 2.0_f64.sqrt();
    let return_sites = return_layer_count(a, shells, max_r);
    let binding = |radius: f64, orientation: &str| {
        field
            .map(|item| item.binding(radius, orientation))
            .unwrap_or(0.0)
    };
    let (total, sigma, spont_sites, _, _, _, _, _, _, _, _, chain_radius) = channel_volumes(
        shells,
        spontaneous,
        scale,
        rates,
        temperature,
        k_mig,
        a,
        omega,
        max_r,
        jump,
        return_sites,
        cutoff + 0.05,
        true,
        &binding,
    );
    let k_rot = rotation_channel(rates, temperature).0 * scale.rotation;
    let SphereVolumes {
        d_graph,
        r_eff,
        sphere,
        decay_volume,
        mean_volume,
        tau_cell,
    } = sphere_volumes(chain_radius, max_r, a, temperature, k_mig, k_rot, field);
    RateVolumes {
        total,
        sigma,
        spont_sites,
        chain_radius,
        d_graph,
        r_eff,
        sphere,
        decay_volume,
        mean_volume,
        tau_cell,
    }
}

/// The volumes one chain radius gives: the Debye-matched radius, its free
/// sphere, and that radius as a sink in the 13³ cell where Table 1
/// measures k.
struct SphereVolumes {
    d_graph: f64,
    r_eff: f64,
    sphere: f64,
    decay_volume: f64,
    mean_volume: f64,
    tau_cell: f64,
}

fn sphere_volumes(
    chain_radius: f64,
    max_r: f64,
    a: f64,
    temperature: f64,
    k_mig: f64,
    k_rot: f64,
    field: Option<&PairField>,
) -> SphereVolumes {
    let omega = a * a * a / 4.0;
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let d_graph = dumbbell_diffusivity(a, k_mig, k_rot).unwrap_or(f64::NAN);
    let nu = (2.0 / 3.0) * d_graph / (a * a);
    let n_atoms = 4.0 * (13usize * 13 * 13) as f64;
    let energy = |radius: f64| {
        field
            .map(|item| item.effective(radius, thermal))
            .unwrap_or(0.0)
    };
    let r_eff = rgsaddle::debye_matched_radius(chain_radius, max_r, thermal, energy)
        .unwrap_or(chain_radius);
    let capture_rate = rgsaddle::smoluchowski_rate(d_graph, r_eff).unwrap_or(f64::NAN);
    let sphere = rgsaddle::volume_from_frequency(capture_rate, nu).unwrap_or(f64::NAN) / omega;
    // Table 1 measures k in the 13³ cell, where the sink's periodic images
    // speed up capture. The same radius is put in that cell here.
    let cell = rgsaddle::periodic_sink_rate(r_eff, 13.0 * a, 1.0).ok();
    let (decay_volume, mean_volume, tau_cell) = match cell {
        Some(rate) => {
            let tau = n_atoms * omega / (d_graph * rate.mean_time);
            (rate.decay * d_graph / nu / omega, n_atoms / (tau * nu), tau)
        }
        None => (f64::NAN, f64::NAN, f64::NAN),
    };
    SphereVolumes {
        d_graph,
        r_eff,
        sphere,
        decay_volume,
        mean_volume,
        tau_cell,
    }
}

fn assemble_run13() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut census =
        std::fs::read_to_string(root.join("defect_pair_run12.csv")).unwrap_or_default();
    census.push('\n');
    census
        .push_str(&std::fs::read_to_string(root.join("defect_pair_run13.csv")).unwrap_or_default());
    let mut rates = catalogue_text();
    rates.push('\n');
    rates.push_str(&census);
    let mut out = File::create(root.join("defect_pair_run13_volume.csv")).expect("volume");
    writeln!(out, "temperature_K,cutoff_A,volume_over_Omega,sigma,spontaneous_sites,kinetic_sites,demkowicz_8788,note").unwrap();
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = demkowicz_lattice(temperature);
        let mut k_mig = migration_rate(&rates, temperature).unwrap_or(f64::NAN);
        if pafi_kappa_unresolved(&rates, temperature)
            && let Some(measured) = md_channel_rate(temperature)
        {
            k_mig = measured;
        }
        let omega = a * a * a / 4.0;
        let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
        let copper = FastCopper::new();
        if (temperature - 300.0).abs() < 1.0 {
            probe_binding(&copper, a);
        }
        let field = Some(pair_field(&copper, a).expect("pair field relaxation"));
        if let Some(item) = field.as_ref() {
            for radius in [8.0_f64, 12.0, 20.0] {
                eprintln!(
                    "field {temperature:.0} K r {radius:.0} Eeff {:.4} eV mean {:.4} eV",
                    item.effective(radius, thermal),
                    item.binding(radius, "100")
                );
            }
        }
        let mut previous = None;
        for cutoff in [10.8_f64, 14.0, 18.0, 22.0, 26.0, 30.0] {
            let shells = shell_table_within(a, 24, cutoff);
            let mut spontaneous = std::collections::HashSet::new();
            for (radius, vector, _) in &shells {
                for orientation in ["100", "010", "001"] {
                    let hat = unit3(*vector);
                    let class_of = |label: &str| {
                        let axis = axis_from_label(label);
                        ((axis[0] * hat[0] + axis[1] * hat[1] + axis[2] * hat[2]).abs() * 1000.0)
                            .round() as i32
                    };
                    let want = class_of(orientation);
                    let matches = |kind: &str, label: &str, line: &str| {
                        let fields: Vec<&str> = line.split(',').collect();
                        fields.len() > 7
                            && fields[0] == kind
                            && fields[1] == "13"
                            && fields[2]
                                .parse::<f64>()
                                .ok()
                                .is_some_and(|t| (t - temperature).abs() < 1.0)
                            && fields[4]
                                .parse::<f64>()
                                .ok()
                                .is_some_and(|r| (r - radius).abs() < 0.02)
                            && fields[5] == label
                    };
                    let mut line = None;
                    for kind in ["quench", "census"] {
                        if let Some(row) =
                            census.lines().find(|row| matches(kind, orientation, row))
                        {
                            line = Some(row);
                            break;
                        }
                        if let Some(row) = census.lines().find(|row| {
                            let fields: Vec<&str> = row.split(',').collect();
                            fields.len() > 5
                                && class_of(fields[5]) == want
                                && matches(kind, fields[5], row)
                        }) {
                            line = Some(row);
                            break;
                        }
                    }
                    if line.is_some_and(|row| row.split(',').nth(6) == Some("spontaneous")) {
                        spontaneous.insert(format!("{radius:.3}-{orientation}"));
                    }
                }
            }
            let table = match temperature as i32 {
                300 => "254-286",
                500 => "161-185",
                700 => "203-210",
                _ => "",
            };
            let max_r = shells.last().map(|(radius, _, _)| *radius).unwrap_or(0.0);
            let jump = a / 2.0_f64.sqrt();
            let return_sites = return_layer_count(a, &shells, max_r);
            let binding = |radius: f64, orientation: &str| {
                field
                    .as_ref()
                    .map(|item| item.binding(radius, orientation))
                    .unwrap_or(0.0)
            };
            let RateVolumes {
                total,
                sigma,
                spont_sites,
                chain_radius,
                d_graph,
                r_eff,
                sphere,
                decay_volume,
                mean_volume,
                tau_cell,
            } = rate_volumes(
                &shells,
                &|id: &str| f64::from(u8::from(spontaneous.contains(id))),
                RateScale::RECORDED,
                &rates,
                temperature,
                k_mig,
                a,
                field.as_ref(),
                cutoff,
            );
            let cell_length = 13.0 * a;
            let energy = |radius: f64| {
                field
                    .as_ref()
                    .map(|item| item.effective(radius, thermal))
                    .unwrap_or(0.0)
            };
            let delta = previous
                .map(|prior: f64| sphere - prior)
                .unwrap_or(f64::NAN);
            writeln!(out, "{temperature},smoluchowski,{sphere:.4},,{spont_sites:.4},{r_eff:.4},{table},cutoff {cutoff:.1}; infinite crystal 24 pi R_eff/a; chain R {chain_radius:.4} before the Debye far field; D_graph {d_ps:.6}; delta {delta:.4}", d_ps = d_graph / 1.0e12).unwrap();
            writeln!(out, "{temperature},cell-decay,{decay_volume:.4},,{spont_sites:.4},{r_eff:.4},{table},cutoff {cutoff:.1}; long-time decay of the uniform survival in the 13^3 cell, L {cell_length:.4}").unwrap();
            writeln!(out, "{temperature},mean-time,{mean_volume:.4},,{spont_sites:.4},{r_eff:.4},{table},cutoff {cutoff:.1}; uniform-start mean time in the 13^3 cell {tau_cell:.6e} s; nu=(2/3)*D_graph/a^2").unwrap();
            writeln!(out, "{temperature},site-sum,{total:.4},{sigma:.4},{spont_sites:.4},,{table},cutoff {cutoff:.1}; this sum grows with the shell count").unwrap();
            eprintln!(
                "run19 {temperature} K R {cutoff:.1} sphere {sphere:.2} (d {delta:.2}) chain {chain_radius:.3} R_eff {r_eff:.3} cell {decay_volume:.2}/{mean_volume:.2} site-sum {total:.2} D {:.4}",
                d_graph / 1.0e12
            );
            if cutoff == 30.0 {
                let mut by_shell: std::collections::BTreeMap<String, usize> = Default::default();
                for key in &spontaneous {
                    let shell = key.split('-').next().unwrap_or_default().to_string();
                    *by_shell.entry(shell).or_default() += 1;
                }
                for (label, need) in [("any", 1usize), ("all", 3)] {
                    let mut isotropic = std::collections::HashSet::new();
                    for (shell, count) in &by_shell {
                        if *count >= need {
                            for orientation in ["100", "010", "001"] {
                                isotropic.insert(format!("{shell}-{orientation}"));
                            }
                        }
                    }
                    let variant = channel_volumes(
                        &shells,
                        &|id: &str| f64::from(u8::from(isotropic.contains(id))),
                        RateScale::RECORDED,
                        &rates,
                        temperature,
                        k_mig,
                        a,
                        omega,
                        max_r,
                        jump,
                        return_sites,
                        cutoff + 0.05,
                        true,
                        &binding,
                    );
                    let r_variant =
                        rgsaddle::debye_matched_radius(variant.11, max_r, thermal, energy)
                            .unwrap_or(variant.11);
                    let v_variant = 24.0 * std::f64::consts::PI * r_variant / a;
                    writeln!(out, "{temperature},isotropic-{label},{v_variant:.4},,{:.4},{r_variant:.4},{table},cutoff {cutoff:.1}; a shell is spontaneous in every orientation when {label} of its orientations are; infinite crystal", variant.2).unwrap();
                    eprintln!(
                        "run19 {temperature} K isotropic-{label} R_eff {r_variant:.3} V {v_variant:.2}"
                    );
                }
                let unbiased = channel_volumes(
                    &shells,
                    &|id: &str| f64::from(u8::from(spontaneous.contains(id))),
                    RateScale::RECORDED,
                    &rates,
                    temperature,
                    k_mig,
                    a,
                    omega,
                    max_r,
                    jump,
                    return_sites,
                    cutoff + 0.05,
                    true,
                    &|_, _| 0.0,
                );
                let v_unbiased = 24.0 * std::f64::consts::PI * unbiased.11 / a;
                writeln!(out, "{temperature},no-strain-bias,{v_unbiased:.4},,{:.4},{:.4},{table},cutoff {cutoff:.1}; hop rates without the dipole interaction and no Debye far field; infinite crystal", unbiased.2, unbiased.11).unwrap();
                eprintln!(
                    "run19 {temperature} K no-strain-bias R_eff {:.3} V {v_unbiased:.2}",
                    unbiased.11
                );
            }
            previous = Some(sphere);
        }
    }
}

/// Radius, recombined sites and total sites of each shell at one
/// temperature of `defect_pair_census.csv`. A row that did not converge
/// stops the assembly.
fn census_fractions(text: &str, temperature: f64) -> Vec<(f64, usize, usize)> {
    let mut shells: Vec<(f64, usize, usize)> = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let Some(temp) = fields.get(1).and_then(|field| field.parse::<f64>().ok()) else {
            continue;
        };
        if (temp - temperature).abs() > 1.0 {
            continue;
        }
        assert_eq!(fields[8], "1", "census row not converged: {line}");
        let radius: f64 = fields[2].parse().expect("radius");
        let sites: usize = fields[3].parse().expect("sites");
        let index = shells
            .iter()
            .position(|shell| (shell.0 - radius).abs() < 1.0e-3)
            .unwrap_or_else(|| {
                shells.push((radius, 0, 0));
                shells.len() - 1
            });
        if fields[4] == "1" {
            shells[index].1 += sites;
        }
        shells[index].2 += sites;
    }
    shells
}

/// The rates run 13 assembles from: the catalogue and the run-12 and run-13
/// rows.
fn recorded_rates() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut text = catalogue_text();
    for name in ["defect_pair_run12.csv", "defect_pair_run13.csv"] {
        text.push_str(&std::fs::read_to_string(root.join(name)).unwrap_or_default());
        text.push('\n');
    }
    text
}

/// The rate catalogue with each recorded isolated-dumbbell PAFI row replaced
/// by the `defect_pair_isolated.csv` row of the same hop and temperature.
fn converged_rates() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let isolated =
        std::fs::read_to_string(root.join("defect_pair_isolated.csv")).unwrap_or_default();
    let key = |line: &str| {
        let fields: Vec<&str> = line.split(',').collect();
        (fields.len() > 5 && fields[0] == "pafi" && fields[1] == "isolated")
            .then(|| (fields[3].to_string(), fields[5].to_string()))
    };
    let fresh: std::collections::HashSet<_> = isolated.lines().filter_map(key).collect();
    let mut kept: String = recorded_rates()
        .lines()
        .filter(|line| key(line).is_none_or(|item| !fresh.contains(&item)))
        .flat_map(|line| [line, "\n"])
        .collect();
    kept.push_str(&isolated);
    kept
}

/// The relative one-sigma error of an isolated-dumbbell PAFI rate: the
/// free-energy error over kT and the relative error of kappa.
fn isolated_spread(text: &str, hop: &str, temperature: f64) -> Option<f64> {
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    text.lines().find_map(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 15
            || fields[0] != "pafi"
            || fields[1] != "isolated"
            || fields[3] != hop
            || fields[12] != "rated"
            || fields[5].parse::<f64>().ok()? != temperature
        {
            return None;
        }
        let free = fields[9].parse::<f64>().ok()? / thermal;
        let words: Vec<&str> = fields[14].split([' ', ';']).collect();
        let after = |name: &str| {
            words
                .windows(2)
                .find(|pair| pair[0] == name)
                .and_then(|pair| pair[1].parse::<f64>().ok())
        };
        let kappa = after("kappa")?;
        let kappa_error = after("error").unwrap_or(0.0);
        Some((free * free + (kappa_error / kappa).powi(2)).sqrt())
    })
}

/// Jumps counted in the direct-MD channel rate at `temperature`.
fn md_channel_jumps(temperature: f64) -> Option<f64> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_diffusivity.csv");
    std::fs::read_to_string(path)
        .ok()?
        .lines()
        .skip(1)
        .find_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            (fields.first()?.parse::<f64>().ok()? == temperature)
                .then(|| fields.get(5)?.parse::<f64>().ok())
                .flatten()
        })
}

/// The run-13 rate model again, with every input that a relaxation decides
/// taken from a converged run: the spontaneous sites from the flow census
/// of every axis orbit in the 13³ cell, and the isolated translation and
/// rotation rates from converged end states. A lumped state of one radius
/// and dumbbell axis is spontaneous in the fraction of its sites whose orbit
/// recombines. The errors are the spread over draws of the translation,
/// rotation and capture rates within their own errors.
///
/// With `recorded` the census is combined with the recorded rates instead,
/// which separates the two changes. The recorded 300 and 500 K isolated
/// rows carry no kappa error, so that variant has no draws.
fn assemble_converged(recorded: bool) {
    let started = Instant::now();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let rates = if recorded {
        recorded_rates()
    } else {
        converged_rates()
    };
    let (name, rates_note) = if recorded {
        ("defect_pair_census_volume.csv", "recorded rates")
    } else {
        (
            "defect_pair_converged_volume.csv",
            "converged isolated rates",
        )
    };
    let draws: usize = if recorded {
        0
    } else {
        env_csv("RGSADDLE_DRAWS", "64")[0].parse().expect("draws")
    };
    let mut out = String::from(
        "temperature_K,estimator,volume_over_Omega,sigma,spontaneous_sites,R_eff_A,demkowicz_8788,note\n",
    );
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = demkowicz_lattice(temperature);
        let fractions = census_fractions(&census, temperature);
        let recombined: usize = fractions.iter().map(|shell| shell.1).sum();
        let spontaneous = |id: &str| {
            let radius: f64 = id
                .split('-')
                .next()
                .and_then(|text| text.parse().ok())
                .unwrap_or(f64::NAN);
            fractions
                .iter()
                .find(|shell| (shell.0 - radius).abs() < 2.0e-3)
                .map(|shell| shell.1 as f64 / shell.2 as f64)
                .unwrap_or(0.0)
        };
        let (mut k_mig, mut mig_spread, mut mig_note) = (
            migration_rate(&rates, temperature).unwrap_or(f64::NAN),
            isolated_spread(&rates, "translate", temperature).unwrap_or(f64::NAN),
            "PAFI translation",
        );
        if pafi_kappa_unresolved(&rates, temperature)
            && let Some(measured) = md_channel_rate(temperature)
        {
            k_mig = measured;
            mig_spread = md_channel_jumps(temperature).map_or(f64::NAN, |n| 1.0 / n.sqrt());
            mig_note = "direct-MD channel rate (PAFI kappa unresolved)";
        }
        let rot_spread = isolated_spread(&rates, "rotate", temperature).unwrap_or(0.0);
        let (k_rot, rot_note) = rotation_channel(&rates, temperature);
        let copper = FastCopper::new();
        let field = pair_field(&copper, a).expect("pair field relaxation");
        let table = match temperature as i32 {
            300 => "254-286",
            500 => "161-185",
            700 => "203-210",
            _ => "",
        };
        eprintln!(
            "converged {temperature} K: k_mig {k_mig:.4e} Hz ({mig_note}, sigma {mig_spread:.3}); k_rot {k_rot:.4e} Hz ({rot_note}, sigma {rot_spread:.3}); census {recombined} recombined sites"
        );
        for cutoff in [10.8_f64, 14.0, 18.0, 22.0, 26.0, 30.0] {
            let shells = shell_table_within(a, 24, cutoff);
            let volumes = |k_mig: f64, scale: RateScale| {
                rate_volumes(
                    &shells,
                    &spontaneous,
                    scale,
                    &rates,
                    temperature,
                    k_mig,
                    a,
                    Some(&field),
                    cutoff,
                )
            };
            let central = volumes(k_mig, RateScale::RECORDED);
            let spread: [f64; 3] = if draws > 0 {
                let mut rng = TinyRng(0x5eed_0000 + temperature as u64);
                let shifts: Vec<[f64; 3]> = (0..draws)
                    .map(|_| [rng.gauss(), rng.gauss(), rng.gauss()])
                    .collect();
                let samples: Vec<[f64; 3]> = shifts
                    .par_iter()
                    .map(|z| {
                        let varied = volumes(
                            k_mig * (z[0] * mig_spread).exp(),
                            RateScale {
                                rotation: (z[1] * rot_spread).exp(),
                                capture: z[2],
                            },
                        );
                        [varied.sphere, varied.decay_volume, varied.mean_volume]
                    })
                    .collect();
                std::array::from_fn(|which| {
                    let mean = samples.iter().map(|s| s[which]).sum::<f64>() / draws as f64;
                    (samples
                        .iter()
                        .map(|s| (s[which] - mean).powi(2))
                        .sum::<f64>()
                        / (draws.max(2) - 1) as f64)
                        .sqrt()
                })
            } else {
                [f64::NAN; 3]
            };
            let RateVolumes {
                total,
                sigma,
                spont_sites,
                chain_radius,
                d_graph,
                r_eff,
                sphere,
                decay_volume,
                mean_volume,
                tau_cell,
            } = central;
            let common = format!(
                "cutoff {cutoff:.1}; flow census; {rates_note}; k_mig {k_mig:.4e} Hz {mig_note}; k_rot {k_rot:.4e} Hz; sigma from {draws} rate draws"
            );
            let spread = spread.map(|sigma| {
                if sigma.is_finite() {
                    format!("{sigma:.4}")
                } else {
                    String::new()
                }
            });
            out.push_str(&format!("{temperature},smoluchowski,{sphere:.4},{},{spont_sites:.4},{r_eff:.4},{table},{common}; infinite crystal; chain R {chain_radius:.4}; D_graph {:.6}\n", spread[0], d_graph / 1.0e12));
            out.push_str(&format!("{temperature},cell-decay,{decay_volume:.4},{},{spont_sites:.4},{r_eff:.4},{table},{common}; slowest decay in the 13^3 cell\n", spread[1]));
            out.push_str(&format!("{temperature},mean-time,{mean_volume:.4},{},{spont_sites:.4},{r_eff:.4},{table},{common}; uniform-start mean time in the 13^3 cell {tau_cell:.6e} s\n", spread[2]));
            out.push_str(&format!("{temperature},site-sum,{total:.4},{sigma:.4},{spont_sites:.4},,{table},{common}; sigma from the capture rates only\n"));
            eprintln!(
                "converged {temperature} K R {cutoff:.1}: sphere {sphere:.2}±{} cell {decay_volume:.2}±{} mean {mean_volume:.2}±{} site-sum {total:.2} R_eff {r_eff:.3} spontaneous {spont_sites:.2} at {:.1} min",
                spread[0],
                spread[1],
                spread[2],
                started.elapsed().as_secs_f64() / 60.0
            );
        }
    }
    std::fs::write(root.join(name), out).expect("volume");
}

#[test]
#[ignore = "V_iv from the flow census and the converged isolated rates"]
fn defect_pair_converged_volume() {
    assemble_converged(false);
}

#[test]
#[ignore = "V_iv from the flow census and the recorded rates"]
fn defect_pair_census_volume() {
    assemble_converged(true);
}

#[allow(clippy::too_many_arguments)]
fn transplanted_rotation(
    copper: &FastCopper,
    checkpoint: &Mutex<File>,
    isolated: &Array1<f64>,
    isolated_dumbbell: [usize; 2],
    temperature: f64,
    a: f64,
    radius: f64,
    vector: [f64; 3],
    orientation: &str,
    translation: f64,
) {
    let repeats = 8usize;
    let PairState {
        positions,
        dumbbell,
        cell,
        recombined,
        ..
    } = placed(
        copper,
        repeats,
        a,
        vector,
        axis_from_label(orientation),
        true,
    );
    if recombined {
        write_tight(
            checkpoint,
            &format!(
                "switch,8,{temperature},{a:.3},{radius:.3},{orientation},spontaneous,,,,,relax recombines"
            ),
        );
        return;
    }
    let reactant = positions;
    let reactant_energy = basin_energy(copper, &reactant, cell);
    let iso_center = centroid(isolated, isolated_dumbbell);
    let pair_center = centroid(&reactant, dumbbell);
    let mut hint = reactant.clone();
    let mut mode = Array1::zeros(reactant.len());
    for (iso, pair) in isolated_dumbbell.into_iter().zip(dumbbell) {
        for c in 0..3 {
            let offset = isolated[3 * iso + c] - iso_center[c];
            hint[3 * pair + c] = pair_center[c] + offset;
            mode[3 * pair + c] = offset;
        }
    }
    let Some(hit) = tight_dimer(copper, cell, hint, mode) else {
        write_tight(
            checkpoint,
            &format!(
                "switch,8,{temperature},{a:.3},{radius:.3},{orientation},not_converged,,,,,transplanted saddle"
            ),
        );
        return;
    };
    let barrier = hit.energy - reactant_energy;
    let assignment = saddle_assignment(&reactant, &hit.positions, dumbbell);
    write_tight(
        checkpoint,
        &format!(
            "switch,8,{temperature},{a:.3},{radius:.3},{orientation},{assignment},{barrier:.6},{:.6e},,,transplanted <110> saddle; translation {translation:.6}; index1 {}",
            hit.force, hit.index1
        ),
    );
    if hit.index1 && assignment == "rotation" && barrier <= translation {
        let (perfect, _) = fcc(repeats, a);
        let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
        let mut product = hit.positions.clone();
        let norm = hit.mode.dot(&hit.mode).sqrt().max(1.0e-8);
        for i in 0..product.len() {
            product[i] += 0.35 * hit.mode[i] / norm;
        }
        product = descend(copper, product, cell, 2.0e-4).expect("push off the saddle relaxation");
        record_pafi(
            copper,
            checkpoint,
            bulk,
            &format!("{radius:.3}"),
            orientation,
            "rotate",
            "tangential",
            temperature,
            &reactant,
            reactant_energy,
            &hit.positions,
            &hit.mode,
            &product,
            cell,
            false,
        );
    }
}

#[test]
#[ignore = "outer cutoff and the strain-mapped <100> to <110> saddle"]
fn defect_pair_run13() {
    let started = Instant::now();
    let job = std::env::var("RUN13").unwrap_or_else(|_| "all".into());
    let copper = FastCopper::new();
    if job == "all" || job == "switch" {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run13.csv");
        let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
        let temperature = 300.0_f64;
        let a = demkowicz_lattice(temperature);
        let translation = neighbor_translation_barrier(temperature).unwrap_or(f64::NAN);
        let job_iso = rotation_job(&copper, 8, a);
        let dumbbell = [job_iso.mobile[0], job_iso.mobile[1]];
        let mut reactant = job_iso.positions.clone();
        reactant = settle(&copper, reactant, job_iso.cell, 2.0e-4).expect("reactant relaxation");
        let (hint, mode) = rotation_hint(&reactant, dumbbell);
        if let Some(hit) = tight_dimer(&copper, job_iso.cell, hint, mode) {
            for (radius, vector, _) in shell_table(a, 8) {
                if radius > 4.0 {
                    continue;
                }
                for orientation in distinct_orientations(vector) {
                    let prior = std::fs::read_to_string(&path).unwrap_or_default();
                    if checkpoint_has(
                        &prior,
                        &format!("switch,8,{temperature},{a:.3},{radius:.3},{orientation},"),
                    ) {
                        continue;
                    }
                    transplanted_rotation(
                        &copper,
                        &checkpoint,
                        &hit.positions,
                        dumbbell,
                        temperature,
                        a,
                        radius,
                        vector,
                        orientation,
                        translation,
                    );
                }
            }
        }
    }
    if job == "all" || job == "outer" {
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run13.csv");
        let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
        let temperature = 300.0_f64;
        let a = demkowicz_lattice(temperature);
        for (radius, vector, count) in shell_table_within(a, 13, 12.5) {
            if radius <= 10.8 {
                continue;
            }
            for orientation in distinct_orientations(vector) {
                let prior = std::fs::read_to_string(&path).unwrap_or_default();
                let key = format!("census,13,{temperature},{a:.3},{radius:.3},{orientation},");
                if checkpoint_has(&prior, &key) {
                    continue;
                }
                let PairState {
                    gap,
                    separation,
                    recombined,
                    ..
                } = placed(&copper, 13, a, vector, axis_from_label(orientation), true);
                let spontaneous = recombined;
                let outcome = if spontaneous { "spontaneous" } else { "bound" };
                write_tight(
                    &checkpoint,
                    &format!(
                        "census,13,{temperature},{a:.3},{radius:.3},{orientation},{outcome},{gap:.3},{separation:.3},,,sites {count}; relax only past 10.8 A"
                    ),
                );
            }
        }
    }
    if job == "all" || job == "pafi" {
        run14_rotation_pafi_700(&copper);
    }
    if job == "all" || job == "assemble" {
        assemble_run13();
    }
    eprintln!(
        "run13 {job} {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

fn run14_rotation_pafi_700(copper: &FastCopper) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run14.csv");
    if !path.exists() {
        let mut header = File::create(&path).expect("run14");
        writeln!(header, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note").unwrap();
    }
    let prior = std::fs::read_to_string(&path).unwrap_or_default();
    if prior.lines().any(|line| {
        line.starts_with("pafi,isolated,100,rotate,tangential,700,")
            && line.split(',').nth(12) == Some("rated")
    }) {
        eprintln!("700 K rotation PAFI already rated");
        return;
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let temperature = 700.0_f64;
    let a = demkowicz_lattice(temperature);
    let repeats = 8usize;
    let (perfect, cell) = fcc(repeats, a);
    let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
    let job = rotation_job(copper, repeats, a);
    let dumbbell = [job.mobile[0], job.mobile[1]];
    let mut reactant = job.positions.clone();
    reactant = settle(copper, reactant, job.cell, 2.0e-4).expect("reactant relaxation");
    let held = FastHeld {
        copper,
        cell: job.cell,
    };
    let (reactant_energy, _) = held
        .eval(reactant.view())
        .unwrap_or((f64::NAN, Array1::zeros(1)));
    let (hint, mode) = rotation_hint(&reactant, dumbbell);
    let Some(hit) = tight_dimer(copper, job.cell, hint, mode) else {
        write_tight(
            &checkpoint,
            "pafi,isolated,100,rotate,tangential,700,,,,,,,not_converged,,dimer failed",
        );
        return;
    };
    write_tight(
        &checkpoint,
        &format!(
            "rotation,isolated,100,rotate,{temperature},{:.6},{:.6e},,,index1 {}; Table 1 lattice a={a:.3}",
            hit.energy - reactant_energy,
            hit.force,
            hit.index1
        ),
    );
    let mut product = hit.positions.clone();
    let norm = hit.mode.dot(&hit.mode).sqrt().max(1.0e-8);
    for i in 0..product.len() {
        product[i] += 0.35 * hit.mode[i] / norm;
    }
    product = descend(copper, product, job.cell, 2.0e-4).expect("push off the saddle relaxation");
    record_pafi(
        copper,
        &checkpoint,
        bulk,
        "isolated",
        "100",
        "rotate",
        "tangential",
        temperature,
        &reactant,
        reactant_energy,
        &hit.positions,
        &hit.mode,
        &product,
        job.cell,
        true,
    );
}

/// The lattice each recorded isolated-dumbbell PAFI row was measured on:
/// ours, except the 700 K rotation, which was run on the Table 1 lattice.
fn isolated_lattice(hop: &str, temperature: f64) -> f64 {
    if hop == "rotate" && temperature as i32 == 700 {
        demkowicz_lattice(temperature)
    } else {
        measured_lattice(temperature)
    }
}

fn isolated_rate(copper: &FastCopper, checkpoint: &Mutex<File>, hop: &str, temperature: f64) {
    let clock = Instant::now();
    let repeats = 8usize;
    let a = isolated_lattice(hop, temperature);
    let (perfect, cell) = fcc(repeats, a);
    let (_, _, bulk) = copper.evaluate(perfect.view(), cell);
    let job = rotation_job(copper, repeats, a);
    let dumbbell = [job.mobile[0], job.mobile[1]];
    let reactant = settle(copper, job.positions.clone(), job.cell, 2.0e-4)
        .expect("isolated dumbbell relaxation");
    let reactant_energy = basin_energy(copper, &reactant, job.cell);
    let found = if hop == "rotate" {
        let (hint, mode) = rotation_hint(&reactant, dumbbell);
        tight_dimer(copper, job.cell, hint, mode)
    } else {
        let mechanisms = find_mechanisms(copper, repeats, a);
        mechanisms
            .iter()
            .find(|item| item.kind == "translate")
            .and_then(|translate| {
                mapped_hops(
                    &reactant,
                    job.cell,
                    centroid(&reactant, dumbbell),
                    axis_of(&reactant, dumbbell),
                    std::slice::from_ref(translate),
                )
                .into_iter()
                .next()
            })
            .and_then(|hop| tight_dimer(copper, job.cell, hop.saddle?, hop.mode?))
    };
    let direction = if hop == "rotate" {
        "tangential"
    } else {
        "toward"
    };
    let Some(hit) = found else {
        write_tight(
            checkpoint,
            &format!(
                "pafi,isolated,100,{hop},{direction},{temperature},,,,,,,not_converged,,dimer failed; a {a:.6}"
            ),
        );
        return;
    };
    let mut product = hit.positions.clone();
    let norm = hit.mode.dot(&hit.mode).sqrt().max(1.0e-8);
    for i in 0..product.len() {
        product[i] += 0.35 * hit.mode[i] / norm;
    }
    let product =
        descend(copper, product, job.cell, 2.0e-4).expect("push off the saddle relaxation");
    record_pafi(
        copper,
        checkpoint,
        bulk,
        "isolated",
        "100",
        hop,
        direction,
        temperature,
        &reactant,
        reactant_energy,
        &hit.positions,
        &hit.mode,
        &product,
        job.cell,
        false,
    );
    eprintln!(
        "isolated {hop} {temperature} K on a {a:.6}: {:.1} min",
        clock.elapsed().as_secs_f64() / 60.0
    );
}

#[test]
#[ignore = "isolated-dumbbell translation and rotation PAFI rates from converged end states"]
fn defect_pair_isolated_rates() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_isolated.csv");
    if !path.exists() {
        std::fs::write(&path, "kind,shell,orientation,hop,direction,temperature_K,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,rate_Hz,outcome,band_force,note\n").expect("isolated");
    }
    let prior = std::fs::read_to_string(&path).expect("isolated");
    let mut jobs = Vec::new();
    for temperature in env_csv("RGSADDLE_TEMPS", "300,500,700") {
        let temperature: f64 = temperature.parse().expect("temperature");
        for hop in env_csv("RGSADDLE_HOPS", "translate,rotate") {
            let direction = if hop == "rotate" {
                "tangential"
            } else {
                "toward"
            };
            if !checkpoint_has(
                &prior,
                &format!("pafi,isolated,100,{hop},{direction},{temperature},"),
            ) {
                jobs.push((hop, temperature));
            }
        }
    }
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    jobs.par_iter()
        .for_each(|(hop, temperature)| isolated_rate(&copper, &checkpoint, hop, *temperature));
    eprintln!(
        "isolated rates wall {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

#[test]
#[ignore = "NVE diffusivity of one <100> dumbbell on the 13^3 Table 1 cell"]
fn defect_pair_diffusivity() {
    let started = Instant::now();
    let a = 3.655_f64;
    let repeats = 13usize;
    let (perfect, cell) = fcc(repeats, a);
    let length = repeats as f64 * a;
    let copper = FastCopper::new();
    let mut positions = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    positions = settle(&copper, positions, cell, 1.0e-2).expect("positions relaxation");
    let lattice = perfect.clone();
    let steps: usize = std::env::var("MD_STEPS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(10000);
    let dt: f64 = std::env::var("MD_DT")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(2.0);
    let mass = 63.546_f64;
    let scale = 0.009648533212_f64;
    let mut rng = 0x5deece66d_u64;
    let normal = |rng: &mut u64| {
        let u1 = {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*rng >> 33) as f64 + 0.5) / 2147483648.0
        };
        let u2 = {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*rng >> 33) as f64 + 0.5) / 2147483648.0
        };
        (-2.0 * u1.max(1.0e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let root =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_diffusivity.csv");
    let mut out = File::create(&root).expect("diffusivity");
    writeln!(out, "temperature_K,time_ps,msd_A2,diffusivity_A2_per_ps,error_A2_per_ps,jumps,channel_rate_Hz,rate_implied_D,correlation,table_D").unwrap();
    let seeds: usize = std::env::var("MD_SEEDS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(2);
    for seed_index in 0..seeds {
        rng = rng.wrapping_add(0x9e37_79b9_u64.wrapping_mul(seed_index as u64 + 1));
        for temperature in [300.0_f64, 500.0, 700.0] {
            let mut x = positions.clone();
            let n = x.len() / 3;
            let k_t = 8.617333262145e-5 * temperature;
            let sigma = (k_t * scale / mass).sqrt();
            let mut v = Array1::zeros(x.len());
            for value in v.iter_mut() {
                *value = sigma * normal(&mut rng);
            }
            let mut com = [0.0; 3];
            for atom in 0..n {
                for c in 0..3 {
                    com[c] += v[3 * atom + c];
                }
            }
            for atom in 0..n {
                for c in 0..3 {
                    v[3 * atom + c] -= com[c] / n as f64;
                }
            }
            let (_, reference_grad, _) = copper.evaluate(x.view(), cell);
            let mut force = eam_force(&x, length);
            let mut mismatch = 0.0_f64;
            for i in 0..force.len() {
                mismatch = mismatch.max((force[i] + reference_grad[i]).abs());
            }
            eprintln!("md force mismatch {mismatch:.3e} eV/A");
            let equil = 2000usize;
            let mut samples = Vec::new();
            let mut jumps = 0usize;
            let mut temperature_sum = 0.0_f64;
            let mut temperature_n = 0.0_f64;
            let mut previous_site = defect_site(&x, &lattice, a, length);
            let mut previous_center = defect_center(&x, &lattice, a, length);
            let mut unwrapped = previous_center;
            let mut origin = previous_center;
            let blocks = 4usize;
            let mut block_msd = vec![0.0; blocks];
            let mut block_origin = vec![unwrapped; blocks];
            let block_steps = steps.saturating_sub(equil) / blocks;
            let probe = Instant::now();
            for step in 0..steps {
                for i in 0..x.len() {
                    v[i] += 0.5 * dt * force[i] * scale / mass;
                    x[i] += dt * v[i];
                }
                force = eam_force(&x, length);
                if step == 0 {
                    eprintln!("md force {} ms", probe.elapsed().as_secs_f64() * 1.0e3);
                }
                for i in 0..x.len() {
                    v[i] += 0.5 * dt * force[i] * scale / mass;
                }
                let tau = if step < equil { 200.0 } else { 5000.0 };
                let retain = (-dt / tau).exp();
                let kick = (k_t * scale / mass * (1.0 - retain * retain)).sqrt();
                for value in v.iter_mut() {
                    *value = retain * *value + kick * normal(&mut rng);
                }
                let mut com = [0.0; 3];
                for atom in 0..n {
                    for c in 0..3 {
                        com[c] += v[3 * atom + c];
                    }
                }
                for atom in 0..n {
                    for c in 0..3 {
                        v[3 * atom + c] -= com[c] / n as f64;
                    }
                }
                if step == equil {
                    previous_center = defect_center(&x, &lattice, a, length);
                    previous_site = defect_site(&x, &lattice, a, length);
                    unwrapped = previous_center;
                    origin = previous_center;
                    block_origin = vec![unwrapped; blocks];
                }
                if step < equil {
                    continue;
                }
                if (step + 1) % 50 == 0 {
                    let mut kinetic = 0.0;
                    for value in v.iter() {
                        kinetic += value * value;
                    }
                    kinetic *= 0.5 * mass / scale;
                    temperature_sum += kinetic / (1.5 * n as f64 * 8.617333262145e-5);
                    temperature_n += 1.0;
                }
                if (step + 1) % 20 == 0 {
                    let center = defect_center(&x, &lattice, a, length);
                    let delta = min_image(center, previous_center, length);
                    unwrapped = [
                        unwrapped[0] + delta[0],
                        unwrapped[1] + delta[1],
                        unwrapped[2] + delta[2],
                    ];
                    let site = defect_site(&x, &lattice, a, length);
                    if site != previous_site {
                        let hop2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                        if hop2 > 1.44 {
                            jumps += 1;
                        }
                        previous_site = site;
                    }
                    previous_center = center;
                    let msd = {
                        let d = [
                            unwrapped[0] - origin[0],
                            unwrapped[1] - origin[1],
                            unwrapped[2] - origin[2],
                        ];
                        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
                    };
                    samples.push(msd);
                    let produced = step.saturating_sub(equil);
                    let block = produced / block_steps.max(1);
                    if block < blocks {
                        if samples.len() == 1
                            || block != produced.saturating_sub(20) / block_steps.max(1)
                        {
                            block_origin[block] = unwrapped;
                        }
                        let d = [
                            unwrapped[0] - block_origin[block][0],
                            unwrapped[1] - block_origin[block][1],
                            unwrapped[2] - block_origin[block][2],
                        ];
                        block_msd[block] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                    }
                }
            }
            let time_ps = steps.saturating_sub(equil) as f64 * dt * 1.0e-3;
            let msd = *samples.last().unwrap_or(&0.0);
            let diffusivity = msd / (6.0 * time_ps);
            let block_time = time_ps / blocks as f64;
            let block_d: Vec<f64> = block_msd
                .iter()
                .map(|value| value / (6.0 * block_time))
                .collect();
            let mean = block_d.iter().sum::<f64>() / blocks as f64;
            let error = (block_d
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / (blocks as f64 * (blocks - 1) as f64))
                .sqrt();
            let channel = jumps as f64 / time_ps * 1.0e12 / 8.0;
            let implied = (2.0 / 3.0) * (channel / 1.0e12) * a * a;
            let correlation = if implied > 0.0 {
                diffusivity / implied
            } else {
                f64::NAN
            };
            let table = match temperature as i32 {
                300 => 0.143,
                500 => 0.526,
                700 => 0.910,
                _ => f64::NAN,
            };
            let measured_t = if temperature_n > 0.0 {
                temperature_sum / temperature_n
            } else {
                f64::NAN
            };
            writeln!(out, "{temperature},{time_ps:.3},{msd:.4},{diffusivity:.4},{error:.4},{jumps},{channel:.6e},{implied:.4},{correlation:.3},{table},seed {seed_index}; measured_T {measured_t:.1}").unwrap();
            eprintln!(
                "md seed {seed_index} {temperature} K measured {measured_t:.1} K D {diffusivity:.4} ± {error:.4} jumps {jumps} channel {channel:.3e} implied {implied:.4} f {correlation:.3} table {table} in {:.1}s",
                started.elapsed().as_secs_f64()
            );
        }
    }
}

fn eam_force(positions: &Array1<f64>, length: f64) -> Array1<f64> {
    let n = positions.len() / 3;
    let nside = ((length / RC).floor() as i32).max(3);
    let bins_n = (nside as usize).pow(3);
    let mut bins = vec![Vec::new(); bins_n];
    let mut which = vec![[0i32; 3]; n];
    for i in 0..n {
        let mut ijk = [0i32; 3];
        for c in 0..3 {
            let mut bin =
                (positions[3 * i + c].rem_euclid(length) / length * nside as f64).floor() as i32;
            if bin >= nside {
                bin = nside - 1;
            }
            if bin < 0 {
                bin = 0;
            }
            ijk[c] = bin;
        }
        which[i] = ijk;
        let slot = ((ijk[0] * nside + ijk[1]) * nside + ijk[2]) as usize;
        bins[slot].push(i);
    }
    let cutoff2 = RC * RC;
    use rayon::prelude::*;
    let (rho, pairs) = (0..n)
        .into_par_iter()
        .fold(
            || (vec![0.0; n], Vec::new()),
            |mut acc, i| {
                let origin = which[i];
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let ix = (origin[0] + dx).rem_euclid(nside);
                            let iy = (origin[1] + dy).rem_euclid(nside);
                            let iz = (origin[2] + dz).rem_euclid(nside);
                            let slot = ((ix * nside + iy) * nside + iz) as usize;
                            for &j in &bins[slot] {
                                if j <= i {
                                    continue;
                                }
                                let mut d = [0.0; 3];
                                let mut r2 = 0.0;
                                for c in 0..3 {
                                    let mut delta = positions[3 * j + c] - positions[3 * i + c];
                                    delta -= length * (delta / length).round();
                                    d[c] = delta;
                                    r2 += delta * delta;
                                }
                                if r2 < cutoff2 && r2 > 1.0e-16 {
                                    let r = r2.sqrt();
                                    let weight = density(r).0;
                                    acc.0[i] += weight;
                                    acc.0[j] += weight;
                                    acc.1.push((i, j, d, r));
                                }
                            }
                        }
                    }
                }
                acc
            },
        )
        .reduce(
            || (vec![0.0; n], Vec::new()),
            |mut left, right| {
                for (slot, value) in left.0.iter_mut().zip(right.0) {
                    *slot += value;
                }
                left.1.extend(right.1);
                left
            },
        );
    let mut fp = vec![0.0; n];
    for i in 0..n {
        fp[i] = embed(rho[i]).1;
    }
    let mut gradient = vec![0.0; positions.len()];
    for (i, j, d, r) in &pairs {
        let (_, dv) = pair(*r);
        let (_, drho) = density(*r);
        let deriv = dv + (fp[*i] + fp[*j]) * drho;
        let inv_r = 1.0 / r;
        for c in 0..3 {
            let force = deriv * d[c] * inv_r;
            gradient[3 * i + c] -= force;
            gradient[3 * j + c] += force;
        }
    }
    Array1::from(gradient).mapv(|value| -value)
}

fn defect_site(
    positions: &Array1<f64>,
    _lattice: &Array1<f64>,
    a: f64,
    length: f64,
) -> (i32, i32, i32) {
    let mut occupied: std::collections::HashMap<(i32, i32, i32), u16> =
        std::collections::HashMap::new();
    for atom in 0..positions.len() / 3 {
        let point = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let key = site_key(point, a, length);
        *occupied.entry(key).or_insert(0) += 1;
    }
    occupied
        .into_iter()
        .max_by_key(|(_, count)| *count)
        .map(|(key, _)| key)
        .unwrap_or((0, 0, 0))
}

fn defect_center(positions: &Array1<f64>, _lattice: &Array1<f64>, a: f64, length: f64) -> [f64; 3] {
    let mut best = [(0.0_f64, [0.0; 3]); 2];
    let half = 0.5 * a;
    for atom in 0..positions.len() / 3 {
        let point = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let key = site_key(point, a, length);
        let site = [
            key.0 as f64 * half,
            key.1 as f64 * half,
            key.2 as f64 * half,
        ];
        let image = min_image(point, site, length);
        let dist2 = image[0] * image[0] + image[1] * image[1] + image[2] * image[2];
        let place = [site[0] + image[0], site[1] + image[1], site[2] + image[2]];
        if dist2 > best[0].0 {
            best[1] = best[0];
            best[0] = (dist2, place);
        } else if dist2 > best[1].0 {
            best[1] = (dist2, place);
        }
    }
    let second = min_image(best[1].1, best[0].1, length);
    [
        best[0].1[0] + 0.5 * second[0],
        best[0].1[1] + 0.5 * second[1],
        best[0].1[2] + 0.5 * second[2],
    ]
}

fn site_key(point: [f64; 3], a: f64, length: f64) -> (i32, i32, i32) {
    let half = 0.5 * a;
    let period = (length / half).round() as i32;
    let mut key = [0i32; 3];
    for c in 0..3 {
        key[c] = (point[c].rem_euclid(length) / half).round() as i32;
        key[c] = key[c].rem_euclid(period);
    }
    if (key[0] + key[1] + key[2]) % 2 != 0 {
        let mut worst = 0;
        let mut worst_err = -1.0;
        for c in 0..3 {
            let err = (point[c].rem_euclid(length) / half - key[c] as f64).abs();
            if err > worst_err {
                worst_err = err;
                worst = c;
            }
        }
        key[worst] = (key[worst] + 1).rem_euclid(period);
    }
    (key[0], key[1], key[2])
}

fn min_image(point: [f64; 3], origin: [f64; 3], length: f64) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..3 {
        let mut delta = point[c] - origin[c];
        delta -= length * (delta / length).round();
        out[c] = delta;
    }
    out
}

fn neighbor_translation_barrier(temperature: f64) -> Option<f64> {
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_translation.csv"),
    )
    .ok()?;
    text.lines().find_map(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6
            || fields[0] != "translation"
            || fields[1] != "isolated"
            || fields[3] != "neighbor"
        {
            return None;
        }
        let temp = fields[4].parse::<f64>().ok()?;
        let barrier = fields[5].parse::<f64>().ok()?;
        ((temp - temperature).abs() < 1.0 && barrier > 0.0).then_some(barrier)
    })
}

/// Tracer diffusivity per (temperature, timestep, friction) from the replica CSV.
///
/// Returns the replica mean of the multi-origin MSD slope, its standard
/// error, the mean committed-jump diffusivity, and the replica count.
#[allow(clippy::type_complexity)]
fn md_replica_diffusivity() -> std::collections::BTreeMap<(i64, i64, i64), (f64, f64, f64, usize)> {
    let path = std::env::var("RUN19_DIFFUSION")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_sia_d.csv")
        });
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut groups: std::collections::BTreeMap<(i64, i64, i64), Vec<(f64, f64)>> =
        Default::default();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        let parse = |i: usize| fields.get(i).and_then(|v| v.parse::<f64>().ok());
        let (Some(t), Some(dt), Some(gamma), Some(d), Some(jump)) =
            (parse(0), parse(3), parse(4), parse(12), parse(14))
        else {
            continue;
        };
        let key = (
            t.round() as i64,
            (dt * 1000.0).round() as i64,
            (gamma * 1000.0).round() as i64,
        );
        groups.entry(key).or_default().push((d, jump));
    }
    groups
        .into_iter()
        .map(|(key, values)| {
            let n = values.len();
            let mean = values.iter().map(|v| v.0).sum::<f64>() / n as f64;
            let var =
                values.iter().map(|v| (v.0 - mean).powi(2)).sum::<f64>() / ((n.max(2) - 1) as f64);
            let jump = values.iter().map(|v| v.1).sum::<f64>() / n as f64;
            (key, (mean, (var / n as f64).sqrt(), jump, n))
        })
        .collect()
}

fn rate_model_radius(temperature: f64) -> Option<f64> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run13_volume.csv");
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .rfind(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            fields.len() > 7
                && fields[1] == "smoluchowski"
                && fields[0]
                    .parse::<f64>()
                    .is_ok_and(|t| (t - temperature).abs() < 1.0)
        })
        .and_then(|line| line.split(',').nth(5)?.parse().ok())
}

/// Largest log-likelihood on a grid, refined by the parabola through its neighbours,
/// and the interval where the profile stays within 1/2 of it.
fn profile_peak(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, f64) {
    let best = (0..xs.len())
        .max_by(|&a, &b| ys[a].total_cmp(&ys[b]))
        .unwrap_or(0);
    let mut peak = xs[best];
    let mut top = ys[best];
    if best > 0 && best + 1 < xs.len() {
        let (y0, y1, y2) = (ys[best - 1], ys[best], ys[best + 1]);
        let h = xs[best + 1] - xs[best];
        let curvature = y0 - 2.0 * y1 + y2;
        if curvature < 0.0 {
            let shift = 0.5 * h * (y0 - y2) / curvature;
            peak += shift;
            top = y1 - 0.125 * (y0 - y2) * (y0 - y2) / curvature;
        }
    }
    let crossing = |range: Vec<usize>| {
        let mut previous = best;
        for i in range {
            if ys[i] < top - 0.5 {
                let (xa, ya, xb, yb) = (xs[previous], ys[previous], xs[i], ys[i]);
                return xa + (top - 0.5 - ya) * (xb - xa) / (yb - ya);
            }
            previous = i;
        }
        f64::NAN
    };
    let low = crossing((0..best).rev().collect());
    let high = crossing((best + 1..xs.len()).collect());
    (peak, low, high, top)
}

#[test]
#[ignore = "fit the direct-MD capture times to a Smoluchowski sink in the periodic cell"]
#[allow(clippy::needless_range_loop, clippy::type_complexity)]
fn direct_md_capture_radius() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let files = std::env::var("RUN19_CAPTURE")
        .unwrap_or_else(|_| "validation/direct_md_long.csv,validation/direct_md_ws.csv".into());
    let diffusion = md_replica_diffusivity();
    let mut groups: std::collections::BTreeMap<
        ((i64, i64, i64), &str),
        Vec<(f64, f64, rgsaddle::CaptureRecord)>,
    > = Default::default();
    for file in files.split(',') {
        let text = std::fs::read_to_string(root.join(file)).expect("capture csv");
        let criterion = if text
            .lines()
            .next()
            .is_some_and(|header| header.contains("ws_defects"))
        {
            "wigner-seitz"
        } else {
            "12-fold"
        };
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split(',').collect();
            let p = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok());
            let (
                Some(t),
                Some(a),
                Some(n),
                Some(x),
                Some(y),
                Some(z),
                Some(dt),
                Some(gamma),
                Some(window),
                Some(captured),
                Some(time),
            ) = (
                p(0),
                p(1),
                p(2),
                p(5),
                p(6),
                p(7),
                p(11),
                p(12),
                p(13),
                p(14),
                p(15),
            )
            else {
                continue;
            };
            let key = (
                t.round() as i64,
                (dt * 1000.0).round() as i64,
                (gamma * 1000.0).round() as i64,
            );
            let record = rgsaddle::CaptureRecord {
                start: [x, y, z],
                time: if captured > 0.5 { time } else { window },
                captured: captured > 0.5,
            };
            groups
                .entry((key, criterion))
                .or_default()
                .push((n * a, a, record));
        }
    }
    let mut out = File::create(root.join("benchmarks/defect_pair_run19_capture_radius.csv"))
        .expect("radius csv");
    writeln!(out, "temperature_K,capture_test,dt_fs,gamma_per_ps,min_start_A,records,captured,D_md,D_md_se,replicas,R_fit_A,R_low_A,R_high_A,R_at_D_minus_A,R_at_D_plus_A,R_joint_A,D_joint,tail_from_ps,relax_ps,tail_events,k_tail_A3_per_ps,k_tail_se,R_tail_A,V_tail_cell,R_rate_model_A,V_inf_fit,V_cell_decay_fit,V_cell_mean_fit,V_cell_decay_at_R_low,V_cell_decay_at_R_high").unwrap();
    for ((key, criterion), rows) in &groups {
        let temperature = key.0 as f64;
        let Some(&(d_md, d_se, _, replicas)) = diffusion.get(key) else {
            eprintln!("no MD diffusivity for {key:?}; skipped");
            continue;
        };
        let (length, lattice) = (rows[0].0, rows[0].1);
        let window = rows.iter().map(|r| r.2.time).fold(0.0, f64::max);
        let r0 = |record: &rgsaddle::CaptureRecord| {
            record.start.iter().map(|v| v * v).sum::<f64>().sqrt()
        };
        let records: Vec<rgsaddle::CaptureRecord> =
            rows.iter().map(|r| r.2).filter(|r| r0(r) > 4.4).collect();
        let mut scale: Vec<f64> = (0..=15).map(|i| 0.5 + 0.1 * i as f64).collect();
        scale.extend((1..=8).map(|i| 2.0 + 0.25 * i as f64));
        let spacing: f64 = std::env::var("RUN19_SPACING")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0);
        let mut diffusivities: Vec<f64> = scale.iter().map(|s| s * d_md).collect();
        diffusivities.push((d_md - d_se).max(0.05 * d_md));
        diffusivities.push(d_md + d_se);
        let radii: Vec<f64> = (0..=36)
            .map(|i| 1.0 + 0.25 * i as f64)
            .filter(|r| *r < 8.0)
            .collect();
        let started = Instant::now();
        let terms: Vec<Vec<Vec<f64>>> = radii
            .iter()
            .map(|&r| {
                rgsaddle::periodic_capture_log_likelihood(
                    r,
                    length,
                    spacing,
                    SAMPLE_INTERVAL_PS,
                    &diffusivities,
                    &records,
                )
                .expect("likelihood")
            })
            .collect();
        eprintln!(
            "{temperature} K {criterion}: {} radii in {:.0}s",
            radii.len(),
            started.elapsed().as_secs_f64()
        );
        let rate_r = rate_model_radius(temperature).unwrap_or(f64::NAN);
        for cut in [4.4_f64, 6.3, 8.1] {
            let keep: Vec<usize> = (0..records.len())
                .filter(|&i| r0(&records[i]) > cut)
                .collect();
            let usable: Vec<usize> = (0..radii.len()).filter(|&j| radii[j] < cut - 0.2).collect();
            let score = |j: usize, d: usize| keep.iter().map(|&i| terms[j][d][i]).sum::<f64>();
            let xs: Vec<f64> = usable.iter().map(|&j| radii[j]).collect();
            let fixed = scale
                .iter()
                .position(|s| (s - 1.0).abs() < 1.0e-9)
                .expect("unit scale");
            let at = |d: usize| {
                profile_peak(
                    &xs,
                    &usable.iter().map(|&j| score(j, d)).collect::<Vec<_>>(),
                )
            };
            let (r_fit, r_low, r_high, _) = at(fixed);
            let (r_minus, ..) = at(diffusivities.len() - 2);
            let (r_plus, ..) = at(diffusivities.len() - 1);
            let mut joint = (f64::MIN, f64::NAN, f64::NAN);
            for d in 0..scale.len() {
                let (r, _, _, top) = at(d);
                if top > joint.0 {
                    joint = (top, r, diffusivities[d]);
                }
            }
            for (j, &r) in xs.iter().enumerate() {
                eprintln!(
                    "{temperature} K cut {cut} R {r:.2} logL {:.3}",
                    score(usable[j], fixed)
                );
            }
            let captured = keep.iter().filter(|&&i| records[i].captured).count();
            let rates =
                |r: f64| rgsaddle::periodic_sink_rate(r, length, 1.0).expect("periodic rate");
            let cell_volume = |r: f64| {
                if r.is_finite() && r > 0.0 {
                    6.0 * rates(r).decay / lattice
                } else {
                    f64::NAN
                }
            };
            let (v_inf, v_decay, v_mean) = if r_fit.is_finite() && r_fit > 0.0 {
                let rate = rates(r_fit);
                (
                    24.0 * std::f64::consts::PI * r_fit / lattice,
                    6.0 * rate.decay / lattice,
                    6.0 * rate.mean_time / lattice,
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };
            let (v_low, v_high) = (cell_volume(r_low), cell_volume(r_high));
            for t0 in [50.0_f64, 100.0, 150.0] {
                let mut events = 0usize;
                let mut exposure = 0.0;
                for &i in &keep {
                    let record = &records[i];
                    if record.time > t0 {
                        exposure += record.time - t0;
                        events += usize::from(record.captured);
                    }
                }
                let lambda = events as f64 / exposure.max(1.0e-12);
                let k_tail = lambda * length.powi(3);
                let k_se = k_tail / (events.max(1) as f64).sqrt();
                let v_tail = 6.0 * k_tail / (d_md * lattice);
                let target = k_tail / d_md;
                // The tail is the slowest mode only after a few of these.
                let relax =
                    length * length / (4.0 * std::f64::consts::PI * std::f64::consts::PI * d_md);
                let (mut lo, mut hi) = (0.2_f64, 0.45 * length - 1.0);
                let r_tail = if events == 0 || rates(hi).decay < target {
                    f64::NAN
                } else {
                    for _ in 0..30 {
                        let mid = 0.5 * (lo + hi);
                        if rates(mid).decay < target {
                            lo = mid
                        } else {
                            hi = mid
                        }
                        if hi - lo < 0.02 {
                            break;
                        }
                    }
                    0.5 * (lo + hi)
                };
                writeln!(out, "{temperature:.0},{criterion},{:.4},{:.4},{cut:.2},{},{captured},{d_md:.6},{d_se:.6},{replicas},{r_fit:.4},{r_low:.4},{r_high:.4},{r_minus:.4},{r_plus:.4},{:.4},{:.6},{t0:.0},{relax:.1},{events},{k_tail:.4},{k_se:.4},{r_tail:.4},{v_tail:.4},{rate_r:.4},{v_inf:.4},{v_decay:.4},{v_mean:.4},{v_low:.4},{v_high:.4}",
                    key.1 as f64 / 1000.0, key.2 as f64 / 1000.0, keep.len(), joint.1, joint.2).unwrap();
                eprintln!(
                    "{temperature} K {criterion} dt {} cut {cut}: R {r_fit:.3} [{r_low:.3}, {r_high:.3}] D±σ {r_minus:.3}/{r_plus:.3} joint R {:.3} D {:.4} tail({t0}) k {k_tail:.2}±{k_se:.2} R {r_tail:.3} V {v_tail:.1} n {events} rate-model {rate_r:.3} V {v_inf:.1}/{v_decay:.1}/{v_mean:.1} window {window}",
                    key.1, joint.1, joint.2
                );
            }
        }
    }
}

const SAMPLE_INTERVAL_PS: f64 = 0.25;

/// Largest difference, in Å, allowed between the FIRE and L-BFGS dipole
/// envelope radii.
const ENVELOPE_AGREEMENT_A: f64 = 0.005;

fn dipole_envelope(
    field: &PairField,
    directions: &[[f64; 3]],
    radius: f64,
    attractive: bool,
) -> f64 {
    let mut value = if attractive { f64::MAX } else { f64::MIN };
    for direction in directions {
        for which in 0..3 {
            let energy = field.interaction(radius, *direction, which);
            value = if attractive {
                value.min(energy)
            } else {
                value.max(energy)
            };
        }
    }
    value
}

fn envelope_radii(
    field: &PairField,
    directions: &[[f64; 3]],
    chain: f64,
    thermal: f64,
) -> (f64, f64) {
    let matched = |attractive: bool| {
        rgsaddle::debye_matched_radius(chain, chain, thermal, |r| {
            dipole_envelope(field, directions, r, attractive)
        })
        .unwrap_or(f64::NAN)
    };
    (matched(true), matched(false))
}

#[test]
#[ignore = "bound the elastic drift on the capture radius by the most attractive dipole envelope"]
fn defect_pair_run19_field() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut out = File::create(root.join("defect_pair_run19_field.csv")).expect("field csv");
    writeln!(out, "temperature_K,radius_A,E_min_eV,E_max_eV,R_rate_model_A,R_attractive_envelope_A,R_repulsive_envelope_A").unwrap();
    let directions = fibonacci_directions(200);
    let mut problems = Vec::new();
    for temperature in [300.0_f64, 500.0, 700.0] {
        let a = demkowicz_lattice(temperature);
        let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
        let field = pair_field(&copper, a).expect("pair field relaxation");
        let check = pair_field_with(&copper, a, &field_relax().with_method(Method::lbfgs()))
            .expect("L-BFGS pair field relaxation");
        let chain = rate_model_radius(temperature).unwrap_or(f64::NAN);
        let (attractive, repulsive) = envelope_radii(&field, &directions, chain, thermal);
        let (check_attractive, check_repulsive) =
            envelope_radii(&check, &directions, chain, thermal);
        let spread = (attractive - check_attractive)
            .abs()
            .max((repulsive - check_repulsive).abs());
        eprintln!(
            "{temperature:.0} K L-BFGS envelopes {check_attractive:.4} / {check_repulsive:.4}, largest difference from FIRE {spread:.2e} A"
        );
        if spread.is_nan() || spread > ENVELOPE_AGREEMENT_A {
            problems.push(format!("{temperature:.0} K: FIRE {attractive:.4}/{repulsive:.4}, L-BFGS {check_attractive:.4}/{check_repulsive:.4}"));
        }
        for radius in [3.0_f64, 4.0, 5.0, 6.0, 8.0, 10.0] {
            let (low, high) = (
                dipole_envelope(&field, &directions, radius, true),
                dipole_envelope(&field, &directions, radius, false),
            );
            writeln!(out, "{temperature:.0},{radius:.1},{low:.6},{high:.6},{chain:.4},{attractive:.4},{repulsive:.4}").unwrap();
            eprintln!("{temperature:.0} K r {radius:.1} E [{low:.5}, {high:.5}] eV");
        }
        eprintln!(
            "{temperature:.0} K R_eff {chain:.4} attractive envelope {attractive:.4} repulsive envelope {repulsive:.4}"
        );
    }
    eprintln!("run19 field {:.1} s", started.elapsed().as_secs_f64());
    assert!(
        problems.is_empty(),
        "FIRE and L-BFGS envelopes differ by more than {ENVELOPE_AGREEMENT_A} A:\n{}",
        problems.join("\n")
    );
}

fn graph_diffusivity(temperature: f64) -> Option<f64> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_run13_volume.csv");
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|line| {
        let fields: Vec<&str> = line.split(',').collect();
        fields.len() > 7
            && fields[1] == "smoluchowski"
            && fields[0]
                .parse::<f64>()
                .is_ok_and(|t| (t - temperature).abs() < 1.0)
    })?;
    let note = line.split_once("D_graph ")?.1;
    note.split(';').next()?.trim().parse().ok()
}

#[test]
#[ignore = "summarise the free-dumbbell replicas into D, the jump diffusivity and f"]
#[allow(clippy::type_complexity)]
fn direct_md_diffusivity_summary() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/direct_md_sia_d.csv");
    let text = std::fs::read_to_string(path).expect("replica csv");
    let mut groups: std::collections::BTreeMap<(i64, i64, i64), Vec<(f64, f64, f64)>> =
        Default::default();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        let p = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok());
        let (Some(t), Some(dt), Some(gamma), Some(jumps), Some(d), Some(jump)) =
            (p(0), p(3), p(4), p(9), p(12), p(14))
        else {
            continue;
        };
        groups
            .entry((
                t.round() as i64,
                (dt * 1000.0).round() as i64,
                (gamma * 1000.0).round() as i64,
            ))
            .or_default()
            .push((d, jump, jumps));
    }
    let stats = |values: &[f64]| {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        (mean, (var / n).sqrt())
    };
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut out = File::create(root.join("defect_pair_run19_diffusivity.csv")).expect("summary");
    writeln!(out, "temperature_K,dt_fs,gamma_per_ps,replicas,jumps,D_msd,D_msd_se,D_jump,D_jump_se,f_md,f_md_se,D_graph,f_graph,f_graph_se").unwrap();
    for ((t, dt, gamma), rows) in &groups {
        let (d, d_se) = stats(&rows.iter().map(|r| r.0).collect::<Vec<_>>());
        let (jump, jump_se) = stats(&rows.iter().map(|r| r.1).collect::<Vec<_>>());
        let jumps: f64 = rows.iter().map(|r| r.2).sum();
        let f_md = d / jump;
        let f_md_se = f_md * ((d_se / d).powi(2) + (jump_se / jump).powi(2)).sqrt();
        let graph = graph_diffusivity(*t as f64).unwrap_or(f64::NAN);
        writeln!(out, "{t},{:.1},{:.1},{},{jumps:.0},{d:.5},{d_se:.5},{jump:.5},{jump_se:.5},{f_md:.4},{f_md_se:.4},{graph:.6},{:.4},{:.4}",
            *dt as f64 / 1000.0, *gamma as f64 / 1000.0, rows.len(), d / graph, d_se / graph).unwrap();
        eprintln!(
            "{t} K dt {dt} gamma {gamma}: n {} D {d:.4}±{d_se:.4} ({:.1}%) D_jump {jump:.4}±{jump_se:.4} f_md {f_md:.3}±{f_md_se:.3} f_graph {:.3}±{:.3}",
            rows.len(),
            100.0 * d_se / d,
            d / graph,
            d_se / graph
        );
    }
}

/// Largest squared relative vector, in units of a/2, of the flow census:
/// the region whose hops the near-shell campaign computes.
const EXPLICIT_NORM2: i64 = 34;

/// fcc lattice vectors in units of a/2 out to `reach` per component.
fn fcc_offsets(reach: i32) -> Vec<rgsaddle::Vec3i> {
    let mut out = Vec::new();
    for x in -reach..=reach {
        for y in -reach..=reach {
            for z in -reach..=reach {
                if (x + y + z).rem_euclid(2) == 0 {
                    out.push([x, y, z]);
                }
            }
        }
    }
    out
}

/// The `<100>` dumbbell next to a vacancy: eight translations that turn
/// the axis into the hop's perpendicular part, and two `<110>` rotation
/// saddles to each other axis, closed under the cubic group.
fn dumbbell_moves(group: &[rgsaddle::Mat3i]) -> Vec<rgsaddle::PairMove> {
    let seed = [
        rgsaddle::PairMove::new("translate", [1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]),
        rgsaddle::PairMove::new("rotate", [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]),
    ];
    rgsaddle::PairMove::orbit(&seed, group)
}

fn dumbbell_network(max_norm2: i64) -> (Vec<rgsaddle::Mat3i>, rgsaddle::PairNetwork) {
    let group = rgsaddle::cubic_group();
    let reach = (max_norm2 as f64).sqrt().ceil() as i32 + 2;
    let network = rgsaddle::PairNetwork::build(
        &group,
        &[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        &dumbbell_moves(&group),
        fcc_offsets(reach),
        max_norm2,
    )
    .expect("dumbbell network");
    (group, network)
}

fn config_label(config: rgsaddle::PairConfig) -> String {
    let [x, y, z] = config.offset;
    let [i, j, k] = config.orientation;
    format!("{x}_{y}_{z}:{i}{j}{k}")
}

fn hop_label(hop: &rgsaddle::PairHop) -> String {
    let [i, j, k] = hop.via;
    format!(
        "{}>{}@{i}{j}{k}",
        config_label(hop.from),
        config_label(hop.to)
    )
}

/// Census orbit of a state: the `(|i|, max(|j|,|k|), min(|j|,|k|))`
/// representative of its member with the dumbbell along x.
fn state_miller(config: rgsaddle::PairConfig, group: &[rgsaddle::Mat3i]) -> String {
    let member = group
        .iter()
        .map(|g| config.transformed(g))
        .find(|image| image.orientation == [1, 0, 0])
        .expect("every orbit holds an x dumbbell");
    let [i, j, k] = member.offset;
    let (high, low) = (j.abs().max(k.abs()), j.abs().min(k.abs()));
    format!("{} {high} {low}", i.abs())
}

/// Census orbits that recombine on the flow at `temperature`, by
/// representative.
fn census_spontaneous(text: &str, temperature: f64) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut seen = 0usize;
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 9
            || fields[0] != "13"
            || fields[1].parse::<f64>().ok() != Some(temperature)
        {
            continue;
        }
        assert_eq!(fields[8], "1", "census row not converged: {line}");
        seen += 1;
        if fields[4] == "1" {
            out.insert(fields[7].to_string());
        }
    }
    assert!(seen > 0, "no 13^3 census rows at {temperature} K");
    out
}

/// The relaxed basin of a vacancy–dumbbell cell, relative to the vacancy:
/// `recombined`, a [`config_label`], or `other:` with the defect counts.
/// Without a vacancy the reference is the site nearest `reference`.
fn pair_basin(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    repeats: usize,
    a: f64,
    reference: [f64; 3],
) -> String {
    let (sites, _) = fcc(repeats, a);
    let occupancy =
        rgsaddle::site_occupancy(sites.view(), positions.view(), cell).expect("site occupancy");
    if occupancy.is_perfect() {
        return "recombined".into();
    }
    let vacancies = occupancy.vacancies();
    let crowded = occupancy.crowded();
    if vacancies.len() > 1 || crowded.len() != 1 || occupancy.counts[crowded[0]] != 2 {
        return format!(
            "other:v{}i{}",
            occupancy.vacancy_count(),
            occupancy.interstitial_count()
        );
    }
    let site = |index: usize| point_of(&sites, index);
    let origin = match vacancies.first() {
        Some(index) => *index,
        None => (0..sites.len() / 3)
            .min_by(|l, r| {
                mic_distance(site(*l), reference, cell).total_cmp(&mic_distance(
                    site(*r),
                    reference,
                    cell,
                ))
            })
            .expect("sites"),
    };
    let mic = |from: [f64; 3], to: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|c| {
            let length = cell[c][c];
            let delta = to[c] - from[c];
            delta - length * (delta / length).round()
        })
    };
    let offset = mic(site(origin), site(crowded[0]));
    let atoms: Vec<usize> = (0..positions.len() / 3)
        .filter(|atom| occupancy.site_of[*atom] == crowded[0])
        .collect();
    let bond = mic(point_of(positions, atoms[0]), point_of(positions, atoms[1]));
    let axis = (0..3)
        .max_by(|l, r| bond[*l].abs().total_cmp(&bond[*r].abs()))
        .expect("axis");
    let mut orientation = [0; 3];
    orientation[axis] = 1;
    let moved = if vacancies.is_empty() || mic_distance(site(origin), reference, cell) < 0.3 {
        ""
    } else {
        "moved:"
    };
    format!(
        "{moved}{}",
        config_label(rgsaddle::PairConfig::new(
            std::array::from_fn(|c| (offset[c] * 2.0 / a).round() as i32),
            orientation,
        ))
    )
}

/// A dumbbell state of the near-shell campaign on the flow.
struct NetworkState {
    positions: Array1<f64>,
    dumbbell: [usize; 2],
    vacancy: [f64; 3],
    cell: [[f64; 3]; 3],
    energy: f64,
    basin: String,
}

fn network_state(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    config: rgsaddle::PairConfig,
    vacancy: bool,
) -> (NetworkState, usize, f64) {
    let shell = config.offset.map(|c| 0.5 * a * c as f64);
    let axis = config.orientation.map(f64::from);
    let pair = placed(copper, repeats, a, shell, axis, vacancy);
    let energy = basin_energy(copper, &pair.positions, pair.cell);
    let basin = pair_basin(&pair.positions, pair.cell, repeats, a, pair.vacancy);
    (
        NetworkState {
            positions: pair.positions,
            dumbbell: pair.dumbbell,
            vacancy: pair.vacancy,
            cell: pair.cell,
            energy,
            basin,
        },
        pair.steps,
        pair.max_force,
    )
}

/// Dimer seed and mode for one move of the dumbbell in `state`, which sits
/// at `source` from the vacancy: the moving atoms halfway to the product of
/// the move, and the product minus the reactant as the mode.
fn hop_guess(
    state: &NetworkState,
    a: f64,
    source: rgsaddle::PairConfig,
    target: rgsaddle::PairConfig,
    rotation: bool,
    via: rgsaddle::Vec3i,
) -> (Array1<f64>, Array1<f64>) {
    let positions = &state.positions;
    let cell = state.cell;
    let lattice = |offset: rgsaddle::Vec3i| -> [f64; 3] {
        std::array::from_fn(|c| state.vacancy[c] + 0.5 * a * offset[c] as f64)
    };
    let site = lattice(source.offset);
    let mic = |point: [f64; 3]| -> [f64; 3] {
        std::array::from_fn(|c| {
            let length = cell[c][c];
            let delta = point[c] - site[c];
            delta - length * (delta / length).round()
        })
    };
    let e = source.orientation.map(f64::from);
    let along = |atom: usize| -> f64 {
        let delta = mic(point_of(positions, atom));
        delta[0] * e[0] + delta[1] * e[1] + delta[2] * e[2]
    };
    let half = 0.5
        * dist(
            point_of(positions, state.dumbbell[0]),
            point_of(positions, state.dumbbell[1]),
        );
    let mut product: Vec<(usize, [f64; 3])> = Vec::new();
    let mut hint: Vec<(usize, [f64; 3])> = Vec::new();
    if rotation {
        let e2 = target.orientation.map(f64::from);
        let v = via.map(f64::from);
        let sigma = (v[0] * e[0] + v[1] * e[1] + v[2] * e[2]).signum()
            * (v[0] * e2[0] + v[1] * e2[1] + v[2] * e2[2]).signum();
        let [first, second] = state.dumbbell;
        let sign = along(first).signum();
        let start: [f64; 3] = std::array::from_fn(|c| sign * e[c]);
        let end: [f64; 3] = std::array::from_fn(|c| sign * sigma * e2[c]);
        let middle = unit3(std::array::from_fn(|c| start[c] + end[c]));
        for (atom, s) in [(first, 1.0), (second, -1.0)] {
            product.push((atom, std::array::from_fn(|c| site[c] + s * half * end[c])));
            hint.push((
                atom,
                std::array::from_fn(|c| site[c] + s * half * middle[c]),
            ));
        }
    } else {
        let d: rgsaddle::Vec3i = std::array::from_fn(|c| target.offset[c] - source.offset[c]);
        let parallel = d[0] * source.orientation[0]
            + d[1] * source.orientation[1]
            + d[2] * source.orientation[2];
        let perpendicular: [f64; 3] =
            std::array::from_fn(|c| (d[c] - parallel * source.orientation[c]) as f64);
        let p = unit3(perpendicular);
        let (leading, staying) = if along(state.dumbbell[0]) * parallel as f64 > 0.0 {
            (state.dumbbell[0], state.dumbbell[1])
        } else {
            (state.dumbbell[1], state.dumbbell[0])
        };
        let landing = lattice(target.offset);
        product.push((staying, site));
        if target.offset == [0, 0, 0] {
            product.push((leading, landing));
        } else {
            let host =
                (0..positions.len() / 3)
                    .filter(|atom| *atom != leading && *atom != staying)
                    .min_by(|l, r| {
                        mic_distance(point_of(positions, *l), landing, cell)
                            .total_cmp(&mic_distance(point_of(positions, *r), landing, cell))
                    })
                    .expect("host atom");
            product.push((leading, std::array::from_fn(|c| landing[c] - half * p[c])));
            product.push((host, std::array::from_fn(|c| landing[c] + half * p[c])));
        }
    }
    let mut seed = positions.clone();
    let mut mode = Array1::zeros(positions.len());
    for (atom, place) in &product {
        let from = point_of(positions, *atom);
        for c in 0..3 {
            let length = cell[c][c];
            let mut delta = place[c] - from[c];
            delta -= length * (delta / length).round();
            mode[3 * atom + c] = delta;
            seed[3 * atom + c] = from[c] + 0.5 * delta;
        }
    }
    for (atom, place) in &hint {
        for c in 0..3 {
            seed[3 * atom + c] = place[c];
        }
    }
    (seed, mode)
}

/// One saddle of the near-shell campaign. The product of [`hop_guess`] is
/// followed down the flow, and a climbing band of five moving images joins
/// the two relaxed ends. The dimer starts from each local band maximum in
/// turn from the reactant side, along the band tangent, and last from the
/// [`hop_guess`] seed along its mode. Each saddle is pushed 0.1 Å (largest
/// atom) each way along the dimer mode and followed down the flow to
/// 1e-3 eV/Å; the two basins are labelled by [`pair_basin`]. The first
/// saddle whose basins are `want` is kept, else the first one found, else
/// none. The band's highest image and the image after it are kept either
/// way, with the relaxed product of the move, the saddle, its mode and the
/// two relaxed pushes.
struct NetworkSaddle {
    energy: f64,
    force: f64,
    index1: bool,
    product: String,
    band_force: f64,
    band_top: f64,
    basins: [String; 2],
    attempt: usize,
    positions: Option<Array1<f64>>,
    mode: Option<Array1<f64>>,
    ends: [Option<Array1<f64>>; 2],
    top: Array1<f64>,
    after_top: Array1<f64>,
    product_positions: Array1<f64>,
}

fn network_saddle(
    copper: &FastCopper,
    state: &NetworkState,
    repeats: usize,
    a: f64,
    hint: (Array1<f64>, Array1<f64>),
    want: &[String; 2],
) -> Result<NetworkSaddle, String> {
    let cell = state.cell;
    let product = descend(copper, &state.positions + &hint.1, cell, 1.0e-3)
        .map_err(|error| format!("product flow: {error}"))?;
    let product_basin = pair_basin(&product, cell, repeats, a, state.vacancy);
    let wrapped = rgsaddle::mic::Cell::from_ase(cell).map_err(|error| error.to_string())?;
    let guess = rgsaddle::linear_path(state.positions.view(), product.view(), 7, Some(&wrapped))
        .map_err(|error| error.to_string())?;
    let (band, band_force) =
        drive_phase(copper, cell, guess, true, Method::fire(FireKind::V2), 200)?;
    let held = FastHeld { copper, cell };
    let energies: Vec<f64> = band
        .rows()
        .into_iter()
        .map(|row| held.eval(row).map(|(energy, _)| energy))
        .collect::<Result<_, _>>()
        .map_err(|error| error.to_string())?;
    let top_index = (1..band.nrows() - 1)
        .max_by(|l, r| energies[*l].total_cmp(&energies[*r]))
        .expect("band images");
    let band_top = energies[top_index];
    let top = band.row(top_index).to_owned();
    let after_top = band.row(top_index + 1).to_owned();
    let mut seeds: Vec<(Array1<f64>, Array1<f64>)> = (1..band.nrows() - 1)
        .filter(|i| energies[*i] > energies[i - 1] && energies[*i] >= energies[i + 1])
        .map(|i| {
            (
                band.row(i).to_owned(),
                &band.row(i + 1).to_owned() - &band.row(i - 1).to_owned(),
            )
        })
        .collect();
    seeds.push(hint);
    let tries = seeds.len();
    let mut want = want.clone();
    want.sort();
    let mut first: Option<NetworkSaddle> = None;
    for (attempt, (seed, mode)) in seeds.into_iter().enumerate() {
        let Some(hit) = tight_dimer(copper, cell, seed, mode) else {
            continue;
        };
        let largest = (0..hit.mode.len() / 3)
            .map(|atom| {
                (0..3)
                    .map(|c| hit.mode[3 * atom + c].powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);
        if largest <= 0.0 {
            continue;
        }
        let push = &hit.mode * (0.1 / largest);
        let relaxed =
            [1.0, -1.0].map(|sign| descend(copper, &hit.positions + &(&push * sign), cell, 1.0e-3));
        let basins = std::array::from_fn(|side| match &relaxed[side] {
            Ok(basin) => pair_basin(basin, cell, repeats, a, state.vacancy),
            Err(error) => format!("unconverged:{error}"),
        });
        let ends = relaxed.map(Result::ok);
        let found = NetworkSaddle {
            energy: hit.energy,
            force: hit.force,
            index1: hit.index1,
            product: product_basin.clone(),
            band_force,
            band_top,
            basins,
            attempt,
            positions: Some(hit.positions),
            mode: Some(push),
            ends,
            top: top.clone(),
            after_top: after_top.clone(),
            product_positions: product.clone(),
        };
        let mut sorted = found.basins.clone();
        sorted.sort();
        if sorted == want {
            return Ok(found);
        }
        first.get_or_insert(found);
    }
    Ok(first.unwrap_or(NetworkSaddle {
        energy: f64::NAN,
        force: f64::NAN,
        index1: false,
        product: product_basin,
        band_force,
        band_top,
        basins: ["none".into(), "none".into()],
        attempt: tries,
        positions: None,
        mode: None,
        ends: [None, None],
        top,
        after_top,
        product_positions: product,
    }))
}

/// Six decimals, blank when not finite.
fn fixed6(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        String::new()
    }
}

const NETWORK_HEADER: &str = "temperature_K,lattice_A,kind,mechanism,hop,source,target,outcome,dE0_eV,energy_eV,max_force_eV_per_A,index1,forward_basin,backward_basin,product_basin,band_force_eV_per_A,band_top_eV,steps_or_seed,seconds";

/// Rows of `defect_pair_network.csv` already written, by temperature, kind
/// and hop or source.
fn network_done(text: &str) -> std::collections::HashSet<(i64, String, String)> {
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let fields: Vec<&str> = line.split(',').collect();
            let temperature = fields.first()?.parse::<f64>().ok()?.round() as i64;
            let kind = fields.get(2)?.to_string();
            let key = if kind == "state" {
                fields.get(5)?
            } else {
                fields.get(4)?
            };
            Some((temperature, kind, key.to_string()))
        })
        .collect()
}

/// The isolated dumbbell in the same cell, for the reference barriers: the
/// dumbbell at (2,0,0) a/2 along x, its translation to (3,1,0) along y and
/// its rotation to y through the (1,1,0) saddle.
fn isolated_network_moves() -> [(&'static str, rgsaddle::PairConfig, rgsaddle::Vec3i); 2] {
    [
        (
            "translate",
            rgsaddle::PairConfig::new([3, 1, 0], [0, 1, 0]),
            [0, 0, 0],
        ),
        (
            "rotate",
            rgsaddle::PairConfig::new([2, 0, 0], [0, 1, 0]),
            [1, 1, 0],
        ),
    ]
}

#[test]
#[ignore = "near-shell saddles of every symmetry-distinct hop in the 8788-atom cell on the Table 1 lattices"]
fn defect_pair_network() {
    let started = Instant::now();
    let repeats = 13usize;
    let temperatures: Vec<f64> = env_csv("RGSADDLE_TEMPS", "300,500,700")
        .iter()
        .map(|item| item.parse().expect("temperature"))
        .collect();
    let limit: usize = env_csv("RGSADDLE_HOP_LIMIT", "100000")[0]
        .parse()
        .expect("hop limit");
    let copper = FastCopper::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_network.csv");
    if !path.exists() {
        std::fs::write(&path, format!("{NETWORK_HEADER}\n")).expect("network");
    }
    let done = network_done(&std::fs::read_to_string(&path).expect("network"));
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let (group, network) = dumbbell_network(EXPLICIT_NORM2);
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let write = |line: String| {
        eprintln!(
            "network {line} at {:.1} min",
            started.elapsed().as_secs_f64() / 60.0
        );
        append_checkpoint(&checkpoint, &line);
    };
    let isolated_source = rgsaddle::PairConfig::new([2, 0, 0], [1, 0, 0]);
    let mut placements = Vec::new();
    for &temperature in &temperatures {
        let spontaneous = census_spontaneous(&census, temperature);
        placements.push((temperature, None));
        for (index, state) in network.states.iter().enumerate() {
            if !spontaneous.contains(&state_miller(state.config, &group)) {
                placements.push((temperature, Some(index)));
            }
        }
    }
    eprintln!(
        "network: {} states, {} hop classes; {} placements on {} threads",
        network.states.len(),
        network.hops.len(),
        placements.len(),
        rayon::current_num_threads()
    );
    let relaxed: Vec<((f64, Option<usize>), NetworkState)> = placements
        .par_iter()
        .map(|&(temperature, index)| {
            let clock = Instant::now();
            let a = demkowicz_lattice(temperature);
            let config = index.map_or(isolated_source, |i| network.states[i].config);
            let (state, steps, force) = network_state(&copper, repeats, a, config, index.is_some());
            let source = index.map_or("isolated".to_string(), |i| network.id(i));
            if !done.contains(&(temperature as i64, "state".into(), source.clone())) {
                write(format!(
                    "{temperature},{a:.3},state,,,{source},{},{},,{:.6},{force:.3e},,{},,,,,{steps},{:.1}",
                    config_label(config),
                    if state.basin == config_label(config) {
                        "bound"
                    } else {
                        "left"
                    },
                    state.energy,
                    state.basin,
                    clock.elapsed().as_secs_f64()
                ));
            }
            ((temperature, index), state)
        })
        .collect();
    let lookup = |temperature: f64, index: Option<usize>| -> &NetworkState {
        &relaxed
            .iter()
            .find(|(key, _)| key.0 == temperature && key.1 == index)
            .expect("relaxed state")
            .1
    };
    struct HopJob {
        temperature: f64,
        source: Option<usize>,
        hop: Option<usize>,
        kind: String,
        from: rgsaddle::PairConfig,
        to: rgsaddle::PairConfig,
        via: rgsaddle::Vec3i,
        expect: String,
        target: String,
    }
    let mut jobs: Vec<HopJob> = Vec::new();
    for &temperature in &temperatures {
        for (kind, to, via) in isolated_network_moves() {
            if !done.contains(&(temperature as i64, "isolated".into(), kind.into())) {
                jobs.push(HopJob {
                    temperature,
                    source: None,
                    hop: None,
                    kind: kind.into(),
                    from: isolated_source,
                    to,
                    via,
                    expect: config_label(to),
                    target: config_label(to),
                });
            }
        }
        let spontaneous = census_spontaneous(&census, temperature);
        let is_spontaneous =
            |i: usize| spontaneous.contains(&state_miller(network.states[i].config, &group));
        let mut claimed = std::collections::HashSet::new();
        for (index, state) in network.states.iter().enumerate() {
            if is_spontaneous(index) {
                continue;
            }
            for edge in &state.edges {
                let (expect, target) = match edge.target {
                    rgsaddle::PairTarget::Outside => continue,
                    rgsaddle::PairTarget::Contact => ("recombined".to_string(), "contact".into()),
                    rgsaddle::PairTarget::State(j) if is_spontaneous(j) => {
                        ("recombined".to_string(), network.id(j))
                    }
                    rgsaddle::PairTarget::State(j) => (config_label(edge.config), network.id(j)),
                };
                if !claimed.insert(edge.hop) {
                    continue;
                }
                let hop = &network.hops[edge.hop];
                if done.contains(&(temperature as i64, "hop".into(), hop_label(hop))) {
                    continue;
                }
                jobs.push(HopJob {
                    temperature,
                    source: Some(index),
                    hop: Some(edge.hop),
                    kind: hop.kind.clone(),
                    from: state.config,
                    to: edge.config,
                    via: edge.via,
                    expect,
                    target,
                });
            }
        }
    }
    jobs.truncate(limit);
    eprintln!("network: {} saddle searches", jobs.len());
    jobs.par_iter().for_each(|job| {
        let clock = Instant::now();
        let a = demkowicz_lattice(job.temperature);
        let state = lookup(job.temperature, job.source);
        let rotation = job.kind == "rotate";
        let source_label = config_label(job.from);
        let found = network_saddle(
            &copper,
            state,
            repeats,
            a,
            hop_guess(state, a, job.from, job.to, rotation, job.via),
            &[source_label.clone(), job.expect.clone()],
        );
        let (outcome, columns) = match &found {
            Err(error) => ("no_saddle", format!(",,,,,,{},,,", error.replace(',', ";"))),
            Ok(saddle) => {
                let mut ends = saddle.basins.clone();
                ends.sort();
                let mut want = [source_label.clone(), job.expect.clone()];
                want.sort();
                let outcome = if ends == want && state.basin == source_label {
                    "connected"
                } else {
                    "misconnected"
                };
                (
                    outcome,
                    format!(
                        "{},{},{},{},{},{},{},{:.3e},{:.6},{}",
                        fixed6(saddle.energy - state.energy),
                        fixed6(saddle.energy),
                        if saddle.force.is_finite() {
                            format!("{:.3e}", saddle.force)
                        } else {
                            String::new()
                        },
                        saddle.index1,
                        saddle.basins[0],
                        saddle.basins[1],
                        saddle.product,
                        saddle.band_force,
                        saddle.band_top,
                        saddle.attempt
                    ),
                )
            }
        };
        let (kind, hop, source) = match job.hop {
            Some(h) => (
                "hop",
                hop_label(&network.hops[h]),
                network.id(job.source.expect("source")),
            ),
            None => ("isolated", job.kind.clone(), "isolated".to_string()),
        };
        write(format!(
            "{},{a:.3},{kind},{},{hop},{source},{},{outcome},{columns},{:.1}",
            job.temperature,
            job.kind,
            job.target,
            clock.elapsed().as_secs_f64()
        ));
    });
    eprintln!(
        "network wall {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

#[test]
fn dumbbell_network_matches_census_orbits() {
    let group = rgsaddle::cubic_group();
    let (_, network) = dumbbell_network(EXPLICIT_NORM2);
    let a = 3.631;
    let mut orbits: Vec<(String, usize)> = network
        .states
        .iter()
        .map(|state| (state_miller(state.config, &group), state.multiplicity / 3))
        .collect();
    orbits.sort();
    let mut census: Vec<(String, usize)> = axis_orbits(a, 13)
        .into_iter()
        .map(|(_, vector, sites)| {
            let (i, j, k) = miller(vector, a);
            (format!("{i} {j} {k}"), sites)
        })
        .collect();
    census.sort();
    assert_eq!(orbits, census);
}

/// Saddle and state energies of one temperature of
/// `defect_pair_network.csv`. A hop's saddle is the verified dimer saddle,
/// else the highest image of a band of at most 0.02 eV/Å whose relaxed
/// product is the hop's target; `band` names the hops that used the band.
struct NetworkInputs {
    saddles: std::collections::HashMap<String, f64>,
    band: std::collections::HashSet<String>,
    states: std::collections::HashMap<String, f64>,
    isolated: [f64; 2],
    rows: usize,
}

/// The configuration of a [`config_label`], `None` for any other basin.
fn parse_config_label(label: &str) -> Option<rgsaddle::PairConfig> {
    let (offset, orientation) = label.split_once(':')?;
    let offset: Vec<i32> = offset
        .split('_')
        .map(|c| c.parse().ok())
        .collect::<Option<_>>()?;
    let orientation: Vec<i32> = orientation
        .chars()
        .map(|c| c.to_digit(10).map(|d| d as i32))
        .collect::<Option<_>>()?;
    (offset.len() == 3 && orientation.len() == 3).then(|| {
        rgsaddle::PairConfig::new(
            [offset[0], offset[1], offset[2]],
            [orientation[0], orientation[1], orientation[2]],
        )
    })
}

fn network_inputs(
    text: &str,
    temperature: f64,
    group: &[rgsaddle::Mat3i],
    spontaneous: &std::collections::HashSet<String>,
) -> NetworkInputs {
    let mut inputs = NetworkInputs {
        saddles: Default::default(),
        band: Default::default(),
        states: Default::default(),
        isolated: [f64::NAN; 2],
        rows: 0,
    };
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 19 || fields[0].parse::<f64>().ok() != Some(temperature) {
            continue;
        }
        inputs.rows += 1;
        let number = |index: usize| fields[index].parse::<f64>().ok();
        match (fields[2], fields[7]) {
            ("state", "bound") => {
                inputs
                    .states
                    .insert(fields[5].to_string(), number(9).expect("state energy"));
            }
            ("isolated", "connected") => {
                let which = usize::from(fields[3] == "rotate");
                inputs.isolated[which] = number(8).expect("isolated barrier");
            }
            ("hop", "connected") => {
                inputs
                    .saddles
                    .insert(fields[4].to_string(), number(9).expect("saddle energy"));
            }
            ("hop", _) => {
                let target_spontaneous = fields[6] == "contact" || spontaneous.contains(fields[6]);
                let product = fields[14];
                let lands = if target_spontaneous {
                    product == "recombined"
                } else {
                    parse_config_label(product)
                        .is_some_and(|config| config_label(config.canonical(group)) == fields[6])
                };
                if let (true, Some(force), Some(top)) = (lands, number(15), number(16))
                    && force <= 0.02
                {
                    inputs.saddles.insert(fields[4].to_string(), top);
                    inputs.band.insert(fields[4].to_string());
                }
            }
            _ => {}
        }
    }
    inputs
}

/// Spread of the barrier-shift model against the recorded near-shell PAFI
/// rates. Each rated near-shell translation or capture row of a file is set
/// against that file's isolated translation at the same temperature; the
/// model predicts `ln(k/k_iso) = -(E0 - E0_iso)/kT`, PAFI gives
/// `ln(nu/nu_iso) - (F - F_iso)/kT`. Returns the root mean square of the
/// difference and the number of rows.
fn barrier_shift_spread() -> (f64, usize) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut residuals = Vec::new();
    for name in [
        "defect_pair_capture.csv",
        "defect_pair_finish.csv",
        "defect_pair_local.csv",
        "defect_pair_run11.csv",
        "defect_pair_run9.csv",
        "defect_pair_viv.csv",
    ] {
        let text = std::fs::read_to_string(root.join(name)).unwrap_or_default();
        let rows: Vec<Vec<&str>> = text
            .lines()
            .map(|line| line.split(',').collect::<Vec<_>>())
            .filter(|f| f.len() >= 13 && f[0] == "pafi" && f[12] == "rated")
            .collect();
        let values = |f: &[&str]| -> Option<(f64, f64, f64, f64)> {
            Some((
                f[5].parse().ok()?,
                f[7].parse().ok()?,
                f[8].parse().ok()?,
                f[10].parse().ok()?,
            ))
        };
        for row in &rows {
            if row[1] == "isolated" || !(row[3].starts_with("translate") || row[3] == "capture") {
                continue;
            }
            let Some((temperature, zero, free, nu)) = values(row) else {
                continue;
            };
            let Some((_, zero_iso, free_iso, nu_iso)) = rows
                .iter()
                .filter(|f| f[1] == "isolated" && f[3] == "translate")
                .filter_map(|f| values(f))
                .find(|v| v.0 == temperature)
            else {
                continue;
            };
            let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
            let pafi = (nu / nu_iso).ln() - (free - free_iso) / thermal;
            let model = -(zero - zero_iso) / thermal;
            residuals.push(pafi - model);
        }
    }
    let n = residuals.len();
    let rms = (residuals.iter().map(|r| r * r).sum::<f64>() / n.max(1) as f64).sqrt();
    (rms, n)
}

/// Rates of the per-vector network. `capture` multiplies every rate taken
/// from a near-shell saddle; `mig` and `rot` scale the isolated channels.
#[derive(Clone, Copy)]
struct NetworkScale {
    mig: f64,
    rot: f64,
    capture: f64,
}

/// Per-class rates of the per-vector network inside the census radius.
/// `computed` holds `ln(k/k_kind)` of one direction of a hop class, by hop
/// label and source state id, and replaces the barrier-shift model there;
/// `shift` adds to `ln k` of both directions of a hop class, whatever its
/// rate source.
#[derive(Clone, Default)]
struct ClassRates {
    computed: std::collections::HashMap<(String, String), f64>,
    shift: std::collections::HashMap<String, f64>,
}

/// One per-vector volume and what built it.
struct NetworkVolumes {
    total: f64,
    spont_sites: f64,
    chain_radius: f64,
    max_r: f64,
    states: usize,
    explicit_edges: usize,
    band_edges: usize,
    elastic_edges: usize,
    fallback_edges: usize,
    computed_edges: usize,
    sphere: SphereVolumes,
}

/// The per-vector absorbing chain at one cutoff. Inside the census radius
/// every state is spontaneous or bound from the census, and with `explicit`
/// each hop out of a bound state there runs at
/// `k_kind exp(-(E_saddle - E_state - E0_kind)/kT)`, `E0_kind` the isolated
/// barrier in the same cell, unless `classes` holds a computed rate for
/// that direction, which then runs at `k_kind exp(computed)` and takes no
/// `capture` factor. Any other hop runs at `k_kind` times half the
/// change of the vector- and orientation-resolved dipole interaction,
/// `exp(-(E_b(to) - E_b(from))/2kT)`. With `matched` the chain is matched
/// to the field-free crystal outside the cutoff by
/// [`rgsaddle::matched_capture_radius`] over the sites within one
/// nearest-neighbour distance of it, and that radius is the chain radius.
/// Without it, hops past the cutoff enter the return layer of
/// [`rgsaddle::embed_sphere_return`] and come back to their source, and
/// the chain radius is the site-weighted median of `P r` over the sites
/// past 8 Å and 0.55 of the cutoff, as the lumped model takes it.
#[allow(clippy::too_many_arguments)]
fn network_volumes(
    network: &rgsaddle::PairNetwork,
    group: &[rgsaddle::Mat3i],
    spontaneous: &std::collections::HashSet<String>,
    explicit: Option<&NetworkInputs>,
    field: &PairField,
    temperature: f64,
    a: f64,
    k_mig: f64,
    k_rot: f64,
    scale: NetworkScale,
    classes: &ClassRates,
    matched: bool,
) -> NetworkVolumes {
    let omega = a * a * a / 4.0;
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let half = 0.5 * a;
    let radius_of = |config: rgsaddle::PairConfig| {
        half * (rgsaddle::pair_network::norm2(config.offset) as f64).sqrt()
    };
    let binding = |config: rgsaddle::PairConfig| -> f64 {
        let radius = radius_of(config);
        if radius <= 0.0 {
            return 0.0;
        }
        let direction = config.offset.map(|c| half * c as f64 / radius);
        let which = config.orientation.iter().position(|c| *c != 0).unwrap_or(0);
        field.interaction(radius, direction, which)
    };
    let is_spontaneous = |i: usize| {
        let config = network.states[i].config;
        rgsaddle::pair_network::norm2(config.offset) <= EXPLICIT_NORM2
            && spontaneous.contains(&state_miller(config, group))
    };
    let counts = std::cell::Cell::new([0usize; 5]);
    let tally = |slot: usize| {
        let mut value = counts.get();
        value[slot] += 1;
        counts.set(value);
    };
    let rate = |i: usize, edge: &rgsaddle::PairEdge| -> f64 {
        let hop = &network.hops[edge.hop];
        let rotation = hop.kind == "rotate";
        let base = if rotation {
            k_rot * scale.rot
        } else {
            k_mig * scale.mig
        };
        let source = network.states[i].config;
        let inside = rgsaddle::pair_network::norm2(source.offset) <= EXPLICIT_NORM2
            && match edge.target {
                rgsaddle::PairTarget::State(j) => {
                    rgsaddle::pair_network::norm2(network.states[j].config.offset) <= EXPLICIT_NORM2
                }
                rgsaddle::PairTarget::Contact => true,
                rgsaddle::PairTarget::Outside => false,
            };
        let label = hop_label(hop);
        let class = if inside {
            classes.shift.get(&label).map_or(1.0, |shift| shift.exp())
        } else {
            1.0
        };
        if inside && let Some(inputs) = explicit {
            if let Some(computed) = classes.computed.get(&(label.clone(), network.id(i))) {
                tally(4);
                return base * class * computed.exp();
            }
            let reference = inputs.isolated[usize::from(rotation)];
            if let (Some(saddle), Some(state), true) = (
                inputs.saddles.get(&label),
                inputs.states.get(&network.id(i)),
                reference.is_finite(),
            ) {
                tally(usize::from(inputs.band.contains(&label)));
                let barrier = (saddle - state).max(0.0);
                return base * class * scale.capture * (-(barrier - reference) / thermal).exp();
            }
            tally(3);
        } else {
            tally(2);
        }
        let shift = match edge.target {
            rgsaddle::PairTarget::Contact => 0.0,
            _ => 0.5 * (binding(edge.config) - binding(source)),
        };
        base * class * (-shift / thermal).exp()
    };
    let (sites, exits) = network
        .sites(&is_spontaneous, &rate)
        .expect("network sites");
    let max_r = network
        .states
        .iter()
        .map(|state| radius_of(state.config))
        .fold(0.0, f64::max);
    let radius: std::collections::HashMap<String, f64> = (0..network.states.len())
        .map(|i| (network.id(i), radius_of(network.states[i].config)))
        .collect();
    let (solved, chain_radius) = if matched {
        let outer: Vec<(String, f64)> = radius
            .iter()
            .filter(|(_, r)| **r >= max_r - a / 2.0_f64.sqrt())
            .map(|(id, r)| (id.clone(), *r))
            .collect();
        let hops: Vec<rgsaddle::OuterHop> = exits
            .iter()
            .map(|exit| rgsaddle::OuterHop {
                source: exit.source.clone(),
                rate: exit.rate,
                distance: half * (exit.norm2 as f64).sqrt(),
            })
            .collect();
        let found =
            rgsaddle::matched_capture_radius(&sites, &hops, &outer, omega).expect("matched chain");
        (found.volume, found.radius)
    } else {
        let landings: Vec<rgsaddle::SphereLanding> = exits
            .iter()
            .map(|exit| rgsaddle::SphereLanding {
                source: exit.source.clone(),
                boundary: exit.source.clone(),
                rate: exit.rate,
                distance: half * (exit.norm2 as f64).sqrt(),
            })
            .collect();
        let embedded =
            rgsaddle::embed_sphere_return(&sites, max_r, &landings).expect("return layer");
        let solved = rgsaddle::kinetic_recombination_volume(&embedded, omega).expect("volume");
        let mut band: Vec<(f64, f64)> = solved
            .placements
            .iter()
            .filter_map(|place| {
                let r = *radius.get(&place.id)?;
                (r >= 8.0 && r >= 0.55 * max_r).then_some((place.probability * r, place.count))
            })
            .collect();
        band.sort_by(|l, r| l.0.total_cmp(&r.0));
        let weight: f64 = band.iter().map(|item| item.1).sum();
        let mut running = 0.0;
        let chain_radius = band
            .iter()
            .find(|item| {
                running += item.1;
                running >= 0.5 * weight
            })
            .map_or(f64::NAN, |item| item.0);
        (solved, chain_radius)
    };
    let [
        explicit_edges,
        band_edges,
        elastic_edges,
        fallback_edges,
        computed_edges,
    ] = counts.get();
    NetworkVolumes {
        total: solved.total_sites,
        spont_sites: solved.spontaneous_sites,
        chain_radius,
        max_r,
        states: network.states.len(),
        explicit_edges,
        band_edges,
        elastic_edges,
        fallback_edges,
        computed_edges,
        sphere: sphere_volumes(
            chain_radius,
            max_r,
            a,
            temperature,
            k_mig * scale.mig,
            k_rot * scale.rot,
            Some(field),
        ),
    }
}

/// Standard deviation of each of the three volumes over `samples`.
fn draw_spread(samples: &[[f64; 3]]) -> [f64; 3] {
    let n = samples.len();
    if n < 2 {
        return [f64::NAN; 3];
    }
    std::array::from_fn(|which| {
        let mean = samples.iter().map(|s| s[which]).sum::<f64>() / n as f64;
        (samples
            .iter()
            .map(|s| (s[which] - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64)
            .sqrt()
    })
}

fn blank_or(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.4}")
    } else {
        String::new()
    }
}

/// The rates and inputs of the per-vector network at one temperature: the
/// flow census, the converged isolated rates and the near-shell saddles of
/// `defect_pair_network.csv` on the Table 1 lattice.
struct NetworkContext {
    temperature: f64,
    a: f64,
    spontaneous: std::collections::HashSet<String>,
    spontaneous_ids: std::collections::HashSet<String>,
    inputs: NetworkInputs,
    k_mig: f64,
    mig_spread: f64,
    mig_note: &'static str,
    k_rot: f64,
    rot_spread: f64,
    field: PairField,
}

fn network_context(
    temperature: f64,
    census: &str,
    saddles: &str,
    rates: &str,
    group: &[rgsaddle::Mat3i],
    explicit_network: &rgsaddle::PairNetwork,
) -> NetworkContext {
    let a = demkowicz_lattice(temperature);
    let spontaneous = census_spontaneous(census, temperature);
    let spontaneous_ids: std::collections::HashSet<String> = (0..explicit_network.states.len())
        .filter(|i| spontaneous.contains(&state_miller(explicit_network.states[*i].config, group)))
        .map(|i| explicit_network.id(i))
        .collect();
    let inputs = network_inputs(saddles, temperature, group, &spontaneous_ids);
    let (mut k_mig, mut mig_spread, mut mig_note) = (
        migration_rate(rates, temperature).unwrap_or(f64::NAN),
        isolated_spread(rates, "translate", temperature).unwrap_or(f64::NAN),
        "PAFI translation",
    );
    if pafi_kappa_unresolved(rates, temperature)
        && let Some(measured) = md_channel_rate(temperature)
    {
        k_mig = measured;
        mig_spread = md_channel_jumps(temperature).map_or(f64::NAN, |n| 1.0 / n.sqrt());
        mig_note = "direct-MD channel rate (PAFI kappa unresolved)";
    }
    let rot_spread = isolated_spread(rates, "rotate", temperature).unwrap_or(0.0);
    let (k_rot, _) = rotation_channel(rates, temperature);
    let copper = FastCopper::new();
    let field = pair_field(&copper, a).expect("pair field relaxation");
    NetworkContext {
        temperature,
        a,
        spontaneous,
        spontaneous_ids,
        inputs,
        k_mig,
        mig_spread,
        mig_note,
        k_rot,
        rot_spread,
        field,
    }
}

impl NetworkContext {
    fn volumes(
        &self,
        network: &rgsaddle::PairNetwork,
        group: &[rgsaddle::Mat3i],
        explicit: bool,
        scale: NetworkScale,
        classes: &ClassRates,
        matched: bool,
    ) -> NetworkVolumes {
        network_volumes(
            network,
            group,
            &self.spontaneous,
            explicit.then_some(&self.inputs),
            &self.field,
            self.temperature,
            self.a,
            self.k_mig,
            self.k_rot,
            scale,
            classes,
            matched,
        )
    }

    /// The per-vector network out to `cutoff` Å.
    fn network_to(&self, cutoff: f64) -> rgsaddle::PairNetwork {
        let max_norm2 = ((cutoff / (0.5 * self.a)).powi(2) + 1.0e-6).floor() as i64;
        dumbbell_network(max_norm2).1
    }
}

const UNIT_SCALE: NetworkScale = NetworkScale {
    mig: 1.0,
    rot: 1.0,
    capture: 1.0,
};

/// One hop class inside the census radius with a direction out of a bound
/// state, and where its model rate comes from: `saddle`, `band` or
/// `fallback` to the dipole rate.
struct ExplicitClass {
    label: String,
    kind: String,
    source: &'static str,
    /// Bound source states of its directions inside the census radius.
    sources: Vec<String>,
}

fn explicit_classes(
    context: &NetworkContext,
    network: &rgsaddle::PairNetwork,
) -> Vec<ExplicitClass> {
    let mut classes: Vec<ExplicitClass> = Vec::new();
    for (index, state) in network.states.iter().enumerate() {
        let id = network.id(index);
        if context.spontaneous_ids.contains(&id) {
            continue;
        }
        for edge in &state.edges {
            if edge.target == rgsaddle::PairTarget::Outside {
                continue;
            }
            let hop = &network.hops[edge.hop];
            let label = hop_label(hop);
            if let Some(class) = classes.iter_mut().find(|class| class.label == label) {
                if !class.sources.contains(&id) {
                    class.sources.push(id.clone());
                }
                continue;
            }
            let source = if !context.inputs.saddles.contains_key(&label) {
                "fallback"
            } else if context.inputs.band.contains(&label) {
                "band"
            } else {
                "saddle"
            };
            classes.push(ExplicitClass {
                label,
                kind: hop.kind.clone(),
                source,
                sources: vec![id.clone()],
            });
        }
    }
    classes
}

/// `d V / d ln k` of each class for the smoluchowski, cell-decay and
/// mean-time volumes, by central differences of 0.1 in `ln k` on both
/// directions of the class.
fn class_sensitivities(
    context: &NetworkContext,
    network: &rgsaddle::PairNetwork,
    group: &[rgsaddle::Mat3i],
    classes: &[ExplicitClass],
    base: &ClassRates,
) -> Vec<[f64; 3]> {
    let step = 0.1;
    classes
        .par_iter()
        .map(|class| {
            let volume = |shift: f64| {
                let mut rates = base.clone();
                *rates.shift.entry(class.label.clone()).or_insert(0.0) += shift;
                let sphere = context
                    .volumes(network, group, true, UNIT_SCALE, &rates, true)
                    .sphere;
                [sphere.sphere, sphere.decay_volume, sphere.mean_volume]
            };
            let (up, down) = (volume(step), volume(-step));
            std::array::from_fn(|which| (up[which] - down[which]) / (2.0 * step))
        })
        .collect()
}

/// PAFI rates of one temperature of `defect_pair_network_pafi.csv`. Each
/// rated direction of a hop class gets
/// `ln(k/k_kind) = ln(nu/nu_iso) - (F - F_iso)/kT`, against the isolated
/// hop of the same kind in the same cell. The isolated hop ends in a state
/// equivalent to its start, so its two directions measure one barrier:
/// `F_iso` is their inverse-variance mean and `nu_iso` the mean of the two
/// end frequencies, and half the gap between the two directions is the
/// drift of one integrated profile along its path. That drift is added in
/// quadrature to the error of every direction of the same kind, in `sigma`,
/// and to the isolated error, which is kept apart by kind because every
/// class of a kind shares it. `residuals` sets each rated direction against
/// the barrier-shift model of [`network_volumes`].
struct PafiInputs {
    computed: std::collections::BTreeMap<(String, String), (f64, f64, bool)>,
    isolated_sigma: [f64; 2],
    drift: [f64; 2],
    residuals: Vec<f64>,
    rated: usize,
    seconds: f64,
}

/// Largest band force a PAFI path may keep and still be rated, the gate of
/// the recorded PAFI rows.
const PAFI_BAND_FORCE: f64 = 0.01;

fn pafi_inputs(
    text: &str,
    temperature: f64,
    spontaneous_ids: &std::collections::HashSet<String>,
    saddles: &NetworkInputs,
    network: &rgsaddle::PairNetwork,
) -> PafiInputs {
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let rows: Vec<Vec<&str>> = text
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect::<Vec<_>>())
        .filter(|f| f.len() >= 21 && f[0].parse::<f64>().ok() == Some(temperature))
        .collect();
    let number = |f: &[&str], index: usize| f[index].parse::<f64>().unwrap_or(f64::NAN);
    let usable = |f: &[&str]| f[7] == "rated";
    let mut isolated = [(f64::NAN, f64::NAN, f64::NAN); 2];
    let mut drift = [f64::NAN; 2];
    for f in rows.iter().filter(|f| f[2] == "isolated" && usable(f)) {
        let kind = usize::from(f[3] == "rotate");
        let (forward, forward_error) = (number(f, 12), number(f, 13));
        let (back, back_error) = (number(f, 15), number(f, 16));
        let (weight_f, weight_b) = (forward_error.powi(-2), back_error.powi(-2));
        let free = (weight_f * forward + weight_b * back) / (weight_f + weight_b);
        drift[kind] = 0.5 * (forward - back).abs();
        let error = ((weight_f + weight_b).recip() + drift[kind].powi(2)).sqrt();
        isolated[kind] = (free, error, 0.5 * (number(f, 14) + number(f, 17)));
    }
    let bound: std::collections::HashSet<String> = (0..network.states.len())
        .map(|i| network.id(i))
        .filter(|id| !spontaneous_ids.contains(id))
        .collect();
    let mut inputs = PafiInputs {
        computed: Default::default(),
        isolated_sigma: isolated.map(|reference| reference.1),
        drift,
        residuals: Vec::new(),
        rated: 0,
        seconds: rows
            .iter()
            .map(|f| number(f, 20))
            .filter(|v| v.is_finite())
            .sum(),
    };
    for f in rows.iter().filter(|f| f[2] == "hop" && usable(f)) {
        let rotation = f[3] == "rotate";
        inputs.rated += 1;
        let label = f[4].to_string();
        let mut directions = vec![(f[5].to_string(), 12)];
        if bound.contains(f[6]) && f[6] != f[5] {
            directions.push((f[6].to_string(), 15));
        }
        let (free_iso, _, nu_iso) = isolated[usize::from(rotation)];
        let path = drift[usize::from(rotation)];
        for (source, column) in directions {
            let (free, sigma, nu) = (
                number(f, column),
                number(f, column + 1),
                number(f, column + 2),
            );
            if ![free, sigma, nu, free_iso, nu_iso, path]
                .iter()
                .all(|v| v.is_finite())
            {
                continue;
            }
            let computed = (nu / nu_iso).ln() - (free.max(0.0) - free_iso) / thermal;
            let reference = saddles.isolated[usize::from(rotation)];
            if let (Some(saddle), Some(state)) =
                (saddles.saddles.get(&label), saddles.states.get(&source))
            {
                let model = -((saddle - state).max(0.0) - reference) / thermal;
                inputs.residuals.push(computed - model);
            }
            inputs.computed.insert(
                (label.clone(), source),
                (computed, sigma.hypot(path) / thermal, rotation),
            );
        }
    }
    inputs
}

impl PafiInputs {
    /// Root mean square of the residuals, `None` below two.
    fn model_sigma(&self) -> Option<f64> {
        (self.residuals.len() >= 2).then(|| {
            (self.residuals.iter().map(|r| r * r).sum::<f64>() / self.residuals.len() as f64).sqrt()
        })
    }

    fn central(&self) -> ClassRates {
        ClassRates {
            computed: self
                .computed
                .iter()
                .map(|(key, value)| (key.clone(), value.0))
                .collect(),
            shift: Default::default(),
        }
    }

    /// The computed rates moved by `z`: one normal per rated direction in
    /// key order, then one per isolated kind.
    fn drawn(&self, z: &[f64], thermal: f64) -> ClassRates {
        let n = self.computed.len();
        ClassRates {
            computed: self
                .computed
                .iter()
                .enumerate()
                .map(|(index, (key, (value, sigma, rotation)))| {
                    let kind = usize::from(*rotation);
                    let common = z[n + kind] * self.isolated_sigma[kind] / thermal;
                    (key.clone(), value + z[index] * sigma + common)
                })
                .collect(),
            shift: Default::default(),
        }
    }

    /// Whether every direction of `class` out of a bound state is computed.
    fn covers(&self, class: &ExplicitClass) -> bool {
        class.sources.iter().all(|source| {
            self.computed
                .contains_key(&(class.label.clone(), source.clone()))
        })
    }
}

const SENSITIVITY_HEADER: &str = "temperature_K,stage,hop,mechanism,rate_source,dV_smoluchowski_dlnk,dV_cell_decay_dlnk,dV_mean_time_dlnk,rank,remaining_Omega,selected,sigma_lnk,tolerance_Omega";

/// Tolerance of the coverage: the smaller Table 1 error at 8788 atoms.
fn table_i_tolerance(temperature: f64) -> f64 {
    table_i_8788(temperature).map_or(f64::NAN, |[(_, alpha), (_, beta)]| alpha.min(beta))
}

/// `defect_pair_network_sensitivity.csv`: every near-shell hop class ranked
/// by `|dV_cell/d ln k|` of the 30 Å cell-decay volume. Stage `model` uses
/// the barrier-shift rates and the spread of [`barrier_shift_spread`];
/// classes are covered by [`rgsaddle::sensitivity_cover`] until the rest,
/// all moved by that spread together, shift the volume by less than the
/// smaller Table 1 error. Stage `pafi` repeats it with the rated classes of
/// `defect_pair_network_pafi.csv` in place and the spread of the model
/// against them, and covers only the classes still on the model.
fn write_sensitivities() {
    let started = Instant::now();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let saddles = std::fs::read_to_string(root.join("defect_pair_network.csv")).expect("network");
    let pafi_text =
        std::fs::read_to_string(root.join("defect_pair_network_pafi.csv")).unwrap_or_default();
    let rates = converged_rates();
    let (model_sigma, _) = barrier_shift_spread();
    let group = rgsaddle::cubic_group();
    let (_, explicit_network) = dumbbell_network(EXPLICIT_NORM2);
    let mut out = format!("{SENSITIVITY_HEADER}\n");
    for temperature in [300.0_f64, 500.0, 700.0] {
        let context = network_context(
            temperature,
            &census,
            &saddles,
            &rates,
            &group,
            &explicit_network,
        );
        let network = context.network_to(30.0);
        let classes = explicit_classes(&context, &explicit_network);
        let tolerance = table_i_tolerance(temperature);
        let pafi = pafi_inputs(
            &pafi_text,
            temperature,
            &context.spontaneous_ids,
            &context.inputs,
            &explicit_network,
        );
        let mut stages = vec![("model", ClassRates::default(), model_sigma, None)];
        if pafi.rated > 0 {
            stages.push((
                "pafi",
                pafi.central(),
                pafi.model_sigma().unwrap_or(model_sigma),
                Some(&pafi),
            ));
        }
        for (stage, base, sigma, computed) in &stages {
            let sensitivity = class_sensitivities(&context, &network, &group, &classes, base);
            let on_model: Vec<usize> = (0..classes.len())
                .filter(|i| computed.is_none_or(|pafi| !pafi.covers(&classes[*i])))
                .collect();
            let cover = rgsaddle::sensitivity_cover(
                &on_model
                    .iter()
                    .map(|i| sensitivity[*i][1])
                    .collect::<Vec<_>>(),
                *sigma,
                tolerance,
            );
            let mut rank_of = vec![None; classes.len()];
            for (rank, position) in cover.order.iter().enumerate() {
                rank_of[on_model[*position]] = Some(rank);
            }
            let mut listed: Vec<usize> = (0..classes.len()).collect();
            listed.sort_by(|l, r| {
                sensitivity[*r][1]
                    .abs()
                    .total_cmp(&sensitivity[*l][1].abs())
                    .then(l.cmp(r))
            });
            for index in listed {
                let class = &classes[index];
                let (rank, remaining, selected) = match rank_of[index] {
                    Some(rank) => (
                        (rank + 1).to_string(),
                        format!("{:.4}", cover.remaining[rank]),
                        if rank < cover.covered { "yes" } else { "no" },
                    ),
                    None => (String::new(), String::new(), "computed"),
                };
                let source = if rank_of[index].is_none() {
                    "pafi"
                } else {
                    class.source
                };
                out.push_str(&format!(
                    "{temperature},{stage},{},{},{source},{:.4},{:.4},{:.4},{rank},{remaining},{selected},{sigma:.4},{tolerance}\n",
                    class.label,
                    class.kind,
                    sensitivity[index][0],
                    sensitivity[index][1],
                    sensitivity[index][2],
                ));
            }
            eprintln!(
                "sensitivity {temperature} K {stage}: {} classes, {} on the model, cover {} for sigma {sigma:.4} and tolerance {tolerance} at {:.1} min",
                classes.len(),
                on_model.len(),
                cover.covered,
                started.elapsed().as_secs_f64() / 60.0
            );
        }
    }
    std::fs::write(root.join("defect_pair_network_sensitivity.csv"), out).expect("sensitivity");
}

#[test]
#[ignore = "rank the near-shell hop classes by the sensitivity of V_iv to their rates"]
fn defect_pair_network_sensitivity() {
    write_sensitivities();
}

/// One PAFI path of a hop class: the band from the bound state through the
/// verified dimer saddle to the relaxed push on the target side, else
/// through the top of the campaign band to its relaxed product. When the
/// target recombines, the band ends just past the saddle instead and only
/// the forward direction is kept.
struct ClassPafi {
    path: &'static str,
    d_e0: f64,
    zero: f64,
    band_force: f64,
    forward: (f64, f64),
    nu_forward: f64,
    backward: (f64, f64),
    nu_backward: f64,
    product: String,
    samples: usize,
}

/// A basin label that is the configuration `expect` up to the group, or
/// the same outcome.
fn same_basin(basin: &str, expect: &str, group: &[rgsaddle::Mat3i]) -> bool {
    basin == expect
        || matches!(
            (parse_config_label(basin), parse_config_label(expect)),
            (Some(l), Some(r)) if l.canonical(group) == r.canonical(group)
        )
}

/// Searches the saddle again as [`network_saddle`] does, then joins the
/// bound state, the saddle and the target basin by a climbing band of
/// thirteen images with energy-weighted springs, and integrates the
/// projected average force along it at `temperature`. A recombining target
/// is several eV down, so its band ends at the saddle pushed 0.3 Å
/// (largest atom) toward it, or at the campaign band image after the top.
/// The forward barrier is the free-energy maximum above the reactant, the
/// backward one the same maximum above the last plane, and each attempt
/// frequency comes from the curvature along the path at its end. A
/// maximum on an end plane carries the error of the next plane in, not
/// zero.
#[allow(clippy::too_many_arguments)]
fn class_pafi(
    copper: &FastCopper,
    state: &NetworkState,
    repeats: usize,
    a: f64,
    temperature: f64,
    bulk: [[f64; 3]; 3],
    hint: (Array1<f64>, Array1<f64>),
    want: &[String; 2],
    group: &[rgsaddle::Mat3i],
) -> Result<ClassPafi, String> {
    let cell = state.cell;
    let expect = &want[1];
    let saddle = network_saddle(copper, state, repeats, a, hint, want)?;
    let mut ends = saddle.basins.clone();
    ends.sort();
    let mut wanted = want.clone();
    wanted.sort();
    let connected = ends == wanted && state.basin == want[0];
    let recombines = expect == "recombined";
    let (path, peak, downhill) = match (&saddle.positions, &saddle.mode, connected) {
        (Some(positions), Some(mode), true) => {
            let side = usize::from(saddle.basins[0] != *expect);
            let end = if recombines {
                let sign = if side == 0 { 1.0 } else { -1.0 };
                Some(positions + &(mode * (3.0 * sign)))
            } else {
                saddle.ends[side].clone()
            };
            (
                "dimer",
                positions.clone(),
                end.ok_or("saddle push did not relax")?,
            )
        }
        _ if saddle.band_force <= 0.02 && same_basin(&saddle.product, expect, group) => (
            "band",
            saddle.top.clone(),
            if recombines {
                saddle.after_top.clone()
            } else {
                saddle.product_positions.clone()
            },
        ),
        _ => {
            return Err(format!(
                "no path to {expect}: basins {} {}; band product {} at {:.3e} eV/A",
                saddle.basins[0], saddle.basins[1], saddle.product, saddle.band_force
            ));
        }
    };
    let product = if recombines {
        saddle.product.clone()
    } else {
        pair_basin(&downhill, cell, repeats, a, state.vacancy)
    };
    let held = FastHeld { copper, cell };
    let config = survey_config(bulk);
    let segment = config
        .band
        .clone()
        .with_force_tol(2.0e-3)
        .with_time_step(0.04);
    let config = config
        .with_band(segment)
        .with_max_band_steps(240)
        .with_images(13);
    let (band, _, band_force) = rgsaddle::segment_band(
        &held,
        &config,
        &state.positions,
        &peak,
        &downhill,
        cell,
        0.1,
        5.0,
    )
    .map_err(|error| format!("band {error}"))?;
    let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs())
        .map_or(f64::NAN, |profile| profile.barrier);
    let d_e0 = held
        .eval(peak.view())
        .map_or(f64::NAN, |(energy, _)| energy - state.energy);
    let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    let pafi = config
        .pafi
        .clone()
        .with_thermal(thermal)
        .with_samples(16)
        .with_thermalize(20)
        .with_max_rounds(4)
        .with_target_error(0.2 * thermal);
    let (profile, _) = rgsaddle::pafi_profile(&held, &band, &band.arcs(), &pafi)
        .map_err(|error| format!("pafi {error}"))?;
    let (start, end) = band.bounds();
    let frequency = |arc: f64| -> f64 {
        band.geometry(arc).ok().map_or(f64::NAN, |frame| {
            rgsaddle::attempt_frequency(
                curvature_along(&held, &frame.positions, &frame.tangent),
                63.546,
                rgsaddle::angular_scale_rad_s(),
            )
            .unwrap_or(f64::NAN)
        })
    };
    let last = profile.free_energy.len() - 1;
    let forward_error = if profile.dividing == 0 {
        profile.free_energy_error[1]
    } else {
        profile.barrier_error
    };
    let (backward, nu_backward) = if recombines {
        ((f64::NAN, f64::NAN), f64::NAN)
    } else if profile.dividing == last {
        let span = profile.arc[last] - profile.arc[last - 1];
        let error =
            0.5 * span * profile.gradient_error[last - 1].hypot(profile.gradient_error[last]);
        ((0.0, error), frequency(end))
    } else {
        (profile.reverse_barrier(), frequency(end))
    };
    Ok(ClassPafi {
        path,
        d_e0,
        zero,
        band_force,
        forward: (profile.barrier, forward_error),
        nu_forward: frequency(start),
        backward,
        nu_backward,
        product,
        samples: profile.samples,
    })
}

const PAFI_HEADER: &str = "temperature_K,lattice_A,kind,mechanism,hop,source,target,outcome,path,dE0_eV,zero_T_eV,band_force_eV_per_A,dF_forward_eV,dF_forward_error_eV,nu_forward_Hz,dF_backward_eV,dF_backward_error_eV,nu_backward_Hz,product_basin,samples,seconds";

/// Scientific notation with six digits, blank when not finite.
fn sci6(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6e}")
    } else {
        String::new()
    }
}

#[test]
#[ignore = "PAFI rates of the near-shell hop classes that carry the capture weight, 8788-atom cell"]
fn defect_pair_network_pafi() {
    let started = Instant::now();
    let repeats = 13usize;
    let temperatures: Vec<f64> = env_csv("RGSADDLE_TEMPS", "300,500,700")
        .iter()
        .map(|item| item.parse().expect("temperature"))
        .collect();
    let limit: usize = env_csv("RGSADDLE_HOP_LIMIT", "100000")[0]
        .parse()
        .expect("hop limit");
    let copper = FastCopper::new();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let path = root.join("defect_pair_network_pafi.csv");
    if !path.exists() {
        std::fs::write(&path, format!("{PAFI_HEADER}\n")).expect("pafi");
    }
    let done = network_done(&std::fs::read_to_string(&path).expect("pafi"));
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let ranking = std::fs::read_to_string(root.join("defect_pair_network_sensitivity.csv"))
        .expect("run defect_pair_network_sensitivity first");
    let selected: std::collections::HashSet<(i64, String)> = ranking
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect::<Vec<_>>())
        .filter(|f| f.len() > 10 && f[10] == "yes")
        .filter_map(|f| Some((f[0].parse::<f64>().ok()?.round() as i64, f[2].to_string())))
        .collect();
    let (group, network) = dumbbell_network(EXPLICIT_NORM2);
    let checkpoint = Mutex::new(File::options().append(true).open(&path).expect("append"));
    let write = |line: String| {
        eprintln!(
            "pafi {line} at {:.1} min",
            started.elapsed().as_secs_f64() / 60.0
        );
        append_checkpoint(&checkpoint, &line);
    };
    let isolated_source = rgsaddle::PairConfig::new([2, 0, 0], [1, 0, 0]);
    struct PafiJob {
        temperature: f64,
        source: Option<usize>,
        hop: Option<usize>,
        kind: String,
        from: rgsaddle::PairConfig,
        to: rgsaddle::PairConfig,
        via: rgsaddle::Vec3i,
        expect: String,
        target: String,
    }
    let mut jobs: Vec<PafiJob> = Vec::new();
    for &temperature in &temperatures {
        for (kind, to, via) in isolated_network_moves() {
            if !done.contains(&(temperature as i64, "isolated".into(), kind.into())) {
                jobs.push(PafiJob {
                    temperature,
                    source: None,
                    hop: None,
                    kind: kind.into(),
                    from: isolated_source,
                    to,
                    via,
                    expect: config_label(to),
                    target: config_label(to),
                });
            }
        }
        let spontaneous = census_spontaneous(&census, temperature);
        let is_spontaneous =
            |i: usize| spontaneous.contains(&state_miller(network.states[i].config, &group));
        let mut claimed = std::collections::HashSet::new();
        for (index, state) in network.states.iter().enumerate() {
            if is_spontaneous(index) {
                continue;
            }
            for edge in &state.edges {
                let (expect, target) = match edge.target {
                    rgsaddle::PairTarget::Outside => continue,
                    rgsaddle::PairTarget::Contact => ("recombined".to_string(), "contact".into()),
                    rgsaddle::PairTarget::State(j) if is_spontaneous(j) => {
                        ("recombined".to_string(), network.id(j))
                    }
                    rgsaddle::PairTarget::State(j) => (config_label(edge.config), network.id(j)),
                };
                if !claimed.insert(edge.hop) {
                    continue;
                }
                let label = hop_label(&network.hops[edge.hop]);
                if !selected.contains(&(temperature as i64, label.clone()))
                    || done.contains(&(temperature as i64, "hop".into(), label))
                {
                    continue;
                }
                jobs.push(PafiJob {
                    temperature,
                    source: Some(index),
                    hop: Some(edge.hop),
                    kind: network.hops[edge.hop].kind.clone(),
                    from: state.config,
                    to: edge.config,
                    via: edge.via,
                    expect,
                    target,
                });
            }
        }
    }
    jobs.truncate(limit);
    let mut placements: Vec<(f64, Option<usize>)> = jobs
        .iter()
        .map(|job| (job.temperature, job.source))
        .collect();
    placements.sort_by(|l, r| l.0.total_cmp(&r.0).then(l.1.cmp(&r.1)));
    placements.dedup();
    eprintln!(
        "pafi: {} jobs from {} placements on {} threads",
        jobs.len(),
        placements.len(),
        rayon::current_num_threads()
    );
    let relaxed: Vec<((f64, Option<usize>), NetworkState)> = placements
        .par_iter()
        .map(|&(temperature, index)| {
            let a = demkowicz_lattice(temperature);
            let config = index.map_or(isolated_source, |i| network.states[i].config);
            let (state, _, _) = network_state(&copper, repeats, a, config, index.is_some());
            ((temperature, index), state)
        })
        .collect();
    let bulk: Vec<(f64, [[f64; 3]; 3])> = temperatures
        .iter()
        .map(|&temperature| {
            let (perfect, cell) = fcc(repeats, demkowicz_lattice(temperature));
            (temperature, copper.evaluate(perfect.view(), cell).2)
        })
        .collect();
    jobs.par_iter().for_each(|job| {
        let clock = Instant::now();
        let a = demkowicz_lattice(job.temperature);
        let state = &relaxed
            .iter()
            .find(|(key, _)| key.0 == job.temperature && key.1 == job.source)
            .expect("relaxed state")
            .1;
        let stress = bulk
            .iter()
            .find(|item| item.0 == job.temperature)
            .expect("bulk stress")
            .1;
        let rotation = job.kind == "rotate";
        let found = class_pafi(
            &copper,
            state,
            repeats,
            a,
            job.temperature,
            stress,
            hop_guess(state, a, job.from, job.to, rotation, job.via),
            &[config_label(job.from), job.expect.clone()],
            &group,
        );
        let columns = match &found {
            Err(error) => format!("no_path,,,,,,,,,,,,{},", error.replace(',', ";")),
            Ok(found) => format!(
                "{},{},{},{},{:.3e},{},{},{},{},{},{},{},{}",
                if found.band_force <= PAFI_BAND_FORCE {
                    "rated"
                } else {
                    "unconverged"
                },
                found.path,
                fixed6(found.d_e0),
                fixed6(found.zero),
                found.band_force,
                fixed6(found.forward.0),
                fixed6(found.forward.1),
                sci6(found.nu_forward),
                fixed6(found.backward.0),
                fixed6(found.backward.1),
                sci6(found.nu_backward),
                found.product,
                found.samples
            ),
        };
        let (kind, hop, source) = match job.hop {
            Some(h) => (
                "hop",
                hop_label(&network.hops[h]),
                network.id(job.source.expect("source")),
            ),
            None => ("isolated", job.kind.clone(), "isolated".to_string()),
        };
        write(format!(
            "{},{a:.3},{kind},{},{hop},{source},{},{columns},{:.1}",
            job.temperature,
            job.kind,
            job.target,
            clock.elapsed().as_secs_f64()
        ));
    });
    eprintln!(
        "pafi wall {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

/// V_iv from the per-vector network: the flow census, the converged
/// isolated rates, the near-shell saddles of `defect_pair_network.csv` and
/// the PAFI rates of `defect_pair_network_pafi.csv`. A rated PAFI direction
/// replaces the barrier-shift model there; the other near-shell rates keep
/// the model. The errors are the spread over the 64 draws of the isolated
/// translation and rotation rates within their own errors, a common factor
/// `exp(z sigma)` on every near-shell rate still on the model, `sigma` the
/// root mean square of the model against the PAFI directions, and an
/// independent normal on each PAFI direction within its free-energy error
/// plus one per isolated kind within the isolated PAFI error, both with the
/// drift of [`PafiInputs`] added. At the 30 Å
/// cutoff the change from the recorded converged-campaign volume is split
/// into five steps: the lumped model without its near-shell rows; the
/// per-vector network with dipole-biased rates only and the lumped model's
/// return layer and chain radius; the same network matched to the crystal
/// outside the cutoff; the matched network with the near-shell saddles,
/// its errors with the spread of [`barrier_shift_spread`]; and the same
/// with the PAFI rates.
fn assemble_network() {
    let started = Instant::now();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let saddles = std::fs::read_to_string(root.join("defect_pair_network.csv")).expect("network");
    let pafi_text =
        std::fs::read_to_string(root.join("defect_pair_network_pafi.csv")).expect("pafi");
    let before =
        std::fs::read_to_string(root.join("defect_pair_converged_volume.csv")).expect("before");
    let rates = converged_rates();
    let isolated_only: String = rates
        .lines()
        .filter(|line| line.split(',').nth(1) == Some("isolated"))
        .flat_map(|line| [line, "\n"])
        .collect();
    let draws: usize = env_csv("RGSADDLE_DRAWS", "64")[0].parse().expect("draws");
    let (recorded_sigma, recorded_rows) = barrier_shift_spread();
    let group = rgsaddle::cubic_group();
    let (_, explicit_network) = dumbbell_network(EXPLICIT_NORM2);
    eprintln!("barrier-shift spread {recorded_sigma:.4} in ln k over {recorded_rows} PAFI rows");
    let mut out = String::from(
        "temperature_K,estimator,volume_over_Omega,sigma,spontaneous_sites,R_eff_A,demkowicz_8788,note\n",
    );
    for temperature in [300.0_f64, 500.0, 700.0] {
        let context = network_context(
            temperature,
            &census,
            &saddles,
            &rates,
            &group,
            &explicit_network,
        );
        let NetworkContext {
            a,
            k_mig,
            mig_spread,
            mig_note,
            k_rot,
            rot_spread,
            ..
        } = context;
        let inputs = &context.inputs;
        let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
        let pafi = pafi_inputs(
            &pafi_text,
            temperature,
            &context.spontaneous_ids,
            &context.inputs,
            &explicit_network,
        );
        let model_sigma = pafi.model_sigma().unwrap_or(recorded_sigma);
        let rated = pafi.central();
        let table = match temperature as i32 {
            300 => "254-286",
            500 => "161-185",
            700 => "203-210",
            _ => "",
        };
        eprintln!(
            "network {temperature} K: k_mig {k_mig:.4e} ({mig_note}, sigma {mig_spread:.3}); k_rot {k_rot:.4e} (sigma {rot_spread:.3}); {} saddles ({} from the band), {} bound states, isolated E0 {:.6}/{:.6} eV; {} PAFI classes, {} directions, model sigma {model_sigma:.4} over {}",
            inputs.saddles.len(),
            inputs.band.len(),
            inputs.states.len(),
            inputs.isolated[0],
            inputs.isolated[1],
            pafi.rated,
            pafi.computed.len(),
            pafi.residuals.len()
        );
        let mut rng = TinyRng(0x5eed_0000 + temperature as u64);
        let shifts: Vec<[f64; 3]> = (0..draws)
            .map(|_| [rng.gauss(), rng.gauss(), rng.gauss()])
            .collect();
        let mut class_rng = TinyRng(0x5eed_9af1_0000 + temperature as u64);
        let class_shifts: Vec<Vec<f64>> = (0..draws)
            .map(|_| {
                (0..pafi.computed.len() + 2)
                    .map(|_| class_rng.gauss())
                    .collect()
            })
            .collect();
        let scale_of = |z: &[f64; 3], sigma: f64| NetworkScale {
            mig: (z[0] * mig_spread).exp(),
            rot: (z[1] * rot_spread).exp(),
            capture: (z[2] * sigma).exp(),
        };
        let spread_of = |network: &rgsaddle::PairNetwork, explicit: bool, with_pafi: bool| {
            let samples: Vec<[f64; 3]> = (0..draws)
                .into_par_iter()
                .map(|draw| {
                    let (classes, sigma) = if with_pafi {
                        (pafi.drawn(&class_shifts[draw], thermal), model_sigma)
                    } else {
                        (ClassRates::default(), recorded_sigma)
                    };
                    let varied = context
                        .volumes(
                            network,
                            &group,
                            explicit,
                            scale_of(&shifts[draw], sigma),
                            &classes,
                            true,
                        )
                        .sphere;
                    [varied.sphere, varied.decay_volume, varied.mean_volume]
                })
                .collect();
            draw_spread(&samples)
        };
        let common = format!(
            "per-vector network; flow census; near-shell saddles of defect_pair_network.csv; {} PAFI classes ({} directions) of defect_pair_network_pafi.csv; k_mig {k_mig:.4e} Hz {mig_note}; k_rot {k_rot:.4e} Hz; sigma from {draws} draws (translation, rotation, model factor sigma {model_sigma:.4} over {} PAFI directions, PAFI free-energy errors)",
            pafi.rated,
            pafi.computed.len(),
            pafi.residuals.len()
        );
        let mut at_30 = None;
        for cutoff in [10.8_f64, 14.0, 18.0, 22.0, 26.0, 30.0] {
            let network = context.network_to(cutoff);
            let central = context.volumes(&network, &group, true, UNIT_SCALE, &rated, true);
            let spread = spread_of(&network, true, true).map(blank_or);
            let SphereVolumes {
                d_graph,
                r_eff,
                sphere,
                decay_volume,
                mean_volume,
                tau_cell,
            } = central.sphere;
            let spont = central.spont_sites;
            let detail = format!(
                "cutoff {cutoff:.1}; {common}; {} states to {:.4} A; edges {} pafi {} saddle {} band {} dipole {} fallback; chain R {:.4}",
                central.states,
                central.max_r,
                central.computed_edges,
                central.explicit_edges,
                central.band_edges,
                central.elastic_edges,
                central.fallback_edges,
                central.chain_radius
            );
            out.push_str(&format!("{temperature},smoluchowski,{sphere:.4},{},{spont:.4},{r_eff:.4},{table},{detail}; infinite crystal; D_graph {:.6}\n", spread[0], d_graph / 1.0e12));
            out.push_str(&format!("{temperature},cell-decay,{decay_volume:.4},{},{spont:.4},{r_eff:.4},{table},{detail}; slowest decay in the 13^3 cell\n", spread[1]));
            out.push_str(&format!("{temperature},mean-time,{mean_volume:.4},{},{spont:.4},{r_eff:.4},{table},{detail}; uniform-start mean time in the 13^3 cell {tau_cell:.6e} s\n", spread[2]));
            out.push_str(&format!(
                "{temperature},site-sum,{:.4},,{spont:.4},,{table},{detail}\n",
                central.total
            ));
            eprintln!(
                "network {temperature} K R {cutoff:.1}: sphere {sphere:.2}±{} cell {decay_volume:.2}±{} mean {mean_volume:.2}±{} R_eff {r_eff:.3} states {} edges {}/{}/{}/{}/{} at {:.1} min",
                spread[0],
                spread[1],
                spread[2],
                central.states,
                central.computed_edges,
                central.explicit_edges,
                central.band_edges,
                central.elastic_edges,
                central.fallback_edges,
                started.elapsed().as_secs_f64() / 60.0
            );
            if cutoff == 30.0 {
                at_30 = Some((network, central.sphere, spread));
            }
        }
        let (network, with_pafi, pafi_spread) = at_30.expect("30 A cutoff");
        let estimators = ["smoluchowski", "cell-decay", "mean-time"];
        let recorded: Vec<(f64, String)> = estimators
            .iter()
            .map(|name| {
                before
                    .lines()
                    .map(|line| line.split(',').collect::<Vec<_>>())
                    .find(|f| {
                        f.len() > 7
                            && f[0].parse::<f64>().ok() == Some(temperature)
                            && f[1] == *name
                            && f[7].starts_with("cutoff 30.0;")
                    })
                    .map(|f| (f[2].parse().expect("volume"), f[3].to_string()))
                    .expect("recorded volume at 30 A")
            })
            .collect();
        let fractions = census_fractions(&census, temperature);
        let lumped = rate_volumes(
            &shell_table_within(a, 24, 30.0),
            &|id: &str| {
                let radius: f64 = id
                    .split('-')
                    .next()
                    .and_then(|text| text.parse().ok())
                    .unwrap_or(f64::NAN);
                fractions
                    .iter()
                    .find(|shell| (shell.0 - radius).abs() < 2.0e-3)
                    .map(|shell| shell.1 as f64 / shell.2 as f64)
                    .unwrap_or(0.0)
            },
            RateScale::RECORDED,
            &isolated_only,
            temperature,
            k_mig,
            a,
            Some(&context.field),
            30.0,
        );
        let plain = ClassRates::default();
        let sphere_return = context
            .volumes(&network, &group, false, UNIT_SCALE, &plain, false)
            .sphere;
        let elastic = context
            .volumes(&network, &group, false, UNIT_SCALE, &plain, true)
            .sphere;
        let elastic_spread = spread_of(&network, false, false).map(blank_or);
        let explicit = context
            .volumes(&network, &group, true, UNIT_SCALE, &plain, true)
            .sphere;
        let explicit_spread = spread_of(&network, true, false).map(blank_or);
        let steps: [(&str, [f64; 3], [String; 3], &str); 6] = [
            (
                "before",
                [recorded[0].0, recorded[1].0, recorded[2].0],
                [
                    recorded[0].1.clone(),
                    recorded[1].1.clone(),
                    recorded[2].1.clone(),
                ],
                "defect_pair_converged_volume.csv: lumped radius/axis states with the recorded near-shell capture, rotation, away and strain rows",
            ),
            (
                "lumped-isolated",
                [lumped.sphere, lumped.decay_volume, lumped.mean_volume],
                [String::new(), String::new(), String::new()],
                "the same lumped model with every near-shell row dropped: isolated rates and the orientation-averaged dipole bias",
            ),
            (
                "vector-return",
                [
                    sphere_return.sphere,
                    sphere_return.decay_volume,
                    sphere_return.mean_volume,
                ],
                [String::new(), String::new(), String::new()],
                "per-vector network; every hop at the isolated rate with the vector- and orientation-resolved dipole bias; return layer and median P r as in the lumped model",
            ),
            (
                "vector-dipole",
                [elastic.sphere, elastic.decay_volume, elastic.mean_volume],
                elastic_spread,
                "the same network matched to the field-free crystal outside the cutoff",
            ),
            (
                "vector-saddles",
                [explicit.sphere, explicit.decay_volume, explicit.mean_volume],
                explicit_spread,
                "the matched network with the 13^3 near-shell saddles inside the census radius; near-shell factor sigma from the recorded PAFI rows",
            ),
            (
                "vector-pafi",
                [
                    with_pafi.sphere,
                    with_pafi.decay_volume,
                    with_pafi.mean_volume,
                ],
                pafi_spread,
                "the same network with the PAFI rates of the covered classes; model factor sigma from the PAFI directions",
            ),
        ];
        for (step, values, spreads, note) in &steps {
            for (which, name) in estimators.iter().enumerate() {
                out.push_str(&format!(
                    "{temperature},{step}:{name},{:.4},{},,,{table},cutoff 30.0; {note}\n",
                    values[which], spreads[which]
                ));
            }
            eprintln!(
                "breakdown {temperature} K {step}: sphere {:.2}±{} cell {:.2}±{} mean {:.2}±{}",
                values[0], spreads[0], values[1], spreads[1], values[2], spreads[2]
            );
        }
    }
    std::fs::write(root.join("defect_pair_network_volume.csv"), out).expect("volume");
    eprintln!(
        "network assembly {:.1} min",
        started.elapsed().as_secs_f64() / 60.0
    );
}

#[test]
#[ignore = "V_iv from the per-vector network with the near-shell saddles and PAFI rates"]
fn defect_pair_network_volume() {
    assemble_network();
}

/// The 8788-atom alpha and beta volumes of [`table_i`] with their errors.
fn table_i_8788(temperature: f64) -> Option<[(f64, f64); 2]> {
    let text = table_i(temperature);
    let tail = text.split("8788").nth(1)?;
    let mut pairs = tail.split(';').next()?.split(" and ").map(|item| {
        let item = item
            .trim_start_matches([':', ' '])
            .trim_start_matches("atoms: ");
        let (value, error) = item.split_once('±')?;
        Some((value.trim().parse().ok()?, error.trim().parse().ok()?))
    });
    Some([pairs.next()??, pairs.next()??])
}

/// `defect_pair_network_notes.txt`, written only from
/// `defect_pair_network.csv`, `defect_pair_network_sensitivity.csv`,
/// `defect_pair_network_pafi.csv`, `defect_pair_network_volume.csv` and the
/// census that marks the spontaneous states.
fn network_notes() -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let census = std::fs::read_to_string(root.join("defect_pair_census.csv")).expect("census");
    let group = rgsaddle::cubic_group();
    let (_, network) = dumbbell_network(EXPLICIT_NORM2);
    let saddles = std::fs::read_to_string(root.join("defect_pair_network.csv")).expect("network");
    let volumes =
        std::fs::read_to_string(root.join("defect_pair_network_volume.csv")).expect("volumes");
    let pafi_text =
        std::fs::read_to_string(root.join("defect_pair_network_pafi.csv")).expect("pafi");
    let ranking = std::fs::read_to_string(root.join("defect_pair_network_sensitivity.csv"))
        .expect("sensitivity");
    let ranked: Vec<Vec<&str>> = ranking
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect();
    let rows: Vec<Vec<&str>> = saddles
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect();
    let volume_rows: Vec<Vec<&str>> = volumes
        .lines()
        .skip(1)
        .map(|line| line.split(',').collect())
        .collect();
    let volume = |temperature: &str, estimator: &str, cutoff: &str| -> (f64, String) {
        volume_rows
            .iter()
            .find(|f| {
                f[0] == temperature
                    && f[1] == estimator
                    && f[7].starts_with(&format!("cutoff {cutoff};"))
            })
            .map(|f| (f[2].parse().expect("volume"), f[3].to_string()))
            .unwrap_or((f64::NAN, String::new()))
    };
    let mut notes = vec![
        "Near-shell saddles of every symmetry-distinct vacancy-dumbbell hop in the 13^3 (8788-site) cell on the Table 1 lattices, from defect_pair_network.csv; the sensitivity of the cell-decay V_iv to each hop class, from defect_pair_network_sensitivity.csv; PAFI rates of the classes that carry it, from defect_pair_network_pafi.csv; and the per-vector V_iv of defect_pair_network_volume.csv. Generated by network_notes in tests/mishin_cu.rs. A negative barrier is a saddle below a bound state that is flat to the 1e-3 eV/A tolerance; the assembly takes it as zero. A PAFI direction runs at k_kind (nu/nu_iso) exp(-(F - F_iso)/kT) against the isolated hop of the same kind in the same cell.".to_string(),
    ];
    for temperature in ["300", "500", "700"] {
        let here: Vec<&Vec<&str>> = rows.iter().filter(|f| f[0] == temperature).collect();
        if here.is_empty() {
            continue;
        }
        let count = |kind: &str, outcome: &str| {
            here.iter()
                .filter(|f| f[2] == kind && (outcome.is_empty() || f[7] == outcome))
                .count()
        };
        let seconds: f64 = here
            .iter()
            .filter_map(|f| f.get(18).and_then(|v| v.parse::<f64>().ok()))
            .sum();
        let isolated = |mechanism: &str| {
            here.iter()
                .find(|f| f[2] == "isolated" && f[3] == mechanism && f[7] == "connected")
                .map_or("not connected".to_string(), |f| format!("{} eV", f[8]))
        };
        let spontaneous = census_spontaneous(&census, temperature.parse().expect("temperature"));
        let spontaneous_ids: std::collections::HashSet<String> = (0..network.states.len())
            .filter(|i| spontaneous.contains(&state_miller(network.states[*i].config, &group)))
            .map(|i| network.id(i))
            .collect();
        let inputs = network_inputs(
            &saddles,
            temperature.parse().expect("temperature"),
            &group,
            &spontaneous_ids,
        );
        let lattice = here.first().map_or("", |f| f[1]);
        notes.push(format!(
            "T {temperature} K, a {lattice} A: {} bound states relaxed by ODE12r to 1e-3 eV/A ({} left their placement); {} hop classes searched, {} connected by the dimer and both pushes, {} misconnected and {} without a band; {} misconnected hops use the top of a band of at most 0.02 eV/A that ends in the target basin and {} fall back to the dipole rate; isolated translation {}, rotation {}; {:.1} thread-hours.",
            count("state", "bound") - usize::from(here.iter().any(|f| f[2] == "state" && f[5] == "isolated")),
            count("state", "left"),
            count("hop", ""),
            count("hop", "connected"),
            count("hop", "misconnected"),
            count("hop", "no_saddle"),
            inputs.band.len(),
            count("hop", "") - inputs.saddles.len(),
            isolated("translate"),
            isolated("rotate"),
            seconds / 3600.0
        ));
        let t: f64 = temperature.parse().expect("temperature");
        let pafi = pafi_inputs(&pafi_text, t, &spontaneous_ids, &inputs, &network);
        let pafi_rows: Vec<Vec<&str>> = pafi_text
            .lines()
            .skip(1)
            .map(|line| line.split(',').collect::<Vec<_>>())
            .filter(|f| f.len() >= 21 && f[0] == temperature)
            .collect();
        let pafi_count = |kind: &str, outcome: &str, path: &str| {
            pafi_rows
                .iter()
                .filter(|f| {
                    f[2] == kind
                        && (outcome.is_empty() || f[7] == outcome)
                        && (path.is_empty() || f[8] == path)
                })
                .count()
        };
        let stage = |name: &str| -> Vec<&Vec<&str>> {
            ranked
                .iter()
                .filter(|f| f.len() > 12 && f[0] == temperature && f[1] == name)
                .collect()
        };
        let model = stage("model");
        let after = stage("pafi");
        let reference = |mechanism: &str| {
            pafi_rows
                .iter()
                .find(|f| f[2] == "isolated" && f[3] == mechanism && f[7] == "rated")
                .map_or("not rated".to_string(), |f| {
                    format!(
                        "dF {}±{} eV forward and {}±{} eV back, nu {} Hz",
                        f[12], f[13], f[15], f[16], f[14]
                    )
                })
        };
        let mean_residual = pafi.residuals.iter().sum::<f64>() / pafi.residuals.len().max(1) as f64;
        notes.push(format!(
            "T {temperature} K PAFI: the model-stage cover (sigma {} in ln k, tolerance {} Omega) selects {} of {} hop classes and the PAFI-stage covers of later passes {} more; {} hop rows, {} rated ({} through the dimer saddle, {} through the band top), {} with an unconverged band and {} without a path; {} rated directions; isolated translation {}, rotation {}; half the gap between the two isolated directions, {:.4} eV translating and {:.4} eV rotating, is added to the error of every direction of that kind; the model against the {} rated directions with a model rate: mean {mean_residual:+.4}, root mean square {} in ln k; {:.1} job-hours, the summed wall time of jobs run eight at a time on eight threads.",
            model.first().map_or("", |f| f[11]),
            model.first().map_or("", |f| f[12]),
            model.iter().filter(|f| f[10] == "yes").count(),
            model.len(),
            pafi_count("hop", "", "") - model.iter().filter(|f| f[10] == "yes").count(),
            pafi_count("hop", "", ""),
            pafi_count("hop", "rated", ""),
            pafi_count("hop", "rated", "dimer"),
            pafi_count("hop", "rated", "band"),
            pafi_count("hop", "unconverged", ""),
            pafi_count("hop", "no_path", ""),
            pafi.computed.len(),
            reference("translate"),
            reference("rotate"),
            pafi.drift[0],
            pafi.drift[1],
            pafi.residuals.len(),
            pafi.model_sigma().map_or(String::new(), |sigma| format!("{sigma:.4}")),
            pafi.seconds / 3600.0
        ));
        if let Some(first) = after.first() {
            let sigma: f64 = first[11].parse().expect("sigma");
            let on_model: Vec<&&Vec<&str>> = after.iter().filter(|f| f[10] != "computed").collect();
            let rest: f64 = on_model
                .iter()
                .map(|f| f[6].parse::<f64>().expect("sensitivity").abs())
                .sum();
            notes.push(format!(
                "T {temperature} K after PAFI: {} classes computed; the {} still on the model, all moved by sigma {sigma:.4}, shift the cell-decay volume by at most {:.2} Omega against the tolerance {} Omega; {} more selected.",
                after.len() - on_model.len(),
                on_model.len(),
                sigma * rest,
                first[12],
                on_model.iter().filter(|f| f[10] == "yes").count()
            ));
        }
        let top: Vec<String> = model
            .iter()
            .take(8)
            .map(|f| format!("{} ({}) {}", f[2], f[4], f[6]))
            .collect();
        notes.push(format!(
            "T {temperature} K largest dV_cell/d ln k before PAFI (Omega): {}.",
            top.join(", ")
        ));
        for mechanism in ["translate", "rotate"] {
            let mut barriers: Vec<f64> = here
                .iter()
                .filter(|f| f[2] == "hop" && f[3] == mechanism && f[7] == "connected")
                .filter_map(|f| f[8].parse().ok())
                .collect();
            barriers.sort_by(f64::total_cmp);
            if let (Some(low), Some(high)) = (barriers.first(), barriers.last()) {
                notes.push(format!(
                    "T {temperature} K {mechanism}: {} connected barriers from {low:.4} to {high:.4} eV, median {:.4} eV.",
                    barriers.len(),
                    barriers[barriers.len() / 2]
                ));
            }
        }
    }
    for temperature in ["300", "500", "700"] {
        let t: f64 = temperature.parse().expect("temperature");
        let reference = table_i_8788(t)
            .map(|[(alpha, ea), (beta, eb)]| {
                format!("Table 1 alpha {alpha}±{ea}, beta {beta}±{eb}")
            })
            .unwrap_or_default();
        let at = |estimator: &str| {
            let (value, sigma) = volume(temperature, estimator, "30.0");
            if sigma.is_empty() {
                format!("{value:.2}")
            } else {
                format!("{value:.2}±{sigma}")
            }
        };
        notes.push(format!(
            "T {temperature} K at the 30 A cutoff: cell-decay {}, mean-time {}, free sphere {} Omega; {reference}.",
            at("cell-decay"),
            at("mean-time"),
            at("smoluchowski")
        ));
        let decay: Vec<String> = ["10.8", "14.0", "18.0", "22.0", "26.0", "30.0"]
            .iter()
            .map(|cutoff| {
                format!(
                    "{cutoff} {:.2}",
                    volume(temperature, "cell-decay", cutoff).0
                )
            })
            .collect();
        notes.push(format!(
            "T {temperature} K cell-decay by cutoff (A Omega): {}.",
            decay.join(", ")
        ));
        for estimator in ["cell-decay", "mean-time", "smoluchowski"] {
            let steps = [
                "before",
                "lumped-isolated",
                "vector-return",
                "vector-dipole",
                "vector-saddles",
                "vector-pafi",
            ];
            let values: Vec<f64> = steps
                .iter()
                .map(|step| volume(temperature, &format!("{step}:{estimator}"), "30.0").0)
                .collect();
            let moves: Vec<String> = steps
                .windows(2)
                .zip(values.windows(2))
                .map(|(names, pair)| format!("{} {:+.2}", names[1], pair[1] - pair[0]))
                .collect();
            notes.push(format!(
                "T {temperature} K {estimator} breakdown at 30 A: before {:.2}; {}; total {:+.2}.",
                values[0],
                moves.join(", "),
                values[5] - values[0]
            ));
        }
    }
    let mut text = notes.join("\n");
    text.push('\n');
    text
}

#[test]
#[ignore = "write defect_pair_network_notes.txt from the network CSVs"]
fn defect_pair_network_notes() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_network_notes.txt");
    std::fs::write(path, network_notes()).expect("notes");
}

#[test]
fn network_notes_match_csv() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_network_notes.txt");
    let committed = std::fs::read_to_string(path).expect("notes");
    assert_eq!(
        committed,
        network_notes(),
        "defect_pair_network_notes.txt is stale: run defect_pair_network_notes"
    );
}

#[test]
fn table_i_8788_parses_every_temperature() {
    assert_eq!(table_i_8788(300.0), Some([(254.0, 21.0), (286.0, 28.0)]));
    assert_eq!(table_i_8788(500.0), Some([(185.0, 23.0), (161.0, 15.0)]));
    assert_eq!(table_i_8788(700.0), Some([(203.0, 10.0), (210.0, 21.0)]));
}
