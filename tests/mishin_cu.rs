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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{
    BOLTZMANN_EV_PER_K, BandConfig, BandSession, BandStatus, BandSurface, Cell, CellMask, CiConfig,
    DefectSource, ElasticDipole, ElasticMultipole, FieldSolver, FmmSolver, IndexCheck, JointChart,
    Kelvin, LINEAR_FOLD_EV, MinModeConfig, MinModeStatus, PointSurface, SaddleError, SaddleSeed,
    SolidDimer, SolidDimerConfig, SolidState, SpringKind, Strain, StrainResponse, Trust,
    TrustPolicy, cubic_rotations, index_at, joint_coordinate,
};

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

fn relax(
    copper: &Copper,
    positions: &mut Array1<f64>,
    cell: [[f64; 3]; 3],
    tol: f64,
    max_steps: usize,
) -> usize {
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
    let mut done = 0;
    for _ in 0..max_steps {
        let oracle = Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
            let (energy, gradient, _) = copper.evaluate(x, cell);
            (energy, gradient)
        });
        solver.step(&oracle, positions).expect("relax");
        done += 1;
        let (_, gradient, _) = copper.evaluate(positions.view(), cell);
        if max_abs(&gradient) < tol {
            break;
        }
    }
    done
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
    let search = MinModeConfig {
        method: Method::Fire { kind: FireKind::V2 },
        force_tol: 2e-3,
        max_move: 0.008,
        dr: 1e-3,
        rotation_tol: 1e-3,
        max_rotations: 12,
        seed_curvature: false,
        ..MinModeConfig::default()
    };
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
    let search = MinModeConfig {
        force_tol: 2e-3,
        ..MinModeConfig::default()
    };
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
        let other_steps = relax(&copper, &mut other, cell, 1e-3, 200);
        let (e_other, g_other, _) = copper.evaluate(other.view(), cell);
        let (_, _, dir_other) = closest_pair(&other, cell);
        let (other_100, other_110, _, other_sep) = axis_report(dir_other);
        let mut minimum = sheared;
        let relax_steps = relax(&copper, &mut minimum, cell, 1e-3, 200);
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
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
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
            0 => Some(0.34700693547728179),
            20 => Some(0.16255571725275786),
            30 => Some(0.088051942122888249),
            40 => Some(0.027924583692424676),
            _ => None,
        };
    }
    match key {
        0 => Some(0.4738130973207717),
        20 => Some(0.2435460895545134),
        30 => Some(0.14829201811522807),
        40 => Some(0.067862165555311549),
        50 => Some(0.0077937374149428251),
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
    relax(copper, &mut other, cell, 1e-3, 200);
    let relax_110_calls = copper.calls() - before_110;
    let mut minimum = placed_100;
    let before_min = copper.calls();
    relax(copper, &mut minimum, cell, 1e-3, 200);
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
        copper,
        &nudged,
        cell,
        &connect,
        mask,
        external,
        cube,
        e_min,
        min_100,
        0.0,
        2500,
    );
    let search_calls = copper.calls() - before_search;
    let (_, sad_gradient, sigma_sad) = copper.evaluate(row.positions.view(), row.cell);
    let mode = minimum_image_difference(&row.positions, &minimum, cell);
    let index1 = row.status == MinModeStatus::Converged
        && !row.rejected
        && row.negative_modes == Some(1)
        && row.curvature < 0.0
        && row.angle_110 < 15.0;
    let accepted = if xy_path { index1 && row.switched } else { index1 };
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
    relax(&copper, &mut bulk, cube, 1e-3, 200);
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
        zero.barrier,
        zero.calls,
        captured.p_sad_xy,
        captured.multipole,
        zero_wall
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
        .with_policy(TrustPolicy {
            quadratic: Some(coeff),
            error_tolerance: LINEAR_FOLD_EV,
            floor: 0.0,
            ..TrustPolicy::default()
        })
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
                        "variable-cell enthalpy is a different ensemble from the dipole line".to_string(),
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
        let (method_barrier, trust, refine_calls, refine_ok, note) =
            if matches!(estimate.trust, Trust::Linear | Trust::Explicit) {
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

    let fixed: Vec<_> = tallies.iter().filter(|row| row.fixed && !row.batch).collect();
    let variable: Vec<_> = tallies.iter().filter(|row| !row.fixed && !row.batch).collect();
    let explicit_fixed: usize = fixed.iter().map(|row| row.explicit_calls).sum::<usize>();
    let method_fixed: usize =
        build_calls + fixed.iter().map(|row| row.method_calls).sum::<usize>();
    let explicit_variable: usize = variable.iter().map(|row| row.explicit_calls).sum::<usize>();
    let batch_explicit: usize = batch_explicit_once.iter().map(|calls| calls * 50).sum::<usize>();
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
        pos.push([
            positions[3 * i],
            positions[3 * i + 1],
            positions[3 * i + 2],
        ]);
        force.push([
            -gradient[3 * i],
            -gradient[3 * i + 1],
            -gradient[3 * i + 2],
        ]);
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
        format!(
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
    let n = positions.len() / 3;
    let lengths = [0, 1, 2].map(|row| {
        (cell[row][0] * cell[row][0] + cell[row][1] * cell[row][1] + cell[row][2] * cell[row][2])
            .sqrt()
    });
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
    let neighbors = |i: usize| {
        let mut pairs = Vec::new();
        for sx in -nimg..=nimg {
            for sy in -nimg..=nimg {
                for sz in -nimg..=nimg {
                    let target = [
                        (frac[i][0] - sx as f64).rem_euclid(1.0),
                        (frac[i][1] - sy as f64).rem_euclid(1.0),
                        (frac[i][2] - sz as f64).rem_euclid(1.0),
                    ];
                    let origin = frac_bin_ijk(target, nside);
                    let mut slots = Vec::with_capacity(8);
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
                    for slot in slots {
                        for &j in &bins[slot] {
                            if j < i || (j == i && !positive_image(sx, sy, sz)) {
                                continue;
                            }
                            let shift = [
                                sx as f64 * cell[0][0] + sy as f64 * cell[1][0] + sz as f64 * cell[2][0],
                                sx as f64 * cell[0][1] + sy as f64 * cell[1][1] + sz as f64 * cell[2][1],
                                sx as f64 * cell[0][2] + sy as f64 * cell[1][2] + sz as f64 * cell[2][2],
                            ];
                            let d = [
                                positions[3 * j] + shift[0] - positions[3 * i],
                                positions[3 * j + 1] + shift[1] - positions[3 * i + 1],
                                positions[3 * j + 2] + shift[2] - positions[3 * i + 2],
                            ];
                            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                            if r2 > 1e-16 && r2 < cutoff2 {
                                pairs.push(Pair { i, j, d, r: r2.sqrt() });
                            }
                        }
                    }
                }
            }
        }
        pairs
    };
    let pairs: Vec<Pair> = (0..n).into_par_iter().flat_map_iter(neighbors).collect();
    accumulate_eam(n, positions.len(), &pairs, det(cell).abs())
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

struct FastCopper {
    calls: AtomicUsize,
}

impl FastCopper {
    fn new() -> Self {
        Self { calls: AtomicUsize::new(0) }
    }

    fn evaluate(&self, positions: ArrayView1<f64>, cell: [[f64; 3]; 3]) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
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

    fn eval_in_cell(&self, x: ArrayView1<f64>, cell: &[f64; 9]) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient, _) = self.copper.evaluate(x, unflat(cell));
        Ok((energy, gradient))
    }

    fn cauchy_stress(&self, x: ArrayView1<f64>, cell: &[f64; 9]) -> Result<Option<[f64; 9]>, SaddleError> {
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

fn relax_fast(copper: &FastCopper, positions: &mut Array1<f64>, cell: [[f64; 3]; 3], tol: f64, max_steps: usize) {
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control { maxiter: usize::MAX, gtol: 0.0, istep: 0.08, ..Control::default() },
        positions.len(),
    );
    solver.set_atom_maxmove(0.08);
    for _ in 0..max_steps {
        let oracle = Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
            let (energy, gradient, _) = copper.evaluate(x, cell);
            (energy, gradient)
        });
        solver.step(&oracle, positions).expect("relax");
        let (_, gradient, _) = copper.evaluate(positions.view(), cell);
        if max_abs(&gradient) < tol {
            break;
        }
    }
}

fn shell_table(a: f64, repeats: usize) -> Vec<(f64, [f64; 3], usize)> {
    let max_r = 10.8_f64.min(repeats as f64 * a * 0.48);
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
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt().max(1.0e-8);
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
        Self {
            state: seed | 1,
        }
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
            let search = MinModeConfig {
                force_tol: 0.04,
                max_move: 0.1,
                dr: 1.0e-3,
                max_rotations: 8,
                ..MinModeConfig::default()
            };
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
                    report.status, report.max_force, report.iteration, report.negative_modes, report.rejected
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
            let mut plus = &saddle + &step;
            let mut minus = &saddle - &step;
            relax_fast(copper, &mut plus, cell, 1.5e-3, 70);
            relax_fast(copper, &mut minus, cell, 1.5e-3, 70);
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
            let (saddle_now, _) = held.eval(saddle.view()).unwrap_or((f64::NAN, Array1::zeros(0)));
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
    recombined: bool,
    gap: f64,
    separation: f64,
    key: (i32, i32, i32),
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

fn occupied(positions: &Array1<f64>, site: [f64; 3], cell: [[f64; 3]; 3]) -> bool {
    let n = positions.len() / 3;
    (0..n).any(|atom| mic_distance(point_of(positions, atom), site, cell) < 0.8)
}

fn relax_pair(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    vacancy: bool,
) -> (Array1<f64>, [f64; 3], [usize; 2], [[f64; 3]; 3], f64, f64, bool) {
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let built = build_pair(repeats, a, shell, vacancy);
    let mut positions = built.positions;
    relax_fast(copper, &mut positions, cell, 1.0e-3, 60);
    let gap = dist(point_of(&positions, built.dumbbell[0]), point_of(&positions, built.dumbbell[1]));
    let separation = dist(centroid(&positions, built.dumbbell), built.vacancy);
    let recombined = vacancy
        && (occupied(&positions, built.vacancy, cell)
            || !(1.2..2.5).contains(&gap)
            || separation < 1.6);
    (positions, built.vacancy, built.dumbbell, cell, gap, separation, recombined)
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
        500 => "4000: 309±18 and 287±34; 5324: 169±17 and 180±20; 6912: 188±11 and 191±19; 8788: 185±23 and 161±15; 16384: 202±13 and 206±22",
        600 => "4000: 218±13 and 238±22; 5324: 170±8 and 177±15; 6912: 198±10 and 203±20; 8788: 186±14 and 191±18; 16384: 183±9 and 194±18",
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
    let search = MinModeConfig {
        force_tol: 2.0e-2,
        max_move: 0.1,
        dr: 1.0e-3,
        max_rotations: 8,
        ..MinModeConfig::default()
    };
    rgsaddle::DefectPairConfig {
        dimer: SolidDimerConfig::frozen(search)
            .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1)),
        band: rgsaddle::BandConfig {
            force_tol: 0.01,
            max_move: 0.06,
            time_step: 0.08,
            method: Method::Fire { kind: FireKind::V2 },
            spring: rgsaddle::SpringKind::Uniform { k: 1.0 },
            climbing: Some(rgsaddle::CiConfig::always()),
            min_mode: None,
            autoneb: None,
            zoom: None,
            quickmin: false,
            ..rgsaddle::BandConfig::default()
        },
        images: 9,
        max_dimer_steps: 100,
        max_band_steps: 120,
        max_relax_steps: 50,
        relax_tol: 1.0e-3,
        relax_step: 0.08,
        max_move: 0.08,
        seed_fraction: 0.45,
        push: 0.3,
        connect_rmsd: 1.5,
        direction_tol: 0.08,
        pafi: rgsaddle::PafiConfig {
            thermalize: 40,
            stride: 16,
            samples,
            blocks: 6,
            time_step: 0.01,
            mobility: 0.2,
            seed,
            fix_com: true,
            retain: 4,
            target_error: Some(target),
            max_rounds: 4,
            ..rgsaddle::PafiConfig::default()
        },
        shooting: rgsaddle::ShootingConfig {
            friction: 2.0,
            time_step: 0.03,
            steps: 80,
            record_every: 20,
            shots: 32,
            blocks: 4,
            seed: seed.wrapping_add(9),
            plateau_record: 2,
            mass: 63.546,
            ..rgsaddle::ShootingConfig::default()
        },
        temperatures: temperatures.to_vec(),
        boltzmann: rgsaddle::BOLTZMANN_EV_PER_K,
        mass: 63.546,
        angular_scale: rgsaddle::angular_scale_rad_s(),
        harmonic_step: 1.0e-3,
        harmonic_atoms: Some(harmonic),
        planes: 11,
        zero_t_tol: 0.02,
        bulk_stress: bulk,
        ..rgsaddle::DefectPairConfig::default()
    }
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
    std::fs::write(root.join("defect_pair_cu.json"), campaign.catalogue.to_json()).unwrap();
    let mut shells = File::create(root.join("defect_pair_shells.csv")).expect("shells");
    writeln!(
        shells,
        "cell_repeats,temperature_K,radius_A,sites,recombined,gap_A,separation_A,miller"
    )
    .unwrap();
    for row in &campaign.shells {
        writeln!(
            shells,
            "{},{},{:.4},{},{},{:.4},{:.4},{} {} {}",
            row.repeats,
            row.temperature,
            row.radius,
            row.count,
            row.recombined as u8,
            row.gap,
            row.separation,
            row.key.0,
            row.key.1,
            row.key.2
        )
        .unwrap();
    }
    let mut lattice = File::create(root.join("defect_pair_lattice.csv")).expect("lattice");
    writeln!(lattice, "temperature_K,a_A,npt_repeats").unwrap();
    for (temperature, a) in &campaign.lattice {
        writeln!(lattice, "{temperature},{a:.6},6").unwrap();
    }
    std::fs::write(root.join("defect_pair_cu_notes.txt"), campaign.notes.join("\n")).unwrap();
}

fn rate_or_harmonic(entry: &rgsaddle::RateEntry) -> Option<f64> {
    if let Some(rate) = entry.rate.filter(|value| value.is_finite() && *value > 0.0) {
        return Some(rate);
    }
    let barrier = entry.d_f?;
    let prefactor = entry.nu.or(entry.vineyard)?;
    let thermal = entry.temperature * rgsaddle::BOLTZMANN_EV_PER_K;
    if !(thermal.is_finite() && thermal > 0.0 && prefactor.is_finite() && prefactor > 0.0) {
        return entry.harmonic_rate.filter(|value| value.is_finite() && *value > 0.0);
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
    let mut sizes: Vec<u32> = campaign.catalogue.entries.iter().map(|entry| entry.cell_repeats).collect();
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
    let mut viv = String::from("cell_repeats,temperature_K,spontaneous_sites,capture_volume_Omega,rate_volume_Omega,k_mig_Hz,table_1\n");
    for repeats in sizes {
        for temperature in &temps {
            let shells: Vec<&ShellRow> = campaign
                .shells
                .iter()
                .filter(|row| row.repeats == repeats as usize && row.temperature.round() as i64 == *temperature)
                .collect();
            if shells.is_empty() {
                continue;
            }
            let spontaneous: f64 = shells.iter().filter(|row| row.recombined).map(|row| row.count as f64).sum();
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
                let values: Vec<f64> = isolated.iter().filter_map(|entry| rate_or_harmonic(entry)).collect();
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
                let group: Vec<&&rgsaddle::RateEntry> = entries.iter().filter(|entry| entry.shell == shell).collect();
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
                    .filter(|entry| entry.direction.as_str() == "toward" && rate_or_harmonic(entry).is_some())
                    .count();
                if let Some(k_mig) = k_mig
                    && n_toward > 0
                    && count > 0.0
                {
                    for entry in group.iter().filter(|entry| entry.direction.as_str() == "toward") {
                        if let Some(rate) = rate_or_harmonic(entry) {
                            weighted += count / n_toward as f64 * rate / k_mig;
                        }
                    }
                }
            }
            let line = format!(
                "{repeats},{temperature},{spontaneous:.1},{capture:.1},{weighted:.1},{},{} (shells with rates {shells_used})",
                k_mig.map(|value| format!("{value:.6e}")).unwrap_or_default(),
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
        .map(|entry| (entry.cell_repeats, entry.shell.clone(), entry.temperature.round() as i64))
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
            let barrier = rows.iter().filter_map(|entry| entry.d_f).sum::<f64>() / rows.len() as f64;
            let error = rows
                .iter()
                .filter_map(|entry| entry.d_f_error)
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
                / rows.len() as f64;
            let static_barrier = rows.iter().map(|entry| entry.d_e0).sum::<f64>() / rows.len() as f64;
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
    for entry in entries.iter().filter(|entry| entry.shell == "isolated" && entry.d_f.is_some()) {
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
        catalogue: rgsaddle::RateCatalogue { entries: Vec::new() },
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
        let rate_this = rate_sizes.contains(&repeats) && std::env::var("RGSADDLE_SCAN_ONLY").is_err();
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
                let (positions, vacancy, dumbbell, cell, _gap, _separation, recombined) =
                    relax_pair(&copper, repeats, reference.1, vector, true);
                if recombined {
                    continue;
                }
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
            let mut positions = isolated.positions;
            let length = repeats as f64 * reference.1;
            let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
            relax_fast(&copper, &mut positions, cell, 1.0e-3, 60);
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
            let shells = shell_table(a, repeats);
            for (radius, vector, count) in &shells {
                let (_positions, _vacancy, _dumbbell, _cell, gap, separation, recombined) =
                    relax_pair(&copper, repeats, a, *vector, true);
                let row = ShellRow {
                    repeats,
                    temperature,
                    radius: *radius,
                    count: *count,
                    recombined,
                    gap,
                    separation,
                    key: miller(*vector, a),
                };
                if recombined {
                    campaign.notes.push(format!(
                        "repeats {repeats} T {temperature} shell {radius:.3} A recombined, {count} sites, gap {gap:.3} separation {separation:.3}"
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
    campaign.catalogue.entries.extend(store.into_inner().expect("rates"));
    campaign.notes.extend(asymmetry_lines(&campaign.catalogue.entries));
    campaign.notes.extend(viv_lines(&campaign));
    let evaluations: usize = campaign.catalogue.entries.iter().map(|entry| entry.pafi_evaluations).sum();
    campaign.notes.push(format!(
        "PAFI force evaluations {evaluations}. Wall time {:.1} min. Sizes requested for rates: {}. Sizes scanned: {}.",
        started.elapsed().as_secs_f64() / 60.0,
        rate_sizes.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(","),
        sizes.iter().map(|value| value.to_string()).collect::<Vec<_>>().join(",")
    ));
    write_campaign(&campaign);
    eprintln!("{}", campaign.notes.join("\n"));
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
    let (positions, vacancy, dumbbell, cell, _gap, _separation, recombined) =
        relax_pair(copper, job.repeats, job.a, vector, vacancy_flag);
    if recombined {
        return Ok(Vec::new());
    }
    let fixed = if vacancy_flag {
        vacancy
    } else {
        let center = centroid(&positions, dumbbell);
        [center[0] + 6.0, center[1], center[2]]
    };
    let mut hops = Vec::new();
    for (index, escape) in job.escapes.iter().enumerate() {
        let kind = if escape.axis_dot < 0.6 { "rotate" } else { "translate" };
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
        if moved < 0.04 * 0.04 && moved_s < 0.04 * 0.04 && atom != dumbbell[0] && atom != dumbbell[1] {
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
        kind: if shift > 0.6 { "translate".into() } else { "rotate".into() },
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
    let search = MinModeConfig {
        force_tol: 0.04,
        max_move: 0.1,
        dr: 1.0e-3,
        max_rotations: 8,
        ..MinModeConfig::default()
    };
    let dimer_config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1));
    let mut dimer = SolidDimer::new(dimer_config, seed, cell, mode).ok()?;
    let held = FastHeld { copper, cell };
    let report = dimer.run(&held, 80).ok()?;
    if report.status != MinModeStatus::Converged || report.rejected || report.negative_modes != Some(1) {
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
    let mut plus = &saddle + &step;
    let mut minus = &saddle - &step;
    relax_fast(copper, &mut plus, cell, 1.5e-3, 50);
    relax_fast(copper, &mut minus, cell, 1.5e-3, 50);
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
    if back > 1.0 || (back_energy - reactant_energy).abs() > 0.08 || product_energy < reactant_energy - 0.5 {
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
    relax_fast(copper, &mut reactant, cell, 1.0e-3, 60);
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
                mechanism.kind, mechanism.barrier, mechanism.atoms.len()
            );
            let _ = std::io::stderr().flush();
            found.push(mechanism);
        }
    }
    let mut kept = Vec::new();
    for kind in ["translate", "rotate"] {
        if let Some(best) = found.iter().filter(|item| item.kind == kind).min_by(|left, right| {
            left.barrier.partial_cmp(&right.barrier).unwrap()
        }) {
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
                if hops.iter().any(|prior: &rgsaddle::HopTemplate| hausdorff(&prior.product, &hop.product, cell) < 0.45) {
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
    let mut config = cu_config(temperatures, 16, 0.01, bulk, harmonic, 41);
    config.finite_temperature = finite;
    config.zero_t_tol = 0.01;
    config.pafi.max_rounds = 3;
    config.pafi.samples = 16;
    config.pafi.thermalize = 20;
    config.max_dimer_steps = 60;
    config.max_band_steps = 80;
    config.images = 7;
    config.connect_rmsd = 0.9;
    let setup = rgsaddle::PairSetup {
        positions: job.positions.clone(),
        cell: job.cell,
        fixed: job.fixed,
        mobile: job.mobile.clone(),
        shell: job.shell.clone(),
        cell_repeats: job.repeats as u32,
        hops: vec![job.hop.clone()],
    };
    rgsaddle::defect_pair_rates(&FastHeld { copper, cell: job.cell }, &setup, &config)
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
        fixed: [job.fixed[0] * factor, job.fixed[1] * factor, job.fixed[2] * factor],
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
    eprintln!("mechanisms {} in {:.0}s", mechanisms.len(), started.elapsed().as_secs_f64());
    let _ = std::io::stderr().flush();
    assert!(!mechanisms.is_empty(), "the isolated dumbbell produced no hop");
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
            let key = ((axis[0] * shell_hat[0] + axis[1] * shell_hat[1] + axis[2] * shell_hat[2]).abs() * 1000.0).round() as i32;
            let shell_key = miller(*vector, a500);
            let mark = (shell_key, key);
            if seen_orientation.contains(&mark) {
                continue;
            }
            seen_orientation.push(mark);
            let (mut positions, vacancy, dumbbell, cell, gap, separation, recombined) =
                relax_pair_axis(&copper, repeats, a500, *vector, axis, true);
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
            job.shell == entry.shell && job.orientation == entry.orientation && job.hop.name == entry.hop
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
            .flat_map(|job| run_science_job(&copper, &job, &[temperature], true).unwrap_or_default())
            .collect();
        thermal_rows.extend(batch);
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks");
    let mut csv = File::create(root.join("defect_pair_hops.csv")).expect("hops");
    writeln!(csv, "shell_A,orientation,hop,final_state,direction,temperature_K,cell_repeats,outcome,connected,index1,band_force_eV_per_A,dE0_eV,zero_T_eV,dF_eV,dF_error_eV,nu_Hz,vineyard_Hz,kappa,rate_Hz,separation_before_A,separation_after_A").unwrap();
    let mut catalogue = rgsaddle::RateCatalogue { entries: Vec::new() };
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
        mechanisms.iter().map(|item| format!("{} {:.4}", item.kind, item.barrier)).collect::<Vec<_>>().join(", "),
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
            let (_positions, _vacancy, _dumbbell, _cell, gap, separation, recombined) =
                relax_pair_axis(copper, *repeats, *a, *vector, *axis, true);
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

fn relax_pair_axis(
    copper: &FastCopper,
    repeats: usize,
    a: f64,
    shell: [f64; 3],
    axis: [f64; 3],
    vacancy: bool,
) -> (Array1<f64>, [f64; 3], [usize; 2], [[f64; 3]; 3], f64, f64, bool) {
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let built = build_oriented(repeats, a, shell, axis, vacancy);
    let mut positions = built.positions;
    relax_fast(copper, &mut positions, cell, 1.0e-3, 60);
    let gap = dist(point_of(&positions, built.dumbbell[0]), point_of(&positions, built.dumbbell[1]));
    let separation = dist(centroid(&positions, built.dumbbell), built.vacancy);
    let recombined = vacancy
        && (occupied(&positions, built.vacancy, cell)
            || !(1.2..2.5).contains(&gap)
            || separation < 1.6);
    (positions, built.vacancy, built.dumbbell, cell, gap, separation, recombined)
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
    pairs.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut atom_used = vec![false; n];
    let mut site_used = vec![false; sites.len()];
    let mut product = reactant.clone();
    let mut assigned = 0usize;
    for (distance, atom, site) in pairs {
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
    rgsaddle::BandConfig {
        force_tol: 0.02,
        max_move: 0.08,
        time_step: 0.08,
        method: Method::Fire { kind: FireKind::V2 },
        spring: rgsaddle::SpringKind::Uniform { k: 1.0 },
        climbing: climbing.then(rgsaddle::CiConfig::always),
        min_mode: None,
        autoneb: None,
        zoom: None,
        quickmin: false,
        ..rgsaddle::BandConfig::default()
    }
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
    let mut neb = rgsaddle::SolidNeb::frozen(capture_band(false), guess).map_err(|error| error.to_string())?;
    let mut force = f64::MAX;
    for _ in 0..50 {
        let report = neb.step(&held).map_err(|error| error.to_string())?;
        force = report.max_force;
        if force < 0.05 {
            break;
        }
    }
    let relaxed = neb.cartesian().map_err(|error| error.to_string())?;
    let mut neb = rgsaddle::SolidNeb::frozen(capture_band(true), relaxed).map_err(|error| error.to_string())?;
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
    let search = MinModeConfig {
        force_tol: 0.03,
        max_move: 0.08,
        dr: 1.0e-3,
        max_rotations: 6,
        ..MinModeConfig::default()
    };
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
    let (reactant, vacancy, dumbbell, cell, _gap, _separation, recombined) =
        relax_pair_axis(copper, repeats, a, vector, axis, true);
    if recombined {
        return Err("reactant recombined".into());
    }
    let center = centroid(&reactant, dumbbell);
    let hops = mapped_hops(&reactant, cell, center, axis, mechanisms);
    let held = FastHeld { copper, cell };
    let (reactant_energy, _) = held.eval(reactant.view()).map_err(|error| error.to_string())?;
    let mobile = [dumbbell[0], dumbbell[1]];
    let before = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
    let mut note = String::from("no toward channel");
    let mut chosen: Option<CaptureRow> = None;
    for hop in hops {
        let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile).unwrap_or(before);
        if after > before - 0.05 {
            continue;
        }
        let mut product = hop.product.clone();
        relax_fast(copper, &mut product, cell, 1.0e-3, 40);
        let (product_energy, _) = held.eval(product.view()).map_err(|error| error.to_string())?;
        let drop = reactant_energy - product_energy;
        let Some(hint) = hop.saddle.clone() else { continue };
        let Some(mode) = hop.mode.clone() else { continue };
        let Some((saddle_energy, index1)) = polish_index(copper, &hint, &mode, cell) else {
            note = format!("toward drop {drop:.3} eV; dimer did not stay on the seed");
            continue;
        };
        if !index1 {
            note = format!("toward drop {drop:.3} eV; saddle is not index-1");
            continue;
        }
        let d_e0 = saddle_energy - reactant_energy;
        let (band, band_force) = relax_neb(copper, &reactant, &hint, cell).unwrap_or((Array2::zeros((0, 0)), f64::MAX));
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
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_hops_static.csv");
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut pairs: Vec<(String, String)> = Vec::new();
    for line in text.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 7 || fields[6] != "rated" {
            continue;
        }
        let Ok(radius) = fields[0].parse::<f64>() else { continue };
        let key = format!("{radius:.3}");
        if pairs.iter().any(|(shell, orientation)| shell == &key && orientation == fields[1]) {
            continue;
        }
        pairs.push((key, fields[1].to_string()));
    }
    pairs.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
    pairs
        .into_iter()
        .filter_map(|(shell, orientation)| shell.parse::<f64>().ok().map(|radius| (radius, orientation)))
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
            && fields[9].parse::<f64>().ok().is_some_and(|error| error <= 0.01)
    })
}

fn rotation_job(copper: &FastCopper, repeats: usize, a: f64) -> SciJob {
    let (positions, vacancy, dumbbell, cell, _, _, _) =
        relax_pair_axis(copper, repeats, a, [a, 0.0, 0.0], [1.0, 0.0, 0.0], false);
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
    eprintln!("mechanisms {} in {:.0}s", mechanisms.len(), started.elapsed().as_secs_f64());
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
        let translate: Vec<Mechanism> = mechanisms.iter().filter(|item| item.kind == "translate").cloned().collect();
        let (positions, vacancy, dumbbell, cell, _, _, _) =
            relax_pair_axis(&copper, repeats, a500, [a500, 0.0, 0.0], [1.0, 0.0, 0.0], false);
        let center = centroid(&positions, dumbbell);
        if let Some(hop) = mapped_hops(&positions, cell, center, [1.0, 0.0, 0.0], &translate).into_iter().next() {
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
            let (positions, vacancy, dumbbell, cell, _, _, recombined) =
                relax_pair_axis(&copper, repeats, a500, vector, axis, true);
            if recombined {
                continue;
            }
            let center = centroid(&positions, dumbbell);
            for hop in mapped_hops(&positions, cell, center, axis, &mechanisms) {
                let mobile = [dumbbell[0], dumbbell[1]];
                let before = rgsaddle::pair_separation(positions.view(), cell, vacancy, &mobile).unwrap_or(0.0);
                let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile).unwrap_or(0.0);
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
                (row.shell.parse::<f64>().unwrap_or(0.0) - radius).abs() < 0.03 && row.orientation == *orientation
            });
            let prior = existing
                .lines()
                .find(|line| line.starts_with(&format!("capture,{radius:.3},{orientation},")));
            let recomputed = if saved.is_none() && !prior.is_some_and(|line| line.contains("barrierless")) {
                capture_one(&copper, repeats, a500, radius, axis, &mechanisms).ok()
            } else {
                None
            };
            let barrier = saved.map(|row| row.d_e0).or(recomputed.as_ref().map(|row| row.d_e0));
            if saved.or(recomputed.as_ref()).is_some_and(|row| {
                row.d_e0 >= BOLTZMANN_EV_PER_K * 500.0 && row.band_force <= 0.05
            }) {
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
    eprintln!("{} pafi jobs in {:.0}s", jobs.len(), started.elapsed().as_secs_f64());
    for temperature in [500.0, 300.0] {
        let a = measured_lattice(temperature);
        for job in &jobs {
            let scaled = scale_job(job, a);
            let fresh = std::fs::read_to_string(&path).unwrap_or_default();
            if pafi_stored(&fresh, &scaled.shell, &scaled.orientation, &scaled.hop.name, temperature) {
                eprintln!("skip pafi {} {} {} {temperature}", scaled.shell, scaled.orientation, scaled.hop.name);
                continue;
            }
            let entries = run_science_job(&copper, &scaled, &[temperature], true).unwrap_or_default();
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
    eprintln!("capture run finished in {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

fn drive_phase(
    copper: &FastCopper,
    cell: [[f64; 3]; 3],
    positions: Array2<f64>,
    climbing: bool,
    method: Method,
    steps: usize,
) -> Result<(Array2<f64>, f64), String> {
    let mut config = capture_band(climbing);
    config.force_tol = 0.01;
    config.time_step = if matches!(method, Method::Ode12r) { 0.05 } else { 0.08 };
    config.method = method;
    let held = FastHeld { copper, cell };
    let mut neb = rgsaddle::SolidNeb::frozen(config, positions).map_err(|error| error.to_string())?;
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
    let idpp = rgsaddle::IdppConfig {
        max_iterations: 40,
        grad_tol: 5.0e-2,
        max_move: 0.15,
        memory: 8,
    };
    let guess = rgsaddle::idpp_path(reactant.view(), product.view(), 17, Some(&wrapped), &idpp)
        .or_else(|_| rgsaddle::linear_path(reactant.view(), product.view(), 17, Some(&wrapped)))
        .map_err(|error| error.to_string())?;
    let (relaxed, relaxed_force) = drive_phase(
        copper,
        cell,
        guess,
        false,
        Method::Fire { kind: FireKind::V2 },
        70,
    )?;
    let (climbed, climbed_force) = drive_phase(
        copper,
        cell,
        relaxed.clone(),
        true,
        Method::Fire { kind: FireKind::V2 },
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
    drop: f64,
    rise: f64,
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
    let (reactant, vacancy, dumbbell, cell, _, _, recombined) =
        relax_pair_axis(copper, repeats, a, vector, axis, true);
    if recombined {
        return Err("reactant recombined on relaxation".into());
    }
    let mut product = crystal_image(&reactant, repeats, a);
    relax_fast(copper, &mut product, cell, 1.0e-3, 25);
    let held = FastHeld { copper, cell };
    let (reactant_energy, _) = held.eval(reactant.view()).map_err(|error| error.to_string())?;
    let (product_energy, _) = held.eval(product.view()).map_err(|error| error.to_string())?;
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
        format!("barrierless; rise {rise:.4} eV; integral {:.4} eV; drop {drop:.3} eV", zero.barrier)
    } else if band_force > 0.05 {
        format!("band not converged; rise {rise:.4} eV; integral {:.4} eV; drop {drop:.3} eV", zero.barrier)
    } else if index1 {
        format!("index-1; rise {rise:.4} eV; drop {drop:.3} eV", )
    } else {
        format!("neb maximum, index not confirmed; rise {rise:.4} eV; drop {drop:.3} eV")
    };
    Ok(NebCapture {
        d_e0,
        zero_t: zero.barrier,
        band_force,
        drop,
        rise,
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
    let mut config = cu_config(&[temperature], 24, 0.01, bulk, harmonic, 41);
    config.finite_temperature = true;
    config.zero_t_tol = 0.01;
    config.pafi.max_rounds = 4;
    config.pafi.samples = 24;
    config.pafi.thermalize = 20;
    config.max_dimer_steps = 70;
    config.max_band_steps = 220;
    config.band.force_tol = 8.0e-3;
    config.images = 9;
    config.connect_rmsd = 1.0;
    let setup = rgsaddle::PairSetup {
        positions: job.positions.clone(),
        cell: job.cell,
        fixed: job.fixed,
        mobile: job.mobile.clone(),
        shell: job.shell.clone(),
        cell_repeats: job.repeats as u32,
        hops: vec![job.hop.clone()],
    };
    rgsaddle::defect_pair_rates(&FastHeld { copper, cell: job.cell }, &setup, &config)
        .map(|found| found.entries)
        .map_err(|error| error.to_string())
}

fn finish_line(kind: &str, entry: &rgsaddle::RateEntry) -> String {
    format!(
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
        format!("kappa {}", num_or(&entry.kappa))
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
                let line = format!("capture,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,{error}");
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
                orientation: if (*radius - 3.639).abs() < 0.02 { "010".into() } else { "100".into() },
                final_state: "crystal".into(),
            },
            a: a500,
        });
    }
    for job in jobs.iter_mut() {
        let orientation = if (job.radius - 3.639).abs() < 0.02 { "010" } else { "100" };
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
        let (positions, vacancy, dumbbell, cell, _, _, _) =
            relax_pair_axis(&copper, repeats, a500, vector, axis, true);
        let center = centroid(&positions, dumbbell);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before = rgsaddle::pair_separation(positions.view(), cell, vacancy, &mobile).unwrap_or(0.0);
        let mut index = 0usize;
        for mut hop in mapped_hops(&positions, cell, center, axis, &mechanisms) {
            let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile).unwrap_or(before);
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
    eprintln!("{} finish jobs in {:.0}s", jobs.len(), started.elapsed().as_secs_f64());
    for temperature in [500.0_f64, 300.0] {
        let a = measured_lattice(temperature);
        for job in &jobs {
            if job.hop.name == "rotate" && temperature < 400.0 {
                continue;
            }
            if job.hop.name == "translate" && (job.radius - 3.639).abs() < 0.02 && temperature > 400.0 {
                continue;
            }
            let scaled = scale_job(job, a);
            let fresh = std::fs::read_to_string(&path).unwrap_or_default();
            let marker = format!("pafi,{},{},{},", scaled.shell, scaled.orientation, scaled.hop.name);
            if fresh.lines().any(|line| {
                let fields: Vec<&str> = line.split(',').collect();
                fields.len() > 12
                    && line.starts_with(&marker)
                    && fields[5].parse::<f64>().ok() == Some(temperature)
                    && fields[12] == "rated"
                    && fields[9].parse::<f64>().ok().is_some_and(|error| error <= 0.01)
            }) {
                eprintln!("skip {}", marker);
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
    eprintln!("finish run {:.1} min", started.elapsed().as_secs_f64() / 60.0);
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
) -> Vec<rgsaddle::DimerSeed> {
    let center = centroid(reactant, dumbbell);
    let length = cell[0][0];
    let mut rng = TinyRng(0x5eed);
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
    drop(push_atoms);
    for kick in rotations {
        seeds.push(rgsaddle::DimerSeed {
            positions: reactant + &kick,
            mode: kick,
        });
    }
    while seeds.len() < 32 {
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
    (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt().max(1.0e-8)
}

fn reaches_recombination(
    reactant: &Array1<f64>,
    reactant_energy: f64,
    vacancy: [f64; 3],
    cell: [[f64; 3]; 3],
    basin: &Array1<f64>,
    basin_energy: f64,
) -> bool {
    let drop = reactant_energy - basin_energy;
    if drop > 1.5 {
        return true;
    }
    let gap_gone = !basin_is_dumbbell(basin, vacancy, cell);
    drop > 0.4 && gap_gone && occupied(basin, vacancy, cell)
}

fn basin_is_dumbbell(positions: &Array1<f64>, vacancy: [f64; 3], cell: [[f64; 3]; 3]) -> bool {
    let n = positions.len() / 3;
    let mut close = 0usize;
    for atom in 0..n {
        if mic_distance(point_of(positions, atom), vacancy, cell) < 2.2 {
            close += 1;
        }
    }
    close == 0
}

fn survey_config(bulk: [[f64; 3]; 3]) -> rgsaddle::DefectPairConfig {
    let mut config = cu_config(&[500.0], 8, 0.01, bulk, Vec::new(), 7);
    config.finite_temperature = false;
    config.max_dimer_steps = 40;
    config.max_relax_steps = 70;
    config.relax_tol = 2.0e-3;
    config.push = 0.35;
    config.dimer = {
        let search = MinModeConfig {
            force_tol: 0.04,
            max_move: 0.1,
            dr: 1.0e-3,
            max_rotations: 6,
            ..MinModeConfig::default()
        };
        SolidDimerConfig::frozen(search)
            .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(0))
    };
    config
}

fn curvature_along(held: &FastHeld, positions: &Array1<f64>, tangent: &Array1<f64>) -> f64 {
    let scale = tangent.dot(tangent).sqrt().max(1.0e-12);
    let step = tangent / scale * 1.0e-3;
    let (e0, _) = held.eval(positions.view()).unwrap_or((0.0, Array1::zeros(1)));
    let (ep, _) = held.eval((positions + &step).view()).unwrap_or((e0, Array1::zeros(1)));
    let (em, _) = held.eval((positions - &step).view()).unwrap_or((e0, Array1::zeros(1)));
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
            append_checkpoint(&checkpoint, &format!("survey,{radius:.3},{orientation},capture,toward,500,,,,,,not_converged,0,no shell"));
            continue;
        };
        let (reactant, vacancy, dumbbell, cell, gap, separation, recombined) =
            relax_pair_axis(&copper, repeats, a500, vector, axis_from_label(orientation), true);
        if recombined {
            let line = format!(
                "survey,{radius:.3},{orientation},capture,toward,500,0.000000,0.000000,,,,,barrierless,0,recombines on relaxation; gap {gap:.3} separation {separation:.3}"
            );
            eprintln!("{line}");
            append_checkpoint(&checkpoint, &line);
            continue;
        }
        let held = FastHeld { copper: &copper, cell };
        let (reactant_energy, _) = held.eval(reactant.view()).expect("reactant");
        let config = survey_config(bulk);
        let seeds = local_seeds(&reactant, cell, dumbbell, vacancy);
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
            let plus_re = reaches_recombination(&reactant, reactant_energy, vacancy, cell, &saddle.plus, saddle.plus_energy);
            let minus_re = reaches_recombination(&reactant, reactant_energy, vacancy, cell, &saddle.minus, saddle.minus_energy);
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
        let sign = if best.plus_energy < best.minus_energy { 1.0 } else { -1.0 };
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
                let trial_energy = held.eval(trial.view()).map(|(energy, _)| energy).unwrap_or(f64::NAN);
                if trial_energy.is_finite() && trial_energy > best.saddle_energy - 0.3 {
                    start = trial;
                    break;
                }
                step *= 0.5;
            }
        }
        let floor = best.saddle_energy - 0.6;
        let (downhill, down_energy) = rgsaddle::descend_limited(&held, start, cell, &config, floor).expect("descend");
        let mut band_config = config.clone();
        band_config.band.force_tol = 2.0e-3;
        band_config.max_band_steps = 180;
        band_config.images = 13;
        band_config.zero_t_tol = 0.02;
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
            (rgsaddle::ArcPath::from_positions(&[reactant.clone(), best.saddle.clone()]).expect("path"), false, f64::MAX)
        });
        let zero = rgsaddle::zero_temperature_barrier(&held, &band, &band.arcs()).ok();
        let zero_v = zero.as_ref().map(|profile| profile.barrier).unwrap_or(f64::NAN);
        let integral_ok = zero.as_ref().is_some_and(|profile| (profile.barrier - d_e0).abs() < 0.05);
        let outcome = if converged && force <= 2.0e-3 && integral_ok && down_energy > reactant_energy - 1.5 {
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
            let held_t = FastHeld { copper: &copper, cell: scaled_cell };
            let thermal = temperature * rgsaddle::BOLTZMANN_EV_PER_K;
            let mut pafi = band_config.pafi.clone();
            pafi.thermal = thermal;
            pafi.samples = 20;
            pafi.max_rounds = 4;
            pafi.target_error = Some(0.01);
            pafi.thermalize = 20;
            let arcs = scaled.arcs();
            let Ok((profile, planes)) = rgsaddle::pafi_profile(&held_t, &scaled, &arcs, &pafi) else {
                append_checkpoint(&checkpoint, &format!("pafi,{radius:.3},{orientation},capture,toward,{temperature},,,,,,,not_converged,0,pafi failed"));
                continue;
            };
            let dividing = profile.dividing;
            let geometry = scaled.geometry(profile.arc[dividing]).expect("geometry");
            let mut shooting = band_config.shooting.clone();
            shooting.thermal = thermal;
            shooting.mass = 63.546;
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
            let kappa = transmission.as_ref().map(|value| value.kappa).unwrap_or(f64::NAN);
            let reactant_t = &reactant * factor;
            let lambda = curvature_along(&held_t, &reactant_t, &geometry.tangent);
            let nu = rgsaddle::attempt_frequency(lambda, 63.546, rgsaddle::angular_scale_rad_s()).ok();
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
    let translate: Vec<Mechanism> = mechanisms.into_iter().filter(|item| item.kind == "translate").collect();
    if !translate.is_empty() {
        let vector = shell_table(a500, repeats)
            .into_iter()
            .find(|(value, _, _)| (*value - 3.639).abs() < 0.04)
            .map(|(_, vector, _)| vector)
            .expect("3.639");
        let a300 = measured_lattice(300.0);
        let factor = a300 / a500;
        let vector = [vector[0] * factor, vector[1] * factor, vector[2] * factor];
        let (positions, vacancy, dumbbell, cell300, _, _, _) =
            relax_pair_axis(&copper, repeats, a300, vector, axis_from_label("010"), true);
        let center = centroid(&positions, dumbbell);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before = rgsaddle::pair_separation(positions.view(), cell300, vacancy, &mobile).unwrap_or(0.0);
        for mut hop in mapped_hops(&positions, cell300, center, [0.0, 1.0, 0.0], &translate) {
            let after = rgsaddle::pair_separation(hop.product.view(), cell300, vacancy, &mobile).unwrap_or(before);
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
    eprintln!("local capture {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

/// Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys.
/// 136, 074103 (2012), eqs. (12)–(20): the band coordinate between images
/// is {J ε̄, ΔR} with J = V̄^{1/3} N^{1/6} fixed from the mean endpoint
/// volume, ε̄ the symmetrized right strain, and ΔR the minimum-image
/// displacement in the average cell. The cell force is −(V/J)(σ + P I).
/// `BandConfig::solid` is that band. P = 0 leaves the enthalpy as the
/// energy. A frozen cell is the ordinary CI-NEB (`solid` unset).
struct CrossJob {
    kind: &'static str,
    shell: String,
    orientation: String,
    direction: String,
    lattice: f64,
    lattice_note: String,
    reactant: Array1<f64>,
    product: Array1<f64>,
    saddle: Array1<f64>,
    mode: Array1<f64>,
    cell: [[f64; 3]; 3],
    csv: Vec<f64>,
    relax_product: bool,
    energy_weighted: bool,
}

struct EamBand<'a> {
    copper: &'a FastCopper,
    fixed: Option<[[f64; 3]; 3]>,
}

impl BandSurface for EamBand<'_> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let cell = self.fixed.ok_or_else(|| {
            SaddleError::Invalid("fixed-cell band has no cell".into())
        })?;
        let rows: Vec<Array1<f64>> = positions.outer_iter().map(|row| row.to_owned()).collect();
        let evals: Vec<_> = rows
            .par_iter()
            .map(|row| self.copper.evaluate(row.view(), cell))
            .collect();
        for (i, (energy, gradient, _)) in evals.into_iter().enumerate() {
            energies[i] = energy;
            gradients.row_mut(i).assign(&gradient);
        }
        Ok(())
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let rows: Vec<Array1<f64>> = positions.outer_iter().map(|row| row.to_owned()).collect();
        let held: Vec<[[f64; 3]; 3]> = cells.iter().map(|cell| cell.0).collect();
        let evals: Vec<_> = rows
            .par_iter()
            .zip(held.par_iter())
            .map(|(row, cell)| self.copper.evaluate(row.view(), *cell))
            .collect();
        for (i, (energy, gradient, stress)) in evals.into_iter().enumerate() {
            energies[i] = energy;
            gradients.row_mut(i).assign(&gradient);
            for a in 0..3 {
                for b in 0..3 {
                    stresses[(i, 3 * a + b)] = stress[a][b];
                }
            }
        }
        Ok(())
    }
}

fn cross_num(value: f64, digits: usize) -> String {
    if value.is_finite() {
        format!("{:.*}", digits, value)
    } else {
        String::new()
    }
}

fn csv_quote(text: &str) -> String {
    if text.contains(',') || text.contains('"') {
        format!("\"{}\"", text.replace('"', "'"))
    } else {
        text.to_string()
    }
}

fn det3(m: [[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn inv3(m: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(m);
    if d.abs() < 1e-18 {
        return None;
    }
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    c[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
    c[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    c[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    c[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    c[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    c[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    c[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
    c[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    Some(c)
}

fn frac_of(point: [f64; 3], inv: [[f64; 3]; 3]) -> [f64; 3] {
    let mut s = [0.0; 3];
    for c in 0..3 {
        s[c] = point[0] * inv[0][c] + point[1] * inv[1][c] + point[2] * inv[2][c];
    }
    s
}

fn cart_from_frac(frac: [f64; 3], cell: [[f64; 3]; 3]) -> [f64; 3] {
    let mut r = [0.0; 3];
    for c in 0..3 {
        r[c] = frac[0] * cell[0][c] + frac[1] * cell[1][c] + frac[2] * cell[2][c];
    }
    r
}

fn blend_cell(left: [[f64; 3]; 3], right: [[f64; 3]; 3], t: f64) -> [[f64; 3]; 3] {
    let mut cell = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            cell[i][j] = (1.0 - t) * left[i][j] + t * right[i][j];
        }
    }
    cell
}

fn blend_positions(
    left: &Array1<f64>,
    left_cell: [[f64; 3]; 3],
    right: &Array1<f64>,
    right_cell: [[f64; 3]; 3],
    cell: [[f64; 3]; 3],
    t: f64,
) -> Array1<f64> {
    let inv_left = inv3(left_cell).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let inv_right = inv3(right_cell).unwrap_or(inv_left);
    let n = left.len() / 3;
    let mut out = Array1::zeros(left.len());
    for atom in 0..n {
        let s = frac_of(point_of(left, atom), inv_left);
        let u = frac_of(point_of(right, atom), inv_right);
        let mut frac = [0.0; 3];
        for c in 0..3 {
            let mut delta = u[c] - s[c];
            delta -= delta.round();
            frac[c] = s[c] + t * delta;
        }
        let r = cart_from_frac(frac, cell);
        for c in 0..3 {
            out[3 * atom + c] = r[c];
        }
    }
    out
}

fn into_cell(positions: &Array1<f64>, from: [[f64; 3]; 3], into: [[f64; 3]; 3]) -> Array1<f64> {
    blend_positions(positions, from, positions, from, into, 0.0)
}

fn geometry_gap(reference: &Array1<f64>, other: &Array1<f64>, cell: [[f64; 3]; 3]) -> (f64, f64) {
    let n = reference.len() / 3;
    let mut mean = [0.0; 3];
    let mut delta = vec![[0.0; 3]; n];
    for atom in 0..n {
        for c in 0..3 {
            let length = cell[c][c].abs().max(1.0);
            let mut d = other[3 * atom + c] - reference[3 * atom + c];
            d -= length * (d / length).round();
            delta[atom][c] = d;
            mean[c] += d;
        }
    }
    for c in 0..3 {
        mean[c] /= n.max(1) as f64;
    }
    let mut acc = 0.0_f64;
    let mut max = 0.0_f64;
    for atom in 0..n {
        let mut norm = 0.0;
        for c in 0..3 {
            let d = delta[atom][c] - mean[c];
            norm += d * d;
        }
        acc += norm;
        max = max.max(norm.sqrt());
    }
    (((acc / n.max(1) as f64).sqrt()), max)
}

fn cell_rms(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> f64 {
    let mut acc = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = left[i][j] - right[i][j];
            acc += d * d;
        }
    }
    (acc / 9.0).sqrt()
}

fn cube_a(cell: [[f64; 3]; 3]) -> f64 {
    det3(cell).abs().cbrt() / 8.0
}

fn neb_config(
    spring: SpringKind,
    climbing: Option<CiConfig>,
    tol: f64,
    solid: Option<SolidState>,
) -> BandConfig {
    BandConfig {
        spring,
        climbing,
        force_tol: tol,
        max_move: 0.06,
        time_step: 0.08,
        method: Method::Lbfgs { memory: 10 },
        min_mode: None,
        autoneb: None,
        zoom: None,
        quickmin: false,
        remove_translation: true,
        solid,
        ..BandConfig::default()
    }
}

fn band_images(
    reactant: &Array1<f64>,
    reactant_cell: [[f64; 3]; 3],
    saddle: &Array1<f64>,
    saddle_cell: [[f64; 3]; 3],
    product: &Array1<f64>,
    product_cell: [[f64; 3]; 3],
) -> (Array2<f64>, Vec<Cell>) {
    let n = 7usize;
    let mid = 3usize;
    let mut positions = Array2::zeros((n, reactant.len()));
    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        let (row, cell) = if i <= mid {
            let t = i as f64 / mid as f64;
            let cell = blend_cell(reactant_cell, saddle_cell, t);
            let row = blend_positions(reactant, reactant_cell, saddle, saddle_cell, cell, t);
            (row, cell)
        } else {
            let t = (i - mid) as f64 / (n - 1 - mid) as f64;
            let cell = blend_cell(saddle_cell, product_cell, t);
            let row = blend_positions(saddle, saddle_cell, product, product_cell, cell, t);
            (row, cell)
        };
        positions.row_mut(i).assign(&row);
        cells.push(Cell(cell));
    }
    (positions, cells)
}

struct BandEnd {
    positions: Array2<f64>,
    cells: Vec<[[f64; 3]; 3]>,
    force: f64,
    steps: usize,
    converged: bool,
}

fn drive_band(
    copper: &FastCopper,
    start: Array2<f64>,
    start_cells: Vec<Cell>,
    fixed: Option<[[f64; 3]; 3]>,
    spring: SpringKind,
) -> Result<BandEnd, String> {
    let surface = EamBand { copper, fixed };
    let climb_limit = env_usize("RGSADDLE_SSNEB_CLIMB", 220);
    let mut positions = start;
    let mut cells = start_cells;
    let mut steps = 0usize;
    let mut best_force = f64::MAX;
    let mut best_positions = positions.clone();
    let mut best_cells: Vec<[[f64; 3]; 3]> = cells.iter().map(|cell| cell.0).collect();
    for (phase, (climbing, tol, limit)) in [
        (0u8, (None, 0.2_f64, 40usize)),
        (1, (Some(CiConfig::after(0.1)), 1.5e-3, climb_limit)),
    ] {
        let solid = if fixed.is_none() {
            Some(SolidState::new(cells.clone()).with_pressure(0.0))
        } else {
            None
        };
        let mut session = BandSession::new(neb_config(spring.clone(), climbing, tol, solid), positions.clone())
            .map_err(|error| error.to_string())?;
        for _ in 0..limit {
            let report = session.step(&surface).map_err(|error| error.to_string())?;
            steps += 1;
            let improved = phase == 1 && report.max_force < best_force;
            if improved || phase == 0 {
                positions = session.positions().to_owned();
                let fallback = fixed.unwrap_or_else(|| cells.first().map(|cell| cell.0).unwrap_or([[0.0; 3]; 3]));
                let n_images = positions.nrows();
                cells = session
                    .image_cells()
                    .map(|image| image.to_vec())
                    .unwrap_or_else(|| vec![Cell(fallback); n_images]);
            }
            if improved {
                best_force = report.max_force;
                best_positions = positions.clone();
                best_cells = cells.iter().map(|cell| cell.0).collect();
            }
            if report.status != BandStatus::Running || report.max_force < tol {
                if phase == 1 && report.max_force < best_force {
                    best_force = report.max_force;
                    best_positions = session.positions().to_owned();
                    best_cells = session
                        .image_cells()
                        .map(|image| image.iter().map(|cell| cell.0).collect())
                        .unwrap_or_else(|| best_cells.clone());
                }
                break;
            }
        }
    }
    if !best_force.is_finite() {
        return Err("band force was not finite".into());
    }
    Ok(BandEnd {
        positions: best_positions,
        cells: best_cells,
        force: best_force,
        steps,
        converged: best_force < 1.5e-3,
    })
}

fn energies_of(copper: &FastCopper, positions: &Array2<f64>, cells: &[[[f64; 3]; 3]]) -> Array1<f64> {
    let mut energies = Array1::zeros(positions.nrows());
    for (i, cell) in cells.iter().enumerate() {
        energies[i] = copper.evaluate(positions.row(i), *cell).0;
    }
    energies
}

fn highest_interior(energies: &Array1<f64>) -> usize {
    let mut best = 1usize;
    for i in 2..energies.len().saturating_sub(1) {
        if energies[i] > energies[best] {
            best = i;
        }
    }
    best
}

fn unit_mode(mode: &Array1<f64>, n: usize) -> Array1<f64> {
    let mut out = if mode.len() == n {
        mode.clone()
    } else {
        Array1::zeros(n)
    };
    let norm = out.dot(&out).sqrt();
    if norm < 1e-8 {
        out = Array1::zeros(n);
        out[0] = 1.0;
    } else {
        out /= norm;
    }
    out
}

struct TightSaddle {
    positions: Array1<f64>,
    mode: Array1<f64>,
    energy: f64,
    force: f64,
    index: i64,
    eigenvalue: f64,
    converged: bool,
    note: String,
}

fn tighten_dimer(
    copper: &FastCopper,
    cell: [[f64; 3]; 3],
    guess: &Array1<f64>,
    mode: &Array1<f64>,
) -> TightSaddle {
    let search = MinModeConfig {
        force_tol: 2.0e-3,
        max_move: 0.08,
        dr: 1.0e-3,
        max_rotations: 8,
        ..MinModeConfig::default()
    };
    let config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(1));
    let mode = unit_mode(mode, guess.len());
    let held = FastHeld { copper, cell };
    let built = SolidDimer::new(config, guess.clone(), cell, mode);
    let Ok(mut dimer) = built else {
        let (energy, _) = held.eval(guess.view()).unwrap_or((f64::NAN, Array1::zeros(1)));
        return TightSaddle {
            positions: guess.clone(),
            mode: unit_mode(&Array1::zeros(0), guess.len()),
            energy,
            force: f64::NAN,
            index: -1,
            eigenvalue: f64::NAN,
            converged: false,
            note: "dimer setup failed".into(),
        };
    };
    let report = dimer.run(&held, env_usize("RGSADDLE_SSNEB_DIMER", 100));
    let positions = dimer.positions().to_owned();
    let mode = dimer.mode().to_owned();
    let (energy, gradient) = held.eval(positions.view()).unwrap_or((f64::NAN, Array1::zeros(1)));
    match report {
        Ok(report) => {
            let index = report.negative_modes.map(|count| count as i64).unwrap_or(-1);
            let eigenvalue = report.eigenvalues.first().copied().unwrap_or(report.curvature);
            let converged = report.status == MinModeStatus::Converged && !report.rejected && index == 1;
            let note = if converged {
                String::new()
            } else {
                format!("dimer {:?} rejected {} index {index}", report.status, report.rejected)
            };
            TightSaddle {
                positions,
                mode,
                energy,
                force: report.max_force.min(max_abs(&gradient)),
                index,
                eigenvalue,
                converged,
                note,
            }
        }
        Err(error) => TightSaddle {
            positions,
            mode,
            energy,
            force: max_abs(&gradient),
            index: -1,
            eigenvalue: f64::NAN,
            converged: false,
            note: format!("dimer {error}"),
        },
    }
}

fn projected_index(
    copper: &FastCopper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    variable: bool,
) -> (i64, f64) {
    let search = MinModeConfig {
        force_tol: 1.0e-2,
        max_move: 0.1,
        dr: 1.0e-3,
        max_rotations: 4,
        ..MinModeConfig::default()
    };
    let check = IndexCheck::first_order().with_step(1.0e-3).with_escapes(0);
    let config = if variable {
        SolidDimerConfig::new(search).with_index_check(check)
    } else {
        SolidDimerConfig::frozen(search).with_index_check(check)
    };
    let mode = unit_mode(&Array1::zeros(0), positions.len());
    let held = FastHeld { copper, cell };
    let Ok(dimer) = SolidDimer::new(config, positions.clone(), cell, mode) else {
        return (-1, f64::NAN);
    };
    match dimer.index(&held) {
        Ok(report) => (report.negative as i64, report.values.first().copied().unwrap_or(f64::NAN)),
        Err(_) => (-1, f64::NAN),
    }
}

fn joint_value(chart: &JointChart, copper: &FastCopper, q: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let Ok((positions, cell)) = chart.geometry(q) else {
        return (1.0e6, Array1::zeros(q.len()));
    };
    let (energy, gradient, stress) = copper.evaluate(positions.view(), cell);
    match chart.gradient(q, gradient.view(), stress) {
        Ok(joint) => (chart.enthalpy(energy, q).unwrap_or(energy), joint),
        Err(_) => (1.0e6, Array1::zeros(q.len())),
    }
}

fn relax_joint(
    copper: &FastCopper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Result<(Array1<f64>, [[f64; 3]; 3], f64, f64), String> {
    let n = positions.len() / 3;
    let chart = JointChart::new(cell, n, CellMask::all(), [[0.0; 3]; 3], 1.0)
        .map_err(|error| error.to_string())?;
    let mut q = joint_coordinate(&chart, CellMask::all(), positions.view(), cell)
        .map_err(|error| error.to_string())?;
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            istep: 0.08,
            ..Control::default()
        },
        q.len(),
    );
    solver.set_atom_maxmove(0.08);
    let mut force = f64::MAX;
    for _ in 0..80 {
        let oracle = Oracle::unbounded(q.len(), |x: ArrayView1<f64>| joint_value(&chart, copper, x));
        solver.step(&oracle, &mut q).map_err(|error| error.to_string())?;
        force = max_abs(&joint_value(&chart, copper, q.view()).1);
        if force < 2.0e-3 {
            break;
        }
    }
    let (relaxed, relaxed_cell) = chart.geometry(q.view()).map_err(|error| error.to_string())?;
    let energy = copper.evaluate(relaxed.view(), relaxed_cell).0;
    Ok((relaxed, relaxed_cell, energy, force))
}

fn judge(converged: bool, index: i64, rmsd: f64, delta: f64) -> String {
    let mut reasons = Vec::new();
    if !converged {
        reasons.push("not converged");
    }
    if index != 1 {
        reasons.push("not index 1");
    }
    if !rmsd.is_finite() || rmsd > 0.05 {
        reasons.push("geometry");
    }
    if !delta.is_finite() || delta.abs() > 0.002 {
        reasons.push("barrier");
    }
    if reasons.is_empty() {
        "recovered".into()
    } else {
        reasons.join("; ")
    }
}

fn nearest_csv(barriers: &[f64], value: f64) -> f64 {
    barriers
        .iter()
        .copied()
        .filter(|barrier| barrier.is_finite())
        .min_by(|left, right| {
            (left - value)
                .abs()
                .partial_cmp(&(right - value).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(f64::NAN)
}

fn cross_row(copper: &FastCopper, job: &CrossJob) -> String {
    let mut notes = Vec::new();
    if job.direction == "recombined" {
        notes.push("pair recombines on relaxation; no saddle to cross-check".into());
        return format!(
            "shell,{shell},{orientation},recombined,{lattice:.6},{note},{n},,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,{tail}",
            shell = job.shell,
            orientation = job.orientation,
            lattice = job.lattice,
            note = csv_quote(&job.lattice_note),
            n = job.reactant.len() / 3,
            tail = csv_quote(&notes.join(" | ")),
        );
    }
    let reactant = {
        let mut positions = job.reactant.clone();
        let force = max_abs(&copper.evaluate(positions.view(), job.cell).1);
        if force > 2.0e-3 {
            relax_fast(copper, &mut positions, job.cell, 1.0e-3, 80);
        }
        positions
    };
    let reactant_energy = copper.evaluate(reactant.view(), job.cell).0;
    let product = if job.relax_product {
        let mut positions = job.product.clone();
        relax_fast(copper, &mut positions, job.cell, 1.0e-3, 80);
        positions
    } else {
        job.product.clone()
    };
    let tight = tighten_dimer(copper, job.cell, &job.saddle, &job.mode);
    if !tight.note.is_empty() {
        notes.push(tight.note.clone());
    }
    let dimer_barrier = tight.energy - reactant_energy;
    let csv = nearest_csv(&job.csv, dimer_barrier);
    let (fixed_positions, fixed_cells) =
        band_images(&reactant, job.cell, &tight.positions, job.cell, &product, job.cell);
    let spring = if job.energy_weighted {
        SpringKind::EnergyWeighted { k_min: 0.1, k_max: 5.0 }
    } else {
        SpringKind::Uniform { k: 1.0 }
    };
    let fixed = drive_band(copper, fixed_positions, fixed_cells, Some(job.cell), spring.clone());
    let (free_reactant, free_reactant_cell, free_reactant_energy, free_product, free_product_cell) =
        match (
            relax_joint(copper, &reactant, job.cell),
            relax_joint(copper, &product, job.cell),
        ) {
            (Ok(left), Ok(right)) => {
                if left.3 > 0.05 || right.3 > 0.05 {
                    notes.push(format!(
                        "cell relax force reactant {:.3} product {:.3}",
                        left.3, right.3
                    ));
                }
                (left.0, left.1, left.2, right.0, right.1)
            }
            (left, right) => {
                notes.push(format!(
                    "cell relax failed: {} / {}",
                    left.as_ref().err().map(|e| e.as_str()).unwrap_or("ok"),
                    right.as_ref().err().map(|e| e.as_str()).unwrap_or("ok")
                ));
                (reactant.clone(), job.cell, reactant_energy, product.clone(), job.cell)
            }
        };
    let (free_positions, free_cells) = band_images(
        &free_reactant,
        free_reactant_cell,
        &tight.positions,
        job.cell,
        &free_product,
        free_product_cell,
    );
    let free = drive_band(copper, free_positions, free_cells, None, spring);
    let mut fixed_barrier = f64::NAN;
    let mut fixed_force = f64::NAN;
    let mut fixed_steps = 0usize;
    let mut fixed_converged = false;
    let mut fixed_index = -1i64;
    let mut fixed_eigen = f64::NAN;
    let mut fixed_rmsd = f64::NAN;
    let mut fixed_max = f64::NAN;
    if let Ok(end) = &fixed {
        let energies = energies_of(copper, &end.positions, &end.cells);
        let ci = highest_interior(&energies);
        fixed_barrier = energies[ci] - energies[0];
        fixed_force = end.force;
        fixed_steps = end.steps;
        fixed_converged = end.converged;
        let (index, eigen) = projected_index(copper, &end.positions.row(ci).to_owned(), end.cells[ci], false);
        fixed_index = index;
        fixed_eigen = eigen;
        let (rmsd, max) = geometry_gap(&tight.positions, &end.positions.row(ci).to_owned(), job.cell);
        fixed_rmsd = rmsd;
        fixed_max = max;
    } else if let Err(error) = &fixed {
        notes.push(format!("fixed band {error}"));
    }
    let mut free_barrier = f64::NAN;
    let mut free_force = f64::NAN;
    let mut free_steps = 0usize;
    let mut free_converged = false;
    let mut free_index = -1i64;
    let mut free_eigen = f64::NAN;
    let mut free_rmsd = f64::NAN;
    let mut free_max = f64::NAN;
    let mut free_cell = f64::NAN;
    let mut free_a = f64::NAN;
    if let Ok(end) = &free {
        let energies = energies_of(copper, &end.positions, &end.cells);
        let ci = highest_interior(&energies);
        free_barrier = energies[ci] - energies[0];
        free_force = end.force;
        free_steps = end.steps;
        free_converged = end.converged;
        let mapped = into_cell(&end.positions.row(ci).to_owned(), end.cells[ci], job.cell);
        let (rmsd, max) = geometry_gap(&tight.positions, &mapped, job.cell);
        free_rmsd = rmsd;
        free_max = max;
        free_cell = cell_rms(end.cells[ci], job.cell);
        free_a = cube_a(end.cells[ci]);
        let (index, eigen) = projected_index(copper, &end.positions.row(ci).to_owned(), end.cells[ci], true);
        free_index = index;
        free_eigen = eigen;
    } else if let Err(error) = &free {
        notes.push(format!("free band {error}"));
    }
    let fixed_delta = fixed_barrier - dimer_barrier;
    let free_delta = free_barrier - dimer_barrier;
    let fixed_result = judge(fixed_converged, fixed_index, fixed_rmsd, fixed_delta);
    let free_result = judge(free_converged, free_index, free_rmsd, free_delta);
    if (cube_a(free_reactant_cell) - job.lattice).abs() > 0.002 {
        notes.push(format!(
            "zero-stress reactant lattice {:.5} A",
            cube_a(free_reactant_cell)
        ));
    }
    let _ = free_reactant_energy;
    format!(
        "{kind},{shell},{orientation},{direction},{lattice:.6},{note},{n},\
{dimer:.6},{dimer_force:.6e},{dimer_index},{dimer_eigen:.6e},{dimer_ok},\
{csv},{dimer_csv:.6},\
{fixed_b:.6},{fixed_force:.6e},{fixed_steps},{fixed_ok},{fixed_index},{fixed_eigen:.6e},{fixed_rmsd:.6},{fixed_max:.6},{fixed_delta:.6},{fixed_result},\
{free_b:.6},{free_force:.6e},{free_steps},{free_ok},{free_index},{free_eigen:.6e},{free_rmsd:.6},{free_max:.6},{free_cell:.6},{free_a:.6},{free_delta:.6},{free_fixed:.6},{free_result},\
{relax:.6},{tail}",
        kind = job.kind,
        shell = job.shell,
        orientation = job.orientation,
        direction = job.direction,
        lattice = job.lattice,
        note = csv_quote(&job.lattice_note),
        n = reactant.len() / 3,
        dimer = dimer_barrier,
        dimer_force = tight.force,
        dimer_index = tight.index,
        dimer_eigen = tight.eigenvalue,
        dimer_ok = tight.converged as u8,
        csv = cross_num(csv, 6),
        dimer_csv = if csv.is_finite() { dimer_barrier - csv } else { f64::NAN },
        fixed_b = fixed_barrier,
        fixed_ok = fixed_converged as u8,
        fixed_result = csv_quote(&fixed_result),
        free_b = free_barrier,
        free_ok = free_converged as u8,
        free_fixed = free_barrier - fixed_barrier,
        free_result = csv_quote(&free_result),
        relax = reactant_energy - free_reactant_energy,
        tail = csv_quote(&notes.join(" | ")),
    )
}

fn wanted(kind: &str) -> bool {
    match std::env::var("RGSADDLE_SSNEB_ONLY") {
        Ok(value) if !value.is_empty() && value != "all" => {
            value.split(',').any(|item| item.trim() == kind)
        }
        _ => true,
    }
}

fn shell_wanted(shell: &str) -> bool {
    match std::env::var("RGSADDLE_SSNEB_SHELLS") {
        Ok(value) if !value.is_empty() && value != "all" => {
            value.split(',').any(|item| (item.trim().parse::<f64>().ok().unwrap_or(-1.0) - shell.parse::<f64>().unwrap_or(-2.0)).abs() < 0.02)
        }
        _ => true,
    }
}

struct Channel {
    kind: String,
    reactant: Array1<f64>,
    saddle: Array1<f64>,
    product: Array1<f64>,
    mode: Array1<f64>,
    cell: [[f64; 3]; 3],
    barrier: f64,
    dumbbell: [usize; 2],
}

fn isolate_channels(copper: &FastCopper, repeats: usize, a: f64) -> Vec<Channel> {
    let built = build_oriented(repeats, a, [a, 0.0, 0.0], [1.0, 0.0, 0.0], false);
    let length = repeats as f64 * a;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut reactant = built.positions;
    relax_fast(copper, &mut reactant, cell, 1.0e-3, 80);
    let center = centroid(&reactant, built.dumbbell);
    let held = FastHeld { copper, cell };
    let (reactant_energy, _) = held.eval(reactant.view()).unwrap_or((0.0, Array1::zeros(1)));
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
    let axis = axis_of(&reactant, built.dumbbell);
    let half = dist(point_of(&reactant, built.dumbbell[0]), center).max(0.8);
    for plane in [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        let target = unit3([
            axis[0] + plane[0],
            axis[1] + plane[1],
            axis[2] + plane[2],
        ]);
        let mut turned = Array1::zeros(reactant.len());
        for (atom, sign) in built.dumbbell.into_iter().zip([-1.0, 1.0]) {
            let here = point_of(&reactant, atom);
            for c in 0..3 {
                turned[3 * atom + c] = center[c] + sign * half * target[c] - here[c];
            }
        }
        kicks.push(turned);
    }
    let mut found = Vec::new();
    for kick in kicks {
        let mode_norm = kick.dot(&kick).sqrt();
        if mode_norm < 1.0e-6 {
            continue;
        }
        let mode = kick.clone() / mode_norm;
        let rotation_seed = mode_norm > 0.9;
        let search = MinModeConfig {
            force_tol: if rotation_seed { 5.0e-3 } else { 0.04 },
            max_move: 0.1,
            dr: 1.0e-3,
            max_rotations: 8,
            ..MinModeConfig::default()
        };
        let config = SolidDimerConfig::frozen(search).with_index_check(
            IndexCheck::first_order()
                .with_step(1.0e-3)
                .with_escapes(if rotation_seed { 4 } else { 1 }),
        );
        let Ok(mut dimer) = SolidDimer::new(config, &reactant + &kick, cell, mode) else {
            continue;
        };
        let Ok(report) = dimer.run(&held, if rotation_seed { 250 } else { 80 }) else {
            continue;
        };
        if report.status != MinModeStatus::Converged || report.rejected || report.negative_modes != Some(1) {
            continue;
        }
        let saddle = dimer.positions().to_owned();
        let unstable = dimer.mode().to_owned();
        let mut peak = 0.0_f64;
        for atom in 0..n {
            let mut acc = 0.0;
            for c in 0..3 {
                acc += unstable[3 * atom + c] * unstable[3 * atom + c];
            }
            peak = peak.max(acc.sqrt());
        }
        if peak < 1.0e-8 {
            continue;
        }
        let step = &unstable * (0.45 / peak);
        let mut plus = &saddle + &step;
        let mut minus = &saddle - &step;
        relax_fast(copper, &mut plus, cell, 1.5e-3, 50);
        relax_fast(copper, &mut minus, cell, 1.5e-3, 50);
        let gap_plus = hausdorff(&plus, &reactant, cell);
        let gap_minus = hausdorff(&minus, &reactant, cell);
        let (product, back, back_pos) = if gap_plus < gap_minus {
            (minus, gap_plus, plus)
        } else {
            (plus, gap_minus, minus)
        };
        let Ok((back_energy, _)) = held.eval(back_pos.view()) else {
            continue;
        };
        let Ok((product_energy, _)) = held.eval(product.view()) else {
            continue;
        };
        let Ok((saddle_energy, _)) = held.eval(saddle.view()) else {
            continue;
        };
        if back > 1.0 || (back_energy - reactant_energy).abs() > 0.08 || product_energy < reactant_energy - 0.5 {
            continue;
        }
        let shift = dist(centroid(&product, built.dumbbell), center);
        let product_axis = axis_of(&product, built.dumbbell);
        let overlap = (product_axis[0] * axis[0] + product_axis[1] * axis[1] + product_axis[2] * axis[2]).abs();
        let kind = if shift > 0.6 {
            "translate"
        } else if overlap < 0.7 {
            "rotate"
        } else {
            continue;
        };
        found.push(Channel {
            kind: kind.into(),
            reactant: reactant.clone(),
            saddle,
            product,
            mode: unstable,
            cell,
            barrier: saddle_energy - reactant_energy,
            dumbbell: built.dumbbell,
        });
    }
    let mut kept = Vec::new();
    for kind in ["translate", "rotate"] {
        if let Some(best) = found
            .iter()
            .filter(|item| item.kind == kind)
            .min_by(|left, right| left.barrier.partial_cmp(&right.barrier).unwrap())
        {
            kept.push(Channel {
                kind: best.kind.clone(),
                reactant: best.reactant.clone(),
                saddle: best.saddle.clone(),
                product: best.product.clone(),
                mode: best.mode.clone(),
                cell: best.cell,
                barrier: best.barrier,
                dumbbell: best.dumbbell,
            });
        }
    }
    kept
}

fn load_rated_hops() -> Vec<(String, String, String, f64)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/defect_pair_hops_static.csv");
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if index == 0 || line.is_empty() {
            continue;
        }
        let columns: Vec<&str> = line.split(',').collect();
        if columns.len() < 9 || columns[6] != "rated" {
            continue;
        }
        if let Ok(barrier) = columns[8].parse::<f64>() {
            rows.push((
                columns[0].to_string(),
                columns[1].to_string(),
                columns[3].to_string(),
                barrier,
            ));
        }
    }
    rows
}

fn cross_header() -> &'static str {
    "kind,shell,orientation,direction,lattice_A,lattice_note,n_atoms,\
dimer_barrier_eV,dimer_force,dimer_index,dimer_eigenvalue,dimer_converged,\
csv_barrier_eV,dimer_minus_csv_eV,\
fixed_barrier_eV,fixed_force,fixed_steps,fixed_converged,fixed_index,fixed_eigenvalue,fixed_rmsd_A,fixed_max_A,fixed_minus_dimer_eV,fixed_result,\
free_barrier_eV,free_force,free_steps,free_converged,free_index,free_eigenvalue,free_rmsd_A,free_max_A,free_cell_rms_A,free_a_A,free_minus_dimer_eV,free_minus_fixed_eV,free_result,\
reactant_cell_relax_eV,note"
}

#[test]
#[ignore = "8x8x8 Mishin SS-NEB cross-check; writes benchmarks/ssneb_crosscheck.csv"]
fn ssneb_crosscheck_cu() {
    let started = Instant::now();
    let copper = FastCopper::new();
    let repeats = 8usize;
    let (sample, sample_cell) = fcc(repeats, 3.615);
    let eval_started = Instant::now();
    let _ = copper.evaluate(sample.view(), sample_cell);
    eprintln!(
        "one 8x8x8 eval {:.3} s",
        eval_started.elapsed().as_secs_f64()
    );
    let mut jobs = Vec::new();
    let mut channels_500: Vec<Channel> = Vec::new();
    if wanted("isolated") {
        let specs = [
            (3.615_f64, "Mishin 0 K lattice", f64::NAN, f64::NAN),
            (3.628311, "300 K NPT lattice", 0.094614, 0.236061),
            (3.638666, "500 K NPT lattice", 0.090966, 0.233710),
        ];
        let only_a = std::env::var("RGSADDLE_SSNEB_A").ok().and_then(|text| text.parse::<f64>().ok());
        let specs: Vec<_> = specs
            .into_iter()
            .filter(|spec| only_a.map(|a| (spec.0 - a).abs() < 1.0e-4).unwrap_or(true))
            .collect();
        let channels: Vec<_> = specs
            .par_iter()
            .map(|(lattice, note, translate, rotate)| {
                let found = isolate_channels(&copper, repeats, *lattice);
                (*lattice, *note, *translate, *rotate, found)
            })
            .collect();
        for (lattice, note, translate_csv, rotate_csv, found) in channels {
            if (lattice - 3.638666).abs() < 1.0e-5 {
                channels_500 = found
                    .iter()
                    .map(|channel| Channel {
                        kind: channel.kind.clone(),
                        reactant: channel.reactant.clone(),
                        saddle: channel.saddle.clone(),
                        product: channel.product.clone(),
                        mode: channel.mode.clone(),
                        cell: channel.cell,
                        barrier: channel.barrier,
                        dumbbell: channel.dumbbell,
                    })
                    .collect();
            }
            for channel in found {
                let (direction, csv) = if channel.kind == "translate" {
                    ("toward", translate_csv)
                } else {
                    ("tangential", rotate_csv)
                };
                eprintln!(
                    "isolated {} {lattice:.4} loose barrier {:.4} eV",
                    channel.kind, channel.barrier
                );
                jobs.push(CrossJob {
                    kind: "isolated",
                    shell: "isolated".into(),
                    orientation: "100".into(),
                    direction: direction.into(),
                    lattice,
                    lattice_note: note.into(),
                    reactant: channel.reactant,
                    product: channel.product,
                    saddle: channel.saddle,
                    mode: channel.mode,
                    cell: channel.cell,
                    csv: if csv.is_finite() { vec![csv] } else { Vec::new() },
                    relax_product: true,
                    energy_weighted: false,
                });
            }
        }
    }
    if wanted("shell") || wanted("capture") {
        let a500 = 3.638666_f64;
        let channels = if channels_500.is_empty() {
            isolate_channels(&copper, repeats, a500)
        } else {
            channels_500
        };
        assert!(!channels.is_empty(), "the 500 K dumbbell produced no channel");
        let mechanisms: Vec<Mechanism> = channels
            .iter()
            .map(|channel| {
                capture_mechanism(
                    &copper,
                    &channel.reactant,
                    channel.cell,
                    channel.dumbbell,
                    &channel.saddle,
                    &channel.product,
                    &channel.mode,
                    channel.barrier,
                )
            })
            .collect();
        if wanted("shell") {
            let rated = load_rated_hops();
            let mut keys = Vec::new();
            for (shell, orientation, _, _) in &rated {
                let key = (shell.clone(), orientation.clone());
                if !keys.contains(&key) && shell_wanted(shell) {
                    keys.push(key);
                }
            }
            let built: Vec<_> = keys
                .par_iter()
                .filter_map(|(shell, orientation)| {
                    let radius: f64 = shell.parse().ok()?;
                    let vector = shell_table(a500, repeats)
                        .into_iter()
                        .find(|(value, _, _)| (*value - radius).abs() < 0.05)
                        .map(|(_, vector, _)| vector)?;
                    let (reactant, vacancy, dumbbell, cell, _, _, recombined) = relax_pair_axis(
                        &copper,
                        repeats,
                        a500,
                        vector,
                        axis_from_label(orientation),
                        true,
                    );
                    Some((shell.clone(), orientation.clone(), reactant, vacancy, dumbbell, cell, recombined))
                })
                .collect();
            for (shell, orientation, reactant, vacancy, dumbbell, cell, recombined) in built {
                if recombined {
                    jobs.push(CrossJob {
                        kind: "shell",
                        shell,
                        orientation,
                        direction: "recombined".into(),
                        lattice: a500,
                        lattice_note: "500 K NPT lattice; pair recombines on relaxation".into(),
                        reactant: reactant.clone(),
                        product: reactant.clone(),
                        saddle: reactant,
                        mode: Array1::zeros(0),
                        cell,
                        csv: Vec::new(),
                        relax_product: false,
                        energy_weighted: false,
                    });
                    continue;
                }
                let center = centroid(&reactant, dumbbell);
                let axis = axis_from_label(&orientation);
                let hops = mapped_hops(&reactant, cell, center, axis, &mechanisms);
                let mobile = [dumbbell[0], dumbbell[1]];
                let before = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
                for hop in hops {
                    let after = rgsaddle::pair_separation(hop.product.view(), cell, vacancy, &mobile).unwrap_or(before);
                    let Ok(direction) = rgsaddle::classify_separation(before, after, 0.08) else {
                        continue;
                    };
                    let csv: Vec<f64> = rated
                        .iter()
                        .filter(|(s, o, d, _)| s == &shell && o == &orientation && d == direction.as_str())
                        .map(|(_, _, _, barrier)| *barrier)
                        .collect();
                    if csv.is_empty() {
                        continue;
                    }
                    let Some(saddle) = hop.saddle.clone() else { continue };
                    let Some(mode) = hop.mode.clone() else { continue };
                    jobs.push(CrossJob {
                        kind: "shell",
                        shell: shell.clone(),
                        orientation: orientation.clone(),
                        direction: direction.as_str().into(),
                        lattice: a500,
                        lattice_note: "500 K NPT lattice; static barrier from defect_pair_hops_static.csv".into(),
                        reactant: reactant.clone(),
                        product: hop.product,
                        saddle,
                        mode,
                        cell,
                        csv,
                        relax_product: true,
                        energy_weighted: false,
                    });
                }
            }
        }
        if wanted("capture") {
            let radius = 5.146_f64;
            let vector = shell_table(a500, repeats)
                .into_iter()
                .find(|(value, _, _)| (*value - radius).abs() < 0.05)
                .map(|(_, vector, _)| vector)
                .expect("5.146 shell");
            let (reactant, vacancy, dumbbell, cell, _, _, recombined) =
                relax_pair_axis(&copper, repeats, a500, vector, axis_from_label("001"), true);
            assert!(!recombined, "5.146 001 recombined on relaxation");
            let held = FastHeld { copper: &copper, cell };
            let (reactant_energy, _) = held.eval(reactant.view()).expect("capture reactant");
            let config = survey_config(cell);
            let seeds = local_seeds(&reactant, cell, dumbbell, vacancy);
            let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
            let mut kept = Vec::new();
            for saddle in &found {
                let plus_re = reaches_recombination(
                    &reactant, reactant_energy, vacancy, cell, &saddle.plus, saddle.plus_energy,
                );
                let minus_re = reaches_recombination(
                    &reactant, reactant_energy, vacancy, cell, &saddle.minus, saddle.minus_energy,
                );
                let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0
                    || (saddle.plus_energy - reactant_energy).abs() < 0.2;
                let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0
                    || (saddle.minus_energy - reactant_energy).abs() < 0.2;
                if (plus_back && minus_re) || (minus_back && plus_re) {
                    kept.push(saddle);
                }
            }
            kept.sort_by(|left, right| {
                (left.saddle_energy - reactant_energy - 0.071269)
                    .abs()
                    .partial_cmp(&(right.saddle_energy - reactant_energy - 0.071269).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if let Some(best) = kept.first() {
                let sign = if best.plus_energy < best.minus_energy { 1.0 } else { -1.0 };
                let mut peak = 0.0_f64;
                for atom in 0..best.mode.len() / 3 {
                    let mut acc = 0.0;
                    for c in 0..3 {
                        acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
                    }
                    peak = peak.max(acc.sqrt());
                }
                let mut start = best.saddle.clone();
                if peak > 1.0e-8 {
                    start = &best.saddle + &(&best.mode * (sign * 0.04 / peak));
                }
                let floor = best.saddle_energy - 0.6;
                let (downhill, _) = rgsaddle::descend_limited(&held, start, cell, &config, floor)
                    .unwrap_or((best.saddle.clone(), best.saddle_energy));
                jobs.push(CrossJob {
                    kind: "capture",
                    shell: "5.146".into(),
                    orientation: "001".into(),
                    direction: "toward".into(),
                    lattice: a500,
                    lattice_note: "500 K NPT lattice; defect_pair_local.csv capture 0.071269 eV".into(),
                    reactant,
                    product: downhill,
                    saddle: best.saddle.clone(),
                    mode: best.mode.clone(),
                    cell,
                    csv: vec![0.071269],
                    relax_product: false,
                    energy_weighted: true,
                });
            }
        }
    }
    let max_jobs = env_usize("RGSADDLE_SSNEB_MAX", usize::MAX);
    if jobs.len() > max_jobs {
        jobs.truncate(max_jobs);
    }
    assert!(!jobs.is_empty(), "no SS-NEB cross-check jobs");
    eprintln!("{} jobs after {:.1} s", jobs.len(), started.elapsed().as_secs_f64());
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_crosscheck.csv");
    let mut file = File::create(&path).expect("cross-check csv");
    writeln!(file, "{}", cross_header()).unwrap();
    file.flush().unwrap();
    let sink = Mutex::new(file);
    jobs.par_iter().for_each(|job| {
        let row = cross_row(&copper, job);
        eprintln!("{row}");
        let mut handle = sink.lock().unwrap_or_else(|poison| poison.into_inner());
        writeln!(handle, "{row}").unwrap();
        handle.flush().unwrap();
    });
    let text = std::fs::read_to_string(&path).unwrap();
    if wanted("isolated") {
        assert!(text.lines().any(|line| line.starts_with("isolated,isolated,100,toward")), "missing free translation");
        assert!(text.lines().any(|line| line.starts_with("isolated,isolated,100,tangential")), "missing rotation");
    }
    if wanted("capture") {
        assert!(text.lines().any(|line| line.starts_with("capture,5.146,001")), "missing capture");
    }
    if wanted("shell") {
        assert!(text.lines().any(|line| line.starts_with("shell,")), "missing shell hops");
    }
    eprintln!("cross-check {:.1} min", started.elapsed().as_secs_f64() / 60.0);
}

#[derive(Clone)]
struct BandStat {
    barrier: f64,
    force: f64,
    steps: usize,
    calls: usize,
    converged: bool,
}

fn band_stat(
    copper: &FastCopper,
    positions: Array2<f64>,
    cells: Vec<Cell>,
    fixed: Option<[[f64; 3]; 3]>,
    spring: SpringKind,
) -> BandStat {
    let before = copper.calls.load(Ordering::Relaxed);
    let end = drive_band(copper, positions.clone(), cells.clone(), fixed, spring);
    let calls = copper.calls.load(Ordering::Relaxed) - before;
    let Ok(end) = end else {
        return BandStat {
            barrier: f64::NAN,
            force: f64::NAN,
            steps: 0,
            calls,
            converged: false,
        };
    };
    let energies = energies_of(copper, &end.positions, &end.cells);
    let ci = highest_interior(&energies);
    BandStat {
        barrier: energies[ci] - energies[0],
        force: end.force,
        steps: end.steps,
        calls,
        converged: end.converged,
    }
}

fn size_pair(copper: &FastCopper, job: &CrossJob) -> (BandStat, BandStat) {
    let mut reactant = job.reactant.clone();
    if max_abs(&copper.evaluate(reactant.view(), job.cell).1) > 2.0e-3 {
        relax_fast(copper, &mut reactant, job.cell, 1.0e-3, 80);
    }
    let product = if job.relax_product {
        let mut positions = job.product.clone();
        relax_fast(copper, &mut positions, job.cell, 1.0e-3, 80);
        positions
    } else {
        job.product.clone()
    };
    let tight = if job.kind == "shell" {
        None
    } else {
        Some(tighten_dimer(copper, job.cell, &job.saddle, &job.mode))
    };
    let spring = if job.energy_weighted {
        SpringKind::EnergyWeighted { k_min: 0.1, k_max: 5.0 }
    } else {
        SpringKind::Uniform { k: 1.0 }
    };
    let saddle_guess = tight
        .as_ref()
        .map(|row| row.positions.clone())
        .unwrap_or_else(|| blend_positions(&reactant, job.cell, &product, job.cell, job.cell, 0.5));
    let (fixed_positions, fixed_cells) =
        band_images(&reactant, job.cell, &saddle_guess, job.cell, &product, job.cell);
    let fixed = band_stat(copper, fixed_positions, fixed_cells, Some(job.cell), spring.clone());
    let (free_reactant, free_cell, free_product, free_product_cell) =
        match (relax_joint(copper, &reactant, job.cell), relax_joint(copper, &product, job.cell)) {
            (Ok(left), Ok(right)) => (left.0, left.1, right.0, right.1),
            _ => (reactant, job.cell, product, job.cell),
        };
    let free_mid = blend_positions(
        &free_reactant,
        free_cell,
        &free_product,
        free_product_cell,
        blend_cell(free_cell, free_product_cell, 0.5),
        0.5,
    );
    let (free_positions, free_cells) = band_images(
        &free_reactant,
        free_cell,
        if job.kind == "shell" { &free_mid } else { &saddle_guess },
        job.cell,
        &free_product,
        free_product_cell,
    );
    let free = band_stat(copper, free_positions, free_cells, None, spring);
    (fixed, free)
}

fn quench_ends(
    copper: &FastCopper,
    saddle: &Array1<f64>,
    mode: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Option<(Array1<f64>, Array1<f64>)> {
    let mut peak = 0.0_f64;
    for atom in 0..saddle.len() / 3 {
        let mut acc = 0.0;
        for c in 0..3 {
            acc += mode[3 * atom + c] * mode[3 * atom + c];
        }
        peak = peak.max(acc.sqrt());
    }
    if peak < 1.0e-8 {
        return None;
    }
    let step = mode * (0.45 / peak);
    let mut plus = saddle + &step;
    let mut minus = saddle - &step;
    relax_fast(copper, &mut plus, cell, 1.5e-3, 80);
    relax_fast(copper, &mut minus, cell, 1.5e-3, 80);
    Some((plus, minus))
}

struct QuenchedAway {
    product: Array1<f64>,
    saddle: Array1<f64>,
    mode: Array1<f64>,
    separation: f64,
    reaction: f64,
}

fn away_shells(
    copper: &FastCopper,
    repeats: usize,
    lattice: f64,
    note: &str,
    mechanisms: &[Mechanism],
) -> Vec<CrossJob> {
    let mut jobs = Vec::new();
    for (r_over_a, orientation, shell) in [(1.00009_f64, "010", "3.639"), (1.2246, "100", "4.456")] {
        let vector = shell_vector(lattice, repeats, r_over_a);
        let (reactant, vacancy, dumbbell, cell, _, _, recombined) =
            relax_pair_axis(copper, repeats, lattice, vector, axis_from_label(orientation), true);
        if recombined {
            eprintln!("shell {shell} {repeats} {lattice:.4} recombined");
            continue;
        }
        let center = centroid(&reactant, dumbbell);
        let hops = mapped_hops(&reactant, cell, center, axis_from_label(orientation), mechanisms);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
        let reactant_energy = copper.evaluate(reactant.view(), cell).0;
        let mut found: Vec<QuenchedAway> = Vec::new();
        for hop in hops {
            let Some(saddle) = hop.saddle.clone() else { continue };
            let Some(mode) = hop.mode.clone() else { continue };
            let Some((plus, minus)) = quench_ends(copper, &saddle, &mode, cell) else { continue };
            for (product, other) in [(&plus, &minus), (&minus, &plus)] {
                let after = rgsaddle::pair_separation(product.view(), cell, vacancy, &mobile).unwrap_or(before);
                let back = rgsaddle::pair_separation(other.view(), cell, vacancy, &mobile).unwrap_or(before);
                let Ok(kind) = rgsaddle::classify_separation(before, after, 0.08) else { continue };
                let back_home = (back - before).abs() < 0.30 || hausdorff(other, &reactant, cell) < 2.0;
                if kind.as_str() != "away" || !back_home {
                    continue;
                }
                if found.iter().any(|kept| (kept.separation - after).abs() < 0.15) {
                    continue;
                }
                let reaction = copper.evaluate(product.view(), cell).0 - reactant_energy;
                eprintln!(
                    "away candidate {shell} {orientation} repeats {repeats} a {lattice:.4} sep {before:.3}->{after:.3} back {back:.3} reaction {reaction:.4}"
                );
                found.push(QuenchedAway {
                    product: product.clone(),
                    saddle,
                    mode,
                    separation: after,
                    reaction,
                });
                break;
            }
        }
        if found.is_empty() {
            eprintln!("shell {shell} {repeats} {lattice:.4} no quench-matched away hop");
            continue;
        }
        for item in found {
            jobs.push(CrossJob {
                kind: "shell",
                shell: shell.into(),
                orientation: orientation.into(),
                direction: format!("away;sep={:.3};dE={:.4}", item.separation, item.reaction),
                lattice,
                lattice_note: format!(
                    "{note};vac={:.5},{:.5},{:.5};db={},{}",
                    vacancy[0], vacancy[1], vacancy[2], dumbbell[0], dumbbell[1]
                ),
                reactant: reactant.clone(),
                product: item.product,
                saddle: item.saddle,
                mode: item.mode,
                cell,
                csv: Vec::new(),
                relax_product: false,
                energy_weighted: false,
            });
        }
    }
    jobs
}

fn shell_vector(a: f64, repeats: usize, r_over_a: f64) -> [f64; 3] {
    let target = r_over_a * a;
    shell_table(a, repeats)
        .into_iter()
        .min_by(|left, right| {
            (left.0 - target)
                .abs()
                .partial_cmp(&(right.0 - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, vector, _)| vector)
        .unwrap_or([a, 0.0, 0.0])
}

fn push_isolated(jobs: &mut Vec<CrossJob>, channel: Channel, lattice: f64, note: &str) {
    let (direction, kind_note) = if channel.kind == "translate" {
        ("toward", "isolated dumbbell translation")
    } else {
        ("tangential", "isolated dumbbell rotation")
    };
    jobs.push(CrossJob {
        kind: "isolated",
        shell: "isolated".into(),
        orientation: "100".into(),
        direction: direction.into(),
        lattice,
        lattice_note: format!("{note}; {kind_note}"),
        reactant: channel.reactant,
        product: channel.product,
        saddle: channel.saddle,
        mode: channel.mode,
        cell: channel.cell,
        csv: Vec::new(),
        relax_product: true,
        energy_weighted: false,
    });
}

fn size_jobs(copper: &FastCopper, repeats: usize, lattice: f64, temperature: f64) -> Vec<CrossJob> {
    let note = format!("{temperature:.0} K NPT lattice");
    let mut jobs = Vec::new();
    let channels = isolate_channels(copper, repeats, lattice);
    let mechanisms: Vec<Mechanism> = channels
        .iter()
        .map(|channel| {
            capture_mechanism(
                copper,
                &channel.reactant,
                channel.cell,
                channel.dumbbell,
                &channel.saddle,
                &channel.product,
                &channel.mode,
                channel.barrier,
            )
        })
        .collect();
    for channel in channels {
        if channel.kind == "translate" || channel.kind == "rotate" {
            push_isolated(&mut jobs, channel, lattice, &note);
        }
    }
    jobs.extend(away_shells(copper, repeats, lattice, &note, &mechanisms));
    let vector = shell_vector(lattice, repeats, 1.4142);
    let (reactant, vacancy, dumbbell, cell, _, _, recombined) =
        relax_pair_axis(copper, repeats, lattice, vector, axis_from_label("001"), true);
    if !recombined {
        let held = FastHeld { copper, cell };
        let (reactant_energy, _) = held.eval(reactant.view()).unwrap_or((0.0, Array1::zeros(1)));
        let config = survey_config(cell);
        let seeds = local_seeds(&reactant, cell, dumbbell, vacancy);
        let found = rgsaddle::seeded_saddles(&held, &config, cell, &seeds);
        let mut kept = Vec::new();
        for saddle in &found {
            let plus_re = reaches_recombination(
                &reactant, reactant_energy, vacancy, cell, &saddle.plus, saddle.plus_energy,
            );
            let minus_re = reaches_recombination(
                &reactant, reactant_energy, vacancy, cell, &saddle.minus, saddle.minus_energy,
            );
            let plus_back = hausdorff(&saddle.plus, &reactant, cell) < 1.0;
            let minus_back = hausdorff(&saddle.minus, &reactant, cell) < 1.0;
            if (plus_back && minus_re) || (minus_back && plus_re) {
                kept.push(saddle);
            }
        }
        kept.sort_by(|left, right| {
            left.saddle_energy
                .partial_cmp(&right.saddle_energy)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(best) = kept.first() {
            let sign = if best.plus_energy < best.minus_energy { 1.0 } else { -1.0 };
            let mut peak = 0.0_f64;
            for atom in 0..best.mode.len() / 3 {
                let mut acc = 0.0;
                for c in 0..3 {
                    acc += best.mode[3 * atom + c] * best.mode[3 * atom + c];
                }
                peak = peak.max(acc.sqrt());
            }
            let mut start = best.saddle.clone();
            if peak > 1.0e-8 {
                start = &best.saddle + &(&best.mode * (sign * 0.04 / peak));
            }
            let floor = best.saddle_energy - 0.6;
            let (downhill, _) = rgsaddle::descend_limited(&held, start, cell, &config, floor)
                .unwrap_or((best.saddle.clone(), best.saddle_energy));
            jobs.push(CrossJob {
                kind: "capture",
                shell: "5.146".into(),
                orientation: "001".into(),
                direction: "toward".into(),
                lattice,
                lattice_note: note,
                reactant,
                product: downhill,
                saddle: best.saddle.clone(),
                mode: best.mode.clone(),
                cell,
                csv: Vec::new(),
                relax_product: false,
                energy_weighted: true,
            });
        }
    }
    jobs
}

fn split_away(direction: &str) -> (&str, String, String) {
    let mut parts = direction.split(';');
    let head = parts.next().unwrap_or(direction);
    let mut separation = String::new();
    let mut reaction = String::new();
    for part in parts {
        if let Some(value) = part.strip_prefix("sep=") {
            separation = value.to_string();
        }
        if let Some(value) = part.strip_prefix("dE=") {
            reaction = value.to_string();
        }
    }
    (head, separation, reaction)
}

fn shell_bundle(copper: &FastCopper, repeats: usize, lattice: f64, temperature: f64) -> Vec<CrossJob> {
    let note = format!("{temperature:.0} K NPT lattice");
    let channels = isolate_channels(copper, repeats, lattice);
    let mechanisms: Vec<Mechanism> = channels
        .iter()
        .map(|channel| {
            capture_mechanism(
                copper,
                &channel.reactant,
                channel.cell,
                channel.dumbbell,
                &channel.saddle,
                &channel.product,
                &channel.mode,
                channel.barrier,
            )
        })
        .collect();
    away_shells(copper, repeats, lattice, &note, &mechanisms)
}

fn write_size_row(
    file: &Mutex<File>,
    repeats: usize,
    temperature: f64,
    job: &CrossJob,
    mode: &str,
    stat: &BandStat,
) {
    let (direction, separation, reaction) = split_away(&job.direction);
    let line = format!(
        "{repeats},{},{temperature:.0},{kind},{shell},{orientation},{direction},{mode},{},{:.6},{:.6e},{},{},{},{separation},{reaction}",
        job.lattice,
        job.reactant.len() / 3,
        stat.barrier,
        stat.force,
        stat.calls,
        stat.steps,
        stat.converged as u8,
        kind = job.kind,
        shell = job.shell,
        orientation = job.orientation,
    );
    eprintln!("{line}");
    let mut handle = file.lock().unwrap_or_else(|poison| poison.into_inner());
    writeln!(handle, "{line}").unwrap();
    handle.flush().unwrap();
}

#[test]
#[ignore = "10x10x10 and 12x12x12 Mishin SS-NEB; writes benchmarks/ssneb_size.csv"]
fn ssneb_size_cu() {
    let copper = FastCopper::new();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_size.csv");
    let mut file = File::create(&path).expect("size csv");
    writeln!(
        file,
        "repeats,lattice_A,temperature_K,kind,shell,orientation,direction,cell,n_atoms,barrier_eV,max_force,force_calls,steps,converged,product_sep_A,reaction_eV"
    )
    .unwrap();
    let sink = Mutex::new(file);
    let specs = [
        (10usize, 3.628311_f64, 300.0_f64),
        (10, 3.638666, 500.0),
        (12, 3.628311, 300.0),
        (12, 3.638666, 500.0),
    ];
    let work: Vec<_> = specs
        .par_iter()
        .map(|(repeats, lattice, temperature)| {
            eprintln!("building {repeats}x{repeats}x{repeats} at {lattice}");
            (*repeats, *temperature, size_jobs(&copper, *repeats, *lattice, *temperature))
        })
        .collect();
    let rows: Vec<_> = work
        .par_iter()
        .flat_map(|(repeats, temperature, jobs)| {
            jobs.par_iter()
                .map(|job| {
                    let local = FastCopper::new();
                    let (fixed, free) = size_pair(&local, job);
                    (*repeats, *temperature, job, fixed, free)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    for (repeats, temperature, job, fixed, free) in rows {
        write_size_row(&sink, repeats, temperature, job, "fixed", &fixed);
        write_size_row(&sink, repeats, temperature, job, "free", &free);
    }
}

#[test]
fn ssneb_size_small_is_reproducible() {
    let copper = FastCopper::new();
    let lattice = 3.628311_f64;
    let jobs = size_jobs(&copper, 4, lattice, 300.0);
    let job = jobs
        .iter()
        .find(|job| job.kind == "isolated" && job.direction == "toward")
        .expect("4x4x4 translation");
    let first = size_pair(&copper, job).0;
    let second = size_pair(&copper, job).0;
    assert!(first.converged && second.converged, "forces {} {}", first.force, second.force);
    assert!((first.barrier - second.barrier).abs() < 1.0e-6, "{} {}", first.barrier, second.barrier);
    assert!((first.barrier - 0.05).abs() < 0.15, "barrier {}", first.barrier);
    assert_eq!(first.steps, second.steps);
    assert_eq!(first.calls, second.calls);
}

#[test]
#[ignore = "quench-matched 3.639 and 4.456 shells at 8, 10 and 12; writes benchmarks/ssneb_shell_repair.csv"]
fn ssneb_shell_repair() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_shell_repair.csv");
    let mut file = File::create(&path).expect("shell repair csv");
    writeln!(
        file,
        "repeats,lattice_A,temperature_K,kind,shell,orientation,direction,cell,n_atoms,barrier_eV,max_force,force_calls,steps,converged,product_sep_A,reaction_eV"
    )
    .unwrap();
    let sink = Mutex::new(file);
    let specs = [
        (8usize, 3.628311_f64, 300.0_f64),
        (8, 3.638666, 500.0),
        (10, 3.628311, 300.0),
        (10, 3.638666, 500.0),
        (12, 3.628311, 300.0),
        (12, 3.638666, 500.0),
    ];
    let work: Vec<_> = specs
        .par_iter()
        .map(|(repeats, lattice, temperature)| {
            eprintln!("shells {repeats} at {lattice}");
            let copper = FastCopper::new();
            (*repeats, *temperature, shell_bundle(&copper, *repeats, *lattice, *temperature))
        })
        .collect();
    let rows: Vec<_> = work
        .par_iter()
        .flat_map(|(repeats, temperature, jobs)| {
            jobs.par_iter()
                .map(|job| {
                    let local = FastCopper::new();
                    let (fixed, free) = size_pair(&local, job);
                    (*repeats, *temperature, job, fixed, free)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    let mut missed = Vec::new();
    for (repeats, temperature, job, fixed, free) in &rows {
        write_size_row(&sink, *repeats, *temperature, job, "fixed", fixed);
        write_size_row(&sink, *repeats, *temperature, job, "free", free);
        if !fixed.converged {
            missed.push(format!("{repeats} {temperature} {} fixed force {}", job.shell, fixed.force));
        }
        if !free.converged {
            missed.push(format!("{repeats} {temperature} {} free force {}", job.shell, free.force));
        }
    }
    assert!(missed.is_empty(), "force tolerance missed: {missed:?}");
}

fn note_frame(note: &str) -> Option<([f64; 3], [usize; 2])> {
    let mut vacancy = None;
    let mut dumbbell = None;
    for part in note.split(';') {
        if let Some(text) = part.strip_prefix("vac=") {
            let mut values = text.split(',');
            let x = values.next()?.parse().ok()?;
            let y = values.next()?.parse().ok()?;
            let z = values.next()?.parse().ok()?;
            vacancy = Some([x, y, z]);
        }
        if let Some(text) = part.strip_prefix("db=") {
            let mut values = text.split(',');
            let a = values.next()?.parse().ok()?;
            let b = values.next()?.parse().ok()?;
            dumbbell = Some([a, b]);
        }
    }
    Some((vacancy?, dumbbell?))
}

fn mic_delta(a: [f64; 3], b: [f64; 3], cell: [[f64; 3]; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for c in 0..3 {
        let length = cell[c][c];
        let mut delta = a[c] - b[c];
        delta -= length * (delta / length).round();
        out[c] = delta;
    }
    out
}

fn transplant_product(
    donor: &CrossJob,
    acceptor: &Array1<f64>,
    acceptor_center: [f64; 3],
    acceptor_cell: [[f64; 3]; 3],
) -> Option<Array1<f64>> {
    let (_, dumbbell) = note_frame(&donor.lattice_note)?;
    let donor_center = centroid(&donor.reactant, dumbbell);
    let mut image = acceptor.clone();
    let n_donor = donor.reactant.len() / 3;
    let n = acceptor.len() / 3;
    for atom in 0..n {
        let offset = mic_delta(point_of(acceptor, atom), acceptor_center, acceptor_cell);
        let radius = (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
        if radius > 6.5 {
            continue;
        }
        let mut best = None;
        let mut best_d = 0.45_f64;
        for other in 0..n_donor {
            let there = mic_delta(point_of(&donor.reactant, other), donor_center, donor.cell);
            let mut acc = 0.0;
            for c in 0..3 {
                let d = offset[c] - there[c];
                acc += d * d;
            }
            let distance = acc.sqrt();
            if distance < best_d {
                best_d = distance;
                best = Some(other);
            }
        }
        let Some(other) = best else { continue };
        let delta = mic_delta(point_of(&donor.product, other), point_of(&donor.reactant, other), donor.cell);
        for c in 0..3 {
            image[3 * atom + c] += delta[c];
        }
    }
    Some(image)
}

#[test]
#[ignore = "transplant the 12^3 500 K 4.456 ends onto 8^3 and 10^3"]
fn ssneb_4456_fill() {
    let lattice = 3.638666_f64;
    let copper = FastCopper::new();
    eprintln!("donor 12 at {lattice}");
    let donor = shell_bundle(&copper, 12, lattice, 500.0);
    let donors: Vec<_> = donor.into_iter().filter(|job| job.shell == "4.456").collect();
    assert!(!donors.is_empty(), "12^3 500 K produced no 4.456 end");
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_4456_fill.csv");
    let mut file = File::create(&path).expect("fill csv");
    writeln!(
        file,
        "repeats,lattice_A,temperature_K,kind,shell,orientation,direction,cell,n_atoms,barrier_eV,max_force,force_calls,steps,converged,product_sep_A,reaction_eV"
    )
    .unwrap();
    let sink = Mutex::new(file);
    let mut jobs = Vec::new();
    for repeats in [8usize, 10] {
        let vector = shell_vector(lattice, repeats, 1.2246);
        let (reactant, vacancy, dumbbell, cell, _, _, recombined) =
            relax_pair_axis(&copper, repeats, lattice, vector, axis_from_label("100"), true);
        assert!(!recombined, "{repeats} 4.456 recombined");
        let center = centroid(&reactant, dumbbell);
        let mobile = [dumbbell[0], dumbbell[1]];
        let before = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
        for donor_job in &donors {
            let Some(mut product) = transplant_product(donor_job, &reactant, center, cell) else { continue };
            relax_fast(&copper, &mut product, cell, 1.0e-3, 80);
            let after = rgsaddle::pair_separation(product.view(), cell, vacancy, &mobile).unwrap_or(0.0);
            let (_, sep_text, _) = split_away(&donor_job.direction);
            let wanted: f64 = sep_text.parse().unwrap_or(0.0);
            eprintln!("transplant {repeats} donor sep {wanted:.3} got {before:.3}->{after:.3}");
            if (after - wanted).abs() > 0.08 {
                continue;
            }
            let reaction = copper.evaluate(product.view(), cell).0 - copper.evaluate(reactant.view(), cell).0;
            jobs.push((
                repeats,
                CrossJob {
                    kind: "shell",
                    shell: "4.456".into(),
                    orientation: "100".into(),
                    direction: format!("away;sep={after:.3};dE={reaction:.4}"),
                    lattice,
                    lattice_note: format!("500 K NPT lattice; transplanted from 12^3"),
                    reactant: reactant.clone(),
                    product,
                    saddle: reactant.clone(),
                    mode: Array1::zeros(reactant.len()),
                    cell,
                    csv: Vec::new(),
                    relax_product: false,
                    energy_weighted: false,
                },
            ));
        }
    }
    assert!(
        jobs.iter().any(|(repeats, _)| *repeats == 10),
        "10^3 500 K 4.456 transplant did not match an end"
    );
    for (repeats, job) in &jobs {
        let local = FastCopper::new();
        let (fixed, free) = size_pair(&local, job);
        write_size_row(&sink, *repeats, 500.0, job, "fixed", &fixed);
        write_size_row(&sink, *repeats, 500.0, job, "free", &free);
        assert!(fixed.converged, "fixed force {}", fixed.force);
        assert!(free.converged, "free force {}", free.force);
    }
}

fn fcc_neighbor_steps(a: f64) -> Vec<[f64; 3]> {
    let half = 0.5 * a;
    let signs = [-1.0_f64, 1.0];
    let mut steps = Vec::new();
    for &left in &signs {
        for &right in &signs {
            steps.push([left * half, right * half, 0.0]);
            steps.push([left * half, 0.0, right * half]);
            steps.push([0.0, left * half, right * half]);
        }
    }
    steps
}

fn dumbbell_axis(positions: &Array1<f64>, dumbbell: [usize; 2], cell: [[f64; 3]; 3]) -> [f64; 3] {
    let center = centroid(positions, dumbbell);
    unit3(mic_delta(point_of(positions, dumbbell[0]), center, cell))
}

fn axis_overlap(left: [f64; 3], right: [f64; 3]) -> f64 {
    let left = unit3(left);
    let right = unit3(right);
    (left[0] * right[0] + left[1] * right[1] + left[2] * right[2]).abs()
}

fn axis_name(axis: [f64; 3]) -> String {
    let axis = unit3(axis);
    format!(
        "{}{}{}",
        if axis[0].abs() > 0.6 { 1 } else { 0 },
        if axis[1].abs() > 0.6 { 1 } else { 0 },
        if axis[2].abs() > 0.6 { 1 } else { 0 }
    )
}

fn pair_recombined(
    positions: &Array1<f64>,
    vacancy: [f64; 3],
    dumbbell: [usize; 2],
    cell: [[f64; 3]; 3],
) -> bool {
    let gap = dist(point_of(positions, dumbbell[0]), point_of(positions, dumbbell[1]));
    let separation = dist(centroid(positions, dumbbell), vacancy);
    occupied(positions, vacancy, cell) || !(1.2..2.5).contains(&gap) || separation < 1.6
}

fn shift_dumbbell(positions: &Array1<f64>, dumbbell: [usize; 2], delta: [f64; 3]) -> Array1<f64> {
    let mut moved = positions.clone();
    for atom in dumbbell {
        for c in 0..3 {
            moved[3 * atom + c] += delta[c];
        }
    }
    moved
}

fn rotate_dumbbell(positions: &Array1<f64>, dumbbell: [usize; 2], axis: [f64; 3]) -> Array1<f64> {
    let center = centroid(positions, dumbbell);
    let half = dist(point_of(positions, dumbbell[0]), center).max(0.8);
    let axis = unit3(axis);
    let mut moved = positions.clone();
    for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
        for c in 0..3 {
            moved[3 * atom + c] = center[c] + sign * half * axis[c];
        }
    }
    moved
}

struct QuenchedEnd {
    positions: Array1<f64>,
    separation: f64,
    reaction: f64,
    recombined: bool,
}

fn quench_end(
    copper: &FastCopper,
    guess: Array1<f64>,
    vacancy: [f64; 3],
    dumbbell: [usize; 2],
    cell: [[f64; 3]; 3],
    reactant_energy: f64,
) -> QuenchedEnd {
    let mut positions = guess;
    relax_fast(copper, &mut positions, cell, 1.5e-3, 60);
    let mobile = [dumbbell[0], dumbbell[1]];
    let separation = rgsaddle::pair_separation(positions.view(), cell, vacancy, &mobile).unwrap_or(0.0);
    let recombined = pair_recombined(&positions, vacancy, dumbbell, cell);
    let reaction = copper.evaluate(positions.view(), cell).0 - reactant_energy;
    QuenchedEnd {
        positions,
        separation,
        reaction,
        recombined,
    }
}

struct ThirteenHop {
    shell: String,
    orientation: String,
    mechanism: &'static str,
    sites: usize,
    reactant_sep: f64,
    end_sep: f64,
    reaction: f64,
    job: CrossJob,
}

fn hop_job(
    mechanism: &'static str,
    shell: &str,
    orientation: &str,
    lattice: f64,
    reactant: &Array1<f64>,
    end: &QuenchedEnd,
    cell: [[f64; 3]; 3],
) -> CrossJob {
    CrossJob {
        kind: "shell",
        shell: shell.into(),
        orientation: orientation.into(),
        direction: format!(
            "{mechanism};sep={:.3};dE={:.4}",
            end.separation, end.reaction
        ),
        lattice,
        lattice_note: "300 K NPT lattice".into(),
        reactant: reactant.clone(),
        product: end.positions.clone(),
        saddle: reactant.clone(),
        mode: Array1::zeros(reactant.len()),
        cell,
        csv: Vec::new(),
        relax_product: false,
        energy_weighted: false,
    }
}

fn distinct_end(ends: &[QuenchedEnd], end: &QuenchedEnd) -> bool {
    ends.iter().all(|kept| (kept.separation - end.separation).abs() > 0.12)
}

fn shell_frame(
    copper: &FastCopper,
    repeats: usize,
    lattice: f64,
    shell: [f64; 3],
    axis: [f64; 3],
) -> (Array1<f64>, [f64; 3], [usize; 2], [[f64; 3]; 3], f64, bool) {
    let (mut reactant, vacancy, dumbbell, cell, _, _, recombined) =
        relax_pair_axis(copper, repeats, lattice, shell, axis, true);
    if recombined {
        relax_fast(copper, &mut reactant, cell, 1.5e-3, 20);
    }
    let mobile = [dumbbell[0], dumbbell[1]];
    let separation = rgsaddle::pair_separation(reactant.view(), cell, vacancy, &mobile).unwrap_or(0.0);
    let gone = pair_recombined(&reactant, vacancy, dumbbell, cell);
    (reactant, vacancy, dumbbell, cell, separation, gone)
}

fn shell_hops(copper: &FastCopper, repeats: usize, lattice: f64, shell: [f64; 3], sites: usize) -> Vec<ThirteenHop> {
    let radius = (shell[0] * shell[0] + shell[1] * shell[1] + shell[2] * shell[2]).sqrt();
    let label = format!("{radius:.3}");
    let axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut frames = Vec::new();
    for axis in axes {
        frames.push((axis, shell_frame(copper, repeats, lattice, shell, axis)));
    }
    let (axis, (reactant, vacancy, dumbbell, cell, reactant_sep, gone)) = frames
        .into_iter()
        .min_by(|left, right| {
            let left_err = if left.1.5 { 100.0 } else { (left.1.4 - radius).abs() };
            let right_err = if right.1.5 { 100.0 } else { (right.1.4 - radius).abs() };
            left_err.partial_cmp(&right_err).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("three <100> axes");
    let orientation = axis_name(axis);
    let _ = gone;
    let reactant_energy = copper.evaluate(reactant.view(), cell).0;
    if pair_recombined(&reactant, vacancy, dumbbell, cell) {
        eprintln!("shell {label} recombines on relaxation");
        let end = QuenchedEnd {
            positions: reactant.clone(),
            separation: reactant_sep,
            reaction: 0.0,
            recombined: true,
        };
        return vec![ThirteenHop {
            shell: label,
            orientation,
            mechanism: "capture",
            sites,
            reactant_sep,
            end_sep: reactant_sep,
            reaction: 0.0,
            job: hop_job("capture", &format!("{radius:.3}"), &axis_name(axis), lattice, &reactant, &end, cell),
        }];
    }
    let mut hops = Vec::new();
    let rotation_axis = if axis[2].abs() > 0.6 {
        [0.0, 1.0, 1.0]
    } else {
        [1.0, 1.0, 0.0]
    };
    let rotated = quench_end(
        copper,
        rotate_dumbbell(&reactant, dumbbell, rotation_axis),
        vacancy,
        dumbbell,
        cell,
        reactant_energy,
    );
    let rotated_axis = dumbbell_axis(&rotated.positions, dumbbell, cell);
    let rotated_shift = (rotated.separation - reactant_sep).abs();
    if !rotated.recombined && rotated_shift < 0.45 && axis_overlap(rotated_axis, axis) < 0.85 {
        hops.push(ThirteenHop {
            shell: label.clone(),
            orientation: orientation.clone(),
            mechanism: "rotation",
            sites,
            reactant_sep,
            end_sep: rotated.separation,
            reaction: rotated.reaction,
            job: hop_job("rotation", &label, &orientation, lattice, &reactant, &rotated, cell),
        });
    }
    if !hops.iter().any(|hop| hop.mechanism == "rotation")
        && let Some(end) = dimer_rotation(copper, &reactant, dumbbell, cell, rotation_axis)
    {
        let now = dumbbell_axis(&end.positions, dumbbell, cell);
        let sep = rgsaddle::pair_separation(end.positions.view(), cell, vacancy, &dumbbell)
            .unwrap_or(reactant_sep);
        if (sep - reactant_sep).abs() < 0.45 && axis_overlap(now, axis) < 0.85 {
            let sep = sep;
            eprintln!("shell {label} dimer rotation axis {} sep {sep:.3}", axis_name(now));
            hops.push(ThirteenHop {
                shell: label.clone(),
                orientation: orientation.clone(),
                mechanism: "rotation",
                sites,
                reactant_sep,
                end_sep: sep,
                reaction: end.reaction,
                job: hop_job("rotation", &label, &orientation, lattice, &reactant, &end, cell),
            });
        }
    }
    let mut toward: Vec<QuenchedEnd> = Vec::new();
    let mut away: Vec<QuenchedEnd> = Vec::new();
    let mut tangential: Vec<QuenchedEnd> = Vec::new();
    for step in fcc_neighbor_steps(lattice) {
        let end = quench_end(
            copper,
            shift_dumbbell(&reactant, dumbbell, step),
            vacancy,
            dumbbell,
            cell,
            reactant_energy,
        );
        if hausdorff(&end.positions, &reactant, cell) < 0.35 {
            continue;
        }
        if end.recombined || end.separation + 0.08 < reactant_sep {
            if distinct_end(&toward, &end) {
                toward.push(end);
            }
            continue;
        }
        if end.separation > reactant_sep + 0.08 {
            if distinct_end(&away, &end) {
                away.push(end);
            }
            continue;
        }
        if distinct_end(&tangential, &end) {
            tangential.push(end);
        }
    }
    let capture = toward.into_iter().min_by(|left, right| {
        left.separation.partial_cmp(&right.separation).unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(end) = capture {
        hops.push(ThirteenHop {
            shell: label.clone(),
            orientation: orientation.clone(),
            mechanism: "capture",
            sites,
            reactant_sep,
            end_sep: end.separation,
            reaction: end.reaction,
            job: hop_job("capture", &label, &orientation, lattice, &reactant, &end, cell),
        });
    }
    let away_shell = (radius / lattice - 1.224744871).abs() < 0.04;
    if away_shell && away.len() < 2 {
        for step in fcc_neighbor_steps(lattice) {
            for extra in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                if axis_name(extra) == orientation {
                    continue;
                }
                let shifted = shift_dumbbell(&reactant, dumbbell, step);
                let end = quench_end(
                    copper,
                    rotate_dumbbell(&shifted, dumbbell, extra),
                    vacancy,
                    dumbbell,
                    cell,
                    reactant_energy,
                );
                if end.recombined || end.separation <= reactant_sep + 0.08 || !distinct_end(&away, &end) {
                    continue;
                }
                away.push(end);
            }
        }
    }
    eprintln!(
        "shell {label} {orientation} sep {reactant_sep:.3} away {}",
        away.len()
    );
    if away_shell {
        away.sort_by(|left, right| right.separation.partial_cmp(&left.separation).unwrap_or(std::cmp::Ordering::Equal));
        for end in away.into_iter().take(2) {
            hops.push(ThirteenHop {
                shell: label.clone(),
                orientation: orientation.clone(),
                mechanism: "away",
                sites,
                reactant_sep,
                end_sep: end.separation,
                reaction: end.reaction,
                job: hop_job("away", &label, &orientation, lattice, &reactant, &end, cell),
            });
        }
    }
    let translation = tangential.into_iter().find(|end| {
        let shift = dist(centroid(&end.positions, dumbbell), centroid(&reactant, dumbbell));
        shift > 1.5 && axis_overlap(dumbbell_axis(&end.positions, dumbbell, cell), axis) > 0.7
    });
    if let Some(end) = translation {
        hops.push(ThirteenHop {
            shell: label.clone(),
            orientation: orientation.clone(),
            mechanism: "translation",
            sites,
            reactant_sep,
            end_sep: end.separation,
            reaction: end.reaction,
            job: hop_job("translation", &label, &orientation, lattice, &reactant, &end, cell),
        });
    }
    eprintln!(
        "shell {label} {orientation} built {} hops, reactant sep {reactant_sep:.3}",
        hops.len()
    );
    hops
}

fn isolated_hops(copper: &FastCopper, repeats: usize, lattice: f64) -> Vec<ThirteenHop> {
    let axis = [1.0, 0.0, 0.0];
    let built = build_oriented(repeats, lattice, [lattice, 0.0, 0.0], axis, false);
    let length = repeats as f64 * lattice;
    let cell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut reactant = built.positions;
    relax_fast(copper, &mut reactant, cell, 1.0e-3, 80);
    let dumbbell = built.dumbbell;
    let energy = copper.evaluate(reactant.view(), cell).0;
    let origin = centroid(&reactant, dumbbell);
    let mut hops = Vec::new();
    for step in fcc_neighbor_steps(lattice) {
        let end = quench_end(
            copper,
            shift_dumbbell(&reactant, dumbbell, step),
            origin,
            dumbbell,
            cell,
            energy,
        );
        let shift = dist(centroid(&end.positions, dumbbell), origin);
        let now = dumbbell_axis(&end.positions, dumbbell, cell);
        eprintln!(
            "isolated translation try shift {shift:.3} axis {}",
            axis_name(now)
        );
        if shift > 2.0 && axis_overlap(now, axis) > 0.75 {
            hops.push(ThirteenHop {
                shell: "isolated".into(),
                orientation: "100".into(),
                mechanism: "translation",
                sites: 1,
                reactant_sep: 0.0,
                end_sep: shift,
                reaction: end.reaction,
                job: hop_job("translation", "isolated", "100", lattice, &reactant, &end, cell),
            });
            break;
        }
    }
    let rotated = quench_end(
        copper,
        rotate_dumbbell(&reactant, dumbbell, [1.0, 1.0, 0.0]),
        origin,
        dumbbell,
        cell,
        energy,
    );
    let shift = dist(centroid(&rotated.positions, dumbbell), origin);
    let now = dumbbell_axis(&rotated.positions, dumbbell, cell);
    eprintln!(
        "isolated rotation quench shift {shift:.3} axis {} overlap110 {:.3}",
        axis_name(now),
        axis_overlap(now, [1.0, 1.0, 0.0])
    );
    if shift < 0.5 && axis_overlap(now, [1.0, 1.0, 0.0]) > 0.75 {
        hops.push(ThirteenHop {
            shell: "isolated".into(),
            orientation: "100".into(),
            mechanism: "rotation",
            sites: 1,
            reactant_sep: 0.0,
            end_sep: shift,
            reaction: rotated.reaction,
            job: hop_job("rotation", "isolated", "100", lattice, &reactant, &rotated, cell),
        });
    } else if let Some(end) = dimer_rotation(copper, &reactant, dumbbell, cell, [1.0, 1.0, 0.0]) {
        let end_shift = dist(centroid(&end.positions, dumbbell), origin);
        let now = dumbbell_axis(&end.positions, dumbbell, cell);
        eprintln!(
            "isolated dimer rotation shift {end_shift:.3} axis {}",
            axis_name(now)
        );
        if end_shift < 0.5 && axis_overlap(now, axis) < 0.85 {
        hops.push(ThirteenHop {
            shell: "isolated".into(),
            orientation: "100".into(),
            mechanism: "rotation",
            sites: 1,
            reactant_sep: 0.0,
            end_sep: end_shift,
            reaction: end.reaction,
            job: hop_job("rotation", "isolated", "100", lattice, &reactant, &end, cell),
        });
        }
    }
    hops
}

fn dimer_rotation(
    copper: &FastCopper,
    reactant: &Array1<f64>,
    dumbbell: [usize; 2],
    cell: [[f64; 3]; 3],
    target: [f64; 3],
) -> Option<QuenchedEnd> {
    let center = centroid(reactant, dumbbell);
    let half = dist(point_of(reactant, dumbbell[0]), center).max(0.8);
    let target = unit3(target);
    let mut kick = Array1::zeros(reactant.len());
    for (atom, sign) in dumbbell.into_iter().zip([-1.0, 1.0]) {
        let here = point_of(reactant, atom);
        for c in 0..3 {
            kick[3 * atom + c] = 0.55 * (center[c] + sign * half * target[c] - here[c]);
        }
    }
    let mode_norm = kick.dot(&kick).sqrt();
    if mode_norm < 1.0e-6 {
        return None;
    }
    let mode = &kick / mode_norm;
    let search = MinModeConfig {
        force_tol: 5.0e-3,
        max_move: 0.1,
        dr: 1.0e-3,
        max_rotations: 8,
        ..MinModeConfig::default()
    };
    let config = SolidDimerConfig::frozen(search)
        .with_index_check(IndexCheck::first_order().with_step(1.0e-3).with_escapes(2));
    let mut dimer = SolidDimer::new(config, reactant + &kick, cell, mode).ok()?;
    let held = FastHeld { copper, cell };
    let report = dimer.run(&held, 120).ok()?;
    if report.status != MinModeStatus::Converged || report.rejected || report.negative_modes != Some(1) {
        eprintln!("isolated rotation dimer {:?}", report.status);
        return None;
    }
    let saddle = dimer.positions().to_owned();
    let unstable = dimer.mode().to_owned();
    let mut peak = 0.0_f64;
    for atom in 0..saddle.len() / 3 {
        let mut acc = 0.0;
        for c in 0..3 {
            acc += unstable[3 * atom + c] * unstable[3 * atom + c];
        }
        peak = peak.max(acc.sqrt());
    }
    if peak < 1.0e-8 {
        return None;
    }
    let step = &unstable * (0.45 / peak);
    let mut plus = &saddle + &step;
    let mut minus = &saddle - &step;
    relax_fast(copper, &mut plus, cell, 1.5e-3, 60);
    relax_fast(copper, &mut minus, cell, 1.5e-3, 60);
    let product = if hausdorff(&plus, reactant, cell) < hausdorff(&minus, reactant, cell) {
        minus
    } else {
        plus
    };
    let energy = copper.evaluate(reactant.view(), cell).0;
    Some(QuenchedEnd {
        separation: 0.0,
        reaction: copper.evaluate(product.view(), cell).0 - energy,
        recombined: false,
        positions: product,
    })
}

struct ThirteenRow {
    shell: String,
    orientation: String,
    mechanism: String,
    cell: String,
    n_atoms: usize,
    barrier: f64,
    end_sep: f64,
    reactant_sep: f64,
    force: f64,
    calls: usize,
    steps: usize,
    converged: bool,
}

#[test]
fn fast_eam_matches_a_sheared_cell() {
    let (positions, _) = fcc(2, 3.628311);
    let cell = [
        [7.3, 0.15, -0.08],
        [0.04, 7.2, 0.12],
        [-0.05, 0.07, 7.4],
    ];
    let slow = eam_eval(positions.view(), cell);
    let fast = eam_eval_fast(positions.view(), cell);
    assert!((slow.0 - fast.0).abs() < 1.0e-8, "energy {} vs {}", slow.0, fast.0);
    for (left, right) in slow.1.iter().zip(fast.1.iter()) {
        assert!((left - right).abs() < 1.0e-8, "force {left} vs {right}");
    }
}

#[test]
#[ignore = "13x13x13 300 K shells inside 7 A; writes benchmarks/ssneb_shell_300K_13.csv"]
fn ssneb_shell_300k_13() {
    let lattice = 3.628311_f64;
    let repeats = 13usize;
    let copper = FastCopper::new();
    let (sample, sample_cell) = fcc(repeats, lattice);
    let started = Instant::now();
    let _ = copper.evaluate(sample.view(), sample_cell);
    eprintln!("13^3 eval {:.3} s, atoms {}", started.elapsed().as_secs_f64(), sample.len() / 3);
    let shells: Vec<_> = shell_table(lattice, repeats)
        .into_iter()
        .filter(|(radius, _, _)| *radius < 7.0)
        .collect();
    assert!(shells.len() >= 5, "expected the neighbor shells inside 7 A");
    let mut hops: Vec<ThirteenHop> = shells
        .par_iter()
        .flat_map(|(radius, vector, sites)| {
            let _ = radius;
            let local = FastCopper::new();
            shell_hops(&local, repeats, lattice, *vector, *sites)
        })
        .collect();
    hops.extend(isolated_hops(&copper, repeats, lattice));
    assert!(hops.iter().any(|hop| hop.mechanism == "rotation" && hop.shell == "isolated"));
    assert!(hops.iter().any(|hop| hop.mechanism == "away"), "no 4.456 away end");
    let measured: Vec<_> = hops
        .par_iter()
        .map(|hop| {
            if hop.mechanism == "capture" && hop.reaction == 0.0 && (hop.end_sep - hop.reactant_sep).abs() < 1.0e-8 {
                let zero = BandStat {
                    barrier: 0.0,
                    force: 0.0,
                    steps: 0,
                    calls: 0,
                    converged: true,
                };
                return (hop, zero.clone(), zero);
            }
            let local = FastCopper::new();
            let (fixed, free) = size_pair(&local, &hop.job);
            (hop, fixed, free)
        })
        .collect();
    let mut rows = Vec::new();
    for (hop, fixed, free) in &measured {
        for (mode, stat) in [("fixed", fixed), ("free", free)] {
            rows.push(ThirteenRow {
                shell: hop.shell.clone(),
                orientation: hop.orientation.clone(),
                mechanism: hop.mechanism.to_string(),
                cell: mode.into(),
                n_atoms: hop.job.reactant.len() / 3,
                barrier: stat.barrier,
                end_sep: hop.end_sep,
                reactant_sep: hop.reactant_sep,
                force: stat.force,
                calls: stat.calls,
                steps: stat.steps,
                converged: stat.converged,
            });
        }
    }
    let reference = |mechanism: &str, cell: &str| -> Option<f64> {
        rows.iter()
            .find(|row| row.shell == "isolated" && row.mechanism == mechanism && row.cell == cell && row.converged)
            .map(|row| row.barrier)
    };
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_shell_300K_13.csv");
    let mut file = File::create(&path).expect("300 K 13^3 csv");
    writeln!(
        file,
        "shell_A,orientation,mechanism,cell,n_atoms,barrier_eV,end_sep_A,reactant_sep_A,max_force,force_calls,steps,converged,free_barrier_eV,barrier_minus_free_eV,rotation_below_translation"
    )
    .unwrap();
    for row in &rows {
        let free_kind = if row.mechanism == "rotation" { "rotation" } else { "translation" };
        let free_barrier = if row.shell == "isolated" { None } else { reference(free_kind, &row.cell) };
        let translation = reference("translation", &row.cell);
        let below = if row.mechanism == "rotation" && row.shell != "isolated" && row.converged {
            translation.map(|value| if row.barrier < value { "1" } else { "0" })
        } else {
            None
        };
        let delta = free_barrier.map(|value| row.barrier - value);
        writeln!(
            file,
            "{shell},{orientation},{mechanism},{cell},{n},{barrier:.6},{end:.3},{reactant:.3},{force:.6e},{calls},{steps},{ok},{free},{minus},{below}",
            shell = row.shell,
            orientation = row.orientation,
            mechanism = row.mechanism,
            cell = row.cell,
            n = row.n_atoms,
            barrier = row.barrier,
            end = row.end_sep,
            reactant = row.reactant_sep,
            force = row.force,
            calls = row.calls,
            steps = row.steps,
            ok = row.converged as u8,
            free = free_barrier.map(|value| format!("{value:.6}")).unwrap_or_default(),
            minus = delta.map(|value| format!("{value:.6}")).unwrap_or_default(),
            below = below.unwrap_or(""),
        )
        .unwrap();
        eprintln!(
            "{shell} {orientation} {mechanism} {cell} barrier {barrier:.4} end {end:.3} force {force:.3e} below {below}",
            shell = row.shell,
            orientation = row.orientation,
            mechanism = row.mechanism,
            cell = row.cell,
            barrier = row.barrier,
            end = row.end_sep,
            force = row.force,
            below = below.unwrap_or("-"),
        );
    }
    let require = |shell: &str, mechanism: &str, cell: &str| {
        rows.iter().any(|row| {
            row.shell == shell && row.mechanism == mechanism && row.cell == cell && row.converged
        })
    };
    assert!(require("isolated", "translation", "fixed") && require("isolated", "translation", "free"));
    assert!(require("isolated", "rotation", "fixed") && require("isolated", "rotation", "free"));
    let away_fixed = rows
        .iter()
        .filter(|row| row.shell == "4.444" && row.mechanism == "away" && row.cell == "fixed" && row.converged)
        .count();
    assert!(away_fixed >= 2, "4.444 A away ends converged in fixed cell: {away_fixed}");
}

#[test]
#[ignore = "nearest 4.444 away ends and the isolated translation on the 13^3 cell"]
fn ssneb_300k_13_finish() {
    let lattice = 3.628311_f64;
    let repeats = 13usize;
    let copper = FastCopper::new();
    let mut jobs = Vec::new();
    let shell = shell_vector(lattice, repeats, 1.224744871);
    let (reactant, vacancy, dumbbell, cell, sep, gone) =
        shell_frame(&copper, repeats, lattice, shell, [1.0, 0.0, 0.0]);
    assert!(!gone, "4.444 shell recombined");
    let energy = copper.evaluate(reactant.view(), cell).0;
    let mut away = Vec::new();
    for step in fcc_neighbor_steps(lattice) {
        let end = quench_end(
            &copper,
            shift_dumbbell(&reactant, dumbbell, step),
            vacancy,
            dumbbell,
            cell,
            energy,
        );
        eprintln!(
            "4.444 try sep {:.3}->{:.3} dE {:.4} recombined {}",
            sep, end.separation, end.reaction, end.recombined as u8
        );
        if end.recombined || end.separation < sep + 0.08 || end.reaction.abs() > 1.0 {
            continue;
        }
        if distinct_end(&away, &end) {
            away.push(end);
        }
    }
    away.sort_by(|left, right| left.separation.partial_cmp(&right.separation).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("4.444 kept {} away ends", away.len());
    for end in away.into_iter().take(2) {
        jobs.push(ThirteenHop {
            shell: "4.444".into(),
            orientation: "100".into(),
            mechanism: "away",
            sites: 24,
            reactant_sep: sep,
            end_sep: end.separation,
            reaction: end.reaction,
            job: hop_job("away", "4.444", "100", lattice, &reactant, &end, cell),
        });
    }
    let axis = [1.0, 0.0, 0.0];
    let built = build_oriented(repeats, lattice, [lattice, 0.0, 0.0], axis, false);
    let length = repeats as f64 * lattice;
    let icell = [[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]];
    let mut isolated = built.positions;
    relax_fast(&copper, &mut isolated, icell, 1.0e-3, 80);
    let idumb = built.dumbbell;
    let ienergy = copper.evaluate(isolated.view(), icell).0;
    let origin = centroid(&isolated, idumb);
    let mut translated = None;
    for step in fcc_neighbor_steps(lattice) {
        let end = quench_end(
            &copper,
            shift_dumbbell(&isolated, idumb, step),
            origin,
            idumb,
            icell,
            ienergy,
        );
        let shift = dist(centroid(&end.positions, idumb), origin);
        let now = dumbbell_axis(&end.positions, idumb, icell);
        eprintln!(
            "isolated try shift {shift:.3} axis {} dE {:.4}",
            axis_name(now),
            end.reaction
        );
        let reaction = end.reaction.abs();
        let better = translated.as_ref().is_none_or(|(_, _, best)| reaction < *best);
        if shift > 2.0 && axis_overlap(now, axis) > 0.75 && better {
            translated = Some((end, shift, reaction));
        }
    }
    let (end, shift, _) = translated.expect("isolated translation end");
    eprintln!("isolated translation chosen shift {shift:.3} dE {:.4}", end.reaction);
    jobs.push(ThirteenHop {
        shell: "isolated".into(),
        orientation: "100".into(),
        mechanism: "translation",
        sites: 1,
        reactant_sep: 0.0,
        end_sep: shift,
        reaction: end.reaction,
        job: hop_job("translation", "isolated", "100", lattice, &isolated, &end, icell),
    });
    let mode = &end.positions - &isolated;
    let mid = blend_positions(&isolated, icell, &end.positions, icell, icell, 0.5);
    let mechanism = capture_mechanism(
        &copper,
        &isolated,
        icell,
        idumb,
        &mid,
        &end.positions,
        &mode,
        end.reaction,
    );
    let center = centroid(&reactant, dumbbell);
    let mapped = mapped_hops(&reactant, cell, center, [1.0, 0.0, 0.0], &[mechanism]);
    let mut mapped_away = Vec::new();
    for hop in mapped {
        let mut product = hop.product;
        relax_fast(&copper, &mut product, cell, 1.5e-3, 60);
        let after = rgsaddle::pair_separation(product.view(), cell, vacancy, &dumbbell).unwrap_or(sep);
        let reaction = copper.evaluate(product.view(), cell).0 - energy;
        let recombined = pair_recombined(&product, vacancy, dumbbell, cell);
        eprintln!("mapped sep {sep:.3}->{after:.3} dE {reaction:.4} recombined {}", recombined as u8);
        if recombined || after < sep + 0.08 || reaction.abs() > 0.5 {
            continue;
        }
        let end = QuenchedEnd {
            positions: product,
            separation: after,
            reaction,
            recombined: false,
        };
        if distinct_end(&mapped_away, &end) {
            mapped_away.push(end);
        }
    }
    mapped_away.sort_by(|left, right| {
        left.separation.partial_cmp(&right.separation).unwrap_or(std::cmp::Ordering::Equal)
    });
    for end in mapped_away.into_iter().take(2) {
        jobs.push(ThirteenHop {
            shell: "4.444".into(),
            orientation: "100".into(),
            mechanism: "away",
            sites: 24,
            reactant_sep: sep,
            end_sep: end.separation,
            reaction: end.reaction,
            job: hop_job("away", "4.444", "100", lattice, &reactant, &end, cell),
        });
    }
    let measured: Vec<_> = jobs
        .par_iter()
        .map(|hop| {
            let local = FastCopper::new();
            let (fixed, free) = size_pair(&local, &hop.job);
            (hop, fixed, free)
        })
        .collect();
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_shell_300K_13_finish.csv");
    let mut file = File::create(&path).expect("finish csv");
    writeln!(
        file,
        "shell_A,orientation,mechanism,cell,n_atoms,barrier_eV,end_sep_A,reactant_sep_A,max_force,force_calls,steps,converged,reaction_eV"
    )
    .unwrap();
    for (hop, fixed, free) in &measured {
        for (mode, stat) in [("fixed", fixed), ("free", free)] {
            writeln!(
                file,
                "{shell},{orientation},{mechanism},{mode},{n},{barrier:.6},{end:.3},{reactant:.3},{force:.6e},{calls},{steps},{ok},{reaction:.4}",
                shell = hop.shell,
                orientation = hop.orientation,
                mechanism = hop.mechanism,
                n = hop.job.reactant.len() / 3,
                barrier = stat.barrier,
                end = hop.end_sep,
                reactant = hop.reactant_sep,
                force = stat.force,
                calls = stat.calls,
                steps = stat.steps,
                ok = stat.converged as u8,
                reaction = hop.reaction,
            )
            .unwrap();
            eprintln!(
                "{shell} {mechanism} {mode} {barrier:.4} end {end:.3} force {force:.3e} ok {ok}",
                shell = hop.shell,
                mechanism = hop.mechanism,
                barrier = stat.barrier,
                end = hop.end_sep,
                force = stat.force,
                ok = stat.converged as u8,
            );
        }
    }
}
