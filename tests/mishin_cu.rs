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
    SaddleError, SaddleSeed, SolidDimer, SolidDimerConfig, Strain, StrainResponse, Trust,
    TrustPolicy, cubic_rotations, index_at,
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
    if cell[0][1].abs() + cell[0][2].abs() + cell[1][0].abs() + cell[1][2].abs() + cell[2][0].abs()
        + cell[2][1].abs()
        > 1.0e-8
    {
        return eam_eval(positions, cell);
    }
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
    for i in 0..n {
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
                                pairs.push(Pair { i, j, d, r: r2.sqrt() });
                            }
                        }
                    }
                }
            }
        }
    }
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
    keep.push([sia_pos[0] - offset, sia_pos[1], sia_pos[2]]);
    keep.push([sia_pos[0] + offset, sia_pos[1], sia_pos[2]]);
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
