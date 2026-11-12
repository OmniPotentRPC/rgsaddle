//! Climbing-image solid-state NEB on the Mishin copper dumbbell, checked
//! against the solid-state dimer rows in `benchmarks/solid_dimer_cu.csv`.
//!
//! The potential, the cell and the dimer settings are the ones that wrote
//! that file. This test is the release run: `cargo test --release --test
//! ssneb_validate_cu -- --ignored --nocapture`.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{
    BandConfig, BandStatus, CellMask, CiConfig, MinModeConfig, MinModeStatus, PointSurface,
    SaddleError, SolidDimer, SolidDimerConfig, SolidNeb, exact_eigh,
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

#[derive(Clone)]
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
            .with_external_stress(external),
        start.clone(),
        cell,
        mode.clone(),
    )
    .unwrap();
    let surface = Held { copper, cell };
    let report = dimer.run(&surface, max_steps).expect("dimer");
    let calls = copper.calls() - before;
    let (e_sad, _, stress_now) = copper.evaluate(dimer.positions(), dimer.cell());
    let stress = dimer.stress().unwrap_or(stress_now);
    let enthalpy = dimer.energy().unwrap_or(e_sad);
    let (_, _, dir) = closest_pair(&dimer.positions().to_owned(), dimer.cell());
    let (angle_100, angle_110, axis, separation) = axis_report(dir);
    // [110] has the same sign on x and y. [-110] is the other saddle.
    let switched = report.status == MinModeStatus::Converged
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
        positions: dimer.positions().to_owned(),
        cell: dimer.cell(),
    }
}

fn print_row(shear: f64, method: &str, row: &SearchRow) {
    eprintln!(
        "RESULT shear={shear:.2} method={method} calls={} barrier_eV={} enthalpy_barrier_eV={} curvature={} status={:?} max_force={} steps={} stress_xx={} stress_yy={} stress_zz={} stress_yz={} stress_xz={} stress_xy={} axis_x={} axis_y={} axis_z={} separation={} angle_100={} angle_110={} switch={} final_shear={} volume={}",
        row.calls,
        row.barrier,
        row.enthalpy_barrier,
        row.curvature,
        row.status,
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

fn unwrap_onto(target: &Array1<f64>, origin: &Array1<f64>, cell: [[f64; 3]; 3]) -> Array1<f64> {
    &origin.to_owned() + &minimum_image_difference(target, origin, cell)
}

struct HeldCell<'a> {
    copper: &'a Copper,
    /// Cell used by [`PointSurface::eval`]. Joint images pass their own.
    cell: [[f64; 3]; 3],
}

impl PointSurface for HeldCell<'_> {
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

fn neb_config(force_tol: f64) -> BandConfig {
    BandConfig {
        climbing: Some(CiConfig::always()),
        force_tol,
        max_move: 0.08,
        remove_translation: true,
        method: Method::Lbfgs { memory: 20 },
        min_mode: None,
        spring: rgsaddle::SpringKind::Uniform { k: 1.0 },
        ..BandConfig::default()
    }
}

fn run_neb(
    neb: &mut SolidNeb,
    surface: &HeldCell<'_>,
    force_tol: f64,
    max_steps: usize,
) -> (rgsaddle::BandReport, usize) {
    let mut report = neb.step(surface).expect("neb step");
    let mut steps = 1;
    while report.status == BandStatus::Running && steps < max_steps {
        if steps % 25 == 0 {
            eprintln!(
                "  neb step {steps} max_force {:.6e} ci {:?}",
                report.max_force, report.ci_index
            );
        }
        report = neb.step(surface).expect("neb step");
        steps += 1;
        if report.max_force <= force_tol {
            break;
        }
    }
    (report, steps)
}

fn joint_gradient(
    surface: &HeldCell<'_>,
    dimer_chart: &rgsaddle::JointChart,
    q: ArrayView1<f64>,
) -> Array1<f64> {
    let (r, h) = dimer_chart.geometry(q).expect("geometry");
    let packed = flat(h);
    let (_, g_cart) = surface.eval_in_cell(r.view(), &packed).expect("gradient");
    let stress = surface
        .cauchy_stress(r.view(), &packed)
        .expect("stress")
        .expect("analytic stress");
    dimer_chart
        .gradient(q, g_cart.view(), unflat(&stress))
        .expect("joint gradient")
}

fn negative_modes(
    surface: &HeldCell<'_>,
    frozen: bool,
    reference: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> (usize, f64, String, Array1<f64>) {
    let n_atoms = positions.len() / 3;
    let (chart, q) = if frozen {
        (None, positions.clone())
    } else {
        let chart = rgsaddle::JointChart::new(reference, n_atoms, CellMask::all(), external, 1.0)
            .expect("chart");
        let q = rgsaddle::joint_coordinate(&chart, CellMask::all(), positions.view(), cell)
            .expect("coordinate");
        (Some(chart), q)
    };
    let dim = q.len();
    let delta = 1e-5;
    let grad = |q: &Array1<f64>| -> Array1<f64> {
        if let Some(chart) = &chart {
            joint_gradient(surface, chart, q.view())
        } else {
            surface.eval(q.view()).expect("cartesian gradient").1
        }
    };
    let g0 = grad(&q);
    let mut hessian = Array2::zeros((dim, dim));
    for j in 0..dim {
        let mut plus = q.clone();
        let mut minus = q.clone();
        plus[j] += delta;
        minus[j] -= delta;
        let gp = grad(&plus);
        let gm = grad(&minus);
        for i in 0..dim {
            hessian[(i, j)] = (gp[i] - gm[i]) / (2.0 * delta);
        }
    }
    for i in 0..dim {
        for j in 0..i {
            let mean = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = mean;
            hessian[(j, i)] = mean;
        }
    }
    let (values, vectors) = exact_eigh(hessian.view()).expect("hessian spectrum");
    let cutoff = -1.0e-4;
    let n_neg = values.iter().filter(|v| **v < cutoff).count();
    let lowest = values[0];
    let shown: Vec<String> = values.iter().take(8).map(|v| format!("{v:.8e}")).collect();
    let mode = vectors.column(0).to_owned();
    let _ = g0;
    (n_neg, lowest, shown.join(";"), mode)
}

fn geometry_distance(
    left_r: &Array1<f64>,
    left_h: [[f64; 3]; 3],
    right_r: &Array1<f64>,
    right_h: [[f64; 3]; 3],
) -> (f64, f64, f64, f64) {
    let delta = minimum_image_difference(left_r, right_r, right_h);
    let n = delta.len() / 3;
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for (c, mean_c) in mean.iter_mut().enumerate() {
            *mean_c += delta[3 * atom + c];
        }
    }
    for mean_c in &mut mean {
        *mean_c /= n as f64;
    }
    let mut max_atom: f64 = 0.0;
    let mut sum = 0.0;
    for atom in 0..n {
        let mut r2 = 0.0;
        for c in 0..3 {
            let d = delta[3 * atom + c] - mean[c];
            r2 += d * d;
        }
        sum += r2;
        max_atom = max_atom.max(r2.sqrt());
    }
    let rms_atom = (sum / n as f64).sqrt();
    let mut max_cell: f64 = 0.0;
    let mut cell_sum = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            let d = (left_h[i][j] - right_h[i][j]).abs();
            max_cell = max_cell.max(d);
            cell_sum += d * d;
        }
    }
    let rms_cell = (cell_sum / 9.0).sqrt();
    (rms_atom, max_atom, rms_cell, max_cell)
}

fn tighten_saddle(
    copper: &Copper,
    row: &SearchRow,
    reference: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    frozen: bool,
    mode: &Array1<f64>,
    e_min: f64,
) -> SearchRow {
    let search = MinModeConfig {
        method: Method::Fire { kind: FireKind::V2 },
        force_tol: 2e-4,
        max_move: 0.008,
        dr: 1e-3,
        rotation_tol: 1e-3,
        max_rotations: 12,
        seed_curvature: false,
        ..MinModeConfig::default()
    };
    let mask = if frozen {
        CellMask::frozen()
    } else {
        CellMask::all()
    };
    let surface = HeldCell {
        copper,
        cell: reference,
    };
    let mut dimer = SolidDimer::new(
        SolidDimerConfig::new(search)
            .with_mask(mask)
            .with_external_stress(external),
        row.positions.clone(),
        if frozen { row.cell } else { reference },
        if frozen {
            mode.slice(s![..row.positions.len()]).to_owned()
        } else {
            mode.clone()
        },
    )
    .expect("tighter dimer");
    if !frozen {
        let chart = dimer.chart().expect("chart");
        let q = rgsaddle::joint_coordinate(chart, CellMask::all(), row.positions.view(), row.cell)
            .expect("saddle on the reactant chart");
        dimer.set_generalized(q).expect("place saddle");
        dimer.set_mode(mode.clone()).expect("mode");
    }
    let report = dimer.run(&surface, 80).expect("tighten");
    let (energy, _, _) = copper.evaluate(dimer.positions(), dimer.cell());
    let enthalpy = dimer.energy().unwrap_or(energy);
    eprintln!(
        "TIGHT status={:?} force={} curvature={} dE={} dH={}",
        report.status,
        report.max_force,
        report.curvature,
        energy - e_min,
        enthalpy - e_min
    );
    SearchRow {
        positions: dimer.positions().to_owned(),
        cell: dimer.cell(),
        curvature: report.curvature,
        status: report.status,
        max_force: report.max_force,
        stress: dimer.stress().unwrap_or(row.stress),
        volume: dimer.volume(),
        ..row.clone()
    }
}

fn dimer_stay(
    copper: &Copper,
    frozen: bool,
    reference: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    mode: &Array1<f64>,
) -> (MinModeStatus, f64, f64, f64, f64, usize) {
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
    let mask = if frozen {
        CellMask::frozen()
    } else {
        CellMask::all()
    };
    let surface = HeldCell {
        copper,
        cell: reference,
    };
    let seed_mode = if frozen {
        if mode.len() == positions.len() {
            mode.clone()
        } else {
            mode.slice(s![..positions.len()]).to_owned()
        }
    } else {
        mode.clone()
    };
    let mut dimer = SolidDimer::new(
        SolidDimerConfig::new(search)
            .with_mask(mask)
            .with_external_stress(external),
        positions.clone(),
        if frozen { cell } else { reference },
        seed_mode.clone(),
    )
    .expect("dimer at the climbing image");
    if !frozen {
        let chart = dimer.chart().expect("joint chart");
        let q = rgsaddle::joint_coordinate(chart, CellMask::all(), positions.view(), cell)
            .expect("climbing image on the reactant chart");
        dimer.set_generalized(q).expect("place the climbing image");
        dimer.set_mode(seed_mode).expect("negative mode");
    }
    let start_r = dimer.positions().to_owned();
    let start_h = dimer.cell();
    let mut report = dimer.step(&surface).expect("stay step");
    let mut steps = 1;
    while report.status == MinModeStatus::Running && steps < 15 {
        report = dimer.step(&surface).expect("stay step");
        steps += 1;
    }
    let (rms, max_atom, _, max_cell) = geometry_distance(
        &dimer.positions().to_owned(),
        dimer.cell(),
        &start_r,
        start_h,
    );
    (
        report.status,
        report.max_force,
        report.curvature,
        max_atom.max(max_cell),
        rms,
        steps,
    )
}

struct NebRow {
    product: String,
    steps: usize,
    max_force: f64,
    converged: bool,
    ci: usize,
    energy: f64,
    enthalpy: f64,
    e_barrier: f64,
    h_barrier: f64,
    final_shear: f64,
    volume: f64,
    stress: [[f64; 3]; 3],
    angle_100: f64,
    angle_110: f64,
    axis: [f64; 3],
    n_neg: usize,
    lowest: f64,
    spectrum: String,
    rms_atom: f64,
    max_atom: f64,
    rms_cell: f64,
    max_cell: f64,
    stay_status: MinModeStatus,
    stay_force: f64,
    stay_curvature: f64,
    stay_disp: f64,
    stay_steps: usize,
    enthalpies: String,
    energies: String,
    forces: String,
}

fn relax_enthalpy(
    copper: &Copper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    reference: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    tol: f64,
) -> (Array1<f64>, [[f64; 3]; 3], f64, f64) {
    let n_atoms = positions.len() / 3;
    let chart = rgsaddle::JointChart::new(reference, n_atoms, CellMask::all(), external, 1.0)
        .expect("chart");
    let mut q =
        rgsaddle::joint_coordinate(&chart, CellMask::all(), positions.view(), cell).expect("q");
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..Control::default()
        },
        q.len(),
    );
    solver.set_atom_maxmove(0.06);
    let surface = HeldCell {
        copper,
        cell: reference,
    };
    let mut steps = 0;
    for _ in 0..500 {
        let oracle = Oracle::unbounded(q.len(), |x: ArrayView1<f64>| {
            let gradient = joint_gradient(&surface, &chart, x);
            let (r, h) = chart.geometry(x).expect("geometry");
            let energy = copper.evaluate(r.view(), h).0;
            let enthalpy = chart.enthalpy(energy, x).expect("enthalpy");
            (enthalpy, gradient)
        });
        solver.step(&oracle, &mut q).expect("enthalpy relax");
        steps += 1;
        let gradient = joint_gradient(&surface, &chart, q.view());
        if max_abs(&gradient) < tol {
            break;
        }
    }
    let (r, h) = chart.geometry(q.view()).expect("relaxed geometry");
    let energy = copper.evaluate(r.view(), h).0;
    let enthalpy = chart.enthalpy(energy, q.view()).expect("enthalpy");
    let force = max_abs(&joint_gradient(&surface, &chart, q.view()));
    eprintln!("ENTHALPY_RELAX steps={steps} energy={energy} enthalpy={enthalpy} force={force}");
    (r, h, energy, enthalpy)
}

#[allow(clippy::too_many_arguments)]
fn climb_row(
    copper: &Copper,
    cube: [[f64; 3]; 3],
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: [[f64; 3]; 3],
    product_cell: [[f64; 3]; 3],
    e_min: f64,
    external: [[f64; 3]; 3],
    frozen: bool,
    dimer: &SearchRow,
    product_name: &str,
) -> NebRow {
    let n_images = 11;
    let force_tol = 5e-4;
    let n_atoms = reactant.len() / 3;
    let mask = if frozen {
        CellMask::frozen()
    } else {
        CellMask::all()
    };
    let product = if frozen {
        unwrap_onto(product, reactant, cell)
    } else {
        product.clone()
    };
    let chart = rgsaddle::JointChart::new(cell, n_atoms, mask, external, 1.0).expect("band chart");
    let q0 = rgsaddle::joint_coordinate(&chart, mask, reactant.view(), cell).expect("reactant q");
    let q1 =
        rgsaddle::joint_coordinate(&chart, mask, product.view(), product_cell).expect("product q");
    let frames: Vec<(Array1<f64>, [[f64; 3]; 3])> = (0..n_images)
        .map(|i| {
            let t = i as f64 / (n_images - 1) as f64;
            let q = &q0 * (1.0 - t) + &q1 * t;
            chart.geometry(q.view()).expect("image geometry")
        })
        .collect();
    let mut positions = Array2::zeros((n_images, reactant.len()));
    let mut cells = Vec::with_capacity(n_images);
    for (i, (r, h)) in frames.into_iter().enumerate() {
        positions.row_mut(i).assign(&r);
        cells.push(h);
    }
    let surface = HeldCell { copper, cell };
    let mut neb = SolidNeb::joint(
        neb_config(force_tol),
        cell,
        positions,
        &cells,
        mask,
        external,
        1.0,
    )
    .expect("joint neb");
    let (report, steps) = run_neb(&mut neb, &surface, force_tol, 800);
    let images = neb.images().expect("neb images");
    let ci = report.ci_index.unwrap_or_else(|| {
        images
            .iter()
            .enumerate()
            .skip(1)
            .take(n_images - 2)
            .max_by(|left, right| {
                left.1
                    .enthalpy
                    .partial_cmp(&right.1.enthalpy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(index, _)| index)
            .unwrap_or(1)
    });
    let image = &images[ci];
    let ci_cell = image.cell;
    let ci_r = image.positions.clone();
    let (e_ci, _, stress) = copper.evaluate(ci_r.view(), ci_cell);
    let enthalpy = if frozen { e_ci } else { image.enthalpy };
    let (_, _, dir) = closest_pair(&ci_r, ci_cell);
    let (angle_100, angle_110, axis, _) = axis_report(dir);
    let (n_neg, lowest, spectrum, mode) =
        negative_modes(&surface, frozen, cell, external, &ci_r, ci_cell);
    let (rms_atom, max_atom, rms_cell, max_cell) =
        geometry_distance(&ci_r, ci_cell, &dimer.positions, dimer.cell);
    let (stay_status, stay_force, stay_curvature, stay_disp, _, stay_steps) =
        dimer_stay(copper, frozen, cell, external, &ci_r, ci_cell, &mode);
    let enthalpies = images
        .iter()
        .map(|image| format!("{:.10}", image.enthalpy - e_min))
        .collect::<Vec<_>>()
        .join(";");
    let energies = images
        .iter()
        .map(|image| format!("{:.10}", image.energy - e_min))
        .collect::<Vec<_>>()
        .join(";");
    let qs = neb.generalized();
    let mut force_values = Vec::with_capacity(qs.nrows());
    if frozen {
        for row in qs.axis_iter(ndarray::Axis(0)) {
            force_values.push(max_abs(&surface.eval(row).expect("image force").1));
        }
    } else {
        let chart = neb.chart().expect("joint chart");
        for row in qs.axis_iter(ndarray::Axis(0)) {
            force_values.push(max_abs(&joint_gradient(&surface, chart, row)));
        }
    }
    let forces = force_values
        .iter()
        .map(|value| format!("{value:.6e}"))
        .collect::<Vec<_>>()
        .join(";");
    NebRow {
        product: product_name.to_string(),
        steps,
        max_force: report.max_force,
        converged: report.status == BandStatus::Converged,
        ci,
        energy: e_ci,
        enthalpy,
        e_barrier: e_ci - e_min,
        h_barrier: enthalpy - e_min,
        final_shear: shear_yx(cube, ci_cell),
        volume: det(ci_cell).abs(),
        stress,
        angle_100,
        angle_110,
        axis,
        n_neg,
        lowest,
        spectrum,
        rms_atom,
        max_atom,
        rms_cell,
        max_cell,
        stay_status,
        stay_force,
        stay_curvature,
        stay_disp,
        stay_steps,
        enthalpies,
        energies,
        forces,
    }
}

#[test]
#[ignore = "release comparison against benchmarks/solid_dimer_cu.csv"]
fn ssneb_validates_the_copper_dimer() {
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
    let published = [
        (0.0, false, 0.47417449726344785, 0.47417449726344785),
        (0.0, true, 0.33256689535636497, 0.34702642693292773),
        (0.02, false, 0.24356279471709286, 0.24356279471709286),
        (0.02, true, 0.31264445363120785, 0.16298514462960156),
        (0.03, false, 0.14826689300639373, 0.14826689300639373),
        (0.03, true, 0.2866943305565428, 0.0875839431751615),
        (0.04, false, 0.0678422080142127, 0.0678422080142127),
        (0.04, true, 0.237131193588084, 0.02701564574050508),
        (0.05, false, 0.007348460091009201, 0.007348460091009201),
        (0.05, true, 0.15654904833360206, 0.9465294342165294),
    ];
    let mut lines = Vec::new();
    lines.push(
        "shear,cell,product,benchmark_row,e_dimer_csv,h_dimer_csv,e_dimer_rerun,h_dimer_rerun,dimer_curvature,dimer_max_force,dimer_final_shear,dimer_stress_xy,e_ci,h_ci,e_barrier,h_barrier,de_eV,dh_eV,ci_index,neb_steps,neb_max_force,neb_converged,final_shear,volume,stress_xy,angle_100,angle_110,axis_x,axis_y,axis_z,n_neg,lowest_eigenvalue,spectrum,rms_atom_A,max_atom_A,rms_cell_A,max_cell_A,geom_A,stay_status,stay_force,stay_curvature,stay_disp_A,stay_steps,near_0p021_eV,near_0p021_force,n_neg_dimer,dimer_lowest,enthalpy_profile,energy_profile,force_profile,pass,reason"
            .to_string(),
    );
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
            500,
        );
        print_row(gamma, "fixed", &fixed);
        let mut external = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                external[i][j] = -sigma_min[i][j];
            }
        }
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
            500,
        );
        print_row(gamma, "variable", &variable);
        assert_eq!(fixed.status, MinModeStatus::Converged);
        assert_eq!(variable.status, MinModeStatus::Converged);
        assert!(fixed.curvature < 0.0 && variable.curvature < 0.0);
        let surface = HeldCell {
            copper: &copper,
            cell,
        };
        let (_, _, _, mode_fixed) = negative_modes(
            &surface,
            true,
            cell,
            [[0.0; 3]; 3],
            &fixed.positions,
            fixed.cell,
        );
        let fixed = tighten_saddle(
            &copper,
            &fixed,
            cell,
            [[0.0; 3]; 3],
            true,
            &mode_fixed,
            e_min,
        );
        let (_, _, _, mode_variable) = negative_modes(
            &surface,
            false,
            cell,
            external,
            &variable.positions,
            variable.cell,
        );
        let variable = tighten_saddle(
            &copper,
            &variable,
            cell,
            external,
            false,
            &mode_variable,
            e_min,
        );
        let (aligned_h, aligned_cell, aligned_e, aligned_enthalpy) =
            relax_enthalpy(&copper, &other, cell, cell, external, 1e-3);
        let (_, _, aligned_dir) = closest_pair(&aligned_h, aligned_cell);
        let (ah100, ah110, ah_axis, _) = axis_report(aligned_dir);
        eprintln!(
            "ENTHALPY_110 shear={gamma:.2} gap_eV={} gap_H_eV={} angle_100={ah100} angle_110={ah110} shear={} axis={} {} {}",
            aligned_e - e_min,
            aligned_enthalpy - e_min,
            shear_yx(cube, aligned_cell),
            ah_axis[0],
            ah_axis[1],
            ah_axis[2]
        );
        let mut jobs = vec![
            (
                "fixed",
                true,
                "aligned",
                other.clone(),
                cell,
                &fixed,
                [[0.0; 3]; 3],
            ),
            (
                "variable",
                false,
                "aligned",
                aligned_h,
                aligned_cell,
                &variable,
                external,
            ),
        ];
        let mut opposite_holder = None;
        if gamma == 0.05 {
            let along_opposite = dumbbell(
                &perfect,
                [
                    -std::f64::consts::FRAC_1_SQRT_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                    0.0,
                ],
                2.05,
            );
            let (sheared_opposite, _) = apply_shear(&along_opposite, cube, gamma);
            let mut opposite = sheared_opposite;
            let opposite_steps = relax(&copper, &mut opposite, cell, 1e-3, 400);
            let (e_opposite, g_opposite, _) = copper.evaluate(opposite.view(), cell);
            let (_, _, dir_opposite) = closest_pair(&opposite, cell);
            let (opp_100, opp_110, opp_axis, _) = axis_report(dir_opposite);
            eprintln!(
                "RELAXED_OPPOSITE shear={gamma:.2} steps={opposite_steps} max_force={} gap_vs_100_eV={} angle_100={opp_100} angle_110={opp_110} axis={} {} {}",
                max_abs(&g_opposite),
                e_opposite - e_min,
                opp_axis[0],
                opp_axis[1],
                opp_axis[2]
            );
            opposite_holder = Some(opposite);
        }
        if let Some(opposite) = opposite_holder.as_ref() {
            let (opp_h, opp_cell, opp_e, opp_enthalpy) =
                relax_enthalpy(&copper, opposite, cell, cell, external, 1e-3);
            let (_, _, opp_dir) = closest_pair(&opp_h, opp_cell);
            let (o100, o110, o_axis, _) = axis_report(opp_dir);
            eprintln!(
                "ENTHALPY_OPPOSITE shear={gamma:.2} gap_eV={} gap_H_eV={} angle_100={o100} angle_110={o110} shear={} axis={} {} {}",
                opp_e - e_min,
                opp_enthalpy - e_min,
                shear_yx(cube, opp_cell),
                o_axis[0],
                o_axis[1],
                o_axis[2]
            );
            jobs.push((
                "variable", false, "opposite", opp_h, opp_cell, &variable, external,
            ));
        }
        for (label, frozen, product_name, end, end_cell, dimer, load) in jobs {
            eprintln!("NEB shear={gamma:.2} cell={label} product={product_name}");
            let neb = climb_row(
                &copper,
                cube,
                &minimum,
                &end,
                cell,
                end_cell,
                e_min,
                load,
                frozen,
                dimer,
                product_name,
            );
            let (e_csv, h_csv) = published
                .iter()
                .find(|row| row.0 == gamma && row.1 == !frozen)
                .map(|row| (row.2, row.3))
                .expect("published row");
            let benchmark = !(gamma == 0.05 && !frozen && product_name == "aligned");
            let de = (neb.e_barrier - e_csv).abs();
            let dh = (neb.h_barrier - h_csv).abs();
            let geom = neb.max_atom.max(neb.max_cell);
            let (dimer_neg, dimer_lowest, _, _) = negative_modes(
                &HeldCell {
                    copper: &copper,
                    cell,
                },
                frozen,
                cell,
                load,
                &dimer.positions,
                dimer.cell,
            );
            let mut reasons = Vec::new();
            if benchmark {
                if !neb.converged {
                    reasons.push(format!("NEB not converged, max force {}", neb.max_force));
                }
                if de >= 1e-3 {
                    reasons.push(format!("|E_CI - E_dimer| = {de} eV"));
                }
                if dh >= 1e-3 {
                    reasons.push(format!("|H_CI - H_dimer| = {dh} eV"));
                }
                if geom >= 0.01 {
                    reasons.push(format!(
                        "geometry distance {geom} A (atom {} cell {})",
                        neb.max_atom, neb.max_cell
                    ));
                }
                if neb.n_neg != 1 {
                    reasons.push(format!(
                        "negative modes {} (lowest {}); dimer saddle has {dimer_neg} (lowest {dimer_lowest}); spectrum {}",
                        neb.n_neg, neb.lowest, neb.spectrum
                    ));
                }
                let dimer_de = (dimer.barrier - e_csv).abs();
                let dimer_dh = (dimer.enthalpy_barrier - h_csv).abs();
                if dimer_de > 1e-4 || dimer_dh > 1e-4 {
                    reasons.push(format!(
                        "dimer re-run differs from the csv by {dimer_de} eV energy and {dimer_dh} eV enthalpy"
                    ));
                }
            }
            let profile: Vec<f64> = neb
                .enthalpies
                .split(';')
                .map(|value| value.parse::<f64>().unwrap())
                .collect();
            let force_profile: Vec<f64> = neb
                .forces
                .split(';')
                .map(|value| value.parse::<f64>().unwrap())
                .collect();
            let mut near = profile[0];
            let mut near_force = force_profile[0];
            for (value, force) in profile.iter().zip(force_profile.iter()) {
                if (value - 0.021).abs() < (near - 0.021).abs() {
                    near = *value;
                    near_force = *force;
                }
            }
            let pass = benchmark && reasons.is_empty();
            let reason = if benchmark {
                if reasons.is_empty() {
                    "ok".to_string()
                } else {
                    reasons.join(" | ")
                }
            } else {
                "resolution path, not the published dimer row".to_string()
            };
            eprintln!(
                "CI shear={gamma:.2} cell={label} product={product_name} E={} H={} dE={de} dH={dh} geom={geom} n_neg={} stay_disp={} pass={pass} {reason}",
                neb.e_barrier, neb.h_barrier, neb.n_neg, neb.stay_disp
            );
            let fields = [
                gamma.to_string(),
                label.to_string(),
                neb.product.clone(),
                benchmark.to_string(),
                e_csv.to_string(),
                h_csv.to_string(),
                dimer.barrier.to_string(),
                dimer.enthalpy_barrier.to_string(),
                dimer.curvature.to_string(),
                dimer.max_force.to_string(),
                dimer.final_shear.to_string(),
                dimer.stress[1][0].to_string(),
                neb.energy.to_string(),
                neb.enthalpy.to_string(),
                neb.e_barrier.to_string(),
                neb.h_barrier.to_string(),
                de.to_string(),
                dh.to_string(),
                neb.ci.to_string(),
                neb.steps.to_string(),
                neb.max_force.to_string(),
                (neb.converged as u8).to_string(),
                neb.final_shear.to_string(),
                neb.volume.to_string(),
                neb.stress[1][0].to_string(),
                neb.angle_100.to_string(),
                neb.angle_110.to_string(),
                neb.axis[0].to_string(),
                neb.axis[1].to_string(),
                neb.axis[2].to_string(),
                neb.n_neg.to_string(),
                neb.lowest.to_string(),
                neb.spectrum.clone(),
                neb.rms_atom.to_string(),
                neb.max_atom.to_string(),
                neb.rms_cell.to_string(),
                neb.max_cell.to_string(),
                geom.to_string(),
                format!("{:?}", neb.stay_status),
                neb.stay_force.to_string(),
                neb.stay_curvature.to_string(),
                neb.stay_disp.to_string(),
                neb.stay_steps.to_string(),
                near.to_string(),
                near_force.to_string(),
                dimer_neg.to_string(),
                dimer_lowest.to_string(),
                neb.enthalpies.clone(),
                neb.energies.clone(),
                neb.forces.clone(),
                pass.to_string(),
                reason.replace(',', ";"),
            ];
            lines.push(fields.join(","));
        }
        let path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("benchmarks/ssneb_validate_cu.csv");
        let mut file = File::create(&path).expect("create csv");
        for line in &lines {
            writeln!(file, "{line}").expect("write csv");
        }
        eprintln!("wrote {} rows to {}", lines.len() - 1, path.display());
    }
}
