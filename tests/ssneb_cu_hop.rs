//! Mishin copper <100> to <110> dumbbell hop, fixed cell and free cell.
//!
//! The cell, the separation and the potential are the zero-shear row of
//! `feat/ssneb-defect-check` (`tests/ssneb_validate_cu.rs`, a 3x3x3 cell).
//! That branch's joint-chart driver is not on this develop. The hop runs
//! on the absolute-cell solid-state band that is.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Control, FireKind, Method, Oracle, Solver};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig, SaddleError, SolidState,
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

fn positive_image(sx: i32, sy: i32, sz: i32) -> bool {
    sx > 0 || (sx == 0 && (sy > 0 || (sy == 0 && sz > 0)))
}

fn eam_eval(positions: ArrayView1<f64>, cell: [[f64; 3]; 3]) -> (f64, Array1<f64>, [[f64; 3]; 3]) {
    let n = positions.len() / 3;
    let min_len = [0, 1, 2]
        .map(|row| {
            (cell[row][0] * cell[row][0]
                + cell[row][1] * cell[row][1]
                + cell[row][2] * cell[row][2])
                .sqrt()
        })
        .into_iter()
        .fold(f64::MAX, f64::min);
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
                        let d = [
                            rj[0] - ri[0]
                                + sx as f64 * cell[0][0]
                                + sy as f64 * cell[1][0]
                                + sz as f64 * cell[2][0],
                            rj[1] - ri[1]
                                + sx as f64 * cell[0][1]
                                + sy as f64 * cell[1][1]
                                + sz as f64 * cell[2][1],
                            rj[2] - ri[2]
                                + sx as f64 * cell[0][2]
                                + sy as f64 * cell[1][2]
                                + sz as f64 * cell[2][2],
                        ];
                        let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                        if r2 > 1e-16 && r2 < cutoff2 {
                            pairs.push((i, j, d, r2.sqrt()));
                        }
                    }
                }
            }
        }
    }
    let mut rho = vec![0.0; n];
    for &(i, j, _, r) in &pairs {
        let weight = density(r).0;
        rho[i] += weight;
        if i != j {
            rho[j] += weight;
        } else {
            rho[i] += weight;
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
    for &(i, j, d, r) in &pairs {
        let (v, dv) = pair(r);
        let (_, drho) = density(r);
        let inv_r = 1.0 / r;
        if i != j {
            energy += v;
            let deriv = dv + (fp[i] + fp[j]) * drho;
            for c in 0..3 {
                let force = deriv * d[c] * inv_r;
                gradient[3 * i + c] -= force;
                gradient[3 * j + c] += force;
            }
            for a in 0..3 {
                for b in 0..3 {
                    virial[a][b] += deriv * d[a] * d[b] * inv_r;
                }
            }
        } else {
            energy += v;
            let deriv = dv + 2.0 * fp[i] * drho;
            for a in 0..3 {
                for b in 0..3 {
                    virial[a][b] += deriv * d[a] * d[b] * inv_r;
                }
            }
        }
    }
    let volume = (cell[0][0] * (cell[1][1] * cell[2][2] - cell[1][2] * cell[2][1])
        - cell[0][1] * (cell[1][0] * cell[2][2] - cell[1][2] * cell[2][0])
        + cell[0][2] * (cell[1][0] * cell[2][1] - cell[1][1] * cell[2][0]))
        .abs();
    let mut stress = [[0.0; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            stress[a][b] = virial[a][b] / volume;
        }
    }
    (energy, gradient, stress)
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

fn max_abs(g: &Array1<f64>) -> f64 {
    g.iter().fold(0.0, |m, v| m.max(v.abs()))
}

fn relax(positions: &mut Array1<f64>, cell: [[f64; 3]; 3]) {
    let mut solver = Solver::new(
        Method::Fire { kind: FireKind::V2 },
        Control {
            gtol: 1e-3,
            maxiter: 1,
            istep: 0.05,
            maxmove: None,
            ..Control::default()
        },
        positions.len(),
    );
    solver.set_atom_maxmove(0.08);
    for _ in 0..200 {
        if max_abs(&eam_eval(positions.view(), cell).1) < 2e-3 {
            break;
        }
        let held = cell;
        let oracle = Oracle::unbounded(positions.len(), move |x: ArrayView1<f64>| {
            let (e, g, _) = eam_eval(x, held);
            (e, g)
        });
        solver.step(&oracle, positions).expect("relax");
    }
}

struct Copper {
    cell: [[f64; 3]; 3],
}

impl BandSurface for Copper {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.eval_cells(
            positions,
            &[],
            energies,
            gradients,
            &mut Array2::zeros((0, 9)),
        )
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for row in 0..positions.nrows() {
            let h = if cells.is_empty() {
                self.cell
            } else {
                cells[row].0
            };
            let (e, g, s) = eam_eval(positions.row(row), h);
            energies[row] = e;
            for k in 0..g.len() {
                gradients[(row, k)] = g[k];
            }
            if !stresses.is_empty() {
                for a in 0..3 {
                    for b in 0..3 {
                        stresses[(row, 3 * a + b)] = s[a][b];
                    }
                }
            }
        }
        Ok(())
    }
}

fn climb(
    method: Method,
    free: bool,
    start: &Array1<f64>,
    end: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> (usize, usize, f64, bool) {
    let n_images = 7;
    let mut positions = Array2::zeros((n_images, start.len()));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        for k in 0..start.len() {
            positions[(i, k)] = (1.0 - t) * start[k] + t * end[k];
        }
    }
    let e0 = eam_eval(start.view(), cell).0;
    let solid = free.then(|| SolidState::new(vec![Cell(cell); n_images]));
    let mut band = BandSession::new(
        BandConfig {
            climbing: Some(CiConfig::after(0.5)),
            force_tol: 1e-3,
            max_move: 0.08,
            min_mode: None,
            spring: rgsaddle::SpringKind::Uniform { k: 1.0 },
            method,
            solid,
            ..BandConfig::default()
        },
        positions,
    )
    .unwrap();
    let surface = Copper { cell };
    let mut rows = 0usize;
    let mut report = band.step(&surface).unwrap();
    rows += report.surface_rows;
    let mut steps = 1usize;
    while report.status == BandStatus::Running && steps < 800 {
        report = band.step(&surface).unwrap();
        rows += report.surface_rows;
        steps += 1;
    }
    let ci = report.ci_index.unwrap_or(1);
    let h = band.image_cells().map(|c| c[ci].0).unwrap_or(cell);
    let barrier = eam_eval(band.positions().row(ci), h).0 - e0;
    (steps, rows, barrier, report.status == BandStatus::Converged)
}

#[test]
fn mishin_dumbbell_hop_fixed_and_free() {
    let (perfect, cell) = fcc(1, 3.615);
    let (energy, gradient, _) = eam_eval(perfect.view(), cell);
    assert!(
        (energy / 4.0 + 3.54).abs() < 5e-3,
        "cohesive {}",
        energy / 4.0
    );
    assert!(
        max_abs(&gradient) < 1e-4,
        "perfect force {}",
        max_abs(&gradient)
    );

    let n = 3;
    let (perfect, cell) = fcc(n, 3.615);
    let mut along_100 = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let mut along_110 = dumbbell(
        &perfect,
        [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
        ],
        2.05,
    );
    relax(&mut along_100, cell);
    relax(&mut along_110, cell);
    let e100 = eam_eval(along_100.view(), cell).0;
    let e110 = eam_eval(along_110.view(), cell).0;
    let gap = e110 - e100;
    eprintln!(
        "ends e100 {e100} f100 {} e110 {e110} f110 {} gap {gap}",
        max_abs(&eam_eval(along_100.view(), cell).1),
        max_abs(&eam_eval(along_110.view(), cell).1)
    );
    assert!((gap - 0.474177).abs() < 2e-3, "endpoint gap {gap}");
    for (label, method) in [
        ("fire2", Method::Fire { kind: FireKind::V2 }),
        ("lbfgs", Method::Lbfgs { memory: 10 }),
    ] {
        for free in [false, true] {
            let (steps, rows, barrier, ok) =
                climb(method.clone(), free, &along_100, &along_110, cell);
            let cell_name = if free { "free" } else { "fixed" };
            eprintln!("cu,{label},{cell_name},{steps},{rows},{barrier:.6},{ok}");
        }
    }
}
