//! Reproduction of Table II of Makri, Ortner and Kermode, J. Chem. Phys.
//! 150, 094109 (2019): vacancy migration in a 107-atom Cu fcc supercell
//! under a Morse potential (A = 4, epsilon = 1, r0 = 2.55 A), a path of 5
//! images, NEB, force evaluations per image to the residual tolerances
//! 1e-1 and 1e-3 eV/A. The paper reports, for NEB with the ode12r step
//! selection, 8 and 27 evaluations per image without the preconditioner
//! and 8 and 19 with the exponential preconditioner (A = 3, r_cut = 2.2
//! r0). The static step rows (8 / 57 and 7 / 37) are not repeated here;
//! the L-BFGS band of this crate is run beside the integrators.
//!
//! What the paper does not state and this file fixes: the spring
//! constant (1 eV/A^2 here, which the metric reads as the dimensionless
//! kappa = k / mu of the paper), the Morse cutoff (5.3 A, half the cell
//! minus a margin, with the energy shifted to zero at the cutoff), the
//! opening step (the band default) and the endpoint relaxation (L-BFGS
//! to 1e-7 eV/A).

use std::sync::atomic::{AtomicUsize, Ordering};

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::{Control, Method, Solver};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell, ExpBandPrecon, SaddleError, SpringKind,
};

const R0: f64 = 2.55;
const MORSE_A: f64 = 4.0;
const EPS: f64 = 1.0;
const CUTOFF: f64 = 5.3;
const CELLS: usize = 3;

fn lattice() -> f64 {
    R0 * 2.0_f64.sqrt()
}

fn box_length() -> f64 {
    CELLS as f64 * lattice()
}

/// Morse pair energy and its radial derivative, shifted to zero at the cutoff.
fn pair(r: f64) -> (f64, f64) {
    if r >= CUTOFF {
        return (0.0, 0.0);
    }
    let e = |d: f64| {
        EPS * ((-2.0 * MORSE_A * (d / R0 - 1.0)).exp() - 2.0 * (-MORSE_A * (d / R0 - 1.0)).exp())
    };
    let de = (-2.0 * MORSE_A / R0)
        * EPS
        * ((-2.0 * MORSE_A * (r / R0 - 1.0)).exp() - (-MORSE_A * (r / R0 - 1.0)).exp());
    (e(r) - e(CUTOFF), de)
}

fn wrap(d: f64, l: f64) -> f64 {
    d - l * (d / l).round()
}

/// Energy and gradient of one configuration under the minimum image.
fn morse(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let l = box_length();
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for a in 0..n {
        for b in a + 1..n {
            let d = [
                wrap(x[3 * b] - x[3 * a], l),
                wrap(x[3 * b + 1] - x[3 * a + 1], l),
                wrap(x[3 * b + 2] - x[3 * a + 2], l),
            ];
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let (e, de) = pair(r);
            energy += e;
            for k in 0..3 {
                let g = de * d[k] / r;
                grad[3 * b + k] += g;
                grad[3 * a + k] -= g;
            }
        }
    }
    (energy, grad)
}

/// fcc sites of a CELLS^3 conventional supercell.
fn sites() -> Vec<[f64; 3]> {
    let a = lattice();
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut out = Vec::new();
    for i in 0..CELLS {
        for j in 0..CELLS {
            for k in 0..CELLS {
                for b in basis {
                    out.push([
                        a * (i as f64 + b[0]),
                        a * (j as f64 + b[1]),
                        a * (k as f64 + b[2]),
                    ]);
                }
            }
        }
    }
    out
}

/// Reactant and product: the vacancy at the central site, and the
/// nearest neighbour that hops into it.
fn endpoints() -> (Array1<f64>, Array1<f64>) {
    let all = sites();
    let l = box_length();
    let centre = [l / 2.0, l / 2.0, l / 2.0];
    let dist = |p: &[f64; 3], q: &[f64; 3]| {
        (0..3)
            .map(|k| wrap(p[k] - q[k], l).powi(2))
            .sum::<f64>()
            .sqrt()
    };
    let vacancy = (0..all.len())
        .min_by(|&i, &j| {
            dist(&all[i], &centre)
                .partial_cmp(&dist(&all[j], &centre))
                .unwrap()
        })
        .unwrap();
    let hopper = (0..all.len())
        .filter(|&i| i != vacancy)
        .min_by(|&i, &j| {
            dist(&all[i], &all[vacancy])
                .partial_cmp(&dist(&all[j], &all[vacancy]))
                .unwrap()
        })
        .unwrap();
    let kept: Vec<usize> = (0..all.len()).filter(|&i| i != vacancy).collect();
    let mut reactant = Array1::zeros(3 * kept.len());
    let mut product = Array1::zeros(3 * kept.len());
    for (slot, &site) in kept.iter().enumerate() {
        for k in 0..3 {
            reactant[3 * slot + k] = all[site][k];
            product[3 * slot + k] = if site == hopper {
                all[vacancy][k]
            } else {
                all[site][k]
            };
        }
    }
    (reactant, product)
}

struct Lattice {
    bounds: Bounds<f64>,
    dim: usize,
}

impl Objective<f64> for Lattice {
    fn dim(&self) -> usize {
        self.dim
    }
    fn bounds(&self) -> &Bounds<f64> {
        &self.bounds
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        morse(x).0
    }
}

impl Gradient<f64> for Lattice {
    fn dim(&self) -> usize {
        self.dim
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        morse(x).1
    }
}

impl DifferentiableObjective<f64> for Lattice {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        morse(x)
    }
}

fn relax(mut x: Array1<f64>) -> Array1<f64> {
    let dim = x.len();
    let obj = Lattice {
        bounds: Bounds::new(
            Array1::from_elem(dim, -1e6),
            Array1::from_elem(dim, 1e6),
            0.0,
        ),
        dim,
    };
    let control = Control {
        maxiter: 1,
        gtol: 0.0,
        istep: 0.1,
        ..Control::default()
    };
    let mut solver = Solver::new(Method::Lbfgs { memory: 10 }, control, dim);
    for _ in 0..2000 {
        let rep = solver.step(&obj, &mut x).unwrap();
        if morse(x.view())
            .1
            .iter()
            .fold(0.0_f64, |m, g| m.max(g.abs()))
            < 1e-7
        {
            let _ = rep;
            break;
        }
    }
    x
}

struct Surface {
    rows: AtomicUsize,
}

impl BandSurface for Surface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = morse(row);
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        self.rows.fetch_add(positions.nrows(), Ordering::Relaxed);
        Ok(())
    }
}

fn initial_band(reactant: &Array1<f64>, product: &Array1<f64>, n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, reactant.len()));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        let row = reactant * (1.0 - t) + product * t;
        band.row_mut(i).assign(&row);
    }
    band
}

/// Evaluations per interior image to reach max |F_perp| <= 1e-1 and
/// <= 1e-3 eV/A, counting every surface row the band asked for.
fn evaluations(
    method: Method,
    precon: Option<ExpBandPrecon>,
    spring: f64,
    band: &Array2<f64>,
) -> (f64, f64, usize) {
    let n_images = band.nrows();
    let l = box_length();
    let label = format!(
        "{method:?}{} k={spring}",
        if precon.is_some() { "-Exp" } else { "" }
    );
    let config = BandConfig {
        spring: SpringKind::Uniform { k: spring },
        climbing: None,
        cell: Some(Cell([[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]])),
        force_tol: 1e-3,
        max_move: 0.0,
        remove_translation: false,
        method,
        min_mode: None,
        exp_precon: precon,
        ..BandConfig::default()
    };
    let surface = Surface {
        rows: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config, band.clone()).unwrap();
    let interior = (n_images - 2) as f64;
    let mut coarse = None;
    let mut steps = 0;
    for _ in 0..1500 {
        let report = session.step(&surface).unwrap();
        steps += 1;
        let per_image = surface.rows.load(Ordering::Relaxed) as f64 / interior;
        if steps == 1 || steps % 100 == 0 {
            eprintln!(
                "{label}: step {steps}, {per_image:.1} evaluations per image, max force {:.4}, mu {:?}",
                report.max_force,
                session.exp_scale()
            );
        }
        if coarse.is_none() && report.max_force <= 1e-1 {
            coarse = Some(per_image);
        }
        if report.status == BandStatus::Converged {
            return (coarse.unwrap_or(per_image), per_image, steps);
        }
    }
    panic!("{label} did not converge the vacancy band in 1500 steps");
}

#[test]
fn ode12r_reproduces_the_cu_vacancy_counts_of_table_ii() {
    let (reactant, product) = endpoints();
    assert_eq!(reactant.len(), 3 * 107);
    let reactant = relax(reactant);
    let product = relax(product);
    let residual = |x: &Array1<f64>| {
        morse(x.view())
            .1
            .iter()
            .fold(0.0_f64, |m, g| m.max(g.abs()))
    };
    eprintln!(
        "endpoints: max |g| {:.2e} and {:.2e}, energies {:.6} and {:.6}",
        residual(&reactant),
        residual(&product),
        morse(reactant.view()).0,
        morse(product.view()).0
    );
    let band = initial_band(&reactant, &product, 5);
    let exp = ExpBandPrecon {
        a: 3.0,
        c_stab: 0.1,
        mu: None,
        r_cut_factor: 2.2,
    };
    let plain = evaluations(Method::Ode12r, None, 1.0, &band);
    let precon = evaluations(Method::Ode12r, Some(exp), 1.0, &band);
    let lbfgs = evaluations(Method::Lbfgs { memory: 20 }, None, 1.0, &band);
    eprintln!(
        "evaluations per image (1e-1, 1e-3, steps): ODE12r {plain:?}, ODE12r-Exp {precon:?}, L-BFGS {lbfgs:?}; paper NEB ode12r 8 / 27, NEB(p) ode12r 8 / 19"
    );
    // Measured: ODE12r 8.67 and 26.67 (paper 8 and 27), ODE12r-Exp 7.33
    // and 19.33 (paper 8 and 19), L-BFGS 7.67 and 19.67, with k = 1
    // eV/A^2 (kappa = k / mu = 0.83 at the estimated mu = 1.21 eV/A^2).
    // The metric count moves with the spring: k = 0.3 gives 20.33, k = 3
    // gives 77.33, k = 10 gives 199.33. The bounds leave 15 percent over
    // the paper's rows.
    assert!(
        plain.0 <= 10.0,
        "ODE12r took {} evaluations per image to 1e-1",
        plain.0
    );
    assert!(
        plain.1 <= 31.0,
        "ODE12r took {} evaluations per image",
        plain.1
    );
    assert!(
        precon.0 <= 10.0,
        "ODE12r-Exp took {} evaluations per image to 1e-1",
        precon.0
    );
    assert!(
        precon.1 <= 22.0,
        "ODE12r-Exp took {} evaluations per image",
        precon.1
    );
    assert!(
        precon.1 < plain.1,
        "the metric should cost less: {} against {}",
        precon.1,
        plain.1
    );
}
