//! A linear LJ7 rearrangement draws atoms through each other. FIRE on
//! that path does not reach the force tolerance. [`interpolated_path`]
//! replaces the overlapping images with IDPP, and the same FIRE run
//! converges.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, CiConfig, IdppConfig, SaddleError,
    SpringKind, interpolated_path, linear_path,
};

struct LennardJones;

impl BandSurface for LennardJones {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, grad) = lennard_jones(row);
            energies[i] = energy;
            gradients.row_mut(i).assign(&grad);
        }
        Ok(())
    }
}

fn lennard_jones(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for a in 0..n {
        for b in (a + 1)..n {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = x[3 * b + k] - x[3 * a + k];
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1e-24 {
                continue;
            }
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r12 = inv_r6 * inv_r6;
            energy += 4.0 * (inv_r12 - inv_r6);
            let scale = 24.0 * (inv_r6 - 2.0 * inv_r12) / r2;
            for k in 0..3 {
                grad[3 * a + k] -= scale * d[k];
                grad[3 * b + k] += scale * d[k];
            }
        }
    }
    (energy, grad)
}

fn project_translation(force: &mut Array1<f64>) {
    let n = force.len() / 3;
    if n < 2 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for (c, mean_c) in mean.iter_mut().enumerate() {
            *mean_c += force[3 * atom + c];
        }
    }
    for mean_c in &mut mean {
        *mean_c /= n as f64;
    }
    for atom in 0..n {
        for (c, mean_c) in mean.iter().enumerate() {
            force[3 * atom + c] -= mean_c;
        }
    }
}

fn minimize(x: &mut Array1<f64>) {
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Lbfgs { memory: 10 },
        rgmin::Control {
            gtol: 1e-6,
            maxiter: 2000,
            maxmove: Some(0.15),
            ..rgmin::Control::default()
        },
        x.len(),
    );
    solver.set_accept(rgmin::Accept::Energy);
    solver.set_atom_maxmove(0.15);
    let oracle = rgmin::Oracle::unbounded(x.len(), |q: ArrayView1<f64>| {
        let (energy, mut grad) = lennard_jones(q);
        project_translation(&mut grad);
        (energy, grad)
    });
    for _ in 0..800 {
        solver.step(&oracle, x).unwrap();
        let (_, mut grad) = lennard_jones(x.view());
        project_translation(&mut grad);
        if grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) < 1e-4 {
            break;
        }
    }
}

fn align(reactant: &Array1<f64>, product: &mut Array1<f64>) {
    let n = reactant.len() / 3;
    let mut used = vec![false; n];
    let mut matched = Array1::zeros(reactant.len());
    for i in 0..n {
        let mut best = 0;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let mut d = 0.0;
            for k in 0..3 {
                let delta = reactant[3 * i + k] - product[3 * j + k];
                d += delta * delta;
            }
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        used[best] = true;
        for k in 0..3 {
            matched[3 * i + k] = product[3 * best + k];
        }
    }
    product.assign(&matched);
}

fn lj7_endpoints() -> (Array1<f64>, Array1<f64>) {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let radius = nn / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let mut penta = Array1::zeros(21);
    for k in 0..5 {
        let ang = 2.0 * std::f64::consts::PI * k as f64 / 5.0;
        penta[3 * k] = radius * ang.cos();
        penta[3 * k + 1] = radius * ang.sin();
    }
    let h = (nn * nn - radius * radius).max(0.2).sqrt();
    penta[17] = h;
    penta[20] = -h;
    let a = nn / 2.0_f64.sqrt();
    let mut octa = Array1::zeros(21);
    let axis = [
        [a, 0.0, 0.0],
        [-a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, -a, 0.0],
        [0.0, 0.0, a],
        [0.0, 0.0, -a],
    ];
    for (i, p) in axis.iter().enumerate() {
        octa[3 * i] = p[0];
        octa[3 * i + 1] = p[1];
        octa[3 * i + 2] = p[2];
    }
    octa[18] = a;
    octa[19] = a;
    octa[20] = a;
    minimize(&mut penta);
    minimize(&mut octa);
    align(&penta, &mut octa);
    (penta, octa)
}

fn min_pair(x: ArrayView1<f64>) -> f64 {
    let n = x.len() / 3;
    let mut best = f64::INFINITY;
    for a in 0..n {
        for b in (a + 1)..n {
            let mut r2 = 0.0;
            for k in 0..3 {
                let d = x[3 * b + k] - x[3 * a + k];
                r2 += d * d;
            }
            best = best.min(r2.sqrt());
        }
    }
    best
}

fn fire_calls(path: Array2<f64>, cap: usize) -> (BandStatus, usize, Option<usize>) {
    let config = BandConfig {
        spring: SpringKind::Uniform { k: 5.0 },
        climbing: Some(CiConfig::always()),
        force_tol: 1e-8,
        max_move: 0.2,
        time_step: 1.0 / 10.1805055,
        min_mode: None,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, path).unwrap();
    let mut calls = 0;
    let mut status = BandStatus::Running;
    let mut ci = None;
    for _ in 0..cap {
        let report = session.step(&LennardJones).unwrap();
        calls += report.surface_rows;
        status = report.status;
        ci = report.ci_index;
        let Some(ev) = session.evaluation() else {
            break;
        };
        let n = ev.projected.nrows();
        let mut band = 0.0_f64;
        let mut ci_force = None;
        for i in 0..n {
            let row = ev.projected.row(i);
            let nrm = row.dot(&row).sqrt();
            band = band.max(nrm);
            if ci == Some(i + 1) {
                ci_force = Some(nrm);
            }
        }
        let scalar = match ci_force {
            Some(local) if band <= 0.1 => local,
            _ => band,
        };
        if scalar < 0.01 {
            return (BandStatus::Converged, calls, ci);
        }
        if status != BandStatus::Running {
            break;
        }
    }
    (status, calls, ci)
}

#[test]
fn linear_lj7_fire_does_not_converge_when_atoms_cross() {
    let (reactant, product) = lj7_endpoints();
    let linear = linear_path(reactant.view(), product.view(), 7, None).unwrap();
    let nn = min_pair(reactant.view()).min(min_pair(product.view()));
    let closest = (1..6)
        .map(|i| min_pair(linear.row(i)))
        .fold(f64::MAX, f64::min);
    assert!(
        closest < 0.5 * nn,
        "the straight path should overlap, closest {closest}, nn {nn}"
    );
    let (status, calls, _) = fire_calls(linear, 400);
    assert_ne!(
        status,
        BandStatus::Converged,
        "linear FIRE converged in {calls} calls; the overlap fixture is gone"
    );
}

#[test]
fn interpolated_lj7_fire_converges() {
    let (reactant, product) = lj7_endpoints();
    let built = interpolated_path(
        reactant.view(),
        product.view(),
        7,
        None,
        &IdppConfig::default(),
    )
    .unwrap();
    assert!(
        built.idpp,
        "an overlapping interpolation must fall back to IDPP"
    );
    let nn = min_pair(reactant.view()).min(min_pair(product.view()));
    let closest = (1..6)
        .map(|i| min_pair(built.positions.row(i)))
        .fold(f64::MAX, f64::min);
    assert!(closest > 0.5 * nn, "closest {closest}, nn {nn}");
    let (status, calls, ci) = fire_calls(built.positions, 2500);
    assert_eq!(
        status,
        BandStatus::Converged,
        "FIRE did not converge in {calls} calls, ci={ci:?}"
    );
    assert!(ci.is_some(), "no climbing image");
}
