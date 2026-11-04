//! How a GS2 branch of the intrinsic reaction coordinate (IRC) ends on two
//! saddles that eOn hands to rgsaddle.
//!
//! The fixtures carry a saddle and its lowest mode as eOn's minimum-mode
//! search refines them, one atom per line: fixed flag, mass, position,
//! mode. The first line has the saddle energy, the eigenvalue and the
//! orthogonal box.
//!
//! - `lj13_saddle.txt`: the 13-atom Lennard-Jones cluster (LJ13) of eOn's
//!   `neb_lj13` test data, shifted 12-6 with u0 1, psi 1 and cutoff 15.
//! - `pt_saddle.txt`: the Morse Pt adatom saddle of eOn's systems tutorial,
//!   one free atom over a fixed slab of 335, Morse De 0.7102, a 1.6047,
//!   re 2.897 and cutoff 9.5, shifted to zero at the cutoff.
//!
//! An energy rise halves the sphere radius down to 1e-4 and ends that step
//! on its last accepted point. A branch in a basin whose force is at or
//! below the threshold must report a minimum all the same, as both LJ13
//! branches must at each threshold. A halved radius must not let a point
//! beside the saddle pass for a minimum, as on the Pt adatom branches.

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{IrcConfig, IrcDirection, IrcReport, IrcSession, PointSurface, SaddleError};

#[derive(Clone, Copy)]
enum Pair {
    LennardJones,
    Morse,
}

impl Pair {
    fn cutoff(self) -> f64 {
        match self {
            Pair::LennardJones => 15.0,
            Pair::Morse => 9.5,
        }
    }

    /// Unshifted pair energy, and `-dU/dr / r`.
    fn term(self, r2: f64) -> (f64, f64) {
        match self {
            Pair::LennardJones => {
                let a = (1.0 / r2).powi(3);
                let b = 4.0 * a;
                (b * (a - 1.0), 6.0 * b / r2 * (2.0 * a - 1.0))
            }
            Pair::Morse => {
                let (de, alpha, re) = (0.7102, 1.6047, 2.897);
                let r = r2.sqrt();
                let d = 1.0 - (-alpha * (r - re)).exp();
                (de * d * d - de, 2.0 * de * alpha * d * (d - 1.0) / r)
            }
        }
    }
}

struct Fixture {
    pair: Pair,
    cell: [f64; 3],
    positions: Vec<[f64; 3]>,
    free: Vec<usize>,
    masses: Array1<f64>,
    saddle: Array1<f64>,
    mode: Array1<f64>,
}

impl Fixture {
    fn load(name: &str, pair: Pair) -> Self {
        let path = format!("{}/tests/fixtures/irc/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(path).unwrap();
        let mut lines = text.lines();
        let head: Vec<f64> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        let mut positions = Vec::new();
        let (mut free, mut masses, mut saddle, mut mode) = (vec![], vec![], vec![], vec![]);
        for (i, line) in lines.enumerate() {
            let v: Vec<f64> = line
                .split_whitespace()
                .map(|v| v.parse().unwrap())
                .collect();
            positions.push([v[2], v[3], v[4]]);
            if v[0] == 0.0 {
                free.push(i);
                masses.push(v[1]);
                saddle.extend_from_slice(&v[2..5]);
                mode.extend_from_slice(&v[5..8]);
            }
        }
        Self {
            pair,
            cell: [head[2], head[3], head[4]],
            positions,
            free,
            masses: Array1::from(masses),
            saddle: Array1::from(saddle),
            mode: Array1::from(mode),
        }
    }

    fn session(&self, dx: f64, force_tol: f64, direction: IrcDirection) -> IrcSession {
        IrcSession::new(
            IrcConfig {
                dx,
                force_tol,
                ..IrcConfig::default()
            },
            self.saddle.clone(),
            self.masses.clone(),
            self.mode.clone(),
            direction,
        )
        .unwrap()
    }

    fn mw_distance(&self, x: ArrayView1<f64>) -> f64 {
        let mut sum = 0.0;
        for k in 0..x.len() {
            let d = x[k] - self.saddle[k];
            sum += self.masses[k / 3] * d * d;
        }
        sum.sqrt()
    }

    /// Lowest eigenvalue of the central-difference Hessian of the free
    /// coordinates. With no fixed atom, translations and rotations are
    /// projected out: off a stationary point a rotation of the cluster has
    /// a curvature of the order of the force, of either sign.
    fn lowest_curvature(&self, x: ArrayView1<f64>) -> f64 {
        let n = x.len();
        let h = 1e-5;
        let mut hess = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            let mut xp = x.to_owned();
            let mut xm = x.to_owned();
            xp[i] += h;
            xm[i] -= h;
            let gp = self.eval(xp.view()).unwrap().1;
            let gm = self.eval(xm.view()).unwrap().1;
            for j in 0..n {
                hess[(j, i)] = (gp[j] - gm[j]) / (2.0 * h);
            }
        }
        let mut sym = (&hess + &hess.t()) * 0.5;
        if self.free.len() == self.positions.len() {
            let q = internal_basis(x);
            sym = q.t().dot(&sym).dot(&q);
        }
        let evals = jacobi_eigenvalues(sym);
        evals.into_iter().fold(f64::INFINITY, f64::min)
    }
}

/// Orthonormal columns spanning the complement of the three translations
/// and three rotations of a free cluster at `x`.
fn internal_basis(x: ArrayView1<f64>) -> Array2<f64> {
    let n = x.len();
    let atoms = n / 3;
    let mut centre = [0.0; 3];
    for a in 0..atoms {
        for k in 0..3 {
            centre[k] += x[3 * a + k] / atoms as f64;
        }
    }
    let mut candidates = Vec::new();
    for k in 0..3 {
        let mut t = Array1::zeros(n);
        for a in 0..atoms {
            t[3 * a + k] = 1.0;
        }
        candidates.push(t);
    }
    for k in 0..3 {
        let mut r = Array1::zeros(n);
        for a in 0..atoms {
            let p = [
                x[3 * a] - centre[0],
                x[3 * a + 1] - centre[1],
                x[3 * a + 2] - centre[2],
            ];
            let mut unit = [0.0; 3];
            unit[k] = 1.0;
            r[3 * a] = unit[1] * p[2] - unit[2] * p[1];
            r[3 * a + 1] = unit[2] * p[0] - unit[0] * p[2];
            r[3 * a + 2] = unit[0] * p[1] - unit[1] * p[0];
        }
        candidates.push(r);
    }
    for i in 0..n {
        let mut e = Array1::zeros(n);
        e[i] = 1.0;
        candidates.push(e);
    }
    let mut basis: Vec<Array1<f64>> = Vec::new();
    for mut v in candidates {
        for b in &basis {
            let c = b.dot(&v);
            v.scaled_add(-c, b);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-8 {
            basis.push(v / norm);
        }
    }
    let internal = &basis[6..];
    let mut q = Array2::zeros((n, internal.len()));
    for (j, v) in internal.iter().enumerate() {
        q.column_mut(j).assign(v);
    }
    q
}

impl PointSurface for Fixture {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut r = self.positions.clone();
        for (j, &atom) in self.free.iter().enumerate() {
            r[atom] = [x[3 * j], x[3 * j + 1], x[3 * j + 2]];
        }
        let mut is_free = vec![false; r.len()];
        for &atom in &self.free {
            is_free[atom] = true;
        }
        let rc = self.pair.cutoff();
        let shift = self.pair.term(rc * rc).0;
        let mut energy = 0.0;
        let mut force = vec![[0.0; 3]; r.len()];
        // Pairs of two fixed atoms add a constant and no free force.
        for &i in &self.free {
            for j in 0..r.len() {
                if j == i || (is_free[j] && j < i) {
                    continue;
                }
                let d: [f64; 3] = std::array::from_fn(|k| {
                    let dk = r[i][k] - r[j][k];
                    dk - self.cell[k] * (dk / self.cell[k]).round()
                });
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 >= rc * rc {
                    continue;
                }
                let (u, fscale) = self.pair.term(r2);
                energy += u - shift;
                for (k, dk) in d.iter().enumerate() {
                    force[i][k] += fscale * dk;
                    force[j][k] -= fscale * dk;
                }
            }
        }
        let mut gradient = Array1::zeros(x.len());
        for (j, &atom) in self.free.iter().enumerate() {
            for k in 0..3 {
                gradient[3 * j + k] = -force[atom][k];
            }
        }
        Ok((energy, gradient))
    }
}

fn jacobi_eigenvalues(mut a: Array2<f64>) -> Vec<f64> {
    let n = a.nrows();
    for _ in 0..100 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += a[(p, q)] * a[(p, q)];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[(p, q)].abs() < 1e-300 {
                    continue;
                }
                let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * a[(p, q)]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[(k, p)], a[(k, q)]);
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[(p, k)], a[(q, k)]);
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
            }
        }
    }
    (0..n).map(|i| a[(i, i)]).collect()
}

/// eOn's loop: step until rgsaddle reports a minimum or `max_steps` run out.
fn follow(session: &mut IrcSession, surface: &Fixture, max_steps: usize) -> (usize, IrcReport) {
    let mut report = session.step(surface).unwrap();
    let mut steps = 1;
    while !report.at_minimum && steps < max_steps {
        report = session.step(surface).unwrap();
        steps += 1;
    }
    (steps, report)
}

/// Follows one LJ13 branch to a minimum: the converged flag, the force
/// at or below `force_tol`, and positive internal curvature where it stops.
fn lj13_branch_reaches_a_minimum(lj13: &Fixture, force_tol: f64, direction: IrcDirection) {
    let mut session = lj13.session(0.1, force_tol, direction);
    let (steps, report) = follow(&mut session, lj13, 1000);
    assert!(
        report.at_minimum,
        "{direction:?} at force_tol {force_tol}: no minimum after {steps} steps, \
         last force {:.4e}, arc {:.4}",
        report.max_force, report.arc
    );
    assert!(report.max_force <= force_tol);
    let curvature = lj13.lowest_curvature(session.position());
    assert!(
        curvature > 0.0,
        "{direction:?} at force_tol {force_tol}: curvature {curvature:.4e} after {steps} steps"
    );
}

#[test]
fn lj13_reverse_branch_under_the_force_threshold_reports_a_minimum() {
    // The force falls under 0.005 after a rejected trial has halved the
    // radius, so no later step ends on an interior one.
    let lj13 = Fixture::load("lj13_saddle.txt", Pair::LennardJones);
    lj13_branch_reaches_a_minimum(&lj13, 0.005, IrcDirection::Reverse);
}

#[test]
#[ignore = "both branches at three thresholds, run in release"]
fn lj13_branches_under_the_force_threshold_report_a_minimum() {
    let lj13 = Fixture::load("lj13_saddle.txt", Pair::LennardJones);
    for force_tol in [0.05, 0.01, 0.005] {
        for direction in [IrcDirection::Forward, IrcDirection::Reverse] {
            lj13_branch_reaches_a_minimum(&lj13, force_tol, direction);
        }
    }
}

#[test]
fn pt_adatom_point_beside_the_saddle_is_not_a_minimum() {
    let pt = Fixture::load("pt_saddle.txt", Pair::Morse);
    for (dx, force_tol) in [(0.1, 0.05), (0.1, 0.01), (0.5, 0.05), (0.5, 0.01)] {
        for direction in [IrcDirection::Forward, IrcDirection::Reverse] {
            let mut session = pt.session(dx, force_tol, direction);
            let (steps, report) = follow(&mut session, &pt, 1000);
            if !report.at_minimum {
                continue;
            }
            let x = session.position().to_owned();
            let curvature = pt.lowest_curvature(x.view());
            assert!(
                curvature > 0.0,
                "{direction:?} at dx {dx} force_tol {force_tol}: minimum after {steps} steps at \
                 {:.4} kicks from the saddle, force {:.4e}, curvature {curvature:.4e}",
                pt.mw_distance(x.view()) / dx,
                report.max_force
            );
        }
    }
}
