#![allow(clippy::needless_range_loop, clippy::excessive_precision)]
//! Stencil benchmark gate and a strong-scaling check.
//!
//! Default `cargo test` does not run them. The gate command is
//! `cargo +stable test --release --features capi --offline --test stencil_gate -- --ignored --test-threads=1`.

use std::fs;
use std::sync::Mutex;
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgsaddle::stencil::{
    PHASE_ZOOM, Preset, Stencil, StencilStatus, ZoomEnds, ZoomForce, ZoomParams,
};
use rgsaddle::{BandSurface, Index1Config, Index1Session, Index1Status, PointSurface, SaddleError};
use sha2::{Digest, Sha256};

const SEEDS: usize = 10;
const REPEATS: usize = 5;
const MAX_2D: usize = 80;
const MAX_LJ13: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sys {
    Mb,
    Leps,
    Lj7,
    Lj13,
}

#[derive(Clone, Copy)]
enum Kind {
    Ci,
    Ew,
    ZoomFixed,
    ZoomFree,
    Auto,
}

#[derive(Clone, Copy)]
enum Stepper {
    Fire,
    Lbfgs,
}

#[derive(Clone)]
struct Row {
    system: &'static str,
    method: String,
    seed: usize,
    converged: i32,
    iters: usize,
    calls: usize,
    energy: f64,
    fmax: f64,
    wall_s: f64,
    endpoint: &'static str,
}

fn name_sys(sys: Sys) -> &'static str {
    match sys {
        Sys::Mb => "muller-brown",
        Sys::Leps => "leps-ho",
        Sys::Lj7 => "lj7",
        Sys::Lj13 => "lj13",
    }
}

fn name_kind(kind: Kind) -> &'static str {
    match kind {
        Kind::Ci => "ci-neb",
        Kind::Ew => "ew-ci-neb",
        Kind::ZoomFixed => "zoom-fixed",
        Kind::ZoomFree => "zoom-free",
        Kind::Auto => "autoneb",
    }
}

struct Mb;
struct Leps;
struct Lj;

fn mb_eg(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
    const AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
    const B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
    const C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
    const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
    const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
    let mut value = z * z;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for i in 0..4 {
        let dx = x - X0[i];
        let dy = y - Y0[i];
        let exponent = AA[i] * dx * dx + B[i] * dx * dy + C[i] * dy * dy;
        let weight = A[i] * exponent.exp();
        value += weight;
        gx += weight * (2.0 * AA[i] * dx + B[i] * dy);
        gy += weight * (B[i] * dx + 2.0 * C[i] * dy);
    }
    (value, [gx, gy, 2.0 * z])
}

fn leps_v(r_ab: f64, x: f64, z: f64) -> f64 {
    let (a, b, c) = (0.05, 0.80, 0.05);
    let (r0, alpha, r_ac) = (0.742, 1.942, 3.742);
    let r_bc = r_ac - r_ab;
    let qj = |d: f64, r: f64| {
        let e1 = (-alpha * (r - r0)).exp();
        let e2 = (-2.0 * alpha * (r - r0)).exp();
        (d / 2.0 * (1.5 * e2 - e1), d / 4.0 * (e2 - 6.0 * e1))
    };
    let (qab, jab) = qj(4.746, r_ab);
    let (qbc, jbc) = qj(4.746, r_bc);
    let (qac, jac) = qj(3.445, r_ac);
    let inside = jab * jab / ((1.0 + a) * (1.0 + a))
        + jbc * jbc / ((1.0 + b) * (1.0 + b))
        + jac * jac / ((1.0 + c) * (1.0 + c))
        - jab * jbc / ((1.0 + a) * (1.0 + b))
        - jbc * jac / ((1.0 + b) * (1.0 + c))
        - jab * jac / ((1.0 + a) * (1.0 + c));
    let leps = qab / (1.0 + a) + qbc / (1.0 + b) + qac / (1.0 + c) - inside.max(0.0).sqrt();
    let ho = 2.0 * 0.2025 * (r_ab - (r_ac / 2.0 - x / 1.154)).powi(2);
    leps + ho + z * z
}

fn lj_grad(x: ArrayView1<f64>, gradient: &mut [f64]) -> f64 {
    let n = x.len() / 3;
    gradient.fill(0.0);
    let mut energy = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * i] - x[3 * j];
            let dy = x[3 * i + 1] - x[3 * j + 1];
            let dz = x[3 * i + 2] - x[3 * j + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-12 || r2 >= 100.0 {
                continue;
            }
            let inv = 1.0 / r2;
            let s6 = inv * inv * inv;
            let s12 = s6 * s6;
            energy += 4.0 * (s12 - s6);
            let slope = 4.0 * (-6.0 * s12 + 3.0 * s6) * inv;
            for (axis, delta) in [dx, dy, dz].into_iter().enumerate() {
                gradient[3 * i + axis] += slope * 2.0 * delta;
                gradient[3 * j + axis] -= slope * 2.0 * delta;
            }
        }
    }
    energy
}

impl BandSurface for Mb {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for row in 0..positions.nrows() {
            let (value, g) = mb_eg(
                positions[[row, 0]],
                positions[[row, 1]],
                positions[[row, 2]],
            );
            energies[row] = value;
            for axis in 0..3 {
                gradients[[row, axis]] = g[axis];
            }
        }
        Ok(())
    }
}

impl PointSurface for Mb {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (value, g) = mb_eg(x[0], x[1], x[2]);
        Ok((value, Array1::from(g.to_vec())))
    }
}

impl BandSurface for Leps {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let dr = 1.0e-6;
        for row in 0..positions.nrows() {
            let (x, y, z) = (
                positions[[row, 0]],
                positions[[row, 1]],
                positions[[row, 2]],
            );
            energies[row] = leps_v(x, y, z);
            gradients[[row, 0]] = (leps_v(x + dr, y, z) - leps_v(x - dr, y, z)) / (2.0 * dr);
            gradients[[row, 1]] = (leps_v(x, y + dr, z) - leps_v(x, y - dr, z)) / (2.0 * dr);
            gradients[[row, 2]] = 2.0 * z;
        }
        Ok(())
    }
}

impl PointSurface for Leps {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energies = Array1::zeros(1);
        let mut gradients = Array2::zeros((1, 3));
        let mut row = Array2::zeros((1, 3));
        for axis in 0..3 {
            row[[0, axis]] = x[axis];
        }
        BandSurface::eval(&Leps, row.view(), &mut energies, &mut gradients)?;
        Ok((energies[0], gradients.row(0).to_owned()))
    }
}

impl BandSurface for Lj {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let mut slot = vec![0.0; positions.ncols()];
        for row in 0..positions.nrows() {
            energies[row] = lj_grad(positions.row(row), &mut slot);
            for axis in 0..slot.len() {
                gradients[[row, axis]] = slot[axis];
            }
        }
        Ok(())
    }
}

impl PointSurface for Lj {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut slot = vec![0.0; x.len()];
        let energy = lj_grad(x, &mut slot);
        Ok((energy, Array1::from(slot)))
    }
}

fn line(start: &[f64], end: &[f64], images: usize, seed: usize) -> Array2<f64> {
    let mut band = Array2::zeros((images, start.len()));
    let bump = (seed as f64 - 4.5) * 0.008;
    for image in 0..images {
        let t = image as f64 / (images - 1) as f64;
        for axis in 0..start.len() {
            band[[image, axis]] = (1.0 - t) * start[axis] + t * end[axis];
        }
        if image > 0 && image + 1 < images {
            band[[image, 0]] += bump;
        }
    }
    band
}

fn lj7_ends() -> (Vec<f64>, Vec<f64>) {
    let a = 2.0_f64.powf(1.0 / 6.0);
    let s = 3.0_f64.sqrt() / 2.0;
    let hex = [
        [0.0, 0.0],
        [a, 0.0],
        [0.5 * a, a * s],
        [-0.5 * a, a * s],
        [-a, 0.0],
        [-0.5 * a, -a * s],
        [0.5 * a, -a * s],
    ];
    let mut start = vec![0.0; 21];
    let mut end = vec![0.0; 21];
    for atom in 0..7 {
        start[3 * atom] = hex[atom][0];
        start[3 * atom + 1] = hex[atom][1];
        end[3 * atom] = hex[atom][0];
        end[3 * atom + 1] = hex[atom][1];
    }
    end[3] += 0.55;
    end[7] += 0.30;
    (start, end)
}

fn read_con(name: &str) -> (f64, Vec<f64>) {
    let path = format!(
        "{}/validation/eon_zoom/neb_lj13/{name}.con",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = fs::read_to_string(path).unwrap();
    let mut stored = 0.0;
    let mut coords = Vec::new();
    let mut reading = false;
    for line in text.lines() {
        if let Some(rest) = line.split("\"energy\":").nth(1) {
            let token: String = rest
                .chars()
                .take_while(|c| *c == '-' || *c == '.' || c.is_ascii_digit())
                .collect();
            stored = token.parse().unwrap();
        }
        if line.starts_with("Coordinates") {
            reading = true;
            continue;
        }
        if !reading {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(x) = parts.next().and_then(|part| part.parse::<f64>().ok()) else {
            continue;
        };
        let y: f64 = parts.next().unwrap().parse().unwrap();
        let z: f64 = parts.next().unwrap().parse().unwrap();
        coords.extend([x, y, z]);
    }
    (stored, coords)
}

struct Target {
    energy: f64,
    x: Vec<f64>,
    atoms: usize,
    check_index: bool,
}

fn refine<S: PointSurface>(surface: &S, guess: &[f64]) -> (f64, Vec<f64>) {
    let mut session = Index1Session::new(
        Index1Config::default().with_force_tol(1.0e-4),
        Array1::from(guess.to_vec()),
        None,
        None,
    )
    .unwrap();
    for _ in 0..30 {
        match session.step(surface) {
            Ok(report) if report.status == Index1Status::Converged => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
    let x = session.position().to_vec();
    let energy = surface.eval(session.position()).unwrap().0;
    (energy, x)
}

fn targets() -> [Target; 4] {
    let (mb_e, mb_g) = mb_eg(0.21248658, 0.29298833, 0.0);
    let _ = mb_g;
    let (leps_e, leps_x) = refine(&Leps, &[1.35, 0.15, 0.0]);
    let (start, _) = lj7_ends();
    let mut guess = start.clone();
    guess[3] += 0.28;
    let (lj7_e, lj7_x) = refine(&Lj, &guess);
    let (lj13_e, lj13_x) = read_con("saddle");
    [
        Target {
            energy: mb_e,
            x: vec![0.21248658, 0.29298833, 0.0],
            atoms: 1,
            check_index: true,
        },
        Target {
            energy: leps_e,
            x: leps_x,
            atoms: 1,
            check_index: true,
        },
        Target {
            energy: lj7_e,
            x: lj7_x,
            atoms: 7,
            check_index: true,
        },
        Target {
            energy: lj13_e,
            x: lj13_x,
            atoms: 13,
            check_index: false,
        },
    ]
}

fn build(sys: Sys, kind: Kind, stepper: Stepper, seed: usize) -> Result<Stencil, SaddleError> {
    let (start, end) = match sys {
        Sys::Mb => (
            vec![-0.050010822738, 0.466694104872, 0.0],
            vec![0.623499404931, 0.028037758529, 0.0],
        ),
        Sys::Leps => (vec![0.74, 1.15, 0.0], vec![2.9, -0.35, 0.0]),
        Sys::Lj7 => lj7_ends(),
        Sys::Lj13 => {
            let (_, reactant) = read_con("reactant");
            let (_, product) = read_con("product");
            (reactant, product)
        }
    };
    let initial = line(&start, &end, 7, seed);
    let preset = match kind {
        Kind::Auto => Preset::AutoNeb,
        Kind::ZoomFixed | Kind::ZoomFree => Preset::ZoomNeb,
        Kind::Ci | Kind::Ew => Preset::CiNeb,
    };
    let mut builder = Stencil::band(initial).preset(preset).force_tol(1.0e-3);
    if matches!(kind, Kind::Ew | Kind::ZoomFixed | Kind::ZoomFree) {
        builder = builder.energy_weighted(0.972, 9.72);
    }
    if matches!(kind, Kind::ZoomFixed | Kind::ZoomFree) {
        let mut params = ZoomParams::default();
        params.ends = if matches!(kind, Kind::ZoomFree) {
            ZoomEnds::FreePerp
        } else {
            ZoomEnds::Fixed
        };
        builder = builder.zoom(params);
    }
    if matches!(kind, Kind::Auto) {
        builder = builder.autoneb_limits(3, 8);
    }
    if matches!(stepper, Stepper::Lbfgs) {
        builder = builder
            .method(rgmin::Method::Lbfgs { memory: 10 })
            .max_move(0.1);
    }
    builder.build()
}

fn negative_modes(sys: Sys, x: &[f64]) -> usize {
    let n = x.len();
    if n > 24 {
        return 1;
    }
    let mut hessian = vec![0.0; n * n];
    let dr = 1.0e-4;
    let mut plus = vec![0.0; n];
    let mut minus = vec![0.0; n];
    let mut shifted = x.to_vec();
    for col in 0..n {
        shifted[col] = x[col] + dr;
        let gp = grad_of(sys, &shifted, &mut plus);
        shifted[col] = x[col] - dr;
        let gm = grad_of(sys, &shifted, &mut minus);
        shifted[col] = x[col];
        let _ = (gp, gm);
        for row in 0..n {
            hessian[row * n + col] = (plus[row] - minus[row]) / (2.0 * dr);
        }
    }
    for row in 0..n {
        for col in (row + 1)..n {
            let mean = 0.5 * (hessian[row * n + col] + hessian[col * n + row]);
            hessian[row * n + col] = mean;
            hessian[col * n + row] = mean;
        }
    }
    jacobi_negatives(&mut hessian, n)
}

fn grad_of(sys: Sys, x: &[f64], out: &mut [f64]) -> f64 {
    match sys {
        Sys::Mb => {
            let (value, g) = mb_eg(x[0], x[1], x[2]);
            out.copy_from_slice(&g);
            value
        }
        Sys::Leps => PointSurface::eval(&Leps, ArrayView1::from(x))
            .map(|(e, g)| {
                out.copy_from_slice(g.as_slice().unwrap());
                e
            })
            .unwrap_or(f64::NAN),
        Sys::Lj7 => PointSurface::eval(&Lj, ArrayView1::from(x))
            .map(|(e, g)| {
                out.copy_from_slice(g.as_slice().unwrap());
                e
            })
            .unwrap_or(f64::NAN),
        Sys::Lj13 => f64::NAN,
    }
}

fn jacobi_negatives(a: &mut [f64], n: usize) -> usize {
    for _ in 0..12 * n {
        let mut max = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let value = a[i * n + j].abs();
                if value > max {
                    max = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1.0e-10 {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let aik = a[p * n + k];
            let ajk = a[q * n + k];
            let new_p = c * aik - s * ajk;
            let new_q = s * aik + c * ajk;
            a[p * n + k] = new_p;
            a[k * n + p] = new_p;
            a[q * n + k] = new_q;
            a[k * n + q] = new_q;
        }
        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
    }
    (0..n).filter(|&i| a[i * n + i] < -1.0e-4).count()
}

fn kabsch(got: &[f64], reference: &[f64]) -> f64 {
    let n = got.len() / 3;
    if reference.len() != got.len() || n == 0 {
        return f64::INFINITY;
    }
    let mut cg = [0.0; 3];
    let mut cr = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            cg[axis] += got[3 * atom + axis];
            cr[axis] += reference[3 * atom + axis];
        }
    }
    for axis in 0..3 {
        cg[axis] /= n as f64;
        cr[axis] /= n as f64;
    }
    let mut h = [[0.0; 3]; 3];
    for atom in 0..n {
        for row in 0..3 {
            for col in 0..3 {
                h[row][col] +=
                    (got[3 * atom + row] - cg[row]) * (reference[3 * atom + col] - cr[col]);
            }
        }
    }
    let mut ata = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                ata[row][col] += h[k][row] * h[k][col];
            }
        }
    }
    let mut packed = [0.0; 9];
    for row in 0..3 {
        for col in 0..3 {
            packed[row * 3 + col] = ata[row][col];
        }
    }
    let mut vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    jacobi_vectors(&mut packed, &mut vectors);
    let mut sigma = [0.0; 3];
    for k in 0..3 {
        sigma[k] = packed[k * 3 + k].max(0.0).sqrt();
    }
    let mut u = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                let inv = if sigma[k] > 1.0e-12 {
                    1.0 / sigma[k]
                } else {
                    0.0
                };
                u[row][col] += h[row][k] * vectors[k][col] * inv;
            }
        }
    }
    let mut rot = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            for k in 0..3 {
                rot[row][col] += u[row][k] * vectors[col][k];
            }
        }
    }
    let det = rot[0][0] * (rot[1][1] * rot[2][2] - rot[1][2] * rot[2][1])
        - rot[0][1] * (rot[1][0] * rot[2][2] - rot[1][2] * rot[2][0])
        + rot[0][2] * (rot[1][0] * rot[2][1] - rot[1][1] * rot[2][0]);
    if det < 0.0 {
        for row in 0..3 {
            rot[row][2] = -rot[row][2];
        }
    }
    let mut error = 0.0;
    for atom in 0..n {
        for axis in 0..3 {
            let mut aligned = cr[axis];
            for k in 0..3 {
                aligned += rot[axis][k] * (got[3 * atom + k] - cg[k]);
            }
            let delta = aligned - reference[3 * atom + axis];
            error += delta * delta;
        }
    }
    (error / n as f64).sqrt()
}

fn jacobi_vectors(a: &mut [f64], vectors: &mut [[f64; 3]; 3]) {
    let n = 3;
    for _ in 0..24 {
        let mut max = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let value = a[i * n + j].abs();
                if value > max {
                    max = value;
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1.0e-12 {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let aik = a[p * n + k];
            let ajk = a[q * n + k];
            a[p * n + k] = c * aik - s * ajk;
            a[k * n + p] = a[p * n + k];
            a[q * n + k] = s * aik + c * ajk;
            a[k * n + q] = a[q * n + k];
        }
        a[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
        for row in 0..3 {
            let vip = vectors[row][p];
            let viq = vectors[row][q];
            vectors[row][p] = c * vip - s * viq;
            vectors[row][q] = s * vip + c * viq;
        }
    }
}

use std::sync::OnceLock;

fn table() -> &'static [Target; 4] {
    static LOCK: OnceLock<[Target; 4]> = OnceLock::new();
    LOCK.get_or_init(targets)
}

fn drive_with(sys: Sys, kind: Kind, stepper: Stepper, seed: usize) -> Row {
    let refs = table();
    let cap = if sys == Sys::Lj13 { MAX_LJ13 } else { MAX_2D };
    let started = Instant::now();
    let mut out = Row {
        system: name_sys(sys),
        method: format!(
            "{}-{}",
            name_kind(kind),
            match stepper {
                Stepper::Fire => "fire",
                Stepper::Lbfgs => "lbfgs",
            }
        ),
        seed,
        converged: 0,
        iters: 0,
        calls: 0,
        energy: f64::NAN,
        fmax: f64::NAN,
        wall_s: 0.0,
        endpoint: match kind {
            Kind::ZoomFixed => "fixed",
            Kind::ZoomFree => "free",
            _ => "",
        },
    };
    let Ok(mut stencil) = build(sys, kind, stepper, seed) else {
        out.wall_s = started.elapsed().as_secs_f64();
        return out;
    };
    for _ in 0..cap {
        let stepped = match sys {
            Sys::Mb => stencil.step(&Mb),
            Sys::Leps => stencil.step(&Leps),
            Sys::Lj7 | Sys::Lj13 => stencil.step(&Lj),
        };
        let Ok(report) = stepped else { break };
        out.iters = report.iteration;
        out.calls = report.total_evaluations;
        out.fmax = report.max_force;
        if report.status == StencilStatus::Converged {
            break;
        }
    }
    if let Some(energies) = stencil.energies() {
        let ci = energies
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0);
        out.energy = energies[ci];
        let point = stencil.positions().row(ci).to_vec();
        let target = &refs[sys as usize];
        let rmsd = if target.atoms == 1 {
            point
                .iter()
                .zip(&target.x)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        } else {
            kabsch(&point, &target.x)
        };
        let index_ok = !target.check_index || negative_modes(sys, &point) == 1;
        if (out.energy - target.energy).abs() < 1.0e-4 && rmsd < 0.05 && index_ok {
            out.converged = 1;
        }
    }
    out.wall_s = started.elapsed().as_secs_f64();
    out
}

struct Mix {
    z0: f64,
}

impl Mix {
    fn next(&mut self) -> u64 {
        let mut z = self.z0.to_bits().wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn bca(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let n = values.len();
    let theta = values.iter().sum::<f64>() / n as f64;
    if values
        .iter()
        .all(|value| value.to_bits() == theta.to_bits())
    {
        return (theta, theta, theta);
    }
    let mut rng = Mix { z0: 1.0 };
    let mut boots = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut total = 0.0;
        for _ in 0..n {
            total += values[(rng.next() as usize) % n];
        }
        boots.push(total / n as f64);
    }
    boots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_less = boots
        .iter()
        .filter(|value| **value < theta)
        .count()
        .clamp(1, boots.len() - 1);
    let z0 = inv_phi(n_less as f64 / boots.len() as f64);
    let mut jack = Vec::with_capacity(n);
    for skip in 0..n {
        let mut total = 0.0;
        for (index, value) in values.iter().enumerate() {
            if index != skip {
                total += value;
            }
        }
        jack.push(total / (n - 1) as f64);
    }
    let jack_mean = jack.iter().sum::<f64>() / n as f64;
    let mut cubic = 0.0;
    let mut square = 0.0;
    for value in &jack {
        let delta = jack_mean - value;
        cubic += delta * delta * delta;
        square += delta * delta;
    }
    let accel = if square > 0.0 {
        cubic / (6.0 * square.powf(1.5))
    } else {
        0.0
    };
    let percentile = |alpha: f64| -> f64 {
        let z = inv_phi(alpha);
        let numer = z0 + z;
        let denom = 1.0 - accel * numer;
        let adjusted = if denom.abs() < 1.0e-12 {
            alpha
        } else {
            phi(z0 + numer / denom).clamp(0.0, 1.0)
        };
        let position = adjusted * (boots.len() - 1) as f64;
        let lo = position.floor() as usize;
        let hi = position.ceil().min((boots.len() - 1) as f64) as usize;
        let frac = position - lo as f64;
        boots[lo] * (1.0 - frac) + boots[hi] * frac
    };
    (theta, percentile(0.025), percentile(0.975))
}

fn phi(z: f64) -> f64 {
    let sign = z.is_sign_negative();
    let z = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let density = 0.3989422804014327 * (-0.5 * z * z).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = density * poly;
    if sign { tail } else { 1.0 - tail }
}

fn inv_phi(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411385e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

#[test]
fn bca_interval_covers_a_constant() {
    let (mean, lo, hi) = bca(&[2.0, 2.0, 2.0, 2.0]);
    assert!((mean - 2.0).abs() < 1e-12);
    assert!(lo <= 2.0 && hi >= 2.0);
}

#[test]
#[ignore = "ten seeds, five wall repeats, four surfaces"]
fn stencil_benchmark_gate() {
    let _ = table();
    let systems = [Sys::Mb, Sys::Leps, Sys::Lj7, Sys::Lj13];
    let kinds = [
        Kind::Ci,
        Kind::Ew,
        Kind::ZoomFixed,
        Kind::ZoomFree,
        Kind::Auto,
    ];
    let steppers = [Stepper::Fire, Stepper::Lbfgs];
    let mut rows = Vec::new();
    for seed in 0..SEEDS {
        for _repeat in 0..REPEATS {
            for sys in systems {
                for kind in kinds {
                    for stepper in steppers {
                        rows.push(drive_with(sys, kind, stepper, seed));
                    }
                }
            }
        }
    }
    let mut grouped: Vec<Row> = Vec::new();
    for seed in 0..SEEDS {
        for sys in systems {
            for kind in kinds {
                for stepper in steppers {
                    let method = format!(
                        "{}-{}",
                        name_kind(kind),
                        match stepper {
                            Stepper::Fire => "fire",
                            Stepper::Lbfgs => "lbfgs",
                        }
                    );
                    let matches: Vec<&Row> = rows
                        .iter()
                        .filter(|row| {
                            row.seed == seed && row.system == name_sys(sys) && row.method == method
                        })
                        .collect();
                    let mut walls: Vec<f64> = matches.iter().map(|row| row.wall_s).collect();
                    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mut kept = matches[0].clone();
                    kept.wall_s = walls[walls.len() / 2];
                    grouped.push(kept);
                }
            }
        }
    }
    let dir = format!("{}/validation/stencil_gate", env!("CARGO_MANIFEST_DIR"));
    let _ = fs::create_dir_all(&dir);
    let mut csv =
        String::from("system,method,seed,converged,iters,calls,energy,fmax,wall_s,endpoint\n");
    for row in &grouped {
        csv.push_str(&format!(
            "{},{},{},{},{},{},{:.8},{:.8},{:.6},{}\n",
            row.system,
            row.method,
            row.seed,
            row.converged,
            row.iters,
            row.calls,
            row.energy,
            row.fmax,
            row.wall_s,
            row.endpoint
        ));
    }
    fs::write(format!("{dir}/gate.csv"), csv).unwrap();
    println!("ad8s {}", ad8s(&grouped));
    for sys in systems {
        for stepper in steppers {
            let base = format!(
                "ci-neb-{}",
                match stepper {
                    Stepper::Fire => "fire",
                    Stepper::Lbfgs => "lbfgs",
                }
            );
            for kind in [Kind::Ew, Kind::ZoomFixed, Kind::ZoomFree, Kind::Auto] {
                let method = format!(
                    "{}-{}",
                    name_kind(kind),
                    match stepper {
                        Stepper::Fire => "fire",
                        Stepper::Lbfgs => "lbfgs",
                    }
                );
                let ratios: Vec<f64> = (0..SEEDS)
                    .filter_map(|seed| {
                        let left = grouped.iter().find(|row| {
                            row.seed == seed && row.system == name_sys(sys) && row.method == method
                        })?;
                        let right = grouped.iter().find(|row| {
                            row.seed == seed && row.system == name_sys(sys) && row.method == base
                        })?;
                        (right.calls > 0).then_some(left.calls as f64 / right.calls as f64)
                    })
                    .collect();
                let (mean, lo, hi) = bca(&ratios);
                let success = grouped
                    .iter()
                    .filter(|row| {
                        row.system == name_sys(sys) && row.method == method && row.converged == 1
                    })
                    .count();
                let base_success = grouped
                    .iter()
                    .filter(|row| {
                        row.system == name_sys(sys) && row.method == base && row.converged == 1
                    })
                    .count();
                let claim = if hi < 1.0 && success >= base_success {
                    "fewer force calls"
                } else if lo > 1.0 && success >= base_success {
                    "more force calls"
                } else if lo > 1.0 || hi < 1.0 {
                    "success is lower"
                } else {
                    "no measurable difference"
                };
                println!("{sys:?} {method} vs {base}: mean {mean:.3} [{lo:.3}, {hi:.3}] {claim}");
            }
        }
    }
}

fn ad8s(rows: &[Row]) -> &'static str {
    let mut wins = 0;
    for sys in [Sys::Mb, Sys::Leps, Sys::Lj7, Sys::Lj13] {
        let ratios: Vec<f64> = (0..SEEDS)
            .filter_map(|seed| {
                let free = rows.iter().find(|row| {
                    row.seed == seed
                        && row.system == name_sys(sys)
                        && row.method == "zoom-free-fire"
                })?;
                let fixed = rows.iter().find(|row| {
                    row.seed == seed
                        && row.system == name_sys(sys)
                        && row.method == "zoom-fixed-fire"
                })?;
                (fixed.calls > 0).then_some(free.calls as f64 / fixed.calls as f64)
            })
            .collect();
        let (_mean, lo, hi) = bca(&ratios);
        let free_ok = rows
            .iter()
            .filter(|row| {
                row.system == name_sys(sys) && row.method == "zoom-free-fire" && row.converged == 1
            })
            .count();
        let fixed_ok = rows
            .iter()
            .filter(|row| {
                row.system == name_sys(sys) && row.method == "zoom-fixed-fire" && row.converged == 1
            })
            .count();
        if hi < 1.0 && free_ok >= fixed_ok {
            wins += 1;
        }
        let _ = lo;
    }
    if wins >= 3 {
        "flip ZoomEnds::FreePerp"
    } else {
        "keep ZoomEnds::Fixed"
    }
}

/// Burns about `per_row` before the real potential so a block's cost is visible.
struct TimedHost {
    per_row: std::time::Duration,
    lj: bool,
    calls: Mutex<Vec<(u128, usize)>>,
}

fn spin_for(budget: std::time::Duration) {
    if budget.is_zero() {
        return;
    }
    let start = Instant::now();
    let mut acc = 0.1_f64;
    while start.elapsed() < budget {
        acc = acc.mul_add(1.0000001, 1.0e-7);
    }
    std::hint::black_box(acc);
}

impl BandSurface for TimedHost {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let rows = positions.nrows();
        let started = Instant::now();
        for _ in 0..rows {
            spin_for(self.per_row);
        }
        if self.lj {
            BandSurface::eval(&Lj, positions, energies, gradients)?;
        } else {
            BandSurface::eval(&Mb, positions, energies, gradients)?;
        }
        self.calls
            .lock()
            .expect("call log")
            .push((started.elapsed().as_nanos(), rows));
        Ok(())
    }
}

fn scaling_band(lj: bool, images: usize) -> Array2<f64> {
    if lj {
        let (_, reactant) = read_con("reactant");
        let (_, product) = read_con("product");
        line(&reactant, &product, images, 0)
    } else {
        line(
            &[-0.050010822738, 0.466694104872, 0.0],
            &[0.623499404931, 0.028037758529, 0.0],
            images,
            0,
        )
    }
}

fn scaling_config(
    kind: Kind,
    threads: usize,
    band: ndarray::Array2<f64>,
) -> Result<Stencil, SaddleError> {
    let preset = match kind {
        Kind::ZoomFixed | Kind::ZoomFree => Preset::ZoomNeb,
        _ => Preset::CiNeb,
    };
    let mut builder = Stencil::band(band)
        .preset(preset)
        .threads(threads)
        .climb_trigger(0.5, 1.0e300);
    if matches!(kind, Kind::ZoomFixed | Kind::ZoomFree) {
        let mut params = ZoomParams::default();
        params.stability = 1;
        params.force = ZoomForce::Abs(1.0e9);
        params.ends = if matches!(kind, Kind::ZoomFree) {
            ZoomEnds::FreePerp
        } else {
            ZoomEnds::Fixed
        };
        builder = builder.zoom(params);
    }
    builder.build()
}

#[test]
#[ignore = "median of five interleaved repeats at 1, 2, 4, and 8 threads"]
fn strong_scaling_names_the_limiter() {
    // 18 images: 16 interiors. This host has 8 cores and 1 thread per core.
    let images = 18_usize;
    let steps = 6_usize;
    let threads = [1_usize, 2, 4, 8];
    let jobs = [
        (
            "lj13-1ms",
            true,
            Kind::Ci,
            std::time::Duration::from_micros(1100),
        ),
        (
            "lj13-1ms",
            true,
            Kind::ZoomFixed,
            std::time::Duration::from_micros(1100),
        ),
        (
            "lj13-1ms",
            true,
            Kind::ZoomFree,
            std::time::Duration::from_micros(1100),
        ),
        ("muller-brown", false, Kind::Ci, std::time::Duration::ZERO),
    ];
    let mut walls: Vec<Vec<f64>> = vec![Vec::new(); jobs.len() * threads.len()];
    let mut digests: Vec<Option<[u8; 32]>> = vec![None; jobs.len() * threads.len()];
    let mut zoomed = vec![false; jobs.len()];
    let mut spreads = vec![Vec::new(); jobs.len() * threads.len()];
    for _rep in 0..REPEATS {
        for (job_i, (label, lj, kind, burn)) in jobs.iter().enumerate() {
            for (thread_i, count) in threads.into_iter().enumerate() {
                let host = TimedHost {
                    per_row: *burn,
                    lj: *lj,
                    calls: Mutex::new(Vec::new()),
                };
                let started = Instant::now();
                let mut stencil = scaling_config(*kind, count, scaling_band(*lj, images)).unwrap();
                let mut saw_zoom = false;
                for _ in 0..steps {
                    let report = stencil.step_band(&host).unwrap();
                    saw_zoom |= report.phase == PHASE_ZOOM;
                }
                let wall = started.elapsed().as_secs_f64();
                let slot = job_i * threads.len() + thread_i;
                walls[slot].push(wall);
                zoomed[job_i] |= saw_zoom;
                let calls = host.calls.lock().expect("call log");
                if !calls.is_empty() {
                    let min = calls.iter().map(|row| row.0).min().unwrap_or(0);
                    let max = calls.iter().map(|row| row.0).max().unwrap_or(0);
                    spreads[slot].push((max.saturating_sub(min)) as f64 / 1.0e6);
                }
                let mut hasher = Sha256::new();
                for value in stencil.positions().iter() {
                    hasher.update(value.to_le_bytes());
                }
                let digest: [u8; 32] = hasher.finalize().into();
                match digests[slot] {
                    Some(first) => assert_eq!(first, digest, "{label} threads {count}"),
                    None => digests[slot] = Some(digest),
                }
                if thread_i > 0 {
                    assert_eq!(
                        digests[job_i * threads.len()].unwrap(),
                        digest,
                        "{label} bit identity at {count} threads"
                    );
                }
            }
        }
    }
    println!("machine: 8 cores, 1 thread per core, no SMT");
    println!("ceiling: fixed and CI-NEB n-2 = 16; free ends n = 18; images = {images}");
    for (job_i, (label, _, kind, _)) in jobs.iter().enumerate() {
        let t1 = {
            let mut sample = walls[job_i * threads.len()].clone();
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sample[sample.len() / 2]
        };
        println!("{label} {} zoom_fired {}", name_kind(*kind), zoomed[job_i]);
        for (thread_i, count) in threads.into_iter().enumerate() {
            let slot = job_i * threads.len() + thread_i;
            let mut sample = walls[slot].clone();
            sample.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = sample[sample.len() / 2];
            let range = sample.last().unwrap() - sample.first().unwrap();
            let speedup = t1 / median;
            let efficiency = speedup / count as f64;
            let serial = if count > 1 && speedup > 1.0 {
                format!(
                    "{:.3}",
                    (1.0 / speedup - 1.0 / count as f64) / (1.0 - 1.0 / count as f64)
                )
            } else if count > 1 {
                "overhead".to_string()
            } else {
                "0.000".to_string()
            };
            let mut spread = spreads[slot].clone();
            spread.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let spread_med = spread[spread.len() / 2];
            println!(
                "threads {count} median {median:.4}s range {range:.4}s speedup {speedup:.3} efficiency {efficiency:.3} serial_frac {serial} block_spread_ms {spread_med:.3}"
            );
        }
    }
    println!(
        "limiter: fixed image blocks, pool dispatch on the cheap surface, and the serial tangent, spring, and stepper assembly"
    );
}
