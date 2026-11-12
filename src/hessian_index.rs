//! Finite-difference Hessian index on the true surface.
//!
//! A converged minimum-mode point is a first-order saddle only when
//! this count is one. Rigid translation and rotation of a free cluster
//! are projected out before the count, the same shift used when a
//! connection checks the vibrational index.

use ndarray::{Array1, Array2, ArrayView1};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Negative eigenvalues below this are the unstable modes. Rigid-body
/// eigenvalues sit near zero and are not counted.
const NEGATIVE: f64 = -1e-3;

pub struct HessianIndex {
    pub negative: usize,
    pub lowest: f64,
    /// Next eigenvalue after `lowest`. Positive when the point is index one.
    pub second: f64,
    pub mode: Array1<f64>,
    pub evaluations: usize,
}

/// Central-difference Hessian index at `position`.
///
/// `project_rigid` removes translation and rotation of a free molecule
/// (at least two atoms). A periodic cell or a single free atom leaves
/// the Hessian as it is.
pub fn finite_difference_index<S: PointSurface>(
    surface: &S,
    position: ArrayView1<f64>,
    project_rigid: bool,
) -> Result<HessianIndex, SaddleError> {
    let n = position.len();
    if n == 0 || !position.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Invalid(
            "a Hessian index needs a finite geometry".into(),
        ));
    }
    let step = 1e-4;
    let mut hess = Array2::<f64>::zeros((n, n));
    let mut evaluations = 0usize;
    for col in 0..n {
        let mut plus = position.to_owned();
        let mut minus = position.to_owned();
        plus[col] += step;
        minus[col] -= step;
        let (_, gp) = surface.eval(plus.view())?;
        let (_, gm) = surface.eval(minus.view())?;
        evaluations += 2;
        for row in 0..n {
            hess[(row, col)] = (gp[row] - gm[row]) / (2.0 * step);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = mean;
            hess[(j, i)] = mean;
        }
    }
    if project_rigid && n.is_multiple_of(3) && n >= 6 {
        hess = project_rigid_modes(&hess, position.as_slice().unwrap());
    }
    let (evals, evecs) = exact_eigh(hess.view())?;
    let mut negative = 0usize;
    for value in evals.iter() {
        if *value < NEGATIVE {
            negative += 1;
        }
    }
    let mode = evecs.column(0).to_owned();
    Ok(HessianIndex {
        negative,
        lowest: evals[0],
        second: evals.get(1).copied().unwrap_or(0.0),
        mode,
        evaluations,
    })
}

fn project_rigid_modes(hess: &Array2<f64>, pos: &[f64]) -> Array2<f64> {
    let modes = rigid_basis(pos);
    let n = hess.nrows();
    if modes.is_empty() {
        return hess.clone();
    }
    let mut out = Array2::zeros((n, n));
    for j in 0..n {
        let mut col = hess.column(j).to_owned();
        project_out(&mut col, &modes);
        for i in 0..n {
            out[(i, j)] = col[i];
        }
    }
    for i in 0..n {
        let mut row = out.row(i).to_owned();
        project_out(&mut row, &modes);
        for j in 0..n {
            out[(i, j)] = row[j];
        }
    }
    for mode in &modes {
        for i in 0..n {
            for j in 0..n {
                out[(i, j)] += mode[i] * mode[j];
            }
        }
    }
    out
}

fn project_out(v: &mut Array1<f64>, modes: &[Vec<f64>]) {
    for mode in modes {
        let c: f64 = mode.iter().zip(v.iter()).map(|(m, x)| m * x).sum();
        for (x, m) in v.iter_mut().zip(mode) {
            *x -= c * *m;
        }
    }
}

fn rigid_basis(pos: &[f64]) -> Vec<Vec<f64>> {
    let atoms = pos.len() / 3;
    if atoms < 2 {
        return Vec::new();
    }
    let mut center = [0.0; 3];
    for i in 0..atoms {
        for k in 0..3 {
            center[k] += pos[3 * i + k];
        }
    }
    for slot in &mut center {
        *slot /= atoms as f64;
    }
    let mut seeds = Vec::new();
    for axis in 0..3 {
        let mut translation = vec![0.0; pos.len()];
        for i in 0..atoms {
            translation[3 * i + axis] = 1.0;
        }
        seeds.push(translation);
        let mut rotation = vec![0.0; pos.len()];
        for i in 0..atoms {
            let p = [
                pos[3 * i] - center[0],
                pos[3 * i + 1] - center[1],
                pos[3 * i + 2] - center[2],
            ];
            let rot = match axis {
                0 => [0.0, p[2], -p[1]],
                1 => [-p[2], 0.0, p[0]],
                _ => [p[1], -p[0], 0.0],
            };
            for k in 0..3 {
                rotation[3 * i + k] = rot[k];
            }
        }
        seeds.push(rotation);
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for seed in seeds {
        let mut mode = seed;
        for kept in &basis {
            let c: f64 = kept.iter().zip(mode.iter()).map(|(a, b)| a * b).sum();
            for (m, k) in mode.iter_mut().zip(kept) {
                *m -= c * *k;
            }
        }
        let norm = mode.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 1e-8 {
            for m in &mut mode {
                *m /= norm;
            }
            basis.push(mode);
        }
    }
    basis
}
