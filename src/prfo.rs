//! Restricted-step partitioned rational function optimization.
//!
//! Baker, *J. Comput. Chem.* 7 (1986) 385,
//! <https://doi.org/10.1002/jcc.540070402>, splits an index-1 step
//! into two rational-function problems: a 2 by 2 problem that
//! maximizes along the lowest mode, and one problem that minimizes
//! on the complement. Besalu and Bofill, *Theor. Chem. Acc.* 100
//! (1998) 265, <https://doi.org/10.1007/s002140050387>, scale that
//! augmented Hessian by `alpha` and solve `alpha` so the step lies
//! inside the trust sphere. `alpha = 1` is the pure partition. A
//! step already inside the sphere is kept. The sphere is the
//! Euclidean norm of the Cartesian displacement.
//!
//! The i-PI Nichols shift is a different displacement and stays in
//! [`crate::nichols`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;
use crate::nichols::{mass_weight, masses_or_ones, sym_eigh};

/// Which stationary structure the partition is aimed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrfoKind {
    /// Maximize along the lowest mode and minimize along the rest.
    Index1,
    /// Minimize along every mode. One restricted-step RFO.
    Minimize,
}

fn secular_minimize(evals: &[f64], gradient: &[f64], alpha: f64) -> Result<f64, SaddleError> {
    let e0 = evals[0];
    let mut g2 = 0.0;
    for g in gradient {
        g2 += g * g;
    }
    let mut mu = 0.5 * (e0 - (e0 * e0 + 4.0 * alpha * g2).sqrt());
    for _ in 0..50 {
        let mut f = -mu / alpha;
        let mut fp = -1.0 / alpha;
        for (&ei, &gi) in evals.iter().zip(gradient.iter()) {
            let d = mu - ei;
            f += gi * gi / d;
            fp += -(gi * gi) / (d * d);
        }
        if !f.is_finite() || !fp.is_finite() {
            return Err(SaddleError::NonFinite("RFO secular equation"));
        }
        if f.abs() < 1e-14 * (1.0 + mu.abs()) {
            break;
        }
        let mut mu_new = mu - f / fp;
        if !mu_new.is_finite() || mu_new >= e0 {
            mu_new = 0.5 * (mu + e0) - 1e-12;
        }
        mu = mu_new;
    }
    if !mu.is_finite() || mu >= e0 {
        return Err(SaddleError::NonFinite("RFO level shift"));
    }
    Ok(mu)
}

fn maximize_root(curvature: f64, gradient: f64, alpha: f64) -> f64 {
    let disc = (curvature * curvature + 4.0 * alpha * gradient * gradient).sqrt();
    0.5 * (curvature + disc)
}

/// Partitioned RFO step in the eigenbasis at a fixed scaling `alpha`.
///
/// `evals` ascend. `gradient` is the energy gradient in that basis.
/// [`PrfoKind::Index1`] maximizes mode 0. [`PrfoKind::Minimize`]
/// minimizes every mode.
pub fn partitioned_rfo_eigen(
    evals: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    alpha: f64,
    kind: PrfoKind,
) -> Result<Array1<f64>, SaddleError> {
    let n = evals.len();
    if n == 0 || gradient.len() != n {
        return Err(SaddleError::Shape(
            "eigenvalues and the eigenbasis gradient must agree".into(),
        ));
    }
    if !(alpha.is_finite() && alpha > 0.0) {
        return Err(SaddleError::Invalid(
            "RFO scaling must be positive and finite".into(),
        ));
    }
    if kind == PrfoKind::Index1 && n < 2 {
        return Err(SaddleError::Shape(
            "an index-1 partition needs two or more modes".into(),
        ));
    }
    let mut step = Array1::<f64>::zeros(n);
    let min_from = if kind == PrfoKind::Index1 {
        let mu = maximize_root(evals[0], gradient[0], alpha);
        let denom = mu - evals[0];
        step[0] = if gradient[0].abs() < 1e-16 || denom.abs() < 1e-18 {
            0.0
        } else {
            gradient[0] / denom
        };
        1
    } else {
        0
    };
    if min_from < n {
        let eps: Vec<f64> = evals.iter().skip(min_from).copied().collect();
        let g: Vec<f64> = gradient.iter().skip(min_from).copied().collect();
        let mu = secular_minimize(&eps, &g, alpha)?;
        for i in min_from..n {
            step[i] = gradient[i] / (mu - evals[i]);
        }
    }
    if step.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("partitioned RFO step"));
    }
    Ok(step)
}

fn l2(v: ArrayView1<f64>) -> f64 {
    v.dot(&v).sqrt()
}

fn cartesian_of(evecs: &Array2<f64>, step_eig: &Array1<f64>, mass: &Array1<f64>) -> Array1<f64> {
    let n = mass.len();
    let mut step = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..step_eig.len() {
            acc += evecs[(i, k)] * step_eig[k];
        }
        step[i] = acc / mass[i].sqrt();
    }
    step
}

/// Baker restricted-step partition.
///
/// `evals` and `gradient` are in the orthonormal eigenbasis. `evecs`
/// has those eigenvectors as columns. `mass` is the per-coordinate
/// mass that takes the mass-weighted step back to Cartesian
/// coordinates. The trust radius bounds the Cartesian Euclidean norm.
pub fn restricted_partitioned_rfo(
    evals: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    evecs: ArrayView2<f64>,
    mass: ArrayView1<f64>,
    trust_radius: f64,
    kind: PrfoKind,
) -> Result<Array1<f64>, SaddleError> {
    if !(trust_radius.is_finite() && trust_radius > 0.0) {
        return Err(SaddleError::Invalid(
            "trust radius must be positive and finite".into(),
        ));
    }
    let at = |alpha: f64| -> Result<Array1<f64>, SaddleError> {
        let step_eig = partitioned_rfo_eigen(evals, gradient, alpha, kind)?;
        let step = cartesian_of(&evecs.to_owned(), &step_eig, &mass.to_owned());
        if step.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("restricted P-RFO step"));
        }
        Ok(step)
    };
    let pure = at(1.0)?;
    if l2(pure.view()) <= trust_radius {
        return Ok(pure);
    }
    let mut lo = 1.0;
    let mut hi = 2.0;
    let mut bracketed = false;
    for _ in 0..40 {
        let trial = at(hi)?;
        if l2(trial.view()) <= trust_radius {
            bracketed = true;
            break;
        }
        hi *= 2.0;
    }
    if !bracketed {
        return Err(SaddleError::Invalid(
            "no RFO scaling puts the step inside the trust radius".into(),
        ));
    }
    for _ in 0..70 {
        let mid = 0.5 * (lo + hi);
        let trial = at(mid)?;
        if l2(trial.view()) > trust_radius {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    at(hi)
}

/// Restricted-step partition from a Cartesian energy Hessian and gradient.
pub fn restricted_prfo_displacement(
    hessian: ArrayView2<f64>,
    gradient: ArrayView1<f64>,
    masses: Option<ArrayView1<f64>>,
    trust_radius: f64,
    kind: PrfoKind,
) -> Result<Array1<f64>, SaddleError> {
    let n = gradient.len();
    if hessian.nrows() != n || hessian.ncols() != n {
        return Err(SaddleError::Shape(
            "Hessian must be square and match the gradient".into(),
        ));
    }
    if gradient.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("gradient"));
    }
    let mass = masses_or_ones(n, masses)?;
    let weighted = mass_weight(hessian, &mass)?;
    let (evals, evecs) = sym_eigh(weighted.view())?;
    let mut gradient_eig = Array1::<f64>::zeros(n);
    for k in 0..n {
        let mut acc = 0.0;
        for i in 0..n {
            acc += evecs[(i, k)] * gradient[i] / mass[i].sqrt();
        }
        gradient_eig[k] = acc;
    }
    restricted_partitioned_rfo(
        evals.view(),
        gradient_eig.view(),
        evecs.view(),
        mass.view(),
        trust_radius,
        kind,
    )
}
