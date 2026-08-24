//! Sella `QuasiNewton` stepper waist.
//!
//! `sella.optimize.stepper.QuasiNewton`. Algebra is
//! [`rgmin::qn_get_s`] / [`rgmin::qn_restricted`]. This module is the
//! rgsaddle factory (`get_stepper("qn")`), not a session and not
//! QuasiNewtonIRC.
//!
//! Ambient reductions go through [`rgmin::vecops`]. A host that
//! wants a point on a set calls [`QuasiNewton::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

pub use rgmin::{qn_get_s, qn_restricted};

use ndarray::{Array1, Array2};
use rgmin::Manifold;

/// Sella `QuasiNewton.alpha0`. Slope is \(-1\): larger \(\alpha\)
/// shortens the step.
pub const ALPHA0: f64 = 0.0;
/// Sella `QuasiNewton.alphamin`.
pub const ALPHA_MIN: f64 = 0.0;
/// Sella `QuasiNewton.alphamax`.
pub const ALPHA_MAX: f64 = f64::INFINITY;
/// Sella `QuasiNewton.slope`.
pub const SLOPE: f64 = -1.0;
/// Sella `newton_safe`. QN is Newton-safe on \(\|s(\alpha)\|\).
pub const NEWTON_SAFE: bool = true;

/// Names [`matches`] accepts. Sella `QuasiNewton.synonyms`.
pub const SYNONYMS: &[&str] = &[
    "qn",
    "quasi-newton",
    "quasi newton",
    "newton",
    "mmf",
    "minimum mode following",
    "minimum-mode following",
    "dimer",
];

/// True when `name` is a Sella QuasiNewton synonym.
pub fn matches(name: &str) -> bool {
    let n = name.trim().to_ascii_lowercase();
    SYNONYMS.iter().any(|s| *s == n)
}

/// Sella `get_stepper`. Only QuasiNewton lives here; RFO / P-RFO
/// are their own tickets.
pub fn get_stepper(name: &str) -> Option<StepperKind> {
    if matches(name) {
        Some(StepperKind::QuasiNewton)
    } else {
        None
    }
}

/// Named Sella stepper this factory can mint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepperKind {
    /// `sella.optimize.stepper.QuasiNewton`.
    QuasiNewton,
}

/// Sella `QuasiNewton`: spectrum, gradient, and saddle `order`.
#[derive(Clone, Debug)]
pub struct QuasiNewton {
    evals: Array1<f64>,
    evecs: Array2<f64>,
    g: Array1<f64>,
    order: usize,
}

impl QuasiNewton {
    /// `evals` / `evecs` from a symmetric Hessian; `order` is the
    /// number of uphill modes Sella flips (`L_i ← -|L_i|`).
    pub fn new(evals: &Array1<f64>, evecs: &Array2<f64>, g: &Array1<f64>, order: usize) -> Self {
        Self {
            evals: evals.clone(),
            evecs: evecs.clone(),
            g: g.clone(),
            order,
        }
    }

    /// Sella `QuasiNewton.get_s`: step and \(ds/d\alpha\).
    pub fn get_s(&self, alpha: f64) -> (Array1<f64>, Array1<f64>) {
        qn_get_s(&self.evals, &self.evecs, &self.g, self.order, alpha)
    }

    /// Sella TrustRegion + QuasiNewton: \(\|s\| \le \delta\).
    pub fn restricted(&self, delta: f64) -> Array1<f64> {
        qn_restricted(&self.evals, &self.evecs, &self.g, self.order, delta)
    }

    /// Riemannian QN step: Riemannian gradient, tangent projection, retract.
    pub fn step_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        egrad: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g = man.egrad2rgrad(x, egrad);
        let (s, _) = qn_get_s(&self.evals, &self.evecs, &g, self.order, alpha);
        let v = man.project(x, &s);
        man.retract(x, &v)
    }

    /// Vector transport of the QN increment from `x` to `x_to`.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        egrad: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g = man.egrad2rgrad(x, egrad);
        let (s, _) = qn_get_s(&self.evals, &self.evecs, &g, self.order, alpha);
        let v = man.project(x, &s);
        man.transport(x, x_to, &v)
    }
}

/// Tangent QN increment: rgrad, `get_s`, project.
pub fn qn_tangent<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    egrad: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> Array1<f64> {
    let g = man.egrad2rgrad(x, egrad);
    let (s, _) = qn_get_s(evals, evecs, &g, order, alpha);
    man.project(x, &s)
}

/// Riemannian QN step: egrad → rgrad, `get_s`, project, retract.
pub fn retract_qn<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    egrad: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> Array1<f64> {
    let s = qn_tangent(man, x, evals, evecs, egrad, order, alpha);
    man.retract(x, &s)
}

/// Alias of [`retract_qn`].
pub fn qn_retract<M: Manifold>(
    man: &M,
    x: &Array1<f64>,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    egrad: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> Array1<f64> {
    retract_qn(man, x, evals, evecs, egrad, order, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rgmin::vecops::{dot, nrm2};
    use rgmin::ManifoldKind;

    #[test]
    fn matches_qn_and_rejects_rfo() {
        assert!(matches("QN"));
        assert!(matches(" quasi-newton "));
        assert!(matches("minimum-mode following"));
        assert!(!matches("rfo"));
        assert!(!matches("prfo"));
        assert!(get_stepper("qn") == Some(StepperKind::QuasiNewton));
        assert!(get_stepper("rfo").is_none());
    }

    #[test]
    fn contract_bounds_match_sella() {
        assert_eq!(ALPHA0, 0.0);
        assert_eq!(ALPHA_MIN, 0.0);
        assert!(ALPHA_MAX.is_infinite());
        assert_eq!(SLOPE, -1.0);
        assert!(NEWTON_SAFE);
    }

    #[test]
    fn alpha_zero_is_signed_newton() {
        let evals = array![4.0, -2.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![8.0, 2.0];
        let (s, _) = QuasiNewton::new(&evals, &evecs, &g, 1).get_s(0.0);
        // Sella flips the first `order` slots: L = (-|4|, |-2|) = (-4, 2);
        // s = -g/L = (2, -1).
        assert!((s[0] - 2.0).abs() < 1e-12);
        assert!((s[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn restricted_clips_a_long_newton_step() {
        let evals = array![1.0, 1.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![4.0, 0.0];
        let s = QuasiNewton::new(&evals, &evecs, &g, 0).restricted(0.5);
        let n = nrm2(s.view());
        assert!((n - 0.5).abs() < 1e-9, "||s||={n}");
    }

    #[test]
    fn qn_step_on_the_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let evals = Array1::ones(3);
        let evecs = Array2::<f64>::eye(3);
        let egrad = array![1.0, 0.2, -0.3];
        let stepper = QuasiNewton::new(&evals, &evecs, &egrad, 0);
        let y = stepper.step_on(&man, &x, &egrad, ALPHA0);
        let n = nrm2(y.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={y:?}");
        assert!(y.iter().all(|v| v.is_finite()));

        let g_r = man.egrad2rgrad(&x, &egrad);
        let (s, _) = qn_get_s(&evals, &evecs, &g_r, 0, ALPHA0);
        let v = man.project(&x, &s);
        assert!(
            dot(x.view(), v.view()).abs() < 1e-12,
            "step is not tangent: x·v={}",
            dot(x.view(), v.view())
        );

        let w = stepper.transport_step(&man, &x, &y, &egrad, ALPHA0);
        assert!(
            dot(y.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(y.view(), w.view())
        );
    }

    #[test]
    fn qn_step_on_rigid_quotient_is_horizontal() {
        let man = ManifoldKind::RigidQuotient;
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let evals = Array1::ones(9);
        let evecs = Array2::<f64>::eye(9);
        let egrad = array![0.4, 0.1, 0.0, 0.2, -0.3, 0.0, -0.1, 0.05, 0.0];
        let stepper = QuasiNewton::new(&evals, &evecs, &egrad, 0);
        let y = stepper.step_on(&man, &x, &egrad, ALPHA0);
        assert_eq!(y.len(), 9);
        assert!(y.iter().all(|v| v.is_finite()));
        let dx = &y - &x;
        let horiz = man.project(&x, &dx);
        let leak = nrm2((&dx - &horiz).view());
        assert!(
            leak < 1e-12,
            "retracted increment left the horizontal: {leak}"
        );
    }
}
