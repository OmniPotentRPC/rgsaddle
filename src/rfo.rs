//! Sella `RationalFunctionOptimization` stepper contract.
//!
//! Algebra is [`rgmin::rfo_get_s`]: the `order`-th mode of the
//! Banerjee-augmented matrix
//! \(\alpha\begin{bmatrix}\alpha H & g\\ g^\top & 0\end{bmatrix}\).
//! [`rgmin::NewtonKind::Rfo`] is a different Banerjee form: lambda
//! iteration on \((H-\lambda I)d=-g\). [`rgmin::NewtonKind::Shifted`]
//! is a shift on unaugmented `H`. Alpha lives in \([0, 1]\);
//! \(\alpha = 1\) is the unrestricted step. Sella `newton_safe` is
//! false, so the trust-region companion is bisection
//! ([`rgmin::rfo_restricted`]), not Newton on \(\|s(\alpha)\|\).
//!
//! Ambient reductions go through [`rgmin::vecops`]. A host that
//! wants a point on a set calls [`RationalFunctionOptimization::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

use ndarray::{Array1, Array2};
use rgmin::{Manifold, rfo_get_s, rfo_restricted};

/// Sella `RationalFunctionOptimization` synonyms.
pub const SYNONYMS: &[&str] = &["rfo", "rational function optimization"];

/// Sella `RationalFunctionOptimization.get_s` waist.
///
/// Holds the saddle `order`. `alpha` is an argument of [`Self::get_s`],
/// not stored state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RationalFunctionOptimization {
    /// Sella `order`: 0 for a minimum, 1 for a first-order saddle.
    pub order: usize,
}

impl Default for RationalFunctionOptimization {
    fn default() -> Self {
        Self { order: 0 }
    }
}

impl RationalFunctionOptimization {
    /// Sella `alpha0`.
    pub const ALPHA0: f64 = 1.0;
    /// Sella `alphamin`.
    pub const ALPHAMIN: f64 = 0.0;
    /// Sella `alphamax`.
    pub const ALPHAMAX: f64 = 1.0;
    /// Sella `slope`: \(\|s\|\) grows with alpha.
    pub const SLOPE: f64 = 1.0;
    /// Sella `newton_safe`. RFO is not Newton-safe on \(\|s(\alpha)\|\).
    pub const NEWTON_SAFE: bool = false;

    /// Stepper at the given saddle order.
    pub fn new(order: usize) -> Self {
        Self { order }
    }

    /// Exact Sella `BaseStepper.match`.
    pub fn match_name(name: &str) -> bool {
        SYNONYMS.iter().any(|s| *s == name)
    }

    /// Factory: `Some` when `name` is an RFO synonym.
    pub fn from_name(name: &str, order: usize) -> Option<Self> {
        Self::match_name(name).then_some(Self { order })
    }

    /// Sella `get_s(alpha)`. The increment is ambient Euclidean.
    pub fn get_s(&self, h: &Array2<f64>, g: &Array1<f64>, alpha: f64) -> Array1<f64> {
        rfo_get_s(h, g, self.order, alpha)
    }

    /// Trust-region companion: \(\|s\|\le\delta\), alpha bisected in \([0,1]\).
    pub fn restricted(&self, h: &Array2<f64>, g: &Array1<f64>, delta: f64) -> Array1<f64> {
        rfo_restricted(h, g, self.order, delta)
    }

    /// Riemannian RFO step: Riemannian gradient, tangent projection, retract.
    pub fn step_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        h: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g_r = man.egrad2rgrad(x, g);
        let s = self.get_s(h, &g_r, alpha);
        let v = man.project(x, &s);
        man.retract(x, &v)
    }

    /// Vector transport of the RFO increment from `x` to `x_to`.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        h: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g_r = man.egrad2rgrad(x, g);
        let s = self.get_s(h, &g_r, alpha);
        let v = man.project(x, &s);
        man.transport(x, x_to, &v)
    }
}

/// Factory matching Sella `get_stepper` for the RFO synonyms only.
pub fn rfo_stepper(name: &str, order: usize) -> Option<RationalFunctionOptimization> {
    RationalFunctionOptimization::from_name(name, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rgmin::ManifoldKind;
    use rgmin::vecops::{dot, nrm2};

    #[test]
    fn factory_matches_sella_synonyms() {
        assert!(RationalFunctionOptimization::match_name("rfo"));
        assert!(RationalFunctionOptimization::match_name(
            "rational function optimization"
        ));
        assert!(rfo_stepper("rfo", 1).is_some());
        assert!(rfo_stepper("qn", 0).is_none());
        assert!(rfo_stepper("prfo", 1).is_none());
    }

    #[test]
    fn contract_bounds_match_sella() {
        assert_eq!(RationalFunctionOptimization::ALPHA0, 1.0);
        assert_eq!(RationalFunctionOptimization::ALPHAMIN, 0.0);
        assert_eq!(RationalFunctionOptimization::ALPHAMAX, 1.0);
        assert_eq!(RationalFunctionOptimization::SLOPE, 1.0);
        assert!(!RationalFunctionOptimization::NEWTON_SAFE);
    }

    #[test]
    fn order_zero_on_identity_points_downhill() {
        let h = Array2::<f64>::eye(2);
        let g = array![2.0, 0.0];
        let s = RationalFunctionOptimization::new(0).get_s(&h, &g, 1.0);
        assert!(s[0] < 0.0, "s={s:?}");
        assert!(s[0].abs() > 1e-8);
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn order_selects_a_different_augmented_mode() {
        let h = Array2::<f64>::eye(2);
        let g = array![2.0, 0.5];
        let s0 = RationalFunctionOptimization::new(0).get_s(&h, &g, 1.0);
        let s1 = RationalFunctionOptimization::new(1).get_s(&h, &g, 1.0);
        let diff = nrm2((&s0 - &s1).view());
        assert!(diff > 1e-8, "order ignored: s0={s0:?} s1={s1:?}");
    }

    #[test]
    fn step_grows_with_alpha() {
        let h = Array2::<f64>::eye(2);
        let g = array![2.0, 0.0];
        let stepper = RationalFunctionOptimization::new(0);
        let n_lo = nrm2(stepper.get_s(&h, &g, 0.2).view());
        let n_hi = nrm2(stepper.get_s(&h, &g, 0.8).view());
        assert!(n_hi > n_lo, "slope flipped: {n_lo} vs {n_hi}");
        let n0 = nrm2(stepper.get_s(&h, &g, 0.0).view());
        assert!(n0 < 1e-12, "alpha=0 must vanish: {n0}");
    }

    #[test]
    fn rfo_step_on_the_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let h = Array2::<f64>::eye(3);
        let g = array![1.0, 0.2, -0.3];
        let stepper = RationalFunctionOptimization::new(0);
        let y = stepper.step_on(&man, &x, &h, &g, 1.0);
        let n = nrm2(y.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={y:?}");
        assert!(y.iter().all(|v| v.is_finite()));

        let g_r = man.egrad2rgrad(&x, &g);
        let s = stepper.get_s(&h, &g_r, 1.0);
        let v = man.project(&x, &s);
        assert!(
            dot(x.view(), v.view()).abs() < 1e-12,
            "step is not tangent: x·v={}",
            dot(x.view(), v.view())
        );

        let w = stepper.transport_step(&man, &x, &y, &h, &g, 1.0);
        assert!(
            dot(y.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(y.view(), w.view())
        );
    }

    #[test]
    fn rfo_step_on_rigid_quotient_is_horizontal() {
        let man = ManifoldKind::RigidQuotient;
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let h = Array2::<f64>::eye(9);
        let g = array![0.4, 0.1, 0.0, 0.2, -0.3, 0.0, -0.1, 0.05, 0.0];
        let stepper = RationalFunctionOptimization::new(0);
        let y = stepper.step_on(&man, &x, &h, &g, 1.0);
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

    #[test]
    fn restricted_clips_a_long_rfo_step() {
        let h = Array2::<f64>::eye(2);
        let g = array![10.0, 0.0];
        let stepper = RationalFunctionOptimization::new(0);
        let full = nrm2(stepper.get_s(&h, &g, 1.0).view());
        let delta = 0.05;
        assert!(full > delta, "unrestricted ||s||={full}");
        let s = stepper.restricted(&h, &g, delta);
        let n = nrm2(s.view());
        assert!(n <= delta + 1e-10, "||s||={n} delta={delta} full={full}");
        assert!(s[0] < 0.0, "clipped step must stay downhill: {s:?}");
        assert!(s.iter().all(|v| v.is_finite()));
    }
}
