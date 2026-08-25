//! Sella `PartitionedRationalFunctionOptimization` stepper contract.
//!
//! Algebra is [`rgmin::rfo_get_s`] on the first `order` modes (uphill,
//! Banerjee index = `order`) plus [`rgmin::rfo_get_s`] on the downhill
//! complement (Banerjee index 0). The trust-region companion is
//! [`rgmin::prfo_restricted`]. Distinct from
//! [`crate::rfo::RationalFunctionOptimization`] (one augmented matrix
//! on the whole space) and from [`rgmin::NewtonKind::Rfo`].
//!
//! Alpha lives in \([0, 1]\); \(\alpha = 1\) is the unrestricted
//! partitioned step. Sella `newton_safe` is false, so the trust
//! companion bisects, it does not Newton on \(\|s(\alpha)\|\).
//!
//! Ambient reductions go through [`rgmin::vecops`]. A host that
//! wants a point on a set calls
//! [`PartitionedRationalFunctionOptimization::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

use ndarray::{Array1, Array2};
use rgmin::vecops::{axpy, dot};
use rgmin::{Manifold, prfo_restricted, rfo_get_s};

/// Sella `PartitionedRationalFunctionOptimization` synonyms.
pub const SYNONYMS: &[&str] = &[
    "prfo",
    "p-rfo",
    "partitioned rational function optimization",
];

/// Sella `PartitionedRationalFunctionOptimization.get_s` waist.
///
/// Holds the saddle `order`. Spectrum and gradient are arguments of
/// [`Self::get_s`], not stored state. `alpha` is an argument too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionedRationalFunctionOptimization {
    /// Sella `order`: number of uphill modes. 1 for a first-order saddle.
    pub order: usize,
}

impl Default for PartitionedRationalFunctionOptimization {
    fn default() -> Self {
        Self { order: 0 }
    }
}

impl PartitionedRationalFunctionOptimization {
    /// Sella `alpha0`. Inherited from RFO.
    pub const ALPHA0: f64 = 1.0;
    /// Sella `alphamin`.
    pub const ALPHAMIN: f64 = 0.0;
    /// Sella `alphamax`.
    pub const ALPHAMAX: f64 = 1.0;
    /// Sella `slope`: \(\|s\|\) grows with alpha.
    pub const SLOPE: f64 = 1.0;
    /// Sella `newton_safe`. P-RFO is not Newton-safe on \(\|s(\alpha)\|\).
    pub const NEWTON_SAFE: bool = false;

    /// Stepper at the given saddle order.
    pub fn new(order: usize) -> Self {
        Self { order }
    }

    /// Exact Sella `BaseStepper.match`.
    pub fn match_name(name: &str) -> bool {
        SYNONYMS.iter().any(|s| *s == name)
    }

    /// Factory: `Some` when `name` is a P-RFO synonym.
    pub fn from_name(name: &str, order: usize) -> Option<Self> {
        Self::match_name(name).then_some(Self { order })
    }

    /// Sella `get_s(alpha)`. The increment is ambient Euclidean.
    pub fn get_s(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        prfo_get_s(evals, evecs, g, self.order, alpha)
    }

    /// Trust-region companion: \(\|s\|\le\delta\) via [`prfo_restricted`].
    pub fn restricted(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        delta: f64,
    ) -> Array1<f64> {
        prfo_restricted(evals, evecs, g, self.order, delta)
    }

    /// Riemannian P-RFO step: Riemannian gradient, tangent projection, retract.
    pub fn step_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g_r = man.egrad2rgrad(x, g);
        let s = self.get_s(evals, evecs, &g_r, alpha);
        let v = man.project(x, &s);
        man.retract(x, &v)
    }

    /// Vector transport of the P-RFO increment from `x` to `x_to`.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let g_r = man.egrad2rgrad(x, g);
        let s = self.get_s(evals, evecs, &g_r, alpha);
        let v = man.project(x, &s);
        man.transport(x, x_to, &v)
    }
}

/// Sella `PartitionedRationalFunctionOptimization.get_s`.
///
/// Project `g` onto the first `order` eigenvectors and onto the
/// complement. Run RFO in each block, then lift with [`axpy`].
pub fn prfo_get_s(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> Array1<f64> {
    let n = g
        .len()
        .min(evals.len())
        .min(evecs.nrows())
        .min(evecs.ncols());
    let order = order.min(n);
    if n == 0 {
        return Array1::zeros(g.len());
    }
    let mut s = Array1::zeros(g.len());
    if order > 0 {
        let mut gmax = Array1::zeros(order);
        let mut hmax = Array2::<f64>::zeros((order, order));
        for i in 0..order {
            hmax[(i, i)] = evals[i];
            gmax[i] = dot(evecs.column(i), g.view());
        }
        let smax = rfo_get_s(&hmax, &gmax, order, alpha);
        for i in 0..order {
            axpy(smax[i], evecs.column(i), &mut s);
        }
    }
    let nmin = n - order;
    if nmin > 0 {
        let mut gmin = Array1::zeros(nmin);
        let mut hmin = Array2::<f64>::zeros((nmin, nmin));
        for i in 0..nmin {
            hmin[(i, i)] = evals[order + i];
            gmin[i] = dot(evecs.column(order + i), g.view());
        }
        let smin = rfo_get_s(&hmin, &gmin, 0, alpha);
        for i in 0..nmin {
            axpy(smin[i], evecs.column(order + i), &mut s);
        }
    }
    s
}

/// Factory matching Sella `get_stepper` for the P-RFO synonyms only.
pub fn prfo_stepper(name: &str, order: usize) -> Option<PartitionedRationalFunctionOptimization> {
    PartitionedRationalFunctionOptimization::from_name(name, order)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rgmin::ManifoldKind;
    use rgmin::vecops::{dot, nrm2};

    #[test]
    fn factory_matches_sella_synonyms() {
        assert!(PartitionedRationalFunctionOptimization::match_name("prfo"));
        assert!(PartitionedRationalFunctionOptimization::match_name("p-rfo"));
        assert!(PartitionedRationalFunctionOptimization::match_name(
            "partitioned rational function optimization"
        ));
        assert!(prfo_stepper("prfo", 1).is_some());
        assert!(prfo_stepper("qn", 0).is_none());
        assert!(prfo_stepper("rfo", 1).is_none());
        assert!(prfo_stepper("rational function optimization", 1).is_none());
    }

    #[test]
    fn contract_bounds_match_sella() {
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHA0, 1.0);
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHAMIN, 0.0);
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHAMAX, 1.0);
        assert_eq!(PartitionedRationalFunctionOptimization::SLOPE, 1.0);
        assert!(!PartitionedRationalFunctionOptimization::NEWTON_SAFE);
        assert_eq!(PartitionedRationalFunctionOptimization::default().order, 0);
    }

    #[test]
    fn order_one_moves_along_the_soft_mode() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let s = PartitionedRationalFunctionOptimization::new(1).get_s(&evals, &evecs, &g, 1.0);
        assert!(s[0].abs() > 1e-8, "no uphill component: {s:?}");
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn partition_differs_from_unpartitioned_rfo() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let h = {
            let mut m = Array2::<f64>::zeros((2, 2));
            m[(0, 0)] = -1.0;
            m[(1, 1)] = 4.0;
            m
        };
        let s_p = PartitionedRationalFunctionOptimization::new(1).get_s(&evals, &evecs, &g, 1.0);
        let s_r = crate::rfo::RationalFunctionOptimization::new(1).get_s(&h, &g, 1.0);
        let diff = nrm2((&s_p - &s_r).view());
        assert!(
            diff > 1e-8,
            "P-RFO collapsed to RFO: s_p={s_p:?} s_r={s_r:?}"
        );
    }

    #[test]
    fn step_grows_with_alpha() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let stepper = PartitionedRationalFunctionOptimization::new(1);
        let n_lo = nrm2(stepper.get_s(&evals, &evecs, &g, 0.2).view());
        let n_hi = nrm2(stepper.get_s(&evals, &evecs, &g, 0.8).view());
        assert!(n_hi > n_lo, "slope flipped: {n_lo} vs {n_hi}");
        let n0 = nrm2(stepper.get_s(&evals, &evecs, &g, 0.0).view());
        assert!(n0 < 1e-12, "alpha=0 must vanish: {n0}");
    }

    #[test]
    fn prfo_step_on_the_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let evals = array![-1.0, 2.0, 3.0];
        let evecs = Array2::<f64>::eye(3);
        let g = array![1.0, 0.2, -0.3];
        let stepper = PartitionedRationalFunctionOptimization::new(1);
        let y = stepper.step_on(&man, &x, &evals, &evecs, &g, 1.0);
        let n = nrm2(y.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={y:?}");
        assert!(y.iter().all(|v| v.is_finite()));

        let g_r = man.egrad2rgrad(&x, &g);
        let s = stepper.get_s(&evals, &evecs, &g_r, 1.0);
        let v = man.project(&x, &s);
        assert!(
            dot(x.view(), v.view()).abs() < 1e-12,
            "step is not tangent: x·v={}",
            dot(x.view(), v.view())
        );

        let w = stepper.transport_step(&man, &x, &y, &evals, &evecs, &g, 1.0);
        assert!(
            dot(y.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(y.view(), w.view())
        );
    }

    #[test]
    fn prfo_step_on_rigid_quotient_is_horizontal() {
        let man = ManifoldKind::RigidQuotient;
        let x = array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut evals = Array1::ones(9);
        evals[0] = -1.0;
        let evecs = Array2::<f64>::eye(9);
        let g = array![0.4, 0.1, 0.0, 0.2, -0.3, 0.0, -0.1, 0.05, 0.0];
        let stepper = PartitionedRationalFunctionOptimization::new(1);
        let y = stepper.step_on(&man, &x, &evals, &evecs, &g, 1.0);
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
    fn restricted_clips_a_long_prfo_step() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![10.0, 8.0];
        let stepper = PartitionedRationalFunctionOptimization::new(1);
        let full = nrm2(stepper.get_s(&evals, &evecs, &g, 1.0).view());
        let delta = 0.05;
        assert!(full > delta, "unrestricted ||s||={full}");
        let s = stepper.restricted(&evals, &evecs, &g, delta);
        let n = nrm2(s.view());
        assert!(n <= delta + 1e-10, "||s||={n} delta={delta} full={full}");
        assert!(s.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn order_zero_is_downhill_rfo_on_the_whole_space() {
        let evals = array![2.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![2.0, 0.0];
        let s = PartitionedRationalFunctionOptimization::new(0).get_s(&evals, &evecs, &g, 1.0);
        assert!(s[0] < 0.0, "s={s:?}");
        assert!(s[0].abs() > 1e-8);
        let h = {
            let mut m = Array2::<f64>::zeros((2, 2));
            m[(0, 0)] = 2.0;
            m[(1, 1)] = 4.0;
            m
        };
        let s_r = crate::rfo::RationalFunctionOptimization::new(0).get_s(&h, &g, 1.0);
        let diff = nrm2((&s - &s_r).view());
        assert!(diff < 1e-10, "order-0 P-RFO must match RFO: {diff}");
    }
}
