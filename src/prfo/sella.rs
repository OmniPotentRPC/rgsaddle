//! Sella `PartitionedRationalFunctionOptimization` stepper contract.
//!
//! Algebra splits the Hessian eigenbasis into the `order` uphill
//! modes and the downhill complement, then runs
//! [`rgmin::rfo_get_s`] in each block. The restricted companion is
//! [`rgmin::prfo_restricted`]. Sella inherits the RFO alpha
//! contract: \(\alpha\in[0,1]\), slope \(+1\), `newton_safe` false.
//! Distinct from [`crate::rfo::RationalFunctionOptimization`] (one
//! Banerjee eigenproblem on the full space) and from
//! [`rgmin::NewtonKind::Rfo`].
//!
//! Ambient reductions go through [`rgmin::vecops`]. A host that
//! wants a point on a set calls
//! [`PartitionedRationalFunctionOptimization::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

use ndarray::{Array1, Array2};
use rgmin::vecops::{axpy, dot, nrm2};
use rgmin::{Manifold, prfo_restricted, rfo_get_s};

/// Widest partition that stays on cyclic Jacobi.
///
/// At this width and below the step is the [`rgmin::rfo_get_s`] call it
/// was before, so the small P-RFO fixtures and an index-1 uphill block
/// keep Jacobi. Any wider partition takes faer, in a test as in a climb:
/// the LJ38 and Pt-island downhill blocks, and the 59-wide downhill block
/// behind `secular_root_matches_the_augmented_step_on_a_wide_diagonal`.
const AUGMENTED_JACOBI_LIMIT: usize = 32;

/// Floor on the last augmented coordinate, matching rgmin `rfo_get_s`.
const AUGMENTED_DENOM_FLOOR: f64 = 1e-12;

const TRUST_ITERS: usize = 64;

#[cfg(test)]
thread_local! {
    static FAER_STEPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Sella `TrustRegion` + P-RFO: one alpha search on \(\|s\|\le\delta\).
///
/// Distinct from dest [`prfo_restricted`], which restricts each
/// eigenblock then clips the sum. Sella Optimizer `rs=tr` is this
/// search on the full partitioned step. The returned displacement is clipped
/// to the physical radius independently of the alpha-search tolerance.
/// A nonfinite or negative radius returns a zero displacement.
pub fn prfo_trust_region(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    delta: f64,
) -> Array1<f64> {
    let Ok(trust) = crate::restricted::TrustRegion::new(delta) else {
        return Array1::zeros(g.len());
    };
    let stepper = PartitionedRationalFunctionOptimization::new(order);
    let s1 = stepper.get_s(
        evals,
        evecs,
        g,
        PartitionedRationalFunctionOptimization::ALPHA0,
    );
    let n1 = nrm2(s1.view());
    if n1 <= delta {
        return s1;
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    let mut best = s1;
    for _ in 0..TRUST_ITERS {
        let mid = 0.5 * (lo + hi);
        let s = stepper.get_s(evals, evecs, g, mid);
        let val = nrm2(s.view());
        best = s;
        if (val - delta).abs() <= 1e-10 {
            return trust.clip(&best);
        }
        if val > delta {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    trust.clip(&best)
}

use crate::rfo::RationalFunctionOptimization;

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PartitionedRationalFunctionOptimization {
    /// Sella `order`: number of uphill modes (1 for a first-order saddle).
    pub order: usize,
}

impl PartitionedRationalFunctionOptimization {
    /// Sella `alpha0` (inherited from RFO).
    pub const ALPHA0: f64 = RationalFunctionOptimization::ALPHA0;
    /// Sella `alphamin`.
    pub const ALPHAMIN: f64 = RationalFunctionOptimization::ALPHAMIN;
    /// Sella `alphamax`.
    pub const ALPHAMAX: f64 = RationalFunctionOptimization::ALPHAMAX;
    /// Sella `slope`: \(\|s\|\) grows with alpha.
    pub const SLOPE: f64 = RationalFunctionOptimization::SLOPE;
    /// Sella `newton_safe`. P-RFO is not Newton-safe on \(\|s(\alpha)\|\).
    pub const NEWTON_SAFE: bool = RationalFunctionOptimization::NEWTON_SAFE;

    /// Stepper at the given saddle order.
    pub fn new(order: usize) -> Self {
        Self { order }
    }

    /// Exact Sella `BaseStepper.match`.
    pub fn match_name(name: &str) -> bool {
        SYNONYMS.contains(&name)
    }

    /// Factory: `Some` when `name` is a P-RFO synonym.
    pub fn from_name(name: &str, order: usize) -> Option<Self> {
        Self::match_name(name).then_some(Self { order })
    }

    /// Sella `get_s(alpha)`. The increment is ambient Euclidean.
    ///
    /// `evals` / `evecs` are the symmetric Hessian spectrum. The first
    /// `order` columns are the uphill block (Sella `Vmax`); the rest
    /// are the downhill complement (`Vmin`). Each block is diagonal, so
    /// the Banerjee augmented matrix is an arrowhead. A block wider
    /// than [`AUGMENTED_JACOBI_LIMIT`] asks faer for that one eigenpair.
    /// A narrower block keeps [`rgmin::rfo_get_s`].
    pub fn get_s(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        alpha: f64,
    ) -> Array1<f64> {
        let n = g.len();
        if n == 0 || evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
            return Array1::zeros(n);
        }
        let order = self.order.min(n);
        let mut s = Array1::zeros(n);
        if order > 0 {
            let mut gmax = Array1::zeros(order);
            let mut lam = Array1::zeros(order);
            for i in 0..order {
                lam[i] = evals[i];
                gmax[i] = dot(evecs.column(i), g.view());
            }
            // The whole uphill block is maximized: the highest augmented mode.
            let smax = diagonal_rfo_step(&lam, &gmax, order, alpha);
            for i in 0..order {
                axpy(smax[i], evecs.column(i), &mut s);
            }
        }
        let nmin = n - order;
        if nmin > 0 {
            let mut gmin = Array1::zeros(nmin);
            let mut lam = Array1::zeros(nmin);
            for i in 0..nmin {
                lam[i] = evals[order + i];
                gmin[i] = dot(evecs.column(order + i), g.view());
            }
            let smin = diagonal_rfo_step(&lam, &gmin, 0, alpha);
            for i in 0..nmin {
                axpy(smin[i], evecs.column(order + i), &mut s);
            }
        }
        s
    }

    /// Dest companion: per-block RFO then a Euclidean clip.
    pub fn restricted(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        delta: f64,
    ) -> Array1<f64> {
        prfo_restricted(evals, evecs, g, self.order, delta)
    }

    /// Sella Optimizer `TrustRegion` + P-RFO (`rs=tr`).
    pub fn trust_region(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        delta: f64,
    ) -> Array1<f64> {
        prfo_trust_region(evals, evecs, g, self.order, delta)
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
    #[expect(
        clippy::too_many_arguments,
        reason = "The public transport API names the two points and the spectral step inputs."
    )]
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

/// One RFO step on a diagonal Hessian block.
///
/// Banerjee, Adams, Simons, and Shepard, J. Phys. Chem. 89, 52 (1985),
/// doi:10.1021/j100247a015, take the displacement from one eigenvector
/// of the augmented matrix `α [α H, g; gᵀ, 0]`. Baker, J. Comput. Chem.
/// 7, 385 (1986), doi:10.1002/jcc.540070402, runs that problem once on
/// the uphill modes and once on the complement. `mode` is the index in
/// the ascending augmented spectrum: `0` minimizes, and `diag.len()`
/// maximizes. A wide block is an arrowhead. faer reduces it to
/// symmetric tridiagonal form and returns that eigenpair. The step is
/// `α v_i / v_last`, with the same floor on `v_last` as rgmin.
fn diagonal_rfo_step(diag: &Array1<f64>, g: &Array1<f64>, mode: usize, alpha: f64) -> Array1<f64> {
    let n = g.len();
    if n == 0 || diag.len() != n {
        return Array1::zeros(n);
    }
    #[cfg(feature = "faer")]
    if n > AUGMENTED_JACOBI_LIMIT
        && let Some(step) = faer_arrowhead_rfo(diag, g, mode, alpha)
    {
        #[cfg(test)]
        FAER_STEPS.with(|count| count.set(count.get() + 1));
        return step;
    }
    let mut h = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        h[(i, i)] = diag[i];
    }
    rfo_get_s(&h, g, mode, alpha)
}

/// Lowest or highest eigenpair of a diagonal augmented Hessian.
///
/// faer's self-adjoint solver tridiagonalizes the arrowhead and calls
/// its symmetric tridiagonal eigensolver. `None` if that solve fails,
/// in which case the caller keeps cyclic Jacobi.
#[cfg(feature = "faer")]
fn faer_arrowhead_rfo(
    diag: &Array1<f64>,
    g: &Array1<f64>,
    mode: usize,
    alpha: f64,
) -> Option<Array1<f64>> {
    let n = g.len();
    let m = n + 1;
    let alpha = alpha.max(0.0);
    let mut aug = faer::Mat::<f64>::zeros(m, m);
    let scale = alpha * alpha;
    for i in 0..n {
        aug[(i, i)] = diag[i] * scale;
        let border = g[i] * alpha;
        aug[(i, n)] = border;
        aug[(n, i)] = border;
    }
    let par = faer::Par::Seq;
    let params = faer::Spec::<faer::linalg::evd::SelfAdjointEvdParams, f64>::default();
    let mut values = faer::diag::Diag::<f64>::zeros(m);
    let mut vectors = faer::Mat::<f64>::zeros(m, m);
    let mut buf =
        faer::dyn_stack::MemBuffer::new(faer::linalg::evd::self_adjoint_evd_scratch::<f64>(
            m,
            faer::linalg::evd::ComputeEigenvectors::Yes,
            par,
            params,
        ));
    faer::linalg::evd::self_adjoint_evd(
        aug.as_ref(),
        values.as_mut(),
        Some(vectors.as_mut()),
        par,
        faer::dyn_stack::MemStack::new(&mut buf),
        params,
    )
    .ok()?;
    let col = mode.min(n);
    let mut denom = vectors[(n, col)];
    if denom.abs() < AUGMENTED_DENOM_FLOOR {
        denom = if denom == 0.0 {
            AUGMENTED_DENOM_FLOOR
        } else {
            denom.signum() * AUGMENTED_DENOM_FLOOR
        };
    }
    let mut step = Array1::zeros(n);
    for i in 0..n {
        step[i] = vectors[(i, col)] * alpha / denom;
    }
    step.iter().all(|v| v.is_finite()).then_some(step)
}

/// Sella partitioned rational-function step in an orthonormal eigenbasis.
///
/// The first `order` columns form the uphill block. Invalid dimensions
/// return a zero displacement, as in the stepper interface.
pub fn prfo_get_s(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> Array1<f64> {
    PartitionedRationalFunctionOptimization::new(order).get_s(evals, evecs, g, alpha)
}

/// Factory matching Sella `get_stepper` for the P-RFO synonyms only.
pub fn prfo_stepper(name: &str, order: usize) -> Option<PartitionedRationalFunctionOptimization> {
    PartitionedRationalFunctionOptimization::from_name(name, order)
}

/// P-RFO displacement and its derivative with respect to alpha.
///
/// Sella's order-1 stepper is Newton-safe: both partitions follow one
/// extremal root of a diagonal augmented matrix. A partition of 50 or
/// more solves that root from the arrowhead secular equation. A smaller
/// partition uses the same Jacobi augmented eigenproblem as
/// [`rgmin::rfo_get_s`]. The derivative is the differentiated secular
/// equation, `dmu/dalpha = -2 alpha mu / (alpha^2 + s·s)`.
pub(crate) fn prfo_step_and_derivative(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    order: usize,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let n = g.len();
    if n == 0 || evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
        return (Array1::zeros(n), Array1::zeros(n));
    }
    let order = order.min(n);
    let mut s = Array1::zeros(n);
    let mut dsda = Array1::zeros(n);
    if order > 0 {
        let (gmax, lmax) = partition(evals, evecs, g, 0, order);
        let (smax, dsmax) = extremal_diagonal_rfo(&lmax, &gmax, true, alpha);
        accumulate(evecs, 0, &smax, &mut s);
        accumulate(evecs, 0, &dsmax, &mut dsda);
    }
    if order < n {
        let (gmin, lmin) = partition(evals, evecs, g, order, n - order);
        let (smin, dsmin) = extremal_diagonal_rfo(&lmin, &gmin, false, alpha);
        accumulate(evecs, order, &smin, &mut s);
        accumulate(evecs, order, &dsmin, &mut dsda);
    }
    (s, dsda)
}

fn partition(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    g: &Array1<f64>,
    start: usize,
    width: usize,
) -> (Array1<f64>, Array1<f64>) {
    let mut g_block = Array1::zeros(width);
    let mut lam = Array1::zeros(width);
    for i in 0..width {
        lam[i] = evals[start + i];
        g_block[i] = dot(evecs.column(start + i), g.view());
    }
    (g_block, lam)
}

fn accumulate(evecs: &Array2<f64>, start: usize, block: &Array1<f64>, out: &mut Array1<f64>) {
    for (i, value) in block.iter().enumerate() {
        axpy(*value, evecs.column(start + i), out);
    }
}

/// One extremal root of a diagonal RFO partition, plus `ds/dalpha`.
fn extremal_diagonal_rfo(
    evals: &Array1<f64>,
    g: &Array1<f64>,
    upper: bool,
    alpha: f64,
) -> (Array1<f64>, Array1<f64>) {
    let n = g.len();
    if n == 0 || alpha == 0.0 || evals.len() != n {
        return (Array1::zeros(n), Array1::zeros(n));
    }
    let step = if n >= 50 {
        secular_rfo_step(evals, g, upper, alpha)
            .unwrap_or_else(|| jacobi_rfo_step(evals, g, upper, alpha))
    } else {
        jacobi_rfo_step(evals, g, upper, alpha)
    };
    let deriv = rfo_alpha_derivative(alpha, &step, g);
    (step, deriv)
}

fn jacobi_rfo_step(evals: &Array1<f64>, g: &Array1<f64>, upper: bool, alpha: f64) -> Array1<f64> {
    let n = g.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        h[(i, i)] = evals[i];
    }
    let order = if upper { n } else { 0 };
    rfo_get_s(&h, g, order, alpha)
}

/// Arrowhead secular root for a diagonal RFO block of dimension >= 50.
fn secular_rfo_step(
    evals: &Array1<f64>,
    g: &Array1<f64>,
    upper: bool,
    alpha: f64,
) -> Option<Array1<f64>> {
    let n = g.len();
    let alpha2 = alpha * alpha;
    let mut diagonal = vec![0.0; n];
    let mut coupling2 = vec![0.0; n];
    let mut abs_diag = 0.0_f64;
    let mut g_l1 = 0.0_f64;
    for i in 0..n {
        diagonal[i] = alpha2 * evals[i];
        coupling2[i] = alpha2 * g[i] * g[i];
        abs_diag = abs_diag.max(diagonal[i].abs());
        g_l1 += (alpha * g[i]).abs();
    }
    let scale = 1.0_f64.max(abs_diag).max(g_l1);
    let theta = secular_root(&diagonal, &coupling2, upper, scale)?;
    let root_scale = theta.abs().max(abs_diag).max(1.0);
    let mut step = Array1::zeros(n);
    for i in 0..n {
        let gap = theta - diagonal[i];
        if gap.abs() < 1e-6 * root_scale {
            return None;
        }
        step[i] = alpha2 * g[i] / gap;
    }
    Some(step)
}

fn secular_root(diagonal: &[f64], coupling2: &[f64], upper: bool, scale: f64) -> Option<f64> {
    let mut pole: f64 = 0.0;
    if upper {
        for &value in diagonal {
            pole = pole.max(value);
        }
        pole = pole.max(0.0);
    } else {
        pole = diagonal.first().copied().unwrap_or(0.0);
        for &value in diagonal {
            pole = pole.min(value);
        }
        pole = pole.min(0.0);
    }
    let secular = |theta: f64| -> f64 {
        let mut sum = -theta;
        for i in 0..diagonal.len() {
            sum += coupling2[i] / (theta - diagonal[i]);
        }
        sum
    };
    let gap = 1e-12 * scale.max(1.0);
    let inside = if upper { pole + gap } else { pole - gap };
    let f_inside = secular(inside);
    let inside_ok = if upper {
        f_inside > 0.0
    } else {
        f_inside < 0.0
    };
    if !inside_ok || !f_inside.is_finite() {
        return None;
    }
    let mut span = scale.max(1.0);
    let mut outside = if upper { pole + span } else { pole - span };
    let mut f_outside = secular(outside);
    for _ in 0..60 {
        let outside_ok = if upper {
            f_outside < 0.0
        } else {
            f_outside > 0.0
        };
        if outside_ok && f_outside.is_finite() {
            break;
        }
        span *= 2.0;
        outside = if upper { pole + span } else { pole - span };
        f_outside = secular(outside);
        if !f_outside.is_finite() {
            return None;
        }
    }
    let outside_ok = if upper {
        f_outside < 0.0
    } else {
        f_outside > 0.0
    };
    if !outside_ok {
        return None;
    }
    brentq(secular, inside, outside)
}

/// SciPy `brentq` on a sign-changing bracket. `xtol` matches Sella's secular solve.
fn brentq<F: Fn(f64) -> f64>(f: F, mut a: f64, mut b: f64) -> Option<f64> {
    let mut fa = f(a);
    let mut fb = f(b);
    if !fa.is_finite() || !fb.is_finite() || fa * fb > 0.0 {
        return None;
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;
    let xtol = 5e-15;
    for _ in 0..80 {
        if fb * fc > 0.0 {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            let (b0, fb0) = (b, fb);
            a = b0;
            b = c;
            c = b0;
            fa = fb0;
            fb = fc;
            fc = fb0;
        }
        let tol = 4.0 * f64::EPSILON * b.abs() + xtol;
        let m = 0.5 * (c - b);
        if m.abs() <= tol || fb == 0.0 {
            return Some(b);
        }
        if e.abs() >= tol && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, mut q) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let q_ratio = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * m * q_ratio * (q_ratio - r) - (b - a) * (r - 1.0)),
                    (q_ratio - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            let mut p = p;
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        a = b;
        fa = fb;
        if d.abs() > tol {
            b += d;
        } else if m > 0.0 {
            b += tol;
        } else {
            b -= tol;
        }
        fb = f(b);
        if !fb.is_finite() {
            return None;
        }
    }
    Some(b)
}

/// `ds/dalpha` from `(H - mu I) s = -g` with `mu = g·s / alpha^2`.
fn rfo_alpha_derivative(alpha: f64, step: &Array1<f64>, g: &Array1<f64>) -> Array1<f64> {
    let n = step.len().min(g.len());
    let mut deriv = Array1::zeros(n);
    if alpha == 0.0 {
        return deriv;
    }
    let mut projection = 0.0;
    let mut length_sq = 0.0;
    for i in 0..n {
        projection += g[i] * step[i];
        length_sq += step[i] * step[i];
    }
    let alpha2 = alpha * alpha;
    let denom = alpha2 + length_sq;
    if !denom.is_finite() || denom.abs() < 1e-30 || alpha2 < 1e-30 {
        return deriv;
    }
    let mu = projection / alpha2;
    let dmu = -2.0 * alpha * mu / denom;
    if !dmu.is_finite() {
        return deriv;
    }
    for i in 0..n {
        if g[i] != 0.0 {
            deriv[i] = -step[i] * step[i] * dmu / g[i];
        }
    }
    deriv
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use rgmin::ManifoldKind;
    use rgmin::rfo_get_s;
    use rgmin::vecops::{dot, nrm2};

    #[test]
    fn factory_matches_sella_synonyms() {
        assert!(PartitionedRationalFunctionOptimization::match_name("prfo"));
        assert!(PartitionedRationalFunctionOptimization::match_name("p-rfo"));
        assert!(PartitionedRationalFunctionOptimization::match_name(
            "partitioned rational function optimization"
        ));
        assert!(prfo_stepper("prfo", 1).is_some());
        assert!(prfo_stepper("rfo", 1).is_none());
        assert!(prfo_stepper("qn", 0).is_none());
    }

    #[test]
    fn contract_bounds_match_sella_rfo() {
        assert_eq!(
            PartitionedRationalFunctionOptimization::ALPHA0,
            RationalFunctionOptimization::ALPHA0
        );
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHA0, 1.0);
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHAMIN, 0.0);
        assert_eq!(PartitionedRationalFunctionOptimization::ALPHAMAX, 1.0);
        assert_eq!(PartitionedRationalFunctionOptimization::SLOPE, 1.0);
        const {
            assert!(!PartitionedRationalFunctionOptimization::NEWTON_SAFE);
        };
    }

    #[test]
    fn order_zero_matches_rfo_on_the_full_space() {
        let evals = array![2.0, 5.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![2.0, 0.4];
        let h = Array2::from_diag(&evals);
        let s_prfo = PartitionedRationalFunctionOptimization::new(0).get_s(&evals, &evecs, &g, 1.0);
        let s_rfo = rfo_get_s(&h, &g, 0, 1.0);
        let diff = nrm2((&s_prfo - &s_rfo).view());
        assert!(
            diff < 1e-10,
            "order-0 P-RFO != RFO: {s_prfo:?} vs {s_rfo:?}"
        );
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
    fn order_partitions_differ_from_full_space_rfo() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let h = Array2::from_diag(&evals);
        let s_prfo = PartitionedRationalFunctionOptimization::new(1).get_s(&evals, &evecs, &g, 1.0);
        let s_rfo = rfo_get_s(&h, &g, 1, 1.0);
        let diff = nrm2((&s_prfo - &s_rfo).view());
        assert!(diff > 1e-8, "P-RFO collapsed to RFO: {s_prfo:?} {s_rfo:?}");
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
        let evals = Array1::from(vec![-1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]);
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
        let g = array![8.0, 4.0];
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
    fn free_step_uses_the_requested_uphill_partition() {
        let evals = array![-1.0, 4.0];
        let evecs = Array2::<f64>::eye(2);
        let g = array![0.5, 0.4];
        let step = prfo_get_s(&evals, &evecs, &g, 1, 1.0);
        let uphill_shift = 0.5 * (-1.0 + 2.0_f64.sqrt());
        let downhill_shift = 0.5 * (4.0 - 16.64_f64.sqrt());
        let expected = array![
            -g[0] / (evals[0] - uphill_shift),
            -g[1] / (evals[1] - downhill_shift),
        ];
        assert!(nrm2((&step - &expected).view()) < 1e-12);
        assert!(step[0] > 0.0);
        assert!(step[1] < 0.0);
    }

    #[test]
    fn alpha_derivative_matches_a_finite_difference() {
        let evals = array![-1.0, 4.0, 9.0];
        let evecs = Array2::<f64>::eye(3);
        let g = array![0.5, -0.2, 0.7];
        let alpha = 0.4;
        let (step, deriv) = prfo_step_and_derivative(&evals, &evecs, &g, 1, alpha);
        let (ahead, _) = prfo_step_and_derivative(&evals, &evecs, &g, 1, alpha + 1e-6);
        let (behind, _) = prfo_step_and_derivative(&evals, &evecs, &g, 1, alpha - 1e-6);
        let numeric = (&ahead - &behind) / 2e-6;
        let err = nrm2((&deriv - &numeric).view()) / nrm2(numeric.view()).max(1.0);
        assert!(
            err < 1e-5,
            "derivative mismatch {err}: {deriv:?} vs {numeric:?}"
        );
        assert!(nrm2((&step - &prfo_get_s(&evals, &evecs, &g, 1, alpha)).view()) < 1e-10);
    }

    /// `f` and the number of augmented blocks it solved with faer.
    fn count_faer_steps<T>(f: impl FnOnce() -> T) -> (T, usize) {
        FAER_STEPS.with(|count| count.set(0));
        let out = f();
        (out, FAER_STEPS.with(|count| count.get()))
    }

    /// A non-trivial orthonormal frame, so the partition gradient is not
    /// the ambient gradient.
    #[cfg(feature = "faer")]
    fn rotated_frame(n: usize) -> Array2<f64> {
        let mut evecs = Array2::<f64>::eye(n);
        let (c, s) = (0.6, 0.8);
        for row in 0..n {
            let a = evecs[(row, 2)];
            let b = evecs[(row, 5)];
            evecs[(row, 2)] = c * a - s * b;
            evecs[(row, 5)] = s * a + c * b;
        }
        evecs
    }

    /// The 35-wide downhill block takes faer, the 1-wide uphill block Jacobi.
    #[cfg(feature = "faer")]
    #[test]
    fn wide_diagonal_partition_matches_jacobi() {
        let n = 36;
        let mut evals = Array1::zeros(n);
        let mut g = Array1::zeros(n);
        for i in 0..n {
            evals[i] = -1.2 + 0.17 * i as f64;
            g[i] = ((i * 13) % 9) as f64 * 0.04 - 0.15;
        }
        let evecs = rotated_frame(n);
        let alpha = 0.6;
        let (got, faer_steps) = count_faer_steps(|| {
            PartitionedRationalFunctionOptimization::new(1).get_s(&evals, &evecs, &g, alpha)
        });
        assert_eq!(faer_steps, 1, "the downhill block fell back to Jacobi");
        let mut gmax = Array1::zeros(1);
        gmax[0] = dot(evecs.column(0), g.view());
        let mut hmax = Array2::<f64>::zeros((1, 1));
        hmax[(0, 0)] = evals[0];
        let smax = rfo_get_s(&hmax, &gmax, 1, alpha);
        let nmin = n - 1;
        let mut gmin = Array1::zeros(nmin);
        let mut hmin = Array2::<f64>::zeros((nmin, nmin));
        for i in 0..nmin {
            hmin[(i, i)] = evals[i + 1];
            gmin[i] = dot(evecs.column(i + 1), g.view());
        }
        let smin = rfo_get_s(&hmin, &gmin, 0, alpha);
        let mut expect = Array1::zeros(n);
        axpy(smax[0], evecs.column(0), &mut expect);
        for i in 0..nmin {
            axpy(smin[i], evecs.column(i + 1), &mut expect);
        }
        let err = nrm2((&got - &expect).view());
        assert!(err < 1e-10, "faer left Jacobi by {err}");
    }

    /// The whole 40-wide space is the uphill block, so faer returns the
    /// highest augmented eigenpair.
    ///
    /// The step `s_i = α v_i / v_last` of an eigenvalue `α² λ` of
    /// `[α² D, α g; α gᵀ, 0]` is `g_i / (λ - d_i)`, and `λ` solves
    /// `Σ g_i² / (λ - d_i) = α² λ`. Choosing `g_i² = c w_i (λ - d_i)`
    /// with `c = α² λ / Σ w` makes a chosen `λ` above every `d_i` the
    /// root. The secular function decreases on `(max d, ∞)`, so it is the
    /// only root there and the highest eigenvalue.
    #[cfg(feature = "faer")]
    #[test]
    fn wide_uphill_partition_takes_the_highest_eigenpair() {
        let n = 40;
        let alpha = 0.6;
        let lambda = 3.5;
        let mut evals = Array1::zeros(n);
        let mut weights = Array1::zeros(n);
        for i in 0..n {
            evals[i] = -1.6 + 0.07 * i as f64;
            weights[i] = 1.0 + ((i * 7) % 5) as f64;
        }
        let c = alpha * alpha * lambda / weights.sum();
        let mut gblock = Array1::zeros(n);
        for i in 0..n {
            let sign = if i % 3 == 0 { -1.0 } else { 1.0 };
            gblock[i] = sign * (c * weights[i] * (lambda - evals[i])).sqrt();
        }
        let evecs = rotated_frame(n);
        let mut g = Array1::zeros(n);
        for i in 0..n {
            axpy(gblock[i], evecs.column(i), &mut g);
        }
        let (got, faer_steps) = count_faer_steps(|| {
            PartitionedRationalFunctionOptimization::new(n).get_s(&evals, &evecs, &g, alpha)
        });
        assert_eq!(faer_steps, 1, "the uphill block fell back to Jacobi");
        let mut gmax = Array1::zeros(n);
        let mut hmax = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            hmax[(i, i)] = evals[i];
            gmax[i] = dot(evecs.column(i), g.view());
        }
        let smax = rfo_get_s(&hmax, &gmax, n, alpha);
        let mut jacobi = Array1::zeros(n);
        let mut analytic = Array1::zeros(n);
        for i in 0..n {
            axpy(smax[i], evecs.column(i), &mut jacobi);
            let step = gblock[i] / (lambda - evals[i]);
            axpy(step, evecs.column(i), &mut analytic);
        }
        let err = nrm2((&got - &jacobi).view());
        assert!(err < 1e-10, "faer left Jacobi by {err}");
        let err = nrm2((&got - &analytic).view());
        assert!(err < 1e-10, "faer left the analytic eigenpair by {err}");
    }

    /// With faer, the reference's 59-wide downhill block is faer's solve.
    /// Without it, that block is Jacobi.
    #[test]
    fn secular_root_matches_the_augmented_step_on_a_wide_diagonal() {
        let n = 60;
        let mut evals = Array1::zeros(n);
        let mut g = Array1::zeros(n);
        for i in 0..n {
            evals[i] = 0.2 + 0.15 * i as f64;
            g[i] = ((i * 17) % 11) as f64 * 0.05 - 0.2;
        }
        evals[0] = -1.4;
        g[0] = 0.35;
        let evecs = Array2::<f64>::eye(n);
        let (secular, _) = prfo_step_and_derivative(&evals, &evecs, &g, 1, 0.7);
        let (augmented, faer_steps) = count_faer_steps(|| prfo_get_s(&evals, &evecs, &g, 1, 0.7));
        assert_eq!(faer_steps, usize::from(cfg!(feature = "faer")));
        let err = nrm2((&secular - &augmented).view());
        assert!(err < 1e-8, "secular left the augmented step by {err}");
    }

    #[test]
    fn brentq_finds_the_square_root_of_two() {
        let root = brentq(|x| x * x - 2.0, 0.0, 2.0).unwrap();
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-12, "root {root}");
    }
}
