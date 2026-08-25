//! Sella `QuasiNewton` stepper and `hessian_update` waist.
//!
//! `sella.optimize.stepper.QuasiNewton`. Algebra is
//! [`rgmin::qn_get_s`] / [`rgmin::qn_restricted`]. This module is the
//! rgsaddle factory (`get_stepper("qn")`), not a session and not
//! QuasiNewtonIRC.
//!
//! Sella `hessian_update.py` lives here as [`HessUpdate`]: BFGS is
//! [`rgmin::bfgs_hessian_update`], TS-BFGS is [`rgmin::ts_bfgs_update`]
//! (`|B|` in the secant weight). [`symmetrize_y_cols`] is Sella
//! `symmetrize_Y`. Reductions go through [`rgmin::vecops`]. A host
//! that wants a point on a set calls [`QuasiNewton::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

pub use rgmin::{qn_get_s, qn_restricted};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::Manifold;
use rgmin::vecops::{dot, nrm2};

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

/// Sella `hessian_update` method. ABI 0 = BFGS, 1 = TS-BFGS.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HessUpdate {
    #[default]
    Bfgs,
    TsBfgs,
}

impl HessUpdate {
    /// C / Python ordinal. Unknown values stay out of the enum.
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::Bfgs => 0,
            Self::TsBfgs => 1,
        }
    }

    /// Inverse of [`Self::to_abi`]. Unknown ordinals are `None`.
    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::Bfgs),
            1 => Some(Self::TsBfgs),
            _ => None,
        }
    }
}

/// Sella `symmetrize_Y` on one pair. A single column is already
/// consistent, so `Y` is returned unchanged.
pub fn symmetrize_y(s: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
    let _ = s;
    y.clone()
}

/// Sella `symmetrize_Y(S, Y, symm)`. `symm` is 0 / 1 / 2; default 2.
///
/// `k = 1` (or a shape mismatch) returns `Y`. Reductions are
/// [`dot`] on the columns.
pub fn symmetrize_y_cols(s: ArrayView2<f64>, y: ArrayView2<f64>, symm: i32) -> Array2<f64> {
    if s.ncols() <= 1 || s.nrows() != y.nrows() || s.ncols() != y.ncols() {
        return y.to_owned();
    }
    match symm {
        0 => symmetrize_y_tril(s, y, false),
        1 => symmetrize_y_tril(s, y, true),
        _ => {
            let mut out = y.to_owned();
            let dy = symmetrize_y2(s, y);
            out += &dy;
            out
        }
    }
}

/// Sella `hessian_update.update_H` on one pair.
///
/// `Y` is passed through [`symmetrize_y`] first (`symm=2` is a
/// no-op for one column). BFGS is Nocedal--Wright 6.19. TS-BFGS
/// is Sella `_MS_TS_BFGS`: `|B|` in the secant weight so a
/// negative mode is not forced positive.
pub fn update_hessian(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>, kind: HessUpdate) {
    if nrm2(s.view()) < 1e-8 {
        return;
    }
    let ytilde = symmetrize_y(s, y);
    match kind {
        HessUpdate::Bfgs => rgmin::bfgs_hessian_update(b, s, &ytilde),
        HessUpdate::TsBfgs => rgmin::ts_bfgs_update(b, s, &ytilde),
    }
}

/// Column-wise update after [`symmetrize_y_cols`].
pub fn update_hessian_cols(
    b: &mut Array2<f64>,
    s: ArrayView2<f64>,
    y: ArrayView2<f64>,
    kind: HessUpdate,
    symm: i32,
) {
    let ytilde = symmetrize_y_cols(s, y, symm);
    for j in 0..s.ncols() {
        let sj = s.column(j).to_owned();
        let yj = ytilde.column(j).to_owned();
        update_hessian(b, &sj, &yj, kind);
    }
}

fn gram(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let k = a.ncols();
    let mut g = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            g[(i, j)] = dot(a.column(i), b.column(j));
        }
    }
    g
}

fn tril_strict(m: &Array2<f64>) -> Array2<f64> {
    let k = m.nrows();
    let mut t = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..i {
            t[(i, j)] = m[(i, j)];
        }
    }
    t
}

fn solve_ax_eq_b(a: &Array2<f64>, rhs: &Array2<f64>) -> Option<Array2<f64>> {
    let n = a.nrows();
    let m = rhs.ncols();
    if n == 0 || a.ncols() != n || rhs.nrows() != n {
        return None;
    }
    let mut aug = Array2::<f64>::zeros((n, n + m));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        for j in 0..m {
            aug[(i, n + j)] = rhs[(i, j)];
        }
    }
    for k in 0..n {
        let mut piv = k;
        let mut best = aug[(k, k)].abs();
        for i in (k + 1)..n {
            let v = aug[(i, k)].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best < 1e-16 {
            return None;
        }
        if piv != k {
            for j in 0..(n + m) {
                let tmp = aug[(k, j)];
                aug[(k, j)] = aug[(piv, j)];
                aug[(piv, j)] = tmp;
            }
        }
        let akk = aug[(k, k)];
        for i in (k + 1)..n {
            let f = aug[(i, k)] / akk;
            for j in k..(n + m) {
                aug[(i, j)] -= f * aug[(k, j)];
            }
        }
    }
    let mut x = Array2::<f64>::zeros((n, m));
    for i in (0..n).rev() {
        for j in 0..m {
            let mut acc = aug[(i, n + j)];
            for k in (i + 1)..n {
                acc -= aug[(i, k)] * x[(k, j)];
            }
            x[(i, j)] = acc / aug[(i, i)];
        }
    }
    Some(x)
}

fn matmul_cols(a: ArrayView2<f64>, x: &Array2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let k = a.ncols();
    let m = x.ncols();
    let mut out = Array2::<f64>::zeros((n, m));
    for j in 0..m {
        for t in 0..k {
            let xt = x[(t, j)];
            if xt == 0.0 {
                continue;
            }
            for i in 0..n {
                out[(i, j)] += a[(i, t)] * xt;
            }
        }
    }
    out
}

fn symmetrize_y_tril(s: ArrayView2<f64>, y: ArrayView2<f64>, use_y: bool) -> Array2<f64> {
    let sty = gram(s, y);
    let yts = gram(y, s);
    let mut asy = sty.clone();
    for i in 0..asy.nrows() {
        for j in 0..asy.ncols() {
            asy[(i, j)] -= yts[(i, j)];
        }
    }
    let rhs = tril_strict(&asy).t().to_owned();
    let a = if use_y { gram(s, y) } else { gram(s, s) };
    let Some(x) = solve_ax_eq_b(&a, &rhs) else {
        return y.to_owned();
    };
    let left = if use_y { y } else { s };
    let mut out = y.to_owned();
    out += &matmul_cols(left, &x);
    out
}

/// Sella `symmetrize_Y2`: sequential least-squares correction.
fn symmetrize_y2(s: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
    let n = s.nrows();
    let k = s.ncols();
    let mut dy = Array2::<f64>::zeros((n, k));
    let yts = gram(y, s);
    let sts = gram(s, s);
    let mut dyts = Array2::<f64>::zeros((k, k));
    for i in 1..k {
        let mut a = Array2::<f64>::zeros((i, i));
        let mut rhs = Array2::<f64>::zeros((i, 1));
        for p in 0..i {
            for q in 0..i {
                a[(p, q)] = sts[(p, q)];
            }
            rhs[(p, 0)] = yts[(i, p)] - yts[(p, i)] - dyts[(p, i)];
        }
        let Some(sol) = solve_ax_eq_b(&a, &rhs) else {
            continue;
        };
        for t in 0..i {
            let c = -sol[(t, 0)];
            for r in 0..n {
                dy[(r, i)] += s[(r, t)] * c;
            }
            for q in 0..k {
                dyts[(i, q)] += -sts[(q, t)] * sol[(t, 0)];
            }
        }
    }
    dy
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rgmin::ManifoldKind;
    use rgmin::vecops::{dot, nrm2};

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
    fn hess_update_abi_is_bfgs_then_ts_bfgs() {
        assert_eq!(HessUpdate::Bfgs.to_abi(), 0);
        assert_eq!(HessUpdate::TsBfgs.to_abi(), 1);
        assert_eq!(HessUpdate::try_from_abi(0), Some(HessUpdate::Bfgs));
        assert_eq!(HessUpdate::try_from_abi(1), Some(HessUpdate::TsBfgs));
        assert_eq!(HessUpdate::try_from_abi(2), None);
        assert_eq!(HessUpdate::default(), HessUpdate::Bfgs);
    }

    #[test]
    fn symmetrize_y_one_col_is_identity() {
        let s = array![0.1, -0.2, 0.3];
        let y = array![0.4, 0.0, -0.1];
        let yt = symmetrize_y(&s, &y);
        assert!((yt[0] - y[0]).abs() < 1e-16);
        assert!((yt[1] - y[1]).abs() < 1e-16);
        assert!((yt[2] - y[2]).abs() < 1e-16);
        let sm = Array2::from_shape_vec((3, 1), s.to_vec()).unwrap();
        let ym = Array2::from_shape_vec((3, 1), y.to_vec()).unwrap();
        let yc = symmetrize_y_cols(sm.view(), ym.view(), 2);
        assert!((yc[(0, 0)] - y[0]).abs() < 1e-16);
    }

    #[test]
    fn symmetrize_y2_leaves_a_symmetric_pair() {
        let mut s = Array2::<f64>::zeros((3, 2));
        s[(0, 0)] = 1.0;
        s[(1, 1)] = 1.0;
        let mut h = Array2::<f64>::eye(3);
        h[(0, 0)] = 2.0;
        h[(1, 1)] = 3.0;
        h[(2, 2)] = 4.0;
        let y = h.dot(&s);
        let yt = symmetrize_y_cols(s.view(), y.view(), 2);
        for i in 0..3 {
            for j in 0..2 {
                assert!((yt[(i, j)] - y[(i, j)]).abs() < 1e-12, "yt={yt:?} y={y:?}");
            }
        }
    }

    #[test]
    fn ts_bfgs_keeps_a_negative_mode() {
        let mut b = Array2::<f64>::zeros((2, 2));
        b[(0, 0)] = -1.0;
        b[(1, 1)] = 4.0;
        let s = array![0.0, 0.1];
        let y = array![0.0, 0.4];
        update_hessian(&mut b, &s, &y, HessUpdate::TsBfgs);
        let (evals, _) = crate::eigensolve::exact_eigh(b.view()).unwrap();
        assert!(
            evals.iter().any(|&e| e < 0.0),
            "TS-BFGS must keep the saddle mode: {evals:?}"
        );
    }

    #[test]
    fn tiny_step_is_a_noop() {
        let mut b = Array2::<f64>::eye(2);
        b[(0, 0)] = 2.0;
        let before = b.clone();
        let s = array![1e-12, 0.0];
        let y = array![2e-12, 0.0];
        update_hessian(&mut b, &s, &y, HessUpdate::TsBfgs);
        assert!((b[(0, 0)] - before[(0, 0)]).abs() < 1e-16);
        assert!((b[(1, 1)] - before[(1, 1)]).abs() < 1e-16);
    }

    #[test]
    fn bfgs_secant_on_a_posdef_pair() {
        let mut b = Array2::<f64>::eye(2);
        let s = array![0.2, 0.0];
        let y = array![0.8, 0.0];
        update_hessian(&mut b, &s, &y, HessUpdate::Bfgs);
        let bs = b.dot(&s);
        assert!((bs[0] - y[0]).abs() < 1e-10, "Bs={bs:?} y={y:?}");
        assert!(bs[1].abs() < 1e-10);
    }

    #[test]
    fn ts_bfgs_qn_step_stays_on_the_sphere() {
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let mut b = Array2::<f64>::eye(3);
        b[(0, 0)] = -1.0;
        let s = array![0.0, 0.0, 0.1];
        let y = array![0.0, 0.0, 0.2];
        update_hessian(&mut b, &s, &y, HessUpdate::TsBfgs);
        let (evals, evecs) = crate::eigensolve::exact_eigh(b.view()).unwrap();
        let egrad = array![0.4, 0.1, -0.2];
        let stepper = QuasiNewton::new(&evals, &evecs, &egrad, 1);
        let ynew = stepper.step_on(&man, &x, &egrad, ALPHA0);
        let n = nrm2(ynew.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={ynew:?}");
        assert!(ynew.iter().all(|v| v.is_finite()));

        let g_r = man.egrad2rgrad(&x, &egrad);
        let (step, _) = qn_get_s(&evals, &evecs, &g_r, 1, ALPHA0);
        let v = man.project(&x, &step);
        assert!(
            dot(x.view(), v.view()).abs() < 1e-12,
            "step is not tangent: x·v={}",
            dot(x.view(), v.view())
        );
        let w = stepper.transport_step(&man, &x, &ynew, &egrad, ALPHA0);
        assert!(
            dot(ynew.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(ynew.view(), w.view())
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
