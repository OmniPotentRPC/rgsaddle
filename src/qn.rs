//! Sella `QuasiNewton` stepper waist and `hessian_update`.
//!
//! `sella.optimize.stepper.QuasiNewton`. Algebra is
//! [`rgmin::qn_get_s`] / [`rgmin::qn_restricted`]. This module is the
//! rgsaddle factory (`get_stepper("qn")`), not a session and not
//! QuasiNewtonIRC.
//!
//! Hessian pairs go through [`update_h`]: Sella `BFGS` or `TS-BFGS`
//! after [`symmetrize_y`]. rgmin owns the single-pair BFGS algebra;
//! dest is the TS-aware waist (`|B|` in the secant weight).
//!
//! Ambient reductions go through [`rgmin::vecops`]. A host that
//! wants a point on a set calls [`QuasiNewton::step_on`]:
//! `egrad2rgrad`, `project`, `retract`.

pub use rgmin::{qn_get_s, qn_restricted};

use ndarray::{Array1, Array2, ArrayView2};
use rgmin::Manifold;
use rgmin::vecops::nrm2;

const STEP_FLOOR: f64 = 1e-8;
const PIVOT_FLOOR: f64 = 1e-12;

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

/// How [`update_h`] treats the secant pair. Sella
/// `hessian_update.update_H` methods `BFGS` and `TS-BFGS`.
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

    /// Apply this method to `b` in place.
    pub fn apply(self, b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
        update_h(b, s, y, self);
    }
}

/// Sella `hessian_update.symmetrize_Y`.
///
/// Columns of `s` and `y` are secant pairs. `None` or a single pair
/// returns `y` unchanged. `symm` 0 / 1 / 2 match Sella.
pub fn symmetrize_y(s: ArrayView2<f64>, y: ArrayView2<f64>, symm: Option<i32>) -> Array2<f64> {
    let k = s.ncols();
    if symm.is_none() || k <= 1 || s.nrows() != y.nrows() || y.ncols() != k {
        return y.to_owned();
    }
    match symm.unwrap() {
        0 => {
            let sts = s.t().dot(&s);
            let rhs = tril_skew_t(s.t().dot(&y), y.t().dot(&s));
            match solve_right(&sts, &rhs) {
                Some(x) => y.to_owned() + s.dot(&x),
                None => y.to_owned(),
            }
        }
        1 => {
            let sty = s.t().dot(&y);
            let rhs = tril_skew_t(sty.clone(), y.t().dot(&s));
            match solve_right(&sty, &rhs) {
                Some(x) => y.to_owned() + y.dot(&x),
                None => y.to_owned(),
            }
        }
        2 => y.to_owned() + symmetrize_y2(s, y),
        _ => y.to_owned(),
    }
}

/// Sella `hessian_update.update_H` for one pair. `||s|| < 1e-8` is a no-op.
pub fn update_h(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>, method: HessUpdate) {
    let n = s.len();
    if n == 0 || y.len() != n || b.nrows() != n || b.ncols() != n {
        return;
    }
    if nrm2(s.view()) < STEP_FLOOR {
        return;
    }
    match method {
        HessUpdate::Bfgs => rgmin::bfgs_hessian_update(b, s, y),
        HessUpdate::TsBfgs => rgmin::ts_bfgs_update(b, s, y),
    }
}

/// Multi-column Sella `update_H`: [`symmetrize_y`] then the MS formula.
pub fn update_h_ms(
    b: &mut Array2<f64>,
    s: ArrayView2<f64>,
    y: ArrayView2<f64>,
    method: HessUpdate,
    symm: Option<i32>,
) {
    if s.ncols() == 0 || s.nrows() != b.nrows() || b.nrows() != b.ncols() {
        return;
    }
    if s.ncols() == 1 {
        let sv = s.column(0).to_owned();
        let yv = y.column(0).to_owned();
        update_h(b, &sv, &yv, method);
        return;
    }
    let yt = symmetrize_y(s, y, symm);
    match method {
        HessUpdate::Bfgs => ms_bfgs(b, s, yt.view()),
        HessUpdate::TsBfgs => ms_ts_bfgs(b, s, yt.view()),
    }
}

/// `np.tril(STY - YTS, -1).T`.
fn tril_skew_t(sty: Array2<f64>, yts: Array2<f64>) -> Array2<f64> {
    let k = sty.nrows();
    let mut rhs = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..i {
            rhs[(j, i)] = sty[(i, j)] - yts[(i, j)];
        }
    }
    rhs
}

/// Sella `hessian_update.symmetrize_Y2`.
fn symmetrize_y2(s: ArrayView2<f64>, y: ArrayView2<f64>) -> Array2<f64> {
    let n = s.nrows();
    let k = s.ncols();
    let mut dy = Array2::<f64>::zeros((n, k));
    let yts = y.t().dot(&s);
    let mut dyts = Array2::<f64>::zeros((k, k));
    let sts = s.t().dot(&s);
    for i in 1..k {
        let mut a = Array2::<f64>::zeros((i, i));
        let mut rhs = Array1::<f64>::zeros(i);
        for p in 0..i {
            for q in 0..i {
                a[(p, q)] = sts[(p, q)];
            }
            rhs[p] = yts[(i, p)] - yts[(p, i)] - dyts[(p, i)];
        }
        let Some(sol) = solve_vec(&a, &rhs) else {
            continue;
        };
        for r in 0..n {
            let mut acc = 0.0;
            for p in 0..i {
                acc += s[(r, p)] * sol[p];
            }
            dy[(r, i)] = -acc;
        }
        for q in 0..k {
            let mut acc = 0.0;
            for p in 0..i {
                acc += sts[(q, p)] * sol[p];
            }
            dyts[(i, q)] = -acc;
        }
    }
    dy
}

/// Sella `_MS_BFGS` increment, added into `b`.
fn ms_bfgs(b: &mut Array2<f64>, s: ArrayView2<f64>, y: ArrayView2<f64>) {
    let yts = y.t().dot(&s);
    let yt = y.t().to_owned();
    let Some(term_y) = solve_right(&yts, &yt) else {
        return;
    };
    let bs = b.dot(&s);
    let stbs = s.t().dot(&bs);
    let stb = s.t().dot(b);
    let Some(term_b) = solve_right(&stbs, &stb) else {
        return;
    };
    let delta = y.dot(&term_y) - bs.dot(&term_b);
    *b = &*b + &delta;
    symmetrize_in_place(b);
}

/// Sella `_MS_TS_BFGS` increment, added into `b`.
fn ms_ts_bfgs(b: &mut Array2<f64>, s: ArrayView2<f64>, y: ArrayView2<f64>) {
    let Ok((lams, vecs)) = crate::eigensolve::exact_eigh(b.view()) else {
        return;
    };
    let bs = b.dot(&s);
    let j = &y - &bs;
    let x1 = s.t().dot(&y).dot(&y.t());
    let vts = vecs.t().dot(&s);
    let n = vecs.nrows();
    let k = s.ncols();
    let mut scaled = vts;
    for i in 0..n {
        let a = lams[i].abs();
        for c in 0..k {
            scaled[(i, c)] *= a;
        }
    }
    let absbs = vecs.dot(&scaled);
    let x2 = s.t().dot(&absbs).dot(&absbs.t());
    let xs = &x1 + &x2;
    let xss = xs.dot(&s);
    let Some(ut) = solve_right(&xss, &xs) else {
        return;
    };
    let u = ut.t().to_owned();
    let ujt = u.dot(&j.t());
    let jts = j.t().dot(&s);
    let delta = &ujt + &ujt.t() - u.dot(&jts).dot(&u.t());
    *b = &*b + &delta;
    symmetrize_in_place(b);
}

fn symmetrize_in_place(b: &mut Array2<f64>) {
    let n = b.nrows();
    for i in 0..n {
        for k in i + 1..n {
            let v = 0.5 * (b[(i, k)] + b[(k, i)]);
            b[(i, k)] = v;
            b[(k, i)] = v;
        }
    }
}

fn solve_right(a: &Array2<f64>, rhs: &Array2<f64>) -> Option<Array2<f64>> {
    let k = a.nrows();
    if a.ncols() != k || rhs.nrows() != k {
        return None;
    }
    let m = rhs.ncols();
    let mut out = Array2::<f64>::zeros((k, m));
    for j in 0..m {
        let col = rhs.column(j).to_owned();
        let x = solve_vec(a, &col)?;
        for i in 0..k {
            out[(i, j)] = x[i];
        }
    }
    Some(out)
}

fn solve_vec(a: &Array2<f64>, rhs: &Array1<f64>) -> Option<Array1<f64>> {
    let n = rhs.len();
    if a.nrows() != n || a.ncols() != n {
        return None;
    }
    let mut m = a.clone();
    let mut x = rhs.clone();
    for k in 0..n {
        let mut piv = k;
        let mut best = m[(k, k)].abs();
        for i in (k + 1)..n {
            let v = m[(i, k)].abs();
            if v > best {
                best = v;
                piv = i;
            }
        }
        if best <= PIVOT_FLOOR {
            return None;
        }
        if piv != k {
            for j in 0..n {
                let tmp = m[(k, j)];
                m[(k, j)] = m[(piv, j)];
                m[(piv, j)] = tmp;
            }
            let tmp = x[k];
            x[k] = x[piv];
            x[piv] = tmp;
        }
        let akk = m[(k, k)];
        for i in (k + 1)..n {
            let f = m[(i, k)] / akk;
            m[(i, k)] = 0.0;
            for j in (k + 1)..n {
                m[(i, j)] -= f * m[(k, j)];
            }
            x[i] -= f * x[k];
        }
    }
    for i in (0..n).rev() {
        let mut acc = x[i];
        for j in (i + 1)..n {
            acc -= m[(i, j)] * x[j];
        }
        let di = m[(i, i)];
        if di.abs() <= PIVOT_FLOOR {
            return None;
        }
        x[i] = acc / di;
    }
    Some(x)
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

    #[test]
    fn hess_update_abi_is_bfgs_then_ts_bfgs() {
        assert_eq!(HessUpdate::Bfgs.to_abi(), 0);
        assert_eq!(HessUpdate::TsBfgs.to_abi(), 1);
        assert_eq!(HessUpdate::try_from_abi(0), Some(HessUpdate::Bfgs));
        assert_eq!(HessUpdate::try_from_abi(1), Some(HessUpdate::TsBfgs));
        assert!(HessUpdate::try_from_abi(2).is_none());
    }

    #[test]
    fn tiny_step_leaves_the_hessian() {
        let mut b = Array2::<f64>::eye(2);
        b[(0, 0)] = 3.0;
        let before = b.clone();
        update_h(
            &mut b,
            &array![1e-12, 0.0],
            &array![1.0, 0.0],
            HessUpdate::Bfgs,
        );
        assert_eq!(b, before);
    }

    #[test]
    fn bfgs_on_a_convex_pair_keeps_posdef() {
        let mut b = Array2::<f64>::eye(2);
        let s = array![0.2, 0.0];
        let y = array![0.8, 0.0];
        HessUpdate::Bfgs.apply(&mut b, &s, &y);
        let (evals, _) = crate::eigensolve::exact_eigh(b.view()).unwrap();
        assert!(evals.iter().all(|&e| e > 0.0), "evals={evals:?}");
        let bs = b.dot(&s);
        assert!((bs[0] - y[0]).abs() < 1e-10, "secant B s != y: {bs:?}");
    }

    #[test]
    fn ts_bfgs_keeps_a_negative_mode() {
        let mut b = Array2::<f64>::zeros((2, 2));
        b[(0, 0)] = -1.0;
        b[(1, 1)] = 4.0;
        let s = array![0.0, 0.1];
        let y = array![0.0, 0.4];
        update_h(&mut b, &s, &y, HessUpdate::TsBfgs);
        let (evals, _) = crate::eigensolve::exact_eigh(b.view()).unwrap();
        assert!(
            evals.iter().any(|&e| e < 0.0),
            "TS-BFGS must keep the saddle mode: {evals:?}"
        );
    }

    #[test]
    fn symmetrize_y_is_noop_for_one_pair() {
        let mut s = Array2::<f64>::zeros((2, 1));
        let mut y = Array2::<f64>::zeros((2, 1));
        s[(0, 0)] = 1.0;
        y[(0, 0)] = 2.0;
        y[(1, 0)] = 3.0;
        let out = symmetrize_y(s.view(), y.view(), Some(2));
        assert!((out[(0, 0)] - 2.0).abs() < 1e-15);
        assert!((out[(1, 0)] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn symmetrize_y2_makes_yts_symmetric() {
        let mut s = Array2::<f64>::zeros((2, 2));
        let mut y = Array2::<f64>::zeros((2, 2));
        s[(0, 0)] = 1.0;
        s[(1, 1)] = 1.0;
        y[(0, 0)] = 1.0;
        y[(0, 1)] = 2.0;
        y[(1, 1)] = 1.0;
        let yt = symmetrize_y(s.view(), y.view(), Some(2));
        let yts = yt.t().dot(&s);
        let sty = s.t().dot(&yt);
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (yts[(i, j)] - sty[(i, j)]).abs() < 1e-10,
                    "Y^T S not symmetric at ({i},{j}): {yts:?}"
                );
            }
        }
    }

    #[test]
    fn update_h_ms_one_column_matches_update_h() {
        let mut a = Array2::<f64>::eye(2);
        let mut c = Array2::<f64>::eye(2);
        let s = array![0.25, -0.1];
        let y = array![0.5, -0.2];
        update_h(&mut a, &s, &y, HessUpdate::TsBfgs);
        let mut sm = Array2::<f64>::zeros((2, 1));
        let mut ym = Array2::<f64>::zeros((2, 1));
        sm[(0, 0)] = s[0];
        sm[(1, 0)] = s[1];
        ym[(0, 0)] = y[0];
        ym[(1, 0)] = y[1];
        update_h_ms(&mut c, sm.view(), ym.view(), HessUpdate::TsBfgs, Some(2));
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[(i, j)] - c[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn ts_bfgs_then_qn_step_stays_on_the_sphere() {
        let mut b = Array2::<f64>::eye(3);
        b[(0, 0)] = -2.0;
        let s = array![0.0, 0.15, 0.0];
        let y = array![0.0, 0.3, 0.0];
        HessUpdate::TsBfgs.apply(&mut b, &s, &y);
        let (evals, evecs) = crate::eigensolve::exact_eigh(b.view()).unwrap();
        let man = ManifoldKind::Sphere;
        let x = array![0.0, 1.0, 0.0];
        let egrad = array![0.4, 0.1, -0.2];
        let ypt = retract_qn(&man, &x, &evals, &evecs, &egrad, 1, ALPHA0);
        let n = nrm2(ypt.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={ypt:?}");
        let g = man.egrad2rgrad(&x, &egrad);
        let (step, _) = qn_get_s(&evals, &evecs, &g, 1, ALPHA0);
        let v = man.project(&x, &step);
        assert!(dot(x.view(), v.view()).abs() < 1e-12);
        let w = man.transport(&x, &ypt, &v);
        assert!(dot(ypt.view(), w.view()).abs() < 1e-12);
    }
}
