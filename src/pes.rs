//! Cartesian PES wrapper: position, gradient, Cartesian Hessian, kick.
//!
//! Sella `peswrapper.PES` without internals. The Hessian is a dense
//! Cartesian matrix updated in `(s, y)`, matching Sella `_update_H`.
//! [`Self::with_proj`] hangs Sella `proj_trans` /
//! `proj_rot` (`fix_translation` / `fix_rotation`) so
//! [`Self::project`] / [`Self::retract`] / [`Self::transport`] stay
//! on that set. [`Self::new`] is the unconstrained Hessian store
//! [`crate::InternalPes`] wraps. IRC keeps a separate MW model on
//! [`crate::irc::IrcSession`].

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::Manifold;
use rgmin::vecops::{axpy, dot};

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

pub use crate::qn::HessUpdate;

/// Dense Cartesian Hessian. Starts as the identity, like Sella's
/// uninitialized model before the first secant.
///
/// Pair updates are the BFGS and TS-BFGS maps in rgmin. [`Self::install`]
/// replaces the matrix after a multi-secant update the pair maps cannot
/// express.
#[derive(Clone, Debug)]
pub struct CartesianHessian {
    b: Array2<f64>,
}

impl CartesianHessian {
    fn identity(n: usize) -> Self {
        Self { b: Array2::eye(n) }
    }

    /// Stored Cartesian Hessian.
    pub fn hessian(&self) -> &Array2<f64> {
        &self.b
    }

    fn forget(&mut self) {
        let n = self.b.nrows();
        self.b = Array2::eye(n);
    }

    /// Rank-1 seed `B = I + (lambda - 1) u u^T` for a unit mode.
    fn seed_mode(&mut self, u: &Array1<f64>, lam: f64) {
        let n = self.b.nrows().min(u.len());
        let mut nrm = 0.0;
        for i in 0..n {
            nrm += u[i] * u[i];
        }
        nrm = nrm.sqrt();
        if nrm <= 1e-16 {
            return;
        }
        let scale = (lam - 1.0) / (nrm * nrm);
        for i in 0..n {
            for j in 0..n {
                self.b[(i, j)] += scale * u[i] * u[j];
            }
        }
    }

    fn update(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        if dot(y.view(), s.view()) <= 1e-12 {
            return;
        }
        rgmin::bfgs_hessian_update(&mut self.b, s, y);
    }

    fn update_ts(&mut self, s: &Array1<f64>, y: &Array1<f64>) {
        rgmin::ts_bfgs_update(&mut self.b, s, y);
    }

    /// Replace the matrix. The shape must match the living frame.
    pub fn install(&mut self, b: Array2<f64>) -> Result<(), SaddleError> {
        if b.nrows() != self.b.nrows() || b.ncols() != self.b.ncols() {
            return Err(SaddleError::Shape(
                "installed Hessian must match the Cartesian frame".into(),
            ));
        }
        if !b.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("installed Hessian"));
        }
        self.b = b;
        Ok(())
    }
}

/// Cartesian geometry plus a persistent Cartesian Hessian.
pub struct CartesianPes {
    x: Array1<f64>,
    masses: Array1<f64>,
    hess: CartesianHessian,
    update: HessUpdate,
    /// Sella `PES.cons` when `proj_trans` / `proj_rot` hang
    /// translation / rotation equalities. Absent on [`Self::new`].
    cons: Option<Constraints>,
    proj_trans: bool,
    proj_rot: bool,
}

impl CartesianPes {
    /// `x` is 3N Cartesian; `masses` is length N.
    pub fn new(x: Array1<f64>, masses: Array1<f64>) -> Result<Self, SaddleError> {
        if x.is_empty() || !x.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "saddle must be a nonzero 3N Cartesian".into(),
            ));
        }
        if masses.len() * 3 != x.len() {
            return Err(SaddleError::Shape(
                "masses (N) must match the 3N Cartesian".into(),
            ));
        }
        let n = x.len();
        Ok(Self {
            x,
            masses,
            hess: CartesianHessian::identity(n),
            update: HessUpdate::Bfgs,
            cons: None,
            proj_trans: false,
            proj_rot: false,
        })
    }

    /// Sella `PES(proj_trans, proj_rot)`: hang `fix_translation`
    /// and/or `fix_rotation` on the living Cartesian frame.
    ///
    /// Rotation needs two atoms; a smaller frame keeps `proj_rot`
    /// as a no-op. [`Self::new`] stays unconstrained so
    /// [`crate::InternalPes`] does not inherit trans/rot.
    pub fn with_proj(
        x: Array1<f64>,
        masses: Array1<f64>,
        proj_trans: bool,
        proj_rot: bool,
    ) -> Result<Self, SaddleError> {
        let mut pes = Self::new(x, masses)?;
        pes.set_proj(proj_trans, proj_rot)?;
        Ok(pes)
    }

    /// Replace the trans/rot chart. Both false drops it.
    pub fn set_proj(&mut self, proj_trans: bool, proj_rot: bool) -> Result<(), SaddleError> {
        self.proj_trans = proj_trans;
        self.proj_rot = proj_rot;
        if !proj_trans && !proj_rot {
            self.cons = None;
            return Ok(());
        }
        let n = self.masses.len();
        let mut cons = Constraints::new(n)?.keep_rotation_residual();
        if proj_trans {
            cons.fix_com(self.x.view())?;
        }
        if proj_rot && n >= 2 {
            cons.fix_orient(None, self.x.view())?;
        }
        self.cons = Some(cons);
        Ok(())
    }

    pub fn proj_trans(&self) -> bool {
        self.proj_trans
    }

    pub fn proj_rot(&self) -> bool {
        self.proj_rot
    }

    /// Living trans/rot chart, if any.
    pub fn constraints(&self) -> Option<&Constraints> {
        self.cons.as_ref()
    }

    /// Project `v` onto `ker(J)` at the living point. Identity when
    /// neither flag is set.
    pub fn project(&self, v: &Array1<f64>) -> Array1<f64> {
        match &self.cons {
            Some(c) => c.project(&self.x, v),
            None => v.clone(),
        }
    }

    /// Retract along `v` and restore onto the trans/rot set.
    pub fn retract(&self, v: &Array1<f64>) -> Array1<f64> {
        match &self.cons {
            Some(c) => c.retract(&self.x, v),
            None => {
                let mut y = self.x.clone();
                axpy(1.0, v.view(), &mut y);
                y
            }
        }
    }

    /// Transport `v` from `x_from` to `x_to` on the trans/rot set.
    pub fn transport(
        &self,
        x_from: &Array1<f64>,
        x_to: &Array1<f64>,
        v: &Array1<f64>,
    ) -> Array1<f64> {
        match &self.cons {
            Some(c) => c.transport(x_from, x_to, v),
            None => v.clone(),
        }
    }

    pub fn set_update(&mut self, update: HessUpdate) {
        self.update = update;
    }

    pub fn update_kind(&self) -> HessUpdate {
        self.update
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Per-atom masses, length N.
    pub fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    pub fn hessian(&self) -> &CartesianHessian {
        &self.hess
    }

    /// Replace the Cartesian Hessian after a multi-secant update.
    pub fn install_hessian(&mut self, b: Array2<f64>) -> Result<(), SaddleError> {
        self.hess.install(b)
    }

    /// Seed one Cartesian Hessian mode before a quasi-Newton session.
    pub fn seed_mode(&mut self, mode: ArrayView1<f64>, curvature: f64) -> Result<(), SaddleError> {
        if mode.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "seed mode must match the 3N Cartesian frame".into(),
            ));
        }
        if mode.iter().any(|value| !value.is_finite()) || !curvature.is_finite() {
            return Err(SaddleError::NonFinite("Cartesian Hessian mode seed"));
        }
        let norm = mode.dot(&mode).sqrt();
        if norm <= 1e-16 {
            return Err(SaddleError::Shape("seed mode must be nonzero".into()));
        }
        self.hess.seed_mode(&mode.to_owned(), curvature);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.hess.forget();
    }

    /// Add `d` into the living Cartesian frame. No surface eval.
    pub fn displace(&mut self, d: ArrayView1<f64>) -> Result<(), SaddleError> {
        if d.len() != self.x.len() {
            return Err(SaddleError::Shape(
                "displace d must match the 3N frame".into(),
            ));
        }
        axpy(1.0, d, &mut self.x);
        Ok(())
    }

    /// Sella `PES.kick`: displace, eval, store a Cartesian secant pair.
    ///
    /// Sella `_update_H(dx, dg)` is Cartesian. IRC keeps its own model.
    pub fn kick<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let (_, g0) = surface.eval(self.x.view())?;
        self.kick_from(surface, d, g0.view())
    }

    /// [`Self::kick`] when the caller already evaluated `g0` at `x`.
    pub fn kick_from<S: PointSurface>(
        &mut self,
        surface: &S,
        d: ArrayView1<f64>,
        g0: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        if g0.len() != self.x.len() {
            return Err(SaddleError::Shape("kick g0 must match the 3N frame".into()));
        }
        let n = self.x.len().min(d.len());
        for i in 0..n {
            self.x[i] += d[i];
        }
        let (e, g1) = surface.eval(self.x.view())?;
        if !g1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("pes gradient"));
        }
        let y = &g1 - &g0;
        let s = d.slice(ndarray::s![..n]).to_owned();
        match self.update {
            HessUpdate::Bfgs => self.hess.update(&s, &y),
            HessUpdate::TsBfgs => self.hess.update_ts(&s, &y),
        }
        Ok((e, g1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rgmin::vecops::nrm2;

    struct Well;
    impl PointSurface for Well {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok(((t * t - 1.0).powi(2), g))
        }
    }

    fn water() -> Array1<f64> {
        Array1::from(vec![0.0, 0.0, 0.0, 0.96, 0.0, 0.0, -0.24, 0.93, 0.0])
    }

    fn com(x: ArrayView1<f64>) -> [f64; 3] {
        let n = x.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..n {
            c[0] += x[3 * i];
            c[1] += x[3 * i + 1];
            c[2] += x[3 * i + 2];
        }
        let inv = 1.0 / n as f64;
        [c[0] * inv, c[1] * inv, c[2] * inv]
    }

    #[test]
    fn kick_moves_and_records_a_pair() {
        let mut pes = CartesianPes::new(Array1::zeros(6), Array1::from(vec![1.0, 1.0])).unwrap();
        let mut d = Array1::zeros(6);
        d[0] = 0.1;
        let (e, g) = pes.kick(&Well, d.view()).unwrap();
        assert!((pes.position()[0] - 0.1).abs() < 1e-14);
        assert!(e > 0.0);
        assert!(g[0] < 0.0);
        assert!(pes.hessian().hessian().nrows() == 6);
        assert!(!pes.proj_trans());
        assert!(!pes.proj_rot());
        assert!(pes.constraints().is_none());
    }

    #[test]
    fn proj_trans_kills_a_rigid_translation() {
        let pes = CartesianPes::with_proj(water(), Array1::from(vec![1.0, 1.0, 1.0]), true, false)
            .unwrap();
        assert!(pes.proj_trans());
        assert!(!pes.proj_rot());
        let v = Array1::from(vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let p = pes.project(&v);
        assert!(nrm2(p.view()) < 1e-10, "translation survived: {p:?}");
    }

    #[test]
    fn proj_trans_rot_retract_stays_on_the_set() {
        let x = water();
        let pes =
            CartesianPes::with_proj(x.clone(), Array1::from(vec![16.0, 1.0, 1.0]), true, true)
                .unwrap();
        assert!(pes.proj_trans());
        assert!(pes.proj_rot());
        let c0 = com(pes.position());
        let mut step = Array1::zeros(9);
        step[0] = 0.2;
        step[4] = -0.15;
        step[8] = 0.05;
        let v = pes.project(&step);
        let v_h = pes.project(&v);
        assert!(nrm2((&v - &v_h).view()) < 1e-12);
        let y = pes.retract(&v);
        let r = pes.constraints().unwrap().residual_norm(y.view()).unwrap();
        assert!(r < 1e-8, "retract left the set: r={r}");
        let c1 = com(y.view());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-10
                && (c0[1] - c1[1]).abs() < 1e-10
                && (c0[2] - c1[2]).abs() < 1e-10,
            "COM drifted {c0:?} -> {c1:?}"
        );
        let vt = pes.transport(&x, &y, &v);
        let vt_h = pes.constraints().unwrap().project(&y, &vt);
        assert!(nrm2((&vt - &vt_h).view()) < 1e-10);
    }
}
