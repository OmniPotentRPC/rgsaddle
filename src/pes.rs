//! Cartesian PES wrapper: position, gradient, Cartesian BFGS Hessian, kick.
//!
//! Sella `peswrapper.PES` without internals. The Hessian lives in
//! rgmin [`BfgsModel`] and is updated in Cartesian `(s, y)`, matching
//! Sella `_update_H`. [`Self::with_proj`] hangs Sella `proj_trans` /
//! `proj_rot` (`fix_translation` / `fix_rotation`) so
//! [`Self::project`] / [`Self::retract`] / [`Self::transport`] stay
//! on that set. [`Self::new`] is the unconstrained Hessian store
//! [`crate::InternalPes`] wraps. IRC keeps a separate MW model on
//! [`crate::irc::IrcSession`].

use ndarray::{Array1, ArrayView1};
use rgmin::BfgsModel;
use rgmin::Manifold;
use rgmin::vecops::axpy;

use crate::constraints::Constraints;
use crate::error::SaddleError;
use crate::minmode::PointSurface;

pub use crate::qn::HessUpdate;

/// Cartesian geometry plus a persistent MW Hessian.
pub struct CartesianPes {
    x: Array1<f64>,
    masses: Array1<f64>,
    hess: BfgsModel,
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
        if x.is_empty() || x.len() % 3 != 0 {
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
            hess: BfgsModel::identity(n),
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

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    /// Per-atom masses, length N.
    pub fn masses(&self) -> ArrayView1<'_, f64> {
        self.masses.view()
    }

    pub fn hessian(&self) -> &BfgsModel {
        &self.hess
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

    /// Sella `PES.kick`: displace, eval, store a Cartesian BFGS pair.
    ///
    /// Sella `_update_H(dx, dg)` is Cartesian. IRC keeps its own MW
    /// [`BfgsModel`] on [`crate::irc::IrcSession`].
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
