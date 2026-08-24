//! Sella order-0 session: QN + TrustRegion over [`CartesianPes`].
//!
//! `sella.optimize.optimize.Sella` with `order=0`, `method=qn`,
//! `eig=false`. One step is eval, project, `qn_restricted`, retract,
//! transport, `PES.kick`, then the `delta0` / `sigma` / `rho` trust
//! schedule. The geometry is [`ManifoldKind::RigidQuotient`] (Sella
//! Cartesian `fix_translation` + `fix_rotation`). The host owns the
//! loop. `run` is a convenience.

use ndarray::Array1;
use rgmin::qn_restricted;
use rgmin::vecops::{axpy, dot, vdot, vnrm2, vnrminf, Vector};
use rgmin::{Manifold, ManifoldKind};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::pes::CartesianPes;

/// Sella `_default_kwargs['minimum']` plus a force gate.
pub struct SellaMinConfig {
    /// Sella `delta0`. The living trust radius starts here.
    pub delta: f64,
    pub sigma_inc: f64,
    pub sigma_dec: f64,
    pub rho_inc: f64,
    pub rho_dec: f64,
    pub delta_min: f64,
    pub force_tol: f64,
}

impl Default for SellaMinConfig {
    fn default() -> Self {
        Self {
            delta: 1e-1,
            sigma_inc: 1.15,
            sigma_dec: 0.90,
            rho_inc: 1.035,
            rho_dec: 100.0,
            delta_min: 1e-4,
            force_tol: 1e-3,
        }
    }
}

pub struct SellaMinReport {
    pub energy: f64,
    pub max_force: f64,
    pub at_minimum: bool,
    pub rho: f64,
    pub delta: f64,
}

pub struct SellaMinSession {
    pes: CartesianPes,
    config: SellaMinConfig,
    manifold: ManifoldKind,
    delta: f64,
    rho: f64,
}

impl SellaMinSession {
    pub fn new(
        config: SellaMinConfig,
        x: Array1<f64>,
        masses: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        // SE(3) quotient needs leftover internals: N >= 3.
        let manifold = if x.len() >= 9 {
            ManifoldKind::RigidQuotient
        } else {
            ManifoldKind::Euclidean
        };
        // Sella optimize.py: TrustRegion delta = delta0 * n_free.
        let n_free = if x.len() >= 9 { x.len() - 6 } else { x.len() };
        let delta = config.delta * n_free as f64;
        Ok(Self {
            pes: CartesianPes::new(x, masses)?,
            config,
            manifold,
            delta,
            rho: 1.0,
        })
    }

    pub fn position(&self) -> ndarray::ArrayView1<'_, f64> {
        self.pes.position()
    }

    /// Living Sella trust radius.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Last accepted `df_actual / df_pred`. `1` when the model is mute.
    pub fn rho(&self) -> f64 {
        self.rho
    }

    pub fn reset(&mut self) {
        self.pes.reset();
        let n = self.pes.position().len();
        let n_free = if n >= 9 { n - 6 } else { n };
        self.delta = self.config.delta * n_free as f64;
        self.rho = 1.0;
    }

    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<SellaMinReport, SaddleError> {
        let x = self.pes.position().to_owned();
        let (energy, g) = surface.eval(x.view())?;
        if !g.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella gradient"));
        }
        let g_r = self.manifold.egrad2rgrad(&x, &g);
        let vg = Vector::from_host(g_r.clone());
        let max_force = atom_fmax(&g_r);
        if max_force <= self.config.force_tol {
            return Ok(SellaMinReport {
                energy,
                max_force,
                at_minimum: true,
                rho: self.rho,
                delta: self.delta,
            });
        }
        let (evals, evecs) = self.pes.hessian().eigh();
        let mut s = qn_restricted(&evals, &evecs, &g_r, 0, self.delta);
        s = self.manifold.project(&x, &s);
        let sn = vnrm2(&Vector::from_host(s.clone()));
        if sn > self.delta && sn > 0.0 {
            s.mapv_inplace(|v| v * (self.delta / sn));
        }
        let vs = Vector::from_host(s.clone());
        let x1 = self.manifold.retract(&x, &s);
        let s1 = self.manifold.transport(&x, &x1, &s);
        if !s1.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("sella transport"));
        }
        let mut d = x1;
        axpy(-1.0, x.view(), &mut d);
        let e0 = energy;
        let hs = self.pes.hessian().hessian().dot(&s);
        let pred = vdot(&vg, &vs) + 0.5 * dot(s.view(), hs.view());
        let (energy, g1) = self.pes.kick(surface, d.view())?;
        let x_new = self.pes.position().to_owned();
        let g1_r = self.manifold.egrad2rgrad(&x_new, &g1);
        let max_force = atom_fmax(&g1_r);
        if pred.abs() >= 1e-14 {
            self.rho = (energy - e0) / pred;
            self.delta = update_trust(self.delta, self.rho, vnrm2(&vs), &self.config);
        } else {
            self.rho = 1.0;
        }
        Ok(SellaMinReport {
            energy,
            max_force,
            at_minimum: max_force <= self.config.force_tol,
            rho: self.rho,
            delta: self.delta,
        })
    }

    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<SellaMinReport, SaddleError> {
        let mut report = self.step(surface)?;
        let mut n = 1;
        while !report.at_minimum && n < max_steps {
            report = self.step(surface)?;
            n += 1;
        }
        Ok(report)
    }
}

/// Sella `PES.converged`: max over atoms of `||F_i||_2`.
fn atom_fmax(g: &Array1<f64>) -> f64 {
    let mut m = 0.0;
    let n = g.len() / 3;
    for i in 0..n {
        let fx = g[3 * i];
        let fy = g[3 * i + 1];
        let fz = g[3 * i + 2];
        let nrm = (fx * fx + fy * fy + fz * fz).sqrt();
        if nrm > m {
            m = nrm;
        }
    }
    if n == 0 {
        vnrminf(&Vector::from_host(g.clone()))
    } else {
        m
    }
}

/// Sella `optimize.py` trust update (`order=0` defaults).
fn update_trust(delta: f64, rho: f64, smag: f64, cfg: &SellaMinConfig) -> f64 {
    if rho < 1.0 / cfg.rho_dec || rho > cfg.rho_dec {
        (smag * cfg.sigma_dec).max(cfg.delta_min)
    } else if rho > 1.0 / cfg.rho_inc && rho < cfg.rho_inc {
        (cfg.sigma_inc * smag).max(delta)
    } else {
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::minmode::PointSurface;
    use ndarray::{Array1, ArrayView1};
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

    fn com(x: ArrayView1<f64>) -> [f64; 3] {
        let nat = x.len() / 3;
        let mut c = [0.0; 3];
        for i in 0..nat {
            c[0] += x[3 * i];
            c[1] += x[3 * i + 1];
            c[2] += x[3 * i + 2];
        }
        let n = nat as f64;
        [c[0] / n, c[1] / n, c[2] / n]
    }

    #[test]
    fn qn_trust_reaches_the_well() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0]),
        )
        .unwrap();
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.25);
        assert!(report.rho.is_finite());
        assert!(report.delta > 0.0);
    }

    #[test]
    fn qn_step_stays_on_the_rigid_quotient() {
        let mut x = Array1::zeros(9);
        x[0] = 0.2;
        x[4] = 1.0;
        x[8] = 1.0;
        let mut sess = SellaMinSession::new(
            SellaMinConfig::default(),
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
        )
        .unwrap();
        let x0 = sess.position().to_owned();
        let c0 = com(x0.view());
        let report = sess.step(&Well).unwrap();
        assert!(report.energy.is_finite());
        let x1 = sess.position().to_owned();
        let c1 = com(x1.view());
        assert!(
            (c0[0] - c1[0]).abs() < 1e-10
                && (c0[1] - c1[1]).abs() < 1e-10
                && (c0[2] - c1[2]).abs() < 1e-10,
            "COM drifted {c0:?} -> {c1:?}"
        );
        let d = &x1 - &x0;
        let d_h = ManifoldKind::RigidQuotient.project(&x0, &d);
        let err = nrm2((&d - &d_h).view());
        assert!(err < 1e-10, "step left the horizontal space: {err}");
        let v = ManifoldKind::RigidQuotient.project(&x0, &Array1::from_elem(9, 0.1));
        let vt = ManifoldKind::RigidQuotient.transport(&x0, &x1, &v);
        let vt_h = ManifoldKind::RigidQuotient.project(&x1, &vt);
        assert!(nrm2((&vt - &vt_h).view()) < 1e-10);
    }

    #[test]
    fn qn_trust_reaches_the_well_with_unequal_masses() {
        let mut x = Array1::zeros(6);
        x[0] = 0.2;
        let mut sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.2,
                force_tol: 0.05,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 16.0]),
        )
        .unwrap();
        let report = sess.run(&Well, 40).unwrap();
        assert!(report.at_minimum, "force {}", report.max_force);
        assert!((sess.position()[0].abs() - 1.0).abs() < 0.25);
    }

    #[test]
    fn trust_radius_starts_at_delta0_times_n_free() {
        let x = Array1::zeros(9);
        let sess = SellaMinSession::new(
            SellaMinConfig {
                delta: 0.1,
                ..SellaMinConfig::default()
            },
            x,
            Array1::from(vec![1.0, 1.0, 1.0]),
        )
        .unwrap();
        assert!((sess.delta() - 0.3).abs() < 1e-14, "delta={}", sess.delta());
    }

    #[test]
    fn atom_fmax_is_the_largest_per_atom_norm() {
        let mut g = Array1::zeros(6);
        g[0] = 0.0008;
        g[1] = 0.0008;
        g[2] = 0.0008;
        let n = atom_fmax(&g);
        assert!(n > 1e-3, "component inf-norm would pass 1e-3; atom |F|={n}");
        assert!((n - (3.0_f64 * 0.0008 * 0.0008).sqrt()).abs() < 1e-14);
    }

    #[test]
    fn update_trust_shrinks_on_a_bad_rho() {
        let cfg = SellaMinConfig::default();
        let shrunk = update_trust(0.2, 0.0, 0.2, &cfg);
        assert!(shrunk < 0.2, "bad rho must shrink: {shrunk}");
        assert!((shrunk - 0.2 * cfg.sigma_dec).abs() < 1e-14);
        let grown = update_trust(0.2, 1.0, 0.2, &cfg);
        assert!(grown > 0.2, "good rho must grow: {grown}");
        assert!((grown - cfg.sigma_inc * 0.2).abs() < 1e-14);
    }
}
