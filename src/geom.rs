//! Geometry a Sella session retracts on.
//!
//! Default is [`ManifoldKind::RigidQuotient`] (Sella Cartesian
//! `fix_translation` + `fix_rotation`). A host that already built a
//! [`Constraints`] chart passes that instead; the session then
//! projects / retracts / transports on `ker(J)`.

use ndarray::{Array1, Array2};
use rgmin::{Manifold, ManifoldKind};

use crate::constraints::Constraints;

/// Either a named rgmin packing or a live equality chart.
#[derive(Clone, Debug)]
pub enum SellaGeom {
    Kind(ManifoldKind),
    Chart(Constraints),
}

impl SellaGeom {
    /// Rigid quotient when the frame has at least three atoms.
    pub fn cartesian(n: usize) -> Self {
        if n >= 9 {
            Self::Kind(ManifoldKind::RigidQuotient)
        } else {
            Self::Kind(ManifoldKind::Euclidean)
        }
    }

    /// Independent coordinates for the Sella `delta0 * n_free` trust.
    pub fn n_free(&self, n: usize) -> usize {
        match self {
            Self::Kind(ManifoldKind::RigidQuotient) if n >= 9 => n.saturating_sub(6).max(1),
            Self::Kind(_) => n.max(1),
            Self::Chart(c) => n.saturating_sub(c.equalities().len()).max(1),
        }
    }

    pub fn chart(&self) -> Option<&Constraints> {
        match self {
            Self::Chart(c) => Some(c),
            Self::Kind(_) => None,
        }
    }

    /// Sella `Ufree`: orthonormal columns spanning the tangent.
    ///
    /// QN / P-RFO run in this chart (`U^T H U`, `U^T g`) and the
    /// increment maps back as `U s_free`.
    pub fn ufree(&self, x: &Array1<f64>) -> Array2<f64> {
        use rgmin::vecops::{axpy, dot, nrm2};
        let n = x.len();
        let mut cols: Vec<Array1<f64>> = Vec::new();
        for i in 0..n {
            let mut e = Array1::zeros(n);
            e[i] = 1.0;
            let mut v = self.project(x, &e);
            for b in &cols {
                let c = dot(v.view(), b.view());
                axpy(-c, b.view(), &mut v);
            }
            let nn = nrm2(v.view());
            if nn > 1e-10 {
                v.mapv_inplace(|a| a / nn);
                cols.push(v);
            }
        }
        let k = cols.len();
        let mut u = Array2::zeros((n, k));
        for (j, col) in cols.into_iter().enumerate() {
            for i in 0..n {
                u[(i, j)] = col[i];
            }
        }
        u
    }
}

/// `y = U^T v`.
pub fn u_t_vec(u: &Array2<f64>, v: &Array1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(u.ncols());
    for j in 0..u.ncols() {
        let mut acc = 0.0;
        for i in 0..u.nrows() {
            acc += u[(i, j)] * v[i];
        }
        y[j] = acc;
    }
    y
}

/// `y = U x`.
pub fn u_vec(u: &Array2<f64>, x: &Array1<f64>) -> Array1<f64> {
    let mut y = Array1::zeros(u.nrows());
    for i in 0..u.nrows() {
        let mut acc = 0.0;
        for j in 0..u.ncols() {
            acc += u[(i, j)] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// `U^T H U`.
pub fn u_t_h_u(u: &Array2<f64>, h: &ndarray::Array2<f64>) -> Array2<f64> {
    let n = u.nrows();
    let k = u.ncols();
    let mut hu = Array2::<f64>::zeros((n, k));
    for i in 0..n {
        for j in 0..k {
            let mut acc = 0.0;
            for t in 0..n {
                acc += h[(i, t)] * u[(t, j)];
            }
            hu[(i, j)] = acc;
        }
    }
    let mut out = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0;
            for t in 0..n {
                acc += u[(t, i)] * hu[(t, j)];
            }
            out[(i, j)] = acc;
        }
    }
    out
}

impl Manifold for SellaGeom {
    fn required_dim(&self, n: usize) -> Result<(), usize> {
        match self {
            Self::Kind(k) => k.required_dim(n),
            Self::Chart(c) => c.required_dim(n),
        }
    }

    fn project(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Kind(k) => k.project(x, v),
            Self::Chart(c) => c.project(x, v),
        }
    }

    fn egrad2rgrad(&self, x: &Array1<f64>, egrad: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Kind(k) => k.egrad2rgrad(x, egrad),
            Self::Chart(c) => c.egrad2rgrad(x, egrad),
        }
    }

    fn retract(&self, x: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Kind(k) => k.retract(x, v),
            Self::Chart(c) => c.retract(x, v),
        }
    }

    fn transport(&self, x_from: &Array1<f64>, x_to: &Array1<f64>, v: &Array1<f64>) -> Array1<f64> {
        match self {
            Self::Kind(k) => k.transport(x_from, x_to, v),
            Self::Chart(c) => c.transport(x_from, x_to, v),
        }
    }
}

/// Sella `optimize.py` trust numbers (`delta0` lives on the session).
#[derive(Clone, Copy, Debug)]
pub struct TrustSchedule {
    pub sigma_inc: f64,
    pub sigma_dec: f64,
    pub rho_inc: f64,
    pub rho_dec: f64,
    pub delta_min: f64,
}

impl TrustSchedule {
    /// `_default_kwargs['minimum']`.
    pub fn minimum() -> Self {
        Self {
            sigma_inc: 1.15,
            sigma_dec: 0.90,
            rho_inc: 1.035,
            rho_dec: 100.0,
            delta_min: 1e-4,
        }
    }

    /// `_default_kwargs['saddle']`.
    pub fn saddle() -> Self {
        Self {
            sigma_inc: 1.15,
            sigma_dec: 0.65,
            rho_inc: 1.035,
            rho_dec: 5.0,
            delta_min: 1e-4,
        }
    }
}

/// Sella `optimize.py` trust update.
pub fn update_trust(delta: f64, rho: f64, smag: f64, sch: &TrustSchedule) -> f64 {
    if rho < 1.0 / sch.rho_dec || rho > sch.rho_dec {
        (smag * sch.sigma_dec).max(sch.delta_min)
    } else if rho > 1.0 / sch.rho_inc && rho < sch.rho_inc {
        (sch.sigma_inc * smag).max(delta)
    } else {
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Constraints;
    use crate::internal::pack_cart;
    use ndarray::Array1;
    use rgmin::vecops::nrm2;

    #[test]
    fn cartesian_picks_the_quotient_at_three_atoms() {
        match SellaGeom::cartesian(9) {
            SellaGeom::Kind(ManifoldKind::RigidQuotient) => {}
            other => panic!("expected RigidQuotient, got {other:?}"),
        }
        match SellaGeom::cartesian(6) {
            SellaGeom::Kind(ManifoldKind::Euclidean) => {}
            other => panic!("expected Euclidean, got {other:?}"),
        }
        assert_eq!(SellaGeom::cartesian(9).n_free(9), 3);
        assert_eq!(SellaGeom::cartesian(6).n_free(6), 6);
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let u = SellaGeom::cartesian(9).ufree(&x);
        assert_eq!(u.ncols(), 3, "RigidQuotient Ufree is 3N-6");
    }

    #[test]
    fn chart_n_free_drops_the_equalities() {
        let x = pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]);
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let geom = SellaGeom::Chart(cons);
        assert_eq!(geom.n_free(9), 6);
        let shift = Array1::from_elem(9, 0.1);
        let v = geom.project(&x, &shift);
        assert!(nrm2(v.view()) < 1e-12, "COM chart must kill a translation");
        let u = geom.ufree(&x);
        assert_eq!(u.ncols(), 6);
        // A translation is orthogonal to every free column.
        let t = Array1::from_elem(9, 1.0);
        let t = geom.project(&x, &t);
        assert!(nrm2(t.view()) < 1e-12);
    }

    #[test]
    fn update_trust_matches_sella_schedule() {
        let sch = TrustSchedule::minimum();
        let shrunk = update_trust(0.2, 0.0, 0.2, &sch);
        assert!((shrunk - 0.2 * sch.sigma_dec).abs() < 1e-14);
        let grown = update_trust(0.2, 1.0, 0.2, &sch);
        assert!((grown - sch.sigma_inc * 0.2).abs() < 1e-14);
    }
}
