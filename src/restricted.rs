//! Sella `MaxInternalStep`: per-coordinate clip on an internals chart.
//!
//! `restricted_step.py` `MaxInternalStep`. `cons(s) = max_i |s_i w_i|`
//! with Sella packing weights (`wx` trans/rot, `wb` bonds, `wa`
//! angles, `wd` dihedrals, `wo` other). The clip runs on the
//! internals increment *before* a Euclidean trust radius. Distinct
//! from [`rgmin::ras_clip`] (per-atom Cartesian) and from
//! `TrustRegion` (`||s||`). Algebra is [`rgmin::vecops`].

use ndarray::{Array1, Array2};
use rgmin::Manifold;
use rgmin::qn_get_s;

use crate::SaddleError;
use crate::constraints::{Constraints, Equality, InternalCounts};

/// Named Sella restricted step this crate dests on an internals chart.
///
/// `TrustRegion` is `||s||` (rgmin `qn_restricted`). `MaxInternalStep`
/// is the per-coordinate clip. `RestrictedAtomicStep` is rgmin
/// `ras_clip` and is incompatible with internals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestrictedKind {
    /// Sella `TrustRegion`: `cons(s) = ||s||`.
    TrustRegion,
    /// Sella `MaxInternalStep`: `cons(s) = max |s_i w_i|`.
    MaxInternalStep,
}

/// Per-slot Sella weights (`wx`, `wb`, `wa`, `wd`, `wo`).
///
/// Rotations reuse `translation` (`wx`), matching Sella
/// `_get_weights`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InternalWeights {
    pub translation: f64,
    pub bond: f64,
    pub angle: f64,
    pub dihedral: f64,
    pub other: f64,
}

impl Default for InternalWeights {
    fn default() -> Self {
        Self {
            translation: 1.0,
            bond: 1.0,
            angle: 1.0,
            dihedral: 1.0,
            other: 1.0,
        }
    }
}

/// Sella `MaxInternalStep`.
#[derive(Clone, Debug)]
pub struct MaxInternalStep {
    delta: f64,
    weights: Array1<f64>,
}

impl MaxInternalStep {
    /// Clip radius `delta` and Sella packing `counts`.
    pub fn new(
        delta: f64,
        counts: InternalCounts,
        weights: InternalWeights,
    ) -> Result<Self, SaddleError> {
        if delta < 0.0 {
            return Err(SaddleError::Shape(
                "MaxInternalStep delta must be non-negative".into(),
            ));
        }
        let w = pack_weights(counts, weights);
        if w.is_empty() {
            return Err(SaddleError::Shape(
                "MaxInternalStep needs at least one internal".into(),
            ));
        }
        Ok(Self { delta, weights: w })
    }

    /// Trust radius.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Packed per-coordinate weights, Sella order.
    pub fn weights(&self) -> &Array1<f64> {
        &self.weights
    }

    /// Sella `MaxInternalStep.cons`: `max |s_i w_i|`.
    pub fn cons(&self, s: &Array1<f64>) -> Result<f64, SaddleError> {
        if s.len() != self.weights.len() {
            return Err(SaddleError::Shape(
                "MaxInternalStep step length must match the internals packing".into(),
            ));
        }
        Ok(cons_max(s, &self.weights))
    }

    /// Scale `s` so `cons(s) <= delta`. The Sella per-coordinate clip.
    pub fn clip(&self, s: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
        if s.len() != self.weights.len() {
            return Err(SaddleError::Shape(
                "MaxInternalStep step length must match the internals packing".into(),
            ));
        }
        Ok(mis_clip(s, &self.weights, self.delta))
    }

    /// Unrestricted QN step, then the per-coordinate clip.
    ///
    /// The clip sits in front of a Euclidean trust radius: a host
    /// that still wants `||s|| <= delta_tr` applies `qn_restricted`
    /// to the clipped increment.
    pub fn restrict_qn(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let n = self.weights.len();
        if g.len() != n || evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
            return Err(SaddleError::Shape(
                "MaxInternalStep QN spectrum must match the internals packing".into(),
            ));
        }
        let (s, _) = qn_get_s(evals, evecs, g, order, 0.0);
        self.clip(&s)
    }

    /// Riemannian clip: project, clip in the chart, retract, transport.
    pub fn step_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        s: &Array1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let v = man.project(x, s);
        let c = self.clip(&v)?;
        Ok(man.retract(x, &c))
    }
}

/// Sella packing: `wx` trans, `wb` bonds, `wa` angles, `wd`
/// dihedrals, `wo` other, `wx` rotations.
pub fn pack_weights(counts: InternalCounts, w: InternalWeights) -> Array1<f64> {
    let n = counts.nint();
    let mut out = Array1::zeros(n);
    let mut k = 0;
    for _ in 0..counts.ntrans {
        out[k] = w.translation;
        k += 1;
    }
    for _ in 0..counts.nbonds {
        out[k] = w.bond;
        k += 1;
    }
    for _ in 0..counts.nangles {
        out[k] = w.angle;
        k += 1;
    }
    for _ in 0..counts.ndihedrals {
        out[k] = w.dihedral;
        k += 1;
    }
    for _ in 0..counts.nother {
        out[k] = w.other;
        k += 1;
    }
    for _ in 0..counts.nrotations {
        out[k] = w.translation;
        k += 1;
    }
    out
}

/// Per-equality weights in chart order (not Sella count packing).
pub fn weights_for_equalities(chart: &Constraints) -> Array1<f64> {
    weights_for_equalities_with(chart, InternalWeights::default())
}

/// [`weights_for_equalities`] with host weights.
pub fn weights_for_equalities_with(chart: &Constraints, w: InternalWeights) -> Array1<f64> {
    let eqs = chart.equalities();
    let mut out = Array1::zeros(eqs.len());
    for (i, eq) in eqs.iter().enumerate() {
        out[i] = match eq {
            Equality::Translation { .. } | Equality::Rotation { .. } => w.translation,
            Equality::Bond { .. } => w.bond,
            Equality::Angle { .. } => w.angle,
            Equality::Dihedral { .. } => w.dihedral,
            Equality::Displacement { .. } => w.other,
        };
    }
    out
}

/// `max_i |s_i w_i|`.
pub fn cons_max(s: &Array1<f64>, w: &Array1<f64>) -> f64 {
    let n = s.len().min(w.len());
    let mut m = 0.0;
    for i in 0..n {
        let v = (s[i] * w[i]).abs();
        if v > m {
            m = v;
        }
    }
    m
}

/// Scale `s` so `max |s_i w_i| <= delta`.
pub fn mis_clip(s: &Array1<f64>, w: &Array1<f64>, delta: f64) -> Array1<f64> {
    let val = cons_max(s, w);
    if val <= delta || val <= 1e-16 {
        return s.clone();
    }
    s * (delta / val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
    use ndarray::Array2;
    use rgmin::Manifold;
    use rgmin::vecops::nrm2;

    fn water() -> Array1<f64> {
        pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    }

    #[test]
    fn chart_weights_follow_equality_order() {
        let x = water();
        let mut chart = Constraints::new(3).unwrap();
        chart.fix_com(x.view()).unwrap();
        chart.fix_bond(0, 1, x.view(), None).unwrap();
        let w = weights_for_equalities(&chart);
        assert_eq!(w.len(), 4);
        assert!((w[0] - 1.0).abs() < 1e-14);
        assert!((w[3] - 1.0).abs() < 1e-14);
        let w2 = weights_for_equalities_with(
            &chart,
            InternalWeights {
                translation: 2.0,
                bond: 3.0,
                ..InternalWeights::default()
            },
        );
        assert!((w2[0] - 2.0).abs() < 1e-14);
        assert!((w2[3] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn weights_follow_sella_packing() {
        let c = InternalCounts {
            ntrans: 1,
            nbonds: 2,
            nangles: 1,
            ndihedrals: 0,
            nother: 1,
            nrotations: 2,
        };
        let w = pack_weights(
            c,
            InternalWeights {
                translation: 2.0,
                bond: 3.0,
                angle: 4.0,
                dihedral: 5.0,
                other: 6.0,
            },
        );
        assert_eq!(w.len(), 7);
        assert!((w[0] - 2.0).abs() < 1e-14);
        assert!((w[1] - 3.0).abs() < 1e-14);
        assert!((w[2] - 3.0).abs() < 1e-14);
        assert!((w[3] - 4.0).abs() < 1e-14);
        assert!((w[4] - 6.0).abs() < 1e-14);
        assert!((w[5] - 2.0).abs() < 1e-14);
        assert!((w[6] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn clip_caps_the_largest_weighted_coordinate() {
        let mis = MaxInternalStep::new(
            0.1,
            InternalCounts {
                ntrans: 2,
                nbonds: 1,
                ..InternalCounts::default()
            },
            InternalWeights::default(),
        )
        .unwrap();
        let s = Array1::from(vec![0.4, -0.05, 0.2]);
        let c = mis.clip(&s).unwrap();
        assert!((mis.cons(&c).unwrap() - 0.1).abs() < 1e-14);
        assert!((c[0] - 0.1).abs() < 1e-14);
        assert!((c[2] - 0.05).abs() < 1e-14);
        assert!(c[1].abs() < 0.1);
    }

    #[test]
    fn short_step_is_not_scaled() {
        let mis = MaxInternalStep::new(
            0.5,
            InternalCounts {
                ntrans: 2,
                ..InternalCounts::default()
            },
            InternalWeights::default(),
        )
        .unwrap();
        let s = Array1::from(vec![0.1, -0.2]);
        let c = mis.clip(&s).unwrap();
        assert!((c[0] - 0.1).abs() < 1e-14);
        assert!((c[1] + 0.2).abs() < 1e-14);
    }

    #[test]
    fn restrict_qn_clips_a_long_newton_step() {
        let n = 3;
        let mis = MaxInternalStep::new(
            0.2,
            InternalCounts {
                ntrans: n,
                ..InternalCounts::default()
            },
            InternalWeights::default(),
        )
        .unwrap();
        let evals = Array1::ones(n);
        let evecs = Array2::<f64>::eye(n);
        let g = Array1::from(vec![4.0, 0.5, -1.0]);
        let s = mis.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        assert!(mis.cons(&s).unwrap() <= 0.2 + 1e-14);
        assert!(s[0] < 0.0);
    }

    #[test]
    fn clip_then_retract_stays_on_the_constraint_set() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let mis = MaxInternalStep::new(
            0.05,
            InternalCounts {
                ntrans: 9,
                ..InternalCounts::default()
            },
            InternalWeights::default(),
        )
        .unwrap();
        let mut s = Array1::zeros(9);
        s[0] = 0.4;
        s[4] = -0.3;
        let y = mis.step_on(&cons, &x, &s).unwrap();
        assert!(
            cons.residual_norm(y.view()).unwrap() < 1e-10,
            "clipped retract left the set"
        );
        let v = cons.project(&x, &s);
        let c = mis.clip(&v).unwrap();
        assert!(mis.cons(&c).unwrap() <= 0.05 + 1e-14);
        let t = cons.transport(&x, &y, &c);
        let t_h = cons.project(&y, &t);
        assert!(nrm2((&t - &t_h).view()) < 1e-12);
    }

    #[test]
    fn empty_counts_are_shape() {
        match MaxInternalStep::new(0.1, InternalCounts::default(), InternalWeights::default()) {
            Err(SaddleError::Shape(_)) => {}
            other => panic!("{other:?}"),
        }
    }
}
