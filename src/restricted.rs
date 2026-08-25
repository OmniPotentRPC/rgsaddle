//! Sella restricted steps: `TrustRegion`, `RestrictedAtomicStep`,
//! and `MaxInternalStep`.
//!
//! `restricted_step.py` `TrustRegion` is `cons(s) = ||s||`. Distinct
//! from `IRCTrustRegion` (`||(s + d1) * sqrt(m)||`,
//! [`rgmin::IrcTrust`] / [`rgmin::qn_irc_restricted`]). The QN
//! companion is [`rgmin::qn_restricted`].
//!
//! `RestrictedAtomicStep` is `cons(s) = max_i ||s_i||` on 3N
//! Cartesian ([`rgmin::ras_clip`]). Sella refuses it on internals.
//!
//! `MaxInternalStep` is the per-coordinate clip
//! `cons(s) = max_i |s_i w_i|` with Sella packing weights (`wx`
//! trans/rot, `wb` bonds, `wa` angles, `wd` dihedrals, `wo` other).
//! That clip runs on the internals increment *before* a Euclidean
//! trust radius. Distinct from RAS (per-atom Cartesian). Algebra is
//! [`rgmin::vecops`] so `par` applies.

use ndarray::{Array1, Array2};
use rgmin::qn_get_s;
use rgmin::qn_restricted;
use rgmin::ras_clip;
use rgmin::vecops::{axpy, nrm2, nrminf};
use rgmin::Manifold;

use crate::constraints::{Constraints, Equality, InternalCounts};
use crate::SaddleError;

/// Named Sella restricted step this crate dests.
///
/// `TrustRegion` is `||s||` (rgmin `qn_restricted`).
/// `RestrictedAtomicStep` is per-atom Cartesian (`ras_clip`).
/// `MaxInternalStep` is the internals per-coordinate clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RestrictedKind {
    /// Sella `TrustRegion`: `cons(s) = ||s||`.
    TrustRegion,
    /// Sella `MaxInternalStep`: `cons(s) = max |s_i w_i|`.
    MaxInternalStep,
    /// Sella `RestrictedAtomicStep`: `cons(s) = max_i ||s_i||`.
    RestrictedAtomicStep,
}

impl RestrictedKind {
    pub const fn to_abi(self) -> i32 {
        match self {
            Self::TrustRegion => 0,
            Self::MaxInternalStep => 1,
            Self::RestrictedAtomicStep => 2,
        }
    }

    pub const fn try_from_abi(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::TrustRegion),
            1 => Some(Self::MaxInternalStep),
            2 => Some(Self::RestrictedAtomicStep),
            _ => None,
        }
    }

    /// Sella `get_restricted_step` names that this crate dests.
    pub fn from_name(name: &str) -> Option<Self> {
        let n = name.trim().to_ascii_lowercase();
        if TrustRegion::match_name(&n) {
            Some(Self::TrustRegion)
        } else if RestrictedAtomicStep::match_name(&n) {
            Some(Self::RestrictedAtomicStep)
        } else if matches!(n.as_str(), "mis" | "max internal step") {
            Some(Self::MaxInternalStep)
        } else {
            None
        }
    }

    /// Sella `RestrictedAtomicStep` refuses an internals chart.
    pub fn refuse_internal(self) -> Result<(), SaddleError> {
        match self {
            Self::RestrictedAtomicStep => Err(RestrictedAtomicStep::incompatible_internal()),
            Self::TrustRegion | Self::MaxInternalStep => Ok(()),
        }
    }

    /// Cartesian clip: RAS is per-atom, otherwise Euclidean `||s||`.
    pub fn clip_cartesian(&self, s: &Array1<f64>, delta: f64) -> Result<Array1<f64>, SaddleError> {
        match self {
            Self::RestrictedAtomicStep => Ok(RestrictedAtomicStep::new(delta)?.clip(s)),
            Self::TrustRegion | Self::MaxInternalStep => Ok(TrustRegion::new(delta)?.clip(s)),
        }
    }

    /// Ambient QN increment before a Cartesian clip.
    ///
    /// RAS is unrestricted [`qn_get_s`] so [`Self::clip_cartesian`]
    /// can bind `max_i ||s_i||`. `qn_restricted` first would leave
    /// `||s|| <= delta`, and ras_clip would be a no-op.
    pub fn qn_step(
        self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        order: usize,
        delta: f64,
    ) -> Array1<f64> {
        match self {
            Self::RestrictedAtomicStep => qn_get_s(evals, evecs, g, order, 0.0).0,
            Self::TrustRegion | Self::MaxInternalStep => {
                qn_restricted(evals, evecs, g, order, delta)
            }
        }
    }

    /// RAS binds the per-atom clip on an unrestricted ambient step.
    pub const fn wants_unrestricted_ambient(self) -> bool {
        matches!(self, Self::RestrictedAtomicStep)
    }
}

/// Sella `TrustRegion` synonyms.
pub const TRUST_SYNONYMS: &[&str] = &[
    "tr",
    "trust region",
    "trust-region",
    "trust radius",
    "trust-radius",
];

/// Sella `RestrictedAtomicStep` synonyms.
pub const RAS_SYNONYMS: &[&str] = &["ras", "restricted atomic step"];

/// Sella `TrustRegion`: `cons(s) = ||s||`, target `delta`.
///
/// Distinct from [`rgmin::IrcTrust`] (`||(s + d1) * sqrt(m)||`).
/// The QN companion is [`rgmin::qn_restricted`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrustRegion {
    delta: f64,
}

impl TrustRegion {
    /// Euclidean trust radius `delta`.
    pub fn new(delta: f64) -> Result<Self, SaddleError> {
        if delta < 0.0 {
            return Err(SaddleError::Shape(
                "TrustRegion delta must be non-negative".into(),
            ));
        }
        Ok(Self { delta })
    }

    /// Trust radius.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Exact Sella `TrustRegion.match`.
    pub fn match_name(name: &str) -> bool {
        let n = name.trim().to_ascii_lowercase();
        TRUST_SYNONYMS.iter().any(|s| *s == n)
    }

    /// Sella `TrustRegion.cons`: `||s||` through vecops.
    pub fn cons(&self, s: &Array1<f64>) -> f64 {
        nrm2(s.view())
    }

    /// Scale `s` so `||s|| <= delta`.
    pub fn clip(&self, s: &Array1<f64>) -> Array1<f64> {
        let n = nrm2(s.view());
        if n <= self.delta || n <= 1e-16 {
            return s.clone();
        }
        let mut out = Array1::zeros(s.len());
        axpy(self.delta / n, s.view(), &mut out);
        out
    }

    /// Sella `TrustRegion` + `QuasiNewton.get_s`: [`qn_restricted`].
    pub fn restrict_qn(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let n = g.len();
        if evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
            return Err(SaddleError::Shape(
                "TrustRegion QN spectrum must match the gradient".into(),
            ));
        }
        Ok(qn_restricted(evals, evecs, g, order, self.delta))
    }

    /// Riemannian QN + trust: rgrad, `qn_restricted`, project, clip, retract.
    pub fn restrict_qn_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        egrad: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let g = man.egrad2rgrad(x, egrad);
        let s = self.restrict_qn(evals, evecs, &g, order)?;
        Ok(self.step_on(man, x, &s))
    }

    /// Riemannian clip: project, clip, retract.
    pub fn step_on<M: Manifold>(&self, man: &M, x: &Array1<f64>, s: &Array1<f64>) -> Array1<f64> {
        let v = man.project(x, s);
        let c = self.clip(&v);
        man.retract(x, &c)
    }

    /// Vector transport of the clipped projected increment.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        s: &Array1<f64>,
    ) -> Array1<f64> {
        let v = man.project(x, s);
        let c = self.clip(&v);
        man.transport(x, x_to, &c)
    }
}

/// Sella `RestrictedAtomicStep`: `cons(s) = max_i ||s_i||`.
///
/// `s` is 3N Cartesian. Incompatible with internals (`pes.int`).
/// The clip dests [`ras_clip`]. Distinct from [`TrustRegion`]
/// (`||s||`) and from [`MaxInternalStep`] (`max |s_i w_i|`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RestrictedAtomicStep {
    delta: f64,
}

impl RestrictedAtomicStep {
    /// Per-atom Cartesian radius `delta`.
    pub fn new(delta: f64) -> Result<Self, SaddleError> {
        if delta < 0.0 {
            return Err(SaddleError::Shape(
                "RestrictedAtomicStep delta must be non-negative".into(),
            ));
        }
        Ok(Self { delta })
    }

    /// Trust radius.
    pub fn delta(&self) -> f64 {
        self.delta
    }

    /// Sella error when the PES carries internals.
    pub fn incompatible_internal() -> SaddleError {
        SaddleError::Shape(
            "Internal coordinates are not compatible with RestrictedAtomicStep".into(),
        )
    }

    /// Exact Sella `RestrictedAtomicStep.match`.
    pub fn match_name(name: &str) -> bool {
        let n = name.trim().to_ascii_lowercase();
        RAS_SYNONYMS.iter().any(|s| *s == n)
    }

    /// Sella `RestrictedAtomicStep.cons`: largest per-atom `||s_i||`.
    ///
    /// Per-atom 3-norms go through vecops so `par` applies.
    pub fn cons(&self, s: &Array1<f64>) -> f64 {
        ras_cons(s)
    }

    /// Scale `s` so `max_i ||s_i|| <= delta`. Dest of [`ras_clip`].
    pub fn clip(&self, s: &Array1<f64>) -> Array1<f64> {
        ras_clip(s, self.delta)
    }

    /// Unrestricted QN step, then the per-atom clip.
    pub fn restrict_qn(
        &self,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        g: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let n = g.len();
        if evals.len() != n || evecs.nrows() != n || evecs.ncols() != n {
            return Err(SaddleError::Shape(
                "RestrictedAtomicStep QN spectrum must match the gradient".into(),
            ));
        }
        let (s, _) = qn_get_s(evals, evecs, g, order, 0.0);
        Ok(self.clip(&s))
    }

    /// Riemannian QN + RAS: rgrad, QN, project, clip, retract.
    pub fn restrict_qn_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        egrad: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let g = man.egrad2rgrad(x, egrad);
        let s = self.restrict_qn(evals, evecs, &g, order)?;
        Ok(self.step_on(man, x, &s))
    }

    /// Riemannian clip: project, clip, retract.
    pub fn step_on<M: Manifold>(&self, man: &M, x: &Array1<f64>, s: &Array1<f64>) -> Array1<f64> {
        let v = man.project(x, s);
        let c = self.clip(&v);
        man.retract(x, &c)
    }

    /// Vector transport of the clipped projected increment.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        s: &Array1<f64>,
    ) -> Array1<f64> {
        let v = man.project(x, s);
        let c = self.clip(&v);
        man.transport(x, x_to, &c)
    }
}

/// `max_i ||s_i||` over packed 3-vectors. Remainder uses [`nrm2`].
pub fn ras_cons(s: &Array1<f64>) -> f64 {
    let atoms = s.len() / 3;
    if atoms == 0 {
        return nrm2(s.view());
    }
    let mut maxn = 0.0;
    for i in 0..atoms {
        let r = nrm2(s.slice(ndarray::s![3 * i..3 * i + 3]));
        if r > maxn {
            maxn = r;
        }
    }
    maxn
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

    /// Riemannian clip: project, clip in the chart, retract.
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

    /// Vector transport of the clipped projected increment.
    pub fn transport_step<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        x_to: &Array1<f64>,
        s: &Array1<f64>,
    ) -> Result<Array1<f64>, SaddleError> {
        let v = man.project(x, s);
        let c = self.clip(&v)?;
        Ok(man.transport(x, x_to, &c))
    }

    /// Riemannian QN + clip: rgrad, unrestricted QN, project, clip, retract.
    pub fn restrict_qn_on<M: Manifold>(
        &self,
        man: &M,
        x: &Array1<f64>,
        evals: &Array1<f64>,
        evecs: &Array2<f64>,
        egrad: &Array1<f64>,
        order: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let g = man.egrad2rgrad(x, egrad);
        let s = self.restrict_qn(evals, evecs, &g, order)?;
        self.step_on(man, x, &s)
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

/// `max_i |s_i w_i|` through vecops `nrminf` so `par` applies.
pub fn cons_max(s: &Array1<f64>, w: &Array1<f64>) -> f64 {
    if s.len() == w.len() {
        return nrminf((s * w).view());
    }
    let n = s.len().min(w.len());
    if n == 0 {
        return 0.0;
    }
    nrminf((&s.slice(ndarray::s![..n]) * &w.slice(ndarray::s![..n])).view())
}

/// Scale `s` so `max |s_i w_i| <= delta`.
pub fn mis_clip(s: &Array1<f64>, w: &Array1<f64>, delta: f64) -> Array1<f64> {
    let val = cons_max(s, w);
    if val <= delta || val <= 1e-16 {
        return s.clone();
    }
    let mut out = Array1::zeros(s.len());
    axpy(delta / val, s.view(), &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::Constraints;
    use crate::internal::pack_cart;
    use ndarray::{array, Array2};
    use rgmin::vecops::{dot, nrm2};
    use rgmin::{Manifold, ManifoldKind};

    fn water() -> Array1<f64> {
        pack_cart(&[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    }

    #[test]
    fn chart_weights_follow_equality_order() {
        let x = water();
        let mut chart = Constraints::new(3).unwrap();
        chart.fix_com(x.view()).unwrap();
        chart.fix_bond([0, 1], x.view(), None).unwrap();
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
        let t = mis.transport_step(&cons, &x, &y, &s).unwrap();
        let t_h = cons.project(&y, &t);
        assert!(nrm2((&t - &t_h).view()) < 1e-12);
    }

    #[test]
    fn restrict_qn_on_stays_on_the_constraint_set() {
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
        let evals = Array1::ones(9);
        let evecs = Array2::<f64>::eye(9);
        let mut egrad = Array1::zeros(9);
        egrad[0] = 4.0;
        egrad[4] = -0.3;
        let y = mis
            .restrict_qn_on(&cons, &x, &evals, &evecs, &egrad, 0)
            .unwrap();
        assert!(
            cons.residual_norm(y.view()).unwrap() < 1e-10,
            "QN clip retract left the set"
        );
        let g = cons.egrad2rgrad(&x, &egrad);
        let s = mis.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        assert!(mis.cons(&s).unwrap() <= 0.05 + 1e-12);
        let t = mis.transport_step(&cons, &x, &y, &s).unwrap();
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

    #[test]
    fn trust_synonyms_match_sella() {
        assert!(TrustRegion::match_name("tr"));
        assert!(TrustRegion::match_name("Trust-Region"));
        assert!(TrustRegion::match_name("trust radius"));
        assert!(TrustRegion::match_name("trust-radius"));
        assert!(!TrustRegion::match_name("ras"));
        assert!(!TrustRegion::match_name("irc"));
        assert_eq!(
            RestrictedKind::from_name("tr"),
            Some(RestrictedKind::TrustRegion)
        );
        assert_eq!(
            RestrictedKind::from_name("max internal step"),
            Some(RestrictedKind::MaxInternalStep)
        );
        assert_eq!(
            RestrictedKind::from_name("ras"),
            Some(RestrictedKind::RestrictedAtomicStep)
        );
        assert_eq!(
            RestrictedKind::from_name("Restricted Atomic Step"),
            Some(RestrictedKind::RestrictedAtomicStep)
        );
        assert!(RestrictedKind::from_name("irc").is_none());
    }

    #[test]
    fn trust_cons_is_euclidean_norm() {
        let tr = TrustRegion::new(0.5).unwrap();
        let s = Array1::from(vec![3.0, 4.0]);
        assert!((tr.cons(&s) - 5.0).abs() < 1e-14);
    }

    #[test]
    fn trust_cons_is_not_irc_mass_weighted() {
        let tr = TrustRegion::new(1.0).unwrap();
        let s = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let d1 = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((tr.cons(&s) - 1.0).abs() < 1e-14);
        let irc_like = nrm2((&s + &d1).view());
        assert!((irc_like - 2.0).abs() < 1e-14);
        assert!((tr.cons(&s) - irc_like).abs() > 0.5);
    }

    #[test]
    fn trust_clip_caps_the_norm() {
        let tr = TrustRegion::new(0.5).unwrap();
        let s = Array1::from(vec![3.0, 4.0]);
        let c = tr.clip(&s);
        assert!((tr.cons(&c) - 0.5).abs() < 1e-14);
        assert!((c[0] - 0.3).abs() < 1e-14);
        assert!((c[1] - 0.4).abs() < 1e-14);
    }

    #[test]
    fn trust_short_step_is_not_scaled() {
        let tr = TrustRegion::new(0.5).unwrap();
        let s = Array1::from(vec![0.1, -0.2]);
        let c = tr.clip(&s);
        assert!((c[0] - 0.1).abs() < 1e-14);
        assert!((c[1] + 0.2).abs() < 1e-14);
    }

    #[test]
    fn trust_restrict_qn_clips_a_long_newton_step() {
        let tr = TrustRegion::new(0.5).unwrap();
        let evals = Array1::ones(2);
        let evecs = Array2::<f64>::eye(2);
        let g = Array1::from(vec![4.0, 0.0]);
        let s = tr.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        assert!((tr.cons(&s) - 0.5).abs() < 1e-9, "||s||={}", tr.cons(&s));
        assert!(s[0] < 0.0);
    }

    #[test]
    fn trust_step_on_the_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let tr = TrustRegion::new(0.2).unwrap();
        let x = array![0.0, 1.0, 0.0];
        let s = Array1::from(vec![4.0, 0.2, -0.3]);
        let y = tr.step_on(&man, &x, &s);
        let n = nrm2(y.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={y:?}");
        let v = man.project(&x, &s);
        let c = tr.clip(&v);
        assert!(tr.cons(&c) <= 0.2 + 1e-14);
        assert!(dot(x.view(), c.view()).abs() < 1e-12);
        let w = tr.transport_step(&man, &x, &y, &s);
        assert!(
            dot(y.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(y.view(), w.view())
        );
    }

    #[test]
    fn trust_restrict_qn_on_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let tr = TrustRegion::new(0.15).unwrap();
        let x = array![0.0, 1.0, 0.0];
        let evals = Array1::ones(3);
        let evecs = Array2::<f64>::eye(3);
        let egrad = array![1.0, 0.2, -0.3];
        let y = tr
            .restrict_qn_on(&man, &x, &evals, &evecs, &egrad, 0)
            .unwrap();
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
        let g = man.egrad2rgrad(&x, &egrad);
        let s = tr.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        let v = man.project(&x, &s);
        let c = tr.clip(&v);
        assert!(tr.cons(&c) <= 0.15 + 1e-12);
        let w = man.transport(&x, &y, &c);
        assert!(dot(y.view(), w.view()).abs() < 1e-12);
    }

    #[test]
    fn trust_clip_then_retract_stays_on_the_constraint_set() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let tr = TrustRegion::new(0.05).unwrap();
        let mut s = Array1::zeros(9);
        s[0] = 0.4;
        s[4] = -0.3;
        let y = tr.step_on(&cons, &x, &s);
        assert!(
            cons.residual_norm(y.view()).unwrap() < 1e-10,
            "clipped retract left the set"
        );
        let v = cons.project(&x, &s);
        let c = tr.clip(&v);
        assert!(tr.cons(&c) <= 0.05 + 1e-14);
        let t = tr.transport_step(&cons, &x, &y, &s);
        let t_h = cons.project(&y, &t);
        assert!(nrm2((&t - &t_h).view()) < 1e-12);
    }

    #[test]
    fn trust_negative_delta_is_shape() {
        match TrustRegion::new(-0.1) {
            Err(SaddleError::Shape(_)) => {}
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ras_synonyms_match_sella() {
        assert!(RestrictedAtomicStep::match_name("ras"));
        assert!(RestrictedAtomicStep::match_name("Restricted Atomic Step"));
        assert!(!RestrictedAtomicStep::match_name("tr"));
        assert!(!RestrictedAtomicStep::match_name("mis"));
        assert_eq!(RestrictedKind::RestrictedAtomicStep.to_abi(), 2);
        assert_eq!(
            RestrictedKind::try_from_abi(2),
            Some(RestrictedKind::RestrictedAtomicStep)
        );
    }

    #[test]
    fn ras_cons_is_the_largest_per_atom_norm() {
        let ras = RestrictedAtomicStep::new(0.1).unwrap();
        let s = Array1::from(vec![0.3, 0.4, 0.0, 0.3, 0.4, 0.0]);
        assert!((ras.cons(&s) - 0.5).abs() < 1e-14);
        let eucl = nrm2(s.view());
        assert!((eucl - (0.5_f64).sqrt()).abs() < 1e-14);
        assert!((ras.cons(&s) - eucl).abs() > 0.2);
    }

    #[test]
    fn ras_cons_is_not_irc_mass_weighted() {
        let ras = RestrictedAtomicStep::new(1.0).unwrap();
        let s = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let d1 = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert!((ras.cons(&s) - 1.0).abs() < 1e-14);
        let irc_like = nrm2((&s + &d1).view());
        assert!((irc_like - 2.0).abs() < 1e-14);
        assert!((ras.cons(&s) - irc_like).abs() > 0.5);
    }

    #[test]
    fn ras_clip_caps_the_largest_atom() {
        let ras = RestrictedAtomicStep::new(0.1).unwrap();
        let s = Array1::from(vec![0.4, 0.3, 0.0, 0.05, 0.0, 0.0]);
        let c = ras.clip(&s);
        assert!((ras.cons(&c) - 0.1).abs() < 1e-12);
        let r1 = (c[3] * c[3] + c[4] * c[4] + c[5] * c[5]).sqrt();
        assert!(r1 < 0.1);
    }

    #[test]
    fn ras_short_step_is_not_scaled() {
        let ras = RestrictedAtomicStep::new(0.5).unwrap();
        let s = Array1::from(vec![0.1, 0.0, 0.0, -0.2, 0.0, 0.0]);
        let c = ras.clip(&s);
        assert!((c[0] - 0.1).abs() < 1e-14);
        assert!((c[3] + 0.2).abs() < 1e-14);
    }

    #[test]
    fn ras_restrict_qn_clips_a_long_newton_step() {
        let ras = RestrictedAtomicStep::new(0.1).unwrap();
        let evals = Array1::ones(6);
        let evecs = Array2::<f64>::eye(6);
        let g = Array1::from(vec![4.0, 0.0, 0.0, 0.5, 0.0, 0.0]);
        let s = ras.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        assert!(ras.cons(&s) <= 0.1 + 1e-12, "cons={}", ras.cons(&s));
        assert!(s[0] < 0.0);
    }

    #[test]
    fn ras_qn_step_is_not_qn_restricted_then_clip() {
        let evals = Array1::ones(6);
        let evecs = Array2::<f64>::eye(6);
        let g = Array1::from(vec![4.0, 0.0, 0.0, 0.5, 0.0, 0.0]);
        let delta = 0.1;
        let ras = RestrictedKind::RestrictedAtomicStep.qn_step(&evals, &evecs, &g, 0, delta);
        let ras = RestrictedKind::RestrictedAtomicStep
            .clip_cartesian(&ras, delta)
            .unwrap();
        let tr = qn_restricted(&evals, &evecs, &g, 0, delta);
        let tr = RestrictedKind::RestrictedAtomicStep
            .clip_cartesian(&tr, delta)
            .unwrap();
        assert!(ras_cons(&ras) <= delta + 1e-12);
        assert!(ras_cons(&tr) <= delta + 1e-12);
        let d = nrm2((&ras - &tr).view());
        assert!(
            d > 1e-8,
            "RAS qn_step+clip matched qn_restricted+clip: {d} ras={ras:?} tr={tr:?}"
        );
        assert!(RestrictedKind::RestrictedAtomicStep.wants_unrestricted_ambient());
        assert!(!RestrictedKind::TrustRegion.wants_unrestricted_ambient());
    }

    #[test]
    fn ras_step_on_the_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let ras = RestrictedAtomicStep::new(0.2).unwrap();
        let x = array![0.0, 1.0, 0.0];
        let s = Array1::from(vec![4.0, 0.2, -0.3]);
        let y = ras.step_on(&man, &x, &s);
        let n = nrm2(y.view());
        assert!((n - 1.0).abs() < 1e-12, "||y||={n} y={y:?}");
        let v = man.project(&x, &s);
        let c = ras.clip(&v);
        assert!(ras.cons(&c) <= 0.2 + 1e-14);
        assert!(dot(x.view(), c.view()).abs() < 1e-12);
        let w = ras.transport_step(&man, &x, &y, &s);
        assert!(
            dot(y.view(), w.view()).abs() < 1e-12,
            "transported step leaves T_y: y·w={}",
            dot(y.view(), w.view())
        );
    }

    #[test]
    fn ras_restrict_qn_on_sphere_stays_on_the_set() {
        let man = ManifoldKind::Sphere;
        let ras = RestrictedAtomicStep::new(0.15).unwrap();
        let x = array![0.0, 1.0, 0.0];
        let evals = Array1::ones(3);
        let evecs = Array2::<f64>::eye(3);
        let egrad = array![1.0, 0.2, -0.3];
        let y = ras
            .restrict_qn_on(&man, &x, &evals, &evecs, &egrad, 0)
            .unwrap();
        assert!((nrm2(y.view()) - 1.0).abs() < 1e-12);
        let g = man.egrad2rgrad(&x, &egrad);
        let s = ras.restrict_qn(&evals, &evecs, &g, 0).unwrap();
        let v = man.project(&x, &s);
        let c = ras.clip(&v);
        assert!(ras.cons(&c) <= 0.15 + 1e-12);
        let w = man.transport(&x, &y, &c);
        assert!(dot(y.view(), w.view()).abs() < 1e-12);
    }

    #[test]
    fn ras_clip_then_retract_stays_on_the_constraint_set() {
        let x = water();
        let mut cons = Constraints::new(3).unwrap();
        cons.fix_com(x.view()).unwrap();
        let ras = RestrictedAtomicStep::new(0.05).unwrap();
        let mut s = Array1::zeros(9);
        s[0] = 0.4;
        s[4] = -0.3;
        let y = ras.step_on(&cons, &x, &s);
        assert!(
            cons.residual_norm(y.view()).unwrap() < 1e-10,
            "clipped retract left the set"
        );
        let v = cons.project(&x, &s);
        let c = ras.clip(&v);
        assert!(ras.cons(&c) <= 0.05 + 1e-14);
        let t = ras.transport_step(&cons, &x, &y, &s);
        let t_h = cons.project(&y, &t);
        assert!(nrm2((&t - &t_h).view()) < 1e-12);
    }

    #[test]
    fn ras_is_incompatible_with_internals() {
        match RestrictedKind::RestrictedAtomicStep.refuse_internal() {
            Err(SaddleError::Shape(msg)) => {
                assert!(msg.contains("Internal coordinates"));
            }
            other => panic!("{other:?}"),
        }
        RestrictedKind::TrustRegion.refuse_internal().unwrap();
        RestrictedKind::MaxInternalStep.refuse_internal().unwrap();
    }

    #[test]
    fn ras_negative_delta_is_shape() {
        match RestrictedAtomicStep::new(-0.1) {
            Err(SaddleError::Shape(_)) => {}
            other => panic!("{other:?}"),
        }
    }
}
