//! Automatic internals on [`InternalPes`]: build, watch, rebuild.
//!
//! [`InternalPes::auto`] builds the chart with
//! [`InternalTopology::from_geometry`]. A hand-specified chart, or a
//! [`crate::VocnFound`] set loaded through [`InternalPes::from_find`],
//! stays as given and is never rebuilt: that is the override path.
//!
//! An automatic chart is rebuilt at the accepted Cartesian geometry
//! when any of these holds:
//!
//! - the Wilson matrix is singular against the rigid quotient, or its
//!   evaluation fails ([`RebuildReason::SingularJacobian`]);
//! - a bend in the chart comes within the near-linear window, or a bend
//!   a dummy or improper dihedral stands in for leaves it
//!   ([`RebuildReason::NearLinearBend`]);
//! - a centre with three neighbours comes within the near-planar window
//!   with no improper dihedral on it ([`RebuildReason::NearPlanar`]);
//! - a back-transformation does not converge, or the dummy pins cannot
//!   be met ([`RebuildReason::BackTransform`]).
//!
//! The Hessian is carried across by way of the Cartesian curvature of
//! the real atoms, `H_x = J_old^T H_q J_old`, and read back in the new
//! chart, `H_q' = (J_new^+)^T H_x J_new^+`. The Sella guess curvature
//! goes on `ker(J_new^T)`. Cartesian directions the new chart spans and
//! the old one does not get zero curvature. When the old Wilson matrix
//! cannot be evaluated the Hessian is reseeded from the guess.

use ndarray::{Array1, Array2, ArrayView1, s};
use rgmin::BfgsModel;

use super::{InternalPes, guess_curvature};
use crate::constraints::{Constraints, Equality};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::mic::Cell;
use crate::minmode::PointSurface;
use crate::pes::{CartesianPes, HessUpdate};
use crate::topology::{
    AtomMap, InternalTopology, LinearBend, PlanarWatch, TopologyCheck, TopologyConfig, Wilson,
    arm_angle, check_chart, is_bent,
};

/// Why an automatic chart was rebuilt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RebuildReason {
    /// The caller asked for it.
    Requested,
    /// The Wilson matrix lost rank against the rigid quotient, or its
    /// evaluation failed.
    SingularJacobian,
    /// A bend entered, or a replaced bend left, the near-linear window.
    NearLinearBend,
    /// A centre with three neighbours came within the near-planar
    /// window without an improper dihedral to cover its umbrella motion.
    NearPlanar,
    /// A back-transformation ended without converging, or the dummy
    /// atoms could not be put back on their pins.
    BackTransform,
}

impl std::fmt::Display for RebuildReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Requested => "requested",
            Self::SingularJacobian => "singular Jacobian",
            Self::NearLinearBend => "near-linear bend",
            Self::NearPlanar => "near-planar centre",
            Self::BackTransform => "back-transformation failed",
        })
    }
}

/// How the model Hessian crossed a rebuild.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HessianCarry {
    /// Moved through the Cartesian curvature of the real atoms.
    Transformed,
    /// Reseeded from the guess curvature of the new chart.
    Reseeded,
}

/// What one rebuild changed.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct RebuildReport {
    /// What triggered it.
    pub reason: RebuildReason,
    /// Topology epoch after the rebuild; the first build is epoch 0.
    pub epoch: u64,
    /// Internals before.
    pub n_int_before: usize,
    /// Internals after.
    pub n_int_after: usize,
    /// Dummy atoms before.
    pub n_dummy_before: usize,
    /// Dummy atoms after.
    pub n_dummy_after: usize,
    /// How the Hessian was carried.
    pub hessian: HessianCarry,
}

/// State of an automatic chart.
pub(super) struct AutoState {
    elements: Vec<u8>,
    lattice: Option<[[f64; 3]; 3]>,
    config: TopologyConfig,
    pins: Constraints,
    map: AtomMap,
    bends: Vec<LinearBend>,
    watch: Vec<PlanarWatch>,
    epoch: u64,
    pending: Option<RebuildReason>,
}

impl AutoState {
    fn from_topology(topology: &InternalTopology, elements: &[u8], epoch: u64) -> Self {
        Self {
            elements: elements.to_vec(),
            lattice: topology.constraints().lattice(),
            config: topology.config().clone(),
            pins: topology.pins().clone(),
            map: topology.atom_map().clone(),
            bends: topology.linear_bends().to_vec(),
            watch: topology.planar_watch().to_vec(),
            epoch,
            pending: None,
        }
    }

    pub(super) fn sv_tol(&self) -> f64 {
        self.config.sv_tol
    }

    pub(super) fn follow_lattice(&mut self, lattice: [[f64; 3]; 3]) {
        if self.pins.is_periodic() {
            self.pins.set_lattice(lattice);
        }
        if self.lattice.is_some() {
            self.lattice = Some(lattice);
        }
    }

    pub(super) fn retarget_lattice(&mut self, lattice: [[f64; 3]; 3]) -> Result<(), SaddleError> {
        if self.pins.is_periodic() || self.pins.lattice().is_some() {
            self.pins.retarget_lattice(lattice)?;
        }
        if self.lattice.is_some() {
            self.lattice = Some(lattice);
        }
        Ok(())
    }
}

/// Most times a failed back-transformation is retried with the internal
/// step halved before the chart is declared stale.
const SHRINKS: usize = 5;

fn wrap_pi(d: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    d - tau * (d / tau).round()
}

fn rms(v: &Array1<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.dot(v) / v.len() as f64).sqrt()
}

fn all_finite(v: &Array1<f64>) -> bool {
    v.iter().all(|a| a.is_finite())
}

impl InternalPes {
    /// Automatic internals: bonds by covalent radius, angles,
    /// dihedrals, and dummy atoms on near-linear bends, from the
    /// geometry `x` of atoms `elements` (atomic numbers). `cell` makes
    /// the search periodic with the cell held fixed.
    ///
    /// The chart is rebuilt as the geometry moves; see the module
    /// notes. Use [`InternalPes::new`] or [`InternalPes::from_find`] to
    /// keep a chart fixed.
    pub fn auto(
        x: Array1<f64>,
        masses: Array1<f64>,
        elements: &[u8],
        cell: Option<&Cell>,
        config: &TopologyConfig,
    ) -> Result<Self, SaddleError> {
        let topology = InternalTopology::from_geometry_with(elements, x.view(), cell, config)?;
        Self::from_topology(x, masses, &topology, elements)
    }

    /// Wrap a topology built at `x` for atoms `elements`.
    pub fn from_topology(
        x: Array1<f64>,
        masses: Array1<f64>,
        topology: &InternalTopology,
        elements: &[u8],
    ) -> Result<Self, SaddleError> {
        if elements.len() != topology.n_real() || 3 * elements.len() != x.len() {
            return Err(SaddleError::Shape(
                "topology, elements and positions must describe the same atoms".into(),
            ));
        }
        let chart = topology.constraints().clone();
        if chart.equalities().is_empty() {
            return Err(SaddleError::Shape("internals chart is empty".into()));
        }
        let nint = chart.equalities().len();
        let mut pes = Self {
            cart: CartesianPes::new(x, masses)?,
            chart,
            hess: BfgsModel::identity(nint),
            update: HessUpdate::Bfgs,
            dummies: topology.dummy_positions().to_owned(),
            auto: Some(Box::new(AutoState::from_topology(topology, elements, 0))),
        };
        pes.guess_hessian();
        Ok(pes)
    }

    /// Whether the chart was built automatically and may be rebuilt.
    pub fn is_auto(&self) -> bool {
        self.auto.is_some()
    }

    /// How many times the chart has been rebuilt. Zero for a chart
    /// that has not been, and for a hand-specified one.
    pub fn topology_epoch(&self) -> u64 {
        self.auto.as_ref().map_or(0, |a| a.epoch)
    }

    /// A rebuild flagged by a step, waiting for
    /// [`Self::rebuild_if_stale`].
    pub fn pending_rebuild(&self) -> Option<RebuildReason> {
        self.auto.as_ref().and_then(|a| a.pending)
    }

    /// Real and dummy slots of the frame; `None` for a hand-specified
    /// chart.
    pub fn atom_map(&self) -> Option<&AtomMap> {
        self.auto.as_ref().map(|a| &a.map)
    }

    /// The pins that hold the dummies; `None` for a hand-specified
    /// chart.
    pub fn pins(&self) -> Option<&Constraints> {
        self.auto.as_ref().map(|a| &a.pins)
    }

    /// The topology rules; `None` for a hand-specified chart.
    pub fn topology_config(&self) -> Option<&TopologyConfig> {
        self.auto.as_ref().map(|a| &a.config)
    }

    /// Rank and completeness of the chart at the current geometry.
    pub fn topology_check(&self) -> Result<TopologyCheck, SaddleError> {
        let n_real = self.cart.position().len() / 3;
        let (pins, tol) = match &self.auto {
            Some(a) => (Some(&a.pins), a.config.sv_tol),
            None => (None, crate::topology::PAPER_SV_TOL),
        };
        let frame = self.frame();
        check_chart(&self.chart, pins, frame.view(), n_real, tol)
    }

    /// Whether the chart is out of date at the current geometry.
    /// `None` for a hand-specified chart.
    pub fn stale_reason(&self) -> Result<Option<RebuildReason>, SaddleError> {
        let Some(auto) = &self.auto else {
            return Ok(None);
        };
        if let Some(r) = auto.pending {
            return Ok(Some(r));
        }
        let frame = self.frame();
        let tol = auto.config.linear_tol;
        let Ok(values) = self.chart.values(frame.view()) else {
            return Ok(Some(RebuildReason::SingularJacobian));
        };
        if !matches!(self.topology_check(), Ok(c) if c.complete) {
            return Ok(Some(RebuildReason::SingularJacobian));
        }
        for (eq, v) in self.chart.equalities().iter().zip(values.iter()) {
            if let Equality::Angle { .. } = eq
                && !is_bent(*v, tol)
            {
                return Ok(Some(RebuildReason::NearLinearBend));
            }
        }
        for bend in &auto.bends {
            let theta = arm_angle(
                frame.view(),
                auto.lattice,
                bend.centre,
                bend.arms[0],
                bend.arms[1],
            );
            if is_bent(theta, tol) {
                return Ok(Some(RebuildReason::NearLinearBend));
            }
        }
        let full_turn = std::f64::consts::TAU;
        for w in &auto.watch {
            let [p, q, r] = w.arms;
            let angle = |u, v| arm_angle(frame.view(), auto.lattice, w.centre, u, v);
            let sum = angle(p, q) + angle(q, r) + angle(p, r);
            if full_turn - sum < auto.config.planar_tol {
                return Ok(Some(RebuildReason::NearPlanar));
            }
        }
        Ok(None)
    }

    /// Rebuild the chart if [`Self::stale_reason`] says so. Call it
    /// between steps, before the gradient is read in the old chart.
    pub fn rebuild_if_stale(&mut self) -> Result<Option<RebuildReport>, SaddleError> {
        match self.stale_reason()? {
            Some(reason) => self.rebuild_topology(reason).map(Some),
            None => Ok(None),
        }
    }

    /// Rebuild the chart at the current geometry and carry the
    /// Hessian across. Errors for a hand-specified chart.
    pub fn rebuild_topology(
        &mut self,
        reason: RebuildReason,
    ) -> Result<RebuildReport, SaddleError> {
        let Some(auto) = &self.auto else {
            return Err(SaddleError::Invalid(
                "a hand-specified chart is never rebuilt".into(),
            ));
        };
        let elements = auto.elements.clone();
        let lattice = auto.lattice;
        let config = auto.config.clone();
        let epoch = auto.epoch + 1;
        let x = self.cart.position().to_owned();
        let topology = InternalTopology::from_lattice(&elements, x.view(), lattice, &config)?;
        let new_chart = topology.constraints().clone();
        let new_frame = topology.frame(x.view())?;
        let j_new = effective_jacobian(
            &new_chart,
            topology.pins(),
            &new_frame,
            x.len(),
            config.sv_tol,
        )?;

        let old_frame = self.frame();
        let (h_new, carry) = match self.effective_jacobian(&old_frame) {
            Ok(j_old) => {
                let h_old = self.hess.hessian();
                let h = transport_hessian(h_old, &j_old, &j_new, &new_chart, config.sv_tol)?;
                (h, HessianCarry::Transformed)
            }
            Err(_) => (guess_matrix(&new_chart), HessianCarry::Reseeded),
        };

        let report = RebuildReport {
            reason,
            epoch,
            n_int_before: self.chart.equalities().len(),
            n_int_after: new_chart.equalities().len(),
            n_dummy_before: self.dummies.len() / 3,
            n_dummy_after: topology.n_dummy(),
            hessian: carry,
        };
        self.hess = hessian_model(&h_new)?;
        self.chart = new_chart;
        self.dummies = topology.dummy_positions().to_owned();
        self.auto = Some(Box::new(AutoState::from_topology(
            &topology, &elements, epoch,
        )));
        Ok(report)
    }

    fn flag(&mut self, reason: RebuildReason) {
        if let Some(a) = &mut self.auto
            && a.pending.is_none()
        {
            a.pending = Some(reason);
        }
    }

    fn dihedral_mask(&self) -> Vec<bool> {
        self.chart
            .equalities()
            .iter()
            .map(|e| matches!(e, Equality::Dihedral { .. }))
            .collect()
    }

    /// The Jacobian of the internals with respect to the real atoms
    /// alone, at the frame `frame`: dummies follow the real atoms along
    /// their pins, `dd = -J_d^+ J_r dx`.
    pub(super) fn effective_jacobian(
        &self,
        frame: &Array1<f64>,
    ) -> Result<Array2<f64>, SaddleError> {
        let Some(auto) = &self.auto else {
            return Err(SaddleError::Invalid(
                "effective Jacobian needs an automatic chart".into(),
            ));
        };
        effective_jacobian(
            &self.chart,
            &auto.pins,
            frame,
            self.cart.position().len(),
            auto.config.sv_tol,
        )
    }

    /// Put the dummies on their pins for the real positions `x`,
    /// starting from `dummies`: Gauss-Newton on the dummy slots alone.
    /// The flag says whether the pins are met.
    fn settle(
        &self,
        x: ArrayView1<f64>,
        dummies: &Array1<f64>,
    ) -> Result<(Array1<f64>, bool), SaddleError> {
        let Some(auto) = &self.auto else {
            return Ok((dummies.clone(), true));
        };
        if auto.pins.equalities().is_empty() {
            return Ok((dummies.clone(), true));
        }
        let cfg = &auto.config;
        let n = x.len();
        let mut frame = crate::topology::join_frame(x, dummies.view());
        for _ in 0..cfg.back_transform_iterations {
            let r = auto.pins.residual(frame.view())?;
            if r.iter().all(|v| v.abs() < cfg.back_transform_tol) {
                return Ok((frame.slice(s![n..]).to_owned(), true));
            }
            let j = auto.pins.jacobian(frame.view())?;
            let jd = j.slice(s![.., n..]).to_owned();
            if !jd.iter().all(|v| v.is_finite()) {
                break;
            }
            let step = Wilson::new(&jd, cfg.sv_tol)?.solve(&r.mapv(|v| -v));
            if !all_finite(&step) {
                break;
            }
            for k in 0..step.len() {
                frame[n + k] += step[k];
            }
        }
        let r = auto.pins.residual(frame.view())?;
        let met = r.iter().all(|v| v.abs() < cfg.back_transform_tol);
        Ok((frame.slice(s![n..]).to_owned(), met))
    }

    /// Iterative back-transformation of the target internals `target`
    /// from the real positions `x0`: Newton steps on the real atoms with
    /// the dummies slaved to them. Returns the positions, the settled
    /// dummies, and whether both the solve and the pins converged. A
    /// failed run returns the best iterate it saw.
    fn back_transform(
        &self,
        x0: &Array1<f64>,
        target: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>, bool), SaddleError> {
        let Some(auto) = &self.auto else {
            return Err(SaddleError::Invalid(
                "back-transform needs an automatic chart".into(),
            ));
        };
        let cfg = &auto.config;
        let dih = self.dihedral_mask();
        let mut x = x0.clone();
        let mut dummies = self.dummies.clone();
        let mut best = (x0.clone(), dummies.clone());
        let mut best_res = f64::INFINITY;
        for _ in 0..cfg.back_transform_iterations {
            let (d, met) = self.settle(x.view(), &dummies)?;
            dummies = d;
            let frame = crate::topology::join_frame(x.view(), dummies.view());
            let Ok(q) = self.chart.values(frame.view()) else {
                return Ok((best.0, best.1, false));
            };
            let mut r = target - &q;
            for (k, wrap) in dih.iter().enumerate() {
                if *wrap {
                    r[k] = wrap_pi(r[k]);
                }
            }
            let res = rms(&r);
            if !res.is_finite() {
                return Ok((best.0, best.1, false));
            }
            if res < best_res {
                best_res = res;
                best = (x.clone(), dummies.clone());
            }
            let Ok(j) = self.effective_jacobian(&frame) else {
                return Ok((best.0, best.1, false));
            };
            let dx = Wilson::new(&j, cfg.sv_tol)?.solve(&r);
            if !all_finite(&dx) {
                return Ok((best.0, best.1, false));
            }
            x += &dx;
            if rms(&dx) < cfg.back_transform_tol {
                let (d, met2) = self.settle(x.view(), &dummies)?;
                return Ok((x, d, met && met2));
            }
        }
        Ok((best.0, best.1, false))
    }

    /// Back-transform `dq` and install the settled dummy positions.
    /// Returns the move of the real atoms; an unconverged solve flags
    /// the chart for a rebuild.
    pub(super) fn move_auto(&mut self, dq: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let n_int = self.chart.equalities().len();
        if dq.len() != n_int {
            return Err(SaddleError::Shape(
                "internals increment must match the chart".into(),
            ));
        }
        let frame0 = self.frame();
        let q0 = self.chart.values(frame0.view())?;
        let x0 = self.cart.position().to_owned();
        let mut scale = 1.0;
        let mut first: Option<(Array1<f64>, Array1<f64>)> = None;
        let mut settled = None;
        for _ in 0..SHRINKS {
            let target = &q0 + &(&dq * scale);
            let (x1, dummies, converged) = self.back_transform(&x0, &target)?;
            if converged {
                settled = Some((x1, dummies));
                break;
            }
            first.get_or_insert((x1, dummies));
            scale *= 0.5;
        }
        let (x1, dummies) = match settled {
            Some(done) => done,
            None => {
                self.flag(RebuildReason::BackTransform);
                first.ok_or_else(|| SaddleError::Solver("back-transform made no attempt".into()))?
            }
        };
        self.dummies = dummies;
        Ok(&x1 - &x0)
    }

    /// [`Self::kick`] on an automatic chart: iterative back-transform
    /// with the dummies slaved to the real atoms, pair update from the
    /// achieved move.
    pub(super) fn kick_auto<S: PointSurface>(
        &mut self,
        surface: &S,
        dq: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), SaddleError> {
        let sv_tol = self
            .auto
            .as_ref()
            .map_or(crate::topology::PAPER_SV_TOL, |a| a.config.sv_tol);
        let n_int = self.chart.equalities().len();
        if dq.len() != n_int {
            return Err(SaddleError::Shape(
                "internals increment must match the chart".into(),
            ));
        }
        let (_, g0) = surface.eval(self.cart.position())?;
        let frame0 = self.frame();
        let q0 = self.chart.values(frame0.view())?;
        let j0 = self.effective_jacobian(&frame0)?;
        let g0_int = Wilson::new(&j0, sv_tol)?.gradient(&g0);
        let dx = self.move_auto(dq)?;
        let (e, g1) = self.cart.kick_from(surface, dx.view(), g0.view())?;
        let frame1 = self.frame();
        let q1 = self.chart.values(frame1.view());
        let j1 = self.effective_jacobian(&frame1);
        match (q1, j1) {
            (Ok(q1), Ok(j1)) => {
                let g1_int = Wilson::new(&j1, sv_tol)?.gradient(&g1);
                let mut step = &q1 - &q0;
                for (k, wrap) in self.dihedral_mask().iter().enumerate() {
                    if *wrap {
                        step[k] = wrap_pi(step[k]);
                    }
                }
                let y = &g1_int - &g0_int;
                match self.update {
                    HessUpdate::Bfgs => self.hess.update(&step, &y),
                    HessUpdate::TsBfgs => self.hess.update_ts(&step, &y),
                }
            }
            _ => self.flag(RebuildReason::SingularJacobian),
        }
        Ok((e, g1))
    }
}

/// The Jacobian of the chart's internals with respect to the real atoms
/// alone: `B_r + B_d D` with the dummy response `D = -J_d^+ J_r` to
/// keep the pins of `pins`. `n_real_slots` is `3 n_real`.
pub(crate) fn effective_jacobian(
    chart: &Constraints,
    pins: &Constraints,
    frame: &Array1<f64>,
    n_real_slots: usize,
    sv_tol: f64,
) -> Result<Array2<f64>, SaddleError> {
    let b = chart.jacobian(frame.view())?;
    if !b.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("internals Jacobian"));
    }
    let n = n_real_slots;
    let b_r = b.slice(s![.., ..n]).to_owned();
    if pins.equalities().is_empty() || b.ncols() == n {
        return Ok(b_r);
    }
    let jp = pins.jacobian(frame.view())?;
    let jd = jp.slice(s![.., n..]).to_owned();
    let jr = jp.slice(s![.., ..n]).to_owned();
    let d = Wilson::new(&jd, sv_tol)?.pinv().dot(&jr).mapv(|v| -v);
    let b_d = b.slice(s![.., n..]).to_owned();
    Ok(b_r + b_d.dot(&d))
}

fn guess_matrix(chart: &Constraints) -> Array2<f64> {
    let n = chart.equalities().len();
    let mut h = Array2::zeros((n, n));
    for (i, eq) in chart.equalities().iter().enumerate() {
        h[(i, i)] = guess_curvature(eq);
    }
    h
}

/// A [`BfgsModel`] holding `h`, by seeding every eigenmode.
fn hessian_model(h: &Array2<f64>) -> Result<BfgsModel, SaddleError> {
    let n = h.nrows();
    let (lam, vec) = exact_eigh(h.view())?;
    let mut model = BfgsModel::identity(n);
    for k in 0..n {
        let u = vec.column(k).to_owned();
        model.seed_mode(&u, lam[k]);
    }
    Ok(model)
}

/// Carry the internals Hessian `h_old` of a chart into another chart.
///
/// `j_old` and `j_new` are the chart Jacobians with respect to the real
/// atoms alone (`3 n_real` columns, dummies slaved to them). The real
/// Cartesian curvature `J_old^T H J_old` is kept and read back through
/// the new chart, `H' = J_new^+T H_x J_new^+`. The guess curvature goes
/// on `ker(J_new^T)`. A Cartesian direction in the row space of `J_new`
/// and not of `J_old` arrives with zero curvature.
pub(crate) fn transport_hessian(
    h_old: &Array2<f64>,
    j_old: &Array2<f64>,
    j_new: &Array2<f64>,
    chart_new: &Constraints,
    sv_tol: f64,
) -> Result<Array2<f64>, SaddleError> {
    let hx = j_old.t().dot(h_old).dot(j_old);
    let w = Wilson::new(j_new, sv_tol)?;
    let u = w.range_basis();
    let pc = w.pinv_cols();
    let inner = pc.t().dot(&hx).dot(&pc);
    let mut h = u.dot(&inner).dot(&u.t());
    let m = j_new.nrows();
    let comp = Array2::<f64>::eye(m) - u.dot(&u.t());
    let guess = guess_matrix(chart_new);
    h += &comp.dot(&guess).dot(&comp);
    let ht = h.t().to_owned();
    Ok((h + ht) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internal::pack_cart;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1_u64 << 53) as f64) - 0.5
    }

    fn random(rows: usize, cols: usize, state: &mut u64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |_| lcg(state))
    }

    fn symmetric(n: usize, state: &mut u64) -> Array2<f64> {
        let a = random(n, n, state);
        let at = a.t().to_owned();
        (a + at) * 0.5
    }

    fn chart_of(n_eq: usize) -> Constraints {
        let x = pack_cart(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.2, 0.0],
            [0.3, 0.2, 1.1],
        ]);
        let mut c = Constraints::new(4).unwrap();
        let bonds = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]];
        for b in bonds.iter().cycle().take(n_eq) {
            c.fix_bond(*b, x.view(), None).unwrap();
        }
        c
    }

    #[test]
    fn a_transported_hessian_keeps_the_cartesian_curvature_it_can_see() {
        let mut state = 7_u64;
        let (p, m_old, m_new) = (9, 6, 11);
        let j_old = random(m_old, p, &mut state);
        let j_new = random(m_new, p, &mut state);
        let h_old = symmetric(m_old, &mut state);
        let chart = chart_of(m_new);
        let h_new = transport_hessian(&h_old, &j_old, &j_new, &chart, f64::EPSILON.sqrt()).unwrap();
        let before = j_old.t().dot(&h_old).dot(&j_old);
        let after = j_new.t().dot(&h_new).dot(&j_new);
        let scale = before.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let worst = (&before - &after)
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(
            worst < 1.0e3 * f64::EPSILON * scale,
            "curvature drifted by {worst}"
        );
        let asym = (&h_new - &h_new.t())
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(asym < f64::EPSILON * scale);
    }

    #[test]
    fn directions_the_old_chart_never_saw_get_the_guess_curvature() {
        let mut state = 11_u64;
        let (p, m_new) = (9, 12);
        let j_old = random(4, p, &mut state);
        let mut j_new = random(m_new, p, &mut state);
        for c in 0..p {
            j_new[(m_new - 1, c)] = j_new[(0, c)] + j_new[(1, c)];
        }
        let h_old = symmetric(4, &mut state);
        let chart = chart_of(m_new);
        let h_new = transport_hessian(&h_old, &j_old, &j_new, &chart, f64::EPSILON.sqrt()).unwrap();
        let mut left = Array1::zeros(m_new);
        left[0] = 1.0;
        left[1] = 1.0;
        left[m_new - 1] = -1.0;
        let w = Wilson::new(&j_new, f64::EPSILON.sqrt()).unwrap();
        assert_eq!(w.rank(), p);
        let seen = left.dot(&j_new).iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(seen < f64::EPSILON.sqrt());
        let guess = guess_matrix(&chart);
        let got = left.dot(&h_new).dot(&left);
        let want = left.dot(&guess).dot(&left);
        assert!((got - want).abs() < 1.0e3 * f64::EPSILON * want.abs());
    }
}
