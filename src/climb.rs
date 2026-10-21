//! The climbing image off the band.
//!
//! Once the armed climbing image has stayed the band's highest image
//! for [`ClimbRefine::stable_steps`] evaluations, the band hands it
//! off: the other images stop, and the climbing image alone is
//! refined to the saddle, one surface evaluation per step instead of
//! `n_images - 2`. This is the hand-off of the optimal climbing image
//! NEB (Goswami, Gunde and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063): the band supplies the tangent, a
//! Lanczos run started on it supplies the lowest mode, and an image
//! whose lowest curvature is not negative, or whose mode turns away
//! from the tangent, goes back to the band.
//!
//! The refinement step is Baker's partitioned rational function step
//! (J. Comput. Chem. 7, 385 (1986)): maximize along the followed mode,
//! minimize along the rest, inside a per-atom trust radius. The model
//! Hessian is a pair-stretch model in the manner of Lindh et al.
//! (Chem. Phys. Lett. 241, 423 (1995)): every pair within twice the
//! nearest-neighbour distance adds `w(r) e e^T` on the bond direction
//! `e`, with the exponential weight `w(r) = exp(-A (r / r_nn - 1))`
//! and the diagonal stabilisation of Packwood et al. (J. Chem. Phys.
//! 144, 164109 (2016)). A pair term's Hessian is `phi'' e e^T` plus
//! `(phi' / r)(I - e e^T)`, and `phi'` is small near bonded
//! distances, so the stiffness lies along bonds. An isotropic model
//! charges motion across bonds as much as along them and holds the
//! soft modes too stiff, which slows the step. The scale is the
//! smallest ratio of measured to model curvature over the positive
//! Lanczos directions, so the model is not stiffer than the surface
//! along any sampled direction, and the Krylov block is then
//! imposed exactly. Each step adds a symmetric rank-two secant update
//! of the TS-BFGS family (Bofill, Int. J. Quantum Chem. 94, 324
//! (2003)), `E = j u^T + u j^T - (j.s) u u^T` with `j = y - B s` and
//! `u = M s / (s^T M s)`, where `M = y y^T + (s^T |B| s) |B|` when
//! `y.s > 0` and `(s^T |B| s) |B|` otherwise. The mode is followed by
//! overlap; a short Lanczos run from it refreshes the model when the
//! followed curvature has lost most of its value or the mode turned.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rgmin::precon::ExpParams;

use crate::band::Cell;
use crate::error::SaddleError;

/// Hand-off rule and refinement constants of the climbing image.
#[derive(Clone, Copy, Debug)]
pub struct ClimbRefine {
    /// Evaluations after arming that must keep the same climbing
    /// image before the hand-off.
    pub stable_steps: usize,
    /// Krylov vectors of the first lowest-mode estimate.
    pub krylov: usize,
    /// Krylov vectors of a refresh.
    pub refresh_krylov: usize,
    /// Relative Ritz residual that ends a Krylov run.
    pub residual_tol: f64,
    /// Forward-difference length of a Hessian action.
    pub finite_difference: f64,
    /// A refresh runs once the followed curvature is above this
    /// fraction of the curvature the last Krylov run measured.
    pub curvature_retained: f64,
    /// A refresh runs once the followed mode's overlap with the
    /// previous one is below this.
    pub min_overlap: f64,
    /// Steps between two refreshes.
    pub refresh_interval: usize,
    /// Smallest per-atom trust radius. The largest is the band's
    /// `max_move`.
    pub trust_min: f64,
    /// Lowest `|cos|` between the first mode and the band tangent.
    /// OCI-NEB's value 0.85 sits above its lower bound `1 / sqrt 2`.
    pub min_alignment: f64,
    /// After a refused hand-off the next one waits until the climbing
    /// force is under this fraction of the refused one.
    pub retry_factor: f64,
}

impl Default for ClimbRefine {
    fn default() -> Self {
        Self {
            stable_steps: 1,
            krylov: 20,
            refresh_krylov: 12,
            residual_tol: 0.1,
            finite_difference: 1e-3,
            curvature_retained: 0.3,
            min_overlap: 0.8,
            refresh_interval: 4,
            trust_min: 1e-3,
            min_alignment: 0.85,
            retry_factor: 0.5,
        }
    }
}

impl ClimbRefine {
    pub(crate) fn validate(&self) -> Result<(), SaddleError> {
        let positive = [
            self.residual_tol,
            self.finite_difference,
            self.trust_min,
            self.retry_factor,
        ];
        if self.krylov < 2
            || self.refresh_krylov < 2
            || positive.iter().any(|v| !(v.is_finite() && *v > 0.0))
            || !(0.0..=1.0).contains(&self.curvature_retained)
            || !(0.0..=1.0).contains(&self.min_overlap)
            || !(0.0..=1.0).contains(&self.min_alignment)
        {
            return Err(SaddleError::Invalid(
                "climbing-image refinement constants out of range".into(),
            ));
        }
        Ok(())
    }
}

/// What one refinement step did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ClimbOutcome {
    /// A Hessian action at a displaced point; the image did not move.
    Probed,
    /// The image moved to a new evaluated point.
    Moved,
    /// The lowest curvature is not negative, or the first mode is not
    /// along the band; the image goes back to the band where it stands.
    Refused,
}

enum Stage {
    Krylov {
        first: bool,
        limit: usize,
        basis: Vec<Array1<f64>>,
        images: Vec<Array1<f64>>,
        alpha: Vec<f64>,
        beta: Vec<f64>,
        next: Array1<f64>,
    },
    Follow,
}

/// Single-image saddle refinement. One surface evaluation per step.
pub(crate) struct Climber {
    cfg: ClimbRefine,
    image: usize,
    x: Array1<f64>,
    g: Array1<f64>,
    e: f64,
    tangent: Array1<f64>,
    cell: Option<Cell>,
    translation: bool,
    rotation: bool,
    rigid: Option<Array2<f64>>,
    b: Array2<f64>,
    mode: Array1<f64>,
    mode_curvature: f64,
    mode_overlap: f64,
    reference_curvature: f64,
    trust: f64,
    max_move: f64,
    iteration: usize,
    last_refresh: Option<usize>,
    stage: Stage,
}

impl Climber {
    /// Start at band image `image` with its evaluated energy and
    /// gradient. `tangent` is the band direction through the image.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        cfg: ClimbRefine,
        image: usize,
        x: Array1<f64>,
        e: f64,
        g: Array1<f64>,
        tangent: Array1<f64>,
        cell: Option<Cell>,
        remove_translation: bool,
        max_move: f64,
    ) -> Result<Self, SaddleError> {
        let n = x.len();
        let atoms = n / 3;
        let translation = remove_translation && atoms >= 2;
        let rotation = translation && cell.is_none() && atoms >= 3;
        let rigid = rigid_basis(x.view(), translation, rotation);
        let mut tau = project(&tangent, rigid.as_ref());
        let norm = tau.dot(&tau).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            return Err(SaddleError::Invalid(
                "climbing image has no tangent to refine along".into(),
            ));
        }
        tau /= norm;
        let trust = if max_move > 0.0 {
            max_move
        } else {
            f64::INFINITY
        };
        Ok(Self {
            stage: Stage::Krylov {
                first: true,
                limit: cfg.krylov,
                basis: Vec::new(),
                images: Vec::new(),
                alpha: Vec::new(),
                beta: Vec::new(),
                next: tau.clone(),
            },
            cfg,
            image,
            x,
            g,
            e,
            tangent: tau,
            cell,
            translation,
            rotation,
            rigid,
            b: Array2::zeros((n, n)),
            mode: Array1::zeros(n),
            mode_curvature: 0.0,
            mode_overlap: 1.0,
            reference_curvature: -1e-3,
            trust,
            max_move: if max_move > 0.0 {
                max_move
            } else {
                f64::INFINITY
            },
            iteration: 0,
            last_refresh: None,
        })
    }

    pub(crate) fn image(&self) -> usize {
        self.image
    }

    pub(crate) fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub(crate) fn gradient(&self) -> ArrayView1<'_, f64> {
        self.g.view()
    }

    pub(crate) fn energy(&self) -> f64 {
        self.e
    }

    /// The followed mode, zero before the first Krylov run ends.
    pub(crate) fn mode(&self) -> ArrayView1<'_, f64> {
        self.mode.view()
    }

    /// Curvature along the followed mode, `None` before the first
    /// Krylov run ends.
    pub(crate) fn curvature(&self) -> Option<f64> {
        matches!(self.stage, Stage::Follow).then_some(self.mode_curvature)
    }

    /// One evaluation: a Hessian action while a Krylov run is open,
    /// else one partitioned RFO step. `eval` is called once.
    pub(crate) fn step<F>(&mut self, mut eval: F) -> Result<ClimbOutcome, SaddleError>
    where
        F: FnMut(ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError>,
    {
        match &self.stage {
            Stage::Krylov { next, .. } => {
                let dr = self.cfg.finite_difference;
                let probe = &self.x + &(next * dr);
                let (_, gp) = eval(probe.view())?;
                check_finite(gp.view())?;
                let hv = project(&((&gp - &self.g) / dr), self.rigid.as_ref());
                Ok(self.absorb(hv))
            }
            Stage::Follow => self.follow(&mut eval),
        }
    }

    fn absorb(&mut self, hv: Array1<f64>) -> ClimbOutcome {
        let tol = self.cfg.residual_tol;
        let Stage::Krylov {
            first,
            limit,
            basis,
            images,
            alpha,
            beta,
            next,
        } = &mut self.stage
        else {
            unreachable!("absorb outside a Krylov run");
        };
        let v = next.clone();
        let a = v.dot(&hv);
        let mut w = &hv - &(&v * a);
        if let (Some(prev), Some(b)) = (basis.last(), beta.last()) {
            w -= &(prev * *b);
        }
        basis.push(v);
        images.push(hv);
        alpha.push(a);
        for q in basis.iter() {
            let c = q.dot(&w);
            w -= &(q * c);
        }
        let k = basis.len();
        let mut t = Array2::zeros((k, k));
        for i in 0..k {
            t[(i, i)] = alpha[i];
            if i + 1 < k {
                t[(i, i + 1)] = beta[i];
                t[(i + 1, i)] = beta[i];
            }
        }
        let (lams, vecs) = sym_eigh(t.view());
        let lam = lams[0];
        let y = vecs.column(0);
        let mut vec = Array1::<f64>::zeros(self.x.len());
        let mut hvec = Array1::<f64>::zeros(self.x.len());
        for i in 0..k {
            vec.scaled_add(y[i], &basis[i]);
            hvec.scaled_add(y[i], &images[i]);
        }
        let residual = &hvec - &(&vec * lam);
        let rn = residual.dot(&residual).sqrt();
        let wn = w.dot(&w).sqrt();
        let done = (k >= 2 && rn < tol * lam.abs().max(1e-8)) || k >= *limit || wn < 1e-10;
        if !done {
            beta.push(wn);
            *next = w / wn;
            return ClimbOutcome::Probed;
        }
        let vn = vec.dot(&vec).sqrt();
        vec /= vn;
        let first = *first;
        let basis = std::mem::take(basis);
        let images = std::mem::take(images);
        if first {
            let alignment = vec.dot(&self.tangent).abs();
            if lam.is_nan() || lam >= 0.0 || alignment < self.cfg.min_alignment {
                return ClimbOutcome::Refused;
            }
            self.seed(&basis, &images);
        } else {
            impose(&mut self.b, &basis, &images);
        }
        self.mode = vec;
        self.mode_curvature = lam;
        self.mode_overlap = 1.0;
        self.reference_curvature = lam.min(-1e-3);
        self.stage = Stage::Follow;
        ClimbOutcome::Probed
    }

    /// `B = mu P` at the image, `mu` the smallest ratio `v.Hv / v.Pv`
    /// over the Krylov vectors of positive curvature, then the block
    /// imposed exactly.
    fn seed(&mut self, basis: &[Array1<f64>], images: &[Array1<f64>]) {
        let mut p = pair_model(self.x.view(), self.cell.as_ref());
        if let Some(q) = &self.rigid {
            p = project_both(&p, q);
        }
        let mut mu = f64::INFINITY;
        for (v, hv) in basis.iter().zip(images) {
            let rq = v.dot(hv);
            let pv = v.dot(&p.dot(v));
            if rq > 0.0 && pv > 0.0 {
                mu = mu.min(rq / pv);
            }
        }
        if !mu.is_finite() {
            mu = 1.0;
        }
        self.b = p * mu;
        impose(&mut self.b, basis, images);
    }

    fn follow<F>(&mut self, eval: &mut F) -> Result<ClimbOutcome, SaddleError>
    where
        F: FnMut(ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError>,
    {
        let step = self.prfo_step();
        let predicted = self.g.dot(&step) + 0.5 * step.dot(&self.b.dot(&step));
        let trial = &self.x + &step;
        let (e_new, g_new) = eval(trial.view())?;
        if !e_new.is_finite() {
            return Err(SaddleError::NonFinite("climbing image energy"));
        }
        check_finite(g_new.view())?;
        let s = project(&step, self.rigid.as_ref());
        let y = project(&(&g_new - &self.g), self.rigid.as_ref());
        ts_bfgs(&mut self.b, &s, &y);
        let actual = e_new - self.e;
        let ratio = if predicted.abs() > 1e-14 {
            actual / predicted
        } else {
            1.0
        };
        let moved = per_atom_max(step.view());
        if !(0.25..=4.0).contains(&ratio) {
            self.trust = (0.5 * moved).max(self.cfg.trust_min);
        } else if ratio > 0.75 && ratio < 1.33 && moved > 0.9 * self.trust {
            self.trust = (2.0 * self.trust).min(self.max_move);
        }
        self.x = trial;
        self.g = g_new;
        self.e = e_new;
        self.rigid = rigid_basis(self.x.view(), self.translation, self.rotation);
        self.iteration += 1;
        let rested = self
            .last_refresh
            .is_none_or(|at| self.iteration - at >= self.cfg.refresh_interval);
        if rested
            && (self.mode_curvature > self.cfg.curvature_retained * self.reference_curvature
                || self.mode_overlap < self.cfg.min_overlap)
        {
            self.last_refresh = Some(self.iteration);
            self.stage = Stage::Krylov {
                first: false,
                limit: self.cfg.refresh_krylov,
                basis: Vec::new(),
                images: Vec::new(),
                alpha: Vec::new(),
                beta: Vec::new(),
                next: project(&self.mode, self.rigid.as_ref()),
            };
            if let Stage::Krylov { next, .. } = &mut self.stage {
                let nn = next.dot(next).sqrt();
                if nn > 0.0 {
                    *next /= nn;
                }
            }
        }
        Ok(ClimbOutcome::Moved)
    }

    /// Partitioned RFO in the eigenbasis of the model: the eigenvector
    /// closest to the followed mode is maximized, the rest minimized.
    /// Rigid directions sit at a large shift, so they take no step.
    fn prfo_step(&mut self) -> Array1<f64> {
        let n = self.x.len();
        let g = project(&self.g, self.rigid.as_ref());
        let mut model = match &self.rigid {
            Some(q) => project_both(&self.b, q),
            None => self.b.clone(),
        };
        if let Some(q) = &self.rigid {
            let scale = model.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let shift = 1e3 * scale;
            model += &(q.dot(&q.t()) * shift);
        }
        let (w, u) = sym_eigh(model.view());
        let overlaps = u.t().dot(&self.mode);
        let mut k = 0;
        for i in 1..n {
            if overlaps[i].abs() > overlaps[k].abs() {
                k = i;
            }
        }
        self.mode_overlap = overlaps[k].abs();
        let sign = if overlaps[k] < 0.0 { -1.0 } else { 1.0 };
        self.mode = u.column(k).to_owned() * sign;
        self.mode_curvature = w[k];
        let gt = u.t().dot(&g);
        let mut d = Array1::<f64>::zeros(n);
        let (lam, gk) = (w[k], gt[k]);
        let up = 0.5 * (lam + (lam * lam + 4.0 * gk * gk).sqrt());
        if (lam - up).abs() > 1e-14 {
            d[k] = -gk / (lam - up);
        }
        let rest: Vec<usize> = (0..n).filter(|&i| i != k).collect();
        let down = rfo_lowest(
            &rest.iter().map(|&i| w[i]).collect::<Vec<_>>(),
            &rest.iter().map(|&i| gt[i]).collect::<Vec<_>>(),
        );
        for &i in &rest {
            let den = w[i] - down;
            if den.abs() > 1e-14 {
                d[i] = -gt[i] / den;
            }
        }
        let mut step = u.dot(&d);
        let m = per_atom_max(step.view());
        if m > self.trust {
            step *= self.trust / m;
        }
        step
    }
}

/// Lowest eigenvalue of the arrowhead `[[diag(w), g], [g^T, 0]]`: the
/// root of `l + sum g_i^2 / (w_i - l)` below `min w`.
fn rfo_lowest(w: &[f64], g: &[f64]) -> f64 {
    if w.is_empty() {
        return 0.0;
    }
    let w_min = w.iter().copied().fold(f64::INFINITY, f64::min);
    let g_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
    let f = |l: f64| {
        l + w
            .iter()
            .zip(g)
            .map(|(wi, gi)| gi * gi / (wi - l))
            .sum::<f64>()
    };
    let mut lo = w_min.min(0.0) - g_norm - 1e-12 * (1.0 + w_min.abs());
    let mut hi = w_min;
    let gap = 1e-14 * (1.0 + w_min.abs());
    if f(hi - gap) <= 0.0 {
        return w_min;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// `B += R V^T + V R^T - V (V^T R) V^T` with `R = HV - B V` on an
/// orthonormalized block, so that `B V = HV` afterwards.
fn impose(b: &mut Array2<f64>, basis: &[Array1<f64>], images: &[Array1<f64>]) {
    let (v, hv) = orthonormal_block(basis, images);
    if v.ncols() == 0 {
        return;
    }
    let r = &hv - &b.dot(&v);
    let m = v.t().dot(&r);
    *b += &r.dot(&v.t());
    *b += &v.dot(&r.t());
    *b -= &v.dot(&m).dot(&v.t());
    symmetrize(b);
}

/// Modified Gram-Schmidt on the basis, the same combinations on the
/// images; a dependent column is dropped.
fn orthonormal_block(basis: &[Array1<f64>], images: &[Array1<f64>]) -> (Array2<f64>, Array2<f64>) {
    let n = basis.first().map_or(0, |v| v.len());
    let mut vs: Vec<Array1<f64>> = Vec::new();
    let mut hs: Vec<Array1<f64>> = Vec::new();
    for (v0, h0) in basis.iter().zip(images) {
        let mut v = v0.clone();
        let mut h = h0.clone();
        for (q, hq) in vs.iter().zip(&hs) {
            let c = q.dot(&v);
            v -= &(q * c);
            h -= &(hq * c);
        }
        let nv = v.dot(&v).sqrt();
        if nv > 1e-8 {
            vs.push(v / nv);
            hs.push(h / nv);
        }
    }
    let mut v = Array2::zeros((n, vs.len()));
    let mut hv = Array2::zeros((n, vs.len()));
    for (j, (a, b)) in vs.iter().zip(&hs).enumerate() {
        v.column_mut(j).assign(a);
        hv.column_mut(j).assign(b);
    }
    (v, hv)
}

fn ts_bfgs(b: &mut Array2<f64>, s: &Array1<f64>, y: &Array1<f64>) {
    if s.dot(s) < 1e-20 {
        return;
    }
    let j = y - &b.dot(s);
    let (lams, vecs) = sym_eigh(b.view());
    let coeffs = vecs.t().dot(s);
    let abs_s = vecs.dot(&(&coeffs * &lams.mapv(f64::abs)));
    let s_abs_s = s.dot(&abs_s);
    let ys = y.dot(s);
    let ms = if ys > 0.0 {
        y * ys + &(&abs_s * s_abs_s)
    } else {
        &abs_s * s_abs_s
    };
    let sms = s.dot(&ms);
    if sms.is_nan() || sms <= 1e-30 {
        return;
    }
    let u = ms / sms;
    let js = j.dot(s);
    let n = s.len();
    for r in 0..n {
        for c in 0..n {
            b[(r, c)] += j[r] * u[c] + u[r] * j[c] - js * u[r] * u[c];
        }
    }
    symmetrize(b);
}

fn symmetrize(b: &mut Array2<f64>) {
    let n = b.nrows();
    for r in 0..n {
        for c in 0..r {
            let m = 0.5 * (b[(r, c)] + b[(c, r)]);
            b[(r, c)] = m;
            b[(c, r)] = m;
        }
    }
}

/// `(I - Q Q^T) A (I - Q Q^T)`.
fn project_both(a: &Array2<f64>, q: &Array2<f64>) -> Array2<f64> {
    let aq = a.dot(q);
    let qt_a = q.t().dot(a);
    let qt_aq = q.t().dot(&aq);
    let mut out = a - &aq.dot(&q.t()) - &q.dot(&qt_a);
    out += &q.dot(&qt_aq).dot(&q.t());
    out
}

fn project(v: &Array1<f64>, q: Option<&Array2<f64>>) -> Array1<f64> {
    match q {
        Some(q) => v - &q.dot(&q.t().dot(v)),
        None => v.clone(),
    }
}

/// Orthonormal translations and rotations about the centroid at `x`.
fn rigid_basis(x: ArrayView1<f64>, translation: bool, rotation: bool) -> Option<Array2<f64>> {
    if !translation {
        return None;
    }
    let atoms = x.len() / 3;
    let mut centre = [0.0; 3];
    for a in 0..atoms {
        for c in 0..3 {
            centre[c] += x[3 * a + c] / atoms as f64;
        }
    }
    let mut seeds: Vec<Array1<f64>> = Vec::new();
    for axis in 0..3 {
        let mut t = Array1::zeros(x.len());
        for a in 0..atoms {
            t[3 * a + axis] = 1.0;
        }
        seeds.push(t);
        if rotation {
            let mut r = Array1::zeros(x.len());
            for a in 0..atoms {
                let p = [
                    x[3 * a] - centre[0],
                    x[3 * a + 1] - centre[1],
                    x[3 * a + 2] - centre[2],
                ];
                let mut e = [0.0; 3];
                e[axis] = 1.0;
                r[3 * a] = p[1] * e[2] - p[2] * e[1];
                r[3 * a + 1] = p[2] * e[0] - p[0] * e[2];
                r[3 * a + 2] = p[0] * e[1] - p[1] * e[0];
            }
            seeds.push(r);
        }
    }
    let mut kept: Vec<Array1<f64>> = Vec::new();
    for mut v in seeds {
        let scale = v.dot(&v).sqrt();
        for q in &kept {
            let c = q.dot(&v);
            v -= &(q * c);
        }
        let nv = v.dot(&v).sqrt();
        if nv > 1e-8 * scale.max(1.0) {
            kept.push(v / nv);
        }
    }
    let mut q = Array2::zeros((x.len(), kept.len()));
    for (j, v) in kept.iter().enumerate() {
        q.column_mut(j).assign(v);
    }
    Some(q)
}

/// Dense pair-stretch model at unit scale: `w(r) e e^T` on every pair
/// within twice the nearest-neighbour distance `r_nn` (the largest
/// nearest-neighbour distance over atoms), minimum-image distances
/// under a cell, and Packwood's `c_stab` on the diagonal.
fn pair_model(x: ArrayView1<f64>, cell: Option<&Cell>) -> Array2<f64> {
    let atoms = x.len() / 3;
    let mut pairs = Vec::with_capacity(atoms * atoms.saturating_sub(1) / 2);
    let mut nearest = vec![f64::INFINITY; atoms];
    for i in 0..atoms {
        for j in i + 1..atoms {
            let mut diff = Array1::from(vec![
                x[3 * j] - x[3 * i],
                x[3 * j + 1] - x[3 * i + 1],
                x[3 * j + 2] - x[3 * i + 2],
            ]);
            if let Some(cell) = cell {
                cell.minimum_image(&mut diff);
            }
            let r = diff.dot(&diff).sqrt();
            nearest[i] = nearest[i].min(r);
            nearest[j] = nearest[j].min(r);
            pairs.push((i, j, r, diff));
        }
    }
    let r_nn = nearest
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0_f64, f64::max);
    let r_nn = if r_nn > 0.0 { r_nn } else { 1.0 };
    let params = ExpParams::default();
    let mut p = Array2::eye(x.len()) * params.c_stab;
    for (i, j, r, diff) in pairs {
        if r.is_nan() || r >= 2.0 * r_nn {
            continue;
        }
        let e = diff / r;
        let w = (-params.a * (r / r_nn - 1.0)).exp();
        for a in 0..3 {
            for b in 0..3 {
                let v = w * e[a] * e[b];
                p[(3 * i + a, 3 * i + b)] += v;
                p[(3 * j + a, 3 * j + b)] += v;
                p[(3 * i + a, 3 * j + b)] -= v;
                p[(3 * j + a, 3 * i + b)] -= v;
            }
        }
    }
    p
}

fn per_atom_max(v: ArrayView1<f64>) -> f64 {
    v.exact_chunks(3)
        .into_iter()
        .map(|a| a.dot(&a).sqrt())
        .fold(0.0_f64, f64::max)
}

fn check_finite(g: ArrayView1<f64>) -> Result<(), SaddleError> {
    if g.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(SaddleError::NonFinite("climbing image gradient"))
    }
}

/// Symmetric eigendecomposition, eigenvalues ascending, eigenvectors
/// as columns.
fn sym_eigh(a: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    let mat = faer::Mat::from_fn(n, n, |i, j| 0.5 * (a[(i, j)] + a[(j, i)]));
    match mat.self_adjoint_eigen(faer::Side::Lower) {
        Ok(evd) => {
            let s = evd.S();
            let u = evd.U();
            let mut values = Array1::zeros(n);
            let mut vectors = Array2::zeros((n, n));
            for c in 0..n {
                values[c] = s[c];
                for r in 0..n {
                    vectors[(r, c)] = u[(r, c)];
                }
            }
            (values, vectors)
        }
        Err(_) => (a.diag().to_owned(), Array2::eye(n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfo_lowest_solves_the_secular_equation() {
        let w = [1.0, 2.5, 4.0];
        let g = [0.3, -0.2, 0.5];
        let l = rfo_lowest(&w, &g);
        let f = l + w.iter().zip(&g).map(|(a, b)| b * b / (a - l)).sum::<f64>();
        assert!(f.abs() < 1e-10, "residual {f}");
        assert!(l < 1.0);
    }

    #[test]
    fn imposed_block_matches_its_images() {
        let mut b = Array2::<f64>::eye(4) * 3.0;
        let v = vec![
            Array1::from(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 1.0, 0.0]),
        ];
        let h = Array2::from_shape_vec(
            (4, 4),
            vec![
                -2.0, 0.5, 0.0, 0.0, 0.5, 4.0, 1.0, 0.0, 0.0, 1.0, 5.0, 0.2, 0.0, 0.0, 0.2, 6.0,
            ],
        )
        .unwrap();
        let hv: Vec<Array1<f64>> = v.iter().map(|x| h.dot(x)).collect();
        impose(&mut b, &v, &hv);
        for (x, hx) in v.iter().zip(&hv) {
            let diff = &b.dot(x) - hx;
            assert!(diff.dot(&diff).sqrt() < 1e-12);
        }
    }

    #[test]
    fn ts_bfgs_meets_the_secant_condition() {
        let mut b = Array2::<f64>::eye(3);
        b[(0, 0)] = -1.0;
        let s = Array1::from(vec![0.1, 0.2, -0.1]);
        let y = Array1::from(vec![-0.05, 0.5, -0.3]);
        ts_bfgs(&mut b, &s, &y);
        let diff = &b.dot(&s) - &y;
        assert!(diff.dot(&diff).sqrt() < 1e-12);
    }

    fn quadratic(
        h: &Array2<f64>,
    ) -> impl Fn(ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let h = h.clone();
        move |x| {
            let g = h.dot(&x);
            Ok((0.5 * x.dot(&g), g))
        }
    }

    fn coupled(diag: &[f64]) -> Array2<f64> {
        let n = diag.len();
        let mut h = Array2::from_diag(&Array1::from(diag.to_vec()));
        for i in 0..n - 1 {
            h[(i, i + 1)] = 0.1;
            h[(i + 1, i)] = 0.1;
        }
        h
    }

    #[test]
    fn climber_converges_on_a_quadratic_saddle() {
        let h = coupled(&[-1.0, 2.0, 3.0, 2.5, 4.0, 1.5]);
        let f = quadratic(&h);
        let x0 = Array1::from(vec![0.3, 0.1, -0.1, 0.05, 0.02, -0.04]);
        let (e0, g0) = f(x0.view()).unwrap();
        let tangent = Array1::from(vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.0]);
        let mut climber = Climber::new(
            ClimbRefine::default(),
            1,
            x0,
            e0,
            g0,
            tangent,
            None,
            false,
            0.2,
        )
        .unwrap();
        let mut moved = 0;
        while climber.gradient().dot(&climber.gradient()).sqrt() > 1e-8 {
            let outcome = climber.step(&f).unwrap();
            assert_ne!(outcome, ClimbOutcome::Refused);
            if outcome == ClimbOutcome::Moved {
                moved += 1;
            }
            assert!(
                moved < 50,
                "|g| {}",
                climber.gradient().dot(&climber.gradient())
            );
        }
        assert!(climber.position().iter().all(|v| v.abs() < 1e-6));
        assert!(climber.curvature().unwrap() < 0.0);
    }

    #[test]
    fn climber_refuses_a_minimum() {
        let h = coupled(&[1.0, 2.0, 3.0, 2.5, 4.0, 1.5]);
        let f = quadratic(&h);
        let x0 = Array1::from(vec![0.3, 0.1, -0.1, 0.05, 0.02, -0.04]);
        let (e0, g0) = f(x0.view()).unwrap();
        let tangent = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut climber = Climber::new(
            ClimbRefine::default(),
            1,
            x0.clone(),
            e0,
            g0,
            tangent,
            None,
            false,
            0.2,
        )
        .unwrap();
        let outcome = loop {
            match climber.step(&f).unwrap() {
                ClimbOutcome::Probed => continue,
                other => break other,
            }
        };
        assert_eq!(outcome, ClimbOutcome::Refused);
        assert_eq!(climber.position(), x0.view());
    }

    #[test]
    fn rigid_basis_spans_translations_and_rotations() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.0, 1.2, 0.0, 0.3, 0.2, 0.9,
        ]);
        let q = rigid_basis(x.view(), true, true).unwrap();
        assert_eq!(q.ncols(), 6);
        let gram = q.t().dot(&q);
        for i in 0..6 {
            for j in 0..6 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((gram[(i, j)] - want).abs() < 1e-12);
            }
        }
    }
}
