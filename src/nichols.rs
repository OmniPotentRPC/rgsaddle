//! Index-1 Newton: the Nichols displacement and the Hessian updates
//! that go with a saddle search.
//!
//! The displacement and the Powell update follow i-PI
//! `ipi/utils/mintools.py` (`nichols`, `Powell`) under the MIT option
//! of i-PI's dual MIT/GPL license.
//!
//! Copyright (C) 2013, Joshua More and Michele Ceriotti
//! Algorithms implemented by Michele Ceriotti and Benjamin Helfrecht, 2015
//!
//! The level shift is Simons and Nichols, *Int. J. Quantum Chem.* 38
//! (1990) 263-276, <https://doi.org/10.1002/qua.560382435>.
//! Powell's symmetric update is Fletcher, *Practical Methods of
//! Optimization*, 2nd ed. (1987).
//! Bofill's mix of that update with the symmetric rank-one term is
//! Bofill, *J. Comput. Chem.* 15 (1994) 1-11,
//! <https://doi.org/10.1002/jcc.540150102>.
//!
//! rgmin's Banerjee RFO minimizes a scalar and accepts a step only
//! when the value falls. An index-1 step climbs the lowest mode, so
//! the displacement here is the Nichols shift on the supplied
//! spectrum rather than that minimizer.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;
use crate::minmode::PointSurface;

/// Which stationary point the Nichols shift is aimed at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NicholsMode {
    /// Level shift for a minimum. `trust_radius` bounds the step in
    /// the eigenbasis, as i-PI `nichols(..., mode=0)`.
    Minimize,
    /// Index-1 saddle: climb the lowest mode, descend the rest.
    /// i-PI `nichols(..., mode=1)`. The trust radius does not enter
    /// this shift; [`Index1Session`] caps the Cartesian step.
    Index1,
}

/// How [`Index1Session`] advances the Cartesian energy Hessian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HessianUpdate {
    /// Powell symmetric Broyden (i-PI `Powell`).
    Powell,
    /// Bofill mix of Powell and the symmetric rank-one update.
    #[default]
    Bofill,
}

/// Stepping controls for [`Index1Session`].
#[derive(Clone, Debug)]
pub struct Index1Config {
    pub update: HessianUpdate,
    pub mode: NicholsMode,
    /// Euclidean bound on the Cartesian step.
    pub trust_radius: f64,
    /// Stop when every |gradient| component is at or under this.
    pub force_tol: f64,
    /// Central-difference step used when no Hessian was supplied.
    pub fd_dr: f64,
}

impl Default for Index1Config {
    fn default() -> Self {
        Self {
            update: HessianUpdate::Bofill,
            mode: NicholsMode::Index1,
            trust_radius: 0.2,
            force_tol: 1e-5,
            fd_dr: 1e-4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Index1Status {
    Running,
    Converged,
}

#[derive(Clone, Debug)]
pub struct Index1Report {
    pub status: Index1Status,
    pub max_force: f64,
    /// Lowest eigenvalue of the Hessian that produced this step.
    pub curvature: f64,
    pub iteration: usize,
    /// Euclidean length of the Cartesian step just taken. Zero when
    /// the gradient was already under `force_tol`.
    pub max_step: f64,
}

/// Cyclic Jacobi eigendecomposition. Eigenvalues ascend. Columns of
/// the second matrix are the eigenvectors.
pub(crate) fn sym_eigh(
    hessian: ArrayView2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n {
        return Err(SaddleError::Shape(
            "Hessian must be square and nonempty".into(),
        ));
    }
    let mut a = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            if !v.is_finite() {
                return Err(SaddleError::NonFinite("Hessian"));
            }
            a[(i, j)] = v;
        }
    }
    let mut v = Array2::<f64>::eye(n);
    if n > 1 {
        for _ in 0..64 {
            let mut off = 0.0;
            for p in 0..n {
                for q in (p + 1)..n {
                    off += a[(p, q)] * a[(p, q)];
                }
            }
            let scale = a.diag().iter().fold(1.0_f64, |m, d| m.max(d.abs()));
            if off.sqrt() <= 1e-15 * scale {
                break;
            }
            for p in 0..n {
                for q in (p + 1)..n {
                    let apq = a[(p, q)];
                    if apq.abs() <= 1e-300 {
                        continue;
                    }
                    let theta = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                    let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                    let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;
                    for i in 0..n {
                        let aip = a[(i, p)];
                        let aiq = a[(i, q)];
                        a[(i, p)] = c * aip - s * aiq;
                        a[(i, q)] = s * aip + c * aiq;
                    }
                    for i in 0..n {
                        let api = a[(p, i)];
                        let aqi = a[(q, i)];
                        a[(p, i)] = c * api - s * aqi;
                        a[(q, i)] = s * api + c * aqi;
                    }
                    for i in 0..n {
                        let vip = v[(i, p)];
                        let viq = v[(i, q)];
                        v[(i, p)] = c * vip - s * viq;
                        v[(i, q)] = s * vip + c * viq;
                    }
                }
            }
        }
    }
    let evals = a.diag().to_owned();
    if evals.iter().any(|e| !e.is_finite()) {
        return Err(SaddleError::NonFinite("Hessian eigenvalues"));
    }
    sort_spectrum(evals, v)
}

fn sort_spectrum(
    evals: Array1<f64>,
    evecs: Array2<f64>,
) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    let nmode = evals.len();
    let mut order: Vec<usize> = (0..nmode).collect();
    order.sort_by(|&i, &j| {
        evals[i]
            .partial_cmp(&evals[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let sorted_evals = Array1::from_iter(order.iter().map(|&k| evals[k]));
    let mut sorted = Array2::<f64>::zeros((evecs.nrows(), nmode));
    for (col, &k) in order.iter().enumerate() {
        sorted.column_mut(col).assign(&evecs.column(k));
    }
    Ok((sorted_evals, sorted))
}

pub(crate) fn masses_or_ones(
    n: usize,
    masses: Option<ArrayView1<f64>>,
) -> Result<Array1<f64>, SaddleError> {
    match masses {
        None => Ok(Array1::ones(n)),
        Some(m) => {
            if m.len() != n {
                return Err(SaddleError::Shape(
                    "masses must match the gradient length".into(),
                ));
            }
            if m.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                return Err(SaddleError::Invalid(
                    "masses must be positive and finite".into(),
                ));
            }
            Ok(m.to_owned())
        }
    }
}

/// Nichols displacement from a spectrum.
///
/// `gradient` is the Cartesian energy gradient. i-PI passes forces,
/// the negation of this vector. `evecs` is row-major with eigenvectors
/// in the columns; it may be rectangular when the host has dropped
/// external modes (`n` by `n - m` in i-PI). `evals` and the columns
/// are the eigenpairs of the mass-weighted Hessian. `masses` is the
/// per-coordinate mass; pass `None` for unit masses.
///
/// For [`NicholsMode::Index1`] the returned step is i-PI `nichols`
/// with `mode=1`: `trust_radius` is ignored. For
/// [`NicholsMode::Minimize`] `trust_radius` is i-PI `big_step`.
pub fn nichols_step(
    gradient: ArrayView1<f64>,
    evals: ArrayView1<f64>,
    evecs: ArrayView2<f64>,
    masses: Option<ArrayView1<f64>>,
    trust_radius: f64,
    mode: NicholsMode,
) -> Result<Array1<f64>, SaddleError> {
    let n = gradient.len();
    let nmode = evals.len();
    if n == 0 || evecs.nrows() != n || evecs.ncols() != nmode || nmode == 0 {
        return Err(SaddleError::Shape(
            "gradient, eigenvalues, and eigenvector columns must agree".into(),
        ));
    }
    if gradient.iter().any(|v| !v.is_finite()) || evals.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("Nichols spectrum"));
    }
    if mode == NicholsMode::Index1 && nmode < 2 {
        return Err(SaddleError::Shape(
            "an index-1 Nichols step needs two or more modes".into(),
        ));
    }
    if mode == NicholsMode::Minimize && !(trust_radius.is_finite() && trust_radius > 0.0) {
        return Err(SaddleError::Invalid(
            "minimize Nichols step needs a positive finite trust radius".into(),
        ));
    }
    let mass = masses_or_ones(n, masses)?;
    let (evals, evecs) = sort_spectrum(evals.to_owned(), evecs.to_owned())?;

    // f_mw = F / sqrt(m), F = -gradient. g_E = -W^T f_mw.
    let mut g_e = Array1::<f64>::zeros(nmode);
    for k in 0..nmode {
        let mut acc = 0.0;
        for i in 0..n {
            let f_mw = -gradient[i] / mass[i].sqrt();
            acc += f_mw * evecs[(i, k)];
        }
        g_e[k] = -acc;
    }

    let (alpha, lamb) = match mode {
        NicholsMode::Minimize => {
            let mut dx_e = eigen_step(1.0, 0.0, &g_e, &evals)?;
            let norm2 = dx_e.dot(&dx_e);
            if evals[0] < 0.0 || norm2 > trust_radius * trust_radius {
                let lamb = evals[0] - (g_e[0] / trust_radius).abs();
                dx_e = eigen_step(1.0, lamb, &g_e, &evals)?;
            }
            return cartesian_step(&evecs, &dx_e, &mass);
        }
        NicholsMode::Index1 => index1_shift(evals[0], evals[1]),
    };
    let dx_e = eigen_step(alpha, lamb, &g_e, &evals)?;
    cartesian_step(&evecs, &dx_e, &mass)
}

fn index1_shift(d0: f64, d1: f64) -> (f64, f64) {
    if d0 > 0.0 {
        if d1 / 2.0 > d0 {
            (1.0, (2.0 * d0 + d1) / 4.0)
        } else {
            ((d1 - d0) / d1, (3.0 * d0 + d1) / 4.0)
        }
    } else if d1 < 0.0 {
        if d1 >= d0 / 2.0 {
            (1.0, (d0 + 2.0 * d1) / 4.0)
        } else {
            ((d0 - d1) / d1, (d0 + 3.0 * d1) / 4.0)
        }
    } else {
        (1.0, (d0 + d1) / 4.0)
    }
}

fn eigen_step(
    alpha: f64,
    lamb: f64,
    g_e: &Array1<f64>,
    evals: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let mut dx_e = Array1::<f64>::zeros(g_e.len());
    for k in 0..g_e.len() {
        dx_e[k] = alpha * g_e[k] / (lamb - evals[k]);
    }
    if dx_e.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("Nichols eigenstep"));
    }
    Ok(dx_e)
}

fn cartesian_step(
    evecs: &Array2<f64>,
    dx_e: &Array1<f64>,
    mass: &Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let n = evecs.nrows();
    let mut dx = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..dx_e.len() {
            acc += evecs[(i, k)] * dx_e[k];
        }
        dx[i] = acc / mass[i].sqrt();
    }
    if dx.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("Nichols step"));
    }
    Ok(dx)
}

/// Nichols displacement from a Cartesian energy Hessian and gradient.
///
/// The Hessian is mass-weighted, diagonalized, and passed to
/// [`nichols_step`]. Unit masses leave the matrix unchanged. The
/// result matches i-PI `nichols` on that spectrum.
pub fn nichols_displacement(
    hessian: ArrayView2<f64>,
    gradient: ArrayView1<f64>,
    masses: Option<ArrayView1<f64>>,
    trust_radius: f64,
    mode: NicholsMode,
) -> Result<Array1<f64>, SaddleError> {
    let n = gradient.len();
    if hessian.nrows() != n || hessian.ncols() != n {
        return Err(SaddleError::Shape(
            "Hessian must be square and match the gradient".into(),
        ));
    }
    let mass = masses_or_ones(n, masses)?;
    let weighted = mass_weight(hessian, &mass)?;
    let (evals, evecs) = sym_eigh(weighted.view())?;
    nichols_step(
        gradient,
        evals.view(),
        evecs.view(),
        Some(mass.view()),
        trust_radius,
        mode,
    )
}

pub(crate) fn mass_weight(
    hessian: ArrayView2<f64>,
    mass: &Array1<f64>,
) -> Result<Array2<f64>, SaddleError> {
    let n = mass.len();
    let mut weighted = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            if !v.is_finite() {
                return Err(SaddleError::NonFinite("Hessian"));
            }
            weighted[(i, j)] = v / (mass[i].sqrt() * mass[j].sqrt());
        }
    }
    Ok(weighted)
}

/// Scale `step` so its largest absolute component is `trust_radius`.
///
/// This is the cap `NicholsOptimizer` applies after `nichols`. A step
/// already inside the cap is unchanged.
pub fn cap_max_abs(step: &mut Array1<f64>, trust_radius: f64) -> Result<(), SaddleError> {
    if !(trust_radius.is_finite() && trust_radius > 0.0) {
        return Err(SaddleError::Invalid(
            "trust radius must be positive and finite".into(),
        ));
    }
    let mut peak = 0.0_f64;
    for v in step.iter() {
        if !v.is_finite() {
            return Err(SaddleError::NonFinite("step"));
        }
        peak = peak.max(v.abs());
    }
    if peak > trust_radius {
        let scale = trust_radius / peak;
        step.mapv_inplace(|v| v * scale);
    }
    Ok(())
}

fn prepare_update(
    h: &Array2<f64>,
    step: ArrayView1<f64>,
    dg: ArrayView1<f64>,
) -> Result<Option<(Array1<f64>, f64)>, SaddleError> {
    let n = step.len();
    if h.nrows() != n || h.ncols() != n || dg.len() != n || n == 0 {
        return Err(SaddleError::Shape(
            "Hessian update vectors must match the Hessian".into(),
        ));
    }
    if step.iter().any(|v| !v.is_finite())
        || dg.iter().any(|v| !v.is_finite())
        || h.iter().any(|v| !v.is_finite())
    {
        return Err(SaddleError::NonFinite("Hessian update"));
    }
    let ss = step.dot(&step);
    // A zero step has no secant pair. i-PI divides by d·d; skipping
    // leaves H unchanged, which is the pair's only finite value.
    if ss < 1e-30 {
        return Ok(None);
    }
    let mut xi = dg.to_owned();
    xi -= &h.dot(&step);
    Ok(Some((xi, ss)))
}

/// Powell symmetric Broyden update of a Cartesian energy Hessian.
///
/// `step` is the Cartesian displacement. `dg` is the change in the
/// energy gradient. i-PI passes the change in the force with the
/// opposite subtraction order, which is this same vector.
pub fn powell_update(
    h: &mut Array2<f64>,
    step: ArrayView1<f64>,
    dg: ArrayView1<f64>,
) -> Result<(), SaddleError> {
    let Some((xi, ss)) = prepare_update(h, step, dg)? else {
        return Ok(());
    };
    let ddi = 1.0 / ss;
    let xix = xi.dot(&step);
    let n = step.len();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += ddi * (xi[i] * step[j] + step[i] * xi[j] - xix * step[i] * step[j] * ddi);
        }
    }
    Ok(())
}

/// Bofill Hessian update: `(1 - phi)` Powell plus `phi` times the
/// symmetric rank-one term.
///
/// `phi = (s^T xi)^2 / ((s^T s)(xi^T xi))` with `xi = dg - H s`.
pub fn bofill_update(
    h: &mut Array2<f64>,
    step: ArrayView1<f64>,
    dg: ArrayView1<f64>,
) -> Result<(), SaddleError> {
    let Some((xi, ss)) = prepare_update(h, step, dg)? else {
        return Ok(());
    };
    let xixi = xi.dot(&xi);
    if xixi < 1e-30 {
        return Ok(());
    }
    let xix = xi.dot(&step);
    let phi = (xix * xix) / (ss * xixi);
    let ddi = 1.0 / ss;
    let powell_scale = (1.0 - phi) * ddi;
    let rank_ss = (1.0 - phi) * xix * ddi * ddi;
    let ms = xix / (ss * xixi);
    let n = step.len();
    for i in 0..n {
        for j in 0..n {
            h[(i, j)] += powell_scale * (xi[i] * step[j] + step[i] * xi[j])
                - rank_ss * step[i] * step[j]
                + ms * xi[i] * xi[j];
        }
    }
    Ok(())
}

fn finite_difference_hessian<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    dr: f64,
) -> Result<Array2<f64>, SaddleError> {
    if !(dr.is_finite() && dr > 0.0) {
        return Err(SaddleError::Invalid(
            "finite-difference step must be positive and finite".into(),
        ));
    }
    let n = x.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for j in 0..n {
        let mut xp = x.to_owned();
        let mut xm = x.to_owned();
        xp[j] += dr;
        xm[j] -= dr;
        let (_, gp) = surface.eval(xp.view())?;
        let (_, gm) = surface.eval(xm.view())?;
        if gp.len() != n || gm.len() != n {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        for i in 0..n {
            h[(i, j)] = (gp[i] - gm[i]) / (2.0 * dr);
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let v = 0.5 * (h[(i, j)] + h[(j, i)]);
            h[(i, j)] = v;
            h[(j, i)] = v;
        }
    }
    if h.iter().any(|v| !v.is_finite()) {
        return Err(SaddleError::NonFinite("finite-difference Hessian"));
    }
    Ok(h)
}

fn check_config(config: &Index1Config, n: usize) -> Result<(), SaddleError> {
    if !(config.trust_radius.is_finite() && config.trust_radius > 0.0) {
        return Err(SaddleError::Invalid(
            "trust radius must be positive and finite".into(),
        ));
    }
    if !(config.force_tol.is_finite() && config.force_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "force tolerance must be finite and non-negative".into(),
        ));
    }
    if !(config.fd_dr.is_finite() && config.fd_dr > 0.0) {
        return Err(SaddleError::Invalid(
            "finite-difference step must be positive and finite".into(),
        ));
    }
    if config.mode == NicholsMode::Index1 && n < 2 {
        return Err(SaddleError::Shape(
            "an index-1 search needs two or more coordinates".into(),
        ));
    }
    Ok(())
}

/// Index-1 (or minimizing) search on one geometry.
///
/// The host owns the loop, as with the band and the minimum-mode
/// session. Each [`Index1Session::step`] diagonalizes the current
/// Cartesian Hessian, takes a restricted-step partitioned RFO
/// displacement, and updates the Hessian with Powell or Bofill.
/// The trust radius shrinks when the rational-function model is a
/// poor prediction and grows when the step sat on the sphere and
/// the model agreed.
pub struct Index1Session {
    config: Index1Config,
    x: Array1<f64>,
    energy: f64,
    gradient: Array1<f64>,
    gradient_valid: bool,
    hessian: Array2<f64>,
    have_hessian: bool,
    masses: Array1<f64>,
    trust: f64,
    trust_max: f64,
    iteration: usize,
}

impl Index1Session {
    /// `hessian` is the Cartesian energy Hessian, or `None` to build
    /// one by central differences on the first step. `masses` is per
    /// coordinate; `None` is unit mass.
    pub fn new(
        config: Index1Config,
        x: Array1<f64>,
        hessian: Option<Array2<f64>>,
        masses: Option<Array1<f64>>,
    ) -> Result<Self, SaddleError> {
        if x.is_empty() {
            return Err(SaddleError::Shape("position must be nonempty".into()));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        check_config(&config, x.len())?;
        let masses = masses_or_ones(x.len(), masses.as_ref().map(|m| m.view()))?;
        let (hessian, have_hessian) = match hessian {
            None => (Array2::<f64>::zeros((x.len(), x.len())), false),
            Some(h) => {
                if h.nrows() != x.len() || h.ncols() != x.len() {
                    return Err(SaddleError::Shape("Hessian must match the position".into()));
                }
                if h.iter().any(|v| !v.is_finite()) {
                    return Err(SaddleError::NonFinite("Hessian"));
                }
                (h, true)
            }
        };
        let trust = config.trust_radius;
        Ok(Self {
            config,
            energy: 0.0,
            gradient: Array1::zeros(x.len()),
            gradient_valid: false,
            hessian,
            have_hessian,
            masses,
            trust,
            trust_max: (trust * 4.0).max(trust),
            iteration: 0,
            x,
        })
    }

    pub fn position(&self) -> ArrayView1<'_, f64> {
        self.x.view()
    }

    pub fn hessian(&self) -> Option<ArrayView2<'_, f64>> {
        if self.have_hessian {
            Some(self.hessian.view())
        } else {
            None
        }
    }

    pub fn set_hessian(&mut self, hessian: Array2<f64>) -> Result<(), SaddleError> {
        if hessian.nrows() != self.x.len() || hessian.ncols() != self.x.len() {
            return Err(SaddleError::Shape("Hessian must match the position".into()));
        }
        if hessian.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("Hessian"));
        }
        self.hessian = hessian;
        self.have_hessian = true;
        Ok(())
    }

    pub fn set_position(&mut self, x: Array1<f64>) -> Result<(), SaddleError> {
        if x.len() != self.x.len() {
            return Err(SaddleError::Shape("position length must stay fixed".into()));
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        self.x = x;
        self.gradient_valid = false;
        Ok(())
    }

    /// Drop the Hessian, the gradient cache, and the grown trust
    /// radius at a surface-epoch boundary.
    pub fn reset(&mut self) {
        self.have_hessian = false;
        self.gradient_valid = false;
        self.hessian.fill(0.0);
        self.gradient.fill(0.0);
        self.trust = self.config.trust_radius;
    }

    fn ensure_gradient<S: PointSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        if self.gradient_valid {
            return Ok(());
        }
        let (energy, g) = surface.eval(self.x.view())?;
        self.energy = energy;
        if g.len() != self.x.len() {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if g.iter().any(|v| !v.is_finite()) {
            return Err(SaddleError::NonFinite("gradient"));
        }
        self.gradient = g;
        self.gradient_valid = true;
        Ok(())
    }

    fn curvature(&self) -> Result<f64, SaddleError> {
        if !self.have_hessian {
            return Ok(0.0);
        }
        let weighted = mass_weight(self.hessian.view(), &self.masses)?;
        let (evals, _) = sym_eigh(weighted.view())?;
        Ok(evals[0])
    }

    fn adjust_trust(&mut self, actual: f64, predicted: f64, step_norm: f64) {
        if predicted.abs() <= 1e-12 {
            return;
        }
        let rho = actual / predicted;
        if !(rho.is_finite()) {
            return;
        }
        if rho < 0.25 {
            self.trust = (self.trust * 0.5).max(1e-4);
        } else if rho > 0.75 && step_norm >= 0.9 * self.trust {
            self.trust = (self.trust * 2.0).min(self.trust_max);
        }
    }

    /// One restricted-step partitioned RFO displacement and one Hessian update.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<Index1Report, SaddleError> {
        self.ensure_gradient(surface)?;
        let max_force = self.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if max_force <= self.config.force_tol {
            return Ok(Index1Report {
                status: Index1Status::Converged,
                max_force,
                curvature: self.curvature()?,
                iteration: self.iteration,
                max_step: 0.0,
            });
        }
        if !self.have_hessian {
            self.hessian = finite_difference_hessian(surface, self.x.view(), self.config.fd_dr)?;
            self.have_hessian = true;
        }
        let kind = match self.config.mode {
            NicholsMode::Index1 => crate::prfo_restricted::PrfoKind::Index1,
            NicholsMode::Minimize => crate::prfo_restricted::PrfoKind::Minimize,
        };
        let (dx, curvature) = crate::prfo_restricted::restricted_prfo_with_lowest(
            self.hessian.view(),
            self.gradient.view(),
            Some(self.masses.view()),
            self.trust,
            kind,
        )?;
        let max_step = dx.dot(&dx).sqrt();
        let hdx = self.hessian.dot(&dx);
        let quadratic = self.gradient.dot(&dx) + 0.5 * dx.dot(&hdx);
        let predicted = quadratic / (1.0 + dx.dot(&dx));
        let energy_old = self.energy;
        let x_new = &self.x + &dx;
        let (energy_new, g_new) = surface.eval(x_new.view())?;
        if g_new.len() != self.x.len() {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if g_new.iter().any(|v| !v.is_finite()) || !energy_new.is_finite() {
            return Err(SaddleError::NonFinite("gradient"));
        }
        self.adjust_trust(energy_new - energy_old, predicted, max_step);
        let dg = &g_new - &self.gradient;
        match self.config.update {
            HessianUpdate::Powell => powell_update(&mut self.hessian, dx.view(), dg.view())?,
            HessianUpdate::Bofill => bofill_update(&mut self.hessian, dx.view(), dg.view())?,
        }
        self.x = x_new;
        self.energy = energy_new;
        self.gradient = g_new;
        self.gradient_valid = true;
        self.iteration += 1;
        let max_force = self.gradient.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let status = if max_force <= self.config.force_tol {
            Index1Status::Converged
        } else {
            Index1Status::Running
        };
        Ok(Index1Report {
            status,
            max_force,
            curvature,
            iteration: self.iteration,
            max_step,
        })
    }

    /// Convenience loop over [`Index1Session::step`].
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<Index1Report, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == Index1Status::Running && self.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}
