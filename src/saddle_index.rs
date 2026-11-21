//! Saddle index of a converged point, and the escape from a saddle of
//! higher index along its extra negative modes.
//!
//! The searches checked here are the OCI-NEB refinement of the climbing
//! image, Goswami, Gunde, and Jónsson, Front. Chem. 14, 1807063 (2026),
//! doi:10.3389/fchem.2026.1807063, and the min-mode search, the dimer
//! that Goswami, Masterov, Kamath, Peña-Torres, and Jónsson, J. Chem.
//! Theory Comput. 21, 7935 (2025), doi:10.1021/acs.jctc.5c00866,
//! accelerate with a Gaussian process in eOn. Both stop where the force
//! vanishes and the curvature they follow is negative. A start with a
//! symmetry the surface shares can stop them on a saddle of higher
//! index, since a mode the symmetry keeps out of the force never moves
//! the point. Zhang and Du, SIAM J. Numer. Anal. 50, 1899 (2012),
//! doi:10.1137/110843149, Theorem 3.2, show that any critical point
//! other than an index-1 saddle is a fixed point of the Euler scheme of
//! their shrinking dimer dynamics, though not a contractive one. A
//! first-order saddle has exactly one negative Hessian eigenvalue; eOn
//! drops a process whose saddle Hessian, built on the moved atoms for
//! the prefactor, has any other count, and the index-1 saddle of Gould,
//! Ortner, and Packwood, Math. Comp. 85, 2939 (2016),
//! doi:10.1090/mcom/3096, is positive definite off its one negative
//! mode.
//!
//! [`saddle_index`] estimates the lowest curvatures on the free
//! coordinates by block Lanczos on finite-difference Hessian actions:
//! Rayleigh-Ritz on a Krylov space grown by the residuals of the lowest
//! Ritz pairs and reorthogonalized in full. The space starts from the
//! search's own mode and from pseudo-random vectors, so a mode that a
//! symmetric search never sees still enters it. An action costs one
//! gradient (forward against the known centre) or two (central), as a
//! step of the Lanczos estimate of Olsen, Kroes, Henkelman, Arnaldsson,
//! and Jónsson, J. Chem. Phys. 121, 9776 (2004), doi:10.1063/1.1809574,
//! or a dimer rotation with the extrapolated gradient of Heyden, Bell,
//! and Keil, J. Chem. Phys. 123, 224101 (2005), doi:10.1063/1.2104507,
//! and Kästner and Sherwood, J. Chem. Phys. 128, 014106 (2008),
//! doi:10.1063/1.2815812. The Ritz values are those of the symmetrized
//! operator on the space, so by Cauchy interlacing the k-th bounds that
//! operator's k-th eigenvalue from above and the count of negative Ritz
//! values never exceeds its index. A block of two seeds or more carries
//! a degenerate pair.
//!
//! The escape is the downward search of Yin, Wang, Chen, Zhang, and
//! Zhang, Phys. Rev. Lett. 124, 090601 (2020),
//! doi:10.1103/PhysRevLett.124.090601, which leaves an index-k saddle
//! along unstable eigenvectors for saddles of lower index. The negative
//! mode that overlaps the reference most is the reaction mode and stays:
//! the band tangent, as the OCI-NEB alignment reads it, or the search's
//! own mode. The point moves along every other negative mode, each to
//! its lower-energy side, and the index-1 search starts again there
//! seeded with the reaction mode. It maximizes along that mode and
//! minimizes along the rest, the high-index saddle dynamics of Yin,
//! Zhang, and Zhang, SIAM J. Sci. Comput. 41, A3576 (2019),
//! doi:10.1137/19M1253356, at index one.
//!
//! The solid-state dimer carries its own check on the generalized
//! coordinate of atoms and cell, [`crate::IndexCheck`].

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::{FiniteDifference, PointSurface};
use crate::stats::Rng;

/// A curvature strictly below the negation of this value, in energy per
/// length squared, counts as a negative mode.
///
/// [`SaddleIndexCheck::negative_tol`] defaults to this tolerance, and
/// [`crate::IndexKSession`] reports convergence only when every requested
/// curvature is below the same bound. A converged index-k search therefore
/// certifies as that index.
pub const NEGATIVE_CURVATURE_TOL: f64 = 1.0e-3;

pub(crate) fn counts_as_negative(curvature: f64) -> bool {
    curvature < -NEGATIVE_CURVATURE_TOL
}

/// Controls of the saddle-index certification.
///
/// The defaults resolve the four lowest curvatures, and the first
/// non-negative one past every negative curvature found, to a Ritz
/// residual of 1e-2 with blocks of two forward-difference actions of
/// displacement 1e-3, within 100 actions. A curvature under
/// [`NEGATIVE_CURVATURE_TOL`] counts as negative. Lengths are the host's.
///
/// Non-exhaustive: start from [`SaddleIndexCheck::default`] and chain
/// the `with_*` setters.
///
/// ```
/// use rgsaddle::{SaddleEscape, SaddleIndexCheck};
///
/// let check = SaddleIndexCheck::default()
///     .with_block(3)
///     .with_escape(SaddleEscape::default().with_push(0.05));
/// assert_eq!(check.block, 3);
/// assert_eq!(check.escape.map(|e| e.push), Some(0.05));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SaddleIndexCheck {
    /// Vectors that enter the Krylov space per expansion, the start
    /// vectors included. Two or more carry a degenerate pair.
    pub block: usize,
    /// Lowest curvatures resolved. The first non-negative curvature
    /// past every negative one is resolved as well.
    pub lowest: usize,
    /// Residual norm `|H u - θ u|` each resolved Ritz pair meets, in
    /// energy per length squared.
    pub residual_tol: f64,
    /// A curvature under `-negative_tol`, in energy per length squared,
    /// counts as negative.
    pub negative_tol: f64,
    /// Hessian actions one certification may spend.
    pub max_actions: usize,
    /// Finite-difference displacement of the actions.
    pub dr: f64,
    /// Forward differences cost one gradient per action with the centre
    /// known, central differences two.
    pub difference: FiniteDifference,
    /// Seed of the pseudo-random start vectors.
    pub seed: u64,
    /// Push off a saddle of index two or more and search again. `None`,
    /// the default, reports the index and stops there.
    pub escape: Option<SaddleEscape>,
}

with_setters!(SaddleIndexCheck {
    with_block => block: val usize,
    with_lowest => lowest: val usize,
    with_residual_tol => residual_tol: val f64,
    with_negative_tol => negative_tol: val f64,
    with_max_actions => max_actions: val usize,
    with_dr => dr: val f64,
    with_difference => difference: val FiniteDifference,
    with_seed => seed: val u64,
    with_escape => escape: opt SaddleEscape,
});

impl Default for SaddleIndexCheck {
    fn default() -> Self {
        Self {
            block: 2,
            lowest: 4,
            residual_tol: 1e-2,
            negative_tol: NEGATIVE_CURVATURE_TOL,
            max_actions: 100,
            dr: 1e-3,
            difference: FiniteDifference::Forward,
            seed: 0x5add_1e00_0000_0001,
            escape: None,
        }
    }
}

impl SaddleIndexCheck {
    pub(crate) fn validate(&self) -> Result<(), SaddleError> {
        let positive = |v: f64| v.is_finite() && v > 0.0;
        if !(self.block > 0
            && self.lowest > 0
            && self.max_actions >= self.block
            && positive(self.dr)
            && positive(self.residual_tol)
            && self.negative_tol.is_finite()
            && self.negative_tol >= 0.0)
        {
            return Err(SaddleError::Invalid(
                "saddle-index check needs a positive block, count, step and residual \
                 tolerance, an action budget of at least one block, and a finite \
                 non-negative curvature threshold"
                    .into(),
            ));
        }
        if let Some(escape) = &self.escape {
            escape.validate()?;
        }
        Ok(())
    }
}

/// Escape from a saddle of index two or more.
///
/// The point moves `push` (largest per-atom displacement, in the host's
/// length unit) along the extra negative modes, each to its
/// lower-energy side, and the index-1 search starts again there. The
/// second attempt pushes twice as far, the third four times, and so
/// on. The defaults push 0.1 and allow two attempts.
///
/// Non-exhaustive: start from [`SaddleEscape::default`] and chain the
/// `with_*` setters.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct SaddleEscape {
    /// Largest per-atom displacement of the first push.
    pub push: f64,
    /// Pushes before a saddle of higher index is reported as it is.
    pub max_attempts: usize,
}

with_setters!(SaddleEscape {
    with_push => push: val f64,
    with_max_attempts => max_attempts: val usize,
});

impl Default for SaddleEscape {
    fn default() -> Self {
        Self {
            push: 0.1,
            max_attempts: 2,
        }
    }
}

impl SaddleEscape {
    fn validate(&self) -> Result<(), SaddleError> {
        if !(self.push.is_finite() && self.push > 0.0 && self.max_attempts > 0) {
            return Err(SaddleError::Invalid(
                "saddle escape needs a positive finite push and at least one attempt".into(),
            ));
        }
        Ok(())
    }

    /// Push length of attempt `attempt`, counted from zero.
    pub(crate) fn push_at(&self, attempt: usize) -> f64 {
        self.push * 2f64.powi(attempt.min(1000) as i32)
    }
}

/// The certified lowest curvatures of a converged point.
#[derive(Clone, Debug, PartialEq)]
pub struct SaddleIndex {
    /// Curvatures under `-negative_tol`: the saddle index, 1 for a
    /// first-order saddle.
    pub index: usize,
    /// Lowest Ritz values in ascending order, energy per length
    /// squared: every negative one and the first non-negative one, at
    /// least [`SaddleIndexCheck::lowest`] when the free space holds them.
    pub curvatures: Vec<f64>,
    /// Residual norms `|H u - θ u|` of those Ritz pairs.
    pub residuals: Vec<f64>,
    /// Every listed pair met the residual tolerance, or the Krylov space
    /// filled the free space. False when the action budget ran out
    /// first; the index then still bounds that of the symmetrized
    /// finite-difference Hessian from below.
    pub converged: bool,
    /// Surface evaluations of this certification.
    pub evaluations: usize,
}

/// Where one certification runs.
pub(crate) struct Probe<'a> {
    pub(crate) x: ArrayView1<'a, f64>,
    pub(crate) gradient: ArrayView1<'a, f64>,
    /// One entry per coordinate, true where the host holds it; empty
    /// for none.
    pub(crate) held: Vec<bool>,
    /// Directions off the free space, of any length.
    pub(crate) excluded: Vec<Array1<f64>>,
    /// Start vectors ahead of the pseudo-random ones.
    pub(crate) hints: Vec<Array1<f64>>,
}

/// A certificate and the Ritz vectors of its curvatures.
#[derive(Clone, Debug)]
pub(crate) struct Certificate {
    pub(crate) index: SaddleIndex,
    pub(crate) vectors: Vec<Array1<f64>>,
}

/// The coordinates not held, orthogonal to the excluded directions.
struct FreeSpace {
    held: Vec<bool>,
    basis: Vec<Array1<f64>>,
}

impl FreeSpace {
    fn new(held: Vec<bool>, excluded: Vec<Array1<f64>>) -> Self {
        let mut space = Self {
            held,
            basis: Vec::with_capacity(excluded.len()),
        };
        for mut v in excluded {
            space.zero_held(&mut v);
            if orthonormalize(&mut v, &space.basis) {
                space.basis.push(v);
            }
        }
        space
    }

    fn zero_held(&self, v: &mut Array1<f64>) {
        for (c, &held) in v.iter_mut().zip(&self.held) {
            if held {
                *c = 0.0;
            }
        }
    }

    fn project(&self, v: &mut Array1<f64>) {
        self.zero_held(v);
        for _ in 0..2 {
            for q in &self.basis {
                let c = q.dot(v);
                v.scaled_add(-c, q);
            }
        }
    }

    fn dimension(&self) -> usize {
        let free = self.held.iter().filter(|&&h| !h).count();
        free.saturating_sub(self.basis.len())
    }
}

/// Orthonormalize `v` against `basis` in two passes. False when what
/// is left is under 1e-8 of where it started.
fn orthonormalize(v: &mut Array1<f64>, basis: &[Array1<f64>]) -> bool {
    let start = v.dot(v).sqrt();
    if !(start.is_finite() && start > 0.0) {
        return false;
    }
    for _ in 0..2 {
        for b in basis {
            let c = b.dot(v);
            v.scaled_add(-c, b);
        }
    }
    let left = v.dot(v).sqrt();
    if left <= 1e-8 * start {
        return false;
    }
    *v /= left;
    true
}

/// Largest norm over consecutive coordinate triples; a trailing pair
/// or single coordinate counts as one more group.
pub(crate) fn largest_atom(v: ArrayView1<f64>) -> f64 {
    let n = v.len();
    (0..n)
        .step_by(3)
        .map(|i| (i..n.min(i + 3)).map(|k| v[k] * v[k]).sum::<f64>().sqrt())
        .fold(0.0, f64::max)
}

fn checked(
    gradient: Array1<f64>,
    n: usize,
    what: &'static str,
) -> Result<Array1<f64>, SaddleError> {
    if gradient.len() != n {
        return Err(SaddleError::Shape(format!(
            "{what} has {} entries for {n} coordinates",
            gradient.len()
        )));
    }
    if !gradient.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite(what));
    }
    Ok(gradient)
}

/// `H v` at `x`: the surface's own action when it has one, else a
/// finite difference of gradients that adds to `calls`.
fn action<S: PointSurface>(
    check: &SaddleIndexCheck,
    surface: &S,
    x: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    v: &Array1<f64>,
    calls: &mut usize,
) -> Result<Array1<f64>, SaddleError> {
    let n = x.len();
    if let Some(hv) = surface.hessian_vector(x, v.view())? {
        return checked(hv, n, "surface Hessian action");
    }
    let dr = check.dr;
    let plus = &x + &(v * dr);
    *calls += 1;
    let forward = checked(surface.eval(plus.view())?.1, n, "certification gradient")?;
    Ok(match check.difference {
        FiniteDifference::Forward => (forward - gradient) / dr,
        FiniteDifference::Central => {
            let minus = &x - &(v * dr);
            *calls += 1;
            let backward = checked(surface.eval(minus.view())?.1, n, "certification gradient")?;
            (forward - backward) / (2.0 * dr)
        }
    })
}

/// Spectrum of the symmetric projected matrix, ascending. It is solved
/// once per expansion and grows to `max_actions`, where the dense
/// Jacobi of [`crate::eigensolve::exact_eigh`] costs more than the
/// actions of a cheap surface, so faer solves it when the feature is
/// on.
fn ritz(t: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), SaddleError> {
    #[cfg(feature = "faer")]
    {
        crate::eigensolve::faer_eigh(t)
    }
    #[cfg(not(feature = "faer"))]
    {
        crate::eigensolve::exact_eigh(t.view())
    }
}

/// Block Lanczos on the free coordinates of `probe`.
pub(crate) fn certify<S: PointSurface>(
    check: &SaddleIndexCheck,
    surface: &S,
    probe: Probe<'_>,
) -> Result<Certificate, SaddleError> {
    let n = probe.x.len();
    if probe.gradient.len() != n
        || (!probe.held.is_empty() && probe.held.len() != n)
        || probe.excluded.iter().any(|v| v.len() != n)
        || probe.hints.iter().any(|v| v.len() != n)
    {
        return Err(SaddleError::Shape(
            "saddle-index probe vectors must match the position".into(),
        ));
    }
    let mut held = if probe.held.is_empty() {
        vec![false; n]
    } else {
        probe.held
    };
    let mut rng = Rng::new(check.seed);
    let mut calls = 0;
    // A host that zeroes the gradient of atoms it fixes, without saying
    // so, leaves those coordinates exactly zero in the gradient and in
    // every action, while the symmetrized projection would still couple
    // them to the free block. One forward difference along those
    // coordinates alone tells them from free ones that are zero here.
    let loaded = probe
        .gradient
        .iter()
        .zip(&held)
        .any(|(g, &h)| !h && *g != 0.0);
    if loaded {
        let candidates: Vec<usize> = (0..n)
            .filter(|&i| !held[i] && probe.gradient[i] == 0.0)
            .collect();
        if !candidates.is_empty() {
            let mut v = Array1::zeros(n);
            for &i in &candidates {
                v[i] = 2.0 * rng.uniform() - 1.0;
            }
            let forward = SaddleIndexCheck {
                difference: FiniteDifference::Forward,
                ..*check
            };
            let a = action(&forward, surface, probe.x, probe.gradient, &v, &mut calls)?;
            for &i in &candidates {
                held[i] = a[i] == 0.0;
            }
        }
    }
    let space = FreeSpace::new(held, probe.excluded);
    let mut spent = 0;
    let mut basis: Vec<Array1<f64>> = Vec::new();
    let mut images: Vec<Array1<f64>> = Vec::new();
    let mut fresh: Vec<Array1<f64>> = probe.hints.into_iter().take(check.block).collect();
    while fresh.len() < check.block {
        fresh.push((0..n).map(|_| 2.0 * rng.uniform() - 1.0).collect());
    }
    loop {
        let before = basis.len();
        for mut v in fresh.drain(..) {
            if spent >= check.max_actions {
                break;
            }
            space.project(&mut v);
            if !orthonormalize(&mut v, &basis) {
                continue;
            }
            let a = action(check, surface, probe.x, probe.gradient, &v, &mut calls)?;
            spent += 1;
            basis.push(v);
            images.push(a);
        }
        let added = basis.len() - before;
        for a in &mut images[before..] {
            space.project(a);
        }
        let m = basis.len();
        if m == 0 {
            return Ok(Certificate {
                index: SaddleIndex {
                    index: 0,
                    curvatures: Vec::new(),
                    residuals: Vec::new(),
                    converged: true,
                    evaluations: calls,
                },
                vectors: Vec::new(),
            });
        }
        let mut t = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..=i {
                let value = 0.5 * (basis[i].dot(&images[j]) + basis[j].dot(&images[i]));
                t[(i, j)] = value;
                t[(j, i)] = value;
            }
        }
        let (theta, y) = ritz(&t)?;
        let negative = theta.iter().filter(|&&v| v < -check.negative_tol).count();
        let target = check.lowest.max(negative + 1);
        let want = m.min(target);
        let keep = m.min(want + check.block);
        let mut vectors = Vec::with_capacity(keep);
        let mut residuals = Vec::with_capacity(keep);
        let mut residual_vectors = Vec::with_capacity(keep);
        for c in 0..keep {
            let mut u = Array1::zeros(n);
            let mut au = Array1::zeros(n);
            for k in 0..m {
                u.scaled_add(y[(k, c)], &basis[k]);
                au.scaled_add(y[(k, c)], &images[k]);
            }
            let r = au - &(&u * theta[c]);
            residuals.push(r.dot(&r).sqrt());
            vectors.push(u);
            residual_vectors.push(r);
        }
        let met = m >= target && residuals[..want].iter().all(|r| *r <= check.residual_tol);
        let full = m >= space.dimension();
        if met || full || added == 0 || spent >= check.max_actions {
            vectors.truncate(want);
            residuals.truncate(want);
            return Ok(Certificate {
                index: SaddleIndex {
                    index: negative,
                    curvatures: theta.iter().take(want).copied().collect(),
                    residuals,
                    converged: met || full,
                    evaluations: calls,
                },
                vectors,
            });
        }
        let mut order: Vec<usize> = (0..want)
            .filter(|&c| residuals[c] > check.residual_tol)
            .collect();
        order.extend(want..keep);
        fresh = order
            .into_iter()
            .take(check.block)
            .map(|c| residual_vectors[c].clone())
            .collect();
        while fresh.len() < check.block {
            fresh.push((0..n).map(|_| 2.0 * rng.uniform() - 1.0).collect());
        }
    }
}

/// Where the escape starts the index-1 search again.
pub(crate) struct Push {
    pub(crate) start: Array1<f64>,
    pub(crate) reaction: Array1<f64>,
    pub(crate) evaluations: usize,
}

/// Move `x` by `push` along the negative modes of `certificate` other
/// than the one closest to `reference`, each to its lower-energy side.
/// `None` below index two.
pub(crate) fn escape_push<S: PointSurface>(
    surface: &S,
    x: ArrayView1<f64>,
    certificate: &Certificate,
    reference: ArrayView1<f64>,
    push: f64,
) -> Result<Option<Push>, SaddleError> {
    let k = certificate.index.index.min(certificate.vectors.len());
    if k < 2 {
        return Ok(None);
    }
    let negative = &certificate.vectors[..k];
    let mut reaction = 0;
    let mut best = f64::NEG_INFINITY;
    for (j, u) in negative.iter().enumerate() {
        let overlap = u.dot(&reference).abs();
        if overlap > best {
            best = overlap;
            reaction = j;
        }
    }
    let mut direction = Array1::zeros(x.len());
    let mut evaluations = 0;
    for (j, u) in negative.iter().enumerate() {
        let scale = largest_atom(u.view());
        if j == reaction || scale.is_nan() || scale <= 0.0 {
            continue;
        }
        let step = u * (push / scale);
        let (up, _) = surface.eval((&x + &step).view())?;
        let (down, _) = surface.eval((&x - &step).view())?;
        evaluations += 2;
        if !(up.is_finite() && down.is_finite()) {
            return Err(SaddleError::NonFinite("escape probe energy"));
        }
        direction.scaled_add(if down < up { -1.0 } else { 1.0 }, u);
    }
    let length = largest_atom(direction.view());
    if length.is_nan() || length <= 0.0 {
        return Ok(None);
    }
    direction *= push / length;
    Ok(Some(Push {
        start: &x + &direction,
        reaction: negative[reaction].clone(),
        evaluations,
    }))
}

/// Certify the saddle index of `x`: the count of curvatures under
/// `-check.negative_tol` on the free coordinates.
///
/// `gradient` is the energy gradient at `x` when the host has it; one
/// surface evaluation finds it otherwise. `mode`, the search's own mode
/// when there is one, joins the pseudo-random start vectors. `fixed`
/// holds one entry per atom (`x` is `3N` long), true for an atom that
/// does not move, or is empty. The directions of
/// [`PointSurface::excluded_modes`] are left out as well, and so is a
/// coordinate that is exactly zero in the gradient and in every action
/// of the first block, as a host that zeroes the force on its fixed
/// atoms leaves it. A surface with [`PointSurface::hessian_vector`]
/// supplies the actions itself. The escape of `check` is not read.
///
/// `x` is the flat `3N` position in Angstrom, `gradient` is in eV/Angstrom,
/// and the curvatures returned are in eV/Angstrom^2.
///
/// # Errors
///
/// [`SaddleError::Invalid`] when `check` fails its validation (a zero block or
/// count, a non-positive or non-finite step or residual tolerance, a budget
/// below one block, or a negative or non-finite curvature threshold) or `x` is
/// empty or not finite; [`SaddleError::Shape`] when `fixed` is not empty and
/// does not hold `x.len() / 3` entries, or `mode`, a supplied gradient or an
/// excluded mode differs in length from `x`; [`SaddleError::NonFinite`] for a
/// non-finite gradient or action; and the errors of `surface`.
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{PointSurface, SaddleError, SaddleIndexCheck, saddle_index};
///
/// /// Two double wells and a harmonic z: a maximum in the x-y plane
/// /// at the origin, index 2.
/// struct Wells;
/// impl PointSurface for Wells {
///     fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (a, b, c) = (x[0], x[1], x[2]);
///         let e = (a.powi(4) - 2.0 * a * a) / 4.0 + (b.powi(4) - b * b) / 4.0 + c * c / 2.0;
///         Ok((e, array![a.powi(3) - a, b.powi(3) - 0.5 * b, c]))
///     }
/// }
///
/// let check = SaddleIndexCheck::default();
/// let top = saddle_index(&check, &Wells, array![0.0, 0.0, 0.0].view(), None, None, &[])?;
/// assert_eq!(top.index, 2);
/// assert!((top.curvatures[0] + 1.0).abs() < 1e-2);
/// assert!((top.curvatures[1] + 0.5).abs() < 1e-2);
/// let side = saddle_index(&check, &Wells, array![0.0, 0.5f64.sqrt(), 0.0].view(), None, None, &[])?;
/// assert_eq!(side.index, 1);
/// # Ok::<(), SaddleError>(())
/// ```
pub fn saddle_index<S: PointSurface>(
    check: &SaddleIndexCheck,
    surface: &S,
    x: ArrayView1<f64>,
    gradient: Option<ArrayView1<f64>>,
    mode: Option<ArrayView1<f64>>,
    fixed: &[bool],
) -> Result<SaddleIndex, SaddleError> {
    check.validate()?;
    let n = x.len();
    if n == 0 || !x.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Invalid(
            "saddle index needs a nonempty finite position".into(),
        ));
    }
    if !fixed.is_empty() && 3 * fixed.len() != n {
        return Err(SaddleError::Shape(format!(
            "fixed-atom mask has {} entries for {n} coordinates",
            fixed.len()
        )));
    }
    if mode.is_some_and(|m| m.len() != n) {
        return Err(SaddleError::Shape("mode must match the position".into()));
    }
    let held: Vec<bool> = if fixed.iter().any(|&f| f) {
        (0..n).map(|k| fixed[k / 3]).collect()
    } else {
        Vec::new()
    };
    let mut calls = 0;
    let mut gradient = match gradient {
        Some(g) => checked(g.to_owned(), n, "gradient")?,
        None => {
            calls += 1;
            checked(surface.eval(x)?.1, n, "gradient")?
        }
    };
    for (g, &h) in gradient.iter_mut().zip(&held) {
        if h {
            *g = 0.0;
        }
    }
    let rows = surface.excluded_modes(x)?;
    if rows.nrows() > 0 && rows.ncols() != n {
        return Err(SaddleError::Shape(
            "excluded modes must match the position".into(),
        ));
    }
    let excluded = rows.outer_iter().map(|r| r.to_owned()).collect();
    let hints = mode.map(|m| vec![m.to_owned()]).unwrap_or_default();
    let mut certificate = certify(
        check,
        surface,
        Probe {
            x,
            gradient: gradient.view(),
            held,
            excluded,
            hints,
        },
    )?;
    certificate.index.evaluations += calls;
    Ok(certificate.index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// `(a^4 - 2a^2)/4 + (b^4 - 2 c b^2)/4 + z^2/2`: index 2 at the
    /// origin, index 1 at `(0, ±sqrt c, 0)` and `(±1, 0, 0)`.
    struct Wells {
        c: f64,
        calls: AtomicUsize,
    }

    impl PointSurface for Wells {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let (a, b, z) = (x[0], x[1], x[2]);
            let c = self.c;
            let e =
                (a.powi(4) - 2.0 * a * a) / 4.0 + (b.powi(4) - 2.0 * c * b * b) / 4.0 + z * z / 2.0;
            Ok((e, array![a.powi(3) - a, b.powi(3) - c * b, z]))
        }
    }

    fn wells() -> Wells {
        Wells {
            c: 0.5,
            calls: AtomicUsize::new(0),
        }
    }

    #[test]
    fn the_double_well_top_is_index_two_and_its_sides_index_one() {
        let surface = wells();
        let check = SaddleIndexCheck::default();
        let cases = [
            (array![0.0, 0.0, 0.0], 2, vec![-1.0, -0.5, 1.0]),
            (array![0.0, 0.5f64.sqrt(), 0.0], 1, vec![-1.0, 1.0, 1.0]),
            (array![1.0, 0.0, 0.0], 1, vec![-0.5, 1.0, 2.0]),
            (array![1.0, 0.5f64.sqrt(), 0.0], 0, vec![1.0, 1.0, 2.0]),
        ];
        for (x, index, curvatures) in cases {
            surface.calls.store(0, Ordering::Relaxed);
            let found = saddle_index(&check, &surface, x.view(), None, None, &[]).unwrap();
            assert_eq!(found.index, index, "{x}: {found:?}");
            assert!(found.converged);
            assert_eq!(found.curvatures.len(), 3);
            for (got, want) in found.curvatures.iter().zip(&curvatures) {
                assert!((got - want).abs() < 5e-3, "{x}: {got} against {want}");
            }
            assert_eq!(found.evaluations, surface.calls.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn a_symmetric_mode_alone_still_finds_the_hidden_negative_mode() {
        let surface = wells();
        let check = SaddleIndexCheck::default().with_block(1);
        let mode = array![1.0, 0.0, 0.0];
        let found = saddle_index(
            &check,
            &surface,
            array![0.0, 0.0, 0.0].view(),
            None,
            Some(mode.view()),
            &[],
        )
        .unwrap();
        assert_eq!(found.index, 2, "{found:?}");
    }

    #[test]
    fn fixed_coordinates_and_host_zeroed_ones_are_left_out() {
        // Six coordinates: the wells on the first three and a second
        // copy of a, coupled to the first, that the host fixes.
        struct Coupled {
            zero_fixed: bool,
        }
        impl PointSurface for Coupled {
            fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                let (a, b, z, p) = (x[0], x[1], x[2], x[3]);
                let e = (a.powi(4) - 2.0 * a * a) / 4.0
                    + (b.powi(4) - b * b) / 4.0
                    + z * z / 2.0
                    + 4.0 * (a - p).powi(2)
                    + 2.0 * (x[4] * x[4] + x[5] * x[5]);
                let mut g = array![
                    a.powi(3) - a + 8.0 * (a - p),
                    b.powi(3) - 0.5 * b,
                    z,
                    -8.0 * (a - p),
                    4.0 * x[4],
                    4.0 * x[5]
                ];
                if self.zero_fixed {
                    g[3] = 0.0;
                    g[4] = 0.0;
                    g[5] = 0.0;
                }
                Ok((e, g))
            }
        }
        let check = SaddleIndexCheck::default();
        let x = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        // With the second atom held, a sees 3 a^2 - 1 + 8 = 7 at the
        // origin and only b is negative.
        let masked = saddle_index(
            &check,
            &Coupled { zero_fixed: false },
            x.view(),
            None,
            None,
            &[false, true],
        )
        .unwrap();
        assert_eq!(masked.index, 1, "{masked:?}");
        assert!((masked.curvatures[0] + 0.5).abs() < 5e-3);
        // Off the origin the gradient is loaded, so the host's zeros
        // read as held coordinates without a mask.
        let off = array![0.01, 0.02, 0.01, 0.0, 0.0, 0.0];
        let zeroed = saddle_index(
            &check,
            &Coupled { zero_fixed: true },
            off.view(),
            None,
            None,
            &[],
        )
        .unwrap();
        let masked_off = saddle_index(
            &check,
            &Coupled { zero_fixed: false },
            off.view(),
            None,
            None,
            &[false, true],
        )
        .unwrap();
        assert_eq!(zeroed.index, 1, "{zeroed:?}");
        assert_eq!(zeroed.index, masked_off.index);
        for (a, b) in zeroed.curvatures.iter().zip(&masked_off.curvatures) {
            assert!((a - b).abs() < 5e-3, "{a} against {b}");
        }
    }

    #[test]
    fn an_exhausted_budget_is_not_converged() {
        let surface = wells();
        let check = SaddleIndexCheck::default()
            .with_block(1)
            .with_max_actions(1)
            .with_residual_tol(1e-12);
        let found = saddle_index(
            &check,
            &surface,
            array![0.3, 0.1, 0.2].view(),
            None,
            None,
            &[],
        )
        .unwrap();
        assert!(!found.converged);
        assert_eq!(found.curvatures.len(), 1);
    }

    #[test]
    fn the_escape_push_keeps_the_mode_closest_to_the_reference() {
        let surface = wells();
        let check = SaddleIndexCheck::default();
        let x = array![0.0, 0.0, 0.0];
        let g = surface.eval(x.view()).unwrap().1;
        let certificate = certify(
            &check,
            &surface,
            Probe {
                x: x.view(),
                gradient: g.view(),
                held: Vec::new(),
                excluded: Vec::new(),
                hints: Vec::new(),
            },
        )
        .unwrap();
        assert_eq!(certificate.index.index, 2);
        let push = escape_push(
            &surface,
            x.view(),
            &certificate,
            array![1.0, 0.0, 0.0].view(),
            0.1,
        )
        .unwrap()
        .expect("index 2 pushes");
        assert!(push.reaction[0].abs() > 0.99, "{}", push.reaction);
        assert!((push.start[1].abs() - 0.1).abs() < 1e-4, "{}", push.start);
        assert!(push.start[0].abs() < 1e-4 && push.start[2].abs() < 1e-4);
        assert_eq!(push.evaluations, 2);
    }

    #[test]
    fn invalid_checks_are_refused() {
        let surface = wells();
        let x = array![0.0, 0.0, 0.0];
        for check in [
            SaddleIndexCheck::default().with_block(0),
            SaddleIndexCheck::default().with_lowest(0),
            SaddleIndexCheck::default().with_max_actions(1),
            SaddleIndexCheck::default().with_dr(0.0),
            SaddleIndexCheck::default().with_residual_tol(f64::NAN),
            SaddleIndexCheck::default().with_negative_tol(-1.0),
            SaddleIndexCheck::default().with_escape(SaddleEscape::default().with_push(0.0)),
            SaddleIndexCheck::default().with_escape(SaddleEscape::default().with_max_attempts(0)),
        ] {
            assert!(matches!(
                saddle_index(&check, &surface, x.view(), None, None, &[]),
                Err(SaddleError::Invalid(_))
            ));
        }
        assert!(matches!(
            saddle_index(
                &SaddleIndexCheck::default(),
                &surface,
                x.view(),
                None,
                None,
                &[true, false]
            ),
            Err(SaddleError::Shape(_))
        ));
    }
}
