//! Index-k minimum-mode step.
//!
//! The one-mode search reflects the force along a single direction,
//! `F - 2 (F · v) v`, once that curvature is negative, and climbs the
//! mode alone while it is not (Henkelman and Jónsson, J. Chem. Phys.
//! 111, 7010 (1999), doi:10.1063/1.480097; Goswami, Gunde, and Jónsson,
//! Front. Chem. 14, 1807063 (2026), doi:10.3389/fchem.2026.1807063,
//! Equation 3). An index-k saddle is a maximum on a k-dimensional
//! subspace and a minimum on its complement. The step reflects in that
//! whole subspace,
//! `F_t = F - 2 Σ_i v_i (v_i · F)`,
//! and climbs inside the subspace while any of the k curvatures is
//! still non-negative. That is the same rule, applied to the block:
//! the orthogonal complement is kept only after every sought curvature
//! is negative.
//!
//! The k lowest pairs come from a local block iteration behind
//! `LowestEigenpairs`. LOBPCG is Knyazev's locally
//! optimal block preconditioned conjugate gradient (SIAM J. Sci.
//! Comput. 23, 517 (2001), doi:10.1137/S1064827500366124), warm-started
//! with the previous block and its conjugate direction.
//! Block Lanczos is a reorthogonalized block Krylov iteration
//! (Rayleigh–Ritz on the block Krylov space). Either can later be
//! replaced by rgmin's `lowest_eigenpairs` without changing the step.
//! For `k = 1` the step is the existing dimer or Lanczos translation,
//! so the position, the mode, and the translated force match
//! [`crate::minmode::MinModeSession::step`] bit for bit.
//!
//! Gould, Ortner, and Packwood, Math. Comp. 85, 2939 (2016),
//! doi:10.1090/mcom/3096, analyse the dimer with preconditioning and a
//! line search. The third author is Packwood. The index-k dimer those
//! authors' shrinking-length analysis rests on is Zhang and Du, SIAM J.
//! Numer. Anal. 50, 1899 (2012), doi:10.1137/110843149, and the
//! high-index generalisation is Yin, Zhang, and Zhang, SIAM J. Sci.
//! Comput. 41, A3576 (2019), doi:10.1137/19M1253356. A Gaussian-process
//! surrogate supplies the Hessian action the way Goswami, Masterov,
//! Kamath, Peña-Torres, and Jónsson, J. Chem. Theory Comput. 21, 7935
//! (2025), doi:10.1021/acs.jctc.5c00866, do: implement
//! [`PointSurface::hessian_vector`], or wrap a [`HessianProduct`] in
//! [`WithHessian`].

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip, s};
use rgmin::Oracle;

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::force::ForceGate;
use crate::linalg::modified_gram_schmidt;
use crate::minmode::{
    FiniteDifference, MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface,
    build_solver, translated_gradient,
};

/// Which block iteration finds the k lowest pairs when `k > 1`.
///
/// `k = 1` does not use this: the step is the dimer or Lanczos
/// translation selected by [`MinModeConfig::kind`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlockEigenKind {
    /// Knyazev LOBPCG, with the previous conjugate direction kept.
    #[default]
    Lobpcg,
    /// Reorthogonalized block Lanczos.
    BlockLanczos,
}

impl BlockEigenKind {
    /// Discriminant used by the C configuration.
    ///
    /// `0` and `1` are [`MinModeKind`] dimer and Lanczos. `2` asks for
    /// LOBPCG even when the one-mode kind would have been the dimer.
    pub fn to_abi(self) -> i32 {
        match self {
            Self::Lobpcg => 2,
            Self::BlockLanczos => 1,
        }
    }
}

/// Hessian-vector product at a geometry.
///
/// A Gaussian-process surrogate implements this with its mean Hessian.
/// [`PointSurface::hessian_vector`] is the same callback; [`WithHessian`]
/// adapts a product that is not already a surface.
pub trait HessianProduct: Sync {
    /// Hessian action `H(x) v`.
    fn product(
        &self,
        x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Array1<f64>, SaddleError>;
}

/// A [`PointSurface`] whose Hessian action is [`HessianProduct`].
pub struct WithHessian<'a, S, H> {
    /// Energy and gradient.
    pub surface: &'a S,
    /// Hessian-vector product used instead of a finite difference.
    pub hessian: &'a H,
}

impl<S: PointSurface, H: HessianProduct> PointSurface for WithHessian<'_, S, H> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.surface.eval(x)
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        vector: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        Ok(Some(self.hessian.product(x, vector)?))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        self.surface.excluded_modes(x)
    }
}

/// Options for [`indexk_step`] and [`IndexKSession`].
///
/// `search` is the one-mode configuration: kind, finite-difference
/// width, translator, force tolerance, and per-atom cap. `k = 1` uses
/// it unchanged.
#[derive(Clone, Debug)]
pub struct IndexKConfig {
    /// Number of lowest modes. At least 1 and at most the dimension.
    pub k: usize,
    /// One-mode estimator, finite difference, and translator.
    pub search: MinModeConfig,
    /// Residual at which the block iteration stops. The default is
    /// [`MinModeConfig::rotation_tol`].
    pub residual_tol: f64,
    /// Cap on block iterations when `k > 1`.
    pub max_block_iterations: usize,
    /// Block solver used when `k > 1`.
    pub block: BlockEigenKind,
}

impl IndexKConfig {
    /// `k` together with a one-mode configuration.
    ///
    /// Lanczos selects block Lanczos. The dimer selects LOBPCG, which
    /// is the block rotation a dimer does not have.
    pub fn new(k: usize, search: MinModeConfig) -> Self {
        let residual_tol = search.rotation_tol;
        let block = match search.kind {
            MinModeKind::Lanczos => BlockEigenKind::BlockLanczos,
            MinModeKind::Dimer => BlockEigenKind::Lobpcg,
        };
        Self {
            k,
            search,
            residual_tol,
            max_block_iterations: 30,
            block,
        }
    }
}

impl Default for IndexKConfig {
    fn default() -> Self {
        Self::new(1, MinModeConfig::default())
    }
}

/// One index-k translation.
#[derive(Clone, Debug)]
pub struct IndexKReport {
    /// Running until every sought curvature is negative and the force is under tolerance.
    pub status: MinModeStatus,
    /// Geometry after the step.
    pub position: Array1<f64>,
    /// Force norm at `position`, in the session's gate.
    pub max_force: f64,
    /// Orthonormal modes, shape `(n, k)`, one column per mode, lowest
    /// curvature first.
    pub modes: Array2<f64>,
    /// Curvature of each column of `modes`, lowest first.
    pub curvatures: Array1<f64>,
    /// Effective gradient the step followed.
    pub translated_force: Array1<f64>,
    /// Dimer rotations, Lanczos actions, or block Hessian products.
    pub rotations: usize,
    /// Steps taken by this session, including this one when it translated.
    pub iteration: usize,
    /// Surface evaluations during this call, the centre included.
    pub evaluations: usize,
}

/// Householder reflection of `force` through the columns of `modes`.
///
/// `modes` has shape `(n, k)` and is orthonormal. The result is
/// `F - 2 Σ_i v_i (v_i · F)`. For `k = 1` this is
/// [`crate::stencil::reflect`], the same subtraction as
/// `force - v * dot(force, v) * 2`. `out` is a new vector.
pub fn reflect_force(
    force: ArrayView1<f64>,
    modes: ArrayView2<f64>,
) -> Result<Array1<f64>, SaddleError> {
    check_modes(force, modes)?;
    if modes.ncols() == 1 {
        let axis = modes.column(0).to_owned();
        return Ok(crate::stencil::reflect(force, axis.view()));
    }
    let mut out = force.to_owned();
    for column in 0..modes.ncols() {
        let axis = modes.column(column);
        let twice = 2.0 * force.dot(&axis);
        Zip::from(&mut out)
            .and(axis)
            .for_each(|slot, &component| *slot -= component * twice);
    }
    Ok(out)
}

/// Translated force of an index-k step.
///
/// Every curvature negative: [`reflect_force`]. Any curvature
/// non-negative: the component outside the block is dropped and each
/// column is reversed, `F_t = -Σ_i v_i (v_i · F)`. For `k = 1` this is
/// the one-mode translated gradient.
pub fn translated_force(
    force: ArrayView1<f64>,
    modes: ArrayView2<f64>,
    curvatures: ArrayView1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    check_modes(force, modes)?;
    if curvatures.len() != modes.ncols() {
        return Err(SaddleError::Shape(
            "curvatures must have one value per mode".into(),
        ));
    }
    if !curvatures.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("index-k curvature"));
    }
    if modes.ncols() == 1 {
        let gradient = force.to_owned();
        let mode = modes.column(0).to_owned();
        return Ok(translated_gradient(&gradient, &mode, curvatures[0]));
    }
    if curvatures.iter().all(|value| *value < 0.0) {
        return reflect_force(force, modes);
    }
    let mut out = Array1::zeros(force.len());
    for column in 0..modes.ncols() {
        let axis = modes.column(column);
        let parallel = force.dot(&axis);
        Zip::from(&mut out)
            .and(axis)
            .for_each(|slot, &component| *slot -= component * parallel);
    }
    Ok(out)
}

fn check_modes(force: ArrayView1<f64>, modes: ArrayView2<f64>) -> Result<(), SaddleError> {
    if modes.ncols() == 0 {
        return Err(SaddleError::Invalid("index k must be at least 1".into()));
    }
    if modes.nrows() != force.len() || force.is_empty() {
        return Err(SaddleError::Shape(
            "force and modes must share a nonzero length".into(),
        ));
    }
    if !force.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("index-k force"));
    }
    Ok(())
}

/// One index-k minimum-mode step from `position` with the host's `force`.
///
/// `force` is the energy gradient at `position`. `modes`, when set, is
/// a warm start of shape `(n, k)`. The Hessian action is
/// [`PointSurface::hessian_vector`] when that returns a vector, and the
/// search's finite difference otherwise. The translator state starts
/// empty; a host that keeps it uses [`IndexKSession`].
pub fn indexk_step<S: PointSurface>(
    position: Array1<f64>,
    force: Array1<f64>,
    surface: &S,
    k: usize,
    modes: Option<Array2<f64>>,
    options: &MinModeConfig,
) -> Result<IndexKReport, SaddleError> {
    let mut session = IndexKSession::new(
        IndexKConfig::new(k, options.clone()),
        position.clone(),
        modes,
    )?;
    session.set_position(position, Some(force))?;
    session.step(surface)
}

/// Stepping index-k saddle search.
///
/// # Examples
///
/// An index-2 saddle of `V = -x² - ½y² + z²`. The Hessian action is
/// analytic, the product a surrogate supplies. The start lies off the
/// origin.
///
/// ```
/// use ndarray::{Array1, ArrayView1, array};
/// use rgsaddle::{
///     IndexKConfig, IndexKSession, MinModeConfig, MinModeStatus, PointSurface, SaddleError,
/// };
///
/// struct IndexTwo;
///
/// impl PointSurface for IndexTwo {
///     fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
///         let (x, y, z) = (r[0], r[1], r[2]);
///         let energy = -x * x - 0.5 * y * y + z * z;
///         Ok((energy, array![-2.0 * x, -y, 2.0 * z]))
///     }
///
///     fn hessian_vector(
///         &self,
///         _x: ArrayView1<f64>,
///         v: ArrayView1<f64>,
///     ) -> Result<Option<Array1<f64>>, SaddleError> {
///         Ok(Some(array![-2.0 * v[0], -v[1], 2.0 * v[2]]))
///     }
/// }
///
/// let start = array![0.25, -0.2, 0.15];
/// let mut config = IndexKConfig::new(2, MinModeConfig::default());
/// config.search.force_tol = 1e-6;
/// config.search.max_move = 0.1;
/// config.residual_tol = 1e-8;
/// let mut session = IndexKSession::new(config, start, None)?;
/// let report = session.run(&IndexTwo, 400)?;
/// assert_eq!(report.status, MinModeStatus::Converged);
/// assert!(session.position().iter().all(|c| c.abs() < 1e-3));
/// assert!(report.curvatures.iter().all(|c| *c < 0.0));
/// # Ok::<(), SaddleError>(())
/// ```
pub struct IndexKSession {
    engine: Engine,
    modes: Array2<f64>,
    curvatures: Array1<f64>,
    translated: Array1<f64>,
}

enum Engine {
    One(Box<MinModeSession>),
    Block(Box<BlockSession>),
}

struct TrialPoint {
    x: Array1<f64>,
    energy: f64,
    gradient: Array1<f64>,
}

type Product<'a> = dyn FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> + 'a;

struct Centre {
    x: Array1<f64>,
    energy: Option<f64>,
    gradient: Array1<f64>,
}

struct BlockSession {
    config: IndexKConfig,
    x: Array1<f64>,
    modes: Array2<f64>,
    curvatures: Array1<f64>,
    solver: rgmin::Solver,
    iteration: usize,
    centre: Option<Centre>,
    conjugate: Option<Array2<f64>>,
    surface_error: Mutex<Option<SaddleError>>,
    force_gate: ForceGate,
}

impl IndexKSession {
    /// `modes` is an optional warm start, shape `(n, k)`, and may be
    /// omitted. A missing block is the first `k` coordinate axes.
    /// `k = 1` opens a [`MinModeSession`].
    pub fn new(
        config: IndexKConfig,
        x: Array1<f64>,
        modes: Option<Array2<f64>>,
    ) -> Result<Self, SaddleError> {
        validate_config(&config, x.len())?;
        if !x.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        let seed = prepare_seed(x.len(), config.k, modes)?;
        if config.k == 1 {
            let mode = seed.column(0).to_owned();
            let session = MinModeSession::new(config.search, x, mode)?;
            let stored = column_matrix(session.mode());
            let curvatures = Array1::from_elem(1, f64::NAN);
            return Ok(Self {
                engine: Engine::One(Box::new(session)),
                modes: stored,
                curvatures,
                translated: Array1::zeros(0),
            });
        }
        crate::band::check_force_driven(&config.search.method)?;
        let n = x.len();
        let solver = build_solver(
            &config.search.method,
            n,
            config.search.max_move,
            1.0,
            false,
            true,
        );
        let block = BlockSession {
            config,
            x,
            modes: seed.clone(),
            curvatures: Array1::from_elem(seed.ncols(), f64::NAN),
            solver,
            iteration: 0,
            centre: None,
            conjugate: None,
            surface_error: Mutex::new(None),
            force_gate: ForceGate::LinfNorm,
        };
        Ok(Self {
            engine: Engine::Block(Box::new(block)),
            modes: seed.clone(),
            curvatures: Array1::from_elem(seed.ncols(), f64::NAN),
            translated: Array1::zeros(0),
        })
    }

    /// Current geometry.
    pub fn position(&self) -> ArrayView1<'_, f64> {
        match &self.engine {
            Engine::One(session) => session.position(),
            Engine::Block(session) => session.x.view(),
        }
    }

    /// Columns are the modes from the last step, or the seed before it.
    pub fn modes(&self) -> ArrayView2<'_, f64> {
        self.modes.view()
    }

    /// Curvature of each mode from the last step. NaN before the first.
    pub fn curvatures(&self) -> ArrayView1<'_, f64> {
        self.curvatures.view()
    }

    /// Effective gradient of the last step. Empty before the first.
    /// Effective gradient of the last step. Empty before the first.
    pub fn translated_force(&self) -> ArrayView1<'_, f64> {
        self.translated.view()
    }

    /// Move to `x`. `gradient` is the energy gradient there when the host has one.
    pub fn set_position(
        &mut self,
        x: Array1<f64>,
        gradient: Option<Array1<f64>>,
    ) -> Result<(), SaddleError> {
        match &mut self.engine {
            Engine::One(session) => session.set_position(x, gradient),
            Engine::Block(session) => session.set_position(x, gradient),
        }
    }

    /// One index-k step: refresh the k lowest modes, translate the force, move.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IndexKReport, SaddleError> {
        match &mut self.engine {
            Engine::One(session) => {
                let report = session.step(surface)?;
                let mode = session.mode().to_owned();
                self.modes = column_matrix(mode.view());
                self.curvatures = Array1::from_elem(1, report.curvature);
                self.translated = session
                    .translated_force()
                    .map(|force| force.to_owned())
                    .ok_or_else(|| {
                        SaddleError::Solver(
                            "min-mode step did not record a translated force".into(),
                        )
                    })?;
                Ok(IndexKReport {
                    status: report.status,
                    position: session.position().to_owned(),
                    max_force: report.max_force,
                    modes: self.modes.clone(),
                    curvatures: self.curvatures.clone(),
                    translated_force: self.translated.clone(),
                    rotations: report.rotations,
                    iteration: report.iteration,
                    evaluations: report.evaluations,
                })
            }
            Engine::Block(session) => {
                let report = session.step(surface)?;
                self.modes = report.modes.clone();
                self.curvatures = report.curvatures.clone();
                self.translated = report.translated_force.clone();
                Ok(report)
            }
        }
    }

    /// Convenience loop over [`IndexKSession::step`].
    pub fn run<S: PointSurface>(
        &mut self,
        surface: &S,
        max_steps: usize,
    ) -> Result<IndexKReport, SaddleError> {
        let mut report = self.step(surface)?;
        while report.status == MinModeStatus::Running && report.iteration < max_steps {
            report = self.step(surface)?;
        }
        Ok(report)
    }
}

impl BlockSession {
    fn set_position(
        &mut self,
        x: Array1<f64>,
        gradient: Option<Array1<f64>>,
    ) -> Result<(), SaddleError> {
        if x.len() != self.x.len() {
            return Err(SaddleError::Shape("position length must stay fixed".into()));
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("position"));
        }
        self.centre = match gradient {
            Some(gradient) => {
                if gradient.len() != x.len() {
                    return Err(SaddleError::Shape(
                        "gradient must match the position".into(),
                    ));
                }
                if !gradient.iter().all(|value| value.is_finite()) {
                    return Err(SaddleError::NonFinite("index-k force"));
                }
                Some(Centre {
                    x: x.clone(),
                    energy: None,
                    gradient,
                })
            }
            None => None,
        };
        self.x = x;
        Ok(())
    }

    fn step<S: PointSurface>(&mut self, surface: &S) -> Result<IndexKReport, SaddleError> {
        let calls = AtomicUsize::new(0);
        let gradient = self.centre_gradient(surface, &calls)?;
        let excluded = surface.excluded_modes(self.x.view())?;
        let (mut modes, curvatures, actions) =
            self.lowest_modes(surface, &calls, gradient.view(), excluded.view())?;
        let previous = self.modes.clone();
        align_signs(&mut modes, previous.view());
        let eff = translated_force(gradient.view(), modes.view(), curvatures.view())?;
        let all_negative = curvatures.iter().all(|value| *value < 0.0);
        let max_force_here = self.force_gate.value(gradient.view());
        if all_negative && max_force_here <= self.config.search.force_tol {
            self.modes = modes.clone();
            self.curvatures = curvatures.clone();
            return Ok(self.report(
                MinModeStatus::Converged,
                max_force_here,
                modes,
                curvatures,
                eff,
                actions,
                calls.load(Ordering::Relaxed),
            ));
        }
        let arrived = if all_negative {
            self.translate(surface, &calls, &gradient, &eff, &modes, &curvatures)?
        } else {
            let direction = ascent_unit(gradient.view(), modes.view());
            let climbed = crate::minmode::convex_displacement(
                self.x.view(),
                direction.view(),
                direction.view(),
                self.config.search.max_move,
            );
            calls.fetch_add(1, Ordering::Relaxed);
            let (energy, gradient) = surface.eval(climbed.view())?;
            TrialPoint {
                x: climbed,
                energy,
                gradient,
            }
        };
        let TrialPoint {
            x,
            energy,
            gradient: arrived_gradient,
        } = arrived;
        let mut curvatures = curvatures;
        let mut actions = actions;
        let max_force = self.force_gate.value(arrived_gradient.view());
        if max_force <= self.config.search.force_tol && x != self.x {
            let excluded = surface.excluded_modes(x.view())?;
            let previous = modes.clone();
            self.modes = previous.clone();
            let (refreshed, values, extra) = lowest_at(
                self,
                surface,
                &calls,
                x.view(),
                arrived_gradient.view(),
                excluded.view(),
            )?;
            let mut refreshed = refreshed;
            align_signs(&mut refreshed, previous.view());
            modes = refreshed;
            curvatures = values;
            actions += extra;
        }
        self.x = x;
        self.modes = modes.clone();
        self.curvatures = curvatures.clone();
        self.iteration += 1;
        self.centre = Some(Centre {
            x: self.x.clone(),
            energy: Some(energy),
            gradient: arrived_gradient,
        });
        let status = if max_force <= self.config.search.force_tol
            && curvatures.iter().all(|value| *value < 0.0)
        {
            MinModeStatus::Converged
        } else {
            MinModeStatus::Running
        };
        Ok(self.report(
            status,
            max_force,
            modes,
            curvatures,
            eff,
            actions,
            calls.load(Ordering::Relaxed),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn report(
        &self,
        status: MinModeStatus,
        max_force: f64,
        modes: Array2<f64>,
        curvatures: Array1<f64>,
        translated_force: Array1<f64>,
        rotations: usize,
        evaluations: usize,
    ) -> IndexKReport {
        IndexKReport {
            status,
            position: self.x.clone(),
            max_force,
            modes,
            curvatures,
            translated_force,
            rotations,
            iteration: self.iteration,
            evaluations,
        }
    }

    fn centre_gradient<S: PointSurface>(
        &mut self,
        surface: &S,
        calls: &AtomicUsize,
    ) -> Result<Array1<f64>, SaddleError> {
        if let Some(centre) = &self.centre
            && centre.x == self.x
        {
            return Ok(centre.gradient.clone());
        }
        calls.fetch_add(1, Ordering::Relaxed);
        let (energy, gradient) = surface.eval(self.x.view())?;
        if gradient.len() != self.x.len() {
            return Err(SaddleError::Shape("surface gradient length changed".into()));
        }
        if !energy.is_finite() || !gradient.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("index-k force"));
        }
        self.centre = Some(Centre {
            x: self.x.clone(),
            energy: Some(energy),
            gradient: gradient.clone(),
        });
        Ok(gradient)
    }

    fn lowest_modes<S: PointSurface>(
        &mut self,
        surface: &S,
        calls: &AtomicUsize,
        gradient: ArrayView1<f64>,
        excluded: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, usize), SaddleError> {
        let x = self.x.clone();
        lowest_at(self, surface, calls, x.view(), gradient, excluded)
    }

    fn translate<S: PointSurface>(
        &mut self,
        surface: &S,
        calls: &AtomicUsize,
        gradient: &Array1<f64>,
        eff: &Array1<f64>,
        modes: &Array2<f64>,
        curvatures: &Array1<f64>,
    ) -> Result<TrialPoint, SaddleError> {
        if let Ok(mut slot) = self.surface_error.lock() {
            slot.take();
        }
        let start = self.x.clone();
        let start_energy = self
            .centre
            .as_ref()
            .filter(|centre| centre.x == start)
            .and_then(|centre| centre.energy);
        let error_slot = &self.surface_error;
        let last: Mutex<Option<TrialPoint>> = Mutex::new(None);
        let modes = modes.clone();
        let curvatures = curvatures.clone();
        let g0 = gradient.clone();
        let eff0 = eff.clone();
        let oracle = Oracle::unbounded(self.x.len(), |xv: ArrayView1<f64>| {
            let answer = if let Some(energy) = start_energy
                && xv == start.view()
            {
                Ok((energy, g0.clone(), eff0.clone()))
            } else {
                calls.fetch_add(1, Ordering::Relaxed);
                surface.eval(xv).and_then(|(energy, gradient)| {
                    let eff = translated_force(gradient.view(), modes.view(), curvatures.view())?;
                    Ok((energy, gradient, eff))
                })
            };
            match answer {
                Ok((energy, gradient, eff)) => {
                    if let Ok(mut slot) = last.lock() {
                        *slot = Some(TrialPoint {
                            x: xv.to_owned(),
                            energy,
                            gradient,
                        });
                    }
                    (energy, eff)
                }
                Err(err) => {
                    if let Ok(mut slot) = error_slot.lock()
                        && slot.is_none()
                    {
                        *slot = Some(err);
                    }
                    (f64::INFINITY, Array1::zeros(xv.len()))
                }
            }
        });
        let mut x = self.x.clone();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        if let Some(err) = self
            .surface_error
            .lock()
            .ok()
            .and_then(|mut slot| slot.take())
        {
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        if let Some(point) = last.into_inner().ok().flatten()
            && point.x == x
        {
            return Ok(point);
        }
        calls.fetch_add(1, Ordering::Relaxed);
        let (energy, gradient) = surface.eval(x.view())?;
        Ok(TrialPoint {
            x,
            energy,
            gradient,
        })
    }
}

fn lowest_at<S: PointSurface>(
    session: &mut BlockSession,
    surface: &S,
    calls: &AtomicUsize,
    x: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    excluded: ArrayView2<f64>,
) -> Result<(Array2<f64>, Array1<f64>, usize), SaddleError> {
    let mut seed = session.modes.clone();
    if excluded.nrows() > 0 {
        for column in 0..seed.ncols() {
            let mut vector = seed.column(column).to_owned();
            deflate(&mut vector, excluded);
            seed.column_mut(column).assign(&vector);
        }
    }
    let seed = orthonormal_columns(seed.view())?;
    let config = session.config.search.clone();
    let kind = session.config.block;
    let tol = session.config.residual_tol;
    let iterations = session.config.max_block_iterations;
    let x = x.to_owned();
    let gradient = gradient.to_owned();
    let mut conjugate = session.conjugate.clone();
    let block = {
        let mut apply = |vector: ArrayView1<f64>| {
            let mut action =
                hessian_action(surface, calls, x.view(), gradient.view(), vector, &config)?;
            if excluded.nrows() > 0 {
                deflate(&mut action, excluded);
            }
            Ok(action)
        };
        solve_block(
            kind,
            &mut conjugate,
            seed.view(),
            tol,
            iterations,
            &mut apply,
        )?
    };
    session.conjugate = conjugate;
    Ok((block.modes, block.values, block.actions))
}

fn validate_config(config: &IndexKConfig, n: usize) -> Result<(), SaddleError> {
    if config.k == 0 {
        return Err(SaddleError::Invalid("index k must be at least 1".into()));
    }
    if config.k > n {
        return Err(SaddleError::Invalid("index k exceeds the dimension".into()));
    }
    if !(config.search.dr.is_finite() && config.search.dr > 0.0) {
        return Err(SaddleError::Invalid(
            "finite-difference step must be positive and finite".into(),
        ));
    }
    if !(config.residual_tol.is_finite() && config.residual_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "residual tolerance must be finite and non-negative".into(),
        ));
    }
    if config.max_block_iterations == 0 {
        return Err(SaddleError::Invalid(
            "block iteration cap must be positive".into(),
        ));
    }
    Ok(())
}

fn prepare_seed(
    n: usize,
    k: usize,
    modes: Option<Array2<f64>>,
) -> Result<Array2<f64>, SaddleError> {
    let seed = match modes {
        Some(modes) => {
            if modes.nrows() != n || modes.ncols() != k {
                return Err(SaddleError::Shape(
                    "warm start must have shape (n, k)".into(),
                ));
            }
            if !modes.iter().all(|value| value.is_finite()) {
                return Err(SaddleError::NonFinite("index-k mode"));
            }
            modes
        }
        None => coordinate_seed(n, k),
    };
    orthonormal_columns(seed.view())
}

fn coordinate_seed(n: usize, k: usize) -> Array2<f64> {
    let mut seed = Array2::zeros((n, k));
    for column in 0..k {
        seed[(column, column)] = 1.0;
    }
    seed
}

fn column_matrix(mode: ArrayView1<f64>) -> Array2<f64> {
    let mut modes = Array2::zeros((mode.len(), 1));
    modes.column_mut(0).assign(&mode);
    modes
}

fn orthonormal_columns(seed: ArrayView2<f64>) -> Result<Array2<f64>, SaddleError> {
    let k = seed.ncols();
    let orth = modified_gram_schmidt(seed, None, 1e-12);
    if orth.ncols() != k {
        return Err(SaddleError::Invalid(
            "warm start must span k independent modes".into(),
        ));
    }
    Ok(orth)
}

fn align_signs(modes: &mut Array2<f64>, previous: ArrayView2<f64>) {
    let columns = modes.ncols().min(previous.ncols());
    for column in 0..columns {
        if modes.column(column).dot(&previous.column(column)) < 0.0 {
            modes.column_mut(column).mapv_inplace(|value| -value);
        }
    }
}

fn deflate(vector: &mut Array1<f64>, excluded: ArrayView2<f64>) {
    for row in excluded.rows() {
        let norm = row.dot(&row);
        if norm > 1e-24 {
            let coefficient = vector.dot(&row) / norm;
            Zip::from(&mut *vector)
                .and(row)
                .for_each(|slot, &component| *slot -= coefficient * component);
        }
    }
}

fn ascent_unit(force: ArrayView1<f64>, modes: ArrayView2<f64>) -> Array1<f64> {
    let mut direction = Array1::<f64>::zeros(force.len());
    for column in 0..modes.ncols() {
        let axis = modes.column(column);
        let parallel = force.dot(&axis);
        for (slot, &component) in direction.iter_mut().zip(axis.iter()) {
            *slot += component * parallel;
        }
    }
    let norm = direction.dot(&direction).sqrt();
    if norm > 1e-14 {
        direction /= norm;
        direction
    } else {
        modes.column(0).to_owned()
    }
}

fn hessian_action<S: PointSurface>(
    surface: &S,
    calls: &AtomicUsize,
    x: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    vector: ArrayView1<f64>,
    config: &MinModeConfig,
) -> Result<Array1<f64>, SaddleError> {
    if let Some(action) = surface.hessian_vector(x, vector)? {
        if action.len() != x.len() {
            return Err(SaddleError::Shape(
                "surface Hessian action length changed".into(),
            ));
        }
        if !action.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("surface Hessian action"));
        }
        return Ok(action);
    }
    let dr = config.dr;
    let mut plus = x.to_owned();
    plus.scaled_add(dr, &vector);
    calls.fetch_add(1, Ordering::Relaxed);
    let (_, forward) = surface.eval(plus.view())?;
    match config.difference {
        FiniteDifference::Forward => {
            let mut action = forward;
            Zip::from(&mut action)
                .and(gradient)
                .for_each(|slot, &centre| *slot = (*slot - centre) / dr);
            Ok(action)
        }
        FiniteDifference::Central => {
            let mut minus = x.to_owned();
            minus.scaled_add(-dr, &vector);
            calls.fetch_add(1, Ordering::Relaxed);
            let (_, backward) = surface.eval(minus.view())?;
            let width = 2.0 * dr;
            let mut action = forward;
            Zip::from(&mut action)
                .and(&backward)
                .for_each(|slot, &other| *slot = (*slot - other) / width);
            Ok(action)
        }
    }
}

pub(crate) struct EigenBlock {
    modes: Array2<f64>,
    values: Array1<f64>,
    actions: usize,
}

/// Lowest `k` eigenpairs of a symmetric matrix-free product.
///
/// The local solvers below implement this. rgmin's upcoming
/// `lowest_eigenpairs` can implement it the same way; the index-k step
/// only calls the trait.
pub(crate) trait LowestEigenpairs {
    fn lowest_eigenpairs(
        &mut self,
        seed: ArrayView2<f64>,
        tol: f64,
        max_iterations: usize,
        apply: &mut Product<'_>,
    ) -> Result<EigenBlock, SaddleError>;
}

struct LobpcgEigenpairs {
    conjugate: Option<Array2<f64>>,
}

struct BlockLanczosEigenpairs;

fn solve_block(
    kind: BlockEigenKind,
    conjugate: &mut Option<Array2<f64>>,
    seed: ArrayView2<f64>,
    tol: f64,
    max_iterations: usize,
    apply: &mut Product<'_>,
) -> Result<EigenBlock, SaddleError> {
    match kind {
        BlockEigenKind::Lobpcg => {
            let mut solver = LobpcgEigenpairs {
                conjugate: conjugate.take(),
            };
            let block = solver.lowest_eigenpairs(seed, tol, max_iterations, apply)?;
            *conjugate = solver.conjugate;
            Ok(block)
        }
        BlockEigenKind::BlockLanczos => {
            *conjugate = None;
            BlockLanczosEigenpairs.lowest_eigenpairs(seed, tol, max_iterations, apply)
        }
    }
}

impl LowestEigenpairs for LobpcgEigenpairs {
    fn lowest_eigenpairs(
        &mut self,
        seed: ArrayView2<f64>,
        tol: f64,
        max_iterations: usize,
        apply: &mut Product<'_>,
    ) -> Result<EigenBlock, SaddleError> {
        let k = seed.ncols();
        let mut x = orthonormal_columns(seed)?;
        let mut ax = apply_columns(&mut *apply, x.view())?;
        let mut actions = k;
        for _ in 0..max_iterations {
            let (values, vectors) = exact_eigh(gram(x.view(), ax.view())?.view())?;
            x = matmul(x.view(), vectors.view());
            ax = matmul(ax.view(), vectors.view());
            let values = values.slice(s![..k]).to_owned();
            let mut residual = ax.clone();
            for column in 0..k {
                let theta = values[column];
                Zip::from(residual.column_mut(column))
                    .and(x.column(column))
                    .for_each(|slot, &component| *slot -= theta * component);
            }
            let max_residual = (0..k)
                .map(|column| residual.column(column).dot(&residual.column(column)).sqrt())
                .fold(0.0, f64::max);
            if max_residual <= tol {
                self.conjugate = Some(residual);
                return Ok(EigenBlock {
                    modes: x,
                    values,
                    actions,
                });
            }
            let corrections = modified_gram_schmidt(residual.view(), Some(x.view()), 1e-10);
            if corrections.ncols() == 0 {
                return Ok(EigenBlock {
                    modes: x,
                    values,
                    actions,
                });
            }
            let mut blocks = vec![x.clone(), corrections.clone()];
            let mut actions_blocks =
                vec![ax.clone(), apply_columns(&mut *apply, corrections.view())?];
            actions += corrections.ncols();
            if let Some(previous) = &self.conjugate {
                let locked = hstack(&[x.view(), corrections.view()]);
                let direction = modified_gram_schmidt(previous.view(), Some(locked.view()), 1e-10);
                if direction.ncols() > 0 {
                    actions += direction.ncols();
                    actions_blocks.push(apply_columns(&mut *apply, direction.view())?);
                    blocks.push(direction);
                }
            }
            let basis = hstack(&blocks.iter().map(|block| block.view()).collect::<Vec<_>>());
            let applied = hstack(
                &actions_blocks
                    .iter()
                    .map(|block| block.view())
                    .collect::<Vec<_>>(),
            );
            let (ritz, vectors) = exact_eigh(gram(basis.view(), applied.view())?.view())?;
            let width = basis.ncols();
            let take = k.min(width);
            let coeff = vectors.slice(s![.., ..take]).to_owned();
            let updated = matmul(basis.view(), coeff.view());
            let updated_action = matmul(applied.view(), coeff.view());
            let tail = if basis.ncols() > k {
                let coeff_tail = vectors.slice(s![k.., ..take]).to_owned();
                let span = basis.slice(s![.., k..]).to_owned();
                Some(matmul(span.view(), coeff_tail.view()))
            } else {
                None
            };
            x = updated;
            ax = updated_action;
            self.conjugate = tail;
            if max_residual <= tol || take < k {
                return Ok(EigenBlock {
                    modes: x,
                    values: ritz.slice(s![..take]).to_owned(),
                    actions,
                });
            }
        }
        let (values, vectors) = exact_eigh(gram(x.view(), ax.view())?.view())?;
        Ok(EigenBlock {
            modes: matmul(x.view(), vectors.slice(s![.., ..k]).to_owned().view()),
            values: values.slice(s![..k]).to_owned(),
            actions,
        })
    }
}

impl LowestEigenpairs for BlockLanczosEigenpairs {
    fn lowest_eigenpairs(
        &mut self,
        seed: ArrayView2<f64>,
        tol: f64,
        max_iterations: usize,
        apply: &mut Product<'_>,
    ) -> Result<EigenBlock, SaddleError> {
        let k = seed.ncols();
        let n = seed.nrows();
        let mut basis = orthonormal_columns(seed)?;
        let mut applied = apply_columns(&mut *apply, basis.view())?;
        let mut actions = k;
        let mut last_width = k;
        for _ in 1..max_iterations.max(1) {
            if basis.ncols() >= n {
                break;
            }
            let start = applied.ncols() - last_width;
            let newest = applied.slice(s![.., start..]).to_owned();
            let orth = modified_gram_schmidt(newest.view(), Some(basis.view()), 1e-10);
            if orth.ncols() == 0 {
                break;
            }
            let aorth = apply_columns(&mut *apply, orth.view())?;
            actions += orth.ncols();
            last_width = orth.ncols();
            basis = hstack(&[basis.view(), orth.view()]);
            applied = hstack(&[applied.view(), aorth.view()]);
            let residual = block_residual(basis.view(), applied.view(), k)?;
            if residual <= tol {
                break;
            }
        }
        let (values, vectors) = exact_eigh(gram(basis.view(), applied.view())?.view())?;
        let take = k.min(basis.ncols());
        Ok(EigenBlock {
            modes: matmul(
                basis.view(),
                vectors.slice(s![.., ..take]).to_owned().view(),
            ),
            values: values.slice(s![..take]).to_owned(),
            actions,
        })
    }
}

fn block_residual(
    basis: ArrayView2<f64>,
    applied: ArrayView2<f64>,
    k: usize,
) -> Result<f64, SaddleError> {
    let (values, vectors) = exact_eigh(gram(basis, applied)?.view())?;
    let take = k.min(basis.ncols());
    let modes = matmul(basis, vectors.slice(s![.., ..take]).to_owned().view());
    let amodes = matmul(applied, vectors.slice(s![.., ..take]).to_owned().view());
    let mut max_residual = 0.0_f64;
    for column in 0..take {
        let mut residual = amodes.column(column).to_owned();
        let theta = values[column];
        Zip::from(&mut residual)
            .and(modes.column(column))
            .for_each(|slot, &component| *slot -= theta * component);
        max_residual = max_residual.max(residual.dot(&residual).sqrt());
    }
    Ok(max_residual)
}

fn apply_columns(
    apply: &mut Product<'_>,
    vectors: ArrayView2<f64>,
) -> Result<Array2<f64>, SaddleError> {
    let mut out = Array2::zeros(vectors.raw_dim());
    for column in 0..vectors.ncols() {
        let action = apply(vectors.column(column))?;
        if action.len() != vectors.nrows() {
            return Err(SaddleError::Shape("Hessian action length changed".into()));
        }
        out.column_mut(column).assign(&action);
    }
    Ok(out)
}

fn gram(x: ArrayView2<f64>, ax: ArrayView2<f64>) -> Result<Array2<f64>, SaddleError> {
    if x.ncols() != ax.ncols() || x.nrows() != ax.nrows() {
        return Err(SaddleError::Shape(
            "Hessian action block does not match the basis".into(),
        ));
    }
    let k = x.ncols();
    let mut matrix = Array2::zeros((k, k));
    for row in 0..k {
        for column in row..k {
            let value = 0.5
                * (x.column(row).dot(&ax.column(column)) + x.column(column).dot(&ax.column(row)));
            matrix[(row, column)] = value;
            matrix[(column, row)] = value;
        }
    }
    Ok(matrix)
}

fn matmul(left: ArrayView2<f64>, right: ArrayView2<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((left.nrows(), right.ncols()));
    for column in 0..right.ncols() {
        for row in 0..left.nrows() {
            let mut sum = 0.0;
            for index in 0..left.ncols() {
                sum += left[(row, index)] * right[(index, column)];
            }
            out[(row, column)] = sum;
        }
    }
    out
}

fn hstack(blocks: &[ArrayView2<f64>]) -> Array2<f64> {
    let rows = blocks.first().map(|block| block.nrows()).unwrap_or(0);
    let cols = blocks.iter().map(|block| block.ncols()).sum();
    let mut out = Array2::zeros((rows, cols));
    let mut column = 0;
    for block in blocks {
        for index in 0..block.ncols() {
            out.column_mut(column).assign(&block.column(index));
            column += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{BlockEigenKind, reflect_force, solve_block};
    use crate::error::SaddleError;
    use ndarray::{Array1, Array2, ArrayView1};

    fn diagonal_apply(
        values: &[f64],
    ) -> impl FnMut(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        let values = values.to_vec();
        move |vector: ArrayView1<f64>| {
            let mut out = Array1::zeros(vector.len());
            for (slot, (&component, &value)) in out.iter_mut().zip(vector.iter().zip(values.iter()))
            {
                *slot = component * value;
            }
            Ok(out)
        }
    }

    #[test]
    fn lobpcg_recovers_the_two_lowest_of_a_diagonal() {
        let mut apply = diagonal_apply(&[-2.0, -1.0, 4.0, 7.0]);
        let mut seed = Array2::zeros((4, 2));
        seed[(0, 0)] = 0.6;
        seed[(1, 0)] = 0.8;
        seed[(2, 1)] = 0.3;
        seed[(3, 1)] = 0.4;
        seed[(1, 1)] = 0.5;
        let mut conjugate = None;
        let block = solve_block(
            BlockEigenKind::Lobpcg,
            &mut conjugate,
            seed.view(),
            1e-10,
            20,
            &mut apply,
        )
        .unwrap();
        assert!((block.values[0] + 2.0).abs() < 1e-8);
        assert!((block.values[1] + 1.0).abs() < 1e-8);
        assert!(block.modes.column(0)[0].abs() > 0.99);
        assert!(block.modes.column(1)[1].abs() > 0.99);
    }

    #[test]
    fn block_lanczos_recovers_the_same_diagonal() {
        let mut apply = diagonal_apply(&[-2.0, -1.0, 4.0, 7.0]);
        let mut seed = Array2::zeros((4, 2));
        seed[(0, 0)] = 0.4;
        seed[(1, 0)] = 0.5;
        seed[(2, 0)] = 0.2;
        seed[(3, 0)] = 0.1;
        seed[(0, 1)] = -0.2;
        seed[(1, 1)] = 0.3;
        seed[(2, 1)] = 0.7;
        seed[(3, 1)] = 0.4;
        let mut conjugate = None;
        let block = solve_block(
            BlockEigenKind::BlockLanczos,
            &mut conjugate,
            seed.view(),
            1e-10,
            8,
            &mut apply,
        )
        .unwrap();
        assert!((block.values[0] + 2.0).abs() < 1e-8);
        assert!((block.values[1] + 1.0).abs() < 1e-8);
    }

    #[test]
    fn reflect_one_column_matches_the_householder_spelling() {
        let force = Array1::from_vec(vec![1.0, -4.0, 0.5]);
        let mut modes = Array2::zeros((3, 1));
        modes[(0, 0)] = 1.0;
        let out = reflect_force(force.view(), modes.view()).unwrap();
        let reflected = crate::stencil::reflect(force.view(), modes.column(0));
        for (left, right) in out.iter().zip(reflected.iter()) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }
}
