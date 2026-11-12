//! Climbing-image solid-state NEB on the dimer chart.
//!
//! Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys.
//! 136, 074103 (2012), doi:10.1063/1.3684549, put atom and cell
//! displacements in one vector and spring the band in that vector.
//! The coordinate here is the one [`crate::SolidDimer`] already steps
//! on: Xiao, Sheppard, Rogal, and Henkelman, J. Chem. Phys. 140,
//! 174104 (2014), doi:10.1063/1.4873437, with
//!
//! `J = V^{1/3} N^{1/6}`,
//!
//! `h = h_0 (I + ε)`, `r = r_0 (I + ε)`, and the free strains stored
//! as `J ε`. The enthalpy is `E + V_0 σ_ext : ε`, the same sign as
//! [`crate::JointChart`]: a positive diagonal stress pushes the cell
//! inward. The tangent, the spring and the climbing image are the
//! band's, applied to that joint vector.
//!
//! A frozen cell is not a second implementation. [`SolidNeb::frozen`]
//! is a [`crate::BandSession`], so its positions match that session
//! for the same configuration and the same surface.
//!
//! The absolute-cell block on [`crate::BandConfig::solid`] is the
//! other solid-state band: hydrostatic pressure, cells stored as
//! matrices. This session is the dimer chart, including a general
//! applied stress.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1, s};
use rayon::prelude::*;
use rgmin::{Accept, Control, Method, Oracle, Solver};

use crate::band::{
    BandConfig, BandReport, BandSession, BandStatus, BandSurface, check_force_driven,
    remove_net_translation,
};
use crate::cell_log::{M3, m3_inv, m3_mul, m3_to_row9};
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::projection::{ProjectionKind, climbing_image_force, dneb_component, force_perp};
use crate::solid_dimer::{CellMask, JointChart, STRAIN_VOIGT, finite_difference_cauchy_stress};
use crate::solid_state::zero_upper;
use crate::spring::SpringKind;
use crate::tangent::compute_tangent;

/// One image of a joint-chart band, after the step that evaluated it.
#[derive(Clone, Debug)]
pub struct SolidImage {
    /// Cartesian positions.
    pub positions: Array1<f64>,
    /// Cell, lattice vectors as rows.
    pub cell: M3,
    /// Joint coordinate. A frozen image is the Cartesian position.
    pub generalized: Array1<f64>,
    /// Potential energy.
    pub energy: f64,
    /// Enthalpy on the chart. A frozen image reports the energy.
    pub enthalpy: f64,
    /// Cauchy stress. Empty on a frozen image; the cell did not enter.
    pub stress: Option<M3>,
}

/// Climbing-image NEB on atoms, or on atoms plus cell strain.
pub struct SolidNeb {
    inner: Inner,
}

enum Inner {
    Frozen(Box<BandSession>),
    Joint(Box<JointRun>),
}

struct JointRun {
    config: BandConfig,
    method: Method,
    chart: JointChart,
    qs: Array2<f64>,
    state: JointState,
    solver: Solver,
    iteration: usize,
    unmoved: usize,
}

struct JointState {
    armed: AtomicBool,
    ci_index: AtomicI64,
    baseline: Mutex<Option<f64>>,
    last_max_force: Mutex<f64>,
    rows: AtomicUsize,
    error: Mutex<Option<SaddleError>>,
    eval: Mutex<Option<JointEval>>,
}

#[derive(Clone)]
struct JointEval {
    interior: Array1<f64>,
    energy: Array1<f64>,
    enthalpy: Array1<f64>,
    gradient: Array2<f64>,
    stress: Vec<M3>,
    cartesian: Array2<f64>,
    cells: Vec<M3>,
    endpoints: bool,
}

impl JointState {
    fn new() -> Self {
        Self {
            armed: AtomicBool::new(false),
            ci_index: AtomicI64::new(-1),
            baseline: Mutex::new(None),
            last_max_force: Mutex::new(f64::INFINITY),
            rows: AtomicUsize::new(0),
            error: Mutex::new(None),
            eval: Mutex::new(None),
        }
    }

    fn ci(&self) -> Option<usize> {
        let index = self.ci_index.load(Ordering::Relaxed);
        (index >= 0).then_some(index as usize)
    }

    fn last_max_force(&self) -> f64 {
        *self
            .last_max_force
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }

    fn take_error(&self) -> Option<SaddleError> {
        self.error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take()
    }

    fn record_error(&self, err: SaddleError) {
        *self
            .error
            .lock()
            .unwrap_or_else(|poison| poison.into_inner()) = Some(err);
    }
}

impl SolidNeb {
    /// Fixed-cell band. `config.solid` must be clear: the cell is not
    /// a degree of freedom, and the session is the ordinary band.
    pub fn frozen(config: BandConfig, positions: Array2<f64>) -> Result<Self, SaddleError> {
        if config.solid.is_some() {
            return Err(SaddleError::Invalid(
                "a frozen solid-state NEB has no cell block; clear BandConfig::solid".into(),
            ));
        }
        Ok(Self {
            inner: Inner::Frozen(Box::new(BandSession::new(config, positions)?)),
        })
    }

    /// Variable-cell band on one [`JointChart`].
    ///
    /// `reference` is `h_0`, lower triangular, shared by every image.
    /// `positions` and `cells` are the Cartesian geometry of each
    /// image, endpoints included, already unwrapped into that chart:
    /// the joint vector is `r_0` then `J ε`, with no minimum image
    /// applied inside the step. `external` is the Cauchy stress in the
    /// enthalpy `E + V_0 σ_ext : ε`. `weight` scales `J`; 1 is the
    /// 2012 value.
    ///
    /// The atomic min-mode refinement, the zoom, the exponential
    /// metric, AutoNEB and [`BandConfig::solid`] belong to the atomic
    /// or absolute-cell band. A joint chart leaves them unset, and
    /// `min_mode` is `None`.
    pub fn joint(
        config: BandConfig,
        reference: M3,
        positions: Array2<f64>,
        cells: &[M3],
        mask: CellMask,
        external: M3,
        weight: f64,
    ) -> Result<Self, SaddleError> {
        if config.solid.is_some()
            || config.min_mode.is_some()
            || config.zoom.is_some()
            || config.exp_precon.is_some()
            || config.autoneb.is_some()
            || config.cell.is_some()
        {
            return Err(SaddleError::Invalid(
                "a joint-chart NEB leaves solid, min_mode, zoom, exp_precon, autoneb and cell unset"
                    .into(),
            ));
        }
        if matches!(
            config.spring,
            SpringKind::OnsagerMachlup { .. } | SpringKind::OnsagerMachlupLive { .. }
        ) {
            return Err(SaddleError::Invalid(
                "a joint-chart NEB does not use Onsager-Machlup springs".into(),
            ));
        }
        let n_images = positions.nrows();
        let n_cart = positions.ncols();
        if n_images < 3 || n_cart == 0 || !n_cart.is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "band needs >= 3 images of 3N dof; got {n_images} x {n_cart}"
            )));
        }
        if cells.len() != n_images {
            return Err(SaddleError::Shape(format!(
                "joint-chart NEB needs {n_images} cells; got {}",
                cells.len()
            )));
        }
        if let SpringKind::Weighted { ks } = &config.spring
            && ks.len() != n_images - 1
        {
            return Err(SaddleError::Shape(format!(
                "weighted springs need n_images - 1 = {} constants; got {}",
                n_images - 1,
                ks.len()
            )));
        }
        if !(config.time_step.is_finite() && config.time_step > 0.0) {
            return Err(SaddleError::Invalid(
                "band time step must be positive and finite".into(),
            ));
        }
        let method = if config.quickmin {
            Method::QuickMin
        } else {
            config.method.clone()
        };
        check_force_driven(&method)?;
        let chart = JointChart::new(reference, n_cart / 3, mask, external, weight)?;
        let mut qs = Array2::zeros((n_images, chart.dimension()));
        for (i, cell) in cells.iter().enumerate() {
            let q = joint_coordinate(&chart, mask, positions.row(i), *cell)?;
            qs.row_mut(i).assign(&q);
        }
        let interior = (n_images - 2) * chart.dimension();
        let solver = build_solver(&config, &method, interior);
        Ok(Self {
            inner: Inner::Joint(Box::new(JointRun {
                config,
                method,
                chart,
                qs,
                state: JointState::new(),
                solver,
                iteration: 0,
                unmoved: 0,
            })),
        })
    }

    /// True when the cell is not a degree of freedom.
    pub fn is_frozen(&self) -> bool {
        matches!(self.inner, Inner::Frozen(_))
    }

    /// Jacobian of a joint chart. `None` when the cell is frozen.
    pub fn jacobian(&self) -> Option<f64> {
        match &self.inner {
            Inner::Frozen(_) => None,
            Inner::Joint(run) => Some(run.chart.jacobian()),
        }
    }

    /// Chart of a joint band. `None` when the cell is frozen.
    pub fn chart(&self) -> Option<&JointChart> {
        match &self.inner {
            Inner::Frozen(_) => None,
            Inner::Joint(run) => Some(&run.chart),
        }
    }

    /// Cartesian positions, one image per row.
    pub fn cartesian(&self) -> Result<Array2<f64>, SaddleError> {
        match &self.inner {
            Inner::Frozen(band) => Ok(band.positions().to_owned()),
            Inner::Joint(run) => {
                let n_cart = run.chart.dimension() - run.chart.n_strain();
                let mut out = Array2::zeros((run.qs.nrows(), n_cart));
                for (i, q) in run.qs.axis_iter(ndarray::Axis(0)).enumerate() {
                    let (r, _) = run.chart.geometry(q)?;
                    out.row_mut(i).assign(&r);
                }
                Ok(out)
            }
        }
    }

    /// Cell of every image. `None` when the cell is frozen: the host
    /// holds that cell, and the band never reads it.
    pub fn cells(&self) -> Result<Option<Vec<M3>>, SaddleError> {
        match &self.inner {
            Inner::Frozen(_) => Ok(None),
            Inner::Joint(run) => {
                let mut cells = Vec::with_capacity(run.qs.nrows());
                for q in run.qs.axis_iter(ndarray::Axis(0)) {
                    cells.push(run.chart.geometry(q)?.1);
                }
                Ok(Some(cells))
            }
        }
    }

    /// Joint coordinates, one image per row. A frozen band is the
    /// Cartesian position.
    pub fn generalized(&self) -> Array2<f64> {
        match &self.inner {
            Inner::Frozen(band) => band.positions().to_owned(),
            Inner::Joint(run) => run.qs.clone(),
        }
    }

    /// Images at the last evaluation. `None` before the first step.
    pub fn images(&self) -> Option<Vec<SolidImage>> {
        match &self.inner {
            Inner::Frozen(band) => {
                let eval = band.evaluation()?;
                let pos = band.positions();
                Some(
                    (0..pos.nrows())
                        .map(|i| SolidImage {
                            positions: pos.row(i).to_owned(),
                            cell: [[0.0; 3]; 3],
                            generalized: pos.row(i).to_owned(),
                            energy: eval.energies[i],
                            enthalpy: eval.energies[i],
                            stress: None,
                        })
                        .collect(),
                )
            }
            Inner::Joint(run) => {
                let slot = run.state.eval.lock().ok()?;
                let eval = slot.as_ref()?;
                Some(
                    (0..eval.energy.len())
                        .map(|i| SolidImage {
                            positions: eval.cartesian.row(i).to_owned(),
                            cell: eval.cells[i],
                            generalized: run.qs.row(i).to_owned(),
                            energy: eval.energy[i],
                            enthalpy: eval.enthalpy[i],
                            stress: Some(eval.stress[i]),
                        })
                        .collect(),
                )
            }
        }
    }

    /// One band step on a frozen cell, through the ordinary band surface.
    ///
    /// A joint chart has no batch Cartesian surface; it steps with
    /// [`Self::step`].
    pub fn step_surface<S: BandSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        match &mut self.inner {
            Inner::Frozen(band) => band.step(surface),
            Inner::Joint(_) => Err(SaddleError::Invalid(
                "a joint-chart band steps through PointSurface".into(),
            )),
        }
    }

    /// One band step.
    ///
    /// A frozen cell calls [`PointSurface::eval`]. A joint chart calls
    /// [`PointSurface::eval_in_cell`] and [`PointSurface::cauchy_stress`],
    /// and finite-differences the stress when the surface returns
    /// `None`.
    pub fn step<S: PointSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        match &mut self.inner {
            Inner::Frozen(band) => {
                let adapter = PointBand { surface };
                band.step(&adapter)
            }
            Inner::Joint(run) => run.step(surface),
        }
    }
}

/// Joint coordinate of one Cartesian geometry on `chart`.
///
/// `h = h_0 (I + ε)` with the upper triangle of `ε` clear, and
/// `r = r_0 (I + ε)`. A strain the mask does not free has to be zero.
pub fn joint_coordinate(
    chart: &JointChart,
    mask: CellMask,
    positions: ArrayView1<f64>,
    cell: M3,
) -> Result<Array1<f64>, SaddleError> {
    let n_cart = chart.dimension() - chart.n_strain();
    if positions.len() != n_cart {
        return Err(SaddleError::Shape(
            "joint coordinate position length".into(),
        ));
    }
    let prod = m3_mul(m3_inv(chart.reference())?, cell);
    let mut eps = prod;
    for (i, row) in eps.iter_mut().enumerate() {
        row[i] -= 1.0;
    }
    if eps[0][1].abs() > 1e-8 || eps[0][2].abs() > 1e-8 || eps[1][2].abs() > 1e-8 {
        return Err(SaddleError::Invalid(
            "joint coordinate cell is not h0 (I + ε) with ε lower triangular".into(),
        ));
    }
    zero_upper(&mut eps);
    let inv_m = m3_inv(prod)?;
    let mut q = Array1::zeros(chart.dimension());
    for atom in 0..n_cart / 3 {
        let r = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for a in 0..3 {
            q[3 * atom + a] = r[0] * inv_m[0][a] + r[1] * inv_m[1][a] + r[2] * inv_m[2][a];
        }
    }
    let mut k = 0;
    let jacobian = chart.jacobian();
    for (&(i, j), on) in STRAIN_VOIGT.iter().zip(mask.flags()) {
        if on {
            q[n_cart + k] = jacobian * eps[i][j];
            k += 1;
        } else if eps[i][j].abs() > 1e-8 {
            return Err(SaddleError::Invalid(
                "joint coordinate moves a strain the mask holds".into(),
            ));
        }
    }
    if k != chart.n_strain() {
        return Err(SaddleError::Invalid(
            "joint coordinate mask does not match the chart".into(),
        ));
    }
    let (back_r, back_h) = chart.geometry(q.view())?;
    for (left, right) in positions.iter().zip(back_r.iter()) {
        if (left - right).abs() > 1e-6 {
            return Err(SaddleError::Invalid(
                "joint coordinate did not reproduce the positions".into(),
            ));
        }
    }
    for i in 0..3 {
        for j in 0..3 {
            if (cell[i][j] - back_h[i][j]).abs() > 1e-6 {
                return Err(SaddleError::Invalid(
                    "joint coordinate did not reproduce the cell".into(),
                ));
            }
        }
    }
    Ok(q)
}

struct PointBand<'a, S> {
    surface: &'a S,
}

impl<S: PointSurface> BandSurface for PointBand<'_, S> {
    fn eval(
        &self,
        positions: ndarray::ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.rows().into_iter().enumerate() {
            let (energy, gradient) = self.surface.eval(row)?;
            if gradient.len() != row.len() {
                return Err(SaddleError::Shape(
                    "frozen solid-state NEB gradient length".into(),
                ));
            }
            energies[i] = energy;
            gradients.row_mut(i).assign(&gradient);
        }
        Ok(())
    }
}

fn build_solver(config: &BandConfig, method: &Method, interior_dof: usize) -> Solver {
    let control = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: config.time_step,
        maxmove: None,
        ..Control::default()
    };
    let mut solver = Solver::new(method.clone(), control, interior_dof);
    solver.set_atom_maxmove(config.max_move);
    if config.quickmin {
        solver.set_quickmin_options(config.quickmin_options.clone());
    }
    if matches!(method, Method::Lbfgs { .. }) {
        solver.set_accept(Accept::Step);
        solver.set_lbfgs_neb_guards(true);
    }
    solver
}

fn strip_translation(force: &mut Array1<f64>, n_cart: usize) {
    if n_cart < 6 || !n_cart.is_multiple_of(3) || n_cart > force.len() {
        return;
    }
    let mut atomic = force.slice(s![..n_cart]).to_owned();
    remove_net_translation(&mut atomic);
    force.slice_mut(s![..n_cart]).assign(&atomic);
}

fn max_abs(force: &Array1<f64>) -> f64 {
    force.iter().fold(0.0_f64, |m, c| m.max(c.abs()))
}

fn unpack_stress(packed: [f64; 9]) -> Result<M3, SaddleError> {
    if !packed.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("solid-state stress"));
    }
    let mut stress = [[0.0; 3]; 3];
    for i in 0..9 {
        stress[i / 3][i % 3] = packed[i];
    }
    Ok(stress)
}

struct ImageEval {
    energy: f64,
    enthalpy: f64,
    gradient: Array1<f64>,
    stress: M3,
    cartesian: Array1<f64>,
    cell: M3,
}

fn eval_image<S: PointSurface>(
    chart: &JointChart,
    surface: &S,
    q: ArrayView1<f64>,
) -> Result<ImageEval, SaddleError> {
    let (r, h) = chart.geometry(q)?;
    let flat = m3_to_row9(h);
    let (energy, g_cart) = surface.eval_in_cell(r.view(), &flat)?;
    if g_cart.len() != r.len() {
        return Err(SaddleError::Shape("joint-chart NEB gradient length".into()));
    }
    let stress = match surface.cauchy_stress(r.view(), &flat)? {
        Some(packed) => unpack_stress(packed)?,
        None => finite_difference_cauchy_stress(surface, r.view(), h, 1e-5)?,
    };
    let gradient = chart.gradient(q, g_cart.view(), stress)?;
    let enthalpy = chart.enthalpy(energy, q)?;
    Ok(ImageEval {
        energy,
        enthalpy,
        gradient,
        stress,
        cartesian: r,
        cell: h,
    })
}

fn load_eval<S: PointSurface>(
    chart: &JointChart,
    surface: &S,
    state: &JointState,
    q_first: ArrayView1<f64>,
    q_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<JointEval, SaddleError> {
    let n_cart = chart.dimension() - chart.n_strain();
    let dim = chart.dimension();
    let n_interior = x.len() / dim;
    let n_images = n_interior + 2;
    let mut slot = state
        .eval
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some(prev) = slot.as_ref()
        && prev.endpoints
        && prev.interior == x
    {
        return Ok(prev.clone());
    }
    let mut energy = Array1::zeros(n_images);
    let mut enthalpy = Array1::zeros(n_images);
    let mut gradient = Array2::zeros((n_images, dim));
    let mut stress = vec![[[0.0; 3]; 3]; n_images];
    let mut cartesian = Array2::zeros((n_images, n_cart));
    let mut cells = vec![[[0.0; 3]; 3]; n_images];
    let have_ends = slot.as_ref().is_some_and(|prev| prev.endpoints);
    if have_ends {
        let prev = slot.as_ref().expect("endpoint cache");
        for end in [0, n_images - 1] {
            energy[end] = prev.energy[end];
            enthalpy[end] = prev.enthalpy[end];
            gradient.row_mut(end).assign(&prev.gradient.row(end));
            stress[end] = prev.stress[end];
            cartesian.row_mut(end).assign(&prev.cartesian.row(end));
            cells[end] = prev.cells[end];
        }
    }
    let indices: Vec<usize> = if have_ends {
        (1..n_images - 1).collect()
    } else {
        (0..n_images).collect()
    };
    // Images do not share state. The assembly below still writes them
    // in band order.
    let computed = indices
        .par_iter()
        .map(|&i| {
            let q = if i == 0 {
                q_first
            } else if i + 1 == n_images {
                q_last
            } else {
                x.slice(s![(i - 1) * dim..i * dim])
            };
            eval_image(chart, surface, q).map(|image| (i, image))
        })
        .collect::<Result<Vec<_>, _>>()?;
    state.rows.fetch_add(computed.len(), Ordering::Relaxed);
    for (i, image) in computed {
        energy[i] = image.energy;
        enthalpy[i] = image.enthalpy;
        gradient.row_mut(i).assign(&image.gradient);
        stress[i] = image.stress;
        cartesian.row_mut(i).assign(&image.cartesian);
        cells[i] = image.cell;
    }
    let eval = JointEval {
        interior: x.to_owned(),
        energy,
        enthalpy,
        gradient,
        stress,
        cartesian,
        cells,
        endpoints: true,
    };
    *slot = Some(eval.clone());
    Ok(eval)
}

#[allow(clippy::too_many_arguments)]
fn image_force(
    config: &BandConfig,
    spring: &SpringKind,
    i: usize,
    climb: bool,
    qs: &[Array1<f64>],
    enthalpy: &Array1<f64>,
    gradients: &Array2<f64>,
    n_cart: usize,
) -> Result<Array1<f64>, SaddleError> {
    let diff_next = &qs[i + 1] - &qs[i];
    let diff_prev = &qs[i] - &qs[i - 1];
    let dist_next = diff_next.dot(&diff_next).sqrt();
    let dist_prev = diff_prev.dot(&diff_prev).sqrt();
    let tangent = compute_tangent(
        config.tangent,
        diff_next.view(),
        diff_prev.view(),
        enthalpy[i],
        enthalpy[i - 1],
        enthalpy[i + 1],
    )
    .map_err(|err| SaddleError::Invalid(format!("image {i}: {err}")))?;
    let force = -&gradients.row(i);
    let spring = spring.compute(
        i,
        tangent.view(),
        dist_next,
        dist_prev,
        diff_next.view(),
        diff_prev.view(),
    );
    let mut projected = if climb {
        let dneb = if config.projection == ProjectionKind::DoublyNudged {
            let perpendicular = force_perp(force.view(), tangent.view());
            dneb_component(spring.full.view(), tangent.view(), perpendicular.view())
        } else {
            Array1::zeros(force.len())
        };
        climbing_image_force(force.view(), tangent.view(), dneb.view())
    } else {
        config
            .projection
            .project(force.view(), tangent.view(), &spring)
    };
    if config.remove_translation {
        strip_translation(&mut projected, n_cart);
    }
    Ok(projected)
}

fn climb_target(climbing: bool, max_i: usize) -> Option<usize> {
    // The highest interior image climbs, including when a constrained
    // endpoint lies above it. An endpoint cell is held, so it need not
    // be a minimum of the enthalpy, and the saddle can sit against it.
    climbing.then_some(max_i)
}

fn assemble<S: PointSurface>(
    config: &BandConfig,
    chart: &JointChart,
    state: &JointState,
    surface: &S,
    q_first: ArrayView1<f64>,
    q_last: ArrayView1<f64>,
    x: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>, f64), SaddleError> {
    let eval = load_eval(chart, surface, state, q_first, q_last, x)?;
    let dim = chart.dimension();
    let n_cart = dim - chart.n_strain();
    let n_images = eval.energy.len();
    let mut qs = Vec::with_capacity(n_images);
    qs.push(q_first.to_owned());
    for i in 0..n_images - 2 {
        qs.push(x.slice(s![i * dim..(i + 1) * dim]).to_owned());
    }
    qs.push(q_last.to_owned());
    let mut max_i = 1;
    for i in 2..n_images - 1 {
        if eval.enthalpy[i] > eval.enthalpy[max_i] {
            max_i = i;
        }
    }
    let target = climb_target(config.climbing.is_some(), max_i);
    let climbing = config.climbing.is_some();
    let armed = climbing && state.armed.load(Ordering::Relaxed);
    if armed {
        state
            .ci_index
            .store(target.map_or(-1, |i| i as i64), Ordering::Relaxed);
    }
    let ci_at = if armed { target } else { None };
    let spring = config.spring.at_energies(eval.enthalpy.view());
    let mut projected = Array1::zeros(x.len());
    for i in 1..n_images - 1 {
        let force = image_force(
            config,
            &spring,
            i,
            ci_at == Some(i),
            &qs,
            &eval.enthalpy,
            &eval.gradient,
            n_cart,
        )?;
        projected
            .slice_mut(s![(i - 1) * dim..i * dim])
            .assign(&force);
    }
    let mut max_component = max_abs(&projected);
    let baseline = {
        let mut slot = state
            .baseline
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *slot.get_or_insert(max_component)
    };
    if let Some(ci) = &config.climbing
        && !armed
        && (max_component < baseline * ci.trigger_factor || max_component < ci.trigger_force)
    {
        state.armed.store(true, Ordering::Relaxed);
        state
            .ci_index
            .store(target.map_or(-1, |i| i as i64), Ordering::Relaxed);
        if let Some(ci_i) = target {
            let force = image_force(
                config,
                &spring,
                ci_i,
                true,
                &qs,
                &eval.enthalpy,
                &eval.gradient,
                n_cart,
            )?;
            projected
                .slice_mut(s![(ci_i - 1) * dim..ci_i * dim])
                .assign(&force);
            max_component = max_abs(&projected);
        }
    }
    if !projected.dot(&projected).is_finite() {
        return Err(SaddleError::NonFinite("band force norm"));
    }
    *state
        .last_max_force
        .lock()
        .unwrap_or_else(|poison| poison.into_inner()) = max_component;
    let pseudo = eval.enthalpy.slice(s![1..n_images - 1]).sum();
    Ok((pseudo, projected, max_component))
}

impl JointRun {
    fn pack(&self) -> Array1<f64> {
        let n_images = self.qs.nrows();
        let dim = self.qs.ncols();
        let mut x = Array1::zeros((n_images - 2) * dim);
        for i in 1..n_images - 1 {
            x.slice_mut(s![(i - 1) * dim..i * dim])
                .assign(&self.qs.row(i));
        }
        x
    }

    fn scatter(&mut self, x: ArrayView1<f64>) {
        let n_images = self.qs.nrows();
        let dim = self.qs.ncols();
        for i in 1..n_images - 1 {
            self.qs
                .row_mut(i)
                .assign(&x.slice(s![(i - 1) * dim..i * dim]));
        }
    }

    fn step<S: PointSurface>(&mut self, surface: &S) -> Result<BandReport, SaddleError> {
        let n_images = self.qs.nrows();
        let dim = self.qs.ncols();
        let interior = (n_images - 2) * dim;
        let q_first = self.qs.row(0).to_owned();
        let q_last = self.qs.row(n_images - 1).to_owned();
        self.state.rows.store(0, Ordering::Relaxed);
        self.state.take_error();
        let config = &self.config;
        let chart = &self.chart;
        let state = &self.state;
        let oracle = Oracle::unbounded(interior, |x: ArrayView1<f64>| {
            match assemble(
                config,
                chart,
                state,
                surface,
                q_first.view(),
                q_last.view(),
                x,
            ) {
                Ok((energy, force, _)) => (energy, -force),
                Err(err) => {
                    state.record_error(err);
                    (f64::INFINITY, Array1::zeros(interior))
                }
            }
        });
        let mut x = self.pack();
        let before = x.clone();
        let stepped = self.solver.step(&oracle, &mut x);
        drop(oracle);
        if let Some(err) = self.state.take_error() {
            self.solver.forget_evaluation();
            return Err(err);
        }
        stepped.map_err(|err| SaddleError::Solver(err.to_string()))?;
        if x == before && matches!(self.method, Method::Ode12r) && self.unmoved < 64 {
            self.unmoved += 1;
        } else {
            self.unmoved = 0;
            let max_force = self.state.last_max_force();
            if !x.iter().all(|v| v.is_finite()) {
                return Err(SaddleError::NonFinite("band step"));
            }
            if x == before && max_force > self.config.force_tol {
                return Err(SaddleError::Stalled(format!(
                    "band unchanged with max force {max_force:e} above {:e}",
                    self.config.force_tol
                )));
            }
        }
        self.scatter(x.view());
        self.iteration += 1;
        let cached = self
            .state
            .eval
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_ref()
            .is_some_and(|eval| eval.interior == x);
        if !cached {
            assemble(
                &self.config,
                &self.chart,
                &self.state,
                surface,
                q_first.view(),
                q_last.view(),
                x.view(),
            )?;
        }
        let max_force = self.state.last_max_force();
        let status = if max_force <= self.config.force_tol {
            BandStatus::Converged
        } else {
            BandStatus::Running
        };
        Ok(BandReport {
            status,
            max_force,
            ci_index: self.state.ci(),
            iteration: self.iteration,
            surface_rows: self.state.rows.load(Ordering::Relaxed),
            curvature: None,
        })
    }
}
