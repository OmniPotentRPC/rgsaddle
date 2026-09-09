//! Basin constrained dimer forces from Xiao et al., JCP 141, 164111 (2014).
use std::cell::{Cell, RefCell};
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rgmin::{ApplyHessian, EigenParams, EigensolverKind};
use crate::SaddleError;

#[derive(Clone, Debug)]
pub struct KappaDimerConfig {
    /// Switching length in the position units; beta * kappa is dimensionless.
    pub beta: f64,
    pub eigen: EigenParams,
}

impl Default for KappaDimerConfig {
    fn default() -> Self {
        Self { beta: 5.0, eigen: EigenParams {
            kind: EigensolverKind::Dimer, tol: 1e-6, max_iter: 100,
            ..EigenParams::default()
        }}
    }
}

#[derive(Clone, Debug)]
pub struct KappaDimerForce {
    pub force: Array1<f64>,
    pub tangent_mode: Array1<f64>,
    pub tangent_dimension: usize,
    pub tangent_curvature: f64,
    pub kappa: f64,
    pub gamma_parallel: f64,
    pub gamma_perpendicular: f64,
    pub residual: f64,
    pub actions: usize,
}

/// Equation (6) in the complement of the force and the excluded modes,
/// followed by the equation (7) force. Rows of `excluded` are Cartesian
/// symmetry or constraint directions. The Hessian callback receives full
/// Cartesian vectors; it need not assemble a matrix.
pub fn kappa_dimer_force<F>(
    gradient: ArrayView1<f64>, mode: ArrayView1<f64>, excluded: ArrayView2<f64>,
    apply_hessian: F, config: &KappaDimerConfig,
) -> Result<KappaDimerForce, SaddleError>
where F: Fn(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    let n = gradient.len();
    if n == 0 || mode.len() != n || excluded.ncols() != n {
        return Err(SaddleError::Shape("kappa gradient, mode and excluded directions".into()));
    }
    if !config.beta.is_finite() || config.beta <= 0.0 ||
        !config.eigen.tol.is_finite() || config.eigen.tol <= 0.0 {
        return Err(SaddleError::Solver("kappa beta and eigensolver tolerance must be positive and finite".into()));
    }
    if !gradient.iter().chain(mode.iter()).chain(excluded.iter()).all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("kappa input"));
    }
    let mut space = Complement::new(n);
    for direction in excluded.rows() {
        space.exclude(direction);
    }
    let g = space.project(gradient);
    let force_norm = norm(g.view());
    let mut tau = space.project(mode);
    let mode_norm = norm(tau.view());
    if mode_norm == 0.0 {
        return Err(SaddleError::Shape("kappa mode has no unconstrained component".into()));
    }
    tau /= mode_norm;
    let parallel = &tau * g.dot(&tau);
    let mut result = KappaDimerForce {
        force: Array1::zeros(n), tangent_mode: Array1::zeros(n),
        tangent_dimension: n-space.reflectors.len(), tangent_curvature: 0.0,
        kappa: 0.0, gamma_parallel: 0.0, gamma_perpendicular: 0.5,
        residual: 0.0, actions: 0,
    };
    if force_norm == 0.0 { return Ok(result); }
    space.exclude(g.view());
    result.tangent_dimension = space.dimension();
    if space.dimension() == 0 {
        result.kappa = f64::NEG_INFINITY;
        result.gamma_parallel = 1.0;
        result.gamma_perpendicular = 0.0;
        result.force = parallel;
        return Ok(result);
    }
    let h = ReducedHessian {
        space: &space, action: &apply_hessian, failure: RefCell::new(None), actions: Cell::new(0),
    };
    // A mixed seed excites tangent modes even if tau is parallel to the force.
    let seed = Array1::from_iter((0..space.dimension()).map(|i| ((i+1) as f64 * 1.618033988749895).sin()));
    let eigenpair = rgmin::lowest_mode(&h, Array1::zeros(space.dimension()).view(), seed.view(), &config.eigen);
    if let Some(error) = h.failure.take() { return Err(error); }
    let eigenpair = eigenpair.map_err(|error| SaddleError::Solver(error.to_string()))?;
    let mut c = eigenpair.vector;
    let c_norm = norm(c.view());
    if !c_norm.is_finite() || c_norm == 0.0 {
        return Err(SaddleError::NonFinite("kappa tangent mode"));
    }
    c /= c_norm;
    let hc = h.apply_hessian(Array1::zeros(c.len()).view(), c.view());
    if let Some(error) = h.failure.take() { return Err(error); }
    let nu = c.dot(&hc);
    let residual = norm((&hc-&(&c*nu)).view());
    if !nu.is_finite() || !residual.is_finite() {
        return Err(SaddleError::NonFinite("kappa tangent eigenpair"));
    }
    if residual > config.eigen.tol * norm(hc.view()).max(1.0) {
        return Err(SaddleError::Solver(format!("kappa tangent eigenpair residual {residual} exceeds tolerance")));
    }
    result.tangent_mode = space.lift(c.view());
    result.tangent_curvature = nu;
    result.kappa = -nu / force_norm;
    (result.gamma_parallel, result.gamma_perpendicular) = coefficients(config.beta * result.kappa);
    result.force = -(&g-&parallel)*result.gamma_perpendicular + parallel*result.gamma_parallel;
    result.residual = residual;
    result.actions = h.actions.get();
    if !result.force.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("kappa force"));
    }
    Ok(result)
}

fn norm(v: ArrayView1<f64>) -> f64 {
    v.iter().fold(0.0_f64, |a, &b| a.hypot(b))
}

/// Equation (7), retaining small parallel coefficients around kappa=0.
fn coefficients(z: f64) -> (f64, f64) {
    if z >= 0.0 {
        let e = (-z).exp();
        ((-z).exp_m1() / (1.0+e), 1.0/(1.0+e))
    } else {
        let e = z.exp();
        (-z.exp_m1() / (1.0+e), e/(1.0+e))
    }
}

/// The trailing columns of a product of Householder reflections. Each
/// excluded direction consumes one leading coordinate. Storage is O(n*r)
/// for r excluded directions; Hessian vectors stay in Cartesian coordinates.
struct Complement {
    n: usize,
    reflectors: Vec<Array1<f64>>,
}
impl Complement {
    fn new(n: usize) -> Self { Self { n, reflectors: Vec::new() } }
    fn dimension(&self) -> usize { self.n - self.reflectors.len() }
    fn reflect(v: &mut Array1<f64>, offset: usize, axis: &Array1<f64>) {
        let dot = v.slice(s![offset..]).dot(axis);
        for (i, &a) in axis.iter().enumerate() { v[offset+i] -= 2.0*dot*a; }
    }
    fn forward(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut out=v.to_owned();
        for (offset,axis) in self.reflectors.iter().enumerate() {
            Self::reflect(&mut out,offset,axis);
        }
        out
    }
    fn reduce(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.forward(v).slice(s![self.reflectors.len()..]).to_owned()
    }
    fn lift(&self, v: ArrayView1<f64>) -> Array1<f64> {
        let mut out=Array1::zeros(self.n);
        out.slice_mut(s![self.reflectors.len()..]).assign(&v);
        for (offset,axis) in self.reflectors.iter().enumerate().rev() {
            Self::reflect(&mut out,offset,axis);
        }
        out
    }
    fn project(&self, v: ArrayView1<f64>) -> Array1<f64> {
        self.lift(self.reduce(v).view())
    }
    fn exclude(&mut self, v: ArrayView1<f64>) {
        let input_norm=norm(v);
        if input_norm == 0.0 || self.dimension() == 0 { return; }
        let unit=&v/input_norm;
        let mut axis=self.reduce(unit.view());
        let tail_norm=norm(axis.view());
        // Dependent rotations of a linear molecule do not reduce its dimension.
        if tail_norm <= 64.0*f64::EPSILON { return; }
        axis /= tail_norm;
        axis[0] += if axis[0] >= 0.0 { 1.0 } else { -1.0 };
        let axis_norm=norm(axis.view());
        axis /= axis_norm;
        self.reflectors.push(axis);
    }
}

struct ReducedHessian<'a, F> {
    space: &'a Complement,
    action: &'a F,
    failure: RefCell<Option<SaddleError>>,
    actions: Cell<usize>,
}
impl<F> ApplyHessian for ReducedHessian<'_, F>
where F: Fn(ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
    fn apply_hessian(&self, _: ArrayView1<f64>, v: ArrayView1<f64>) -> Array1<f64> {
        self.actions.set(self.actions.get()+1);
        let hv=(self.action)(self.space.lift(v).view()).and_then(|hv| {
            if hv.len()!=self.space.n { return Err(SaddleError::Shape("kappa Hessian action".into())); }
            if !hv.iter().all(|v| v.is_finite()) { return Err(SaddleError::NonFinite("kappa Hessian action")); }
            Ok(self.space.reduce(hv.view()))
        });
        match hv {
            Ok(v) => v,
            Err(error) => { self.failure.replace(Some(error)); Array1::zeros(v.len()) }
        }
    }
}
