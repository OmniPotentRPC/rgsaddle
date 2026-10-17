//! Image dependent pair potential initial path.
//!
//! Goswami, MethodsX (2026), doi:10.1016/j.mex.2026.103899.
//! Smidstrup, Pedersen, Stokbro, and Jonsson, J. Chem. Phys. 140,
//! 214106 (2014), doi:10.1063/1.4878664.
//!
//! Linear interpolation of the Cartesian endpoints can place two atoms
//! on top of each other. The path below keeps the endpoints and relaxes
//! every interior image toward the linearly interpolated pair distances
//! before any surface call. Distances and the endpoint step use the
//! minimum image.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;
use crate::mic::{self, Cell};

/// Per-image stop for [`idpp_path`].
///
/// The defaults are 5000 L-BFGS steps, a gradient L2 norm of `1e-3`,
/// a per-atom step cap of `0.1`, and 20 L-BFGS pairs.
#[derive(Clone, Debug)]
pub struct IdppConfig {
    pub max_iterations: usize,
    pub grad_tol: f64,
    pub max_move: f64,
    pub memory: usize,
}

impl Default for IdppConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            grad_tol: 1e-3,
            max_move: 0.1,
            memory: 20,
        }
    }
}

/// Straight path through the minimum image.
///
/// `n_images` counts the endpoints. Row 0 is `reactant` and the last
/// row is `product`. Interior row `i` is
/// `reactant + (i / (n_images - 1)) * mic(product - reactant)`.
/// `cell` is `None` in free space.
pub fn linear_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
) -> Result<Array2<f64>, SaddleError> {
    check_ends(reactant, product, n_images)?;
    let dof = reactant.len();
    let mut sep = &product - &reactant;
    if let Some(cell) = cell {
        mic::wrap_difference(cell, &mut sep);
    }
    let denom = (n_images - 1) as f64;
    sep /= denom;
    let mut path = Array2::zeros((n_images, dof));
    path.row_mut(0).assign(&reactant);
    path.row_mut(n_images - 1).assign(&product);
    for i in 1..n_images - 1 {
        let row = &reactant + &sep * (i as f64);
        path.row_mut(i).assign(&row);
    }
    Ok(path)
}

/// IDPP path. Same arguments as [`linear_path`].
///
/// Interior image `i` (fraction `xi = i / (n_images - 1)`) is minimized
/// on
///
/// `S = 1/2 sum_{a<b} (r_ab - d_ab)^2 / r_ab^4`,
///
/// where `r_ab` is the current minimum-image distance and `d_ab` is
/// `(1 - xi) r_ab(reactant) + xi r_ab(product)`. A pair closer than
/// `1e-4` is held at that floor so the weight stays finite. The
/// endpoints are copied through. The minimizer is L-BFGS. A step that
/// hits the iteration cap is kept.
pub fn idpp_path(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
    cell: Option<&Cell>,
    config: &IdppConfig,
) -> Result<Array2<f64>, SaddleError> {
    check_config(config)?;
    let mut path = linear_path(reactant, product, n_images, cell)?;
    let d_init = distance_matrix(reactant, cell)?;
    let d_final = distance_matrix(product, cell)?;
    let denom = (n_images - 1) as f64;
    for i in 1..n_images - 1 {
        let xi = (i as f64) / denom;
        let target = &d_init * (1.0 - xi) + &d_final * xi;
        let mut image = path.row(i).to_owned();
        relax_image(&mut image, target.view(), cell, config)?;
        path.row_mut(i).assign(&image);
    }
    Ok(path)
}

fn check_config(config: &IdppConfig) -> Result<(), SaddleError> {
    if config.max_iterations == 0 {
        return Err(SaddleError::Invalid(
            "IDPP needs at least one iteration".into(),
        ));
    }
    if !(config.grad_tol > 0.0 && config.grad_tol.is_finite()) {
        return Err(SaddleError::Invalid(
            "IDPP gradient tolerance must be finite and positive".into(),
        ));
    }
    if !(config.max_move > 0.0 && config.max_move.is_finite()) {
        return Err(SaddleError::Invalid(
            "IDPP max move must be finite and positive".into(),
        ));
    }
    if config.memory == 0 {
        return Err(SaddleError::Invalid("IDPP L-BFGS memory is empty".into()));
    }
    Ok(())
}

fn check_ends(
    reactant: ArrayView1<f64>,
    product: ArrayView1<f64>,
    n_images: usize,
) -> Result<(), SaddleError> {
    let dof = reactant.len();
    if product.len() != dof || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "IDPP endpoints must be matching 3N vectors; got {dof} and {}",
            product.len()
        )));
    }
    if n_images < 3 {
        return Err(SaddleError::Shape(format!(
            "IDPP path needs at least 3 images; got {n_images}"
        )));
    }
    if !reactant.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP reactant"));
    }
    if !product.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP product"));
    }
    Ok(())
}

/// Floor on a pair distance. Below it the `1/r^4` weight is undefined.
const R_FLOOR: f64 = 1e-4;

fn distance_matrix(pos: ArrayView1<f64>, cell: Option<&Cell>) -> Result<Array2<f64>, SaddleError> {
    let n = pos.len() / 3;
    let mut d = Array2::zeros((n, n));
    let mut diff = Array1::zeros(3);
    for i in 0..n {
        for j in (i + 1)..n {
            let r = pair_distance(pos, i, j, cell, &mut diff)?;
            d[(i, j)] = r;
            d[(j, i)] = r;
        }
    }
    Ok(d)
}

fn pair_distance(
    pos: ArrayView1<f64>,
    i: usize,
    j: usize,
    cell: Option<&Cell>,
    diff: &mut Array1<f64>,
) -> Result<f64, SaddleError> {
    for c in 0..3 {
        diff[c] = pos[3 * i + c] - pos[3 * j + c];
    }
    if let Some(cell) = cell {
        mic::wrap_difference(cell, diff);
    }
    let r2 = diff.dot(diff);
    if !r2.is_finite() {
        return Err(SaddleError::NonFinite("IDPP pair distance"));
    }
    Ok(r2.sqrt())
}

fn idpp_value_grad(
    pos: ArrayView1<f64>,
    target: ArrayView2<f64>,
    cell: Option<&Cell>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let n = pos.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(pos.len());
    let mut diff = Array1::zeros(3);
    for i in 0..n {
        for j in (i + 1)..n {
            for c in 0..3 {
                diff[c] = pos[3 * i + c] - pos[3 * j + c];
            }
            if let Some(cell) = cell {
                mic::wrap_difference(cell, &mut diff);
            }
            let mut r = diff.dot(&diff).sqrt();
            if !r.is_finite() {
                return Err(SaddleError::NonFinite("IDPP pair distance"));
            }
            if r < R_FLOOR {
                r = R_FLOOR;
            }
            let delta = r - target[(i, j)];
            let r2 = r * r;
            let r4 = r2 * r2;
            energy += 0.5 * delta * delta / r4;
            // dS/dr = delta * (1 - 2 delta / r) / r^4
            let d_s_dr = (delta * (1.0 - 2.0 * delta / r)) / r4;
            let scale = d_s_dr / r;
            for c in 0..3 {
                let g = scale * diff[c];
                grad[3 * i + c] += g;
                grad[3 * j + c] -= g;
            }
        }
    }
    if !energy.is_finite() || !grad.iter().all(|g: &f64| g.is_finite()) {
        return Err(SaddleError::NonFinite("IDPP objective"));
    }
    Ok((energy, grad))
}

fn relax_image(
    image: &mut Array1<f64>,
    target: ArrayView2<f64>,
    cell: Option<&Cell>,
    config: &IdppConfig,
) -> Result<(), SaddleError> {
    use rgmin::{Accept, Control, Method, Oracle, Solver};

    let dim = image.len();
    let target = target.to_owned();
    let cell = cell.copied();
    let oracle = Oracle::unbounded(dim, move |y: ArrayView1<f64>| {
        match idpp_value_grad(y, target.view(), cell.as_ref()) {
            Ok(pair) => pair,
            Err(_) => (f64::NAN, Array1::from_elem(y.len(), f64::NAN)),
        }
    });
    let mut solver = Solver::new(
        Method::Lbfgs {
            memory: config.memory,
        },
        Control {
            maxiter: config.max_iterations,
            gtol: config.grad_tol,
            istep: 1.0,
            maxmove: None,
            ftol_rel: None,
        },
        dim,
    );
    solver.set_atom_maxmove(config.max_move);
    solver.set_accept(Accept::Energy);
    for _ in 0..config.max_iterations {
        let report = solver
            .step(&oracle, image)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        if !image.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("IDPP image"));
        }
        if report.grad_norm < config.grad_tol {
            return Ok(());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn gradient_matches_a_central_difference() {
        let pos = array![0.0, 0.0, 0.0, 1.2, 0.1, 0.0, 0.4, 1.1, 0.3];
        let cell = Cell::ortho(6.0, 6.0, 6.0).unwrap();
        let mut target = distance_matrix(pos.view(), Some(&cell)).unwrap();
        target *= 1.4;
        let (_e, grad) = idpp_value_grad(pos.view(), target.view(), Some(&cell)).unwrap();
        let eps = 1e-6;
        for k in 0..pos.len() {
            let mut plus = pos.clone();
            let mut minus = pos.clone();
            plus[k] += eps;
            minus[k] -= eps;
            let (ep, _) = idpp_value_grad(plus.view(), target.view(), Some(&cell)).unwrap();
            let (em, _) = idpp_value_grad(minus.view(), target.view(), Some(&cell)).unwrap();
            assert_abs_diff_eq!(grad[k], (ep - em) / (2.0 * eps), epsilon = 1e-6);
        }
    }
}
