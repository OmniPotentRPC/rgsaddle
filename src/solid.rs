//! The cell block of the generalized solid-state band (G-SSNEB) of
//! Sheppard, Xiao, Chemelewski, Johnson and Henkelman, J. Chem. Phys.
//! 136, 074103 (2012), doi:10.1063/1.3684549.
//!
//! A solid-state band row is `3N + 9` long: the Cartesian positions of
//! `N` atoms, then the cell row-major (row `i` is lattice vector
//! `a_i`). Cells are lower triangular (`a_1` on x, `a_2` in the xy
//! plane), which fixes the three rigid rotations of the cell; a host
//! puts its cells in that form before it builds the band.
//!
//! The band acts on the joint vector of each image:
//!
//! * between images `A` and `B` the atomic block is the wrapped
//!   fractional displacement times the mean cell, `ds (h_A + h_B)/2`,
//!   and the cell block is the strain averaged in the two cells,
//!   `J/2 (h_A^-1 + h_B^-1)(h_B - h_A)`;
//! * at an image the atomic block of the force is `F = -g` and the
//!   cell block is `-(V/J)(sigma + p I)`, with the Cauchy stress
//!   `sigma = (1/V) dE/deps` for the right strain `h <- h (I + eps)`
//!   at fixed fractional coordinates.
//!
//! The upper triangle of both cell blocks, the rotational part, is
//! cleared. `J = (V/N)^(1/3) sqrt(N) w` with `V` the mean endpoint
//! volume: the scaled strain has length units comparable to an
//! atomic displacement, and the scaled stress times the scaled strain
//! is work (`validation/ssneb_jacobian.py`). The same tangent, spring,
//! projection and climbing image then act on the long vector, and the
//! projected joint force goes back to positions and cell as the
//! deformation `eps = F_cell / J`: `dr = F_atoms + r eps`,
//! `dh = h eps`.

use ndarray::{Array1, ArrayView1, ArrayViewMut1};

use crate::error::SaddleError;

/// Solid-state band settings. Construct with [`SolidStateConfig::new`]
/// (or `default`) and the `with_*` setters; fields may be added.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SolidStateConfig {
    /// Multiplies the Jacobian (eOn `solid_state_weight`). 1 gives a
    /// cell strain the weight of an atomic displacement.
    pub weight: f64,
    /// Hydrostatic external pressure, energy per length cubed. The
    /// band then relaxes the enthalpy `E + p V`; 0 leaves the
    /// potential-energy surface.
    pub pressure: f64,
}

impl Default for SolidStateConfig {
    fn default() -> Self {
        Self {
            weight: 1.0,
            pressure: 0.0,
        }
    }
}

impl SolidStateConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_pressure(mut self, pressure: f64) -> Self {
        self.pressure = pressure;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), SaddleError> {
        if !(self.weight.is_finite() && self.weight > 0.0) {
            return Err(SaddleError::Invalid(
                "solid-state weight must be positive and finite".into(),
            ));
        }
        if !self.pressure.is_finite() {
            return Err(SaddleError::Invalid(
                "solid-state pressure must be finite".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) type M3 = [[f64; 3]; 3];

/// Upper-triangle entries `(0,1)`, `(0,2)`, `(1,2)`: the cell rotations.
const UPPER: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

fn clear_upper(m: &mut M3) {
    for (r, c) in UPPER {
        m[r][c] = 0.0;
    }
}

pub(crate) fn det3(h: &M3) -> f64 {
    h[0][0] * (h[1][1] * h[2][2] - h[1][2] * h[2][1])
        - h[0][1] * (h[1][0] * h[2][2] - h[1][2] * h[2][0])
        + h[0][2] * (h[1][0] * h[2][1] - h[1][1] * h[2][0])
}

fn inv3(h: &M3) -> Option<M3> {
    let det = det3(h);
    let scale = h.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !(det.is_finite() && det.abs() > 1e-12 * scale.powi(3)) {
        return None;
    }
    let d = 1.0 / det;
    Some([
        [
            (h[1][1] * h[2][2] - h[1][2] * h[2][1]) * d,
            (h[0][2] * h[2][1] - h[0][1] * h[2][2]) * d,
            (h[0][1] * h[1][2] - h[0][2] * h[1][1]) * d,
        ],
        [
            (h[1][2] * h[2][0] - h[1][0] * h[2][2]) * d,
            (h[0][0] * h[2][2] - h[0][2] * h[2][0]) * d,
            (h[0][2] * h[1][0] - h[0][0] * h[1][2]) * d,
        ],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * d,
            (h[0][1] * h[2][0] - h[0][0] * h[2][1]) * d,
            (h[0][0] * h[1][1] - h[0][1] * h[1][0]) * d,
        ],
    ])
}

fn mul3(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|r| std::array::from_fn(|c| (0..3).map(|k| a[r][k] * b[k][c]).sum()))
}

/// Number of atoms of a solid-state row of length `len`.
pub(crate) fn atoms_of(len: usize) -> usize {
    (len - 9) / 3
}

/// The cell stored in the last nine entries of a band row.
pub(crate) fn cell_of(row: ArrayView1<f64>) -> M3 {
    let o = row.len() - 9;
    std::array::from_fn(|r| std::array::from_fn(|c| row[o + 3 * r + c]))
}

pub(crate) fn volume_of(row: ArrayView1<f64>) -> f64 {
    det3(&cell_of(row)).abs()
}

/// A band row's cell must be nonsingular and lower triangular to
/// within `1e-12` of its largest entry.
pub(crate) fn check_row(row: ArrayView1<f64>, image: usize) -> Result<(), SaddleError> {
    let h = cell_of(row);
    let scale = h.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
    if UPPER.iter().any(|&(r, c)| h[r][c].abs() > 1e-12 * scale) {
        return Err(SaddleError::Invalid(format!(
            "image {image}: a solid-state cell must be lower triangular \
             (a1 on x, a2 in the xy plane)"
        )));
    }
    if inv3(&h).is_none() {
        return Err(SaddleError::Invalid(format!(
            "image {image}: singular solid-state cell"
        )));
    }
    Ok(())
}

/// `J = (V/N)^(1/3) sqrt(N) w`, `V` the mean endpoint volume.
pub fn jacobian(mean_volume: f64, n_atoms: usize, weight: f64) -> Result<f64, SaddleError> {
    if !(mean_volume.is_finite() && mean_volume > 0.0) || n_atoms == 0 || !(weight > 0.0) {
        return Err(SaddleError::Invalid(
            "solid-state Jacobian needs a positive volume, atom count and weight".into(),
        ));
    }
    let n = n_atoms as f64;
    Ok((mean_volume / n).cbrt() * n.sqrt() * weight)
}

/// Jacobian of a band from its two endpoint rows.
pub(crate) fn band_jacobian(
    first: ArrayView1<f64>,
    last: ArrayView1<f64>,
    weight: f64,
) -> Result<f64, SaddleError> {
    let mean = 0.5 * (volume_of(first) + volume_of(last));
    jacobian(mean, atoms_of(first.len()), weight)
}

fn wrap_fraction(v: f64) -> f64 {
    let w = v - v.floor();
    if w >= 0.5 { w - 1.0 } else { w }
}

/// Joint displacement from row `from` to row `to`: the wrapped
/// fractional displacement times the mean cell, then the scaled strain
/// averaged in the two cells with its upper triangle cleared.
pub(crate) fn joint_displacement(
    from: ArrayView1<f64>,
    to: ArrayView1<f64>,
    jacobian: f64,
) -> Result<Array1<f64>, SaddleError> {
    let n = atoms_of(from.len());
    let (ha, hb) = (cell_of(from), cell_of(to));
    let (ia, ib) = match (inv3(&ha), inv3(&hb)) {
        (Some(a), Some(b)) => (a, b),
        _ => return Err(SaddleError::Invalid("singular solid-state cell".into())),
    };
    let mean: M3 = std::array::from_fn(|r| std::array::from_fn(|c| 0.5 * (ha[r][c] + hb[r][c])));
    let mut out = Array1::zeros(from.len());
    for a in 0..n {
        let ra = [from[3 * a], from[3 * a + 1], from[3 * a + 2]];
        let rb = [to[3 * a], to[3 * a + 1], to[3 * a + 2]];
        let ds: [f64; 3] = std::array::from_fn(|c| {
            let sa: f64 = (0..3).map(|k| ra[k] * ia[k][c]).sum();
            let sb: f64 = (0..3).map(|k| rb[k] * ib[k][c]).sum();
            wrap_fraction(sb - sa)
        });
        for c in 0..3 {
            out[3 * a + c] = (0..3).map(|k| ds[k] * mean[k][c]).sum();
        }
    }
    let dh: M3 = std::array::from_fn(|r| std::array::from_fn(|c| jacobian * (hb[r][c] - ha[r][c])));
    let (sa, sb) = (mul3(&ia, &dh), mul3(&ib, &dh));
    let mut strain: M3 = std::array::from_fn(|r| std::array::from_fn(|c| 0.5 * (sa[r][c] + sb[r][c])));
    clear_upper(&mut strain);
    write_cell(out.view_mut(), &strain);
    Ok(out)
}

fn write_cell(mut row: ArrayViewMut1<f64>, m: &M3) {
    let o = row.len() - 9;
    for r in 0..3 {
        for c in 0..3 {
            row[o + 3 * r + c] = m[r][c];
        }
    }
}

/// Joint force at an image: `F = -g` on the atoms, `-(V/J)(sigma + p I)`
/// on the cell with the upper triangle cleared. `gradient` is the row
/// the surface returned (gradient, then the Cauchy stress row-major).
pub(crate) fn joint_force(
    row: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    jacobian: f64,
    pressure: f64,
) -> Array1<f64> {
    let volume = volume_of(row);
    let mut force = -&gradient;
    let stress = cell_of(gradient);
    let mut cell: M3 = std::array::from_fn(|r| {
        std::array::from_fn(|c| {
            let p = if r == c { pressure } else { 0.0 };
            -(volume / jacobian) * (stress[r][c] + p)
        })
    });
    clear_upper(&mut cell);
    write_cell(force.view_mut(), &cell);
    force
}

/// Enthalpy `E + p V` of a row.
pub(crate) fn enthalpy(energy: f64, row: ArrayView1<f64>, pressure: f64) -> f64 {
    if pressure == 0.0 {
        energy
    } else {
        energy + pressure * volume_of(row)
    }
}

/// The projected joint force at a row as the move of positions and
/// cell the band solver takes: `eps = F_cell / J` (lower triangular),
/// `dr = F_atoms + r eps`, `dh = h eps`.
pub(crate) fn cartesian_step(row: ArrayView1<f64>, joint: ArrayView1<f64>, jacobian: f64) -> Array1<f64> {
    let n = atoms_of(row.len());
    let mut eps = cell_of(joint);
    for v in eps.iter_mut().flatten() {
        *v /= jacobian;
    }
    clear_upper(&mut eps);
    let mut out = joint.to_owned();
    for a in 0..n {
        for c in 0..3 {
            out[3 * a + c] += (0..3).map(|k| row[3 * a + k] * eps[k][c]).sum::<f64>();
        }
    }
    let mut dh = mul3(&cell_of(row), &eps);
    clear_upper(&mut dh);
    write_cell(out.view_mut(), &dh);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array1;

    fn row(positions: &[f64], cell: M3) -> Array1<f64> {
        let mut v = positions.to_vec();
        v.extend(cell.iter().flatten());
        Array1::from(v)
    }

    #[test]
    fn jacobian_scales_with_the_volume_per_atom() {
        // 8 atoms in 64 A^3: (64/8)^(1/3) sqrt(8) = 2 sqrt 8.
        let j = jacobian(64.0, 8, 1.0).unwrap();
        assert_abs_diff_eq!(j, 2.0 * 8f64.sqrt(), epsilon = 1e-12);
        assert_abs_diff_eq!(jacobian(64.0, 8, 2.5).unwrap(), 2.5 * j, epsilon = 1e-12);
        assert!(jacobian(0.0, 8, 1.0).is_err());
    }

    #[test]
    fn a_fixed_cell_gives_the_minimum_image_displacement_and_no_strain() {
        let h = [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]];
        let a = row(&[0.1, 0.2, 0.3, 3.9, 4.8, 5.9], h);
        let b = row(&[0.3, 0.2, 0.3, 0.1, 0.1, 0.0], h);
        let d = joint_displacement(a.view(), b.view(), 7.0).unwrap();
        let want = [0.2, 0.0, 0.0, 0.2, 0.3, 0.1];
        for (g, w) in d.iter().zip(want) {
            assert_abs_diff_eq!(*g, w, epsilon = 1e-12);
        }
        for v in d.slice(ndarray::s![6..]).iter() {
            assert_abs_diff_eq!(*v, 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn a_homogeneous_strain_is_the_averaged_strain_times_the_jacobian() {
        // h_B = h_A (I + e) with fractional coordinates fixed: the atomic
        // block vanishes, the cell block is J/2 (e + (I+e)^-1 e).
        let ha = [[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.5, 0.2, 4.0]];
        let e = [[0.02, 0.0, 0.0], [0.01, -0.01, 0.0], [0.0, 0.03, 0.015]];
        let ie: M3 = std::array::from_fn(|r| std::array::from_fn(|c| e[r][c] + f64::from(r == c)));
        let hb = mul3(&ha, &ie);
        let ra = [1.0, 1.5, 2.0];
        let rb: Vec<f64> = (0..3).map(|c| (0..3).map(|k| ra[k] * ie[k][c]).sum()).collect();
        let a = row(&ra, ha);
        let b = row(&rb, hb);
        let j = 3.0;
        let d = joint_displacement(a.view(), b.view(), j).unwrap();
        for c in 0..3 {
            assert_abs_diff_eq!(d[c], 0.0, epsilon = 1e-12);
        }
        let back = mul3(&inv3(&ie).unwrap(), &e);
        let got = cell_of(d.view());
        for r in 0..3 {
            for c in 0..3 {
                let want = if c > r { 0.0 } else { 0.5 * j * (e[r][c] + back[r][c]) };
                assert_abs_diff_eq!(got[r][c], want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn the_cell_force_is_minus_the_volume_stress_over_the_jacobian() {
        let h = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        let r = row(&[0.0, 0.0, 0.0], h);
        let s = [[0.1, 0.2, 0.3], [0.2, 0.4, 0.5], [0.3, 0.5, 0.6]];
        let g = row(&[1.0, -2.0, 0.5], s);
        let f = joint_force(r.view(), g.view(), 2.0, 0.01);
        assert_abs_diff_eq!(f[0], -1.0, epsilon = 1e-15);
        let c = cell_of(f.view());
        // V = 24, V/J = 12.
        assert_abs_diff_eq!(c[0][0], -12.0 * 0.11, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1][0], -12.0 * 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(c[2][1], -12.0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[0][1], 0.0);
        assert_abs_diff_eq!(c[1][2], 0.0);
    }

    #[test]
    fn the_step_deforms_positions_with_the_cell() {
        let h = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.0, 0.0, 4.0]];
        let r = row(&[1.0, 1.0, 1.0], h);
        let mut joint = Array1::zeros(12);
        joint[0] = 0.1;
        // eps = F/J with J = 2: eps_00 = 0.05, eps_20 = 0.1.
        joint[3] = 0.1;
        joint[3 + 6] = 0.2;
        let step = cartesian_step(r.view(), joint.view(), 2.0);
        // dr = F + r eps = (0.1 + 1*0.05 + 1*0.1, 0, 0).
        assert_abs_diff_eq!(step[0], 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(step[1], 0.0);
        let dh = cell_of(step.view());
        // dh = h eps: row 0 (0.1,0,0), row 2 (1*0.05 + 4*0.1, 0, 0).
        assert_abs_diff_eq!(dh[0][0], 0.1, epsilon = 1e-15);
        assert_abs_diff_eq!(dh[2][0], 0.45, epsilon = 1e-15);
        assert_abs_diff_eq!(dh[0][1], 0.0);
    }

    #[test]
    fn an_upper_entry_is_refused() {
        let h = [[2.0, 0.1, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
        assert!(check_row(row(&[0.0, 0.0, 0.0], h).view(), 1).is_err());
    }
}
