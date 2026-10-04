//! Solid-state band block.
//!
//! Sheppard, Xiao, Chemelewski, Johnson, and Henkelman, J. Chem. Phys.
//! 136, 074103 (2012), put the strain next to the atomic displacement
//! and the stress next to the atomic force. The Jacobian that balances
//! the two blocks is fixed from the mean endpoint volume. The log chart
//! in [`crate::cell_log`] is the Sella cell coordinate; this module does
//! not use it. The 3x3 product, determinant and inverse come from there.

use ndarray::{Array1, ArrayView1};

use crate::cell_log::{M3, m3_det, m3_inv, m3_mul};
use crate::error::SaddleError;

/// True when `value` is not strictly above `floor`; NaN is not above
/// anything, so it fails every bound written with this test.
fn not_above(value: f64, floor: f64) -> bool {
    value.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater)
}

/// `J = (V / N)^{1/3} N^{1/2} weight`, with `V` the mean endpoint volume.
pub fn solid_state_jacobian(
    mean_volume: f64,
    n_atoms: usize,
    weight: f64,
) -> Result<f64, SaddleError> {
    if not_above(mean_volume, 0.0) || n_atoms < 1 || not_above(weight, 0.0) || !weight.is_finite() {
        return Err(SaddleError::Invalid(
            "solid-state Jacobian needs a positive volume, atom count, and weight".into(),
        ));
    }
    let n = n_atoms as f64;
    Ok((mean_volume / n).cbrt() * n.sqrt() * weight)
}

pub(crate) fn zero_upper(cell: &mut M3) {
    cell[0][1] = 0.0;
    cell[0][2] = 0.0;
    cell[1][2] = 0.0;
}

fn m3_scale(s: f64, a: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = s * a[i][j];
        }
    }
    c
}

fn m3_add(a: M3, b: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

fn m3_sub(a: M3, b: M3) -> M3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rotate the lab frame so the first lattice vector lies on x and the
/// second lies in the xy plane. Fractional coordinates and the sign of
/// the volume are unchanged. The upper triangle is the rotational part
/// and is cleared.
pub fn orient_lower_triangular(
    cell: &mut M3,
    positions: &mut Array1<f64>,
) -> Result<(), SaddleError> {
    let a = cell[0];
    let b = cell[1];
    let na = norm3(a);
    if not_above(na, 1e-12) {
        return Err(SaddleError::Invalid(
            "solid-state cell has a vanishing first lattice vector".into(),
        ));
    }
    let e1 = [a[0] / na, a[1] / na, a[2] / na];
    let along = dot3(b, e1);
    let b_perp = [
        b[0] - along * e1[0],
        b[1] - along * e1[1],
        b[2] - along * e1[2],
    ];
    let nb = norm3(b_perp);
    if not_above(nb, 1e-12) {
        return Err(SaddleError::Invalid(
            "solid-state cell has parallel lattice vectors".into(),
        ));
    }
    let e2 = [b_perp[0] / nb, b_perp[1] / nb, b_perp[2] / nb];
    let e3 = cross3(e1, e2);
    let rotation = [
        [e1[0], e2[0], e3[0]],
        [e1[1], e2[1], e3[1]],
        [e1[2], e2[2], e3[2]],
    ];
    let mut oriented = m3_mul(*cell, rotation);
    zero_upper(&mut oriented);
    if not_above(m3_det(oriented).abs(), 1e-18) {
        return Err(SaddleError::Invalid(
            "solid-state orientation produced a singular cell".into(),
        ));
    }
    if positions.len().is_multiple_of(3) {
        for atom in 0..positions.len() / 3 {
            let r = [
                positions[3 * atom],
                positions[3 * atom + 1],
                positions[3 * atom + 2],
            ];
            for c in 0..3 {
                positions[3 * atom + c] =
                    r[0] * rotation[0][c] + r[1] * rotation[1][c] + r[2] * rotation[2][c];
            }
        }
    }
    *cell = oriented;
    Ok(())
}

fn wrap_frac(value: f64) -> f64 {
    let mut wrapped = value - value.floor();
    if wrapped >= 0.5 {
        wrapped -= 1.0;
    }
    wrapped
}

fn fractional(position: ArrayView1<f64>, cell: M3) -> Result<Array1<f64>, SaddleError> {
    let inv = m3_inv(cell)?;
    let n = position.len() / 3;
    let mut out = Array1::zeros(position.len());
    for atom in 0..n {
        let r = [
            position[3 * atom],
            position[3 * atom + 1],
            position[3 * atom + 2],
        ];
        for c in 0..3 {
            out[3 * atom + c] = r[0] * inv[0][c] + r[1] * inv[1][c] + r[2] * inv[2][c];
        }
    }
    Ok(out)
}

/// Displacement of `to` relative to `from` in the joint metric.
///
/// Atomic rows are fractional minimum-image displacements mapped
/// through the average cell. The cell block is the averaged right
/// strain of the Jacobian-scaled cell difference, with the upper
/// triangle cleared.
pub fn joint_displacement(
    h_from: M3,
    pos_from: ArrayView1<f64>,
    h_to: M3,
    pos_to: ArrayView1<f64>,
    jacobian: f64,
) -> Result<(Array1<f64>, M3), SaddleError> {
    if not_above(jacobian, 0.0) {
        return Err(SaddleError::Invalid(
            "solid-state displacement needs a positive Jacobian".into(),
        ));
    }
    if pos_from.len() != pos_to.len() || !pos_from.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "solid-state displacement positions differ".into(),
        ));
    }
    let mut frac = &fractional(pos_to, h_to)? - &fractional(pos_from, h_from)?;
    for value in frac.iter_mut() {
        *value = wrap_frac(*value);
    }
    let average = m3_scale(0.5, m3_add(h_from, h_to));
    let mut atomic = Array1::zeros(frac.len());
    for atom in 0..frac.len() / 3 {
        let s = [frac[3 * atom], frac[3 * atom + 1], frac[3 * atom + 2]];
        for c in 0..3 {
            atomic[3 * atom + c] =
                s[0] * average[0][c] + s[1] * average[1][c] + s[2] * average[2][c];
        }
    }
    let dh = m3_scale(jacobian, m3_sub(h_to, h_from));
    let left = m3_mul(m3_inv(h_from)?, dh);
    let right = m3_mul(m3_inv(h_to)?, dh);
    let mut cell = m3_scale(0.5, m3_add(left, right));
    zero_upper(&mut cell);
    Ok((atomic, cell))
}

/// Cell part of the band force: `-(V / J) (σ + P I)`, upper triangle cleared.
///
/// `cauchy` is the Cauchy stress. `pressure` is hydrostatic and is added
/// before the volume factor, so a positive pressure pushes the cell inward.
pub fn cell_neb_force(
    cauchy: M3,
    volume: f64,
    jacobian: f64,
    pressure: f64,
) -> Result<M3, SaddleError> {
    if not_above(volume, 0.0) || not_above(jacobian, 0.0) || !pressure.is_finite() {
        return Err(SaddleError::Invalid(
            "solid-state cell force needs a positive volume and Jacobian".into(),
        ));
    }
    let mut stress = cauchy;
    for (i, row) in stress.iter_mut().enumerate() {
        row[i] += pressure;
    }
    let mut force = m3_scale(-(volume / jacobian), stress);
    zero_upper(&mut force);
    Ok(force)
}

/// `E + P Tr(h0^{-1} (h - h0)) V0`. A zero pressure returns `energy`.
pub fn solid_state_enthalpy(
    energy: f64,
    cell: M3,
    reference: M3,
    pressure: f64,
) -> Result<f64, SaddleError> {
    if pressure == 0.0 {
        return Ok(energy);
    }
    if !pressure.is_finite() || !energy.is_finite() {
        return Err(SaddleError::NonFinite("solid-state enthalpy"));
    }
    let strain = m3_mul(m3_inv(reference)?, m3_sub(cell, reference));
    let volume = m3_det(reference).abs();
    Ok(energy + pressure * (strain[0][0] + strain[1][1] + strain[2][2]) * volume)
}

/// One steepest step in Cartesian coordinates and the cell matrix.
///
/// `delta_h = h (F_cell / J)` and `delta_r = F_atomic + r (F_cell / J)`.
/// The upper triangle of `delta_h` is zero.
pub fn cartesian_step(
    positions: ArrayView1<f64>,
    cell: M3,
    atomic_force: ArrayView1<f64>,
    cell_force: M3,
    jacobian: f64,
) -> Result<(Array1<f64>, M3), SaddleError> {
    if not_above(jacobian, 0.0) {
        return Err(SaddleError::Invalid(
            "solid-state step needs a positive Jacobian".into(),
        ));
    }
    if positions.len() != atomic_force.len() {
        return Err(SaddleError::Shape(
            "solid-state step force length mismatch".into(),
        ));
    }
    let mut strain = m3_scale(1.0 / jacobian, cell_force);
    zero_upper(&mut strain);
    let mut delta_h = m3_mul(cell, strain);
    zero_upper(&mut delta_h);
    let mut delta_r = atomic_force.to_owned();
    for atom in 0..positions.len() / 3 {
        let r = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            delta_r[3 * atom + c] +=
                r[0] * strain[0][c] + r[1] * strain[1][c] + r[2] * strain[2][c];
        }
    }
    Ok((delta_r, delta_h))
}

pub(crate) fn pack_joint(atomic: ArrayView1<f64>, cell: M3) -> Array1<f64> {
    let mut out = Array1::zeros(atomic.len() + 9);
    out.slice_mut(ndarray::s![..atomic.len()]).assign(&atomic);
    let flat = [
        cell[0][0], cell[0][1], cell[0][2], cell[1][0], cell[1][1], cell[1][2], cell[2][0],
        cell[2][1], cell[2][2],
    ];
    for (i, value) in flat.into_iter().enumerate() {
        out[atomic.len() + i] = value;
    }
    out
}

pub(crate) fn unpack_cell(flat: &[f64]) -> M3 {
    let mut cell = [[0.0; 3]; 3];
    for i in 0..9 {
        cell[i / 3][i % 3] = flat[i];
    }
    zero_upper(&mut cell);
    cell
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn diag(x: f64, y: f64, z: f64) -> M3 {
        [[x, 0.0, 0.0], [0.0, y, 0.0], [0.0, 0.0, z]]
    }

    #[test]
    fn a_nan_bound_input_is_refused() {
        let nan = f64::NAN;
        assert!(solid_state_jacobian(nan, 8, 1.0).is_err());
        assert!(solid_state_jacobian(64.0, 8, nan).is_err());
        let id = diag(1.0, 1.0, 1.0);
        assert!(cell_neb_force(id, nan, 1.0, 0.0).is_err());
        assert!(cell_neb_force(id, 1.0, nan, 0.0).is_err());
        let pos = array![0.0, 0.0, 0.0];
        assert!(cartesian_step(pos.view(), id, pos.view(), id, nan).is_err());
        assert!(joint_displacement(id, pos.view(), id, pos.view(), nan).is_err());
        let mut cell = [[nan, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut positions = array![0.0, 0.0, 0.0];
        assert!(orient_lower_triangular(&mut cell, &mut positions).is_err());
    }

    #[test]
    fn jacobian_is_the_cell_volume_scaling() {
        let value = solid_state_jacobian(64.0, 8, 2.0).unwrap();
        assert_abs_diff_eq!(value, 8.0 * 2.0_f64.sqrt(), epsilon = 1e-12);
        assert!(solid_state_jacobian(0.0, 8, 1.0).is_err());
    }

    #[test]
    fn diagonal_cell_orientation_keeps_positions() {
        let mut cell = diag(5.0, 4.0, 6.0);
        let mut positions = array![1.0, 2.0, 3.0, 0.5, 0.2, 0.1];
        let held = positions.clone();
        orient_lower_triangular(&mut cell, &mut positions).unwrap();
        assert_abs_diff_eq!(cell[0][0], 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cell[1][1], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cell[2][2], 6.0, epsilon = 1e-12);
        for (left, right) in positions.iter().zip(held.iter()) {
            assert_abs_diff_eq!(left, right, epsilon = 1e-12);
        }
        assert_abs_diff_eq!(cell[0][1], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn cell_force_divides_the_stress_by_the_jacobian() {
        let cauchy = diag(1.0, 0.0, 0.0);
        let force = cell_neb_force(cauchy, 8.0, 2.0, 0.5).unwrap();
        // σ + P I = diag(1.5, 0.5, 0.5); -(V/J) = -4.
        assert_abs_diff_eq!(force[0][0], -6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(force[1][1], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(force[2][2], -2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(force[0][1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(force[1][0], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn averaged_strain_scales_the_cell_difference_by_the_jacobian() {
        let h_from = diag(4.0, 4.0, 4.0);
        let h_to = diag(6.0, 4.0, 4.0);
        let (atomic, cell) = joint_displacement(
            h_from,
            array![2.0, 2.0, 2.0].view(),
            h_to,
            array![3.0, 2.0, 2.0].view(),
            2.0,
        )
        .unwrap();
        // Same fractional coordinate: the atomic block is zero.
        // dh = J (h_to - h_from) = diag(4, 0, 0).
        // 0.5 (h_from^{-1} dh + h_to^{-1} dh) = diag(5/6, 0, 0).
        assert_abs_diff_eq!(atomic[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cell[0][0], 5.0 / 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cell[1][1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cell[0][1], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn equal_cells_reproduce_the_cartesian_minimum_image() {
        let h = diag(10.0, 8.0, 6.0);
        let (atomic, cell) = joint_displacement(
            h,
            array![0.0, 0.0, 0.0].view(),
            h,
            array![9.0, -5.0, 2.0].view(),
            3.0,
        )
        .unwrap();
        assert_abs_diff_eq!(atomic[0], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(atomic[1], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(atomic[2], 2.0, epsilon = 1e-12);
        for row in cell {
            for value in row {
                assert_abs_diff_eq!(value, 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn positive_pressure_adds_only_the_trace_term() {
        let reference = diag(5.0, 4.0, 4.0);
        let cell = diag(6.0, 4.0, 4.0);
        let plain = solid_state_enthalpy(1.5, cell, reference, 0.0).unwrap();
        assert_abs_diff_eq!(plain, 1.5, epsilon = 1e-15);
        // h0^{-1} (h - h0) = diag(0.2, 0, 0), Tr = 0.2, V0 = 80.
        let enthalpy = solid_state_enthalpy(1.5, cell, reference, 0.02).unwrap();
        assert_abs_diff_eq!(enthalpy, 1.5 + 0.02 * 0.2 * 80.0, epsilon = 1e-12);
    }
}
