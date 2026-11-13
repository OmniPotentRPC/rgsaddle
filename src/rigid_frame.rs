//! Rigid translations and rotations for the band, the min-mode climb,
//! and the connect surface.
//!
//! rgmin 0.5.2 exposes [`rgmin::RigidWorkspace`], and its rank test stops
//! at `1e-10` times the leading pivot. That cutoff drops the real
//! rotation of a near-linear molecule and keeps the axial roundoff of a
//! molecule far from the origin. This module is the same column-pivoted
//! QR with the cutoff `max(3N, k) ε max(|R_11|, ||x||_∞)`, rotations
//! about the mass-weighted centre of mass, and frozen atoms zeroed
//! before the rank test.

use ndarray::Array2;

/// One image's rigid frame.
pub(crate) struct RigidFrame;

impl RigidFrame {
    pub(crate) fn new() -> Self {
        Self
    }

    /// Project `force` in place. Below two atoms the force is kept:
    /// a one-atom image has no internal force apart from translation.
    pub(crate) fn project(
        &mut self,
        force: &mut [f64],
        pos: &[f64],
        masses: Option<&[f64]>,
        frozen: Option<&[bool]>,
        rotation: bool,
    ) {
        let n = force.len();
        if n < 6 || n != pos.len() || n % 3 != 0 {
            return;
        }
        let q = modes(pos, masses, frozen, rotation && n >= 6);
        let scale = mass_scale(masses, n / 3, n);
        let mut w = vec![0.0; n];
        for i in 0..n {
            w[i] = force[i] * scale[i];
        }
        let rank = q.len() / n;
        for k in 0..rank {
            let col = &q[k * n..(k + 1) * n];
            let mut dot = 0.0;
            for i in 0..n {
                dot += w[i] * col[i];
            }
            for i in 0..n {
                w[i] -= dot * col[i];
            }
        }
        for i in 0..n {
            if scale[i] > 0.0 {
                force[i] = w[i] / scale[i];
            }
        }
        zero_frozen(force, frozen);
    }

    /// Orthonormal columns, or `None` below two atoms.
    pub(crate) fn basis(
        &mut self,
        pos: &[f64],
        masses: Option<&[f64]>,
        frozen: Option<&[bool]>,
        rotation: bool,
    ) -> Option<Array2<f64>> {
        let n = pos.len();
        if n < 6 || n % 3 != 0 {
            return None;
        }
        let raw = modes(pos, masses, frozen, rotation && n >= 6);
        let nat_rank = if raw.is_empty() { 0 } else { raw.len() / n };
        let scale = mass_scale(masses, n / 3, n);
        let mut q = Array2::zeros((n, nat_rank));
        for k in 0..nat_rank {
            for i in 0..n {
                let cart = if scale[i] > 0.0 {
                    raw[k * n + i] / scale[i]
                } else {
                    0.0
                };
                q[(i, k)] = cart;
            }
        }
        Some(q)
    }
}

fn modes(pos: &[f64], masses: Option<&[f64]>, frozen: Option<&[bool]>, rotation: bool) -> Vec<f64> {
    let n = pos.len();
    let nat = n / 3;
    let k = if rotation { 6 } else { 3 };
    let com = centre_of_mass(pos, masses, nat);
    let mut cols = generators(pos, nat, rotation, com);
    cols = admissible(cols, n, k, frozen, nat, coordinate_scale(pos));
    let ncols = if cols.is_empty() { 0 } else { cols.len() / n };
    if ncols == 0 {
        return Vec::new();
    }
    let scale = mass_scale(masses, nat, n);
    for j in 0..ncols {
        for i in 0..n {
            cols[j * n + i] *= scale[i];
        }
    }
    let qr = col_pivoted_qr(cols, n, ncols, coordinate_scale(pos));
    thin_q(&qr)
}

fn centre_of_mass(pos: &[f64], masses: Option<&[f64]>, nat: usize) -> [f64; 3] {
    let mut com = [0.0; 3];
    let mut total = 0.0;
    let mass_ok = masses.is_some_and(|m| m.len() == nat);
    for atom in 0..nat {
        let mass = if mass_ok {
            masses.unwrap()[atom].max(0.0)
        } else {
            1.0
        };
        total += mass;
        com[0] += mass * pos[3 * atom];
        com[1] += mass * pos[3 * atom + 1];
        com[2] += mass * pos[3 * atom + 2];
    }
    if total > 0.0 {
        com[0] /= total;
        com[1] /= total;
        com[2] /= total;
    }
    com
}

fn coordinate_scale(pos: &[f64]) -> f64 {
    pos.iter().fold(0.0, |scale, value| scale.max(value.abs()))
}

fn generators(pos: &[f64], nat: usize, rotation: bool, com: [f64; 3]) -> Vec<f64> {
    let n = nat * 3;
    let k = if rotation { 6 } else { 3 };
    let mut a = vec![0.0; n * k];
    for axis in 0..3 {
        for atom in 0..nat {
            a[axis * n + 3 * atom + axis] = 1.0;
        }
    }
    if rotation {
        for atom in 0..nat {
            let x = pos[3 * atom] - com[0];
            let y = pos[3 * atom + 1] - com[1];
            let z = pos[3 * atom + 2] - com[2];
            let base = 3 * atom;
            a[3 * n + base + 1] = -z;
            a[3 * n + base + 2] = y;
            a[4 * n + base] = z;
            a[4 * n + base + 2] = -x;
            a[5 * n + base] = -y;
            a[5 * n + base + 1] = x;
        }
    }
    a
}

fn admissible(
    cols: Vec<f64>,
    n: usize,
    k: usize,
    frozen: Option<&[bool]>,
    nat: usize,
    coord: f64,
) -> Vec<f64> {
    let Some(mask) = frozen.filter(|m| m.len() == nat) else {
        return cols;
    };
    let mut rows = Vec::new();
    for (atom, &is_frozen) in mask.iter().enumerate() {
        if is_frozen {
            rows.extend([3 * atom, 3 * atom + 1, 3 * atom + 2]);
        }
    }
    if rows.is_empty() {
        return cols;
    }
    let mf = rows.len();
    let mut constraint = vec![0.0; mf * k];
    for j in 0..k {
        for (r, &i) in rows.iter().enumerate() {
            constraint[j * mf + r] = cols[j * n + i];
        }
    }
    let qr = col_pivoted_qr(constraint, mf, k, coord);
    let nullity = k - qr.rank;
    if nullity == 0 {
        return Vec::new();
    }
    let z = kernel_basis(&qr);
    let mut kept = vec![0.0; n * nullity];
    for j in 0..nullity {
        for t in 0..k {
            let coef = z[j * k + t];
            if coef == 0.0 {
                continue;
            }
            for i in 0..n {
                kept[j * n + i] += coef * cols[t * n + i];
            }
        }
        for &i in &rows {
            kept[j * n + i] = 0.0;
        }
    }
    kept
}

fn mass_scale(masses: Option<&[f64]>, nat: usize, n: usize) -> Vec<f64> {
    let mut scale = vec![1.0; n];
    let Some(masses) = masses.filter(|m| m.len() == nat) else {
        return scale;
    };
    for atom in 0..nat {
        let s = masses[atom].max(0.0).sqrt();
        scale[3 * atom] = s;
        scale[3 * atom + 1] = s;
        scale[3 * atom + 2] = s;
    }
    scale
}

fn zero_frozen(v: &mut [f64], frozen: Option<&[bool]>) {
    let Some(mask) = frozen else {
        return;
    };
    for (atom, &is_frozen) in mask.iter().enumerate() {
        if is_frozen {
            for c in 0..3 {
                let i = 3 * atom + c;
                if i < v.len() {
                    v[i] = 0.0;
                }
            }
        }
    }
}

struct Qr {
    a: Vec<f64>,
    tau: Vec<f64>,
    pivot: Vec<usize>,
    m: usize,
    n: usize,
    rank: usize,
}

#[allow(
    clippy::needless_range_loop,
    reason = "the pivot step indexes the column, the pivot and the reflector together"
)]
fn col_pivoted_qr(mut a: Vec<f64>, m: usize, n: usize, coord_scale: f64) -> Qr {
    let mut pivot: Vec<usize> = (0..n).collect();
    let mut tau = vec![0.0; n];
    let steps = m.min(n);
    for k in 0..steps {
        let mut best_j = k;
        let mut best = -1.0;
        for j in k..n {
            let mut nrm2 = 0.0;
            for i in k..m {
                let v = a[j * m + i];
                nrm2 += v * v;
            }
            if nrm2 > best {
                best = nrm2;
                best_j = j;
            }
        }
        if best_j != k {
            for i in 0..m {
                a.swap(k * m + i, best_j * m + i);
            }
            pivot.swap(k, best_j);
        }
        tau[k] = householder(&mut a[k * m + k..k * m + m]);
        if tau[k] != 0.0 {
            for j in k + 1..n {
                let mut dot = a[j * m + k];
                for i in k + 1..m {
                    dot += a[k * m + i] * a[j * m + i];
                }
                let t = tau[k] * dot;
                a[j * m + k] -= t;
                for i in k + 1..m {
                    a[j * m + i] -= t * a[k * m + i];
                }
            }
        }
    }
    let mut rmax = 0.0_f64;
    for i in 0..steps {
        rmax = rmax.max(a[i * m + i].abs());
    }
    let tol = (m.max(n) as f64) * f64::EPSILON * rmax.max(coord_scale);
    let mut rank = 0;
    while rank < steps && a[rank * m + rank].abs() > tol {
        rank += 1;
    }
    Qr {
        a,
        tau,
        pivot,
        m,
        n,
        rank,
    }
}

fn householder(x: &mut [f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    let mut sigma = 0.0;
    for v in x.iter().skip(1) {
        sigma += v * v;
    }
    let alpha = x[0];
    if sigma == 0.0 {
        if alpha >= 0.0 {
            return 0.0;
        }
        x[0] = -alpha;
        return 2.0;
    }
    let nrm = alpha.hypot(sigma.sqrt());
    let beta = if alpha >= 0.0 { -nrm } else { nrm };
    let tau = (beta - alpha) / beta;
    let inv = 1.0 / (alpha - beta);
    for v in x.iter_mut().skip(1) {
        *v *= inv;
    }
    x[0] = beta;
    tau
}

fn thin_q(qr: &Qr) -> Vec<f64> {
    let m = qr.m;
    let rank = qr.rank;
    let mut q = vec![0.0; m * rank];
    for j in 0..rank {
        q[j * m + j] = 1.0;
    }
    for k in (0..rank).rev() {
        let tau = qr.tau[k];
        if tau == 0.0 {
            continue;
        }
        for j in 0..rank {
            let col = j * m;
            let mut dot = q[col + k];
            for i in k + 1..m {
                dot += qr.a[k * m + i] * q[col + i];
            }
            let t = tau * dot;
            q[col + k] -= t;
            for i in k + 1..m {
                q[col + i] -= t * qr.a[k * m + i];
            }
        }
    }
    q
}

#[allow(
    clippy::needless_range_loop,
    reason = "back-substitution indexes R and the unknown together"
)]
fn kernel_basis(qr: &Qr) -> Vec<f64> {
    let k = qr.n;
    let r = qr.rank;
    let nullity = k - r;
    let mut z = vec![0.0; k * nullity];
    if nullity == 0 {
        return z;
    }
    if r == 0 {
        for i in 0..k {
            z[i * k + qr.pivot[i]] = 1.0;
        }
        return z;
    }
    for j in 0..nullity {
        let pcol = r + j;
        let mut y = vec![0.0; r];
        for i in (0..r).rev() {
            let mut sum = qr.a[pcol * qr.m + i];
            for c in i + 1..r {
                sum -= qr.a[c * qr.m + i] * y[c];
            }
            y[i] = sum / qr.a[i * qr.m + i];
        }
        let col = j * k;
        for i in 0..r {
            z[col + qr.pivot[i]] = -y[i];
        }
        z[col + qr.pivot[pcol]] = 1.0;
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn norm(v: &Array1<f64>) -> f64 {
        v.dot(v).sqrt()
    }

    fn centroid(x: &[f64]) -> [f64; 3] {
        let nat = x.len() / 3;
        let mut c = [0.0; 3];
        for atom in 0..nat {
            c[0] += x[3 * atom];
            c[1] += x[3 * atom + 1];
            c[2] += x[3 * atom + 2];
        }
        let s = nat as f64;
        [c[0] / s, c[1] / s, c[2] / s]
    }

    fn mass_com(x: &[f64], masses: &[f64]) -> [f64; 3] {
        let mut c = [0.0; 3];
        let mut total = 0.0;
        for (atom, &mass) in masses.iter().enumerate() {
            total += mass;
            c[0] += mass * x[3 * atom];
            c[1] += mass * x[3 * atom + 1];
            c[2] += mass * x[3 * atom + 2];
        }
        [c[0] / total, c[1] / total, c[2] / total]
    }

    fn rotation(x: &[f64], omega: [f64; 3], origin: [f64; 3]) -> Array1<f64> {
        let mut v = Array1::zeros(x.len());
        for atom in 0..x.len() / 3 {
            let r = [
                x[3 * atom] - origin[0],
                x[3 * atom + 1] - origin[1],
                x[3 * atom + 2] - origin[2],
            ];
            v[3 * atom] = omega[1] * r[2] - omega[2] * r[1];
            v[3 * atom + 1] = omega[2] * r[0] - omega[0] * r[2];
            v[3 * atom + 2] = omega[0] * r[1] - omega[1] * r[0];
        }
        v
    }

    fn project_vec(
        pos: &[f64],
        mut force: Array1<f64>,
        masses: Option<&[f64]>,
        frozen: Option<&[bool]>,
        rotation: bool,
    ) -> Array1<f64> {
        RigidFrame::new().project(force.as_slice_mut().unwrap(), pos, masses, frozen, rotation);
        force
    }

    #[test]
    fn a_near_linear_rotation_is_removed() {
        let delta = 1.0e-10;
        let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, delta, 0.0];
        let origin = centroid(&pos);
        let mode = rotation(&pos, [1.0, 0.0, 0.0], origin);
        let before = norm(&mode);
        assert!(before > 1e-12 && before < 1e-8, "{before}");
        let left = project_vec(&pos, mode, None, None, true);
        assert!(
            norm(&left) < 1e-6 * before,
            "near-linear rotation survived {before} -> {}",
            norm(&left)
        );
    }

    #[test]
    fn a_far_chain_does_not_keep_the_axial_roundoff() {
        let axis = {
            let v = [1.0 / 3.0, 2.0_f64.sqrt(), std::f64::consts::PI / 5.0];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let shift = [8.0e6, -2.0e6, 4.0e6 * std::f64::consts::PI];
        let nat = 24;
        let mut pos = vec![0.0; 3 * nat];
        for atom in 0..nat {
            let t = -3.0 + 6.0 * atom as f64 / (nat - 1) as f64;
            for c in 0..3 {
                pos[3 * atom + c] = shift[c] + t * axis[c];
            }
        }
        let origin = centroid(&pos);
        let mut trial = [1.0, 0.0, 0.0];
        if axis[0].abs() > 0.9 {
            trial = [0.0, 1.0, 0.0];
        }
        let e1 = {
            let c = [
                axis[1] * trial[2] - axis[2] * trial[1],
                axis[2] * trial[0] - axis[0] * trial[2],
                axis[0] * trial[1] - axis[1] * trial[0],
            ];
            let n = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
            [c[0] / n, c[1] / n, c[2] / n]
        };
        let e2 = [
            axis[1] * e1[2] - axis[2] * e1[1],
            axis[2] * e1[0] - axis[0] * e1[2],
            axis[0] * e1[1] - axis[1] * e1[0],
        ];
        let mut axial = rotation(&pos, axis, origin);
        for mode in [rotation(&pos, e1, origin), rotation(&pos, e2, origin)] {
            let n2 = mode.dot(&mode);
            let coef = axial.dot(&mode) / n2;
            axial = &axial - &mode * coef;
        }
        for axis_i in 0..3 {
            let mut t = Array1::zeros(pos.len());
            for atom in 0..nat {
                t[3 * atom + axis_i] = 1.0;
            }
            let n2 = t.dot(&t);
            let coef = axial.dot(&t) / n2;
            axial = &axial - &t * coef;
        }
        assert!(norm(&axial) > 1e-12, "roundoff column {}", norm(&axial));
        let left = project_vec(&pos, axial.clone(), None, None, true);
        let kept = left.dot(&axial) / axial.dot(&axial);
        assert!(
            kept > 0.5,
            "a spurious axial mode was ranked in, coefficient {kept}"
        );
    }

    #[test]
    fn a_frozen_slab_has_no_rigid_mode() {
        let pos = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.4, 0.6, 1.8];
        let frozen = [true, true, true, false];
        let force = Array1::from(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.15, 0.35,
        ]);
        let left = project_vec(&pos, force, None, Some(&frozen), true);
        assert!(
            left.iter().take(9).all(|c| c.abs() < 1e-12),
            "slab force {left:?}"
        );
        assert!((left[9] - 0.2).abs() < 1e-10, "{left:?}");
        assert!((left[10] + 0.15).abs() < 1e-10, "{left:?}");
        assert!((left[11] - 0.35).abs() < 1e-10, "{left:?}");
        let q = RigidFrame::new()
            .basis(&pos, None, Some(&frozen), true)
            .expect("four atoms");
        assert_eq!(q.ncols(), 0, "frozen slab kept {} modes", q.ncols());
    }

    #[test]
    fn one_frozen_atom_keeps_the_rotations_about_it() {
        let pos = [0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.1, 1.1, 0.0, 0.2, -0.3, 0.9];
        let frozen = [true, false, false, false];
        let spin = rotation(&pos, [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]);
        let left = project_vec(&pos, spin, None, Some(&frozen), true);
        assert!(
            norm(&left) < 1e-8,
            "rotation about the pinned atom survived"
        );
        let mut radial = Array1::zeros(12);
        radial[3] = 1.0;
        let kept = project_vec(&pos, radial, None, Some(&frozen), true);
        assert!(kept.iter().take(3).all(|c| c.abs() < 1e-12));
        assert!(
            (kept[3] - 1.0).abs() < 1e-8,
            "radial push removed {}",
            kept[3]
        );
    }

    #[test]
    fn unequal_masses_centre_rotations_on_the_mass_weighted_com() {
        let pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.0, 0.0];
        let masses = [12.0, 1.0, 1.0];
        let origin = mass_com(&pos, &masses);
        let mode = rotation(&pos, [0.0, 0.0, 1.0], origin);
        let left = project_vec(&pos, mode, Some(&masses), None, true);
        assert!(
            norm(&left) < 1e-8,
            "mass-weighted rotation survived {}",
            norm(&left)
        );
    }
}
