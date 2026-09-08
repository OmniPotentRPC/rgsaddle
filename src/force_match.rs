//! Sella `force_match.pyx`: pair-harmonic seed Hessian, in Rust.
//!
//! The Cython matcher fits Buckingham / Morse / LJ / bond terms to a
//! residual gradient. This port is the bond arm: covalent-radius
//! pairs, linear `k_ij` at fixed `r0` (covalent sum), least-squares
//! against the Cartesian force, then the analytic pair Hessian.
//! Hosts that need the other pair kinds call this again with a
//! different pair list; there is no Python hot path.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;

/// Default covalent radii (Å), indexed by Z when known, else 0.70.
pub fn covalent_radius(z: u8) -> f64 {
    match z {
        1 => 0.31,
        6 => 0.76,
        7 => 0.71,
        8 => 0.66,
        9 => 0.57,
        14 => 1.11,
        15 => 1.07,
        16 => 1.05,
        17 => 1.02,
        _ => 0.70,
    }
}

/// Pairs with `r <= scale * (rcov_i + rcov_j)`.
///
/// Sella `force_match.pyx` skips `dij > 1.5 * rcov`, so equality is a
/// bond.
pub fn covalent_pairs(
    x: ArrayView1<f64>,
    z: &[u8],
    scale: f64,
) -> Result<Vec<[usize; 2]>, SaddleError> {
    if !x.len().is_multiple_of(3) || x.len() / 3 != z.len() {
        return Err(SaddleError::Shape(
            "force_match frame is not 3N with matching Z".into(),
        ));
    }
    let n = z.len();
    let mut pairs = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[3 * j] - x[3 * i];
            let dy = x[3 * j + 1] - x[3 * i + 1];
            let dz = x[3 * j + 2] - x[3 * i + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let cut = scale * (covalent_radius(z[i]) + covalent_radius(z[j]));
            if r > 1e-12 && r <= cut {
                pairs.push([i, j]);
            }
        }
    }
    Ok(pairs)
}

/// Fit pair spring constants so `F_ff ≈ -g` in the least-squares sense.
///
/// Pair `i-j` contributes `k (r - r0) u` on `j` and the opposite on
/// `i`, with `r0 = rcov_i + rcov_j`.
pub fn fit_bond_ks(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    pairs: &[[usize; 2]],
) -> Result<Array1<f64>, SaddleError> {
    if g.len() != x.len() {
        return Err(SaddleError::Shape(
            "force_match gradient must match the 3N frame".into(),
        ));
    }
    let nlin = pairs.len();
    let ndof = x.len();
    let mut j = Array2::<f64>::zeros((ndof, nlin));
    for (p, pair) in pairs.iter().enumerate() {
        let (col, _) = pair_force_col(x, z, *pair)?;
        for i in 0..ndof {
            j[(i, p)] = col[i];
        }
    }
    // Normal equations (J^T J) k = J^T (-g).
    let mut a = Array2::<f64>::zeros((nlin, nlin));
    let mut rhs = Array1::zeros(nlin);
    let mut target = g.to_owned();
    target.mapv_inplace(|v| -v);
    for i in 0..nlin {
        for k in 0..nlin {
            let mut acc = 0.0;
            for t in 0..ndof {
                acc += j[(t, i)] * j[(t, k)];
            }
            a[(i, k)] = acc;
        }
        a[(i, i)] += 1e-12;
        let mut acc = 0.0;
        for t in 0..ndof {
            acc += j[(t, i)] * target[t];
        }
        rhs[i] = acc;
    }
    Ok(solve_spd(&a, &rhs))
}

/// Analytic pair-harmonic Hessian at the current geometry.
pub fn bond_hessian(
    x: ArrayView1<f64>,
    z: &[u8],
    pairs: &[[usize; 2]],
    ks: ArrayView1<f64>,
) -> Result<Array2<f64>, SaddleError> {
    if ks.len() != pairs.len() {
        return Err(SaddleError::Shape(
            "force_match k vector must match the pair list".into(),
        ));
    }
    let n = x.len();
    let mut h = Array2::<f64>::zeros((n, n));
    for (p, pair) in pairs.iter().enumerate() {
        let i = pair[0];
        let j = pair[1];
        let dx = [
            x[3 * j] - x[3 * i],
            x[3 * j + 1] - x[3 * i + 1],
            x[3 * j + 2] - x[3 * i + 2],
        ];
        let r2 = dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2];
        if r2 <= f64::MIN_POSITIVE {
            return Err(SaddleError::NonFinite("force_match bond"));
        }
        let r = r2.sqrt();
        let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
        let k = ks[p];
        let u = [dx[0] / r, dx[1] / r, dx[2] / r];
        // H_ii = k [ uu^T + (r-r0)/r (I - uu^T) ]
        let stretch = (r - r0) / r;
        for a in 0..3 {
            for b in 0..3 {
                let hab = k * (u[a] * u[b] + stretch * ((a == b) as i32 as f64 - u[a] * u[b]));
                h[(3 * i + a, 3 * i + b)] += hab;
                h[(3 * j + a, 3 * j + b)] += hab;
                h[(3 * i + a, 3 * j + b)] -= hab;
                h[(3 * j + a, 3 * i + b)] -= hab;
            }
        }
    }
    Ok(h)
}

/// Seed Hessian, fitted bond stiffnesses, and the corresponding atom pairs.
pub type ForceMatchedHessian = (Array2<f64>, Array1<f64>, Vec<[usize; 2]>);

/// Fit `k` and return the seed Hessian in one call.
pub fn force_match_hessian(
    x: ArrayView1<f64>,
    g: ArrayView1<f64>,
    z: &[u8],
    scale: f64,
) -> Result<ForceMatchedHessian, SaddleError> {
    let pairs = covalent_pairs(x, z, scale)?;
    if pairs.is_empty() {
        return Ok((Array2::eye(x.len()), Array1::zeros(0), pairs));
    }
    let ks = fit_bond_ks(x, g, z, &pairs)?;
    let h = bond_hessian(x, z, &pairs, ks.view())?;
    Ok((h, ks, pairs))
}

fn pair_force_col(
    x: ArrayView1<f64>,
    z: &[u8],
    pair: [usize; 2],
) -> Result<(Array1<f64>, f64), SaddleError> {
    let i = pair[0];
    let j = pair[1];
    let dx = [
        x[3 * j] - x[3 * i],
        x[3 * j + 1] - x[3 * i + 1],
        x[3 * j + 2] - x[3 * i + 2],
    ];
    let r = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
    if r <= f64::MIN_POSITIVE {
        return Err(SaddleError::NonFinite("force_match bond"));
    }
    let r0 = covalent_radius(z[i]) + covalent_radius(z[j]);
    let factor = r - r0;
    let mut col = Array1::zeros(x.len());
    for a in 0..3 {
        let u = dx[a] / r;
        col[3 * i + a] = -factor * u;
        col[3 * j + a] = factor * u;
    }
    Ok((col, r))
}

fn solve_spd(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let n = b.len();
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut acc = a[(i, j)];
            for k in 0..j {
                acc -= l[(i, k)] * l[(j, k)];
            }
            if i == j {
                l[(i, i)] = acc.max(1e-30).sqrt();
            } else {
                l[(i, j)] = acc / l[(j, j)];
            }
        }
    }
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let mut acc = b[i];
        for k in 0..i {
            acc -= l[(i, k)] * y[k];
        }
        y[i] = acc / l[(i, i)];
    }
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut acc = y[i];
        for k in (i + 1)..n {
            acc -= l[(k, i)] * x[k];
        }
        x[i] = acc / l[(i, i)];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use rgmin::vecops::dot;

    #[test]
    fn water_pairs_are_the_two_oh_bonds() {
        let x = Array1::from(vec![
            0.0, 0.0, 0.0, // O
            0.96, 0.0, 0.0, // H
            -0.24, 0.93, 0.0, // H
        ]);
        let z = [8u8, 1, 1];
        let pairs = covalent_pairs(x.view(), &z, 1.3).unwrap();
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&[0, 1]));
        assert!(pairs.contains(&[0, 2]));
    }

    #[test]
    fn cutoff_equality_is_a_bond() {
        let rc = covalent_radius(6);
        let scale = 1.5;
        let r = scale * (rc + rc);
        let x = Array1::from(vec![0.0, 0.0, 0.0, r, 0.0, 0.0]);
        let z = [6u8, 6];
        let pairs = covalent_pairs(x.view(), &z, scale).unwrap();
        assert_eq!(pairs, vec![[0, 1]]);
    }

    #[test]
    fn fitted_hessian_is_symmetric_and_finite() {
        let x = Array1::from(vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0]);
        let g = Array1::from(vec![-0.2, 0.0, 0.0, 0.2, 0.0, 0.0]);
        let z = [6u8, 6];
        let (h, ks, pairs) = force_match_hessian(x.view(), g.view(), &z, 1.5).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!(ks[0].is_finite());
        for i in 0..6 {
            for j in 0..6 {
                assert!((h[(i, j)] - h[(j, i)]).abs() < 1e-12);
                assert!(h[(i, j)].is_finite());
            }
        }
        // Residual force along the bond should be smaller than |g|.
        let (col, _) = pair_force_col(x.view(), &z, pairs[0]).unwrap();
        let mut pred = Array1::zeros(6);
        for i in 0..6 {
            pred[i] = ks[0] * col[i];
        }
        let mut target = g.clone();
        target.mapv_inplace(|v| -v);
        let err = {
            let mut d = pred.clone();
            for i in 0..6 {
                d[i] -= target[i];
            }
            dot(d.view(), d.view()).sqrt()
        };
        assert!(err < 1e-8, "LS residual {err}");
    }
}
