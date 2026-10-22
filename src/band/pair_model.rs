//! Model Hessian of one band image.
//!
//! The Krylov vectors of a Lanczos estimate carry measured Hessian
//! actions; [`impose`] makes the model match them exactly.

use ndarray::{Array1, Array2};

/// Make `b` match every measured pair `(v, Hv)` of `block`: with
/// `V` the orthonormalised vectors and `R = HV - BV`,
/// `B <- B + R V' + V R' - V (V'R) V'`. Directions orthogonal to the
/// block keep their model curvature.
pub(super) fn impose(b: &mut Array2<f64>, block: &[(Array1<f64>, Array1<f64>)]) {
    let n = b.nrows();
    let mut vs: Vec<Array1<f64>> = Vec::new();
    let mut hs: Vec<Array1<f64>> = Vec::new();
    for (v0, h0) in block {
        let mut v = v0.clone();
        let mut h = h0.clone();
        for (q, hq) in vs.iter().zip(&hs) {
            let c = q.dot(&v);
            v -= &(q * c);
            h -= &(hq * c);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-8 {
            vs.push(v / norm);
            hs.push(h / norm);
        }
    }
    if vs.is_empty() {
        return;
    }
    let mut v = Array2::zeros((n, vs.len()));
    let mut hv = Array2::zeros((n, vs.len()));
    for (j, (a, h)) in vs.iter().zip(&hs).enumerate() {
        v.column_mut(j).assign(a);
        hv.column_mut(j).assign(h);
    }
    let r = &hv - &b.dot(&v);
    let m = v.t().dot(&r);
    let next = &*b + &r.dot(&v.t()) + &v.dot(&r.t()) - &v.dot(&m).dot(&v.t());
    *b = (&next + &next.t()) * 0.5;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_imposed_model_matches_its_block() {
        let n = 6;
        let mut b = Array2::eye(n);
        let h = Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j {
                1.0 + i as f64
            } else {
                0.1 / (1.0 + (i + j) as f64)
            }
        });
        let v1 = Array1::from(vec![1.0, 0.5, 0.0, 0.0, 0.2, 0.0]);
        let v2 = Array1::from(vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.3]);
        let block = vec![(v1.clone(), h.dot(&v1)), (v2.clone(), h.dot(&v2))];
        impose(&mut b, &block);
        for (v, hv) in &block {
            let bv = b.dot(v);
            assert!((&bv - hv).iter().all(|e| e.abs() < 1e-12), "{bv} vs {hv}");
        }
        assert!((&b - &b.t()).iter().all(|v| v.abs() < 1e-14));
        let mut off = Array1::zeros(n);
        off[3] = 1.0;
        assert!((off.dot(&b.dot(&off)) - 1.0).abs() < 1e-12);
    }
}
