#![expect(
    clippy::needless_range_loop,
    reason = "the SQNM subspace products follow the reference index loops"
)]
#![expect(
    clippy::too_many_arguments,
    reason = "ConstrainedOpt::step matches the COMPASS argument list"
)]

//! SQNM with linear constraints, as in COMPASS `ConstrainedOpt`.
//!
//! Sommer-Jörgensen and Goedecker, J. Chem. Phys. 160, 064110 (2024).
//! Schaefer, Ghasemi, Roy, and Goedecker, J. Chem. Phys. 142, 034112 (2015).

use faer::Mat;

#[derive(Clone)]
pub struct ConstrainedOpt {
    n: usize,
    alpha0: f64,
    maxrise: f64,
    nhistx: usize,
    trustradius: Option<f64>,
    alpha: f64,
    position: Vec<Vec<f64>>,
    force: Vec<Vec<f64>>,
    energy_old: Option<f64>,
    nhist: usize,
    pub trustradius_applied: bool,
}

pub struct Step {
    pub position: Vec<f64>,
    pub displacement: Vec<f64>,
    pub displacement2: Option<Vec<f64>>,
    pub ndim: usize,
}

impl ConstrainedOpt {
    pub fn new(
        n: usize,
        alpha0: f64,
        maxrise: f64,
        nhistx: usize,
        trustradius: Option<f64>,
    ) -> Self {
        let cols = nhistx + 1;
        Self {
            n,
            alpha0,
            maxrise,
            nhistx,
            trustradius,
            alpha: alpha0,
            position: vec![vec![0.0; n]; cols],
            force: vec![vec![0.0; n]; cols],
            energy_old: None,
            nhist: 0,
            trustradius_applied: false,
        }
    }

    pub fn reset_history(&mut self) {
        self.nhist = 0;
        self.alpha = self.alpha0;
    }

    pub fn step(
        &mut self,
        pos: &[f64],
        energy: f64,
        force: &[f64],
        projection_w: &[Vec<f64>],
        projection_b: &[f64],
        constraint_w_in: Option<&[f64]>,
        constraint_b_in: f64,
        curv2: f64,
    ) -> Step {
        let n = self.n;
        self.position[self.nhist].copy_from_slice(pos);
        self.force[self.nhist].copy_from_slice(force);

        let mut incomplete = false;
        if !projection_w.is_empty() {
            let (gram, _) = gram_factor(projection_w);
            for i in 0..=self.nhist {
                let mut r = mat_transpose_vec(projection_w, &self.position[i]);
                for (rk, bk) in r.iter_mut().zip(projection_b) {
                    *rk -= *bk;
                }
                let solved = solve_spd(&gram, &r);
                let mut delta = mat_vec(projection_w, &solved);
                for d in &mut delta {
                    *d = -*d;
                }
                if i == self.nhist
                    && let Some(radius) = self.trustradius
                {
                    let maxd = max_atom(&delta);
                    if maxd > radius {
                        let scale = radius / maxd;
                        for d in &mut delta {
                            *d *= scale;
                        }
                        incomplete = true;
                    }
                }
                for k in 0..n {
                    self.position[i][k] += delta[k];
                }
                let r0 = mat_transpose_vec(projection_w, &self.force[i]);
                let solved = solve_spd(&gram, &r0);
                let corr = mat_vec(projection_w, &solved);
                for k in 0..n {
                    self.force[i][k] -= corr[k];
                }
            }
        }

        let mut constraint_w = constraint_w_in.map(|w| w.to_vec());
        let mut _constraint_b = constraint_b_in;
        if let Some(w_in) = constraint_w_in
            && !projection_w.is_empty()
        {
            let (gram, _) = gram_factor(projection_w);
            let r0 = mat_transpose_vec(projection_w, w_in);
            let solved = solve_spd(&gram, &r0);
            let corr = mat_vec(projection_w, &solved);
            let mut w = w_in.to_vec();
            for k in 0..n {
                w[k] -= corr[k];
            }
            let coeff = dot(projection_b, &solved);
            _constraint_b -= coeff;
            constraint_w = Some(w);
        }

        if incomplete {
            self.alpha = self.alpha0;
            let pos_now = self.position[self.nhist].clone();
            let force_now = self.force[self.nhist].clone();
            self.position[0] = pos_now;
            self.force[0] = force_now;
            self.nhist = 0;
        }

        let unconstrained = constraint_w_in.is_none() && projection_b.is_empty();
        if unconstrained
            && let Some(old) = self.energy_old
            && energy - old > self.maxrise
            && self.nhist > 0
        {
            self.alpha = (0.5 * self.alpha).max(self.alpha0);
            let prev = self.nhist - 1;
            let pos_prev = self.position[prev].clone();
            let force_prev = self.force[prev].clone();
            self.position[0] = pos_prev;
            self.force[0] = force_prev;
            self.nhist = 0;
            self.energy_old = None;
        } else {
            self.energy_old = Some(energy);
        }

        if self.nhist > 0 {
            let mut v1 = self.force[self.nhist - 1].clone();
            let mut v2 = self.force[self.nhist].clone();
            if let Some(w) = constraint_w.as_ref() {
                let wn = dot(w, w);
                if wn > 0.0 {
                    let c1 = dot(w, &v1) / wn;
                    let c2 = dot(w, &v2) / wn;
                    for k in 0..n {
                        v1[k] -= c1 * w[k];
                        v2[k] -= c2 * w[k];
                    }
                }
            }
            let n1 = norm(&v1);
            let n2 = norm(&v2);
            let cosangle = if n1 > 0.0 && n2 > 0.0 {
                dot(&v1, &v2) / (n1 * n2)
            } else {
                0.0
            };
            if cosangle > 0.2 {
                if self.trustradius_applied {
                    self.alpha = (self.alpha * 0.99).max(self.alpha0);
                } else {
                    self.alpha = (self.alpha * 1.05).min(self.alpha0 * 1000.0);
                }
            } else {
                self.alpha = (self.alpha * 0.85).max(self.alpha0);
            }
        }
        self.trustradius_applied = false;

        let hist = self.nhist;
        let mut disps = Vec::with_capacity(hist);
        let mut fdifs = Vec::with_capacity(hist);
        let mut norms = Vec::with_capacity(hist);
        for i in 0..hist {
            let mut d = vec![0.0; n];
            let mut f = vec![0.0; n];
            for k in 0..n {
                d[k] = self.position[i + 1][k] - self.position[i][k];
                f[k] = self.force[i + 1][k] - self.force[i][k];
            }
            norms.push(norm(&d));
            disps.push(d);
            fdifs.push(f);
        }
        let mut overlap = vec![vec![0.0; hist]; hist];
        for i in 0..hist {
            for j in 0..hist {
                if norms[i] > 0.0 && norms[j] > 0.0 {
                    overlap[i][j] = dot(&disps[i], &disps[j]) / (norms[i] * norms[j]);
                }
            }
        }
        let (ov_eval, ov_evec) = if hist == 0 {
            (Vec::new(), Vec::new())
        } else {
            eigh(&overlap)
        };
        let mut basis: Vec<Vec<f64>> = Vec::new();
        let mut fdiff: Vec<Vec<f64>> = Vec::new();
        if hist > 0 {
            let max_eval = ov_eval[hist - 1];
            for idim in 0..hist {
                if max_eval > 0.0 && ov_eval[idim] / max_eval > 1.0e-4 {
                    let scale = ov_eval[idim].sqrt();
                    let mut bcol = vec![0.0; n];
                    let mut fcol = vec![0.0; n];
                    for j in 0..hist {
                        if norms[j] == 0.0 {
                            continue;
                        }
                        let c = ov_evec[j][idim] / scale / norms[j];
                        for k in 0..n {
                            bcol[k] += disps[j][k] * c;
                            fcol[k] += fdifs[j][k] * c;
                        }
                    }
                    for k in 0..n {
                        fcol[k] = -fcol[k];
                    }
                    basis.push(bcol);
                    fdiff.push(fcol);
                }
            }
        }
        let ndim = basis.len();
        let mut hessian = vec![vec![0.0; ndim]; ndim];
        for a in 0..ndim {
            for b in 0..ndim {
                hessian[a][b] = dot(&basis[a], &fdiff[b]);
            }
        }
        for a in 0..ndim {
            for b in a + 1..ndim {
                let v = 0.5 * (hessian[a][b] + hessian[b][a]);
                hessian[a][b] = v;
                hessian[b][a] = v;
            }
        }
        let (h_eval, h_evec) = if ndim == 0 {
            (Vec::new(), Vec::new())
        } else {
            eigh(&hessian)
        };
        let mut diagonal_basis = vec![vec![0.0; ndim]; n];
        let mut diagonal_fdiff = vec![vec![0.0; ndim]; n];
        for k in 0..n {
            for col in 0..ndim {
                let mut b = 0.0;
                let mut f = 0.0;
                for a in 0..ndim {
                    b += basis[a][k] * h_evec[a][col];
                    f += fdiff[a][k] * h_evec[a][col];
                }
                diagonal_basis[k][col] = b;
                diagonal_fdiff[k][col] = f;
            }
        }
        // Store basis as columns for dots: rebuild column vecs.
        let mut dbasis: Vec<Vec<f64>> = vec![vec![0.0; n]; ndim];
        let mut dfdiff: Vec<Vec<f64>> = vec![vec![0.0; n]; ndim];
        for col in 0..ndim {
            for k in 0..n {
                dbasis[col][k] = diagonal_basis[k][col];
                dfdiff[col][k] = diagonal_fdiff[k][col];
            }
        }
        let mut res = vec![0.0; ndim];
        for i in 0..ndim {
            let mut acc = 0.0;
            for k in 0..n {
                let t = dfdiff[i][k] - h_eval[i] * dbasis[i][k];
                acc += t * t;
            }
            res[i] = acc.sqrt();
        }

        self.nhist += 1;
        if self.nhist > self.nhistx {
            self.nhist = self.nhistx;
            for i in 0..self.nhist {
                let pos_next = self.position[i + 1].clone();
                let force_next = self.force[i + 1].clone();
                self.position[i] = pos_next;
                self.force[i] = force_next;
            }
        }
        let current = self.nhist - 1;
        let mut gradient = self.force[current].clone();
        for g in &mut gradient {
            *g = -*g;
        }

        let mut displacement = gradient.clone();
        // displacement = (I - B B^T) gradient
        for col in &dbasis {
            let c = dot(col, &gradient);
            for k in 0..n {
                displacement[k] -= c * col[k];
            }
        }
        let complement = (self.alpha.powi(-2) + curv2 * curv2).powf(-0.5);
        for d in &mut displacement {
            *d *= complement;
        }

        let mut hinv_diag = vec![0.0; ndim];
        for i in 0..ndim {
            hinv_diag[i] = 1.0 / (h_eval[i] * h_eval[i] + res[i] * res[i] + curv2 * curv2).sqrt();
        }
        let displacement2 = if let Some(w) = constraint_w.as_ref() {
            let mut add = vec![0.0; n];
            for i in 0..ndim {
                let c = dot(&dbasis[i], &gradient) * hinv_diag[i];
                for k in 0..n {
                    add[k] += c * dbasis[i][k];
                }
            }
            for k in 0..n {
                displacement[k] += add[k];
            }
            let mut hinvw = w.clone();
            for col in &dbasis {
                let c = dot(col, w);
                for k in 0..n {
                    hinvw[k] -= c * col[k];
                }
            }
            for v in &mut hinvw {
                *v *= complement;
            }
            for i in 0..ndim {
                let c = dot(&dbasis[i], w) * hinv_diag[i];
                for k in 0..n {
                    hinvw[k] += c * dbasis[i][k];
                }
            }
            let sinv = -1.0 / dot(&hinvw, w);
            let along = dot(&hinvw, &gradient);
            for k in 0..n {
                displacement[k] += sinv * hinvw[k] * along;
            }
            let wn = norm(w);
            let mut d2 = hinvw;
            for v in &mut d2 {
                *v /= wn;
            }
            Some(d2)
        } else {
            for i in 0..ndim {
                let c = dot(&dbasis[i], &gradient) * hinv_diag[i];
                for k in 0..n {
                    displacement[k] += c * dbasis[i][k];
                }
            }
            None
        };

        if let Some(radius) = self.trustradius {
            let maxd = max_atom(&displacement);
            if maxd > radius {
                let scale = radius / maxd;
                for d in &mut displacement {
                    *d *= scale;
                }
                self.trustradius_applied = true;
            }
        }

        Step {
            position: self.position[current].clone(),
            displacement,
            displacement2,
            ndim,
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn max_atom(v: &[f64]) -> f64 {
    let mut best = 0.0;
    for chunk in v.chunks(3) {
        let n = chunk.iter().map(|x| x * x).sum::<f64>().sqrt();
        if n > best {
            best = n;
        }
    }
    best
}

fn mat_vec(cols: &[Vec<f64>], coef: &[f64]) -> Vec<f64> {
    if cols.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; cols[0].len()];
    for (col, c) in cols.iter().zip(coef) {
        for (o, v) in out.iter_mut().zip(col) {
            *o += *c * *v;
        }
    }
    out
}

fn mat_transpose_vec(cols: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    cols.iter().map(|col| dot(col, v)).collect()
}

fn gram_factor(cols: &[Vec<f64>]) -> (Vec<Vec<f64>>, usize) {
    let p = cols.len();
    let mut g = vec![vec![0.0; p]; p];
    for i in 0..p {
        for j in 0..p {
            g[i][j] = dot(&cols[i], &cols[j]);
        }
    }
    (g, p)
}

fn solve_spd(gram: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let p = rhs.len();
    if p == 0 {
        return Vec::new();
    }
    if p == 1 {
        return vec![rhs[0] / gram[0][0]];
    }
    // Gaussian elimination. Constraint counts stay at one or two.
    let mut a = gram.to_vec();
    let mut b = rhs.to_vec();
    for i in 0..p {
        let mut pivot = i;
        for r in i + 1..p {
            if a[r][i].abs() > a[pivot][i].abs() {
                pivot = r;
            }
        }
        a.swap(i, pivot);
        b.swap(i, pivot);
        let div = a[i][i];
        for c in i..p {
            a[i][c] /= div;
        }
        b[i] /= div;
        for r in 0..p {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for c in i..p {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    b
}

fn eigh(matrix: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut m = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            m[(i, j)] = matrix[i][j];
        }
    }
    let eig = m.self_adjoint_eigen(faer::Side::Lower).unwrap();
    let mut evals = vec![0.0; n];
    let mut evecs = vec![vec![0.0; n]; n];
    for k in 0..n {
        evals[k] = eig.S()[k];
        for i in 0..n {
            evecs[i][k] = eig.U()[(i, k)];
        }
    }
    (evals, evecs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrained_quadratic_matches_compass() {
        let mut opt = ConstrainedOpt::new(3, 0.001, f64::INFINITY, 10, Some(0.05));
        let mut pos = vec![1.0, 0.4, 0.0];
        let w = vec![0.0, 1.0, 0.0];
        let pz = vec![vec![0.0, 0.0, 1.0]];
        let expect = [
            [0.998, 0.4, 0.0],
            [0.948, 0.4, 0.0],
            [0.898, 0.4, 0.0],
            [0.848, 0.4, 0.0],
            [0.798, 0.4, 0.0],
            [0.748, 0.4, 0.0],
        ];
        for target in expect {
            let energy = pos[0] * pos[0] + 0.25 * pos[1] * pos[1];
            let force = vec![-2.0 * pos[0], -0.5 * pos[1], 0.0];
            let step = opt.step(&pos, energy, &force, &pz, &[0.0], Some(&w), 0.0, 0.0);
            for k in 0..3 {
                pos[k] = step.position[k] - step.displacement[k];
            }
            for k in 0..3 {
                assert!(
                    (pos[k] - target[k]).abs() < 1e-9,
                    "pos {pos:?} target {target:?}"
                );
            }
        }
    }
}
