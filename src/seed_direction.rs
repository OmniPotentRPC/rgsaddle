//! Direction seeding helpers for minimum-mode saddle searches.
//!
//! [`soft_mode_then_full_directions`] computes a sparse Hessian at a
//! minimum, extracts the lowest spectrum with the built-in Lanczos solver, picks
//! the soft-subspace rank from the largest spectral gap, and emits search
//! directions in two phases:
//!
//! 1. low-discrepancy-like draws in that soft subspace, then
//! 2. isotropic full-space draws for verification coverage.

use ndarray::{Array1, Array2, ArrayView1};

use crate::{PointSurface, SaddleError, SparseHessianOpts, SparsityPattern, sparse_hessian};

/// Soft-mode-first direction plan.
#[derive(Clone, Debug)]
pub struct SoftModeDirectionPlan {
    /// Soft-subspace rank selected from the Hessian spectrum.
    pub k: usize,
    /// Lowest-to-highest eigenvalues of the recovered Hessian.
    pub eigenvalues: Array1<f64>,
    /// Seed directions: soft-subspace stage first, then full-space verification.
    pub directions: Vec<Array1<f64>>,
}

/// Pick the soft-subspace rank from the largest adjacent relative spectral gap.
///
/// For ascending eigenvalues `λ₀ ≤ λ₁ ≤ ...`, this returns
/// `argmax_i (λᵢ₊₁-λᵢ) + 1`, which lies in `[1, n]`.
///
/// If the largest and second-largest normalized gaps are tied at machine
/// precision, there is no resolved split and the full space rank `n` is used.
#[must_use]
pub fn spectral_gap_rank(eigenvalues: ArrayView1<'_, f64>) -> usize {
    if eigenvalues.len() <= 1 {
        return eigenvalues.len().max(1);
    }
    let mut best_i = 0usize;
    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    for i in 0..(eigenvalues.len() - 1) {
        let score = eigenvalues[i + 1] - eigenvalues[i];
        if score >= best {
            second = best;
            best = score;
            best_i = i;
        } else if score > second {
            second = score;
        }
    }
    if !best.is_finite() || best <= 0.0 {
        return eigenvalues.len();
    }
    let ulp = f64::EPSILON * (best.abs() + second.abs()).max(1.0);
    if second.is_finite() && (best - second) <= ulp {
        return eigenvalues.len();
    }
    best_i + 1
}

/// Soft-mode-subspace-first seed directions with full-space verification fallback.
///
/// `phase_count` controls each phase size: the plan returns
/// `2 * phase_count` directions.
pub fn soft_mode_then_full_directions<S: PointSurface>(
    surface: &S,
    minimum: ArrayView1<'_, f64>,
    phase_count: usize,
    seed: u64,
) -> Result<SoftModeDirectionPlan, SaddleError> {
    if phase_count == 0 {
        return Err(SaddleError::Invalid(
            "soft-mode direction seeding needs at least one seed".into(),
        ));
    }
    let dof = minimum.len();
    if dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "soft-mode direction seeding expects a non-empty 3N Cartesian point".into(),
        ));
    }
    let n_atoms = dof / 3;
    let mut pairs = Vec::new();
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            pairs.push((i, j));
        }
    }
    let pattern = SparsityPattern::from_adjacency(n_atoms, &pairs)?;
    let hessian = sparse_hessian(surface, minimum, &pattern, SparseHessianOpts::default())?;
    let eig = hessian.lowest_eigenpairs(dof)?;
    let eigenvalues = eig.values;
    let eigenvectors = eig.vectors;
    let k = spectral_gap_rank(eigenvalues.view());
    let basis = if k == dof {
        Array2::eye(dof)
    } else {
        eigenvectors.slice(ndarray::s![.., ..k]).to_owned()
    };

    let mut rng = SplitMix64::new(seed);
    let mut directions = Vec::with_capacity(2 * phase_count);
    for i in 0..phase_count {
        directions.push(sample_in_subspace(&basis, minimum, i, &mut rng));
    }
    for _ in 0..phase_count {
        directions.push(sample_on_sphere(dof, &mut rng));
    }
    Ok(SoftModeDirectionPlan {
        k,
        eigenvalues,
        directions,
    })
}

fn sample_in_subspace(
    basis: &Array2<f64>,
    _minimum: ArrayView1<'_, f64>,
    _index: usize,
    rng: &mut SplitMix64,
) -> Array1<f64> {
    let k = basis.ncols();
    let mut coeffs = Array1::zeros(k);
    for c in &mut coeffs {
        *c = rng.normal();
    }
    let mut v = basis.dot(&coeffs);
    if v.iter().all(|x| x.abs() < 1.0e-14) {
        v = basis.column(0).to_owned();
    }
    unit(v)
}

fn sample_on_sphere(dim: usize, rng: &mut SplitMix64) -> Array1<f64> {
    let mut v = Array1::zeros(dim);
    for c in &mut v {
        *c = rng.normal();
    }
    unit(v)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView1, array};

    use super::{hessian_mode_sweep_directions, spectral_gap_rank};
    use crate::{PointSurface, SaddleError};

    struct DiagonalHarmonic;

    impl PointSurface for DiagonalHarmonic {
        fn eval(
            &self,
            x: ArrayView1<'_, f64>,
        ) -> Result<(f64, Array1<f64>), SaddleError> {
            let diagonal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let gradient = &diagonal * &x;
            Ok((0.5 * x.dot(&gradient), gradient))
        }
    }

    #[test]
    fn hessian_chirp_visits_each_nonrigid_mode_with_overlap_floor() {
        let minimum = Array1::zeros(6);
        let weights = array![[0.0, 1.0], [1.0, 0.0]];
        let plan = hessian_mode_sweep_directions(
            &DiagonalHarmonic,
            minimum.view(),
            1,
            0.25,
            Some(weights.view()),
        )
        .expect("mode sweep");
        assert_eq!(plan.mode_indices, vec![1, 2, 3, 4, 5]);
        assert_eq!(plan.directions.len(), 5);
        assert!((plan.minimum_squared_mode_weight - 0.04).abs() < 1.0e-14);
        for (direction, &mode) in plan.directions.iter().zip(&plan.mode_indices) {
            assert!(
                direction[mode].powi(2) + 1.0e-14
                    >= plan.minimum_squared_mode_weight
            );
        }
    }

    #[test]
    fn spectral_gap_rank_uses_full_space_when_gaps_are_degenerate() {
        let values = array![1.0, 2.0, 3.0, 4.0];
        assert_eq!(spectral_gap_rank(values.view()), 4);
    }

    #[test]
    fn spectral_gap_rank_uses_full_space_when_no_gap_is_resolved() {
        let values = array![1.0, 1.5, 2.0, 2.5, 3.0];
        // All adjacent gaps are equal.
        assert_eq!(spectral_gap_rank(values.view()), 5);
    }

    #[test]
    fn spectral_gap_rank_picks_the_dominant_split() {
        let values = array![1.0, 1.1, 1.2, 4.0, 4.5];
        assert_eq!(spectral_gap_rank(values.view()), 3);
    }
}

fn unit(mut v: Array1<f64>) -> Array1<f64> {
    let norm2 = v.dot(&v);
    if !norm2.is_finite() || norm2 <= 1.0e-24 {
        v.fill(0.0);
        v[0] = 1.0;
        return v;
    }
    let norm = norm2.sqrt();
    v.mapv_inplace(|x| x / norm);
    v
}

#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
    spare: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn uniform01(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        let u1 = self.uniform01().max(1.0e-16);
        let u2 = self.uniform01();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f64::consts::TAU * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        self.spare = Some(z1);
        z0
    }
}
