//! Direction seeding helpers for minimum-mode saddle searches.
//!
//! [`soft_mode_then_full_directions`] computes a sparse Hessian at a
//! minimum, extracts the lowest spectrum with the built-in Lanczos solver, picks
//! the soft-subspace rank from the largest spectral gap, and emits search
//! directions in two phases:
//!
//! 1. pseudorandom Gaussian draws in that soft subspace, then
//! 2. isotropic full-space draws for verification coverage.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::eigensolve::DENSE_JACOBI_LIMIT;
use crate::{
    PointSurface, SaddleError, SparseHessianOpts, SparsityPattern, exact_eigh, sparse_hessian,
};

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

/// Deterministic soft-to-stiff Hessian-mode sweep.
#[derive(Clone, Debug)]
pub struct ModeSweepDirectionPlan {
    /// Lowest-to-highest eigenvalues of the recovered Hessian.
    pub eigenvalues: Array1<f64>,
    /// Eigenvector column used for each direction.
    pub mode_indices: Vec<usize>,
    /// One direction for every non-rigid mode.
    pub directions: Vec<Array1<f64>>,
    /// Proven lower bound on squared overlap with each selected mode.
    pub minimum_squared_mode_weight: f64,
}

/// Select a deterministic soft-to-stiff schedule over an available mode band.
///
/// The returned values are positions within the band, `0..total_modes`, not
/// eigenvector columns. A short schedule takes the midpoint of each
/// equal-width stratum of the mode index, so the longest run of unsampled
/// modes, ends included, is at most `ceil(total_modes / draws)`. A schedule
/// at least as long as the mode band visits every mode once before cycling.
/// The result is integer arithmetic and does not depend on the platform.
pub fn mode_sweep_schedule(total_modes: usize, draws: usize) -> Result<Vec<usize>, SaddleError> {
    if total_modes == 0 || draws == 0 {
        return Err(SaddleError::Invalid(
            "mode sweep schedule needs at least one mode and one draw".into(),
        ));
    }
    if draws >= total_modes {
        return Ok((0..draws).map(|draw| draw % total_modes).collect());
    }
    let total_modes = total_modes as u128;
    let draws = draws as u128;
    Ok((0..draws)
        .map(|draw| (((2 * draw + 1) * total_modes) / (2 * draws)) as usize)
        .collect())
}

/// Build one localized seed for every non-rigid Hessian mode, soft to stiff.
///
/// The Hessian is recovered by central differences over an all-pairs
/// sparsity pattern and diagonalized in full. That costs `6 N` gradient
/// evaluations, `O((3N)^3)` time and `O((3N)^2)` memory for `N` atoms, and
/// builds every direction whatever schedule the caller then picks from
/// [`mode_sweep_schedule`]. Above 128 degrees of freedom the diagonalization
/// needs the `faer` feature; without it this returns an error.
///
/// `rigid_modes` is the number of lowest eigenpairs skipped. It assumes the
/// rigid modes are the lowest of a stable minimum. Within a degenerate
/// eigenvalue the modes are an arbitrary orthonormal basis. Each eigenvector
/// takes the sign that makes its first largest-magnitude component
/// positive, so the plan does not depend on the sign the eigensolver chose.
///
/// `localization_weights`, when present, has one row per localization centre
/// and one column per atom. Entries must lie in `[0, 1]`; rows are assigned
/// cyclically to modes. The direction for mode `e_m` is
/// `(L e_m + carrier e_m) / ||L e_m + carrier e_m||`, so its squared overlap
/// with `e_m` is at least `(carrier / (1 + carrier))²`.
pub fn hessian_mode_sweep_directions<S: PointSurface>(
    surface: &S,
    minimum: ArrayView1<'_, f64>,
    rigid_modes: usize,
    carrier: f64,
    localization_weights: Option<ArrayView2<'_, f64>>,
) -> Result<ModeSweepDirectionPlan, SaddleError> {
    let dof = minimum.len();
    if dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "mode sweep expects a non-empty 3N Cartesian point".into(),
        ));
    }
    if rigid_modes >= dof {
        return Err(SaddleError::Invalid(
            "mode sweep needs at least one non-rigid mode".into(),
        ));
    }
    if !(carrier.is_finite() && carrier > 0.0) {
        return Err(SaddleError::Invalid(
            "mode sweep carrier must be finite and positive".into(),
        ));
    }
    if !cfg!(feature = "faer") && dof > DENSE_JACOBI_LIMIT {
        return Err(SaddleError::Solver(
            "mode sweep above 128 degrees of freedom requires the faer feature".into(),
        ));
    }
    let n_atoms = dof / 3;
    if let Some(weights) = localization_weights {
        if weights.nrows() == 0 || weights.ncols() != n_atoms {
            return Err(SaddleError::Shape(
                "localization weights must have shape (centres, atoms)".into(),
            ));
        }
        if weights
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(SaddleError::Invalid(
                "localization weights must be finite and lie in [0, 1]".into(),
            ));
        }
    }
    let mut pairs = Vec::new();
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            pairs.push((i, j));
        }
    }
    let pattern = SparsityPattern::from_adjacency(n_atoms, &pairs)?;
    let hessian = sparse_hessian(surface, minimum, &pattern, SparseHessianOpts::default())?;
    let (eigenvalues, eigenvectors) = exact_eigh(hessian.to_dense().view())?;
    let mut directions = Vec::with_capacity(dof - rigid_modes);
    let mode_indices = (rigid_modes..dof).collect::<Vec<_>>();
    for (sweep_index, &mode_index) in mode_indices.iter().enumerate() {
        let mut mode = eigenvectors.column(mode_index).to_owned();
        fix_sign(&mut mode);
        let mut localized = mode.clone();
        if let Some(weights) = localization_weights {
            let row = weights.row(sweep_index % weights.nrows());
            for atom in 0..n_atoms {
                for axis in 0..3 {
                    localized[3 * atom + axis] *= row[atom];
                }
            }
        }
        directions.push(checked_unit(localized + &(mode * carrier))?);
    }
    Ok(ModeSweepDirectionPlan {
        eigenvalues,
        mode_indices,
        directions,
        minimum_squared_mode_weight: (carrier / (1.0 + carrier)).powi(2),
    })
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
    use ndarray::{Array1, Array2, ArrayView1, array};

    use super::{hessian_mode_sweep_directions, mode_sweep_schedule, spectral_gap_rank};
    use crate::{PointSurface, SaddleError, exact_eigh};

    struct DiagonalHarmonic;

    impl PointSurface for DiagonalHarmonic {
        fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let diagonal = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let gradient = &diagonal * &x;
            Ok((0.5 * x.dot(&gradient), gradient))
        }
    }

    struct LargeHarmonic;

    impl PointSurface for LargeHarmonic {
        fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            Ok((0.5 * x.dot(&x), x.to_owned()))
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
            assert!(direction[mode].powi(2) + 1.0e-14 >= plan.minimum_squared_mode_weight);
        }
    }

    #[test]
    fn hessian_chirp_supports_more_than_lanczos_full_dimension() {
        let minimum = Array1::zeros(99);
        let plan = hessian_mode_sweep_directions(&LargeHarmonic, minimum.view(), 3, 0.25, None)
            .expect("large mode sweep");
        assert_eq!(plan.directions.len(), 96);
    }

    #[test]
    fn hessian_chirp_normalizes_large_finite_carriers() {
        let plan = hessian_mode_sweep_directions(
            &DiagonalHarmonic,
            Array1::zeros(6).view(),
            1,
            1.0e200,
            None,
        )
        .expect("large finite carrier");

        assert_eq!(plan.minimum_squared_mode_weight, 1.0);
        for (direction, &mode) in plan.directions.iter().zip(&plan.mode_indices) {
            assert!(direction.iter().all(|value| value.is_finite()));
            assert!(direction[mode].abs() > 1.0 - 1.0e-12);
        }
    }

    #[test]
    fn short_chirp_stratifies_the_whole_mode_band() {
        assert_eq!(mode_sweep_schedule(10, 4).unwrap(), vec![1, 3, 6, 8]);
        assert_eq!(
            mode_sweep_schedule(3, 7).unwrap(),
            vec![0, 1, 2, 0, 1, 2, 0]
        );
        assert!(mode_sweep_schedule(0, 1).is_err());
        assert!(mode_sweep_schedule(3, 0).is_err());
    }

    #[test]
    fn stratified_schedule_handles_the_largest_mode_count() {
        let total = usize::MAX;
        let expected = vec![
            ((total as u128) / 4) as usize,
            ((3 * total as u128) / 4) as usize,
        ];
        assert_eq!(mode_sweep_schedule(total, 2).unwrap(), expected);
    }

    #[test]
    fn stratified_schedule_bounds_the_unsampled_run_for_every_size() {
        for modes in 1..=40_usize {
            for draws in 1..=60_usize {
                let schedule = mode_sweep_schedule(modes, draws).unwrap();
                assert_eq!(schedule.len(), draws);
                assert!(schedule.iter().all(|&mode| mode < modes));
                let mut visited = vec![false; modes];
                for &mode in &schedule {
                    visited[mode] = true;
                }
                if draws >= modes {
                    assert!(visited.iter().all(|&seen| seen), "{modes} {draws}");
                    continue;
                }
                assert!(schedule.windows(2).all(|pair| pair[0] < pair[1]));
                let mut run = 0;
                let mut longest = 0;
                for seen in visited {
                    run = if seen { 0 } else { run + 1 };
                    longest = longest.max(run);
                }
                assert!(
                    longest <= modes.div_ceil(draws),
                    "{modes} {draws} {longest}"
                );
            }
        }
    }

    /// Symmetric matrix with a spread, non-degenerate spectrum.
    fn dense_symmetric(n: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, n), |(i, j)| {
            let coupling = 0.05 * ((i + 1) as f64 * (j + 1) as f64).sin();
            if i == j {
                1.0 + 0.1 * i as f64 + coupling
            } else {
                coupling
            }
        })
    }

    struct DenseQuadratic(Array2<f64>);

    impl PointSurface for DenseQuadratic {
        fn eval(&self, x: ArrayView1<'_, f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let gradient = self.0.dot(&x);
            Ok((0.5 * x.dot(&gradient), gradient))
        }
    }

    #[test]
    fn localized_directions_are_unit_and_respect_the_overlap_floor() {
        let hessian = dense_symmetric(12);
        let (values, vectors) = exact_eigh(hessian.view()).unwrap();
        let weights = array![
            [1.0, 0.0, 0.5, 0.25],
            [0.0, 1.0, 0.75, 0.0],
            [0.3, 0.3, 0.3, 0.3]
        ];
        let carrier = 0.25;
        let plan = hessian_mode_sweep_directions(
            &DenseQuadratic(hessian),
            Array1::zeros(12).view(),
            2,
            carrier,
            Some(weights.view()),
        )
        .unwrap();
        assert_eq!(plan.directions.len(), 10);
        assert_eq!(plan.mode_indices, (2..12).collect::<Vec<_>>());
        for (a, b) in plan.eigenvalues.iter().zip(values.iter()) {
            assert!((a - b).abs() < 1.0e-6, "{a} {b}");
        }
        let floor = plan.minimum_squared_mode_weight;
        let mut loosest = 1.0_f64;
        for (direction, &mode) in plan.directions.iter().zip(&plan.mode_indices) {
            assert!((direction.dot(direction) - 1.0).abs() < 1.0e-12);
            let overlap = direction.dot(&vectors.column(mode)).powi(2);
            assert!(
                overlap + 1.0e-9 >= floor,
                "mode {mode}: {overlap} < {floor}"
            );
            loosest = loosest.min(overlap);
        }
        assert!(loosest < 0.9, "localization never changed a direction");
    }

    #[cfg(feature = "faer")]
    #[test]
    fn unlocalized_sweep_returns_each_eigenvector_above_the_dense_limit() {
        let n = 135;
        let hessian = dense_symmetric(n);
        let surface = DenseQuadratic(hessian.clone());
        let minimum = Array1::zeros(n);
        let plan = hessian_mode_sweep_directions(&surface, minimum.view(), 0, 0.25, None).unwrap();
        assert_eq!(plan.directions.len(), n);
        assert!(
            plan.eigenvalues
                .to_vec()
                .windows(2)
                .all(|pair| pair[0] <= pair[1])
        );
        for (direction, &mode) in plan.directions.iter().zip(&plan.mode_indices) {
            let residual = hessian.dot(direction) - direction * plan.eigenvalues[mode];
            assert!(residual.dot(&residual).sqrt() < 1.0e-5, "mode {mode}");
            let peak = direction
                .iter()
                .fold(0.0_f64, |worst, x| worst.max(x.abs()));
            let first = direction
                .iter()
                .position(|x| x.abs() > 0.999 * peak)
                .unwrap();
            assert!(direction[first] > 0.0, "mode {mode} sign");
        }
        let again = hessian_mode_sweep_directions(&surface, minimum.view(), 0, 0.25, None).unwrap();
        for (left, right) in plan.directions.iter().zip(&again.directions) {
            assert_eq!(left, right);
        }
    }

    #[cfg(not(feature = "faer"))]
    #[test]
    fn mode_sweep_without_faer_reports_the_dense_limit() {
        let n = 135;
        let surface = DenseQuadratic(dense_symmetric(n));
        let error = hessian_mode_sweep_directions(&surface, Array1::zeros(n).view(), 0, 0.25, None)
            .unwrap_err();
        assert!(matches!(error, SaddleError::Solver(_)));
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

fn fix_sign(v: &mut Array1<f64>) {
    let peak = v.iter().fold(0.0_f64, |worst, x| worst.max(x.abs()));
    let first = v
        .iter()
        .position(|x| x.abs() >= peak * (1.0 - 1.0e-6))
        .unwrap_or(0);
    if v[first] < 0.0 {
        v.mapv_inplace(|x| -x);
    }
}

fn unit(mut v: Array1<f64>) -> Array1<f64> {
    let scale = v
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if !scale.is_finite() || scale <= 1.0e-12 {
        v.fill(0.0);
        v[0] = 1.0;
        return v;
    }
    v.mapv_inplace(|x| x / scale);
    let norm = v.dot(&v).sqrt();
    v.mapv_inplace(|x| x / norm);
    v
}

fn checked_unit(mut v: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    if v.iter().any(|value| !value.is_finite()) {
        return Err(SaddleError::NonFinite("mode sweep direction"));
    }
    let scale = v
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if scale == 0.0 {
        return Err(SaddleError::Invalid(
            "mode sweep produced a zero direction".into(),
        ));
    }
    v.mapv_inplace(|value| value / scale);
    let norm = v.dot(&v).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(SaddleError::NonFinite("mode sweep direction norm"));
    }
    v.mapv_inplace(|value| value / norm);
    Ok(v)
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
