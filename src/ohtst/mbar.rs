//! The multistate Bennett acceptance ratio (Shirts and Chodera, J. Chem.
//! Phys. 129, 124105 (2008)) and the overlap matrix of its weights
//! (Klimovich, Shirts and Mobley, J. Comput.-Aided Mol. Des. 29, 397
//! (2015)).
//!
//! With `N_k` samples from each state and the reduced potential
//! `u_k(x_n)` of every sample in every state, the reduced free energies
//! solve
//!
//! ```text
//! f_k = -ln Σ_n exp(-u_k(x_n)) / Σ_l N_l exp(f_l - u_l(x_n)),
//! ```
//!
//! the stationary point of the convex
//! `F(f) = Σ_n ln Σ_l N_l exp(f_l - u_l(x_n)) - Σ_k N_k f_k`. Newton's
//! method with a backtracking line search minimizes it with `f_0 = 0`.
//! With the weights `W_nk = exp(f_k - u_k(x_n)) / Σ_l N_l exp(f_l - u_l(x_n))`
//! and the thin decomposition `W = U Σ Vᵀ`, the asymptotic covariance of
//! `f` is `V Σ (I - Σ Vᵀ N V Σ)⁺ Σ Vᵀ` (Shirts and Chodera, Eq. D6), and the
//! overlap matrix is `O = Wᵀ W N`. Its rows sum to one, and `O_ij` is the
//! chance that a sample of state `i` is taken for one of state `j`.

use ndarray::{Array1, Array2, Axis, s};

use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;

/// Free energies of every state, `f_0 = 0`, their covariance and the
/// overlap matrix.
pub(crate) struct Mbar {
    pub(crate) free_energies: Array1<f64>,
    pub(crate) covariance: Array2<f64>,
    pub(crate) overlap: Array2<f64>,
}

/// Solve MBAR for the reduced potentials `u`, one row per sample and one
/// column per state, with `counts[k]` samples drawn from state `k`,
/// starting from `initial`.
pub(crate) fn mbar(
    u: &Array2<f64>,
    counts: &[usize],
    initial: &[f64],
) -> Result<Mbar, SaddleError> {
    let (samples, states) = u.dim();
    if states < 2 || counts.len() != states || initial.len() != states {
        return Err(SaddleError::Shape(
            "MBAR needs two states or more and one count and start per state".into(),
        ));
    }
    let total = counts
        .iter()
        .try_fold(0usize, |acc, &count| acc.checked_add(count))
        .ok_or_else(|| {
            SaddleError::Invalid("MBAR needs samples from every state, as many as the rows".into())
        })?;
    if counts.contains(&0) || total != samples {
        return Err(SaddleError::Invalid(
            "MBAR needs samples from every state, as many as the rows".into(),
        ));
    }
    if u.iter().any(|v| v.is_nan() || *v == f64::NEG_INFINITY)
        || u.rows()
            .into_iter()
            .any(|row| row.iter().all(|v| !v.is_finite()))
    {
        return Err(SaddleError::NonFinite("MBAR reduced potential"));
    }
    if !initial.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("MBAR initial free energies"));
    }
    let n: Array1<f64> = counts.iter().map(|&c| c as f64).collect();
    let log_n = n.mapv(f64::ln);
    let mut f = Array1::from_iter(initial.iter().map(|x| x - initial[0]));
    if !f.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("MBAR initial free energies"));
    }
    let mut converged = false;
    for _ in 0..200 {
        let (objective, w) = evaluate(u, &log_n, &f);
        let column: Array1<f64> = w.sum_axis(Axis(0));
        let residual = column.iter().fold(0.0_f64, |m, c| m.max((c - 1.0).abs()));
        if residual < 1.0e-10 {
            converged = true;
            break;
        }
        let gradient: Array1<f64> = (&n * &(&column - 1.0)).slice(s![1..]).to_owned();
        let scaled = &w.slice(s![.., 1..]) * &n.slice(s![1..]);
        let mut hessian = -scaled.t().dot(&scaled);
        for k in 1..states {
            hessian[(k - 1, k - 1)] += n[k] * column[k];
        }
        let (values, vectors) = exact_eigh(hessian.view())?;
        let largest = values.iter().fold(0.0_f64, |m, x| m.max(*x));
        let mut step = Array1::zeros(states - 1);
        for (i, &value) in values.iter().enumerate() {
            if value > 1.0e-14 * largest {
                let v = vectors.column(i);
                step.scaled_add(-v.dot(&gradient) / value, &v);
            }
        }
        // Close to the root the decrease of F is below its rounding, so the
        // full Newton step is taken there.
        let slope = gradient.dot(&step);
        let mut t = 1.0;
        let mut trial = f.clone();
        loop {
            trial
                .slice_mut(s![1..])
                .assign(&(&f.slice(s![1..]) + &(&step * t)));
            if residual < 1.0e-4 {
                break;
            }
            let (value, _) = evaluate(u, &log_n, &trial);
            if value <= objective + 1.0e-4 * t * slope || t < 1.0e-12 {
                break;
            }
            t *= 0.5;
        }
        f = trial;
    }
    if !converged {
        return Err(SaddleError::Solver("MBAR did not converge".into()));
    }
    let (_, w) = evaluate(u, &log_n, &f);
    let gram = w.t().dot(&w);
    let (squares, v) = exact_eigh(gram.view())?;
    let sigma = squares.mapv(|x| x.max(0.0).sqrt());
    let mut inner = Array2::<f64>::eye(states);
    let vnv = v.t().dot(&(&v * &n.view().insert_axis(Axis(1))));
    for i in 0..states {
        for j in 0..states {
            inner[(i, j)] -= sigma[i] * vnv[(i, j)] * sigma[j];
        }
    }
    let (alphas, q) = exact_eigh(inner.view())?;
    let top = alphas.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut pseudo = Array2::zeros((states, states));
    for (i, &alpha) in alphas.iter().enumerate() {
        if alpha.abs() > 1.0e-10 * top {
            let column = q.column(i).insert_axis(Axis(1));
            pseudo = pseudo + column.dot(&column.t()) / alpha;
        }
    }
    let vs = &v * &sigma.view().insert_axis(Axis(0));
    let covariance = vs.dot(&pseudo).dot(&vs.t());
    let overlap = &gram * &n.view().insert_axis(Axis(0));
    if !f.iter().all(|v| v.is_finite())
        || !covariance.iter().all(|v| v.is_finite())
        || !overlap.iter().all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("MBAR output"));
    }
    Ok(Mbar {
        free_energies: f,
        covariance,
        overlap,
    })
}

/// `F(f)` and the weights `W`.
fn evaluate(u: &Array2<f64>, log_n: &Array1<f64>, f: &Array1<f64>) -> (f64, Array2<f64>) {
    let mut w = Array2::zeros(u.dim());
    let mut total = 0.0;
    for (row, mut out) in u.rows().into_iter().zip(w.rows_mut()) {
        let largest = row
            .iter()
            .zip(log_n.iter().zip(f.iter()))
            .map(|(u, (ln, f))| ln + f - u)
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row
            .iter()
            .zip(log_n.iter().zip(f.iter()))
            .map(|(u, (ln, f))| (ln + f - u - largest).exp())
            .sum();
        let log_denominator = largest + sum.ln();
        total += log_denominator;
        for ((o, u), f) in out.iter_mut().zip(row.iter()).zip(f.iter()) {
            *o = (f - u - log_denominator).exp();
        }
    }
    let counts = log_n.mapv(f64::exp);
    (total - counts.dot(f), w)
}

/// Settings of [`mbar_free_energies`].
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct MbarConfig {
    /// Starting reduced free energies, one per state. `None` starts at zero.
    /// The solved energies are shifted so the first state is zero.
    pub initial: Option<Array1<f64>>,
}

impl MbarConfig {
    /// Set [`Self::initial`].
    #[must_use]
    pub fn with_initial(mut self, initial: impl Into<Option<Array1<f64>>>) -> Self {
        self.initial = initial.into();
        self
    }
}

/// Reduced free energies from MBAR, with `f_0 = 0`, their covariance, and
/// the overlap matrix whose rows sum to one.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct MbarFreeEnergy {
    /// Reduced free energy of every state.
    pub free_energies: Array1<f64>,
    /// Asymptotic covariance of [`Self::free_energies`].
    pub covariance: Array2<f64>,
    /// Overlap matrix. `O_ij` is the chance a sample of state `i` is taken
    /// for one of state `j`.
    pub overlap: Array2<f64>,
}

/// Solve MBAR for reduced potentials `reduced`, one row per sample and one
/// column per state, with `counts[k]` samples drawn from state `k`.
pub fn mbar_free_energies(
    reduced: &Array2<f64>,
    counts: &[usize],
    config: &MbarConfig,
) -> Result<MbarFreeEnergy, SaddleError> {
    let states = reduced.ncols();
    let initial = match &config.initial {
        Some(start) => start.to_vec(),
        None => vec![0.0; states],
    };
    let solved = mbar(reduced, counts, &initial)?;
    Ok(MbarFreeEnergy {
        free_energies: solved.free_energies,
        covariance: solved.covariance,
        overlap: solved.overlap,
    })
}

/// The overlap `O_ab` of two states from the forward works `w_F` of the
/// samples of `a` and the reverse works `w_R` of those of `b`, in units of
/// `kT`, at `Δf = f_b - f_a`.
pub(crate) fn pair_overlap(forward: &[f64], reverse: &[f64], difference: f64) -> f64 {
    let (na, nb) = (forward.len(), reverse.len());
    let mut u = Array2::zeros((na + nb, 2));
    for (i, w) in forward.iter().enumerate() {
        u[(i, 1)] = *w;
    }
    for (i, w) in reverse.iter().enumerate() {
        u[(na + i, 0)] = *w;
    }
    let log_n = Array1::from(vec![(na as f64).ln(), (nb as f64).ln()]);
    let (_, weights) = evaluate(&u, &log_n, &Array1::from(vec![0.0, difference]));
    weights.column(0).dot(&weights.column(1)) * nb as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::Rng;

    /// Harmonic states `u_k = κ_k (x - c_k)² / 2` have
    /// `f_k = ln(κ_k / 2π) / 2`.
    #[test]
    fn mbar_recovers_harmonic_free_energies() {
        let stiffness = [1.0, 1.6, 2.5, 4.0];
        let centre = [0.0, 0.3, 0.5, 0.9];
        let per_state = 1500;
        let mut rng = Rng::new(7);
        let mut xs = Vec::new();
        for k in 0..4 {
            for _ in 0..per_state {
                xs.push(centre[k] + rng.normal() / f64::sqrt(stiffness[k]));
            }
        }
        let u = Array2::from_shape_fn((xs.len(), 4), |(n, k)| {
            0.5 * stiffness[k] * (xs[n] - centre[k]).powi(2)
        });
        let result = mbar(&u, &[per_state; 4], &[0.0; 4]).unwrap();
        let exact = |k: usize| 0.5 * (stiffness[k] / std::f64::consts::TAU).ln();
        for k in 1..4 {
            let delta = result.free_energies[k];
            let c = &result.covariance;
            let error = (c[(k, k)] + c[(0, 0)] - 2.0 * c[(0, k)]).sqrt();
            let want = exact(k) - exact(0);
            assert!(error > 0.0 && error < 0.05, "{error}");
            assert!(
                (delta - want).abs() < 4.0 * error,
                "{k}: {delta} ± {error} against {want}"
            );
        }
        for row in result.overlap.rows() {
            assert!((row.sum() - 1.0).abs() < 1e-10);
        }
        assert!(result.overlap[(0, 1)] > result.overlap[(0, 3)]);
        let started = mbar(&u, &[per_state; 4], &[0.0, 1.0, -2.0, 3.0]).unwrap();
        assert!(
            (&started.free_energies - &result.free_energies)
                .iter()
                .all(|v| v.abs() < 1e-9)
        );
    }

    #[test]
    fn pair_overlap_spans_identical_to_disjoint_states() {
        let zeros = vec![0.0; 50];
        assert!((pair_overlap(&zeros, &zeros, 0.0) - 0.5).abs() < 1e-12);
        let far: Vec<f64> = vec![60.0; 50];
        assert!(pair_overlap(&far, &far, 0.0) < 1e-20);
    }

    #[test]
    fn mbar_rejects_overflowing_count_sum() {
        let u = Array2::zeros((2, 2));
        let err = mbar_free_energies(&u, &[usize::MAX, 2], &MbarConfig::default()).unwrap_err();
        assert!(matches!(err, SaddleError::Invalid(_)), "{err:?}");
    }

    #[test]
    fn mbar_rejects_non_finite_initial_and_all_infinite_row() {
        let u = Array2::zeros((2, 2));
        let cfg = MbarConfig::default().with_initial(Array1::from(vec![0.0, f64::NAN]));
        let err = mbar_free_energies(&u, &[1, 1], &cfg).unwrap_err();
        assert!(
            matches!(err, SaddleError::NonFinite("MBAR initial free energies")),
            "{err:?}"
        );
        let inf_row =
            Array2::from_shape_vec((2, 2), vec![f64::INFINITY, f64::INFINITY, 0.0, 0.0]).unwrap();
        let err = mbar_free_energies(&inf_row, &[1, 1], &MbarConfig::default()).unwrap_err();
        assert!(
            matches!(err, SaddleError::NonFinite("MBAR reduced potential")),
            "{err:?}"
        );
    }
}
