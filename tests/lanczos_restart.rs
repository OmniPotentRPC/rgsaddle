//! Thick-restart Lanczos on a diagonal quadratic.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1};
use rgsaddle::{MinModeConfig, MinModeKind, MinModeSession, PointSurface, SaddleError};

struct Diag {
    evals: Vec<f64>,
    actions: AtomicUsize,
}

impl Diag {
    fn new(evals: Vec<f64>) -> Self {
        Self {
            evals,
            actions: AtomicUsize::new(0),
        }
    }
}

impl PointSurface for Diag {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..x.len() {
            energy += 0.5 * self.evals[i] * x[i] * x[i];
            gradient[i] = self.evals[i] * x[i];
        }
        Ok((energy, gradient))
    }

    fn hessian_vector(
        &self,
        _x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        self.actions.fetch_add(1, Ordering::Relaxed);
        let mut hv = Array1::zeros(v.len());
        for i in 0..v.len() {
            hv[i] = self.evals[i] * v[i];
        }
        Ok(Some(hv))
    }
}

fn config(keep: usize, krylov: usize, gap: f64) -> MinModeConfig {
    MinModeConfig {
        kind: MinModeKind::Lanczos,
        krylov_dim: krylov,
        lanczos_restart_keep: keep,
        lanczos_recycle_gap: gap,
        rotation_tol: 1e-10,
        max_rotations: 400,
        ..MinModeConfig::default()
    }
}

fn clustered(n: usize) -> Vec<f64> {
    let mut evals = vec![-1.0, -0.999];
    for i in 2..n {
        evals.push(i as f64);
    }
    evals
}

fn bytes(mode: &Array1<f64>) -> Vec<u8> {
    mode.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn thick_restart_resolves_a_cluster_the_single_pass_misses() {
    let n = 24;
    let surface = Diag::new(clustered(n));
    let seed = Array1::from_elem(n, 1.0);
    let x = Array1::zeros(n);
    let mut plain = MinModeSession::new(config(0, 4, 0.2), x.clone(), seed.clone()).unwrap();
    let missed = plain.estimate_mode(&surface).unwrap();
    assert!(
        !missed.converged,
        "single pass residual {}",
        missed.curvature
    );
    assert_eq!(missed.rotations, 4);

    let mut thick = MinModeSession::new(config(2, 4, 0.2), x, seed).unwrap();
    let found = thick.estimate_mode(&surface).unwrap();
    let mut res = 0.0;
    for i in 0..n {
        let d = surface.evals[i] * found.mode[i] - found.curvature * found.mode[i];
        res += d * d;
    }
    assert!(
        found.converged,
        "curvature {} actions {} m0 {} m1 {} true_res {} plain {}",
        found.curvature,
        found.rotations,
        found.mode[0],
        found.mode[1],
        res.sqrt(),
        missed.curvature
    );
    assert!((found.curvature + 1.0).abs() < 1e-6, "{}", found.curvature);
    assert!(found.mode[0].abs() > 0.9, "{}", found.mode[0]);
}

#[test]
fn zero_keep_matches_the_single_pass_on_a_separated_spectrum() {
    let n = 8;
    let mut evals = vec![-2.0];
    for i in 1..n {
        evals.push(0.5 * i as f64);
    }
    let surface = Diag::new(evals);
    let seed = Array1::from_elem(n, 1.0);
    let x = Array1::zeros(n);
    let mut keep0 = MinModeSession::new(config(0, 8, 0.2), x.clone(), seed.clone()).unwrap();
    let mut keep2 = MinModeSession::new(config(2, 8, 0.2), x, seed).unwrap();
    let a = keep0.estimate_mode(&surface).unwrap();
    let b = keep2.estimate_mode(&surface).unwrap();
    assert!(
        a.converged && b.converged,
        "a {} rot {} curv {} b {} rot {} curv {}",
        a.converged,
        a.rotations,
        a.curvature,
        b.converged,
        b.rotations,
        b.curvature
    );
    assert_eq!(a.rotations, b.rotations);
    assert_eq!(bytes(&a.mode), bytes(&b.mode));
    assert_eq!(a.curvature.to_bits(), b.curvature.to_bits());
}

#[test]
fn recycling_spends_one_fresh_action_and_not_a_stale_one() {
    let n = 16;
    let surface = Diag::new(clustered(n));
    let seed = Array1::from_elem(n, 1.0);
    let x = Array1::zeros(n);
    let mut session = MinModeSession::new(config(2, 6, 0.2), x.clone(), seed).unwrap();
    let first = session.estimate_mode(&surface).unwrap();
    assert!(first.converged);
    let before = surface.actions.load(Ordering::Relaxed);
    let gradient = Array1::zeros(n);
    session.set_position(x.clone(), Some(gradient)).unwrap();
    let second = session.estimate_mode(&surface).unwrap();
    let spent = surface.actions.load(Ordering::Relaxed) - before;
    assert_eq!(spent, second.rotations);
    assert_eq!(
        second.rotations, 2,
        "one action on the mode and one on the recycled vector"
    );

    let mut cold = MinModeSession::new(config(2, 6, 0.0), x, first.mode).unwrap();
    let once = cold.estimate_mode(&surface).unwrap();
    assert_eq!(once.rotations, 1);
}

#[test]
fn modes_match_across_thread_counts() {
    let n = 12;
    let evals = clustered(n);
    let seed = Array1::from_elem(n, 1.0);
    let x = Array1::zeros(n);
    let mut digests = Vec::new();
    for threads in [1usize, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let digest = pool.install(|| {
            let surface = Diag::new(evals.clone());
            let mut session =
                MinModeSession::new(config(2, 4, 0.2), x.clone(), seed.clone()).unwrap();
            let est = session.estimate_mode(&surface).unwrap();
            (bytes(&est.mode), est.curvature.to_bits(), est.converged)
        });
        digests.push(digest);
    }
    assert!(digests.iter().all(|item| item.2));
    assert!(
        digests
            .windows(2)
            .all(|pair| pair[0].0 == pair[1].0 && pair[0].1 == pair[1].1)
    );
}
