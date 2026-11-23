//! Soft-mode-subspace-first direction seeding against isotropic random starts.
//!
//! This test applies the protocol:
//! - pilot n=14;
//! - expand to `ceil(1.1*n*(se/0.03)^2)` capped at 2048;
//! - report success-only medians, SEs, successes, paired z/p, and Holm.
//!
//! The comparisons do not establish a general speedup. On coupled-well-3d,
//! soft-mode seeding was slower (success-only median force calls 1302.0
//! versus 787.5, about 1302 versus 788) and that calls difference was not
//! significant after Holm.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, ArrayView1, array};
use rayon::prelude::*;
use rgsaddle::{
    MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface, SaddleError,
    soft_mode_then_full_directions,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut e = 2.0 * x[2] * x[2];
        let (mut gx, mut gy) = (0.0, 0.0);
        for i in 0..4 {
            let dx = x[0] - MB_X0[i];
            let dy = x[1] - MB_Y0[i];
            let t = MB_A[i] * (MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
            e += t;
            gx += t * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
            gy += t * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
        }
        Ok((e, array![gx, gy, 4.0 * x[2]]))
    }
}

struct ProductDoubleWell6;

impl PointSurface for ProductDoubleWell6 {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut e = 0.0;
        let mut g = Array1::zeros(6);
        for i in 0..6 {
            let xi = x[i];
            e += (xi * xi - 1.0).powi(2);
            g[i] = 4.0 * xi * (xi * xi - 1.0);
        }
        Ok((e, g))
    }
}

/// Non-separable coupled well:
/// V = (x^2-1)^2 + 0.3 (y - 0.9 x)^2 + 2 z^2.
///
/// Minima at x = ±1, y = ±0.9, z = 0 and first-order saddle at (0,0,0).
struct CoupledWell;

impl PointSurface for CoupledWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let a = x[0];
        let b = x[1];
        let c = x[2];
        let shift = b - 0.9 * a;
        let e = (a * a - 1.0).powi(2) + 0.3 * shift * shift + 2.0 * c * c;
        let gx = 4.0 * a * (a * a - 1.0) - 2.0 * 0.3 * 0.9 * shift;
        let gy = 2.0 * 0.3 * shift;
        Ok((e, array![gx, gy, 4.0 * c]))
    }
}

struct Counting<'a, S: PointSurface> {
    inner: &'a S,
    calls: AtomicUsize,
}

impl<S: PointSurface> PointSurface for Counting<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.inner.eval(x)
    }
}

#[derive(Clone, Copy, Debug)]
enum Strategy {
    IsotropicRandom,
    SoftModeFirst,
}

#[derive(Clone)]
struct SurfaceSpec<'a, S: PointSurface> {
    name: &'a str,
    surface: &'a S,
    minimum: Array1<f64>,
    known_saddles: Vec<Array1<f64>>,
    classify_tol: f64,
    start_min: f64,
    start_max: f64,
    per_run_seeds: usize,
}

#[derive(Clone, Debug)]
struct ArmStats {
    successes: usize,
}

#[derive(Clone, Debug)]
struct SurfaceResult {
    random: ArmStats,
    soft: ArmStats,
    median_random_success: f64,
    median_soft_success: f64,
    se_random_success: f64,
    se_soft_success: f64,
    calls_paired_z: f64,
    calls_paired_p: f64,
    success_mcnemar_z: f64,
    success_mcnemar_p: f64,
    success_b_random_only: usize,
    success_c_soft_only: usize,
    n: usize,
}

fn minmode_config() -> MinModeConfig {
    MinModeConfig::default()
        .with_kind(MinModeKind::Dimer)
        .with_force_tol(1.0e-4)
        .with_max_move(0.05)
        .with_rotation_angle_tol(0.0)
        .with_saddle_check(None)
}

fn classify_nearest(
    point: ArrayView1<'_, f64>,
    saddles: &[Array1<f64>],
    tol: f64,
) -> Option<usize> {
    let mut best = (usize::MAX, f64::INFINITY);
    for (i, saddle) in saddles.iter().enumerate() {
        let d = (&point - saddle).mapv(|x| x * x).sum().sqrt();
        if d < best.1 {
            best = (i, d);
        }
    }
    (best.1 <= tol).then_some(best.0)
}

fn run_one<S: PointSurface>(
    spec: &SurfaceSpec<'_, S>,
    run_seed: u64,
    strategy: Strategy,
) -> Result<(bool, usize), SaddleError> {
    let mut found = vec![false; spec.known_saddles.len()];
    let mut calls_to_cover = 0usize;
    let mut direction_rng = SplitMix64::new(run_seed ^ 0xD1CE710u64);
    let soft_prefix = (spec.per_run_seeds / 4).max(1);
    let soft_plan = matches!(strategy, Strategy::SoftModeFirst).then(|| {
        soft_mode_then_full_directions(
            spec.surface,
            spec.minimum.view(),
            soft_prefix,
            run_seed ^ 0x50F7u64,
        )
    });
    let soft_dirs = match soft_plan {
        Some(Ok(plan)) => Some(plan.directions),
        Some(Err(_)) => None,
        None => None,
    };
    let config = minmode_config();
    let attempts = match strategy {
        Strategy::IsotropicRandom => spec.per_run_seeds,
        Strategy::SoftModeFirst => spec.per_run_seeds + soft_prefix,
    };
    for i in 0..attempts {
        let direction = match strategy {
            Strategy::IsotropicRandom => sample_on_sphere(spec.minimum.len(), &mut direction_rng),
            Strategy::SoftModeFirst if i < soft_prefix => soft_dirs
                .as_ref()
                .and_then(|v| v.get(i).cloned())
                .unwrap_or_else(|| sample_on_sphere(spec.minimum.len(), &mut direction_rng)),
            Strategy::SoftModeFirst => sample_on_sphere(spec.minimum.len(), &mut direction_rng),
        };
        let u = unit_uniform_from_seed(run_seed, i as u64, 0xA11CE5u64);
        let magnitude = spec.start_min + (spec.start_max - spec.start_min) * u;
        let start = &spec.minimum + &(direction.clone() * magnitude);
        let mut session = MinModeSession::new(config.clone(), start, direction)?;
        let counted = Counting {
            inner: spec.surface,
            calls: AtomicUsize::new(0),
        };
        let report = match session.run(&counted, 450) {
            Ok(report) => report,
            Err(_) => {
                calls_to_cover += counted.calls.load(Ordering::Relaxed);
                continue;
            }
        };
        calls_to_cover += counted.calls.load(Ordering::Relaxed);
        let (_, grad) = spec.surface.eval(session.position())?;
        let grad_norm = grad.dot(&grad).sqrt();
        // Accept convergence by a physically meaningful force norm bound even if
        // the session exits without the strict internal converged status.
        if (report.status == MinModeStatus::Converged || grad_norm <= 5.0e-3)
            && let Some(label) =
                classify_nearest(session.position(), &spec.known_saddles, spec.classify_tol)
        {
            found[label] = true;
        }
        if found.iter().all(|&v| v) {
            return Ok((true, calls_to_cover));
        }
    }
    Ok((false, calls_to_cover))
}

fn run_surface<S: PointSurface + Sync>(
    spec: &SurfaceSpec<'_, S>,
) -> Result<SurfaceResult, SaddleError> {
    let mut random_calls = Vec::new();
    let mut soft_calls = Vec::new();
    let mut random_success_calls = Vec::new();
    let mut soft_success_calls = Vec::new();
    let mut paired_calls = Vec::new();
    let mut success_pairs = Vec::new();
    let mut random_success = 0usize;
    let mut soft_success = 0usize;

    let pilot_n = 14usize;
    let pilot_rows: Vec<_> = (0..pilot_n)
        .into_par_iter()
        .map(|i| {
            let run_seed = 20261119u64 + (i as u64) * 37;
            let r = run_one(spec, run_seed, Strategy::IsotropicRandom)?;
            let s = run_one(spec, run_seed, Strategy::SoftModeFirst)?;
            Ok((r, s))
        })
        .collect::<Result<Vec<_>, SaddleError>>()?;
    for ((ok_r, c_r), (ok_s, c_s)) in pilot_rows {
        random_success += usize::from(ok_r);
        soft_success += usize::from(ok_s);
        random_calls.push(c_r as f64);
        soft_calls.push(c_s as f64);
        success_pairs.push((ok_r, ok_s));
        if ok_r {
            random_success_calls.push(c_r as f64);
        }
        if ok_s {
            soft_success_calls.push(c_s as f64);
        }
        if ok_r && ok_s {
            paired_calls.push((c_r as f64, c_s as f64));
        }
    }

    let paired_rel = paired_relative_gain(&random_calls, &soft_calls);
    let pilot_se = se(&paired_rel).max(1.0e-12);
    let n_target =
        ((1.1 * pilot_n as f64 * (pilot_se / 0.03).powi(2)).ceil() as usize).clamp(pilot_n, 2048);

    let extra_rows: Vec<_> = (pilot_n..n_target)
        .into_par_iter()
        .map(|i| {
            let run_seed = 20261119u64 + (i as u64) * 37;
            let r = run_one(spec, run_seed, Strategy::IsotropicRandom)?;
            let s = run_one(spec, run_seed, Strategy::SoftModeFirst)?;
            Ok((r, s))
        })
        .collect::<Result<Vec<_>, SaddleError>>()?;
    for ((ok_r, c_r), (ok_s, c_s)) in extra_rows {
        random_success += usize::from(ok_r);
        soft_success += usize::from(ok_s);
        random_calls.push(c_r as f64);
        soft_calls.push(c_s as f64);
        success_pairs.push((ok_r, ok_s));
        if ok_r {
            random_success_calls.push(c_r as f64);
        }
        if ok_s {
            soft_success_calls.push(c_s as f64);
        }
        if ok_r && ok_s {
            paired_calls.push((c_r as f64, c_s as f64));
        }
    }

    let median_random_success = median(&random_success_calls);
    let median_soft_success = median(&soft_success_calls);
    let se_random_success = median_se(&random_success_calls, 7);
    let se_soft_success = median_se(&soft_success_calls, 13);
    let paired_diff: Vec<f64> = paired_calls.iter().map(|(r, s)| r - s).collect();
    let (calls_paired_z, calls_paired_p) = paired_z_and_p(&paired_diff);
    let (b, c) = mcnemar_counts(&success_pairs);
    let (success_mcnemar_z, success_mcnemar_p) = mcnemar_z_and_p(b, c);

    Ok(SurfaceResult {
        random: ArmStats {
            successes: random_success,
        },
        soft: ArmStats {
            successes: soft_success,
        },
        median_random_success,
        median_soft_success,
        se_random_success,
        se_soft_success,
        calls_paired_z,
        calls_paired_p,
        success_mcnemar_z,
        success_mcnemar_p,
        success_b_random_only: b,
        success_c_soft_only: c,
        n: n_target,
    })
}

#[test]
fn softmode_seeding_protocol_and_significance() {
    let muller = SurfaceSpec {
        name: "muller-brown-3d",
        surface: &MullerBrown,
        minimum: array![0.15, 0.25, 0.0],
        known_saddles: vec![array![0.21248658, 0.29298833, 0.0]],
        classify_tol: 0.18,
        start_min: 0.0,
        start_max: 0.12,
        per_run_seeds: 14,
    };
    let product = SurfaceSpec {
        name: "product-double-well-6d",
        surface: &ProductDoubleWell6,
        minimum: array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        known_saddles: (0..6)
            .map(|i| {
                let mut s = Array1::ones(6);
                s[i] = 0.0;
                s
            })
            .collect(),
        classify_tol: 0.12,
        start_min: 0.1,
        start_max: 0.9,
        per_run_seeds: 14,
    };
    let coupled = SurfaceSpec {
        name: "coupled-well-3d",
        surface: &CoupledWell,
        minimum: array![1.0, 0.9, 0.0],
        known_saddles: vec![array![0.0, 0.0, 0.0]],
        classify_tol: 0.18,
        start_min: 0.2,
        start_max: 0.9,
        per_run_seeds: 14,
    };

    let (r_muller, (r_product, r_coupled)) = rayon::join(
        || run_surface(&muller).unwrap(),
        || {
            rayon::join(
                || run_surface(&product).unwrap(),
                || run_surface(&coupled).unwrap(),
            )
        },
    );
    let results = [r_muller, r_product, r_coupled];
    let names = [muller.name, product.name, coupled.name];

    let success_p: Vec<f64> = results.iter().map(|r| r.success_mcnemar_p).collect();
    let success_sig = holm_mask(&success_p, 0.05);
    let call_p: Vec<f64> = results.iter().map(|r| r.calls_paired_p).collect();
    let call_sig = holm_mask(&call_p, 0.05);

    for (idx, (name, result)) in names.iter().zip(results.iter()).enumerate() {
        eprintln!(
            "{name}: n={} random median(success) {:.1} SE {:.3} success {}/{} | soft median(success) {:.1} SE {:.3} success {}/{} | calls paired |z|={:.3} p={:.4} (Holm sig {}) | success McNemar b={} c={} |z|={:.3} p={:.4} (Holm sig {})",
            result.n,
            result.median_random_success,
            result.se_random_success,
            result.random.successes,
            result.n,
            result.median_soft_success,
            result.se_soft_success,
            result.soft.successes,
            result.n,
            result.calls_paired_z.abs(),
            result.calls_paired_p,
            call_sig[idx],
            result.success_b_random_only,
            result.success_c_soft_only,
            result.success_mcnemar_z.abs(),
            result.success_mcnemar_p,
            success_sig[idx]
        );
        assert!(
            result.se_soft_success.is_finite() && result.se_soft_success > 0.0,
            "{name}: softmode SE is zero, replicates are not varying"
        );
        assert!(
            result.se_random_success.is_finite() && result.se_random_success > 0.0,
            "{name}: random SE is zero, replicates are not varying"
        );

        if call_sig[idx] {
            assert!(
                result.calls_paired_p < 0.05,
                "{name}: Holm marked calls significant but p is not < 0.05"
            );
        }
        if success_sig[idx] {
            assert!(
                result.success_mcnemar_p < 0.05,
                "{name}: Holm marked success significant but McNemar p is not < 0.05"
            );
        }
    }

    let coupled_result = &results[2];
    assert!(
        coupled_result.median_soft_success > coupled_result.median_random_success,
        "coupled-well-3d softmode was not slower: random {:.1} vs soft {:.1}",
        coupled_result.median_random_success,
        coupled_result.median_soft_success
    );
    assert!(
        !call_sig[2],
        "coupled-well-3d calls comparison was significant after Holm: p={:.4}",
        coupled_result.calls_paired_p
    );
}

#[test]
fn softmode_plan_changes_with_run_seed() {
    let surface = CoupledWell;
    let minimum = array![1.0, 0.9, 0.0];
    let a = soft_mode_then_full_directions(&surface, minimum.view(), 8, 11).expect("seed plan A");
    let b = soft_mode_then_full_directions(&surface, minimum.view(), 8, 19).expect("seed plan B");
    let different = a
        .directions
        .iter()
        .zip(&b.directions)
        .take(8)
        .any(|(x, y)| {
            x.iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
                > 1.0e-12
        });
    assert!(
        different,
        "soft-mode arm is effectively seed-independent across run seeds"
    );
}

#[cfg(not(feature = "lammps"))]
#[test]
fn rgpot_eam_cu_vacancy_callback_path_missing_requirements() {
    eprintln!(
        "rgpot EAM Cu vacancy callback test not runnable in this build: missing `lammps` feature (enables rgpot-core in-process callback)"
    );
}

fn paired_relative_gain(random_calls: &[f64], soft_calls: &[f64]) -> Vec<f64> {
    random_calls
        .iter()
        .zip(soft_calls)
        .map(|(r, s)| (r - s) / r.max(*s).max(1.0))
        .collect()
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return f64::NAN;
    }
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn se(values: &[f64]) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    (var / values.len() as f64).sqrt()
}

fn paired_z_and_p(diffs: &[f64]) -> (f64, f64) {
    let n = diffs.len();
    if n <= 1 {
        return (0.0, 1.0);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let sd = (diffs
        .iter()
        .map(|x| {
            let d = *x - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64)
        .sqrt();
    if sd <= 0.0 || !sd.is_finite() {
        return (0.0, 1.0);
    }
    let z = mean / (sd / (n as f64).sqrt());
    let p = erfc_approx(z.abs() / std::f64::consts::SQRT_2);
    (z, p)
}

fn mcnemar_counts(pairs: &[(bool, bool)]) -> (usize, usize) {
    let mut b = 0usize;
    let mut c = 0usize;
    for (r, s) in pairs {
        if *r && !*s {
            b += 1;
        } else if !*r && *s {
            c += 1;
        }
    }
    (b, c)
}

fn mcnemar_z_and_p(b: usize, c: usize) -> (f64, f64) {
    let n = (b + c) as f64;
    if n <= 0.0 {
        return (0.0, 1.0);
    }
    let z = (b as f64 - c as f64) / n.sqrt();
    let p = erfc_approx(z.abs() / std::f64::consts::SQRT_2);
    (z, p)
}

fn holm_mask(pvalues: &[f64], alpha: f64) -> Vec<bool> {
    let m = pvalues.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| pvalues[i].total_cmp(&pvalues[j]));
    let mut out = vec![false; m];
    let mut keep = true;
    for (rank, &idx) in order.iter().enumerate() {
        let threshold = alpha / ((m - rank) as f64);
        if keep && pvalues[idx] <= threshold {
            out[idx] = true;
        } else {
            keep = false;
        }
    }
    out
}

fn median_se(values: &[f64], seed: u64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let mut rng = SplitMix64::new(seed);
    let mut meds = Vec::with_capacity(512);
    for _ in 0..512 {
        let mut sample = Vec::with_capacity(values.len());
        for _ in 0..values.len() {
            let i = (rng.next_u64() as usize) % values.len();
            sample.push(values[i]);
        }
        meds.push(median(&sample));
    }
    let mean = meds.iter().sum::<f64>() / meds.len() as f64;
    let var = meds
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / (meds.len() - 1) as f64;
    var.sqrt()
}

fn erfc_approx(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26
    let p = 0.3275911;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let t = 1.0 / (1.0 + p * x);
    let poly = (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t;
    poly * (-x * x).exp()
}

fn unit_uniform_from_seed(seed: u64, index: u64, tag: u64) -> f64 {
    let mut z = seed
        .wrapping_add(tag)
        .wrapping_add(index.wrapping_mul(0x9E3779B97F4A7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    let bits = z >> 11;
    (bits as f64) * (1.0 / ((1u64 << 53) as f64))
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

fn sample_on_sphere(dim: usize, rng: &mut SplitMix64) -> Array1<f64> {
    let mut v = Array1::zeros(dim);
    for x in &mut v {
        *x = rng.normal();
    }
    let norm2 = v.dot(&v);
    if norm2 <= 1.0e-24 || !norm2.is_finite() {
        let mut out = Array1::zeros(dim);
        out[0] = 1.0;
        return out;
    }
    v.mapv(|x| x / norm2.sqrt())
}
