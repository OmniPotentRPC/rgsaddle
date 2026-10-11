//! Force-call budget comparison, the rank-invariant digest, and the
//! seed-2 climbing trace.
//!
//! The comparison and the trace are ignored. Run them in release:
//! `cargo test --release --test autoneb_gate -- --ignored --test-threads=1`

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::autoneb::AutoNebStage;
use rgsaddle::{
    AutoNeb, AutoNebTuning, BandConfig, BandSession, BandStatus, BandSurface, CiConfig, SaddleError,
};
use sha2::{Digest, Sha256};

const SEEDS: u64 = 32;
const BOOTSTRAP: usize = 10_000;

#[derive(Clone, Copy)]
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        let px = 2.0 * MB_AA[i] * dx + MB_B[i] * dy;
        let py = MB_B[i] * dx + 2.0 * MB_C[i] * dy;
        energy += weight;
        gx += weight * px;
        gy += weight * py;
    }
    (energy, [gx, gy])
}

#[derive(Clone, Copy)]
struct MullerBrown;

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (energy, g) = muller_brown(row[0], row[1]);
            energies[i] = energy;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = 0.0;
        }
        Ok(())
    }
}

struct Budget<S> {
    inner: S,
    rows: AtomicUsize,
    limit: usize,
}

impl<S> Budget<S> {
    fn new(inner: S, limit: usize) -> Self {
        Self {
            inner,
            rows: AtomicUsize::new(0),
            limit,
        }
    }

    fn rows(&self) -> usize {
        self.rows.load(Ordering::Relaxed)
    }
}

impl<S: BandSurface> BandSurface for Budget<S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let added = positions.nrows();
        let previous = self.rows.fetch_add(added, Ordering::Relaxed);
        if previous >= self.limit || previous + added > self.limit {
            return Err(SaddleError::Invalid("force-call budget".into()));
        }
        self.inner.eval(positions, energies, gradients)
    }
}

fn is_budget(err: &SaddleError) -> bool {
    matches!(err, SaddleError::Invalid(message) if message.contains("force-call budget"))
}

/// Counter-based unit in (-1, 1), keyed by `(seed, image, coordinate)`.
fn unit(seed: u64, image: usize, coord: usize) -> f64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (image as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (coord as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    let mantissa = (z >> 11) as f64 / ((1u64 << 53) as f64);
    mantissa.mul_add(2.0, -1.0)
}

fn perturb(points: [[f64; 3]; 3], seed: u64) -> Array2<f64> {
    let mut band = Array2::from_shape_vec((3, 3), points.into_iter().flatten().collect()).unwrap();
    for image in 0..3 {
        for coord in 0..2 {
            band[(image, coord)] += 0.02 * unit(seed, image, coord);
        }
    }
    band
}

fn through_seven(three: &Array2<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((7, 3));
    for i in 0..7 {
        let t = i as f64 / 6.0;
        let (left, right, mix) = if t <= 0.5 {
            (0, 1, t / 0.5)
        } else {
            (1, 2, (t - 0.5) / 0.5)
        };
        for coord in 0..3 {
            out[(i, coord)] = three[(left, coord)].mul_add(1.0 - mix, three[(right, coord)] * mix);
        }
    }
    out
}

fn config(force_tol: f64, full_band: bool) -> BandConfig {
    BandConfig {
        force_tol,
        max_move: 0.1,
        method: rgmin::Method::Lbfgs { memory: 10 },
        climbing: Some(CiConfig {
            trigger_factor: 1.0,
            trigger_force: f64::INFINITY,
        }),
        autoneb: Some(AutoNebTuning {
            n_simul: 3,
            n_max: 7,
            local_force_tol: 1e-1,
            local_max_steps: 100_000,
            climb_max_steps: 100_000,
            space_energy_ratio: 0.5,
            climb: true,
            full_band,
            threads: 1,
        }),
        ..BandConfig::default()
    }
}

struct Outcome {
    converged: bool,
    censored: bool,
    iters: usize,
    calls: usize,
    energy: f64,
    fmax: f64,
    reference: bool,
    stages: Vec<(AutoNebStage, usize)>,
}

fn near(pos: ArrayView2<f64>, ci: usize, saddles: &[[f64; 2]]) -> bool {
    saddles.iter().any(|saddle| {
        let dx = pos[(ci, 0)] - saddle[0];
        let dy = pos[(ci, 1)] - saddle[1];
        (dx * dx + dy * dy).sqrt() < 0.05 && pos[(ci, 2)].abs() < 0.05
    })
}

fn run_auto<S: BandSurface + Copy>(
    surface: S,
    three: &Array2<f64>,
    force_tol: f64,
    full_band: bool,
    budget: usize,
    saddles: &[[f64; 2]],
) -> Outcome {
    let mut search = AutoNeb::new(config(force_tol, full_band), three.clone()).unwrap();
    let counted = Budget::new(surface, budget);
    let result = search.run(&counted);
    let calls = counted.rows();
    let censored = result.as_ref().err().is_some_and(is_budget);
    if let Err(err) = &result
        && !is_budget(err)
    {
        panic!("AutoNEB failed: {err}");
    }
    let (converged, iters, fmax, ci) = match result {
        Ok(report) => (
            report.status == BandStatus::Converged,
            report.iteration,
            report.max_force,
            report.ci_index,
        ),
        Err(_) => {
            let last = search.stages().last();
            let step = last.and_then(|stage| stage.steps.last());
            (
                false,
                step.map(|step| step.iteration).unwrap_or(0),
                step.map(|step| step.max_force).unwrap_or(f64::NAN),
                last.and_then(|stage| stage.ci_index),
            )
        }
    };
    let energy = ci
        .and_then(|index| search.energies().get(index).copied().flatten())
        .unwrap_or(f64::NAN);
    let reference = ci.is_some_and(|index| near(search.positions(), index, saddles));
    Outcome {
        converged: converged && !censored,
        censored,
        iters,
        calls,
        energy,
        fmax,
        reference: reference && converged && !censored,
        stages: search
            .stages()
            .iter()
            .map(|stage| (stage.stage, stage.calls))
            .collect(),
    }
}

fn run_cineb<S: BandSurface + Copy>(
    surface: S,
    seven: &Array2<f64>,
    force_tol: f64,
    budget: usize,
    saddles: &[[f64; 2]],
) -> Outcome {
    let mut band = config(force_tol, false);
    band.autoneb = None;
    let mut session = BandSession::new(band, seven.clone()).unwrap();
    let counted = Budget::new(surface, budget);
    let mut censored = false;
    let mut report = None;
    loop {
        match session.step(&counted) {
            Ok(step) => {
                let done = step.status == BandStatus::Converged;
                report = Some(step);
                if done {
                    break;
                }
            }
            Err(err) if is_budget(&err) => {
                censored = true;
                break;
            }
            Err(err) => panic!("CI-NEB failed: {err}"),
        }
    }
    let calls = counted.rows();
    let (converged, iters, fmax, ci, energy) = if let Some(report) = report {
        let energy = report
            .ci_index
            .and_then(|index| session.evaluation().map(|eval| eval.energies[index]));
        (
            report.status == BandStatus::Converged,
            report.iteration,
            report.max_force,
            report.ci_index,
            energy.unwrap_or(f64::NAN),
        )
    } else {
        (false, 0, f64::NAN, None, f64::NAN)
    };
    let reference = ci.is_some_and(|index| near(session.positions(), index, saddles));
    Outcome {
        converged: converged && !censored,
        censored,
        iters,
        calls,
        energy,
        fmax,
        reference: reference && converged && !censored,
        stages: Vec::new(),
    }
}

struct Row {
    system: &'static str,
    method: &'static str,
    seed: u64,
    outcome: Outcome,
    wall: f64,
    endpoint: &'static str,
}

fn csv_row(row: &Row) -> String {
    format!(
        "{system},{method},{seed},{converged},{iters},{calls},{energy:.8},{fmax:.8e},{wall:.6},{endpoint},{censored},{reference}",
        system = row.system,
        method = row.method,
        seed = row.seed,
        converged = u8::from(row.outcome.converged),
        iters = row.outcome.iters,
        calls = row.outcome.calls,
        energy = row.outcome.energy,
        fmax = row.outcome.fmax,
        wall = row.wall,
        endpoint = row.endpoint,
        censored = u8::from(row.outcome.censored),
        reference = u8::from(row.outcome.reference),
    )
}

fn median_range(values: &mut [f64]) -> (f64, f64, f64) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    };
    (median, values[0], *values.last().unwrap())
}

struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn phi(z: f64) -> f64 {
    let sign = z.is_sign_negative();
    let z = z.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let density = 0.3989422804014327 * (-0.5 * z * z).exp();
    let poly = t
        * (0.319381530
            + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = density * poly;
    if sign { tail } else { 1.0 - tail }
}

#[allow(clippy::excessive_precision)]
fn inv_phi(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411385e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
}

fn bca_mean(values: &[f64]) -> (f64, f64, f64) {
    let theta = mean(values);
    let n = values.len();
    let mut rng = SplitMix(0xA17E_BEEF_5EED);
    let mut boots = Vec::with_capacity(BOOTSTRAP);
    for _ in 0..BOOTSTRAP {
        let mut total = 0.0;
        for _ in 0..n {
            total += values[rng.below(n)];
        }
        boots.push(total / n as f64);
    }
    boots.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_less = boots.iter().filter(|value| **value < theta).count();
    let z0 = inv_phi(n_less as f64 / boots.len() as f64);
    let mut jack = Vec::with_capacity(n);
    for skip in 0..n {
        let mut total = 0.0;
        for (index, value) in values.iter().enumerate() {
            if index != skip {
                total += value;
            }
        }
        jack.push(total / (n - 1) as f64);
    }
    let jack_mean = mean(&jack);
    let mut cubic = 0.0;
    let mut square = 0.0;
    for value in &jack {
        let delta = jack_mean - value;
        cubic += delta * delta * delta;
        square += delta * delta;
    }
    let accel = if square > 0.0 {
        cubic / (6.0 * square.powf(1.5))
    } else {
        0.0
    };
    let percentile = |alpha: f64| -> f64 {
        let z = inv_phi(alpha);
        let numer = z0 + z;
        let adjusted = phi(z0 + numer / (1.0 - accel * numer));
        let adjusted = adjusted.clamp(0.0, 1.0);
        let position = adjusted * (boots.len() - 1) as f64;
        let lo = position.floor() as usize;
        let hi = position.ceil() as usize;
        let frac = position - lo as f64;
        boots[lo].mul_add(1.0 - frac, boots[hi] * frac)
    };
    (theta, percentile(0.025), percentile(0.975))
}

struct SurfaceSpec<'a, S> {
    name: &'static str,
    endpoint: &'static str,
    force_tol: f64,
    budget: usize,
    points: [[f64; 3]; 3],
    saddles: &'a [[f64; 2]],
    surface: S,
}

fn compare_one<S: BandSurface + Copy>(spec: &SurfaceSpec<'_, S>, rows: &mut Vec<Row>) {
    for seed in 0..SEEDS {
        let three = perturb(spec.points, seed);
        let seven = through_seven(&three);
        let started = Instant::now();
        let cineb = run_cineb(
            spec.surface,
            &seven,
            spec.force_tol,
            spec.budget,
            spec.saddles,
        );
        rows.push(Row {
            system: spec.name,
            method: "cineb",
            seed,
            outcome: cineb,
            wall: started.elapsed().as_secs_f64(),
            endpoint: spec.endpoint,
        });
        let started = Instant::now();
        let auto = run_auto(
            spec.surface,
            &three,
            spec.force_tol,
            false,
            spec.budget,
            spec.saddles,
        );
        rows.push(Row {
            system: spec.name,
            method: "autoneb",
            seed,
            outcome: auto,
            wall: started.elapsed().as_secs_f64(),
            endpoint: spec.endpoint,
        });
        let started = Instant::now();
        let full = run_auto(
            spec.surface,
            &three,
            spec.force_tol,
            true,
            spec.budget,
            spec.saddles,
        );
        rows.push(Row {
            system: spec.name,
            method: "autoneb_full",
            seed,
            outcome: full,
            wall: started.elapsed().as_secs_f64(),
            endpoint: spec.endpoint,
        });
    }
}

fn summarize(rows: &[Row]) {
    let systems = ["double_well", "muller_bc", "muller_ab"];
    let methods = ["cineb", "autoneb", "autoneb_full"];
    for system in systems {
        for method in methods {
            let matched: Vec<&Row> = rows
                .iter()
                .filter(|row| row.system == system && row.method == method)
                .collect();
            let successes: Vec<&Row> = matched
                .iter()
                .copied()
                .filter(|row| row.outcome.converged)
                .collect();
            let censored = matched.iter().filter(|row| row.outcome.censored).count();
            let refs = matched.iter().filter(|row| row.outcome.reference).count();
            eprintln!(
                "{system} {method}: n={} successes={} censored={} reference={}",
                matched.len(),
                successes.len(),
                censored,
                refs
            );
            if !successes.is_empty() {
                let mut calls: Vec<f64> = successes
                    .iter()
                    .map(|row| row.outcome.calls as f64)
                    .collect();
                let (median, lo, hi) = median_range(&mut calls);
                eprintln!("  calls among successes median={median:.1} range={lo:.1}..{hi:.1}");
            }
        }
        for method in ["autoneb", "autoneb_full"] {
            let mut ratios = Vec::new();
            for seed in 0..SEEDS {
                let base = rows
                    .iter()
                    .find(|row| row.system == system && row.method == "cineb" && row.seed == seed);
                let other = rows
                    .iter()
                    .find(|row| row.system == system && row.method == method && row.seed == seed);
                if let (Some(base), Some(other)) = (base, other)
                    && base.outcome.converged
                    && other.outcome.converged
                    && base.outcome.calls > 0
                {
                    ratios.push(other.outcome.calls as f64 / base.outcome.calls as f64);
                }
            }
            if ratios.len() >= 2 {
                let (mean_ratio, lo, hi) = bca_mean(&ratios);
                eprintln!(
                    "  {method}/cineb call ratio on {n} paired successes: mean={mean_ratio:.3} BCa95=[{lo:.3}, {hi:.3}]",
                    n = ratios.len()
                );
            } else {
                eprintln!("  {method}/cineb paired successes: {}", ratios.len());
            }
            for stage in [
                AutoNebStage::Initial,
                AutoNebStage::Insertion,
                AutoNebStage::Climb,
                AutoNebStage::FullBand,
            ] {
                let mut calls = Vec::new();
                for row in rows.iter().filter(|row| {
                    row.system == system && row.method == method && row.outcome.converged
                }) {
                    let total: usize = row
                        .outcome
                        .stages
                        .iter()
                        .filter(|(kind, _)| *kind == stage)
                        .map(|(_, calls)| *calls)
                        .sum();
                    if total > 0 || stage == AutoNebStage::FullBand && method == "autoneb_full" {
                        calls.push(total as f64);
                    }
                }
                if calls.is_empty() {
                    continue;
                }
                let (median, lo, hi) = median_range(&mut calls);
                eprintln!(
                    "  {method} stage {stage:?} calls median={median:.1} range={lo:.1}..{hi:.1} n={}",
                    calls.len()
                );
            }
        }
    }
}

fn old_seed_two() {
    // The previous protocol's Müller-Brown seed 2: a sine bump, then
    // `seed * 0.01 * sin(0.17 i)` on the interiors. The sine is that
    // protocol's bump, not the counter used for the budget comparison.
    let start = [-0.05001082_f64, 0.46669410];
    let end = [0.62349940_f64, 0.02803776];
    let dx = end[0] - start[0];
    let dy = end[1] - start[1];
    let norm = (dx * dx + dy * dy).sqrt();
    let (px, py) = (-dy / norm, dx / norm);
    let mut band = Array2::zeros((7, 3));
    for i in 0..7 {
        let t = i as f64 / 6.0;
        let bump = 0.15 * (std::f64::consts::PI * t).sin();
        band[(i, 0)] = start[0] + dx * t + bump * px;
        band[(i, 1)] = start[1] + dy * t + bump * py;
    }
    for i in 1..6 {
        band[(i, 1)] += 2.0 * 0.01 * ((i as f64) * 0.17).sin();
    }
    let mut three = Array2::zeros((3, 3));
    for (k, index) in [0, 3, 6].into_iter().enumerate() {
        three.row_mut(k).assign(&band.row(index));
    }
    let mut cfg = config(1e-2, true);
    cfg.climbing = Some(CiConfig {
        trigger_factor: 0.5,
        trigger_force: 0.0,
    });
    if let Some(tuning) = cfg.autoneb.as_mut() {
        tuning.local_max_steps = 40;
        tuning.climb_max_steps = 800;
        tuning.local_force_tol = 1e-1;
    }
    let mut search = AutoNeb::new(cfg, three).unwrap();
    let report = search.run(&MullerBrown).unwrap();
    eprintln!(
        "seed2 status={:?} calls={} iters={} sessions_opened={}",
        report.status,
        report.surface_rows,
        report.iteration,
        search.sessions_opened()
    );
    for stage in search.stages() {
        if stage.stage != AutoNebStage::FullBand {
            continue;
        }
        eprintln!(
            "seed2 full band opened_session={} calls={} steps={}",
            stage.opened_session,
            stage.calls,
            stage.steps.len()
        );
        let mut previous = None;
        let mut moved = false;
        for step in &stage.steps {
            if previous.is_some() && previous != step.ci_index {
                moved = true;
            }
            previous = step.ci_index;
            eprintln!(
                "seed2 full step iter={} ci={:?} fmax={:.6e}",
                step.iteration, step.ci_index, step.max_force
            );
        }
        eprintln!(
            "seed2 climbing image moved={moved} session restarted={}",
            stage.opened_session
        );
    }
}

#[test]
#[ignore = "release budget comparison; --test-threads=1"]
fn gate_force_call_budget() {
    let dw = [0.0, 0.0];
    let ts_bc = [0.212486582001, 0.292988325083];
    let ts_ac = [-0.822001558732, 0.624312802418];
    let mut rows = Vec::new();
    compare_one(
        &SurfaceSpec {
            name: "double_well",
            endpoint: "minima",
            force_tol: 1e-3,
            budget: 2_000,
            points: [[-1.0, 0.0, 0.0], [0.0, 0.3, 0.0], [1.0, 0.0, 0.0]],
            saddles: &[dw],
            surface: DoubleWell,
        },
        &mut rows,
    );
    compare_one(
        &SurfaceSpec {
            name: "muller_bc",
            endpoint: "B-C",
            force_tol: 1e-2,
            budget: 5_000,
            points: [
                [-0.050010822738, 0.466694104872, 0.0],
                [0.286744291096, 0.247365931700, 0.0],
                [0.623499404931, 0.028037758529, 0.0],
            ],
            saddles: &[ts_bc],
            surface: MullerBrown,
        },
        &mut rows,
    );
    compare_one(
        &SurfaceSpec {
            name: "muller_ab",
            endpoint: "A-B",
            force_tol: 1e-2,
            budget: 5_000,
            points: [
                [-0.558223634633, 1.44172584195, 0.0],
                [-0.050010822738, 0.466694104872, 0.0],
                [0.623499404931, 0.028037758529, 0.0],
            ],
            saddles: &[ts_ac, ts_bc],
            surface: MullerBrown,
        },
        &mut rows,
    );
    let mut lines = vec![
        "system,method,seed,converged,iters,calls,energy,fmax,wall_s,endpoint,censored,reference"
            .to_string(),
    ];
    for row in &rows {
        lines.push(csv_row(row));
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/autoneb_gate.csv");
    fs::write(&path, lines.join("\n") + "\n").unwrap();
    eprintln!("csv {}", path.display());
    summarize(&rows);
    old_seed_two();
    let (check, _, _) = bca_mean(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert!((check - 3.0).abs() < 1e-9);
    assert!((inv_phi(0.975) - 1.95996398454).abs() < 1e-6);
}

fn feed_f64(hasher: &mut Sha256, value: f64) {
    hasher.update(value.to_le_bytes());
}

fn stage_byte(stage: AutoNebStage) -> u8 {
    match stage {
        AutoNebStage::Initial => 0,
        AutoNebStage::Insertion => 1,
        AutoNebStage::Climb => 2,
        AutoNebStage::FullBand => 3,
    }
}

fn digest_search(search: &AutoNeb, surface_rows: usize) -> String {
    let mut hasher = Sha256::new();
    for index in search.insertions() {
        hasher.update((*index as u64).to_le_bytes());
    }
    hasher.update((search.sessions_opened() as u64).to_le_bytes());
    for stage in search.stages() {
        let ci = stage.ci_index.map(|index| index as i64).unwrap_or(-1);
        hasher.update(ci.to_le_bytes());
    }
    hasher.update((surface_rows as u64).to_le_bytes());
    for value in search.positions().iter() {
        feed_f64(&mut hasher, *value);
    }
    for energy in search.energies() {
        match energy {
            Some(value) => {
                hasher.update([1]);
                feed_f64(&mut hasher, *value);
            }
            None => hasher.update([0]),
        }
    }
    let mut samples = search.samples();
    samples.sort_by_key(|sample| (stage_byte(sample.stage), sample.step, sample.image));
    for sample in samples {
        hasher.update([stage_byte(sample.stage)]);
        hasher.update(sample.step.to_le_bytes());
        hasher.update((sample.image as u64).to_le_bytes());
        for value in sample.position {
            feed_f64(&mut hasher, value);
        }
    }
    format!("{:x}", hasher.finalize())
}

fn hash_band<S: BandSurface + Copy>(
    surface: S,
    points: [[f64; 3]; 3],
    force_tol: f64,
    threads: usize,
) -> String {
    let mut cfg = config(force_tol, false);
    if let Some(tuning) = cfg.autoneb.as_mut() {
        tuning.threads = threads;
        tuning.n_max = 6;
        tuning.local_max_steps = 12;
        tuning.climb_max_steps = 20;
    }
    let initial = Array2::from_shape_vec((3, 3), points.into_iter().flatten().collect()).unwrap();
    let mut search = AutoNeb::new(cfg, initial).unwrap();
    let report = search.run(&surface).unwrap();
    digest_search(&search, report.surface_rows)
}

#[test]
fn autoneb_digest_matches_at_one_two_three_and_eight_threads() {
    let dw = [[-1.0, 0.0, 0.0], [0.0, 0.3, 0.0], [1.0, 0.0, 0.0]];
    let mb = [
        [-0.050010822738, 0.466694104872, 0.0],
        [0.286744291096, 0.247365931700, 0.0],
        [0.623499404931, 0.028037758529, 0.0],
    ];
    for (name, points, tol, surface) in [
        ("double_well", dw, 1e-2, SurfaceKind::Well),
        ("muller_brown", mb, 5e-2, SurfaceKind::Muller),
    ] {
        let one = hash_kind(surface, points, tol, 1);
        eprintln!("sha256 {name} threads=1 {one}");
        for threads in [2, 3, 8] {
            let other = hash_kind(surface, points, tol, threads);
            assert_eq!(other, one, "{name} threads={threads}");
            eprintln!("sha256 {name} threads={threads} {other}");
        }
    }
}

#[derive(Clone, Copy)]
enum SurfaceKind {
    Well,
    Muller,
}

fn hash_kind(kind: SurfaceKind, points: [[f64; 3]; 3], tol: f64, threads: usize) -> String {
    match kind {
        SurfaceKind::Well => hash_band(DoubleWell, points, tol, threads),
        SurfaceKind::Muller => hash_band(MullerBrown, points, tol, threads),
    }
}
