//! Shared sampling statistics: the noise stream and block standard errors.

use crate::SaddleError;

/// Automated equilibration and correlation report for one scalar series.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EquilibrationReport {
    /// Records discarded after the engine's minimum warm-up floor.
    pub discard: usize,
    /// Engine steps discarded: floor plus `discard * stride`.
    pub equilibration_steps: usize,
    /// Integrated autocorrelation time of the retained tail, in records.
    pub tau: f64,
    /// `(N - discard) / g`, with `g = max(1, 2 tau)`.
    pub n_eff: f64,
    /// Absolute difference between the retained tail's half means.
    pub split_half_drift: f64,
    /// Root-sum-square block standard error of the half means.
    pub split_half_error: f64,
    /// Multiple of [`Self::split_half_error`] used by the drift test.
    pub sigma_multiple: f64,
    /// Whether the split-half drift test passed.
    pub drift_passed: bool,
}

/// Equal-width batch layout for `len` records.
///
/// The width is at least `len / blocks` and at least `min_width`. The
/// remainder `len % width` is dropped, so every retained block has that
/// width and none is empty. Returns `(block_count, width)` when at least
/// two blocks fit. Ceiling the width instead (`div_ceil`) turns an odd tail
/// at two blocks into a single block: 15 records become width 8, and only
/// one block fits.
pub(crate) fn batch_blocks(len: usize, blocks: usize, min_width: usize) -> Option<(usize, usize)> {
    if blocks < 2 || len < 2 {
        return None;
    }
    let width = (len / blocks).max(min_width).max(1);
    let block_count = len / width;
    if block_count < 2 {
        return None;
    }
    Some((block_count, width))
}

/// Chodera-style automated equilibration followed by a split-half drift test.
///
/// Candidate discards maximize `(N-t0)/g(t0)`. For long series a
/// coarse-to-fine scan bounds the cost of repeated autocorrelation estimates.
/// The effective-sample floor is the block count passed by the caller,
/// `n_eff >= blocks`: 20 for the plane and thermodynamic-integration
/// samplers, 5 for PAFI, and the adaptive string's own count. It is not a
/// literal shared by every caller.
///
/// The drift threshold is four block standard errors. On a stationary
/// Gaussian series the statistic is a two-sample *t*. With `ν` degrees of
/// freedom the two-sided false-rejection probability at four standard
/// errors is the Student-*t* tail `I_{ν/(ν+16)}(ν/2, 1/2)`. Two block means
/// in each half give `ν = 2` and a 5.72% tail. Seven in each half (a
/// 14-record production sample) give `ν = 12` and a 0.176% tail. That tail
/// is left in place: the threshold is an absolute four-standard-error
/// screen, and the pilot already spends the family-wise budget.
pub(crate) fn equilibrate(
    values: &[f64],
    warmup_floor: usize,
    stride: usize,
    blocks: usize,
) -> Result<EquilibrationReport, SaddleError> {
    const K: f64 = 4.0;
    if values.len() < 4 {
        return Err(SaddleError::Invalid(
            "equilibration detection needs at least four records".into(),
        ));
    }
    // Do not let the optimizer manufacture stationarity from a tiny suffix.
    // A quarter of the trajectory also leaves enough data for the
    // independent split-half diagnostic.
    let minimum_tail = (2 * blocks)
        .max(8)
        .max(values.len().div_ceil(4))
        .min(values.len());
    let last = values.len().saturating_sub(minimum_tail);
    let step = (last / 128).max(1);
    let mut best = (0usize, 0.0, 0.5);
    let evaluate = |t0: usize| {
        let tau = autocorr_time(&values[t0..]);
        let g = (2.0 * tau).max(1.0);
        ((values.len() - t0) as f64 / g, tau)
    };
    for t0 in (0..=last).step_by(step) {
        let (n_eff, tau) = evaluate(t0);
        if n_eff > best.1 {
            best = (t0, n_eff, tau);
        }
    }
    let lo = best.0.saturating_sub(step);
    let hi = (best.0 + step).min(last);
    for t0 in lo..=hi {
        let (n_eff, tau) = evaluate(t0);
        if n_eff > best.1 {
            best = (t0, n_eff, tau);
        }
    }
    let tail = &values[best.0..];
    let middle = tail.len() / 2;
    let halves = [&tail[..middle], &tail[middle..]];
    let half_blocks = blocks.max(2).min(middle.max(2));
    let (mean_a, error_a, _, _) = block_mean(halves[0], half_blocks);
    let (mean_b, error_b, _, _) = block_mean(halves[1], half_blocks);
    let drift = (mean_a - mean_b).abs();
    let error = error_a.hypot(error_b);
    let roundoff = 32.0 * f64::EPSILON * mean_a.abs().max(mean_b.abs()).max(1.0);
    let passed = drift <= (K * error).max(roundoff);
    let report = EquilibrationReport {
        discard: best.0,
        equilibration_steps: warmup_floor.saturating_add(best.0.saturating_mul(stride)),
        tau: best.2,
        n_eff: best.1,
        split_half_drift: drift,
        split_half_error: error,
        sigma_multiple: K,
        drift_passed: passed,
    };
    // The caller passes `blocks`. The floor is that count, not a literal.
    if report.n_eff < blocks as f64 {
        return Err(SaddleError::InsufficientEffectiveSamples {
            equilibration_steps: report.equilibration_steps,
            n_eff: report.n_eff,
            required: blocks as f64,
        });
    }
    let correlation_width = ((2.0 * report.tau).ceil() as usize).max(1);
    if batch_blocks(tail.len(), blocks, correlation_width).is_none() {
        return Err(SaddleError::InsufficientEffectiveSamples {
            equilibration_steps: report.equilibration_steps,
            n_eff: report.n_eff,
            required: 2.0 * correlation_width as f64,
        });
    }
    if !passed {
        return Err(SaddleError::NotEquilibrated {
            equilibration_steps: report.equilibration_steps,
            drift,
            standard_error: error,
            sigma_multiple: K,
        });
    }
    Ok(report)
}

/// Block mean of a correlated series: `(mean, standard error, τ, n_eff)`.
///
/// The block is at least twice the integrated autocorrelation time, so a
/// correlated series is not treated as independent.
pub(crate) fn block_mean(values: &[f64], blocks: usize) -> (f64, f64, f64, f64) {
    let tau = autocorr_time(values);
    let correlated = (2.0 * tau).ceil() as usize;
    let width = (values.len() / blocks).max(correlated).max(1);
    let n_blocks = (values.len() / width).max(2).min(values.len());
    let width = values.len() / n_blocks;
    let mut means = Vec::with_capacity(n_blocks);
    for block in 0..n_blocks {
        let slice = &values[block * width..(block + 1) * width];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        means.push(mean);
    }
    let mean = means.iter().sum::<f64>() / n_blocks as f64;
    let variance = if n_blocks < 2 {
        0.0
    } else {
        means
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f64>()
            / (n_blocks - 1) as f64
    };
    let n_eff = values.len() as f64 / (2.0 * tau).max(1.0);
    (mean, (variance / n_blocks as f64).sqrt(), tau, n_eff)
}

/// Sokal's integrated autocorrelation time, in samples. The sum stops at
/// the first non-positive autocorrelation.
pub(crate) fn autocorr_time(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 4 {
        return 0.5;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let mut variance = 0.0;
    for value in values {
        let d = value - mean;
        variance += d * d;
    }
    variance /= n as f64;
    if variance < 1e-30 {
        return 0.5;
    }
    let mut tau = 0.5;
    let mut lag = 1usize;
    // A 6-point half has `n / 4 == 1`, so `lag < n / 4` never runs. Keep lag 1
    // there; longer series stay on the original cutoff.
    while lag < n / 4 || (n / 4 == 1 && lag == 1) {
        let mut cov = 0.0;
        for i in 0..(n - lag) {
            cov += (values[i] - mean) * (values[i + lag] - mean);
        }
        cov /= (n - lag) as f64;
        let rho = cov / variance;
        if rho <= 0.0 {
            break;
        }
        tau += rho;
        lag += 1;
    }
    tau.max(0.5)
}

/// xoshiro256++. The stream is deterministic for a seed.
pub(crate) struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut s = [0; 4];
        for slot in &mut s {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut mixed = z;
            mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            *slot = mixed ^ (mixed >> 31);
        }
        Self { s }
    }

    pub(crate) fn u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    }

    pub(crate) fn normal(&mut self) -> f64 {
        let u1 = ((self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))).max(1.0e-16);
        let u2 = (self.u64() >> 11) as f64 * (1.0 / ((1u64 << 53) as f64));
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_adjacent_seeds_start_different_streams() {
        for seed in [0, 2, 10, 42, u64::MAX - 1] {
            let mut even = Rng::new(seed);
            let mut odd = Rng::new(seed + 1);
            assert_ne!(even.u64(), odd.u64());
        }
    }

    #[test]
    fn harmonic_ou_relaxation_is_discarded() {
        // Overdamped harmonic relaxation with unit sampling interval:
        // x(t+1) = rho x(t) + noise and exact relaxation time -1/ln(rho).
        let relaxation = 20.0;
        let rho = (-1.0_f64 / relaxation).exp();
        let sigma = (1.0 - rho * rho).sqrt();
        let mut rng = Rng::new(17);
        let mut x = 100.0;
        let mut values = Vec::with_capacity(2_000);
        for _ in 0..2_000 {
            x = rho * x + sigma * rng.normal();
            values.push(x);
        }
        let report = equilibrate(&values, relaxation as usize, 1, 10).unwrap();
        assert_eq!(report.equilibration_steps, relaxation as usize);
        assert!(report.tau > 5.0 && report.tau < 60.0, "tau={}", report.tau);
        assert!(report.drift_passed);
    }

    #[test]
    fn a_deliberately_drifting_series_is_rejected() {
        let values: Vec<f64> = (0..400)
            .map(|i| 0.01 * i as f64 + if i % 2 == 0 { 0.001 } else { -0.001 })
            .collect();
        let result = equilibrate(&values, 25, 2, 8);
        assert!(
            matches!(
                result,
                Err(SaddleError::NotEquilibrated { .. }
                    | SaddleError::InsufficientEffectiveSamples { .. })
            ),
            "{result:?}"
        );
    }

    /// One-sided allowance under the normal approximation to a binomial count.
    /// Three standard errors: `3 sqrt(k (n-k) / n)`, rounded up. For the
    /// 15-record, 2-block cell at `n = 2000` and `k = 1892` this is 31 counts.
    /// The same cell measured independently at 1884 differs by 8 counts, and
    /// `sqrt(1892 * 108 / 2000) = 10.1`, so that gap is 0.8 standard errors.
    fn binomial_lower(passes: usize, trials: usize) -> usize {
        let variance = passes as f64 * (trials - passes) as f64 / trials as f64;
        let allowance = (3.0 * variance.sqrt()).ceil() as usize;
        passes.saturating_sub(allowance)
    }

    #[test]
    fn odd_lengths_keep_two_equal_blocks() {
        for len in (13..=30).filter(|len| len % 2 == 1) {
            let (blocks, width) = batch_blocks(len, 2, 1).expect("two blocks");
            assert_eq!(blocks, 2, "len {len}");
            assert_eq!(width, len / 2, "len {len}");
            assert_eq!(blocks * width + len % 2, len);
        }
        // Ceiling the width is the regression: an odd tail then holds one block.
        for len in 13usize..=30 {
            if len % 2 == 0 {
                continue;
            }
            let width = len.div_ceil(2);
            assert!(len / width < 2, "len {len}");
        }
    }

    /// Pass counts for `Rng::new(seed)`, seeds `1..=200`, standard normal
    /// draws, `equilibrate` with that block count. Rows are record counts 13
    /// through 30. Columns are block counts 2 through 7.
    ///
    /// The allowance is [`binomial_lower`]: three standard errors of the
    /// binomial count, rounded up. The widest allowance on this table is 14
    /// counts (179 of 200). Replacing the floor width by `div_ceil` drops the
    /// two-block odd-length cells from about 180 passes to about 10, which
    /// is outside every one of those allowances. That is the ceiling layout.
    #[test]
    fn iid_batch_pass_rates_match_the_equal_width_layout() {
        const TRIALS: usize = 200;
        const BASE: [[usize; 6]; 18] = [
            [188, 198, 200, 200, 198, 189],
            [190, 198, 200, 200, 200, 195],
            [184, 199, 199, 200, 199, 194],
            [183, 198, 198, 200, 200, 198],
            [185, 198, 198, 200, 200, 199],
            [191, 198, 199, 200, 200, 200],
            [189, 198, 199, 199, 200, 200],
            [194, 196, 199, 199, 200, 200],
            [190, 196, 199, 199, 200, 200],
            [189, 198, 199, 199, 200, 200],
            [194, 196, 200, 200, 200, 200],
            [190, 198, 199, 200, 200, 200],
            [190, 197, 199, 200, 200, 200],
            [193, 197, 199, 200, 200, 200],
            [194, 197, 199, 200, 200, 200],
            [179, 198, 200, 200, 200, 200],
            [182, 194, 199, 199, 200, 200],
            [190, 195, 199, 199, 200, 200],
        ];
        for (row, records) in (13..=30).enumerate() {
            for (col, blocks) in (2..=7).enumerate() {
                let mut passes = 0usize;
                for seed in 1..=TRIALS as u64 {
                    let mut rng = Rng::new(seed);
                    let values: Vec<f64> = (0..records).map(|_| rng.normal()).collect();
                    if equilibrate(&values, 0, 1, blocks).is_ok() {
                        passes += 1;
                    }
                }
                let floor = binomial_lower(BASE[row][col], TRIALS);
                assert!(
                    passes >= floor,
                    "records {records} blocks {blocks}: {passes} passes, base {}, allowance {floor}",
                    BASE[row][col]
                );
            }
        }
    }

    /// Canonical cell from the review: 15 records, 2 blocks, 2000 seeds.
    /// The floor width passes 1892 with this generator. An independent count
    /// of 1884 is 0.8 binomial standard errors lower
    /// (`sqrt(1892 * 108 / 2000) = 10.1`).
    #[test]
    fn fifteen_records_and_two_blocks_keep_the_base_pass_rate() {
        let mut passes = 0usize;
        for seed in 1..=2000 {
            let mut rng = Rng::new(seed);
            let values: Vec<f64> = (0..15).map(|_| rng.normal()).collect();
            if equilibrate(&values, 0, 1, 2).is_ok() {
                passes += 1;
            }
        }
        let floor = binomial_lower(1892, 2000);
        assert!(
            passes >= floor,
            "{passes} passes, base 1892, allowance {floor}"
        );
    }

    fn thinned_ar1(phi: f64, records: usize, stride: usize, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        let innov = (1.0 - phi * phi).sqrt();
        let mut x = rng.normal();
        let mut values = Vec::with_capacity(records);
        let mut step = 0usize;
        while values.len() < records {
            x = phi * x + innov * rng.normal();
            step += 1;
            if step.is_multiple_of(stride) {
                values.push(x);
            }
        }
        values
    }

    /// Stationary AR(1), recorded every 5th step (the default production
    /// stride), 80 records, 20 blocks, seeds 1..=300.
    ///
    /// At `phi = 0.9` the floor refuses 65 and at `phi = 0.95` it refuses 207.
    /// The review's 60 and 213 differ by 5 and 6. The binomial standard
    /// errors are `sqrt(65 * 235 / 300) = 7.1` and `sqrt(207 * 93 / 300) = 8.0`,
    /// so both gaps are under one standard error. The acceptance window is
    /// three of those standard errors, 22 and 24 counts. A literal floor of
    /// 4 refuses none of these runs: every retained `n_eff` is at least 4.
    #[test]
    fn caller_block_floor_refuses_at_the_base_rate() {
        let expected = [(0.9_f64, 65usize, 22usize), (0.95, 207, 24)];
        for (phi, base, allowance) in expected {
            let mut refused = 0usize;
            for seed in 1..=300 {
                let values = thinned_ar1(phi, 80, 5, seed);
                match equilibrate(&values, 0, 1, 20) {
                    Ok(_) => {}
                    Err(SaddleError::InsufficientEffectiveSamples { required, .. }) => {
                        assert_eq!(required, 20.0);
                        refused += 1;
                    }
                    Err(error) => panic!("phi {phi} seed {seed}: {error}"),
                }
            }
            let low = base.saturating_sub(allowance);
            let high = base + allowance;
            assert!(
                (low..=high).contains(&refused),
                "phi {phi}: refused {refused}, base {base}, window {low}..={high}"
            );
        }
    }
}
