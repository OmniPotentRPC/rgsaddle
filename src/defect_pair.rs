//! Saddles and finite-temperature rates of a defect pair.
//!
//! The caller supplies a potential, a periodic cell, one relaxed
//! configuration, and a catalogue of hops (the product geometry of each
//! symmetry-distinct move). Nothing here is specific to a material. For
//! each hop the search
//!
//! 1. minimizes the reactant and the product,
//! 2. climbs from the reactant side of that hop with the solid-state dimer
//!    ([`crate::SolidDimer`]), on a frozen cell or on the joint chart,
//! 3. requires one negative projected eigenvalue from [`crate::index_at`],
//! 4. pushes off both sides of the unstable mode, minimizes, and keeps the
//!    hop only when those minima are the reactant and the product,
//! 5. builds a climbing-image solid-state NEB through that saddle, in two
//!    phases: a plain relaxation, then the climb,
//! 6. integrates the projected average force along that path
//!    ([`crate::pafi`]) and shoots the Bennett–Chandler factor
//!    ([`crate::bennett`]) from the free-energy maximum.
//!
//! The rate at temperature `T` is
//!
//! `k(T) = κ ν exp(-ΔF / kT)`.
//!
//! `ΔF` is the PAFI potential of mean force from the reactant hyperplane
//! to the dividing surface. `ν` is the reactant attempt frequency along
//! the path tangent, `ν = (ω / 2π)` with `ω = sqrt(τ · H τ / m)`. `κ` is
//! the Bennett–Chandler plateau. In the harmonic limit `ΔF` holds the
//! transverse entropy and this expression agrees with Vineyard's rate
//! when `κ = 1`. The Vineyard rate `ν_h exp(-ΔE0 / kT)` is reported beside
//! it. Vineyard, J. Phys. Chem. Solids 3, 121 (1957),
//! doi:10.1016/0022-3697(57)90059-8.
//!
//! A hop is `toward`, `away`, or `tangential` by the change in the
//! minimum-image separation of the mobile-defect centroid from the fixed
//! defect. The shell label is the caller's.
//!
//! [`RateCatalogue`] is the document `rgsaddle.defect_pair_rates`,
//! version 1. An amsel absorbing-Markov step reads `rate_Hz` as the
//! first-order exit rate of one channel. The channel key is
//! `(shell, direction, hop, temperature, cell_repeats)`.

use ndarray::{Array1, Array2, ArrayView1, s};
use rayon::prelude::*;
use rgmin::{FireKind, Method};

use crate::band::{BandConfig, BandStatus, CiConfig};
use crate::bennett::ShootingConfig;
use crate::cell_log::{M3, m3_det, m3_inv, m3_mul, m3_to_row9};
use crate::eigensolve::exact_eigh;
use crate::error::SaddleError;
use crate::minmode::{MinModeStatus, PointSurface};
#[cfg(feature = "ohtst")]
use crate::ohtst::{Counting, TransmissionConfig};
use crate::pafi::{self, ArcPath, PafiConfig};
use crate::relax::{RelaxConfig, max_atom_force, relax_to_tolerance};
use crate::solid_dimer::{
    CellMask, JointChart, SolidDimer, SolidDimerConfig, finite_difference_cauchy_stress,
};
use crate::ssneb::SolidNeb;
use crate::strain_rate::{
    self, ElasticDipole, Json, double_dot, obj_get, parse_json, required_f64, required_matrix,
    required_str, split_translational_modes,
};

/// Which way the mobile defect moves relative to the fixed defect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectionClass {
    /// The separation shrinks by more than the tolerance.
    Toward,
    /// The separation grows by more than the tolerance.
    Away,
    /// The separation changes by at most the tolerance.
    Tangential,
}

impl DirectionClass {
    /// `toward`, `away`, or `tangential`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Toward => "toward",
            Self::Away => "away",
            Self::Tangential => "tangential",
        }
    }

    fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "toward" => Ok(Self::Toward),
            "away" => Ok(Self::Away),
            "tangential" => Ok(Self::Tangential),
            _ => Err(SaddleError::Invalid(
                "direction must be toward, away, or tangential".into(),
            )),
        }
    }
}

/// Classify a hop by the change in separation.
///
/// `after - before < -tol` is toward the fixed defect, `> tol` is away,
/// and the rest is tangential.
pub fn classify_separation(
    before: f64,
    after: f64,
    tol: f64,
) -> Result<DirectionClass, SaddleError> {
    if !(before.is_finite() && after.is_finite() && tol.is_finite() && tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "separation and tolerance must be finite, and the tolerance non-negative".into(),
        ));
    }
    let delta = after - before;
    if delta < -tol {
        Ok(DirectionClass::Toward)
    } else if delta > tol {
        Ok(DirectionClass::Away)
    } else {
        Ok(DirectionClass::Tangential)
    }
}

/// `k = 4 π D R`, the Smoluchowski rate for hitting a sphere.
pub fn smoluchowski_rate(diffusivity: f64, radius: f64) -> Result<f64, SaddleError> {
    if !(diffusivity.is_finite() && diffusivity >= 0.0 && radius.is_finite() && radius >= 0.0) {
        return Err(SaddleError::Invalid(
            "diffusivity and radius must be finite and non-negative".into(),
        ));
    }
    Ok(4.0 * std::f64::consts::PI * diffusivity * radius)
}

/// Continuum capture probability `R_eff / distance` inverted for the radius.
pub fn effective_capture_radius(probability: f64, distance: f64) -> Result<f64, SaddleError> {
    if !(probability.is_finite() && (0.0..=1.0).contains(&probability)) {
        return Err(SaddleError::Invalid(
            "the capture probability must lie in [0, 1]".into(),
        ));
    }
    if !(distance.is_finite() && distance > 0.0) {
        return Err(SaddleError::Invalid(
            "the distance from the absorber must be finite and positive".into(),
        ));
    }
    Ok(probability * distance)
}

/// Volume in `k = frequency * volume`, the relation used for a migration frequency.
pub fn volume_from_frequency(rate: f64, frequency: f64) -> Result<f64, SaddleError> {
    if !(rate.is_finite() && rate >= 0.0 && frequency.is_finite() && frequency > 0.0) {
        return Err(SaddleError::Invalid(
            "the rate must be non-negative and the frequency positive".into(),
        ));
    }
    Ok(rate / frequency)
}

fn boltzmann_factor(energy: f64, thermal: f64) -> f64 {
    if !(thermal.is_finite() && thermal > 0.0) {
        return 1.0;
    }
    let argument = energy / thermal;
    if argument > 80.0 {
        80.0_f64.exp()
    } else if argument < -80.0 {
        0.0
    } else {
        argument.exp()
    }
}

/// `∫_inner^∞ exp(E(r)/kT) r^{-2} dr`, with `E(∞) = 0`.
///
/// The substitution `u = 1/r` turns the integrand into `exp(E(1/u)/kT)`.
/// At `u = 0` the factor is 1. A vanishing interaction integrates to `1/inner`.
pub fn debye_resistance<F>(inner: f64, thermal: f64, energy: F) -> Result<f64, SaddleError>
where
    F: Fn(f64) -> f64,
{
    if !(inner.is_finite() && inner > 0.0 && thermal.is_finite() && thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "the Debye inner radius and kT must be finite and positive".into(),
        ));
    }
    let panels = 512usize;
    let step = (1.0 / inner) / panels as f64;
    let mut sum = 0.0;
    for i in 0..=panels {
        let u = step * i as f64;
        let factor = if i == 0 {
            1.0
        } else {
            let value = energy(1.0 / u);
            if !value.is_finite() {
                return Err(SaddleError::NonFinite("Debye interaction"));
            }
            boltzmann_factor(value, thermal)
        };
        let weight = if i == 0 || i == panels {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * factor;
    }
    let resistance = sum * step / 3.0;
    if !(resistance.is_finite() && resistance > 0.0) {
        return Err(SaddleError::NonFinite("Debye resistance"));
    }
    Ok(resistance)
}

/// Debye–Smoluchowski radius, the reciprocal of [`debye_resistance`].
///
/// A perfect absorber at `inner` with `E = 0` has this radius equal to `inner`.
pub fn debye_radius<F>(inner: f64, thermal: f64, energy: F) -> Result<f64, SaddleError>
where
    F: Fn(f64) -> f64,
{
    Ok(1.0 / debye_resistance(inner, thermal, energy)?)
}

/// Capture probability at `distance` for a perfect absorber at `inner`.
///
/// `E = 0` gives `inner / distance`. An attractive interaction raises it.
pub fn debye_hit_probability<F>(
    inner: f64,
    distance: f64,
    thermal: f64,
    energy: F,
) -> Result<f64, SaddleError>
where
    F: Fn(f64) -> f64,
{
    if !(distance.is_finite() && distance > 0.0) {
        return Err(SaddleError::Invalid(
            "the starting distance must be finite and positive".into(),
        ));
    }
    if distance <= inner {
        return Ok(1.0);
    }
    let far = debye_resistance(distance, thermal, &energy)?;
    let all = debye_resistance(inner, thermal, energy)?;
    Ok((far / all).clamp(0.0, 1.0))
}

/// Replace the free exterior of a chain by a strained one.
///
/// `chain_radius` is the effective radius of the discrete problem with a
/// free exterior beyond `cutoff`. The free piece `1/cutoff` is removed and
/// the Debye integral is put in its place. `E = 0` returns `chain_radius`.
pub fn debye_matched_radius<F>(
    chain_radius: f64,
    cutoff: f64,
    thermal: f64,
    energy: F,
) -> Result<f64, SaddleError>
where
    F: Fn(f64) -> f64,
{
    if !(chain_radius.is_finite() && chain_radius > 0.0 && cutoff.is_finite() && cutoff > 0.0) {
        return Err(SaddleError::Invalid(
            "the chain radius and the cutoff must be finite and positive".into(),
        ));
    }
    let exterior = debye_resistance(cutoff, thermal, energy)?;
    let inverse = 1.0 / chain_radius - 1.0 / cutoff + exterior;
    if !(inverse.is_finite() && inverse > 0.0) {
        return Err(SaddleError::Invalid(
            "the matched Debye radius is not positive".into(),
        ));
    }
    Ok(1.0 / inverse)
}

/// Survival against a perfectly absorbing sphere at the origin of a periodic cube.
///
/// Times are reduced, `s = D t` in Å². `at_starts[i][j]` is the probability
/// that a walker from `starts[i]` has not touched any image of the sphere by
/// `reduced_times[j]`. `cell_mean[j]` is the same average over a uniform start
/// in the cell, inside the sphere included.
#[derive(Debug, Clone)]
pub struct PeriodicSinkSurvival {
    /// Sample times `s = D t`, in Å².
    pub reduced_times: Vec<f64>,
    /// Survival from each start, one series per start.
    pub at_starts: Vec<Vec<f64>>,
    /// Survival from a uniform start in the cell.
    pub cell_mean: Vec<f64>,
}

struct OctantGrid {
    nodes: usize,
    side: usize,
    spacing: f64,
    half: f64,
    absorbing: Vec<bool>,
    neighbors: Vec<[usize; 6]>,
    coupling: Vec<[f64; 6]>,
    sink: Vec<f64>,
    weight: Vec<f64>,
}

impl OctantGrid {
    /// The backward equation `∂S/∂s = ∇²S` on `[0, L/2]³`.
    ///
    /// Mirror symmetry of the sphere and of the lattice of images makes every
    /// face of the octant reflecting. Next to the sphere the second difference
    /// uses the distance to the surface (Shortley–Weller), and that link is
    /// stepped implicitly so the explicit limit stays `h²/6`.
    fn new(radius: f64, cell_length: f64, spacing: f64) -> Result<Self, SaddleError> {
        if !(radius.is_finite() && radius > 0.0 && cell_length.is_finite() && spacing.is_finite()) {
            return Err(SaddleError::Invalid(
                "radius, cell length and spacing must be finite".into(),
            ));
        }
        let half = 0.5 * cell_length;
        if !(spacing > 0.0 && radius + spacing < half) {
            return Err(SaddleError::Invalid(
                "the sphere must fit inside half the cell with one spacing to spare".into(),
            ));
        }
        let m = (half / spacing).round().max(2.0) as usize;
        let h = half / m as f64;
        let side = m + 1;
        let nodes = side * side * side;
        let index = |i: usize, j: usize, k: usize| (i * side + j) * side + k;
        let coordinate = |n: usize| n as f64 * h;
        let mut absorbing = vec![false; nodes];
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    let r2 = coordinate(i).powi(2) + coordinate(j).powi(2) + coordinate(k).powi(2);
                    absorbing[index(i, j, k)] = r2 <= radius * radius;
                }
            }
        }
        let mut neighbors = vec![[0usize; 6]; nodes];
        let mut coupling = vec![[0.0; 6]; nodes];
        let mut sink = vec![0.0; nodes];
        let mut weight = vec![0.0; nodes];
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    let here = index(i, j, k);
                    let mut w = 1.0;
                    for n in [i, j, k] {
                        if n == 0 || n == m {
                            w *= 0.5;
                        }
                    }
                    weight[here] = w;
                    if absorbing[here] {
                        continue;
                    }
                    let point = [coordinate(i), coordinate(j), coordinate(k)];
                    let ijk = [i, j, k];
                    for axis in 0..3 {
                        let mut gaps = [h; 2];
                        let mut targets = [None; 2];
                        for (side_index, sign) in [-1i64, 1].into_iter().enumerate() {
                            let raw = ijk[axis] as i64 + sign;
                            let mirrored = if raw < 0 {
                                1
                            } else if raw > m as i64 {
                                m - 1
                            } else {
                                raw as usize
                            };
                            let mut other = ijk;
                            other[axis] = mirrored;
                            let target = index(other[0], other[1], other[2]);
                            if absorbing[target] {
                                let along = point[axis];
                                let across: f64 = (0..3)
                                    .filter(|&a| a != axis)
                                    .map(|a| point[a] * point[a])
                                    .sum();
                                let surface = (radius * radius - across).max(0.0).sqrt();
                                let gap = (along + sign as f64 * surface).abs();
                                gaps[side_index] = gap.clamp(1.0e-3 * h, h);
                            } else {
                                targets[side_index] = Some(target);
                            }
                        }
                        let span = gaps[0] + gaps[1];
                        for side_index in 0..2 {
                            let factor = 2.0 / (span * gaps[side_index]);
                            let slot = 2 * axis + side_index;
                            match targets[side_index] {
                                Some(target) => {
                                    neighbors[here][slot] = target;
                                    coupling[here][slot] = factor;
                                }
                                None => {
                                    neighbors[here][slot] = here;
                                    sink[here] += factor;
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(Self {
            nodes,
            side,
            spacing: h,
            half,
            absorbing,
            neighbors,
            coupling,
            sink,
            weight,
        })
    }

    fn stable_step(&self) -> f64 {
        self.spacing * self.spacing / 6.0
    }

    fn advance(&self, current: &[f64], next: &mut [f64], step: f64) {
        next.par_iter_mut().enumerate().for_each(|(node, value)| {
            if self.absorbing[node] {
                *value = 0.0;
                return;
            }
            let here = current[node];
            let mut flow = 0.0;
            for slot in 0..6 {
                flow += self.coupling[node][slot] * (current[self.neighbors[node][slot]] - here);
            }
            *value = (here + step * flow) / (1.0 + step * self.sink[node]);
        });
    }

    fn mean(&self, values: &[f64]) -> f64 {
        let total: f64 = self.weight.iter().sum();
        values
            .iter()
            .zip(&self.weight)
            .map(|(v, w)| v * w)
            .sum::<f64>()
            / total
    }

    fn sample(&self, values: &[f64], start: [f64; 3]) -> f64 {
        let length = 2.0 * self.half;
        let folded = start.map(|x| {
            let wrapped = x.rem_euclid(length);
            let reflected = if wrapped > self.half {
                length - wrapped
            } else {
                wrapped
            };
            reflected / self.spacing
        });
        let m = self.side - 1;
        let base = folded.map(|u| (u.floor() as usize).min(m - 1));
        let frac = [0, 1, 2].map(|a| folded[a] - base[a] as f64);
        let mut value = 0.0;
        for corner in 0..8 {
            let mut w = 1.0;
            let mut at = [0usize; 3];
            for a in 0..3 {
                let up = (corner >> a) & 1 == 1;
                at[a] = base[a] + usize::from(up);
                w *= if up { frac[a] } else { 1.0 - frac[a] };
            }
            value += w * values[(at[0] * self.side + at[1]) * self.side + at[2]];
        }
        value
    }
}

/// [`PeriodicSinkSurvival`] for a sphere of `radius` in a cube of `cell_length`.
///
/// `spacing` is the grid step; `reduced_times` must be non-negative and
/// increasing. Nothing here depends on the lattice or the defects.
pub fn periodic_sink_survival(
    radius: f64,
    cell_length: f64,
    spacing: f64,
    starts: &[[f64; 3]],
    reduced_times: &[f64],
) -> Result<PeriodicSinkSurvival, SaddleError> {
    if reduced_times.windows(2).any(|pair| pair[1] < pair[0])
        || reduced_times.iter().any(|t| !(t.is_finite() && *t >= 0.0))
    {
        return Err(SaddleError::Invalid(
            "reduced times must be finite, non-negative and increasing".into(),
        ));
    }
    let grid = OctantGrid::new(radius, cell_length, spacing)?;
    let mut current: Vec<f64> = grid
        .absorbing
        .iter()
        .map(|&a| if a { 0.0 } else { 1.0 })
        .collect();
    let mut next = vec![0.0; grid.nodes];
    let limit = grid.stable_step();
    let mut clock = 0.0;
    let mut at_starts = vec![Vec::with_capacity(reduced_times.len()); starts.len()];
    let mut cell_mean = Vec::with_capacity(reduced_times.len());
    for &target in reduced_times {
        while clock < target {
            let step = limit.min(target - clock);
            grid.advance(&current, &mut next, step);
            std::mem::swap(&mut current, &mut next);
            clock += step;
            if target - clock < 1.0e-12 * limit {
                clock = target;
            }
        }
        for (series, &start) in at_starts.iter_mut().zip(starts) {
            series.push(grid.sample(&current, start));
        }
        cell_mean.push(grid.mean(&current));
    }
    Ok(PeriodicSinkSurvival {
        reduced_times: reduced_times.to_vec(),
        at_starts,
        cell_mean,
    })
}

/// One trajectory: separation vector at time zero, then the first sample at
/// which the pair counted as recombined, or the end of the window.
#[derive(Debug, Clone, Copy)]
pub struct CaptureRecord {
    /// Separation vector at time zero.
    pub start: [f64; 3],
    /// First recombined sample, or the end of the window.
    pub time: f64,
    /// Whether the pair recombined inside the window.
    pub captured: bool,
}

/// Log-likelihood terms of capture records under a sphere of `radius` in a periodic cube.
///
/// A capture at `time` happened after the previous sample at
/// `time - interval`. One grid solve in reduced time serves every entry of
/// `diffusivities`. Entry `[d][i]` is the log-probability of record `i` at
/// `diffusivities[d]`; a sum over any subset of records is its log-likelihood.
pub fn periodic_capture_log_likelihood(
    radius: f64,
    cell_length: f64,
    spacing: f64,
    interval: f64,
    diffusivities: &[f64],
    records: &[CaptureRecord],
) -> Result<Vec<Vec<f64>>, SaddleError> {
    if !(interval.is_finite() && interval > 0.0)
        || diffusivities.iter().any(|d| !(d.is_finite() && *d > 0.0))
        || records
            .iter()
            .any(|r| !(r.time.is_finite() && r.time >= 0.0))
    {
        return Err(SaddleError::Invalid(
            "interval, diffusivities and times must be finite and positive".into(),
        ));
    }
    let mut starts: Vec<[f64; 3]> = Vec::new();
    let mut slot = Vec::with_capacity(records.len());
    for record in records {
        let index = starts
            .iter()
            .position(|s| *s == record.start)
            .unwrap_or_else(|| {
                starts.push(record.start);
                starts.len() - 1
            });
        slot.push(index);
    }
    let fastest = diffusivities.iter().copied().fold(0.0, f64::max);
    let slowest = diffusivities.iter().copied().fold(f64::MAX, f64::min);
    let latest = records.iter().map(|r| r.time).fold(0.0, f64::max);
    let resolution = 0.25 * slowest * interval;
    let count = ((fastest * latest / resolution).ceil() as usize).max(1);
    let grid_times: Vec<f64> = (0..=count).map(|j| j as f64 * resolution).collect();
    let survival = periodic_sink_survival(radius, cell_length, spacing, &starts, &grid_times)?;
    let at = |series: &[f64], reduced: f64| {
        let u = (reduced / resolution).clamp(0.0, count as f64);
        let low = (u.floor() as usize).min(count.saturating_sub(1));
        let frac = u - low as f64;
        series[low] * (1.0 - frac) + series[(low + 1).min(count)] * frac
    };
    Ok(diffusivities
        .iter()
        .map(|&d| {
            records
                .iter()
                .zip(&slot)
                .map(|(record, &index)| {
                    let series = &survival.at_starts[index];
                    let late = at(series, d * record.time);
                    let p = if record.captured {
                        let before = if record.time - interval <= 0.0 {
                            1.0
                        } else {
                            at(series, d * (record.time - interval))
                        };
                        before - late
                    } else {
                        late
                    };
                    p.max(1.0e-300).ln()
                })
                .collect()
        })
        .collect())
}

/// Rate constants a periodic cell measures for a sphere of `radius`.
///
/// Both are `k / D` in Å, so `k = D * value`. `decay` is `V λ` with `λ` the
/// long-time decay of the cell-mean survival in reduced time. `mean_time` is
/// `V / ∫ S ds` over a uniform start. An infinite cell gives `4 π radius` for
/// both.
#[derive(Debug, Clone, Copy)]
pub struct PeriodicSinkRate {
    /// `V λ`, from the long-time decay of the cell-mean survival.
    pub decay: f64,
    /// `V / ∫ S ds` over a uniform start.
    pub mean_time: f64,
}

/// [`PeriodicSinkRate`] of a sphere of `radius` in a periodic cube of side
/// `cell_length`, on a grid of `spacing`.
pub fn periodic_sink_rate(
    radius: f64,
    cell_length: f64,
    spacing: f64,
) -> Result<PeriodicSinkRate, SaddleError> {
    let grid = OctantGrid::new(radius, cell_length, spacing)?;
    let volume = cell_length.powi(3);
    let mut current: Vec<f64> = grid
        .absorbing
        .iter()
        .map(|&a| if a { 0.0 } else { 1.0 })
        .collect();
    let mut next = vec![0.0; grid.nodes];
    let step = grid.stable_step();
    let relax = cell_length * cell_length / (2.0 * std::f64::consts::PI * std::f64::consts::PI);
    let horizon = 6.0 * relax;
    let mut clock = 0.0;
    let mut integral = 0.0;
    let mut previous = grid.mean(&current);
    let mut checkpoint = None;
    while clock < horizon {
        grid.advance(&current, &mut next, step);
        std::mem::swap(&mut current, &mut next);
        clock += step;
        let mean = grid.mean(&current);
        integral += 0.5 * step * (previous + mean);
        previous = mean;
        if checkpoint.is_none() && clock >= 0.5 * horizon {
            checkpoint = Some((clock, mean));
        }
    }
    let (early_clock, early_mean) =
        checkpoint.ok_or_else(|| SaddleError::Solver("periodic sink horizon too short".into()))?;
    if !(previous > 0.0 && early_mean > previous) {
        return Err(SaddleError::NonFinite("periodic sink decay"));
    }
    let decay = (early_mean / previous).ln() / (clock - early_clock);
    integral += previous / decay;
    Ok(PeriodicSinkRate {
        decay: volume * decay,
        mean_time: volume / integral,
    })
}

/// Factor on one direction of a shell pair after a saddle shift.
///
/// `source_factor` is the Boltzmann factor of the away barrier measured on
/// the shell the hop leaves. An away hop uses that factor. The reverse,
/// toward hop uses the destination's factor, which is this shell's factor
/// when the pair is viewed from the other end. A tangential hop is not that
/// barrier. The two directions of one pair then carry the same factor, and
/// `n_i k_ij = n_j k_ji` survives when the geometric weights already balance.
pub fn shell_pair_factor(
    class: DirectionClass,
    source_factor: f64,
    destination_factor: f64,
) -> f64 {
    match class {
        DirectionClass::Away => source_factor,
        DirectionClass::Toward => destination_factor,
        DirectionClass::Tangential => 1.0,
    }
}

/// Minimum-image separation between `fixed` and the centroid of `mobile`.
pub fn pair_separation(
    positions: ArrayView1<f64>,
    cell: M3,
    fixed: [f64; 3],
    mobile: &[usize],
) -> Result<f64, SaddleError> {
    let center = mobile_center(positions, cell, mobile)?;
    Ok(norm3(mic(sub3(center, fixed), cell)?))
}

/// A dimer start. `mode` is the initial unstable direction, not a displacement
/// that has to be normalized by the caller.
#[derive(Clone, Debug)]
pub struct DimerSeed {
    /// Start geometry, Cartesian.
    pub positions: Array1<f64>,
    /// Initial unstable direction.
    pub mode: Array1<f64>,
}

/// An index-1 saddle and the two basins reached by pushing along its mode.
#[derive(Clone, Debug)]
pub struct BasinSaddle {
    /// Saddle geometry, Cartesian.
    pub saddle: Array1<f64>,
    /// Energy at the saddle.
    pub saddle_energy: f64,
    /// Lowest projected eigenvalue at the saddle.
    pub negative_eigenvalue: Option<f64>,
    /// Unstable mode at the saddle. The lower basin lies along one sign of this.
    pub mode: Array1<f64>,
    /// Minimum reached from `saddle + push`.
    pub plus: Array1<f64>,
    /// Minimum reached from `saddle - push`.
    pub minus: Array1<f64>,
    /// Energy of `plus`.
    pub plus_energy: f64,
    /// Energy of `minus`.
    pub minus_energy: f64,
}

/// One symmetry-distinct hop. `product` is a Cartesian geometry of the
/// same length as the reactant, before it is minimized.
#[derive(Clone, Debug)]
pub struct HopTemplate {
    /// Caller label, stored on every row.
    pub name: String,
    /// Product geometry, Cartesian.
    pub product: Array1<f64>,
    /// Start the dimer here when the caller already has a saddle guess.
    pub saddle: Option<Array1<f64>>,
    /// Initial unstable direction. Used with [`Self::saddle`].
    pub mode: Option<Array1<f64>>,
    /// Caller label for the mobile-defect orientation. Empty if it has none.
    pub orientation: String,
    /// Caller label for the product basin.
    pub final_state: String,
}

/// The relaxed pair and the hops to search.
#[derive(Clone, Debug)]
pub struct PairSetup {
    /// Relaxed reactant, Cartesian, three entries per atom.
    pub positions: Array1<f64>,
    /// Periodic cell, rows are lattice vectors.
    pub cell: M3,
    /// Fixed defect, Cartesian, in the same frame as `positions`.
    pub fixed: [f64; 3],
    /// Atoms whose centroid is the mobile defect.
    pub mobile: Vec<usize>,
    /// Caller label, stored on every row. A shell index or a distance.
    pub shell: String,
    /// Cell repeats, stored on every row. Zero if the cell is not a cube of repeats.
    pub cell_repeats: u32,
    /// Catalogue searched from this reactant.
    pub hops: Vec<HopTemplate>,
}

/// Frozen cell, or the joint chart with every lower-triangular strain free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellMode {
    /// The cell is held fixed.
    Frozen,
    /// Every lower-triangular strain is free.
    Variable,
}

/// Search, sampling, and the unit system of the rate.
///
/// `temperatures` are in the unit implied by `boltzmann`: kelvin when
/// `boltzmann` is [`crate::BOLTZMANN_EV_PER_K`]. `mass` and `angular_scale`
/// convert `sqrt(λ / mass)` into a frequency. For eV, Å and amu,
/// `angular_scale` is [`crate::angular_scale_rad_s`].
///
/// Non-exhaustive: start from [`DefectPairConfig::default`] and chain
/// the `with_*` setters.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct DefectPairConfig {
    /// Saddle search. The cell mask is set from `cell_mode`.
    pub dimer: SolidDimerConfig,
    /// Frozen or variable cell.
    pub cell_mode: CellMode,
    /// Applied load with the sign of a pressure, positive inward, as
    /// [`SolidDimerConfig::external_stress`]. Nonzero makes barriers
    /// enthalpies.
    pub external_stress: M3,
    /// Subtracted in `P = -V (σ - σ_bulk)`. Zero if the perfect cell is unstressed.
    pub bulk_stress: M3,
    /// Climbing-image band settings.
    pub band: BandConfig,
    /// Band images, including the ends. At least 3.
    pub images: usize,
    /// Most dimer steps per hop.
    pub max_dimer_steps: usize,
    /// Most band steps per phase.
    pub max_band_steps: usize,
    /// Every minimization: the reactant, the products and the pushes off
    /// a saddle. It runs to `relax.force_tol` on the largest per-atom force
    /// and reaching `relax.max_steps` is an error. The pushes decide which
    /// minima a saddle connects, so the default follows the steepest-descent
    /// flow with [`Method::Ode12r`].
    pub relax: RelaxConfig,
    /// Dimer start, this fraction of the way from reactant to product.
    pub seed_fraction: f64,
    /// Largest atomic displacement of the push off each side of the saddle.
    pub push: f64,
    /// Largest point-cloud gap that still names a minimum.
    pub connect_rmsd: f64,
    /// Separation change below which a hop is tangential.
    pub direction_tol: f64,
    /// `thermal` is replaced by `boltzmann * T` for each temperature.
    pub pafi: PafiConfig,
    /// Unconstrained forward/backward shooting. `thermal` is replaced the
    /// same way and [`Self::mass`] is installed on every coordinate.
    /// Present when the `ohtst` feature supplies the shooter.
    #[cfg(feature = "ohtst")]
    pub transmission: TransmissionConfig,
    /// Legacy one-sided shooting settings retained for callers that run
    /// [`crate::bennett_chandler`] themselves.
    pub shooting: ShootingConfig,
    /// Temperatures, in the unit of `boltzmann`.
    pub temperatures: Vec<f64>,
    /// Boltzmann constant in energy per temperature.
    pub boltzmann: f64,
    /// Atomic mass in the unit of `angular_scale`.
    pub mass: f64,
    /// Converts `sqrt(λ / mass)` to an angular frequency.
    pub angular_scale: f64,
    /// Central finite-difference step of the harmonic Hessian.
    pub harmonic_step: f64,
    /// Atoms in the harmonic Hessian. `None` is the whole cell.
    pub harmonic_atoms: Option<Vec<usize>>,
    /// Hyperplanes, including the ends. Zero uses every NEB image.
    pub planes: usize,
    /// When false, the search still reports ΔE0, the band and the
    /// zero-temperature integral, and does not sample a thermal profile.
    pub finite_temperature: bool,
    /// A converged path whose zero-temperature PAFI integral misses `dE0`
    /// by more than this does not receive a finite-temperature rate.
    pub zero_t_tol: f64,
}

with_setters!(DefectPairConfig {
    with_dimer => dimer: val SolidDimerConfig,
    with_cell_mode => cell_mode: val CellMode,
    with_external_stress => external_stress: val M3,
    with_bulk_stress => bulk_stress: val M3,
    with_band => band: val BandConfig,
    with_images => images: val usize,
    with_max_dimer_steps => max_dimer_steps: val usize,
    with_max_band_steps => max_band_steps: val usize,
    with_relax => relax: val RelaxConfig,
    with_seed_fraction => seed_fraction: val f64,
    with_push => push: val f64,
    with_connect_rmsd => connect_rmsd: val f64,
    with_direction_tol => direction_tol: val f64,
    with_pafi => pafi: val PafiConfig,
    with_shooting => shooting: val ShootingConfig,
    with_temperatures => temperatures: val Vec<f64>,
    with_boltzmann => boltzmann: val f64,
    with_mass => mass: val f64,
    with_angular_scale => angular_scale: val f64,
    with_harmonic_step => harmonic_step: val f64,
    with_harmonic_atoms => harmonic_atoms: opt Vec<usize>,
    with_planes => planes: val usize,
    with_finite_temperature => finite_temperature: val bool,
    with_zero_t_tol => zero_t_tol: val f64,
});

#[cfg(feature = "ohtst")]
impl DefectPairConfig {
    /// Set [`Self::transmission`].
    #[must_use]
    pub fn with_transmission(mut self, transmission: TransmissionConfig) -> Self {
        self.transmission = transmission;
        self
    }
}

impl Default for DefectPairConfig {
    fn default() -> Self {
        Self {
            dimer: SolidDimerConfig::frozen(crate::MinModeConfig::default())
                .with_index_check(crate::IndexCheck::first_order()),
            cell_mode: CellMode::Frozen,
            external_stress: [[0.0; 3]; 3],
            bulk_stress: [[0.0; 3]; 3],
            band: BandConfig::default()
                .with_min_mode(None)
                .with_autoneb(None)
                .with_zoom(None)
                .with_climbing(CiConfig::always()),
            images: 7,
            max_dimer_steps: 80,
            max_band_steps: 80,
            relax: RelaxConfig::default()
                .with_method(Method::Ode12r)
                .with_max_atom_move(0.08),
            seed_fraction: 0.35,
            push: 0.15,
            connect_rmsd: 0.3,
            direction_tol: 0.05,
            pafi: PafiConfig::default(),
            #[cfg(feature = "ohtst")]
            transmission: TransmissionConfig::default().with_counting(Counting::Final),
            shooting: ShootingConfig::default(),
            temperatures: vec![300.0],
            boltzmann: strain_rate::BOLTZMANN_EV_PER_K,
            mass: 1.0,
            angular_scale: strain_rate::angular_scale_rad_s(),
            harmonic_step: 1.0e-3,
            harmonic_atoms: None,
            planes: 0,
            finite_temperature: true,
            zero_t_tol: 0.02,
        }
    }
}

/// Why a hop did not receive a rate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HopOutcome {
    /// Connected index-1 saddle with a rate.
    Rated,
    /// The dimer or the band did not converge, or the zero-temperature integral missed `dE0`.
    NotConverged,
    /// The dimer stopped on a point that is not index 1.
    NotIndex1,
    /// The push does not return to the reactant and the product.
    Disconnected,
}

impl HopOutcome {
    /// `rated`, `not_converged`, `not_index1`, or `disconnected`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rated => "rated",
            Self::NotConverged => "not_converged",
            Self::NotIndex1 => "not_index1",
            Self::Disconnected => "disconnected",
        }
    }

    fn parse(text: &str) -> Result<Self, SaddleError> {
        match text {
            "rated" => Ok(Self::Rated),
            "not_converged" => Ok(Self::NotConverged),
            "not_index1" => Ok(Self::NotIndex1),
            "disconnected" => Ok(Self::Disconnected),
            _ => Err(SaddleError::Invalid("unknown hop outcome".into())),
        }
    }
}

/// One channel at one temperature.
#[derive(Clone, Debug)]
pub struct RateEntry {
    /// Caller label from [`PairSetup::shell`].
    pub shell: String,
    /// Change of the pair separation.
    pub direction: DirectionClass,
    /// Name from [`HopTemplate::name`].
    pub hop: String,
    /// Temperature of this row.
    pub temperature: f64,
    /// From [`PairSetup::cell_repeats`].
    pub cell_repeats: u32,
    /// Atoms in the cell.
    pub n_atoms: usize,
    /// Pair separation of the reactant.
    pub separation_before: f64,
    /// Pair separation of the product.
    pub separation_after: f64,
    /// Why this row does or does not carry a rate.
    pub outcome: HopOutcome,
    /// Both pushes return to the named minima.
    pub connected: bool,
    /// One negative projected eigenvalue at the saddle.
    pub index1: bool,
    /// The climbing band met its force tolerance.
    pub band_converged: bool,
    /// Minimized reactant energy.
    pub reactant_energy: f64,
    /// Minimized product energy.
    pub product_energy: f64,
    /// Energy at the dimer saddle.
    pub saddle_energy: f64,
    /// Potential, or enthalpy when an external stress is set.
    pub d_e0: f64,
    /// PAFI free-energy barrier.
    pub d_f: Option<f64>,
    /// Standard error of `d_f`.
    pub d_f_error: Option<f64>,
    /// `ΔE0 - kT ln(ν_h / ν)`.
    pub d_f_harmonic: Option<f64>,
    /// Attempt frequency along the path tangent, in the `angular_scale` unit.
    pub nu: Option<f64>,
    /// Vineyard prefactor in the same unit.
    pub vineyard: Option<f64>,
    /// Bennett–Chandler transmission at the plateau.
    pub kappa: Option<f64>,
    /// Standard error of `kappa`.
    pub kappa_error: Option<f64>,
    /// Vineyard rate `ν_h exp(-ΔE0 / kT)`.
    pub harmonic_rate: Option<f64>,
    /// `κ ν exp(-ΔF / kT)`.
    pub rate: Option<f64>,
    /// `∫ (∇E · n) ds` with the system held on the path. On a minimum-energy
    /// path this is `dE0`.
    pub zero_t: Option<f64>,
    /// Max band force, in energy per length, after the climbing phase.
    pub band_force: f64,
    /// Recorded PAFI samples, summed over planes.
    pub pafi_samples: usize,
    /// Force evaluations spent on the PAFI sampling.
    pub pafi_evaluations: usize,
    /// Cauchy stress at the saddle.
    pub saddle_stress: M3,
    /// `-V (σ_saddle - σ_bulk)` at the saddle.
    pub saddle_dipole: M3,
    /// Lowest projected eigenvalue at the saddle.
    pub negative_eigenvalue: Option<f64>,
    /// Mobile-defect orientation, as supplied on the hop.
    pub orientation: String,
    /// Product basin, as supplied on the hop.
    pub final_state: String,
}

/// The document written by [`defect_pair_rates`].
#[derive(Clone, Debug)]
pub struct RateCatalogue {
    /// One row per channel and temperature.
    pub entries: Vec<RateEntry>,
}

impl RateCatalogue {
    /// Append every entry of `other`.
    pub fn merge(&mut self, other: RateCatalogue) {
        self.entries.extend(other.entries);
    }

    /// `rgsaddle.defect_pair_rates` version 1.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"format\": \"rgsaddle.defect_pair_rates\",\n");
        out.push_str("  \"version\": 1,\n");
        out.push_str("  \"rate\": \"k = kappa * nu * exp(-dF/kT)\",\n");
        out.push_str("  \"nu\": \"(1/2pi) * sqrt((tau·H·tau)/mass) * angular_scale\",\n");
        out.push_str(
            "  \"dF\": \"PAFI potential of mean force from the reactant to the dividing surface\",\n",
        );
        out.push_str("  \"kappa\": \"Bennett-Chandler positive-flux transmission coefficient\",\n");
        out.push_str("  \"harmonic_rate\": \"Vineyard prefactor * exp(-dE0/kT)\",\n");
        out.push_str("  \"entries\": [\n");
        for (i, entry) in self.entries.iter().enumerate() {
            out.push_str("    {\n");
            push_kv(&mut out, "shell", &json_string(&entry.shell), true);
            push_kv(
                &mut out,
                "direction",
                &json_string(entry.direction.as_str()),
                true,
            );
            push_kv(&mut out, "hop", &json_string(&entry.hop), true);
            push_kv(&mut out, "temperature", &num(entry.temperature), true);
            push_kv(
                &mut out,
                "cell_repeats",
                &entry.cell_repeats.to_string(),
                true,
            );
            push_kv(&mut out, "n_atoms", &entry.n_atoms.to_string(), true);
            push_kv(
                &mut out,
                "separation_before",
                &num(entry.separation_before),
                true,
            );
            push_kv(
                &mut out,
                "separation_after",
                &num(entry.separation_after),
                true,
            );
            push_kv(
                &mut out,
                "outcome",
                &json_string(entry.outcome.as_str()),
                true,
            );
            push_kv(
                &mut out,
                "connected",
                if entry.connected { "true" } else { "false" },
                true,
            );
            push_kv(
                &mut out,
                "index1",
                if entry.index1 { "true" } else { "false" },
                true,
            );
            push_kv(
                &mut out,
                "band_converged",
                if entry.band_converged {
                    "true"
                } else {
                    "false"
                },
                true,
            );
            push_kv(
                &mut out,
                "reactant_energy",
                &num(entry.reactant_energy),
                true,
            );
            push_kv(&mut out, "product_energy", &num(entry.product_energy), true);
            push_kv(&mut out, "saddle_energy", &num(entry.saddle_energy), true);
            push_kv(&mut out, "dE0", &num(entry.d_e0), true);
            push_kv(&mut out, "dF", &opt_num(entry.d_f), true);
            push_kv(&mut out, "dF_error", &opt_num(entry.d_f_error), true);
            push_kv(&mut out, "dF_harmonic", &opt_num(entry.d_f_harmonic), true);
            push_kv(&mut out, "nu", &opt_num(entry.nu), true);
            push_kv(&mut out, "vineyard", &opt_num(entry.vineyard), true);
            push_kv(&mut out, "kappa", &opt_num(entry.kappa), true);
            push_kv(&mut out, "kappa_error", &opt_num(entry.kappa_error), true);
            push_kv(
                &mut out,
                "harmonic_rate",
                &opt_num(entry.harmonic_rate),
                true,
            );
            push_kv(&mut out, "rate_Hz", &opt_num(entry.rate), true);
            push_kv(&mut out, "zero_T", &opt_num(entry.zero_t), true);
            push_kv(&mut out, "band_force", &num(entry.band_force), true);
            push_kv(
                &mut out,
                "pafi_samples",
                &entry.pafi_samples.to_string(),
                true,
            );
            push_kv(
                &mut out,
                "pafi_evaluations",
                &entry.pafi_evaluations.to_string(),
                true,
            );
            push_kv(
                &mut out,
                "saddle_stress",
                &matrix_json(entry.saddle_stress),
                true,
            );
            push_kv(
                &mut out,
                "saddle_dipole",
                &matrix_json(entry.saddle_dipole),
                true,
            );
            push_kv(
                &mut out,
                "negative_eigenvalue",
                &opt_num(entry.negative_eigenvalue),
                true,
            );
            push_kv(
                &mut out,
                "orientation",
                &json_string(&entry.orientation),
                true,
            );
            push_kv(
                &mut out,
                "final_state",
                &json_string(&entry.final_state),
                false,
            );
            out.push_str("    }");
            if i + 1 != self.entries.len() {
                out.push(',');
            }
            out.push('\n');
        }
        out.push_str("  ]\n}\n");
        out
    }

    /// Parse a document this crate wrote.
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let obj = value.as_object().ok_or_else(|| {
            SaddleError::Invalid("defect-pair catalogue must be an object".into())
        })?;
        let kind = obj_get(obj, "format").and_then(Json::as_str).unwrap_or("");
        if kind != "rgsaddle.defect_pair_rates" {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.defect_pair_rates".into(),
            ));
        }
        let rows = obj_get(obj, "entries")
            .and_then(Json::as_array)
            .ok_or_else(|| SaddleError::Invalid("catalogue is missing entries".into()))?;
        let mut entries = Vec::with_capacity(rows.len());
        for row in rows {
            let item = row.as_object().ok_or_else(|| {
                SaddleError::Invalid("a catalogue entry must be an object".into())
            })?;
            entries.push(RateEntry {
                shell: required_str(item, "shell")?.to_string(),
                direction: DirectionClass::parse(required_str(item, "direction")?)?,
                hop: required_str(item, "hop")?.to_string(),
                temperature: required_f64(item, "temperature")?,
                cell_repeats: required_f64(item, "cell_repeats")? as u32,
                n_atoms: required_f64(item, "n_atoms")? as usize,
                separation_before: required_f64(item, "separation_before")?,
                separation_after: required_f64(item, "separation_after")?,
                outcome: HopOutcome::parse(required_str(item, "outcome")?)?,
                connected: flag(item, "connected")?,
                index1: flag(item, "index1")?,
                band_converged: flag(item, "band_converged")?,
                reactant_energy: required_f64(item, "reactant_energy")?,
                product_energy: required_f64(item, "product_energy")?,
                saddle_energy: required_f64(item, "saddle_energy")?,
                d_e0: required_f64(item, "dE0")?,
                d_f: optional_f64(item, "dF")?,
                d_f_error: optional_f64(item, "dF_error")?,
                d_f_harmonic: optional_f64(item, "dF_harmonic")?,
                nu: optional_f64(item, "nu")?,
                vineyard: optional_f64(item, "vineyard")?,
                kappa: optional_f64(item, "kappa")?,
                kappa_error: optional_f64(item, "kappa_error")?,
                harmonic_rate: optional_f64(item, "harmonic_rate")?,
                rate: optional_f64(item, "rate_Hz")?,
                zero_t: optional_f64(item, "zero_T")?,
                band_force: optional_f64(item, "band_force")?.unwrap_or(0.0),
                pafi_samples: optional_f64(item, "pafi_samples")?.unwrap_or(0.0) as usize,
                pafi_evaluations: optional_f64(item, "pafi_evaluations")?.unwrap_or(0.0) as usize,
                saddle_stress: required_matrix(item, "saddle_stress")?,
                saddle_dipole: required_matrix(item, "saddle_dipole")?,
                negative_eigenvalue: optional_f64(item, "negative_eigenvalue")?,
                orientation: optional_str(item, "orientation").unwrap_or("").to_string(),
                final_state: optional_str(item, "final_state").unwrap_or("").to_string(),
            });
        }
        Ok(Self { entries })
    }
}

/// Find every hop's saddle and its finite-temperature rates.
pub fn defect_pair_rates<S: PointSurface + Sync>(
    surface: &S,
    setup: &PairSetup,
    config: &DefectPairConfig,
) -> Result<RateCatalogue, SaddleError> {
    check_setup(setup)?;
    check_config(config)?;
    let n_atoms = setup.positions.len() / 3;
    let (reactant, reactant_energy) = relax(surface, setup.positions.clone(), setup.cell, config)?;
    let mut entries = Vec::new();
    for hop in &setup.hops {
        let (product, product_energy) = relax(surface, hop.product.clone(), setup.cell, config)?;
        let before = pair_separation(reactant.view(), setup.cell, setup.fixed, &setup.mobile)?;
        let after = pair_separation(product.view(), setup.cell, setup.fixed, &setup.mobile)?;
        let direction = classify_separation(before, after, config.direction_tol)?;
        let searched = search_hop(
            surface,
            setup,
            config,
            hop,
            &reactant,
            reactant_energy,
            &product,
            product_energy,
        )?;
        for &temperature in &config.temperatures {
            let mut entry = blank_entry(
                setup,
                hop,
                direction,
                temperature,
                n_atoms,
                before,
                after,
                reactant_energy,
                product_energy,
            );
            fill_entry(surface, config, &searched, temperature, &mut entry)?;
            entries.push(entry);
        }
    }
    Ok(RateCatalogue { entries })
}

struct Searched {
    outcome: HopOutcome,
    connected: bool,
    index1: bool,
    band_converged: bool,
    saddle: Array1<f64>,
    saddle_energy: f64,
    d_e0: f64,
    stress: M3,
    dipole: M3,
    negative: Option<f64>,
    path: Option<ArcPath>,
    reactant: Array1<f64>,
    reactant_energy: f64,
    band_force: f64,
}

#[allow(clippy::too_many_arguments)]
fn search_hop<S: PointSurface>(
    surface: &S,
    setup: &PairSetup,
    config: &DefectPairConfig,
    hop: &HopTemplate,
    reactant: &Array1<f64>,
    reactant_energy: f64,
    product: &Array1<f64>,
    _product_energy: f64,
) -> Result<Searched, SaddleError> {
    let failed = |outcome: HopOutcome| -> Searched {
        Searched {
            outcome,
            connected: false,
            index1: false,
            band_converged: false,
            saddle: reactant.clone(),
            saddle_energy: reactant_energy,
            d_e0: 0.0,
            stress: [[0.0; 3]; 3],
            dipole: [[0.0; 3]; 3],
            negative: None,
            path: None,
            reactant: reactant.clone(),
            reactant_energy,
            band_force: 0.0,
        }
    };
    let product_mode = mic_positions(product, reactant, setup.cell)?;
    if norm1(&product_mode) < 1.0e-8 && hop.mode.is_none() {
        return Ok(failed(HopOutcome::NotConverged));
    }
    let mut seeds = Vec::new();
    if let Some(hint) = &hop.saddle {
        let mode = hop.mode.clone().unwrap_or_else(|| product_mode.clone());
        seeds.push((hint.clone(), mode));
    }
    if let Ok(mixed) = mix(reactant, product, config.seed_fraction, setup.cell) {
        seeds.push((mixed, product_mode));
    }
    if seeds.is_empty() {
        return Ok(failed(HopOutcome::NotConverged));
    }
    let dimer_config = config
        .dimer
        .clone()
        .with_mask(match config.cell_mode {
            CellMode::Frozen => CellMask::frozen(),
            CellMode::Variable => CellMask::all(),
        })
        .with_external_stress(config.external_stress);
    let mut found_saddle = None;
    let mut last_negative = None;
    let mut saw_index = false;
    let mut disconnected = false;
    for (seed, mode) in seeds {
        if norm1(&mode) < 1.0e-8 {
            continue;
        }
        let mut dimer = SolidDimer::new(dimer_config.clone(), seed, setup.cell, mode)?;
        let report = dimer.run(surface, config.max_dimer_steps)?;
        if report.status != MinModeStatus::Converged {
            continue;
        }
        let index1 = !report.rejected && report.negative_modes == Some(1);
        if !index1 {
            saw_index = true;
            last_negative = report.eigenvalues.first().copied();
            continue;
        }
        let saddle = dimer.positions().to_owned();
        let saddle_cell = dimer.cell();
        let push = cartesian_push(&dimer, config.push)?;
        let (plus, _) = relax(surface, &saddle + &push, saddle_cell, config)?;
        let (minus, _) = relax(surface, &saddle - &push, saddle_cell, config)?;
        if links(
            &plus,
            &minus,
            reactant,
            product,
            setup.cell,
            config.connect_rmsd,
        )? {
            found_saddle = Some((dimer, saddle, saddle_cell, report));
            break;
        }
        disconnected = true;
    }
    let Some((dimer, saddle, _found_cell, report)) = found_saddle else {
        if disconnected {
            return Ok(failed(HopOutcome::Disconnected));
        }
        if saw_index {
            let mut row = failed(HopOutcome::NotIndex1);
            row.negative = last_negative;
            return Ok(row);
        }
        return Ok(failed(HopOutcome::NotConverged));
    };
    let saddle_cell = dimer.cell();
    let push = cartesian_push(&dimer, config.push)?;
    let (plus, _) = relax(surface, &saddle + &push, saddle_cell, config)?;
    let (minus, _) = relax(surface, &saddle - &push, saddle_cell, config)?;
    let connected = links(
        &plus,
        &minus,
        reactant,
        product,
        setup.cell,
        config.connect_rmsd,
    )?;
    let (saddle_energy, _) = surface.eval_in_cell(saddle.view(), &m3_to_row9(saddle_cell))?;
    let d_e0 = enthalpy(
        saddle_energy,
        saddle_cell,
        setup.cell,
        config.external_stress,
    )? - enthalpy(
        reactant_energy,
        setup.cell,
        setup.cell,
        config.external_stress,
    )?;
    let stress = cauchy(surface, saddle.view(), saddle_cell)?;
    let volume = m3_det(saddle_cell).abs();
    let dipole = ElasticDipole::from_stress(volume, stress, config.bulk_stress)?.tensor();
    if !connected {
        return Ok(Searched {
            outcome: HopOutcome::Disconnected,
            connected: false,
            index1: true,
            band_converged: false,
            saddle,
            saddle_energy,
            d_e0,
            stress,
            dipole,
            negative: report.eigenvalues.first().copied(),
            path: None,
            reactant: reactant.clone(),
            reactant_energy,
            band_force: 0.0,
        });
    }
    let (path, band_converged, band_force) = climb_path(
        surface,
        config,
        reactant,
        &saddle,
        product,
        setup.cell,
        saddle_cell,
    )?;
    Ok(Searched {
        outcome: HopOutcome::Rated,
        connected: true,
        index1: true,
        band_converged,
        saddle,
        saddle_energy,
        d_e0,
        stress,
        dipole,
        negative: report.eigenvalues.first().copied(),
        path: Some(path),
        reactant: reactant.clone(),
        reactant_energy,
        band_force,
    })
}

fn fill_entry<S: PointSurface + Sync>(
    surface: &S,
    config: &DefectPairConfig,
    searched: &Searched,
    temperature: f64,
    entry: &mut RateEntry,
) -> Result<(), SaddleError> {
    entry.outcome = searched.outcome;
    entry.connected = searched.connected;
    entry.index1 = searched.index1;
    entry.band_converged = searched.band_converged;
    entry.band_force = searched.band_force;
    entry.saddle_energy = searched.saddle_energy;
    entry.d_e0 = searched.d_e0;
    entry.saddle_stress = searched.stress;
    entry.saddle_dipole = searched.dipole;
    entry.negative_eigenvalue = searched.negative;
    entry.reactant_energy = searched.reactant_energy;
    let Some(path) = searched.path.as_ref() else {
        return Ok(());
    };
    if !searched.band_converged {
        entry.outcome = HopOutcome::NotConverged;
        return Ok(());
    }
    let thermal = config.boltzmann * temperature;
    if !(thermal.is_finite() && thermal > 0.0) {
        return Err(SaddleError::Invalid(
            "kT must be finite and positive".into(),
        ));
    }
    let harmonic = harmonic_rates(
        surface,
        config,
        searched.reactant.view(),
        searched.saddle.view(),
        path,
        searched.d_e0,
        thermal,
    )?;
    entry.nu = harmonic.nu;
    entry.vineyard = harmonic.vineyard;
    entry.d_f_harmonic = harmonic.d_f_harmonic;
    entry.harmonic_rate = harmonic.rate;
    let arcs = plane_arcs(path, config.planes);
    let zero = pafi::zero_temperature_barrier(surface, path, &arcs)?;
    entry.zero_t = Some(zero.barrier);
    if (zero.barrier - searched.d_e0).abs() > config.zero_t_tol {
        entry.outcome = HopOutcome::NotConverged;
        entry.d_f = Some(zero.barrier);
        entry.d_f_error = Some((zero.barrier - searched.d_e0).abs());
        return Ok(());
    }
    if !config.finite_temperature {
        return Ok(());
    }
    let pafi_config = config.pafi.clone().with_thermal(thermal);
    let (profile, planes) = pafi::pafi_profile(surface, path, &arcs, &pafi_config)?;
    entry.d_f = Some(profile.barrier);
    entry.d_f_error = Some(profile.barrier_error);
    entry.pafi_samples = profile.samples;
    entry.pafi_evaluations = planes.iter().map(|plane| plane.evaluations).sum();
    let (kappa, kappa_error) = shoot(surface, config, path, &profile, &planes, thermal)?;
    entry.kappa = Some(kappa);
    entry.kappa_error = Some(kappa_error);
    if let Some(nu) = harmonic.nu
        && (0.0..=1.0).contains(&kappa)
    {
        entry.rate = Some(arrhenius(kappa * nu, profile.barrier, thermal));
    }
    Ok(())
}

struct Harmonic {
    nu: Option<f64>,
    vineyard: Option<f64>,
    d_f_harmonic: Option<f64>,
    rate: Option<f64>,
}

fn harmonic_rates<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    reactant: ArrayView1<f64>,
    saddle: ArrayView1<f64>,
    path: &ArcPath,
    d_e0: f64,
    thermal: f64,
) -> Result<Harmonic, SaddleError> {
    let atoms = config.harmonic_atoms.as_deref();
    let (min_h, min_vals, min_vecs) = hessian(surface, reactant, atoms, config.harmonic_step)?;
    let (_sad_h, sad_vals, sad_vecs) = hessian(surface, saddle, atoms, config.harmonic_step)?;
    let (min_positive, sad_positive) = if atoms.is_none() {
        let translations = surface.excluded_modes(reactant)?;
        let min_split = split_translational_modes(&min_vals, min_vecs.view(), translations.view())?;
        let sad_translations = surface.excluded_modes(saddle)?;
        let sad_split =
            split_translational_modes(&sad_vals, sad_vecs.view(), sad_translations.view())?;
        (min_split.positive, sad_split.positive)
    } else {
        (positives(&min_vals), positives(&sad_vals))
    };
    let geometry = path.geometry(path.bounds().0)?;
    let tangent = unit(&geometry.tangent)?;
    let lambda = curvature_along(&min_h, tangent.view(), atoms, reactant.len());
    let nu = if lambda > 0.0 {
        Some(attempt_frequency(
            lambda,
            config.mass,
            config.angular_scale,
        )?)
    } else {
        None
    };
    let vineyard = if !min_positive.is_empty()
        && !sad_positive.is_empty()
        && min_positive.len() > sad_positive.len()
    {
        vineyard_frequency(
            &min_positive,
            &sad_positive,
            config.mass,
            config.angular_scale,
        )
        .ok()
    } else {
        None
    };
    let d_f_harmonic = match (nu, vineyard) {
        (Some(nu), Some(vineyard)) if vineyard > 0.0 && nu > 0.0 => {
            Some(d_e0 - thermal * (vineyard / nu).ln())
        }
        _ => None,
    };
    let rate = vineyard.map(|prefactor| arrhenius(prefactor, d_e0, thermal));
    Ok(Harmonic {
        nu,
        vineyard,
        d_f_harmonic,
        rate,
    })
}

#[cfg(feature = "ohtst")]
fn shoot<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    _path: &pafi::ArcPath,
    profile: &pafi::FreeEnergyProfile,
    planes: &[pafi::HyperplaneAverage],
    thermal: f64,
) -> Result<(f64, f64), SaddleError> {
    let shooting = config
        .transmission
        .clone()
        .with_thermal(thermal)
        .with_masses(vec![config.mass; planes[profile.dividing].plane.dim()]);
    let coefficient = pafi::pafi_transmission(surface, profile, planes, &shooting)?;
    Ok((coefficient.kappa, coefficient.error))
}

#[cfg(not(feature = "ohtst"))]
fn shoot<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    path: &pafi::ArcPath,
    profile: &pafi::FreeEnergyProfile,
    planes: &[pafi::HyperplaneAverage],
    thermal: f64,
) -> Result<(f64, f64), SaddleError> {
    let dividing = profile.dividing;
    let geometry = path.geometry(profile.arc[dividing])?;
    let starts = if planes[dividing].retained.is_empty() {
        vec![geometry.positions.clone()]
    } else {
        planes[dividing].retained.clone()
    };
    let shooting = config
        .shooting
        .clone()
        .with_thermal(thermal)
        .with_mass(config.mass);
    // The product side is the side of increasing arc length.
    let normal = product_tangent(path, geometry.arc)?;
    let transmission = crate::bennett::bennett_chandler(
        surface,
        &starts,
        geometry.positions.view(),
        normal.view(),
        &shooting,
    )?;
    Ok((transmission.kappa, transmission.error))
}

fn climb_path<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    product: &Array1<f64>,
    cell: M3,
    saddle_cell: M3,
) -> Result<(ArcPath, bool, f64), SaddleError> {
    let images = config.images.max(3);
    let mid = images / 2;
    let idpp = crate::idpp::IdppConfig::default()
        .with_max_iterations(2000)
        .with_grad_tol(1.0e-3)
        .with_max_move(0.2)
        .with_memory(12);
    let wrapped = crate::mic::Cell::from_ase(cell)
        .map_err(|error| SaddleError::Invalid(format!("IDPP cell: {error}")))?;
    let left = crate::idpp::idpp_path(
        reactant.view(),
        saddle.view(),
        mid + 1,
        Some(&wrapped),
        &idpp,
    )?;
    let right = crate::idpp::idpp_path(
        saddle.view(),
        product.view(),
        images - mid,
        Some(&wrapped),
        &idpp,
    )?;
    let mut band = Array2::zeros((images, reactant.len()));
    for i in 0..=mid {
        band.row_mut(i).assign(&left.row(i));
    }
    for i in 1..(images - mid) {
        band.row_mut(mid + i).assign(&right.row(i));
    }
    let _ = saddle_cell;
    // L-BFGS with climbing armed on the first step follows the highest
    // image. An IDPP guess can put that image on a collision, several eV
    // above the dimer saddle, and the two-loop step then runs up the
    // wall. The guards in the band reset a capped step; they do not make
    // that image the saddle. Relax with climbing off, then climb. If the
    // quasi-Newton climb still runs away, repeat it with FIRE, or with
    // ODE12r when the caller was already on FIRE.
    let plain = bare_band(&config.band).with_climbing(None);
    let start_cells = vec![cell; images];
    let (mut relaxed, mut relaxed_cells, mut relaxed_force) = run_band(
        surface,
        config,
        plain,
        band.clone(),
        cell,
        Some(start_cells.clone()),
        false,
    )?;
    if relaxed_force > config.band.force_tol * 5.0
        && !matches!(config.band.method, Method::Fire { .. })
    {
        let fire = bare_band(&config.band)
            .with_climbing(None)
            .with_method(Method::fire(FireKind::V2))
            .with_time_step(config.band.time_step.min(0.1));
        let (pos, cells, force) =
            run_band(surface, config, fire, band, cell, Some(start_cells), false)?;
        if force < relaxed_force {
            relaxed = pos;
            relaxed_cells = cells;
            relaxed_force = force;
        }
    }
    let climbing = bare_band(&config.band).with_climbing(CiConfig::always());
    let (climbed, _, climbed_force) = run_band(
        surface,
        config,
        climbing.clone(),
        relaxed.clone(),
        cell,
        Some(relaxed_cells.clone()),
        true,
    )?;
    let (cartesian, force) =
        if climbed_force <= config.band.force_tol || climbed_force <= relaxed_force {
            (climbed, climbed_force)
        } else {
            let retry = climbing
                .with_method(if matches!(config.band.method, Method::Fire { .. }) {
                    Method::Ode12r
                } else {
                    Method::fire(FireKind::V2)
                })
                .with_time_step(config.band.time_step.min(0.1));
            let (again, _, again_force) = run_band(
                surface,
                config,
                retry,
                relaxed,
                cell,
                Some(relaxed_cells),
                true,
            )?;
            if again_force < climbed_force {
                (again, again_force)
            } else {
                (climbed, climbed_force)
            }
        };
    let (saddle_energy, _) = surface.eval(saddle.view())?;
    let mut highest = f64::NEG_INFINITY;
    for row in cartesian.rows() {
        let (energy, _) = surface.eval(row)?;
        highest = highest.max(energy);
    }
    let mut frames = Vec::with_capacity(cartesian.nrows());
    for row in cartesian.rows() {
        frames.push(row.to_owned());
    }
    let tol = config.band.force_tol;
    let converged = force <= tol && highest <= saddle_energy + config.zero_t_tol.max(0.05);
    Ok((ArcPath::from_positions(&frames)?, converged, force))
}

fn bare_band(config: &BandConfig) -> BandConfig {
    config
        .clone()
        .with_min_mode(None)
        .with_autoneb(None)
        .with_zoom(None)
        .with_solid(None)
        .with_exp_precon(None)
        .with_cell(None)
}

fn run_band<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    band_config: BandConfig,
    positions: Array2<f64>,
    reference: M3,
    cells: Option<Vec<M3>>,
    abort_on_spike: bool,
) -> Result<(Array2<f64>, Vec<M3>, f64), SaddleError> {
    let tol = band_config.force_tol;
    let n_images = positions.nrows();
    let image_cells = cells.unwrap_or_else(|| vec![reference; n_images]);
    let mut neb = match config.cell_mode {
        CellMode::Frozen => SolidNeb::frozen(band_config, positions)?,
        CellMode::Variable => SolidNeb::joint(
            band_config,
            reference,
            positions,
            &image_cells,
            CellMask::all(),
            config.external_stress,
            1.0,
        )?,
    };
    let mut steps = 0usize;
    let mut report = neb.step(surface)?;
    steps += 1;
    let mut best_force = report.max_force;
    let mut best = neb.cartesian()?;
    let mut best_cells = neb.cells()?.unwrap_or_default();
    while report.status == BandStatus::Running && steps < config.max_band_steps {
        report = neb.step(surface)?;
        steps += 1;
        if report.max_force < best_force {
            best_force = report.max_force;
            best = neb.cartesian()?;
            best_cells = neb.cells()?.unwrap_or_default();
        } else if abort_on_spike
            && best_force < tol.max(0.05)
            && report.max_force > best_force * 8.0
        {
            break;
        }
    }
    let (cartesian, found_cells, force) = if report.max_force <= best_force {
        (
            neb.cartesian()?,
            neb.cells()?.unwrap_or_default(),
            report.max_force,
        )
    } else {
        (best, best_cells, best_force)
    };
    Ok((cartesian, found_cells, force))
}

/// Run every seed. Index-1 saddles are returned with both pushed basins.
/// A seed that does not converge, or that is not index 1, is omitted.
pub fn seeded_saddles<S: PointSurface + Sync>(
    surface: &S,
    config: &DefectPairConfig,
    cell: M3,
    seeds: &[DimerSeed],
) -> Vec<BasinSaddle> {
    seeds
        .par_iter()
        .filter_map(|seed| one_seed(surface, config, cell, seed).ok())
        .collect()
}

fn one_seed<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    cell: M3,
    seed: &DimerSeed,
) -> Result<BasinSaddle, SaddleError> {
    if seed.mode.dot(&seed.mode) < 1.0e-16 {
        return Err(SaddleError::Invalid("dimer seed mode vanishes".into()));
    }
    let mut dimer = SolidDimer::new(
        config.dimer.clone(),
        seed.positions.clone(),
        cell,
        seed.mode.clone(),
    )?;
    let report = dimer.run(surface, config.max_dimer_steps)?;
    if report.status != MinModeStatus::Converged
        || report.rejected
        || report.negative_modes != Some(1)
    {
        return Err(SaddleError::Invalid("seed is not an index-1 saddle".into()));
    }
    let saddle = dimer.positions().to_owned();
    let mode = dimer.mode().to_owned();
    let push = cartesian_push(&dimer, config.push)?;
    let (plus, plus_energy) = relax(surface, &saddle + &push, cell, config)?;
    let (minus, minus_energy) = relax(surface, &saddle - &push, cell, config)?;
    let (saddle_energy, _) = surface.eval(saddle.view())?;
    Ok(BasinSaddle {
        saddle,
        saddle_energy,
        negative_eigenvalue: report.eigenvalues.first().copied(),
        mode,
        plus,
        minus,
        plus_energy,
        minus_energy,
    })
}

/// Follow `config.relax` from `positions` until the energy is at or below
/// `floor` or the largest per-atom force is below `config.relax.force_tol`.
///
/// The floor ends a descent from a saddle well before a deep minimum, so a
/// band can end there instead of across the whole drop. Reaching
/// `config.relax.max_steps` first, or a non-finite energy, is an error.
pub fn descend_to_floor<S: PointSurface>(
    surface: &S,
    mut positions: Array1<f64>,
    cell: M3,
    config: &DefectPairConfig,
    floor: f64,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let relax = &config.relax;
    relax.check()?;
    let flat = m3_to_row9(cell);
    let (mut energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("descent start energy"));
    }
    if energy <= floor || max_atom_force(gradient.view()) < relax.force_tol {
        return Ok((positions, energy));
    }
    let mut solver = rgmin::Solver::new(
        relax.method.clone(),
        rgmin::Control::default()
            .with_maxiter(usize::MAX)
            .with_gtol(0.0)
            .with_istep(relax.initial_step),
        positions.len(),
    );
    solver.set_atom_maxmove(relax.max_atom_move);
    let oracle = rgmin::Oracle::unbounded(positions.len(), |x: ArrayView1<f64>| {
        surface
            .eval_in_cell(x, &flat)
            .unwrap_or_else(|_| (f64::NAN, Array1::from_elem(x.len(), f64::NAN)))
    });
    let mut largest = f64::INFINITY;
    for _ in 0..relax.max_steps {
        solver
            .step(&oracle, &mut positions)
            .map_err(|error| SaddleError::Solver(error.to_string()))?;
        let (next, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
        if !next.is_finite() {
            return Err(SaddleError::NonFinite("descent energy"));
        }
        energy = next;
        largest = max_atom_force(gradient.view());
        if energy <= floor || largest < relax.force_tol {
            return Ok((positions, energy));
        }
    }
    Err(SaddleError::Solver(format!(
        "descent not converged: energy {energy:.6} above the floor {floor:.6} and largest atom force {largest:.3e} not below {:.1e} after {} steps",
        relax.force_tol, relax.max_steps
    )))
}

/// How a placement enters the recombination volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteOutcome {
    /// The placement recombines on relaxation, or its re-referenced capture
    /// barrier is at most [`SPONTANEOUS_BARRIER_EV`]. It is a spontaneous
    /// site: capture probability 1, and it is not a missing rate.
    Spontaneous,
    /// Capture competes with the other hops in [`VolumeSite::rates`].
    Kinetic,
}

impl SiteOutcome {
    /// `spontaneous` or `kinetic`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Spontaneous => "spontaneous",
            Self::Kinetic => "kinetic",
        }
    }
}

/// A re-referenced barrier at or below this, in eV, is spontaneous capture.
pub const SPONTANEOUS_BARRIER_EV: f64 = 0.001;

/// True when the placement is not a bound minimum, or the barrier measured
/// from the lowest bound state is at most [`SPONTANEOUS_BARRIER_EV`].
pub fn capture_is_spontaneous(recombines_on_relaxation: bool, barrier_ev: Option<f64>) -> bool {
    capture_is_spontaneous_within(recombines_on_relaxation, barrier_ev, SPONTANEOUS_BARRIER_EV)
}

/// Same test with an explicit threshold, in eV, for the sensitivity of V_iv.
pub fn capture_is_spontaneous_within(
    recombines_on_relaxation: bool,
    barrier_ev: Option<f64>,
    threshold_ev: f64,
) -> bool {
    recombines_on_relaxation
        || barrier_ev.is_some_and(|value| value.is_finite() && value <= threshold_ev)
}

/// One shell in a kinetic recombination-volume sum.
///
/// `rates` are `(destination, rate)`. A destination equal to another site's
/// `id` is a hop to that shell. Any other destination is absorbing.
/// `"recombined"` is the capture absorber, and so is the id of a
/// [`SiteOutcome::Spontaneous`] shell. Every other unknown name is escape.
/// [`SiteOutcome::Spontaneous`] is captured with probability 1 even when
/// `rates` is empty. Those sites are counted, not dropped.
#[derive(Clone, Debug)]
pub struct VolumeSite {
    /// Shell name. Other sites hop to it by this name.
    pub id: String,
    /// Multiplicity of the shell.
    pub count: f64,
    /// Spontaneous or kinetic.
    pub outcome: SiteOutcome,
    /// Outgoing `(destination, rate)` pairs.
    pub rates: Vec<(String, f64)>,
}

/// One placement's share of the recombination volume.
#[derive(Clone, Debug)]
pub struct VolumePlacement {
    /// Shell name from [`VolumeSite::id`].
    pub id: String,
    /// Multiplicity of the shell.
    pub count: f64,
    /// Spontaneous or kinetic.
    pub outcome: SiteOutcome,
    /// Probability of recombination before escape.
    pub probability: f64,
    /// `count * probability`.
    pub sites: f64,
}

/// Kinetic recombination volume, in the same units as `atomic_volume`.
///
/// The transient shells form an absorbing Markov chain. `kinetic_sites` is
/// the multiplicity-weighted probability of reaching recombination before
/// escape. `spontaneous_sites` is the multiplicity of shells that are not
/// minima. `total_sites` is their sum, and `volume` is that sum times
/// `atomic_volume`.
#[derive(Clone, Debug)]
pub struct RecombinationVolume {
    /// `total_sites * atomic_volume`.
    pub volume: f64,
    /// Multiplicity of the shells that are not minima.
    pub spontaneous_sites: f64,
    /// Multiplicity-weighted capture probability of the bound shells.
    pub kinetic_sites: f64,
    /// `spontaneous_sites + kinetic_sites`.
    pub total_sites: f64,
    /// Every input placement, spontaneous ones included.
    pub placements: Vec<VolumePlacement>,
}

/// Absorbing-chain recombination volume.
pub fn kinetic_recombination_volume(
    sites: &[VolumeSite],
    atomic_volume: f64,
) -> Result<RecombinationVolume, SaddleError> {
    if !(atomic_volume.is_finite() && atomic_volume > 0.0) {
        return Err(SaddleError::Invalid(
            "atomic volume must be finite and positive".into(),
        ));
    }
    let mut seen = std::collections::HashSet::new();
    for site in sites {
        if !seen.insert(site.id.as_str()) {
            return Err(SaddleError::Invalid(format!(
                "site id {} is repeated",
                site.id
            )));
        }
    }
    let mut spontaneous = 0.0;
    let mut placements = Vec::with_capacity(sites.len());
    let mut transient: Vec<&VolumeSite> = Vec::new();
    let mut spontaneous_ids = std::collections::HashSet::new();
    for site in sites {
        if !(site.count.is_finite() && site.count >= 0.0) {
            return Err(SaddleError::Invalid(
                "site multiplicity must be finite and non-negative".into(),
            ));
        }
        if site.outcome == SiteOutcome::Spontaneous {
            spontaneous += site.count;
            spontaneous_ids.insert(site.id.as_str());
            placements.push(VolumePlacement {
                id: site.id.clone(),
                count: site.count,
                outcome: SiteOutcome::Spontaneous,
                probability: 1.0,
                sites: site.count,
            });
        } else {
            transient.push(site);
        }
    }
    let n = transient.len();
    let mut absorption = vec![0.0; n];
    if n > 0 {
        let index: std::collections::HashMap<&str, usize> = transient
            .iter()
            .enumerate()
            .map(|(i, site)| (site.id.as_str(), i))
            .collect();
        let mut matrix = vec![vec![0.0; n]; n];
        let mut rhs = vec![0.0; n];
        for (i, site) in transient.iter().enumerate() {
            let mut total = 0.0;
            for (_, rate) in &site.rates {
                if !(rate.is_finite() && *rate >= 0.0) {
                    return Err(SaddleError::Invalid(
                        "a hop rate must be finite and non-negative".into(),
                    ));
                }
                total += *rate;
            }
            if total <= 0.0 {
                return Err(SaddleError::Invalid(format!(
                    "site {} has no outgoing rate",
                    site.id
                )));
            }
            matrix[i][i] = 1.0;
            for (destination, rate) in &site.rates {
                let probability = *rate / total;
                // A spontaneous shell is already captured. Naming it is the
                // same absorber as "recombined"; it is not an escape.
                if destination == "recombined" || spontaneous_ids.contains(destination.as_str()) {
                    rhs[i] += probability;
                } else if let Some(&j) = index.get(destination.as_str()) {
                    matrix[i][j] -= probability;
                }
            }
        }
        absorption = solve_linear(matrix, rhs)?;
    }
    let mut kinetic = 0.0;
    for (site, probability) in transient.iter().zip(&absorption) {
        if !(probability.is_finite() && *probability >= -1.0e-8 && *probability <= 1.0 + 1.0e-8) {
            return Err(SaddleError::Invalid(format!(
                "capture probability for {} is not in [0, 1]",
                site.id
            )));
        }
        let probability = probability.clamp(0.0, 1.0);
        let sites = site.count * probability;
        kinetic += sites;
        placements.push(VolumePlacement {
            id: site.id.clone(),
            count: site.count,
            outcome: SiteOutcome::Kinetic,
            probability,
            sites,
        });
    }
    let total = spontaneous + kinetic;
    Ok(RecombinationVolume {
        volume: total * atomic_volume,
        spontaneous_sites: spontaneous,
        kinetic_sites: kinetic,
        total_sites: total,
        placements,
    })
}

/// The free crystal beyond the last discrete shell.
///
/// [`Self::boundary`] hops outward at [`Self::outward_rate`] into a return
/// layer of [`Self::return_count`] sites. A fraction
/// [`Self::return_probability`] of the trajectories that reach that layer
/// come back to the boundary; the rest escape to infinity. The outward hop
/// is then a first-passage step, not an absorbing loss.
#[derive(Clone, Debug)]
pub struct FreeMigration {
    /// Id of the outermost discrete shell.
    pub boundary: String,
    /// Total rate from that shell into the return layer.
    pub outward_rate: f64,
    /// Fraction that comes back from the return layer to the boundary.
    pub return_probability: f64,
    /// Sites in the return layer.
    pub return_count: f64,
}

/// Id of the return layer inserted by [`embed_free_migration`].
pub const FREE_CRYSTAL: &str = "free-crystal";

/// Continuum probability of hitting a sphere of radius `radius` before
/// infinity, starting one jump farther out.
///
/// For three-dimensional diffusion the potential is `1/r`, so the
/// probability of reaching `radius` from `radius + jump` is
/// `radius / (radius + jump)`.
pub fn spherical_return_probability(radius: f64, jump: f64) -> Result<f64, SaddleError> {
    if !(jump.is_finite() && jump > 0.0) {
        return Err(SaddleError::Invalid(
            "the jump length must be finite and positive".into(),
        ));
    }
    sphere_hit_probability(radius, radius + jump)
}

/// Probability of hitting a sphere of `radius` before infinity, starting at `distance`.
///
/// Inside or on the sphere the probability is 1. Outside it is `radius / distance`.
pub fn sphere_hit_probability(radius: f64, distance: f64) -> Result<f64, SaddleError> {
    if !(radius.is_finite() && radius > 0.0 && distance.is_finite() && distance > 0.0) {
        return Err(SaddleError::Invalid(
            "the sphere radius and the starting distance must be finite and positive".into(),
        ));
    }
    if distance <= radius {
        Ok(1.0)
    } else {
        Ok(radius / distance)
    }
}

/// One hop that leaves the discrete region and lands at `distance` from the origin.
///
/// `source` is the shell the hop leaves. `boundary` is the shell a trajectory
/// re-enters when it hits the sphere again. `rate` is the hop rate.
#[derive(Clone, Debug)]
pub struct SphereLanding {
    /// Shell the hop leaves.
    pub source: String,
    /// Shell re-entered when the sphere is hit again.
    pub boundary: String,
    /// Hop rate.
    pub rate: f64,
    /// Landing distance from the origin.
    pub distance: f64,
}

/// Send every outward hop into one return layer.
///
/// An `"escaped"` edge is not a loss. Each landing in `landings` is a hop
/// into the layer, and the layer comes back to that landing's boundary with
/// probability [`sphere_hit_probability`] or else escapes to infinity. Hops
/// from shells inside the cutoff are the same kind of hop: absorbing them
/// makes the effective radius fall as the cutoff grows.
pub fn embed_sphere_return(
    sites: &[VolumeSite],
    radius: f64,
    landings: &[SphereLanding],
) -> Result<Vec<VolumeSite>, SaddleError> {
    if !(radius.is_finite() && radius > 0.0) {
        return Err(SaddleError::Invalid(
            "the sphere radius must be finite and positive".into(),
        ));
    }
    if landings.is_empty() {
        return Err(SaddleError::Invalid(
            "sphere return needs at least one outward landing".into(),
        ));
    }
    let mut out: Vec<VolumeSite> = sites
        .iter()
        .map(|site| {
            let mut site = site.clone();
            site.rates
                .retain(|(destination, _)| destination != "escaped");
            site
        })
        .collect();
    if out.iter().any(|site| site.id == FREE_CRYSTAL) {
        return Err(SaddleError::Invalid(
            "the shell list already has a return layer".into(),
        ));
    }
    let mut back: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    let mut leave = 0.0;
    for landing in landings {
        if !(landing.rate.is_finite() && landing.rate > 0.0) {
            return Err(SaddleError::Invalid(
                "an outward landing rate must be finite and positive".into(),
            ));
        }
        if !out.iter().any(|site| site.id == landing.source) {
            return Err(SaddleError::Invalid(format!(
                "landing source {} is not in the shell list",
                landing.source
            )));
        }
        if !out.iter().any(|site| site.id == landing.boundary) {
            return Err(SaddleError::Invalid(format!(
                "landing boundary {} is not in the shell list",
                landing.boundary
            )));
        }
        let source = out
            .iter_mut()
            .find(|site| site.id == landing.source)
            .expect("source was checked");
        if source.outcome != SiteOutcome::Kinetic {
            return Err(SaddleError::Invalid(format!(
                "landing source {} is not a kinetic shell",
                landing.source
            )));
        }
        if let Some(edge) = source
            .rates
            .iter_mut()
            .find(|(destination, _)| destination == FREE_CRYSTAL)
        {
            edge.1 += landing.rate;
        } else {
            source.rates.push((FREE_CRYSTAL.to_string(), landing.rate));
        }
        let hit = sphere_hit_probability(radius, landing.distance)?;
        *back.entry(landing.boundary.clone()).or_insert(0.0) += landing.rate * hit;
        leave += landing.rate * (1.0 - hit);
    }
    let mut rates: Vec<(String, f64)> = back.into_iter().filter(|(_, rate)| *rate > 0.0).collect();
    if leave > 0.0 {
        rates.push(("escaped".to_string(), leave));
    }
    if rates.is_empty() {
        return Err(SaddleError::Invalid(
            "the return layer has no outgoing rate".into(),
        ));
    }
    out.push(VolumeSite {
        id: FREE_CRYSTAL.to_string(),
        count: 1.0,
        outcome: SiteOutcome::Kinetic,
        rates,
    });
    Ok(out)
}

/// One hop from a site of the discrete region to `distance` from the origin,
/// outside the region.
#[derive(Clone, Debug, PartialEq)]
pub struct OuterHop {
    /// Site the hop leaves.
    pub source: String,
    /// Hop rate.
    pub rate: f64,
    /// Landing distance from the origin.
    pub distance: f64,
}

/// A site chain matched to the infinite crystal outside it.
#[derive(Clone, Debug)]
pub struct MatchedCapture {
    /// Capture radius `R` with `P(r) = R / r` outside the region.
    pub radius: f64,
    /// The chain solved with every outer hop captured with probability
    /// `radius / distance`.
    pub volume: RecombinationVolume,
}

/// Capture radius of a site chain whose outside is the field-free infinite
/// crystal.
///
/// Outside the region the capture probability is `R / r`, so an outer hop
/// to `distance` is captured with probability `R / distance` and escapes
/// otherwise. `R` is fixed by asking the multiplicity-weighted mean of
/// `P r` over the `outer` sites (`(id, radius)`) to equal `R`. Capture
/// probabilities are affine in `R`, so two solves give `R` in closed form
/// and a third returns the matched chain.
///
/// [`embed_sphere_return`] returns a hop that lands just past the sphere
/// with probability close to 1, which holds the outer sites high; matching
/// does not need the landing to be far from the region.
pub fn matched_capture_radius(
    sites: &[VolumeSite],
    hops: &[OuterHop],
    outer: &[(String, f64)],
    atomic_volume: f64,
) -> Result<MatchedCapture, SaddleError> {
    if hops.is_empty() || outer.is_empty() {
        return Err(SaddleError::Invalid(
            "matching needs outer hops and outer sites".into(),
        ));
    }
    let nearest = hops
        .iter()
        .map(|hop| hop.distance)
        .fold(f64::INFINITY, f64::min);
    if !(nearest.is_finite() && nearest > 0.0) {
        return Err(SaddleError::Invalid(
            "outer hop distances must be finite and positive".into(),
        ));
    }
    let with_radius = |radius: f64| -> Result<RecombinationVolume, SaddleError> {
        let mut chain: Vec<VolumeSite> = sites.to_vec();
        for hop in hops {
            if !(hop.rate.is_finite() && hop.rate > 0.0) {
                return Err(SaddleError::Invalid(
                    "an outer hop rate must be finite and positive".into(),
                ));
            }
            let site = chain
                .iter_mut()
                .find(|site| site.id == hop.source)
                .ok_or_else(|| {
                    SaddleError::Invalid(format!("outer hop source {} is not a site", hop.source))
                })?;
            if site.outcome != SiteOutcome::Kinetic {
                return Err(SaddleError::Invalid(format!(
                    "outer hop source {} is not a kinetic site",
                    hop.source
                )));
            }
            let caught = radius / hop.distance;
            if caught > 0.0 {
                site.rates.push(("recombined".into(), hop.rate * caught));
            }
            if caught < 1.0 {
                site.rates
                    .push(("escaped".into(), hop.rate * (1.0 - caught)));
            }
        }
        kinetic_recombination_volume(&chain, atomic_volume)
    };
    let moments = |volume: &RecombinationVolume| -> Result<(f64, f64), SaddleError> {
        let mut weight = 0.0;
        let mut sum = 0.0;
        for (id, radius) in outer {
            let place = volume
                .placements
                .iter()
                .find(|place| &place.id == id)
                .ok_or_else(|| SaddleError::Invalid(format!("outer site {id} is not a site")))?;
            weight += place.count;
            sum += place.count * place.probability * radius;
        }
        Ok((weight, sum))
    };
    let (weight, base) = moments(&with_radius(0.0)?)?;
    let (_, probe) = moments(&with_radius(nearest)?)?;
    let slope = (probe - base) / nearest;
    let denominator = weight - slope;
    if !(weight > 0.0 && denominator > 0.0) {
        return Err(SaddleError::Invalid(
            "the outer sites cannot be matched to an outside capture radius".into(),
        ));
    }
    let radius = base / denominator;
    if !(radius.is_finite() && radius >= 0.0 && radius <= nearest) {
        return Err(SaddleError::Invalid(format!(
            "matched capture radius {radius} is outside [0, {nearest}]"
        )));
    }
    Ok(MatchedCapture {
        radius,
        volume: with_radius(radius)?,
    })
}

/// Unit path tangent at `arc`, pointing from the reactant toward the product.
///
/// Frames are stored reactant-first, so increasing arc length is the product
/// direction. A tangent that points the other way is reversed. The sign is
/// the path's, not a choice made after looking at `κ`.
pub fn product_tangent(path: &crate::pafi::ArcPath, arc: f64) -> Result<Array1<f64>, SaddleError> {
    let geometry = path.geometry(arc)?;
    let (start, end) = path.bounds();
    let span = end - start;
    let step = (span * 1.0e-4).max(1.0e-8);
    let forward = if arc + step <= end {
        path.geometry(arc + step)?.positions - &geometry.positions
    } else {
        &geometry.positions - &path.geometry((arc - step).max(start))?.positions
    };
    let mut tangent = geometry.tangent;
    if tangent.dot(&forward) < 0.0 {
        tangent.mapv_inplace(|value| -value);
    }
    unit(&tangent)
}

/// Mean time to absorption on the same rate graph as [`kinetic_recombination_volume`].
///
/// Spontaneous shells and every non-transient destination (`"recombined"`,
/// `"escaped"`, anything that is not another site's id) are absorbing, with
/// time 0. The returned time is the multiplicity-weighted mean over every
/// input site, spontaneous ones included. Rates are in Hz when the time is
/// in seconds. This is the mean-time input of a recombination volume: for a
/// cell of `n_atoms` and a migration frequency `nu`, `n_atoms / (mean_time * nu)`.
pub fn mean_absorption_time(sites: &[VolumeSite]) -> Result<f64, SaddleError> {
    let mut transient: Vec<&VolumeSite> = Vec::new();
    let mut weight = 0.0;
    for site in sites {
        if !(site.count.is_finite() && site.count >= 0.0) {
            return Err(SaddleError::Invalid(
                "site multiplicity must be finite and non-negative".into(),
            ));
        }
        weight += site.count;
        if site.outcome != SiteOutcome::Spontaneous {
            transient.push(site);
        }
    }
    if weight <= 0.0 {
        return Err(SaddleError::Invalid("the cell has no sites".into()));
    }
    let n = transient.len();
    let mut times = vec![0.0; sites.len()];
    if n > 0 {
        let index: std::collections::HashMap<&str, usize> = transient
            .iter()
            .enumerate()
            .map(|(i, site)| (site.id.as_str(), i))
            .collect();
        let mut matrix = vec![vec![0.0; n]; n];
        let rhs = vec![1.0; n];
        for (i, site) in transient.iter().enumerate() {
            let mut total = 0.0;
            for (_, rate) in &site.rates {
                if !(rate.is_finite() && *rate >= 0.0) {
                    return Err(SaddleError::Invalid(
                        "a hop rate must be finite and non-negative".into(),
                    ));
                }
                total += *rate;
            }
            if total <= 0.0 {
                return Err(SaddleError::Invalid(format!(
                    "site {} has no outgoing rate",
                    site.id
                )));
            }
            matrix[i][i] = total;
            for (destination, rate) in &site.rates {
                if let Some(&j) = index.get(destination.as_str()) {
                    matrix[i][j] -= *rate;
                }
            }
            let _ = rhs[i];
        }
        let solved = solve_linear(matrix, rhs)?;
        for (site, time) in transient.iter().zip(&solved) {
            if !(time.is_finite() && *time >= 0.0) {
                return Err(SaddleError::Invalid(format!(
                    "mean time for {} is not finite and non-negative",
                    site.id
                )));
            }
            if let Some(slot) = sites.iter().position(|item| item.id == site.id) {
                times[slot] = *time;
            }
        }
    }
    let mut accum = 0.0;
    for (site, time) in sites.iter().zip(&times) {
        accum += site.count * *time;
    }
    Ok(accum / weight)
}

/// Embed `sites` in free migration.
///
/// The boundary's absorbing `"escaped"` edge is replaced by a hop into
/// [`FREE_CRYSTAL`]. The inserted layer hops back to the boundary and onward
/// to infinity. `kinetic_recombination_volume` on the result is the capture
/// probability of that first-passage problem.
pub fn embed_free_migration(
    sites: &[VolumeSite],
    migration: &FreeMigration,
) -> Result<Vec<VolumeSite>, SaddleError> {
    if !(migration.outward_rate.is_finite() && migration.outward_rate > 0.0) {
        return Err(SaddleError::Invalid(
            "the outward rate must be finite and positive".into(),
        ));
    }
    if !(migration.return_probability.is_finite()
        && (0.0..=1.0).contains(&migration.return_probability)
        && migration.return_probability < 1.0)
    {
        return Err(SaddleError::Invalid(
            "the return probability must lie in [0, 1)".into(),
        ));
    }
    if !(migration.return_count.is_finite() && migration.return_count >= 0.0) {
        return Err(SaddleError::Invalid(
            "the return-layer multiplicity must be finite and non-negative".into(),
        ));
    }
    let mut found = false;
    let mut out = Vec::with_capacity(sites.len() + 1);
    for site in sites {
        let mut site = site.clone();
        if site.id == migration.boundary {
            if site.outcome != SiteOutcome::Kinetic {
                return Err(SaddleError::Invalid(format!(
                    "boundary {} is not a kinetic shell",
                    migration.boundary
                )));
            }
            found = true;
            site.rates
                .retain(|(destination, _)| destination != "escaped");
            site.rates
                .push((FREE_CRYSTAL.to_string(), migration.outward_rate));
        }
        out.push(site);
    }
    if !found {
        return Err(SaddleError::Invalid(format!(
            "boundary {} is not in the shell list",
            migration.boundary
        )));
    }
    let back = migration.outward_rate * migration.return_probability;
    let leave = migration.outward_rate * (1.0 - migration.return_probability);
    out.push(VolumeSite {
        id: FREE_CRYSTAL.to_string(),
        count: migration.return_count,
        outcome: SiteOutcome::Kinetic,
        rates: vec![
            (migration.boundary.clone(), back),
            ("escaped".to_string(), leave),
        ],
    });
    Ok(out)
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>, SaddleError> {
    let n = rhs.len();
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][col].abs() < 1.0e-14 {
            return Err(SaddleError::Invalid(
                "the shell rate graph is singular".into(),
            ));
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        let scale = matrix[col][col];
        for value in &mut matrix[col] {
            *value /= scale;
        }
        rhs[col] /= scale;
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            for col2 in col..n {
                matrix[row][col2] -= factor * matrix[col][col2];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    Ok(rhs)
}

/// Climbing band from the reactant through `saddle` to a nearby downhill
/// geometry. Springs are energy-weighted so images stay near the barrier.
#[allow(clippy::too_many_arguments)]
pub fn segment_band<S: PointSurface>(
    surface: &S,
    config: &DefectPairConfig,
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    downhill: &Array1<f64>,
    cell: M3,
    k_min: f64,
    k_max: f64,
) -> Result<(ArcPath, bool, f64), SaddleError> {
    let band = config
        .band
        .clone()
        .with_spring(crate::spring::SpringKind::EnergyWeighted { k_min, k_max });
    let config = config.clone().with_band(band);
    climb_path(surface, &config, reactant, saddle, downhill, cell, cell)
}

fn cartesian_push(dimer: &SolidDimer, push: f64) -> Result<Array1<f64>, SaddleError> {
    let n = dimer.positions().len();
    let mode = dimer.mode().to_owned();
    let q = dimer.generalized().to_owned();
    let mut step = Array1::zeros(n);
    if let Some(chart) = dimer.chart() {
        let eps = chart.strain(q.view())?;
        let deform = deform_of(eps);
        for atom in 0..n / 3 {
            let row = [mode[3 * atom], mode[3 * atom + 1], mode[3 * atom + 2]];
            for c in 0..3 {
                step[3 * atom + c] =
                    row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
            }
        }
    } else {
        let length = mode.len().min(n);
        step.slice_mut(s![..length])
            .assign(&mode.slice(s![..length]));
    }
    let scale = max_atomic(&step);
    if scale == 0.0 {
        return Err(SaddleError::Invalid(
            "the unstable mode has no atomic component".into(),
        ));
    }
    Ok(step * (push / scale))
}

/// Largest per-atom length. `push` is this length, not the 3N Euclidean norm,
/// so a localized mode on a large cell still leaves the saddle.
fn max_atomic(step: &Array1<f64>) -> f64 {
    let mut scale = 0.0_f64;
    let n = step.len() / 3;
    for atom in 0..n {
        let mut acc = 0.0;
        for c in 0..3 {
            let value = step[3 * atom + c];
            acc += value * value;
        }
        scale = scale.max(acc.sqrt());
    }
    scale
}

fn deform_of(eps: M3) -> M3 {
    [
        [1.0 + eps[0][0], eps[0][1], eps[0][2]],
        [eps[1][0], 1.0 + eps[1][1], eps[1][2]],
        [eps[2][0], eps[2][1], 1.0 + eps[2][2]],
    ]
}

fn links(
    plus: &Array1<f64>,
    minus: &Array1<f64>,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: M3,
    tol: f64,
) -> Result<bool, SaddleError> {
    let plus_r = hausdorff(plus, reactant, cell)?;
    let plus_p = hausdorff(plus, product, cell)?;
    let minus_r = hausdorff(minus, reactant, cell)?;
    let minus_p = hausdorff(minus, product, cell)?;
    let forward = plus_p <= tol && minus_r <= tol;
    let backward = plus_r <= tol && minus_p <= tol;
    Ok(forward || backward)
}

/// Largest gap between the two point clouds. Identical atoms make an index
/// RMSD depend on which atom of a split site moved; the clouds do not.
fn hausdorff(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> Result<f64, SaddleError> {
    Ok(directed_gap(left, right, cell)?.max(directed_gap(right, left, cell)?))
}

fn directed_gap(left: &Array1<f64>, right: &Array1<f64>, cell: M3) -> Result<f64, SaddleError> {
    let n = left.len() / 3;
    let m = right.len() / 3;
    if n == 0 || n != m {
        return Err(SaddleError::Shape(
            "connectivity geometries must have the same number of atoms".into(),
        ));
    }
    let mut worst = 0.0;
    for i in 0..n {
        let origin = [left[3 * i], left[3 * i + 1], left[3 * i + 2]];
        let mut best = f64::MAX;
        for j in 0..m {
            let delta = mic(
                [
                    right[3 * j] - origin[0],
                    right[3 * j + 1] - origin[1],
                    right[3 * j + 2] - origin[2],
                ],
                cell,
            )?;
            let r = norm3(delta);
            if r < best {
                best = r;
            }
        }
        if best > worst {
            worst = best;
        }
    }
    Ok(worst)
}

fn relax<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &DefectPairConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    if config.cell_mode == CellMode::Variable {
        return relax_joint(surface, positions, cell, config);
    }
    let done = relax_to_tolerance(surface, positions, cell, &config.relax)?;
    Ok((done.positions, done.energy))
}

fn relax_joint<S: PointSurface>(
    surface: &S,
    positions: Array1<f64>,
    cell: M3,
    config: &DefectPairConfig,
) -> Result<(Array1<f64>, f64), SaddleError> {
    let n_atoms = positions.len() / 3;
    let chart = JointChart::new(cell, n_atoms, CellMask::all(), config.external_stress, 1.0)?;
    let mut q = Array1::zeros(chart.dimension());
    q.slice_mut(s![..positions.len()]).assign(&positions);
    let relax = &config.relax;
    let mut solver = rgmin::Solver::new(
        relax.method.clone(),
        rgmin::Control::default()
            .with_maxiter(relax.max_steps)
            .with_gtol(relax.force_tol)
            .with_stop_norm(rgmin::StopNorm::MaxAtom)
            .with_istep(relax.initial_step),
        q.len(),
    );
    solver.set_atom_maxmove(relax.max_atom_move);
    let oracle = rgmin::Oracle::unbounded(q.len(), |q: ArrayView1<f64>| {
        match joint_gradient(surface, &chart, q) {
            Ok(pair) => pair,
            Err(_) => (f64::NAN, Array1::zeros(q.len())),
        }
    });
    let report = solver
        .run(&oracle, &mut q)
        .map_err(|error| SaddleError::Solver(error.to_string()))?;
    let (energy, gradient) = joint_gradient(surface, &chart, q.view())?;
    let largest = max_atom_force(gradient.view());
    if !(energy.is_finite() && largest < relax.force_tol) {
        return Err(SaddleError::Solver(format!(
            "joint relaxation not converged: largest generalized force {largest:.3e} is not below {:.1e} after {} of {} steps ({:?})",
            relax.force_tol, report.steps, relax.max_steps, report.status
        )));
    }
    let (positions, _) = chart.geometry(q.view())?;
    Ok((positions, energy))
}

fn joint_gradient<S: PointSurface>(
    surface: &S,
    chart: &JointChart,
    q: ArrayView1<f64>,
) -> Result<(f64, Array1<f64>), SaddleError> {
    let (positions, cell) = chart.geometry(q)?;
    let flat = m3_to_row9(cell);
    let (energy, gradient) = surface.eval_in_cell(positions.view(), &flat)?;
    let stress = cauchy(surface, positions.view(), cell)?;
    let generalized = chart.gradient(q, gradient.view(), stress)?;
    Ok((chart.enthalpy(energy, q)?, generalized))
}

fn cauchy<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    cell: M3,
) -> Result<M3, SaddleError> {
    let flat = m3_to_row9(cell);
    if let Some(raw) = surface.cauchy_stress(positions, &flat)? {
        return Ok(unflat(&raw));
    }
    finite_difference_cauchy_stress(surface, positions, cell, 1.0e-5)
}

#[allow(clippy::type_complexity)]
fn hessian<S: PointSurface>(
    surface: &S,
    positions: ArrayView1<f64>,
    atoms: Option<&[usize]>,
    step: f64,
) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), SaddleError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(SaddleError::Invalid(
            "harmonic difference step must be positive".into(),
        ));
    }
    let coords: Vec<usize> = match atoms {
        None => (0..positions.len()).collect(),
        Some(atoms) => {
            let mut coords = Vec::with_capacity(atoms.len() * 3);
            for &atom in atoms {
                if atom >= positions.len() / 3 {
                    return Err(SaddleError::Invalid(
                        "harmonic atom is outside the cell".into(),
                    ));
                }
                coords.extend([3 * atom, 3 * atom + 1, 3 * atom + 2]);
            }
            coords
        }
    };
    if coords.is_empty() {
        return Err(SaddleError::Invalid("the harmonic region is empty".into()));
    }
    let n = coords.len();
    let mut matrix = Array2::zeros((n, n));
    for (col, &coordinate) in coords.iter().enumerate() {
        let mut plus = positions.to_owned();
        let mut minus = positions.to_owned();
        plus[coordinate] += step;
        minus[coordinate] -= step;
        let (_, g_plus) = surface.eval(plus.view())?;
        let (_, g_minus) = surface.eval(minus.view())?;
        for (row, &other) in coords.iter().enumerate() {
            matrix[(row, col)] = (g_plus[other] - g_minus[other]) / (2.0 * step);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let mean = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = mean;
            matrix[(j, i)] = mean;
        }
    }
    let (values, vectors) = exact_eigh(matrix.view())?;
    Ok((matrix, values.to_vec(), vectors))
}

fn curvature_along(
    hessian: &Array2<f64>,
    tangent: ArrayView1<f64>,
    atoms: Option<&[usize]>,
    n_full: usize,
) -> f64 {
    let coords: Vec<usize> = match atoms {
        None => (0..n_full).collect(),
        Some(atoms) => atoms
            .iter()
            .flat_map(|atom| [3 * atom, 3 * atom + 1, 3 * atom + 2])
            .collect(),
    };
    let mut lambda = 0.0;
    for (i, &ci) in coords.iter().enumerate() {
        for (j, &cj) in coords.iter().enumerate() {
            lambda += tangent[ci] * hessian[(i, j)] * tangent[cj];
        }
    }
    lambda
}

fn positives(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .copied()
        .filter(|value| *value > 1.0e-4)
        .collect()
}

fn vineyard_frequency(
    min_lambda: &[f64],
    sad_lambda: &[f64],
    mass: f64,
    angular: f64,
) -> Result<f64, SaddleError> {
    if !(mass.is_finite() && mass > 0.0 && angular.is_finite() && angular > 0.0) {
        return Err(SaddleError::Invalid(
            "Vineyard mass and angular scale must be positive".into(),
        ));
    }
    if min_lambda.len() <= sad_lambda.len() {
        return Err(SaddleError::Invalid(
            "the saddle product is not shorter than the minimum product".into(),
        ));
    }
    let mut log_ratio = 0.0;
    for lambda in min_lambda {
        if !(lambda.is_finite() && *lambda > 0.0) {
            return Err(SaddleError::Invalid(
                "minimum eigenvalues in the Vineyard product must be positive".into(),
            ));
        }
        log_ratio += 0.5 * lambda.ln();
    }
    for lambda in sad_lambda {
        if !(lambda.is_finite() && *lambda > 0.0) {
            return Err(SaddleError::Invalid(
                "saddle eigenvalues in the Vineyard product must be positive".into(),
            ));
        }
        log_ratio -= 0.5 * lambda.ln();
    }
    let extra = (min_lambda.len() - sad_lambda.len()) as f64;
    log_ratio -= 0.5 * extra * mass.ln();
    log_ratio -= extra * (2.0 * std::f64::consts::PI).ln();
    let frequency = log_ratio.exp() * angular.powf(extra);
    if !frequency.is_finite() || frequency <= 0.0 {
        return Err(SaddleError::NonFinite("Vineyard prefactor"));
    }
    Ok(frequency)
}

/// `ν = angular * sqrt(λ / mass) / 2π`.
pub fn attempt_frequency(lambda: f64, mass: f64, angular: f64) -> Result<f64, SaddleError> {
    if !(lambda.is_finite() && lambda > 0.0 && mass.is_finite() && mass > 0.0) {
        return Err(SaddleError::Invalid(
            "attempt frequency needs a positive curvature and mass".into(),
        ));
    }
    if !(angular.is_finite() && angular > 0.0) {
        return Err(SaddleError::Invalid(
            "angular scale must be positive".into(),
        ));
    }
    Ok(angular * (lambda / mass).sqrt() / std::f64::consts::TAU)
}

fn arrhenius(prefactor: f64, barrier: f64, thermal: f64) -> f64 {
    let exponent = -barrier / thermal;
    if exponent < -700.0 {
        0.0
    } else {
        prefactor * exponent.exp()
    }
}

fn plane_arcs(path: &ArcPath, planes: usize) -> Vec<f64> {
    let nodes = path.arcs();
    if planes < 2 {
        return nodes;
    }
    let (start, end) = path.bounds();
    (0..planes)
        .map(|i| start + (end - start) * (i as f64) / (planes - 1) as f64)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn blank_entry(
    setup: &PairSetup,
    hop: &HopTemplate,
    direction: DirectionClass,
    temperature: f64,
    n_atoms: usize,
    before: f64,
    after: f64,
    reactant_energy: f64,
    product_energy: f64,
) -> RateEntry {
    RateEntry {
        shell: setup.shell.clone(),
        direction,
        hop: hop.name.clone(),
        temperature,
        cell_repeats: setup.cell_repeats,
        n_atoms,
        separation_before: before,
        separation_after: after,
        outcome: HopOutcome::NotConverged,
        connected: false,
        index1: false,
        band_converged: false,
        reactant_energy,
        product_energy,
        saddle_energy: reactant_energy,
        d_e0: 0.0,
        d_f: None,
        d_f_error: None,
        d_f_harmonic: None,
        nu: None,
        vineyard: None,
        kappa: None,
        kappa_error: None,
        harmonic_rate: None,
        rate: None,
        zero_t: None,
        band_force: 0.0,
        pafi_samples: 0,
        pafi_evaluations: 0,
        saddle_stress: [[0.0; 3]; 3],
        saddle_dipole: [[0.0; 3]; 3],
        negative_eigenvalue: None,
        orientation: hop.orientation.clone(),
        final_state: hop.final_state.clone(),
    }
}

fn check_setup(setup: &PairSetup) -> Result<(), SaddleError> {
    let n = setup.positions.len();
    if n == 0 || !n.is_multiple_of(3) || !setup.positions.iter().all(|v| v.is_finite()) {
        return Err(SaddleError::Shape(
            "defect-pair positions must be a finite 3N vector".into(),
        ));
    }
    if m3_det(setup.cell).abs() < 1.0e-18 {
        return Err(SaddleError::Invalid(
            "defect-pair cell must have positive volume".into(),
        ));
    }
    if setup.mobile.is_empty() || setup.mobile.iter().any(|atom| atom >= &(n / 3)) {
        return Err(SaddleError::Invalid(
            "mobile atoms must be a non-empty set of indices".into(),
        ));
    }
    if setup.hops.is_empty() {
        return Err(SaddleError::Invalid("the hop catalogue is empty".into()));
    }
    for hop in &setup.hops {
        if hop.product.len() != n {
            return Err(SaddleError::Shape(
                "a hop product must have the reactant length".into(),
            ));
        }
        if hop.name.is_empty() {
            return Err(SaddleError::Invalid("a hop needs a name".into()));
        }
    }
    Ok(())
}

fn check_config(config: &DefectPairConfig) -> Result<(), SaddleError> {
    if config.dimer.index.is_none() {
        return Err(SaddleError::Invalid(
            "defect-pair search requires the solid-dimer index check".into(),
        ));
    }
    if config.images < 3 {
        return Err(SaddleError::Invalid(
            "the climbing image band needs at least three images".into(),
        ));
    }
    if config.temperatures.is_empty()
        || config
            .temperatures
            .iter()
            .any(|t| !(t.is_finite() && *t > 0.0))
    {
        return Err(SaddleError::Invalid(
            "at least one positive finite temperature is required".into(),
        ));
    }
    if !(config.boltzmann.is_finite() && config.boltzmann > 0.0) {
        return Err(SaddleError::Invalid(
            "Boltzmann's constant must be positive".into(),
        ));
    }
    if !(config.seed_fraction.is_finite() && (0.0..1.0).contains(&config.seed_fraction)) {
        return Err(SaddleError::Invalid(
            "the dimer seed fraction must lie in [0, 1)".into(),
        ));
    }
    if !(config.push.is_finite() && config.push > 0.0 && config.connect_rmsd > 0.0) {
        return Err(SaddleError::Invalid(
            "the push and the connectivity tolerance must be positive".into(),
        ));
    }
    config.relax.check()?;
    if !(config.zero_t_tol.is_finite() && config.zero_t_tol > 0.0) {
        return Err(SaddleError::Invalid(
            "the zero-temperature tolerance must be positive".into(),
        ));
    }
    Ok(())
}

fn enthalpy(energy: f64, cell: M3, reference: M3, external: M3) -> Result<f64, SaddleError> {
    let eps = strain_between(reference, cell)?;
    let volume = m3_det(reference).abs();
    Ok(energy + volume * double_dot(external, eps))
}

fn strain_between(reference: M3, cell: M3) -> Result<M3, SaddleError> {
    let prod = m3_mul(m3_inv(reference)?, cell);
    Ok([
        [prod[0][0] - 1.0, prod[0][1], prod[0][2]],
        [prod[1][0], prod[1][1] - 1.0, prod[1][2]],
        [prod[2][0], prod[2][1], prod[2][2] - 1.0],
    ])
}

fn mix(
    start: &Array1<f64>,
    end: &Array1<f64>,
    fraction: f64,
    cell: M3,
) -> Result<Array1<f64>, SaddleError> {
    let delta = mic_positions(end, start, cell)?;
    Ok(start + delta * fraction)
}

fn mic_positions(
    end: &Array1<f64>,
    start: &Array1<f64>,
    cell: M3,
) -> Result<Array1<f64>, SaddleError> {
    if end.len() != start.len() {
        return Err(SaddleError::Shape(
            "geometries in one hop must share a length".into(),
        ));
    }
    let mut out = Array1::zeros(start.len());
    for atom in 0..start.len() / 3 {
        let delta = mic(
            [
                end[3 * atom] - start[3 * atom],
                end[3 * atom + 1] - start[3 * atom + 1],
                end[3 * atom + 2] - start[3 * atom + 2],
            ],
            cell,
        )?;
        for c in 0..3 {
            out[3 * atom + c] = delta[c];
        }
    }
    Ok(out)
}

fn mobile_center(
    positions: ArrayView1<f64>,
    cell: M3,
    mobile: &[usize],
) -> Result<[f64; 3], SaddleError> {
    let origin = mobile[0];
    let base = [
        positions[3 * origin],
        positions[3 * origin + 1],
        positions[3 * origin + 2],
    ];
    let mut acc = [0.0; 3];
    for &atom in mobile {
        let point = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        let delta = mic(sub3(point, base), cell)?;
        for c in 0..3 {
            acc[c] += base[c] + delta[c];
        }
    }
    let scale = mobile.len() as f64;
    Ok([acc[0] / scale, acc[1] / scale, acc[2] / scale])
}

#[allow(clippy::needless_range_loop)]
fn mic(delta: [f64; 3], cell: M3) -> Result<[f64; 3], SaddleError> {
    let inv = m3_inv(cell)?;
    let mut s = [0.0; 3];
    for i in 0..3 {
        s[i] = delta[0] * inv[0][i] + delta[1] * inv[1][i] + delta[2] * inv[2][i];
        s[i] -= s[i].round();
    }
    Ok([
        s[0] * cell[0][0] + s[1] * cell[1][0] + s[2] * cell[2][0],
        s[0] * cell[0][1] + s[1] * cell[1][1] + s[2] * cell[2][1],
        s[0] * cell[0][2] + s[1] * cell[1][2] + s[2] * cell[2][2],
    ])
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn norm1(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn unit(v: &Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let scale = norm1(v);
    if scale == 0.0 {
        return Err(SaddleError::Invalid("a path tangent vanishes".into()));
    }
    Ok(v / scale)
}

fn unflat(cell: &[f64; 9]) -> M3 {
    [
        [cell[0], cell[1], cell[2]],
        [cell[3], cell[4], cell[5]],
        [cell[6], cell[7], cell[8]],
    ]
}

fn flag(obj: &[(String, Json)], key: &str) -> Result<bool, SaddleError> {
    let value = required_f64(obj, key)?;
    Ok(value > 0.5)
}

fn optional_str<'a>(obj: &'a [(String, Json)], key: &str) -> Option<&'a str> {
    obj_get(obj, key).and_then(Json::as_str)
}

fn optional_f64(obj: &[(String, Json)], key: &str) -> Result<Option<f64>, SaddleError> {
    match obj_get(obj, key) {
        None | Some(Json::Null) => Ok(None),
        Some(value) => value.as_f64().map(Some).ok_or_else(|| {
            SaddleError::Invalid(format!("json field {key} must be a number or null"))
        }),
    }
}

fn num(value: f64) -> String {
    if value == 0.0 {
        "0".to_string()
    } else {
        format!("{value:.16e}")
    }
}

fn opt_num(value: Option<f64>) -> String {
    match value {
        Some(value) => num(value),
        None => "null".to_string(),
    }
}

fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn matrix_json(matrix: M3) -> String {
    format!(
        "[[{},{},{}],[{},{},{}],[{},{},{}]]",
        num(matrix[0][0]),
        num(matrix[0][1]),
        num(matrix[0][2]),
        num(matrix[1][0]),
        num(matrix[1][1]),
        num(matrix[1][2]),
        num(matrix[2][0]),
        num(matrix[2][1]),
        num(matrix[2][2])
    )
}

fn push_kv(out: &mut String, key: &str, value: &str, comma: bool) {
    out.push_str("      \"");
    out.push_str(key);
    out.push_str("\": ");
    out.push_str(value);
    if comma {
        out.push(',');
    }
    out.push('\n');
}
