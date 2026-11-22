//! Reversible work between hyperplanes.
//!
//! The free energy of a plane changes by `dA = ∂A/∂p · dp + ∂A/∂ν · dν`
//! (Jóhannesson and Jónsson, Eqs. 4, 18 and 19). [`reversible_work_along`]
//! integrates it along a [`PlanePath`]: PAFI planes along a curve, straight
//! legs between given planes, the optimizer's own iterates, or any chain of
//! them. A node samples its plane with [`crate::ohtst::sample_plane`] and gives the
//! derivative of `A` along the path's `[dp/ds; dν/ds]` from the block
//! means, so the nodes are independent and the error of the sum is
//! `sqrt(Σ wᵢ² σᵢ²)` over the quadrature weights. Each node starts from
//! the last configuration of the one before. `A` is a function of the
//! plane, so any path between two planes gives the same work.
//! [`reversible_work`] takes straight legs through given planes.
//!
//! With [`ReversibleWorkConfig::bar_samples`] the same nodes also give the
//! work without quadrature error, by Bennett's acceptance ratio between
//! neighbours (J. Comput. Phys. 22, 245 (1976)). A map `T` from plane `a`
//! onto plane `b` with Jacobian `J` gives
//! `Z_b / Z_a = J ⟨exp(-ΔV/kT)⟩_a`, `ΔV = V(T q) - V(q)`, and the reverse
//! average runs over `b` with `T⁻¹` (Eqs. 7.2 and 12.3 of the OH-TST
//! design survey). [`ReversibleWorkConfig::transport`] picks the rigid map
//! or the Gaussian optimal-transport map, which removes the loss of
//! overlap from in-plane means and covariances that change between the
//! planes. The Gaussian map keeps only the share of its fitted change that
//! the samples resolve, reported as [`WorkNode::transport_weight`], so in
//! many dimensions with few samples it falls back toward the rigid map
//! instead of adding noise. Each pair reports its overlap, the
//! off-diagonal element of the two-state overlap matrix, and pairs below
//! [`ReversibleWorkConfig::min_overlap`] are named in
//! [`ReversibleWork::warnings`].
//!
//! With [`ReversibleWorkConfig::mbar`] every node's configurations are
//! carried onto every plane by the chain of maps, and MBAR (Shirts and
//! Chodera, J. Chem. Phys. 129, 124105 (2008)) combines them all. BAR and
//! MBAR use the configurations each node retains, so their spacing should
//! exceed the correlation time; the Gaussian map is fitted to a second,
//! interleaved set. [`ReversibleWorkConfig::estimate`] picks the estimate
//! that [`ReversibleWork::work`] reports, and so the one the rate uses.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::ohtst::mbar::{mbar, pair_overlap};
use crate::ohtst::path::PlanePath;
use crate::ohtst::plane::Hyperplane;
use crate::ohtst::sample::{
    AnalyticSurfaceSampler, LangevinConfig, PlaneSample, sample_plane_with_sampler,
};
use crate::ohtst::transport::{PlaneMap, Transport};
use crate::projector::HyperplaneSampler;

/// Which estimate [`ReversibleWork::work`] holds.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum WorkEstimate {
    /// Thermodynamic integration over the nodes.
    #[default]
    Integration,
    /// BAR between neighbouring nodes, summed along the path.
    Bar,
    /// MBAR across all nodes.
    Mbar,
}

/// Settings of [`reversible_work`] and [`reversible_work_along`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ReversibleWorkConfig {
    /// Sampling of each node.
    pub sampler: LangevinConfig,
    /// Burn-in of a node that starts from the previous node's last
    /// configuration.
    pub warm_burn_in: usize,
    /// Nodes on each segment of the path, ends included.
    pub nodes: usize,
    /// Composite Simpson weights; they need an odd node count. Otherwise
    /// trapezoid.
    pub simpson: bool,
    /// Configurations each node keeps for BAR and MBAR; zero turns both
    /// off. A larger [`LangevinConfig::retain`] wins.
    pub bar_samples: usize,
    /// The map that carries configurations between planes. A Gaussian map
    /// retains twice [`Self::bar_samples`] and fits to every other one.
    pub transport: Transport,
    /// Solve MBAR across all nodes. That evaluates every retained
    /// configuration on every plane, so its cost grows with the square of
    /// the node count.
    pub mbar: bool,
    /// The estimate in [`ReversibleWork::work`] and
    /// [`ReversibleWork::error`].
    pub estimate: WorkEstimate,
    /// Neighbouring nodes that overlap less than this are named in
    /// [`ReversibleWork::warnings`].
    pub min_overlap: f64,
}

impl Default for ReversibleWorkConfig {
    fn default() -> Self {
        Self {
            sampler: LangevinConfig::default(),
            warm_burn_in: 100,
            nodes: 9,
            simpson: true,
            bar_samples: 128,
            transport: Transport::Rigid,
            mbar: false,
            estimate: WorkEstimate::Integration,
            min_overlap: 0.03,
        }
    }
}

impl ReversibleWorkConfig {
    /// Set [`Self::bar_samples`].
    #[must_use]
    pub fn with_bar_samples(mut self, bar_samples: usize) -> Self {
        self.bar_samples = bar_samples;
        self
    }

    /// Set [`Self::sampler`].
    #[must_use]
    pub fn with_sampler(mut self, sampler: LangevinConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// Set [`Self::warm_burn_in`].
    #[must_use]
    pub fn with_warm_burn_in(mut self, warm_burn_in: usize) -> Self {
        self.warm_burn_in = warm_burn_in;
        self
    }

    /// Set [`Self::nodes`].
    #[must_use]
    pub fn with_nodes(mut self, nodes: usize) -> Self {
        self.nodes = nodes;
        self
    }

    /// Set [`Self::simpson`].
    #[must_use]
    pub fn with_simpson(mut self, simpson: bool) -> Self {
        self.simpson = simpson;
        self
    }

    /// Set [`Self::transport`].
    #[must_use]
    pub fn with_transport(mut self, transport: Transport) -> Self {
        self.transport = transport;
        self
    }

    /// Set [`Self::mbar`].
    #[must_use]
    pub fn with_mbar(mut self, mbar: bool) -> Self {
        self.mbar = mbar;
        self
    }

    /// Set [`Self::estimate`].
    #[must_use]
    pub fn with_estimate(mut self, estimate: WorkEstimate) -> Self {
        self.estimate = estimate;
        self
    }

    /// Set [`Self::min_overlap`].
    #[must_use]
    pub fn with_min_overlap(mut self, min_overlap: f64) -> Self {
        self.min_overlap = min_overlap;
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        self.sampler.check()?;
        if self.nodes < 2 || (self.simpson && self.nodes.is_multiple_of(2)) {
            return Err(SaddleError::Invalid(
                "a segment needs two nodes or more, and an odd count for Simpson".into(),
            ));
        }
        if !(0.0..1.0).contains(&self.min_overlap) {
            return Err(SaddleError::Invalid(
                "the overlap threshold must lie in [0, 1)".into(),
            ));
        }
        if let Transport::Gaussian(gaussian) = &self.transport {
            gaussian.check()?;
        }
        let needs_samples = self.mbar || self.estimate != WorkEstimate::Integration;
        if needs_samples && self.bar_samples == 0 {
            return Err(SaddleError::Invalid(
                "BAR and MBAR need retained configurations; set bar_samples".into(),
            ));
        }
        if self.estimate == WorkEstimate::Mbar && !self.mbar {
            return Err(SaddleError::Invalid(
                "the MBAR estimate needs mbar on".into(),
            ));
        }
        Ok(())
    }
}

/// One node of [`ReversibleWork`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct WorkNode {
    /// Segment index.
    pub leg: usize,
    /// Position on the segment, from 0 to 1.
    pub s: f64,
    /// The sampled plane.
    pub plane: Hyperplane,
    /// `dA/ds` on the segment.
    pub derivative: f64,
    /// Block standard error of [`Self::derivative`].
    pub error: f64,
    /// Work from the first plane to this node by the trapezoid rule.
    pub work: f64,
    /// Work from the first plane to this node by BAR between neighbours.
    pub bar_work: Option<f64>,
    /// Overlap with the node before, from the BAR works of the pair.
    pub overlap: Option<f64>,
    /// Share of the fitted Gaussian change that the map from the node
    /// before carries, from 0 for the rigid map to 1; see
    /// [`crate::GaussianTransport::shrink_to_rigid`].
    pub transport_weight: Option<f64>,
    /// Work from the first plane to this node by MBAR.
    pub mbar_work: Option<f64>,
}

/// `A(last plane) - A(first plane)`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ReversibleWork {
    /// The work by [`Self::estimate`].
    pub work: f64,
    /// Its standard error.
    pub error: f64,
    /// The estimate in [`Self::work`].
    pub estimate: WorkEstimate,
    /// The work by thermodynamic integration.
    pub integrated_work: f64,
    /// Its standard error.
    pub integrated_error: f64,
    /// Every node in path order. A node shared by two segments is sampled
    /// once.
    pub nodes: Vec<WorkNode>,
    /// The sample at the last plane.
    pub end: PlaneSample,
    /// Surface evaluations over all nodes.
    pub evaluations: usize,
    /// The sampling temperature `kT`.
    pub thermal: f64,
    /// The work by BAR between neighbouring nodes.
    pub bar_work: Option<f64>,
    /// Asymptotic standard error of [`Self::bar_work`].
    pub bar_error: Option<f64>,
    /// The work by MBAR across all nodes.
    pub mbar_work: Option<f64>,
    /// Asymptotic standard error of [`Self::mbar_work`].
    pub mbar_error: Option<f64>,
    /// The MBAR overlap matrix of the nodes; its rows sum to one.
    pub overlap: Option<Array2<f64>>,
    /// Neighbouring nodes that overlap less than
    /// [`ReversibleWorkConfig::min_overlap`], and Gaussian maps that fell
    /// back to the rigid one.
    pub warnings: Vec<String>,
}

fn weights(nodes: usize, simpson: bool) -> Vec<f64> {
    let h = 1.0 / (nodes - 1) as f64;
    (0..nodes)
        .map(|i| {
            let ends = i == 0 || i == nodes - 1;
            if simpson {
                h / 3.0
                    * if ends {
                        1.0
                    } else if i % 2 == 1 {
                        4.0
                    } else {
                        2.0
                    }
            } else if ends {
                h / 2.0
            } else {
                h
            }
        })
        .collect()
}

/// The configurations a node keeps for BAR and MBAR.
struct Kept {
    plane: Hyperplane,
    evaluation: Vec<Array1<f64>>,
    energies: Vec<f64>,
    fit: Vec<Array1<f64>>,
    /// Energies of [`Self::evaluation`] carried onto the next plane.
    up: Vec<f64>,
    /// Energies of [`Self::evaluation`] carried back onto the plane before.
    down: Vec<f64>,
}

/// The work to move from `planes[0]` to the last plane on straight legs
/// through the planes between.
pub fn reversible_work<E: HyperplaneSampler>(
    sampler: &mut E,
    planes: &[Hyperplane],
    config: &ReversibleWorkConfig,
) -> Result<ReversibleWork, SaddleError> {
    reversible_work_along(sampler, &PlanePath::legs(planes)?, config)
}

/// The work to move from the first plane of `path` to its last.
pub fn reversible_work_along<E: HyperplaneSampler>(
    sampler: &mut E,
    path: &PlanePath,
    config: &ReversibleWorkConfig,
) -> Result<ReversibleWork, SaddleError> {
    config.check()?;
    let weights = weights(config.nodes, config.simpson);
    let h = 1.0 / (config.nodes - 1) as f64;
    let gaussian = matches!(config.transport, Transport::Gaussian(_));
    let retain = config.bar_samples * if gaussian { 2 } else { 1 };
    let mut nodes: Vec<WorkNode> = Vec::new();
    // The latest sample and the weighted tangent it carries into the sum;
    // the node shared by two segments carries both segments' weights.
    let mut open: Option<(PlaneSample, Array1<f64>)> = None;
    let mut work = 0.0;
    let mut variance = 0.0;
    let mut evaluations = 0;
    let mut sampled = 0u64;
    let first = path.start()?;
    let held = sampler.constrained_directions(&first)?;
    let kt = config.sampler.thermal;
    let mut kept: Vec<Kept> = Vec::new();
    let mut maps: Vec<PlaneMap> = Vec::new();
    let mut warnings = Vec::new();
    let (mut bar_work, mut bar_variance) = (0.0, 0.0);
    for k in 0..path.len() {
        let mut cumulative = nodes.last().map_or(0.0, |node| node.work);
        let mut before = None;
        for (i, weight) in weights.iter().enumerate() {
            let s = i as f64 * h;
            let tangent = path.tangent(k, s)?;
            let mut overlap = None;
            let mut transport_weight = None;
            if i > 0 || k == 0 {
                if let Some((sample, direction)) = open.take() {
                    let (value, error) = sample.directional(direction.view())?;
                    work += value;
                    variance += error * error;
                }
                let burn_in = if sampled > 0 {
                    config.warm_burn_in
                } else {
                    config.sampler.burn_in
                };
                let sampling = config
                    .sampler
                    .clone()
                    .with_seed(config.sampler.seed.wrapping_add(sampled))
                    .with_retain(config.sampler.retain.max(retain))
                    .with_burn_in(burn_in);
                let plane = path.plane(k, s)?;
                let sample = sample_plane_with_sampler(sampler, &plane, &sampling)?;
                sampled += 1;
                evaluations += sample.evaluations;
                if config.bar_samples > 0 {
                    let (evaluation, fit) = if gaussian {
                        let (even, odd): (Vec<_>, Vec<_>) = sample
                            .retained
                            .iter()
                            .enumerate()
                            .partition(|(j, _)| j % 2 == 0);
                        (
                            even.into_iter().map(|(_, q)| q.clone()).collect(),
                            odd.into_iter().map(|(_, q)| q.clone()).collect(),
                        )
                    } else {
                        (sample.retained.clone(), Vec::new())
                    };
                    let energies = evaluation
                        .iter()
                        .map(|q| energy(sampler, q, &mut evaluations))
                        .collect::<Result<Vec<_>, _>>()?;
                    let mut current = Kept {
                        plane: sample.plane.clone(),
                        evaluation,
                        energies,
                        fit,
                        up: Vec::new(),
                        down: Vec::new(),
                    };
                    let index = kept.len();
                    if let Some(previous) = kept.last_mut() {
                        let map = PlaneMap::fit(
                            &config.transport,
                            &previous.plane,
                            &current.plane,
                            &held,
                            &previous.fit,
                            &current.fit,
                        )?;
                        let jacobian = map.log_jacobian();
                        let mut forward = Vec::with_capacity(previous.evaluation.len());
                        for (q, e) in previous.evaluation.iter().zip(&previous.energies) {
                            let mapped = energy(sampler, &map.forward(q), &mut evaluations)?;
                            previous.up.push(mapped);
                            forward.push((mapped - e) / kt - jacobian);
                        }
                        let mut reverse = Vec::with_capacity(current.evaluation.len());
                        for (q, e) in current.evaluation.iter().zip(&current.energies) {
                            let mapped = energy(sampler, &map.backward(q), &mut evaluations)?;
                            current.down.push(mapped);
                            reverse.push((mapped - e) / kt + jacobian);
                        }
                        let (difference, error) = bennett(&forward, &reverse)?;
                        bar_work += kt * difference;
                        bar_variance += (kt * error).powi(2);
                        let pair = pair_overlap(&forward, &reverse, difference);
                        if map.fallback {
                            warnings.push(format!(
                                "the Gaussian map from node {} to node {index} fell back to the \
                                 rigid map: a fitted covariance is not positive definite",
                                index - 1
                            ));
                        }
                        if pair < config.min_overlap {
                            warnings.push(format!(
                                "nodes {} and {index} overlap {pair:.3e} under BAR, below {}; \
                                 place planes closer{}",
                                index - 1,
                                config.min_overlap,
                                if gaussian {
                                    ""
                                } else {
                                    " or use the Gaussian transport map"
                                }
                            ));
                        }
                        overlap = Some(pair);
                        transport_weight = gaussian.then(|| map.weight());
                        maps.push(map);
                    }
                    kept.push(current);
                }
                open = Some((sample, Array1::zeros(tangent.len())));
            }
            let (sample, direction) = open
                .as_mut()
                .ok_or_else(|| SaddleError::Invalid("a work segment has no sample".into()))?;
            direction.scaled_add(*weight, &tangent);
            let (derivative, error) = sample.directional(tangent.view())?;
            if let Some(previous) = before {
                cumulative += 0.5 * h * (previous + derivative);
            }
            before = Some(derivative);
            if i > 0 || k == 0 {
                nodes.push(WorkNode {
                    leg: k,
                    s,
                    plane: sample.plane.clone(),
                    derivative,
                    error,
                    work: cumulative,
                    bar_work: (config.bar_samples > 0).then_some(bar_work),
                    overlap,
                    transport_weight,
                    mbar_work: None,
                });
            }
        }
    }
    let (end, direction) =
        open.ok_or_else(|| SaddleError::Invalid("a work path has no sample".into()))?;
    let (value, error) = end.directional(direction.view())?;
    work += value;
    variance += error * error;

    let mut mbar_estimate = None;
    let mut overlap_matrix = None;
    if config.mbar {
        let initial: Vec<f64> = nodes
            .iter()
            .map(|node| node.bar_work.unwrap_or(0.0) / kt)
            .collect();
        let (u, counts) = reduced_potentials(sampler, &kept, &maps, kt, &mut evaluations)?;
        let solved = mbar(&u, &counts, &initial)?;
        let last = kept.len() - 1;
        let c = &solved.covariance;
        let spread = (c[(last, last)] + c[(0, 0)] - 2.0 * c[(0, last)]).max(0.0);
        for (node, f) in nodes.iter_mut().zip(solved.free_energies.iter()) {
            node.mbar_work = Some(kt * f);
        }
        mbar_estimate = Some((kt * solved.free_energies[last], kt * spread.sqrt()));
        overlap_matrix = Some(solved.overlap);
    }
    let bar = (config.bar_samples > 0).then(|| (bar_work, bar_variance.sqrt()));
    let integrated = (work, variance.sqrt());
    let (selected, selected_error) = match config.estimate {
        WorkEstimate::Integration => integrated,
        WorkEstimate::Bar => {
            bar.ok_or_else(|| SaddleError::Solver("BAR estimate was not computed".into()))?
        }
        WorkEstimate::Mbar => mbar_estimate
            .ok_or_else(|| SaddleError::Solver("MBAR estimate was not computed".into()))?,
    };
    Ok(ReversibleWork {
        work: selected,
        error: selected_error,
        estimate: config.estimate,
        integrated_work: integrated.0,
        integrated_error: integrated.1,
        nodes,
        end,
        evaluations,
        thermal: kt,
        bar_work: bar.map(|b| b.0),
        bar_error: bar.map(|b| b.1),
        mbar_work: mbar_estimate.map(|m| m.0),
        mbar_error: mbar_estimate.map(|m| m.1),
        overlap: overlap_matrix,
        warnings,
    })
}

/// Analytic-surface compatibility wrapper for [`reversible_work`].
#[deprecated(
    since = "0.5.0",
    note = "construct AnalyticSurfaceSampler and call reversible_work"
)]
pub fn reversible_work_on_surface<S: PointSurface>(
    surface: &S,
    planes: &[Hyperplane],
    start: Option<ArrayView1<f64>>,
    config: &ReversibleWorkConfig,
) -> Result<ReversibleWork, SaddleError> {
    let mut sampler = AnalyticSurfaceSampler::new(surface, config.sampler.clone());
    if let Some(start) = start {
        sampler = sampler.with_start(start.to_owned());
    }
    reversible_work(&mut sampler, planes, config)
}

/// Analytic-surface compatibility wrapper for [`reversible_work_along`].
#[deprecated(
    since = "0.5.0",
    note = "construct AnalyticSurfaceSampler and call reversible_work_along"
)]
pub fn reversible_work_along_on_surface<S: PointSurface>(
    surface: &S,
    path: &PlanePath,
    start: Option<ArrayView1<f64>>,
    config: &ReversibleWorkConfig,
) -> Result<ReversibleWork, SaddleError> {
    let mut sampler = AnalyticSurfaceSampler::new(surface, config.sampler.clone());
    if let Some(start) = start {
        sampler = sampler.with_start(start.to_owned());
    }
    reversible_work_along(&mut sampler, path, config)
}

/// `u_k = V(Φ_k Φ_i⁻¹ q) / kT - ln |det Φ_k|` for every configuration `q`
/// of every node `i` and every node `k`, where `Φ_k` is the chain of maps
/// from the first plane to plane `k`.
fn reduced_potentials<E: HyperplaneSampler>(
    sampler: &mut E,
    kept: &[Kept],
    maps: &[PlaneMap],
    kt: f64,
    evaluations: &mut usize,
) -> Result<(Array2<f64>, Vec<usize>), SaddleError> {
    let states = kept.len();
    let mut log_jacobian = vec![0.0; states];
    for k in 1..states {
        log_jacobian[k] = log_jacobian[k - 1] + maps[k - 1].log_jacobian();
    }
    let counts: Vec<usize> = kept.iter().map(|node| node.evaluation.len()).collect();
    let mut u = Array2::zeros((counts.iter().sum(), states));
    let mut row = 0;
    for (i, node) in kept.iter().enumerate() {
        for (j, q) in node.evaluation.iter().enumerate() {
            u[(row, i)] = node.energies[j] / kt - log_jacobian[i];
            let mut image = q.clone();
            for k in i + 1..states {
                image = maps[k - 1].forward(&image);
                let e = if k == i + 1 {
                    node.up[j]
                } else {
                    energy_or_infinity(sampler, &image, evaluations)?
                };
                u[(row, k)] = e / kt - log_jacobian[k];
            }
            let mut image = q.clone();
            for k in (0..i).rev() {
                image = maps[k].backward(&image);
                let e = if k + 1 == i {
                    node.down[j]
                } else {
                    energy_or_infinity(sampler, &image, evaluations)?
                };
                u[(row, k)] = e / kt - log_jacobian[k];
            }
            row += 1;
        }
    }
    Ok((u, counts))
}

fn energy<E: HyperplaneSampler>(
    sampler: &mut E,
    q: &Array1<f64>,
    evaluations: &mut usize,
) -> Result<f64, SaddleError> {
    let value = sampler.energy(q.view())?.ok_or_else(|| {
        SaddleError::Invalid("BAR and MBAR need sampler energy evaluations".into())
    })?;
    *evaluations += 1;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(SaddleError::NonFinite("energy of a mapped configuration"))
    }
}

/// The energy of a configuration carried onto a distant plane, where an
/// unphysical one has no weight.
fn energy_or_infinity<E: HyperplaneSampler>(
    sampler: &mut E,
    q: &Array1<f64>,
    evaluations: &mut usize,
) -> Result<f64, SaddleError> {
    *evaluations += 1;
    Ok(match sampler.energy(q.view()) {
        Ok(Some(value)) if value.is_finite() => value,
        _ => f64::INFINITY,
    })
}

/// Bennett's acceptance ratio: `ΔA / kT` from forward works `w_F` on
/// samples of `a` and reverse works `w_R` on samples of `b`, in units of
/// `kT`, with its asymptotic standard error (Bennett's Eq. 10).
fn bennett(forward: &[f64], reverse: &[f64]) -> Result<(f64, f64), SaddleError> {
    if forward.is_empty() || reverse.is_empty() {
        return Err(SaddleError::Invalid(
            "BAR needs retained configurations on both planes".into(),
        ));
    }
    let (nf, nr) = (forward.len() as f64, reverse.len() as f64);
    let m = (nf / nr).ln();
    let fermi = |x: f64| 1.0 / (1.0 + x.exp());
    let balance = |df: f64| -> f64 {
        let left: f64 = forward.iter().map(|w| fermi(m + w - df)).sum();
        let right: f64 = reverse.iter().map(|w| fermi(-m + w + df)).sum();
        left - right
    };
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let (mut low, mut high) = (-mean(reverse) - 1.0, mean(forward) + 1.0);
    if low > high {
        std::mem::swap(&mut low, &mut high);
    }
    let mut width = 1.0;
    while balance(low) > 0.0 {
        low -= width;
        width *= 2.0;
        if width > 1.0e6 {
            return Err(SaddleError::Solver("BAR did not bracket".into()));
        }
    }
    width = 1.0;
    while balance(high) < 0.0 {
        high += width;
        width *= 2.0;
        if width > 1.0e6 {
            return Err(SaddleError::Solver("BAR did not bracket".into()));
        }
    }
    for _ in 0..200 {
        let middle = 0.5 * (low + high);
        if balance(middle) > 0.0 {
            high = middle;
        } else {
            low = middle;
        }
        if high - low < 1.0e-12 * (1.0 + middle.abs()) {
            break;
        }
    }
    let df = 0.5 * (low + high);
    let spread = |values: Vec<f64>| {
        let n = values.len() as f64;
        let first = values.iter().sum::<f64>() / n;
        let second = values.iter().map(|f| f * f).sum::<f64>() / n;
        (second / (first * first) - 1.0) / n
    };
    let variance = spread(forward.iter().map(|w| fermi(m + w - df)).collect())
        + spread(reverse.iter().map(|w| fermi(-m + w + df)).collect());
    let error = if variance.is_finite() {
        variance.max(0.0).sqrt()
    } else {
        f64::INFINITY
    };
    Ok((df, error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ohtst::transport::GaussianTransport;
    use ndarray::array;

    /// `V = ½ qᵀ K q`, where `A(p, ν) = (kT/2) ln(νᵀK⁻¹ν) + (ν·p)²/(2 νᵀK⁻¹ν)`.
    struct Bowl;

    const K: [f64; 3] = [1.0, 2.0, 0.5];

    impl PointSurface for Bowl {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let g = array![K[0] * q[0], K[1] * q[1], K[2] * q[2]];
            Ok((0.5 * q.dot(&g), g))
        }
    }

    fn exact(plane: &Hyperplane, kt: f64) -> f64 {
        let nu = plane.normal();
        let s: f64 = (0..3).map(|i| nu[i] * nu[i] / K[i]).sum();
        let a = nu.dot(&plane.point());
        0.5 * kt * s.ln() + a * a / (2.0 * s)
    }

    fn bowl_sampler(kt: f64) -> LangevinConfig {
        LangevinConfig::default()
            .with_thermal(kt)
            .with_time_step(0.4)
            .with_burn_in(200)
            .with_stride(4)
            .with_records(2000)
    }

    #[test]
    fn work_matches_the_closed_form_on_a_bent_path() {
        let kt = 0.3;
        let planes = [
            Hyperplane::new(array![-0.6, 0.1, 0.0], array![1.0, 0.0, 0.0]).unwrap(),
            Hyperplane::new(array![-0.1, 0.3, 0.2], array![1.0, 0.6, 0.2]).unwrap(),
            Hyperplane::new(array![0.3, -0.1, 0.4], array![0.6, 0.2, 1.0]).unwrap(),
        ];
        let config = ReversibleWorkConfig::default().with_sampler(bowl_sampler(kt));
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, config.sampler.clone());
        let result = reversible_work(&mut sampler, &planes, &config).unwrap();
        let expected = exact(&planes[2], kt) - exact(&planes[0], kt);
        assert!(
            (result.work - expected).abs() < 4.0 * result.error + 1.0e-4,
            "work {} ± {}, closed form {expected}",
            result.work,
            result.error
        );
        assert_eq!(result.work, result.integrated_work);
        assert_eq!(result.nodes.len(), 2 * config.nodes - 1);
        let middle = &result.nodes[config.nodes - 1];
        let partial = exact(&planes[1], kt) - exact(&planes[0], kt);
        assert!((middle.work - partial).abs() < 0.01);
        assert_eq!(result.end.plane, planes[2]);
        let (bar, bar_error) = (result.bar_work.unwrap(), result.bar_error.unwrap());
        assert!(
            (bar - expected).abs() < 4.0 * bar_error,
            "BAR {bar} ± {bar_error}, closed form {expected}"
        );
        assert!(bar_error < 0.02, "{bar_error}");
        assert!((middle.bar_work.unwrap() - partial).abs() < 0.02);
        assert!(result.nodes[1..].iter().all(|n| n.overlap.unwrap() > 0.1));
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        assert!(result.mbar_work.is_none() && result.overlap.is_none());
    }

    /// PAFI planes along a curve through the bowl, clamped to its end
    /// tangents, against the closed form of `A` at every node; MBAR across
    /// the nodes and BAR with the Gaussian map agree with it.
    #[test]
    fn a_curve_of_planes_with_gaussian_maps_and_mbar() {
        let kt = 0.3;
        let curve = |t: f64| {
            array![
                -0.6 + 0.9 * t,
                0.1 + 0.2 * (std::f64::consts::PI * t).sin(),
                0.4 * t
            ]
        };
        let points: Vec<Array1<f64>> = (0..4).map(|i| curve(i as f64 / 3.0)).collect();
        let slope = |t: f64| {
            array![
                0.9,
                0.2 * std::f64::consts::PI * (std::f64::consts::PI * t).cos(),
                0.4
            ]
        };
        let (start_normal, end_normal) = (slope(0.0), slope(1.0));
        let path =
            PlanePath::curve(&points, Some(start_normal.view()), Some(end_normal.view())).unwrap();
        let (first, last) = (path.start().unwrap(), path.end().unwrap());
        let expected = exact(&last, kt) - exact(&first, kt);
        let config = ReversibleWorkConfig::default()
            .with_sampler(bowl_sampler(kt))
            .with_nodes(3)
            .with_mbar(true)
            .with_estimate(WorkEstimate::Mbar)
            .with_transport(GaussianTransport::default().into());
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, config.sampler.clone());
        let result = reversible_work_along(&mut sampler, &path, &config).unwrap();
        assert_eq!(result.nodes.len(), 7);
        assert_eq!(result.estimate, WorkEstimate::Mbar);
        let checks = [
            ("TI", result.integrated_work, result.integrated_error),
            ("BAR", result.bar_work.unwrap(), result.bar_error.unwrap()),
            (
                "MBAR",
                result.mbar_work.unwrap(),
                result.mbar_error.unwrap(),
            ),
        ];
        for (name, value, error) in checks {
            assert!(
                (value - expected).abs() < 4.0 * error + 1.0e-4,
                "{name} {value} ± {error}, closed form {expected}"
            );
        }
        assert_eq!(result.work, result.mbar_work.unwrap());
        // A quadratic surface leaves the Gaussian map nearly constant
        // works, so the neighbours overlap almost fully.
        assert!(result.nodes[1..].iter().all(|n| n.overlap.unwrap() > 0.4));
        let overlap = result.overlap.as_ref().unwrap();
        assert_eq!(overlap.dim(), (7, 7));
        assert!(
            overlap
                .rows()
                .into_iter()
                .all(|r| (r.sum() - 1.0).abs() < 1e-9)
        );
        for node in &result.nodes {
            let partial = exact(&node.plane, kt) - exact(&first, kt);
            assert!((node.mbar_work.unwrap() - partial).abs() < 0.02);
        }
    }

    /// The rigid map takes the pivot of one plane to the pivot of the
    /// next. With the second pivot 1.41 from its plane's mean, eight
    /// in-plane standard deviations, the pair overlaps poorly and is named
    /// in the warnings; the Gaussian map matches the means and bridges it.
    #[test]
    fn poor_overlap_is_reported_and_the_gaussian_map_bridges_it() {
        let kt = 0.05;
        let planes = [
            Hyperplane::new(array![-0.6, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap(),
            Hyperplane::new(array![1.0, -1.0, 0.0], array![1.0, 1.0, 0.0]).unwrap(),
        ];
        let config = ReversibleWorkConfig::default()
            .with_sampler(bowl_sampler(kt).with_records(800))
            .with_nodes(2)
            .with_simpson(false)
            .with_bar_samples(64);
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, config.sampler.clone());
        let rigid = reversible_work(&mut sampler, &planes, &config).unwrap();
        let overlap = rigid.nodes[1].overlap.unwrap();
        assert!(overlap < 0.03, "{overlap}");
        assert_eq!(rigid.warnings.len(), 1, "{:?}", rigid.warnings);
        assert!(rigid.warnings[0].contains("nodes 0 and 1"));

        let gaussian = config.with_transport(GaussianTransport::default().into());
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, gaussian.sampler.clone());
        let mapped = reversible_work(&mut sampler, &planes, &gaussian).unwrap();
        assert!(
            mapped.nodes[1].overlap.unwrap() > 0.3,
            "{:?}",
            mapped.nodes[1].overlap
        );
        assert!(mapped.warnings.is_empty(), "{:?}", mapped.warnings);
        let expected = exact(&planes[1], kt) - exact(&planes[0], kt);
        let (bar, bar_error) = (mapped.bar_work.unwrap(), mapped.bar_error.unwrap());
        assert!(
            (bar - expected).abs() < 4.0 * bar_error,
            "BAR {bar} ± {bar_error}, closed form {expected}"
        );
    }

    /// Gaussian works that obey Crooks' relation, `w_F ~ N(Δ + σ²/2, σ²)`
    /// and `w_R ~ N(-Δ + σ²/2, σ²)`, give back `Δ`.
    #[test]
    fn bar_recovers_a_known_difference() {
        let mut rng = crate::stats::Rng::new(4);
        let (delta, sigma) = (1.7, 1.2);
        let draw = |rng: &mut crate::stats::Rng, centre: f64, n: usize| -> Vec<f64> {
            (0..n).map(|_| centre + sigma * rng.normal()).collect()
        };
        let forward = draw(&mut rng, delta + 0.5 * sigma * sigma, 4000);
        let reverse = draw(&mut rng, -delta + 0.5 * sigma * sigma, 3000);
        let (estimate, error) = bennett(&forward, &reverse).unwrap();
        assert!(
            (estimate - delta).abs() < 4.0 * error,
            "{estimate} ± {error}"
        );
        assert!(error < 0.05 && error > 0.0);
        assert!(bennett(&forward, &[]).is_err());
    }

    #[test]
    fn opposite_normals_and_bad_settings_are_rejected() {
        let a = Hyperplane::new(array![0.0, 0.0, 0.0], array![1.0, 0.0, 0.0]).unwrap();
        let b = Hyperplane::new(array![0.1, 0.0, 0.0], array![-1.0, 0.0, 0.0]).unwrap();
        let config = ReversibleWorkConfig::default();
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, config.sampler.clone());
        assert!(reversible_work(&mut sampler, &[a.clone(), b], &config).is_err());
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, config.sampler.clone());
        assert!(reversible_work(&mut sampler, std::slice::from_ref(&a), &config).is_err());
        let c = Hyperplane::new(array![0.1, 0.0, 0.0], array![1.0, 0.1, 0.0]).unwrap();
        let pair = [a, c];
        let mbar_off = config.clone().with_estimate(WorkEstimate::Mbar);
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, mbar_off.sampler.clone());
        assert!(reversible_work(&mut sampler, &pair, &mbar_off).is_err());
        let no_samples = config.clone().with_bar_samples(0).with_mbar(true);
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, no_samples.sampler.clone());
        assert!(reversible_work(&mut sampler, &pair, &no_samples).is_err());
        let overlap = config.with_min_overlap(1.0);
        let mut sampler = AnalyticSurfaceSampler::new(&Bowl, overlap.sampler.clone());
        assert!(reversible_work(&mut sampler, &pair, &overlap).is_err());
    }
}
