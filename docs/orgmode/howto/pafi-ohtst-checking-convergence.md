Neither method runs to completion: a sampler reports whether its own records behaved like independent draws, not whether the answer is physically right. This how-to reads the diagnostics each result carries. On PAFI's side it compares a short run that the sampler refuses against a longer one that it accepts, and on OH-TST's a quadrature that does not have enough nodes against one that does.


# PAFI: `tau` and `n_eff`

`HyperplaneAverage::tau` is the integrated autocorrelation time, in records; `HyperplaneAverage::n_eff` is `(samples - discard) / max(1, 2 tau)`, the number of effectively independent draws the retained tail carries. A block standard error computed from a chain with `n_eff` in the single digits is not trustworthy, whatever number it prints, so `pafi_profile` does not print one: it returns `SaddleError::InsufficientEffectiveSamples` for a plane whose tail has fewer effective samples than `PafiConfig::blocks`, 5 by default. The example below sets `blocks` to 10:

```rust
fn pafi_tau_and_n_eff() -> Result<(), SaddleError> {
    let path = ArcPath::from_positions(&[array![-1.0, 0.0], array![0.0, 0.0], array![1.0, 0.0]])?;
    let arcs: Vec<f64> = {
        let (first, last) = path.bounds();
        (0..9)
            .map(|k| first + (last - first) * k as f64 / 8.0)
            .collect()
    };
    for (label, samples, stride) in [("under-sampled", 60, 1), ("settled", 6_000, 4)] {
        let config = PafiConfig::default()
            .with_thermal(0.1)
            .with_samples(samples)
            .with_stride(stride)
            .with_blocks(10)
            .with_seed(9);
        match pafi_profile(&CurvedWell, &path, &arcs, &config) {
            Ok((profile, planes)) => {
                let tau_max = planes.iter().fold(0.0_f64, |m, p| m.max(p.tau));
                let n_eff_min = planes.iter().fold(f64::INFINITY, |m, p| m.min(p.n_eff));
                println!(
                    "PAFI {label}: barrier {:.5} +/- {:.5}, worst tau {:.2}, worst n_eff {:.1}",
                    profile.barrier, profile.barrier_error, tau_max, n_eff_min
                );
            }
            Err(error @ SaddleError::InsufficientEffectiveSamples { .. }) => {
                println!("PAFI {label}: refused, {error}");
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
```

```console
$ cargo run --release --features ohtst --example howto_convergence
```

    PAFI under-sampled: refused, constrained ensemble has only 5.280 effective samples after 222 steps; at least 10.000 are required
    PAFI settled: barrier 0.97572 +/- 0.00268, worst tau 61.87, worst n_eff 48.5

The under-sampled run, 60 samples a step apart, is refused: its tail carries 5.28 effective samples behind a block error computed from 10 blocks, so the block-to-block spread would badly underestimate the true uncertainty, and the sampler says so instead of returning it. The settled run's worst `n_eff`, 48.5, is comfortably above the threshold, and its error bar can be read. A small error bar next to a low `n_eff` means the run needs to be longer; the sampler enforces that for a plane below `blocks`, 10 in this example, and leaves the judgement above that to the caller. It also refuses a tail when blocks at least `ceil(2 tau)` records wide leave fewer than two complete blocks, even if the first effective-sample check passed. (The barrier here is not the fully converged one from `tutorials/pafi`, 1.04418 at the same `kT`: this how-to uses 9 planes on a three-point path and far fewer samples than that tutorial's 49 planes and 30000, to keep two comparable runs fast, not to give the tutorial's own number.)


# OH-TST: the virial check and quadrature overlap

`PlaneSample::virial` is the in-plane virial $\langle (P \nabla V) \cdot (q - p) \rangle$, where $P$ projects onto the plane's free directions; at equilibrium on an unconstrained plane it should sit within error of `free_dimensions * kT` (equipartition), and a run that misses it by several standard errors has a time step too large or a chain that has not equilibrated:

```rust
fn ohtst_virial_and_overlap() -> Result<(), SaddleError> {
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0])?;
    let sampler = LangevinConfig::default()
        .with_thermal(0.1)
        .with_time_step(0.02)
        .with_friction(1.0)
        .with_burn_in(500)
        .with_records(4_000)
        .with_seed(13);
    let sample = sample_plane(&CurvedWell, &plane, None, &sampler)?;
    let expected_virial = sample.free_dimensions as f64 * sample.thermal;
    println!(
        "OH-TST virial: {:.5} ± {:.5} against free_dimensions * kT = {:.5} (tau {:.2}, effective samples {:.1})",
        sample.virial, sample.virial_error, expected_virial, sample.tau, sample.effective_samples
    );
    let deviation = (sample.virial - expected_virial).abs() / sample.virial_error;
```

    OH-TST virial: 0.10537 ± 0.00952 against free_dimensions * kT = 0.10000 (tau 6.08, effective samples 222.4)
    deviation from equipartition: 0.56 standard errors

0.56 standard errors from the expected virial is an unremarkable sampling fluctuation; a run reporting several standard errors off this check, instead, points at the time step or burn-in before the sampled free energy itself is trusted.

`ReversibleWorkConfig::min_overlap` and `ReversibleWork::warnings` are the other convergence signal: when neighbouring nodes along a `PlanePath` share too little configuration-space overlap, BAR and MBAR run out of the shared support their estimators need, and a node pair below the threshold is named in `warnings`. A separate, equally common failure needs no overlap warning at all: too few nodes to resolve the curve between them. `V = (x^2 - 1)^2` has zero slope at both the reactant, $x=-1$, and the saddle, $x=0$, so a two-node trapezoid, which only sees the slope at the two endpoints, badly underestimates everything the curve does between them:

```rust
let reactant_plane = Hyperplane::new(array![-1.0, 0.0], array![1.0, 0.0])?;
let path = PlanePath::legs(&[reactant_plane, plane.clone()])?;
for nodes in [2, 3, 9] {
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler.clone().with_records(1_000))
        .with_nodes(nodes)
        .with_simpson(nodes > 2)
        .with_bar_samples(128);
    let mut work_sampler = AnalyticSurfaceSampler::new(&CurvedWell, config.sampler.clone());
    let work = reversible_work_along(&mut work_sampler, &path, &config)?;
    let worst_overlap = work
        .nodes
        .iter()
        .filter_map(|n| n.overlap)
        .fold(f64::INFINITY, f64::min);
    println!(
        "reactant -> saddle, {nodes} node(s): integrated {:.5} ± {:.5}, worst neighbour overlap {:.3}, warnings: {:?}",
        work.integrated_work, work.integrated_error, worst_overlap, work.warnings
    );
```

    reactant -> saddle, 2 node(s): integrated 0.04503 ± 0.00606, worst neighbour overlap 0.473, warnings: []
    reactant -> saddle, 3 node(s): integrated 1.04012 ± 0.00376, worst neighbour overlap 0.490, warnings: []
    reactant -> saddle, 9 node(s): integrated 1.04531 ± 0.00283, worst neighbour overlap 0.487, warnings: []

Two nodes gives 0.04503, wildly short of the true barrier; three nodes, enough for Simpson's rule to see the curve bend once in the middle, already recovers 1.04012; nine nodes gets to 1.04531, close to the 1.04581 closed form `tutorials/oh-tst` checks its own reversible work against. None of the three runs prints an overlap warning: the worst overlaps are 0.473, 0.490 and 0.487, all above the configured threshold, since the plane is normal to a single, smooth direction here. A low node count is a quadrature error, not an overlap failure, and `warnings` being empty does not mean the integral converged; only comparing node counts against each other, the way this how-to does, or against a closed form when one exists, shows that.


# Next
