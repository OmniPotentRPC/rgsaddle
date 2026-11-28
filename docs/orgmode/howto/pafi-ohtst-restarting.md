PAFI and OH-TST restart differently. PAFI's sampling loop is inside the crate already: ask for a target error and it runs extra rounds on its own. OH-TST's `sample_plane` is a single pass with no loop of its own, so extending a chain is the host's job: carry the last configuration forward and skip the burn-in the second time.


# PAFI: `target_error` and `max_rounds`

`PafiConfig::with_target_error` sets a barrier error to stop at; `with_max_rounds` bounds how many rounds of extra sampling it is allowed to spend trying to reach it. Each extra round samples every plane again from its path point, with a fresh seed derived deterministically from the base seed, plane index and round number, and inverse-variance combines its gradient with the earlier rounds; the profile's reported sample count is the running total across every round so far, not just the last one:

```rust
fn pafi_continuation() -> Result<(), SaddleError> {
    let path = ArcPath::from_positions(&[array![-1.0, 0.0], array![0.0, 0.0], array![1.0, 0.0]])?;
    let arcs: Vec<f64> = {
        let (first, last) = path.bounds();
        (0..9)
            .map(|k| first + (last - first) * k as f64 / 8.0)
            .collect()
    };
    // One round of 800 samples per plane is loose: the target asks for a
    // barrier error under 0.004, tighter than one round alone reaches, so
    // extra rounds run on top of the first.
    let config = PafiConfig::default()
        .with_thermal(0.1)
        .with_samples(800)
        .with_blocks(10)
        .with_target_error(0.004)
        .with_max_rounds(6)
        .with_seed(5);
    let (profile, _) = pafi_profile(&CurvedWell, &path, &arcs, &config)?;
    let one_round = config.samples * arcs.len();
    println!(
        "PAFI: barrier {:.5} +/- {:.5} from {} samples ({} planes x {} in the first round alone)",
        profile.barrier,
        profile.barrier_error,
        profile.samples,
        arcs.len(),
        config.samples
    );
    println!(
        "ran past the first round: {} ({} total samples against {} for one round)",
        profile.samples > one_round,
        profile.samples,
        one_round
    );
    Ok(())
```

One round alone, at `samples = 800`, has a block error loose enough that 0.004 is out of reach; `with_max_rounds(6)` lets it keep sampling past the first round until it gets there or runs out of rounds.

```console
$ cargo run --release --features ohtst --example howto_restart
```

    PAFI: barrier 0.96397 +/- 0.00243 from 21600 samples (9 planes x 800 in the first round alone)
    ran past the first round: true (21600 total samples against 7200 for one round)

21600 recorded samples against 7200 for one round (9 planes times 800): an extra round ran on every plane at once, scaled up from the first round's samples by how far the barrier error still was from the target, and that was enough to bring the error down to 0.00243, under the 0.004 target, so the loop stopped there instead of spending the rest of its 6-round budget. A target the extra rounds cannot reach inside the round budget runs every round and reports whatever error it ends at; nothing here distinguishes that case in the output from stopping early on success, so a caller checking convergence should compare `profile.barrier_error` against the target it asked for, not assume success from the run completing.


# OH-TST: carry `PlaneSample::last` into the next start

`sample_plane` takes an optional `start` configuration on the plane; leaving it `None`, as every other how-to and tutorial on this page does, starts from the plane's own point. Passing `Some(previous.last.view())` instead continues the chain exactly where the last call's Langevin dynamics stopped, so a second call with `with_burn_in(0)` gets useful samples immediately instead of re-equilibrating:

```rust
fn ohtst_restart() -> Result<(), SaddleError> {
    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0])?;
    let sampler = LangevinConfig::default()
        .with_thermal(0.1)
        .with_time_step(0.05)
        .with_friction(1.0)
        .with_stride(3)
        .with_seed(31);

    let first = sample_plane(
        &CurvedWell,
        &plane,
        None,
        &sampler.clone().with_burn_in(500).with_records(500),
    )?;
    println!(
        "first leg: {:.5} ± {:.5} from {} records, last point ({:.4}, {:.4})",
        first.normal_force, first.normal_force_error, first.records, first.last[0], first.last[1]
    );

    // Continue from where the first leg stopped: the configuration is
    // already on the plane and already equilibrated, so burn_in can be
    // zero instead of another 500 steps.
    let second = sample_plane(
        &CurvedWell,
        &plane,
        Some(first.last.view()),
        &sampler.with_burn_in(0).with_records(500).with_seed(32),
    )?;
    println!(
        "continued leg: {:.5} ± {:.5} from {} records starting at step 0 of burn-in",
        second.normal_force, second.normal_force_error, second.records
    );

    let combined = (first.normal_force + second.normal_force) / 2.0;
    let combined_error =
        0.5 * (first.normal_force_error.powi(2) + second.normal_force_error.powi(2)).sqrt();
    println!("combined (equal weight): {combined:.5} ± {combined_error:.5}");
    Ok(())
```

    first leg: -0.03375 ± 0.00408 from 494 records, last point (0.0000, -0.0175)
    continued leg: -0.02608 ± 0.00380 from 494 records starting at step 0 of burn-in
    combined (equal weight): -0.02991 ± 0.00279

Both legs sample the same plane at the same temperature, so an equal-weight average of the two means is a valid combined estimate, and its error, 0.00279, is smaller than either leg's own (0.00408 and 0.00380): the two legs are independent draws from the same stationary distribution (different seeds, 31 and 32), and combining independent estimates of the same quantity always tightens the error, exactly the point of extending a chain rather than discarding the first one's samples. The two legs' individual means, -0.03375 and -0.02608, differ by rather more than either one's own error bar; that is ordinary sampling noise at only 494 records each, not evidence the chain has not equilibrated, since the second leg's burn-in is zero by construction and its mean is not systematically offset from the first's in one direction.


# Next
