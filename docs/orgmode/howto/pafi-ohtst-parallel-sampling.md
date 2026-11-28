PAFI's planes are independent of each other, so `pafi_profile` already samples every plane in parallel on rayon's global thread pool: nothing to opt into. OH-TST's single calls, `basin_density`, `sample_plane` and `reversible_work_along`, each sample one Langevin chain where every node starts from the one before, so they stay sequential inside the call. Independent OH-TST calls are a different story: nothing ties one call to another, so running several of them, over several temperatures or several reaction steps, parallelizes at the host's level with `rayon::par_iter` or `rayon::join`, since `PointSurface` is `Sync`.


# PAFI across a custom thread pool

A `rayon::ThreadPoolBuilder` with a chosen `num_threads` and `pool.install` puts `pafi_profile` on that pool instead of the global one, which is how this how-to compares one worker against every core without changing the profile itself:

```rust
fn pafi_over_planes() -> Result<(), SaddleError> {
    let path = ArcPath::from_positions(&[array![-1.0, 0.0], array![0.0, 0.0], array![1.0, 0.0]])?;
    let arcs: Vec<f64> = {
        let (first, last) = path.bounds();
        (0..33)
            .map(|k| first + (last - first) * k as f64 / 32.0)
            .collect()
    };
    let config = PafiConfig::default()
        .with_thermal(0.1)
        .with_samples(6_000)
        .with_blocks(20)
        .with_seed(5);

    let threads: usize = std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1);
    for workers in [1, threads] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| SaddleError::Solver(e.to_string()))?;
        let start = Instant::now();
        let (profile, _) = pool.install(|| pafi_profile(&CurvedWell, &path, &arcs, &config))?;
        println!(
            "PAFI, {workers} worker(s): barrier {:.5} +/- {:.5} in {:.3} s over {} planes",
            profile.barrier,
            profile.barrier_error,
            start.elapsed().as_secs_f64(),
            arcs.len()
        );
    }
    Ok(())
```


# OH-TST across temperatures, with `par_iter`

`basin_density` at six different temperatures are six unrelated calls. A plain `.iter().map(...)` runs them one after another; swapping it for `.par_iter().map(...)` runs them concurrently, with no other change to the body of the closure:

```rust
fn ohtst_over_temperatures() -> Result<(), SaddleError> {
    let reactant = array![-1.0, 0.0];
    let product = array![1.0, 0.0];
    let temperatures = [0.05, 0.08, 0.1, 0.13, 0.16, 0.2];

    let sequential_start = Instant::now();
    let sequential: Vec<f64> = temperatures
        .iter()
        .map(|&kt| {
            let sampler = LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.02)
                .with_friction(1.0)
                .with_burn_in(500)
                .with_records(6000)
                .with_seed(7);
            basin_density(
                &CurvedWell,
                reactant.view(),
                product.view(),
                &BasinConfig::default().with_sampler(sampler),
            )
            .map(|b| b.density)
        })
        .collect::<Result<_, _>>()?;
    let sequential_time = sequential_start.elapsed().as_secs_f64();

    let parallel_start = Instant::now();
    let parallel: Vec<f64> = temperatures
        .par_iter()
        .map(|&kt| {
            let sampler = LangevinConfig::default()
                .with_thermal(kt)
                .with_time_step(0.02)
                .with_friction(1.0)
                .with_burn_in(500)
                .with_records(6000)
                .with_seed(7);
            basin_density(
                &CurvedWell,
                reactant.view(),
                product.view(),
                &BasinConfig::default().with_sampler(sampler),
            )
            .map(|b| b.density)
        })
        .collect::<Result<_, _>>()?;
    let parallel_time = parallel_start.elapsed().as_secs_f64();

    println!(
        "OH-TST, {} temperatures sequentially: {:.3} s",
        temperatures.len(),
        sequential_time
    );
    println!(
        "OH-TST, {} temperatures with par_iter: {:.3} s",
        temperatures.len(),
        parallel_time
    );
    println!(
        "Z_H/Z_R by temperature, both ways agree: {}",
        sequential == parallel
    );
    for (kt, density) in temperatures.iter().zip(&sequential) {
        println!("  kT = {kt}: Z_H/Z_R = {density:.4}");
    }
    Ok(())
```

Comparing `sequential == parallel` on the resulting `Vec<f64>` works here because every call uses a fixed seed: the two runs draw bit-identical Langevin chains regardless of how many threads ran them, so their results are exactly equal, not just close.


# Run it

```console
$ cargo run --release --features ohtst --example howto_parallel
```

    PAFI, 1 worker(s): barrier 1.03841 +/- 0.00179 in 2.300 s over 33 planes
    PAFI, 8 worker(s): barrier 1.03841 +/- 0.00179 in 0.709 s over 33 planes
    OH-TST, 6 temperatures sequentially: 1.607 s
    OH-TST, 6 temperatures with par_iter: 0.541 s
    Z_H/Z_R by temperature, both ways agree: true
      kT = 0.05: Z_H/Z_R = 4.9987
      kT = 0.08: Z_H/Z_R = 3.9286
      kT = 0.1: Z_H/Z_R = 3.5002
      kT = 0.13: Z_H/Z_R = 3.0550
      kT = 0.16: Z_H/Z_R = 2.7350
      kT = 0.2: Z_H/Z_R = 2.4232

Both PAFI runs report the same barrier and error to the last printed digit, since the pool only changes how the 33 planes' independent chains are scheduled, not the chains themselves. This host had 8 logical cores available but was busy with other builds, so the wall times are far from a clean 8x: 2.300 s falls to 0.709 s, a bit over 3x, and the OH-TST side falls from 1.607 s to 0.541 s, a bit under 3x. Read the ratio, not the seconds, and expect a larger one on an idle machine. `Z_H/Z_R` falls from 5.0 at the coldest temperature to 2.4 at the warmest, since a warmer basin spreads further from the minimum before the plane integral settles, both runs agreeing exactly at every temperature. Each basin integral records 6000 samples: `basin_density` refuses a plane whose effective sample count is under 20, and 2000 records fell short of that at one of these temperatures.


# Next
