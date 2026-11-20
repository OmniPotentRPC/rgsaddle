//! How-to: parallel sampling for PAFI and OH-TST.
//!
//! [`pafi_profile`] samples every plane in parallel on rayon's global
//! pool; nothing to opt into. OH-TST's own calls, [`basin_density`],
//! [`sample_plane`] and [`reversible_work_along`], sample one plane's
//! Langevin chain at a time, because each node starts from the one
//! before. Independent OH-TST calls, over several temperatures or
//! several hops, parallelize at the host's level with `rayon::join` or
//! `par_iter`, since [`PointSurface`] is `Sync`.
//!
//! Run:
//!
//! ```console
//! $ cargo run --release --features ohtst --example howto_parallel
//! ```

use std::time::Instant;

use ndarray::{Array1, ArrayView1, array};
use rayon::prelude::*;
use rgsaddle::{
    ArcPath, BasinConfig, LangevinConfig, PafiConfig, PointSurface, SaddleError, basin_density,
    pafi_profile,
};

/// Same surface as `examples/pafi_tutorial.rs`.
struct CurvedWell;

impl PointSurface for CurvedWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let k = 2.5 + 1.5 * x;
        let well = x * x - 1.0;
        let energy = well * well + 0.5 * k * y * y;
        let dvdx = 4.0 * x * well + 0.75 * y * y;
        let dvdy = k * y;
        Ok((energy, array![dvdx, dvdy]))
    }
}

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
        .with_seed(5)
        .with_fix_com(false);

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
}

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
}

fn main() -> Result<(), SaddleError> {
    pafi_over_planes()?;
    ohtst_over_temperatures()
}
