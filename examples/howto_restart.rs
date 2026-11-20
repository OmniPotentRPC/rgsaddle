//! How-to: restart a PAFI or OH-TST sampling run instead of starting over.
//!
//! PAFI: [`PafiConfig::target_error`] and [`PafiConfig::max_rounds`] run
//! extra sampling rounds automatically, sampling each plane from its path
//! point with a round-derived seed and inverse-variance combining with
//! earlier rounds, until the barrier error meets the target or the round
//! budget runs out. No manual bookkeeping.
//!
//! OH-TST: [`sample_plane`]'s `start` argument and [`PlaneSample::last`]
//! let the host do the same by hand: take the last configuration of one
//! call and pass it as the `start` of the next, with
//! [`LangevinConfig::burn_in`] at zero, to extend a chain without paying
//! for thermalization twice.
//!
//! Run:
//!
//! ```console
//! $ cargo run --release --features ohtst --example howto_restart
//! ```

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    ArcPath, Hyperplane, LangevinConfig, PafiConfig, PointSurface, SaddleError, pafi_profile,
    sample_plane,
};

/// Same surface as `examples/pafi_tutorial.rs`: `V = (x^2 - 1)^2 + k(x)
/// y^2 / 2`, `k(x) = 2.5 + 1.5 x`. The transverse stiffness varies with
/// `x`, so the force along the path fluctuates with `y` and the block
/// error below is not degenerately zero.
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
        .with_seed(5)
        .with_fix_com(false);
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
}

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
}

fn main() -> Result<(), SaddleError> {
    pafi_continuation()?;
    ohtst_restart()
}
