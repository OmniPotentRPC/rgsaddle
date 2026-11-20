//! How-to: checking convergence of a PAFI or OH-TST run.
//!
//! Neither method is "run to completion": a sampler reports whether its
//! records behaved like independent draws, not whether the answer is
//! right. The host reads the diagnostics in the result and compares a
//! short run that has not settled against a longer one that has.
//!
//! PAFI: [`HyperplaneAverage::tau`] is the integrated autocorrelation
//! time, in records, and [`HyperplaneAverage::n_eff`] is
//! `samples / max(1, 2 tau)`; a block error trusted with `n_eff` under
//! about 20 is noisy. OH-TST: [`PlaneSample::tau`] and
//! [`PlaneSample::effective_samples`] are the same pair, and
//! [`PlaneSample::virial`] should sit within error of `free_dimensions *
//! kT` at equilibrium (the equipartition virial on an unconstrained
//! plane); a departure flags a time step that biases the sampling or a
//! chain that never equilibrated. [`ReversibleWork::warnings`] names
//! neighbouring nodes whose overlap falls under
//! [`ReversibleWorkConfig::min_overlap`], where BAR and MBAR run out of
//! shared support.
//!
//! Run:
//!
//! ```console
//! $ cargo run --release --features ohtst --example howto_convergence
//! ```

use ndarray::{Array1, ArrayView1, array};
use rgsaddle::{
    AnalyticSurfaceSampler, ArcPath, Hyperplane, LangevinConfig, PafiConfig, PlanePath,
    PointSurface, ReversibleWorkConfig, SaddleError, pafi_profile, reversible_work_along,
    sample_plane,
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
}

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
    println!("deviation from equipartition: {deviation:.2} standard errors");

    // The same span, the reactant minimum to the saddle, resolved by too
    // few nodes against enough of them. V = (x^2 - 1)^2 has zero slope at
    // both x = -1 and x = 0, so a trapezoid between just the two
    // endpoints badly underestimates the barrier in between; the
    // integration needs enough interior nodes to see the curve bend.
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
    }
    Ok(())
}

fn main() -> Result<(), SaddleError> {
    pafi_tau_and_n_eff()?;
    ohtst_virial_and_overlap()
}
