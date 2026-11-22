//! OH-TST on a separable double well, end to end.
//!
//! `V = (x² − 1)² / 4 + y² / 2` has minima at `(±1, 0)` and its saddle at
//! the origin. The coordinates are already mass-weighted. The plane through
//! the saddle with normal `(1, 0)` is sampled, climbed, and used as one end
//! of a reversible-work leg. A two-state MBAR problem, a short transmission
//! run and the rate from those parts follow.

mod example {
    use ndarray::{Array1, ArrayView1, array};
    use rgsaddle::{
        AnalyticSurfaceDynamics, AnalyticSurfaceSampler, Hyperplane, LangevinConfig, MbarConfig,
        PlaneOptimizer, PlaneOptimizerConfig, PointSurface, ReversibleWorkConfig, SaddleError,
        TransmissionConfig, mbar_free_energies, rate_from_parts, reversible_work, sample_plane,
        transmission_coefficient,
    };

    struct DoubleWell;

    impl PointSurface for DoubleWell {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (x, y) = (q[0], q[1]);
            let well = x * x - 1.0;
            Ok((0.25 * well * well + 0.5 * y * y, array![x * well, y]))
        }
    }

    fn sampler() -> LangevinConfig {
        LangevinConfig::default()
            .with_thermal(0.25)
            .with_time_step(0.05)
            .with_friction(1.0)
            .with_burn_in(4)
            .with_stride(1)
            .with_records(8)
            .with_blocks(2)
            .with_retain(2)
            .with_seed(3)
    }

    pub(super) fn run() -> Result<(), SaddleError> {
        let surface = DoubleWell;
        let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0])?;
        let sample = sample_plane(&surface, &plane, None, &sampler())?;
        println!(
            "sample: normal force {:.4} ± {:.4}, virial {:.4}, acceptance {:.2}",
            sample.normal_force, sample.normal_force_error, sample.virial, sample.acceptance
        );

        let optimizer = PlaneOptimizerConfig::default()
            .with_sampler(sampler())
            .with_norm_test(0.0)
            .with_max_records(8)
            .with_max_iterations(4)
            .with_patience(1)
            .with_max_translation(0.2)
            .with_rotate(false);
        let start = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0])?;
        let optimum = PlaneOptimizer::new(&start, optimizer)?.run(&surface)?;
        println!(
            "optimum: point ({:.4}, {:.4}), iterations {}",
            optimum.plane.point()[0],
            optimum.plane.point()[1],
            optimum.iterations
        );

        let other = Hyperplane::new(array![0.3, 0.0], array![1.0, 0.0])?;
        let work_config = ReversibleWorkConfig::default()
            .with_sampler(sampler())
            .with_nodes(3)
            .with_bar_samples(0);
        let mut engine = AnalyticSurfaceSampler::new(&surface, work_config.sampler.clone());
        let work = reversible_work(&mut engine, &[plane.clone(), other], &work_config)?;
        println!(
            "work: {:.4} ± {:.4} over {} evaluations",
            work.work, work.error, work.evaluations
        );

        let reduced = array![[0.0, 4.0], [0.2, 3.5], [4.0, 0.0], [3.6, 0.1]];
        let mbar = mbar_free_energies(&reduced, &[2, 2], &MbarConfig::default())?;
        println!(
            "mbar: f = ({:.4}, {:.4})",
            mbar.free_energies[0], mbar.free_energies[1]
        );

        let starts = vec![array![0.0, 0.0], array![0.0, 0.2]];
        let transmission = TransmissionConfig::default()
            .with_thermal(0.25)
            .with_time_step(0.05)
            .with_friction(0.0)
            .with_steps(20)
            .with_check_every(5)
            .with_shots(4)
            .with_blocks(2)
            .with_seed(5);
        let mut dynamics = AnalyticSurfaceDynamics::new(&surface);
        let kappa = transmission_coefficient(&mut dynamics, &plane, &starts, &transmission)?;
        println!("kappa: {:.4} ± {:.4}", kappa.kappa, kappa.error);

        let rate = rate_from_parts(0.25, 1.0, 0.0, work.work, work.error)?;
        println!("rate: {:.6} ± {:.6}", rate.rate, rate.error);
        Ok(())
    }
}

fn main() -> Result<(), rgsaddle::SaddleError> {
    example::run()
}
