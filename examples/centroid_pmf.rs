//! Centroid-hyperplane potential of mean force on `(x² − 1)²`.
//!
//! The barrier is the static centroid quantum transition state theory
//! free energy of the ring. It is not a ring-polymer molecular dynamics
//! rate. The path runs from the relaxed reactant through the min-mode
//! saddle to the relaxed product, with two samples on each curvature
//! length of the well. The shared step is the smaller image-local
//! [`RingPolymer::time_step`] bound for the requested bead counts. The
//! production length provides at least six effective samples per block
//! layout for the deterministic seed used here.
//!
//! ```text
//! cargo run --example centroid_pmf
//! ```

use ndarray::{Array1, ArrayView1};
use rgmin::{Control, Method, Oracle, Solver, StopNorm};
use rgsaddle::{
    MinModeConfig, MinModeSession, MinModeStatus, PafiConfig, PafiIntegrator, PointSurface,
    RelaxConfig, RingPolymer, RingSettings, SaddleError, bead_study,
};

struct DoubleWell;

fn quartic(x: f64) -> (f64, f64) {
    let square = x * x;
    ((square - 1.0).powi(2), 4.0 * x * (square - 1.0))
}

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, gradient) = quartic(x[0]);
        Ok((energy, Array1::from_elem(1, gradient)))
    }

    fn hessian_vector(
        &self,
        x: ArrayView1<f64>,
        v: ArrayView1<f64>,
    ) -> Result<Option<Array1<f64>>, SaddleError> {
        let curvature = 12.0 * x[0] * x[0] - 4.0;
        Ok(Some(Array1::from_elem(1, curvature * v[0])))
    }
}

fn main() -> Result<(), SaddleError> {
    let surface = DoubleWell;
    let reactant = relax(&surface, ndarray::array![-1.4])?;
    let product = relax(&surface, ndarray::array![1.25])?;
    let saddle = saddle(&surface, ndarray::array![0.2], ndarray::array![1.0])?;
    let images = images_through(&reactant, &saddle, &product)?;

    let settings = RingSettings::default();
    let blocks = 6;
    let mobility = PafiConfig::default().mobility;
    let one_bound =
        RingPolymer::new(&surface, 1, settings.clone())?.time_step(mobility, &images)?;
    let two_bound =
        RingPolymer::new(&surface, 2, settings.clone())?.time_step(mobility, &images)?;
    let stability_bound = if one_bound < two_bound {
        one_bound
    } else {
        two_bound
    };
    let time_step = stability_bound / 8.0;
    let config = PafiConfig::default()
        .with_thermal(1.0 / settings.beta)
        .with_time_step(time_step)
        .with_blocks(blocks)
        .with_samples(blocks * blocks * 4)
        .with_stride(2)
        .with_thermalize(blocks * 10)
        .with_seed(1)
        .with_integrator(PafiIntegrator::EulerMaruyama)
        .with_fix_com(false)
        .with_retain(0);
    let samples = bead_study(&surface, &images, &[1, 2], &settings, &config)?;
    let (saddle_energy, _) = quartic(saddle[0]);
    println!(
        "min-mode saddle at {:.6}, energy {:.6}",
        saddle[0], saddle_energy
    );
    println!("centroid-hyperplane PMF (static centroid QTST), not an RPMD rate");
    for sample in &samples {
        println!(
            "P={} barrier {:.6} ± {:.6}, n_eff {:.2}",
            sample.beads,
            sample.profile.barrier,
            sample.profile.barrier_error,
            sample.effective_samples
        );
    }
    Ok(())
}

/// Reactant to saddle to product. The saddle is the shared knot.
///
/// The step is two samples of the well's curvature length `1/√|V''|`.
fn images_through(
    reactant: &Array1<f64>,
    saddle: &Array1<f64>,
    product: &Array1<f64>,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    let curvature = (12.0 * reactant[0] * reactant[0] - 4.0).abs();
    let positive = matches!(
        curvature.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Greater)
    );
    if !positive {
        return Err(SaddleError::Invalid(
            "the path spacing needs a positive well curvature".into(),
        ));
    }
    let span = (saddle[0] - reactant[0]).abs();
    let intervals = (2.0 * span * curvature.sqrt()).ceil();
    if !intervals.is_finite() || intervals < 1.0 {
        return Err(SaddleError::Invalid(
            "the reactant and the saddle did not separate".into(),
        ));
    }
    let per_leg = intervals as usize + 1;
    let mut images = blend(reactant, saddle, per_leg);
    let middle = per_leg - 1;
    for (value, bit) in images[middle].iter().zip(saddle.iter()) {
        if value.to_bits() != bit.to_bits() {
            return Err(SaddleError::Invalid(
                "the min-mode saddle is not an image of the path".into(),
            ));
        }
    }
    images.extend(blend(saddle, product, per_leg).into_iter().skip(1));
    Ok(images)
}

fn blend(start: &Array1<f64>, end: &Array1<f64>, count: usize) -> Vec<Array1<f64>> {
    (0..count)
        .map(|index| {
            let fraction = index as f64 / (count - 1) as f64;
            (1.0 - fraction) * start + fraction * end
        })
        .collect()
}

fn relax(surface: &DoubleWell, mut position: Array1<f64>) -> Result<Array1<f64>, SaddleError> {
    let controls = RelaxConfig::default().with_method(Method::lbfgs_with_memory(8));
    let mut solver = Solver::new(
        controls.method.clone(),
        Control::default()
            .with_maxiter(controls.max_steps)
            .with_gtol(controls.force_tol)
            .with_istep(controls.initial_step)
            .with_stop_norm(StopNorm::Inf),
        position.len(),
    );
    solver.set_atom_maxmove(controls.max_atom_move);
    let n = position.len();
    let oracle = Oracle::unbounded(n, |x: ArrayView1<f64>| {
        let (energy, gradient) = quartic(x[0]);
        (energy, ndarray::array![gradient])
    });
    solver
        .run(&oracle, &mut position)
        .map_err(|error| SaddleError::Solver(error.to_string()))?;
    let (_, gradient) = surface.eval(position.view())?;
    let mut largest = 0.0_f64;
    for value in &gradient {
        if value.abs() > largest {
            largest = value.abs();
        }
    }
    if !(largest.is_finite() && largest <= controls.force_tol) {
        return Err(SaddleError::Solver(format!(
            "relaxation left a force component {largest}"
        )));
    }
    Ok(position)
}

fn saddle(
    surface: &DoubleWell,
    start: Array1<f64>,
    mode: Array1<f64>,
) -> Result<Array1<f64>, SaddleError> {
    let mut session = MinModeSession::new(MinModeConfig::default(), start, mode)?;
    let report = session.run(surface, RelaxConfig::default().max_steps)?;
    let negative = matches!(
        report.curvature.partial_cmp(&0.0),
        Some(std::cmp::Ordering::Less)
    );
    if report.status != MinModeStatus::Converged || !negative {
        return Err(SaddleError::Solver(format!(
            "saddle search ended as {:?} with curvature {}",
            report.status, report.curvature
        )));
    }
    Ok(session.position().to_owned())
}
