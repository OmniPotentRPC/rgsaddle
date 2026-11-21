//! Molecular NEB starts built by TRICS and evaluated with rgpot kernels.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::{
    AutoNeb, AutoNebInterpolant, AutoNebTuning, BandConfig, BandSession, BandSurface,
    PathInitializer, SaddleError, TricsStartConfig, initialize_path,
};

struct LennardJones;

fn rgpot_lj_gradient(positions: &[f64], gradient: &mut [f64]) -> f64 {
    gradient.fill(0.0);
    let mut energy = 0.0;
    for a in 0..positions.len() / 3 {
        for b in (a + 1)..positions.len() / 3 {
            let mut delta = [0.0; 3];
            let mut squared = 0.0;
            for component in 0..3 {
                delta[component] = positions[3 * a + component] - positions[3 * b + component];
                squared += delta[component] * delta[component];
            }
            let inverse = 1.0 / squared;
            let inverse_six = inverse * inverse * inverse;
            let inverse_twelve = inverse_six * inverse_six;
            energy += 4.0 * (inverse_twelve - inverse_six);
            let slope = -24.0 * inverse * (2.0 * inverse_twelve - inverse_six);
            for component in 0..3 {
                let contribution = slope * delta[component];
                gradient[3 * a + component] += contribution;
                gradient[3 * b + component] -= contribution;
            }
        }
    }
    energy
}

impl BandSurface for LennardJones {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (image, row) in positions.outer_iter().enumerate() {
            let mut gradient = gradients.row_mut(image);
            let row = row
                .as_slice()
                .ok_or_else(|| SaddleError::Shape("contiguous NEB image required".into()))?;
            let gradient = gradient
                .as_slice_mut()
                .ok_or_else(|| SaddleError::Shape("contiguous NEB gradient required".into()))?;
            energies[image] = rgpot_lj_gradient(row, gradient);
        }
        Ok(())
    }
}

fn distance(row: ArrayView1<f64>, a: usize, b: usize) -> f64 {
    let mut squared = 0.0;
    for component in 0..3 {
        let delta = row[3 * a + component] - row[3 * b + component];
        squared += delta * delta;
    }
    squared.sqrt()
}

fn assert_continuous_and_smooth(path: &Array2<f64>) {
    let atoms = path.ncols() / 3;
    for row in path.outer_iter() {
        for a in 0..atoms {
            for b in (a + 1)..atoms {
                assert!(distance(row, a, b) > 0.35, "atom overlap in {row:?}");
            }
        }
    }

    let steps: Vec<Array1<f64>> = (1..path.nrows())
        .map(|image| &path.row(image) - &path.row(image - 1))
        .collect();
    let lengths: Vec<f64> = steps.iter().map(|step| step.dot(step).sqrt()).collect();
    let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
    let longest = lengths.iter().copied().fold(0.0_f64, f64::max);
    assert!(longest < 2.5 * mean, "uneven image spacing {lengths:?}");

    for pair in steps.windows(2) {
        let cosine = pair[0].dot(&pair[1]) / (pair[0].dot(&pair[0]) * pair[1].dot(&pair[1])).sqrt();
        assert!(cosine > -0.25, "path doubles back with cosine {cosine}");
    }
}

fn run_start(reactant: Array1<f64>, product: Array1<f64>) {
    let initializer = PathInitializer::Trics(TricsStartConfig::default());
    let path = initialize_path(reactant.view(), product.view(), 7, None, &initializer)
        .expect("TRICS start");
    assert_eq!(path.row(0), reactant.view());
    assert_eq!(path.row(path.nrows() - 1), product.view());
    assert_continuous_and_smooth(&path);

    let config = BandConfig::default().with_remove_translation(false);
    let mut neb = BandSession::new(config, path).expect("NEB session");
    let report = neb.step(&LennardJones).expect("NEB step");
    assert!(report.max_force.is_finite());
    assert_continuous_and_smooth(&neb.positions().to_owned());
}

#[test]
fn trics_starts_neb_for_a_triatomic_linear_bend() {
    let reactant = array![-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0];
    let product = array![-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, -0.6, 0.0];
    run_start(reactant, product);
}

#[test]
fn trics_starts_neb_for_a_four_atom_torsion() {
    let reactant = array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.9, 0.8, 0.0, 2.7, 0.8, 0.8];
    let product = array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.9, 0.8, 0.0, 2.7, 0.8, -0.8];
    run_start(reactant, product);
}

#[test]
fn autoneb_uses_trics_for_inserted_images() {
    let endpoints = array![
        [-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0],
        [-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, -0.6, 0.0]
    ];
    let tuning = AutoNebTuning::default()
        .with_n_simul(3)
        .with_n_max(5)
        .with_interpolant(AutoNebInterpolant::Trics(TricsStartConfig::default()));
    let config = BandConfig::default().with_autoneb(Some(tuning));
    let autoneb = AutoNeb::new(config, endpoints).expect("TRICS AutoNEB");
    assert_eq!(autoneb.positions().nrows(), 5);
    assert_continuous_and_smooth(&autoneb.positions().to_owned());
}
