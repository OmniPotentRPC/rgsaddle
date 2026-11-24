//! Molecular NEB starts built by TRICS and evaluated with rgpot kernels.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::mic::Cell;
use rgsaddle::{
    AutoNeb, AutoNebInterpolant, AutoNebTuning, BandConfig, BandReport, BandSession, BandStatus,
    BandSurface, PathInitializer, SaddleError, TricsStartConfig, initialize_path,
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

fn assert_no_overlap(path: &Array2<f64>) {
    let atoms = path.ncols() / 3;
    for row in path.outer_iter() {
        for a in 0..atoms {
            for b in (a + 1)..atoms {
                assert!(distance(row, a, b) > 0.35, "atom overlap in {row:?}");
            }
        }
    }
}

fn assert_continuous_and_smooth(path: &Array2<f64>) {
    assert_no_overlap(path);

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

/// Fraction by which a distance may differ from its endpoint value.
fn relative_spread(path: &Array2<f64>, a: usize, b: usize) -> f64 {
    let first = distance(path.row(0), a, b);
    path.outer_iter()
        .map(|row| (distance(row, a, b) - first).abs() / first)
        .fold(0.0_f64, f64::max)
}

fn path_energies(path: &Array2<f64>) -> Vec<f64> {
    let mut energies = Array1::zeros(path.nrows());
    let mut gradients = Array2::zeros(path.raw_dim());
    LennardJones
        .eval(path.view(), &mut energies, &mut gradients)
        .expect("path energies");
    energies.to_vec()
}

const IMAGES: usize = 7;
const FORCE_TOL: f64 = 0.05;

/// Relax the path as an NEB, returning the converged band and its first
/// report.
fn relax(path: Array2<f64>) -> (BandSession, BandReport, BandReport) {
    let config = BandConfig::default()
        .with_remove_translation(false)
        .with_force_tol(FORCE_TOL);
    let mut neb = BandSession::new(config, path).expect("NEB session");
    let first = neb.step(&LennardJones).expect("first NEB step");
    let last = neb.run(&LennardJones, 2000).expect("NEB run");
    (neb, first, last)
}

fn run_start(reactant: Array1<f64>, product: Array1<f64>, bonds: &[(usize, usize)]) {
    let initializer = PathInitializer::Trics(TricsStartConfig::default());
    let path = initialize_path(reactant.view(), product.view(), IMAGES, None, &initializer)
        .expect("TRICS start");
    assert_eq!(path.nrows(), IMAGES);
    assert_eq!(path.ncols(), reactant.len());
    assert_eq!(path.row(0), reactant.view());
    assert_eq!(path.row(path.nrows() - 1), product.view());
    assert_continuous_and_smooth(&path);
    assert!(path_energies(&path).iter().all(|energy| energy.is_finite()));
    let linear = initialize_path(
        reactant.view(),
        product.view(),
        IMAGES,
        None,
        &PathInitializer::Linear,
    )
    .expect("linear start");
    let mut deformed_by_linear = false;
    for &(a, b) in bonds {
        let trics = relative_spread(&path, a, b);
        let straight = relative_spread(&linear, a, b);
        assert!(trics < 0.05, "TRICS stretched bond {a}-{b} by {trics}");
        if straight > 0.05 {
            assert!(
                trics < 0.5 * straight,
                "TRICS bond {a}-{b} spread {trics} is not below linear {straight}"
            );
        }
        deformed_by_linear |= straight > 0.05;
    }

    assert!(
        deformed_by_linear,
        "the linear start must deform one of the bonds, or the comparison is vacuous"
    );

    let (neb, first, last) = relax(path.clone());
    assert_eq!(last.status, BandStatus::Converged, "{last:?}");
    assert!(last.max_force <= FORCE_TOL, "{last:?}");
    assert!(last.max_force < first.max_force, "{first:?} then {last:?}");
    assert!(last.iteration > 1);

    let relaxed = neb.positions().to_owned();
    assert_eq!(relaxed.dim(), path.dim());
    assert_eq!(relaxed.row(0), reactant.view());
    assert_eq!(relaxed.row(IMAGES - 1), product.view());
    assert_no_overlap(&relaxed);
    let energies = path_energies(&relaxed);
    assert!(energies.iter().all(|energy| energy.is_finite()));
    let barrier = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ends = energies[0].max(energies[IMAGES - 1]);
    assert!(
        barrier >= ends,
        "highest image {barrier} below an endpoint {ends}"
    );
    assert!(
        barrier - ends < 5.0,
        "relaxed path has an unphysical {} barrier",
        barrier - ends
    );
}

#[test]
fn trics_starts_neb_for_a_triatomic_linear_bend() {
    let reactant = array![-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.6, 0.0];
    let product = array![-0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, -0.6, 0.0];
    run_start(reactant, product, &[(1, 2)]);
}

#[test]
fn trics_starts_neb_for_a_four_atom_torsion() {
    let reactant = array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.9, 0.8, 0.0, 2.7, 0.8, 0.8];
    let product = array![0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 1.9, 0.8, 0.0, 2.7, 0.8, -0.8];
    run_start(reactant, product, &[(0, 1), (1, 2), (2, 3)]);
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
        .with_trics(TricsStartConfig::default());
    let config = BandConfig::default().with_autoneb(Some(tuning));
    let autoneb = AutoNeb::new(config, endpoints.clone()).expect("TRICS AutoNEB");
    let chain = autoneb.positions().to_owned();
    assert_eq!(chain.nrows(), 5);
    assert_eq!(chain.row(0), endpoints.row(0));
    assert_eq!(chain.row(4), endpoints.row(1));
    assert_continuous_and_smooth(&chain);
    assert!(
        path_energies(&chain)
            .iter()
            .all(|energy| energy.is_finite())
    );
    assert!(
        relative_spread(&chain, 1, 2) < 0.05,
        "inserted images stretched the 1-2 bond"
    );
}

fn triclinic_cell() -> Cell {
    Cell::from_vectors(
        [6.0, 0.0, 0.0],
        [1.2, 5.0, 0.0],
        [0.3, 0.4, 7.0],
        [0.0, 0.0, 0.0],
    )
    .expect("triclinic cell")
}

fn rows_of(path: &Array2<f64>, atoms: usize) -> usize {
    assert_eq!(path.ncols(), 3 * atoms);
    path.nrows()
}

#[test]
fn trics_refuses_a_triclinic_cell_and_the_cell_aware_starts_honour_it() {
    let reactant = array![0.4, 0.5, 0.5, 5.5, 0.8, 0.6, 3.0, 2.5, 3.5];
    let product = array![5.6, 0.7, 0.8, 0.5, 0.6, 0.4, 3.0, 2.5, 3.5];
    let cell = triclinic_cell();

    // TRICS needs unwrapped coordinates, so a cell is an error and the
    // path is never built across the periodic images.
    let trics = PathInitializer::Trics(TricsStartConfig::default());
    let refused = initialize_path(reactant.view(), product.view(), IMAGES, Some(&cell), &trics);
    assert!(
        matches!(refused, Err(SaddleError::Invalid(_))),
        "{refused:?}"
    );

    // The linear start in the same cell takes the minimum image: atom 0
    // moves 0.8 across the boundary instead of 5.2 through the cell.
    let linear = initialize_path(
        reactant.view(),
        product.view(),
        IMAGES,
        Some(&cell),
        &PathInitializer::Linear,
    )
    .expect("linear start in the cell");
    assert_eq!(rows_of(&linear, 3), IMAGES);
    let step = &linear.row(1) - &linear.row(0);
    let first_atom = (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt();
    assert!(first_atom < 0.3, "first atom stepped {first_atom}");
    assert!((step[0] + 0.8 / 6.0).abs() < 1e-12, "{step:?}");

    // AutoNEB refuses the same combination when the band is created.
    let endpoints = Array2::from_shape_vec(
        (2, 9),
        reactant.iter().chain(product.iter()).copied().collect(),
    )
    .unwrap();
    let tuning = AutoNebTuning::default()
        .with_n_simul(3)
        .with_n_max(5)
        .with_trics(TricsStartConfig::default());
    let config = BandConfig::default()
        .with_cell(rgsaddle::Cell([
            [6.0, 0.0, 0.0],
            [1.2, 5.0, 0.0],
            [0.3, 0.4, 7.0],
        ]))
        .with_autoneb(Some(tuning));
    let refused = AutoNeb::new(config, endpoints);
    assert!(matches!(refused, Err(SaddleError::Invalid(_))));
}

/// A downstream crate that matches `AutoNebInterpolant` without a wildcard
/// arm keeps compiling, because TRICS placement is an option of
/// `AutoNebTuning` and not a variant of the enum.
#[test]
fn autoneb_interpolant_stays_exhaustively_matchable() {
    let describe = |interpolant: &AutoNebInterpolant| match interpolant {
        AutoNebInterpolant::Linear => "linear",
        AutoNebInterpolant::Idpp(_) => "idpp",
    };
    assert_eq!(describe(&AutoNebInterpolant::Linear), "linear");
    assert_eq!(describe(&AutoNebTuning::default().interpolant), "linear");
}
