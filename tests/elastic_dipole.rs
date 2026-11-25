//! Analytic and finite-cell checks for the anisotropic elastic layer.

use rgsaddle::{
    AnisotropicGreen, AnisotropicGreenConfig, DefectDipole, ElasticConstants, ElasticDipole,
    ElasticPairState, ElasticRate, Kelvin, PeriodicElastic, PeriodicElasticConfig, SaddleError,
    arrhenius,
};

fn shear_dipole(value: f64) -> ElasticDipole {
    ElasticDipole::from_tensor([
        [0.0, value, 0.0],
        [value, 0.0, 0.5 * value],
        [0.0, 0.5 * value, 0.0],
    ])
    .expect("finite symmetric dipole")
}

fn cubic_constants() -> ElasticConstants {
    ElasticConstants::cubic(10.0, 4.0, 2.0).expect("stable cubic tensor")
}

fn catalogue_state(energy: f64) -> ElasticPairState {
    ElasticPairState {
        energy,
        first: DefectDipole {
            position: [0.0, 0.0, 0.0],
            dipole: shear_dipole(1.2),
        },
        second: DefectDipole {
            position: [3.0, 1.0, 0.5],
            dipole: shear_dipole(-0.9),
        },
    }
}

fn rate_corrector() -> PeriodicElastic {
    PeriodicElastic::new(
        cubic_constants(),
        PeriodicElasticConfig::default()
            .with_green(AnisotropicGreenConfig::default().with_angular_points(32))
            .with_image_shells(2),
    )
    .expect("periodic corrector")
}

#[test]
fn barnett_quadrature_requires_multiples_of_eight_from_thirty_two() {
    let constants = cubic_constants();
    for points in [8, 10, 12, 14, 16, 18, 20, 24, 28, 36] {
        let error = AnisotropicGreen::new(constants)
            .with_config(AnisotropicGreenConfig::default().with_angular_points(points))
            .expect_err("invalid quadrature should fail");
        assert!(
            error
                .to_string()
                .contains("at least 32 points and a multiple of 8"),
            "points {points} gave {error}"
        );
    }
    let separation = [2.3, -1.1, 0.7];
    let source = shear_dipole(1.7);
    let target = shear_dipole(-0.8);
    for points in [32, 40] {
        let green = AnisotropicGreen::new(constants)
            .with_config(AnisotropicGreenConfig::default().with_angular_points(points))
            .expect("valid quadrature should pass");
        let estimate = green
            .dipole_interaction(separation, source, target)
            .expect("valid quadrature should evaluate dipole_interaction");
        assert!(
            estimate.value.is_finite() && estimate.error.is_finite(),
            "points {points} gave non-finite estimate {estimate:?}"
        );
    }
}

#[test]
fn periodic_elastic_rejects_invalid_angular_mesh_boundaries() {
    let constants = cubic_constants();
    let config = PeriodicElasticConfig::default().with_image_shells(2);
    for points in [12, 16, 18, 20, 24, 36] {
        let error = PeriodicElastic::new(
            constants,
            config.with_green(AnisotropicGreenConfig::default().with_angular_points(points)),
        )
        .expect_err("invalid quadrature should fail");
        assert!(
            error
                .to_string()
                .contains("at least 32 points and a multiple of 8"),
            "points {points} gave {error}"
        );
    }
    let periodic = PeriodicElastic::new(
        constants,
        config.with_green(AnisotropicGreenConfig::default().with_angular_points(32)),
    )
    .expect("boundary quadrature should pass");
    let cell = linkcell::Cell::ortho(12.0, 12.0, 12.0).expect("cubic cell");
    let first = DefectDipole {
        position: [0.0, 0.0, 0.0],
        dipole: shear_dipole(1.2),
    };
    let second = DefectDipole {
        position: [3.0, 1.0, 0.5],
        dipole: shear_dipole(-0.9),
    };
    let correction = periodic
        .pair_correction(&cell, first, second)
        .expect("boundary quadrature should evaluate dipole_interaction");
    assert!(
        correction.infinite.is_finite()
            && correction.periodic.is_finite()
            && correction.error.is_finite(),
        "boundary correction gave non-finite result {correction:?}"
    );
}

#[test]
fn half_mesh_rejected_meshes_are_refused_at_construction() {
    let constants = cubic_constants();
    let periodic_config = PeriodicElasticConfig::default().with_image_shells(2);
    for points in [16, 20, 24, 28, 36] {
        let green_config = AnisotropicGreenConfig::default().with_angular_points(points);
        let green_error = AnisotropicGreen::new(constants)
            .with_config(green_config)
            .expect_err("half-mesh rejection must happen at AnisotropicGreen construction");
        assert!(
            green_error
                .to_string()
                .contains("at least 32 points and a multiple of 8"),
            "points {points} gave {green_error}"
        );
        let periodic_error =
            PeriodicElastic::new(constants, periodic_config.with_green(green_config))
                .expect_err("half-mesh rejection must happen at PeriodicElastic construction");
        assert!(
            periodic_error
                .to_string()
                .contains("at least 32 points and a multiple of 8"),
            "points {points} gave {periodic_error}"
        );
    }
}

#[test]
fn barnett_reduces_to_the_eshelby_kelvin_solution() {
    // Eshelby's isotropic point-force solution is Kelvin's Green function.
    // lambda and mu give nu = lambda/[2(lambda+mu)].
    let lambda = 2.0;
    let mu = 3.0;
    let poisson = lambda / (2.0 * (lambda + mu));
    let source = shear_dipole(1.7);
    let target = shear_dipole(-0.8);
    let separation = [2.3, -1.1, 0.7];
    let analytic = Kelvin::new(mu, poisson)
        .expect("stable isotropic solid")
        .dipole_interaction(separation, source, target)
        .expect("nonzero separation");
    let numeric = AnisotropicGreen::new(
        ElasticConstants::isotropic(lambda, mu).expect("stable isotropic tensor"),
    )
    .with_config(AnisotropicGreenConfig::default().with_angular_points(192))
    .expect("valid quadrature")
    .dipole_interaction(separation, source, target)
    .expect("Barnett interaction");
    let tolerance = f64::EPSILON.sqrt() * analytic.abs().max(1.0);
    assert!(
        (numeric.value - analytic).abs() <= tolerance,
        "Barnett {} Kelvin {} estimated error {}",
        numeric.value,
        analytic,
        numeric.error
    );
}

#[test]
fn cubic_100_green_derivative_matches_dederichs_leibfried() {
    // Dederichs and Leibfried, Phys. Rev. 188, 1175 (1969),
    // doi:10.1103/PhysRev.188.1175, identify <100> as an exact cubic
    // direction.  On x, G_xx = 1/(4 pi C44 r), hence
    // G_xx,xx = 1/(2 pi C44 r^3), including for an anisotropic cubic solid.
    let c11: f64 = 11.0;
    let c12 = 4.0;
    let c44 = 2.0;
    assert!((c11 - c12 - 2.0 * c44).abs() > f64::EPSILON.sqrt());
    let radius = 2.5;
    let green =
        AnisotropicGreen::new(ElasticConstants::cubic(c11, c12, c44).expect("stable cubic tensor"))
            .with_config(AnisotropicGreenConfig::default().with_angular_points(192))
            .expect("valid quadrature");
    let (second, error) = green
        .second_derivative([radius, 0.0, 0.0])
        .expect("Green derivative");
    let published = 1.0 / (2.0 * std::f64::consts::PI * c44 * radius.powi(3));
    let tolerance = f64::EPSILON.sqrt() * published.abs().max(1.0);
    assert!(
        (second[0][0][0][0] - published).abs() <= tolerance,
        "Barnett {} exact {} estimated error {}",
        second[0][0][0][0],
        published,
        error[0][0][0][0]
    );
}

#[test]
fn periodic_pair_correction_removes_the_smoke_scale_cell_drift() {
    let constants = cubic_constants();
    let corrector = PeriodicElastic::new(
        constants,
        PeriodicElasticConfig::default()
            .with_green(AnisotropicGreenConfig::default().with_angular_points(32))
            .with_image_shells(2),
    )
    .expect("periodic corrector");
    let reference = PeriodicElastic::new(
        constants,
        PeriodicElasticConfig::default()
            .with_green(AnisotropicGreenConfig::default().with_angular_points(192))
            .with_image_shells(5),
    )
    .expect("reference periodic corrector");
    let first = DefectDipole {
        position: [0.0, 0.0, 0.0],
        dipole: shear_dipole(1.2),
    };
    let chemical = -0.35;
    let span = |values: &[f64]| {
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    for position in [[3.0, 1.0, 0.5], [2.5, 2.0, 1.0], [4.0, 1.5, 0.75]] {
        let second = DefectDipole {
            position,
            dipole: shear_dipole(-0.9),
        };
        let mut raw = Vec::new();
        let mut corrected = Vec::new();
        let mut reference_infinite = Vec::new();
        let mut max_reference_mismatch = 0.0_f64;
        for length in [9.0, 12.0, 15.0] {
            let cell = linkcell::Cell::ortho(length, length, length).expect("cubic cell");
            let reference_pair = reference
                .pair_correction(&cell, first, second)
                .expect("reference elastic correction");
            let periodic_energy = chemical + reference_pair.periodic;
            let state = ElasticPairState {
                energy: periodic_energy,
                first,
                second,
            };
            let infinite = corrector
                .correct_energy(&cell, state)
                .expect("corrected energy");
            let reference_energy = chemical + reference_pair.infinite;
            let mismatch = (infinite.infinite - reference_energy).abs();
            raw.push(periodic_energy);
            corrected.push(infinite.infinite);
            reference_infinite.push(reference_energy);
            max_reference_mismatch = max_reference_mismatch.max(mismatch);
        }
        assert!(
            span(&reference_infinite) <= f64::EPSILON.sqrt() * chemical.abs().max(1.0),
            "separation {position:?}: {reference_infinite:?}"
        );
        assert!(
            max_reference_mismatch <= 1.0e-4,
            "separation {position:?}: corrected {corrected:?} reference {reference_infinite:?}"
        );
        assert!(
            span(&corrected) < span(&raw),
            "separation {position:?}: raw {raw:?} corrected {corrected:?}"
        );
    }
}

#[test]
fn homogeneous_strain_and_residual_stress_routes_agree() {
    let volume = 17.0;
    let defect = [[0.2, 0.03, 0.0], [0.03, -0.1, 0.0], [0.0, 0.0, 0.04]];
    let bulk = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]];
    let stress = ElasticDipole::from_stress(volume, defect, bulk).expect("stress route");
    let gradient = |tensor: [[f64; 3]; 3]| tensor.map(|row| row.map(|value| volume * value));
    let strain = ElasticDipole::from_energy_strain_gradient(gradient(defect), gradient(bulk))
        .expect("homogeneous-strain route");
    for (from_stress, from_strain) in stress
        .tensor()
        .into_iter()
        .flatten()
        .zip(strain.tensor().into_iter().flatten())
    {
        assert!(
            (from_stress - from_strain).abs() <= 8.0 * f64::EPSILON * from_stress.abs().max(1.0)
        );
    }
}

#[test]
fn indefinite_cubic_acoustic_tensor_is_rejected() {
    // C11 > 0 and C44 < 0 makes every sampled acoustic tensor indefinite
    // with a positive determinant: the [100] eigenvalues are C11, C44, C44,
    // and C12 = 12 keeps the [110] and [111] determinants positive too.
    // A determinant test therefore accepts this tensor.
    let error = ElasticConstants::cubic(10.0, 12.0, -1.0)
        .expect_err("indefinite acoustic tensor must be rejected");
    assert!(
        matches!(
            error,
            SaddleError::Invalid(ref message) if message.contains("not positive definite")
        ),
        "{error}"
    );
}

#[test]
fn correct_rate_catalogue_rejects_non_finite_rates() {
    let corrector = rate_corrector();
    let cell = linkcell::Cell::ortho(12.0, 12.0, 12.0).expect("cubic cell");
    let reactant = catalogue_state(0.0);
    let saddle = catalogue_state(-1.0e6);
    let reactant_energy = corrector
        .correct_energy(&cell, reactant)
        .expect("reactant correction");
    let saddle_energy = corrector
        .correct_energy(&cell, saddle)
        .expect("saddle correction");
    let barrier = saddle_energy.infinite - reactant_energy.infinite;
    assert!(
        barrier < -100.0,
        "barrier {barrier} must overflow the Arrhenius factor at 300 K"
    );
    let error = corrector
        .correct_rate_catalogue(
            &cell,
            &[ElasticRate {
                reactant,
                saddle,
                prefactor_hz: 1.0e12,
                temperature_k: 300.0,
            }],
        )
        .expect_err("an overflowing harmonic rate must be reported");
    assert!(
        matches!(error, SaddleError::NonFinite("corrected harmonic rate")),
        "{error}"
    );
}

#[test]
fn correct_rate_catalogue_keeps_a_finite_arrhenius_rate() {
    let corrector = rate_corrector();
    let cell = linkcell::Cell::ortho(12.0, 12.0, 12.0).expect("cubic cell");
    let prefactor_hz = 1.0e12;
    let temperature_k = 300.0;
    let barrier = 0.5;
    let corrected = corrector
        .correct_rate_catalogue(
            &cell,
            &[ElasticRate {
                reactant: catalogue_state(0.0),
                saddle: catalogue_state(barrier),
                prefactor_hz,
                temperature_k,
            }],
        )
        .expect("a modest barrier has a finite rate");
    assert_eq!(corrected.len(), 1);
    let row = corrected[0];
    assert_eq!(row.raw_barrier, barrier);
    assert!(
        (row.infinite_barrier - barrier).abs() <= 1.0e-8,
        "infinite barrier {} raw {barrier}",
        row.infinite_barrier
    );
    assert!(row.rate_hz.is_finite(), "rate {}", row.rate_hz);
    let expected = arrhenius(prefactor_hz, row.infinite_barrier, temperature_k);
    let scale = expected.abs().max(1.0);
    assert!(
        (row.rate_hz - expected).abs() <= 16.0 * f64::EPSILON * scale,
        "rate {} expected {expected}",
        row.rate_hz
    );
}
