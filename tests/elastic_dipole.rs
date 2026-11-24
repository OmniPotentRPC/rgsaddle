//! Analytic and finite-cell checks for the anisotropic elastic layer.

use rgsaddle::{
    AnisotropicGreen, AnisotropicGreenConfig, DefectDipole, ElasticConstants, ElasticDipole,
    ElasticPairState, Kelvin, PeriodicElastic, PeriodicElasticConfig,
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

#[test]
fn barnett_quadrature_requires_multiples_of_four_from_sixteen() {
    let constants = cubic_constants();
    for points in [8, 10, 12, 14, 18] {
        let error = AnisotropicGreen::new(constants)
            .with_config(AnisotropicGreenConfig::default().with_angular_points(points))
            .expect_err("invalid quadrature should fail");
        assert!(
            error
                .to_string()
                .contains("at least sixteen points and a multiple of four"),
            "points {points} gave {error}"
        );
    }
    for points in [16, 20] {
        AnisotropicGreen::new(constants)
            .with_config(AnisotropicGreenConfig::default().with_angular_points(points))
            .expect("valid quadrature should pass");
    }
}

#[test]
fn periodic_elastic_rejects_invalid_angular_mesh_boundaries() {
    let constants = cubic_constants();
    let config = PeriodicElasticConfig::default().with_image_shells(2);
    for points in [12, 18] {
        let error = PeriodicElastic::new(
            constants,
            config.with_green(AnisotropicGreenConfig::default().with_angular_points(points)),
        )
        .expect_err("invalid quadrature should fail");
        assert!(
            error
                .to_string()
                .contains("at least sixteen points and a multiple of four"),
            "points {points} gave {error}"
        );
    }
    PeriodicElastic::new(
        constants,
        config.with_green(AnisotropicGreenConfig::default().with_angular_points(16)),
    )
    .expect("boundary quadrature should pass");
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
            raw.push(periodic_energy);
            corrected.push(infinite.infinite);
            reference_infinite.push(chemical + reference_pair.infinite);
            max_reference_mismatch = max_reference_mismatch.max(
                (infinite.infinite - (chemical + reference_pair.infinite)).abs(),
            );
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
