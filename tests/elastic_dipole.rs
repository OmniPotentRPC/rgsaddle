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
    let constants = ElasticConstants::cubic(10.0, 4.0, 2.0).expect("stable cubic tensor");
    let corrector = PeriodicElastic::new(
        constants,
        PeriodicElasticConfig::default()
            .with_green(AnisotropicGreenConfig::default().with_angular_points(32))
            .with_image_shells(2),
    )
    .expect("periodic corrector");
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
        for length in [9.0, 12.0, 15.0] {
            let cell = linkcell::Cell::ortho(length, length, length).expect("cubic cell");
            let elastic = corrector
                .pair_correction(&cell, first, second)
                .expect("elastic correction");
            let periodic_energy = chemical + elastic.periodic;
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
        }
        assert!(
            span(&corrected) <= f64::EPSILON.sqrt() * chemical.abs().max(1.0),
            "separation {position:?}: {corrected:?}"
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
