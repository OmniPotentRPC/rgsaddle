use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    BandSurface, FixedEnergySeed, LineNebConfig, LineNebSession, LineNebStatus, SaddleError,
    Spline, fixed_energy_action, fixed_energy_action_force, fixed_energy_mep_seed,
};

fn analytic_surface(positions: ArrayView2<'_, f64>) -> (Array1<f64>, Array2<f64>) {
    let mut energies = Array1::zeros(positions.nrows());
    let mut gradients = Array2::zeros(positions.raw_dim());
    for (i, row) in positions.outer_iter().enumerate() {
        let x = row[0];
        let y = row[1];
        energies[i] = 0.7 * x * x + 0.4 * x * y + 1.1 * y * y + 2.0;
        gradients[(i, 0)] = 1.4 * x + 0.4 * y;
        gradients[(i, 1)] = 0.4 * x + 2.2 * y;
    }
    (energies, gradients)
}

#[test]
fn action_force_matches_an_action_finite_difference() {
    let positions = array![[-1.0, 0.2], [-0.35, 0.55], [0.25, 0.48], [0.9, -0.1],];
    let masses = array![2.0, 5.0];
    let target = 0.3;
    let hbar = 1.7;
    let (energies, gradients) = analytic_surface(positions.view());
    let force = fixed_energy_action_force(
        positions.view(),
        energies.view(),
        gradients.view(),
        masses.view(),
        target,
        hbar,
    )
    .unwrap();

    let displacement = 2.0e-6;
    for image in 1..positions.nrows() - 1 {
        for coordinate in 0..positions.ncols() {
            let mut plus = positions.clone();
            plus[(image, coordinate)] += displacement;
            let (plus_energies, _) = analytic_surface(plus.view());
            let plus_action = fixed_energy_action(
                plus.view(),
                plus_energies.view(),
                masses.view(),
                target,
                hbar,
            )
            .unwrap();

            let mut minus = positions.clone();
            minus[(image, coordinate)] -= displacement;
            let (minus_energies, _) = analytic_surface(minus.view());
            let minus_action = fixed_energy_action(
                minus.view(),
                minus_energies.view(),
                masses.view(),
                target,
                hbar,
            )
            .unwrap();
            let finite_difference = (plus_action - minus_action) / (2.0 * displacement);
            assert!(
                (finite_difference + force[(image, coordinate)]).abs() < 2.0e-8,
                "image {image}, coordinate {coordinate}: FD {finite_difference}, force {}",
                force[(image, coordinate)]
            );
        }
    }
}

#[test]
fn the_action_is_the_trapezoid_over_every_image() {
    let positions = array![[0.0], [1.0], [2.0]];
    let masses = array![1.0];
    let action = fixed_energy_action(
        positions.view(),
        array![4.0, 2.0, 4.0].view(),
        masses.view(),
        1.0,
        1.0,
    )
    .unwrap();
    assert_eq!(action, 6.0_f64.sqrt() + 2.0_f64.sqrt());

    let turning_points = fixed_energy_action(
        positions.view(),
        array![1.0, 3.0, 1.0].view(),
        masses.view(),
        1.0,
        1.0,
    )
    .unwrap();
    assert_eq!(turning_points, 2.0);
}

#[test]
fn an_endpoint_below_the_fixed_energy_is_refused() {
    let positions = array![[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
    let masses = array![1.0, 1.0];
    for energies in [array![0.9, 2.0, 1.0], array![1.0, 2.0, 0.8]] {
        let action =
            fixed_energy_action(positions.view(), energies.view(), masses.view(), 1.0, 1.0);
        assert!(
            matches!(&action, Err(SaddleError::Invalid(message)) if message.contains("endpoint")),
            "{action:?}"
        );
        let force = fixed_energy_action_force(
            positions.view(),
            energies.view(),
            Array2::ones((3, 2)).view(),
            masses.view(),
            1.0,
            1.0,
        );
        assert!(matches!(force, Err(SaddleError::Invalid(_))));
    }
}

#[test]
fn a_session_endpoint_below_the_fixed_energy_has_zero_weight() {
    let target = 0.25;
    let positions = array![[-1.2, 0.0], [-0.5, 0.1], [0.0, 0.1], [0.5, 0.1], [1.2, 0.0]];
    let mut energies = Array1::zeros(5);
    let mut gradients = Array2::zeros((5, 2));
    DoubleWell
        .eval(positions.view(), &mut energies, &mut gradients)
        .unwrap();
    assert!(energies[0] < target && energies[4] < target);
    let masses = array![1.0, 1.0];
    assert!(
        fixed_energy_action(
            positions.view(),
            energies.view(),
            masses.view(),
            target,
            1.0
        )
        .is_err()
    );
    let config = LineNebConfig::default()
        .with_spring_k(1.0)
        .with_endpoint_kappa(1.0)
        .with_force_tol(1.0e-6)
        .with_energy(target)
        .with_masses(masses)
        .with_time_step(0.01)
        .with_reduced_planck(1.0);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let report = session.step(&DoubleWell).unwrap();
    assert!(report.action.is_finite() && report.action > 0.0);
    assert_eq!(report.status, LineNebStatus::Running);
}

#[test]
fn natural_cubic_returns_value_and_derivatives() {
    let spline = Spline::natural(&[0.0, 1.0, 2.0], &array![[1.0], [3.0], [5.0]]).unwrap();
    let sample = spline.sample(0.5).unwrap();
    assert_eq!(sample.value[0], 2.0);
    assert_eq!(sample.first[0], 2.0);
    assert_eq!(sample.second[0], 0.0);
}

struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let x = row[0];
            let y = row[1];
            energies[i] = (x * x - 1.0).powi(2) + y * y;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 2.0 * y;
        }
        Ok(())
    }
}

#[test]
fn a_seed_without_a_budget_is_exhausted() {
    let err = fixed_energy_mep_seed(
        &DoubleWell,
        array![0.0, 0.0].view(),
        array![1.0, 0.0].view(),
        array![1.0, 2.0].view(),
        FixedEnergySeed::default()
            .with_saddle_energy(1.0)
            .with_target_energy(0.25)
            .with_images(9)
            .with_max_evaluations(0),
    );
    assert!(matches!(
        err,
        Err(SaddleError::Invalid(message)) if message.starts_with("work budget exhausted")
    ));
}

#[test]
fn the_default_seed_energies_must_be_set() {
    let err = fixed_energy_mep_seed(
        &DoubleWell,
        array![0.0, 0.0].view(),
        array![1.0, 0.0].view(),
        array![1.0, 2.0].view(),
        FixedEnergySeed::default(),
    );
    assert!(matches!(err, Err(SaddleError::Invalid(_))));
}

#[test]
fn energy_sliced_seed_follows_the_double_well_mep() {
    let path = fixed_energy_mep_seed(
        &DoubleWell,
        array![0.0, 0.0].view(),
        array![1.0, 0.0].view(),
        array![1.0, 2.0].view(),
        FixedEnergySeed::default()
            .with_saddle_energy(1.0)
            .with_target_energy(0.25)
            .with_images(9)
            .with_max_evaluations(20_000),
    )
    .unwrap();
    assert_eq!(path.raw_dim(), ndarray::Ix2(9, 2));
    assert!(path.column(1).iter().all(|value| *value == 0.0));
    for endpoint in [0, 8] {
        let x = path[(endpoint, 0)];
        assert!(((x * x - 1.0).powi(2) - 0.25).abs() <= f64::EPSILON * 8.0);
    }
}

#[test]
fn fixed_energy_double_well_path_converges() {
    let n_images = 9;
    let target: f64 = 0.25;
    let initial_endpoint_energy: f64 = 0.3;
    let turning_point = (1.0_f64 - initial_endpoint_energy.sqrt()).sqrt();
    let mut positions = Array2::zeros((n_images, 2));
    for i in 0..n_images {
        let fraction = i as f64 / (n_images - 1) as f64;
        positions[(i, 0)] = -turning_point + 2.0 * turning_point * fraction;
        positions[(i, 1)] = 0.18 * (std::f64::consts::PI * fraction).sin();
    }
    let config = LineNebConfig::default()
        .with_spring_k(0.4)
        .with_endpoint_kappa(1.5)
        .with_force_tol(2.0e-6)
        .with_energy(target)
        .with_masses(array![1.0, 2.0])
        .with_time_step(0.03)
        .with_reduced_planck(1.0);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let host_force_budget = 80_000;
    let mut rows = 0;
    let report = loop {
        let report = session.step(&DoubleWell).unwrap();
        rows += report.surface_rows;
        if report.status == LineNebStatus::Converged {
            break report;
        }
        assert!(
            rows < host_force_budget,
            "host force budget exhausted: {report:?}"
        );
    };

    let converged = session.positions();
    assert!(converged.column(1).iter().all(|y| y.abs() < 2.0e-5));
    for endpoint in [0, n_images - 1] {
        let x = converged[(endpoint, 0)];
        let y = converged[(endpoint, 1)];
        let energy = (x * x - 1.0).powi(2) + y * y;
        assert!((energy - target).abs() < 2.0e-6);
    }
    assert!(report.max_interior_perpendicular <= 2.0e-6);
    assert!(report.max_endpoint_optimized <= 2.0e-6);
    let mut energies = Array1::zeros(converged.nrows());
    let mut gradients = Array2::zeros(converged.raw_dim());
    DoubleWell
        .eval(converged.view(), &mut energies, &mut gradients)
        .unwrap();
    let converged_action = fixed_energy_action(
        converged.view(),
        energies.view(),
        array![1.0, 2.0].view(),
        target,
        1.0,
    )
    .unwrap();
    assert_eq!(report.action, converged_action);
}

struct InvertedParabola;

impl BandSurface for InvertedParabola {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = 1.0 - row[0] * row[0];
            gradients[(i, 0)] = -2.0 * row[0];
        }
        Ok(())
    }
}

#[test]
fn an_unevenly_spaced_line_converges_only_once_its_springs_balance() {
    // With one coordinate per image the transverse action force is zero
    // and the endpoints sit on the fixed energy, so only the springs
    // can tell this band apart from a converged one.
    let positions = array![[-1.0], [-0.9], [-0.8], [0.5], [1.0]];
    let config = LineNebConfig::default()
        .with_spring_k(1.0)
        .with_endpoint_kappa(1.0)
        .with_force_tol(1.0e-8)
        .with_energy(0.0)
        .with_time_step(0.1)
        .with_reduced_planck(1.0);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let first = session.step(&InvertedParabola).unwrap();
    assert_eq!(first.max_interior_perpendicular, 0.0);
    assert_eq!(first.max_endpoint_optimized, 0.0);
    assert!(first.max_interior_spring > 0.5, "{first:?}");
    assert_eq!(first.status, LineNebStatus::Running);

    let report = loop {
        let report = session.step(&InvertedParabola).unwrap();
        if report.status == LineNebStatus::Converged {
            break report;
        }
        assert!(report.iteration < 100_000, "{report:?}");
    };
    assert!(report.max_interior_spring <= 1.0e-8);
    let converged = session.positions();
    for (image, expected) in [-1.0, -0.5, 0.0, 0.5, 1.0].into_iter().enumerate() {
        assert!(
            (converged[(image, 0)] - expected).abs() < 1.0e-7,
            "image {image}: {}",
            converged[(image, 0)]
        );
    }
}

struct Trough;

impl BandSurface for Trough {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            energies[i] = 1.0 + row[1] * row[1];
            gradients[(i, 0)] = 0.0;
            gradients[(i, 1)] = 2.0 * row[1];
        }
        Ok(())
    }
}

#[test]
fn a_plateau_of_three_equal_energies_still_steps() {
    // Images 1, 2, and 3 lie on the trough floor at one energy, so the
    // weighted tangent at image 2 has no energy difference to scale by.
    let positions = array![[-2.0, 0.5], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [2.0, 0.5]];
    let config = LineNebConfig::default()
        .with_spring_k(1.0)
        .with_endpoint_kappa(1.0)
        .with_force_tol(1.0e-6)
        .with_energy(0.0)
        .with_masses(array![1.0, 1.0])
        .with_time_step(0.05)
        .with_reduced_planck(1.0);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let report = session.step(&Trough).unwrap();
    assert!(report.action.is_finite());
    assert!(report.max_interior_perpendicular.is_finite());
    assert!(report.max_interior_spring.is_finite());
    assert_eq!(report.status, LineNebStatus::Running);
}
