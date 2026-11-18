use ndarray::{Array1, Array2, ArrayView2, array};
use rgsaddle::{
    BandSurface, LineNebConfig, LineNebSession, LineNebStatus, NaturalCubic, SaddleError,
    fixed_energy_action, fixed_energy_action_force, fixed_energy_mep_seed,
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
fn action_and_force_match_the_discrete_li_neb_equations() {
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
    let action = fixed_energy_action(
        positions.view(),
        energies.view(),
        masses.view(),
        target,
        hbar,
    )
    .unwrap();
    assert!((action - 4.451_267_924_111_834).abs() < 2.0e-15);
    let expected = array![
        [0.0, 0.0],
        [0.406_560_449_872_918_34, -2.593_280_341_805_817_5],
        [-0.851_336_823_737_935_1, -2.151_288_222_332_447],
        [0.0, 0.0],
    ];
    for (actual, expected) in force.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 2.0e-15);
    }
}

#[test]
fn action_rejects_an_endpoint_below_the_target_energy() {
    let positions = array![[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]];
    let energies = array![0.9, 2.0, 0.8];
    let error = fixed_energy_action(
        positions.view(),
        energies.view(),
        array![1.0, 1.0].view(),
        1.0,
        1.0,
    )
    .unwrap_err();
    assert!(error.to_string().contains("image 0"));
}

#[test]
fn natural_cubic_returns_value_and_derivatives() {
    let spline =
        NaturalCubic::new(vec![0.0, 1.0, 2.0], vec![vec![1.0], vec![3.0], vec![5.0]]).unwrap();
    let sample = spline.sample(0.5).unwrap();
    assert_eq!(sample.value, vec![2.0]);
    assert_eq!(sample.first, vec![2.0]);
    assert_eq!(sample.second, vec![0.0]);
}

#[test]
fn natural_cubic_handles_nonuniform_vector_data_and_knot_continuity() {
    let spline = NaturalCubic::new(
        vec![0.0, 1.0, 3.0],
        vec![vec![0.0, 1.0], vec![1.0, -1.0], vec![9.0, 2.0]],
    )
    .unwrap();
    let left = spline.sample(0.5).unwrap();
    assert_eq!(left.value, vec![0.3125, -0.21875]);
    assert!((left.first[0] - 0.875).abs() < f64::EPSILON);
    assert!((left.first[1] + 2.145_833_333_333_333_5).abs() < f64::EPSILON);
    assert_eq!(left.second, vec![1.5, 1.75]);
    let right = spline.sample(2.0).unwrap();
    assert_eq!(right.value, vec![4.25, -0.375]);
    assert!((right.first[0] - 4.25).abs() < f64::EPSILON);
    assert!((right.first[1] - 1.791_666_666_666_666_7).abs() < f64::EPSILON);
    assert_eq!(right.second, vec![1.5, 1.75]);
    let knot = spline.sample(1.0).unwrap();
    assert_eq!(knot.value, vec![1.0, -1.0]);
    assert!((knot.first[0] - 2.0).abs() < f64::EPSILON);
    assert!((knot.first[1] + 5.0 / 6.0).abs() < f64::EPSILON);
    assert_eq!(knot.second, vec![3.0, 3.5]);
}

#[test]
fn natural_cubic_rejects_invalid_two_knot_grids() {
    for knots in [vec![0.0, 0.0], vec![f64::NAN, 1.0]] {
        assert!(NaturalCubic::new(knots, vec![vec![0.0], vec![1.0]]).is_err());
    }
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

struct CoupledDoubleWell;

impl BandSurface for CoupledDoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<'_, f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let x = row[0];
            let y = row[1];
            let coupling = y - 0.5 * x;
            energies[i] = (x * x - 1.0).powi(2) + coupling * coupling;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0) - coupling;
            gradients[(i, 1)] = 2.0 * coupling;
        }
        Ok(())
    }
}

#[test]
fn energy_sliced_seed_follows_the_double_well_mep() {
    let path = fixed_energy_mep_seed(
        &DoubleWell,
        array![0.0, 0.0].view(),
        array![1.0, 0.0].view(),
        array![1.0, 2.0].view(),
        1.0,
        0.25,
        9,
    )
    .unwrap();
    assert_eq!(path.raw_dim(), ndarray::Ix2(9, 2));
    assert!(path.column(1).iter().all(|value| *value == 0.0));
    assert!(path[(4, 0)].abs() <= f64::EPSILON);
    assert!(((path[(4, 0)].powi(2) - 1.0).powi(2) - 1.0).abs() <= f64::EPSILON);
    for endpoint in [0, 8] {
        let x = path[(endpoint, 0)];
        assert!(((x * x - 1.0).powi(2) - 0.25).abs() <= f64::EPSILON * 8.0);
    }
}

#[test]
fn energy_sliced_seed_follows_a_curved_coupled_gradient_path() {
    let unstable_eigenvalue = (-1.5_f64 - 34.25_f64.sqrt()) / 2.0;
    let mode = array![2.0 - unstable_eigenvalue, 1.0];
    let path = fixed_energy_mep_seed(
        &CoupledDoubleWell,
        array![0.0, 0.0].view(),
        mode.view(),
        array![1.0, 1.0].view(),
        1.0,
        0.25,
        11,
    )
    .unwrap();
    assert_eq!(path.raw_dim(), ndarray::Ix2(11, 2));
    assert!(path[(5, 0)].abs() <= f64::EPSILON);
    assert!(path[(5, 1)].abs() <= f64::EPSILON);
    for endpoint in [0, 10] {
        let x = path[(endpoint, 0)];
        let coupling = path[(endpoint, 1)] - 0.5 * x;
        let energy = (x * x - 1.0).powi(2) + coupling * coupling;
        assert!((energy - 0.25).abs() < f64::EPSILON * 16.0);
    }
    let left_slope = path[(4, 1)] / path[(4, 0)];
    let endpoint_slope = path[(0, 1)] / path[(0, 0)];
    assert!((left_slope - endpoint_slope).abs() > 0.02);
}

#[test]
fn uneven_on_path_images_do_not_converge_with_a_spring_residual() {
    let turning_point = (1.0_f64 - 0.25_f64.sqrt()).sqrt();
    let target = (turning_point * turning_point - 1.0).powi(2);
    let positions = array![
        [-turning_point, 0.0],
        [-0.65, 0.0],
        [-0.05, 0.0],
        [0.2, 0.0],
        [turning_point, 0.0],
    ];
    let config = LineNebConfig::default()
        .with_energy(target)
        .with_masses(array![1.0, 1.0])
        .with_force_tol(1.0e-8)
        .with_endpoint_force_tol(1.0e-8)
        .with_sum_force_tol(1.0e-8);
    let mut session = LineNebSession::new(config, positions).unwrap();
    let report = session.step(&DoubleWell).unwrap();
    assert_eq!(report.status, LineNebStatus::Running);
    assert!(report.max_interior_perpendicular < f64::EPSILON);
    assert!(report.max_interior_optimized > 0.01);
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
        .with_time_step(0.03);
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
    assert!(converged[(n_images / 2, 0)].abs() < 2.0e-5);
    assert!(converged[(n_images / 2, 1)].abs() < 2.0e-5);
    let saddle_x = converged[(n_images / 2, 0)];
    let saddle_y = converged[(n_images / 2, 1)];
    let saddle_energy = (saddle_x * saddle_x - 1.0).powi(2) + saddle_y * saddle_y;
    assert!((saddle_energy - 1.0).abs() < 2.0e-6);
    for endpoint in [0, n_images - 1] {
        let x = converged[(endpoint, 0)];
        let y = converged[(endpoint, 1)];
        let energy = (x * x - 1.0).powi(2) + y * y;
        assert!((energy - target).abs() < 2.0e-6);
    }
    assert!(report.max_interior_perpendicular <= 2.0e-6);
    assert!(report.max_interior_optimized <= 2.0e-6);
    assert!(report.max_endpoint_optimized <= 2.0e-6);
    assert!(report.sum_interior_perpendicular <= 2.0e-6);
}
