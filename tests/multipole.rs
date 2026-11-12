//! Force multipoles and the quadrupole correction to a barrier.

use rgsaddle::{ElasticMultipole, LocalField, Strain, StrainResponse, quadrupole_energy};

fn shear_pair() -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
    let positions = vec![
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let forces = vec![
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ];
    (positions, forces)
}

#[test]
fn quadrupole_is_the_second_force_moment() {
    let (positions, forces) = shear_pair();
    let moment = ElasticMultipole::about([0.0, 0.0, 0.0], &positions, &forces, 4).unwrap();
    assert_eq!(moment.order(), 4);
    let dipole = moment.dipole().tensor();
    assert!((dipole[0][1] - 2.0).abs() < 1e-12);
    let quadrupole = moment.quadrupole().unwrap();
    // F_y at x = ±1 contributes Q_yxx = 1*1*1 + (-1)*(-1)*(-1) = 1 - 1 = 0
    assert!(quadrupole[1][0][0].abs() < 1e-12);
    let octupole = moment.octupole().unwrap();
    // F_y * X_x^3 at ±1: 1*(1)^3 + (-1)*(-1)^3 = 1 + 1 = 2
    assert!((octupole[1][0][0][0] - 2.0).abs() < 1e-12);
    let shifted = ElasticMultipole::about([0.5, 0.0, 0.0], &positions, &forces, 2).unwrap();
    assert!((shifted.dipole().tensor()[0][1] - dipole[0][1]).abs() < 1e-12);
}

#[test]
fn barrier_adds_the_quadrupole_contraction() {
    let (positions, forces) = shear_pair();
    let saddle = ElasticMultipole::about([0.0; 3], &positions, &forces, 3).unwrap();
    let quiet =
        ElasticMultipole::about([0.0; 3], &[[0.0, 0.0, 0.0]], &[[0.0, 0.0, 0.0]], 3).unwrap();
    let response = StrainResponse::new(1.0, quiet.dipole(), saddle.dipole())
        .unwrap()
        .with_multipoles(quiet, saddle);
    let mut gradient = [[[0.0; 3]; 3]; 3];
    gradient[1][0][0] = 0.25;
    let field = LocalField {
        strain: Strain::zero(),
        gradient: Some(gradient),
    };
    let estimate = response.barrier_in(&field);
    let quadrupole = response_quadrupole_delta(&positions, &forces);
    let expected = 1.0 - quadrupole_energy(quadrupole, gradient);
    assert!(
        (estimate.barrier - expected).abs() < 1e-12,
        "{} vs {expected}",
        estimate.barrier
    );
}

fn response_quadrupole_delta(positions: &[[f64; 3]], forces: &[[f64; 3]]) -> [[[f64; 3]; 3]; 3] {
    ElasticMultipole::about([0.0; 3], positions, forces, 3)
        .unwrap()
        .quadrupole()
        .unwrap()
}
