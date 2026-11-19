use ndarray::Array1;
use rgsaddle::{Cell, NptConfig, npt_lattice};

fn one_atom(a: f64) -> (Array1<f64>, Cell) {
    (
        Array1::from_vec(vec![0.25 * a, 0.25 * a, 0.25 * a]),
        Cell([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
    )
}

#[test]
fn npt_reports_block_standard_errors_and_zero_pressure_convergence() {
    let (mut positions, mut cell) = one_atom(3.6);
    let config = NptConfig::default()
        .with_repeats(1)
        .with_steps(40)
        .with_maximum_steps(40)
        .with_equilibration_steps(20)
        .with_block_steps(5)
        .with_seed(11);

    let report = npt_lattice(&mut positions, &mut cell, 0.0, 63.546, config, |x, _| {
        Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
    })
    .unwrap();

    assert_eq!(report.steps, 40);
    assert_eq!(report.blocks, 4);
    assert_eq!(report.mean_lattice_a, 3.6);
    assert_eq!(report.lattice_standard_error, 0.0);
    assert_eq!(report.mean_pressure, 0.0);
    assert_eq!(report.pressure_standard_error, 0.0);
    assert!(report.pressure_converged);
}

#[test]
fn npt_extends_the_run_until_the_pressure_rule_is_evaluable() {
    let (mut positions, mut cell) = one_atom(3.6);
    let config = NptConfig::default()
        .with_repeats(1)
        .with_steps(20)
        .with_maximum_steps(30)
        .with_extension_steps(10)
        .with_equilibration_steps(10)
        .with_block_steps(5)
        .with_seed(11);

    let report = npt_lattice(&mut positions, &mut cell, 0.0, 63.546, config, |x, _| {
        Ok((
            0.0,
            Array1::zeros(x.len()),
            [[-0.01, 0.0, 0.0], [0.0, -0.01, 0.0], [0.0, 0.0, -0.01]],
        ))
    })
    .unwrap();

    assert_eq!(report.steps, 30);
    assert_eq!(report.blocks, 4);
    assert!(!report.pressure_converged);
}

#[test]
fn npt_pressure_includes_the_kinetic_virial() {
    let (mut positions, mut cell) = one_atom(3.6);
    let config = NptConfig::default()
        .with_repeats(1)
        .with_steps(40)
        .with_maximum_steps(40)
        .with_equilibration_steps(20)
        .with_block_steps(5)
        .with_seed(11);

    let report = npt_lattice(&mut positions, &mut cell, 600.0, 63.546, config, |x, _| {
        Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
    })
    .unwrap();

    assert!(report.mean_pressure > 0.0);
}
