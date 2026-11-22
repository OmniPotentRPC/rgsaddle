use ndarray::Array1;
use rgsaddle::{Cell, NptConfig, NptReport, SaddleError, npt_lattice};

fn one_atom(a: f64) -> (Array1<f64>, Cell) {
    (
        Array1::from_vec(vec![0.25 * a, 0.25 * a, 0.25 * a]),
        Cell([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
    )
}

#[test]
fn npt_reports_block_standard_errors_and_zero_pressure_convergence() {
    let (mut positions, mut cell) = two_atoms(3.6);
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
    let (mut positions, mut cell) = two_atoms(3.6);
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
    let (mut positions, mut cell) = two_atoms(3.6);
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

const KT_300_EV: f64 = rgsaddle::BOLTZMANN_EV_PER_K * 300.0;

fn grid(per_side: usize, spacing: f64) -> (Array1<f64>, Cell) {
    let mut positions = Vec::with_capacity(3 * per_side.pow(3));
    for i in 0..per_side {
        for j in 0..per_side {
            for k in 0..per_side {
                positions.extend([i as f64, j as f64, k as f64].map(|index| index * spacing));
            }
        }
    }
    let edge = per_side as f64 * spacing;
    (
        Array1::from_vec(positions),
        Cell([[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]]),
    )
}

#[test]
fn ideal_gas_pressure_is_the_kinetic_density() {
    let (mut positions, mut cell) = grid(5, 4.0);
    let atoms = positions.len() / 3;
    let volume = 20.0_f64.powi(3);
    let config = NptConfig::default()
        .with_repeats(5)
        .with_steps(4_000)
        .with_maximum_steps(4_000)
        .with_equilibration_steps(1_000)
        .with_block_steps(100)
        .with_compressibility_per_ev_a3(1.0e-12)
        .with_seed(3);

    let report = npt_lattice(&mut positions, &mut cell, 300.0, 63.546, config, |x, _| {
        Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
    })
    .unwrap();

    // The thermostat removes the centre-of-mass momentum, which leaves
    // 3 (N - 1) kinetic degrees of freedom.
    let expected = (atoms - 1) as f64 * KT_300_EV / volume;
    eprintln!(
        "ideal gas: N {atoms} V {volume} P {:.6e} SE {:.2e} expected {expected:.6e} ratio {:.4}",
        report.mean_pressure,
        report.pressure_standard_error,
        report.mean_pressure / expected
    );
    assert!(
        (report.mean_pressure / expected - 1.0).abs() < 0.03,
        "{} vs {expected}",
        report.mean_pressure
    );
    assert!(
        (report.mean_pressure - expected).abs() < 4.0 * report.pressure_standard_error,
        "{} vs {expected}, SE {}",
        report.mean_pressure,
        report.pressure_standard_error
    );
}

/// Atoms tethered to sites that scale with the cell, with no other
/// interaction. At fixed fractional coordinates the tether energy is
/// `k |u|^2 / 2` with `u` scaling like the cell, so the configurational
/// stress is `k u u^T / V`. Equipartition makes its hydrostatic part
/// `N kT / V` and the virial theorem makes the total pressure zero at
/// every volume. Without the kinetic term the sampled pressure is
/// `-N kT / V`.
#[test]
fn harmonic_crystal_has_zero_total_pressure() {
    let (mut positions, mut cell) = grid(6, 3.6);
    let sites = positions.clone();
    let atoms = positions.len() / 3;
    let spring = 2.0;
    let edge0 = cell.0[0][0];
    let volume = edge0.powi(3);
    let config = NptConfig::default()
        .with_repeats(6)
        .with_steps(6_000)
        .with_maximum_steps(6_000)
        .with_equilibration_steps(2_000)
        .with_block_steps(200)
        .with_seed(5);

    let mut configurational = 0.0;
    let mut samples = 0usize;
    let report = npt_lattice(&mut positions, &mut cell, 300.0, 63.546, config, |x, h| {
        let volume = h.0[0][0] * h.0[1][1] * h.0[2][2];
        let scale = h.0[0][0] / edge0;
        let mut gradient = Array1::zeros(x.len());
        let mut stress = [[0.0; 3]; 3];
        let mut energy = 0.0;
        for atom in 0..x.len() / 3 {
            let mut u = [0.0; 3];
            for c in 0..3 {
                u[c] = x[3 * atom + c] - scale * sites[3 * atom + c];
                gradient[3 * atom + c] = spring * u[c];
            }
            energy += 0.5 * spring * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            for a in 0..3 {
                for b in 0..3 {
                    stress[a][b] += spring * u[a] * u[b] / volume;
                }
            }
        }
        configurational += -(stress[0][0] + stress[1][1] + stress[2][2]) / 3.0;
        samples += 1;
        Ok((energy, gradient, stress))
    })
    .unwrap();

    let nkt_over_v = atoms as f64 * KT_300_EV / volume;
    eprintln!(
        "harmonic: N {atoms} N kT/V {nkt_over_v:.6e} P {:.6e} SE {:.2e} P/NkTV {:.4} \
         configurational-only P/NkTV {:.4} a {:.5}",
        report.mean_pressure,
        report.pressure_standard_error,
        report.mean_pressure / nkt_over_v,
        configurational / samples as f64 / nkt_over_v,
        report.mean_lattice_a
    );
    assert!(report.mean_pressure.abs() < 0.1 * nkt_over_v);
    assert!(report.pressure_converged);
    assert!((report.mean_lattice_a / 3.6 - 1.0).abs() < 2.0e-3);
}

#[test]
fn ideal_gas_settles_at_the_target_pressure() {
    let (mut positions, mut cell) = grid(5, 4.0);
    let atoms = positions.len() / 3;
    let target = 2.0e-4;
    let config = NptConfig::default()
        .with_repeats(5)
        .with_steps(20_000)
        .with_maximum_steps(20_000)
        .with_equilibration_steps(5_000)
        .with_block_steps(500)
        .with_target_pressure_ev_a3(target)
        .with_compressibility_per_ev_a3(100.0)
        .with_berendsen_tau_s(2.0e-14)
        .with_seed(9);

    let report = npt_lattice(&mut positions, &mut cell, 300.0, 63.546, config, |x, _| {
        Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
    })
    .unwrap();

    let volume = (atoms - 1) as f64 * KT_300_EV / target;
    let expected = volume.cbrt() / 5.0;
    eprintln!(
        "target pressure: P {:.6e} target {target:.1e} a {:.5} expected {expected:.5}",
        report.mean_pressure, report.mean_lattice_a
    );
    assert!((report.mean_pressure / target - 1.0).abs() < 0.03);
    assert!((report.mean_lattice_a / expected - 1.0).abs() < 0.01);
}

#[test]
fn consecutive_seeds_sample_different_trajectories() {
    let mut pressures = Vec::new();
    for seed in 0..6_u64 {
        let (mut positions, mut cell) = grid(3, 4.0);
        let config = NptConfig::default()
            .with_repeats(3)
            .with_steps(400)
            .with_maximum_steps(400)
            .with_equilibration_steps(100)
            .with_block_steps(50)
            .with_compressibility_per_ev_a3(1.0e-12)
            .with_seed(seed);
        let report = npt_lattice(&mut positions, &mut cell, 300.0, 63.546, config, |x, _| {
            Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
        })
        .unwrap();
        pressures.push(report.mean_pressure);
    }
    for (a, first) in pressures.iter().enumerate() {
        for second in &pressures[a + 1..] {
            assert_ne!(first, second);
        }
    }
}

#[test]
fn configuration_is_built_from_defaults_with_setters() {
    let config = NptConfig::default()
        .with_timestep_s(1.0e-15)
        .with_langevin_gamma_per_s(2.0e12)
        .with_berendsen_tau_s(1.0e-12)
        .with_compressibility_per_ev_a3(0.5)
        .with_target_pressure_ev_a3(1.0e-4)
        .with_seed(0);
    assert_eq!(config.timestep_s, 1.0e-15);
    assert_eq!(config.langevin_gamma_per_s, 2.0e12);
    assert_eq!(config.berendsen_tau_s, 1.0e-12);
    assert_eq!(config.compressibility_per_ev_a3, 0.5);
    assert_eq!(config.target_pressure_ev_a3, 1.0e-4);
    assert_eq!(config.seed, 0);
    assert_eq!(config.repeats, NptConfig::default().repeats);
}

fn two_atoms(a: f64) -> (Array1<f64>, Cell) {
    (
        Array1::from_vec(vec![
            0.25 * a,
            0.25 * a,
            0.25 * a,
            0.75 * a,
            0.75 * a,
            0.75 * a,
        ]),
        Cell([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
    )
}

fn short_config() -> NptConfig {
    NptConfig::default()
        .with_repeats(1)
        .with_steps(40)
        .with_maximum_steps(40)
        .with_equilibration_steps(20)
        .with_block_steps(5)
        .with_seed(11)
}

/// Run on a free two-atom cell and count the evaluator calls.
fn run_free(
    positions: &mut Array1<f64>,
    cell: &mut Cell,
    temperature_k: f64,
    mass_amu: f64,
    config: NptConfig,
) -> (Result<NptReport, SaddleError>, usize) {
    let mut calls = 0;
    let result = npt_lattice(positions, cell, temperature_k, mass_amu, config, |x, _| {
        calls += 1;
        Ok((0.0, Array1::zeros(x.len()), [[0.0; 3]; 3]))
    });
    (result, calls)
}

type Setter = fn(NptConfig, f64) -> NptConfig;

const NON_FINITE: [f64; 3] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];

#[test]
fn non_finite_controls_are_refused_before_any_evaluation() {
    let controls: [(&str, Setter); 5] = [
        ("timestep", NptConfig::with_timestep_s),
        ("friction", NptConfig::with_langevin_gamma_per_s),
        ("coupling time", NptConfig::with_berendsen_tau_s),
        ("compressibility", NptConfig::with_compressibility_per_ev_a3),
        ("target pressure", NptConfig::with_target_pressure_ev_a3),
    ];
    for (name, set) in controls {
        for value in NON_FINITE {
            let (mut positions, mut cell) = two_atoms(3.6);
            let before = (positions.clone(), cell);
            let (result, calls) = run_free(
                &mut positions,
                &mut cell,
                0.0,
                63.546,
                set(short_config(), value),
            );
            assert!(
                matches!(result, Err(SaddleError::Invalid(_))),
                "{name} {value}: {result:?}"
            );
            assert_eq!(calls, 0, "{name} {value}");
            assert_eq!(positions, before.0, "{name} {value}");
            assert_eq!(cell.0, before.1.0, "{name} {value}");
        }
    }
    for value in NON_FINITE {
        for (temperature, mass) in [(value, 63.546), (300.0, value)] {
            let (mut positions, mut cell) = two_atoms(3.6);
            let (result, calls) =
                run_free(&mut positions, &mut cell, temperature, mass, short_config());
            assert!(matches!(result, Err(SaddleError::Invalid(_))), "{result:?}");
            assert_eq!(calls, 0);
        }
    }
}

#[test]
fn a_negative_target_pressure_is_allowed_and_a_non_finite_one_is_not() {
    let (mut positions, mut cell) = two_atoms(3.6);
    let config = short_config().with_target_pressure_ev_a3(-1.0e-4);
    let (result, _) = run_free(&mut positions, &mut cell, 0.0, 63.546, config);
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn non_finite_positions_or_cell_are_refused_without_evaluating() {
    for value in NON_FINITE {
        let (mut positions, mut cell) = two_atoms(3.6);
        positions[4] = value;
        let (result, calls) = run_free(&mut positions, &mut cell, 300.0, 63.546, short_config());
        assert!(
            matches!(result, Err(SaddleError::NonFinite(_))),
            "positions {value}: {result:?}"
        );
        assert_eq!(calls, 0);

        let (mut positions, mut cell) = two_atoms(3.6);
        cell.0[1][1] = value;
        let (result, calls) = run_free(&mut positions, &mut cell, 300.0, 63.546, short_config());
        assert!(
            matches!(result, Err(SaddleError::NonFinite(_))),
            "cell {value}: {result:?}"
        );
        assert_eq!(calls, 0);
    }
}

#[test]
fn a_cell_without_volume_is_refused_rather_than_divided_by() {
    let flat = [
        Cell([[0.0; 3]; 3]),
        Cell([[3.6, 0.0, 0.0], [3.6, 0.0, 0.0], [0.0, 0.0, 3.6]]),
        Cell([[3.6, 0.0, 0.0], [0.0, 3.6, 0.0], [0.0, 0.0, 0.0]]),
    ];
    for start in flat {
        let (mut positions, mut cell) = two_atoms(3.6);
        cell.0 = start.0;
        let (result, calls) = run_free(&mut positions, &mut cell, 300.0, 63.546, short_config());
        assert!(
            matches!(result, Err(SaddleError::Invalid(_))),
            "{start:?}: {result:?}"
        );
        assert_eq!(calls, 0);
        assert_eq!(cell.0, start.0);
    }
}

#[test]
fn a_barostat_that_overflows_is_an_error_and_not_a_nan_report() {
    // The coupling constant overflows to infinity, and infinity times a
    // zero pressure drive is NaN. The run used to return a report whose
    // mean lattice parameter was NaN and whose standard error was zero.
    let (mut positions, mut cell) = two_atoms(3.6);
    let config = short_config()
        .with_compressibility_per_ev_a3(f64::MAX)
        .with_berendsen_tau_s(1.0e-30);
    let (result, _) = run_free(&mut positions, &mut cell, 0.0, 63.546, config);
    assert!(
        matches!(result, Err(SaddleError::NonFinite(_))),
        "{result:?}"
    );
}

#[test]
fn a_non_finite_evaluator_result_is_an_error() {
    let (mut positions, mut cell) = two_atoms(3.6);
    let result = npt_lattice(
        &mut positions,
        &mut cell,
        0.0,
        63.546,
        short_config(),
        |x, _| Ok((0.0, Array1::zeros(x.len()), [[f64::NAN; 3]; 3])),
    );
    assert!(matches!(result, Err(SaddleError::NonFinite(_))));

    let (mut positions, mut cell) = two_atoms(3.6);
    let result = npt_lattice(
        &mut positions,
        &mut cell,
        0.0,
        63.546,
        short_config(),
        |_, _| Ok((0.0, Array1::zeros(1), [[0.0; 3]; 3])),
    );
    assert!(matches!(result, Err(SaddleError::Shape(_))));
}

#[test]
fn a_single_atom_has_no_thermal_motion_and_is_refused() {
    // With one atom the removed centre-of-mass momentum is all of the
    // momentum, so every velocity is zero at any temperature. The run used
    // to return a converged zero pressure with no spread at 1000 K.
    for temperature in [0.0, 300.0, 1000.0] {
        let (mut positions, mut cell) = one_atom(3.6);
        let before = (positions.clone(), cell);
        let (result, calls) = run_free(
            &mut positions,
            &mut cell,
            temperature,
            63.546,
            short_config(),
        );
        assert!(
            matches!(result, Err(SaddleError::Invalid(_))),
            "{temperature} K: {result:?}"
        );
        assert_eq!(calls, 0);
        assert_eq!(positions, before.0);
        assert_eq!(cell.0, before.1.0);
    }
}

#[test]
fn two_atoms_have_thermal_motion_and_a_positive_kinetic_pressure() {
    let (mut positions, mut cell) = two_atoms(3.6);
    let (result, _) = run_free(&mut positions, &mut cell, 1000.0, 63.546, short_config());
    let report = result.unwrap();
    assert!(report.mean_pressure > 0.0);
    assert!(report.pressure_standard_error > 0.0);
}
