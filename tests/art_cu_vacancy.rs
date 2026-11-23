//! ART units regression on a 63-atom Girifalco-Weizer copper vacancy.

#[path = "common/cu_morse.rs"]
mod cu_morse;

use ndarray::Array1;
use rgsaddle::{
    ArtConfig, ArtSeed, ArtStatus, RelaxConfig, SaddleError, atom_mask, relax_to_tolerance,
    run_events,
};

use cu_morse::{CuMorse, ENERGY_SCALE, LENGTH_SCALE};

#[test]
fn metal_scaled_art_finds_three_cu_vacancy_saddles() {
    let (surface, unrelaxed, vacancy) = CuMorse::vacancy_4x4x4();
    assert_eq!(unrelaxed.len(), 3 * 63);
    let quench = RelaxConfig::default()
        .with_force_tol(1.0e-4)
        .with_max_steps(4_000)
        .with_max_atom_move(0.05)
        .with_initial_step(0.02);
    let reactant = relax_to_tolerance(
        &surface,
        unrelaxed,
        [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        &quench,
    )
    .expect("the copper vacancy relaxes")
    .positions;

    let mut neighbours: Vec<(usize, Array1<f64>)> = (0..63)
        .map(|atom| {
            let displacement =
                surface.displacement(reactant.slice(ndarray::s![3 * atom..3 * atom + 3]), vacancy);
            (atom, displacement)
        })
        .collect();
    neighbours
        .sort_by(|(_, left), (_, right)| left.dot(left).partial_cmp(&right.dot(right)).unwrap());
    let seeds: Vec<ArtSeed> = neighbours
        .into_iter()
        .enumerate()
        .filter(|(rank, _)| matches!(rank, 0..=2 | 12))
        .map(|(_, (atom, displacement))| {
            let mut seed = Array1::zeros(reactant.len());
            seed.slice_mut(ndarray::s![3 * atom..3 * atom + 3])
                .assign(&displacement);
            ArtSeed {
                seed,
                mask: atom_mask(reactant.len(), &[atom]).unwrap(),
            }
        })
        .collect();
    assert_eq!(seeds.len(), 4);

    let legacy = ArtConfig::lj_art()
        .with_mode(ArtConfig::lj_art().mode.with_force_tol(0.01))
        .with_perpendicular(ArtConfig::lj_art().perpendicular.with_force_tol(0.08));
    assert!(matches!(
        legacy.validate(),
        Err(SaddleError::Invalid(message))
            if message.contains("perpendicular_force_tol=8e-2 energy/length")
                && message.contains("mode_force_tol=1e-2 energy/length")
    ));

    let mut config = ArtConfig::for_units(ENERGY_SCALE, LENGTH_SCALE)
        .unwrap()
        .with_quench(quench)
        .with_max_climbs(80);
    config.saddle.max_actions = 400;
    config.saddle.residual_tol = 0.01;
    let batch = run_events(&config, &reactant, &seeds, &surface, 800)
        .expect("units-aware ART runs on copper");
    let connected = batch
        .outcomes
        .iter()
        .filter(|outcome| outcome.status == ArtStatus::Connected)
        .count();
    let distinct = connected - batch.duplicates.len();
    eprintln!(
        "Cu vacancy ART: {connected} connected, {distinct} distinct, {} duplicates",
        batch.duplicates.len()
    );
    assert!(
        connected == 3 && distinct == 3,
        "expected three distinct saddles, got {connected} connected and {distinct} distinct: {:?}",
        batch
            .outcomes
            .iter()
            .map(|outcome| (
                outcome.status,
                outcome.saddle_index,
                outcome.certificate_converged,
                outcome.curvature,
                outcome.max_force,
                outcome.climbs
            ))
            .collect::<Vec<_>>()
    );
}
