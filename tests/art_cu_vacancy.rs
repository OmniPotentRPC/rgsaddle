//! ART units regression on a 63-atom Girifalco-Weizer copper vacancy.

#[path = "common/cu_morse.rs"]
mod cu_morse;

use ndarray::Array1;
use rgsaddle::{
    ArtConfig, ArtSeed, ArtStatus, PointSurface, RelaxConfig, SaddleError, atom_mask,
    relax_to_tolerance, run_events,
};

use cu_morse::{CuMorse, ENERGY_SCALE, LENGTH_SCALE};

fn cu_vacancy_setup() -> (CuMorse, Array1<f64>, Vec<ArtSeed>) {
    let (surface, unrelaxed, vacancy) = CuMorse::vacancy_4x4x4();
    assert_eq!(unrelaxed.len(), 3 * 63);
    let reactant_relax = RelaxConfig::default()
        .with_force_tol(1.0e-4)
        .with_max_steps(4_000)
        .with_max_atom_move(0.05)
        .with_initial_step(0.02);
    let reactant = relax_to_tolerance(
        &surface,
        unrelaxed,
        [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]],
        &reactant_relax,
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
    let first_neighbour_square = neighbours[0].1.dot(&neighbours[0].1);
    assert!(neighbours[..12].iter().all(|(_, displacement)| {
        (displacement.dot(displacement) - first_neighbour_square).abs() < 1.0e-8
    }));
    assert!(neighbours[12].1.dot(&neighbours[12].1) > first_neighbour_square + 1.0);
    let seeds: Vec<ArtSeed> = neighbours
        .into_iter()
        .take(4)
        .map(|(atom, displacement)| {
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

    (surface, reactant, seeds)
}

/// Connected events that are not a repeat of an earlier saddle.
///
/// Duplicates are recorded for every event that reached a saddle,
/// whatever its status, so they are not subtracted from the connected
/// count.
fn distinct_connected(events: impl IntoIterator<Item = (ArtStatus, Option<usize>)>) -> usize {
    events
        .into_iter()
        .filter(|(status, duplicate_of)| *status == ArtStatus::Connected && duplicate_of.is_none())
        .count()
}

#[test]
fn distinct_count_ignores_duplicates_of_unconnected_events() {
    let events = [
        (ArtStatus::Reverted, None),
        (ArtStatus::Reverted, Some(0)),
        (ArtStatus::NotDownhill, Some(0)),
        (ArtStatus::Connected, None),
    ];
    let connected = events
        .iter()
        .filter(|(status, _)| *status == ArtStatus::Connected)
        .count();
    let duplicates = events.iter().filter(|(_, of)| of.is_some()).count();
    assert!(connected.checked_sub(duplicates).is_none());
    assert_eq!(distinct_connected(events), 1);
    assert_eq!(
        distinct_connected([
            (ArtStatus::Connected, None),
            (ArtStatus::Connected, Some(0)),
            (ArtStatus::Irreversible, None),
        ]),
        1
    );
    assert_eq!(distinct_connected([]), 0);
}

fn assert_four_cu_vacancy_saddles(
    surface: &CuMorse,
    reactant: &Array1<f64>,
    seeds: &[ArtSeed],
    config: &ArtConfig,
    label: &str,
) {
    let batch =
        run_events(config, reactant, seeds, surface, 800).expect("ART runs on copper vacancy");
    let reactant_energy = surface.eval(reactant.view()).unwrap().0;
    let connected = batch
        .outcomes
        .iter()
        .filter(|outcome| outcome.status == ArtStatus::Connected)
        .count();
    let distinct = distinct_connected(
        batch
            .outcomes
            .iter()
            .map(|outcome| (outcome.status, outcome.duplicate_of)),
    );
    eprintln!(
        "Cu vacancy ART ({label}): {connected} connected, {distinct} distinct, {} duplicates",
        batch.duplicates.len()
    );
    assert_eq!(connected, 4);
    assert_eq!(distinct, 4);
    const CU01_VACANCY_BARRIER_EV: f64 = 0.550_956;
    const BARRIER_TOLERANCE_EV: f64 = 1.0e-5;
    for outcome in &batch.outcomes {
        assert_eq!(outcome.status, ArtStatus::Connected);
        assert_eq!(outcome.saddle_index, Some(1));
        assert!(outcome.certificate_converged);
        assert!(outcome.curvature < -0.5, "curvature {}", outcome.curvature);
        let barrier = outcome.saddle_energy.unwrap() - reactant_energy;
        assert!(
            (barrier - CU01_VACANCY_BARRIER_EV).abs() <= BARRIER_TOLERANCE_EV,
            "Cu01 vacancy barrier {barrier}"
        );
    }
}

#[test]
fn metal_scaled_art_finds_four_cu_vacancy_saddles() {
    let (surface, reactant, seeds) = cu_vacancy_setup();
    let config = ArtConfig::for_units(ENERGY_SCALE, LENGTH_SCALE).unwrap();
    assert_four_cu_vacancy_saddles(&surface, &reactant, &seeds, &config, "for_units");
}

#[test]
fn default_art_finds_four_cu_vacancy_saddles() {
    let (surface, reactant, seeds) = cu_vacancy_setup();
    let config = ArtConfig::default();
    assert_four_cu_vacancy_saddles(&surface, &reactant, &seeds, &config, "default");
}
