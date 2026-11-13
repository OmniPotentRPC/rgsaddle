//! Direction labels, a symmetric toward/away control, and the catalogue document.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    BandConfig, CiConfig, DefectPairConfig, DirectionClass, HopTemplate, IndexCheck, MinModeConfig,
    PairSetup, PointSurface, RateCatalogue, RateEntry, SaddleError, SolidDimerConfig,
    classify_separation, defect_pair_rates, pair_separation,
};

#[test]
fn separation_change_picks_the_direction() {
    assert_eq!(
        classify_separation(4.0, 3.0, 0.05).expect("toward"),
        DirectionClass::Toward
    );
    assert_eq!(
        classify_separation(4.0, 5.2, 0.05).expect("away"),
        DirectionClass::Away
    );
    assert_eq!(
        classify_separation(4.0, 4.02, 0.05).expect("flat"),
        DirectionClass::Tangential
    );
    assert!(classify_separation(1.0, f64::NAN, 0.1).is_err());
}

#[test]
fn pair_separation_uses_the_minimum_image() {
    let mut positions = Array1::zeros(6);
    positions[0] = 0.2;
    positions[3] = 9.6;
    let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
    let separation = pair_separation(positions.view(), cell, [0.0, 0.0, 0.0], &[1]).expect("r");
    assert!((separation - 0.4).abs() < 1e-9, "{separation}");
}

struct Cosine;

impl PointSurface for Cosine {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let phase = std::f64::consts::TAU * x[0];
        let energy = 0.25 * (1.0 - phase.cos()) + x[1] * x[1] + x[2] * x[2];
        let mut gradient = Array1::zeros(3);
        gradient[0] = 0.25 * std::f64::consts::TAU * phase.sin();
        gradient[1] = 2.0 * x[1];
        gradient[2] = 2.0 * x[2];
        Ok((energy, gradient))
    }
}

fn control_config() -> DefectPairConfig {
    let search = MinModeConfig {
        force_tol: 1.0e-4,
        max_move: 0.04,
        dr: 1.0e-3,
        max_rotations: 8,
        ..MinModeConfig::default()
    };
    DefectPairConfig {
        dimer: SolidDimerConfig::frozen(search).with_index_check(IndexCheck::first_order()),
        band: BandConfig {
            force_tol: 1.0e-3,
            max_move: 0.05,
            time_step: 0.05,
            climbing: Some(CiConfig::always()),
            min_mode: None,
            autoneb: None,
            zoom: None,
            ..BandConfig::default()
        },
        images: 5,
        max_dimer_steps: 40,
        max_band_steps: 30,
        max_relax_steps: 20,
        relax_tol: 1.0e-4,
        relax_step: 0.05,
        max_move: 0.05,
        seed_fraction: 0.35,
        push: 0.2,
        connect_rmsd: 0.15,
        direction_tol: 0.05,
        pafi: rgsaddle::PafiConfig {
            thermalize: 20,
            stride: 2,
            samples: 40,
            blocks: 4,
            time_step: 0.01,
            mobility: 1.0,
            seed: 5,
            fix_com: false,
            retain: 8,
            ..rgsaddle::PafiConfig::default()
        },
        shooting: rgsaddle::ShootingConfig {
            friction: 0.0,
            time_step: 0.01,
            steps: 30,
            record_every: 10,
            shots: 24,
            blocks: 4,
            seed: 9,
            plateau_record: 1,
            ..rgsaddle::ShootingConfig::default()
        },
        temperatures: vec![0.05],
        boltzmann: 1.0,
        mass: 1.0,
        angular_scale: 1.0,
        harmonic_step: 1.0e-4,
        planes: 17,
        ..DefectPairConfig::default()
    }
}

#[test]
fn cosine_pafi_barrier_is_one_half() {
    let frames: Vec<Array1<f64>> = (0..9)
        .map(|i| {
            let x = 5.0 - 0.5 * i as f64 / 8.0;
            Array1::from_vec(vec![x, 0.0, 0.0])
        })
        .collect();
    let path = rgsaddle::ArcPath::from_positions(&frames).unwrap();
    let config = rgsaddle::PafiConfig {
        thermal: 0.05,
        thermalize: 2,
        stride: 1,
        samples: 4,
        blocks: 2,
        time_step: 0.01,
        fix_com: false,
        retain: 1,
        seed: 1,
        ..rgsaddle::PafiConfig::default()
    };
    let (profile, _) = rgsaddle::pafi_profile(&Cosine, &path, &path.arcs(), &config).unwrap();
    assert!((profile.barrier - 0.5).abs() < 0.03, "{}", profile.barrier);
}

#[test]
fn a_symmetric_control_has_equal_toward_and_away_rates() {
    let cell = [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]];
    let setup = PairSetup {
        positions: Array1::from_vec(vec![5.0, 0.0, 0.0]),
        cell,
        fixed: [0.0, 0.0, 0.0],
        mobile: vec![0],
        shell: "control".into(),
        cell_repeats: 0,
        hops: vec![
            HopTemplate {
                name: "translate".into(),
                product: Array1::from_vec(vec![4.0, 0.0, 0.0]),
                saddle: None,
                mode: None,
                orientation: String::new(),
                final_state: String::new(),
            },
            HopTemplate {
                name: "translate".into(),
                product: Array1::from_vec(vec![6.0, 0.0, 0.0]),
                saddle: None,
                mode: None,
                orientation: String::new(),
                final_state: String::new(),
            },
        ],
    };
    let catalogue = defect_pair_rates(&Cosine, &setup, &control_config()).expect("rates");
    assert_eq!(catalogue.entries.len(), 2);
    let toward = &catalogue.entries[0];
    let away = &catalogue.entries[1];
    assert_eq!(toward.direction, DirectionClass::Toward);
    assert_eq!(away.direction, DirectionClass::Away);
    assert!(toward.connected && toward.index1, "toward {:?}", toward.outcome);
    assert!(away.connected && away.index1, "away {:?}", away.outcome);
    let barrier = 0.5;
    for entry in [&toward, &away] {
        let d_e0 = entry.d_e0;
        let d_f = entry.d_f.expect("dF");
        assert!((d_e0 - barrier).abs() < 0.02, "dE0 {d_e0}");
        assert!(
        (d_f - barrier).abs() < 0.05,
        "dF {d_f} ± {} dE0 {} nu {:?} vineyard {:?} harm {:?}",
        entry.d_f_error.unwrap(),
        entry.d_e0,
        entry.nu,
        entry.vineyard,
        entry.d_f_harmonic
    );
        let kappa = entry.kappa.expect("kappa");
        assert!((kappa - 1.0).abs() < 0.05, "kappa {kappa}");
    }
    let ratio = toward.rate.expect("toward rate") / away.rate.expect("away rate");
    assert!(
        (ratio - 1.0).abs() < 0.2,
        "toward/away {ratio} dF {} {}",
        toward.d_f.unwrap(),
        away.d_f.unwrap()
    );
}

#[test]
fn the_catalogue_round_trips() {
    let entry = RateEntry {
        shell: "4.42".into(),
        direction: DirectionClass::Tangential,
        hop: "rotate".into(),
        temperature: 300.0,
        cell_repeats: 10,
        n_atoms: 4000,
        separation_before: 4.42,
        separation_after: 4.42,
        outcome: rgsaddle::HopOutcome::Rated,
        connected: true,
        index1: true,
        band_converged: true,
        reactant_energy: -1.5,
        product_energy: -1.4,
        saddle_energy: -1.2,
        d_e0: 0.3,
        d_f: Some(0.27),
        d_f_error: Some(0.01),
        d_f_harmonic: Some(0.29),
        nu: Some(1.2e12),
        vineyard: Some(1.1e12),
        kappa: Some(0.8),
        kappa_error: Some(0.05),
        harmonic_rate: Some(1.0e7),
        rate: Some(2.0e7),
        zero_t: Some(0.3),
        band_force: 0.004,
        pafi_samples: 80,
        pafi_evaluations: 400,
        saddle_stress: [[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]],
        saddle_dipole: [[-1.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, -3.0]],
        negative_eigenvalue: Some(-0.4),
        orientation: "100".into(),
        final_state: "010-at-neighbor".into(),
    };
    let catalogue = RateCatalogue {
        entries: vec![entry],
    };
    let parsed = RateCatalogue::from_json(&catalogue.to_json()).expect("parse");
    let back = &parsed.entries[0];
    assert_eq!(back.shell, "4.42");
    assert_eq!(back.direction, DirectionClass::Tangential);
    assert_eq!(back.hop, "rotate");
    assert_eq!(back.cell_repeats, 10);
    assert_eq!(back.n_atoms, 4000);
    assert!(back.connected && back.index1 && back.band_converged);
    assert!((back.d_e0 - 0.3).abs() < 1e-12);
    assert!((back.d_f.unwrap() - 0.27).abs() < 1e-12);
    assert!((back.d_f_error.unwrap() - 0.01).abs() < 1e-12);
    assert!((back.kappa.unwrap() - 0.8).abs() < 1e-12);
    assert!((back.rate.unwrap() - 2.0e7).abs() < 1.0);
    assert!((back.zero_t.unwrap() - 0.3).abs() < 1e-12);
    assert!((back.band_force - 0.004).abs() < 1e-12);
    assert_eq!(back.pafi_samples, 80);
    assert_eq!(back.pafi_evaluations, 400);
    assert_eq!(back.orientation, "100");
    assert_eq!(back.final_state, "010-at-neighbor");
    assert!((back.saddle_dipole[1][1] + 2.0).abs() < 1e-12);
    assert!((back.negative_eigenvalue.unwrap() + 0.4).abs() < 1e-12);
}

#[test]
fn kinetic_volume_counts_return_from_the_outer_shell() {
    let sites = vec![
        rgsaddle::VolumeSite {
            id: "inner".into(),
            count: 6.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("recombined".into(), 9.0), ("outer".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "outer".into(),
            count: 12.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("inner".into(), 1.0), ("escaped".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "nearest".into(),
            count: 12.0,
            outcome: rgsaddle::SiteOutcome::Spontaneous,
            rates: Vec::new(),
        },
    ];
    let volume = rgsaddle::kinetic_recombination_volume(&sites, 1.0).expect("volume");
    let inner = 0.9 / 0.95;
    let outer = 0.45 / 0.95;
    let expected = 12.0 + 6.0 * inner + 12.0 * outer;
    assert!((volume.total_sites - expected).abs() < 1.0e-9, "{}", volume.total_sites);
    assert!((volume.spontaneous_sites - 12.0).abs() < 1.0e-12);
    assert!((volume.volume - expected).abs() < 1.0e-9);
    let nearest = volume
        .placements
        .iter()
        .find(|site| site.id == "nearest")
        .expect("spontaneous site is kept");
    assert_eq!(nearest.outcome, rgsaddle::SiteOutcome::Spontaneous);
    assert!((nearest.probability - 1.0).abs() < 1.0e-12);
    assert_eq!(nearest.outcome.as_str(), "spontaneous");
    assert!(rgsaddle::capture_is_spontaneous(true, None));
    assert!(rgsaddle::capture_is_spontaneous(false, Some(-0.0028)));
    assert!(rgsaddle::capture_is_spontaneous(false, Some(0.001)));
    assert!(!rgsaddle::capture_is_spontaneous(false, Some(0.0064)));
}

#[test]
fn free_migration_return_raises_the_capture_probability() {
    let bare = vec![rgsaddle::VolumeSite {
        id: "boundary".into(),
        count: 10.0,
        outcome: rgsaddle::SiteOutcome::Kinetic,
        rates: vec![("recombined".into(), 9.0), ("escaped".into(), 1.0)],
    }];
    let without = rgsaddle::kinetic_recombination_volume(&bare, 1.0).expect("bare");
    assert!((without.total_sites - 9.0).abs() < 1.0e-9);
    let embedded = rgsaddle::embed_free_migration(
        &bare,
        &rgsaddle::FreeMigration {
            boundary: "boundary".into(),
            outward_rate: 1.0,
            return_probability: 0.5,
            return_count: 4.0,
        },
    )
    .expect("embed");
    let with = rgsaddle::kinetic_recombination_volume(&embedded, 1.0).expect("embedded");
    // P_b = 0.9 / 0.95, P_layer = 0.5 P_b, volume = 10 P_b + 4 P_layer.
    let probability = 0.9 / 0.95;
    let expected = 10.0 * probability + 4.0 * 0.5 * probability;
    assert!(
        (with.total_sites - expected).abs() < 1.0e-9,
        "{}",
        with.total_sites
    );
    assert!(with.total_sites > without.total_sites);
}

#[test]
fn a_hop_into_a_spontaneous_shell_is_capture() {
    // The neighbor shell recombines on relaxation. A hop onto it is capture,
    // not an escape under an unknown name.
    let sites = vec![
        rgsaddle::VolumeSite {
            id: "outer".into(),
            count: 8.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("touch".into(), 2.0), ("escaped".into(), 2.0)],
        },
        rgsaddle::VolumeSite {
            id: "touch".into(),
            count: 6.0,
            outcome: rgsaddle::SiteOutcome::Spontaneous,
            rates: Vec::new(),
        },
    ];
    let volume = rgsaddle::kinetic_recombination_volume(&sites, 2.0).expect("volume");
    let outer = volume
        .placements
        .iter()
        .find(|site| site.id == "outer")
        .expect("outer");
    assert!((outer.probability - 0.5).abs() < 1.0e-12, "{}", outer.probability);
    assert!((volume.spontaneous_sites - 6.0).abs() < 1.0e-12);
    assert!((volume.kinetic_sites - 4.0).abs() < 1.0e-12, "{}", volume.kinetic_sites);
    assert!((volume.volume - 20.0).abs() < 1.0e-12, "{}", volume.volume);
}

/// Absorbing chain on three sites.
///
/// Site 1 hops to recombination at rate 3 and to site 2 at rate 1.
/// Site 2 hops to site 1 and to site 3 at rate 1 each.
/// Site 3 hops to site 2 at rate 1 and escapes at rate 2.
///
/// The capture probabilities `u` solve
///
/// `u1 = 3/4 + (1/4) u2`,
/// `u2 = (1/2) u1 + (1/2) u3`,
/// `u3 = (1/3) u2`,
///
/// so `u = (15/17, 9/17, 3/17)`.
fn chain_sites() -> Vec<rgsaddle::VolumeSite> {
    vec![
        rgsaddle::VolumeSite {
            id: "s1".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("recombined".into(), 3.0), ("s2".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "s2".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("s1".into(), 1.0), ("s3".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "s3".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("s2".into(), 1.0), ("escaped".into(), 2.0)],
        },
    ]
}

fn probability(volume: &rgsaddle::RecombinationVolume, id: &str) -> f64 {
    volume
        .placements
        .iter()
        .find(|site| site.id == id)
        .expect(id)
        .probability
}

#[test]
fn the_chain_matches_the_closed_form() {
    let volume = rgsaddle::kinetic_recombination_volume(&chain_sites(), 1.0).expect("volume");
    let exact = [15.0 / 17.0, 9.0 / 17.0, 3.0 / 17.0];
    for (id, expected) in [("s1", exact[0]), ("s2", exact[1]), ("s3", exact[2])] {
        let got = probability(&volume, id);
        assert!((got - expected).abs() < 1.0e-12, "{id} {got} {expected}");
    }
    let sites = exact.iter().sum::<f64>();
    assert!((volume.kinetic_sites - sites).abs() < 1.0e-12);
    assert!((volume.total_sites - sites).abs() < 1.0e-12);
}

#[test]
fn the_chain_matches_kinetic_monte_carlo() {
    let sites = chain_sites();
    let volume = rgsaddle::kinetic_recombination_volume(&sites, 1.0).expect("volume");
    let mut rng = 0x5eed_u64;
    let mut next = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 11) as f64 * (1.0 / ((1u64 << 53) as f64))
    };
    let trajectories = 20_000usize;
    for (start, id) in [(0usize, "s1"), (1, "s2"), (2, "s3")] {
        let mut captured = 0usize;
        for _ in 0..trajectories {
            let mut state = start;
            for _step in 0..10_000 {
                let (rates, total) = match state {
                    0 => ([(3.0, true, None), (1.0, false, Some(1))], 4.0),
                    1 => ([(1.0, false, Some(0)), (1.0, false, Some(2))], 2.0),
                    _ => ([(1.0, false, Some(1)), (2.0, false, None)], 3.0),
                };
                let draw = next() * total;
                let mut cumulative = 0.0;
                let mut done = None;
                for (rate, capture, next_state) in rates {
                    cumulative += rate;
                    if draw <= cumulative {
                        if capture {
                            captured += 1;
                            done = Some(true);
                        } else if let Some(next_state) = next_state {
                            state = next_state;
                            done = Some(false);
                        } else {
                            done = Some(true);
                        }
                        break;
                    }
                }
                if done == Some(true) {
                    break;
                }
            }
        }
        let frequency = captured as f64 / trajectories as f64;
        let exact = probability(&volume, id);
        let sigma = (exact * (1.0 - exact) / trajectories as f64).sqrt();
        assert!(
            (frequency - exact).abs() < 4.0 * sigma.max(1.0e-3),
            "{id} kmc {frequency} chain {exact} sigma {sigma}"
        );
    }
}

#[test]
fn the_return_layer_matches_its_two_state_solution() {
    let bare = vec![rgsaddle::VolumeSite {
        id: "boundary".into(),
        count: 1.0,
        outcome: rgsaddle::SiteOutcome::Kinetic,
        rates: vec![("recombined".into(), 1.0), ("escaped".into(), 1.0)],
    }];
    let rho = rgsaddle::spherical_return_probability(2.0, 2.0).expect("rho");
    assert!((rho - 0.5).abs() < 1.0e-15);
    let embedded = rgsaddle::embed_free_migration(
        &bare,
        &rgsaddle::FreeMigration {
            boundary: "boundary".into(),
            outward_rate: 1.0,
            return_probability: rho,
            return_count: 2.0,
        },
    )
    .expect("embed");
    let volume = rgsaddle::kinetic_recombination_volume(&embedded, 1.0).expect("volume");
    // u_L = ρ u_B, u_B = 1/2 + (1/2) u_L, so u_B = 2/3 and u_L = 1/3.
    assert!((probability(&volume, "boundary") - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((probability(&volume, rgsaddle::FREE_CRYSTAL) - 1.0 / 3.0).abs() < 1.0e-12);
    assert!((volume.total_sites - (2.0 / 3.0 + 2.0 / 3.0)).abs() < 1.0e-12);
}

#[test]
fn repeated_site_ids_are_rejected() {
    let sites = vec![
        rgsaddle::VolumeSite {
            id: "same".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("escaped".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "same".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("escaped".into(), 1.0)],
        },
    ];
    assert!(rgsaddle::kinetic_recombination_volume(&sites, 1.0).is_err());
}

#[test]
fn the_product_normal_points_from_reactant_to_product() {
    let frames = vec![
        Array1::from_vec(vec![0.0, 0.0, 0.0]),
        Array1::from_vec(vec![1.0, 0.0, 0.0]),
        Array1::from_vec(vec![1.0, 1.0, 0.0]),
    ];
    let path = rgsaddle::ArcPath::from_positions(&frames).expect("path");
    let normal = rgsaddle::product_tangent(&path, 0.5).expect("normal");
    assert!(normal[0] > 0.9, "{normal:?}");
    assert!(normal[1].abs() < 1.0e-9, "{normal:?}");
    let end = path.bounds().1;
    let last = rgsaddle::product_tangent(&path, end).expect("last");
    assert!(last[1] > 0.9, "{last:?}");
}

#[test]
fn mean_absorption_time_of_one_hop_is_the_inverse_rate() {
    let sites = vec![rgsaddle::VolumeSite {
        id: "only".into(),
        count: 4.0,
        outcome: rgsaddle::SiteOutcome::Kinetic,
        rates: vec![("recombined".into(), 2.0)],
    }];
    let time = rgsaddle::mean_absorption_time(&sites).expect("time");
    assert!((time - 0.5).abs() < 1.0e-9, "{time}");
}

fn fcc_neighbors(a: f64) -> Vec<[f64; 3]> {
    let mut hops = Vec::new();
    for sx in [0.5, -0.5] {
        for sy in [0.5, -0.5] {
            hops.push([sx * a, sy * a, 0.0]);
            hops.push([sx * a, 0.0, sy * a]);
            hops.push([0.0, sx * a, sy * a]);
        }
    }
    hops
}

fn axis_name(axis: [f64; 3]) -> &'static str {
    if axis[0].abs() > 0.5 {
        "100"
    } else if axis[1].abs() > 0.5 {
        "010"
    } else {
        "001"
    }
}

#[test]
fn a_100_dumbbell_translates_on_eight_channels_and_rotates() {
    let hops = fcc_neighbors(1.0);
    let axis = [0.0, 0.0, 1.0];
    let mut targets = Vec::new();
    for hop in &hops {
        if let Some(next) = rgsaddle::dumbbell_translation_axis(axis, *hop) {
            let back = rgsaddle::dumbbell_translation_axis(next, [-hop[0], -hop[1], -hop[2]])
                .expect("the reverse hop is a translation");
            assert_eq!(axis_name(back), "001", "{hop:?} -> {next:?} -> {back:?}");
            assert_ne!(axis_name(next), "001");
            targets.push(axis_name(next));
        }
    }
    assert_eq!(targets.len(), 8, "{targets:?}");
    assert_eq!(targets.iter().filter(|label| **label == "100").count(), 4);
    assert_eq!(targets.iter().filter(|label| **label == "010").count(), 4);
    // The four hops in the plane normal to the axis are not this saddle.
    assert_eq!(hops.len() - targets.len(), 4);
}

#[test]
fn along_line_and_oblique_hops_are_disjoint_and_balance() {
    // Two FCC sites, one nearest-neighbor step apart. Each direction is
    // either along the vacancy line or oblique, never both, and the raw
    // hop counts satisfy n_i w_ij = n_j w_ji.
    let a = 2.0_f64;
    let origin = [0.0, 0.0, 0.0];
    let neighbor = [a * 0.5, 0.0, a * 0.5];
    let hop = [
        neighbor[0] - origin[0],
        neighbor[1] - origin[1],
        neighbor[2] - origin[2],
    ];
    let back = [-hop[0], -hop[1], -hop[2]];
    let axis = [0.0, 0.0, 1.0];
    let forward = rgsaddle::dumbbell_translation_axis(axis, hop).expect("forward");
    let reverse_axis = rgsaddle::dumbbell_translation_axis(forward, back).expect("reverse");
    assert_eq!(axis_name(reverse_axis), "001");
    let r0 = 0.0;
    let r1 = (neighbor[0] * neighbor[0] + neighbor[2] * neighbor[2]).sqrt();
    let hop_len = (hop[0] * hop[0] + hop[2] * hop[2]).sqrt();
    let radial = (r1 - r0).abs() >= 0.5 * hop_len;
    let oblique = (r1 - r0).abs() < 0.5 * hop_len;
    assert!(radial ^ oblique);
    // Equal multiplicities, one hop each way, common rate 3.
    // The away barrier is four times the free rate; both directions take it.
    let factor = rgsaddle::shell_pair_factor(rgsaddle::DirectionClass::Away, 4.0, 1.0);
    let reverse = rgsaddle::shell_pair_factor(rgsaddle::DirectionClass::Toward, 1.0, 4.0);
    assert!((factor - 4.0).abs() < 1.0e-12);
    assert!((reverse - factor).abs() < 1.0e-12, "{reverse}");
    let count = 2.0;
    assert!((count * factor - count * reverse).abs() < 1.0e-12);
    // Link rate 4 both ways, capture 1 on the near site, escape 4 on the far
    // site. u_near = (4/5) u_far + 1/5, u_far = (4/8) u_near, so u = (1/3, 1/6).
    let sites = vec![
        rgsaddle::VolumeSite {
            id: "near-001".into(),
            count,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("far-100".into(), 4.0), ("recombined".into(), 1.0)],
        },
        rgsaddle::VolumeSite {
            id: "far-100".into(),
            count,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("near-001".into(), 4.0), ("escaped".into(), 4.0)],
        },
    ];
    let volume = rgsaddle::kinetic_recombination_volume(&sites, 1.0).expect("volume");
    // u_n = 4/5 u_f + 1/5, u_f = 4/8 u_n. => u_n = 1/3, u_f = 1/6.
    let near = probability(&volume, "near-001");
    let far = probability(&volume, "far-100");
    assert!((near - 1.0 / 3.0).abs() < 1.0e-12, "{near}");
    assert!((far - 1.0 / 6.0).abs() < 1.0e-12, "{far}");
    assert!((volume.kinetic_sites - count * (near + far)).abs() < 1.0e-12);
}

#[test]
fn rotation_is_two_saddles_between_orientation_states() {
    // One shell, two axes, multiplicity 1 each. Two <110> saddles into the
    // other axis, each of rate 1, so the edge is 2. Capture and escape are 2.
    // u_x = (2/4) u_y + 2/4, u_y = (2/4) u_x => u_x = 2/3, u_y = 1/3.
    let sites = vec![
        rgsaddle::VolumeSite {
            id: "shell-100".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("shell-010".into(), 2.0), ("recombined".into(), 2.0)],
        },
        rgsaddle::VolumeSite {
            id: "shell-010".into(),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates: vec![("shell-100".into(), 2.0), ("escaped".into(), 2.0)],
        },
    ];
    let volume = rgsaddle::kinetic_recombination_volume(&sites, 1.0).expect("volume");
    assert!((probability(&volume, "shell-100") - 2.0 / 3.0).abs() < 1.0e-12);
    assert!((probability(&volume, "shell-010") - 1.0 / 3.0).abs() < 1.0e-12);
    assert!((volume.kinetic_sites - 1.0).abs() < 1.0e-12);
}

#[test]
fn the_spontaneous_count_follows_the_census() {
    // A hard-coded 34 would give the same total on both lattices. The share
    // is the shell multiplicity of each orientation the census marks.
    let hot = spontaneous_share(&[(12.0, true), (12.0, false), (6.0, true)]);
    let cold = spontaneous_share(&[(12.0, true), (12.0, true), (6.0, false)]);
    assert!((hot - 6.0).abs() < 1.0e-12, "{hot}");
    assert!((cold - 8.0).abs() < 1.0e-12, "{cold}");
    assert!((hot - cold).abs() > 1.0);
}

#[test]
fn eight_channels_set_the_diffusivity_and_rotation_does_not() {
    let lattice = 3.6_f64;
    let channel = 2.0e10_f64;
    let quiet = rgsaddle::dumbbell_diffusivity(lattice, channel, 0.0).expect("D");
    let spinning = rgsaddle::dumbbell_diffusivity(lattice, channel, 5.0e12).expect("D");
    let expected = (2.0 / 3.0) * channel * lattice * lattice;
    assert!((quiet - expected).abs() / expected < 1.0e-12, "{quiet} {expected}");
    assert!((spinning - quiet).abs() / quiet < 1.0e-12, "{spinning}");
    // Twelve neighbor channels at the same jump length would be 3/2 of this.
    let twelve = 1.5 * expected;
    assert!((quiet - twelve).abs() / twelve > 0.2);
}

#[test]
fn the_smoluchowski_radius_does_not_grow_when_the_return_layer_does() {
    let radius = 5.0_f64;
    let jump = 1.0_f64;
    let distance = radius + jump;
    let back = rgsaddle::spherical_return_probability(radius, jump).expect("return");
    let core = vec![rgsaddle::VolumeSite {
        id: "boundary".into(),
        count: 20.0,
        outcome: rgsaddle::SiteOutcome::Kinetic,
        rates: vec![("recombined".into(), 1.0e6), ("escaped".into(), 1.0)],
    }];
    let mut volumes = Vec::new();
    let mut radii = Vec::new();
    for count in [1.0_f64, 80.0] {
        let embedded = rgsaddle::embed_free_migration(
            &core,
            &rgsaddle::FreeMigration {
                boundary: "boundary".into(),
                outward_rate: 1.0,
                return_probability: back,
                return_count: count,
            },
        )
        .expect("embed");
        let volume = rgsaddle::kinetic_recombination_volume(&embedded, 1.0).expect("volume");
        let layer = volume
            .placements
            .iter()
            .find(|site| site.id == rgsaddle::FREE_CRYSTAL)
            .expect("layer");
        let r_eff = rgsaddle::effective_capture_radius(layer.probability, distance).expect("radius");
        volumes.push(volume.total_sites);
        radii.push(r_eff);
    }
    assert!((volumes[1] - volumes[0]).abs() > 1.0, "{volumes:?}");
    assert!((radii[0] - radius).abs() < 1.0e-3, "{}", radii[0]);
    assert!((radii[1] - radii[0]).abs() < 1.0e-9, "{radii:?}");
    let rate = rgsaddle::smoluchowski_rate(1.5, radii[0]).expect("rate");
    let migration = (2.0 / 3.0) * 1.5 / (4.0 * 4.0);
    let volume = rgsaddle::volume_from_frequency(rate, migration).expect("volume");
    let atomic = 4.0_f64.powi(3) / 4.0;
    let per_atom = volume / atomic;
    let closed = 16.0 * std::f64::consts::PI * radius / ((2.0 / 3.0) * 4.0);
    assert!((per_atom - closed).abs() / closed < 1.0e-4, "{per_atom} {closed}");
}

#[test]
fn pure_diffusion_onto_an_absorbing_sphere_returns_that_radius() {
    let absorber = 4.0_f64;
    for outer in [8.0_f64, 12.0, 16.0] {
        let (sites, landings) = absorbing_sphere(absorber, outer);
        let embedded = rgsaddle::embed_sphere_return(&sites, outer, &landings).expect("embed");
        let volume = rgsaddle::kinetic_recombination_volume(&embedded, 1.0).expect("volume");
        let boundary = probability(&volume, &format!("{outer:.0}"));
        let radius = rgsaddle::effective_capture_radius(boundary, outer).expect("radius");
        assert!(
            (radius - absorber).abs() < 1.0e-8,
            "outer {outer}: {radius} != {absorber}"
        );
        // The same outward hops, absorbed instead of returned, shrink the radius,
        // and the shrinkage gets worse as the cutoff moves out.
        let leaked = rgsaddle::kinetic_recombination_volume(&sites, 1.0).expect("leaked");
        let leaked_radius =
            rgsaddle::effective_capture_radius(probability(&leaked, &format!("{outer:.0}")), outer)
                .expect("leaked radius");
        assert!(
            leaked_radius < absorber * 0.5,
            "outer {outer}: leak {leaked_radius}"
        );
    }
}

#[test]
fn a_vanishing_interaction_is_the_free_diffusion_radius() {
    let inner = 4.0_f64;
    let thermal = 0.025_f64;
    let zero = |_radius: f64| 0.0;
    let radius = rgsaddle::debye_radius(inner, thermal, zero).expect("radius");
    assert!((radius - inner).abs() < 1.0e-9, "{radius}");
    for distance in [4.0, 6.0, 10.0, 25.0] {
        let hit = rgsaddle::debye_hit_probability(inner, distance, thermal, zero).expect("hit");
        let free = if distance <= inner { 1.0 } else { inner / distance };
        assert!((hit - free).abs() < 1.0e-8, "{distance} {hit} {free}");
    }
    let matched = rgsaddle::debye_matched_radius(3.5, 9.0, thermal, zero).expect("matched");
    assert!((matched - 3.5).abs() < 1.0e-8, "{matched}");
}

#[test]
fn a_coulomb_interaction_matches_the_debye_closed_form() {
    let inner = 2.0_f64;
    let thermal = 0.5_f64;
    let alpha = 2.0_f64;
    let attractive = |radius: f64| -alpha / radius;
    let radius = rgsaddle::debye_radius(inner, thermal, attractive).expect("radius");
    let onsager = alpha / thermal;
    let closed = onsager / (1.0 - (-onsager / inner).exp());
    assert!((radius - closed).abs() / closed < 1.0e-6, "{radius} {closed}");
    assert!(radius > inner);
    let repulsive = rgsaddle::debye_radius(inner, thermal, |radius| alpha / radius).expect("repulsive");
    let closed_repulsive = onsager / ((onsager / inner).exp() - 1.0);
    assert!(
        (repulsive - closed_repulsive).abs() / closed_repulsive < 1.0e-6,
        "{repulsive} {closed_repulsive}"
    );
    assert!(repulsive < inner);
}

fn absorbing_sphere(
    absorber: f64,
    outer: f64,
) -> (Vec<rgsaddle::VolumeSite>, Vec<rgsaddle::SphereLanding>) {
    let mut sites = vec![rgsaddle::VolumeSite {
        id: format!("{absorber:.0}"),
        count: 1.0,
        outcome: rgsaddle::SiteOutcome::Spontaneous,
        rates: Vec::new(),
    }];
    let mut landings = Vec::new();
    let first = absorber as i32 + 1;
    let last = outer as i32;
    for n in first..=last {
        let inward = (n - 1) as f64;
        let outward = (n + 1) as f64;
        let mut rates = vec![(format!("{}", n - 1), inward)];
        if n == last {
            // Leaving the sphere is an absorbing loss until the return layer is added.
            rates.push(("escaped".into(), outward));
            landings.push(rgsaddle::SphereLanding {
                source: format!("{n}"),
                boundary: format!("{n}"),
                rate: outward,
                distance: (n + 1) as f64,
            });
        } else {
            rates.push((format!("{}", n + 1), outward));
        }
        sites.push(rgsaddle::VolumeSite {
            id: format!("{n}"),
            count: 1.0,
            outcome: rgsaddle::SiteOutcome::Kinetic,
            rates,
        });
    }
    (sites, landings)
}

fn spontaneous_share(rows: &[(f64, bool)]) -> f64 {
    rows.iter()
        .map(|(count, spontaneous)| if *spontaneous { count / 3.0 } else { 0.0 })
        .sum()
}
