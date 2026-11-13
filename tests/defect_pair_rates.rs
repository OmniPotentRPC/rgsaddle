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
