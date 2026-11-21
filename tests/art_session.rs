//! Lennard-Jones 7, the smallest cluster on which Malek and Mousseau's
//! exit count is still a finite first-order census.
//!
//! Malek and Mousseau, Phys. Rev. E 62, 7723 (2000),
//! doi:10.1103/PhysRevE.62.7723, report that ART nouveau finds the
//! first-order exits from a minimum, that each saddle connects two
//! minima, and that the path is reversible. LJ7 is that statement at a
//! size the test can finish: from the pentagonal bipyramid, several host
//! seeds are followed, and every connected event is index 1 in both
//! directions. A quench pair that falls in one basin is `Reverted` and
//! is not counted as an exit.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::{
    ArtConfig, ArtPhase, ArtSeed, ArtSession, ArtStatus, MinModeConfig, MinModeKind, PointSurface,
    RelaxConfig, SaddleError, connection_verdict, full_mask, relax_to_tolerance,
};

struct Lj7 {
    calls: AtomicUsize,
}

impl PointSurface for Lj7 {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let n = x.len() / 3;
        let mut energy = 0.0;
        let mut gradient = Array1::zeros(x.len());
        for i in 0..n {
            for j in (i + 1)..n {
                let mut square = 0.0;
                let mut delta = [0.0; 3];
                for axis in 0..3 {
                    delta[axis] = x[3 * i + axis] - x[3 * j + axis];
                    square += delta[axis] * delta[axis];
                }
                if square <= 0.0 || !square.is_finite() {
                    return Err(SaddleError::NonFinite("Lennard-Jones overlap"));
                }
                let (pair, slope) = lj_pair(square);
                energy += pair;
                for axis in 0..3 {
                    let component = slope * 2.0 * delta[axis];
                    gradient[3 * i + axis] += component;
                    gradient[3 * j + axis] -= component;
                }
            }
        }
        Ok((energy, gradient))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(rigid_modes(x))
    }
}

fn lj_pair(square: f64) -> (f64, f64) {
    let inverse = 1.0 / square;
    let inverse3 = inverse * inverse * inverse;
    let inverse6 = inverse3 * inverse3;
    let energy = 4.0 * (inverse6 - inverse3);
    let slope = 4.0 * (-6.0 * inverse6 * inverse + 3.0 * inverse3 * inverse);
    (energy, slope)
}

fn rigid_modes(x: ArrayView1<f64>) -> Array2<f64> {
    let n = x.len() / 3;
    let mut com = [0.0; 3];
    for atom in 0..n {
        for axis in 0..3 {
            com[axis] += x[3 * atom + axis];
        }
    }
    for value in &mut com {
        *value /= n as f64;
    }
    let mut rows = Vec::new();
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for atom in 0..n {
            row[3 * atom + axis] = 1.0;
        }
        rows.push(row);
    }
    for axis in 0..3 {
        let mut row = Array1::zeros(x.len());
        for atom in 0..n {
            let relative = [
                x[3 * atom] - com[0],
                x[3 * atom + 1] - com[1],
                x[3 * atom + 2] - com[2],
            ];
            let velocity = match axis {
                0 => [0.0, -relative[2], relative[1]],
                1 => [relative[2], 0.0, -relative[0]],
                _ => [-relative[1], relative[0], 0.0],
            };
            row[3 * atom] = velocity[0];
            row[3 * atom + 1] = velocity[1];
            row[3 * atom + 2] = velocity[2];
        }
        rows.push(row);
    }
    let mut basis = Vec::new();
    for mut row in rows {
        for previous in &basis {
            let overlap = row.dot(previous);
            row = &row - &(previous * overlap);
        }
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 && norm.is_finite() {
            basis.push(row / norm);
        }
    }
    let mut out = Array2::zeros((basis.len(), x.len()));
    for (index, row) in basis.iter().enumerate() {
        out.row_mut(index).assign(row);
    }
    out
}

fn pentagonal_bipyramid() -> Array1<f64> {
    let bond = 2.0_f64.powf(1.0 / 6.0);
    let radius = bond / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let height = (bond * bond - radius * radius).sqrt();
    let mut x = Array1::zeros(21);
    x[2] = height;
    x[5] = -height;
    for vertex in 0..5 {
        let angle = 2.0 * std::f64::consts::PI * (vertex as f64) / 5.0;
        let base = 6 + 3 * vertex;
        x[base] = radius * angle.cos();
        x[base + 1] = radius * angle.sin();
    }
    let surface = Lj7 {
        calls: AtomicUsize::new(0),
    };
    let quench = RelaxConfig::default()
        .with_force_tol(0.000_1)
        .with_max_steps(4_000)
        .with_max_atom_move(0.05)
        .with_initial_step(0.05);
    let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
    relax_to_tolerance(&surface, x, cell, &quench)
        .expect("LJ7 bipyramid relaxes")
        .positions
}

fn lj_config() -> ArtConfig {
    let mode = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_saddle_check(None)
        .with_krylov_dim(12)
        .with_max_rotations(16)
        .with_force_tol(0.02)
        .with_max_move(0.08)
        .with_dr(0.001);
    let perpendicular = RelaxConfig::default()
        .with_force_tol(0.08)
        .with_max_steps(600)
        .with_max_atom_move(0.08)
        .with_initial_step(0.04);
    let quench = RelaxConfig::default()
        .with_force_tol(0.001)
        .with_max_steps(4_000)
        .with_max_atom_move(0.08)
        .with_initial_step(0.05);
    ArtConfig::default()
        .with_mode(mode)
        .with_perpendicular(perpendicular)
        .with_quench(quench)
        .with_eigen_threshold(-0.01)
        .with_push_length(0.55)
        .with_push_step(0.2)
        .with_max_pushes(12)
        .with_max_climbs(40)
        .with_push_over(0.2)
        .with_basin_rmsd(0.04)
        .with_duplicate_rmsd(0.04)
        .with_permute_identical(true)
}

/// Host seed: a fixed splitmix stream, with rigid motion removed so the
/// push is an internal deformation. The session itself draws nothing.
fn splitmix_seed(minimum: &Array1<f64>, which: u64) -> ArtSeed {
    let mut state = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(which.wrapping_add(1));
    let mut raw = Array1::zeros(minimum.len());
    for value in raw.iter_mut() {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut mixed = state;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        mixed ^= mixed >> 31;
        let unit = (mixed as f64) / (u64::MAX as f64);
        *value = unit * 2.0 - 1.0;
    }
    let modes = rigid_modes(minimum.view());
    for row in modes.rows() {
        let overlap = raw.dot(&row);
        raw = &raw - &(&row * overlap);
    }
    ArtSeed {
        seed: raw,
        mask: full_mask(minimum.len()),
    }
}

#[test]
fn lj7_exits_are_index_one_reversible_and_reverts_fail() {
    let minimum = pentagonal_bipyramid();
    let surface = Lj7 {
        calls: AtomicUsize::new(0),
    };
    let (_, gradient) = surface.eval(minimum.view()).unwrap();
    let force = gradient.iter().map(|value| value.abs()).fold(0.0, f64::max);
    assert!(force < 0.001, "reactant force {force}");
    let (_, energy_only) = {
        let (energy, _) = Lj7 {
            calls: AtomicUsize::new(0),
        }
        .eval(minimum.view())
        .unwrap();
        ((), energy)
    };
    // Wales LJ7 pentagonal bipyramid, same 4ε potential, ε = σ = 1.
    assert!(
        (energy_only + 16.505384).abs() < 0.01,
        "reactant energy {energy_only}"
    );

    let seeds = [
        splitmix_seed(&minimum, 0),
        splitmix_seed(&minimum, 1),
        splitmix_seed(&minimum, 2),
        splitmix_seed(&minimum, 3),
        splitmix_seed(&minimum, 4),
        splitmix_seed(&minimum, 5),
    ];
    let config = lj_config();
    surface.calls.store(0, Ordering::Relaxed);
    let mut outcomes = Vec::new();
    let mut lost = 0;
    let mut exposed = 0;
    for seed in &seeds {
        let mut session = ArtSession::new(
            config.clone(),
            minimum.clone(),
            seed.seed.clone(),
            seed.mask.clone(),
        )
        .expect("LJ7 event starts");
        let run = session.run(&surface, 400);
        exposed += session.force_call_count();
        match run {
            Ok(_) => outcomes.push(session.outcome()),
            Err(SaddleError::ArtIndexOneLost { .. }) => lost += 1,
            Err(error) => panic!("LJ7 event: {error}"),
        }
    }
    assert_eq!(exposed, surface.calls.load(Ordering::Relaxed));
    let mut connected = 0;
    let mut reverted = 0;
    let mut lines = Vec::new();
    for (index, outcome) in outcomes.iter().enumerate() {
        lines.push(format!(
            "seed {index}: {:?} pushes {} climbs {} index {:?} curv {:?} back {:?} forward {:?} rev {:?}",
            outcome.status,
            outcome.pushes,
            outcome.climbs,
            outcome.saddle_index,
            outcome.curvatures,
            outcome.back_energy,
            outcome.forward_energy,
            outcome.reversibility
        ));
        if outcome
            .reversibility
            .as_ref()
            .is_some_and(|item| item.reverted)
        {
            assert_eq!(
                outcome.status,
                ArtStatus::Reverted,
                "{}",
                lines.last().unwrap()
            );
            reverted += 1;
        }
        if outcome.status != ArtStatus::Connected {
            continue;
        }
        connected += 1;
        assert_eq!(outcome.saddle_index, Some(1), "{}", lines.last().unwrap());
        assert!(outcome.certificate_converged);
        let reversibility = outcome.reversibility.unwrap();
        assert!(
            reversibility.returns_to_reactant,
            "{}",
            lines.last().unwrap()
        );
        assert!(
            reversibility.product_is_distinct,
            "{}",
            lines.last().unwrap()
        );
        assert!(reversibility.downhill, "{}", lines.last().unwrap());
        assert!(!reversibility.reverted);
        assert!(outcome.back_force.unwrap() < 0.002);
        assert!(outcome.forward_force.unwrap() < 0.002);
        let saddle_energy = outcome.saddle_energy.unwrap();
        assert!(outcome.back_energy.unwrap() < saddle_energy);
        assert!(outcome.forward_energy.unwrap() < saddle_energy);
        for phase in [
            ArtPhase::Push,
            ArtPhase::Perpendicular,
            ArtPhase::Lanczos,
            ArtPhase::Climb,
            ArtPhase::Certify,
            ArtPhase::PushOver,
            ArtPhase::Quench,
            ArtPhase::Report,
        ] {
            assert!(
                outcome.phases.contains(&phase),
                "seed {index} skipped {phase:?}"
            );
        }
    }
    assert!(
        connected >= 1,
        "no first-order exit from the LJ7 bipyramid ({reverted} reverts, {lost} index-one losses)\n{}",
        lines.join("\n")
    );

    // The same index-1 pair, judged with a basin wide enough to contain
    // both minima, is a failed revert and is not a completed exit.
    let event = outcomes
        .iter()
        .find(|item| item.status == ArtStatus::Connected)
        .expect("a connected exit");
    let reversibility = event.reversibility.expect("reversibility");
    let wide = reversibility
        .back_rmsd
        .max(reversibility.forward_rmsd)
        .max(reversibility.separation_rmsd)
        + 0.05;
    let back = event.minimum_back.as_ref().expect("back minimum");
    let forward = event.minimum_forward.as_ref().expect("forward minimum");
    let (failed, again) = connection_verdict(
        minimum.view(),
        back.view(),
        forward.view(),
        event.saddle_energy.expect("saddle energy"),
        event.back_energy.expect("back energy"),
        event.forward_energy.expect("forward energy"),
        wide,
        true,
    )
    .expect("wide basin");
    assert_eq!(failed, ArtStatus::Reverted);
    assert!(again.reverted);
    assert_ne!(failed, ArtStatus::Connected);
}
