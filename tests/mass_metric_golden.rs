//! Stored path values at atom masses 1.008 and 15.999.
//!
//! The figures were read from the metric maps before those maps required
//! equal lengths. A matched length uses the same arithmetic, and each
//! stored value is compared at a relative tolerance of 1e-9.

#![cfg(feature = "ohtst")]

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgsaddle::{
    AnalyticSurfaceDynamics, ArcPath, BandSurface, FixedEnergySeed, FixedEnergyTrajectory,
    Hyperplane, IrcConfig, IrcDirection, IrcKind, IrcSession, LineNebConfig, LineNebSession,
    MassWeighted, PafiConfig, PointSurface, SaddleError, TransmissionConfig, arc_at_time,
    fixed_energy_action, fixed_energy_action_force, fixed_energy_mep_seed,
    mass_weighted_rigid_translations, transmission_coefficient,
};

const LIGHT: f64 = 1.008;
const HEAVY: f64 = 15.999;

fn atom_masses() -> Array1<f64> {
    array![LIGHT, HEAVY]
}

fn coordinate_masses() -> Array1<f64> {
    Array1::from(vec![LIGHT, LIGHT, LIGHT, HEAVY, HEAVY, HEAVY])
}

fn energy_gradient(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    let x0 = x[0];
    let well = 0.25 * (x0 * x0 - 1.0).powi(2);
    let mut transverse = 0.0;
    let mut gradient = Array1::zeros(x.len());
    gradient[0] = x0 * x0 * x0 - x0;
    for index in 1..x.len() {
        transverse += 0.5 * x[index] * x[index];
        gradient[index] = x[index];
    }
    (well + transverse, gradient)
}

struct SaddleWell;

impl PointSurface for SaddleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok(energy_gradient(x))
    }
}

impl BandSurface for SaddleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (image, row) in positions.outer_iter().enumerate() {
            let (energy, gradient) = energy_gradient(row);
            energies[image] = energy;
            gradients.row_mut(image).assign(&gradient);
        }
        Ok(())
    }
}

/// Coupled bowl so a shot along a mixed normal feels both masses.
struct Bowl;

impl PointSurface for Bowl {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let bond = x[0] - x[3];
        let center = x[0] + x[3];
        let mut energy = 0.5 * bond * bond + 0.15 * center * center;
        let mut gradient = Array1::zeros(x.len());
        gradient[0] = bond + 0.3 * center;
        gradient[3] = -bond + 0.3 * center;
        for index in [1, 2, 4, 5] {
            energy += 0.25 * x[index] * x[index];
            gradient[index] = 0.5 * x[index];
        }
        Ok((energy, gradient))
    }
}

fn stored() -> BTreeMap<&'static str, Vec<f64>> {
    let mut map: BTreeMap<&'static str, Vec<f64>> = BTreeMap::new();
    let mut name: Option<&'static str> = None;
    let text = include_str!("data/mass_metric_golden.txt");
    let names = [
        "irc_gs2",
        "irc_morokuma",
        "irc_sella",
        "irc_launch",
        "line_action",
        "line_force",
        "line_session",
        "line_seed",
        "tunnel",
        "pafi",
        "ohtst_surface",
        "plane_maps",
        "shooting",
    ];
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("ITEM ") {
            let key = rest.split_whitespace().next().expect("item name");
            let key = names
                .into_iter()
                .find(|candidate| *candidate == key)
                .unwrap_or_else(|| panic!("unknown item {key}"));
            name = Some(key);
            map.insert(key, Vec::new());
        } else if let Some(key) = name {
            map.get_mut(key)
                .expect("item")
                .push(line.parse().unwrap_or_else(|_| panic!("float {line}")));
        }
    }
    map
}

fn expect_close(stored: &BTreeMap<&'static str, Vec<f64>>, name: &str, got: &[f64]) {
    let expected = stored
        .get(name)
        .unwrap_or_else(|| panic!("missing stored {name}"));
    assert_eq!(got.len(), expected.len(), "{name}");
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        let gap = (got - expected).abs();
        let tolerance = 1.0e-9 * expected.abs();
        assert!(
            gap <= tolerance,
            "{name}[{index}] got {got:.17e} stored {expected:.17e}"
        );
    }
}

fn irc_config(kind: IrcKind) -> IrcConfig {
    IrcConfig::default()
        .with_dx(0.07)
        .with_force_tol(1.0e-12)
        .with_max_inner(8)
        .with_kind(kind)
}

fn launch_mode() -> Array1<f64> {
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;
    mode
}

fn irc_kind(kind: IrcKind, steps: usize) -> Result<Vec<f64>, SaddleError> {
    let mut session = IrcSession::new(
        irc_config(kind),
        Array1::zeros(6),
        atom_masses(),
        launch_mode(),
        IrcDirection::Forward,
    )?;
    let mut out = Vec::new();
    for _ in 0..steps {
        let report = session.step(&SaddleWell)?;
        out.extend(session.position().iter().copied());
        out.push(report.energy);
        out.push(report.max_force);
        out.push(report.arc);
        out.push(report.inner_steps as f64);
        out.push(if report.at_minimum { 1.0 } else { 0.0 });
    }
    Ok(out)
}

fn irc_launch() -> Result<Vec<f64>, SaddleError> {
    let seed = array![0.4, 0.2, -0.1, 0.05, 0.0, 0.3];
    let mut session = IrcSession::from_surface(
        irc_config(IrcKind::SellaSphere)
            .with_dr(1.0e-4)
            .with_krylov_dim(8),
        Array1::zeros(6),
        atom_masses(),
        seed,
        IrcDirection::Forward,
        &SaddleWell,
    )?;
    let report = session.step(&SaddleWell)?;
    let mut out: Vec<f64> = session.position().to_vec();
    out.push(report.energy);
    out.push(report.max_force);
    out.push(report.arc);
    out.push(report.inner_steps as f64);
    Ok(out)
}

fn line_images() -> Array2<f64> {
    array![
        [-0.35, 0.02, -0.01, 0.04, 0.0, 0.01],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.28, -0.03, 0.02, -0.01, 0.05, -0.02],
    ]
}

struct ImageTable {
    positions: Array2<f64>,
    energies: Array1<f64>,
    gradients: Array2<f64>,
}

fn line_table() -> Result<ImageTable, SaddleError> {
    let positions = line_images();
    let mut energies = Array1::zeros(positions.nrows());
    let mut gradients = Array2::zeros(positions.raw_dim());
    BandSurface::eval(&SaddleWell, positions.view(), &mut energies, &mut gradients)?;
    Ok(ImageTable {
        positions,
        energies,
        gradients,
    })
}

fn tunneling() -> Result<Vec<f64>, SaddleError> {
    let positions = line_images();
    let mut gradients = Array2::zeros(positions.raw_dim());
    let mut energies = Array1::zeros(positions.nrows());
    BandSurface::eval(&SaddleWell, positions.view(), &mut energies, &mut gradients)?;
    let trajectory =
        FixedEnergyTrajectory::from_gradients(&positions, &gradients, coordinate_masses().view())?;
    let half = trajectory.half(0.25, 80)?;
    let mut out = Vec::new();
    out.push(half.len() as f64);
    for point in &half {
        out.push(point.time);
        out.push(point.arc);
        out.push(point.velocity);
    }
    let mid = 0.5 * half.last().map(|point| point.time).unwrap_or(0.0);
    out.push(arc_at_time(&half, mid)?);
    out.push(trajectory.period(0.25, 80)?);
    Ok(out)
}

fn pafi_values() -> Result<Vec<f64>, SaddleError> {
    let surface = MassWeighted::per_atom(&SaddleWell, &[LIGHT, HEAVY])?;
    let frames = vec![Array1::zeros(6), array![0.4, 0.0, 0.0, 0.0, 0.0, 0.0]];
    let path = ArcPath::from_positions(&frames)?;
    let arcs = path.arcs();
    let config = PafiConfig::default()
        .with_thermal(0.05)
        .with_mobility(1.0)
        .with_time_step(0.01)
        .with_thermalize(2)
        .with_stride(1)
        .with_samples(4)
        .with_blocks(2)
        .with_seed(5)
        .with_fix_com(false)
        .with_retain(1)
        .with_target_error(None)
        .with_max_rounds(1)
        .with_max_jump(None);
    let (profile, planes) = rgsaddle::pafi_profile(&surface, &path, &arcs, &config)?;
    let mut out = Vec::new();
    out.push(profile.barrier);
    out.push(profile.barrier_error);
    out.extend(profile.free_energy.iter().copied());
    out.extend(profile.gradient.iter().copied());
    for plane in &planes {
        out.push(plane.gradient);
        out.push(plane.error);
        out.push(plane.psi);
    }
    Ok(out)
}

fn ohtst_surface() -> Result<Vec<f64>, SaddleError> {
    let surface = MassWeighted::per_atom(&SaddleWell, &[LIGHT, HEAVY])?;
    let cartesian = array![0.2, -0.1, 0.05, 0.15, 0.0, -0.08];
    let weighted = surface.to_mass_weighted(cartesian.view())?;
    let back = surface.to_cartesian(weighted.view())?;
    let (energy, gradient) = surface.eval(weighted.view())?;
    let mut out = weighted.to_vec();
    out.extend(back.iter().copied());
    out.push(energy);
    out.extend(gradient.iter().copied());
    Ok(out)
}

fn plane_maps() -> Result<Vec<f64>, SaddleError> {
    let masses = [LIGHT, HEAVY];
    let rows = mass_weighted_rigid_translations(&masses, 2)?;
    let mut out = Vec::new();
    out.extend(rows.iter().copied());
    let plane = Hyperplane::new(
        array![0.1, 0.0, 0.0, -0.2, 0.0, 0.0],
        array![1.0, 0.2, -0.1, 0.5, 0.0, 0.3],
    )?;
    let orthogonal = plane.orthogonal_to_mass_weighted_translations(&masses, 2)?;
    out.extend(orthogonal.normal().iter().copied());
    out.extend(orthogonal.point().iter().copied());
    Ok(out)
}

fn shooting() -> Result<Vec<f64>, SaddleError> {
    let plane = Hyperplane::new(
        array![0.12, 0.01, 0.0, 0.05, 0.0, -0.01],
        array![0.55, 0.0, 0.0, 0.35, 0.0, 0.0],
    )?;
    let start = array![0.12, 0.01, 0.0, 0.05, 0.0, -0.01];
    let config = TransmissionConfig::default()
        .with_thermal(0.4)
        .with_friction(0.05)
        .with_time_step(0.1)
        .with_steps(40)
        .with_check_every(10)
        .with_shots(8)
        .with_blocks(2)
        .with_seed(11)
        .with_masses(coordinate_masses().to_vec())
        .with_hold_excluded(false);
    let mut dynamics = AnalyticSurfaceDynamics::new(&Bowl);
    let coefficient = transmission_coefficient(&mut dynamics, &plane, &[start], &config)?;
    let mut out = vec![
        coefficient.kappa,
        coefficient.error,
        coefficient.shots as f64,
        coefficient.unassigned as f64,
        coefficient.reactive as f64,
        coefficient.recrossed as f64,
        coefficient.evaluations as f64,
    ];
    out.extend(coefficient.kappa_of_t.iter().copied());
    Ok(out)
}

#[test]
fn irc_kinds_match_stored_values() {
    let stored = stored();
    for (name, kind) in [
        ("irc_gs2", IrcKind::Gs2),
        ("irc_morokuma", IrcKind::Morokuma),
        ("irc_sella", IrcKind::SellaSphere),
    ] {
        expect_close(&stored, name, &irc_kind(kind, 4).expect(name));
    }
}

#[test]
fn irc_minimum_mode_launch_matches_stored_values() {
    let stored = stored();
    expect_close(
        &stored,
        "irc_launch",
        &irc_launch().expect("minimum-mode launch"),
    );
}

#[test]
fn line_neb_matches_stored_values() {
    let stored = stored();
    let table = line_table().expect("images");
    let masses = coordinate_masses();
    let action = fixed_energy_action(
        table.positions.view(),
        table.energies.view(),
        masses.view(),
        0.05,
        1.0,
    )
    .expect("action");
    expect_close(&stored, "line_action", &[action]);
    let force = fixed_energy_action_force(
        table.positions.view(),
        table.energies.view(),
        table.gradients.view(),
        masses.view(),
        0.05,
        1.0,
    )
    .expect("force");
    expect_close(
        &stored,
        "line_force",
        force.as_slice().expect("force layout"),
    );
    let config = LineNebConfig::default()
        .with_energy(0.05)
        .with_masses(masses)
        .with_time_step(0.05)
        .with_reduced_planck(1.0)
        .with_spring_k(0.4);
    let mut session = LineNebSession::new(config, table.positions).expect("session");
    let report = session.step(&SaddleWell).expect("session step");
    let mut values = session
        .positions()
        .as_slice()
        .expect("position layout")
        .to_vec();
    values.push(report.action);
    values.push(report.max_interior_perpendicular);
    values.push(report.max_interior_spring);
    values.push(report.max_endpoint_optimized);
    values.push(report.iteration as f64);
    expect_close(&stored, "line_session", &values);
    let seed = FixedEnergySeed::default()
        .with_saddle_energy(0.25)
        .with_target_energy(0.2)
        .with_images(3)
        .with_max_evaluations(400);
    let images = fixed_energy_mep_seed(
        &SaddleWell,
        Array1::zeros(6).view(),
        launch_mode().view(),
        coordinate_masses().view(),
        seed,
    )
    .expect("seed");
    expect_close(
        &stored,
        "line_seed",
        images.as_slice().expect("seed layout"),
    );
}

#[test]
fn tunneling_path_matches_stored_values() {
    let stored = stored();
    expect_close(&stored, "tunnel", &tunneling().expect("trajectory"));
}

#[test]
fn pafi_matches_stored_values() {
    let stored = stored();
    expect_close(&stored, "pafi", &pafi_values().expect("PAFI"));
}

#[test]
fn ohtst_surface_and_plane_maps_match_stored_values() {
    let stored = stored();
    expect_close(
        &stored,
        "ohtst_surface",
        &ohtst_surface().expect("mass-weighted surface"),
    );
    expect_close(&stored, "plane_maps", &plane_maps().expect("plane maps"));
}

#[test]
fn ohtst_shooting_matches_stored_values() {
    let stored = stored();
    expect_close(&stored, "shooting", &shooting().expect("shooting"));
}
