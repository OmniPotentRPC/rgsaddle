#![cfg(feature = "lammps")]

use std::path::PathBuf;
use std::sync::Arc;

use capnp::message::Builder;
use capnp::serialize;
use ndarray::{Array1, ArrayView1, array};
use rgpot_core::Potentials_capnp::potential_config;
use rgsaddle::{
    AnalyticSurfaceSampler, Hyperplane, HyperplaneCoordinates, HyperplaneSampleBudget,
    HyperplaneSampler, LammpsHyperplaneSampler, LammpsSamplerConfig, LammpsSystem, LangevinConfig,
    PlanarBoundary, PointSurface, SaddleError, sample_plane_with_sampler,
};

const EPSILON: f64 = 0.2;
const SIGMA: f64 = 2.5;
const TETHER: f64 = 2.0;

fn library_path() -> Option<PathBuf> {
    std::env::var_os("LAMMPC_LIBRARY").map(PathBuf::from)
}

fn lammps_config(
    pair_style: &str,
    pair_coeffs: &[String],
    atomic_number: i32,
    mass: f64,
    boundary: &str,
    extra_setup: &[String],
) -> Vec<u8> {
    let mut message = Builder::new_default();
    {
        let config = message.init_root::<potential_config::Builder>();
        let mut params = config.init_lammps();
        params.set_units_style("metal");
        params.set_atom_style("atomic");
        params.set_pair_style(pair_style);
        params.set_boundary(boundary);
        params.set_newton_pair(true);
        let mut coefficients = params.reborrow().init_pair_coeffs(pair_coeffs.len() as u32);
        for (index, coefficient) in pair_coeffs.iter().enumerate() {
            coefficients.set(index as u32, coefficient);
        }
        let mut type_map = params.reborrow().init_type_to_atomic_number(1);
        type_map.set(0, atomic_number);
        let mut masses = params.reborrow().init_masses(1);
        masses.set(0, mass);
        let mut setup = params.reborrow().init_extra_setup(extra_setup.len() as u32);
        for (index, command) in extra_setup.iter().enumerate() {
            setup.set(index as u32, command);
        }
    }
    serialize::write_message_to_words(&message)
}

fn lj_system() -> LammpsSystem {
    let positions = Array1::from_vec(vec![0.0, 0.0, 0.0, 3.1, 0.0, 0.0]);
    LammpsSystem::default()
        .with_config_bytes(lammps_config(
            "lj/cut 8.0",
            &[format!("1 1 {EPSILON} {SIGMA}")],
            29,
            1.0,
            "f f f",
            &[format!("fix tether all spring/self {TETHER}")],
        ))
        .with_positions(positions)
        .with_atomic_numbers(vec![29, 29])
        .with_masses(vec![1.0, 1.0])
}

fn lj_backend() -> Option<LammpsHyperplaneSampler> {
    let path = library_path()?;
    Some(LammpsHyperplaneSampler::new(
        lj_system(),
        LammpsSamplerConfig::default()
            .with_library_path(path)
            .with_time_step_ps(0.0002)
            .with_friction_per_ps(10.0),
    ))
}

#[test]
fn projected_md_holds_the_plane_over_many_steps() {
    let Some(mut backend) = lj_backend() else {
        return;
    };
    let point = backend.system().positions.clone();
    let plane = Hyperplane::new(
        point,
        Array1::from_vec(vec![-1.0, 0.5, 0.0, 1.0, -0.5, 0.0]),
    )
    .expect("plane");
    let budget = HyperplaneSampleBudget::default()
        .with_thermal(0.01)
        .with_seed(19)
        .with_thermalize(25)
        .with_stride(2)
        .with_samples(100)
        .with_coordinates(HyperplaneCoordinates::MassWeighted)
        .with_positions(true);
    let ensemble = backend
        .sample_hyperplane(&plane, &budget)
        .expect("constrained LAMMPS");
    assert_eq!(ensemble.observations.len(), 100);
    for observation in &ensemble.observations {
        let position = observation.position.as_ref().expect("position");
        assert!(
            plane.distance(position.view()).abs() < 2.0e-11,
            "distance={}",
            plane.distance(position.view())
        );
        let force = observation.projected_force.as_ref().expect("force");
        assert!(plane.normal().dot(force).abs() < 2.0e-10);
    }
}

struct LjTether {
    reference: Array1<f64>,
}

impl PointSurface for LjTether {
    fn eval(&self, position: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let displacement = &position - &self.reference;
        let mut gradient = &displacement * TETHER;
        let delta = [
            position[3] - position[0],
            position[4] - position[1],
            position[5] - position[2],
        ];
        let distance2 = delta.iter().map(|value| value * value).sum::<f64>();
        let distance = distance2.sqrt();
        let sr6 = (SIGMA / distance).powi(6);
        let pair_energy = 4.0 * EPSILON * (sr6 * sr6 - sr6);
        let radial_gradient = 24.0 * EPSILON * (sr6 - 2.0 * sr6 * sr6) / distance2;
        for axis in 0..3 {
            let component = radial_gradient * delta[axis];
            gradient[axis] -= component;
            gradient[3 + axis] += component;
        }
        Ok((
            pair_energy + 0.5 * TETHER * displacement.dot(&displacement),
            gradient,
        ))
    }
}

#[test]
fn lammps_and_analytic_lj_mean_forces_agree_with_their_error_bars() {
    let Some(mut lammps) = lj_backend() else {
        return;
    };
    let reference = lammps.system().positions.clone();
    let mut point = reference.clone();
    point[3] = 3.0;
    let plane = Hyperplane::new(
        point.clone(),
        Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )
    .expect("plane");
    let estimator = LangevinConfig::default()
        .with_thermal(0.01)
        .with_burn_in(1_000)
        .with_stride(5)
        .with_records(4_000)
        .with_blocks(10)
        .with_seed(23)
        .with_retain(0);
    let lammps_sample =
        sample_plane_with_sampler(&mut lammps, &plane, &estimator).expect("LAMMPS estimate");
    let surface = LjTether { reference };
    let mut analytic = AnalyticSurfaceSampler::new(
        &surface,
        estimator.clone().with_time_step(0.01).with_friction(1.0),
    )
    .with_start(point);
    let analytic_sample =
        sample_plane_with_sampler(&mut analytic, &plane, &estimator).expect("analytic estimate");
    let difference = (lammps_sample.normal_force - analytic_sample.normal_force).abs();
    let combined = lammps_sample
        .normal_force_error
        .hypot(analytic_sample.normal_force_error);
    assert!(
        difference <= 5.0 * combined.max(1.0e-3),
        "LAMMPS {} +/- {}, analytic {} +/- {}",
        lammps_sample.normal_force,
        lammps_sample.normal_force_error,
        analytic_sample.normal_force,
        analytic_sample.normal_force_error
    );
}

#[test]
fn copper_eam_alloy_vacancy_plane_smoke() {
    let (Some(library), Some(eam_file)) = (
        library_path(),
        std::env::var_os("RGSADDLE_CU_EAM").map(PathBuf::from),
    ) else {
        return;
    };
    let lattice = 3.615;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ];
    let mut positions = Vec::new();
    for ix in 0..2 {
        for iy in 0..2 {
            for iz in 0..2 {
                for site in basis {
                    if ix == 0 && iy == 0 && iz == 0 && site == [0.0, 0.0, 0.0] {
                        continue;
                    }
                    positions.extend([
                        (ix as f64 + site[0]) * lattice,
                        (iy as f64 + site[1]) * lattice,
                        (iz as f64 + site[2]) * lattice,
                    ]);
                }
            }
        }
    }
    let positions = Array1::from_vec(positions);
    let atoms = positions.len() / 3;
    let eam = eam_file.to_string_lossy();
    let system = LammpsSystem::default()
        .with_config_bytes(lammps_config(
            "eam/alloy",
            &[format!("* * {eam} Cu")],
            29,
            63.546,
            "p p p",
            &[],
        ))
        .with_positions(positions.clone())
        .with_atomic_numbers(vec![29; atoms])
        .with_masses(vec![63.546; atoms])
        .with_box_matrix([
            2.0 * lattice,
            0.0,
            0.0,
            0.0,
            2.0 * lattice,
            0.0,
            0.0,
            0.0,
            2.0 * lattice,
        ]);
    let mut point = positions * 63.546_f64.sqrt();
    let mut normal = Array1::zeros(point.len());
    let hop = [0.0, -1.0, -1.0];
    let hop_norm = 2.0_f64.sqrt();
    for axis in 0..3 {
        point[axis] *= 0.5;
        normal[axis] = hop[axis] / hop_norm;
    }
    let plane = Hyperplane::new(point, normal).expect("vacancy plane");
    let mut backend = LammpsHyperplaneSampler::new(
        system,
        LammpsSamplerConfig::default()
            .with_library_path(library)
            .with_time_step_ps(0.0002)
            .with_friction_per_ps(5.0),
    );
    let config = LangevinConfig::default()
        .with_thermal(0.02585)
        .with_burn_in(200)
        .with_stride(5)
        .with_records(800)
        .with_blocks(4)
        .with_seed(31)
        .with_retain(2);
    let sample = sample_plane_with_sampler(&mut backend, &plane, &config).expect("Cu OH-TST smoke");
    assert!(!sample.retained.is_empty() && sample.retained.len() <= 2);
    assert!(sample.normal_force.is_finite());
    assert!(sample.normal_force_error.is_finite());
    assert!(sample.evaluations >= 4_200);
}

fn sample_lj_with_optional_boundary(with_boundary: bool) -> Result<Vec<Array1<f64>>, SaddleError> {
    let mut backend = lj_backend().ok_or_else(|| {
        SaddleError::Invalid("LAMMPC_LIBRARY is required for the LAMMPS boundary test".into())
    })?;
    let point = backend.system().positions.clone();
    let plane = Hyperplane::new(
        point,
        Array1::from_vec(vec![-1.0, 0.5, 0.0, 1.0, -0.5, 0.0]),
    )?;
    let mut budget = HyperplaneSampleBudget::default()
        .with_thermal(0.02)
        .with_seed(71)
        .with_thermalize(50)
        .with_stride(2)
        .with_samples(400)
        .with_coordinates(HyperplaneCoordinates::Cartesian)
        .with_positions(true);
    if with_boundary {
        let boundary = PlanarBoundary::new(
            array![0.0, 0.0, 0.0, 3.05, 0.0, 0.0],
            array![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )?;
        budget = budget.with_region(Arc::new(boundary));
    }
    let ensemble = backend.sample_hyperplane(&plane, &budget)?;
    ensemble
        .observations
        .into_iter()
        .map(|observation| {
            observation.position.ok_or_else(|| {
                SaddleError::Invalid("LAMMPS boundary test requires retained positions".into())
            })
        })
        .collect()
}

#[test]
fn hard_boundary_callback_confines_lammps_trajectory() {
    if library_path().is_none() {
        eprintln!(
            "SKIPPED hard_boundary_callback_confines_lammps_trajectory: \
             set LAMMPC_LIBRARY to a real liblammpc"
        );
        return;
    }
    let with_boundary =
        sample_lj_with_optional_boundary(true).expect("LAMMPS hard boundary must be enforced");
    let without_boundary =
        sample_lj_with_optional_boundary(false).expect("unbounded LAMMPS sample");
    let leaves_without = without_boundary
        .iter()
        .filter(|position| position[3] > 3.05)
        .count();
    let leaves_with = with_boundary
        .iter()
        .filter(|position| position[3] > 3.05)
        .count();
    assert_eq!(leaves_with, 0);
    assert!(leaves_without > 0);
}
