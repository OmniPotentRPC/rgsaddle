//! LAMMPS constrained molecular dynamics through lammpc.
//!
//! The adapter loads lammpc's in-process profile, creates one persistent
//! LAMMPS session, and advances lammpc's projected BAOAB stepper. LAMMPS
//! supplies the forces. lammpc applies the plane to position, velocity, and
//! force on every step.

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

use libloading::Library;
use ndarray::{Array1, Array2, ArrayView1};
use rgpot_core::profile::{ProfileRequest, encode_force_input, library_candidates};

use crate::ohtst::THERMODYNAMIC_DERIVATIVE;
use crate::strain_rate::{parse_json, required_f64, required_str};
use crate::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, SaddleError,
};

const LAMMPC_ABI_VERSION: c_int = 0;
const LAMMPS_THERMOSTAT_RECORD_FORMAT: &str = "rgsaddle.lammps_thermostat";
const LAMMPS_OBSERVABLE_THERMOSTAT: &str = "lammps.thermostat";
const LAMMPS_OBSERVABLE_EFFECTIVE_DOF: &str = "lammps.effective_degrees_of_freedom";

/// Thermostat used by the constrained LAMMPS backend.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LammpsThermostatKind {
    /// Projected BAOAB Langevin thermostat.
    #[default]
    BaoabLangevin,
}

impl LammpsThermostatKind {
    /// Stable string value used in records.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BaoabLangevin => "baoab_langevin",
        }
    }

    fn observable_code(self) -> f64 {
        match self {
            Self::BaoabLangevin => 1.0,
        }
    }
}

impl std::str::FromStr for LammpsThermostatKind {
    type Err = SaddleError;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        match text {
            "baoab_langevin" => Ok(Self::BaoabLangevin),
            _ => Err(SaddleError::Invalid(format!(
                "unsupported LAMMPS thermostat {text}"
            ))),
        }
    }
}

/// Thermostat metadata written with each LAMMPS sample record.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LammpsThermostatRecord {
    /// Thermostat kind used to generate the record.
    pub thermostat: LammpsThermostatKind,
    /// Effective constrained degrees of freedom, `3N - 1 - excluded_modes`.
    pub effective_degrees_of_freedom: usize,
}

impl LammpsThermostatRecord {
    /// Serialize this metadata to JSON.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"format\":\"{LAMMPS_THERMOSTAT_RECORD_FORMAT}\",\"thermostat\":\"{}\",\"effective_degrees_of_freedom\":{}}}",
            self.thermostat.as_str(),
            self.effective_degrees_of_freedom
        )
    }

    /// Parse a document written by [`Self::to_json`].
    pub fn from_json(text: &str) -> Result<Self, SaddleError> {
        let value = parse_json(text)?;
        let object = value.as_object().ok_or_else(|| {
            SaddleError::Invalid("LAMMPS thermostat record json must be an object".into())
        })?;
        if required_str(object, "format")? != LAMMPS_THERMOSTAT_RECORD_FORMAT {
            return Err(SaddleError::Invalid(
                "json format is not rgsaddle.lammps_thermostat".into(),
            ));
        }
        let thermostat = required_str(object, "thermostat")?.parse()?;
        let dof = required_f64(object, "effective_degrees_of_freedom")?;
        if !dof.is_finite() || dof < 0.0 || dof.fract() != 0.0 {
            return Err(SaddleError::Invalid(
                "effective_degrees_of_freedom must be a nonnegative integer".into(),
            ));
        }
        Ok(Self {
            thermostat,
            effective_degrees_of_freedom: dof as usize,
        })
    }

    fn observables(&self) -> [(String, f64); 2] {
        [
            (
                LAMMPS_OBSERVABLE_THERMOSTAT.to_string(),
                self.thermostat.observable_code(),
            ),
            (
                LAMMPS_OBSERVABLE_EFFECTIVE_DOF.to_string(),
                self.effective_degrees_of_freedom as f64,
            ),
        ]
    }
}

/// Atomistic geometry and lammpc interaction-model configuration.
///
/// `config_bytes` is a flat rgpot `PotentialConfig` carrying its `lammps`
/// arm. `positions` and `box_matrix` use angstrom. `masses` has one value in
/// amu per atom and must match the per-type masses in that configuration.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct LammpsSystem {
    /// Flat rgpot `PotentialConfig` bytes.
    pub config_bytes: Vec<u8>,
    /// Cartesian coordinates, three values per atom, in angstrom.
    pub positions: Array1<f64>,
    /// Atomic numbers in position order.
    pub atomic_numbers: Vec<i32>,
    /// Atomic masses in amu, one per atom.
    pub masses: Vec<f64>,
    /// Optional row-major simulation cell in angstrom.
    pub box_matrix: Option<[f64; 9]>,
}

impl LammpsSystem {
    /// Set [`Self::config_bytes`].
    #[must_use]
    pub fn with_config_bytes(mut self, config_bytes: Vec<u8>) -> Self {
        self.config_bytes = config_bytes;
        self
    }

    /// Set [`Self::positions`].
    #[must_use]
    pub fn with_positions(mut self, positions: Array1<f64>) -> Self {
        self.positions = positions;
        self
    }

    /// Set [`Self::atomic_numbers`].
    #[must_use]
    pub fn with_atomic_numbers(mut self, atomic_numbers: Vec<i32>) -> Self {
        self.atomic_numbers = atomic_numbers;
        self
    }

    /// Set [`Self::masses`].
    #[must_use]
    pub fn with_masses(mut self, masses: Vec<f64>) -> Self {
        self.masses = masses;
        self
    }

    /// Set [`Self::box_matrix`].
    #[must_use]
    pub fn with_box_matrix(mut self, box_matrix: [f64; 9]) -> Self {
        self.box_matrix = Some(box_matrix);
        self
    }

    fn check(&self) -> Result<(), SaddleError> {
        if self.config_bytes.is_empty() {
            return Err(SaddleError::Invalid(
                "LAMMPS PotentialConfig bytes are empty".into(),
            ));
        }
        if self.positions.is_empty() || !self.positions.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(
                "LAMMPS positions need three coordinates per atom".into(),
            ));
        }
        let atoms = self.positions.len() / 3;
        if self.atomic_numbers.len() != atoms || self.masses.len() != atoms {
            return Err(SaddleError::Shape(
                "LAMMPS needs one atomic number and mass per atom".into(),
            ));
        }
        if !self.positions.iter().all(|value| value.is_finite())
            || !self
                .masses
                .iter()
                .all(|mass| mass.is_finite() && *mass > 0.0)
            || self
                .box_matrix
                .is_some_and(|cell| !cell.iter().all(|value| value.is_finite()))
        {
            return Err(SaddleError::NonFinite("LAMMPS system"));
        }
        Ok(())
    }

    fn coordinate_masses(&self) -> impl Iterator<Item = f64> + '_ {
        self.masses.iter().flat_map(|mass| [*mass; 3])
    }
}

/// Dense Taylor harmonic reference for lambda thermodynamic integration.
///
/// The reference supplies `V_harm(x) = E_0 + g_0·(x-x_0) + ½ (x-x_0)ᵀ H (x-x_0)`
/// in the sampler coordinates of the plane passed to lammpc. `hessian` is
/// row-major and matches `point.len()`. The force uses the symmetric part of
/// `hessian`, which is the gradient of [`Self::energy_at`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LammpsThermodynamicReference {
    /// Reference geometry in sampler coordinates.
    pub point: Array1<f64>,
    /// Potential at [`Self::point`].
    pub energy: f64,
    /// Gradient at [`Self::point`].
    pub gradient: Array1<f64>,
    /// Symmetric Hessian at [`Self::point`].
    pub hessian: Array2<f64>,
}

impl LammpsThermodynamicReference {
    /// Taylor model `E0 + g·(x-x0) + ½ (x-x0)·H(x-x0)`.
    #[must_use]
    pub fn new(
        point: Array1<f64>,
        energy: f64,
        gradient: Array1<f64>,
        hessian: Array2<f64>,
    ) -> Self {
        Self {
            point,
            energy,
            gradient,
            hessian,
        }
    }

    fn check_position(&self, position: ArrayView1<f64>) -> Result<(), SaddleError> {
        if position.len() != self.point.len() {
            return Err(SaddleError::Shape(
                "LAMMPS thermodynamic reference differs in length from the position".into(),
            ));
        }
        if self.gradient.len() != self.point.len()
            || self.hessian.dim() != (position.len(), position.len())
        {
            return Err(SaddleError::Shape(
                "LAMMPS thermodynamic reference has inconsistent gradient or Hessian".into(),
            ));
        }
        if !self.energy.is_finite()
            || !position.iter().all(|value| value.is_finite())
            || !self.point.iter().all(|value| value.is_finite())
            || !self.gradient.iter().all(|value| value.is_finite())
            || !self.hessian.iter().all(|value| value.is_finite())
        {
            return Err(SaddleError::NonFinite("LAMMPS thermodynamic reference"));
        }
        Ok(())
    }

    /// `H_sym v` with `H_sym = (H + Hᵀ) / 2`.
    fn symmetric_action(&self, displacement: &Array1<f64>) -> Array1<f64> {
        let forward = self.hessian.dot(displacement);
        let backward = self.hessian.t().dot(displacement);
        0.5 * (&forward + &backward)
    }

    /// Row-major `(H + Hᵀ) / 2` passed to the mixing hook.
    fn symmetrized_hessian(&self) -> Array2<f64> {
        let mut symmetric = self.hessian.clone();
        symmetric += &self.hessian.t();
        symmetric.mapv_inplace(|value| value * 0.5);
        symmetric
    }

    /// Evaluate the Taylor harmonic energy at `position`.
    pub fn energy_at(&self, position: ArrayView1<f64>) -> Result<f64, SaddleError> {
        self.check_position(position)?;
        let displacement = position.to_owned() - &self.point;
        let image = self.symmetric_action(&displacement);
        let value = self.energy + self.gradient.dot(&displacement) + 0.5 * displacement.dot(&image);
        if !value.is_finite() {
            return Err(SaddleError::NonFinite(
                "LAMMPS thermodynamic reference energy",
            ));
        }
        Ok(value)
    }

    /// Harmonic force `-∇V_harm` at `position`.
    pub fn force_at(&self, position: ArrayView1<f64>) -> Result<Array1<f64>, SaddleError> {
        self.check_position(position)?;
        let displacement = position.to_owned() - &self.point;
        let force = -(&self.gradient + &self.symmetric_action(&displacement));
        if !force.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite(
                "LAMMPS thermodynamic reference force",
            ));
        }
        Ok(force)
    }
}

/// Runtime controls for lammpc constrained dynamics.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct LammpsSamplerConfig {
    /// Optional explicit `liblammpc` path.
    pub library_path: Option<PathBuf>,
    /// BAOAB time step in ps.
    pub time_step_ps: f64,
    /// Langevin friction in inverse ps.
    pub friction_per_ps: f64,
    /// Thermostat used by the constrained dynamics.
    pub thermostat: LammpsThermostatKind,
    /// Modes held by the plane construction in addition to the plane normal.
    pub excluded_modes: usize,
    /// Harmonic reference for lambda thermodynamic mixing, when requested.
    pub thermodynamic_reference: Option<LammpsThermodynamicReference>,
}

impl Default for LammpsSamplerConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            time_step_ps: 0.001,
            friction_per_ps: 1.0,
            thermostat: LammpsThermostatKind::BaoabLangevin,
            excluded_modes: 0,
            thermodynamic_reference: None,
        }
    }
}

impl LammpsSamplerConfig {
    /// Set an explicit `liblammpc` path.
    #[must_use]
    pub fn with_library_path(mut self, library_path: impl Into<PathBuf>) -> Self {
        self.library_path = Some(library_path.into());
        self
    }

    /// Set [`Self::time_step_ps`].
    #[must_use]
    pub fn with_time_step_ps(mut self, time_step_ps: f64) -> Self {
        self.time_step_ps = time_step_ps;
        self
    }

    /// Set [`Self::friction_per_ps`].
    #[must_use]
    pub fn with_friction_per_ps(mut self, friction_per_ps: f64) -> Self {
        self.friction_per_ps = friction_per_ps;
        self
    }

    /// Set [`Self::thermostat`].
    #[must_use]
    pub fn with_thermostat(mut self, thermostat: LammpsThermostatKind) -> Self {
        self.thermostat = thermostat;
        self
    }

    /// Set [`Self::excluded_modes`].
    #[must_use]
    pub fn with_excluded_modes(mut self, excluded_modes: usize) -> Self {
        self.excluded_modes = excluded_modes;
        self
    }

    /// Set [`Self::thermodynamic_reference`].
    #[must_use]
    pub fn with_thermodynamic_reference(
        mut self,
        thermodynamic_reference: LammpsThermodynamicReference,
    ) -> Self {
        self.thermodynamic_reference = Some(thermodynamic_reference);
        self
    }

    /// Effective constrained degrees of freedom, `3N - 1 - excluded_modes`.
    pub fn effective_degrees_of_freedom(&self, n_atoms: usize) -> Result<usize, SaddleError> {
        let cartesian = n_atoms
            .checked_mul(3)
            .ok_or_else(|| SaddleError::Invalid("LAMMPS atom count overflow".into()))?;
        let constrained = 1usize
            .checked_add(self.excluded_modes)
            .ok_or_else(|| SaddleError::Invalid("LAMMPS excluded-mode count overflow".into()))?;
        cartesian.checked_sub(constrained).ok_or_else(|| {
            SaddleError::Invalid(format!(
                "LAMMPS needs 3N > 1 + excluded_modes, got N={n_atoms}, excluded_modes={}",
                self.excluded_modes
            ))
        })
    }

    fn thermostat_record(&self, n_atoms: usize) -> Result<LammpsThermostatRecord, SaddleError> {
        Ok(LammpsThermostatRecord {
            thermostat: self.thermostat,
            effective_degrees_of_freedom: self.effective_degrees_of_freedom(n_atoms)?,
        })
    }

    fn check(&self) -> Result<(), SaddleError> {
        if !(self.time_step_ps.is_finite() && self.time_step_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS time step must be finite and positive".into(),
            ));
        }
        if !(self.friction_per_ps.is_finite() && self.friction_per_ps > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS friction must be finite and positive".into(),
            ));
        }
        Ok(())
    }
}

/// A real-LAMMPS implementation of [`HyperplaneSampler`].
pub struct LammpsHyperplaneSampler {
    system: LammpsSystem,
    config: LammpsSamplerConfig,
}

impl LammpsHyperplaneSampler {
    /// Build a sampler. The shared library is loaded when sampling begins.
    pub fn new(system: LammpsSystem, config: LammpsSamplerConfig) -> Self {
        Self { system, config }
    }

    /// Atomistic system used to initialize each ensemble.
    pub fn system(&self) -> &LammpsSystem {
        &self.system
    }

    /// Runtime controls used by lammpc.
    pub fn config(&self) -> &LammpsSamplerConfig {
        &self.config
    }
}

impl HyperplaneSampler for LammpsHyperplaneSampler {
    fn constrained_directions(
        &mut self,
        plane: &Hyperplane,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        let count = self.config.excluded_modes;
        if count == 0 {
            return Ok(Vec::new());
        }
        let dimension = plane.dim();
        if count >= dimension {
            return Err(SaddleError::Invalid(
                "LAMMPS excluded modes do not fit in the plane".into(),
            ));
        }
        let normal = plane.normal();
        let mut seeds = Array2::zeros((count, dimension));
        let mut placed = 0usize;
        for axis in 0..dimension {
            if placed == count {
                break;
            }
            let mut row = Array1::zeros(dimension);
            row[axis] = 1.0;
            let along = normal[axis];
            row.scaled_add(-along, &normal);
            if row.iter().all(|value| *value == 0.0) {
                continue;
            }
            seeds.row_mut(placed).assign(&row);
            placed += 1;
        }
        if placed != count {
            return Err(SaddleError::Invalid(
                "LAMMPS excluded modes do not fit in the plane".into(),
            ));
        }
        let directions = crate::projector::orthonormal_rows(seeds.view(), dimension);
        if directions.len() != count {
            return Err(SaddleError::Invalid(
                "LAMMPS excluded modes are not independent in the plane".into(),
            ));
        }
        Ok(directions)
    }

    fn sample_hyperplane(
        &mut self,
        plane: &Hyperplane,
        budget: &HyperplaneSampleBudget,
    ) -> Result<HyperplaneEnsemble, SaddleError> {
        self.system.check()?;
        self.config.check()?;
        if plane.dim() != self.system.positions.len() {
            return Err(SaddleError::Shape(
                "LAMMPS system and hyperplane dimensions differ".into(),
            ));
        }
        let thermostat = self.config.thermostat_record(self.system.masses.len())?;
        if !(budget.thermal.is_finite() && budget.thermal > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS kT must be finite and positive".into(),
            ));
        }
        if budget.stride == 0 {
            return Err(SaddleError::Invalid(
                "LAMMPS sampling stride must be positive".into(),
            ));
        }

        let request =
            ProfileRequest {
                positions: self.system.positions.as_slice().ok_or_else(|| {
                    SaddleError::Shape("LAMMPS positions must be contiguous".into())
                })?,
                atomic_numbers: &self.system.atomic_numbers,
                box_matrix: self.system.box_matrix.as_ref(),
                length_unit: "angstrom",
                energy_unit: "eV",
            };
        let force_input = encode_force_input(&request)
            .map_err(|error| SaddleError::Surface(error.to_string()))?;
        let mut engine = Engine::load(
            self.config.library_path.as_deref(),
            &self.system.config_bytes,
        )?;
        let md_config = MdConfig {
            time_step_ps: self.config.time_step_ps,
            thermal_energy_ev: budget.thermal,
            friction_per_ps: self.config.friction_per_ps,
            seed: budget.seed,
            coordinates: match budget.coordinates {
                HyperplaneCoordinates::MassWeighted => 0,
                HyperplaneCoordinates::Cartesian => 1,
            },
        };
        engine.initialize(
            &force_input,
            plane.point().as_slice().expect("hyperplane is contiguous"),
            plane.normal().as_slice().expect("hyperplane is contiguous"),
            &md_config,
        )?;
        if let Some(region) = &budget.region {
            let boundaries = region.planar_boundaries().ok_or_else(|| {
                SaddleError::Invalid(
                    "LAMMPS requires a region with engine-portable planar boundaries".into(),
                )
            })?;
            engine.set_boundaries(&boundaries, plane.dim())?;
        }
        let reference = if let Some(lambda) = budget.lambda {
            let reference = self
                .config
                .thermodynamic_reference
                .as_ref()
                .ok_or_else(|| {
                    SaddleError::Invalid(
                        "LAMMPS lambda sampling needs LammpsSamplerConfig::thermodynamic_reference"
                            .into(),
                    )
                })?;
            // A lambda of one adds no harmonic force, so the live LAMMPS
            // potential is already the unmixed full potential. The mixing
            // hook is required only when that force is nonzero.
            if lambda == 1.0 {
                if reference.point.len() != plane.dim()
                    || reference.gradient.len() != plane.dim()
                    || reference.hessian.dim() != (plane.dim(), plane.dim())
                {
                    return Err(SaddleError::Shape(
                        "LAMMPS thermodynamic reference does not match the plane dimension".into(),
                    ));
                }
                reference.energy_at(reference.point.view())?;
            } else {
                engine.set_thermodynamic_coupling(lambda, reference, plane.dim())?;
            }
            Some(reference)
        } else {
            None
        };
        gather_constrained_samples(
            &mut engine,
            &self.system,
            budget,
            plane.dim(),
            &thermostat,
            reference,
        )
    }
}

trait PotentialStep {
    fn step(&mut self, steps: usize) -> Result<(), SaddleError>;
    fn state(&mut self, dimension: usize) -> Result<MdState, SaddleError>;
    fn full_potential_energy(&mut self, position: &[f64]) -> Result<f64, SaddleError>;
}

fn gather_constrained_samples<E: PotentialStep>(
    engine: &mut E,
    system: &LammpsSystem,
    budget: &HyperplaneSampleBudget,
    dimension: usize,
    thermostat: &LammpsThermostatRecord,
    reference: Option<&LammpsThermodynamicReference>,
) -> Result<HyperplaneEnsemble, SaddleError> {
    if reference.is_some() {
        let initial = engine.state(dimension)?;
        engine.full_potential_energy(&initial.positions)?;
    }
    engine.step(budget.thermalize)?;

    let mut observations = Vec::with_capacity(budget.samples);
    let mut accounting = MdObservation::default();
    for _ in 0..budget.samples {
        engine.step(budget.stride)?;
        let state = engine.state(dimension)?;
        accounting = state.accounting;
        // md_state returns Cartesian positions. The Taylor model is stored
        // in the sampler frame of the plane, so the observable uses the
        // same conversion as the returned record.
        let cartesian = state.positions.clone();
        let (position, projected_force, projected_velocity) =
            convert_state(system, budget.coordinates, state);
        let thermodynamic_derivative = if let Some(reference) = reference {
            let harmonic = reference.energy_at(position.view())?;
            let full = engine.full_potential_energy(&cartesian)?;
            let value = full - harmonic;
            if !value.is_finite() {
                return Err(SaddleError::NonFinite("LAMMPS thermodynamic derivative"));
            }
            Some(value)
        } else {
            None
        };
        let mut observables = thermostat
            .observables()
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        if let Some(value) = thermodynamic_derivative {
            observables.insert(THERMODYNAMIC_DERIVATIVE.into(), value);
        }
        observations.push(HyperplaneObservation {
            constraint_force: accounting.constraint_force,
            pafi_gradient: None,
            jacobian: 0.0,
            position: budget.positions.then_some(position),
            projected_force: Some(projected_force),
            projected_velocity: Some(projected_velocity),
            observables,
        });
    }
    Ok(HyperplaneEnsemble {
        observations,
        evaluations: accounting.force_evaluations,
        acceptance: 1.0,
    })
}

fn convert_state(
    system: &LammpsSystem,
    coordinates: HyperplaneCoordinates,
    state: MdState,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    match coordinates {
        HyperplaneCoordinates::Cartesian => (
            Array1::from_vec(state.positions),
            Array1::from_vec(state.forces),
            Array1::from_vec(state.removed_velocities),
        ),
        HyperplaneCoordinates::MassWeighted => {
            let masses: Vec<f64> = system.coordinate_masses().collect();
            let position = state
                .positions
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value * mass.sqrt())
                .collect();
            let force = state
                .forces
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value / mass.sqrt())
                .collect();
            let velocity = state
                .removed_velocities
                .iter()
                .zip(&masses)
                .map(|(value, mass)| value * mass.sqrt())
                .collect();
            (position, force, velocity)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MdConfig {
    time_step_ps: f64,
    thermal_energy_ev: f64,
    friction_per_ps: f64,
    seed: u64,
    coordinates: c_int,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MdObservation {
    constraint_force: f64,
    force_evaluations: usize,
    steps: usize,
}

type AvailableFn = unsafe extern "C" fn() -> c_int;
type AbiVersionFn = unsafe extern "C" fn() -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type FinalizeFn = unsafe extern "C" fn();
type SessionCreateFn = unsafe extern "C" fn(*const c_void, usize) -> *mut c_void;
type SessionDestroyFn = unsafe extern "C" fn(*mut c_void);
type MdInitializeFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_void,
    usize,
    *const f64,
    *const f64,
    usize,
    *const MdConfig,
) -> c_int;
type MdStepFn = unsafe extern "C" fn(*mut c_void, usize) -> c_int;
type MdSetBoundariesFn =
    unsafe extern "C" fn(*mut c_void, *const f64, *const f64, usize, usize) -> c_int;
type MdStateFn = unsafe extern "C" fn(
    *mut c_void,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    usize,
    *mut MdObservation,
) -> c_int;
type MdSetThermodynamicCouplingFn = unsafe extern "C" fn(
    *mut c_void,
    *const MdThermodynamicCoupling,
    *const f64,
    *const f64,
    *const f64,
    usize,
) -> c_int;
type MdPotentialEnergyFn = unsafe extern "C" fn(*mut c_void, *const f64, usize, *mut f64) -> c_int;
type NveStateFn =
    unsafe extern "C" fn(*mut c_void, *mut f64, *mut f64, usize, *mut NveObservation) -> c_int;

#[repr(C)]
struct NveObservation {
    potential_energy_ev: f64,
    kinetic_energy_ev: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MdThermodynamicCoupling {
    lambda: f64,
    reference_energy: f64,
}

struct MdState {
    positions: Vec<f64>,
    forces: Vec<f64>,
    removed_velocities: Vec<f64>,
    accounting: MdObservation,
}

struct Engine {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
    last_error: LastErrorFn,
    md_initialize: MdInitializeFn,
    md_step: MdStepFn,
    md_set_boundaries: Option<MdSetBoundariesFn>,
    md_state: MdStateFn,
    md_set_thermodynamic_coupling: Option<MdSetThermodynamicCouplingFn>,
    md_potential_energy: Option<MdPotentialEnergyFn>,
    nve_state: Option<NveStateFn>,
    _library: Library,
}

impl Engine {
    fn load(path: Option<&Path>, config: &[u8]) -> Result<Self, SaddleError> {
        let explicit = path.and_then(Path::to_str);
        let mut failures = Vec::new();
        let mut loaded = None;
        for candidate in library_candidates("lammpc", explicit) {
            match unsafe { Library::new(&candidate) } {
                Ok(library) => {
                    loaded = Some(library);
                    break;
                }
                Err(error) => failures.push(format!("{candidate}: {error}")),
            }
        }
        let library = loaded.ok_or_else(|| {
            SaddleError::Surface(format!("liblammpc not loaded; {}", failures.join("; ")))
        })?;
        unsafe {
            let available = *library
                .get::<AvailableFn>(b"lammpc_available\0")
                .map_err(load_error)?;
            let abi_version = *library
                .get::<AbiVersionFn>(b"lammpc_abi_version\0")
                .map_err(load_error)?;
            let last_error = *library
                .get::<LastErrorFn>(b"lammpc_last_error\0")
                .map_err(load_error)?;
            let finalize = *library
                .get::<FinalizeFn>(b"lammpc_finalize\0")
                .map_err(load_error)?;
            let create = *library
                .get::<SessionCreateFn>(b"lammpc_session_create_from_config\0")
                .map_err(load_error)?;
            let destroy = *library
                .get::<SessionDestroyFn>(b"lammpc_session_destroy\0")
                .map_err(load_error)?;
            let md_initialize = *library
                .get::<MdInitializeFn>(b"lammpc_session_md_initialize\0")
                .map_err(load_error)?;
            let md_step = *library
                .get::<MdStepFn>(b"lammpc_session_md_step\0")
                .map_err(load_error)?;
            let md_set_boundaries = library
                .get::<MdSetBoundariesFn>(b"lammpc_session_md_set_boundaries\0")
                .ok()
                .map(|symbol| *symbol);
            let md_state = *library
                .get::<MdStateFn>(b"lammpc_session_md_state\0")
                .map_err(load_error)?;
            let md_set_thermodynamic_coupling = library
                .get::<MdSetThermodynamicCouplingFn>(
                    b"lammpc_session_md_set_thermodynamic_coupling\0",
                )
                .ok()
                .map(|symbol| *symbol);
            let md_potential_energy = library
                .get::<MdPotentialEnergyFn>(b"lammpc_session_md_potential_energy\0")
                .ok()
                .map(|symbol| *symbol);
            let nve_state = library
                .get::<NveStateFn>(b"lammpc_session_nve_state\0")
                .ok()
                .map(|symbol| *symbol);
            if available() == 0 {
                return Err(SaddleError::Surface(
                    "lammpc was built without LAMMPS".into(),
                ));
            }
            if abi_version() != LAMMPC_ABI_VERSION {
                return Err(SaddleError::Surface(format!(
                    "lammpc ABI {} does not match {}",
                    abi_version(),
                    LAMMPC_ABI_VERSION
                )));
            }
            let session = create(config.as_ptr().cast(), config.len());
            if session.is_null() {
                return Err(SaddleError::Surface(last_error_message(last_error)));
            }
            Ok(Self {
                session,
                destroy,
                finalize,
                last_error,
                md_initialize,
                md_step,
                md_set_boundaries,
                md_state,
                md_set_thermodynamic_coupling,
                md_potential_energy,
                nve_state,
                _library: library,
            })
        }
    }

    fn set_thermodynamic_coupling(
        &mut self,
        lambda: f64,
        reference: &LammpsThermodynamicReference,
        dimension: usize,
    ) -> Result<(), SaddleError> {
        let set = self.md_set_thermodynamic_coupling.ok_or_else(|| {
            SaddleError::Surface(
                "liblammpc lacks lammpc_session_md_set_thermodynamic_coupling; \
                 thermodynamic integration needs a mixing hook that adds \
                 (1-lambda) F_harmonic in the sampler frame each step, with the \
                 reference geometry, gradient, and Hessian passed at coupling time"
                    .into(),
            )
        })?;
        if reference.point.len() != dimension
            || reference.gradient.len() != dimension
            || reference.hessian.dim() != (dimension, dimension)
        {
            return Err(SaddleError::Shape(
                "LAMMPS thermodynamic reference does not match the plane dimension".into(),
            ));
        }
        if !(lambda.is_finite() && (0.0..=1.0).contains(&lambda)) {
            return Err(SaddleError::Invalid(
                "thermodynamic lambda must lie in [0, 1]".into(),
            ));
        }
        reference.energy_at(reference.point.view())?;
        let coupling = MdThermodynamicCoupling {
            lambda,
            reference_energy: reference.energy,
        };
        let point = reference.point.as_slice().ok_or_else(|| {
            SaddleError::Shape("LAMMPS reference geometry must be contiguous".into())
        })?;
        let gradient = reference.gradient.as_slice().ok_or_else(|| {
            SaddleError::Shape("LAMMPS reference gradient must be contiguous".into())
        })?;
        let symmetric = reference.symmetrized_hessian();
        let hessian = symmetric.as_slice().ok_or_else(|| {
            SaddleError::Shape("LAMMPS reference Hessian must be contiguous".into())
        })?;
        let status = unsafe {
            set(
                self.session,
                &coupling,
                point.as_ptr(),
                gradient.as_ptr(),
                hessian.as_ptr(),
                dimension,
            )
        };
        self.check(status)
    }

    fn full_potential_energy(&mut self, position: &[f64]) -> Result<f64, SaddleError> {
        if let Some(evaluate) = self.md_potential_energy {
            let mut energy = 0.0;
            let status =
                unsafe { evaluate(self.session, position.as_ptr(), position.len(), &mut energy) };
            self.check(status)?;
            return finite_potential(energy);
        }
        let current = self.state(position.len())?;
        let same_geometry = current.positions.len() == position.len()
            && current
                .positions
                .iter()
                .zip(position)
                .all(|(stored, requested)| stored == requested);
        if !same_geometry {
            return Err(missing_potential_symbol());
        }
        self.live_potential_energy(position.len())
    }

    fn live_potential_energy(&mut self, dimension: usize) -> Result<f64, SaddleError> {
        let read = self.nve_state.ok_or_else(missing_potential_symbol)?;
        let mut observation = NveObservation {
            potential_energy_ev: 0.0,
            kinetic_energy_ev: 0.0,
        };
        let status = unsafe {
            read(
                self.session,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dimension,
                &mut observation,
            )
        };
        self.check(status)?;
        finite_potential(observation.potential_energy_ev)
    }

    fn initialize(
        &mut self,
        force_input: &[u8],
        point: &[f64],
        normal: &[f64],
        config: &MdConfig,
    ) -> Result<(), SaddleError> {
        let status = unsafe {
            (self.md_initialize)(
                self.session,
                force_input.as_ptr().cast(),
                force_input.len(),
                point.as_ptr(),
                normal.as_ptr(),
                point.len(),
                config,
            )
        };
        self.check(status)
    }

    fn step(&mut self, steps: usize) -> Result<(), SaddleError> {
        let status = unsafe { (self.md_step)(self.session, steps) };
        self.check(status)
    }

    fn set_boundaries(
        &mut self,
        boundaries: &[crate::PlanarBoundary],
        dimension: usize,
    ) -> Result<(), SaddleError> {
        let set = self.md_set_boundaries.ok_or_else(|| {
            SaddleError::Surface(
                "liblammpc lacks hard-region support; update lammpc with the generic boundary ABI"
                    .into(),
            )
        })?;
        let mut points = Vec::with_capacity(boundaries.len() * dimension);
        let mut normals = Vec::with_capacity(boundaries.len() * dimension);
        for boundary in boundaries {
            if boundary.point().len() != dimension {
                return Err(SaddleError::Shape(
                    "LAMMPS boundary and plane dimensions differ".into(),
                ));
            }
            points.extend(boundary.point());
            normals.extend(boundary.normal());
        }
        let status = unsafe {
            set(
                self.session,
                points.as_ptr(),
                normals.as_ptr(),
                boundaries.len(),
                dimension,
            )
        };
        self.check(status)
    }

    fn state(&mut self, dimension: usize) -> Result<MdState, SaddleError> {
        let mut positions = vec![0.0; dimension];
        let mut velocities = vec![0.0; dimension];
        let mut forces = vec![0.0; dimension];
        let mut removed_velocities = vec![0.0; dimension];
        let mut accounting = MdObservation::default();
        let status = unsafe {
            (self.md_state)(
                self.session,
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
                forces.as_mut_ptr(),
                removed_velocities.as_mut_ptr(),
                dimension,
                &mut accounting,
            )
        };
        self.check(status)?;
        Ok(MdState {
            positions,
            forces,
            removed_velocities,
            accounting,
        })
    }

    fn check(&self, status: c_int) -> Result<(), SaddleError> {
        if status == 0 {
            Ok(())
        } else {
            Err(SaddleError::Surface(last_error_message(self.last_error)))
        }
    }
}

impl PotentialStep for Engine {
    fn step(&mut self, steps: usize) -> Result<(), SaddleError> {
        Engine::step(self, steps)
    }

    fn state(&mut self, dimension: usize) -> Result<MdState, SaddleError> {
        Engine::state(self, dimension)
    }

    fn full_potential_energy(&mut self, position: &[f64]) -> Result<f64, SaddleError> {
        Engine::full_potential_energy(self, position)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            (self.destroy)(self.session);
            self.session = std::ptr::null_mut();
            (self.finalize)();
        }
    }
}

fn finite_potential(energy: f64) -> Result<f64, SaddleError> {
    if !energy.is_finite() {
        return Err(SaddleError::NonFinite("LAMMPS potential energy"));
    }
    Ok(energy)
}

fn missing_potential_symbol() -> SaddleError {
    SaddleError::Surface(
        "liblammpc lacks lammpc_session_md_potential_energy; thermodynamic \
         integration needs the unmixed full potential at the sampled geometry"
            .into(),
    )
}

fn load_error(error: libloading::Error) -> SaddleError {
    SaddleError::Surface(error.to_string())
}

fn last_error_message(last_error: LastErrorFn) -> String {
    let pointer = unsafe { last_error() };
    if pointer.is_null() {
        "lammpc failed without a diagnostic".into()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn thermostat_record_round_trips() {
        let config = LammpsSamplerConfig::default().with_excluded_modes(2);
        let record = config.thermostat_record(4).unwrap();
        assert_eq!(record.thermostat, LammpsThermostatKind::BaoabLangevin);
        assert_eq!(record.effective_degrees_of_freedom, 9);
        let text = record.to_json();
        assert_eq!(LammpsThermostatRecord::from_json(&text).unwrap(), record);
    }

    #[test]
    fn effective_degrees_of_freedom_rejects_overconstrained_systems() {
        let error = LammpsSamplerConfig::default()
            .with_excluded_modes(3)
            .effective_degrees_of_freedom(1)
            .expect_err("3N - 1 - excluded_modes must be nonnegative");
        assert!(matches!(error, SaddleError::Invalid(_)));
    }

    #[test]
    fn thermostat_metadata_maps_to_record_observables() {
        let record = LammpsThermostatRecord {
            thermostat: LammpsThermostatKind::BaoabLangevin,
            effective_degrees_of_freedom: 14,
        };
        let observables: BTreeMap<_, _> = record.observables().into_iter().collect();
        assert_eq!(observables.get(LAMMPS_OBSERVABLE_THERMOSTAT), Some(&1.0));
        assert_eq!(
            observables.get(LAMMPS_OBSERVABLE_EFFECTIVE_DOF),
            Some(&14.0)
        );
    }

    #[test]
    fn taylor_force_is_the_gradient_of_the_symmetric_quadratic() {
        let reference = LammpsThermodynamicReference {
            point: Array1::from_vec(vec![1.0, -2.0]),
            energy: 0.5,
            gradient: Array1::from_vec(vec![0.2, -0.4]),
            hessian: Array2::from_shape_vec((2, 2), vec![2.0, 0.5, -0.1, 3.0]).unwrap(),
        };
        let position = Array1::from_vec(vec![1.3, -1.6]);
        let energy = reference.energy_at(position.view()).unwrap();
        let force = reference.force_at(position.view()).unwrap();
        assert!((energy - 0.754).abs() < 1.0e-12);
        assert!((force[0] + 0.88).abs() < 1.0e-12);
        assert!((force[1] + 0.86).abs() < 1.0e-12);
    }

    #[test]
    fn mass_weighted_records_scale_cartesian_positions() {
        let system = LammpsSystem {
            masses: vec![4.0],
            ..LammpsSystem::default()
        };
        let state = MdState {
            positions: vec![1.5, -2.0, 0.5],
            forces: vec![3.0, 0.0, -1.0],
            removed_velocities: vec![0.25, 0.0, -0.5],
            accounting: MdObservation::default(),
        };
        let (position, force, velocity) =
            convert_state(&system, HyperplaneCoordinates::MassWeighted, state);
        let scale = 2.0;
        assert_eq!(
            position.as_slice().unwrap(),
            [1.5 * scale, -2.0 * scale, 0.5 * scale]
        );
        assert_eq!(force.as_slice().unwrap(), [3.0 / scale, 0.0, -1.0 / scale]);
        assert_eq!(
            velocity.as_slice().unwrap(),
            [0.25 * scale, 0.0, -0.5 * scale]
        );
    }

    #[test]
    fn thermodynamic_coupling_is_two_doubles() {
        assert_eq!(std::mem::size_of::<MdThermodynamicCoupling>(), 16);
        assert_eq!(std::mem::align_of::<MdThermodynamicCoupling>(), 8);
    }

    struct NanAtRest {
        steps: usize,
    }

    impl PotentialStep for NanAtRest {
        fn step(&mut self, steps: usize) -> Result<(), SaddleError> {
            self.steps += steps;
            Ok(())
        }

        fn state(&mut self, dimension: usize) -> Result<MdState, SaddleError> {
            Ok(MdState {
                positions: vec![0.1; dimension],
                forces: vec![0.0; dimension],
                removed_velocities: vec![0.0; dimension],
                accounting: MdObservation::default(),
            })
        }

        fn full_potential_energy(&mut self, _position: &[f64]) -> Result<f64, SaddleError> {
            if self.steps == 0 {
                Err(SaddleError::NonFinite("LAMMPS potential energy"))
            } else {
                Ok(1.0)
            }
        }
    }

    #[test]
    fn excluded_modes_are_constrained_directions() {
        let mut sampler = LammpsHyperplaneSampler::new(
            LammpsSystem::default()
                .with_positions(Array1::zeros(6))
                .with_atomic_numbers(vec![1, 1])
                .with_masses(vec![1.0, 1.0])
                .with_config_bytes(vec![1]),
            LammpsSamplerConfig::default().with_excluded_modes(2),
        );
        let mut normal = Array1::zeros(6);
        normal[0] = 1.0;
        let plane = Hyperplane::new(Array1::zeros(6), normal).unwrap();
        let directions = sampler.constrained_directions(&plane).unwrap();
        assert_eq!(directions.len(), 2);
        assert_eq!(plane.dim() - 1 - directions.len(), 3);
        for direction in &directions {
            assert!(plane.normal().dot(direction).abs() < 1.0e-12);
            assert!((direction.dot(direction) - 1.0).abs() < 1.0e-12);
        }
        assert!(directions[0].dot(&directions[1]).abs() < 1.0e-12);
    }

    #[test]
    fn lambda_sampling_rejects_a_non_finite_energy_before_the_first_step() {
        let reference = LammpsThermodynamicReference {
            point: Array1::zeros(1),
            energy: 0.0,
            gradient: Array1::zeros(1),
            hessian: Array2::zeros((1, 1)),
        };
        let budget = HyperplaneSampleBudget::default()
            .with_thermalize(4)
            .with_stride(1)
            .with_samples(1)
            .with_coordinates(HyperplaneCoordinates::Cartesian);
        let thermostat = LammpsThermostatRecord {
            thermostat: LammpsThermostatKind::BaoabLangevin,
            effective_degrees_of_freedom: 0,
        };
        let mut engine = NanAtRest { steps: 0 };
        let error = gather_constrained_samples(
            &mut engine,
            &LammpsSystem::default(),
            &budget,
            1,
            &thermostat,
            Some(&reference),
        )
        .expect_err("the resting energy is not finite");
        assert!(matches!(error, SaddleError::NonFinite(_)), "{error:?}");
        assert_eq!(engine.steps, 0);
    }
}
