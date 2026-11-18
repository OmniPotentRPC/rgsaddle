//! LAMMPS constrained molecular dynamics through lammpc.
//!
//! The adapter loads lammpc's in-process profile, creates one persistent
//! LAMMPS session, and advances lammpc's projected BAOAB stepper. LAMMPS
//! supplies the forces. lammpc applies the plane to position, velocity, and
//! force on every step.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::slice;

use libloading::Library;
use ndarray::Array1;
use rgpot_core::profile::{ProfileRequest, encode_force_input, library_candidates};

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
}

impl Default for LammpsSamplerConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            time_step_ps: 0.001,
            friction_per_ps: 1.0,
            thermostat: LammpsThermostatKind::BaoabLangevin,
            excluded_modes: 0,
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
            engine.set_boundaries(
                &boundaries,
                plane.normal().as_slice().expect("hyperplane is contiguous"),
                self.system.coordinate_masses(),
                budget.coordinates,
                plane.dim(),
            )?;
        }
        engine.step(budget.thermalize)?;

        let mut observations = Vec::with_capacity(budget.samples);
        let mut accounting = MdObservation::default();
        for _ in 0..budget.samples {
            engine.step(budget.stride)?;
            let state = engine.state(plane.dim())?;
            accounting = state.accounting;
            let (position, projected_force, projected_velocity) =
                convert_state(&self.system, budget.coordinates, state);
            let observables = thermostat.observables().into_iter().collect();
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
// lammpc's md_* and hard-boundary entry points all accept the owning
// LammpcSession handle returned by session_create_from_config.
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
type HardBoundaryCallbackFn =
    unsafe extern "C" fn(*mut c_void, c_int, *mut f64, *mut f64, *const c_void) -> c_int;
type SessionSetHardBoundaryFn =
    unsafe extern "C" fn(*mut c_void, Option<HardBoundaryCallbackFn>, *mut c_void) -> c_int;
type SessionClearHardBoundaryFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type MdStateFn = unsafe extern "C" fn(
    *mut c_void,
    *mut f64,
    *mut f64,
    *mut f64,
    *mut f64,
    usize,
    *mut MdObservation,
) -> c_int;

struct MdState {
    positions: Vec<f64>,
    forces: Vec<f64>,
    removed_velocities: Vec<f64>,
    accounting: MdObservation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundaryCoordinates {
    MassWeighted,
    Cartesian,
}

#[derive(Clone, Debug)]
struct BoundaryFace {
    point: Vec<f64>,
    normal: Vec<f64>,
}

#[derive(Clone, Debug)]
struct HardBoundaryData {
    faces: Vec<BoundaryFace>,
    plane_normal: Vec<f64>,
    coordinates: BoundaryCoordinates,
    sqrt_masses: Vec<f64>,
    callback_calls: usize,
}

const BOUNDARY_ACCEPT: c_int = 0;
const BOUNDARY_REJECT: c_int = 1;
const BOUNDARY_REFLECT: c_int = 2;
const BOUNDARY_ERROR: c_int = -1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct BoundaryBox {
    lo: [f64; 3],
    hi: [f64; 3],
    xy: f64,
    xz: f64,
    yz: f64,
    periodic: [c_int; 3],
    triclinic: c_int,
}

struct Engine {
    session: *mut c_void,
    destroy: SessionDestroyFn,
    finalize: FinalizeFn,
    last_error: LastErrorFn,
    md_initialize: MdInitializeFn,
    md_step: MdStepFn,
    set_hard_boundary: Option<SessionSetHardBoundaryFn>,
    clear_hard_boundary: Option<SessionClearHardBoundaryFn>,
    hard_boundary: Option<Box<HardBoundaryData>>,
    md_state: MdStateFn,
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
            let set_hard_boundary = library
                .get::<SessionSetHardBoundaryFn>(b"lammpc_session_set_hard_boundary\0")
                .ok()
                .map(|symbol| *symbol);
            let clear_hard_boundary = library
                .get::<SessionClearHardBoundaryFn>(b"lammpc_session_clear_hard_boundary\0")
                .ok()
                .map(|symbol| *symbol);
            let md_state = *library
                .get::<MdStateFn>(b"lammpc_session_md_state\0")
                .map_err(load_error)?;
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
                set_hard_boundary,
                clear_hard_boundary,
                hard_boundary: None,
                md_state,
                _library: library,
            })
        }
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
        let callbacks_before = self
            .hard_boundary
            .as_ref()
            .map(|boundary| boundary.callback_calls);
        let status = unsafe { (self.md_step)(self.session, steps) };
        self.check(status)?;
        if let (Some(before), Some(boundary)) = (callbacks_before, self.hard_boundary.as_ref()) {
            let callbacks = boundary.callback_calls.saturating_sub(before);
            if steps > 0 && callbacks != steps {
                return Err(SaddleError::Surface(format!(
                    "liblammpc lacks constrained-MD hard-boundary capability: \
                     md_step requested {steps} steps but invoked the callback {callbacks} times"
                )));
            }
        }
        Ok(())
    }

    fn set_boundaries(
        &mut self,
        boundaries: &[crate::PlanarBoundary],
        plane_normal: &[f64],
        coordinate_masses: impl Iterator<Item = f64>,
        coordinates: HyperplaneCoordinates,
        dimension: usize,
    ) -> Result<(), SaddleError> {
        let set = self
            .set_hard_boundary
            .zip(self.clear_hard_boundary)
            .map(|pair| pair.0)
            .ok_or_else(|| {
                SaddleError::Surface(
                    "liblammpc lacks hard-region support; update lammpc with the generic boundary ABI"
                        .into(),
                )
            })?;
        if plane_normal.len() != dimension {
            return Err(SaddleError::Shape(
                "LAMMPS plane normal and boundary dimensions differ".into(),
            ));
        }
        let sqrt_masses: Vec<f64> = coordinate_masses.map(|mass| mass.sqrt()).collect();
        if sqrt_masses.len() != dimension {
            return Err(SaddleError::Shape(
                "LAMMPS masses and boundary dimensions differ".into(),
            ));
        }
        if sqrt_masses
            .iter()
            .any(|value| !(value.is_finite() && *value > 0.0))
        {
            return Err(SaddleError::Invalid(
                "LAMMPS masses must be finite and positive".into(),
            ));
        }
        let callback_coordinates = match coordinates {
            HyperplaneCoordinates::MassWeighted => BoundaryCoordinates::MassWeighted,
            HyperplaneCoordinates::Cartesian => BoundaryCoordinates::Cartesian,
        };
        let mut faces = Vec::with_capacity(boundaries.len());
        for boundary in boundaries {
            if boundary.point().len() != dimension {
                return Err(SaddleError::Shape(
                    "LAMMPS boundary and plane dimensions differ".into(),
                ));
            }
            faces.push(BoundaryFace {
                point: boundary.point().to_vec(),
                normal: boundary.normal().to_vec(),
            });
        }
        let mut data = Box::new(HardBoundaryData {
            faces,
            plane_normal: plane_normal.to_vec(),
            coordinates: callback_coordinates,
            sqrt_masses,
            callback_calls: 0,
        });
        let status = unsafe {
            set(
                self.session,
                Some(hard_boundary_callback),
                data.as_mut() as *mut HardBoundaryData as *mut c_void,
            )
        };
        self.check(status)?;
        self.hard_boundary = Some(data);
        Ok(())
    }

    fn clear_boundaries(&mut self) {
        if let Some(clear) = self.clear_hard_boundary {
            let _ = unsafe { clear(self.session) };
        }
        self.hard_boundary = None;
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

fn apply_hard_boundary(
    data: &HardBoundaryData,
    positions: &mut [f64],
    velocities: &mut [f64],
    simulation_box: &BoundaryBox,
) -> Result<c_int, SaddleError> {
    if positions.len() != velocities.len() {
        return Err(SaddleError::Shape(
            "LAMMPS boundary callback needs matching position and velocity dimensions".into(),
        ));
    }
    if positions.len() != data.plane_normal.len() || positions.len() != data.sqrt_masses.len() {
        return Err(SaddleError::Shape(
            "LAMMPS boundary callback dimensions differ from boundary data".into(),
        ));
    }
    let mut coordinates = positions.to_vec();
    let mut tangent_velocities = velocities.to_vec();
    if data.coordinates == BoundaryCoordinates::MassWeighted {
        for axis in 0..coordinates.len() {
            coordinates[axis] *= data.sqrt_masses[axis];
            tangent_velocities[axis] *= data.sqrt_masses[axis];
        }
    }
    let mut reflected = false;
    for face in &data.faces {
        let displacement = boundary_displacement(data, &coordinates, &face.point, simulation_box)?;
        let signed = dot(&face.normal, &displacement)?;
        if signed <= 0.0 {
            continue;
        }
        let mut tangent_normal = project_to_tangent(&face.normal, &data.plane_normal)?;
        normalize_in_place(&mut tangent_normal)?;
        let penetration = dot(&tangent_normal, &displacement)?;
        if penetration <= 0.0 {
            return Ok(BOUNDARY_REJECT);
        }
        mirror_position(&mut coordinates, &tangent_normal, penetration);
        mirror_velocity(&mut tangent_velocities, &tangent_normal);
        reflected = true;
    }
    if data.coordinates == BoundaryCoordinates::MassWeighted {
        for axis in 0..coordinates.len() {
            positions[axis] = coordinates[axis] / data.sqrt_masses[axis];
            velocities[axis] = tangent_velocities[axis] / data.sqrt_masses[axis];
        }
    } else {
        positions.copy_from_slice(&coordinates);
        velocities.copy_from_slice(&tangent_velocities);
    }
    if reflected {
        Ok(BOUNDARY_REFLECT)
    } else {
        Ok(BOUNDARY_ACCEPT)
    }
}

fn boundary_displacement(
    data: &HardBoundaryData,
    position: &[f64],
    point: &[f64],
    simulation_box: &BoundaryBox,
) -> Result<Vec<f64>, SaddleError> {
    if position.len() != point.len() || position.len() != data.sqrt_masses.len() {
        return Err(SaddleError::Shape(
            "LAMMPS boundary callback received inconsistent dimensions".into(),
        ));
    }
    if !position.len().is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "LAMMPS boundary callback needs three coordinates per atom".into(),
        ));
    }
    let box_values = [
        simulation_box.lo[0],
        simulation_box.lo[1],
        simulation_box.lo[2],
        simulation_box.hi[0],
        simulation_box.hi[1],
        simulation_box.hi[2],
        simulation_box.xy,
        simulation_box.xz,
        simulation_box.yz,
    ];
    if !box_values.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("LAMMPS boundary box"));
    }
    let lengths = [
        simulation_box.hi[0] - simulation_box.lo[0],
        simulation_box.hi[1] - simulation_box.lo[1],
        simulation_box.hi[2] - simulation_box.lo[2],
    ];
    for length in lengths {
        if !(length.is_finite() && length > 0.0) {
            return Err(SaddleError::Invalid(
                "LAMMPS boundary box has a nonpositive length".into(),
            ));
        }
    }

    let mut displacement = Vec::with_capacity(position.len());
    for atom in 0..(position.len() / 3) {
        let offset = 3 * atom;
        let mut cartesian = [0.0; 3];
        for axis in 0..3 {
            let delta = position[offset + axis] - point[offset + axis];
            cartesian[axis] = if data.coordinates == BoundaryCoordinates::MassWeighted {
                delta / data.sqrt_masses[offset + axis]
            } else {
                delta
            };
        }
        let mut fractional = [
            (cartesian[0]
                - simulation_box.xy * cartesian[1] / lengths[1]
                - simulation_box.xz * cartesian[2] / lengths[2]
                + simulation_box.xy * simulation_box.yz * cartesian[2] / (lengths[1] * lengths[2]))
                / lengths[0],
            (cartesian[1] - simulation_box.yz * cartesian[2] / lengths[2]) / lengths[1],
            cartesian[2] / lengths[2],
        ];
        for axis in 0..3 {
            if simulation_box.periodic[axis] != 0 {
                fractional[axis] -= fractional[axis].round();
            }
        }
        cartesian = [
            lengths[0] * fractional[0]
                + simulation_box.xy * fractional[1]
                + simulation_box.xz * fractional[2],
            lengths[1] * fractional[1] + simulation_box.yz * fractional[2],
            lengths[2] * fractional[2],
        ];
        for axis in 0..3 {
            let delta = if data.coordinates == BoundaryCoordinates::MassWeighted {
                cartesian[axis] * data.sqrt_masses[offset + axis]
            } else {
                cartesian[axis]
            };
            if !delta.is_finite() {
                return Err(SaddleError::NonFinite("LAMMPS boundary displacement"));
            }
            displacement.push(delta);
        }
    }
    Ok(displacement)
}

fn project_to_tangent(vector: &[f64], plane_normal: &[f64]) -> Result<Vec<f64>, SaddleError> {
    if vector.len() != plane_normal.len() {
        return Err(SaddleError::Shape(
            "LAMMPS boundary normal and plane normal dimensions differ".into(),
        ));
    }
    let denominator = dot(plane_normal, plane_normal)?;
    if denominator <= 0.0 {
        return Err(SaddleError::Invalid(
            "LAMMPS plane normal vanishes in boundary callback".into(),
        ));
    }
    let coefficient = dot(vector, plane_normal)? / denominator;
    let mut projected = Vec::with_capacity(vector.len());
    for axis in 0..vector.len() {
        projected.push(vector[axis] - coefficient * plane_normal[axis]);
    }
    Ok(projected)
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, SaddleError> {
    if left.len() != right.len() {
        return Err(SaddleError::Shape(
            "LAMMPS boundary callback dot-product dimensions differ".into(),
        ));
    }
    let mut sum = 0.0;
    for axis in 0..left.len() {
        sum += left[axis] * right[axis];
    }
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(SaddleError::NonFinite(
            "LAMMPS boundary callback dot product",
        ))
    }
}

fn normalize_in_place(vector: &mut [f64]) -> Result<(), SaddleError> {
    let mut norm_sq = 0.0;
    for value in vector.iter() {
        norm_sq += value * value;
    }
    if !norm_sq.is_finite() {
        return Err(SaddleError::NonFinite("LAMMPS boundary tangent normal"));
    }
    if norm_sq <= f64::EPSILON {
        return Err(SaddleError::Invalid(
            "LAMMPS boundary has no tangent-space normal component".into(),
        ));
    }
    let norm = norm_sq.sqrt();
    for value in vector.iter_mut() {
        *value /= norm;
    }
    Ok(())
}

fn mirror_position(position: &mut [f64], normal: &[f64], penetration: f64) {
    for axis in 0..position.len() {
        position[axis] -= 2.0 * penetration * normal[axis];
    }
}

fn mirror_velocity(velocity: &mut [f64], normal: &[f64]) {
    let mut component = 0.0;
    for axis in 0..velocity.len() {
        component += velocity[axis] * normal[axis];
    }
    for axis in 0..velocity.len() {
        velocity[axis] -= 2.0 * component * normal[axis];
    }
}

unsafe extern "C" fn hard_boundary_callback(
    user: *mut c_void,
    n_atoms: c_int,
    positions: *mut f64,
    velocities: *mut f64,
    simulation_box: *const c_void,
) -> c_int {
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user.is_null()
            || positions.is_null()
            || velocities.is_null()
            || simulation_box.is_null()
            || n_atoms < 0
        {
            return Ok(BOUNDARY_ERROR);
        }
        let data = unsafe { &mut *(user as *mut HardBoundaryData) };
        data.callback_calls = data.callback_calls.saturating_add(1);
        let coordinates = 3usize
            .checked_mul(n_atoms as usize)
            .ok_or_else(|| SaddleError::Invalid("LAMMPS atom count overflow".into()))?;
        let positions_slice = unsafe { slice::from_raw_parts_mut(positions, coordinates) };
        let velocities_slice = unsafe { slice::from_raw_parts_mut(velocities, coordinates) };
        let simulation_box = unsafe { &*(simulation_box as *const BoundaryBox) };
        apply_hard_boundary(data, positions_slice, velocities_slice, simulation_box)
    }));
    match run {
        Ok(Ok(status)) => status,
        Ok(Err(_)) | Err(_) => BOUNDARY_ERROR,
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.clear_boundaries();
        unsafe {
            (self.destroy)(self.session);
            self.session = std::ptr::null_mut();
            (self.finalize)();
        }
    }
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

    fn single_face_data(
        point: Vec<f64>,
        normal: Vec<f64>,
        plane_normal: Vec<f64>,
        coordinates: BoundaryCoordinates,
    ) -> HardBoundaryData {
        HardBoundaryData {
            faces: vec![BoundaryFace { point, normal }],
            plane_normal,
            coordinates,
            sqrt_masses: vec![1.0, 1.0, 1.0],
            callback_calls: 0,
        }
    }

    fn nonperiodic_box() -> BoundaryBox {
        BoundaryBox {
            lo: [-10.0; 3],
            hi: [10.0; 3],
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
            periodic: [0; 3],
            triclinic: 0,
        }
    }

    #[test]
    fn hard_boundary_accepts_inside_points() {
        let data = single_face_data(
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            BoundaryCoordinates::Cartesian,
        );
        let mut positions = vec![-0.5, 0.1, 0.0];
        let mut velocities = vec![0.2, 0.3, 0.0];
        let status =
            apply_hard_boundary(&data, &mut positions, &mut velocities, &nonperiodic_box())
                .unwrap();
        assert_eq!(status, BOUNDARY_ACCEPT);
        assert_eq!(positions, vec![-0.5, 0.1, 0.0]);
        assert_eq!(velocities, vec![0.2, 0.3, 0.0]);
    }

    #[test]
    fn hard_boundary_reflects_crossings() {
        let data = single_face_data(
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            BoundaryCoordinates::Cartesian,
        );
        let mut positions = vec![0.2, -0.1, 0.0];
        let mut velocities = vec![0.4, 0.3, 0.0];
        let status =
            apply_hard_boundary(&data, &mut positions, &mut velocities, &nonperiodic_box())
                .unwrap();
        assert_eq!(status, BOUNDARY_REFLECT);
        let expected_positions = [-0.2, -0.1, 0.0];
        let expected_velocities = [-0.4, 0.3, 0.0];
        for axis in 0..positions.len() {
            assert!((positions[axis] - expected_positions[axis]).abs() <= 32.0 * f64::EPSILON);
            assert!((velocities[axis] - expected_velocities[axis]).abs() <= 32.0 * f64::EPSILON);
        }
    }

    #[test]
    fn hard_boundary_callback_rejects_ill_defined_tangent_faces() {
        let mut data = Box::new(single_face_data(
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            BoundaryCoordinates::Cartesian,
        ));
        let mut positions = vec![0.0, 0.5, 0.0];
        let mut velocities = vec![0.0, -0.2, 0.0];
        let simulation_box = nonperiodic_box();
        let status = unsafe {
            hard_boundary_callback(
                data.as_mut() as *mut HardBoundaryData as *mut c_void,
                1,
                positions.as_mut_ptr(),
                velocities.as_mut_ptr(),
                (&raw const simulation_box).cast(),
            )
        };
        assert_eq!(status, BOUNDARY_ERROR);
    }

    #[test]
    fn hard_boundary_mass_weighted_mode_converts_back_to_cartesian() {
        let mut data = single_face_data(
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            BoundaryCoordinates::MassWeighted,
        );
        data.sqrt_masses = vec![2.0, 2.0, 1.0];
        let mut positions = vec![0.2, -0.3, 0.0];
        let mut velocities = vec![0.5, 0.1, 0.0];
        let status =
            apply_hard_boundary(&data, &mut positions, &mut velocities, &nonperiodic_box())
                .unwrap();
        assert_eq!(status, BOUNDARY_REFLECT);
        assert!((positions[0] + 0.2).abs() <= 64.0 * f64::EPSILON);
        assert!((velocities[0] + 0.5).abs() <= 64.0 * f64::EPSILON);
    }

    #[test]
    fn hard_boundary_uses_periodic_minimum_image() {
        let data = single_face_data(
            vec![9.8, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            BoundaryCoordinates::Cartesian,
        );
        let simulation_box = BoundaryBox {
            lo: [0.0; 3],
            hi: [10.0; 3],
            xy: 0.0,
            xz: 0.0,
            yz: 0.0,
            periodic: [1, 0, 0],
            triclinic: 0,
        };
        let mut positions = vec![0.2, 0.0, 0.0];
        let mut velocities = vec![0.5, 0.0, 0.0];
        let status =
            apply_hard_boundary(&data, &mut positions, &mut velocities, &simulation_box).unwrap();
        assert_eq!(status, BOUNDARY_REFLECT);
        assert!((positions[0] + 0.6).abs() <= 64.0 * f64::EPSILON);
        assert!((velocities[0] + 0.5).abs() <= 64.0 * f64::EPSILON);
    }
}
