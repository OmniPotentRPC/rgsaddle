//! Band, minimum-mode, and index-1 Newton saddle mechanics.
//!
//! The band and the minimum-mode search take one [`rgmin::Solver`]
//! step on a force this crate assembled. The host owns the loop.
//! [`band::BandSession::run`] and [`minmode::MinModeSession::run`]
//! are convenience loops over `step`.
//! A climbing band refines its climbing image by the lowest mode once
//! that image is stable (OCI-NEB: Goswami, Gunde, and Jónsson, Front.
//! Chem. 14 (2026), doi:10.3389/fchem.2026.1807063);
//! [`ocineb::OcinebSession`] is that band with the published controls.
//! [`autoneb::AutoNeb`] inserts
//! images where the energy gap or the segment length is worst,
//! relaxes that window, and climbs. The full band runs only when
//! requested.
//! Image blocks are stored by image index after every rank finishes,
//! so the band does not depend on which block finished first.
//!
//! [`lineb::LinebSession`] minimizes the fixed-energy tunneling action
//! of Ásgeirsson, Arnaldsson and Jónsson, J. Chem. Phys. 148, 102334
//! (2018), doi:10.1063/1.5007180, and [`lineb::hqtst_rate`] turns the
//! resulting instanton into a harmonic quantum TST rate.
//!
//! [`nichols::Index1Session`] is the index-1 search on a dense
//! Hessian: Baker's restricted-step partitioned RFO (maximize the
//! lowest mode, minimize the rest, with the step inside a Euclidean
//! trust sphere) and a Powell or Bofill update. The i-PI Nichols
//! shift remains [`nichols::nichols_step`] for a host that already
//! holds a spectrum.
//!
//! Force assembly is pure: tangents (Mills–Jonsson–Schenter simple,
//! Henkelman–Jonsson improved), springs (uniform, energy-weighted,
//! Onsager–Machlup), projections (plain elastic band, NEB,
//! doubly-nudged), and the climbing-image force, ported from eOn's
//! NEBTangent / NEBSpringForce / NEBForceProjection with identical
//! branch structure.
//!
//! Positions are unwrapped Cartesian. [`BandConfig::cell`] makes the
//! band mechanics take minimum-image differences; the host surface still
//! receives the unwrapped positions and owns its potential's periodicity.
//!
//! [`sparse_hessian::sparse_hessian`] recovers a cutoff Hessian by a
//! direct Curtis-Powell-Reid partition: a smallest-last distance-2
//! colouring and one batched gradient difference per colour and
//! Cartesian axis.
//!
//! Each entry point opens with a runnable example on a small analytic
//! surface: [`ArtSession`], [`BandSession`], [`AutoNeb`], [`OcinebSession`],
//! [`MinModeSession`], [`Index1Session`], [`SolidDimer`], [`SolidNeb`],
#![cfg_attr(
    feature = "ohtst",
    doc = "[`Stencil`], and the optional [`ohtst`] module (crate feature `ohtst`). With the `capi` feature the C"
)]
#![cfg_attr(
    not(feature = "ohtst"),
    doc = "[`Stencil`], and the optional `ohtst` module (crate feature `ohtst`). With the `capi` feature the C"
)]
//! band, min-mode and stencil handles have the same examples in `capi`.
//! The guides and tutorials are at
//! <https://omnipotentrpc.github.io/rgsaddle/>.

#![deny(missing_docs)]

#[macro_use]
mod config;

#[cfg(feature = "ohtst")]
pub mod adaptive_ohtst;
pub mod art;
pub mod autoneb;
pub mod band;
pub mod bennett;
#[cfg(feature = "capi")]
pub mod capi;
#[cfg(all(feature = "capi", feature = "ohtst"))]
pub mod capi_hyperplane;
pub mod connect;
pub mod defect_pair;
pub mod elastic_dipole;
pub mod error;
pub mod geodesic;
pub mod idpp;
pub mod indexk;
#[cfg(feature = "lammps")]
pub mod lammps;
pub mod line_neb;
pub mod lineb;
pub mod minmode;
pub mod nichols;
pub mod npt;
pub mod occupancy;
pub mod ocineb;
#[cfg(feature = "ohtst")]
pub mod ohtst;
pub mod pafi;
pub mod pair_network;
pub mod prfo;
pub mod projection;
pub mod projector;
pub mod region;
pub mod relax;
pub mod saddle_index;
pub mod solid_dimer;
pub mod solid_prefactor;
pub mod solid_state;
pub mod spring;
pub mod ssneb;
pub mod stencil;
pub mod strain_rate;
pub mod tangent;

// `translated_gradient` is crate-private, so the golden-bit test lives in
// this module under `cfg(test)`. The Kani harnesses are the same module
// under `cfg(kani)`, which `cargo test` does not set, so the default
// build does not name the Kani crate.
#[cfg(any(test, kani))]
mod kani_proofs;

#[cfg(all(test, feature = "verify"))]
mod differential;

mod spline;
mod sqnm;
mod stats;
#[cfg(feature = "ohtst")]
pub use adaptive_ohtst::{
    ADAPTIVE_OHTST_MAX_SAMPLES, ADAPTIVE_OHTST_PILOT_SAMPLES, ADAPTIVE_OHTST_SAMPLE_SAFETY,
    ADAPTIVE_OHTST_TARGET_LOG_ERROR, AdaptiveOhTstConfig, AdaptiveOhTstEstimate,
    AdaptiveOhTstProfile, AdaptiveOhTstStatus, adaptive_ohtst_sample_count,
    adaptive_ohtst_string_rate,
};
pub use art::{
    ArtBatch, ArtConfig, ArtDuplicate, ArtForceCall, ArtOutcome, ArtPhase, ArtReport,
    ArtReversibility, ArtSeed, ArtSession, ArtStatus, atom_mask, connection_verdict, full_mask,
    run_events, structure_distance,
};
pub use autoneb::{AutoNeb, AutoNebInterpolant, AutoNebReport, AutoNebTuning};
pub use band::{
    BandConfig, BandEvaluation, BandReport, BandSession, BandStatus, BandSurface, Cell, CiConfig,
    DynamicNeb, ExpBandPrecon, FeaNebReport, FeaNebSession, FreeEndConfig, FreeEndReport,
    FreeEndSide, GrowingStringConfig, GrowingStringPhase, GrowingStringReport,
    GrowingStringSession, MinModeRefine, SolidState, StepRefine, StringConfig, StringSession,
    ZoomConfig, climbing_image,
};
pub use bennett::{ShootingConfig, Transmission, bennett_chandler};
pub use connect::{ConnectConfig, ConnectProfile, ConnectReport, connect};
pub use defect_pair::{
    BasinSaddle, CaptureRecord, CellMode, DefectPairConfig, DimerSeed, DirectionClass,
    FREE_CRYSTAL, FreeMigration, HopOutcome, HopTemplate, MatchedCapture, OuterHop, PairSetup,
    PeriodicSinkRate, PeriodicSinkSurvival, RateCatalogue, RateEntry, RecombinationVolume,
    SPONTANEOUS_BARRIER_EV, SiteOutcome, SphereLanding, VolumePlacement, VolumeSite,
    attempt_frequency, capture_is_spontaneous, capture_is_spontaneous_within, classify_separation,
    debye_hit_probability, debye_matched_radius, debye_radius, debye_resistance, defect_pair_rates,
    descend_to_floor, effective_capture_radius, embed_free_migration, embed_sphere_return,
    kinetic_recombination_volume, matched_capture_radius, mean_absorption_time, pair_separation,
    periodic_capture_log_likelihood, periodic_sink_rate, periodic_sink_survival, product_tangent,
    seeded_saddles, segment_band, shell_pair_factor, smoluchowski_rate, sphere_hit_probability,
    spherical_return_probability, volume_from_frequency,
};
pub use elastic::{
    ElasticMultipole, LocalField, Octupole, Quadrupole, StrainGradient, quadrupole_energy,
};
pub use elastic_dipole::{
    AnisotropicGreen, AnisotropicGreenConfig, CorrectedEnergy, CorrectedRate, DefectDipole,
    ElasticConstants, ElasticCorrection, ElasticPairState, ElasticRate, GreenSecondDerivative,
    InteractionEstimate, PeriodicElastic, PeriodicElasticConfig, Tensor4,
};
pub use error::SaddleError;
pub use field::{DefectSource, DirectSolver, FieldSolver, FmmSolver, Kelvin};
pub use geodesic::{
    GeodesicChart, GeodesicConfig, GeodesicLengths, GeodesicReport, MORSE_ALPHA, MORSE_BETA,
    MorsePairChart, geodesic_interpolate, geodesic_lengths, geodesic_path,
};
pub use idpp::{IdppConfig, SidppConfig, idpp_path, linear_path, sidpp_path};
pub use indexk::{
    BlockEigenKind, HessianProduct, IndexKConfig, IndexKReport, IndexKSession, WithHessian,
    indexk_step, reflect_force, translated_force,
};
#[cfg(feature = "lammps")]
pub use lammps::{
    LammpsHyperplaneSampler, LammpsSamplerConfig, LammpsSystem, LammpsThermodynamicReference,
    LammpsThermostatKind, LammpsThermostatRecord,
};
pub use line_neb::{
    EndpointBelowEnergy, FixedEnergySeed, LineNebConfig, LineNebEndpointViolation, LineNebReport,
    LineNebSession, LineNebStatus, fixed_energy_action, fixed_energy_action_force,
    fixed_energy_mep_seed,
};
pub use lineb::{
    Continuation, HqtstRate, LinebCalls, LinebConfig, LinebPath, LinebRateOrbit, LinebReport,
    LinebSession, LinebStatus, LinebTurning, Masses, continue_from_mep,
    continue_from_mep_with_observer, continue_from_path_with_observer, hqtst_rate,
    hqtst_rate_with_orbit,
};
#[cfg(any(test, feature = "lineb-interpolated-hessian"))]
pub use lineb::{
    cartesian_image_hessian, hqtst_rate_from_image_hessians, hqtst_rate_interpolated_hessian,
    hqtst_rate_interpolated_hessian_observed, hqtst_rate_interpolated_hessian_with_step,
};
pub use minmode::{
    ArtNConfig, FiniteDifference, MinModeConfig, MinModeKind, MinModeReport, MinModeSession,
    MinModeStatus, ModeEstimate, PointSurface, RaisedStep, halve_raised_energy,
    rotation_force_error,
};
pub use nichols::{
    HessianUpdate, Index1Config, Index1Report, Index1Session, Index1Status, NicholsMode,
    bofill_update, cap_max_abs, nichols_displacement, nichols_step, powell_update,
};
pub use npt::{NptBarostat, NptConfig, NptReport, npt_lattice};
pub use occupancy::{SiteOccupancy, site_occupancy};
pub use ocineb::{
    MIN_ALIGNMENT, OcinebConfig, OcinebPhase, OcinebReport, OcinebSession, OcinebStatus,
    backoff_threshold, initial_threshold, mode_alignment, should_trigger, success_threshold,
    walk_helped,
};
#[cfg(feature = "ohtst")]
#[allow(deprecated)]
pub use ohtst::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, AnalyticThermodynamicSampler,
    AnharmonicConfig, AnharmonicCorrection, BasinConfig, BasinDensity, BasinEnd, Basins, Counting,
    EndState, Endpoint, ExactGaussianPlaneSampler, FixedCell, GaussianTransport, HarmonicExpansion,
    HarmonicReference, HyperplaneDynamics, HyperplaneObservable, LangevinConfig, MassWeighted,
    OhTstRate, PermutationDistance, PlaneDensity, PlaneIntegrator, PlaneMetric, PlaneOptimizer,
    PlaneOptimizerConfig, PlaneOptimizerStatus, PlaneOptimum, PlanePath, PlaneSample, PlaneTrace,
    QuadraticModel, QuenchConfig, QuenchedTransitionRegion, RateMultiplicity, ReactantDensity,
    ReversibleWork, ReversibleWorkConfig, StateDistance, THERMODYNAMIC_DERIVATIVE,
    ThermodynamicDerivative, ThermodynamicIntegrationConfig, ThermodynamicIntegrationEstimate,
    ThermodynamicIntegrationNode, ThermodynamicSampler, ThermodynamicSurfaceSampler,
    TransmissionCoefficient, TransmissionConfig, TransmissionEstimate, Transport, Uncontrolled,
    WorkEstimate, WorkNode, anharmonic_correction, basin_density, flux_mass, harmonic_energy_at,
    mass_weighted_rigid_translations, ohtst_rate, optimize_plane, plane_transmission,
    reactant_density, restricted_thermodynamic_integration, reversible_work, reversible_work_along,
    reversible_work_along_on_surface, reversible_work_on_surface, sample_plane,
    sample_plane_with_sampler, thermodynamic_integration, transmission_coefficient,
    transmission_coefficient_on_surface,
};
pub use pafi::{
    ArcPath, BalancedBarriers, FixedEnergyTrajectory, FreeEnergyProfile, HyperplaneAverage,
    PafiConfig, PafiIntegrator, PafiMeanForce, PathFrame, PathGeometry, TrajectoryPoint,
    anchored_barriers, arc_at_time, average_hyperplane, average_hyperplane_with_sampler,
    balanced_barriers, integrate_profile, pafi_profile, zero_temperature_barrier,
};
pub use pair_network::{
    Mat3i, PairConfig, PairEdge, PairExit, PairHop, PairMove, PairNetwork, PairStateClass,
    PairTarget, SensitivityCover, SurrogateCalibration, Vec3i, calibrate_surrogate, cubic_group,
    sensitivity_cover,
};
pub use prfo::{PrfoKind, partitioned_rfo_eigen, restricted_prfo_displacement};
pub use projection::ProjectionKind;
pub use projector::{
    Hyperplane, HyperplaneCoordinates, HyperplaneEnsemble, HyperplaneModes, HyperplaneObservation,
    HyperplaneSampleBudget, HyperplaneSampler, ManifoldTangent, ProjectedConstraint,
    TangentProjector, UnitNormal,
};
pub use region::{
    AffineSymmetry, BoundaryViolation, CrystalSymmetryConfig, CrystalSymmetryOperation,
    CrystalSymmetryRegion, PathSlab, PeriodicSymmetryOperations, PlanarBoundary, Region,
    RegionIntersection, SymmetryOperations, SymmetrySource, SymmetryUniqueRegion, reflect_velocity,
};
pub use relax::{
    DEFAULT_FORCE_TOL, DEFAULT_MAX_STEPS, RelaxConfig, Relaxation, max_atom_force,
    relax_to_tolerance,
};
pub use saddle_index::{SaddleEscape, SaddleIndex, SaddleIndexCheck, saddle_index};
pub use solid_dimer::{
    CellMask, IndexCheck, IndexReport, JointChart, STRAIN_VOIGT, SolidDimer, SolidDimerConfig,
    SolidDimerReport, StressCheck, VineyardIndexError, VineyardIndexReport, certify_vineyard_pair,
    check_cauchy_stress, finite_difference_cauchy_stress, index_at, virial,
};
pub use solid_prefactor::{
    JointHessian, JointHessianConfig, JointVineyard, JointVineyardConfig, joint_hessian,
    joint_vineyard_prefactor,
};
pub use spring::SpringKind;
pub use ssneb::{SolidImage, SolidNeb, joint_coordinate};
pub use stats::EquilibrationReport;
pub use stencil::{
    BAND_PRESET_SEAM, BandBuilder, Chain, Climb, EvalPool, Mesh, MeshEvent, PointBuilder, Preset,
    Role, Stencil, StencilReport, StencilStatus,
};
pub use strain_rate::{
    BOLTZMANN_EV_PER_K, CurvatureBound, ElasticDipole, Estimate, HopVariant, LINEAR_FOLD_EV,
    NEGATIVE_CURVATURE, SaddleSeed, SearchReason, Strain, StrainResponse, TranslationalSplit,
    Trust, TrustPolicy, activation_volume, angular_scale_rad_s, arrhenius,
    compliance_from_stiffness, cubic_rotations, double_dot, elastic_dipole, rotation_x,
    split_translational_modes, symmetrize, transform_tensor, vineyard_prefactor,
};
pub use tangent::TangentKind;

pub mod cell_log;
pub mod constraints;
pub mod covalent;
pub mod eigensolve;
pub mod elastic;
pub mod field;
pub mod force;
pub mod force_match;
pub mod geom;
pub mod internal;
pub mod kappa;
pub mod linalg;
pub mod mic;
pub mod pes;
pub mod pes_internal;
pub mod prfo_restricted;
pub mod qn;
pub mod restricted;
pub mod rfo;
pub mod samd;
pub mod seed_direction;
pub mod sella_min;
pub mod sella_saddle;
pub mod topology;
pub mod vocn;

pub use cell_log::{CellChart, expm_3x3, logm_3x3};

pub use constraints::{Constraints, Equality, InternalCounts};

pub use eigensolve::{
    EigenDevice, ExpandKind, dense_logabsdet, eigh_on, exact_eigh, exact_eigvalsh, expand,
    faer_eigh, lowest_on, rayleigh_ritz, rayleigh_ritz_iter, relative_zero_tolerance, solve_dense,
    solve_dense_matrix, spd_logdet,
};

pub use force::ForceGate;

pub use force_match::{covalent_pairs, force_match_hessian};

pub use geom::{SellaGeom, TrustSchedule, trust_update, update_trust};

pub use internal::{CartAxis, Displacement, InternalSlot, Rotation, Translation};

pub mod sparse_hessian;
pub use sparse_hessian::{
    Colouring, LowestEigenpairs, SparseHessian, SparseHessianOpts, SparsityPattern,
    action_pair_blocks, distance2_colouring, eval_displacements, one_atom_hessian,
    prefactor_hessian, sparse_hessian,
};

pub use linalg::{
    NumericalHessian, modified_gram_schmidt, numerical_hvp, numerical_hvp_on, numerical_hvp_proj,
    project_hvp, retract_hvp, transport_hvp,
};

pub use pes::CartesianPes;

pub use pes_internal::{
    CellCartesianPes, CellInternalPes, HessianCarry, InternalPes, RebuildReason, RebuildReport,
    SellaPes, niggli_reduce_cell, niggli_reduce_vectors, place_perp_dummy,
};

pub use covalent::{atomic_number, covalent_radius, element_symbol};
pub use topology::{
    AtomMap, DummyHost, InternalTopology, LinearBend, PlanarWatch, TopologyCheck, TopologyConfig,
    check_chart,
};

pub use qn::{
    HessUpdate, QuasiNewton, StepperKind, get_stepper, retract_qn, symmetrize_y, symmetrize_y_cols,
    symmetrize_y_pair, update_h, update_h_ms, update_hessian, update_hessian_cols,
};

pub use restricted::{
    InternalWeights, MaxInternalStep, RAS_SYNONYMS, RestrictedAtomicStep, RestrictedKind,
    TRUST_SYNONYMS, TrustRegion, mis_clip, ras_cons, weights_for_equalities,
};

pub use rfo::{RationalFunctionOptimization, rfo_stepper};

pub use samd::{
    SamdConfig, SamdReport, SamdSession, VelocitySofteningConfig, VelocitySofteningReport,
    project_velocity, retract_samd, soften_velocity_on, transport_velocity, verlet_increment,
};
pub use seed_direction::{
    ModeSweepDirectionPlan, SoftModeDirectionPlan, hessian_mode_sweep_directions,
    mode_sweep_schedule, soft_mode_then_full_directions, spectral_gap_rank,
};

pub use sella_min::{SellaMinConfig, SellaMinReport, SellaMinSession};

pub use sella_saddle::{SellaSaddleConfig, SellaSaddleReport, SellaSaddleSession};

pub use vocn::{Found as VocnFound, Primitive as VocnPrimitive};

pub use prfo::{
    PartitionedRationalFunctionOptimization, prfo_get_s, prfo_stepper, prfo_trust_region,
};

pub mod irc;
pub use irc::{IrcConfig, IrcDirection, IrcKind, IrcReport, IrcSession};

#[cfg(feature = "capi")]
pub mod capi_kappa;

pub mod afir;
pub use afir::{afir_force, try_afir_force};

pub mod reparam;
pub use reparam::{Reparametrization, Spline, SplineEnds, SplineSample, reparametrize};

pub mod rtr;
pub use rtr::{
    BandForces, BandRtr, RtrConfig, RtrRadius, RtrReport, TcgResult, TcgStop, band_forces,
    reparametrize_equal_arc, truncated_cg,
};

#[cfg(feature = "readcon")]
pub mod io;
#[cfg(feature = "readcon-db")]
pub use io::store_band;
#[cfg(feature = "readcon")]
pub use io::{
    ConImage, ConSpecies, MolecularFrame, frame_from_con, frame_from_con_nth, frames_from_con,
    write_band, write_restart,
};

pub mod gpu;
pub use gpu::{
    DEFAULT_MIN_DIM, GPU_MIN_DIM, GpuPolicy, clear_oom_floor, cuda_available, cuda_device,
    fail_next_cuda_claim, gpu_eigh, gpu_eigh_env, gpu_eigh_t, gpu_eigh_with, gpu_ok, gpu_ok_env,
    gpu_project, gpu_project_env, gpu_project_with, gpu_qr, gpu_qr_env, gpu_qr_with,
    lock_oom_for_test, oom_floor, record_oom, to_gpu, to_gpu_matrix, to_gpu_view,
};

pub mod trust;
