//! Optimal hyperplanar transition state theory (OH-TST).
//!
//! Jóhannesson and Jónsson, J. Chem. Phys. 115, 9644 (2001),
//! doi:10.1063/1.1415499, take the dividing surface to be a hyperplane
//! and move it until its free energy is a maximum. By the variational
//! principle of transition state theory that plane gives the smallest
//! hyperplanar TST rate. Bligaard and Jónsson, Comput. Phys. Commun. 169,
//! 284 (2005), doi:10.1016/j.cpc.2005.03.064, follow the same optimum on
//! two-dimensional surfaces where the free energy of a line is a
//! one-dimensional integral.
//!
//! Coordinates are mass-weighted, `q = M^{1/2} x`. [`MassWeighted`] wraps
//! a Cartesian [`crate::PointSurface`] with its masses; a surface already
//! in these coordinates is used as it is. A [`Hyperplane`] is a point `p`
//! and a unit normal `ν` that points from the reactant to the product. Its
//! free energy is
//!
//! `A(p, ν) = -kT ln ∫ δ(ν · (q - p)) exp(-V(q) / kT) dq`.
//!
//! `A` depends on `p` only through `ν · p`, so the `n - 1` in-plane moves
//! of `p` are a gauge.
//!
//! A rate takes five calls. [`Hyperplane::from_min_mode`] seeds the plane
//! from a converged dimer. [`PlaneOptimizer`] climbs `A`. [`basin_density`]
//! gives `Z_H / Z_R` on a plane through the minimum from parallel planes.
//! [`reversible_work`] gives `ΔA` from that plane to the optimum, by
//! thermodynamic integration and by BAR. [`ohtst_rate`] returns
//! `k = sqrt(kT / 2π) (Z_H / Z_R) exp(-ΔA / kT)`, with unit mass in these
//! coordinates. [`transmission_coefficient`] gives `κ` from trajectories
//! shot from the optimal plane, counted at the end or at the first
//! crossing, with end states from the side of the plane or from a quench
//! into the basins; [`plane_transmission`] is the plain Bennett–Chandler
//! `κ`. Planes are sampled by projected BAOAB ([`PlaneIntegrator`]); a
//! [`HarmonicReference`] adds Stein control variates; and
//! [`anharmonic_correction`] gives the first-order correction to harmonic
//! TST from Hessian actions alone. No full Hessian is formed anywhere.
//! These follow the decisions of Sec. 2 of the OH-TST design survey.
//!
//! Nothing here names a system. A crystal at a fixed volume or lattice
//! parameter for each temperature is a surface evaluated in that cell:
//! wrap it in [`FixedCell`] with the cell for the temperature, then in
//! [`MassWeighted`].
//!
//! # Examples
//!
//! The first two calls on the skewed double well
//! `V = (x² − 1)² + ½ k(x) (y − x)²`, `k(x) = 4 (1 + 0.8 tanh 2x)`, already
//! in mass-weighted coordinates. A dimer converges on the saddle at the
//! origin and seeds the plane. The optimizer then climbs the plane free
//! energy at `kT = 0.1` with short Langevin samples.
//!
//! ```
//! use ndarray::{Array1, ArrayView1, array};
//! use rgsaddle::{
//!     Hyperplane, LangevinConfig, MinModeConfig, MinModeSession, MinModeStatus, PlaneOptimizer,
//!     PlaneOptimizerConfig, PointSurface, SaddleError,
//! };
//!
//! struct SkewedWell;
//!
//! impl PointSurface for SkewedWell {
//!     fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
//!         let (x, y) = (q[0], q[1]);
//!         let (well, tanh, valley) = (x * x - 1.0, (2.0 * x).tanh(), y - x);
//!         let k = 4.0 * (1.0 + 0.8 * tanh);
//!         let energy = well * well + 0.5 * k * valley * valley;
//!         let gx = 4.0 * x * well + 3.2 * (1.0 - tanh * tanh) * valley * valley - k * valley;
//!         Ok((energy, array![gx, k * valley]))
//!     }
//! }
//!
//! let search = MinModeConfig::default().with_force_tol(1e-8);
//! let mut dimer = MinModeSession::new(search, array![0.1, -0.05], array![1.0, 0.3])?;
//! assert_eq!(dimer.run(&SkewedWell, 2000)?.status, MinModeStatus::Converged);
//! let seed = Hyperplane::from_min_mode(&dimer)?.oriented_toward(array![1.0, 1.0].view());
//!
//! let sampler = LangevinConfig::default()
//!     .with_thermal(0.1)
//!     .with_time_step(0.1)
//!     .with_burn_in(200)
//!     .with_records(400)
//!     .with_seed(9);
//! let config = PlaneOptimizerConfig::default()
//!     .with_sampler(sampler)
//!     .with_max_records(400)
//!     .with_max_iterations(10);
//! let optimum = PlaneOptimizer::new(&seed, config)?.run(&SkewedWell)?;
//!
//! let normal = optimum.plane.normal();
//! assert!((normal.dot(&normal) - 1.0).abs() < 1e-12);
//! assert!(normal[0] > 0.0 && normal[1] > 0.0);
//! assert!(optimum.plane.point().iter().all(|c| c.abs() < 0.5));
//! # Ok::<(), SaddleError>(())
//! ```

mod anharmonic;
mod basin;
mod control;
mod gaussian;
mod mbar;
mod optimize;
mod path;
mod plane;
mod rate;
mod reactant;
mod sample;
mod spline;
mod ti;
mod transmission;
mod transport;
mod work;

pub use anharmonic::{AnharmonicConfig, AnharmonicCorrection, anharmonic_correction};
pub use basin::{BasinConfig, BasinDensity, BasinEnd, basin_density};
pub use control::HarmonicReference;
pub use gaussian::{ExactGaussianPlaneSampler, HyperplaneObservable, QuadraticModel};
pub use mbar::{MbarConfig, MbarFreeEnergy, mbar_free_energies};
pub use optimize::{
    PlaneMetric, PlaneOptimizer, PlaneOptimizerConfig, PlaneOptimizerStatus, PlaneOptimum,
    PlaneTrace, optimize_plane,
};
pub use path::PlanePath;
pub use plane::{FixedCell, Hyperplane, MassWeighted, mass_weighted_rigid_translations};
pub use rate::{
    OhTstRate, OhTstRateConfig, PlaneDensity, RateMultiplicity, flux_mass, ohtst_rate,
    ohtst_rate_with_config, plane_transmission, rate_from_parts,
};
pub use reactant::{ReactantDensity, reactant_density};
pub(crate) use sample::sample_plane_pilot_with_sampler;
pub use sample::{
    AnalyticSurfaceSampler, LangevinConfig, PlaneIntegrator, PlaneSample, Uncontrolled,
    sample_plane, sample_plane_with_sampler,
};
pub use ti::{
    AnalyticThermodynamicSampler, HarmonicExpansion, THERMODYNAMIC_DERIVATIVE,
    ThermodynamicDerivative, ThermodynamicIntegrationConfig, ThermodynamicIntegrationEstimate,
    ThermodynamicIntegrationNode, ThermodynamicSampler, ThermodynamicSurfaceSampler,
    harmonic_energy_at, restricted_thermodynamic_integration, thermodynamic_integration,
};
#[allow(deprecated)]
pub use transmission::{
    AnalyticSurfaceDynamics, Basins, Counting, EndState, Endpoint, HyperplaneDynamics,
    PermutationDistance, QuenchConfig, QuenchedTransitionRegion, StateDistance,
    TransmissionCoefficient, TransmissionConfig, TransmissionEstimate, transmission_coefficient,
    transmission_coefficient_on_surface,
};
pub use transport::{GaussianTransport, Transport};
#[allow(deprecated)]
pub use work::{
    ReversibleWork, ReversibleWorkConfig, WorkEstimate, WorkNode, reversible_work,
    reversible_work_along, reversible_work_along_on_surface, reversible_work_on_surface,
};
