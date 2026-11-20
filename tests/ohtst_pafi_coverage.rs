//! Observable-behavior tests for PAFI, OH-TST thermodynamic integration,
//! transmission coefficients, Gaussian reference samplers, reversible work,
//! rate multipliers, and plane optimization on toy surfaces.

#![cfg(feature = "ohtst")]

use std::num::NonZeroUsize;
use std::sync::Arc;

use ndarray::{Array1, Array2, ArrayView1, array};
use rgmin::Method;
use rgsaddle::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, ArcPath, Basins, Counting, EndState,
    ExactGaussianPlaneSampler, FixedCell, GaussianTransport, HarmonicExpansion, HarmonicReference,
    Hyperplane, HyperplaneSampleBudget, HyperplaneSampler, LangevinConfig, MassWeighted, OhTstRate,
    PafiConfig, PermutationDistance, PlanarBoundary, PlaneMetric, PlaneOptimizer,
    PlaneOptimizerConfig, PlanePath, PointSurface, QuadraticModel, QuenchConfig,
    QuenchedTransitionRegion, RateMultiplicity, Region, ReversibleWorkConfig, SaddleError,
    THERMODYNAMIC_DERIVATIVE, ThermodynamicDerivative, ThermodynamicIntegrationConfig,
    ThermodynamicSampler, ThermodynamicSurfaceSampler, TransmissionConfig, Transport, WorkEstimate,
    flux_mass, integrate_profile, mass_weighted_rigid_translations, ohtst_rate, pafi_profile,
    plane_transmission, reactant_density, restricted_thermodynamic_integration,
    reversible_work_along, thermodynamic_integration, transmission_coefficient,
    zero_temperature_barrier,
};

include!("common/ohtst_surfaces.rs");
use ohtst_surfaces::{LepsHo, SkewedWell};

struct Quartic2D;

impl PointSurface for Quartic2D {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        Ok((
            0.5 * x * x + 0.5 * y * y + 0.1 * y.powi(4),
            array![x, y + 0.4 * y.powi(3)],
        ))
    }

    fn excluded_modes(&self, q: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        Ok(Array2::zeros((0, q.len())))
    }
}

struct TwoAtomHarmonic;

impl PointSurface for TwoAtomHarmonic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let dx = x[3] - x[0];
        let dy = x[4] - x[1];
        let dz = x[5] - x[2];
        let e = 0.5 * ((dx * dx - 1.0).powi(2) + dy * dy + dz * dz)
            + 0.05 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        let g_dx = 2.0 * dx * (dx * dx - 1.0);
        Ok((
            e,
            array![
                -g_dx + 0.1 * x[0],
                -dy + 0.1 * x[1],
                -dz + 0.1 * x[2],
                g_dx,
                dy,
                dz
            ],
        ))
    }
}

struct FailingPlaneSurface;

impl PointSurface for FailingPlaneSurface {
    fn eval(&self, _x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Err(SaddleError::Surface("injected surface failure".into()))
    }
}

#[test]
fn pafi_adaptive_profile_zero_temp_barrier_and_region_reflection() {
    let nodes = [array![-1.0, -1.0], array![0.0, 0.0], array![1.0, 1.0]];
    let path = ArcPath::from_positions(&nodes).unwrap();
    let total_arc = *path.arcs().last().unwrap();
    let arcs: Vec<f64> = (0..=8).map(|i| total_arc * i as f64 / 8.0).collect();

    // Zero-temperature integral recovers the potential barrier (1.0) on SkewedWell.
    let zero_t = zero_temperature_barrier(&SkewedWell, &path, &arcs).unwrap();
    assert!(
        (zero_t.barrier - 1.0).abs() < 1e-3,
        "barrier={}",
        zero_t.barrier
    );
    assert!(zero_temperature_barrier(&SkewedWell, &path, &[0.0]).is_err());

    // Hard PlanarBoundary keeps all PAFI samples on y >= -1.15 while containing the path.
    let boundary = Arc::new(
        PlanarBoundary::containing(
            array![0.0, -1.15],
            array![0.0, 1.0],
            array![0.0, 0.0].view(),
        )
        .unwrap(),
    );
    let cfg = PafiConfig::default()
        .with_thermal(0.1)
        .with_time_step(5e-3)
        .with_thermalize(50)
        .with_stride(2)
        .with_samples(60)
        .with_blocks(4)
        .with_seed(7)
        .with_fix_com(false)
        .with_target_error(Some(1e-4))
        .with_max_rounds(2)
        .with_region(boundary);
    let (profile, planes) = pafi_profile(&SkewedWell, &path, &arcs, &cfg).unwrap();
    assert!(
        profile.barrier > 0.5 && profile.barrier < 1.5,
        "barrier={}",
        profile.barrier
    );
    assert!(profile.samples > 60 * arcs.len());
    const ADAPTIVE_GROWTH_BOUND: usize = 8;
    let adaptive_limit = cfg.samples + cfg.samples * ADAPTIVE_GROWTH_BOUND;
    assert!(profile.samples <= adaptive_limit * arcs.len());
    for plane in &planes {
        for q in &plane.retained {
            assert!(q[1] >= -1.15 - 1e-10, "violated boundary y={}", q[1]);
        }
    }

    // Two-atom PAFI exercises fix_com = true path.
    let path2 = ArcPath::from_positions(&[
        array![-0.5, 0.0, 0.0, 0.5, 0.0, 0.0],
        array![0.5, 0.0, 0.0, -0.5, 0.0, 0.0],
    ])
    .unwrap();
    let cfg2 = PafiConfig::default()
        .with_thermal(0.05)
        .with_time_step(2e-3)
        .with_thermalize(20)
        .with_stride(2)
        .with_samples(40)
        .with_blocks(4)
        .with_fix_com(true)
        .with_com_atoms(NonZeroUsize::new(2).expect("two atoms"));
    let (prof2, _) = pafi_profile(&TwoAtomHarmonic, &path2, &[0.0, 0.5, 1.0], &cfg2).unwrap();
    assert!(prof2.barrier.is_finite());

    // integrate_profile error branches.
    assert!(integrate_profile(&[0.0], &[0.0], &[0.0]).is_err());
    assert!(integrate_profile(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]).is_err());
    assert!(integrate_profile(&[0.0, 1.0], &[0.0, f64::NAN], &[0.0, 0.0]).is_err());
    assert!(integrate_profile(&[0.0, 1.0], &[0.0, 1.0], &[0.0, -1.0]).is_err());
}

#[test]
fn ti_harmonic_expansion_surface_sampler_and_restricted_ti() {
    let reference = HarmonicReference::new(array![0.0, 0.0])
        .with_point(array![0.0, 0.0])
        .with_hvp_step(1e-4)
        .with_tolerance(1e-8);
    let expansion =
        HarmonicExpansion::new(&Quartic2D, reference.clone()).with_reference(reference.clone());
    let q = array![0.0, 0.5];
    let (e_exp, g_exp) = expansion.eval(q.view()).unwrap();
    assert!((e_exp - 0.125).abs() < 1e-8);
    assert!((g_exp[1] - 0.5).abs() < 1e-6);
    assert!(
        expansion
            .hessian_vector(q.view(), array![0.0, 1.0].view())
            .unwrap()
            .is_some()
    );
    assert_eq!(expansion.excluded_modes(q.view()).unwrap().nrows(), 0);

    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let ti_cfg = ThermodynamicIntegrationConfig::default()
        .with_thermal(0.4)
        .with_nodes(4)
        .with_thermalize(50)
        .with_stride(2)
        .with_samples(120)
        .with_blocks(4)
        .with_seed(13);

    // ThermodynamicSurfaceSampler runs lambda TI on Quartic2D directly.
    let dynamics = LangevinConfig::default()
        .with_thermal(0.4)
        .with_time_step(0.05)
        .with_friction(1.0);
    let mut surf_sampler =
        ThermodynamicSurfaceSampler::new(&Quartic2D, reference.clone(), dynamics).unwrap();
    let full_e = surf_sampler.energy(q.view()).unwrap().expect("full energy");
    assert!((full_e - 0.13125).abs() < 1e-12, "full_e={full_e}");
    assert!(((full_e - e_exp) - 0.00625).abs() < 1e-8);
    assert!(
        surf_sampler
            .constrained_directions(&plane)
            .unwrap()
            .is_empty()
    );
    let ti_est = thermodynamic_integration(&mut surf_sampler, &plane, &ti_cfg).unwrap();
    assert!(ti_est.free_energy > 0.0 && ti_est.error >= 0.0);

    // Missing lambda fails ThermodynamicSurfaceSampler.
    let no_lambda = HyperplaneSampleBudget::default()
        .with_thermal(0.4)
        .with_seed(1)
        .with_thermalize(10)
        .with_stride(1)
        .with_samples(20);
    assert!(surf_sampler.sample_hyperplane(&plane, &no_lambda).is_err());

    // Restricted TI enforces PlanarBoundary via HarmonicReference sampler.
    let bad_ref = HarmonicReference::new(array![0.0]).with_hvp_step(0.0);
    assert!(
        ThermodynamicSurfaceSampler::new(&Quartic2D, bad_ref, LangevinConfig::default()).is_err()
    );

    // Shifted QuadraticModel setters affect evaluated energy and constrained Gaussian mean.
    let shifted = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]])
        .with_center(array![1.0, -0.5])
        .with_energy(0.3)
        .with_gradient(array![0.2, -0.6])
        .with_hessian(array![[2.0, 0.0], [0.0, 3.0]]);
    let mut shifted_gauss = ExactGaussianPlaneSampler::new(shifted).unwrap();
    // At q = [0.0, 0.5], dq = [-1.0, 1.0]: 0.3 + (0.2*(-1) + -0.6*1) + 0.5*(2*1 + 3*1) = 2.0.
    assert!((shifted_gauss.energy(q.view()).unwrap().unwrap() - 2.0).abs() < 1e-12);

    // ExactGaussianPlaneSampler + restricted_thermodynamic_integration matches analytic moment
    // E_0[0.1 * y^4] = 0.1 * 3 * (0.4)^2 = 0.048 on x = 0.
    let harmonic = QuadraticModel::new(array![0.0, 0.0], array![[1.0, 0.0], [0.0, 1.0]]);
    let deriv = ThermodynamicDerivative::new(&Quartic2D, reference.clone())
        .with_reference(reference.clone());
    let mut gauss = ExactGaussianPlaneSampler::new(harmonic.clone())
        .unwrap()
        .with_observable(THERMODYNAMIC_DERIVATIVE, &deriv);
    assert!((gauss.energy(q.view()).unwrap().unwrap() - 0.125).abs() < 1e-12);
    let r_est = restricted_thermodynamic_integration(&mut gauss, &plane, &ti_cfg).unwrap();
    assert!(
        (r_est.free_energy - 0.048).abs() < 4.0 * r_est.error + 0.015,
        "got {} +- {}",
        r_est.free_energy,
        r_est.error
    );

    // Bare ExactGaussianPlaneSampler omits THERMODYNAMIC_DERIVATIVE until ThermodynamicSampler
    // attaches V_full - V_harmonic = 0.1 * y^4 and strips temporary positions when unrequested.
    let mut bare = ExactGaussianPlaneSampler::new(harmonic).unwrap();
    let bare_ens = bare
        .sample_hyperplane(
            &plane,
            &HyperplaneSampleBudget::default()
                .with_thermal(0.4)
                .with_seed(2)
                .with_samples(4)
                .with_observable(THERMODYNAMIC_DERIVATIVE),
        )
        .unwrap();
    assert!(
        bare_ens
            .observations
            .iter()
            .all(|o| !o.observables.contains_key(THERMODYNAMIC_DERIVATIVE))
    );
    let deriv2 = ThermodynamicDerivative::new(&Quartic2D, reference);
    let mut wrapped = ThermodynamicSampler::new(bare, deriv2);
    let ens_with_pos = wrapped
        .sample_hyperplane(
            &plane,
            &HyperplaneSampleBudget::default()
                .with_thermal(0.4)
                .with_seed(2)
                .with_samples(8)
                .with_positions(true)
                .with_observable(THERMODYNAMIC_DERIVATIVE),
        )
        .unwrap();
    for obs in &ens_with_pos.observations {
        let pos = obs.position.as_ref().expect("position retained");
        let dv = obs.observables[THERMODYNAMIC_DERIVATIVE];
        assert!(
            (dv - 0.1 * pos[1].powi(4)).abs() < 1e-6,
            "dv={dv} pos={pos}"
        );
    }
    let budget_no_pos = HyperplaneSampleBudget::default()
        .with_thermal(0.4)
        .with_seed(2)
        .with_thermalize(0)
        .with_stride(1)
        .with_samples(16)
        .with_positions(false)
        .with_observable(THERMODYNAMIC_DERIVATIVE);
    let ens = wrapped.sample_hyperplane(&plane, &budget_no_pos).unwrap();
    assert_eq!(ens.observations.len(), 16);
    assert!(ens.observations.iter().all(|o| o.position.is_none()));
    for (no_pos, with_pos) in ens.observations.iter().zip(&ens_with_pos.observations) {
        assert!(
            (no_pos.observables[THERMODYNAMIC_DERIVATIVE]
                - with_pos.observables[THERMODYNAMIC_DERIVATIVE])
                .abs()
                < 1e-12
        );
    }

    // Config validation errors.
    assert!(
        thermodynamic_integration(&mut surf_sampler, &plane, &ti_cfg.clone().with_thermal(0.0))
            .is_err()
    );
    assert!(
        thermodynamic_integration(&mut surf_sampler, &plane, &ti_cfg.clone().with_nodes(0))
            .is_err()
    );
    assert!(
        thermodynamic_integration(
            &mut surf_sampler,
            &plane,
            &ti_cfg.clone().with_observable("")
        )
        .is_err()
    );
}

#[test]
fn transmission_quench_permutation_and_quenched_region() {
    let cell = rgsaddle::mic::Cell::from_vectors(
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [0.0; 3],
    )
    .unwrap();
    let pd = PermutationDistance::new(vec![0, 0])
        .with_species(vec![1, 1])
        .with_cell(cell)
        .with_masses(vec![1.0, 1.0])
        .with_atoms(vec![0, 1]);
    let a = array![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
    let b = array![2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert!(pd.distance(a.view(), b.view()).unwrap() < 1e-12);
    assert!(pd.distance(array![0.0, 0.0].view(), b.view()).is_err());
    let bad_atom = PermutationDistance::new(vec![0, 0]).with_atoms(vec![5]);
    assert!(bad_atom.distance(a.view(), b.view()).is_err());
    let bad_mass = PermutationDistance::new(vec![0, 0]).with_masses(vec![1.0, 0.0]);
    assert!(bad_mass.distance(a.view(), b.view()).is_err());

    let quench = QuenchConfig::default()
        .with_method(Method::Ode12r)
        .with_max_steps(200)
        .with_max_move(0.2)
        .with_initial_step(0.05);
    let basins = Basins::new(vec![array![-1.0, -1.0]], vec![array![1.0, 1.0]])
        .with_reactants(vec![array![-1.0, -1.0]])
        .with_products(vec![array![1.0, 1.0]])
        .with_tolerance(0.3)
        .with_quench(quench);
    let qreg = QuenchedTransitionRegion::new(Arc::new(SkewedWell), basins.clone());
    assert_eq!(qreg.basins().reactants.len(), 1);
    let viol = qreg.violation(array![-0.95, -0.95].view()).unwrap();
    assert!(viol.is_some());

    let plane = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let sample_cfg = LangevinConfig::default()
        .with_thermal(0.1)
        .with_time_step(0.05)
        .with_burn_in(100)
        .with_records(200)
        .with_blocks(4)
        .with_retain(32)
        .with_seed(5);
    let sample = rgsaddle::sample_plane(&SkewedWell, &plane, None, &sample_cfg).unwrap();

    let tcfg = TransmissionConfig::default()
        .with_thermal(0.1)
        .with_friction(0.5)
        .with_time_step(0.05)
        .with_steps(80)
        .with_check_every(10)
        .with_commit(2)
        .with_shots(32)
        .with_blocks(4)
        .with_seed(9)
        .with_masses(vec![1.0, 1.0])
        .with_counting(Counting::Final)
        .with_end_state(EndState::from(basins))
        .with_hold_excluded(false);
    let mut dyn_engine = AnalyticSurfaceDynamics::new(&SkewedWell);
    let tc = transmission_coefficient(&mut dyn_engine, &plane, &sample.retained, &tcfg).unwrap();
    assert!(tc.kappa > 0.0 && tc.kappa <= 1.0);
    assert!(tc.error >= 0.0);

    let bc = plane_transmission(
        &SkewedWell,
        &sample,
        &rgsaddle::ShootingConfig::default()
            .with_thermal(0.1)
            .with_shots(16),
    )
    .unwrap();
    assert!(bc.kappa > 0.0 && bc.kappa <= 1.0);
}

#[test]
fn rate_multiplicity_work_bar_mbar_optimizer_and_plane_wrappers() {
    let m_eff = flux_mass(array![1.0, 0.0].view(), array![2.0, 4.0].view()).unwrap();
    assert!((m_eff - 2.0).abs() < 1e-12);
    assert!(flux_mass(array![1.0, 0.0].view(), array![2.0].view()).is_err());

    let mult = RateMultiplicity::default()
        .with_paths(3)
        .with_in_plane_images(4);
    assert_eq!(mult.total().unwrap(), 12);
    assert!(RateMultiplicity::default().with_paths(0).total().is_err());
    assert!(
        RateMultiplicity::default()
            .with_paths(usize::MAX)
            .with_in_plane_images(2)
            .total()
            .is_err()
    );

    let p_react = Hyperplane::new(array![-0.8, -0.8], array![1.0, 0.0]).unwrap();
    // Unstable mode at the SkewedWell origin saddle is (0.8506508, 0.5257311) with curvature 2 - sqrt(20).
    let p_sad = Hyperplane::new(array![0.0, 0.0], array![0.8506508, 0.5257311]).unwrap();
    let sampler_cfg = LangevinConfig::default()
        .with_thermal(0.15)
        .with_time_step(0.05)
        .with_burn_in(100)
        .with_records(200)
        .with_blocks(4)
        .with_seed(11);
    let g_transport = GaussianTransport::default()
        .with_shrinkage(0.1)
        .with_shrink_to_rigid(true)
        .with_rank(1);
    let work_cfg = ReversibleWorkConfig::default()
        .with_sampler(sampler_cfg.clone())
        .with_warm_burn_in(40)
        .with_nodes(5)
        .with_simpson(true)
        .with_bar_samples(32)
        .with_transport(Transport::Gaussian(g_transport))
        .with_mbar(true)
        .with_estimate(WorkEstimate::Mbar)
        .with_min_overlap(0.02);
    let path = PlanePath::legs(&[p_react.clone(), p_sad.clone()]).unwrap();
    let mut sampler = AnalyticSurfaceSampler::new(&SkewedWell, sampler_cfg.clone());
    let work = reversible_work_along(&mut sampler, &path, &work_cfg).unwrap();
    let bar_w = work.bar_work.expect("BAR work");
    let mbar_w = work.mbar_work.expect("MBAR work");
    assert!((work.work - mbar_w).abs() < 1e-12);
    assert!((work.error - work.mbar_error.expect("MBAR error")).abs() < 1e-12);
    assert!(mbar_w > 0.2 && bar_w > 0.2 && work.integrated_work > 0.2);
    assert!((mbar_w - bar_w).abs() < 0.15, "mbar={mbar_w} bar={bar_w}");
    assert!(
        (mbar_w - work.integrated_work).abs() < 0.2,
        "mbar={mbar_w} ti={}",
        work.integrated_work
    );
    assert_eq!(work.overlap.as_ref().expect("MBAR overlap").dim(), (5, 5));

    let wall = Hyperplane::new(array![0.0, 0.0], array![1.0, 0.0]).unwrap();
    let rdens = reactant_density(
        &SkewedWell,
        &p_react,
        array![-1.0, -1.0].view(),
        &[wall],
        &sampler_cfg,
    )
    .unwrap();
    let rate = ohtst_rate(&rdens, &work).unwrap();
    let scaled: OhTstRate = rate.multiplied(&mult).unwrap();
    assert!((scaled.rate - 12.0 * rate.rate).abs() < 1e-12 * scaled.rate.max(1.0));
    let mut wrong_temp = rdens.clone();
    wrong_temp.thermal = 0.5;
    assert!(ohtst_rate(&wrong_temp, &work).is_err());
    let mut wrong_plane = rdens.clone();
    wrong_plane.plane = p_sad.clone();
    assert!(ohtst_rate(&wrong_plane, &work).is_err());

    let opt_cfg = PlaneOptimizerConfig::default()
        .with_sampler(sampler_cfg)
        .with_max_records(200)
        .with_norm_test(0.5)
        .with_max_translation(0.1)
        .with_max_rotation(0.35)
        .with_metric(PlaneMetric::Harmonic(HarmonicReference::new(array![
            0.0, 0.0
        ])))
        .with_metric_rank(2)
        .with_rotation_basis(Array2::eye(2))
        .with_translate(false)
        .with_rotate(true)
        .with_recenter(false)
        .with_confidence(2.0)
        .with_tolerance(1e-3)
        .with_patience(2)
        .with_max_iterations(8);
    let p_tilted = Hyperplane::new(array![0.0, 0.0], array![0.98, 0.199]).unwrap();
    let opt = PlaneOptimizer::new(&p_tilted, opt_cfg.clone()).unwrap();
    let optimum = opt.run(&SkewedWell).unwrap();
    assert!((optimum.plane.normal().dot(&optimum.plane.normal()) - 1.0).abs() < 1e-12);
    let point_shift = optimum.plane.point().to_owned() - p_tilted.point();
    assert!(
        point_shift.dot(&point_shift) < 1e-12,
        "translate=false must keep plane point fixed, got {:?}",
        optimum.plane.point()
    );
    let a_seed = ohtst_surfaces::line_free_energy(
        SkewedWell::energy,
        [0.0, 0.0],
        [p_tilted.normal()[0], p_tilted.normal()[1]],
        0.15,
        1.0,
        3.0,
    );
    let a_opt = ohtst_surfaces::line_free_energy(
        SkewedWell::energy,
        [0.0, 0.0],
        [optimum.plane.normal()[0], optimum.plane.normal()[1]],
        0.15,
        1.0,
        3.0,
    );
    assert!(
        a_opt > a_seed,
        "rotate=true should raise plane free energy ({a_opt} vs {a_seed})"
    );
    let opt_fail = PlaneOptimizer::new(&p_sad, opt_cfg).unwrap();
    assert!(opt_fail.run(&FailingPlaneSurface).is_err());

    let mw = MassWeighted::per_atom(&TwoAtomHarmonic, &[2.0, 3.0]).unwrap();
    assert_eq!(mw.masses().len(), 6);
    assert!(mw.to_mass_weighted(array![1.0, 2.0].view()).is_err());
    assert!(mw.to_cartesian(array![1.0, 2.0].view()).is_err());
    let cell = [5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0];
    let fc = FixedCell::new(&LepsHo, cell).unwrap();
    assert_eq!(
        fc.excluded_modes(array![1.0, 0.0].view()).unwrap().nrows(),
        0
    );
    assert_eq!(
        mass_weighted_rigid_translations(&[1.0, 2.0], 2)
            .unwrap()
            .dim(),
        (3, 6)
    );
}
