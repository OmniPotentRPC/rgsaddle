//! Observable-behavior tests for NEB variants, climbing image,
//! AutoNEB, OCI-NEB, solid-state NEB, and COMPASS connect on analytic
//! toy surfaces.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, array};
use rgmin::Method;
use rgsaddle::{
    AutoNeb, AutoNebInterpolant, AutoNebTuning, BandConfig, BandStatus, BandSurface, Cell,
    CellMask, CiConfig, ConnectConfig, IdppConfig, OcinebConfig, OcinebSession, PointSurface,
    SaddleError, SolidNeb, SpringKind, connect, joint_coordinate, linear_path,
};
#[cfg(feature = "faer")]
use rgsaddle::{
    BandSession, ExpBandPrecon, ForceGate, MinModeKind, MinModeRefine, OcinebPhase, OcinebStatus,
    ProjectionKind, Reparametrization, RtrConfig, SaddleIndexCheck, SolidState, TangentKind,
};

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown_eval(x: f64, y: f64, z: f64) -> (f64, [f64; 3]) {
    let mut e = 2.0 * z * z;
    let (mut gx, mut gy) = (0.0, 0.0);
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let t = MB_A[i] * (MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy).exp();
        e += t;
        gx += t * (2.0 * MB_AA[i] * dx + MB_B[i] * dy);
        gy += t * (MB_B[i] * dx + 2.0 * MB_C[i] * dy);
    }
    (e, [gx, gy, 4.0 * z])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (e, g) = muller_brown_eval(x[0], x[1], x[2]);
        Ok((e, array![g[0], g[1], g[2]]))
    }
}

impl BandSurface for MullerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = muller_brown_eval(row[0], row[1], row[2]);
            energies[i] = e;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = g[2];
        }
        Ok(())
    }
}

struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (a, b, c) = (x[0], x[1], x[2]);
        Ok((
            (a * a - 1.0).powi(2) + 2.0 * b * b + 2.0 * c * c,
            array![4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c],
        ))
    }
}

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (mut e, g) = PointSurface::eval(self, row.slice(ndarray::s![..3]))?;
            gradients[(i, 0)] = g[0];
            gradients[(i, 1)] = g[1];
            gradients[(i, 2)] = g[2];
            for k in 3..row.len() {
                let target = if k == 3 { 3.0 } else { 0.0 };
                let d = row[k] - target;
                e += 2.0 * d * d;
                gradients[(i, k)] = 4.0 * d;
            }
            energies[i] = e;
        }
        Ok(())
    }

    fn eval_cells(
        &self,
        positions: ArrayView2<f64>,
        cells: &[Cell],
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
        stresses: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        assert_eq!(cells.len(), positions.nrows());
        BandSurface::eval(self, positions, energies, gradients)?;
        stresses.fill(0.0);
        Ok(())
    }
}

fn two_barrier_toy(x: &[f64]) -> Result<(f64, Vec<f64>), SaddleError> {
    let (a, b, c) = (x[0], x[1], x[2]);
    let u = a * a * a - 4.0 * a;
    let energy = u * u + 2.0 * b * b + 2.0 * c * c;
    let gradient = vec![2.0 * u * (3.0 * a * a - 4.0), 4.0 * b, 4.0 * c];
    Ok((energy, gradient))
}

#[cfg(feature = "faer")]
#[test]
fn connect_falls_back_through_compass_and_sqnm_when_idpp_hits_well() {
    // With band_images = 1, the sole interior IDPP image lands at the
    // intermediate local minimum x = 0 of two_barrier_toy, so the
    // initial Lanczos climb cannot find a negative mode and connect
    // falls back to COMPASS + SQNM + climb_between.
    let cfg = ConnectConfig::default()
        .with_band_images(1)
        .with_nsteps(6_000)
        .with_minimize_nsteps(1_000)
        .with_n_loops(16)
        .with_quench_nodes(true)
        .with_forcetol(1e-3);
    let left = [-2.0, 0.0, 0.0];
    let right = [2.0, 0.0, 0.0];
    let report = connect(two_barrier_toy, &left, &right, &cfg).unwrap();
    assert_eq!(report.kind, 's');
    assert_eq!(report.hessian_index, 1);
    assert!(report.force_norm <= 1e-3, "|g|={}", report.force_norm);
    assert!((report.energy - 256.0 / 27.0).abs() < 1e-5);
    assert!((report.position[0].abs() - (4.0_f64 / 3.0).sqrt()).abs() < 1e-3);
    assert!(
        report.profile.searches > 0 && report.profile.sqnm_segments > 0,
        "expected COMPASS fallback search, got {:?}",
        report.profile
    );
    assert!(
        report.profile.quenches > 0,
        "expected endpoint/split quenches"
    );
    assert!(report.profile.hessian_checks > 0, "expected Hessian checks");

    // Non-quenched split nodes also converge on the same barrier.
    let no_quench = cfg.clone().with_quench_nodes(false);
    let r2 = connect(two_barrier_toy, &left, &right, &no_quench).unwrap();
    assert_eq!(r2.kind, 's');
    assert_eq!(r2.hessian_index, 1);

    // Invalid connect inputs fail with specific variants before any surface callback.
    let calls = std::cell::Cell::new(0usize);
    let counting = |x: &[f64]| -> Result<(f64, Vec<f64>), SaddleError> {
        calls.set(calls.get() + 1);
        Ok((x.iter().map(|v| 0.5 * v * v).sum(), x.to_vec()))
    };
    assert!(matches!(
        connect(counting, &left, &[2.0, 0.0], &cfg),
        Err(SaddleError::Shape(_))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        connect(counting, &[], &[], &cfg),
        Err(SaddleError::Shape(_))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        connect(counting, &[-2.0, 0.0], &[2.0, 0.0], &cfg),
        Err(SaddleError::Shape(_))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        connect(
            counting,
            &[-2.0, 0.0, 0.0, 1.0],
            &[2.0, 0.0, 0.0, 1.0],
            &cfg
        ),
        Err(SaddleError::Shape(_))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        connect(counting, &[f64::NAN, 0.0, 0.0], &right, &cfg),
        Err(SaddleError::NonFinite(_))
    ));
    assert_eq!(calls.get(), 0);
    assert!(matches!(
        connect(counting, &left, &[2.0, f64::INFINITY, 0.0], &cfg),
        Err(SaddleError::NonFinite(_))
    ));
    assert_eq!(calls.get(), 0);
    for bad_tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            connect(counting, &left, &right, &cfg.clone().with_forcetol(bad_tol)),
            Err(SaddleError::Invalid(_))
        ));
        assert_eq!(calls.get(), 0);
    }
}

#[cfg(feature = "faer")]
#[test]
fn connect_zero_steps_returns_ok() {
    let left = [-2.0, 0.0, 0.0];
    let right = [2.0, 0.0, 0.0];
    let zero_optbench = ConnectConfig::optbench(0);
    let report =
        connect(two_barrier_toy, &left, &right, &zero_optbench).expect("optbench(0) returns Ok");
    assert_eq!(report.position.len(), left.len());
    assert!(report.position.iter().all(|v| v.is_finite()));
    assert!(report.energy.is_finite());
    assert!(report.force_norm.is_finite());
    assert!(matches!(report.kind, 's' | 'u' | 'f'));
    assert!(report.evaluations > 0);

    let zero_nsteps = ConnectConfig::default()
        .with_band_images(1)
        .with_nsteps(0)
        .with_minimize_nsteps(0)
        .with_n_loops(2);
    let capped =
        connect(two_barrier_toy, &left, &right, &zero_nsteps).expect("nsteps == 0 returns Ok");
    assert_eq!(capped.position.len(), left.len());
    assert!(capped.position.iter().all(|v| v.is_finite()));
    assert!(capped.energy.is_finite());
    assert!(capped.force_norm.is_finite());
    assert!(matches!(capped.kind, 's' | 'u' | 'f'));
}

#[test]
fn autoneb_idpp_full_band_and_threads_converge_on_muller_brown() {
    let band = BandConfig::default()
        .with_force_tol(0.05)
        .with_max_move(0.05)
        .with_method(Method::fire(rgmin::FireKind::V2))
        .with_spring(SpringKind::Uniform { k: 5.0 })
        .with_climbing(CiConfig::always())
        .with_autoneb(
            AutoNebTuning::default()
                .with_n_simul(2)
                .with_n_max(7)
                .with_local_force_tol(0.5)
                .with_local_max_steps(30)
                .with_climb_max_steps(100)
                .with_space_energy_ratio(0.5)
                .with_climb(true)
                .with_full_band(true)
                .with_threads(2)
                .with_record_samples(true)
                .with_interpolant(AutoNebInterpolant::Idpp(IdppConfig::default())),
        );
    let initial = array![[-0.050011, 0.466694, 0.0], [0.623499, 0.028038, 0.0]];
    let mut autoneb = AutoNeb::new(band.clone(), initial.clone()).unwrap();
    let report = autoneb.run(&MullerBrown).unwrap();
    assert_eq!(report.status, BandStatus::Converged);
    assert_eq!(report.n_images, 7);
    let ci = autoneb.climbing_image().expect("climbing image");
    let saddle = autoneb.positions().row(ci).to_owned();
    assert!((saddle[0] - 0.2125).abs() < 0.05, "x={}", saddle[0]);
    assert!((saddle[1] - 0.2930).abs() < 0.05, "y={}", saddle[1]);
    assert!(!autoneb.cuts().is_empty());
    assert!(!autoneb.samples().is_empty());
    let full_stage = autoneb
        .stages()
        .iter()
        .find(|s| s.stage == rgsaddle::autoneb::AutoNebStage::FullBand)
        .expect("FullBand stage");
    assert!(full_stage.calls >= 5);
    assert!(
        report.max_force <= 0.05,
        "FullBand converged under band.force_tol, got {}",
        report.max_force
    );

    let window_only = band.clone().with_autoneb(
        band.autoneb
            .clone()
            .unwrap()
            .with_full_band(false)
            .with_record_samples(false),
    );
    let mut an_window = AutoNeb::new(window_only, initial.clone()).unwrap();
    an_window.run(&MullerBrown).unwrap();
    assert!(
        !an_window
            .stages()
            .iter()
            .any(|s| s.stage == rgsaddle::autoneb::AutoNebStage::FullBand)
    );
    let interior_shift: f64 = (1..6)
        .map(|i| {
            let d = &autoneb.positions().row(i) - &an_window.positions().row(i);
            d.dot(&d).sqrt()
        })
        .sum();
    assert!(
        interior_shift > 1e-4,
        "FullBand relaxation must move non-window interior images, shift={interior_shift}"
    );
    // Second run is refused.
    assert!(autoneb.run(&MullerBrown).is_err());

    // Climb disabled still completes and relaxes the full band when requested.
    let no_climb = band.clone().with_autoneb(
        band.autoneb
            .unwrap()
            .with_climb(false)
            .with_full_band(true)
            .with_threads(1)
            .with_record_samples(false),
    );
    let mut an2 = AutoNeb::new(no_climb, initial.clone()).unwrap();
    let r2 = an2.run(&MullerBrown).unwrap();
    assert_eq!(r2.n_images, 7);
    assert!(an2.climbing_image().is_none());

    // Error cases: weighted spring refused, wrong shapes.
    let bad_spring = BandConfig::default()
        .with_spring(SpringKind::EnergyWeighted {
            k_min: 1.0,
            k_max: 5.0,
        })
        .with_autoneb(AutoNebTuning::default());
    assert!(AutoNeb::new(bad_spring, initial).is_err());
    assert!(rgsaddle::autoneb::worst_segment(&[], &[0.0], 1.0, 0.5).is_err());
    assert!(rgsaddle::autoneb::worst_segment(&[1.0], &[0.0, f64::NAN], 1.0, 0.5).is_err());
    assert!(rgsaddle::autoneb::worst_segment(&[1.0], &[0.0, 1.0], 1.0, -0.1).is_err());
}

#[cfg(feature = "faer")]
#[test]
fn neb_variants_and_ocineb_controls_cover_all_projections_and_springs() {
    let b = array![-0.050011, 0.466694, 0.0];
    let c = array![0.623499, 0.028038, 0.0];
    let path = linear_path(b.view(), c.view(), 7, None).unwrap();

    // OnsagerMachlupLive + DoublyNudged + Simple tangent + force gate switch + restart.
    let cfg_om = BandConfig::default()
        .with_tangent(TangentKind::Simple)
        .with_spring(SpringKind::OnsagerMachlupLive { k: 4.0 })
        .with_projection(ProjectionKind::DoublyNudged)
        .with_method(Method::lbfgs_with_memory(10))
        .with_climbing(CiConfig::always())
        .with_min_mode(None)
        .with_force_tol(1e-2)
        .with_max_move(0.05);
    let mut sess = BandSession::new(cfg_om, path.clone()).unwrap();
    assert!(sess.exp_precon().is_none());
    sess.set_force_gate(ForceGate::L2Norm);
    let r1 = sess.step(&MullerBrown).unwrap();
    assert!(r1.max_force.is_finite());
    sess.restart();
    assert!(sess.evaluation().is_some());
    sess.reset();
    assert!(sess.evaluation().is_none());

    // PlainElasticBand + EnergyWeighted + RTR NotAKnot.
    let cfg_peb = BandConfig::default()
        .with_spring(SpringKind::EnergyWeighted {
            k_min: 1.0,
            k_max: 6.0,
        })
        .with_projection(ProjectionKind::PlainElasticBand)
        .with_method(Method::lbfgs_with_memory(10))
        .with_climbing(None)
        .with_force_tol(1e-2)
        .with_max_move(0.05);
    let mut sess_peb = BandSession::new(cfg_peb, path.clone()).unwrap();
    sess_peb
        .set_rtr(Some(
            RtrConfig::default().with_reparametrization(Reparametrization::NotAKnot),
        ))
        .unwrap();
    let r_peb = sess_peb.step(&MullerBrown).unwrap();
    assert!(r_peb.max_force.is_finite());

    // SolidState::with_cells supplies per-image cells that evolve under pressure while
    // rejecting incompatible ExpBandPrecon; ExpBandPrecon estimates positive mu on first step.
    let cell = Cell([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]);
    assert!(matches!(
        BandSession::new(
            BandConfig::default().with_solid(Some(SolidState::new(Vec::new()))),
            path.clone(),
        ),
        Err(SaddleError::Shape(_))
    ));
    let solid = SolidState::new(Vec::new())
        .with_cells(vec![cell; 7])
        .with_pressure(0.05);
    assert!(matches!(
        BandSession::new(
            BandConfig::default()
                .with_solid(Some(solid.clone()))
                .with_exp_precon(Some(ExpBandPrecon::default())),
            path.clone(),
        ),
        Err(SaddleError::Invalid(_))
    ));
    let mut solid_sess = BandSession::new(
        BandConfig::default()
            .with_solid(Some(solid))
            .with_climbing(None)
            .with_max_move(0.05),
        path.clone(),
    )
    .unwrap();
    solid_sess.step(&DoubleWell).unwrap();
    let stepped_cells = solid_sess.image_cells().expect("solid cells");
    assert!(stepped_cells[3].0[0][0] < 20.0);
    assert!((stepped_cells[0].0[0][0] - 20.0).abs() < 1e-12);
    assert!((stepped_cells[6].0[0][0] - 20.0).abs() < 1e-12);

    let two_atom_path = linear_path(
        array![-1.0, 0.2, 0.0, 3.0, 0.0, 0.0].view(),
        array![1.0, 0.2, 0.0, 3.0, 0.0, 0.0].view(),
        5,
        None,
    )
    .unwrap();
    let mut exp_sess = BandSession::new(
        BandConfig::default()
            .with_method(Method::lbfgs_with_memory(10))
            .with_remove_translation(false)
            .with_climbing(CiConfig::always())
            .with_min_mode(None)
            .with_exp_precon(Some(ExpBandPrecon::default()))
            .with_force_tol(1e-3)
            .with_max_move(0.1),
        two_atom_path,
    )
    .unwrap();
    assert!(exp_sess.exp_scale().is_none());
    exp_sess.step(&DoubleWell).unwrap();
    let mu = exp_sess.exp_scale().expect("estimated mu after first step");
    assert!(mu.is_finite() && mu > 0.0);
    let exp_rep = exp_sess.run(&DoubleWell, 400).unwrap();
    assert_eq!(exp_rep.status, BandStatus::Converged);
    let exp_x = exp_sess.positions()[(2, 0)];
    assert!(exp_x.abs() < 1e-3, "saddle x={exp_x}");

    // OcinebSession accessors, Lanczos, refresh, climb_alignment fallback, and saddle check.
    let oci_cfg = OcinebConfig::default()
        .with_minmode(
            rgsaddle::MinModeConfig::default()
                .with_kind(MinModeKind::Lanczos)
                .with_krylov_dim(5)
                .with_force_tol(1e-2)
                .with_max_move(0.05),
        )
        .with_band(
            BandConfig::default()
                .with_climbing(CiConfig::after_rel(0.5))
                .with_force_tol(1e-2)
                .with_max_move(0.05),
        )
        .with_stability_count(2)
        .with_max_mmf_steps(3)
        .with_restore_unhelpful(true);
    let mut oci = OcinebSession::new(oci_cfg, path).unwrap();
    oci.set_force_gate(ForceGate::MaxForceOnAtom);
    oci.set_highs(false);
    oci.set_saddle_check(Some(SaddleIndexCheck::default()))
        .unwrap();
    assert!(oci.evaluation().is_none());
    let rep = oci.run(&MullerBrown, 800).unwrap();
    assert_eq!(rep.status, OcinebStatus::Converged);
    assert!(oci.evaluation().is_some());
    assert!(oci.saddle_index().is_some_and(|s| s.index == 1));
    assert_eq!(oci.saddle_escapes(), 0);
    assert!(oci.saddle_evaluations() > 0);
    oci.reset();
    assert!(oci.evaluation().is_none());
    assert_eq!(oci.phase(), OcinebPhase::Band);

    // MinModeRefine refresh + budget exhaustion on DoubleWell.
    let start_dw = array![-1.0, 0.0, 0.0];
    let end_dw = array![1.0, 0.0, 0.0];
    let mut dw_path = linear_path(start_dw.view(), end_dw.view(), 7, None).unwrap();
    for i in 1..6 {
        dw_path[(i, 1)] = 0.25 * (std::f64::consts::PI * (i as f64) / 6.0).sin();
    }
    let refine = MinModeRefine::default()
        .with_kind(MinModeKind::Dimer)
        .with_refresh_interval(1)
        .with_curvature_retained(0.99)
        .with_min_overlap(0.999)
        .with_max_steps(2)
        .with_climb_alignment(true)
        .with_restore_unhelpful(true);
    let dw_cfg = BandConfig::default()
        .with_climbing(CiConfig::always())
        .with_min_mode(Some(refine))
        .with_force_tol(1e-4)
        .with_max_move(0.05);
    let mut dw_sess = BandSession::new(dw_cfg, dw_path).unwrap();
    let dw_rep = dw_sess.run(&DoubleWell, 800).unwrap();
    assert_eq!(dw_rep.status, BandStatus::Converged);
}

#[test]
fn solid_neb_accessors_and_joint_coordinate_errors() {
    let initial = linear_path(
        array![-1.0, 0.0, 0.0].view(),
        array![1.0, 0.0, 0.0].view(),
        5,
        None,
    )
    .unwrap();
    let cfg = BandConfig::default()
        .with_climbing(CiConfig::always())
        .with_min_mode(None)
        .with_force_tol(1e-3)
        .with_max_move(0.1);
    let mut frozen = SolidNeb::frozen(cfg.clone(), initial).unwrap();
    assert!(frozen.is_frozen());
    assert!(frozen.jacobian().is_none());
    assert!(frozen.chart().is_none());
    assert!(frozen.cells().unwrap().is_none());
    assert!(frozen.images().is_none());
    assert_eq!(frozen.generalized().dim(), (5, 3));
    frozen.step(&DoubleWell).unwrap();
    let imgs = frozen.images().expect("images after step");
    assert_eq!(imgs.len(), 5);
    assert!(imgs.iter().all(|im| im.stress.is_none()));

    // Joint-chart error edges.
    let eye = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
    let diag_mask = CellMask::voigt([true, true, true, false, false, false]);
    let chart = rgsaddle::JointChart::new(eye, 1, diag_mask, [[0.0; 3]; 3], 1.0).unwrap();
    // Wrong position length.
    assert!(joint_coordinate(&chart, diag_mask, array![0.0, 0.0].view(), eye).is_err());
    // Upper-triangular strain is rejected.
    let upper = [[5.0, 0.5, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]];
    assert!(joint_coordinate(&chart, diag_mask, array![0.0, 0.0, 0.0].view(), upper).is_err());
    // Strain component locked by diagonal mask is rejected.
    let sheared = [[5.0, 0.0, 0.0], [0.5, 5.0, 0.0], [0.0, 0.0, 5.0]];
    assert!(joint_coordinate(&chart, diag_mask, array![0.0, 0.0, 0.0].view(), sheared).is_err());
    // Joint SolidNeb refuses step_surface.
    let mut pos = Array2::zeros((5, 3));
    for i in 0..5 {
        pos[(i, 0)] = -1.0 + 0.5 * i as f64;
    }
    let mut joint =
        SolidNeb::joint(cfg, eye, pos, &[eye; 5], diag_mask, [[0.0; 3]; 3], 1.0).unwrap();
    assert!(!joint.is_frozen());
    assert!(joint.jacobian().is_some());
    assert!(joint.chart().is_some());
    assert_eq!(joint.cells().unwrap().unwrap().len(), 5);
    assert_eq!(joint.cartesian().unwrap().dim(), (5, 3));
    assert_eq!(joint.generalized().dim(), (5, 6));
    assert!(joint.step_surface(&DoubleWell).is_err());
}

#[cfg(not(feature = "faer"))]
#[test]
fn connect_and_ocineb_without_faer_return_solver_error() {
    let left = [-2.0, 0.0, 0.0];
    let right = [2.0, 0.0, 0.0];
    let cfg = ConnectConfig::default()
        .with_band_images(1)
        .with_nsteps(10)
        .with_minimize_nsteps(10)
        .with_n_loops(2);
    match connect(two_barrier_toy, &left, &right, &cfg) {
        Err(SaddleError::Solver(msg)) => assert!(msg.contains("faer"), "{msg}"),
        other => panic!("expected SaddleError::Solver, got {other:?}"),
    }

    let b = array![-0.050011, 0.466694, 0.0];
    let c = array![0.623499, 0.028038, 0.0];
    let path = linear_path(b.view(), c.view(), 7, None).unwrap();
    let oci_cfg = OcinebConfig::default()
        .with_band(
            BandConfig::default()
                .with_climbing(CiConfig::after_rel(0.5))
                .with_force_tol(1e-2)
                .with_max_move(0.05),
        )
        .with_stability_count(2);
    let mut oci = OcinebSession::new(oci_cfg, path).unwrap();
    match oci.run(&MullerBrown, 200) {
        Err(SaddleError::Solver(msg)) => assert!(msg.contains("faer"), "{msg}"),
        other => panic!("expected SaddleError::Solver, got {other:?}"),
    }
}
