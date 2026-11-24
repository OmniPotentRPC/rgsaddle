//! Fixed atoms on a band (`BandConfig::with_fixed_atoms`): a Morse
//! fcc(100) slab with its bottom layer frozen and an adatom hopping
//! between hollow sites. The frozen layer never moves under any stepper.
//! The band and its start map onto themselves through the mirror plane
//! of the adatom's row, so the climbing image stops on that mirror, at a
//! stationary point with three negative curvatures; the saddle check
//! reports index 3 there for every stepper, and its escape takes the
//! image off the mirror to the first-order saddle. An all-false mask is
//! the band without a mask.

use rayon::prelude::*;
use rgsaddle::{
    BandConfig, BandRtr, BandSession, BandStatus, BandSurface, ExpBandPrecon, MinModeConfig,
    MinModeKind, MinModeSession, MinModeStatus, PointSurface, RtrConfig, SaddleError, SaddleEscape,
    SaddleIndex, SaddleIndexCheck, band_forces,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/morse_slab.rs"
));

const N_IMAGES: usize = 7;

/// The stationary point on the mirror; its Hessian on the free
/// coordinates has three negative eigenvalues.
const MIRROR_POINT: f64 = -50.6503755;

/// The first-order saddle off the mirror, curvature -0.913 along the hop.
const SADDLE: f64 = -50.6761677;

fn config() -> BandConfig {
    BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(slab_fixed())
}

fn exp() -> ExpBandPrecon {
    ExpBandPrecon::default()
}

fn escape() -> Option<SaddleIndexCheck> {
    Some(SaddleIndexCheck::default().with_escape(Some(SaddleEscape::default())))
}

fn frozen_bits(band: ArrayView2<f64>) -> Vec<u64> {
    let fixed = slab_fixed();
    band.outer_iter()
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .filter(|(k, _)| fixed[k / 3])
                .map(|(_, v)| v.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

fn free_max(gradient: ndarray::ArrayView1<f64>) -> f64 {
    let fixed = slab_fixed();
    gradient
        .iter()
        .enumerate()
        .filter(|(k, _)| !fixed[k / 3])
        .fold(0.0_f64, |m, (_, v)| m.max(v.abs()))
}

/// Largest distance from an atom of `x`, mirrored through the plane
/// `y = r0` of the adatom's row, to the nearest atom of `x` under the
/// cell. The slab and the straight band map onto themselves, so this is
/// zero on every image of the starting band.
fn mirror_error(x: ArrayView1<f64>) -> f64 {
    let side = slab_box();
    let atoms = x.len() / 3;
    (0..atoms)
        .map(|a| {
            let mirrored = [x[3 * a], 2.0 * SLAB_R0 - x[3 * a + 1], x[3 * a + 2]];
            (0..atoms)
                .map(|b| {
                    (0..3)
                        .map(|c| {
                            let d = mirrored[c] - x[3 * b + c];
                            let d = if c < 2 {
                                d - side * (d / side).round()
                            } else {
                                d
                            };
                            d * d
                        })
                        .sum::<f64>()
                        .sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0, f64::max)
}

/// A converged run: the climbing image, its energy and free-atom
/// gradient, the surface rows and the saddle certificate.
struct Converged {
    energy: f64,
    residual: f64,
    rows: usize,
    image: Array1<f64>,
    index: Option<SaddleIndex>,
    escapes: usize,
    check_rows: usize,
}

fn converge(config: BandConfig, rtr: bool, band: &Array2<f64>) -> Converged {
    let surface = SlabSurface::new();
    let mut session = BandSession::new(config, band.clone()).expect("session");
    if rtr {
        session
            .set_rtr(Some(RtrConfig::default().with_radius_max(0.2)))
            .expect("RTR");
    }
    let report = session.run(&surface, 3000).expect("run");
    assert_eq!(
        report.status,
        BandStatus::Converged,
        "max force {} after {} steps",
        report.max_force,
        report.iteration
    );
    assert_eq!(
        frozen_bits(session.positions()),
        frozen_bits(band.view()),
        "a frozen atom moved"
    );
    let ci = session.climbing_image().expect("climbing image");
    let image = session.positions().row(ci).to_owned();
    let (energy, gradient) = slab_energy(image.view());
    assert_eq!(session.saddle_index(), report.saddle_index.as_ref());
    Converged {
        energy,
        residual: free_max(gradient.view()),
        rows: surface.rows(),
        image,
        index: report.saddle_index,
        escapes: session.saddle_escapes(),
        check_rows: session.saddle_evaluations(),
    }
}

fn index_of(run: &Converged) -> usize {
    run.index.as_ref().expect("certified").index
}

fn assert_frozen_atoms_bitwise_unchanged(name: &str, method: rgmin::Method, rtr: bool) {
    let band = slab_band(N_IMAGES);
    let run = converge(config().with_method(method), rtr, &band);
    eprintln!(
        "{name}: energy {:.9}, free |g| {:.2e}, {} rows ({} for the index), index {}",
        run.energy,
        run.residual,
        run.rows,
        run.check_rows,
        index_of(&run)
    );
    assert!(
        run.residual <= 1e-3,
        "{name} climbing image is not stationary"
    );
    assert_eq!(index_of(&run), 3, "{name} stops on the mirror point");
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_fire() {
    assert_frozen_atoms_bitwise_unchanged("FIRE", rgmin::Method::fire(rgmin::FireKind::V2), false);
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_lbfgs() {
    assert_frozen_atoms_bitwise_unchanged("L-BFGS", rgmin::Method::lbfgs_with_memory(20), false);
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_quick_min() {
    assert_frozen_atoms_bitwise_unchanged("quick-min", rgmin::Method::QuickMin, false);
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_ode12r() {
    assert_frozen_atoms_bitwise_unchanged("ODE12r", rgmin::Method::Ode12r, false);
}

#[test]
fn frozen_atoms_are_bitwise_unchanged_by_rtr() {
    assert_frozen_atoms_bitwise_unchanged("RTR", rgmin::Method::fire(rgmin::FireKind::V2), true);
}

#[test]
fn fire_min_mode_refinement_keeps_the_band_mirror() {
    // The band's own climbing image converges on the mirror through the
    // adatom's row. FIRE hands it to the refinement far from convergence
    // and climbs for tens of steps, so a climb model that breaks the
    // mirror carries the image off it, to a lower stationary point. The
    // point on the mirror is not a first-order saddle: the two modes
    // that break the mirror are negative there too.
    let band = slab_band(N_IMAGES);
    let plain = converge(config().with_min_mode(None), false, &band);
    let refined = converge(config(), false, &band);
    let mirror = mirror_error(refined.image.view());
    let curvatures = &refined.index.as_ref().expect("certified").curvatures;
    eprintln!(
        "FIRE climbing image {:.10} with the refinement, {:.10} without; \
         mirror error {mirror:.2e}; curvatures {curvatures:?}",
        refined.energy, plain.energy
    );
    assert!(
        refined.residual <= 1e-3,
        "the climbing image is not stationary"
    );
    assert!(
        mirror < 1e-5,
        "the climbing image left the mirror by {mirror}"
    );
    // Each run stops within tol^2 / |lambda| of the same stationary
    // point, with tol 1e-3 and |lambda| at least 0.037 there.
    assert!(
        (refined.energy - plain.energy).abs() < 1e-4,
        "FIRE with the refinement {} against {} without",
        refined.energy,
        plain.energy
    );
    assert_eq!(index_of(&plain), 3);
    assert_eq!(index_of(&refined), 3);
    assert!((curvatures[0] + 1.379).abs() < 1e-2, "{curvatures:?}");
    assert!((curvatures[1] + 0.164).abs() < 1e-2, "{curvatures:?}");
    assert!((curvatures[2] + 0.059).abs() < 1e-2, "{curvatures:?}");
}

#[test]
fn exp_precon_with_a_frozen_layer_reaches_the_same_mirror_point() {
    let band = slab_band(N_IMAGES);
    let tol = 1e-4;
    let plain = config().with_force_tol(tol);
    let specs = [
        ("FIRE", plain.clone()),
        (
            "FIRE without the refinement",
            plain.clone().with_min_mode(None),
        ),
        (
            "L-BFGS",
            plain
                .clone()
                .with_method(rgmin::Method::lbfgs_with_memory(20)),
        ),
        (
            "ODE12r-Exp",
            plain
                .clone()
                .with_method(rgmin::Method::Ode12r)
                .with_exp_precon(Some(exp())),
        ),
        ("FIRE-Exp", plain.with_exp_precon(Some(exp()))),
    ];
    let runs: Vec<(&str, Converged)> = specs
        .into_par_iter()
        .map(|(name, cfg)| (name, converge(cfg, false, &band)))
        .collect();
    let fire = &runs[0].1;
    let climb_only = &runs[1].1;
    let lbfgs = &runs[2].1;
    let ode = &runs[3].1;
    let fire_exp = &runs[4].1;
    eprintln!(
        "mirror-point energies: FIRE {:.10}, FIRE without the refinement {:.10}, \
         L-BFGS {:.10}, ODE12r-Exp {:.10}, FIRE-Exp {:.10}",
        fire.energy, climb_only.energy, lbfgs.energy, ode.energy, fire_exp.energy
    );
    assert!(ode.residual <= tol && fire_exp.residual <= tol);
    assert!(
        (fire.energy - MIRROR_POINT).abs() < 1e-6,
        "FIRE {}",
        fire.energy
    );
    // A free gradient of at most tol puts the climbing image within
    // tol / |lambda| of the stationary point and its energy within
    // tol^2 / |lambda| of that energy; the softest mode here is well
    // above 1e-2.
    for (name, run) in [
        ("FIRE", fire),
        ("FIRE without the refinement", climb_only),
        ("ODE12r-Exp", ode),
        ("FIRE-Exp", fire_exp),
        ("L-BFGS", lbfgs),
    ] {
        assert!(
            (run.energy - fire.energy).abs() < 1e-6,
            "{name} stops at {} against FIRE {}",
            run.energy,
            fire.energy
        );
        assert_eq!(index_of(run), 3, "{name}");
    }
}

fn assert_escape_leaves_the_mirror(name: &str, config: BandConfig) {
    let band = slab_band(N_IMAGES);
    let tol = 1e-4;
    let run = converge(config.with_saddle_check(escape()), false, &band);
    let index = run.index.as_ref().expect("certified");
    let mirror = mirror_error(run.image.view());
    eprintln!(
        "{name} escape: energy {:.10}, free |g| {:.2e}, {} rows ({} for the index and \
         the push); mirror error {mirror:.3}; curvatures {:?}",
        run.energy, run.residual, run.rows, run.check_rows, index.curvatures
    );
    assert_eq!(index.index, 1, "{name}: {index:?}");
    assert_eq!(run.escapes, 1, "{name}");
    assert!(run.residual <= tol, "{name}");
    assert!((run.energy - SADDLE).abs() < 1e-6, "{name}: {}", run.energy);
    assert!(
        (index.curvatures[0] + 0.913).abs() < 1e-2,
        "{name}: {index:?}"
    );
    assert!(mirror > 0.5, "{name} stayed near the mirror: {mirror}");
}

#[test]
fn fire_escape_leaves_the_mirror_for_the_first_order_saddle() {
    assert_escape_leaves_the_mirror("FIRE", config().with_force_tol(1e-4));
}

#[test]
fn preconditioned_escape_leaves_the_mirror_for_the_first_order_saddle() {
    assert_escape_leaves_the_mirror(
        "ODE12r-Exp",
        config()
            .with_force_tol(1e-4)
            .with_method(rgmin::Method::Ode12r)
            .with_exp_precon(Some(exp())),
    );
}

/// The slab as a point surface for a min-mode search: the frozen layer's
/// gradient is zeroed and its coordinates are excluded from the mode.
struct FrozenPoint;

impl PointSurface for FrozenPoint {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (e, mut g) = slab_energy(x);
        let fixed = slab_fixed();
        for (k, v) in g.iter_mut().enumerate() {
            if fixed[k / 3] {
                *v = 0.0;
            }
        }
        Ok((e, g))
    }

    fn excluded_modes(&self, x: ArrayView1<f64>) -> Result<Array2<f64>, SaddleError> {
        let held: Vec<usize> = (0..x.len()).filter(|k| slab_fixed()[k / 3]).collect();
        let mut rows = Array2::zeros((held.len(), x.len()));
        for (r, &k) in held.iter().enumerate() {
            rows[(r, k)] = 1.0;
        }
        Ok(rows)
    }
}

#[test]
fn a_min_mode_search_from_the_mirror_point_escapes_to_the_first_order_saddle() {
    let band = slab_band(N_IMAGES);
    let surface = SlabSurface::new();
    let mut session =
        BandSession::new(config().with_saddle_check(None), band.clone()).expect("session");
    session.run(&surface, 3000).expect("run");
    let ci = session.climbing_image().expect("climbing image");
    let start = session.positions().row(ci).to_owned();
    let tangent = &session.positions().row(ci + 1) - &session.positions().row(ci - 1);
    let config = MinModeConfig::default()
        .with_kind(MinModeKind::Lanczos)
        .with_force_tol(1e-4)
        .with_saddle_check(escape());
    let mut search = MinModeSession::new(config, start.clone(), tangent).expect("session");
    let mut indices = Vec::new();
    let mut evaluations = 0;
    let mut steps = 0;
    loop {
        let report = search.step(&FrozenPoint).expect("step");
        evaluations += report.evaluations;
        steps += 1;
        if let Some(index) = report.saddle_index {
            indices.push(index.index);
        }
        if report.status == MinModeStatus::Converged || steps == 5000 {
            break;
        }
    }
    let x = search.position();
    let (energy, gradient) = slab_energy(x);
    eprintln!(
        "Lanczos escape: energy {energy:.10}, free |g| {:.2e}, {steps} steps, {evaluations} \
         evaluations ({} for the index and the push), indices {indices:?}",
        free_max(gradient.view()),
        search.saddle_evaluations()
    );
    assert_eq!(indices, [3, 1]);
    assert_eq!(search.saddle_escapes(), 1);
    assert!((energy - SADDLE).abs() < 1e-6, "{energy}");
    let fixed = slab_fixed();
    assert!(
        x.iter()
            .zip(start.iter())
            .enumerate()
            .all(|(k, (a, b))| !fixed[k / 3] || a.to_bits() == b.to_bits()),
        "a frozen atom moved"
    );
}

fn steps(config: BandConfig, band: &Array2<f64>, n: usize) -> (Array2<f64>, Vec<u64>) {
    let surface = SlabSurface::new();
    let mut session = BandSession::new(config, band.clone()).expect("session");
    let mut trace = Vec::new();
    for _ in 0..n {
        let report = session.step(&surface).expect("step");
        trace.push(report.max_force.to_bits());
        trace.push(report.ci_index.unwrap_or(usize::MAX) as u64);
        trace.push(report.curvature.unwrap_or(0.0).to_bits());
    }
    (session.positions().to_owned(), trace)
}

#[test]
fn an_all_false_mask_is_bitwise_the_band_without_a_mask() {
    let band = slab_band(5);
    let none = BandConfig::default().with_cell(Some(slab_cell()));
    let clear = none.clone().with_fixed_atoms(vec![false; slab_atoms()]);
    for precon in [None, Some(exp())] {
        for method in [
            rgmin::Method::fire(rgmin::FireKind::V2),
            rgmin::Method::lbfgs_with_memory(20),
        ] {
            let a = steps(
                none.clone()
                    .with_method(method.clone())
                    .with_exp_precon(precon),
                &band,
                60,
            );
            let b = steps(
                clear.clone().with_method(method).with_exp_precon(precon),
                &band,
                60,
            );
            assert_eq!(a.1, b.1, "reports differ");
            let bits = |m: &Array2<f64>| m.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a.0), bits(&b.0), "positions differ");
        }
    }
}

/// The slab with the frozen layer's gradient zeroed by the host, the
/// only option before the mask existed.
struct HostZeroed(SlabSurface);

impl BandSurface for HostZeroed {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.0.eval(positions, energies, gradients)?;
        let fixed = slab_fixed();
        for mut row in gradients.outer_iter_mut() {
            for (k, g) in row.iter_mut().enumerate() {
                if fixed[k / 3] {
                    *g = 0.0;
                }
            }
        }
        Ok(())
    }
}

#[test]
fn host_zeroed_forces_under_the_metric_move_frozen_atoms_and_the_mask_does_not() {
    let band = slab_band(5);
    let host = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_remove_translation(false)
        .with_exp_precon(Some(exp()));
    let surface = HostZeroed(SlabSurface::new());
    let mut session = BandSession::new(host.clone(), band.clone()).expect("session");
    for _ in 0..5 {
        session.step(&surface).expect("step");
    }
    assert_ne!(
        frozen_bits(session.positions()),
        frozen_bits(band.view()),
        "the full-system metric should couple the frozen layer to the free atoms"
    );
    let mut session =
        BandSession::new(host.with_fixed_atoms(slab_fixed()), band.clone()).expect("session");
    for _ in 0..5 {
        session.step(&surface).expect("step");
    }
    assert_eq!(frozen_bits(session.positions()), frozen_bits(band.view()));
}

#[test]
fn frozen_atoms_carry_no_force_in_the_projection_or_the_report() {
    let mut band = slab_band(5);
    for mut row in band.outer_iter_mut() {
        row[0] += 0.1;
        row[4] -= 0.1;
    }
    let surface = SlabSurface::new();
    let forces = band_forces(&config(), &surface, band.view(), None).expect("forces");
    let dof = band.ncols();
    let fixed = slab_fixed();
    let mut free = 0.0_f64;
    for (k, f) in forces.force.iter().enumerate() {
        if fixed[(k % dof) / 3] {
            assert_eq!(*f, 0.0);
        } else {
            free = free.max(f.abs());
        }
    }
    assert_eq!(forces.max_force, free);
    let mut session = BandSession::new(config(), band.clone()).expect("session");
    let report = session.step(&surface).expect("step");
    let evaluation = session.evaluation();
    let first = evaluation.as_ref().map(|e| e.gradients.clone());
    let gradients = first.expect("evaluation after a step");
    for row in gradients.outer_iter() {
        for (k, g) in row.iter().enumerate() {
            if fixed[k / 3] {
                assert_eq!(*g, 0.0, "a frozen gradient in the evaluation");
            }
        }
    }
    let frozen_gradient = slab_energy(band.row(2))
        .1
        .iter()
        .enumerate()
        .filter(|(k, _)| fixed[k / 3])
        .fold(0.0_f64, |m, (_, v)| m.max(v.abs()));
    assert!(
        frozen_gradient > 1e-2,
        "the test needs a frozen layer under load: {frozen_gradient}"
    );
    assert!(report.max_force.is_finite());
}

#[test]
fn rtr_driver_keeps_the_frozen_layer() {
    let mut band = slab_band(5);
    let start = band.clone();
    let config = config();
    let surface = SlabSurface::new();
    let mut rtr = BandRtr::new(RtrConfig::default().with_radius_max(0.2), true);
    for _ in 0..50 {
        rtr.step(&config, &surface, &mut band).expect("step");
    }
    assert_eq!(frozen_bits(band.view()), frozen_bits(start.view()));
    assert_ne!(band, start);
}

#[test]
fn endpoints_that_differ_on_a_frozen_atom_are_refused() {
    let mut band = slab_band(5);
    let last = band.nrows() - 1;
    band[(last, 2)] += 1e-6;
    let error = BandSession::new(config(), band.clone())
        .err()
        .expect("refused");
    assert!(
        matches!(&error, SaddleError::Invalid(m) if m.contains("fixed atom 0")),
        "{error}"
    );
    let mut fine = band.clone();
    fine[(last, 2)] -= 1e-6;
    fine[(last, 0)] += slab_box();
    BandSession::new(config(), fine).expect("a whole period apart is the same place");
    let mut session = BandSession::new(
        BandConfig::default().with_cell(Some(slab_cell())),
        band.clone(),
    )
    .expect("no mask");
    assert!(matches!(
        session.set_fixed_atoms(slab_fixed()),
        Err(SaddleError::Invalid(_))
    ));
    assert!(session.fixed_atoms().is_empty());
}

#[test]
fn malformed_masks_are_refused() {
    let band = slab_band(5);
    let short = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(vec![true; slab_atoms() - 1]);
    assert!(matches!(
        BandSession::new(short, band.clone()),
        Err(SaddleError::Shape(_))
    ));
    let all = BandConfig::default()
        .with_cell(Some(slab_cell()))
        .with_fixed_atoms(vec![true; slab_atoms()]);
    assert!(matches!(
        BandSession::new(all, band),
        Err(SaddleError::Invalid(_))
    ));
}

#[test]
fn setting_the_mask_mid_run_pins_the_interior_to_the_first_image() {
    let mut band = slab_band(5);
    let free = BandConfig::default().with_cell(Some(slab_cell()));
    for i in 1..band.nrows() - 1 {
        band[(i, 0)] += 0.01 * i as f64;
    }
    let surface = SlabSurface::new();
    let mut session = BandSession::new(free, band.clone()).expect("session");
    for _ in 0..3 {
        session.step(&surface).expect("step");
    }
    session.set_fixed_atoms(slab_fixed()).expect("mask");
    assert_eq!(session.fixed_atoms(), slab_fixed().as_slice());
    let first = band.row(0).to_owned();
    let pinned = |positions: ArrayView2<f64>| {
        let fixed = slab_fixed();
        positions.outer_iter().all(|row| {
            row.iter()
                .zip(first.iter())
                .enumerate()
                .all(|(k, (a, b))| !fixed[k / 3] || a.to_bits() == b.to_bits())
        })
    };
    assert!(pinned(session.positions()));
    for _ in 0..30 {
        session.step(&surface).expect("step");
    }
    assert!(pinned(session.positions()));
    session.set_fixed_atoms(Vec::new()).expect("clear");
    assert!(session.fixed_atoms().is_empty());
}
