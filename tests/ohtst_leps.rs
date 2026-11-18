#![cfg(feature = "ohtst")]

//! OH-TST on the LEPS plus harmonic oscillator surface of Jóhannesson and
//! Jónsson, J. Chem. Phys. 115, 9644 (2001), Appendix B, at
//! `kT = 0.1 E_A` as in their Sec. III B and at lower temperatures.
//!
//! The references are quadrature of the line free energy
//! `A = -kT ln ∫ exp(-V/kT)` and of the reactant integral behind the
//! optimal line, with unit masses, computed in each test and independent of
//! the sampler, and harmonic TST through rgsaddle's dimer and Vineyard
//! functions.
//!
//! Sec. III B, J. Chem. Phys. 115, 9651 (2001), prints
//! `Q‡/Q_{Z_R} = 0.765 ± 0.005`, `⟨|v|⟩ = 0.399` and
//! `k_OH-TST = (4.5 ± 0.2)×10^{-7}` at `k_B T = 0.1 E_A`, with
//! `k = (4.4 ± 0.1)×10^{-7}` from hyperdynamics and `k_hTST = 4.3×10^{-7}`.
//! Eq. (1) on p. 9645 is `k = ⟨|v⊥|⟩ Q‡ / (2 Q_R)`. The product of the two
//! tabulated factors is not that rate, and `ΔA` is drawn in Fig. 8 rather
//! than printed, so the absolute rate is not asserted.

use ndarray::{Array1, array};
use rgsaddle::{
    AnalyticSurfaceDynamics, AnalyticSurfaceSampler, BasinConfig, Basins, Counting, EndState,
    GaussianTransport, Hyperplane, LangevinConfig, MinModeConfig, MinModeSession, MinModeStatus,
    PlaneIntegrator, PlaneOptimizer, PlaneOptimizerConfig, PlanePath, ReversibleWorkConfig,
    ShootingConfig, TransmissionConfig, WorkEstimate, angular_scale_rad_s, basin_density,
    exact_eigh, ohtst_rate, plane_transmission, reactant_density, reversible_work,
    reversible_work_along, transmission_coefficient, vineyard_prefactor,
};

include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/common/ohtst_surfaces.rs"
));

use ohtst_surfaces::{
    LepsHo, descend, descent_path, eigen2, half_plane_integral, hessian, line_draws,
    line_free_energy, stationary,
};

/// Stationary points of the surface and its barrier `E_A`.
struct Geometry {
    reactant: [f64; 2],
    product: [f64; 2],
    saddle: [f64; 2],
    saddle_energy: f64,
    barrier: f64,
}

fn geometry() -> Geometry {
    let reactant = descend(LepsHo::energy, [0.75, 1.3]);
    let product = descend(LepsHo::energy, [3.0, -1.3]);
    let saddle = stationary(LepsHo::energy, [2.02, -0.17]);
    let saddle_energy = LepsHo::energy(saddle[0], saddle[1]).0;
    Geometry {
        reactant,
        product,
        saddle,
        saddle_energy,
        barrier: saddle_energy - LepsHo::energy(reactant[0], reactant[1]).0,
    }
}

impl Geometry {
    /// Angle of the reactant-to-product guideline.
    fn guideline(&self) -> f64 {
        (self.product[1] - self.reactant[1]).atan2(self.product[0] - self.reactant[0])
    }

    /// `Z_R`: the line through the reactant normal to the guideline.
    fn reactant_line(&self) -> Hyperplane {
        line(self.reactant, self.guideline())
    }

    /// The point `offset` along the normal at `theta` from the saddle.
    fn off_saddle(&self, offset: f64, theta: f64) -> [f64; 2] {
        let n = unit(theta);
        [
            self.saddle[0] + offset * n[0],
            self.saddle[1] + offset * n[1],
        ]
    }

    /// Angle of the saddle's unstable mode.
    fn saddle_angle(&self) -> f64 {
        let (_, _, mode) = eigen2(hessian(LepsHo::energy, self.saddle));
        (mode[1] / mode[0]).atan()
    }
}

fn unit(theta: f64) -> [f64; 2] {
    [theta.cos(), theta.sin()]
}

fn line(point: [f64; 2], theta: f64) -> Hyperplane {
    let n = unit(theta);
    Hyperplane::new(array![point[0], point[1]], array![n[0], n[1]]).unwrap()
}

fn angle_of(plane: &Hyperplane) -> f64 {
    plane.normal()[1].atan2(plane.normal()[0])
}

/// Quadrature line free energy, shifted by the saddle energy.
fn free_energy(geometry: &Geometry, point: [f64; 2], theta: f64, kt: f64) -> f64 {
    line_free_energy(
        LepsHo::energy,
        point,
        unit(theta),
        kt,
        geometry.saddle_energy,
        8.0,
    )
}

/// The optimal line by Newton's method on quadrature: its offset from the
/// saddle along its normal, its angle, and the Hessian of `A / kT` in
/// those two coordinates (offset, radians).
fn optimal_line(geometry: &Geometry, kt: f64) -> (f64, f64, [[f64; 2]; 2]) {
    let f = |d: f64, t: f64| free_energy(geometry, geometry.off_saddle(d, t), t, kt) / kt;
    let (mut d, mut t) = (0.0, geometry.saddle_angle());
    let h = 1.0e-3;
    let mut curvature = [[0.0; 2]; 2];
    for _ in 0..6 {
        let f0 = f(d, t);
        let (fdp, fdm, ftp, ftm) = (f(d + h, t), f(d - h, t), f(d, t + h), f(d, t - h));
        let cross =
            (f(d + h, t + h) - f(d + h, t - h) - f(d - h, t + h) + f(d - h, t - h)) / (4.0 * h * h);
        let (gd, gt) = ((fdp - fdm) / (2.0 * h), (ftp - ftm) / (2.0 * h));
        let (hdd, htt) = (
            (fdp - 2.0 * f0 + fdm) / (h * h),
            (ftp - 2.0 * f0 + ftm) / (h * h),
        );
        let det = hdd * htt - cross * cross;
        d -= (htt * gd - cross * gt) / det;
        t -= (hdd * gt - cross * gd) / det;
        curvature = [[hdd, cross], [cross, htt]];
    }
    (d, t, curvature)
}

/// Quadrature OH-TST rate with unit masses, `sqrt(kT/2π) Z‡ / Q_R`, with
/// `Q_R` the integral over the reactant side of the optimal line.
fn quadrature_rate(geometry: &Geometry, kt: f64, offset: f64, theta: f64) -> f64 {
    line_rate(geometry, kt, geometry.off_saddle(offset, theta), theta)
}

/// Quadrature TST rate with unit masses of the line through `point` with
/// its normal at `theta`, `Q_R` the integral behind it.
fn line_rate(geometry: &Geometry, kt: f64, point: [f64; 2], theta: f64) -> f64 {
    let reactant = half_plane_integral(
        LepsHo::energy,
        point,
        unit(theta),
        kt,
        geometry.saddle_energy,
        8.0,
    );
    let plane = (-free_energy(geometry, point, theta, kt) / kt).exp();
    (kt / (2.0 * std::f64::consts::PI)).sqrt() * plane / reactant
}

/// Harmonic TST with unit masses from the Hessian eigenvalues and
/// rgsaddle's Vineyard prefactor, in the surface's time unit.
fn harmonic_rate(geometry: &Geometry, kt: f64) -> f64 {
    let eigenvalues = |at: [f64; 2]| {
        let h = hessian(LepsHo::energy, at);
        exact_eigh(ndarray::arr2(&h).view()).unwrap().0
    };
    let (minimum, saddle) = (eigenvalues(geometry.reactant), eigenvalues(geometry.saddle));
    let prefactor = vineyard_prefactor(&[minimum[0], minimum[1]], &[saddle[1]], 1.0).unwrap()
        / angular_scale_rad_s();
    prefactor * (-geometry.barrier / kt).exp()
}

/// The plane of a dimer run from near the saddle, and its curvature.
fn dimer_plane() -> (Hyperplane, f64) {
    let config = MinModeConfig::default().with_force_tol(1.0e-8);
    let mut session = MinModeSession::new(config, array![1.9, -0.05], array![1.0, 0.2]).unwrap();
    for _ in 0..2000 {
        let report = session.step(&LepsHo).unwrap();
        if report.status == MinModeStatus::Converged {
            return (
                Hyperplane::from_min_mode(&session).unwrap(),
                report.curvature,
            );
        }
    }
    panic!("the dimer did not converge");
}

fn sampler(kt: f64, time_step: f64) -> LangevinConfig {
    LangevinConfig::default()
        .with_thermal(kt)
        .with_time_step(time_step)
        .with_friction(1.0)
        .with_burn_in(300)
        .with_stride(5)
        .with_seed(5)
}

/// A step of 0.3 is past the BAOAB stability limit at the repulsive
/// walls, so these runs use the exact GHMC integrator instead.
fn large_step_sampler(kt: f64) -> LangevinConfig {
    sampler(kt, 0.3)
        .with_integrator(PlaneIntegrator::Ghmc)
        .with_records(1000)
        .with_seed(11)
}

/// Simpson's rule on quadrature `dA/ds` at the nodes of a reversible-work
/// leg from `(p0, θ0)` to `(p1, θ1)`, with the point linear in `s` and the
/// normal turning at a constant rate, in units of `kT`.
fn simpson_on_leg(
    geometry: &Geometry,
    kt: f64,
    (p0, t0): ([f64; 2], f64),
    (p1, t1): ([f64; 2], f64),
    nodes: usize,
) -> f64 {
    let at = |s: f64| {
        let point = [p0[0] + s * (p1[0] - p0[0]), p0[1] + s * (p1[1] - p0[1])];
        free_energy(geometry, point, t0 + s * (t1 - t0), kt) / kt
    };
    let h = 1.0e-4;
    let slopes: Vec<f64> = (0..nodes)
        .map(|i| {
            let s = i as f64 / (nodes - 1) as f64;
            (at(s + h) - at(s - h)) / (2.0 * h)
        })
        .collect();
    let width = 1.0 / (nodes - 1) as f64;
    let inner: f64 = slopes[1..nodes - 1]
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 2 == 0 { 4.0 * v } else { 2.0 * v })
        .sum();
    (slopes[0] + inner + slopes[nodes - 1]) * width / 3.0
}

/// A dimer from near the saddle converges to the paper's saddle point,
/// (2.02083, -0.172881), and its plane is normal to the unstable mode.
/// That is the plane OH-TST starts from.
#[test]
fn leps_dimer_seeds_the_saddle_plane() {
    let geometry = geometry();
    let (plane, curvature) = dimer_plane();
    let point = plane.point();
    assert!((point[0] - 2.02083).abs() < 5.0e-5 && (point[1] + 0.172881).abs() < 5.0e-5);
    let (lowest, _, mode) = eigen2(hessian(LepsHo::energy, geometry.saddle));
    assert!((curvature - lowest).abs() < 1.0e-3 * lowest.abs());
    let along = (plane.normal()[0] * mode[0] + plane.normal()[1] * mode[1]).abs();
    assert!(1.0 - along < 1.0e-8, "normal off the mode by {along}");
}

/// At `kT = 0.1 E_A` the optimizer, translating and turning the dimer's
/// plane, reaches the quadrature optimum, -4.9174° and 0.00092 from the
/// saddle along its normal, within four errors (last force and torque
/// errors over the quadrature curvature). The saddle plane is at -4.6601°;
/// the optimum is resolved from it, the turn toward the ridge that Sec.
/// III B describes.
#[test]
fn leps_optimizer_climbs_from_the_dimer_plane() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (offset, theta, curvature) = optimal_line(&geometry, kt);
    let (seed, _) = dimer_plane();
    let config = PlaneOptimizerConfig::default()
        .with_sampler(large_step_sampler(kt))
        .with_max_iterations(100);
    let optimum = PlaneOptimizer::new(&seed, config)
        .unwrap()
        .run(&LepsHo)
        .unwrap();
    let last = optimum.trace.last().unwrap();
    let found = angle_of(&optimum.plane);
    let shift = optimum
        .plane
        .normal()
        .dot(&(&optimum.plane.point() - &array![geometry.saddle[0], geometry.saddle[1]]));
    let angle_error = last.rotation_error / kt / curvature[1][1].abs();
    let offset_error = last.normal_force_error / kt / curvature[0][0].abs();
    assert!(
        (found - theta).abs() < 4.0 * angle_error,
        "{}° ± {}° against {}°",
        found.to_degrees(),
        angle_error.to_degrees(),
        theta.to_degrees()
    );
    assert!(
        (shift - offset).abs() < 4.0 * offset_error,
        "{shift} ± {offset_error} against {offset}"
    );
    assert!((found - angle_of(&seed)).abs() > 4.0 * angle_error);
}

/// The OH-TST rate at `kT = 0.1 E_A`. The density of the reactant
/// distribution at `Z_R`, the guideline plane through the minimum, matches
/// quadrature (0.6509) both from crossings behind a wall at the optimal
/// plane and from parallel planes. The parallel planes, with control
/// variates from the Hessian at the minimum, gave 0.6516 ± 0.0007 from
/// twelve planes of 4000 records, and the crossings 0.662 ± 0.028 from
/// 16 000. Without the control variates the mean forces on the planes
/// behind and through the minimum have errors 13 to 20 times larger, and
/// those next to the barrier about the same. The reversible work from `Z_R` to the
/// optimal plane matches quadrature (8.3081 kT; Simpson's rule on its
/// slopes at the same nodes is within 0.01 kT of that); and their rate
/// matches the quadrature rate, 3.8589e-5 in the surface's time unit, all
/// within four standard errors. Harmonic TST from the Vineyard prefactor
/// gives 4.0410e-5, a ratio of 0.955.
///
/// The transmission coefficient shot from the optimal plane with the
/// sampler's friction is a probability.
#[test]
fn leps_rate_against_quadrature_and_harmonic_tst() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (offset, theta, _) = optimal_line(&geometry, kt);
    let summit = geometry.off_saddle(offset, theta);
    let optimal = line(summit, theta);
    let start = geometry.reactant_line();
    let reactant = array![geometry.reactant[0], geometry.reactant[1]];

    let exact = quadrature_rate(&geometry, kt, offset, theta);
    let rise = (free_energy(&geometry, summit, theta, kt)
        - free_energy(&geometry, geometry.reactant, geometry.guideline(), kt))
        / kt;
    let density_exact = exact / (kt / (2.0 * std::f64::consts::PI)).sqrt() * rise.exp();

    let density = reactant_density(
        &LepsHo,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    assert!(
        (density.density - density_exact).abs() < 4.0 * density.density_error,
        "density {} ± {} against {density_exact}",
        density.density,
        density.density_error
    );
    let basin = basin_density(
        &LepsHo,
        reactant.view(),
        start.normal(),
        &BasinConfig::default().with_sampler(sampler(kt, 0.1).with_records(4000)),
    )
    .unwrap();
    assert!(
        (basin.density - density_exact).abs()
            < 4.0 * basin.density_error + basin.quadrature_error * basin.density,
        "parallel planes {} ± {} against {density_exact}",
        basin.density,
        basin.density_error
    );
    assert!(basin.density_error < 0.002 * density_exact);

    let nodes = 17;
    let layout = simpson_on_leg(
        &geometry,
        kt,
        (geometry.reactant, geometry.guideline()),
        (summit, theta),
        nodes,
    );
    assert!((layout - rise).abs() < 0.01, "{layout} against {rise}");
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000).with_retain(128))
        .with_nodes(nodes);
    let mut engine =
        AnalyticSurfaceSampler::new(&LepsHo, config.sampler.clone()).with_start(reactant.clone());
    let work = reversible_work(&mut engine, &[start, optimal], &config).unwrap();
    let (value, error) = (work.work / kt, work.error / kt);
    assert!(
        (value - rise).abs() < 4.0 * error,
        "work {value} ± {error} kT against {rise}"
    );

    let rate = ohtst_rate(&density, &work).unwrap();
    assert!(
        (rate.log_rate - exact.ln()).abs() < 4.0 * rate.log_rate_error,
        "rate {} ± {} against {exact}",
        rate.rate,
        rate.error
    );
    let ratio = exact / harmonic_rate(&geometry, kt);
    assert!((ratio - 0.95493).abs() < 1.0e-4, "ratio {ratio}");

    let shooting = ShootingConfig::default()
        .with_thermal(kt)
        .with_friction(1.0)
        .with_time_step(0.05)
        .with_steps(400)
        .with_record_every(40)
        .with_shots(256)
        .with_blocks(8)
        .with_plateau_record(9);
    let kappa = plane_transmission(&LepsHo, &work.end, &shooting).unwrap();
    assert!(
        kappa.kappa > 4.0 * kappa.error && kappa.kappa < 1.0 + 4.0 * kappa.error,
        "{kappa:?}"
    );
}

/// With the orientation held at the guideline normal, as in the paper's
/// Fig. 8(b), the free energy along the guideline peaks 2.3876 from the
/// reactant, 5.7344 kT above `Z_R` by quadrature against 8.3081 kT for
/// the optimal plane. The optimizer with rotation off finds that peak from
/// the plane through the saddle, and the reversible work to it matches.
/// Most of its error comes from the last nodes, where the line crosses the
/// ridge at a slant.
#[test]
fn leps_fixed_orientation_barrier() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let theta = geometry.guideline();
    let n = unit(theta);
    let along = |o: f64| {
        let point = [
            geometry.reactant[0] + o * n[0],
            geometry.reactant[1] + o * n[1],
        ];
        free_energy(&geometry, point, theta, kt) / kt
    };
    let (mut a, mut b) = (2.0, 2.8);
    while b - a > 1.0e-6 {
        let third = (b - a) / 3.0;
        if along(a + third) > along(b - third) {
            b -= third;
        } else {
            a += third;
        }
    }
    let peak = 0.5 * (a + b);
    let h = 1.0e-3;
    let curvature = (along(peak + h) - 2.0 * along(peak) + along(peak - h)) / (h * h);
    let rise = along(peak) - along(0.0);
    assert!((peak - 2.38761).abs() < 1.0e-4 && (rise - 5.73441).abs() < 1.0e-4);

    let saddle_offset = n[0] * (geometry.saddle[0] - geometry.reactant[0])
        + n[1] * (geometry.saddle[1] - geometry.reactant[1]);
    let through = line(
        [
            geometry.reactant[0] + saddle_offset * n[0],
            geometry.reactant[1] + saddle_offset * n[1],
        ],
        theta,
    );
    let config = PlaneOptimizerConfig::default()
        .with_sampler(large_step_sampler(kt))
        .with_rotate(false)
        .with_recenter(false)
        .with_max_iterations(100);
    let optimum = PlaneOptimizer::new(&through, config)
        .unwrap()
        .run(&LepsHo)
        .unwrap();
    let found = optimum
        .plane
        .normal()
        .dot(&(&optimum.plane.point() - &array![geometry.reactant[0], geometry.reactant[1]]));
    let error = optimum.trace.last().unwrap().normal_force_error / kt / curvature.abs();
    assert!(
        (found - peak).abs() < 4.0 * error,
        "{found} ± {error} against {peak}"
    );

    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000))
        .with_nodes(17);
    let mut engine = AnalyticSurfaceSampler::new(&LepsHo, config.sampler.clone())
        .with_start(array![geometry.reactant[0], geometry.reactant[1]]);
    let work = reversible_work(
        &mut engine,
        &[geometry.reactant_line(), optimum.plane.clone()],
        &config,
    )
    .unwrap();
    let top = [
        geometry.reactant[0] + found * n[0],
        geometry.reactant[1] + found * n[1],
    ];
    let exact = along(found) - along(0.0);
    let layout = simpson_on_leg(&geometry, kt, (geometry.reactant, theta), (top, theta), 17);
    assert!((layout - exact).abs() < 0.02, "{layout} against {exact}");
    let (value, error) = (work.work / kt, work.error / kt);
    assert!(
        (value - exact).abs() < 4.0 * error,
        "work {value} ± {error} kT against {exact}"
    );
}

/// As the temperature falls the optimal line turns back to the saddle
/// line and the OH-TST rate approaches harmonic TST: by quadrature the
/// ratio is 0.955 at `kT = 0.1 E_A`, 0.979 at 0.05 and 0.992 at 0.02,
/// and the optimum is 0.053° from the saddle line at 0.02. At
/// `kT = 0.02 E_A` the sampled rate matches the quadrature rate and
/// harmonic TST within four standard errors.
#[test]
fn leps_rate_approaches_harmonic_tst_at_low_temperature() {
    let geometry = geometry();
    let saddle_angle = geometry.saddle_angle();
    let mut previous = f64::INFINITY;
    for fraction in [0.1, 0.05, 0.02] {
        let kt = fraction * geometry.barrier;
        let (offset, theta, _) = optimal_line(&geometry, kt);
        let ratio = quadrature_rate(&geometry, kt, offset, theta) / harmonic_rate(&geometry, kt);
        let gap = 1.0 - ratio;
        assert!(
            gap > 0.0 && gap < previous && gap < 0.5 * fraction,
            "ratio {ratio} at {fraction}"
        );
        previous = gap;
        let turn = (theta - saddle_angle).abs().to_degrees();
        assert!(turn < 3.0 * fraction, "{turn}° at {fraction}");
    }

    let kt = 0.02 * geometry.barrier;
    let (offset, theta, _) = optimal_line(&geometry, kt);
    let optimal = line(geometry.off_saddle(offset, theta), theta);
    let start = geometry.reactant_line();
    let reactant = array![geometry.reactant[0], geometry.reactant[1]];
    let density = reactant_density(
        &LepsHo,
        &start,
        reactant.view(),
        std::slice::from_ref(&optimal),
        &sampler(kt, 0.1).with_records(16_000),
    )
    .unwrap();
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(4000))
        .with_nodes(17);
    let mut engine =
        AnalyticSurfaceSampler::new(&LepsHo, config.sampler.clone()).with_start(reactant);
    let work = reversible_work(&mut engine, &[start, optimal], &config).unwrap();
    let rate = ohtst_rate(&density, &work).unwrap();
    assert!(rate.log_rate_error < 0.25, "{rate:?}");
    for (reference, name) in [
        (quadrature_rate(&geometry, kt, offset, theta), "quadrature"),
        (harmonic_rate(&geometry, kt), "harmonic"),
    ] {
        assert!(
            (rate.log_rate - reference.ln()).abs() < 4.0 * rate.log_rate_error,
            "rate {} ± {} against {name} {reference}",
            rate.rate,
            rate.error
        );
    }
}

/// `κ k_TST` does not depend on the dividing line. Unlike the reactant,
/// the product well is shallow, 1.745 below the saddle against 3.634 from
/// the reactant, so at `kT = 0.1 E_A` trajectories that have settled in it
/// come back within the shooting time: without commitment `κ(t)` of the
/// optimal line falls from 0.7715 at `t = 4` to 0.7217 ± 0.0071 at
/// `t = 20`. A half that stays in one state for six checkpoints a time
/// unit apart is committed, and `κ(t)` is then flat from `t = 6`. A
/// window of three units also commits trajectories that linger at the
/// barrier and come back: it gave 0.78589 ± 0.00734 on the optimal line,
/// against 0.76294 for six and 0.76831 ± 0.00645 for ten.
///
/// With the sampler's friction and that commitment, shots from exact draws
/// on the optimal line and on the line turned by 0.25 rad about its summit,
/// whose quadrature TST rates are 3.8589e-5 and 7.1880e-5, give
/// `κ` = 0.76294 ± 0.00660 and 0.40454 ± 0.00763 at the first crossing,
/// so `κ k_TST` = 2.9441e-5 and 2.9078e-5; by Bennett–Chandler 0.75928 ±
/// 0.00705 and 0.40137 ± 0.00751. Assigned by quench they are 0.76294 ±
/// 0.00660 and 0.40454 ± 0.00766. The separatrix crosses the turned line
/// at a slant, and after one time unit the quench is 0.035 above its
/// plateau there against 0.091 for the side of the line.
#[test]
fn leps_transmission_times_tst_rate_is_line_independent() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (offset, theta, _) = optimal_line(&geometry, kt);
    let summit = geometry.off_saddle(offset, theta);
    let shots = 4096;
    let basins = Basins::new(
        vec![array![geometry.reactant[0], geometry.reactant[1]]],
        vec![array![geometry.product[0], geometry.product[1]]],
    );
    let shooting = TransmissionConfig::default()
        .with_thermal(kt)
        .with_friction(1.0)
        .with_time_step(0.05)
        .with_steps(400)
        .with_check_every(20)
        .with_shots(shots)
        .with_blocks(32);
    let optimal_starts = line_draws(LepsHo::energy, summit, unit(theta), kt, 8.0, shots, 5);
    let free = transmission_coefficient(
        &mut AnalyticSurfaceDynamics::new(&LepsHo),
        &line(summit, theta),
        &optimal_starts,
        &shooting,
    )
    .unwrap();
    let early = free.kappa_of_t[3];
    assert!(
        free.kappa < early - 4.0 * free.error,
        "κ(t) {:?} ± {}",
        free.kappa_of_t,
        free.error
    );

    let committed = shooting.with_commit(6);
    let mut products = Vec::new();
    for angle in [theta, theta + 0.25] {
        let rate = line_rate(&geometry, kt, summit, angle);
        let starts = line_draws(LepsHo::energy, summit, unit(angle), kt, 8.0, shots, 5);
        let plane = line(summit, angle);
        let run = |counting: Counting, end_state: EndState| {
            let config = committed
                .clone()
                .with_counting(counting)
                .with_end_state(end_state);
            transmission_coefficient(
                &mut AnalyticSurfaceDynamics::new(&LepsHo),
                &plane,
                &starts,
                &config,
            )
            .unwrap()
        };
        let first = run(Counting::FirstCrossing, EndState::Side);
        let last = first.kappa_of_t.len() - 1;
        assert!(
            (first.kappa_of_t[7] - first.kappa).abs() < 0.2 * first.error,
            "κ(t) {:?}",
            first.kappa_of_t
        );
        let final_state = run(Counting::Final, EndState::Side);
        let quench = run(Counting::FirstCrossing, basins.clone().into());
        for other in [&final_state, &quench] {
            assert!(
                (other.kappa - first.kappa).abs() < 4.0 * other.error,
                "{} ± {} against {} ± {}",
                other.kappa,
                other.error,
                first.kappa,
                first.error
            );
            assert_eq!(other.unassigned, 0);
        }
        if angle != theta {
            assert!(
                (quench.kappa_of_t[0] - quench.kappa_of_t[last]).abs()
                    < 0.5 * (first.kappa_of_t[0] - first.kappa_of_t[last]).abs(),
                "quench {:?}, side {:?}",
                quench.kappa_of_t,
                first.kappa_of_t
            );
        }
        products.push((rate, first.kappa, first.error, quench.kappa, quench.error));
    }
    let (optimal, turned) = (products[0], products[1]);
    assert!((turned.0 / optimal.0 - 1.86).abs() < 0.01);
    for (a, b) in [
        ((optimal.1, optimal.2), (turned.1, turned.2)),
        ((optimal.3, optimal.4), (turned.3, turned.4)),
    ] {
        let (left, right) = (a.0 * optimal.0, b.0 * turned.0);
        let error = (a.1 * optimal.0).hypot(b.1 * turned.0);
        assert!(
            (left - right).abs() < 4.0 * error,
            "κ k_TST {left:e} against {right:e} ± {error:e}"
        );
    }
}

/// The work from the line through the reactant minimum, normal to its
/// soft mode, to the saddle line along the PAFI planes of the
/// steepest-descent path at `kT` = 0.1 `E_A`, against quadrature of the
/// two lines, 2.92859. Over 13 spline pieces integration gave
/// 2.92179 ± 0.03840 and MBAR across the 27 nodes with Gaussian maps
/// 2.92919 ± 0.00290; BAR gave 2.94235 ± 0.01837 with Gaussian maps and
/// 2.89296 ± 0.03728 with rigid ones. Through the corner of the surface
/// the rigid map loses overlap: the smallest between neighbours is 0.346
/// with it and 0.471 with the Gaussian map, which keeps over 0.9 of its
/// fitted change there.
#[test]
fn leps_work_along_the_minimum_energy_path() {
    let geometry = geometry();
    let kt = 0.1 * geometry.barrier;
    let (saddle_plane, _) = dimer_plane();
    let product = array![geometry.product[0], geometry.product[1]];
    let saddle_plane = saddle_plane.oriented_toward(product.view());
    let (mode, pivot) = (saddle_plane.normal(), saddle_plane.point());
    let mut points = descent_path(
        LepsHo::energy,
        [pivot[0], pivot[1]],
        [-mode[0], -mode[1]],
        0.2,
    );
    points.reverse();
    let reactant = points[0];
    let (_, _, soft) = eigen2(hessian(LepsHo::energy, reactant));
    let along = [points[1][0] - reactant[0], points[1][1] - reactant[1]];
    let sign = (soft[0] * along[0] + soft[1] * along[1]).signum();
    let start_normal = array![sign * soft[0], sign * soft[1]];
    let points: Vec<Array1<f64>> = points.iter().map(|p| array![p[0], p[1]]).collect();
    let path = PlanePath::curve(&points, Some(start_normal.view()), Some(mode)).unwrap();
    let (first, last) = (path.start().unwrap(), path.end().unwrap());
    let exact = free_energy(&geometry, [pivot[0], pivot[1]], angle_of(&last), kt)
        - free_energy(&geometry, reactant, angle_of(&first), kt);
    let config = ReversibleWorkConfig::default()
        .with_sampler(sampler(kt, 0.1).with_records(2000))
        .with_nodes(3);
    let mut engine = AnalyticSurfaceSampler::new(&LepsHo, config.sampler.clone());
    let rigid = reversible_work_along(&mut engine, &path, &config).unwrap();
    let config = config
        .with_transport(GaussianTransport::default().into())
        .with_mbar(true)
        .with_estimate(WorkEstimate::Mbar);
    let mut engine = AnalyticSurfaceSampler::new(&LepsHo, config.sampler.clone());
    let work = reversible_work_along(&mut engine, &path, &config).unwrap();
    let smallest = |work: &rgsaddle::ReversibleWork| {
        work.nodes
            .iter()
            .filter_map(|n| n.overlap)
            .fold(1.0, f64::min)
    };
    let estimates = [
        ("integration", work.integrated_work, work.integrated_error),
        ("BAR", work.bar_work.unwrap(), work.bar_error.unwrap()),
        ("MBAR", work.mbar_work.unwrap(), work.mbar_error.unwrap()),
        (
            "rigid BAR",
            rigid.bar_work.unwrap(),
            rigid.bar_error.unwrap(),
        ),
    ];
    for (name, value, error) in estimates {
        assert!(
            (value - exact).abs() < 4.0 * error,
            "{name} {value} ± {error} against {exact}"
        );
    }
    assert!(work.warnings.is_empty() && rigid.warnings.is_empty());
    assert!(smallest(&rigid) > 0.3 && smallest(&work) > smallest(&rigid));
}

/// Sec. III B, p. 9651, pairs `⟨|v|⟩ = 0.399` and `Q‡/Q_{Z_R} = 0.765 ± 0.005`
/// with `k_OH-TST = (4.5 ± 0.2)×10^{-7}`. Eq. (1) on p. 9645 multiplies
/// `⟨|v⊥|⟩/2` by a ratio of configuration integrals. That product of the
/// two printed factors is not the printed rate: `ΔA` is in Fig. 8, which
/// does not give a number that can be read back.
#[test]
fn section_iii_b_prefactor_is_not_the_printed_rate() {
    let half_speed = 0.399_f64 / 2.0;
    let ratio = 0.765_f64;
    let prefactor = half_speed * ratio;
    let published = 4.5_f64 / 10_000_000.0;
    let uncertainty = 0.2_f64 / 10_000_000.0;
    assert!(
        (prefactor - published).abs() > 1000.0 * uncertainty,
        "{prefactor} against {published} ± {uncertainty}"
    );
    let low = ratio - 0.005;
    let high = ratio + 0.005;
    assert!((low * half_speed - published).abs() > 1000.0 * uncertainty);
    assert!((high * half_speed - published).abs() > 1000.0 * uncertainty);
}
