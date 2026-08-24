//! Two-atom double well: saddle at the origin, minima at x = ±1.
//! Rolling forward and reverse must leave the saddle in opposite ways.

use ndarray::{Array1, ArrayView1};
use rgsaddle::{
    IrcConfig, IrcDirection, IrcSession, PointSurface, SaddleError,
};

struct DoubleWell;

impl PointSurface for DoubleWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        // Atom 0 moves in x; the rest are dummy coordinates.
        let t = x[0];
        let energy = (t * t - 1.0) * (t * t - 1.0);
        let mut g = Array1::zeros(x.len());
        g[0] = 4.0 * t * (t * t - 1.0);
        Ok((energy, g))
    }
}

#[test]
fn forward_and_reverse_leave_the_saddle_opposite_ways() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let mut mode = Array1::zeros(6);
    mode[0] = 1.0;
    let cfg = IrcConfig {
        dx: 0.2,
        force_tol: 1e-3,
        ..IrcConfig::default()
    };
    let mut session = IrcSession::new(
        cfg.clone(),
        saddle.clone(),
        masses.clone(),
        mode.clone(),
        IrcDirection::Forward,
    )
    .unwrap();
    let fwd = session.step(&DoubleWell).unwrap();
    let x_fwd = session.position()[0];
    assert!(x_fwd.abs() > 1e-8, "kick must leave the saddle");
    assert!(fwd.arc > 0.0);

    session.set_direction(IrcDirection::Reverse);
    let _rev = session.step(&DoubleWell).unwrap();
    let x_rev = session.position()[0];
    assert!(
        x_fwd * x_rev < 0.0,
        "forward {x_fwd} and reverse {x_rev} must have opposite signs"
    );
}

#[test]
fn kick_mode_comes_from_matrix_free_lanczos_not_a_full_heev() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig::default(),
        saddle,
        masses,
        seed,
        IrcDirection::Forward,
        &DoubleWell,
    )
    .unwrap();
    let _ = session.step(&DoubleWell).unwrap();
    let x0 = session.position()[0];
    assert!(x0.abs() > 1e-8, "Lanczos kick must leave the saddle");
}

#[test]
fn run_reaches_a_well() {
    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig {
            dx: 0.15,
            force_tol: 0.05,
            ..IrcConfig::default()
        },
        saddle,
        masses,
        seed,
        IrcDirection::Forward,
        &DoubleWell,
    )
    .unwrap();
    let report = session.run(&DoubleWell, 40).unwrap();
    assert!(report.at_minimum, "force {} arc {}", report.max_force, report.arc);
    assert!(
        (session.position()[0].abs() - 1.0).abs() < 0.25,
        "x={}",
        session.position()[0]
    );
    assert!(report.energy < 0.1, "energy {}", report.energy);
}
