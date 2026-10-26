//! rgmin 0.5.1 left `Control::ftol_rel` and `Control::noise_floor` unset.
//! Unset is a strict energy test and no noise-floor stop. 0.5.2 keeps
//! those defaults; the sessions take the two fields from `Control::default()`.

use std::sync::OnceLock;

use eindir_core::{Bounds, DifferentiableObjective, Gradient, Objective};
use ndarray::{Array1, ArrayView1, array};
use rgmin::{Control, Method, NoiseFloor, Solver, Status};

fn bounds() -> &'static Bounds<f64> {
    static BOUNDS: OnceLock<Bounds<f64>> = OnceLock::new();
    BOUNDS.get_or_init(|| Bounds::new(array![-1e6], array![1e6], 0.0))
}

/// `f(x) = x²/2`, gradient `x`. One BFGS step from 1 lands on 0.
struct Quadratic;

impl Objective<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        0.5 * x[0] * x[0]
    }
}

impl Gradient<f64> for Quadratic {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, x: ArrayView1<f64>) -> Array1<f64> {
        array![x[0]]
    }
}

impl DifferentiableObjective<f64> for Quadratic {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

/// Energy rises along the steepest-descent direction.
struct Uphill;

impl Objective<f64> for Uphill {
    fn dim(&self) -> usize {
        1
    }
    fn bounds(&self) -> &Bounds<f64> {
        bounds()
    }
    fn eval(&self, x: ArrayView1<f64>) -> f64 {
        x[0]
    }
}

impl Gradient<f64> for Uphill {
    fn dim(&self) -> usize {
        1
    }
    fn grad(&self, _x: ArrayView1<f64>) -> Array1<f64> {
        array![-1.0]
    }
}

impl DifferentiableObjective<f64> for Uphill {
    fn value_and_gradient(&self, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
        (self.eval(x), self.grad(x))
    }
}

fn bfgs(control: Control) -> Solver {
    Solver::new(Method::Bfgs, control, 1)
}

#[test]
fn ftol_rel_and_noise_floor_default_to_the_0_5_1_behaviour() {
    let control = Control::default();
    assert_eq!(control.ftol_rel, None);
    assert_eq!(control.noise_floor, None);

    // The same fill the band, minimum-mode, stencil, and IRC solvers use.
    let session = Control {
        maxiter: usize::MAX,
        gtol: 0.0,
        istep: 1.0,
        maxmove: None,
        ..Control::default()
    };
    assert_eq!(session.ftol_rel, None);
    assert_eq!(session.noise_floor, None);

    // No noise floor: a gradient at or above `gtol` keeps iterating.
    let n_free = 48;
    assert_eq!(control.classify(control.gtol, n_free), Status::Iterating);
    assert_eq!(control.halt(control.gtol, n_free), None);
    assert_eq!(control.classify(1.0, n_free), Status::Iterating);
    assert_eq!(session.classify(1.0, n_free), Status::Iterating);

    // Installing a floor is what 0.5.1 left off.
    let armed = control.clone().with_noise_floor(NoiseFloor::new(1.0));
    assert_eq!(armed.classify(1.0, n_free), Status::NoiseFloor);
    assert_eq!(armed.halt(1.0, n_free), Some(Status::NoiseFloor));

    // A real decrease is taken while the floor is unset. An energy floor
    // wider than that decrease keeps the start. `c` defaults to 3, so
    // `eps_f = 1` refuses the drop from 1/2 to 0.
    let mut x = array![1.0];
    let moved = bfgs(control.clone()).step(&Quadratic, &mut x).unwrap();
    assert!(moved.value < 0.5, "value {}", moved.value);
    assert!(x[0].abs() < 1.0, "x {}", x[0]);
    assert_ne!(moved.status, Status::NoiseFloor);

    let mut y = array![1.0];
    let floored = control
        .clone()
        .with_noise_floor(NoiseFloor::new(0.0).with_eps_f(1.0));
    let stayed = bfgs(floored).step(&Quadratic, &mut y).unwrap();
    assert_eq!(y[0], 1.0);
    assert_eq!(stayed.value, 0.5);
    assert_eq!(stayed.status, Status::NoiseFloor);

    // Unset `ftol_rel` does not grant a relative slack. A step whose
    // energy rises is refused, which is the strict 0.5.1 test.
    let mut z = array![0.0];
    let refused = bfgs(control).step(&Uphill, &mut z).unwrap();
    assert_eq!(z[0], 0.0);
    assert_eq!(refused.value, 0.0);
    assert_ne!(refused.status, Status::NoiseFloor);
}
