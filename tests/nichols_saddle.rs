//! Nichols displacement against i-PI `mintools.nichols`, and index-1
//! searches on a quadratic saddle and on Muller-Brown.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    HessianUpdate, Index1Config, Index1Session, Index1Status, NicholsMode, PointSurface,
    SaddleError, bofill_update, nichols_displacement, nichols_step, powell_update,
};

fn close(got: &[f64], expect: &[f64], tol: f64) {
    assert_eq!(got.len(), expect.len());
    for (i, (a, b)) in got.iter().zip(expect).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "component {i}: {a} differs from {b} by {}",
            (a - b).abs()
        );
    }
}

/// V = -x^2 + y^2. Saddle at the origin, Hessian diag(-2, 2).
struct Quadratic;

impl PointSurface for Quadratic {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let energy = -x[0] * x[0] + x[1] * x[1];
        Ok((energy, array![-2.0 * x[0], 2.0 * x[1]]))
    }
}

fn quadratic_hessian() -> Array2<f64> {
    array![[-2.0, 0.0], [0.0, 2.0]]
}

#[test]
fn nichols_matches_ipi_on_the_same_hessian_and_gradient() {
    // i-PI `nichols(-g, eigh(H), ones, big_step, mode=1)` at e1460fef.
    let h = array![[1.5, 0.3], [0.3, -0.4]];
    let g = array![0.2, -0.5];
    let dx = nichols_displacement(h.view(), g.view(), None, 0.4, NicholsMode::Index1).unwrap();
    close(
        dx.as_slice().unwrap(),
        &[0.016359918200408982, -0.7334696659850035],
        1e-12,
    );

    let positive = array![[0.2, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 4.0]];
    let gp = array![0.1, -0.2, 0.05];
    let dx =
        nichols_displacement(positive.view(), gp.view(), None, 0.3, NicholsMode::Index1).unwrap();
    close(
        dx.as_slice().unwrap(),
        &[
            0.3636363636363637,
            0.19512195121951223,
            -0.014184397163120569,
        ],
        1e-12,
    );

    let two_neg = array![[-3.0, 0.0, 0.0], [0.0, -0.4, 0.0], [0.0, 0.0, 2.0]];
    let gn = array![0.3, -0.1, 0.2];
    let dx =
        nichols_displacement(two_neg.view(), gn.view(), None, 0.5, NicholsMode::Index1).unwrap();
    close(
        dx.as_slice().unwrap(),
        &[
            0.14634146341463417,
            0.18181818181818185,
            -0.06779661016949153,
        ],
        1e-12,
    );
}

#[test]
fn nichols_on_a_supplied_spectrum_matches_ipi() {
    // Same mixed Hessian, with the spectrum i-PI's eigh produced, so
    // the comparison does not depend on the Jacobi ordering.
    let evals = array![-0.44624294225856376, 1.5462429422585637];
    let evecs = array![
        [0.15234391170286127, -0.9883275431591851],
        [-0.9883275431591851, -0.15234391170286127]
    ];
    let g = array![0.2, -0.5];
    let dx = nichols_step(
        g.view(),
        evals.view(),
        evecs.view(),
        None,
        0.4,
        NicholsMode::Index1,
    )
    .unwrap();
    close(
        dx.as_slice().unwrap(),
        &[0.016359918200408982, -0.7334696659850035],
        1e-12,
    );
}

#[test]
fn minimize_nichols_uses_the_trust_radius() {
    let h = quadratic_hessian();
    let g = array![-0.8, -0.6];
    let dx = nichols_displacement(h.view(), g.view(), None, 0.2, NicholsMode::Minimize).unwrap();
    close(dx.as_slice().unwrap(), &[0.2, 0.075], 1e-12);
}

#[test]
fn mass_weighted_nichols_matches_ipi() {
    let h = quadratic_hessian();
    let g = array![-0.8, -0.6];
    let masses = array![2.0, 8.0];
    let dx = nichols_displacement(
        h.view(),
        g.view(),
        Some(masses.view()),
        1.0,
        NicholsMode::Index1,
    )
    .unwrap();
    close(
        dx.as_slice().unwrap(),
        &[-0.4923076923076925, 0.17142857142857146],
        1e-12,
    );
}

#[test]
fn powell_and_bofill_match_the_reference_update() {
    let step = array![0.2, -0.1];
    let dg = array![-0.05, 0.4];
    let mut powell = array![[2.0, 0.1], [0.1, 3.0]];
    powell_update(&mut powell, step.view(), dg.view()).unwrap();
    close(
        powell.as_slice().unwrap(),
        &[0.9760000000000002, 2.452, 2.452, 0.9040000000000004],
        1e-12,
    );

    let mut bofill = array![[2.0, 0.1], [0.1, 3.0]];
    bofill_update(&mut bofill, step.view(), dg.view()).unwrap();
    close(
        bofill.as_slice().unwrap(),
        &[
            0.81497756097561,
            2.129955121951219,
            2.129955121951219,
            0.25991024390243966,
        ],
        1e-12,
    );
}

#[test]
fn trust_cap_limits_the_quadratic_step() {
    let config = Index1Config {
        update: HessianUpdate::Bofill,
        trust_radius: 0.2,
        force_tol: 1e-10,
        ..Index1Config::default()
    };
    let mut session =
        Index1Session::new(config, array![0.4, -0.3], Some(quadratic_hessian()), None).unwrap();
    let report = session.step(&Quadratic).unwrap();
    assert!((report.max_step - 0.2).abs() < 1e-12);
    close(session.position().as_slice().unwrap(), &[0.2, -0.15], 1e-12);
}

fn quadratic_converges(update: HessianUpdate, hessian: Option<Array2<f64>>) {
    let config = Index1Config {
        update,
        trust_radius: 0.2,
        force_tol: 1e-8,
        fd_dr: 1e-5,
        mode: NicholsMode::Index1,
    };
    let mut session = Index1Session::new(config, array![0.4, -0.3], hessian, None).unwrap();
    let report = session.run(&Quadratic, 20).unwrap();
    assert_eq!(report.status, Index1Status::Converged, "{update:?}");
    close(session.position().as_slice().unwrap(), &[0.0, 0.0], 1e-6);
    assert!(report.curvature < -1.0, "curvature {}", report.curvature);
}

#[test]
fn quadratic_saddle_with_bofill_and_powell() {
    quadratic_converges(HessianUpdate::Bofill, Some(quadratic_hessian()));
    quadratic_converges(HessianUpdate::Powell, Some(quadratic_hessian()));
    quadratic_converges(HessianUpdate::Bofill, None);
}

const MB_A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
const MB_AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
const MB_B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
const MB_C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
const MB_X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
const MB_Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];

fn muller_brown(x: f64, y: f64) -> (f64, [f64; 2], [[f64; 2]; 2]) {
    let mut energy = 0.0;
    let mut gx = 0.0;
    let mut gy = 0.0;
    let mut hxx = 0.0;
    let mut hyy = 0.0;
    let mut hxy = 0.0;
    for i in 0..4 {
        let dx = x - MB_X0[i];
        let dy = y - MB_Y0[i];
        let exponent = MB_AA[i] * dx * dx + MB_B[i] * dx * dy + MB_C[i] * dy * dy;
        let weight = MB_A[i] * exponent.exp();
        let px = 2.0 * MB_AA[i] * dx + MB_B[i] * dy;
        let py = MB_B[i] * dx + 2.0 * MB_C[i] * dy;
        energy += weight;
        gx += weight * px;
        gy += weight * py;
        hxx += weight * (px * px + 2.0 * MB_AA[i]);
        hyy += weight * (py * py + 2.0 * MB_C[i]);
        hxy += weight * (px * py + MB_B[i]);
    }
    (energy, [gx, gy], [[hxx, hxy], [hxy, hyy]])
}

struct MullerBrown;

impl PointSurface for MullerBrown {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (energy, g, _) = muller_brown(x[0], x[1]);
        Ok((energy, array![g[0], g[1]]))
    }
}

fn mb_hessian(x: f64, y: f64) -> Array2<f64> {
    let (_, _, h) = muller_brown(x, y);
    array![[h[0][0], h[0][1]], [h[1][0], h[1][1]]]
}

fn mb_index(x: f64, y: f64) -> i32 {
    let h = mb_hessian(x, y);
    let tr = h[(0, 0)] + h[(1, 1)];
    let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
    let disc = (tr * tr - 4.0 * det).sqrt();
    let low = 0.5 * (tr - disc);
    let high = 0.5 * (tr + disc);
    i32::from(low < 0.0) + i32::from(high < 0.0)
}

#[test]
fn muller_brown_gradient_matches_a_central_difference() {
    let (_, g, h) = muller_brown(-0.7, 0.5);
    let dr = 1e-6;
    let (ep, _, _) = muller_brown(-0.7 + dr, 0.5);
    let (em, _, _) = muller_brown(-0.7 - dr, 0.5);
    assert!(((ep - em) / (2.0 * dr) - g[0]).abs() < 1e-4);
    let (_, gp, _) = muller_brown(-0.7 + dr, 0.5);
    let (_, gm, _) = muller_brown(-0.7 - dr, 0.5);
    assert!(((gp[1] - gm[1]) / (2.0 * dr) - h[1][0]).abs() < 1e-3);
}

fn find_muller_saddle(update: HessianUpdate, start: [f64; 2], target: [f64; 2]) {
    let config = Index1Config {
        update,
        trust_radius: 0.15,
        force_tol: 1e-6,
        fd_dr: 1e-4,
        mode: NicholsMode::Index1,
    };
    let h0 = mb_hessian(start[0], start[1]);
    let mut session =
        Index1Session::new(config, array![start[0], start[1]], Some(h0), None).unwrap();
    let report = session.run(&MullerBrown, 40).unwrap();
    assert_eq!(
        report.status,
        Index1Status::Converged,
        "{update:?} from {start:?} force {}",
        report.max_force
    );
    let x = session.position();
    assert!(
        (x[0] - target[0]).abs() < 1e-4 && (x[1] - target[1]).abs() < 1e-4,
        "{update:?} landed at {x:?}, target {target:?}"
    );
    assert_eq!(mb_index(x[0], x[1]), 1, "{update:?} index at {x:?}");
    assert!(report.curvature < 0.0, "curvature {}", report.curvature);
}

#[test]
fn muller_brown_saddles() {
    // Literature saddles. The starts are displaced; the session has
    // to walk there. TS1 separates the left and middle basins, TS2
    // the middle and right.
    let ts1 = [-0.82200156, 0.62431280];
    let ts2 = [0.21248658, 0.29298833];
    for update in [HessianUpdate::Bofill, HessianUpdate::Powell] {
        find_muller_saddle(update, [-0.7, 0.5], ts1);
        find_muller_saddle(update, [0.3, 0.4], ts2);
    }
}
