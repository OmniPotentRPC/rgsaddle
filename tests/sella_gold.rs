//! Golden-master rgsaddle steppers and sessions against dest Sella gold.
//!
//! `tests/sella_manopt_gold.json` is dest `tests/sella_manopt_gold.json`
//! (zadorlab/sella `optimize/stepper.py`, `restricted_step.py`,
//! `hessian_update.py`) plus SellaMin / SellaSaddle one-step golds
//! minted from Sella Optimizer `rs=tr` on 1-atom Euclidean quadratics.
//! Remint lives next to dest (`tests/sella_manopt_gold.py`). These
//! tests load the frozen JSON only.

use ndarray::{Array1, Array2, ArrayView1, array};
use rgsaddle::{
    PartitionedRationalFunctionOptimization, PointSurface, QuasiNewton,
    RationalFunctionOptimization, SaddleError, SellaMinConfig, SellaMinSession, SellaSaddleConfig,
    SellaSaddleSession,
};

const GOLD: &str = include_str!("sella_manopt_gold.json");
const TOL: f64 = 1e-10;
/// Named tolerance for SellaMin / SellaSaddle one-step vs Sella Optimizer.
const SESSION_TOL: f64 = 1e-8;

fn case_slice(name: &str) -> &'static str {
    let key = format!("\"name\": \"{name}\"");
    let start = GOLD
        .find(&key)
        .unwrap_or_else(|| panic!("missing gold case {name}"));
    let rest = &GOLD[start..];
    let end = rest[1..]
        .find("\n    {")
        .or_else(|| rest.find("\n  ]"))
        .unwrap_or(rest.len());
    &rest[..end + 1]
}

fn json_nums(blob: &str, key: &str) -> Vec<f64> {
    let pat = format!("\"{key}\":");
    let start = blob
        .find(&pat)
        .unwrap_or_else(|| panic!("missing key {key}"));
    let after = &blob[start + pat.len()..];
    let lb = after
        .find('[')
        .unwrap_or_else(|| panic!("{key} is not an array"));
    let mut depth = 0;
    let mut rb = lb;
    for (i, c) in after[lb..].char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    rb = lb + i;
                    break;
                }
            }
            _ => {}
        }
    }
    after[lb + 1..rb]
        .split(|c: char| c == ',' || c == '[' || c == ']' || c.is_whitespace())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .unwrap_or_else(|e| panic!("{key} parse {s}: {e}"))
        })
        .collect()
}

fn json_num(blob: &str, key: &str) -> f64 {
    let pat = format!("\"{key}\": ");
    let start = blob
        .find(&pat)
        .unwrap_or_else(|| panic!("missing number {key}"));
    let after = &blob[start + pat.len()..];
    let end = after
        .find(|c: char| c == ',' || c == '\n' || c == '}')
        .unwrap_or(after.len());
    after[..end]
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("{key} parse: {e}"))
}

fn json_usize(blob: &str, key: &str) -> usize {
    json_num(blob, key) as usize
}

fn mat2(vals: &[f64]) -> Array2<f64> {
    array![[vals[0], vals[1]], [vals[2], vals[3]]]
}

fn vecn(vals: &[f64]) -> Array1<f64> {
    Array1::from(vals.to_vec())
}

fn eigh2(h: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let a = h[(0, 0)];
    let b = h[(0, 1)];
    let c = h[(1, 1)];
    let tr = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let l0 = 0.5 * (tr - disc);
    let l1 = 0.5 * (tr + disc);
    let mut v = Array2::<f64>::zeros((2, 2));
    if b.abs() < 1e-15 && (a - c).abs() < 1e-15 {
        v[(0, 0)] = 1.0;
        v[(1, 1)] = 1.0;
    } else if b.abs() >= (a - c).abs() {
        let n0 = ((l0 - c) * (l0 - c) + b * b).sqrt();
        v[(0, 0)] = (l0 - c) / n0;
        v[(1, 0)] = b / n0;
        let n1 = ((l1 - c) * (l1 - c) + b * b).sqrt();
        v[(0, 1)] = (l1 - c) / n1;
        v[(1, 1)] = b / n1;
    } else {
        let n0 = ((l0 - a) * (l0 - a) + b * b).sqrt();
        v[(0, 0)] = b / n0;
        v[(1, 0)] = (l0 - a) / n0;
        let n1 = ((l1 - a) * (l1 - a) + b * b).sqrt();
        v[(0, 1)] = b / n1;
        v[(1, 1)] = (l1 - a) / n1;
    }
    (array![l0, l1], v)
}

fn assert_close(got: &Array1<f64>, gold: &[f64], name: &str) {
    assert_eq!(got.len(), gold.len(), "{name} len");
    for i in 0..got.len() {
        let err = (got[i] - gold[i]).abs();
        assert!(
            err <= TOL,
            "{name}[{i}] dest={} gold={} err={err}",
            got[i],
            gold[i]
        );
    }
}

fn assert_close_tol(got: &Array1<f64>, gold: &[f64], name: &str, tol: f64) {
    assert_eq!(got.len(), gold.len(), "{name} len");
    for i in 0..got.len() {
        let err = (got[i] - gold[i]).abs();
        assert!(
            err <= tol,
            "{name}[{i}] dest={} gold={} err={err}",
            got[i],
            gold[i]
        );
    }
}

fn assert_scalar(got: f64, gold: f64, name: &str, tol: f64) {
    let err = (got - gold).abs();
    assert!(err <= tol, "{name} dest={got} gold={gold} err={err}");
}

#[test]
fn gold_json_names_sella_stepper() {
    assert!(GOLD.contains("sella/optimize/stepper.py"));
    assert!(GOLD.contains("sella/optimize/restricted_step.py"));
    assert!(GOLD.contains("sella/hessian_update.py"));
    assert!(GOLD.contains("\"kind\": \"rfo\""));
    assert!(GOLD.contains("\"kind\": \"qn\""));
    assert!(GOLD.contains("\"kind\": \"prfo\""));
    assert!(GOLD.contains("\"kind\": \"sella_min\""));
    assert!(GOLD.contains("\"kind\": \"sella_saddle\""));
}

#[test]
fn rgsaddle_rfo_matches_sella_gold() {
    for name in [
        "rfo_eye_order0_alpha0.25",
        "rfo_eye_order0_alpha0.5",
        "rfo_eye_order0_alpha1.0",
        "rfo_saddle_order1_alpha1",
    ] {
        let blob = case_slice(name);
        let h = mat2(&json_nums(blob, "H"));
        let g = vecn(&json_nums(blob, "g"));
        let dest = RationalFunctionOptimization::new(json_usize(blob, "order")).get_s(
            &h,
            &g,
            json_num(blob, "alpha"),
        );
        assert_close(&dest, &json_nums(blob, "s"), name);
    }
}

#[test]
fn rgsaddle_qn_matches_sella_gold() {
    for name in [
        "qn_saddle_order1_alpha0.0",
        "qn_saddle_order1_alpha0.3",
        "qn_saddle_order1_alpha1.0",
    ] {
        let blob = case_slice(name);
        let h = mat2(&json_nums(blob, "H"));
        let g = vecn(&json_nums(blob, "g"));
        let (evals, evecs) = eigh2(&h);
        let (dest, _) = QuasiNewton::new(&evals, &evecs, &g, json_usize(blob, "order"))
            .get_s(json_num(blob, "alpha"));
        assert_close(&dest, &json_nums(blob, "s"), name);
    }
}

#[test]
fn rgsaddle_prfo_matches_sella_gold() {
    let name = "prfo_saddle_order1_alpha1";
    let blob = case_slice(name);
    let h = mat2(&json_nums(blob, "H"));
    let g = vecn(&json_nums(blob, "g"));
    let (evals, evecs) = eigh2(&h);
    let dest = PartitionedRationalFunctionOptimization::new(json_usize(blob, "order")).get_s(
        &evals,
        &evecs,
        &g,
        json_num(blob, "alpha"),
    );
    assert_close(&dest, &json_nums(blob, "s"), name);
}

struct QuadMin;
impl PointSurface for QuadMin {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.5 * x.dot(&x), x.to_owned()))
    }
}

struct QuadSaddle;
impl PointSurface for QuadSaddle {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let e = 0.5 * (-x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        Ok((e, array![-x[0], x[1], x[2]]))
    }
}

#[test]
fn sella_min_onestep_matches_sella_optimizer() {
    let name = "sella_min_quad_onestep";
    let blob = case_slice(name);
    let x0 = vecn(&json_nums(blob, "x0"));
    let mut sess = SellaMinSession::new(
        SellaMinConfig {
            delta: json_num(blob, "delta0"),
            ..SellaMinConfig::default()
        },
        x0,
        Array1::from(vec![1.0]),
    )
    .unwrap();
    let report = sess.step(&QuadMin).unwrap();
    assert_close_tol(
        &sess.position().to_owned(),
        &json_nums(blob, "x1"),
        name,
        SESSION_TOL,
    );
    assert_scalar(
        report.energy,
        json_num(blob, "energy"),
        "sella_min.energy",
        SESSION_TOL,
    );
    assert_scalar(
        report.rho,
        json_num(blob, "rho"),
        "sella_min.rho",
        SESSION_TOL,
    );
    assert_scalar(
        report.delta,
        json_num(blob, "delta"),
        "sella_min.delta",
        SESSION_TOL,
    );
}

#[test]
fn sella_saddle_onestep_matches_sella_optimizer() {
    let name = "sella_saddle_quad_onestep";
    let blob = case_slice(name);
    let x0 = vecn(&json_nums(blob, "x0"));
    let mut sess = SellaSaddleSession::new(
        SellaSaddleConfig {
            delta: json_num(blob, "delta0"),
            eig: false,
            order: json_usize(blob, "order"),
            ..SellaSaddleConfig::default()
        },
        x0,
        Array1::from(vec![1.0]),
    )
    .unwrap();
    let report = sess.step(&QuadSaddle).unwrap();
    assert_close_tol(
        &sess.position().to_owned(),
        &json_nums(blob, "x1"),
        name,
        SESSION_TOL,
    );
    assert_scalar(
        report.energy,
        json_num(blob, "energy"),
        "sella_saddle.energy",
        SESSION_TOL,
    );
    assert_scalar(
        report.rho,
        json_num(blob, "rho"),
        "sella_saddle.rho",
        SESSION_TOL,
    );
    assert_scalar(
        report.delta,
        json_num(blob, "delta"),
        "sella_saddle.delta",
        SESSION_TOL,
    );
}
