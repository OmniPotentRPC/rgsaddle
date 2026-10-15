//! Index-1 refinement on an exact projected Hessian.
//!
//! One call to [`PointSurface::eval`] is a force evaluation. One call
//! to [`PointSurface::projected_hessian`] is a Hessian evaluation.
//! The Python driver in `rgsaddle.tsopt` is the same controller and is
//! what the OptBench entries run. This session is the Rust host.

use ndarray::{Array1, Array2, ArrayView1};

use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::nichols::sym_eigh;
use crate::prfo::{PrfoKind, partitioned_rfo_eigen};

const NEG_CUT: f64 = -1e-6;

/// Result of one index-1 refinement.
#[derive(Clone, Debug)]
pub struct RefineReport {
    pub x: Array1<f64>,
    pub force_calls: usize,
    pub hessian_calls: usize,
    pub converged: bool,
    pub index: i32,
    pub gradient_norm: f64,
}

impl RefineReport {
    pub fn calls(&self) -> usize {
        self.force_calls + self.hessian_calls
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefineKind {
    /// Fast Newton. Rejects a convex landing and a step that flattens
    /// a deep mode without cutting the gradient.
    Default,
    /// Curvature-preserving Newton, including a soft-mode step from a
    /// convex point.
    Ridge,
    /// [`Self::Default`], then [`Self::Ridge`] from the original start
    /// when the first search is not index 1.
    Tuned,
}

struct CapError;

fn l2(v: &Array1<f64>) -> f64 {
    v.dot(v).sqrt()
}

fn prfo_free(
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    gradient: &Array1<f64>,
    delta: f64,
) -> Result<Array1<f64>, SaddleError> {
    let g_eig = evecs.t().dot(gradient);
    let assemble = |alpha: f64| -> Result<Array1<f64>, SaddleError> {
        let coeff = partitioned_rfo_eigen(evals.view(), g_eig.view(), alpha, PrfoKind::Index1)?;
        Ok(evecs.dot(&coeff))
    };
    let step = assemble(1.0)?;
    let nrm = l2(&step);
    if nrm <= delta || delta <= 0.0 {
        return Ok(step);
    }
    let mut lo = 1.0;
    let mut hi = 2.0;
    let mut bracketed = false;
    let mut trial = step.clone();
    for _ in 0..30 {
        trial = assemble(hi)?;
        if l2(&trial) <= delta {
            bracketed = true;
            break;
        }
        hi *= 2.0;
    }
    if !bracketed {
        return Ok(if nrm > 0.0 {
            step * (delta / nrm)
        } else {
            step
        });
    }
    let mut best = trial;
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        trial = assemble(mid)?;
        if l2(&trial) > delta {
            lo = mid;
        } else {
            hi = mid;
            best = trial.clone();
        }
    }
    Ok(best)
}

fn neg_count(evals: &Array1<f64>) -> i32 {
    evals.iter().filter(|v| **v < NEG_CUT).count() as i32
}

fn finish(
    x: &Array1<f64>,
    g: &Array1<f64>,
    h: &Array2<f64>,
    force: usize,
    hess: usize,
    tol: f64,
) -> Result<RefineReport, SaddleError> {
    let (ev, _) = sym_eigh(h.view())?;
    let index = neg_count(&ev);
    let gradient_norm = l2(g);
    Ok(RefineReport {
        x: x.clone(),
        force_calls: force,
        hessian_calls: hess,
        converged: gradient_norm < tol && index == 1,
        index,
        gradient_norm,
    })
}

fn failed(x: &Array1<f64>, force: usize, hess: usize) -> RefineReport {
    RefineReport {
        x: x.clone(),
        force_calls: force,
        hessian_calls: hess,
        converged: false,
        index: -1,
        gradient_norm: 1.0e9,
    }
}

struct Counters {
    force: usize,
    hess: usize,
    cap: usize,
}

impl Counters {
    fn eval<S: PointSurface>(
        &mut self,
        surface: &S,
        x: ArrayView1<f64>,
    ) -> Result<(f64, Array1<f64>), Result<SaddleError, CapError>> {
        if self.force + self.hess >= self.cap {
            return Err(Err(CapError));
        }
        self.force += 1;
        surface.eval(x).map_err(|err| Ok(err))
    }

    fn hess<S: PointSurface>(
        &mut self,
        surface: &S,
        x: ArrayView1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), Result<SaddleError, CapError>> {
        if self.force + self.hess >= self.cap {
            return Err(Err(CapError));
        }
        self.hess += 1;
        match surface.projected_hessian(x).map_err(|err| Ok(err))? {
            Some(pair) => Ok(pair),
            None => Err(Ok(SaddleError::Invalid(
                "index-1 Newton needs projected_hessian".into(),
            ))),
        }
    }
}

fn refine_fast<S: PointSurface>(
    surface: &S,
    x0: Array1<f64>,
    cap: usize,
    force_tol: f64,
) -> Result<RefineReport, SaddleError> {
    let mut c = Counters {
        force: 0,
        hess: 0,
        cap,
    };
    let mut x = x0;
    let run = (|| {
        let (mut energy, mut g) = c.eval(surface, x.view())?;
        let (mut u, mut h) = c.hess(surface, x.view())?;
        let mut delta = 0.18;
        let mut rejects = 0usize;
        for _ in 0..80 {
            let gn = l2(&g);
            let (ev, vecs) = sym_eigh(h.view()).map_err(|err| Ok(err))?;
            let neg = neg_count(&ev);
            let ev0 = ev[0];
            if gn < force_tol && neg == 1 {
                break;
            }
            let gf = u.t().dot(&g);
            let mut limit: f64 = delta;
            if neg == 1 && gn < 1.5 {
                limit = limit.min((0.008_f64).max(0.28 * gn));
            }
            let step = if neg == 0 {
                let v = vecs.column(0).to_owned();
                let sign = if gf.dot(&v) >= 0.0 { 1.0 } else { -1.0 };
                let amp = 0.08_f64.min(limit.max(0.02));
                u.dot(&(v * sign * amp))
            } else {
                let free = prfo_free(&ev, &vecs, &gf, limit).map_err(|err| Ok(err))?;
                u.dot(&free)
            };
            let nrm = l2(&step);
            if nrm < 1e-14 {
                delta = (delta * 0.5).max(1e-4);
                rejects += 1;
                if rejects > 8 {
                    break;
                }
                continue;
            }
            let pred = g.dot(&step) + 0.5 * step.dot(&u.dot(&h.dot(&u.t().dot(&step))));
            let x1 = &x + &step;
            let (e1, g1) = c.eval(surface, x1.view())?;
            let g1n = l2(&g1);
            let rho = if pred.abs() >= 1e-14 {
                (e1 - energy) / pred
            } else {
                1.0
            };
            let rho_ok = (0.15..=6.0).contains(&rho) || (neg == 0 && g1n < gn);
            if !rho_ok {
                delta = delta.min(nrm) * 0.5;
                delta = delta.max(1e-4);
                rejects += 1;
                if rejects > 8 {
                    break;
                }
                continue;
            }
            let (u1, h1) = c.hess(surface, x1.view())?;
            let (ev1, _) = sym_eigh(h1.view()).map_err(|err| Ok(err))?;
            let neg1 = neg_count(&ev1);
            let ev1_0 = ev1[0];
            let collapsed = neg >= 1 && ev0 < -0.25 && ev1_0 > 0.35 * ev0 && g1n > 0.75 * gn;
            let lost = neg1 == 0 && g1n >= force_tol;
            if collapsed || lost {
                delta = delta.min(nrm) * 0.45;
                delta = delta.max(1e-4);
                rejects += 1;
                if rejects > 8 {
                    break;
                }
                continue;
            }
            x = x1;
            energy = e1;
            g = g1;
            u = u1;
            h = h1;
            rejects = 0;
            if (0.65..1.45).contains(&rho)
                && nrm > 0.7 * limit
                && ev1_0 < (-0.2_f64).min(0.85 * ev0)
            {
                delta = delta.max(nrm) * 1.25;
                delta = delta.min(0.4);
            } else if !(0.45..=2.2).contains(&rho) {
                delta = delta.min(nrm) * 0.8;
                delta = delta.max(1e-4);
            }
        }
        finish(&x, &g, &h, c.force, c.hess, force_tol).map_err(|err| Ok(err))
    })();
    match run {
        Ok(report) => Ok(report),
        Err(Ok(err)) => Err(err),
        Err(Err(CapError)) => Ok(failed(&x, c.force, c.hess)),
    }
}

fn pure_mode(
    u: &Array2<f64>,
    vecs: &Array2<f64>,
    gf: &Array1<f64>,
    ev0: f64,
    limit: f64,
) -> Result<Array1<f64>, SaddleError> {
    let g0 = gf.dot(&vecs.column(0));
    let disc = (ev0 * ev0 + 4.0 * g0 * g0).sqrt();
    let mu = 0.5 * (ev0 + disc);
    let denom = mu - ev0;
    let coeff = if g0.abs() < 1e-16 || denom.abs() < 1e-18 {
        0.0
    } else {
        g0 / denom
    };
    let mut pure = u.dot(&(vecs.column(0).to_owned() * coeff));
    let pn = l2(&pure);
    if pn > limit && limit > 0.0 {
        pure *= limit / pn;
    }
    Ok(pure)
}

fn refine_ridge<S: PointSurface>(
    surface: &S,
    x0: Array1<f64>,
    cap: usize,
    force_tol: f64,
) -> Result<RefineReport, SaddleError> {
    let mut c = Counters {
        force: 0,
        hess: 0,
        cap,
    };
    let mut x = x0;
    let run = (|| {
        let (mut energy, mut g) = c.eval(surface, x.view())?;
        let (mut u, mut h) = c.hess(surface, x.view())?;
        let mut delta = 0.18;
        for _ in 0..80 {
            let gn = l2(&g);
            let (ev, vecs) = sym_eigh(h.view()).map_err(|err| Ok(err))?;
            let neg = neg_count(&ev);
            let ev0 = ev[0];
            if gn < force_tol && neg == 1 {
                break;
            }
            let gf = u.t().dot(&g);
            let mut limit: f64 = delta;
            if neg == 1 && gn < 1.5 {
                limit = limit.min((0.01_f64).max(0.3 * gn));
            }
            let (step, pure) = if neg == 0 {
                let free = prfo_free(&ev, &vecs, &gf, limit.max(0.08)).map_err(|err| Ok(err))?;
                let step = u.dot(&free);
                let v = vecs.column(0).to_owned();
                let sign = if gf.dot(&v) >= 0.0 { 1.0 } else { -1.0 };
                let pure = u.dot(&(v * sign * 0.12_f64.min(limit.max(0.04))));
                (step, pure)
            } else {
                let free = prfo_free(&ev, &vecs, &gf, limit).map_err(|err| Ok(err))?;
                let step = u.dot(&free);
                let pure = pure_mode(&u, &vecs, &gf, ev0, limit).map_err(|err| Ok(err))?;
                (step, pure)
            };
            let mut found: Option<(
                Array1<f64>,
                f64,
                Array1<f64>,
                Array2<f64>,
                Array2<f64>,
                f64,
                f64,
            )> = None;
            for cand in [&step, &pure] {
                for scale in [1.0, 0.5, 0.25, 0.125] {
                    let trial = cand * scale;
                    let nrm = l2(&trial);
                    if nrm < 1e-10 {
                        continue;
                    }
                    let pred = g.dot(&trial) + 0.5 * trial.dot(&u.dot(&h.dot(&u.t().dot(&trial))));
                    let x1 = &x + &trial;
                    let (e1, g1) = c.eval(surface, x1.view())?;
                    let g1n = l2(&g1);
                    let rho = if pred.abs() >= 1e-14 {
                        (e1 - energy) / pred
                    } else {
                        1.0
                    };
                    let rho_ok = (0.1..=8.0).contains(&rho) || g1n < gn;
                    if !rho_ok {
                        continue;
                    }
                    let (u1, h1) = c.hess(surface, x1.view())?;
                    let (ev1, _) = sym_eigh(h1.view()).map_err(|err| Ok(err))?;
                    let neg1 = neg_count(&ev1);
                    let ev1_0 = ev1[0];
                    let accept = if g1n < force_tol && neg1 == 1 {
                        true
                    } else if neg1 < 1 {
                        false
                    } else if ev0 < -0.15 {
                        ev1_0 < 0.6 * ev0
                    } else {
                        g1n <= gn * 0.995
                    };
                    if accept {
                        found = Some((trial, e1, g1, u1, h1, rho, nrm));
                        break;
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            let Some((trial, e1, g1, u1, h1, rho, nrm)) = found else {
                delta = (delta * 0.5).max(1e-3);
                if delta <= 1e-3 && neg == 0 {
                    break;
                }
                continue;
            };
            x = &x + &trial;
            energy = e1;
            g = g1;
            u = u1;
            h = h1;
            let (ev_now, _) = sym_eigh(h.view()).map_err(|err| Ok(err))?;
            let ev1_0 = ev_now[0];
            if (0.6..1.5).contains(&rho) && ev1_0 < (-0.3_f64).min(0.9 * ev0) {
                delta = delta.max(nrm) * 1.2;
                delta = delta.min(0.35);
            } else if !(0.4..=2.5).contains(&rho) {
                delta = delta.min(nrm).max(1e-3);
            }
        }
        finish(&x, &g, &h, c.force, c.hess, force_tol).map_err(|err| Ok(err))
    })();
    match run {
        Ok(report) => Ok(report),
        Err(Ok(err)) => Err(err),
        Err(Err(CapError)) => Ok(failed(&x, c.force, c.hess)),
    }
}

/// Refine `x0` to an index-1 saddle.
pub fn refine<S: PointSurface>(
    surface: &S,
    x0: Array1<f64>,
    kind: RefineKind,
    force_tol: f64,
) -> Result<RefineReport, SaddleError> {
    if !(force_tol.is_finite() && force_tol >= 0.0) {
        return Err(SaddleError::Invalid(
            "force tolerance must be finite and non-negative".into(),
        ));
    }
    match kind {
        RefineKind::Default => refine_fast(surface, x0, 80, force_tol),
        RefineKind::Ridge => refine_ridge(surface, x0, 160, force_tol),
        RefineKind::Tuned => {
            let first = refine_fast(surface, x0.clone(), 48, force_tol)?;
            if first.converged {
                return Ok(first);
            }
            let second = refine_ridge(surface, x0, 160, force_tol)?;
            Ok(RefineReport {
                x: second.x,
                force_calls: first.force_calls + second.force_calls,
                hessian_calls: first.hessian_calls + second.hessian_calls,
                converged: second.converged,
                index: second.index,
                gradient_norm: second.gradient_norm,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    struct Quadratic;

    impl PointSurface for Quadratic {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let energy = -x[0] * x[0] + x[1] * x[1];
            Ok((energy, array![-2.0 * x[0], 2.0 * x[1]]))
        }

        fn projected_hessian(
            &self,
            x: ArrayView1<f64>,
        ) -> Result<Option<(Array2<f64>, Array2<f64>)>, SaddleError> {
            let n = x.len();
            Ok(Some((Array2::eye(n), array![[-2.0, 0.0], [0.0, 2.0]])))
        }
    }

    #[test]
    fn quadratic_saddle_is_index_1() {
        let report = refine(&Quadratic, array![0.4, -0.3], RefineKind::Default, 1e-3).unwrap();
        assert!(report.converged, "{report:?}");
        assert_eq!(report.index, 1);
        assert!(report.gradient_norm < 1e-3);
        assert!(report.calls() < 30);
        assert!(report.x[0].abs() < 1e-3);
        assert!(report.x[1].abs() < 1e-3);
    }
}
