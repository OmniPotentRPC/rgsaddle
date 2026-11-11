use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, s};
use rgsaddle::rtr::band_forces;
use rgsaddle::{SaddleError, SpringKind};

use crate::model::{
    Counted, DT, DT_MAX, K_SPRING, LBFGS_MEMORY, MAX_ITER, MAX_MOVE, Optim, Problem, Row, TOL,
    band_config, cap_atoms, convergence, gather, lbfgs_direction, max_atom, scatter,
};

pub(crate) fn assemble_eon(
    case: &Problem,
    positions: &Array2<f64>,
    prev_ci: Option<usize>,
    calls: &mut usize,
) -> Result<(Array1<f64>, Array1<f64>, Option<usize>), SaddleError> {
    let surface = Counted {
        pot: case.pot.clone(),
        cell: case.cell,
        calls: AtomicUsize::new(0),
    };
    // Probe energies without committing the climbing image, then again
    // with the latched image. The second call is free when the surface
    // result is reused by evaluating once and projecting locally.
    let config = band_config(case, SpringKind::Uniform { k: K_SPRING });
    let probed = band_forces(&config, &surface, positions.view(), None)?;
    let ci = eon_ci(&probed.energies, prev_ci);
    let full = if ci.is_some() {
        band_forces(&config, &surface, positions.view(), ci)?
    } else {
        probed
    };
    let rows = surface.calls.load(Ordering::Relaxed);
    let n = positions.nrows();
    // band_forces evaluates every image. A second call with the climbing
    // image set repeats the same positions, so it is one potential
    // evaluation. Endpoints are charged only on the first assembly.
    let assemblies = if ci.is_some() { 2 } else { 1 };
    let once = rows / assemblies;
    let charged = if *calls == 0 {
        once
    } else {
        once.saturating_sub(2.min(n))
    };
    *calls += charged;
    Ok((full.force, full.energies, ci))
}

pub(crate) fn eon_ci(energies: &Array1<f64>, prev: Option<usize>) -> Option<usize> {
    let n = energies.len();
    if n < 3 {
        return None;
    }
    let mut max_i = 1usize;
    for i in 2..n - 1 {
        if energies[i] > energies[max_i] {
            max_i = i;
        }
    }
    let higher_end = energies[0].max(energies[n - 1]);
    if energies[max_i] <= higher_end {
        return None;
    }
    if let Some(prev) = prev
        && prev > 0
        && prev + 1 < n
        && energies[prev] > energies[prev - 1]
        && energies[prev] > energies[prev + 1]
    {
        return Some(prev);
    }
    Some(max_i)
}

pub(crate) fn run_eon(case: &Problem, optim: Optim) -> Result<Row, SaddleError> {
    let started = Instant::now();
    let mut calls = 0usize;
    let (force, energies, ci) = assemble_eon(case, &case.positions, None, &mut calls)?;
    let mut positions = case.positions.clone();
    let mut force = force;
    let mut energies = energies;
    let mut ci = ci;
    let dof = positions.ncols();
    let n_int = positions.nrows() - 2;
    if convergence(&force, dof, ci) < TOL {
        return Ok(Row {
            calls,
            seconds: started.elapsed().as_secs_f64(),
            ci,
            converged: true,
        });
    }
    match optim {
        Optim::Fire => eon_fire(
            case,
            &mut positions,
            &mut force,
            &mut energies,
            &mut ci,
            &mut calls,
            n_int,
            dof,
        )?,
        Optim::Lbfgs => eon_lbfgs(
            case,
            &mut positions,
            &mut force,
            &mut energies,
            &mut ci,
            &mut calls,
            n_int,
            dof,
        )?,
    }
    Ok(Row {
        calls,
        seconds: started.elapsed().as_secs_f64(),
        ci,
        converged: true,
    })
}

pub(crate) fn eon_fire(
    case: &Problem,
    positions: &mut Array2<f64>,
    force: &mut Array1<f64>,
    energies: &mut Array1<f64>,
    ci: &mut Option<usize>,
    calls: &mut usize,
    _n_int: usize,
    dof: usize,
) -> Result<(), SaddleError> {
    let mut vel = Array1::zeros(force.len());
    let mut dt = DT;
    let mut alpha = 0.1;
    let mut n_pos = 0usize;
    for _ in 0..MAX_ITER {
        if dt < 1e-6 {
            return Err(SaddleError::Stalled(format!(
                "fire dt {dt:e} collapsed, calls={calls}"
            )));
        }
        vel = &vel + &*force * dt;
        let mut dx = &vel * dt;
        cap_atoms(&mut dx, MAX_MOVE);
        let mut x = gather(positions);
        x += &dx;
        scatter(positions, &x);
        let (f, e, next) = assemble_eon(case, positions, *ci, calls)?;
        *force = f;
        *energies = e;
        *ci = next;
        let fnorm = force.dot(force).sqrt();
        let power = force.dot(&vel);
        if fnorm > 0.0 {
            let vnorm = vel.dot(&vel).sqrt();
            vel = &vel * (1.0 - alpha) + &*force * (alpha * vnorm / fnorm);
        }
        if power >= 0.0 {
            n_pos += 1;
            if n_pos > 5 {
                dt = (dt * 1.1).min(DT_MAX);
                alpha *= 0.99;
            }
        } else {
            dt *= 0.5;
            vel.fill(0.0);
            alpha = 0.1;
            n_pos = 0;
        }
        if convergence(force, dof, *ci) < TOL {
            return Ok(());
        }
    }
    Err(SaddleError::Stalled(format!(
        "fire not under {TOL}, calls={calls}, last {:.4e}, ci={ci:?}",
        convergence(force, dof, *ci)
    )))
}

pub(crate) fn eon_lbfgs(
    case: &Problem,
    positions: &mut Array2<f64>,
    force: &mut Array1<f64>,
    energies: &mut Array1<f64>,
    ci: &mut Option<usize>,
    calls: &mut usize,
    _n_int: usize,
    dof: usize,
) -> Result<(), SaddleError> {
    let mut s_mem: Vec<Array1<f64>> = Vec::new();
    let mut y_mem: Vec<Array1<f64>> = Vec::new();
    let mut x_prev = gather(positions);
    let mut f_prev = force.clone();
    for _ in 0..MAX_ITER {
        let x = gather(positions);
        let mut negative = false;
        if x != x_prev {
            let s = &x - &x_prev;
            let y = &f_prev - &*force;
            let sy = s.dot(&y);
            if sy < 0.0 || sy.abs() < 1e-30 {
                // eOn curvature = reset: drop the memory and step along
                // the force, capped at the per-atom maximum.
                s_mem.clear();
                y_mem.clear();
                negative = sy < 0.0;
            } else {
                s_mem.push(s);
                y_mem.push(y);
                if s_mem.len() > LBFGS_MEMORY {
                    s_mem.remove(0);
                    y_mem.remove(0);
                }
            }
        }
        let grad = -&*force;
        let mut direction = if negative {
            let mut d = force.clone();
            cap_atoms(&mut d, MAX_MOVE);
            d
        } else {
            let mut d = lbfgs_direction(&grad, &s_mem, &y_mem, 0.01);
            let too_far = max_atom(&d) >= MAX_MOVE;
            let uphill = d.dot(&grad) >= 0.0;
            if too_far || uphill || d.iter().any(|v| !v.is_finite()) {
                s_mem.clear();
                y_mem.clear();
                d = &*force * 0.01;
                cap_atoms(&mut d, MAX_MOVE);
            }
            d
        };
        if s_mem.is_empty() && max_atom(&direction) > MAX_MOVE {
            cap_atoms(&mut direction, MAX_MOVE);
        }
        let e0: f64 = energies.slice(s![1..energies.len() - 1]).sum();
        let mut alpha = 1.0;
        let mut accepted = false;
        let saved = positions.clone();
        let saved_f = force.clone();
        let saved_e = energies.clone();
        let saved_ci = *ci;
        for _ in 0..10 {
            let trial = &x + &direction * alpha;
            scatter(positions, &trial);
            let (f, e, next) = assemble_eon(case, positions, *ci, calls)?;
            let e_acc: f64 = e.slice(s![1..e.len() - 1]).sum();
            if e_acc - e0 <= 1e-8 {
                *force = f;
                *energies = e;
                *ci = next;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            *positions = saved;
            *force = saved_f;
            *energies = saved_e;
            *ci = saved_ci;
            s_mem.clear();
            y_mem.clear();
            let mut fallback = &*force * 0.1;
            cap_atoms(&mut fallback, MAX_MOVE);
            let trial = &x + &fallback;
            scatter(positions, &trial);
            let (f, e, next) = assemble_eon(case, positions, *ci, calls)?;
            *force = f;
            *energies = e;
            *ci = next;
        }
        x_prev = x;
        f_prev = force.clone();
        if convergence(force, dof, *ci) < TOL {
            return Ok(());
        }
    }
    Err(SaddleError::Stalled(format!(
        "lbfgs not under {TOL}, calls={calls}"
    )))
}
