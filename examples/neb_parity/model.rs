use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ndarray::{Array1, Array2, ArrayView2, s};
use rgsaddle::{
    BandConfig, BandSession, BandStatus, BandSurface, Cell as BandCell, CiConfig, MinModeRefine,
    SaddleError, SpringKind,
};

use crate::pot::{Pot, PotEval};

pub(crate) const TOL: f64 = 0.01;

pub(crate) const SLACK: f64 = 10.0;

pub(crate) const MAX_ITER: usize = 2000;

pub(crate) const K_SPRING: f64 = 5.0;

pub(crate) const MAX_MOVE: f64 = 0.2;

pub(crate) const TIME_UNIT: f64 = 10.1805055;

pub(crate) const DT: f64 = 1.0 / TIME_UNIT;

pub(crate) const DT_MAX: f64 = 2.5 / TIME_UNIT;

pub(crate) const LBFGS_MEMORY: usize = 20;

#[derive(Clone, Copy)]
pub(crate) enum Optim {
    Fire,
    Lbfgs,
}

impl Optim {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Optim::Fire => "fire",
            Optim::Lbfgs => "lbfgs",
        }
    }
}

pub(crate) struct Row {
    pub(crate) calls: usize,
    pub(crate) seconds: f64,
    pub(crate) ci: Option<usize>,
    pub(crate) converged: bool,
}

pub(crate) struct Problem {
    pub(crate) name: &'static str,
    pub(crate) path_kind: &'static str,
    pub(crate) held_out: bool,
    pub(crate) positions: Array2<f64>,
    pub(crate) cell: Option<BandCell>,
    /// False when a frozen slab means the band must keep the net force.
    pub(crate) translate: bool,
    pub(crate) pot: Pot,
}

pub(crate) fn lbfgs_direction(
    grad: &Array1<f64>,
    s_mem: &[Array1<f64>],
    y_mem: &[Array1<f64>],
    h0_empty: f64,
) -> Array1<f64> {
    let m = s_mem.len();
    let mut q = grad.clone();
    let mut alpha = vec![0.0; m];
    for i in (0..m).rev() {
        let sy = s_mem[i].dot(&y_mem[i]);
        let rho = if sy.abs() < 1e-30 { 0.0 } else { 1.0 / sy };
        alpha[i] = rho * s_mem[i].dot(&q);
        q = &q - &(&y_mem[i] * alpha[i]);
    }
    let h0 = if m == 0 {
        h0_empty
    } else {
        let sy = s_mem[m - 1].dot(&y_mem[m - 1]);
        let yy = y_mem[m - 1].dot(&y_mem[m - 1]);
        if sy > 0.0 && yy > 0.0 {
            sy / yy
        } else {
            h0_empty
        }
    };
    q *= h0;
    for i in 0..m {
        let sy = s_mem[i].dot(&y_mem[i]);
        let rho = if sy.abs() < 1e-30 { 0.0 } else { 1.0 / sy };
        let beta = rho * y_mem[i].dot(&q);
        q = &q + &(&s_mem[i] * (alpha[i] - beta));
    }
    -q
}

pub(crate) fn project_translation(force: &mut Array1<f64>) {
    let n = force.len() / 3;
    if n < 2 {
        return;
    }
    let mut mean = [0.0; 3];
    for atom in 0..n {
        for (c, m) in mean.iter_mut().enumerate() {
            *m += force[3 * atom + c];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    for atom in 0..n {
        for (c, m) in mean.iter().enumerate() {
            force[3 * atom + c] -= m;
        }
    }
}

pub(crate) fn cap_atoms(step: &mut Array1<f64>, max_move: f64) {
    let mut max = 0.0_f64;
    let n = step.len() / 3;
    for i in 0..n {
        let nrm = (step[3 * i].powi(2) + step[3 * i + 1].powi(2) + step[3 * i + 2].powi(2)).sqrt();
        max = max.max(nrm);
    }
    if max > max_move && max.is_finite() {
        *step *= max_move / max;
    }
}

pub(crate) fn image_norm(force: &Array1<f64>, dof: usize, image: usize) -> f64 {
    let row = force.slice(s![image * dof..(image + 1) * dof]);
    row.dot(&row).sqrt()
}

pub(crate) fn convergence(force: &Array1<f64>, dof: usize, ci: Option<usize>) -> f64 {
    let n = force.len() / dof;
    let mut band = 0.0_f64;
    for i in 0..n {
        band = band.max(image_norm(force, dof, i));
    }
    if let Some(index) = ci {
        let local = image_norm(force, dof, index - 1);
        if SLACK > 0.0 && band > SLACK * TOL {
            band
        } else {
            local
        }
    } else {
        band
    }
}

pub(crate) struct Counted<P> {
    pub(crate) pot: P,
    pub(crate) cell: Option<BandCell>,
    pub(crate) calls: AtomicUsize,
}

impl<P: PotEval + Sync> BandSurface for Counted<P> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        self.calls.fetch_add(positions.nrows(), Ordering::Relaxed);
        for (i, row) in positions.outer_iter().enumerate() {
            let (e, g) = self.pot.eval(row, self.cell);
            energies[i] = e;
            gradients.row_mut(i).assign(&g);
        }
        Ok(())
    }
}

pub(crate) fn band_config(case: &Problem, spring: SpringKind) -> BandConfig {
    BandConfig {
        spring,
        climbing: Some(CiConfig::always()),
        cell: case.cell,
        // The shared stop is the per-image norm below. A component
        // tolerance at the same number can report converged while that
        // norm is still above it, so the session is left running.
        force_tol: 1e-8,
        max_move: MAX_MOVE,
        remove_translation: case.translate,
        method: rgmin::Method::Fire {
            kind: rgmin::FireKind::V1,
        },
        time_step: DT,
        min_mode: None,
        ..BandConfig::default()
    }
}

pub(crate) fn run_rgsaddle(
    case: &Problem,
    optim: Optim,
    per_image: bool,
) -> Result<Row, SaddleError> {
    let mut config = band_config(case, SpringKind::Uniform { k: K_SPRING });
    config.per_image_fire = per_image;
    if matches!(optim, Optim::Lbfgs) {
        config.method = rgmin::Method::Lbfgs {
            memory: LBFGS_MEMORY,
        };
    }
    run_with(case, config)
}

pub(crate) fn run_tuned(case: &Problem, spring: SpringKind, oci: bool) -> Result<Row, SaddleError> {
    let mut config = band_config(case, spring);
    if oci {
        config.min_mode = Some(MinModeRefine::default());
    }
    run_with(case, config)
}

pub(crate) fn run_with(case: &Problem, config: BandConfig) -> Result<Row, SaddleError> {
    let surface = Counted {
        pot: case.pot.clone(),
        cell: case.cell,
        calls: AtomicUsize::new(0),
    };
    let mut session = BandSession::new(config, case.positions.clone())?;
    let started = Instant::now();
    let mut ci = None;
    let mut converged = false;
    for _ in 0..MAX_ITER {
        let report = session.step(&surface)?;
        ci = report.ci_index;
        let Some(ev) = session.evaluation() else {
            break;
        };
        let dof = case.positions.ncols();
        let flat: Array1<f64> = ev.projected.iter().copied().collect();
        let scalar = convergence(&flat, dof, report.ci_index);
        if scalar < TOL {
            converged = true;
            break;
        }
        if report.status == BandStatus::Converged {
            // The session's component tolerance can fire while the
            // per-image norm has not. Keep stepping only while the
            // shared eOn test is unmet; a zero step is a stall.
            break;
        }
    }
    Ok(Row {
        calls: surface.calls.load(Ordering::Relaxed),
        seconds: started.elapsed().as_secs_f64(),
        ci,
        converged,
    })
}

pub(crate) fn scatter(positions: &mut Array2<f64>, flat: &Array1<f64>) {
    let dof = positions.ncols();
    for i in 1..positions.nrows() - 1 {
        positions
            .row_mut(i)
            .assign(&flat.slice(s![(i - 1) * dof..i * dof]));
    }
}

pub(crate) fn gather(positions: &Array2<f64>) -> Array1<f64> {
    let n = positions.nrows();
    positions.slice(s![1..n - 1, ..]).iter().copied().collect()
}

pub(crate) fn max_atom(step: &Array1<f64>) -> f64 {
    let mut max = 0.0_f64;
    let n = step.len() / 3;
    for i in 0..n {
        let nrm = (step[3 * i].powi(2) + step[3 * i + 1].powi(2) + step[3 * i + 2].powi(2)).sqrt();
        max = max.max(nrm);
    }
    max
}
