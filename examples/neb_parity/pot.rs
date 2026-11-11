use ndarray::{Array1, ArrayView1};

use rgsaddle::Cell as BandCell;

pub(crate) trait PotEval: Clone {
    fn eval(&self, x: ArrayView1<f64>, cell: Option<BandCell>) -> (f64, Array1<f64>);
}

#[derive(Clone)]
pub(crate) enum Pot {
    Lj,
    LjVacancy { length: f64 },
    LjCrystal { length: f64, cutoff: f64 },
    MorseSlab { slab: Vec<[f64; 3]> },
}

impl PotEval for Pot {
    fn eval(&self, x: ArrayView1<f64>, _cell: Option<BandCell>) -> (f64, Array1<f64>) {
        match self {
            Pot::Lj => lennard_jones(x, None, None),
            Pot::LjVacancy { length } => lennard_jones(x, Some(*length), None),
            Pot::LjCrystal { length, cutoff } => lennard_jones(x, Some(*length), Some(*cutoff)),
            Pot::MorseSlab { slab } => morse_with_slab(x, slab),
        }
    }
}

pub(crate) fn lennard_jones(
    x: ArrayView1<f64>,
    box_length: Option<f64>,
    cutoff: Option<f64>,
) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    for a in 0..n {
        for b in a + 1..n {
            let mut d = [0.0; 3];
            for k in 0..3 {
                d[k] = x[3 * b + k] - x[3 * a + k];
                if let Some(l) = box_length {
                    d[k] -= l * (d[k] / l).round();
                }
            }
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if r2 < 1e-24 {
                continue;
            }
            if cutoff.is_some_and(|cut| r2 >= cut * cut) {
                continue;
            }
            let r = r2.sqrt();
            let inv_r2 = 1.0 / r2;
            let inv_r6 = inv_r2 * inv_r2 * inv_r2;
            let inv_r12 = inv_r6 * inv_r6;
            let mut pair = 4.0 * (inv_r12 - inv_r6);
            if let Some(cut) = cutoff {
                let inv_c2 = 1.0 / (cut * cut);
                let inv_c6 = inv_c2 * inv_c2 * inv_c2;
                pair -= 4.0 * (inv_c6 * inv_c6 - inv_c6);
            }
            energy += pair;
            let de_dr = 24.0 * (inv_r6 - 2.0 * inv_r12) / r;
            let scale = de_dr / r;
            for k in 0..3 {
                grad[3 * a + k] -= scale * d[k];
                grad[3 * b + k] += scale * d[k];
            }
        }
    }
    (energy, grad)
}

pub(crate) fn morse_pair(r: f64) -> (f64, f64) {
    // Platinum Morse used as a stand-in where an EAM file is not linked:
    // D = 0.7102 eV, alpha = 1.6047 / Å, r0 = 2.897 Å, cutoff 9.5 Å,
    // energy and force shifted to zero at the cutoff.
    const D: f64 = 0.7102;
    const ALPHA: f64 = 1.6047;
    const R0: f64 = 2.897;
    const CUT: f64 = 9.5;
    if r >= CUT {
        return (0.0, 0.0);
    }
    let raw = |q: f64| {
        let x = ALPHA * (q - R0);
        D * ((-2.0 * x).exp() - 2.0 * (-x).exp())
    };
    let draw = |q: f64| {
        let x = ALPHA * (q - R0);
        D * ALPHA * (-2.0 * (-2.0 * x).exp() + 2.0 * (-x).exp())
    };
    (
        raw(r) - raw(CUT) - (r - CUT) * draw(CUT),
        draw(r) - draw(CUT),
    )
}

pub(crate) fn morse_with_slab(x: ArrayView1<f64>, slab: &[[f64; 3]]) -> (f64, Array1<f64>) {
    let n = x.len() / 3;
    let mut energy = 0.0;
    let mut grad = Array1::zeros(x.len());
    let pos = |i: usize| [x[3 * i], x[3 * i + 1], x[3 * i + 2]];
    for a in 0..n {
        for b in a + 1..n {
            contribute_morse(&mut energy, &mut grad, a, b, pos(a), pos(b));
        }
        for site in slab {
            contribute_morse_frozen(&mut energy, &mut grad, a, pos(a), *site);
        }
    }
    (energy, grad)
}

pub(crate) fn contribute_morse(
    energy: &mut f64,
    grad: &mut Array1<f64>,
    a: usize,
    b: usize,
    pa: [f64; 3],
    pb: [f64; 3],
) {
    let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r < 1e-12 {
        return;
    }
    let (e, de) = morse_pair(r);
    *energy += e;
    let scale = de / r;
    for k in 0..3 {
        grad[3 * a + k] -= scale * d[k];
        grad[3 * b + k] += scale * d[k];
    }
}

pub(crate) fn contribute_morse_frozen(
    energy: &mut f64,
    grad: &mut Array1<f64>,
    a: usize,
    pa: [f64; 3],
    site: [f64; 3],
) {
    let d = [site[0] - pa[0], site[1] - pa[1], site[2] - pa[2]];
    let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if r < 1e-12 {
        return;
    }
    let (e, de) = morse_pair(r);
    *energy += e;
    let scale = de / r;
    for k in 0..3 {
        grad[3 * a + k] -= scale * d[k];
    }
}
