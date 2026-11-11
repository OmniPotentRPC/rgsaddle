use ndarray::{Array1, Array2, ArrayView1};

use rgsaddle::{
    Cell as BandCell, IdppConfig, SaddleError, idpp_path, interpolated_path, linear_path,
};

use crate::model::{Problem, project_translation};
use crate::pot::{Pot, PotEval};

pub(crate) fn build_cases() -> Result<Vec<Problem>, SaddleError> {
    let mut cases = Vec::new();
    let (lj7_r, mut lj7_p) = lj7_endpoints()?;
    align_atoms(&lj7_r, &mut lj7_p);
    cases.push(make("lj7", "linear", false, &lj7_r, &lj7_p, None, Pot::Lj)?);
    cases.push(make(
        "lj7-idpp",
        "idpp",
        false,
        &lj7_r,
        &lj7_p,
        None,
        Pot::Lj,
    )?);
    let (vac_r, vac_p, length) = vacancy_endpoints();
    let band_cell = BandCell([[length, 0.0, 0.0], [0.0, length, 0.0], [0.0, 0.0, length]]);
    let mic = linkcell::Cell::ortho(length, length, length)
        .map_err(|e| SaddleError::Invalid(e.to_string()))?;
    let vac_path = linear_path(vac_r.view(), vac_p.view(), 7, Some(&mic))?;
    cases.push(Problem {
        name: "lj-vacancy",
        path_kind: "linear",
        held_out: false,
        positions: relax_ends(vac_path, Pot::LjVacancy { length }, Some(band_cell))?,
        cell: Some(band_cell),
        translate: true,
        pot: Pot::LjVacancy { length },
    });
    let (ad_r, ad_p, slab) = morse_adatom()?;
    let mut adatom = make(
        "morse-adatom",
        "linear",
        false,
        &ad_r,
        &ad_p,
        None,
        Pot::MorseSlab { slab: slab.clone() },
    )?;
    adatom.translate = false;
    cases.push(adatom);
    let (hep_r, hep_p, hep_slab) = morse_heptamer()?;
    let mut heptamer = make(
        "morse-heptamer",
        "linear",
        false,
        &hep_r,
        &hep_p,
        None,
        Pot::MorseSlab { slab: hep_slab },
    )?;
    heptamer.translate = false;
    cases.push(heptamer);
    let (lj13_r, lj13_p) = lj13_endpoints();
    // Held out: the eOn neb_lj13 pair, not used to choose a default.
    cases.push(make(
        "lj13",
        "linear",
        true,
        &lj13_r,
        &lj13_p,
        None,
        Pot::Lj,
    )?);
    for case in &cases {
        let (e0, g0) = case.pot.eval(case.positions.row(0), case.cell);
        let last = case.positions.nrows() - 1;
        let (e1, g1) = case.pot.eval(case.positions.row(last), case.cell);
        let mut peak = f64::NEG_INFINITY;
        for i in 1..last {
            let (e, _) = case.pot.eval(case.positions.row(i), case.cell);
            peak = peak.max(e);
        }
        eprintln!(
            "setup {name}: ends {e0:.6} {e1:.6} peak {peak:.6} endforces {f0:.3e} {f1:.3e}",
            name = case.name,
            f0 = g0.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
            f1 = g1.iter().map(|v| v.abs()).fold(0.0_f64, f64::max),
        );
    }
    Ok(cases)
}

pub(crate) fn make(
    name: &'static str,
    path_kind: &'static str,
    held_out: bool,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: Option<BandCell>,
    pot: Pot,
) -> Result<Problem, SaddleError> {
    let (positions, path_kind) = if path_kind == "idpp" {
        (
            idpp_path(
                reactant.view(),
                product.view(),
                7,
                None,
                &IdppConfig::default(),
            )?,
            path_kind,
        )
    } else {
        let built = interpolated_path(
            reactant.view(),
            product.view(),
            7,
            None,
            &IdppConfig::default(),
        )?;
        let kind = if built.idpp { "idpp-fb" } else { path_kind };
        (built.positions, kind)
    };
    let positions = relax_ends(positions, pot.clone(), cell)?;
    Ok(Problem {
        name,
        path_kind,
        held_out,
        positions,
        cell,
        translate: true,
        pot,
    })
}

pub(crate) fn relax_ends(
    mut path: Array2<f64>,
    pot: Pot,
    cell: Option<BandCell>,
) -> Result<Array2<f64>, SaddleError> {
    for end in [0, path.nrows() - 1] {
        let mut x = path.row(end).to_owned();
        minimize(&mut x, &pot, cell)?;
        path.row_mut(end).assign(&x);
    }
    Ok(path)
}

pub(crate) fn minimize(
    x: &mut Array1<f64>,
    pot: &Pot,
    cell: Option<BandCell>,
) -> Result<(), SaddleError> {
    // Endpoint preparation only. These potential calls are not band calls.
    let pot_oracle = pot.clone();
    let mut solver = rgmin::Solver::new(
        rgmin::Method::Lbfgs { memory: 10 },
        rgmin::Control {
            gtol: 1e-6,
            maxiter: 2000,
            maxmove: Some(0.15),
            ..rgmin::Control::default()
        },
        x.len(),
    );
    solver.set_accept(rgmin::Accept::Energy);
    solver.set_atom_maxmove(0.15);
    let oracle = rgmin::Oracle::unbounded(x.len(), move |q: ArrayView1<f64>| {
        let (energy, mut grad) = pot_oracle.eval(q, cell);
        if !matches!(pot_oracle, Pot::MorseSlab { .. }) {
            project_translation(&mut grad);
        }
        (energy, grad)
    });
    for _ in 0..800 {
        solver
            .step(&oracle, x)
            .map_err(|err| SaddleError::Solver(err.to_string()))?;
        let (_, mut grad) = pot.eval(x.view(), cell);
        if !matches!(pot, Pot::MorseSlab { .. }) {
            project_translation(&mut grad);
        }
        if grad.iter().map(|v| v.abs()).fold(0.0_f64, f64::max) < 1e-4 {
            break;
        }
    }
    Ok(())
}

/// Greedy nearest-neighbour matching so a linear path does not drag
/// unrelated atoms through each other.
pub(crate) fn align_atoms(reactant: &Array1<f64>, product: &mut Array1<f64>) {
    let n = reactant.len() / 3;
    let mut used = vec![false; n];
    let mut matched = Array1::zeros(reactant.len());
    for i in 0..n {
        let mut best = 0usize;
        let mut best_d = f64::INFINITY;
        for j in 0..n {
            if used[j] {
                continue;
            }
            let mut d = 0.0;
            for k in 0..3 {
                let delta = reactant[3 * i + k] - product[3 * j + k];
                d += delta * delta;
            }
            if d < best_d {
                best_d = d;
                best = j;
            }
        }
        used[best] = true;
        for k in 0..3 {
            matched[3 * i + k] = product[3 * best + k];
        }
    }
    product.assign(&matched);
}

pub(crate) fn lj7_endpoints() -> Result<(Array1<f64>, Array1<f64>), SaddleError> {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let radius = nn / (2.0 * (std::f64::consts::PI / 5.0).sin());
    let mut penta = Array1::zeros(21);
    for k in 0..5 {
        let ang = 2.0 * std::f64::consts::PI * k as f64 / 5.0;
        penta[3 * k] = radius * ang.cos();
        penta[3 * k + 1] = radius * ang.sin();
    }
    let h = (nn * nn - radius * radius).max(0.2).sqrt();
    penta[15] = 0.0;
    penta[16] = 0.0;
    penta[17] = h;
    penta[18] = 0.0;
    penta[19] = 0.0;
    penta[20] = -h;
    let a = nn / 2.0_f64.sqrt();
    let mut octa = Array1::zeros(21);
    let axis = [
        [a, 0.0, 0.0],
        [-a, 0.0, 0.0],
        [0.0, a, 0.0],
        [0.0, -a, 0.0],
        [0.0, 0.0, a],
        [0.0, 0.0, -a],
    ];
    for (i, p) in axis.iter().enumerate() {
        octa[3 * i] = p[0];
        octa[3 * i + 1] = p[1];
        octa[3 * i + 2] = p[2];
    }
    octa[18] = a;
    octa[19] = a;
    octa[20] = a;
    minimize(&mut penta, &Pot::Lj, None)?;
    minimize(&mut octa, &Pot::Lj, None)?;
    Ok((penta, octa))
}

pub(crate) fn vacancy_endpoints() -> (Array1<f64>, Array1<f64>, f64) {
    let nn = 2.0_f64.powf(1.0 / 6.0);
    let a = nn * 2.0_f64.sqrt();
    let length = 2.0 * a;
    let basis = [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
    ];
    let mut sites = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                for b in basis {
                    sites.push([
                        a * (i as f64 + b[0]),
                        a * (j as f64 + b[1]),
                        a * (k as f64 + b[2]),
                    ]);
                }
            }
        }
    }
    let vacancy = 0usize;
    let hopper = 1usize;
    let mut reactant = Array1::zeros(3 * (sites.len() - 1));
    let mut product = reactant.clone();
    let mut slot = 0;
    for (index, site) in sites.iter().enumerate() {
        if index == vacancy {
            continue;
        }
        for k in 0..3 {
            reactant[3 * slot + k] = site[k];
            product[3 * slot + k] = if index == hopper {
                sites[vacancy][k]
            } else {
                site[k]
            };
        }
        slot += 1;
    }
    (reactant, product, length)
}

pub(crate) fn morse_adatom() -> Result<(Array1<f64>, Array1<f64>, Vec<[f64; 3]>), SaddleError> {
    let nn = 2.897;
    let a1 = [nn, 0.0, 0.0];
    let a2 = [nn / 2.0, nn * 3.0_f64.sqrt() / 2.0, 0.0];
    let mut slab = Vec::new();
    for i in 0..5 {
        for j in 0..5 {
            slab.push([
                i as f64 * a1[0] + j as f64 * a2[0],
                i as f64 * a1[1] + j as f64 * a2[1],
                0.0,
            ]);
        }
    }
    let hollow = |i: f64, j: f64| {
        let h = nn * (2.0_f64 / 3.0).sqrt();
        [
            (i + 1.0 / 3.0) * a1[0] + (j + 1.0 / 3.0) * a2[0],
            (i + 1.0 / 3.0) * a1[1] + (j + 1.0 / 3.0) * a2[1],
            h,
        ]
    };
    let mut reactant = Array1::from_vec(hollow(2.0, 2.0).to_vec());
    let mut product = Array1::from_vec(hollow(3.0, 2.0).to_vec());
    let pot = Pot::MorseSlab { slab: slab.clone() };
    minimize(&mut reactant, &pot, None)?;
    minimize(&mut product, &pot, None)?;
    Ok((reactant, product, slab))
}

pub(crate) fn morse_heptamer() -> Result<(Array1<f64>, Array1<f64>, Vec<[f64; 3]>), SaddleError> {
    let nn = 2.897;
    let a1 = [nn, 0.0, 0.0];
    let a2 = [nn / 2.0, nn * 3.0_f64.sqrt() / 2.0, 0.0];
    let h = nn * (2.0_f64 / 3.0).sqrt();
    let mut slab = Vec::new();
    for i in -1..6 {
        for j in -1..6 {
            slab.push([
                i as f64 * a1[0] + j as f64 * a2[0],
                i as f64 * a1[1] + j as f64 * a2[1],
                0.0,
            ]);
        }
    }
    let hollow = |i: f64, j: f64| {
        [
            (i + 1.0 / 3.0) * a1[0] + (j + 1.0 / 3.0) * a2[0],
            (i + 1.0 / 3.0) * a1[1] + (j + 1.0 / 3.0) * a2[1],
            h,
        ]
    };
    // Compact heptamer, and the same island with one edge atom in the
    // next hollow: a Pt-island edge hop on the frozen Morse layer.
    let island = [
        (2.0, 2.0),
        (3.0, 2.0),
        (2.0, 3.0),
        (1.0, 2.0),
        (2.0, 1.0),
        (3.0, 1.0),
        (1.0, 3.0),
    ];
    let mut reactant = Array1::zeros(21);
    for (n, (i, j)) in island.iter().enumerate() {
        let p = hollow(*i, *j);
        reactant[3 * n] = p[0];
        reactant[3 * n + 1] = p[1];
        reactant[3 * n + 2] = p[2];
    }
    let mut product = reactant.clone();
    let moved = hollow(4.0, 2.0);
    product[3] = moved[0];
    product[4] = moved[1];
    product[5] = moved[2];
    let pot = Pot::MorseSlab { slab: slab.clone() };
    minimize(&mut reactant, &pot, None)?;
    minimize(&mut product, &pot, None)?;
    Ok((reactant, product, slab))
}

pub(crate) fn lj13_endpoints() -> (Array1<f64>, Array1<f64>) {
    // eOn client/unit_tests/data/systems/neb_lj13, shifted by the
    // reactant centroid. The file energy -39.96535194652229 matches
    // this Lennard-Jones sum.
    let reactant = [
        50.65605413599571,
        52.02801494194084,
        52.32815952876039,
        51.63328851925837,
        52.63600287312886,
        51.229889253477495,
        51.333244273857,
        52.9090293637059,
        52.26479259481949,
        50.554374961599436,
        52.76243907382286,
        51.48100142412469,
        51.53090901655143,
        50.87474699298496,
        51.66828004564838,
        50.95486209140735,
        51.757623001211805,
        51.31162177931877,
        51.224862252592864,
        50.97534048280285,
        50.59374465908034,
        51.28903775355402,
        52.07903151549311,
        50.31902323047677,
        50.45187067438953,
        50.99890228905946,
        51.91996579806674,
        51.74679013611403,
        51.89794777434641,
        52.06032645625532,
        51.45029336810132,
        52.17354869031571,
        53.09829984243966,
        52.03415665719459,
        51.62192845504001,
        51.01853712416401,
        51.23109502801023,
        51.151556846087445,
        52.70224808931864,
    ];
    let product = [
        50.64298032974756,
        52.05588492716968,
        52.32998913139571,
        51.65155569177506,
        52.38264191669297,
        51.1097488754756,
        51.410119862170305,
        52.87369387015567,
        52.1306118818863,
        50.60781977341129,
        52.73824949314509,
        51.39472353743992,
        51.47061592750834,
        50.84757316936427,
        51.61980609151891,
        50.82336495401012,
        51.67645458433605,
        51.32938573615539,
        51.5195945863182,
        51.38180423017856,
        50.55679323837121,
        50.81762731109767,
        52.25695661455614,
        50.38957236293557,
        50.40675743569269,
        50.9691977655336,
        52.04095500915992,
        51.69111787937081,
        51.82900626552174,
        52.04229106911383,
        51.46873878378175,
        52.21618616962847,
        53.05269422188865,
        52.355329166128584,
        51.48592847574357,
        51.25411153493548,
        51.22382283141214,
        51.15252914952926,
        52.74404869782771,
    ];
    let mut r = Array1::from_vec(reactant.to_vec());
    let mut p = Array1::from_vec(product.to_vec());
    let mut c = [0.0; 3];
    for a in 0..13 {
        for k in 0..3 {
            c[k] += r[3 * a + k];
        }
    }
    for k in 0..3 {
        c[k] /= 13.0;
    }
    for a in 0..13 {
        for k in 0..3 {
            r[3 * a + k] -= c[k];
            p[3 * a + k] -= c[k];
        }
    }
    (r, p)
}
