//! Release table for the 109-atom Mishin copper dumbbell.
//!
//! The searches are the fixed-cell and variable-cell dimers of
//! `tests/mishin_cu.rs`. A row is a barrier only when that search
//! stops on a switched index-1 saddle. Otherwise a climbing band
//! decides, and a band that does not rise is recorded as no barrier.

use std::io::{Write, stderr};
use std::path::PathBuf;

use ndarray::{Array1, Array2, ArrayView1};
use rgmin::{Control, Method, Oracle, Solver};
use rgsaddle::mic::Cell as LinkCell;
use rgsaddle::{
    BandConfig, BandStatus, CellMask, CiConfig, ExplicitSide, FiniteDifference, IndexCheck,
    JointChart, LINEAR_FOLD_EV, MinModeStatus, PointSurface, SaddleError, SolidNeb,
    SparseHessianOpts, SparsityPattern, SpringKind, StrainPoint, StrainRateTable,
    activation_volume, compliance_from_stiffness, elastic_dipole, exact_eigh, hydrostatic,
    joint_coordinate, judge_linear, prefactor_hessian, shear_xz, shear_yx,
    split_translational_modes, uniaxial_x, vineyard_prefactor,
};

use super::{Copper, Held, SearchRow, axis_report, closest_pair, det, dumbbell, fcc, max_abs};

const CU_MASS_AMU: f64 = 63.546;
const NO_BARRIER_RISE: f64 = 1.0e-3;

fn log(message: &str) {
    let mut err = stderr();
    let _ = writeln!(err, "{message}");
    let _ = err.flush();
}

fn apply_strain(
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
) -> (Array1<f64>, [[f64; 3]; 3]) {
    let mut deform = eps;
    for i in 0..3 {
        deform[i][i] += 1.0;
    }
    let mut h = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            h[i][j] =
                cell[i][0] * deform[0][j] + cell[i][1] * deform[1][j] + cell[i][2] * deform[2][j];
        }
    }
    let mut out = Array1::zeros(positions.len());
    for atom in 0..positions.len() / 3 {
        let row = [
            positions[3 * atom],
            positions[3 * atom + 1],
            positions[3 * atom + 2],
        ];
        for c in 0..3 {
            out[3 * atom + c] =
                row[0] * deform[0][c] + row[1] * deform[1][c] + row[2] * deform[2][c];
        }
    }
    (out, h)
}

fn cell_shift(row: &SearchRow, start: [[f64; 3]; 3]) -> f64 {
    let mut max = 0.0_f64;
    for i in 0..3 {
        for j in 0..3 {
            max = max.max((row.cell[i][j] - start[i][j]).abs());
        }
    }
    max
}

fn connecting(row: &SearchRow, start: [[f64; 3]; 3]) -> bool {
    let indexed = row.status == MinModeStatus::Converged
        && !row.rejected
        && row.negative_modes == Some(1)
        && row.curvature < 0.0
        && row.barrier.is_finite()
        && row.barrier > 0.0
        && row.enthalpy_barrier.is_finite()
        && row.enthalpy_barrier > 0.0;
    if !indexed {
        return false;
    }
    if row.switched {
        return true;
    }
    // <110> in a plane other than xy. A saddle that leaves the imposed
    // cell, as the 5% variable search does, is not this path.
    row.angle_110 < 20.0 && row.angle_100 > 15.0 && cell_shift(row, start) < 0.2
}

fn index_phrase(row: &SearchRow) -> String {
    if row.rejected {
        format!("rejected index {}", row.negative_modes.unwrap_or(0))
    } else if row.negative_modes == Some(1) && row.status == MinModeStatus::Converged {
        "index 1".to_string()
    } else {
        format!(
            "unresolved status {:?} negative {:?}",
            row.status, row.negative_modes
        )
    }
}

#[derive(Clone)]
struct Geometry {
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
    energy: f64,
    stress: [[f64; 3]; 3],
}

fn positive_lambdas(
    copper: &Copper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
) -> Result<(Vec<f64>, usize), SaddleError> {
    let surface = Held { copper, cell };
    let opts = SparseHessianOpts::default().with_difference(FiniteDifference::Central);
    let link = LinkCell::from_vectors(
        [cell[0][0], cell[0][1], cell[0][2]],
        [cell[1][0], cell[1][1], cell[1][2]],
        [cell[2][0], cell[2][1], cell[2][2]],
        [0.0; 3],
    )
    .map_err(|_| SaddleError::Invalid("hessian cell is singular".into()))?;
    let pattern = SparsityPattern::from_positions(
        positions.view(),
        Some(&link),
        super::RC,
        opts.search_skin(),
    )?;
    let hessian = prefactor_hessian(&surface, positions.view(), &pattern, opts)?;
    let projected = hessian.project_excluded(&surface, positions.view())?;
    let (values, vectors) = exact_eigh(projected.view())?;
    let translations = surface.excluded_modes(positions.view())?;
    let values: Vec<f64> = values.iter().copied().collect();
    let split = split_translational_modes(&values, vectors.view(), translations.view())?;
    Ok((split.positive, split.negative.len()))
}

fn prefactor_hz(minimum: &(Vec<f64>, usize), saddle: &(Vec<f64>, usize)) -> Option<f64> {
    if minimum.1 != 0 || saddle.1 != 1 {
        return None;
    }
    vineyard_prefactor(&minimum.0, &saddle.0, CU_MASS_AMU).ok()
}

fn relax_enthalpy(
    copper: &Copper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    reference: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
) -> Result<Geometry, SaddleError> {
    let n_atoms = positions.len() / 3;
    let chart = JointChart::new(reference, n_atoms, CellMask::all(), external, 1.0)?;
    let mut q = joint_coordinate(&chart, CellMask::all(), positions.view(), cell)?;
    let mut solver = Solver::new(
        Method::Lbfgs { memory: 20 },
        Control {
            maxiter: usize::MAX,
            gtol: 0.0,
            ..Control::default()
        },
        q.len(),
    );
    for _ in 0..400 {
        let oracle = Oracle::unbounded(q.len(), |x: ArrayView1<f64>| {
            let (r, h) = chart.geometry(x).expect("geometry");
            let (energy, gradient, stress) = copper.evaluate(r.view(), h);
            let enthalpy = chart.enthalpy(energy, x).expect("enthalpy");
            let g = chart
                .gradient(x, gradient.view(), stress)
                .expect("joint gradient");
            (enthalpy, g)
        });
        solver.step(&oracle, &mut q).expect("enthalpy relax");
        let (r, h) = chart.geometry(q.view())?;
        let (_, gradient, stress) = copper.evaluate(r.view(), h);
        let g = chart.gradient(q.view(), gradient.view(), stress)?;
        if max_abs(&g) < 1e-3 {
            break;
        }
    }
    let (r, h) = chart.geometry(q.view())?;
    let (energy, _, stress) = copper.evaluate(r.view(), h);
    Ok(Geometry {
        positions: r,
        cell: h,
        energy,
        stress,
    })
}

struct Climbed {
    rise: f64,
    positions: Array1<f64>,
    cell: [[f64; 3]; 3],
}

fn band_climb(
    copper: &Copper,
    reactant: &Array1<f64>,
    product: &Array1<f64>,
    cell: [[f64; 3]; 3],
    product_cell: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    frozen: bool,
) -> Result<Climbed, SaddleError> {
    let n_images = 7;
    let n_cart = reactant.len();
    let diff = super::minimum_image_difference(product, reactant, cell);
    let mut positions = Array2::zeros((n_images, n_cart));
    let mut cells = vec![cell; n_images];
    for image in 0..n_images {
        let t = image as f64 / (n_images - 1) as f64;
        for k in 0..n_cart {
            positions[(image, k)] = reactant[k] + t * diff[k];
        }
        if !frozen {
            for a in 0..3 {
                for b in 0..3 {
                    cells[image][a][b] = (1.0 - t) * cell[a][b] + t * product_cell[a][b];
                }
            }
        }
    }
    let config = BandConfig {
        climbing: Some(CiConfig::always()),
        force_tol: 2e-3,
        max_move: 0.08,
        remove_translation: true,
        method: Method::Lbfgs { memory: 20 },
        min_mode: None,
        spring: SpringKind::Uniform { k: 1.0 },
        ..BandConfig::default()
    };
    let surface = Held { copper, cell };
    let mut neb = if frozen {
        SolidNeb::frozen(config, positions)?
    } else {
        SolidNeb::joint(
            config,
            cell,
            positions,
            &cells,
            CellMask::all(),
            external,
            1.0,
        )?
    };
    let mut report = neb.step(&surface)?;
    let volume0 = det(cell).abs();
    for _ in 1..200 {
        if report.status != BandStatus::Running || report.max_force <= 2e-3 {
            break;
        }
        if let Some(images) = neb.images()
            && images.iter().any(|image| {
                let volume = det(image.cell).abs();
                volume < 0.5 * volume0 || volume > 2.0 * volume0
            })
        {
            log("  band stopped, an image volume left the imposed range");
            break;
        }
        report = neb.step(&surface)?;
    }
    let images = neb
        .images()
        .ok_or_else(|| SaddleError::Invalid("the solid-state band did not evaluate".into()))?;
    let reactant_h = images[0].enthalpy;
    let mut best = 0usize;
    let mut rise = f64::MIN;
    for (index, image) in images.iter().enumerate() {
        let above = image.enthalpy - reactant_h;
        if above > rise {
            rise = above;
            best = index;
        }
    }
    Ok(Climbed {
        rise,
        positions: images[best].positions.clone(),
        cell: images[best].cell,
    })
}

fn explicit_from_band(
    copper: &Copper,
    reactant: &Array1<f64>,
    products: &[Geometry],
    cell: [[f64; 3]; 3],
    external: [[f64; 3]; 3],
    frozen: bool,
    reactant_h: f64,
) -> Result<(ExplicitSide, Option<Geometry>), SaddleError> {
    let mut best_rise = f64::MAX;
    let mut best_side: Option<(ExplicitSide, Geometry)> = None;
    for product in products {
        let Climbed {
            rise,
            positions,
            cell: climbed_cell,
        } = band_climb(
            copper,
            reactant,
            &product.positions,
            cell,
            product.cell,
            external,
            frozen,
        )?;
        log(&format!(
            "  band frozen={frozen} rise={rise:.6e} product_gap={:.6e}",
            product.energy - reactant_h
        ));
        if rise < NO_BARRIER_RISE {
            best_rise = best_rise.min(rise);
            continue;
        }
        log(&format!("  indexing band image rise={rise:.6e}"));
        let surface = Held {
            copper,
            cell: climbed_cell,
        };
        let index = rgsaddle::index_at(
            &surface,
            positions.view(),
            IndexCheck::first_order().with_step(1e-4),
        )?;
        if index.negative == 1 && rise < best_rise {
            best_rise = rise;
            best_side = Some((
                ExplicitSide::barrier(rise, None, "index 1")?,
                Geometry {
                    positions,
                    cell: climbed_cell,
                    energy: reactant_h + rise,
                    stress: [[0.0; 3]; 3],
                },
            ));
        }
    }
    match best_side {
        Some((side, geometry)) => Ok((side, Some(geometry))),
        None => Ok((ExplicitSide::no_barrier(), None)),
    }
}

fn side_from_search(
    row: &SearchRow,
    start: [[f64; 3]; 3],
    frozen: bool,
) -> Result<Option<ExplicitSide>, SaddleError> {
    if !connecting(row, start) {
        return Ok(None);
    }
    let barrier = if frozen {
        row.barrier
    } else {
        row.enthalpy_barrier
    };
    Ok(Some(ExplicitSide::barrier(
        barrier,
        None,
        index_phrase(row),
    )?))
}

struct Prepared {
    minimum: Geometry,
    aligned: Geometry,
    opposite: Geometry,
    nudged: Array1<f64>,
    mode: Array1<f64>,
    min_angle_100: f64,
    min_angle_110: f64,
}

fn prepare(
    copper: &Copper,
    along_100: &Array1<f64>,
    along_110: &Array1<f64>,
    along_opposite: &Array1<f64>,
    cube: [[f64; 3]; 3],
    eps: [[f64; 3]; 3],
) -> Prepared {
    let (sheared, cell) = apply_strain(along_100, cube, eps);
    let (sheared_110, _) = apply_strain(along_110, cube, eps);
    let (sheared_opposite, _) = apply_strain(along_opposite, cube, eps);
    let mut minimum = sheared;
    super::relax(copper, &mut minimum, cell, 1e-3, 250);
    let (e_min, _, sigma_min) = copper.evaluate(minimum.view(), cell);
    let mut aligned = sheared_110;
    super::relax(copper, &mut aligned, cell, 1e-3, 250);
    let (e_aligned, _, sigma_aligned) = copper.evaluate(aligned.view(), cell);
    let mut opposite = sheared_opposite;
    super::relax(copper, &mut opposite, cell, 1e-3, 250);
    let (e_opposite, _, sigma_opposite) = copper.evaluate(opposite.view(), cell);
    let (_, _, dir) = closest_pair(&minimum, cell);
    let (min_100, min_110, _, _) = axis_report(dir);
    let connect = super::minimum_image_difference(&aligned, &minimum, cell);
    let norm = connect.dot(&connect).sqrt().max(1e-12);
    let mut nudged = minimum.clone();
    for (slot, component) in nudged.iter_mut().zip(connect.iter()) {
        *slot += 0.02 * component / norm;
    }
    Prepared {
        minimum: Geometry {
            positions: minimum,
            cell,
            energy: e_min,
            stress: sigma_min,
        },
        aligned: Geometry {
            positions: aligned,
            cell,
            energy: e_aligned,
            stress: sigma_aligned,
        },
        opposite: Geometry {
            positions: opposite,
            cell,
            energy: e_opposite,
            stress: sigma_opposite,
        },
        nudged,
        mode: connect,
        min_angle_100: min_100,
        min_angle_110: min_110,
    }
}

fn search_pair(
    copper: &Copper,
    prepared: &Prepared,
    cube: [[f64; 3]; 3],
) -> (SearchRow, SearchRow) {
    let fixed = super::saddle_from_minimum(
        copper,
        &prepared.nudged,
        prepared.minimum.cell,
        &prepared.mode,
        CellMask::frozen(),
        [[0.0; 3]; 3],
        cube,
        prepared.minimum.energy,
        prepared.min_angle_100,
        prepared.min_angle_110,
        2500,
    );
    let mut external = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            external[i][j] = -prepared.minimum.stress[i][j];
        }
    }
    let variable = super::saddle_from_minimum(
        copper,
        &prepared.nudged,
        prepared.minimum.cell,
        &prepared.mode,
        CellMask::all(),
        external,
        cube,
        prepared.minimum.energy,
        prepared.min_angle_100,
        prepared.min_angle_110,
        2500,
    );
    (fixed, variable)
}

fn voigt_stress(stress: [[f64; 3]; 3]) -> [f64; 6] {
    rgsaddle::strain_rate::voigt(stress)
}

fn stiffness(copper: &Copper) -> Result<[[f64; 6]; 6], SaddleError> {
    let (positions, cell) = fcc(1, 3.615);
    let delta = 1e-5;
    let mut matrix = [[0.0; 6]; 6];
    for column in 0..6 {
        let mut plus = [0.0; 6];
        let mut minus = [0.0; 6];
        plus[column] = delta;
        minus[column] = -delta;
        let sp = voigt_stress(strained_stress(copper, &positions, cell, plus));
        let sm = voigt_stress(strained_stress(copper, &positions, cell, minus));
        for row in 0..6 {
            matrix[row][column] = (sp[row] - sm[row]) / (2.0 * delta);
        }
    }
    Ok(matrix)
}

fn strained_stress(
    copper: &Copper,
    positions: &Array1<f64>,
    cell: [[f64; 3]; 3],
    eps: [f64; 6],
) -> [[f64; 3]; 3] {
    let tensor = rgsaddle::strain_rate::from_voigt(eps);
    let (moved, strained) = apply_strain(positions, cell, tensor);
    copper.evaluate(moved.view(), strained).2
}

fn samples() -> Vec<(&'static str, [[f64; 3]; 3])> {
    let mut rows = vec![("reference", shear_yx(0.0))];
    for gamma in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05] {
        rows.push(("shear_yx", shear_yx(gamma)));
    }
    rows.push(("shear_xz", shear_xz(0.02)));
    rows.push(("uniaxial_x", uniaxial_x(0.01)));
    rows.push(("hydrostatic", hydrostatic(0.01)));
    rows
}

#[test]
#[ignore = "release copper strain-rate table"]
fn cu_strain_rate_table() -> Result<(), SaddleError> {
    let copper = Copper::new();
    let (perfect, cube) = fcc(3, 3.615);
    let mut bulk = perfect.clone();
    super::relax(&copper, &mut bulk, cube, 1e-4, 50);
    let bulk_stress = copper.evaluate(bulk.view(), cube).2;
    let along_100 = dumbbell(&perfect, [1.0, 0.0, 0.0], 2.05);
    let root2 = std::f64::consts::FRAC_1_SQRT_2;
    let along_110 = dumbbell(&perfect, [root2, root2, 0.0], 2.05);
    let along_opposite = dumbbell(&perfect, [-root2, root2, 0.0], 2.05);

    let mut points = Vec::new();
    let mut reference_min_p = None;
    let mut reference_sad_p = None;
    let mut barrier0 = None;
    let mut reference_checked = false;

    for (direction, eps) in samples() {
        log(&format!("STRAIN {direction} {}", strain_label(eps)));
        let prepared = prepare(&copper, &along_100, &along_110, &along_opposite, cube, eps);
        let (fixed, variable) = search_pair(&copper, &prepared, cube);
        log(&format!(
            "  dimer fixed barrier={} switched={} angle110={} shift={} index={} variable enthalpy={} switched={} angle110={} shift={} index={}",
            fixed.barrier,
            fixed.switched as u8,
            fixed.angle_110,
            cell_shift(&fixed, prepared.minimum.cell),
            index_phrase(&fixed),
            variable.enthalpy_barrier,
            variable.switched as u8,
            variable.angle_110,
            cell_shift(&variable, prepared.minimum.cell),
            index_phrase(&variable)
        ));
        let mut external = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                external[i][j] = -prepared.minimum.stress[i][j];
            }
        }
        let (helmholtz, helmholtz_geom) =
            match side_from_search(&fixed, prepared.minimum.cell, true)? {
                Some(side) => (
                    side,
                    Some(Geometry {
                        positions: fixed.positions.clone(),
                        cell: fixed.cell,
                        energy: fixed.barrier + prepared.minimum.energy,
                        stress: fixed.stress,
                    }),
                ),
                None => explicit_from_band(
                    &copper,
                    &prepared.minimum.positions,
                    &[prepared.aligned.clone(), prepared.opposite.clone()],
                    prepared.minimum.cell,
                    [[0.0; 3]; 3],
                    true,
                    prepared.minimum.energy,
                )?,
            };
        let mut products = vec![prepared.aligned.clone(), prepared.opposite.clone()];
        if !connecting(&variable, prepared.minimum.cell) {
            for product in &mut products {
                let relaxed = relax_enthalpy(
                    &copper,
                    &product.positions,
                    product.cell,
                    prepared.minimum.cell,
                    external,
                )?;
                *product = relaxed;
            }
        }
        let (gibbs, gibbs_geom) = match side_from_search(&variable, prepared.minimum.cell, false)? {
            Some(side) => (
                side,
                Some(Geometry {
                    positions: variable.positions.clone(),
                    cell: variable.cell,
                    energy: variable.barrier + prepared.minimum.energy,
                    stress: variable.stress,
                }),
            ),
            None => explicit_from_band(
                &copper,
                &prepared.minimum.positions,
                &products,
                prepared.minimum.cell,
                external,
                false,
                prepared.minimum.energy,
            )?,
        };

        let min_spec =
            positive_lambdas(&copper, &prepared.minimum.positions, prepared.minimum.cell)?;
        let mut helmholtz = helmholtz;
        if let Some(barrier) = helmholtz.barrier
            && let Some(geometry) = helmholtz_geom
        {
            let sad_spec = positive_lambdas(&copper, &geometry.positions, geometry.cell)?;
            if let Some(hz) = prefactor_hz(&min_spec, &sad_spec) {
                helmholtz = ExplicitSide::barrier(barrier, Some(hz), helmholtz.index.clone())?;
            }
        }
        let mut gibbs = gibbs;
        if let Some(barrier) = gibbs.barrier
            && let Some(geometry) = gibbs_geom
        {
            let sad_spec = positive_lambdas(&copper, &geometry.positions, geometry.cell)?;
            if let Some(hz) = prefactor_hz(&min_spec, &sad_spec) {
                gibbs = ExplicitSide::barrier(barrier, Some(hz), gibbs.index.clone())?;
            }
        }

        if direction == "reference" {
            let volume = det(prepared.minimum.cell).abs();
            reference_min_p = Some(elastic_dipole(
                volume,
                prepared.minimum.stress,
                bulk_stress,
            )?);
            reference_sad_p = Some(elastic_dipole(volume, fixed.stress, bulk_stress)?);
            barrier0 = Some(fixed.barrier);
            reference_checked = connecting(&fixed, prepared.minimum.cell);
        }

        let linear = 0.0;
        let (status, error) = judge_linear(linear, helmholtz.barrier, LINEAR_FOLD_EV);
        points.push(StrainPoint {
            direction: direction.to_string(),
            strain: eps,
            linear,
            linear_status: status,
            helmholtz,
            gibbs,
            linear_error: error,
        });
    }

    if !reference_checked {
        return Err(SaddleError::Invalid(
            "the zero-strain fixed saddle was not the <110> path".into(),
        ));
    }
    let minimum = reference_min_p.expect("reference dipole");
    let saddle = reference_sad_p.expect("reference saddle dipole");
    let barrier0 = barrier0.expect("reference barrier");
    let model = rgsaddle::DipoleModel {
        minimum,
        saddle,
        barrier0,
    };
    for point in &mut points {
        let linear = model.linear_barrier(point.strain);
        let (status, error) = judge_linear(linear, point.helmholtz.barrier, LINEAR_FOLD_EV);
        point.linear = linear;
        point.linear_status = status;
        point.linear_error = error;
    }
    let compliance = compliance_from_stiffness(&stiffness(&copper)?)?;
    let omega = activation_volume(minimum, saddle, &compliance)?;
    let table = StrainRateTable::try_new(model, points, Some(omega))?;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    table.save(root.join("benchmarks/strain_rate_table_cu.csv"))?;
    table.save(root.join("benchmarks/strain_rate_table_cu.json"))?;
    log("wrote benchmarks/strain_rate_table_cu.csv");
    for point in table.points() {
        if point.linear_status == rgsaddle::LinearStatus::Failed {
            log(&format!(
                "LINEAR FAILED {} linear={} helmholtz={:?} error={:?}",
                point.direction, point.linear, point.helmholtz.barrier, point.linear_error
            ));
        }
    }
    Ok(())
}

fn strain_label(eps: [[f64; 3]; 3]) -> String {
    format!(
        "[{:.4} {:.4} {:.4}; {:.4} {:.4} {:.4}; {:.4} {:.4} {:.4}]",
        eps[0][0],
        eps[0][1],
        eps[0][2],
        eps[1][0],
        eps[1][1],
        eps[1][2],
        eps[2][0],
        eps[2][1],
        eps[2][2]
    )
}
