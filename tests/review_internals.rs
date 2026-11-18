//! Review fixes for periodic dihedral gradients and a chart whose
//! lattice follows a variable cell.

use ndarray::{Array1, ArrayView1};
use rgsaddle::internal::pack_cart;
use rgsaddle::mic::Cell;
use rgsaddle::minmode::PointSurface;
use rgsaddle::{
    CellInternalPes, Constraints, Equality, InternalPes, InternalTopology, SaddleError,
    TopologyConfig, covalent_radius,
};

fn rows_of(cell: &Cell) -> [[f64; 3]; 3] {
    let origin = cell.origin();
    let axis = |frac: [f64; 3]| {
        let cart = cell.cartesian(frac);
        [
            cart[0] - origin[0],
            cart[1] - origin[1],
            cart[2] - origin[2],
        ]
    };
    [
        axis([1.0, 0.0, 0.0]),
        axis([0.0, 1.0, 0.0]),
        axis([0.0, 0.0, 1.0]),
    ]
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|k| (a[k] - b[k]).abs() < 1.0e-8)
}

struct Quiet;

impl PointSurface for Quiet {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Ok((0.0, Array1::zeros(x.len())))
    }

    fn cell_grad(
        &self,
        _x: ArrayView1<f64>,
        _cell: &[f64; 9],
    ) -> Result<Option<[f64; 9]>, SaddleError> {
        Ok(Some([0.0; 9]))
    }
}

#[test]
fn a_periodic_three_carbon_chain_does_not_leak_rigid_motion() {
    let cell = Cell::ortho(4.3, 6.0, 6.0).unwrap();
    let x = pack_cart(&[[0.0, 0.2, 0.1], [1.4, 1.0, 0.6], [2.7, 0.3, -0.2]]);
    let cfg = TopologyConfig::default().with_require_complete(false);
    let t = InternalTopology::from_geometry_with(&[6, 6, 6], x.view(), Some(&cell), &cfg).unwrap();
    let k = t.check();
    assert!(k.periodic_network, "{k:?}");
    assert_eq!(k.expected_rank, 6);
    assert!(
        k.rank <= k.expected_rank,
        "rank {} of {}, rigid leak {:.3e}",
        k.rank,
        k.expected_rank,
        k.rigid_leak
    );
    assert!(k.rigid_leak < 1.0e-12, "rigid leak {:.3e}", k.rigid_leak);
    let mut repeated = false;
    for (idx, eq) in t.constraints().equalities().iter().enumerate() {
        if let Equality::Dihedral { atoms, .. } = eq {
            let mut seen = [None; 3];
            for (slot, &atom) in atoms.iter().enumerate() {
                if let Some(prev) = seen[atom] {
                    let images = t.constraints().images(idx).expect("periodic dihedral");
                    if images[prev] != images[slot] {
                        repeated = true;
                    }
                }
                seen[atom] = Some(slot);
            }
        }
    }
    assert!(repeated, "a dihedral should visit one atom in two images");
}

#[test]
fn dihedral_gradient_matches_a_central_difference_when_an_image_repeats() {
    let mut chart = Constraints::new(3).unwrap();
    chart.set_lattice([[4.5, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]);
    let x = pack_cart(&[[0.0, 0.0, 0.0], [1.2, 0.5, 0.2], [2.5, -0.3, 0.7]]);
    chart
        .fix_dihedral_image(
            [0, 1, 2, 0],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]],
            x.view(),
            None,
        )
        .unwrap();
    let images = chart.images(0).expect("repeated image");
    assert_eq!(images[0], [0, 0, 0]);
    assert_eq!(images[3], [1, 0, 0]);
    let j = chart.jacobian(x.view()).unwrap();
    let step = 1.0e-7;
    let mut worst = 0.0_f64;
    for col in 0..x.len() {
        let mut xp = x.clone();
        xp[col] += step;
        let ahead = chart.values(xp.view()).unwrap()[0];
        xp[col] = x[col] - step;
        let behind = chart.values(xp.view()).unwrap()[0];
        let fd = (ahead - behind) / (2.0 * step);
        worst = worst.max((j[(0, col)] - fd).abs());
    }
    assert!(
        worst < 1.0e-6,
        "image-repeat dihedral gradient error {worst}"
    );
}

#[test]
fn a_cell_kick_keeps_the_periodic_chart_on_the_living_lattice() {
    let cell = Cell::ortho(4.0, 5.0, 6.0).unwrap();
    let x = pack_cart(&[[0.0, 0.0, 0.0], [1.4, 0.2, 0.0]]);
    let mut chart = Constraints::new(2).unwrap();
    chart.set_lattice(rows_of(&cell));
    chart
        .fix_bond_image([0, 1], [[0, 0, 0], [1, 0, 0]], x.view(), None)
        .unwrap();
    assert!(chart.is_periodic());
    let mut pes = CellInternalPes::new(x, Array1::from(vec![12.0, 12.0]), chart, cell).unwrap();
    let before = pes.internals().internals().unwrap()[0];
    let nint = pes.internals().n_int();
    let mut d = Array1::zeros(pes.packed_len());
    d[nint] = 0.25;
    pes.kick_packed(&Quiet, d.view()).unwrap();
    let lattice = pes.internals().chart().lattice().expect("lattice");
    let living = rows_of(pes.cell());
    assert!(
        close(lattice[0], living[0])
            && close(lattice[1], living[1])
            && close(lattice[2], living[2]),
        "chart {:?} living {:?}",
        lattice,
        living
    );
    assert!((living[0][0] - 4.25).abs() < 1.0e-12);
    let after = pes.internals().internals().unwrap()[0];
    assert!(
        (after - before).abs() > 0.1,
        "chart kept the old bond {before} -> {after}"
    );
}

#[test]
fn niggli_reduction_retargets_periodic_images() {
    let cell =
        Cell::from_vectors([5.0, 0.0, 0.0], [3.5, 0.6, 0.0], [0.2, 0.3, 4.5], [0.0; 3]).unwrap();
    let x = pack_cart(&[[0.0, 0.0, 0.0], [1.3, 0.2, 0.1]]);
    let mut chart = Constraints::new(2).unwrap();
    chart.set_lattice(rows_of(&cell));
    chart
        .fix_bond_image([0, 1], [[0, 0, 0], [1, 0, 0]], x.view(), None)
        .unwrap();
    let before = chart.values(x.view()).unwrap()[0];
    let mut pes =
        CellInternalPes::new(x.clone(), Array1::from(vec![12.0, 12.0]), chart, cell).unwrap();
    let applied = pes.maybe_niggli_reduce(20.0).unwrap();
    assert!(applied);
    let lattice = pes.internals().chart().lattice().expect("lattice");
    let living = rows_of(pes.cell());
    assert!(
        close(lattice[0], living[0])
            && close(lattice[1], living[1])
            && close(lattice[2], living[2]),
        "chart {:?} living {:?}",
        lattice,
        living
    );
    let frame = pes.internals().position().to_owned();
    let after = pes.internals().chart().values(frame.view()).unwrap()[0];
    assert!(
        (after - before).abs() < 1.0e-8,
        "image bond {before} -> {after}"
    );
}

#[test]
fn an_intra_ring_bond_between_one_and_bond_scale_is_kept() {
    let rsum = 2.0 * covalent_radius(6).unwrap();
    let short = 0.9 * rsum;
    let long = 1.12 * rsum;
    assert!(long > rsum && long < 1.25 * rsum);
    let x_coord = (2.0 * short * short - long * long) / (2.0 * short);
    let y_coord = (short * short - x_coord * x_coord).sqrt();
    let x = pack_cart(&[[0.0, 0.0, 0.0], [short, 0.0, 0.0], [x_coord, y_coord, 0.0]]);
    let t = InternalTopology::from_geometry(&[6, 6, 6], x.view(), None).unwrap();
    assert_eq!(t.constraints().counts().nbonds, 3, "{:?}", t.check());
    let mut long_bond = false;
    for eq in t.constraints().equalities() {
        if let Equality::Bond { atoms, target } = eq
            && *atoms == [1, 2]
            && (*target - long).abs() < 1.0e-8
        {
            long_bond = true;
        }
    }
    assert!(long_bond, "the long intra-ring bond was dropped");
}

#[test]
fn a_dihedral_residual_wraps_across_the_branch_cut() {
    let x = pack_cart(&[
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [1.2, -0.95, 0.05],
    ]);
    let mut measured = Constraints::new(4).unwrap();
    measured.fix_dihedral([0, 1, 2, 3], x.view(), None).unwrap();
    let value = measured.values(x.view()).unwrap()[0];
    let pi = std::f64::consts::PI;
    let target = value - 2.0 * pi + 0.2;
    let mut chart = Constraints::new(4).unwrap();
    chart
        .fix_dihedral([0, 1, 2, 3], x.view(), Some(target))
        .unwrap();
    let residual = chart.residual(x.view()).unwrap()[0];
    let raw = value - target;
    assert!(raw.abs() > pi, "raw residual {raw} does not cross the cut");
    assert!(
        (residual + 0.2).abs() < 1.0e-9,
        "wrapped residual {residual}, raw {raw}, value {value}"
    );
}

fn bent_triatomic(theta_deg: f64, r1: f64, r2: f64) -> Array1<f64> {
    let half = (theta_deg / 2.0).to_radians();
    pack_cart(&[
        [-r1 * half.sin(), r1 * half.cos(), 0.0],
        [0.0, 0.0, 0.0],
        [r2 * half.sin(), r2 * half.cos(), 0.0],
    ])
}

#[test]
fn a_back_transform_that_needs_one_halving_still_converges() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let cfg = TopologyConfig::default().with_back_transform_iterations(3);
    let mut pes =
        InternalPes::auto(bent, Array1::from(vec![16.0, 12.0, 16.0]), &z, None, &cfg).unwrap();
    let q0 = pes.internals().unwrap();
    let angle = pes
        .chart()
        .equalities()
        .iter()
        .position(|eq| matches!(eq, Equality::Angle { .. }))
        .expect("angle");
    let mut dq = Array1::zeros(pes.n_int());
    let requested = 0.4;
    dq[angle] = requested;
    pes.displace(dq.view()).unwrap();
    assert_eq!(pes.pending_rebuild(), None);
    let q1 = pes.internals().unwrap();
    let mut acc = 0.0_f64;
    for i in 0..q0.len() {
        let mut d = q1[i] - q0[i];
        if matches!(pes.chart().equalities()[i], Equality::Dihedral { .. }) {
            let pi = std::f64::consts::PI;
            d -= 2.0 * pi * (d / (2.0 * pi)).round();
        }
        acc += d * d;
    }
    let realized = acc.sqrt();
    assert!(
        (realized - 0.5 * requested).abs() < 1.0e-2,
        "step should converge only after halving, realized {realized}"
    );
}
