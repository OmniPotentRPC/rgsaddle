//! Periodic dihedral gradients accumulate when one atom is visited twice.

use ndarray::Array1;
use rgsaddle::internal::pack_cart;
use rgsaddle::mic::Cell;
use rgsaddle::{Constraints, Equality, InternalTopology, TopologyConfig};

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
