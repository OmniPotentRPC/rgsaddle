#![cfg(feature = "readcon")]

use rgsaddle::frame_from_con;

fn vectors(cell: &rgsaddle::mic::Cell) -> [[f64; 3]; 3] {
    [cell.cartesian([1.0,0.0,0.0]), cell.cartesian([0.0,1.0,0.0]), cell.cartesian([0.0,0.0,1.0])]
}

#[test]
fn con_lengths_and_angles_preserve_the_triclinic_metric() {
    let frame = frame_from_con(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/triclinic.con")).unwrap();
    assert_eq!(frame.positions.to_vec(), vec![1.0,2.0,3.0]);
    assert_eq!(frame.masses.to_vec(), vec![12.011]);
    let lattice = vectors(frame.cell.as_ref().unwrap());
    let lengths = [10.0_f64,11.0,12.0];
    for i in 0..3 {
        let squared = lattice[i].iter().map(|v|v*v).sum::<f64>();
        assert!((squared - lengths[i].powi(2)).abs() < 1e-11);
    }
    for (i,j,angle) in [(1,2,80.0_f64),(0,2,100.0),(0,1,60.0)] {
        let dot = lattice[i].iter().zip(lattice[j]).map(|(a,b)|a*b).sum::<f64>();
        assert!((dot/(lengths[i]*lengths[j]) - angle.to_radians().cos()).abs() < 1e-13);
    }
}

#[test]
fn exact_lattice_metadata_takes_precedence_over_rounded_header_angles() {
    let frame = frame_from_con(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/exact_lattice.con")).unwrap();
    let actual = vectors(frame.cell.as_ref().unwrap());
    let expected = [[2.000000000002,0.400000000003,0.0],[0.0,3.000000000004,0.200000000005],[0.100000000006,0.0,4.000000000007]];
    for i in 0..3 { for j in 0..3 { assert!((actual[i][j] - expected[i][j]).abs() < 1e-13); } }
}
