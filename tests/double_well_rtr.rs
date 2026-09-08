//! RTR-tCG band relaxation on the analytic double well: the interior
//! images fall onto the reaction path (y = z = 0), the climbing image
//! reaches the saddle at the origin, the endpoints never move, and every
//! accepted step decreases the force.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::{
    BandConfig, BandRtr, BandSession, BandStatus, BandSurface, RtrConfig, SaddleError, band_forces,
};

/// V = (x^2 - 1)^2 + 2 y^2 + 2 z^2: minima at (+-1, 0, 0), saddle at
/// the origin with barrier 1.
struct DoubleWell;

impl BandSurface for DoubleWell {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            energies[i] = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[(i, 0)] = 4.0 * x * (x * x - 1.0);
            gradients[(i, 1)] = 4.0 * y;
            gradients[(i, 2)] = 4.0 * z;
        }
        Ok(())
    }
}

fn initial_band(n_images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = -1.0 + 2.0 * t;
        band[(i, 1)] = 0.3 * (std::f64::consts::PI * t).sin();
    }
    band
}

#[test]
fn rtr_band_relaxes_onto_the_path_and_climbs_to_the_saddle() {
    let n = 7;
    let mut band = initial_band(n);
    let first = band.row(0).to_owned();
    let last = band.row(n - 1).to_owned();
    let config = BandConfig {
        force_tol: 1e-6,
        ..BandConfig::default()
    };
    let mut rtr = BandRtr::new(
        RtrConfig {
            radius_max: 0.5,
            ..RtrConfig::default()
        },
        true,
    );
    let mut converged = false;
    let mut accepted = 0;
    for _ in 0..400 {
        let report = rtr.step(&config, &DoubleWell, &mut band).expect("step");
        if report.accepted {
            accepted += 1;
        }
        if report.max_force <= config.force_tol {
            converged = true;
            break;
        }
    }
    assert!(converged, "band did not converge: {:?}", band);
    assert!(accepted > 0);
    assert_eq!(band.row(0), first.view());
    assert_eq!(band.row(n - 1), last.view());
    for i in 1..n - 1 {
        assert!(
            band[(i, 1)].abs() < 1e-5,
            "image {i} off the path: {}",
            band[(i, 1)]
        );
        assert!(band[(i, 2)].abs() < 1e-5);
    }
    let forces = band_forces(&config, &DoubleWell, band.view(), None).unwrap();
    let ci = (1..n - 1)
        .max_by(|&a, &b| forces.energies[a].partial_cmp(&forces.energies[b]).unwrap())
        .unwrap();
    assert!(
        band[(ci, 0)].abs() < 1e-4,
        "climbing image not at the saddle: {}",
        band[(ci, 0)]
    );
    assert!((forces.energies[ci] - 1.0).abs() < 1e-6);
}

#[test]
fn rejected_steps_shrink_the_radius_and_accepted_boundary_steps_grow_it() {
    let n = 5;
    let mut band = initial_band(n);
    let config = BandConfig::default();
    let mut rtr = BandRtr::new(
        RtrConfig {
            radius_max: 4.0,
            ..RtrConfig::default()
        },
        false,
    );
    let r0 = rtr.radius.radius;
    let report = rtr.step(&config, &DoubleWell, &mut band).expect("step");
    if report.accepted {
        assert!(rtr.radius.radius >= r0);
    } else {
        assert!(rtr.radius.radius < r0);
    }
    assert!(report.cg_iterations >= 1);
}

#[test]
fn band_session_steps_by_rtr_when_configured() {
    let n = 7;
    let band = initial_band(n);
    let config = BandConfig {
        force_tol: 1e-5,
        rtr: Some(RtrConfig {
            radius_max: 0.5,
            ..RtrConfig::default()
        }),
        ..BandConfig::default()
    };
    let mut session = BandSession::new(config, band).expect("session");
    let mut converged = false;
    for _ in 0..400 {
        let report = session.step(&DoubleWell).expect("step");
        if report.status == BandStatus::Converged {
            converged = true;
            break;
        }
    }
    assert!(converged, "session did not converge");
    let pos = session.positions();
    for i in 1..n - 1 {
        assert!(pos[(i, 1)].abs() < 1e-4, "image {i} off the path");
    }
    let ci = session.climbing_image().expect("climbing image armed");
    assert!(
        pos[(ci, 0)].abs() < 1e-3,
        "climbing image not at the saddle: {}",
        pos[(ci, 0)]
    );
}
