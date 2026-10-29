//! Energy-weighted springs set from the band energies at every
//! evaluation (Ásgeirsson et al., J. Chem. Theory Comput. 17, 4929
//! (2021), doi:10.1021/acs.jctc.1c00462) on Müller-Brown with a
//! harmonic z.

use ndarray::{Array1, Array2, ArrayView2};
use rgsaddle::rtr::band_forces;
use rgsaddle::spring::energy_weighted_constants;
use rgsaddle::{BandConfig, BandSession, BandStatus, BandSurface, SaddleError, SpringKind};

struct MuellerBrown;

impl BandSurface for MuellerBrown {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        const A: [f64; 4] = [-200.0, -100.0, -170.0, 15.0];
        const AA: [f64; 4] = [-1.0, -1.0, -6.5, 0.7];
        const B: [f64; 4] = [0.0, 0.0, 11.0, 0.6];
        const C: [f64; 4] = [-10.0, -10.0, -6.5, 0.7];
        const X0: [f64; 4] = [1.0, 0.0, -0.5, -1.0];
        const Y0: [f64; 4] = [0.0, 0.5, 1.5, 1.0];
        for (i, row) in positions.outer_iter().enumerate() {
            let (x, y, z) = (row[0], row[1], row[2]);
            let mut e = 10.0 * z * z;
            let (mut gx, mut gy) = (0.0, 0.0);
            for k in 0..4 {
                let (dx, dy) = (x - X0[k], y - Y0[k]);
                let t = A[k] * (AA[k] * dx * dx + B[k] * dx * dy + C[k] * dy * dy).exp();
                e += t;
                gx += t * (2.0 * AA[k] * dx + B[k] * dy);
                gy += t * (B[k] * dx + 2.0 * C[k] * dy);
            }
            energies[i] = e;
            gradients[(i, 0)] = gx;
            gradients[(i, 1)] = gy;
            gradients[(i, 2)] = 20.0 * z;
        }
        Ok(())
    }
}

const K_MIN: f64 = 0.97;
const K_MAX: f64 = 9.7;

fn linear_band(n_images: usize) -> Array2<f64> {
    let a = [-0.558223634633024, 1.441725841804669];
    let b = [0.623499404930877, 0.028037758528686];
    let mut band = Array2::zeros((n_images, 3));
    for i in 0..n_images {
        let t = i as f64 / (n_images - 1) as f64;
        band[(i, 0)] = (1.0 - t) * a[0] + t * b[0];
        band[(i, 1)] = (1.0 - t) * a[1] + t * b[1];
    }
    band
}

fn energies_of(positions: ArrayView2<f64>) -> Array1<f64> {
    let mut energies = Array1::zeros(positions.nrows());
    let mut gradients = Array2::zeros(positions.raw_dim());
    MuellerBrown
        .eval(positions, &mut energies, &mut gradients)
        .unwrap();
    energies
}

fn config(spring: SpringKind) -> BandConfig {
    BandConfig {
        spring,
        force_tol: 1e-3,
        max_move: 0.05,
        method: rgmin::Method::Lbfgs { memory: 10 },
        min_mode: None,
        ..BandConfig::default()
    }
}

fn dynamic() -> SpringKind {
    SpringKind::EnergyWeighted {
        k_min: K_MIN,
        k_max: K_MAX,
    }
}

fn frozen_at(positions: ArrayView2<f64>) -> SpringKind {
    SpringKind::Weighted {
        ks: energy_weighted_constants(energies_of(positions).view(), K_MIN, K_MAX),
    }
}

#[test]
fn the_band_force_uses_the_constants_of_the_current_energies() {
    let mut session = BandSession::new(config(dynamic()), linear_band(9)).unwrap();
    for _ in 0..40 {
        session.step(&MuellerBrown).unwrap();
    }
    let here = session.positions().to_owned();
    let start = linear_band(9);
    for ci in [None, Some(3)] {
        let got = band_forces(&config(dynamic()), &MuellerBrown, here.view(), ci).unwrap();
        let want = band_forces(
            &config(frozen_at(here.view())),
            &MuellerBrown,
            here.view(),
            ci,
        )
        .unwrap();
        assert_eq!(got.force, want.force);
        let stale = band_forces(
            &config(frozen_at(start.view())),
            &MuellerBrown,
            here.view(),
            ci,
        )
        .unwrap();
        let moved = (&got.force - &stale.force).mapv(f64::abs).sum();
        assert!(moved > 1e-3, "springs did not follow the band: {moved}");
    }
}

#[test]
fn the_first_step_matches_springs_frozen_at_the_start() {
    let start = linear_band(9);
    let mut live = BandSession::new(config(dynamic()), start.clone()).unwrap();
    let mut frozen = BandSession::new(config(frozen_at(start.view())), start).unwrap();
    live.step(&MuellerBrown).unwrap();
    frozen.step(&MuellerBrown).unwrap();
    assert_eq!(live.positions(), frozen.positions());
    for _ in 0..10 {
        live.step(&MuellerBrown).unwrap();
        frozen.step(&MuellerBrown).unwrap();
    }
    assert_ne!(live.positions(), frozen.positions());
}

#[test]
fn the_climbing_image_reaches_the_higher_mueller_brown_saddle() {
    let mut session = BandSession::new(config(dynamic()), linear_band(9)).unwrap();
    let mut report = session.step(&MuellerBrown).unwrap();
    for _ in 0..5000 {
        if report.status == BandStatus::Converged {
            break;
        }
        report = session.step(&MuellerBrown).unwrap();
    }
    assert_eq!(report.status, BandStatus::Converged, "{}", report.max_force);
    let ci = report.ci_index.expect("climbing image");
    let point = session.positions().row(ci).to_owned();
    assert!((point[0] + 0.822).abs() < 2e-3, "x {}", point[0]);
    assert!((point[1] - 0.624).abs() < 2e-3, "y {}", point[1]);
    let energy = energies_of(session.positions())[ci];
    assert!((energy + 40.66484).abs() < 1e-4, "energy {energy}");
    let pos = session.positions();
    assert_eq!(pos.row(0), linear_band(9).row(0));
    assert_eq!(pos.row(8), linear_band(9).row(8));
}

#[test]
fn energy_weighted_constants_must_be_ordered_and_positive() {
    for (k_min, k_max) in [
        (0.0, 1.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (1.0, f64::INFINITY),
    ] {
        let spring = SpringKind::EnergyWeighted { k_min, k_max };
        assert!(matches!(
            BandSession::new(config(spring), linear_band(5)),
            Err(SaddleError::Invalid(_))
        ));
    }
    let equal = SpringKind::EnergyWeighted {
        k_min: 2.0,
        k_max: 2.0,
    };
    assert!(BandSession::new(config(equal), linear_band(5)).is_ok());
}
