//! One AKMC sweep over a strain that varies from site to site.
//!
//! Four copies of the same hop sit along x. The applied shear grows with
//! the site coordinate. The dipole line is used where it is trusted. Past
//! the radius the event is refined on a one-atom double well, which stands
//! in for the solid-state dimer on the strained saddle.
//!
//! ```text
//! cargo run --example akmc_strain
//! ```

use ndarray::{ArrayView1, array};
use rgsaddle::minmode::PointSurface;
use rgsaddle::{
    ElasticDipole, SaddleError, SaddleSeed, Strain, StrainResponse, Trust, TrustPolicy, arrhenius,
};

struct Well;

impl PointSurface for Well {
    fn eval(&self, r: ArrayView1<f64>) -> Result<(f64, ndarray::Array1<f64>), SaddleError> {
        let (x, y, z) = (r[0], r[1], r[2]);
        let energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
        Ok((energy, array![4.0 * x * (x * x - 1.0), 4.0 * y, 4.0 * z]))
    }
}

fn event() -> StrainResponse {
    let mut saddle = [[0.0; 3]; 3];
    saddle[0][1] = 8.0;
    saddle[1][0] = 8.0;
    StrainResponse::new(
        0.40,
        ElasticDipole::from_tensor([[0.0; 3]; 3]).unwrap(),
        ElasticDipole::from_tensor(saddle).unwrap(),
    )
    .unwrap()
    .with_prefactor(1.0e12, None)
    .unwrap()
    .with_policy(TrustPolicy {
        radius: Some(0.03),
        floor: 0.05,
        ..TrustPolicy::default()
    })
    .unwrap()
}

fn main() -> Result<(), SaddleError> {
    let temperature = 600.0;
    let surface = Well;
    let mut events: Vec<_> = (0..4).map(|_| event()).collect();
    let mut chosen = None;
    for (site, event) in events.iter_mut().enumerate() {
        let shear = 0.015 * site as f64;
        let strain = Strain::shear_yx(shear);
        let mut estimate = event.barrier(&strain);
        if matches!(estimate.trust, Trust::NeedsSearch(_)) {
            estimate = event.refine(
                &strain,
                &surface,
                &SaddleSeed {
                    positions: array![0.35, 0.05, 0.0],
                    cell: [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]],
                    mode: array![1.0, 0.0, 0.0],
                    minimum_energy: 0.0,
                },
                80,
            )?;
        }
        let rate = arrhenius(
            estimate.prefactor_hz.expect("prefactor"),
            estimate.barrier,
            temperature,
        );
        println!(
            "site {site} shear {shear:.3} variant {} barrier {:.6} eV rate {rate:.6e} Hz",
            estimate.variant, estimate.barrier
        );
        let replace = match &chosen {
            None => true,
            Some((_, best, _)) => rate > *best,
        };
        if replace {
            chosen = Some((site, rate, estimate.trust));
        }
    }
    let (site, rate, trust) = chosen.expect("one event");
    println!("execute site {site} at {rate:.6e} Hz ({trust:?})");
    Ok(())
}
