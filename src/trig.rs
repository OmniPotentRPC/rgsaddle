//! Elementary functions for result paths.
//!
//! Host sine, cosine, and inverse functions can differ between x86_64
//! and aarch64. These call the musl port in `libm`, which is the same
//! code on both.

pub(crate) fn sin(x: f64) -> f64 {
    libm::sin(x)
}

pub(crate) fn cos(x: f64) -> f64 {
    libm::cos(x)
}

pub(crate) fn atan(x: f64) -> f64 {
    libm::atan(x)
}

pub(crate) fn acos(x: f64) -> f64 {
    libm::acos(x)
}

pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
    libm::sincos(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bytes() -> Vec<u8> {
        let mut bytes = Vec::new();
        for i in 0..16 {
            let x = (i as f64 - 8.0) * 0.37;
            for value in [sin(x), cos(x), atan(x)] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            let argument = (x * 0.1).clamp(-1.0, 1.0);
            bytes.extend_from_slice(&acos(argument).to_le_bytes());
            let (sine, cosine) = sin_cos(x);
            bytes.extend_from_slice(&sine.to_le_bytes());
            bytes.extend_from_slice(&cosine.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn elementary_results_match_at_one_two_and_eight_threads() {
        let mut digests = Vec::new();
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            digests.push(pool.install(sample_bytes));
        }
        assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
        use sha2::Digest;
        let hash = sha2::Sha256::digest(&digests[0]);
        let hex = hash.iter().map(|b| format!("{b:02x}")).collect::<String>();
        assert_eq!(
            hex,
            "5ced084431f34dae9208550c3b7dfaa7ed41736b7def5700c551dcf2029e1ad2"
        );
    }

    #[test]
    fn dneb_and_dimer_bytes_match_across_threads() {
        use crate::SaddleError;
        use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, PointSurface};
        use crate::projection::dneb_component;
        use ndarray::{Array1, ArrayView1, array};

        struct Quadratic;
        impl PointSurface for Quadratic {
            fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
                let g = array![-2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
                Ok((-x[0] * x[0] + x[1] * x[1] + x[2] * x[2], g))
            }
        }

        let mut digests = Vec::new();
        for threads in [1usize, 2, 3, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let digest = pool.install(|| {
                let dneb = dneb_component(
                    array![0.2, -0.4, 0.1].view(),
                    array![1.0, 0.0, 0.0].view(),
                    array![0.0, 0.3, -0.2].view(),
                );
                let config = MinModeConfig {
                    kind: MinModeKind::Dimer,
                    rotation_tol: 1e-6,
                    max_rotations: 20,
                    ..MinModeConfig::default()
                };
                let mut session =
                    MinModeSession::new(config, array![0.2, -0.1, 0.05], array![0.3, 0.4, 0.5])
                        .unwrap();
                let estimate = session.estimate_mode(&Quadratic).unwrap();
                let mut bytes = Vec::new();
                for value in dneb.iter().chain(estimate.mode.iter()) {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                bytes.extend_from_slice(&estimate.curvature.to_le_bytes());
                bytes
            });
            digests.push(digest);
        }
        assert!(digests.windows(2).all(|pair| pair[0] == pair[1]));
        use sha2::Digest;
        let hex = sha2::Sha256::digest(&digests[0])
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        assert_eq!(
            hex,
            "8616d26dbaaa772f5bc10218d58b6235ab69f865f6ed2207a9ab5961035f304d"
        );
    }
}
