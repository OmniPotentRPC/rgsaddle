//! Band presets on the shared stencil: springs, zoom, refused presets, ids.

use ndarray::{Array1, Array2, ArrayViewMut2};
use rgsaddle::SaddleError;
use rgsaddle::stencil::spring_ew::fill_constants;
use rgsaddle::stencil::{ImageId, PHASE_ZOOM, Preset, Stencil, Surface, ZoomForce, ZoomParams};

fn line(images: usize) -> Array2<f64> {
    let mut band = Array2::zeros((images, 3));
    for image in 0..images {
        let t = image as f64 / (images - 1) as f64;
        band[[image, 0]] = -1.0 + 2.0 * t;
    }
    band
}

struct Well {
    ids: std::sync::Mutex<Vec<Vec<ImageId>>>,
}

impl Surface for Well {
    fn eval(
        &self,
        ids: &[ImageId],
        positions: ndarray::ArrayView2<f64>,
        energies: &mut [f64],
        mut gradients: ArrayViewMut2<f64>,
    ) -> Result<(), SaddleError> {
        self.ids.lock().expect("ids").push(ids.to_vec());
        for (row, energy) in energies.iter_mut().enumerate() {
            let x = positions[[row, 0]];
            let y = positions[[row, 1]];
            let z = positions[[row, 2]];
            *energy = (x * x - 1.0).powi(2) + 2.0 * y * y + 2.0 * z * z;
            gradients[[row, 0]] = 4.0 * x * (x * x - 1.0);
            gradients[[row, 1]] = 4.0 * y;
            gradients[[row, 2]] = 4.0 * z;
        }
        Ok(())
    }
}

fn must_err(result: Result<Stencil, SaddleError>) -> String {
    match result {
        Ok(_) => panic!("expected a build error"),
        Err(err) => err.to_string(),
    }
}

#[test]
fn energy_weighted_constants_match_the_assembly() {
    let energies = [0.0, 0.2, 1.0, 0.4, 0.0];
    let mut ks = vec![0.0; 4];
    fill_constants(&energies, 0.972, 9.72, &mut ks);
    assert!(ks.iter().all(|k| *k >= 0.972 && *k <= 9.72));
    assert!(ks[1] > ks[0]);
}

#[test]
fn refused_presets_name_the_missing_piece() {
    let band = line(5);
    let idpp = must_err(Stencil::band(band.clone()).preset(Preset::Idpp).build());
    assert!(idpp.contains("at least two atoms"));
    let sidpp = must_err(Stencil::band(band.clone()).preset(Preset::SIdpp).build());
    assert!(sidpp.contains("at least two atoms"));
    Stencil::band(band)
        .preset(Preset::OciNeb)
        .build()
        .expect("oci-neb");
    let point = must_err(Stencil::point(Array1::zeros(3)).preset(Preset::Neb).build());
    assert!(point.contains("band preset"));
    for text in [must_err(
        Stencil::band(line(4)).preset(Preset::Idpp).build(),
    )] {
        assert!(!text.contains("rgsaddle#"));
        assert!(!text.contains("feat/"));
    }
}

#[test]
fn zoom_pack_keeps_the_window_end_ids() {
    let well = Well {
        ids: std::sync::Mutex::new(Vec::new()),
    };
    let mut params = ZoomParams::default();
    params.stability = 1;
    params.force = ZoomForce::Abs(1.0e9);
    let mut stencil = Stencil::band(line(7))
        .preset(Preset::ZoomNeb)
        .zoom(params)
        .climb_trigger(0.5, 1.0e300)
        .build()
        .unwrap();
    let mut saw = false;
    for _ in 0..8 {
        let report = stencil.step_surface(&well).unwrap();
        if report.phase == PHASE_ZOOM {
            saw = true;
            assert_eq!(report.evaluations, 0);
            let (lo, hi) = report.zoom_window.expect("window");
            assert!(lo < hi);
            let ids = stencil.image_ids();
            assert_eq!(ids.len(), 7);
            assert_ne!(ids[0], ids[1]);
            assert_ne!(ids[6], ids[5]);
            break;
        }
    }
    assert!(saw, "zoom did not pack");
    let seen = well.ids.lock().expect("ids");
    assert!(seen.iter().any(|row| row.iter().any(|id| *id == 0)));
    assert!(seen.iter().any(|row| row.iter().any(|id| *id == 6)));
}

#[test]
fn autoneb_callbacks_carry_stable_ids() {
    let well = Well {
        ids: std::sync::Mutex::new(Vec::new()),
    };
    let mut stencil = Stencil::band(line(2))
        .preset(Preset::AutoNeb)
        .autoneb_limits(3, 6)
        .force_tol(1.0e-2)
        .build()
        .unwrap();
    let _ = stencil.step_surface(&well).unwrap();
    let seen = well.ids.lock().expect("ids");
    assert!(!seen.is_empty());
    let mut all = Vec::new();
    for row in seen.iter() {
        if row.len() > 1 {
            assert!(
                row.iter().any(|id| *id != 0),
                "a multi-row callback was all zeros: {row:?}"
            );
        }
        let mut sorted = row.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), row.len(), "duplicate ids in {row:?}");
        all.extend(row.iter().copied());
    }
    all.sort_unstable();
    all.dedup();
    assert!(
        all.len() > 1,
        "callbacks did not carry distinct image ids: {all:?}"
    );
}

#[test]
fn ci_neb_dneb_and_weighted_build() {
    let band = line(5);
    Stencil::band(band.clone())
        .preset(Preset::CiNeb)
        .build()
        .unwrap()
        .step_surface(&Well {
            ids: std::sync::Mutex::new(Vec::new()),
        })
        .unwrap();
    Stencil::band(band.clone())
        .preset(Preset::Dneb)
        .build()
        .unwrap()
        .step_surface(&Well {
            ids: std::sync::Mutex::new(Vec::new()),
        })
        .unwrap();
    Stencil::band(band)
        .preset(Preset::EnergyWeightedNeb)
        .build()
        .unwrap()
        .step_surface(&Well {
            ids: std::sync::Mutex::new(Vec::new()),
        })
        .unwrap();
}
