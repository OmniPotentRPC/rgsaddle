//! Zoom-NEB: once the climbing image has kept its index and the band
//! force is low, every image is placed again on a window around it.
//!
//! eOn `NEBZoom.cpp` (TheochemUI/eOn 579738a9): the window grows from
//! the climbing image over the contiguous images above
//! `E_ref + alpha (E_max - E_ref)`, with `E_ref` the lower endpoint and
//! `E_max` the highest image. A window of the climbing image alone, or
//! an `alpha` of 0, takes `offset` images on each side. Unlike eOn, the
//! window then always reaches both neighbours of the climbing image, so
//! a one-sided cut cannot make the climber a fixed endpoint. The
//! window's end images become the fixed endpoints, and the band is placed on it
//! at equal arc length, along cubic Hermite segments or linearly, as
//! the OCI-NEB reparameterization does (Goswami, Gunde, and Jónsson,
//! Front. Chem. 14, 1807063 (2026), doi:10.3389/fchem.2026.1807063,
//! section 2.3.6). Cutting the band down to the barrier region is the
//! adaptive NEB of Maragakis, Andreev, Brumer, Reichman, and Kaxiras,
//! J. Chem. Phys. 117, 4651 (2002), doi:10.1063/1.1495401.

use ndarray::{Array1, Array2, ArrayView2, s};

use super::{BandConfig, BandSession, BandStatus, Cell};
use crate::error::SaddleError;
use crate::spring::SpringKind;

/// Zoom-NEB controls, eOn's `zoom_*` keys. Off unless set on
/// [`BandConfig::zoom`] or through [`BandSession::set_zoom`]. The
/// defaults are the recommended settings: `alpha` 0.5, `stability` 5,
/// and the zoom force at twice `force_tol`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ZoomConfig {
    /// Barrier fraction of the energy window (`zoom_alpha`); 0 takes
    /// the offset window (`zoom_mode = manual`).
    pub alpha: f64,
    /// Images on each side of the climbing image in the offset window.
    pub offset: usize,
    /// The zoom waits for the largest per-image force, each image
    /// reduced by the session's force gate as eOn's `convergenceForce`
    /// does, to fall under this (`zoom_after`). 0 takes twice
    /// `force_tol`; eOn's 0 takes ten times.
    pub after: f64,
    /// Band steps the climbing image keeps its index first.
    pub stability: usize,
    /// Cubic Hermite placement; `false` places linearly.
    pub cubic: bool,
}

impl Default for ZoomConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            offset: 1,
            after: 0.0,
            stability: 5,
            cubic: true,
        }
    }
}

/// The zoom needs a uniform spring, since weighted and Onsager-Machlup
/// constants belong to the images they were built for, and an atomic
/// band, since the window carries no cells.
pub(super) fn check_zoom(config: &BandConfig, zoom: &ZoomConfig) -> Result<(), SaddleError> {
    if !((0.0..1.0).contains(&zoom.alpha)
        && zoom.offset > 0
        && zoom.stability > 0
        && zoom.after.is_finite()
        && zoom.after >= 0.0)
    {
        return Err(SaddleError::Invalid(
            "zoom needs alpha in [0, 1), offset and stability of at least 1, \
             and a finite non-negative force"
                .into(),
        ));
    }
    if config.solid.is_some() || !matches!(config.spring, SpringKind::Uniform { .. }) {
        return Err(SaddleError::Invalid(
            "zoom needs a uniform spring on an atomic band".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct ZoomState {
    prev_ci: Option<usize>,
    stable: usize,
    window: Option<(usize, usize)>,
}

/// Inclusive window `(lo, hi)` around the interior climbing image `ci`:
/// eOn's window, widened to both neighbours of `ci`.
pub(super) fn select_window(energies: &[f64], ci: usize, zoom: &ZoomConfig) -> (usize, usize) {
    let (lo, hi) = eon_window(energies, ci, zoom);
    (
        lo.min(ci.saturating_sub(1)),
        hi.max((ci + 1).min(energies.len() - 1)),
    )
}

fn eon_window(energies: &[f64], ci: usize, zoom: &ZoomConfig) -> (usize, usize) {
    let n = energies.len();
    let offset = (
        ci.saturating_sub(zoom.offset),
        (ci + zoom.offset).min(n - 1),
    );
    let e_ref = energies[0].min(energies[n - 1]);
    let barrier = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max) - e_ref;
    if !(barrier > 0.0 && zoom.alpha > 0.0) {
        return offset;
    }
    let threshold = e_ref + zoom.alpha * barrier;
    let (mut lo, mut hi) = (ci, ci);
    while lo > 0 && energies[lo - 1] > threshold {
        lo -= 1;
    }
    while hi + 1 < n && energies[hi + 1] > threshold {
        hi += 1;
    }
    if hi > lo { (lo, hi) } else { offset }
}

/// `count` images at equal arc length along `rows`; the first and last
/// rows are kept as given. Steps between rows are minimum images.
pub(super) fn resample(
    rows: ArrayView2<f64>,
    count: usize,
    cubic: bool,
    cell: Option<&Cell>,
) -> Array2<f64> {
    let n = rows.nrows();
    let mut chain = rows.to_owned();
    let mut arc = vec![0.0; n];
    for i in 1..n {
        let mut d = &rows.row(i) - &rows.row(i - 1);
        if let Some(cell) = cell {
            cell.minimum_image(&mut d);
        }
        arc[i] = arc[i - 1] + d.dot(&d).sqrt();
        let next = &chain.row(i - 1) + &d;
        chain.row_mut(i).assign(&next);
    }
    let tangent = |i: usize| -> Array1<f64> {
        if i == 0 || i == n - 1 {
            let j = i.max(1);
            return &chain.row(j) - &chain.row(j - 1);
        }
        let t = (&chain.row(i + 1) - &chain.row(i - 1)) * 0.5;
        let norm = t.dot(&t).sqrt();
        if norm > 0.0 {
            t * ((arc[i + 1] - arc[i - 1]) / (2.0 * norm))
        } else {
            t
        }
    };
    let mut out = Array2::zeros((count, rows.ncols()));
    out.row_mut(0).assign(&rows.row(0));
    out.row_mut(count - 1).assign(&rows.row(n - 1));
    for k in 1..count - 1 {
        let target = arc[n - 1] * k as f64 / (count - 1) as f64;
        let lo = (1..n).find(|&j| arc[j] >= target).map_or(0, |j| j - 1);
        let hi = (lo + 1).min(n - 1);
        let span = arc[hi] - arc[lo];
        let f = if span > 1e-12 {
            (target - arc[lo]) / span
        } else {
            0.0
        };
        let (p0, p1) = (chain.row(lo), chain.row(hi));
        let row = if cubic {
            let (f2, f3) = (f * f, f * f * f);
            &p0 * (2.0 * f3 - 3.0 * f2 + 1.0)
                + &tangent(lo) * (f3 - 2.0 * f2 + f)
                + &p1 * (3.0 * f2 - 2.0 * f3)
                + &tangent(hi) * (f3 - f2)
        } else {
            &p0 * (1.0 - f) + &p1 * f
        };
        out.row_mut(k).assign(&row);
    }
    out
}

impl BandSession {
    /// Install, replace, or clear the zoom. The stability count starts
    /// again; a band that has zoomed may zoom once more.
    pub fn set_zoom(&mut self, zoom: Option<ZoomConfig>) -> Result<(), SaddleError> {
        if let Some(zoom) = &zoom {
            check_zoom(&self.config, zoom)?;
        }
        self.config.zoom = zoom;
        self.zoom = ZoomState::default();
        Ok(())
    }

    #[must_use]
    pub fn zoom(&self) -> Option<ZoomConfig> {
        self.config.zoom
    }

    /// The window `(lo, hi)` of the band before the zoom, whose images
    /// `lo` and `hi` became the endpoints; `None` before the zoom.
    #[must_use]
    pub fn zoom_window(&self) -> Option<(usize, usize)> {
        self.zoom.window
    }

    /// A zoom that is set and has not placed the band decides once the
    /// climbing image has kept its index for `stability` steps. Until
    /// then the min-mode refinement does not start, since its steps
    /// would bypass that decision.
    pub(super) fn zoom_pending(&self) -> bool {
        self.config
            .zoom
            .is_some_and(|zoom| self.zoom.window.is_none() && self.zoom.stable < zoom.stability)
    }

    /// After a band step: count the steps the climbing image kept its
    /// index and place the band on the window once the count and the
    /// force allow. The climbing image stays armed, the window's end
    /// energies are kept, and optimizer history and the refinement go.
    /// Returns whether the band moved.
    pub(super) fn zoom_latch(&mut self, status: BandStatus) -> bool {
        let (Some(zoom), None, Some(ci)) = (self.config.zoom, self.zoom.window, self.state.ci())
        else {
            return false;
        };
        if self.zoom.prev_ci == Some(ci) {
            self.zoom.stable += 1;
        } else {
            self.zoom = ZoomState {
                prev_ci: Some(ci),
                stable: 1,
                window: None,
            };
        }
        let limit = if zoom.after > 0.0 {
            zoom.after
        } else {
            2.0 * self.config.force_tol
        };
        if status != BandStatus::Running || self.zoom.stable < zoom.stability {
            return false;
        }
        let x = self.interior_flat();
        let (dof, gate) = (self.positions.ncols(), self.state.force_gate);
        let Some((energies, image_force)) = self
            .state
            .last
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .filter(|last| last.x == x)
            .map(|last| {
                let image_force = last
                    .projected
                    .exact_chunks(dof)
                    .into_iter()
                    .map(|f| gate.value(f))
                    .fold(0.0, f64::max);
                (last.energies.to_vec(), image_force)
            })
        else {
            return false;
        };
        if image_force >= limit {
            return false;
        }
        let n = energies.len();
        if ci == 0 || ci + 1 >= n || energies[ci] <= energies[0].max(energies[n - 1]) {
            return false;
        }
        let (lo, hi) = select_window(&energies, ci, &zoom);
        self.positions = resample(
            self.positions.slice(s![lo..=hi, ..]),
            n,
            zoom.cubic,
            self.config.cell.as_ref(),
        );
        self.state.set_endpoints(Some([energies[lo], energies[hi]]));
        self.state.drop_last();
        self.forget_optimizer();
        self.refine = super::min_mode::RefineState::default();
        if let Some(exp) = self.exp.as_mut() {
            exp.metrics.clear();
        }
        self.zoom.window = Some((lo, hi));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn manual(offset: usize) -> ZoomConfig {
        ZoomConfig {
            alpha: 0.0,
            offset,
            ..ZoomConfig::default()
        }
    }

    #[test]
    fn windows_follow_the_eon_cases() {
        assert_eq!(select_window(&[0.0; 11], 5, &manual(2)), (3, 7));
        let peak = [0.0, 0.2, 0.4, 1.2, 2.0, 1.2, 0.4, 0.2, 0.0];
        assert_eq!(select_window(&peak, 4, &ZoomConfig::default()), (3, 5));
        let spike = [0.0, 0.1, 0.2, 0.3, 5.0, 0.3, 0.2, 0.1, 0.0];
        assert_eq!(select_window(&spike, 4, &ZoomConfig::default()), (3, 5));
        assert_eq!(select_window(&spike, 1, &manual(3)), (0, 4));
    }

    #[test]
    fn a_one_sided_window_is_widened_around_the_climber() {
        // The threshold is 1.0: the right neighbour is above it and the
        // left one is not, so eOn's window ends on the climbing image.
        let shoulder = [0.0, 0.2, 0.5, 0.8, 2.0, 1.5, 0.3, 0.1, 0.0];
        let zoom = ZoomConfig::default();
        assert_eq!(eon_window(&shoulder, 4, &zoom), (4, 5));
        let (lo, hi) = select_window(&shoulder, 4, &zoom);
        assert!(lo < 4 && 4 < hi, "window ({lo}, {hi})");
        assert_eq!((lo, hi), (3, 5));
    }

    #[test]
    fn a_straight_window_is_packed_evenly_by_both_placements() {
        let line = Array2::from_shape_fn((11, 3), |(i, c)| if c == 0 { i as f64 } else { 0.0 });
        for cubic in [true, false] {
            let placed = resample(line.slice(s![3..=7, ..]), 11, cubic, None);
            for (i, row) in placed.outer_iter().enumerate() {
                assert_abs_diff_eq!(row[0], 3.0 + 0.4 * i as f64, epsilon = 1e-9);
            }
        }
    }
}
