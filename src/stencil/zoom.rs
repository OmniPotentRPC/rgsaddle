//! Zoom-NEB window and pack.
//!
//! eOn 3.4.0 selects the window (`client/NEBZoom.cpp`, TheochemUI/eOn,
//! doi:10.5281/zenodo.23166734). The bracket below is a deliberate
//! deviation: after the eOn window, `lo` is at most the image before
//! the climber and `hi` is at least the image after it, so the saddle
//! between those neighbours stays inside the window. eOn can return a
//! window that ends on the climber when only one neighbour is above
//! the threshold.
//!
//! The pack places the same image count on the window polyline.
//! Linear placement is eOn `linearResample`. Cubic placement is eOn
//! `resamplePath`: the tangent at a window end is the chord to its
//! neighbour, an interior tangent is the mean of the two chords,
//! normalised and scaled by half the chord from the previous image to
//! the next, and the Hermite parameter is the chord fraction. On a
//! curve the images are not exactly equally spaced.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::band::Cell;

/// How the window is chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ZoomWindow {
    /// Energy threshold, with an offset fallback.
    Auto { alpha: f64, offset: usize },
    /// `offset` images on each side of the climbing image.
    Manual { offset: usize },
}

impl Default for ZoomWindow {
    fn default() -> Self {
        Self::Auto {
            alpha: 0.5,
            offset: 1,
        }
    }
}

/// How images are placed on the window polyline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Interp {
    /// Chord fraction. ORCA's default.
    Linear,
    /// Cubic Hermite, eOn's default.
    #[default]
    Cubic,
}

/// What the window ends do after the pack.
///
/// [`ZoomEnds::Fixed`] is the default. The window ends stay where the
/// pack put them, as eOn does. [`ZoomEnds::FreePerp`] is ORCA
/// `Free_End_Type Perp`. The default changes to free ends only when
/// the paired force-call ratio against CI-NEB has a BCa 95% interval
/// that excludes 1 on at least three of the four benchmark surfaces
/// and the success rate is not lower.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ZoomEnds {
    /// Frozen window ends. eOn. The default.
    #[default]
    Fixed,
    /// Ends move on the perpendicular force only. ORCA `Free_End_Type Perp`.
    FreePerp,
}

/// When the climbing image is stable enough to pack.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ZoomForce {
    /// Multiple of the climbing-image tolerance. eOn default is 10.
    RelTol(f64),
    /// Absolute force in the host unit.
    Abs(f64),
}

impl Default for ZoomForce {
    fn default() -> Self {
        Self::RelTol(10.0)
    }
}

/// Zoom controls. New fields can be added without a breaking change.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ZoomParams {
    pub stability: usize,
    pub force: ZoomForce,
    pub window: ZoomWindow,
    pub interpolation: Interp,
    pub ends: ZoomEnds,
    /// Iterations after the pack. Zero keeps the caller's cap.
    pub max_iterations: usize,
}

impl Default for ZoomParams {
    fn default() -> Self {
        Self {
            stability: 5,
            force: ZoomForce::default(),
            window: ZoomWindow::default(),
            interpolation: Interp::Cubic,
            ends: ZoomEnds::Fixed,
            max_iterations: 0,
        }
    }
}

/// Stability counter. The count resets to 1 when the index changes.
#[derive(Clone, Debug)]
pub struct ZoomWatch {
    params: ZoomParams,
    last: Option<usize>,
    count: usize,
    armed: bool,
}

impl ZoomWatch {
    pub fn new(params: ZoomParams) -> Self {
        Self {
            params,
            last: None,
            count: 0,
            armed: false,
        }
    }

    pub fn params(&self) -> ZoomParams {
        self.params
    }

    pub fn fired(&self) -> bool {
        self.armed
    }

    pub fn count(&self) -> usize {
        self.count
    }

    /// Record one iteration. Returns true once, on the iteration the
    /// trigger first holds. A climber on an endpoint never triggers.
    pub fn observe(
        &mut self,
        n_images: usize,
        climbing: bool,
        ci: Option<usize>,
        max_force: f64,
        ci_tol: f64,
    ) -> bool {
        if self.armed || !climbing {
            return false;
        }
        let Some(ci) = ci else {
            self.last = None;
            self.count = 0;
            return false;
        };
        if ci == 0 || ci + 1 >= n_images {
            self.last = Some(ci);
            self.count = 0;
            return false;
        }
        if self.last == Some(ci) {
            self.count += 1;
        } else {
            self.last = Some(ci);
            self.count = 1;
        }
        let limit = match self.params.force {
            ZoomForce::RelTol(scale) => scale * ci_tol,
            ZoomForce::Abs(force) => force,
        };
        if self.count >= self.params.stability && max_force <= limit {
            self.armed = true;
            return true;
        }
        false
    }
}

/// eOn window, then the bracket that keeps the climber interior.
///
/// `None` when the climber is an endpoint or the band is shorter than
/// three images.
pub fn select_window(energies: &[f64], ci: usize, window: ZoomWindow) -> Option<(usize, usize)> {
    let n = energies.len();
    if n < 3 || ci == 0 || ci + 1 >= n {
        return None;
    }
    let (lo, hi) = eon_window(energies, ci, window)?;
    let lo = lo.min(ci - 1);
    let hi = hi.max(ci + 1);
    if hi <= lo || hi >= n {
        return None;
    }
    Some((lo, hi))
}

fn manual_window(n: usize, ci: usize, offset: usize) -> (usize, usize) {
    let half = offset.max(1);
    let lo = ci.saturating_sub(half);
    let hi = (ci + half).min(n - 1);
    (lo, hi)
}

fn eon_window(energies: &[f64], ci: usize, window: ZoomWindow) -> Option<(usize, usize)> {
    let n = energies.len();
    let (lo, hi) = match window {
        ZoomWindow::Manual { offset } => manual_window(n, ci, offset),
        ZoomWindow::Auto { alpha, offset } => {
            let e_low = if energies[0] <= energies[n - 1] {
                energies[0]
            } else {
                energies[n - 1]
            };
            let mut e_max = energies[0];
            for energy in energies.iter().skip(1) {
                if *energy > e_max {
                    e_max = *energy;
                }
            }
            let barrier = e_max - e_low;
            let usable = barrier > 0.0 && alpha > 0.0 && alpha < 1.0;
            if !usable {
                manual_window(n, ci, offset)
            } else {
                let threshold = e_low + alpha * barrier;
                let mut lo = ci;
                let mut hi = ci;
                while lo > 0 && energies[lo - 1] > threshold {
                    lo -= 1;
                }
                while hi + 1 < n && energies[hi + 1] > threshold {
                    hi += 1;
                }
                if hi <= lo {
                    manual_window(n, ci, offset)
                } else {
                    (lo, hi)
                }
            }
        }
    };
    if hi <= lo || hi >= n {
        None
    } else {
        Some((lo, hi))
    }
}

/// Place `count` images on the polyline `positions[lo..=hi]`.
///
/// Arc length is the cumulative minimum-image chord, summed in image
/// order. The first and last rows are the window ends.
pub fn pack_window(
    positions: ArrayView2<f64>,
    lo: usize,
    hi: usize,
    count: usize,
    interpolation: Interp,
    cell: Option<&Cell>,
) -> Array2<f64> {
    let m = hi - lo + 1;
    let dof = positions.ncols();
    let mut unwrapped = Array2::zeros((m, dof));
    unwrapped.row_mut(0).assign(&positions.row(lo));
    let mut arc = vec![0.0; m];
    for i in 1..m {
        let mut diff = &positions.row(lo + i) - &positions.row(lo + i - 1);
        if let Some(cell) = cell {
            cell.minimum_image(&mut diff);
        }
        let next = &unwrapped.row(i - 1) + &diff;
        unwrapped.row_mut(i).assign(&next);
        arc[i] = arc[i - 1] + diff.dot(&diff).sqrt();
    }
    let mut out = Array2::zeros((count, dof));
    if count == 0 {
        return out;
    }
    out.row_mut(0).assign(&unwrapped.row(0));
    if count == 1 {
        return out;
    }
    out.row_mut(count - 1).assign(&unwrapped.row(m - 1));
    let total = arc[m - 1];
    let tangents = if interpolation == Interp::Cubic {
        Some(hermite_tangents(unwrapped.view(), &arc))
    } else {
        None
    };
    for i in 1..count - 1 {
        let target = if total > 1e-12 {
            total * (i as f64) / (count - 1) as f64
        } else {
            0.0
        };
        let mut seg = 0usize;
        for (j, length) in arc.iter().enumerate().skip(1) {
            if *length >= target {
                seg = j - 1;
                break;
            }
        }
        let high = (seg + 1).min(m - 1);
        let span = arc[high] - arc[seg];
        let frac_floor = if interpolation == Interp::Cubic {
            1e-10
        } else {
            1e-12
        };
        let frac = if span > frac_floor {
            (target - arc[seg]) / span
        } else {
            0.0
        };
        let placed = if let Some(tangents) = &tangents {
            hermite(
                unwrapped.row(seg),
                tangents.row(seg),
                unwrapped.row(high),
                tangents.row(high),
                frac,
            )
        } else {
            &unwrapped.row(seg) * (1.0 - frac) + &unwrapped.row(high) * frac
        };
        out.row_mut(i).assign(&placed);
    }
    out
}

fn hermite_tangents(points: ArrayView2<f64>, arc: &[f64]) -> Array2<f64> {
    let n = points.nrows();
    let mut tangents = Array2::zeros(points.dim());
    for i in 0..n {
        if i == 0 {
            tangents
                .row_mut(i)
                .assign(&(&points.row(1) - &points.row(0)));
        } else if i + 1 == n {
            tangents
                .row_mut(i)
                .assign(&(&points.row(i) - &points.row(i - 1)));
        } else {
            let d_next = &points.row(i + 1) - &points.row(i);
            let d_prev = &points.row(i) - &points.row(i - 1);
            let mut tangent = (&d_next + &d_prev) * 0.5;
            let local = (arc[i + 1] - arc[i - 1]) / 2.0;
            let norm = tangent.dot(&tangent).sqrt();
            if norm > 0.0 {
                tangent *= local / norm;
            }
            tangents.row_mut(i).assign(&tangent);
        }
    }
    tangents
}

fn hermite(
    p0: ArrayView1<f64>,
    t0: ArrayView1<f64>,
    p1: ArrayView1<f64>,
    t1: ArrayView1<f64>,
    frac: f64,
) -> Array1<f64> {
    let f2 = frac * frac;
    let f3 = f2 * frac;
    let h00 = 2.0 * f3 - 3.0 * f2 + 1.0;
    let h10 = f3 - 2.0 * f2 + frac;
    let h01 = -2.0 * f3 + 3.0 * f2;
    let h11 = f3 - f2;
    &p0 * h00 + &t0 * h10 + &p1 * h01 + &t1 * h11
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn manual_offset_on_a_flat_band() {
        let energies = [0.0; 11];
        assert_eq!(
            select_window(&energies, 5, ZoomWindow::Manual { offset: 2 }),
            Some((3, 7))
        );
    }

    #[test]
    fn auto_window_matches_the_eon_crest() {
        let energies = [0.0, 0.2, 0.4, 1.2, 2.0, 1.2, 0.4, 0.2, 0.0];
        assert_eq!(
            select_window(
                &energies,
                4,
                ZoomWindow::Auto {
                    alpha: 0.5,
                    offset: 1
                }
            ),
            Some((3, 5))
        );
    }

    #[test]
    fn auto_window_falls_back_when_the_neighbours_are_low() {
        let energies = [0.0, 0.1, 0.2, 0.3, 5.0, 0.3, 0.2, 0.1, 0.0];
        assert_eq!(
            select_window(
                &energies,
                4,
                ZoomWindow::Auto {
                    alpha: 0.5,
                    offset: 1
                }
            ),
            Some((3, 5))
        );
    }

    #[test]
    fn an_endpoint_climber_has_no_window() {
        let energies = [1.0, 0.2, 0.0, 0.2, 0.4];
        assert_eq!(
            select_window(&energies, 0, ZoomWindow::Manual { offset: 1 }),
            None
        );
        assert_eq!(
            select_window(
                &energies,
                4,
                ZoomWindow::Auto {
                    alpha: 0.5,
                    offset: 1
                }
            ),
            None
        );
    }

    #[test]
    fn the_bracket_keeps_an_asymmetric_climber_interior() {
        // eOn's auto rule returns [4, 5]. The bracket pulls the left
        // side back to the image before the climber.
        let energies = [0.0, 0.1, 0.2, 0.4, 2.0, 1.5, 0.3, 0.1, 0.0];
        assert_eq!(
            select_window(
                &energies,
                4,
                ZoomWindow::Auto {
                    alpha: 0.5,
                    offset: 1
                }
            ),
            Some((3, 5))
        );
    }

    #[test]
    fn straight_window_packs_at_equal_spacing() {
        let mut rows = Vec::new();
        for i in 0..11 {
            rows.extend_from_slice(&[i as f64, 0.0, 0.0]);
        }
        let positions = Array2::from_shape_vec((11, 3), rows).unwrap();
        for interpolation in [Interp::Cubic, Interp::Linear] {
            let packed = pack_window(positions.view(), 3, 7, 11, interpolation, None);
            assert!((packed[(0, 0)] - 3.0).abs() < 1e-12);
            assert!((packed[(10, 0)] - 7.0).abs() < 1e-12);
            for i in 0..10 {
                let step = packed[(i + 1, 0)] - packed[(i, 0)];
                assert!(
                    (step - 0.4).abs() < 1e-12,
                    "{interpolation:?} step {i} = {step}"
                );
            }
        }
        let linear = pack_window(positions.view(), 3, 7, 11, Interp::Linear, None);
        let cubic = pack_window(positions.view(), 3, 7, 11, Interp::Cubic, None);
        for (left, right) in linear.iter().zip(cubic.iter()) {
            assert!((left - right).abs() < 1e-12);
        }
    }

    #[test]
    fn stability_resets_when_the_index_changes_and_fires_once() {
        let mut watch = ZoomWatch::new(ZoomParams {
            stability: 3,
            force: ZoomForce::Abs(1.0),
            ..ZoomParams::default()
        });
        assert!(!watch.observe(7, true, Some(3), 0.1, 1e-3));
        assert!(!watch.observe(7, true, Some(3), 0.1, 1e-3));
        assert!(!watch.observe(7, true, Some(4), 0.1, 1e-3));
        assert_eq!(watch.count(), 1);
        assert!(!watch.observe(7, true, Some(4), 0.1, 1e-3));
        assert!(watch.observe(7, true, Some(4), 0.1, 1e-3));
        assert!(!watch.observe(7, true, Some(4), 0.1, 1e-3));
        assert!(!watch.observe(7, false, Some(4), 0.1, 1e-3));
        assert!(!watch.observe(7, true, Some(0), 0.0, 1e-3));
    }

    #[test]
    fn curved_pack_matches_the_eon_hermite_golden() {
        let text = include_str!("../../validation/eon_zoom/hermite_golden.json");
        let packed_key = "\"packed\"";
        let start = text.find(packed_key).expect("packed");
        let body = &text[start + packed_key.len()..];
        let open = body.find('[').expect("open");
        let close = body.rfind(']').expect("close");
        let mut numbers = Vec::new();
        let mut token = String::new();
        for ch in body[open..=close].chars() {
            if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' || ch == '+'
            {
                token.push(ch);
            } else if !token.is_empty() {
                numbers.push(token.parse::<f64>().unwrap());
                token.clear();
            }
        }
        assert_eq!(numbers.len(), 11 * 3);
        let dense_key = "\"dense\"";
        let dense_at = text.find(dense_key).expect("dense");
        let dense_body = &text[dense_at + dense_key.len()..text.find(packed_key).unwrap()];
        let mut dense_numbers = Vec::new();
        token.clear();
        for ch in dense_body.chars() {
            if ch.is_ascii_digit() || ch == '.' || ch == '-' || ch == 'e' || ch == 'E' || ch == '+'
            {
                token.push(ch);
            } else if !token.is_empty() {
                dense_numbers.push(token.parse::<f64>().unwrap());
                token.clear();
            }
        }
        let dense = Array2::from_shape_vec((6, 3), dense_numbers).unwrap();
        let got = pack_window(dense.view(), 0, 5, 11, Interp::Cubic, None);
        for (got, want) in got.iter().zip(numbers.iter()) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
        let _ = array![0.0];
    }
}
