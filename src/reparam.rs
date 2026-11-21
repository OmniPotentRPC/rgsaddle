//! Equal arc-length reparametrization of a band or string.
//!
//! Both schemes measure the arc length `s` of the current polyline, the
//! sum of its chords, and keep the endpoints fixed.
//! [`Reparametrization::Linear`] places the interior images at equal
//! steps of `s` on the polyline. [`Reparametrization::NotAKnot`] fits
//! the not-a-knot cubic spline through the images in the parameter `s`,
//! the interpolation of E, Ren, and Vanden-Eijnden, J. Chem. Phys. 126,
//! 164103 (2007), doi:10.1063/1.2720838, Sec. II, where the string is
//! `φ(α)` with `α` the normalized polyline arc length. The paper then
//! takes the images at uniform `α`. Here they go to equal steps of the
//! spline's own arc length, measured by Gauss-Legendre quadrature, so an
//! unevenly spaced band is spaced evenly along the curve in one pass.
//! Both placements lie on the same curve.
//!
//! The spline reproduces a cubic polynomial in `s` exactly, so a band
//! whose images lie on a straight line stays on it. With three distinct
//! images it is the interpolating parabola and with two the chord.
//! Images whose arc lengths agree to `1e-12` of the total are one knot
//! there; the anchor and the last image keep their own positions. In a
//! periodic cell the chords are minimum images: the chain is unwrapped
//! from the first image before the fit, and each placed image is moved
//! by the minimum image of its displacement.
//!
//! ```
//! use ndarray::array;
//! use rgsaddle::{Reparametrization, reparametrize};
//! let mut band = array![[0.0, 0.0], [0.1, 0.0], [0.5, 0.0], [2.0, 0.0]];
//! reparametrize(&mut band, None, None, Reparametrization::NotAKnot);
//! for (j, row) in band.outer_iter().enumerate() {
//!     assert!((row[0] - 2.0 * j as f64 / 3.0).abs() < 1e-12);
//! }
//! ```

use std::cmp::Ordering;

use ndarray::{Array1, Array2};

use crate::band::Cell;

/// How interior images are placed at equal arc length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Reparametrization {
    /// On the polyline through the images.
    #[default]
    Linear,
    /// On the not-a-knot cubic spline through the images in their
    /// polyline arc length (E, Ren, and Vanden-Eijnden 2007).
    NotAKnot,
}

impl Reparametrization {
    /// The discriminant used by the C and Python interfaces.
    pub fn to_abi(self) -> i32 {
        match self {
            Self::Linear => 0,
            Self::NotAKnot => 1,
        }
    }

    /// Inverse of [`Self::to_abi`].
    pub fn from_abi(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Linear),
            1 => Some(Self::NotAKnot),
            _ => None,
        }
    }
}

/// Place the interior images at equal arc length. With `anchor` an
/// interior image, each side of it is spaced on its own and the anchor
/// stays. Endpoints never move. `cell` makes the chords minimum images.
pub fn reparametrize(
    positions: &mut Array2<f64>,
    anchor: Option<usize>,
    cell: Option<&Cell>,
    scheme: Reparametrization,
) {
    match scheme {
        Reparametrization::Linear => crate::rtr::reparametrize_linear(positions, anchor, cell),
        Reparametrization::NotAKnot => not_a_knot_equal_arc(positions, anchor, cell),
    }
}

/// Knots closer than this fraction of the total arc length are one knot.
const COINCIDENT: f64 = 1e-12;

/// [`solve_dense`] takes a pivot at or below this fraction of the largest
/// entry as zero.
#[cfg(test)]
const PIVOT_TOLERANCE: f64 = 1e-14;

/// A target arc this close to an end of the curve, as a fraction of the
/// total arc length, is that end, and one further out is refused.
const ARC_END_TOLERANCE: f64 = 1e-9;

/// [`Spline::at_arc`] stops when the arc at the iterate is within this
/// fraction of the total arc length of the target.
const ARC_MISS_TOLERANCE: f64 = 1e-14;

/// [`Spline::at_arc`] stops when its bracket is narrower than this fraction
/// of the parameter span.
const PARAMETER_BRACKET_TOLERANCE: f64 = 1e-15;

fn not_a_knot_equal_arc(positions: &mut Array2<f64>, anchor: Option<usize>, cell: Option<&Cell>) {
    let n = positions.nrows();
    if n < 3 {
        return;
    }
    let original = positions.clone();
    let mut chain = original.clone();
    let mut s = vec![0.0; n];
    for i in 1..n {
        let mut d = &original.row(i) - &original.row(i - 1);
        if let Some(cell) = cell {
            cell.minimum_image(&mut d);
        }
        let unwrapped = &chain.row(i - 1) + &d;
        chain.row_mut(i).assign(&unwrapped);
        s[i] = s[i - 1] + d.dot(&d).sqrt();
    }
    let total = s[n - 1];
    if !(total.is_finite() && total > 0.0) {
        return;
    }
    let anchor = anchor.filter(|&a| a > 0 && a < n - 1);
    let kept = distinct_knots(&s, anchor, COINCIDENT * total);
    let knots: Vec<f64> = kept.iter().map(|&i| s[i]).collect();
    let mut values = Array2::zeros((kept.len(), chain.ncols()));
    for (row, &i) in kept.iter().enumerate() {
        values.row_mut(row).assign(&chain.row(i));
    }
    let Some(spline) = Spline::fit(&knots, &values) else {
        return;
    };
    // Arc length of the spline itself at each knot. The images go to
    // equal steps of it, so a fit through unevenly spaced images still
    // places them evenly along the curve.
    let arc = spline.knot_arcs();
    let length = arc[arc.len() - 1];
    if !(length.is_finite() && length > 0.0) {
        return;
    }
    let mut targets: Vec<(usize, f64)> = Vec::with_capacity(n);
    let split = anchor.and_then(|a| kept.iter().position(|&i| i == a).map(|k| (a, arc[k])));
    match split {
        Some((a, at)) => {
            let left = at / a as f64;
            targets.extend((1..a).map(|j| (j, j as f64 * left)));
            let right = (length - at) / (n - 1 - a) as f64;
            targets.extend((a + 1..n - 1).map(|j| (j, at + (j - a) as f64 * right)));
        }
        None => {
            let h = length / (n - 1) as f64;
            targets.extend(
                (1..n - 1)
                    .filter(|&j| Some(j) != anchor)
                    .map(|j| (j, j as f64 * h)),
            );
        }
    }
    // A target the curve cannot reach leaves every image where it was.
    let mut placed = Vec::with_capacity(targets.len());
    for (j, target) in targets {
        let Some(at) = spline.at_arc(&arc, target) else {
            return;
        };
        let mut difference = &spline.eval(at) - &original.row(j);
        if let Some(cell) = cell {
            cell.minimum_image(&mut difference);
        }
        placed.push((j, &original.row(j) + &difference));
    }
    for (j, row) in placed {
        positions.row_mut(j).assign(&row);
    }
}

/// Indices of the images that become knots. A knot within `tol` of the
/// previous one is dropped, unless it is the anchor or the last image,
/// which then replaces the previous knot when that one is neither the
/// first image nor the anchor.
fn distinct_knots(s: &[f64], anchor: Option<usize>, tol: f64) -> Vec<usize> {
    let n = s.len();
    let pinned = |i: usize| i == 0 || i == n - 1 || Some(i) == anchor;
    let mut kept: Vec<usize> = vec![0];
    for i in 1..n {
        let last = kept[kept.len() - 1];
        if s[i] - s[last] > tol {
            kept.push(i);
        } else if pinned(i) && !pinned(last) {
            let end = kept.len() - 1;
            kept[end] = i;
        }
    }
    kept
}

/// Interpolating spline through knots `t` (strictly increasing) and the
/// rows of `values`: the chord for two knots, the parabola for three,
/// and the not-a-knot cubic spline for four or more.
#[derive(Clone, Debug)]
pub(crate) struct Spline {
    t: Vec<f64>,
    y: Array2<f64>,
    /// Second derivatives at the knots; zero for the chord.
    m: Array2<f64>,
}

impl Spline {
    pub(crate) fn fit(t: &[f64], y: &Array2<f64>) -> Option<Self> {
        let k = t.len();
        if k < 2
            || y.nrows() != k
            || t.windows(2)
                .any(|w| w[1].partial_cmp(&w[0]) != Some(Ordering::Greater))
        {
            return None;
        }
        let mut m = Array2::zeros(y.raw_dim());
        if k == 3 {
            // The parabola has one second derivative, twice the second
            // divided difference, at every knot.
            let d0 = (&y.row(1) - &y.row(0)) / (t[1] - t[0]);
            let d1 = (&y.row(2) - &y.row(1)) / (t[2] - t[1]);
            let second = (&d1 - &d0) * (2.0 / (t[2] - t[0]));
            for mut row in m.outer_iter_mut() {
                row.assign(&second);
            }
        } else if k >= 4 {
            m = not_a_knot_moments(t, y)?;
        }
        Some(Self {
            t: t.to_vec(),
            y: y.clone(),
            m,
        })
    }

    fn segment(&self, at: f64) -> usize {
        let k = self.t.len();
        let mut seg = 0;
        while seg + 2 < k && self.t[seg + 1] < at {
            seg += 1;
        }
        seg
    }

    /// `|dS/dt|` at `at`, inside segment `seg`.
    fn speed(&self, seg: usize, at: f64) -> f64 {
        let h = self.t[seg + 1] - self.t[seg];
        let b = (at - self.t[seg]) / h;
        let a = 1.0 - b;
        let (ca, cb) = (
            -(3.0 * a * a - 1.0) * h / 6.0,
            (3.0 * b * b - 1.0) * h / 6.0,
        );
        let (y0, y1) = (self.y.row(seg), self.y.row(seg + 1));
        let (m0, m1) = (self.m.row(seg), self.m.row(seg + 1));
        let mut sum = 0.0;
        for col in 0..self.y.ncols() {
            let d = (y1[col] - y0[col]) / h + ca * m0[col] + cb * m1[col];
            sum += d * d;
        }
        sum.sqrt()
    }

    /// Arc length from `from` to `to` inside segment `seg`, by five-point
    /// Gauss-Legendre quadrature on each half of the interval.
    fn arc_between(&self, seg: usize, from: f64, to: f64) -> f64 {
        const NODES: [f64; 5] = [
            0.0,
            -0.538_469_310_105_683,
            0.538_469_310_105_683,
            -0.906_179_845_938_664,
            0.906_179_845_938_664,
        ];
        const WEIGHTS: [f64; 5] = [
            0.568_888_888_888_889,
            0.478_628_670_499_366,
            0.478_628_670_499_366,
            0.236_926_885_056_189,
            0.236_926_885_056_189,
        ];
        let mid = 0.5 * (from + to);
        [(from, mid), (mid, to)]
            .into_iter()
            .map(|(lo, hi)| {
                let (centre, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
                half * NODES
                    .iter()
                    .zip(WEIGHTS)
                    .map(|(&x, w)| w * self.speed(seg, centre + half * x))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Cumulative arc length of the spline at each knot.
    pub(crate) fn knot_arcs(&self) -> Vec<f64> {
        let mut arc = vec![0.0; self.t.len()];
        for seg in 0..self.t.len() - 1 {
            arc[seg + 1] = arc[seg] + self.arc_between(seg, self.t[seg], self.t[seg + 1]);
        }
        arc
    }

    /// First derivative `dS/dt` at parameter `at`.
    pub(crate) fn derivative(&self, at: f64) -> Array1<f64> {
        let seg = self.segment(at);
        let h = self.t[seg + 1] - self.t[seg];
        let b = (at - self.t[seg]) / h;
        let a = 1.0 - b;
        let (ca, cb) = (
            -(3.0 * a * a - 1.0) * h / 6.0,
            (3.0 * b * b - 1.0) * h / 6.0,
        );
        let (y0, y1) = (self.y.row(seg), self.y.row(seg + 1));
        let (m0, m1) = (self.m.row(seg), self.m.row(seg + 1));
        let mut out = Array1::zeros(self.y.ncols());
        for (col, slot) in out.iter_mut().enumerate() {
            *slot = (y1[col] - y0[col]) / h + ca * m0[col] + cb * m1[col];
        }
        out
    }

    /// The parameter at which the spline's arc length is `target`, from
    /// the knot arcs `arc`: Newton on the arc, kept inside a bisection
    /// bracket of the segment.
    ///
    /// A target within [`ARC_END_TOLERANCE`] of an end of the curve, or
    /// beyond it by no more than that, is that end's knot. A target further
    /// out, or one that is not finite, has no parameter and gives `None`,
    /// as does a set of knot arcs that does not rise.
    pub(crate) fn at_arc(&self, arc: &[f64], target: f64) -> Option<f64> {
        let k = self.t.len();
        if arc.len() != k {
            return None;
        }
        let (first, length) = (arc[0], arc[k - 1]);
        let ordered = arc.windows(2).all(|pair| pair[0] <= pair[1]);
        if !(ordered && length.is_finite() && length > first && target.is_finite()) {
            return None;
        }
        let slack = ARC_END_TOLERANCE * (length - first);
        if target < first - slack || target > length + slack {
            return None;
        }
        if target <= first {
            return Some(self.t[0]);
        }
        if target >= length {
            return Some(self.t[k - 1]);
        }
        let mut seg = 0;
        while seg + 2 < k && arc[seg + 1] < target {
            seg += 1;
        }
        let (mut lo, mut hi) = (self.t[seg], self.t[seg + 1]);
        let span = arc[seg + 1] - arc[seg];
        if span.partial_cmp(&0.0) != Some(Ordering::Greater) {
            return Some(lo);
        }
        let mut t = lo + (hi - lo) * ((target - arc[seg]) / span);
        for _ in 0..60 {
            let miss = arc[seg] + self.arc_between(seg, self.t[seg], t) - target;
            if miss.abs() <= ARC_MISS_TOLERANCE * length {
                break;
            }
            if miss > 0.0 {
                hi = t;
            } else {
                lo = t;
            }
            let speed = self.speed(seg, t);
            let newton = t - miss / speed;
            t = if speed > 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if hi - lo <= PARAMETER_BRACKET_TOLERANCE * (self.t[k - 1] - self.t[0]) {
                break;
            }
        }
        Some(t)
    }

    /// The knots.
    pub(crate) fn knots(&self) -> &[f64] {
        &self.t
    }

    /// The interpolated rows, one per knot.
    pub(crate) fn values(&self) -> &Array2<f64> {
        &self.y
    }

    /// `S`, `dS/dt`, `d²S/dt²` and the one-sided `d³S/dt³` at `at`.
    pub(crate) fn derivatives(&self, at: f64) -> [Array1<f64>; 4] {
        let seg = self.segment(at);
        let h = self.t[seg + 1] - self.t[seg];
        let b = (at - self.t[seg]) / h;
        let a = 1.0 - b;
        let (bend_a, bend_b) = ((a * a * a - a) * h * h / 6.0, (b * b * b - b) * h * h / 6.0);
        let (slope_a, slope_b) = (
            -(3.0 * a * a - 1.0) * h / 6.0,
            (3.0 * b * b - 1.0) * h / 6.0,
        );
        let (y0, y1) = (self.y.row(seg), self.y.row(seg + 1));
        let (m0, m1) = (self.m.row(seg), self.m.row(seg + 1));
        let value = &y0 * a + &y1 * b + &m0 * bend_a + &m1 * bend_b;
        let slope = (&y1 - &y0) / h + &m0 * slope_a + &m1 * slope_b;
        let second = &m0 * a + &m1 * b;
        let third = (&m1 - &m0) / h;
        [value, slope, second, third]
    }

    pub(crate) fn eval(&self, at: f64) -> Array1<f64> {
        let seg = self.segment(at);
        let h = self.t[seg + 1] - self.t[seg];
        let b = (at - self.t[seg]) / h;
        let a = 1.0 - b;
        let c = h * h / 6.0;
        let (y0, y1) = (self.y.row(seg), self.y.row(seg + 1));
        let (m0, m1) = (self.m.row(seg), self.m.row(seg + 1));
        let mut out = Array1::zeros(self.y.ncols());
        for (col, slot) in out.iter_mut().enumerate() {
            *slot = a * y0[col]
                + b * y1[col]
                + ((a * a * a - a) * m0[col] + (b * b * b - b) * m1[col]) * c;
        }
        out
    }
}

/// Second derivatives `M` of the not-a-knot cubic spline, `k >= 4`.
///
/// Rows 1..k-2 are the continuity of the first derivative,
/// `h_{i-1} M_{i-1} + 2 (h_{i-1} + h_i) M_i + h_i M_{i+1}
///  = 6 (d_i - d_{i-1})` with `d_i` the chord slopes. The first and
/// last rows ask the third derivative to be continuous at the second
/// and the second-to-last knot (de Boor, A Practical Guide to Splines,
/// Springer (1978), Ch. IV): `h_1 M_0 - (h_0 + h_1) M_1 + h_0 M_2 = 0`
/// and its mirror. Eliminating `M_0` and `M_{k-1}` from those boundary
/// equations leaves a tridiagonal system for `M_1..M_{k-2}`. Thomas
/// elimination solves every coordinate together in `O(k * ncols)`.
fn not_a_knot_moments(t: &[f64], y: &Array2<f64>) -> Option<Array2<f64>> {
    let k = t.len();
    let h: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    let n = k - 2;
    let mut lower = vec![0.0; n];
    let mut diagonal = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut rhs = Array2::zeros((n, y.ncols()));
    for row in 0..n {
        let i = row + 1;
        if row > 0 {
            lower[row] = h[i - 1];
        }
        diagonal[row] = 2.0 * (h[i - 1] + h[i]);
        if row + 1 < n {
            upper[row] = h[i];
        }
        for col in 0..y.ncols() {
            let left = (y[(i, col)] - y[(i - 1, col)]) / h[i - 1];
            let right = (y[(i + 1, col)] - y[(i, col)]) / h[i];
            rhs[(row, col)] = 6.0 * (right - left);
        }
    }

    // M0 = ((h0 + h1) M1 - h0 M2) / h1.
    diagonal[0] += h[0] * (h[0] + h[1]) / h[1];
    upper[0] -= h[0] * h[0] / h[1];
    // M{k-1} = ((h{k-3} + h{k-2}) M{k-2}
    //             - h{k-2} M{k-3}) / h{k-3}.
    lower[n - 1] -= h[k - 2] * h[k - 2] / h[k - 3];
    diagonal[n - 1] += h[k - 2] * (h[k - 3] + h[k - 2]) / h[k - 3];

    let scale = t[k - 1] - t[0];
    if !(scale.is_finite() && scale > 0.0) {
        return None;
    }
    solve_tridiagonal(&lower, &mut diagonal, &upper, &mut rhs, scale)?;
    let mut moments = Array2::zeros(y.raw_dim());
    moments.slice_mut(ndarray::s![1..k - 1, ..]).assign(&rhs);
    for col in 0..y.ncols() {
        moments[(0, col)] = ((h[0] + h[1]) * moments[(1, col)] - h[0] * moments[(2, col)]) / h[1];
        moments[(k - 1, col)] = ((h[k - 3] + h[k - 2]) * moments[(k - 2, col)]
            - h[k - 2] * moments[(k - 3, col)])
            / h[k - 3];
    }
    Some(moments)
}

/// Solve a tridiagonal system in place for all columns of `rhs`.
///
/// `scale` is the knot span. A pivot that is not strictly larger than
/// one unit in the last place of that span is singular: the spacing
/// that produced it is lost in the roundoff of the span.
fn solve_tridiagonal(
    lower: &[f64],
    diagonal: &mut [f64],
    upper: &[f64],
    rhs: &mut Array2<f64>,
    scale: f64,
) -> Option<()> {
    let n = diagonal.len();
    for row in 1..n {
        let pivot = diagonal[row - 1];
        if pivot_vanishes(pivot, scale) {
            return None;
        }
        let factor = lower[row] / pivot;
        diagonal[row] -= factor * upper[row - 1];
        let (source, mut target) =
            rhs.multi_slice_mut((ndarray::s![row - 1, ..], ndarray::s![row, ..]));
        target.scaled_add(-factor, &source);
    }
    let last = diagonal[n - 1];
    if pivot_vanishes(last, scale) {
        return None;
    }
    rhs.row_mut(n - 1).mapv_inplace(|value| value / last);
    for row in (0..n - 1).rev() {
        let pivot = diagonal[row];
        if pivot_vanishes(pivot, scale) {
            return None;
        }
        let coefficient = upper[row];
        let (mut target, source) =
            rhs.multi_slice_mut((ndarray::s![row, ..], ndarray::s![row + 1, ..]));
        target.scaled_add(-coefficient, &source);
        target.mapv_inplace(|value| value / pivot);
    }
    Some(())
}

/// A pivot lost in the roundoff of `scale`, or not finite.
fn pivot_vanishes(pivot: f64, scale: f64) -> bool {
    let gap = pivot.abs();
    let limit = scale * f64::EPSILON;
    !matches!(
        gap.partial_cmp(&limit),
        Some(Ordering::Greater) if gap.is_finite()
    )
}

/// Dense reference used to check the linear-time spline solve.
#[cfg(test)]
fn solve_dense(a: &mut [f64], b: &mut Array2<f64>) -> bool {
    let k = b.nrows();
    let scale = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if !(scale.is_finite() && scale > 0.0) {
        return false;
    }
    for col in 0..k {
        let pivot = (col..k)
            .max_by(|&i, &j| a[i * k + col].abs().total_cmp(&a[j * k + col].abs()))
            .unwrap_or(col);
        if a[pivot * k + col]
            .abs()
            .partial_cmp(&(PIVOT_TOLERANCE * scale))
            != Some(Ordering::Greater)
        {
            return false;
        }
        if pivot != col {
            for c in 0..k {
                a.swap(col * k + c, pivot * k + c);
            }
            for c in 0..b.ncols() {
                b.swap((col, c), (pivot, c));
            }
        }
        for row in col + 1..k {
            let factor = a[row * k + col] / a[col * k + col];
            if factor == 0.0 {
                continue;
            }
            for c in col..k {
                a[row * k + c] -= factor * a[col * k + c];
            }
            let (upper, mut lower) =
                b.multi_slice_mut((ndarray::s![col, ..], ndarray::s![row, ..]));
            lower.scaled_add(-factor, &upper.view());
        }
    }
    for col in (0..k).rev() {
        for row in col + 1..k {
            let coefficient = a[col * k + row];
            if coefficient != 0.0 {
                let (mut target, source) =
                    b.multi_slice_mut((ndarray::s![col, ..], ndarray::s![row, ..]));
                target.scaled_add(-coefficient, &source.view());
            }
        }
        let diagonal = a[col * k + col];
        b.row_mut(col).mapv_inplace(|v| v / diagonal);
    }
    true
}

/// Length of each segment between consecutive images, as `scheme`
/// measures it: the chord for [`Reparametrization::Linear`], the arc of
/// the not-a-knot spline through the images for
/// [`Reparametrization::NotAKnot`]. `cell` makes the chords minimum
/// images. A reparametrized band has equal entries, on each side of an
/// anchor, up to the interpolation error.
pub fn segment_lengths(
    positions: &Array2<f64>,
    cell: Option<&Cell>,
    scheme: Reparametrization,
) -> Vec<f64> {
    let n = positions.nrows();
    let mut chain = positions.clone();
    let mut chords = Vec::with_capacity(n.saturating_sub(1));
    for i in 1..n {
        let mut d = &positions.row(i) - &positions.row(i - 1);
        if let Some(cell) = cell {
            cell.minimum_image(&mut d);
        }
        let unwrapped = &chain.row(i - 1) + &d;
        chain.row_mut(i).assign(&unwrapped);
        chords.push(d.dot(&d).sqrt());
    }
    if scheme == Reparametrization::Linear || n < 3 {
        return chords;
    }
    let mut t = vec![0.0; n];
    for i in 1..n {
        t[i] = t[i - 1] + chords[i - 1];
    }
    let total = t[n - 1];
    if !(total.is_finite() && total > 0.0) {
        return chords;
    }
    let kept = distinct_knots(&t, None, COINCIDENT * total);
    let knots: Vec<f64> = kept.iter().map(|&i| t[i]).collect();
    let mut values = Array2::zeros((kept.len(), chain.ncols()));
    for (row, &i) in kept.iter().enumerate() {
        values.row_mut(row).assign(&chain.row(i));
    }
    let Some(spline) = Spline::fit(&knots, &values) else {
        return chords;
    };
    let arcs = spline.knot_arcs();
    let mut lengths = vec![0.0; n - 1];
    for (k, pair) in kept.windows(2).enumerate() {
        lengths[pair[1] - 1] = arcs[k + 1] - arcs[k];
    }
    lengths
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn cubic(t: f64) -> [f64; 3] {
        [
            1.0 - 2.0 * t + 0.5 * t * t * t,
            0.3 * t * t - t * t * t,
            2.0 + t,
        ]
    }

    #[test]
    fn the_spline_reproduces_a_cubic_on_uneven_knots() {
        let knots = [0.0, 0.13, 0.4, 0.45, 1.1, 1.7, 2.0];
        let mut values = Array2::zeros((knots.len(), 3));
        for (row, &t) in knots.iter().enumerate() {
            values.row_mut(row).assign(&Array1::from(cubic(t).to_vec()));
        }
        let spline = Spline::fit(&knots, &values).unwrap();
        for step in 0..=200 {
            let t = 2.0 * step as f64 / 200.0;
            let got = spline.eval(t);
            for (g, e) in got.iter().zip(cubic(t)) {
                assert_abs_diff_eq!(*g, e, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn four_knots_are_the_cubic_and_three_the_parabola() {
        let four = [0.0, 0.5, 1.2, 2.0];
        let mut values = Array2::zeros((4, 3));
        for (row, &t) in four.iter().enumerate() {
            values.row_mut(row).assign(&Array1::from(cubic(t).to_vec()));
        }
        let spline = Spline::fit(&four, &values).unwrap();
        for (g, e) in spline.eval(0.8).iter().zip(cubic(0.8)) {
            assert_abs_diff_eq!(*g, e, epsilon = 1e-12);
        }
        let parabola = |t: f64| 1.0 - t + 3.0 * t * t;
        let three = [0.0, 0.3, 1.0];
        let values = Array2::from_shape_fn((3, 1), |(i, _)| parabola(three[i]));
        let spline = Spline::fit(&three, &values).unwrap();
        for t in [0.1, 0.5, 0.9] {
            assert_abs_diff_eq!(spline.eval(t)[0], parabola(t), epsilon = 1e-12);
        }
        let two = Spline::fit(&[0.0, 2.0], &array![[0.0], [4.0]]).unwrap();
        assert_abs_diff_eq!(two.eval(0.5)[0], 1.0, epsilon = 1e-15);
    }

    #[test]
    fn a_fit_refuses_repeated_knots() {
        assert!(Spline::fit(&[0.0, 1.0, 1.0, 2.0], &Array2::zeros((4, 1))).is_none());
    }

    #[test]
    fn a_near_duplicate_knot_span_refuses_the_spline() {
        let span = 2.5;
        let gap = span * f64::EPSILON / 8.0;
        let knots = [0.0, gap, 2.0 * gap, 3.0 * gap, span];
        let mut values = Array2::zeros((knots.len(), 3));
        for (row, &t) in knots.iter().enumerate() {
            values.row_mut(row).assign(&Array1::from(cubic(t).to_vec()));
        }
        assert!(Spline::fit(&knots, &values).is_none());

        let separated = span * f64::EPSILON.sqrt();
        let knots = [0.0, separated, 2.0 * separated, 3.0 * separated, span];
        let mut values = Array2::zeros((knots.len(), 1));
        for (row, &t) in knots.iter().enumerate() {
            values[(row, 0)] = t;
        }
        let spline = Spline::fit(&knots, &values).expect("gaps above the span roundoff");
        let at = 0.5 * (knots[3] + span);
        assert_abs_diff_eq!(spline.eval(at)[0], at, epsilon = 1e-12);
    }

    #[test]
    fn tridiagonal_moments_match_dense_not_a_knot_system() {
        let mut seed = 0x6a09_e667_f3bc_c909_u64;
        let mut random = || {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 11) as f64 / (1_u64 << 53) as f64
        };
        for k in 4..40 {
            let mut knots = vec![0.0; k];
            for i in 1..k {
                knots[i] = knots[i - 1] + 0.05 + random();
            }
            let values = Array2::from_shape_fn((k, 17), |_| 2.0 * random() - 1.0);
            let got = not_a_knot_moments(&knots, &values).expect("tridiagonal solve");

            let h: Vec<f64> = knots.windows(2).map(|w| w[1] - w[0]).collect();
            let mut matrix = vec![0.0; k * k];
            let mut want = Array2::zeros(values.raw_dim());
            matrix[0] = h[1];
            matrix[1] = -(h[0] + h[1]);
            matrix[2] = h[0];
            for i in 1..k - 1 {
                matrix[i * k + i - 1] = h[i - 1];
                matrix[i * k + i] = 2.0 * (h[i - 1] + h[i]);
                matrix[i * k + i + 1] = h[i];
                for col in 0..values.ncols() {
                    let left = (values[(i, col)] - values[(i - 1, col)]) / h[i - 1];
                    let right = (values[(i + 1, col)] - values[(i, col)]) / h[i];
                    want[(i, col)] = 6.0 * (right - left);
                }
            }
            let last = (k - 1) * k;
            matrix[last + k - 3] = h[k - 2];
            matrix[last + k - 2] = -(h[k - 3] + h[k - 2]);
            matrix[last + k - 1] = h[k - 3];
            assert!(solve_dense(&mut matrix, &mut want));
            for (actual, expected) in got.iter().zip(want.iter()) {
                assert_abs_diff_eq!(actual, expected, epsilon = 2e-10);
            }
        }
    }

    #[test]
    fn coincident_images_merge_into_one_knot() {
        let s = [0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        assert_eq!(distinct_knots(&s, None, 1e-12), vec![0, 2, 5]);
        assert_eq!(distinct_knots(&s, Some(3), 1e-12), vec![0, 3, 5]);
    }

    fn curve() -> (Spline, Vec<f64>) {
        let knots = [0.0, 0.4, 1.1, 1.5, 2.0];
        let mut values = Array2::zeros((knots.len(), 3));
        for (row, &t) in knots.iter().enumerate() {
            values.row_mut(row).assign(&Array1::from(cubic(t).to_vec()));
        }
        let spline = Spline::fit(&knots, &values).unwrap();
        let arc = spline.knot_arcs();
        (spline, arc)
    }

    #[test]
    fn a_target_on_the_curve_is_found_by_its_arc() {
        let (spline, arc) = curve();
        let length = arc[arc.len() - 1];
        for step in 1..20 {
            let target = length * step as f64 / 20.0;
            let at = spline.at_arc(&arc, target).unwrap();
            assert!((spline.knots()[0]..=spline.knots()[4]).contains(&at));
            let reached = arc_to(&spline, &arc, at);
            assert_abs_diff_eq!(reached, target, epsilon = 1e-12 * length);
        }
    }

    fn arc_to(spline: &Spline, arc: &[f64], at: f64) -> f64 {
        let seg = spline.segment(at);
        arc[seg] + spline.arc_between(seg, spline.knots()[seg], at)
    }

    #[test]
    fn the_ends_of_the_curve_are_its_end_knots() {
        let (spline, arc) = curve();
        let length = arc[arc.len() - 1];
        let (first, last) = (spline.knots()[0], spline.knots()[4]);
        assert_eq!(spline.at_arc(&arc, 0.0), Some(first));
        assert_eq!(spline.at_arc(&arc, length), Some(last));
        let inside = 0.5 * ARC_END_TOLERANCE * length;
        assert_eq!(spline.at_arc(&arc, -inside), Some(first));
        assert_eq!(spline.at_arc(&arc, length + inside), Some(last));
    }

    #[test]
    fn a_target_past_an_end_has_no_parameter() {
        let (spline, arc) = curve();
        let length = arc[arc.len() - 1];
        let outside = 2.0 * ARC_END_TOLERANCE * length;
        assert_eq!(spline.at_arc(&arc, -outside), None);
        assert_eq!(spline.at_arc(&arc, length + outside), None);
        assert_eq!(spline.at_arc(&arc, -length), None);
        assert_eq!(spline.at_arc(&arc, 3.0 * length), None);
        assert_eq!(spline.at_arc(&arc, f64::NAN), None);
        assert_eq!(spline.at_arc(&arc, f64::INFINITY), None);
    }

    #[test]
    fn a_curve_without_length_has_no_parameter() {
        let (spline, _) = curve();
        let flat = vec![1.0; spline.knots().len()];
        assert_eq!(spline.at_arc(&flat, 1.0), None);
        let broken = vec![0.0, 1.0, f64::NAN, 2.0, 3.0];
        assert_eq!(spline.at_arc(&broken, 1.5), None);
    }

    #[test]
    fn images_stay_when_the_curve_cannot_be_walked() {
        let mut positions = Array2::from_elem((5, 2), 1.5);
        let before = positions.clone();
        not_a_knot_equal_arc(&mut positions, None, None);
        assert_eq!(positions, before);
        positions[[2, 0]] = f64::NAN;
        let before = positions.clone();
        not_a_knot_equal_arc(&mut positions, None, None);
        assert!(
            positions
                .iter()
                .zip(before.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}
