//! Cubic splines with closed-form primitives.

/// Cubic spline through `(x_i, y_i)` with ascending `x`. With four knots
/// or more the third derivative is continuous at the second and the
/// second-to-last knot (not-a-knot), so cubics are reproduced and the
/// error is fourth order up to the ends; with three the second derivative
/// vanishes at the ends, and two knots give a line.
#[derive(Clone, Debug)]
pub(crate) struct Spline {
    x: Vec<f64>,
    y: Vec<f64>,
    second: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Spline {
    pub(crate) fn new(x: &[f64], y: &[f64]) -> Self {
        assert!(x.len() >= 2 && x.len() == y.len());
        let k = x.len();
        let mut second = vec![0.0; k];
        if k > 2 {
            let interior = k - 2;
            let mut lower = vec![0.0; interior];
            let mut diagonal = vec![0.0; interior];
            let mut upper = vec![0.0; interior];
            let mut rhs = vec![0.0; interior];
            for i in 1..k - 1 {
                let (h0, h1) = (x[i] - x[i - 1], x[i + 1] - x[i]);
                lower[i - 1] = h0;
                diagonal[i - 1] = 2.0 * (h0 + h1);
                upper[i - 1] = h1;
                rhs[i - 1] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0);
            }
            let not_a_knot = k >= 4;
            let h = |i: usize| x[i + 1] - x[i];
            if not_a_knot {
                let (h0, h1) = (h(0), h(1));
                diagonal[0] += h0 + h0 * h0 / h1;
                upper[0] -= h0 * h0 / h1;
                let (a, b) = (h(k - 3), h(k - 2));
                diagonal[interior - 1] += b + b * b / a;
                lower[interior - 1] -= b * b / a;
            }
            for i in 1..interior {
                let w = lower[i] / diagonal[i - 1];
                diagonal[i] -= w * upper[i - 1];
                rhs[i] -= w * rhs[i - 1];
            }
            second[interior] = rhs[interior - 1] / diagonal[interior - 1];
            for i in (0..interior - 1).rev() {
                second[i + 1] = (rhs[i] - upper[i] * second[i + 2]) / diagonal[i];
            }
            if not_a_knot {
                let (h0, h1) = (h(0), h(1));
                second[0] = second[1] + (second[1] - second[2]) * h0 / h1;
                let (a, b) = (h(k - 3), h(k - 2));
                second[k - 1] = second[k - 2] + (second[k - 2] - second[k - 3]) * b / a;
            }
        }
        let mut spline = Self {
            x: x.to_vec(),
            y: y.to_vec(),
            second,
            cumulative: vec![0.0; k],
        };
        for j in 0..k - 1 {
            spline.cumulative[j + 1] = spline.cumulative[j] + spline.piece_primitive(j, x[j + 1]);
        }
        spline
    }

    fn interval(&self, t: f64) -> usize {
        let last = self.x.len() - 2;
        match self.x.partition_point(|&xi| xi <= t) {
            0 => 0,
            i => (i - 1).min(last),
        }
    }

    fn piece_primitive(&self, j: usize, t: f64) -> f64 {
        let (x0, x1) = (self.x[j], self.x[j + 1]);
        let h = x1 - x0;
        let (a, b) = (x1 - t, t - x0);
        let (m0, m1) = (self.second[j], self.second[j + 1]);
        m0 * (h.powi(4) - a.powi(4)) / (24.0 * h)
            + m1 * b.powi(4) / (24.0 * h)
            + (self.y[j] / h - m0 * h / 6.0) * (h * h - a * a) / 2.0
            + (self.y[j + 1] / h - m1 * h / 6.0) * b * b / 2.0
    }

    pub(crate) fn value(&self, t: f64) -> f64 {
        let j = self.interval(t);
        let (x0, x1) = (self.x[j], self.x[j + 1]);
        let h = x1 - x0;
        let (a, b) = (x1 - t, t - x0);
        let (m0, m1) = (self.second[j], self.second[j + 1]);
        m0 * a.powi(3) / (6.0 * h)
            + m1 * b.powi(3) / (6.0 * h)
            + (self.y[j] / h - m0 * h / 6.0) * a
            + (self.y[j + 1] / h - m1 * h / 6.0) * b
    }

    /// Second derivative of the piece containing `t`. It is linear on
    /// each interval, equal to the stored knot values at the knots.
    pub(crate) fn second_derivative(&self, t: f64) -> f64 {
        let j = self.interval(t);
        let (x0, x1) = (self.x[j], self.x[j + 1]);
        let h = x1 - x0;
        let (a, b) = (x1 - t, t - x0);
        (self.second[j] * a + self.second[j + 1] * b) / h
    }

    /// Derivative of the piece containing `t`.
    pub(crate) fn derivative(&self, t: f64) -> f64 {
        let j = self.interval(t);
        let (x0, x1) = (self.x[j], self.x[j + 1]);
        let h = x1 - x0;
        let (a, b) = (x1 - t, t - x0);
        let (m0, m1) = (self.second[j], self.second[j + 1]);
        -m0 * a * a / (2.0 * h) + m1 * b * b / (2.0 * h) - (self.y[j] / h - m0 * h / 6.0)
            + (self.y[j + 1] / h - m1 * h / 6.0)
    }

    /// `∫ S` from the first knot to `t`.
    #[cfg(any(test, feature = "ohtst"))]
    pub(crate) fn primitive(&self, t: f64) -> f64 {
        let j = self.interval(t);
        self.cumulative[j] + self.piece_primitive(j, t)
    }
}

/// Nodes and weights of the composite Simpson rule with `2 pairs`
/// subintervals on `[a, b]`.
#[cfg(any(test, feature = "ohtst"))]
pub(crate) fn simpson_rule(a: f64, b: f64, pairs: usize) -> impl Iterator<Item = (f64, f64)> {
    let intervals = 2 * pairs;
    let h = (b - a) / intervals as f64;
    (0..=intervals).map(move |i| {
        let weight = if i == 0 || i == intervals {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        (a + h * i as f64, weight * h / 3.0)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_cubic_on_uneven_knots() {
        let x = [-1.0, -0.6, 0.0, 0.5, 1.5, 2.0, 3.5];
        let cubic = |t: f64| 0.8 * t * t * t - 0.9 * t * t + t - 0.3;
        let primitive = |t: f64| 0.2 * t.powi(4) - 0.3 * t.powi(3) + 0.5 * t * t - 0.3 * t;
        let values: Vec<f64> = x.iter().map(|&t| cubic(t)).collect();
        let spline = Spline::new(&x, &values);
        let slope = |t: f64| 2.4 * t * t - 1.8 * t + 1.0;
        let bend = |t: f64| 4.8 * t - 1.8;
        let roundoff = f64::EPSILON * (x.len() as f64).powi(2);
        for t in [-1.0, -0.8, 0.0, 0.3, 1.0, 1.9, 3.5] {
            // Relative to the cubic, its derivative, its second derivative,
            // and its primitive.
            assert!(
                (spline.value(t) - cubic(t)).abs() <= cubic(t).abs() * roundoff,
                "{t}"
            );
            assert!(
                (spline.derivative(t) - slope(t)).abs() <= slope(t).abs() * roundoff,
                "{t}"
            );
            assert!(
                (spline.second_derivative(t) - bend(t)).abs() <= bend(t).abs() * roundoff,
                "{t}"
            );
            let integral = primitive(t) - primitive(-1.0);
            assert!(
                (spline.primitive(t) - integral).abs() <= integral.abs() * roundoff,
                "{t}"
            );
        }
    }

    #[test]
    fn three_knots_are_natural_and_two_a_line() {
        let spline = Spline::new(&[0.0, 2.0], &[1.0, 2.0]);
        assert!((spline.value(1.5) - 1.75).abs() < 1e-15);
        assert!((spline.primitive(2.0) - 3.0).abs() < 1e-15);
        let spline = Spline::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 0.0]);
        assert!((spline.value(1.0) - 1.0).abs() < 1e-15);
        assert!((spline.value(0.5) - spline.value(1.5)).abs() < 1e-15);
        assert!((spline.primitive(2.0) - 2.0 * spline.primitive(1.0)).abs() < 1e-15);
    }

    #[test]
    fn simpson_is_exact_for_cubics() {
        let total: f64 = simpson_rule(-1.0, 2.0, 3)
            .map(|(t, w)| w * (t * t * t - 2.0 * t + 1.0))
            .sum();
        assert!((total - (15.0 / 4.0 - 3.0 + 3.0)).abs() < 1e-13);
    }
}
