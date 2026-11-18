// Two-dimensional surfaces of the OH-TST papers and quadrature references
// for the free energy of a line, shared by the OH-TST tests.

#[allow(dead_code)]
mod ohtst_surfaces {
    use ndarray::{Array1, ArrayView1, array};
    use rgsaddle::{PointSurface, SaddleError};

    /// LEPS plus a harmonic oscillator, Jóhannesson and Jónsson, J. Chem.
    /// Phys. 115, 9644 (2001), Appendix B, with the parameters given there.
    pub struct LepsHo;

    const LEPS_A: f64 = 0.05;
    const LEPS_B: f64 = 0.80;
    const LEPS_C: f64 = 0.05;
    const R0: f64 = 0.742;
    const ALPHA: f64 = 1.942;
    const D_AB: f64 = 4.746;
    const D_BC: f64 = 4.746;
    const D_AC: f64 = 3.445;
    const R_AC: f64 = 3.742;
    const K_C: f64 = 0.2025;
    const C_HO: f64 = 1.154;

    /// `(Q, dQ/dr, d²Q/dr², J, dJ/dr, d²J/dr²)` of one pair.
    fn coulomb_exchange(d: f64, r: f64) -> (f64, f64, f64, f64, f64, f64) {
        let e1 = (-ALPHA * (r - R0)).exp();
        let e2 = e1 * e1;
        let q = 0.5 * d * (1.5 * e2 - e1);
        let dq = 0.5 * d * (-3.0 * ALPHA * e2 + ALPHA * e1);
        let d2q = 0.5 * d * ALPHA * ALPHA * (6.0 * e2 - e1);
        let j = 0.25 * d * (e2 - 6.0 * e1);
        let dj = 0.25 * d * (-2.0 * ALPHA * e2 + 6.0 * ALPHA * e1);
        let d2j = 0.25 * d * ALPHA * ALPHA * (4.0 * e2 - 6.0 * e1);
        (q, dq, d2q, j, dj, d2j)
    }

    impl LepsHo {
        /// Energy and gradient at `(x, y)`. `r_AB = x`, `r_BC = r_AC - x`,
        /// so `r_AC` is fixed and its pair terms are constants.
        pub fn energy(x: f64, y: f64) -> (f64, f64, f64) {
            let (qab, dqab, _, jab, djab, _) = coulomb_exchange(D_AB, x);
            let (qbc, dqbc, _, jbc, djbc, _) = coulomb_exchange(D_BC, R_AC - x);
            let (qac, _, _, jac, _, _) = coulomb_exchange(D_AC, R_AC);
            let (sa, sb, sc) = (1.0 + LEPS_A, 1.0 + LEPS_B, 1.0 + LEPS_C);
            let (u, w, z) = (jab / sa, jbc / sb, jac / sc);
            let s = u * u + w * w + z * z - u * w - w * z - u * z;
            let root = s.sqrt();
            let leps = qab / sa + qbc / sb + qac / sc - root;
            let du = djab / sa;
            let dw = -djbc / sb;
            let ds = (2.0 * u - w - z) * du + (2.0 * w - u - z) * dw;
            let dleps = dqab / sa - dqbc / sb - ds / (2.0 * root);
            let stretch = x - (0.5 * R_AC - y / C_HO);
            let energy = leps + 2.0 * K_C * stretch * stretch;
            (
                energy,
                dleps + 4.0 * K_C * stretch,
                4.0 * K_C * stretch / C_HO,
            )
        }

        /// Analytic Hessian of the Appendix B potential.
        pub fn hessian(x: f64, _y: f64) -> [[f64; 2]; 2] {
            let (_, _, d2qab, jab, djab, d2jab) = coulomb_exchange(D_AB, x);
            let (_, _, d2qbc, jbc, djbc, d2jbc) =
                coulomb_exchange(D_BC, R_AC - x);
            let (_, _, _, jac, _, _) = coulomb_exchange(D_AC, R_AC);
            let (sa, sb, sc) = (1.0 + LEPS_A, 1.0 + LEPS_B, 1.0 + LEPS_C);
            let (u, w, z) = (jab / sa, jbc / sb, jac / sc);
            let (du, dw) = (djab / sa, -djbc / sb);
            let (d2u, d2w) = (d2jab / sa, d2jbc / sb);
            let s = u * u + w * w + z * z - u * w - w * z - u * z;
            let root = s.sqrt();
            let ds = (2.0 * u - w - z) * du + (2.0 * w - u - z) * dw;
            let d2s = 2.0 * (du * du + dw * dw - du * dw)
                + (2.0 * u - w - z) * d2u
                + (2.0 * w - u - z) * d2w;
            let leps_xx = d2qab / sa + d2qbc / sb - d2s / (2.0 * root)
                + ds * ds / (4.0 * root.powi(3));
            let oscillator = 4.0 * K_C;
            [
                [leps_xx + oscillator, oscillator / C_HO],
                [oscillator / C_HO, oscillator / (C_HO * C_HO)],
            ]
        }
    }

    impl PointSurface for LepsHo {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (e, gx, gy) = Self::energy(q[0], q[1]);
            Ok((e, array![gx, gy]))
        }
    }

    /// Symmetric Eckart barrier on a harmonic valley, Eq. 25 of the same
    /// paper and the surface of Bligaard and Jónsson, Comput. Phys. Commun.
    /// 169, 284 (2005): `V0 / cosh²(αx/2) + ½μω² (y - Cx)²`.
    pub struct EckartHo;

    pub const ECKART_V0: f64 = 0.0156;
    const ECKART_ALPHA: f64 = 3.97;
    const ECKART_HALF_MU_OMEGA2: f64 = 1.04e-4;
    pub const ECKART_C: f64 = 10.0;

    impl EckartHo {
        pub fn energy(x: f64, y: f64) -> (f64, f64, f64) {
            let half = 0.5 * ECKART_ALPHA * x;
            let sech2 = 1.0 / half.cosh().powi(2);
            let valley = y - ECKART_C * x;
            let energy = ECKART_V0 * sech2 + ECKART_HALF_MU_OMEGA2 * valley * valley;
            let gx = -ECKART_ALPHA * ECKART_V0 * sech2 * half.tanh()
                - 2.0 * ECKART_HALF_MU_OMEGA2 * ECKART_C * valley;
            let gy = 2.0 * ECKART_HALF_MU_OMEGA2 * valley;
            (energy, gx, gy)
        }
    }

    impl PointSurface for EckartHo {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (e, gx, gy) = Self::energy(q[0], q[1]);
            Ok((e, array![gx, gy]))
        }
    }

    /// Double well in a skewed valley whose stiffness rises toward the
    /// product, so the optimal plane both shifts and turns with `kT`:
    /// `(x² - 1)² + ½ k₀ (1 + α tanh 2x) (y - Cx)²` with `k₀ = 4`, `C = 1`,
    /// `α = 0.8`. The reactant minimum is `(-1, -1)`, the product `(1, 1)`,
    /// and the saddle the origin, one energy unit up.
    pub struct SkewedWell;

    pub const SKEWED_K0: f64 = 4.0;
    pub const SKEWED_C: f64 = 1.0;
    pub const SKEWED_ALPHA: f64 = 0.8;

    impl SkewedWell {
        pub fn energy(x: f64, y: f64) -> (f64, f64, f64) {
            let well = x * x - 1.0;
            let tanh = (2.0 * x).tanh();
            let stiffness = SKEWED_K0 * (1.0 + SKEWED_ALPHA * tanh);
            let valley = y - SKEWED_C * x;
            let energy = well * well + 0.5 * stiffness * valley * valley;
            let gx = 4.0 * x * well
                + SKEWED_K0 * SKEWED_ALPHA * (1.0 - tanh * tanh) * valley * valley
                - SKEWED_C * stiffness * valley;
            let gy = stiffness * valley;
            (energy, gx, gy)
        }
    }

    impl PointSurface for SkewedWell {
        fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let (e, gx, gy) = Self::energy(q[0], q[1]);
            Ok((e, array![gx, gy]))
        }
    }

    /// Adaptive Simpson on `[a, b]` to an absolute tolerance.
    pub fn simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, tol: f64) -> f64 {
        fn rule<F: Fn(f64) -> f64>(
            f: &F,
            (a, b): (f64, f64),
            [fa, fm, fb]: [f64; 3],
            whole: f64,
            tol: f64,
            depth: usize,
        ) -> f64 {
            let m = 0.5 * (a + b);
            let (flm, frm) = (f(0.5 * (a + m)), f(0.5 * (m + b)));
            let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
            let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
            let delta = left + right - whole;
            if depth == 0 || delta.abs() <= 15.0 * tol {
                return left + right + delta / 15.0;
            }
            rule(f, (a, m), [fa, flm, fm], left, 0.5 * tol, depth - 1)
                + rule(f, (m, b), [fm, frm, fb], right, 0.5 * tol, depth - 1)
        }
        let m = 0.5 * (a + b);
        let ends = [f(a), f(m), f(b)];
        let whole = (b - a) / 6.0 * (ends[0] + 4.0 * ends[1] + ends[2]);
        rule(f, (a, b), ends, whole, tol, 40)
    }

    /// `∫ g(t) dt` over the real line for a positive bump `g` that is
    /// negligible past `|t| = extent`: the support is found on a grid and
    /// split into panels before the adaptive rule.
    pub fn bump_integral<F: Fn(f64) -> f64>(g: &F, extent: f64, rel: f64) -> f64 {
        let grid = 4000;
        let step = 2.0 * extent / grid as f64;
        let values: Vec<f64> = (0..=grid)
            .map(|i| g(-extent + step * i as f64))
            .collect();
        let peak = values.iter().cloned().fold(0.0, f64::max);
        let floor = peak * 1.0e-18;
        let first = values.iter().position(|v| *v > floor).unwrap_or(0);
        let last = values.iter().rposition(|v| *v > floor).unwrap_or(grid);
        let lo = -extent + step * first.saturating_sub(1) as f64;
        let hi = -extent + step * (last + 1).min(grid) as f64;
        let panels = 64;
        let width = (hi - lo) / panels as f64;
        let scale = peak * (hi - lo);
        (0..panels)
            .map(|k| {
                let a = lo + width * k as f64;
                simpson(g, a, a + width, rel * scale / panels as f64)
            })
            .sum()
    }

    /// `∫ exp(-(V - shift) / kT)` along the line through `point` normal to
    /// `normal`, for a two-dimensional energy.
    pub fn line_integral(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let norm = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        let (nx, ny) = (normal[0] / norm, normal[1] / norm);
        let (dx, dy) = (-ny, nx);
        let g = |t: f64| {
            let (e, _, _) = energy(point[0] + t * dx, point[1] + t * dy);
            (-(e - shift) / kt).exp()
        };
        bump_integral(&g, extent, 1.0e-11)
    }

    /// Plane free energy of a line, up to the constant `shift`.
    pub fn line_free_energy(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        -kt * line_integral(energy, point, normal, kt, shift, extent).ln()
    }

    /// `count` independent draws from `exp(-V/kT)` on the line through
    /// `point` normal to `normal`, out to `extent` either side, by
    /// inverting the trapezoid cumulative integral on 20 000 intervals.
    /// Uniforms come from SplitMix64 seeded with `seed`.
    pub fn line_draws(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        extent: f64,
        count: usize,
        seed: u64,
    ) -> Vec<Array1<f64>> {
        let norm = normal[0].hypot(normal[1]);
        let (dx, dy) = (-normal[1] / norm, normal[0] / norm);
        let grid = 20_000;
        let step = 2.0 * extent / grid as f64;
        let energies: Vec<f64> = (0..=grid)
            .map(|i| {
                let t = -extent + step * i as f64;
                energy(point[0] + t * dx, point[1] + t * dy).0
            })
            .collect();
        let lowest = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut cumulative = vec![0.0; grid + 1];
        for i in 1..=grid {
            let (a, b) = (energies[i - 1] - lowest, energies[i] - lowest);
            cumulative[i] = cumulative[i - 1] + 0.5 * step * ((-a / kt).exp() + (-b / kt).exp());
        }
        let total = cumulative[grid];
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (((z ^ (z >> 31)) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                let target = u * total;
                let i = cumulative.partition_point(|c| *c < target).clamp(1, grid);
                let within = (target - cumulative[i - 1]) / (cumulative[i] - cumulative[i - 1]);
                let t = -extent + step * ((i - 1) as f64 + within);
                array![point[0] + t * dx, point[1] + t * dy]
            })
            .collect()
    }

    /// `∫ exp(-(V - shift) / kT)` over the half-plane behind the line, the
    /// side opposite to `normal`: lines at signed distance `ξ ≤ 0`.
    pub fn half_plane_integral(
        energy: fn(f64, f64) -> (f64, f64, f64),
        point: [f64; 2],
        normal: [f64; 2],
        kt: f64,
        shift: f64,
        extent: f64,
    ) -> f64 {
        let norm = (normal[0] * normal[0] + normal[1] * normal[1]).sqrt();
        let (nx, ny) = (normal[0] / norm, normal[1] / norm);
        let line = |xi: f64| {
            line_integral(
                energy,
                [point[0] + xi * nx, point[1] + xi * ny],
                [nx, ny],
                kt,
                shift,
                extent,
            )
        };
        let grid = 400;
        let step = extent / grid as f64;
        let values: Vec<f64> = (0..=grid).map(|i| line(-step * i as f64)).collect();
        let peak = values.iter().cloned().fold(0.0, f64::max);
        let deepest = values
            .iter()
            .rposition(|v| *v > peak * 1.0e-18)
            .unwrap_or(grid);
        let lo = -step * (deepest + 1).min(grid) as f64;
        let panels = 32;
        let width = -lo / panels as f64;
        let scale = peak * (-lo);
        (0..panels)
            .map(|k| {
                let a = lo + width * k as f64;
                simpson(&line, a, a + width, 1.0e-9 * scale / panels as f64)
            })
            .sum()
    }

    /// Position of a local minimum of a two-dimensional energy by gradient
    /// descent with a backtracking step, to a gradient tolerance.
    pub fn descend(energy: fn(f64, f64) -> (f64, f64, f64), start: [f64; 2]) -> [f64; 2] {
        let mut p = start;
        let mut step = 1.0e-2;
        for _ in 0..200_000 {
            let (e, gx, gy) = energy(p[0], p[1]);
            if gx.hypot(gy) < 1.0e-12 {
                break;
            }
            loop {
                let trial = [p[0] - step * gx, p[1] - step * gy];
                if energy(trial[0], trial[1]).0 < e {
                    p = trial;
                    step *= 1.2;
                    break;
                }
                step *= 0.5;
                if step < 1.0e-16 {
                    return p;
                }
            }
        }
        p
    }

    /// The steepest-descent path from the saddle `start` along `direction`
    /// to the minimum below it: points `spacing` apart in arc length, by
    /// RK4 on `dx/ds = -∇V/|∇V|` with steps of `spacing / 50`, then the
    /// minimum polished by [`descend`]. The flow is singular at the
    /// minimum, so the integration stops once a step moves less than a
    /// quarter of its length.
    pub fn descent_path(
        energy: fn(f64, f64) -> (f64, f64, f64),
        start: [f64; 2],
        direction: [f64; 2],
        spacing: f64,
    ) -> Vec<[f64; 2]> {
        let flow = |p: [f64; 2]| {
            let (_, gx, gy) = energy(p[0], p[1]);
            let g = gx.hypot(gy);
            [-gx / g, -gy / g]
        };
        let h = spacing / 50.0;
        let mut points = vec![start];
        let mut p = [start[0] + h * direction[0], start[1] + h * direction[1]];
        let mut travelled = h;
        let mut previous = energy(p[0], p[1]).0;
        for _ in 0..1_000_000 {
            let k1 = flow(p);
            let k2 = flow([p[0] + 0.5 * h * k1[0], p[1] + 0.5 * h * k1[1]]);
            let k3 = flow([p[0] + 0.5 * h * k2[0], p[1] + 0.5 * h * k2[1]]);
            let k4 = flow([p[0] + h * k3[0], p[1] + h * k3[1]]);
            let next = [
                p[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
                p[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
            ];
            let e = energy(next[0], next[1]).0;
            let moved = (next[0] - p[0]).hypot(next[1] - p[1]);
            if e >= previous || moved < 0.25 * h {
                break;
            }
            previous = e;
            p = next;
            travelled += moved;
            if travelled >= spacing * points.len() as f64 {
                points.push(p);
            }
        }
        let minimum = descend(energy, p);
        let last = points[points.len() - 1];
        if (minimum[0] - last[0]).hypot(minimum[1] - last[1]) < 0.5 * spacing {
            points.pop();
        }
        points.push(minimum);
        points
    }

    /// Stationary point near `start` by Newton's method on the analytic
    /// gradient, with a central-difference Jacobian.
    pub fn stationary(energy: fn(f64, f64) -> (f64, f64, f64), start: [f64; 2]) -> [f64; 2] {
        let mut p = start;
        let h = 1.0e-6;
        for _ in 0..100 {
            let (_, gx, gy) = energy(p[0], p[1]);
            if gx.hypot(gy) < 1.0e-13 {
                break;
            }
            let (_, axp, ayp) = energy(p[0] + h, p[1]);
            let (_, axm, aym) = energy(p[0] - h, p[1]);
            let (_, bxp, byp) = energy(p[0], p[1] + h);
            let (_, bxm, bym) = energy(p[0], p[1] - h);
            let hxx = (axp - axm) / (2.0 * h);
            let hyx = (ayp - aym) / (2.0 * h);
            let hxy = (bxp - bxm) / (2.0 * h);
            let hyy = (byp - bym) / (2.0 * h);
            let det = hxx * hyy - hxy * hyx;
            p[0] -= (hyy * gx - hxy * gy) / det;
            p[1] -= (hxx * gy - hyx * gx) / det;
        }
        p
    }

    /// Symmetric central-difference Hessian of the analytic gradient.
    pub fn hessian(energy: fn(f64, f64) -> (f64, f64, f64), p: [f64; 2]) -> [[f64; 2]; 2] {
        let h = 1.0e-5;
        let (_, axp, ayp) = energy(p[0] + h, p[1]);
        let (_, axm, aym) = energy(p[0] - h, p[1]);
        let (_, bxp, byp) = energy(p[0], p[1] + h);
        let (_, bxm, bym) = energy(p[0], p[1] - h);
        let off = 0.5 * ((ayp - aym) + (bxp - bxm)) / (2.0 * h);
        [
            [(axp - axm) / (2.0 * h), off],
            [off, (byp - bym) / (2.0 * h)],
        ]
    }

    /// One-dimensional Boltzmann partition function `∫ exp(-V/kT) dy`.
    pub fn boltzmann_line_partition(
        potential: &dyn Fn(f64) -> f64,
        kt: f64,
        extent: f64,
    ) -> f64 {
        bump_integral(&|y| (-potential(y) / kt).exp(), extent, 1.0e-11)
    }

    /// `F_full - F_harm` for a one-dimensional line density with potentials
    /// `full` and `harmonic`.
    pub fn anharmonic_line_free_energy(
        full: &dyn Fn(f64) -> f64,
        harmonic: &dyn Fn(f64) -> f64,
        kt: f64,
        extent: f64,
    ) -> f64 {
        let full_partition = boltzmann_line_partition(full, kt, extent);
        let harmonic_partition = boltzmann_line_partition(harmonic, kt, extent);
        -kt * (full_partition / harmonic_partition).ln()
    }

    /// Eigenvalues (ascending) and the unit eigenvector of the lower one
    /// of a symmetric 2×2 matrix.
    pub fn eigen2(h: [[f64; 2]; 2]) -> (f64, f64, [f64; 2]) {
        let mean = 0.5 * (h[0][0] + h[1][1]);
        let half = 0.5 * (h[0][0] - h[1][1]);
        let radius = half.hypot(h[0][1]);
        let (low, high) = (mean - radius, mean + radius);
        let v = if h[0][1].abs() > 1.0e-300 {
            [h[0][1], low - h[0][0]]
        } else if h[0][0] <= h[1][1] {
            [1.0, 0.0]
        } else {
            [0.0, 1.0]
        };
        let norm = v[0].hypot(v[1]);
        (low, high, [v[0] / norm, v[1] / norm])
    }

    #[test]
    fn analytic_gradients_match_differences() {
        for energy in [
            LepsHo::energy as fn(f64, f64) -> _,
            EckartHo::energy,
            SkewedWell::energy,
        ] {
            for (x, y) in [(0.9, 1.1), (2.0, -0.2), (2.6, -1.0), (0.1, 0.4)] {
                let (_, gx, gy) = energy(x, y);
                let h = 1.0e-6;
                let fx = (energy(x + h, y).0 - energy(x - h, y).0) / (2.0 * h);
                let fy = (energy(x, y + h).0 - energy(x, y - h).0) / (2.0 * h);
                assert!((fx - gx).abs() < 1.0e-6 * (1.0 + gx.abs()));
                assert!((fy - gy).abs() < 1.0e-6 * (1.0 + gy.abs()));
            }
        }
    }
}
