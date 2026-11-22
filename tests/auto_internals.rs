//! Automatic internals: topology on small molecules, the rebuild path,
//! and parity with hand-specified charts on a first-order saddle
//! search.
//!
//! Every surface here is a sum of springs over a hand-written list of
//! bonds and angles plus one torsion or stretch term that makes a known
//! first-order saddle. The list is the test's own: it never reads the
//! topology under test.

use ndarray::{Array1, Array2, ArrayView1};
use rgsaddle::eigensolve::exact_eigh;
use rgsaddle::internal::pack_cart;
use rgsaddle::mic::Cell;
use rgsaddle::minmode::PointSurface;
use rgsaddle::{
    Constraints, Equality, HessUpdate, HessianCarry, InternalPes, InternalTopology, RebuildReason,
    SaddleError, SellaSaddleConfig, SellaSaddleSession, TopologyConfig, check_chart,
};

type Gradient = (f64, Array1<f64>);

/// Springs on a hand-written list of primitives, plus an optional
/// one-dihedral barrier.
struct Springs {
    chart: Constraints,
    k: Vec<f64>,
    barrier: Option<Barrier>,
    umbrella: Option<Umbrella>,
}

/// `V0 ((h/h0)^2 - 1)^2` on the height `h` of atom 0 over the plane
/// of atoms 1, 2 and 3; its gradient is a central difference.
struct Umbrella {
    v0: f64,
    h0: f64,
}

impl Umbrella {
    fn height(x: ArrayView1<f64>) -> f64 {
        let p = |a: usize| [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
        let (n, h1, h2, h3) = (p(0), p(1), p(2), p(3));
        let u = [h2[0] - h1[0], h2[1] - h1[1], h2[2] - h1[2]];
        let v = [h3[0] - h1[0], h3[1] - h1[1], h3[2] - h1[2]];
        let c = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        let nc = (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt();
        let d = [n[0] - h1[0], n[1] - h1[1], n[2] - h1[2]];
        (d[0] * c[0] + d[1] * c[1] + d[2] * c[2]) / nc
    }

    fn energy(&self, x: ArrayView1<f64>) -> f64 {
        let h = Self::height(x);
        self.v0 * ((h / self.h0).powi(2) - 1.0).powi(2)
    }
}

/// `V0/2 (1 + sign cos(n phi))` on one dihedral.
struct Barrier {
    chart: Constraints,
    v0: f64,
    n: f64,
    sign: f64,
}

impl PointSurface for Springs {
    fn eval(&self, x: ArrayView1<f64>) -> Result<Gradient, SaddleError> {
        let r = self.chart.residual(x)?;
        let j = self.chart.jacobian(x)?;
        let mut e = 0.0;
        let mut w = Array1::zeros(r.len());
        for i in 0..r.len() {
            e += 0.5 * self.k[i] * r[i] * r[i];
            w[i] = self.k[i] * r[i];
        }
        let mut g = j.t().dot(&w);
        if let Some(b) = &self.barrier {
            let phi = b.chart.values(x)?[0];
            let jb = b.chart.jacobian(x)?;
            e += 0.5 * b.v0 * (1.0 + b.sign * (b.n * phi).cos());
            let de = -0.5 * b.v0 * b.sign * b.n * (b.n * phi).sin();
            for c in 0..g.len() {
                g[c] += de * jb[(0, c)];
            }
        }
        if let Some(u) = &self.umbrella {
            e += u.energy(x);
            let step = 1.0e-6;
            let mut xp = x.to_owned();
            for c in 0..g.len() {
                let x0 = xp[c];
                xp[c] = x0 + step;
                let ep = u.energy(xp.view());
                xp[c] = x0 - step;
                let em = u.energy(xp.view());
                xp[c] = x0;
                g[c] += (ep - em) / (2.0 * step);
            }
        }
        Ok((e, g))
    }
}

/// Lattice rows and, per atom, the image that puts it back in the
/// reference molecule.
type Periodic<'a> = Option<([[f64; 3]; 3], &'a [[i32; 3]])>;

fn springs(
    x_ref: &Array1<f64>,
    bonds: &[[usize; 2]],
    angles: &[[usize; 3]],
    kb: f64,
    ka: f64,
    barrier: Option<([usize; 4], f64, f64, f64)>,
) -> Springs {
    springs_in(x_ref, bonds, angles, kb, ka, barrier, None)
}

fn springs_in(
    x_ref: &Array1<f64>,
    bonds: &[[usize; 2]],
    angles: &[[usize; 3]],
    kb: f64,
    ka: f64,
    barrier: Option<([usize; 4], f64, f64, f64)>,
    periodic: Periodic<'_>,
) -> Springs {
    let n = x_ref.len() / 3;
    let pick = |atoms: &[usize]| -> Vec<[i32; 3]> {
        atoms
            .iter()
            .map(|&a| periodic.map_or([0; 3], |(_, img)| img[a]))
            .collect()
    };
    let fresh = || {
        let mut c = Constraints::new(n).unwrap();
        if let Some((lattice, _)) = periodic {
            c.set_lattice(lattice);
        }
        c
    };
    let mut chart = fresh();
    let mut k = Vec::new();
    for b in bonds {
        let im = pick(b);
        chart
            .fix_bond_image(*b, [im[0], im[1]], x_ref.view(), None)
            .unwrap();
        k.push(kb);
    }
    for a in angles {
        let im = pick(a);
        chart
            .fix_angle_image(*a, [im[0], im[1], im[2]], x_ref.view(), None)
            .unwrap();
        k.push(ka);
    }
    let barrier = barrier.map(|(atoms, v0, nn, sign)| {
        let mut c = fresh();
        let im = pick(&atoms);
        c.fix_dihedral_image(atoms, [im[0], im[1], im[2], im[3]], x_ref.view(), None)
            .unwrap();
        Barrier {
            chart: c,
            v0,
            n: nn,
            sign,
        }
    });
    Springs {
        chart,
        k,
        barrier,
        umbrella: None,
    }
}

fn angles_at_vertices(bonds: &[[usize; 2]]) -> Vec<[usize; 3]> {
    let n = bonds.iter().flatten().max().map_or(0, |m| m + 1);
    let mut nbr = vec![Vec::new(); n];
    for b in bonds {
        nbr[b[0]].push(b[1]);
        nbr[b[1]].push(b[0]);
    }
    let mut out = Vec::new();
    for (v, list) in nbr.iter().enumerate() {
        for p in 0..list.len() {
            for q in (p + 1)..list.len() {
                out.push([list[p], v, list[q]]);
            }
        }
    }
    out
}

fn rotate(x: &mut Array1<f64>, atoms: &[usize], origin: [f64; 3], axis: [f64; 3], angle: f64) {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let k = [axis[0] / norm, axis[1] / norm, axis[2] / norm];
    let (s, c) = angle.sin_cos();
    for &a in atoms {
        let v = [
            x[3 * a] - origin[0],
            x[3 * a + 1] - origin[1],
            x[3 * a + 2] - origin[2],
        ];
        let kv = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
        let cr = [
            k[1] * v[2] - k[2] * v[1],
            k[2] * v[0] - k[0] * v[2],
            k[0] * v[1] - k[1] * v[0],
        ];
        for d in 0..3 {
            x[3 * a + d] = origin[d] + v[d] * c + cr[d] * s + k[d] * kv * (1.0 - c);
        }
    }
}

/// Kabsch RMSD by Horn's quaternion method.
fn rmsd(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let n = a.len() / 3;
    let centre = |x: &Array1<f64>| {
        let mut c = [0.0; 3];
        for i in 0..n {
            for d in 0..3 {
                c[d] += x[3 * i + d] / n as f64;
            }
        }
        c
    };
    let (ca, cb) = (centre(a), centre(b));
    let mut m = [[0.0; 3]; 3];
    for i in 0..n {
        for p in 0..3 {
            for q in 0..3 {
                m[p][q] += (a[3 * i + p] - ca[p]) * (b[3 * i + q] - cb[q]);
            }
        }
    }
    let mut nm = Array2::zeros((4, 4));
    nm[(0, 0)] = m[0][0] + m[1][1] + m[2][2];
    nm[(0, 1)] = m[1][2] - m[2][1];
    nm[(0, 2)] = m[2][0] - m[0][2];
    nm[(0, 3)] = m[0][1] - m[1][0];
    nm[(1, 1)] = m[0][0] - m[1][1] - m[2][2];
    nm[(1, 2)] = m[0][1] + m[1][0];
    nm[(1, 3)] = m[2][0] + m[0][2];
    nm[(2, 2)] = -m[0][0] + m[1][1] - m[2][2];
    nm[(2, 3)] = m[1][2] + m[2][1];
    nm[(3, 3)] = -m[0][0] - m[1][1] + m[2][2];
    for i in 0..4 {
        for j in 0..i {
            nm[(i, j)] = nm[(j, i)];
        }
    }
    let (_, vec) = exact_eigh(nm.view()).unwrap();
    let q = vec.column(3);
    let (w, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
    let rot = [
        [
            w * w + qx * qx - qy * qy - qz * qz,
            2.0 * (qx * qy - w * qz),
            2.0 * (qx * qz + w * qy),
        ],
        [
            2.0 * (qx * qy + w * qz),
            w * w - qx * qx + qy * qy - qz * qz,
            2.0 * (qy * qz - w * qx),
        ],
        [
            2.0 * (qx * qz - w * qy),
            2.0 * (qy * qz + w * qx),
            w * w - qx * qx - qy * qy + qz * qz,
        ],
    ];
    let mut acc = 0.0;
    for i in 0..n {
        let v = [a[3 * i] - ca[0], a[3 * i + 1] - ca[1], a[3 * i + 2] - ca[2]];
        for p in 0..3 {
            let r = rot[p][0] * v[0] + rot[p][1] * v[1] + rot[p][2] * v[2];
            let d = r - (b[3 * i + p] - cb[p]);
            acc += d * d;
        }
    }
    (acc / n as f64).sqrt()
}

/// Eigenvalues of the central-difference Hessian.
fn fd_spectrum(s: &dyn PointSurface, x: &Array1<f64>) -> Array1<f64> {
    let n = x.len();
    let h = 1.0e-4;
    let mut hess = Array2::zeros((n, n));
    for i in 0..n {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[i] += h;
        xm[i] -= h;
        let gp = s.eval(xp.view()).unwrap().1;
        let gm = s.eval(xm.view()).unwrap().1;
        for j in 0..n {
            hess[(j, i)] = (gp[j] - gm[j]) / (2.0 * h);
        }
    }
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (hess[(i, j)] + hess[(j, i)]);
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
    exact_eigh(hess.view()).unwrap().0
}

fn ethane() -> (Vec<u8>, Array1<f64>) {
    let rcc = 1.54;
    let rch = 1.09;
    let cos_t = -1.0 / 3.0;
    let sin_t = 8.0_f64.sqrt() / 3.0;
    let mut rows = vec![[0.0, 0.0, -rcc / 2.0], [0.0, 0.0, rcc / 2.0]];
    let mut z = vec![6u8, 6];
    for (zc, flip, phase) in [(-rcc / 2.0, 1.0, 0.0), (rcc / 2.0, -1.0, 60.0)] {
        for k in 0..3 {
            let phi = (phase + 120.0 * f64::from(k)).to_radians();
            rows.push([
                rch * sin_t * phi.cos(),
                rch * sin_t * phi.sin(),
                zc + flip * rch * cos_t,
            ]);
            z.push(1);
        }
    }
    (z, pack_cart(&rows))
}

fn masses_for(z: &[u8]) -> Array1<f64> {
    z.iter()
        .map(|&zi| match zi {
            1 => 1.008,
            6 => 12.011,
            7 => 14.007,
            8 => 15.999,
            _ => 20.0,
        })
        .collect()
}

#[test]
fn ethane_topology_counts_and_chart_values_follow_the_geometry() {
    let (z, x) = ethane();
    let t = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
    let c = t.constraints().counts();
    assert_eq!((c.nbonds, c.nangles, c.ndihedrals), (7, 12, 9));
    let q = t.constraints().values(x.view()).unwrap();
    let bond = |a: usize, b: usize| {
        t.constraints()
            .equalities()
            .iter()
            .position(|e| matches!(e, Equality::Bond { atoms, .. } if *atoms == [a, b]))
            .map(|k| q[k])
    };
    assert!((bond(0, 1).unwrap() - 1.54).abs() < 1.0e-12);
    assert!((bond(0, 2).unwrap() - 1.09).abs() < 1.0e-12);
    assert!(t.check().complete);
}

/// A-B-A held planar, with a bend potential `a (cos t - c0)^2`: bent
/// minima, and a maximum at the linear geometry. That geometry is the
/// first-order saddle; the stiff out-of-plane term keeps the second
/// bend direction positive.
struct BendWell {
    kr: f64,
    r0: f64,
    a: f64,
    c0: f64,
    kz: f64,
}

impl PointSurface for BendWell {
    fn eval(&self, x: ArrayView1<f64>) -> Result<Gradient, SaddleError> {
        let u = [x[0] - x[3], x[1] - x[4], x[2] - x[5]];
        let v = [x[6] - x[3], x[7] - x[4], x[8] - x[5]];
        let r1 = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        let r2 = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (r1 * r2);
        let e = 0.5 * self.kr * ((r1 - self.r0).powi(2) + (r2 - self.r0).powi(2))
            + self.a * (cos - self.c0).powi(2)
            + 0.5 * self.kz * (u[2] * u[2] + v[2] * v[2]);
        let de_dcos = 2.0 * self.a * (cos - self.c0);
        let mut g = Array1::zeros(9);
        for d in 0..3 {
            let mut gu = self.kr * (r1 - self.r0) * u[d] / r1
                + de_dcos * (v[d] / (r1 * r2) - cos * u[d] / (r1 * r1));
            let mut gv = self.kr * (r2 - self.r0) * v[d] / r2
                + de_dcos * (u[d] / (r1 * r2) - cos * v[d] / (r2 * r2));
            if d == 2 {
                gu += self.kz * u[2];
                gv += self.kz * v[2];
            }
            g[d] += gu;
            g[6 + d] += gv;
            g[3 + d] -= gu + gv;
        }
        Ok((e, g))
    }
}

fn bent_triatomic(theta_deg: f64, r1: f64, r2: f64) -> Array1<f64> {
    let half = (theta_deg / 2.0).to_radians();
    pack_cart(&[
        [-r1 * half.sin(), r1 * half.cos(), 0.0],
        [0.0, 0.0, 0.0],
        [r2 * half.sin(), r2 * half.cos(), 0.0],
    ])
}

fn angle_at_middle(x: &Array1<f64>) -> f64 {
    let u = [x[0] - x[3], x[1] - x[4], x[2] - x[5]];
    let v = [x[6] - x[3], x[7] - x[4], x[8] - x[5]];
    let cr = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let nc = (cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]).sqrt();
    nc.atan2(u[0] * v[0] + u[1] * v[1] + u[2] * v[2])
        .to_degrees()
}

#[test]
fn a_search_through_the_linear_window_rebuilds_and_finds_the_linear_saddle() {
    let surf = BendWell {
        kr: 4.0,
        r0: 1.1,
        a: 2.0,
        c0: -0.25,
        kz: 5.0,
    };
    let x0 = bent_triatomic(140.0, 1.15, 1.08);
    let z = [8u8, 6, 8];
    let cfg = SellaSaddleConfig::default()
        .with_order(1)
        .with_force_tol(1.0e-5);
    let mut sess = SellaSaddleSession::on_auto_internal(
        cfg,
        x0,
        masses_for(&z),
        &z,
        None,
        &TopologyConfig::default(),
    )
    .unwrap();
    sess.set_update(HessUpdate::TsBfgs);
    assert_eq!(sess.topology_restarts(), 0);
    let mut done = false;
    for _ in 0..200 {
        let rep = sess.step(&surf).unwrap();
        if rep.at_saddle {
            done = true;
            break;
        }
    }
    let x = sess.position().to_owned();
    assert!(done, "no saddle; angle {}", angle_at_middle(&x));
    assert!(sess.topology_restarts() >= 1, "no rebuild happened");
    assert!(angle_at_middle(&x) > 179.9);
    let (e, _) = surf.eval(x.view()).unwrap();
    assert!((e - 2.0 * 0.75 * 0.75).abs() < 1.0e-6, "energy {e}");
    let pes = sess.internal_pes().unwrap();
    assert_eq!(pes.n_dummy(), 1);
    assert!(pes.topology_epoch() >= 1);
    let ev = fd_spectrum(&surf, &x);
    assert_eq!(ev.iter().filter(|l| **l < -1.0e-3).count(), 1);
}

fn dihedral_of(x: &Array1<f64>, atoms: [usize; 4]) -> f64 {
    let mut c = Constraints::new(x.len() / 3).unwrap();
    c.fix_dihedral(atoms, x.view(), None).unwrap();
    c.values(x.view()).unwrap()[0]
}

fn wrap(d: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    d - tau * (d / tau).round()
}

/// Rotate `moved` about the axis until `dihedral` reads `target` (radians).
fn twist_to(
    x: &Array1<f64>,
    moved: &[usize],
    origin: [f64; 3],
    axis: [f64; 3],
    dihedral: [usize; 4],
    target: f64,
) -> Array1<f64> {
    let f = |angle: f64| {
        let mut y = x.clone();
        rotate(&mut y, moved, origin, axis, angle);
        wrap(dihedral_of(&y, dihedral) - target)
    };
    let steps = 720;
    let mut prev = f(0.0);
    for k in 1..=steps {
        let a1 = std::f64::consts::TAU * k as f64 / steps as f64;
        let a0 = std::f64::consts::TAU * (k - 1) as f64 / steps as f64;
        let cur = f(a1);
        if cur == 0.0 {
            let mut y = x.clone();
            rotate(&mut y, moved, origin, axis, a1);
            return y;
        }
        if prev * cur < 0.0 && (prev - cur).abs() < 1.0 {
            let (mut lo, mut hi) = (a0, a1);
            let mut flo = prev;
            for _ in 0..80 {
                let mid = 0.5 * (lo + hi);
                let fm = f(mid);
                if flo * fm <= 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    flo = fm;
                }
            }
            let mut y = x.clone();
            rotate(&mut y, moved, origin, axis, 0.5 * (lo + hi));
            return y;
        }
        prev = cur;
    }
    panic!("no twist reaches the target dihedral");
}

struct Mol {
    name: &'static str,
    z: Vec<u8>,
    x_start: Array1<f64>,
    bonds: Vec<[usize; 2]>,
    surface: Springs,
    hand_dihedrals: Vec<[usize; 4]>,
    ts_energy: Option<f64>,
}

fn methyl_rotor(z_heavy: u8, r_cx: f64, r_xh: f64, ang_cxh_deg: f64, v0: f64) -> Mol {
    // C at the origin, X on +z, H on X in the xz plane, three methyl H.
    let ang = ang_cxh_deg.to_radians();
    let hx = [
        r_xh * (std::f64::consts::PI - ang).sin(),
        0.0,
        r_cx - r_xh * (std::f64::consts::PI - ang).cos(),
    ];
    let cos_h = (109.5_f64).to_radians().cos();
    let sin_h = (109.5_f64).to_radians().sin();
    let mut rows = vec![[0.0; 3], [0.0, 0.0, r_cx], hx];
    for az in [180.0_f64, 300.0, 60.0] {
        let a = az.to_radians();
        rows.push([1.09 * sin_h * a.cos(), 1.09 * sin_h * a.sin(), 1.09 * cos_h]);
    }
    let x_ref = pack_cart(&rows);
    let bonds = vec![[0, 1], [1, 2], [0, 3], [0, 4], [0, 5]];
    let angles = angles_at_vertices(&bonds);
    let surface = springs(
        &x_ref,
        &bonds,
        &angles,
        6.0,
        3.0,
        Some(([2, 1, 0, 3], v0, 3.0, 1.0)),
    );
    // Staggered at 180; the saddle is the eclipsed 120.
    let start = twist_to(
        &x_ref,
        &[3, 4, 5],
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [2, 1, 0, 3],
        130.0_f64.to_radians(),
    );
    Mol {
        name: if z_heavy == 8 {
            "methanol"
        } else {
            "methanethiol"
        },
        z: vec![6, z_heavy, 1, 1, 1, 1],
        x_start: start,
        bonds,
        surface,
        hand_dihedrals: vec![[2, 1, 0, 3], [2, 1, 0, 4], [2, 1, 0, 5]],
        ts_energy: Some(v0),
    }
}

fn molecules() -> Vec<Mol> {
    let mut out = Vec::new();

    let (zc, xe) = ethane();
    let bonds = vec![[0, 1], [0, 2], [0, 3], [0, 4], [1, 5], [1, 6], [1, 7]];
    let angles = angles_at_vertices(&bonds);
    let surface = springs(
        &xe,
        &bonds,
        &angles,
        6.0,
        3.0,
        Some(([2, 0, 1, 5], 1.0, 3.0, 1.0)),
    );
    let start = twist_to(
        &xe,
        &[1, 5, 6, 7],
        [0.0; 3],
        [0.0, 0.0, 1.0],
        [2, 0, 1, 5],
        25.0_f64.to_radians(),
    );
    out.push(Mol {
        name: "ethane",
        z: zc,
        x_start: start,
        bonds,
        surface,
        hand_dihedrals: vec![[2, 0, 1, 5], [3, 0, 1, 6], [4, 0, 1, 7]],
        ts_energy: Some(1.0),
    });

    let hooh_ref = {
        let (c, s) = (100.0_f64.to_radians().cos(), 100.0_f64.to_radians().sin());
        pack_cart(&[
            [0.0, 0.0, 0.0],
            [1.45, 0.0, 0.0],
            [0.97 * c, 0.97 * s, 0.0],
            [1.45 - 0.97 * c, 0.97 * s, 0.0],
        ])
    };
    let bonds = vec![[0, 1], [0, 2], [1, 3]];
    let angles = angles_at_vertices(&bonds);
    let surface = springs(
        &hooh_ref,
        &bonds,
        &angles,
        6.0,
        3.0,
        Some(([2, 0, 1, 3], 1.0, 1.0, -1.0)),
    );
    let start = twist_to(
        &hooh_ref,
        &[3],
        [1.45, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2, 0, 1, 3],
        150.0_f64.to_radians(),
    );
    out.push(Mol {
        name: "hydrogen peroxide",
        z: vec![8, 8, 1, 1],
        x_start: start,
        bonds,
        surface,
        hand_dihedrals: vec![[2, 0, 1, 3]],
        ts_energy: Some(1.0),
    });

    out.push(methyl_rotor(8, 1.43, 0.96, 108.0, 1.0));
    out.push(methyl_rotor(16, 1.82, 1.34, 96.5, 1.0));

    let r = 1.012;
    let a = 106.7_f64.to_radians();
    let rho = r * (a / 2.0).sin() / (std::f64::consts::PI / 3.0).sin();
    let h_below = -(r * r - rho * rho).sqrt();
    let rows = [
        [0.0, 0.0, 0.0],
        [rho, 0.0, h_below],
        [
            rho * (2.0 * std::f64::consts::PI / 3.0).cos(),
            rho * (2.0 * std::f64::consts::PI / 3.0).sin(),
            h_below,
        ],
        [
            rho * (4.0 * std::f64::consts::PI / 3.0).cos(),
            rho * (4.0 * std::f64::consts::PI / 3.0).sin(),
            h_below,
        ],
    ];
    let nh3 = pack_cart(&rows);
    let bonds = vec![[0, 1], [0, 2], [0, 3]];
    let angles = angles_at_vertices(&bonds);
    let mut surface = springs(&nh3, &bonds, &angles, 6.0, 0.3, None);
    surface.umbrella = Some(Umbrella {
        v0: 2.0,
        h0: -h_below,
    });
    out.push(Mol {
        name: "ammonia",
        z: vec![7, 1, 1, 1],
        x_start: {
            let mut y = nh3.clone();
            for a in 1..4 {
                y[3 * a + 2] = h_below * 0.4;
            }
            y
        },
        bonds,
        surface,
        hand_dihedrals: vec![[1, 0, 2, 3]],
        ts_energy: None,
    });
    out
}

struct Outcome {
    x: Array1<f64>,
    energy: f64,
    converged: bool,
    restarts: usize,
}

fn search(mol: &Mol, auto: bool) -> Outcome {
    let cfg = SellaSaddleConfig::default()
        .with_order(1)
        .with_force_tol(1.0e-5);
    let n = mol.z.len();
    let masses = masses_for(&mol.z);
    let x0 = mol.x_start.clone();
    let sess = if auto {
        SellaSaddleSession::on_auto_internal(
            cfg,
            x0,
            masses,
            &mol.z,
            None,
            &TopologyConfig::default(),
        )
        .unwrap()
    } else {
        let mut chart = Constraints::new(n).unwrap();
        for b in &mol.bonds {
            chart.fix_bond(*b, mol.x_start.view(), None).unwrap();
        }
        for a in angles_at_vertices(&mol.bonds) {
            chart.fix_angle(a, mol.x_start.view(), None).unwrap();
        }
        for d in &mol.hand_dihedrals {
            chart.fix_dihedral(*d, mol.x_start.view(), None).unwrap();
        }
        let k = check_chart(&chart, None, mol.x_start.view(), n, 1.0e-6).unwrap();
        assert!(k.complete, "{}: hand chart incomplete: {k:?}", mol.name);
        SellaSaddleSession::on_internal(cfg, x0, masses, chart).unwrap()
    };
    drive(sess, &mol.surface)
}

fn drive<S: PointSurface>(mut sess: SellaSaddleSession, surface: &S) -> Outcome {
    sess.set_update(HessUpdate::TsBfgs);
    let mut converged = false;
    let mut energy = f64::NAN;
    for _ in 0..300 {
        let rep = sess.step(surface).unwrap();
        energy = rep.energy;
        if rep.at_saddle {
            converged = true;
            break;
        }
    }
    Outcome {
        x: sess.position().to_owned(),
        energy,
        converged,
        restarts: sess.topology_restarts(),
    }
}

#[test]
fn auto_internals_find_the_same_saddle_as_a_hand_chart_on_five_molecules() {
    for mol in molecules() {
        let hand = search(&mol, false);
        let auto = search(&mol, true);
        assert!(hand.converged, "{}: hand chart did not converge", mol.name);
        assert!(
            auto.converged,
            "{}: automatic chart did not converge",
            mol.name
        );
        assert!(
            (hand.energy - auto.energy).abs() < 1.0e-6,
            "{}: energies {} vs {}",
            mol.name,
            hand.energy,
            auto.energy
        );
        let d = rmsd(&hand.x, &auto.x);
        assert!(d < 1.0e-3, "{}: rmsd {d}", mol.name);
        if let Some(e) = mol.ts_energy {
            assert!(
                (auto.energy - e).abs() < 1.0e-6,
                "{}: {}",
                mol.name,
                auto.energy
            );
        }
        let ev = fd_spectrum(&mol.surface, &auto.x);
        let neg = ev.iter().filter(|l| **l < -1.0e-3).count();
        assert_eq!(neg, 1, "{}: {} negative modes", mol.name, neg);
        eprintln!(
            "{}: ok, {} rebuilds, E={:.6}, rmsd={d:.2e}",
            mol.name, auto.restarts, auto.energy
        );
    }
}

fn nh3(height: f64) -> Array1<f64> {
    let rho = 0.82;
    let at = |k: f64| {
        let a = k * 2.0 * std::f64::consts::PI / 3.0;
        [rho * a.cos(), rho * a.sin(), 0.0]
    };
    let (a, b, c) = (at(0.0), at(1.0), at(2.0));
    pack_cart(&[[0.0, 0.0, height], a, b, c])
}

const NH3: [u8; 4] = [7, 1, 1, 1];

#[test]
fn a_periodic_search_across_the_cell_boundary_finds_the_free_saddle() {
    let mol = molecules().remove(0);
    assert_eq!(mol.name, "ethane");
    let edge = 10.0;
    let lattice = [[edge, 0.0, 0.0], [0.0, edge, 0.0], [0.0, 0.0, edge]];
    let cell = Cell::ortho(edge, edge, edge).unwrap();
    let n = mol.z.len();
    let mut shifts = vec![[0_i32; 3]; n];
    let mut wrapped = mol.x_start.clone();
    for a in 0..n {
        for d in 0..3 {
            let s = -(mol.x_start[3 * a + d] / edge).floor();
            shifts[a][d] = s as i32;
            wrapped[3 * a + d] += s * edge;
        }
    }
    assert!(shifts.iter().any(|s| s.iter().any(|v| *v != 0)));
    assert!(shifts.iter().any(|s| s.iter().all(|v| *v == 0)));
    let images: Vec<[i32; 3]> = shifts.iter().map(|s| [-s[0], -s[1], -s[2]]).collect();
    let (_, xe) = ethane();
    let mut xe_wrapped = xe.clone();
    for a in 0..n {
        for d in 0..3 {
            xe_wrapped[3 * a + d] += f64::from(shifts[a][d]) * edge;
        }
    }
    let angles = angles_at_vertices(&mol.bonds);
    let surface = springs_in(
        &xe_wrapped,
        &mol.bonds,
        &angles,
        6.0,
        3.0,
        Some(([2, 0, 1, 5], 1.0, 3.0, 1.0)),
        Some((lattice, &images)),
    );

    let free = search(&mol, true);
    assert!(free.converged);

    let naive = InternalTopology::from_geometry(&mol.z, wrapped.view(), None)
        .unwrap()
        .constraints()
        .counts();
    let aware = InternalTopology::from_geometry(&mol.z, wrapped.view(), Some(&cell))
        .unwrap()
        .constraints()
        .counts();
    assert_eq!((aware.nbonds, aware.nangles, aware.ndihedrals), (7, 12, 9));
    assert_ne!((naive.nangles, naive.ndihedrals), (12, 9));

    let cfg = SellaSaddleConfig::default()
        .with_order(1)
        .with_force_tol(1.0e-5);
    let sess = SellaSaddleSession::on_auto_internal(
        cfg,
        wrapped,
        masses_for(&mol.z),
        &mol.z,
        Some(&cell),
        &TopologyConfig::default(),
    )
    .unwrap();
    assert!(sess.internal_pes().unwrap().chart().is_periodic());
    let periodic = drive(sess, &surface);
    assert!(periodic.converged, "periodic search did not converge");
    assert!((periodic.energy - free.energy).abs() < 1.0e-6);

    let mut unwrapped = periodic.x.clone();
    for a in 0..n {
        for d in 0..3 {
            unwrapped[3 * a + d] -= f64::from(shifts[a][d]) * edge;
        }
    }
    let d = rmsd(&unwrapped, &free.x);
    assert!(d < 1.0e-3, "rmsd {d}");
    let ev = fd_spectrum(&surface, &periodic.x);
    assert_eq!(ev.iter().filter(|l| **l < -1.0e-3).count(), 1);
}

#[test]
fn a_singular_chart_is_rebuilt_with_the_hessian_carried() {
    let pyramid = nh3(0.38);
    let flat = nh3(0.0);
    let topo = InternalTopology::from_geometry(&NH3, pyramid.view(), None).unwrap();
    assert_eq!(topo.constraints().counts().ndihedrals, 0);
    let masses = masses_for(&NH3);
    let mut pes = InternalPes::from_topology(flat.clone(), masses.clone(), &topo, &NH3).unwrap();

    let broken = pes.topology_check().unwrap();
    assert!(!broken.complete);
    assert_eq!((broken.rank, broken.expected_rank), (5, 6));
    assert_eq!(
        pes.stale_reason().unwrap(),
        Some(RebuildReason::SingularJacobian)
    );
    let old = pes.b_matrix().unwrap();
    let h_old = pes.hessian().hessian().clone();
    let hx_old = old.t().dot(&h_old).dot(&old);

    let report = pes.rebuild_if_stale().unwrap().unwrap();
    assert_eq!(report.reason, RebuildReason::SingularJacobian);
    assert_eq!(
        (report.epoch, report.n_int_before, report.n_int_after),
        (1, 6, 7)
    );
    assert_eq!(report.hessian, HessianCarry::Transformed);
    assert_eq!(pes.topology_epoch(), 1);
    assert!(pes.topology_check().unwrap().complete);
    assert_eq!(pes.stale_reason().unwrap(), None);

    let new = pes.b_matrix().unwrap();
    let hx_new = new.t().dot(pes.hessian().hessian()).dot(&new);
    let scale = hx_old.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = (&hx_new - &hx_old)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        worst < 1.0e-9 * scale,
        "Cartesian curvature moved by {worst}"
    );

    let fresh_topo = InternalTopology::from_geometry(&NH3, flat.view(), None).unwrap();
    let fresh = InternalPes::from_topology(flat, masses, &fresh_topo, &NH3).unwrap();
    let fb = fresh.b_matrix().unwrap();
    let reseeded = fb.t().dot(fresh.hessian().hessian()).dot(&fb);
    let apart = (&reseeded - &hx_new)
        .iter()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(
        apart > 1.0e-3 * scale,
        "a transported Hessian must differ from the reseed"
    );
}

#[test]
fn a_pyramidal_centre_that_flattens_is_rebuilt_with_an_improper() {
    let topo = InternalTopology::from_geometry(&NH3, nh3(0.38).view(), None).unwrap();
    let masses = masses_for(&NH3);
    let mut pes = InternalPes::from_topology(nh3(0.38), masses.clone(), &topo, &NH3).unwrap();
    assert_eq!(pes.stale_reason().unwrap(), None);

    let shallow = nh3(0.12);
    let mut pes_shallow = InternalPes::from_topology(shallow, masses, &topo, &NH3).unwrap();
    assert!(pes_shallow.topology_check().unwrap().complete);
    assert_eq!(
        pes_shallow.stale_reason().unwrap(),
        Some(RebuildReason::NearPlanar)
    );
    let report = pes_shallow.rebuild_if_stale().unwrap().unwrap();
    assert_eq!(report.reason, RebuildReason::NearPlanar);
    assert_eq!((report.n_int_before, report.n_int_after), (6, 7));
    assert_eq!(pes_shallow.stale_reason().unwrap(), None);
    assert!(pes.rebuild_if_stale().unwrap().is_none());
    assert_eq!(pes.topology_epoch(), 0);
}

#[test]
fn a_bend_that_nears_linear_is_rebuilt_with_a_dummy() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let topo = InternalTopology::from_geometry(&z, bent.view(), None).unwrap();
    assert_eq!(topo.n_dummy(), 0);
    let straight = bent_triatomic(172.0, 1.16, 1.16);
    let mut pes = InternalPes::from_topology(straight, masses_for(&z), &topo, &z).unwrap();
    assert_eq!(
        pes.stale_reason().unwrap(),
        Some(RebuildReason::NearLinearBend)
    );
    let report = pes.rebuild_if_stale().unwrap().unwrap();
    assert_eq!(report.reason, RebuildReason::NearLinearBend);
    assert_eq!((report.n_dummy_before, report.n_dummy_after), (0, 1));
    assert_eq!(pes.n_dummy(), 1);
    assert_eq!(pes.stale_reason().unwrap(), None);
    assert!(pes.topology_check().unwrap().complete);
}

#[test]
fn coincident_atoms_flag_the_chart_and_the_rebuild_refuses() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let topo = InternalTopology::from_geometry(&z, bent.view(), None).unwrap();
    let mut squashed = bent.clone();
    for d in 0..3 {
        squashed[d] = squashed[3 + d];
    }
    let mut pes = InternalPes::from_topology(squashed, masses_for(&z), &topo, &z).unwrap();
    assert_eq!(
        pes.stale_reason().unwrap(),
        Some(RebuildReason::SingularJacobian)
    );
    assert!(pes.rebuild_if_stale().is_err());
    assert_eq!(pes.topology_epoch(), 0);
}

#[test]
fn a_back_transformation_that_cannot_converge_flags_a_rebuild() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let tight = TopologyConfig::default().with_back_transform_iterations(1);
    let mut pes = InternalPes::auto(bent.clone(), masses_for(&z), &z, None, &tight).unwrap();
    let mut dq = Array1::zeros(pes.n_int());
    dq[pes.n_int() - 1] = 0.4;
    pes.displace(dq.view()).unwrap();
    assert_eq!(pes.pending_rebuild(), Some(RebuildReason::BackTransform));
    assert_eq!(
        pes.stale_reason().unwrap(),
        Some(RebuildReason::BackTransform)
    );
    let report = pes.rebuild_if_stale().unwrap().unwrap();
    assert_eq!(report.reason, RebuildReason::BackTransform);
    assert_eq!(pes.pending_rebuild(), None);

    let easy = TopologyConfig::default();
    let mut pes = InternalPes::auto(bent, masses_for(&z), &z, None, &easy).unwrap();
    let mut dq = Array1::zeros(pes.n_int());
    dq[pes.n_int() - 1] = 0.05;
    pes.displace(dq.view()).unwrap();
    assert_eq!(pes.pending_rebuild(), None);
}

#[test]
fn a_hand_chart_is_never_rebuilt() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let mut chart = Constraints::new(3).unwrap();
    chart.fix_bond([0, 1], bent.view(), None).unwrap();
    chart.fix_bond([1, 2], bent.view(), None).unwrap();
    chart.fix_angle([0, 1, 2], bent.view(), None).unwrap();
    let straight = bent_triatomic(176.0, 1.16, 1.16);
    let mut pes = InternalPes::new(straight, masses_for(&z), chart).unwrap();
    assert!(!pes.is_auto());
    assert_eq!(pes.stale_reason().unwrap(), None);
    assert!(pes.rebuild_if_stale().unwrap().is_none());
    assert!(pes.rebuild_topology(RebuildReason::Requested).is_err());
    assert_eq!(pes.topology_epoch(), 0);
    assert!(pes.atom_map().is_none());
}

#[test]
fn a_requested_rebuild_works_on_an_automatic_chart() {
    let z = [8u8, 6, 8];
    let bent = bent_triatomic(120.0, 1.16, 1.16);
    let mut pes =
        InternalPes::auto(bent, masses_for(&z), &z, None, &TopologyConfig::default()).unwrap();
    assert!(pes.is_auto());
    let report = pes.rebuild_topology(RebuildReason::Requested).unwrap();
    assert_eq!((report.epoch, report.reason), (1, RebuildReason::Requested));
    assert_eq!(report.hessian, HessianCarry::Transformed);
    assert_eq!(report.n_int_before, report.n_int_after);
}

#[test]
fn a_hand_bond_and_angle_chart_and_the_dummy_chart_reach_the_same_linear_saddle() {
    let surf = BendWell {
        kr: 4.0,
        r0: 1.1,
        a: 2.0,
        c0: -0.25,
        kz: 5.0,
    };
    let x0 = bent_triatomic(140.0, 1.15, 1.08);
    let z = [8u8, 6, 8];
    let mut chart = Constraints::new(3).unwrap();
    chart.fix_bond([0, 1], x0.view(), None).unwrap();
    chart.fix_bond([1, 2], x0.view(), None).unwrap();
    chart.fix_angle([0, 1, 2], x0.view(), None).unwrap();
    let cfg = || {
        SellaSaddleConfig::default()
            .with_order(1)
            .with_force_tol(1.0e-5)
    };
    let hand = SellaSaddleSession::on_internal(cfg(), x0.clone(), masses_for(&z), chart).unwrap();
    let hand = drive(hand, &surf);
    let auto = SellaSaddleSession::on_auto_internal(
        cfg(),
        x0,
        masses_for(&z),
        &z,
        None,
        &TopologyConfig::default(),
    )
    .unwrap();
    let auto = drive(auto, &surf);
    assert!(hand.converged && auto.converged);
    assert!((hand.energy - auto.energy).abs() < 1.0e-6);
    assert!(angle_at_middle(&hand.x) > 179.9 && angle_at_middle(&auto.x) > 179.9);
    assert!(auto.restarts >= 1);
    assert_eq!(hand.restarts, 0);
}

#[test]
fn the_chart_is_the_same_for_a_rigidly_moved_molecule() {
    let (z, x) = ethane();
    let a = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
    let mut moved = x.clone();
    let atoms: Vec<usize> = (0..z.len()).collect();
    rotate(&mut moved, &atoms, [0.0; 3], [0.3, -0.5, 0.8], 1.1);
    for i in 0..z.len() {
        moved[3 * i] += 4.0;
        moved[3 * i + 1] -= 2.5;
        moved[3 * i + 2] += 0.7;
    }
    let b = InternalTopology::from_geometry(&z, moved.view(), None).unwrap();
    assert_eq!(
        a.constraints().equalities().len(),
        b.constraints().equalities().len()
    );
    for (ea, eb) in a
        .constraints()
        .equalities()
        .iter()
        .zip(b.constraints().equalities())
    {
        match (ea, eb) {
            (Equality::Bond { atoms: p, .. }, Equality::Bond { atoms: q, .. }) => assert_eq!(p, q),
            (Equality::Angle { atoms: p, .. }, Equality::Angle { atoms: q, .. }) => {
                assert_eq!(p, q)
            }
            (Equality::Dihedral { atoms: p, .. }, Equality::Dihedral { atoms: q, .. }) => {
                assert_eq!(p, q);
            }
            _ => panic!("the two charts list different primitives"),
        }
    }
    let qa = a.constraints().values(x.view()).unwrap();
    let qb = b.constraints().values(moved.view()).unwrap();
    for ((eq, u), v) in a
        .constraints()
        .equalities()
        .iter()
        .zip(qa.iter())
        .zip(qb.iter())
    {
        let gap = match eq {
            Equality::Dihedral { .. } => wrap(u - v),
            _ => u - v,
        };
        assert!(gap.abs() < 1.0e-10, "{eq:?}: {u} vs {v}");
    }
    let again = InternalTopology::from_geometry(&z, x.view(), None).unwrap();
    assert_eq!(
        format!("{:?}", a.constraints().equalities()),
        format!("{:?}", again.constraints().equalities())
    );
    assert!(a.check().complete && b.check().complete);
}
