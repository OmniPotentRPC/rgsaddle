//! Elastic field of many defects.
//!
//! The isotropic kernel is Kelvin's Green's function. The strain of a
//! dipole is the second derivative of that kernel, and the strain gradient
//! is the third. A [`DirectSolver`] sums every pair. [`FmmSolver`] is the
//! same sum with a dual tree walk: neighboring leaves are direct, and a
//! well-separated cell is translated into a local expansion
//! (multipole-to-local), which is the fast multipole method of Greengard
//! and Rokhlin, J. Comput. Phys. 73, 325 (1987),
//! doi:10.1016/0021-9991(87)90140-9.

use crate::cell_log::M3;
use crate::elastic::{LocalField, StrainGradient};
use crate::error::SaddleError;
use crate::minmode::PointSurface;
use crate::strain_rate::{ElasticDipole, Estimate, HopInField, SaddleSeed, Strain, StrainResponse};

const PI: f64 = std::f64::consts::PI;

/// One defect's centre and its elastic dipole.
#[derive(Clone, Copy, Debug)]
pub struct DefectSource {
    pub position: [f64; 3],
    pub dipole: ElasticDipole,
}

/// Isotropic Kelvin Green's function.
///
/// `shear_modulus` is μ. `poisson` is ν. The displacement due to a unit
/// point force is
///
/// `G_ij = A [(3 - 4ν) δ_ij / R + r_i r_j / R^3]`,
///
/// `A = 1 / (16 π μ (1 - ν))`.
#[derive(Clone, Copy, Debug)]
pub struct Kelvin {
    shear_modulus: f64,
    poisson: f64,
}

impl Kelvin {
    /// μ > 0 and -1 < ν < 1/2.
    pub fn new(shear_modulus: f64, poisson: f64) -> Result<Self, SaddleError> {
        if !(shear_modulus.is_finite() && shear_modulus > 0.0) {
            return Err(SaddleError::Invalid(
                "shear modulus must be finite and positive".into(),
            ));
        }
        if !(poisson.is_finite() && poisson > -1.0 && poisson < 0.5) {
            return Err(SaddleError::Invalid(
                "Poisson's ratio must lie between -1 and 1/2".into(),
            ));
        }
        Ok(Self {
            shear_modulus,
            poisson,
        })
    }

    fn amplitude(self) -> f64 {
        1.0 / (16.0 * PI * self.shear_modulus * (1.0 - self.poisson))
    }

    /// Symmetric strain at `r` due to a point force at the origin.
    pub fn strain_from_force(&self, r: [f64; 3], force: [f64; 3]) -> Result<M3, SaddleError> {
        let distance = length(r);
        if distance < 1e-12 {
            return Err(SaddleError::Invalid(
                "the Kelvin kernel is singular at the origin".into(),
            ));
        }
        let first = first_derivative(r, 3.0 - 4.0 * self.poisson);
        let mut grad = [[0.0; 3]; 3];
        let scale = self.amplitude();
        for i in 0..3 {
            for m in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    sum += first[i][j][m] * force[j];
                }
                grad[i][m] = scale * sum;
            }
        }
        Ok(symmetrize(grad))
    }

    /// Symmetric strain at `r` due to a dipole at the origin.
    ///
    /// `u_i = -∂_k G_ij P_jk`.
    pub fn strain_from_dipole(
        &self,
        r: [f64; 3],
        dipole: ElasticDipole,
    ) -> Result<M3, SaddleError> {
        let distance = length(r);
        if distance < 1e-12 {
            return Err(SaddleError::Invalid(
                "the Kelvin kernel is singular at the origin".into(),
            ));
        }
        let second = second_derivative(r, 3.0 - 4.0 * self.poisson);
        let p = dipole.tensor();
        let mut grad = [[0.0; 3]; 3];
        let scale = self.amplitude();
        for i in 0..3 {
            for m in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    for k in 0..3 {
                        sum += second[i][j][k][m] * p[j][k];
                    }
                }
                grad[i][m] = -scale * sum;
            }
        }
        Ok(symmetrize(grad))
    }

    /// `∂_n ε_ij` at `r` due to a dipole at the origin.
    pub fn strain_gradient_from_dipole(
        &self,
        r: [f64; 3],
        dipole: ElasticDipole,
    ) -> Result<StrainGradient, SaddleError> {
        let distance = length(r);
        if distance < 1e-12 {
            return Err(SaddleError::Invalid(
                "the Kelvin kernel is singular at the origin".into(),
            ));
        }
        let third = third_derivative(r, 3.0 - 4.0 * self.poisson);
        let p = dipole.tensor();
        let scale = self.amplitude();
        let mut raw = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    let mut sum = 0.0;
                    for j in 0..3 {
                        for k in 0..3 {
                            sum += third[i][j][k][m][n] * p[j][k];
                        }
                    }
                    raw[i][m][n] = -scale * sum;
                }
            }
        }
        let mut gradient = [[[0.0; 3]; 3]; 3];
        for i in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    gradient[i][m][n] = 0.5 * (raw[i][m][n] + raw[m][i][n]);
                }
            }
        }
        Ok(gradient)
    }
}

/// Cubic stiffness, in energy per volume, the same units as a Cauchy stress.
///
/// `c11`, `c12` and `c44` are the three independent moduli. The strain of a
/// dipole is the second derivative of the elastic Green's function,
///
/// `u_i = -∂_k G_ij P_jk`,
///
/// with `G` the line integral of Barnett, Phys. Status Solidi B 49, 741
/// (1972), doi:10.1002/pssb.2220490238, over the circle normal to `r`:
///
/// `G_ij(r) = 1/(8 π² r) ∮ Λ^{-1}_ij(n) ds`,
///
/// `Λ_ik(n) = C_ijkl n_j n_l`. When `C11 - C12 - 2 C44` vanishes the
/// integral is Kelvin's function, and that closed form is used instead.
/// [`Self::periodic_strain`] sums that field over the images of a fixed
/// cell and subtracts the cell average, so the mean strain stays zero.
#[derive(Clone, Copy, Debug)]
pub struct CubicElasticity {
    c11: f64,
    c12: f64,
    c44: f64,
}

impl CubicElasticity {
    /// Positive-definite cubic stiffness. `C11 > |C12|`, `C44 > 0` and
    /// `C11 + 2 C12 > 0`.
    pub fn new(c11: f64, c12: f64, c44: f64) -> Result<Self, SaddleError> {
        let value = Self { c11, c12, c44 };
        if !value.stable() {
            return Err(SaddleError::Invalid(
                "cubic stiffness must be finite and positive definite".into(),
            ));
        }
        Ok(value)
    }

    /// Isotropic solid with shear modulus μ and Poisson ratio ν.
    /// `C44 = μ`, `C12 = 2 μ ν / (1 - 2 ν)`, `C11 = C12 + 2 C44`.
    pub fn isotropic(shear_modulus: f64, poisson: f64) -> Result<Self, SaddleError> {
        let _ = Kelvin::new(shear_modulus, poisson)?;
        let c44 = shear_modulus;
        let c12 = 2.0 * shear_modulus * poisson / (1.0 - 2.0 * poisson);
        let c11 = c12 + 2.0 * c44;
        Self::new(c11, c12, c44)
    }

    fn stable(self) -> bool {
        self.c11.is_finite()
            && self.c12.is_finite()
            && self.c44.is_finite()
            && self.c44 > 0.0
            && self.c11 > self.c12.abs()
            && self.c11 + 2.0 * self.c12 > 0.0
    }

    fn isotropic_kelvin(self) -> Option<Kelvin> {
        let mismatch = (self.c11 - self.c12 - 2.0 * self.c44).abs();
        if mismatch > 1.0e-8 * self.c44 {
            return None;
        }
        let lam = self.c12;
        let mu = self.c44;
        let poisson = lam / (2.0 * (lam + mu));
        Kelvin::new(mu, poisson).ok()
    }

    /// Infinite-medium strain at `r` of a dipole at the origin.
    pub fn strain_of_dipole(
        &self,
        r: [f64; 3],
        dipole: ElasticDipole,
    ) -> Result<Strain, SaddleError> {
        if let Some(kelvin) = self.isotropic_kelvin() {
            return Strain::from_tensor(kelvin.strain_from_dipole(r, dipole)?);
        }
        if length(r) < 1.0e-8 {
            return Err(SaddleError::Invalid(
                "the elastic Green's function is singular at the origin".into(),
            ));
        }
        let second = self.second_derivative(r);
        let p = dipole.tensor();
        let mut raw = [[0.0; 3]; 3];
        for i in 0..3 {
            for m in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    for k in 0..3 {
                        sum += second[k][m][i][j] * p[j][k];
                    }
                }
                raw[i][m] = -sum;
            }
        }
        Strain::from_tensor(symmetrize(raw))
    }

    /// Strain of one dipole in a fixed periodic cell.
    ///
    /// `cell` has the lattice vectors as its rows. `nmax` is the number of
    /// real-space image shells kept on each side of the origin. The sum is
    /// the infinite-medium field of those images, minus its average over the
    /// cell, so the mean strain stays zero. A reciprocal-space pile is not
    /// used: the Fourier transform of a dipole strain does not decay, and
    /// Pasianot, Philos. Mag. Lett. 96, 447 (2016),
    /// doi:10.1080/09500839.2016.1250965, shows that dropping the correction
    /// to a naive image sum leaves a spurious piece in the interaction.
    pub fn periodic_strain(
        &self,
        r: [f64; 3],
        cell: M3,
        dipole: ElasticDipole,
        nmax: usize,
    ) -> Result<Strain, SaddleError> {
        if nmax == 0 {
            return Err(SaddleError::Invalid(
                "the periodic strain keeps at least one image shell".into(),
            ));
        }
        if !cell.iter().flatten().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("periodic cell"));
        }
        let volume = det3(cell).abs();
        if !(volume.is_finite() && volume > 0.0) {
            return Err(SaddleError::Invalid(
                "the periodic cell must have positive volume".into(),
            ));
        }
        let at_point = self.image_sum(r, cell, dipole, nmax)?;
        let mut mean = [[0.0; 3]; 3];
        let samples = 2i32;
        let mut count = 0.0;
        for i in 0..samples {
            for j in 0..samples {
                for k in 0..samples {
                    let point = cart_from_fractional(
                        cell,
                        [
                            (i as f64 + 0.5) / samples as f64,
                            (j as f64 + 0.5) / samples as f64,
                            (k as f64 + 0.5) / samples as f64,
                        ],
                    );
                    let sample = self.image_sum(point, cell, dipole, nmax)?;
                    for a in 0..3 {
                        for b in 0..3 {
                            mean[a][b] += sample[a][b];
                        }
                    }
                    count += 1.0;
                }
            }
        }
        let mut raw = [[0.0; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                raw[a][b] = at_point[a][b] - mean[a][b] / count;
            }
        }
        Strain::from_tensor(raw)
    }

    fn image_sum(
        &self,
        r: [f64; 3],
        cell: M3,
        dipole: ElasticDipole,
        nmax: usize,
    ) -> Result<M3, SaddleError> {
        let mut raw = [[0.0; 3]; 3];
        let limit = nmax as i32;
        for n0 in -limit..=limit {
            for n1 in -limit..=limit {
                for n2 in -limit..=limit {
                    let image = cart_from_fractional(cell, [n0 as f64, n1 as f64, n2 as f64]);
                    let arm = [r[0] - image[0], r[1] - image[1], r[2] - image[2]];
                    if length(arm) < 1.0e-8 {
                        continue;
                    }
                    let piece = self.strain_of_dipole(arm, dipole)?.tensor();
                    for a in 0..3 {
                        for b in 0..3 {
                            raw[a][b] += piece[a][b];
                        }
                    }
                }
            }
        }
        Ok(raw)
    }

    fn green(&self, r: [f64; 3]) -> [[f64; 3]; 3] {
        let radius = length(r);
        let integral = circle_inverse(*self, r);
        let scale = 1.0 / (8.0 * PI * PI * radius);
        scale_matrix(integral, scale)
    }

    fn second_derivative(&self, r: [f64; 3]) -> [[[[f64; 3]; 3]; 3]; 3] {
        let radius = length(r);
        let h = 1.0e-4 * radius;
        let mut out = [[[[0.0; 3]; 3]; 3]; 3];
        for k in 0..3 {
            for m in 0..=k {
                let mixed = if k == m {
                    self.pure_second(r, k, h)
                } else {
                    self.mixed_second(r, k, m, h)
                };
                out[k][m] = mixed;
                if k != m {
                    out[m][k] = mixed;
                }
            }
        }
        out
    }

    fn pure_second(&self, r: [f64; 3], axis: usize, h: f64) -> [[f64; 3]; 3] {
        let mut samples = [r; 5];
        let shifts = [-2.0, -1.0, 0.0, 1.0, 2.0];
        for (sample, shift) in samples.iter_mut().zip(shifts) {
            sample[axis] += shift * h;
        }
        let g = samples.map(|point| self.green(point));
        let mut out = [[0.0; 3]; 3];
        let scale = 1.0 / (12.0 * h * h);
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = scale
                    * (-g[0][i][j] + 16.0 * g[1][i][j] - 30.0 * g[2][i][j] + 16.0 * g[3][i][j]
                        - g[4][i][j]);
            }
        }
        out
    }

    fn mixed_second(&self, r: [f64; 3], axis_k: usize, axis_m: usize, h: f64) -> [[f64; 3]; 3] {
        let signs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)];
        let weights = [1.0, -1.0, -1.0, 1.0];
        let mut out = [[0.0; 3]; 3];
        for (sign, weight) in signs.into_iter().zip(weights) {
            let mut point = r;
            point[axis_k] += sign.0 * h;
            point[axis_m] += sign.1 * h;
            let g = self.green(point);
            for i in 0..3 {
                for j in 0..3 {
                    out[i][j] += weight * g[i][j];
                }
            }
        }
        let scale = 1.0 / (4.0 * h * h);
        for row in &mut out {
            for value in row.iter_mut() {
                *value *= scale;
            }
        }
        out
    }
}

impl StrainResponse {
    /// Barrier of every hop of this defect at `separation` from a source dipole.
    ///
    /// The strain is the infinite-medium field of `source`. Each entry is
    /// one symmetry image. [`StrainResponse::refine_from_dipole`] is the
    /// explicit dimer at that same strain.
    pub fn hops_from_dipole(
        &self,
        elasticity: &CubicElasticity,
        separation: [f64; 3],
        source: ElasticDipole,
    ) -> Result<Vec<HopInField>, SaddleError> {
        let strain = elasticity.strain_of_dipole(separation, source)?;
        Ok(self.hops_at(&strain))
    }

    /// The same images when the minimum and the saddle sit at different points.
    ///
    /// `minimum_at` and `saddle_at` are positions relative to the source
    /// dipole. The barrier is `ΔE0 - P_saddle : ε(saddle) + P_minimum : ε(minimum)`.
    pub fn hops_from_dipole_ends(
        &self,
        elasticity: &CubicElasticity,
        minimum_at: [f64; 3],
        saddle_at: [f64; 3],
        source: ElasticDipole,
    ) -> Result<Vec<HopInField>, SaddleError> {
        let strain_minimum = elasticity.strain_of_dipole(minimum_at, source)?;
        let strain_saddle = elasticity.strain_of_dipole(saddle_at, source)?;
        Ok(self.hops_at_ends(&strain_minimum, &strain_saddle))
    }

    /// [`Self::hops_from_dipole_ends`] with the fixed-cell image sum.
    pub fn hops_from_periodic_ends(
        &self,
        elasticity: &CubicElasticity,
        minimum_at: [f64; 3],
        saddle_at: [f64; 3],
        cell: M3,
        source: ElasticDipole,
        nmax: usize,
    ) -> Result<Vec<HopInField>, SaddleError> {
        let strain_minimum = elasticity.periodic_strain(minimum_at, cell, source, nmax)?;
        let strain_saddle = elasticity.periodic_strain(saddle_at, cell, source, nmax)?;
        Ok(self.hops_at_ends(&strain_minimum, &strain_saddle))
    }

    /// The same barriers in a fixed periodic cell. `cell` has its lattice
    /// vectors as rows, and `nmax` is the reciprocal cutoff of
    /// [`CubicElasticity::periodic_strain`].
    pub fn hops_from_periodic_dipole(
        &self,
        elasticity: &CubicElasticity,
        separation: [f64; 3],
        cell: M3,
        source: ElasticDipole,
        nmax: usize,
    ) -> Result<Vec<HopInField>, SaddleError> {
        let strain = elasticity.periodic_strain(separation, cell, source, nmax)?;
        Ok(self.hops_at(&strain))
    }

    /// Explicit dimer at the infinite-medium strain of `source`.
    ///
    /// The surface must already be bound to the cell `refine` builds from
    /// `seed` and that strain, the same contract as [`StrainResponse::refine`].
    pub fn refine_from_dipole<S: PointSurface>(
        &mut self,
        elasticity: &CubicElasticity,
        separation: [f64; 3],
        source: ElasticDipole,
        surface: &S,
        seed: &SaddleSeed,
        max_steps: usize,
    ) -> Result<Estimate, SaddleError> {
        let strain = elasticity.strain_of_dipole(separation, source)?;
        self.refine(&strain, surface, seed, max_steps)
    }
}

/// Field of every defect at every other defect.
pub trait FieldSolver {
    fn fields(&self, defects: &[DefectSource]) -> Result<Vec<LocalField>, SaddleError>;
}

/// Pair sum. Cost grows as the square of the number of defects.
#[derive(Clone, Copy, Debug)]
pub struct DirectSolver {
    pub green: Kelvin,
}

impl FieldSolver for DirectSolver {
    fn fields(&self, defects: &[DefectSource]) -> Result<Vec<LocalField>, SaddleError> {
        let mut out = Vec::with_capacity(defects.len());
        for (i, target) in defects.iter().enumerate() {
            out.push(pair_field(&self.green, defects, i, target.position)?);
        }
        Ok(out)
    }
}

/// Dual-tree multipole walk.
///
/// `theta` is the opening angle. A source cell of half-width `s` and a
/// target cell of half-width `t` are well separated when the distance
/// between centres exceeds `(s + t) / theta`. `leaf` is the most particles
/// a cell keeps before it splits.
#[derive(Clone, Copy, Debug)]
pub struct FmmSolver {
    pub green: Kelvin,
    pub theta: f64,
    pub leaf: usize,
}

impl FmmSolver {
    /// `theta` 0.22 and leaves of 16.
    pub fn new(green: Kelvin) -> Self {
        Self {
            green,
            theta: 0.22,
            leaf: 16,
        }
    }
}

impl FieldSolver for FmmSolver {
    fn fields(&self, defects: &[DefectSource]) -> Result<Vec<LocalField>, SaddleError> {
        if !(self.theta.is_finite() && self.theta > 0.0 && self.theta < 1.0) {
            return Err(SaddleError::Invalid(
                "the opening angle must lie in (0, 1)".into(),
            ));
        }
        if self.leaf == 0 {
            return Err(SaddleError::Invalid(
                "a leaf must hold at least one defect".into(),
            ));
        }
        if defects.is_empty() {
            return Ok(Vec::new());
        }
        let mut order: Vec<usize> = (0..defects.len()).collect();
        let mut nodes = Vec::new();
        let root = build_node(
            defects,
            &mut order,
            0,
            defects.len(),
            0,
            &mut nodes,
            self.leaf,
        );
        accumulate(&mut nodes, root, defects);
        let mut acc = vec![
            LocalField {
                strain: Strain::zero(),
                gradient: Some([[[0.0; 3]; 3]; 3]),
            };
            defects.len()
        ];
        interact(self, &nodes, root, root, defects, &mut acc)?;
        for (field, defect) in acc.iter_mut().zip(defects) {
            let _ = defect;
            field.strain = Strain::from_tensor(field.strain.tensor())?;
        }
        Ok(acc)
    }
}

fn pair_field(
    green: &Kelvin,
    defects: &[DefectSource],
    index: usize,
    position: [f64; 3],
) -> Result<LocalField, SaddleError> {
    let mut strain = [[0.0; 3]; 3];
    let mut gradient = [[[0.0; 3]; 3]; 3];
    for (other, source) in defects.iter().enumerate() {
        if other == index {
            continue;
        }
        let r = sub_vec(position, source.position);
        let piece = green.strain_from_dipole(r, source.dipole)?;
        let slope = green.strain_gradient_from_dipole(r, source.dipole)?;
        for a in 0..3 {
            for b in 0..3 {
                strain[a][b] += piece[a][b];
                for c in 0..3 {
                    gradient[a][b][c] += slope[a][b][c];
                }
            }
        }
    }
    Ok(LocalField {
        strain: Strain::from_tensor(strain)?,
        gradient: Some(gradient),
    })
}

struct Node {
    center: [f64; 3],
    half: f64,
    children: Vec<usize>,
    dipole: M3,
    /// `moment[a][i][j] = Σ P_ij (x_a - centre_a)`.
    arm: [M3; 3],
    ids: Vec<usize>,
}

fn build_node(
    defects: &[DefectSource],
    order: &mut [usize],
    start: usize,
    count: usize,
    depth: usize,
    nodes: &mut Vec<Node>,
    leaf: usize,
) -> usize {
    let mut lo = [f64::MAX; 3];
    let mut hi = [f64::MIN; 3];
    for slot in start..start + count {
        let p = defects[order[slot]].position;
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let mut center = [0.0; 3];
    let mut half = 0.0_f64;
    for k in 0..3 {
        center[k] = 0.5 * (lo[k] + hi[k]);
        half = half.max(0.5 * (hi[k] - lo[k]));
    }
    half = half.max(1e-12);
    let index = nodes.len();
    nodes.push(Node {
        center,
        half,
        children: Vec::new(),
        dipole: [[0.0; 3]; 3],
        arm: [[[0.0; 3]; 3]; 3],
        ids: Vec::new(),
    });
    if count <= leaf || depth >= 16 || half < 1e-9 {
        nodes[index].ids = order[start..start + count].to_vec();
        return index;
    }
    let mut octants: [Vec<usize>; 8] = Default::default();
    for slot in start..start + count {
        let p = defects[order[slot]].position;
        let mut octant = 0;
        for k in 0..3 {
            if p[k] >= center[k] {
                octant |= 1 << k;
            }
        }
        octants[octant].push(order[slot]);
    }
    let mut cursor = start;
    let mut children = Vec::new();
    for bucket in &octants {
        if bucket.is_empty() {
            continue;
        }
        for (offset, id) in bucket.iter().enumerate() {
            order[cursor + offset] = *id;
        }
        let child = build_node(defects, order, cursor, bucket.len(), depth + 1, nodes, leaf);
        children.push(child);
        cursor += bucket.len();
    }
    let mut ids = Vec::with_capacity(count);
    for child in &children {
        ids.extend_from_slice(&nodes[*child].ids);
    }
    nodes[index].ids = ids;
    nodes[index].children = children;
    index
}

fn accumulate(nodes: &mut [Node], index: usize, defects: &[DefectSource]) {
    let children = nodes[index].children.clone();
    if children.is_empty() {
        let center = nodes[index].center;
        let mut dipole = [[0.0; 3]; 3];
        let mut arm = [[[0.0; 3]; 3]; 3];
        for id in nodes[index].ids.clone() {
            let p = defects[id].dipole.tensor();
            let delta = sub_vec(defects[id].position, center);
            for i in 0..3 {
                for j in 0..3 {
                    dipole[i][j] += p[i][j];
                    for axis in 0..3 {
                        arm[axis][i][j] += p[i][j] * delta[axis];
                    }
                }
            }
        }
        nodes[index].dipole = dipole;
        nodes[index].arm = arm;
        return;
    }
    for child in &children {
        accumulate(nodes, *child, defects);
    }
    let center = nodes[index].center;
    let mut dipole = [[0.0; 3]; 3];
    let mut arm = [[[0.0; 3]; 3]; 3];
    for child in &children {
        let node = &nodes[*child];
        for i in 0..3 {
            for j in 0..3 {
                dipole[i][j] += node.dipole[i][j];
                let offset = node.center[0];
                let _ = offset;
                for axis in 0..3 {
                    arm[axis][i][j] += node.arm[axis][i][j]
                        + node.dipole[i][j] * (node.center[axis] - center[axis]);
                }
            }
        }
    }
    nodes[index].dipole = dipole;
    nodes[index].arm = arm;
}

fn interact(
    solver: &FmmSolver,
    nodes: &[Node],
    source: usize,
    target: usize,
    defects: &[DefectSource],
    acc: &mut [LocalField],
) -> Result<(), SaddleError> {
    if source == target {
        let node = &nodes[source];
        if node.children.is_empty() {
            direct_leaf(solver, node, defects, acc)?;
        } else {
            let children = node.children.clone();
            for i in 0..children.len() {
                for j in i..children.len() {
                    interact(solver, nodes, children[i], children[j], defects, acc)?;
                }
            }
        }
        return Ok(());
    }
    if separated(&nodes[source], &nodes[target], solver.theta) {
        m2l(solver, &nodes[source], &nodes[target], defects, acc)?;
        return Ok(());
    }
    let source_leaf = nodes[source].children.is_empty();
    let target_leaf = nodes[target].children.is_empty();
    if source_leaf && target_leaf {
        direct_between(solver, &nodes[source], &nodes[target], defects, acc)?;
        return Ok(());
    }
    if target_leaf || (!source_leaf && nodes[source].half >= nodes[target].half) {
        for child in nodes[source].children.clone() {
            interact(solver, nodes, child, target, defects, acc)?;
        }
    } else {
        for child in nodes[target].children.clone() {
            interact(solver, nodes, source, child, defects, acc)?;
        }
    }
    Ok(())
}

fn separated(source: &Node, target: &Node, theta: f64) -> bool {
    let distance = length(sub_vec(target.center, source.center));
    distance > (source.half + target.half) / theta
}

fn m2l(
    solver: &FmmSolver,
    source: &Node,
    target: &Node,
    defects: &[DefectSource],
    acc: &mut [LocalField],
) -> Result<(), SaddleError> {
    let r = sub_vec(target.center, source.center);
    let dipole = ElasticDipole::from_tensor(source.dipole)?;
    let mut strain = solver.green.strain_from_dipole(r, dipole)?;
    let gradient = solver.green.strain_gradient_from_dipole(r, dipole)?;
    for axis in 0..3 {
        let moment = ElasticDipole::from_tensor(source.arm[axis])?;
        let slope = solver.green.strain_gradient_from_dipole(r, moment)?;
        for i in 0..3 {
            for j in 0..3 {
                strain[i][j] -= slope[i][j][axis];
            }
        }
    }
    let order = particle_order(target, defects);
    for id in order {
        let delta = sub_vec(defects[id].position, target.center);
        let mut local = strain;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    local[i][j] += gradient[i][j][k] * delta[k];
                }
            }
        }
        add_field(&mut acc[id], local, gradient);
    }
    Ok(())
}

fn direct_leaf(
    solver: &FmmSolver,
    node: &Node,
    defects: &[DefectSource],
    acc: &mut [LocalField],
) -> Result<(), SaddleError> {
    let ids = particle_order(node, defects);
    for (a, i) in ids.iter().enumerate() {
        for j in ids.iter().skip(a + 1) {
            add_pair(solver, defects, *i, *j, acc)?;
        }
    }
    Ok(())
}

fn direct_between(
    solver: &FmmSolver,
    source: &Node,
    target: &Node,
    defects: &[DefectSource],
    acc: &mut [LocalField],
) -> Result<(), SaddleError> {
    let left = particle_order(source, defects);
    let right = particle_order(target, defects);
    for i in &left {
        for j in &right {
            add_pair(solver, defects, *i, *j, acc)?;
        }
    }
    Ok(())
}

fn particle_order(node: &Node, defects: &[DefectSource]) -> Vec<usize> {
    let _ = defects;
    node.ids.clone()
}

fn add_pair(
    solver: &FmmSolver,
    defects: &[DefectSource],
    i: usize,
    j: usize,
    acc: &mut [LocalField],
) -> Result<(), SaddleError> {
    let rij = sub_vec(defects[i].position, defects[j].position);
    let strain = solver.green.strain_from_dipole(rij, defects[j].dipole)?;
    let gradient = solver
        .green
        .strain_gradient_from_dipole(rij, defects[j].dipole)?;
    add_field(&mut acc[i], strain, gradient);
    let rji = [-rij[0], -rij[1], -rij[2]];
    let strain = solver.green.strain_from_dipole(rji, defects[i].dipole)?;
    let gradient = solver
        .green
        .strain_gradient_from_dipole(rji, defects[i].dipole)?;
    add_field(&mut acc[j], strain, gradient);
    Ok(())
}

fn add_field(field: &mut LocalField, strain: M3, gradient: StrainGradient) {
    let mut tensor = field.strain.tensor();
    for i in 0..3 {
        for j in 0..3 {
            tensor[i][j] += strain[i][j];
        }
    }
    field.strain = Strain::from_tensor(tensor).unwrap_or(field.strain);
    if let Some(slot) = field.gradient.as_mut() {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    slot[i][j][k] += gradient[i][j][k];
                }
            }
        }
    }
}

fn first_derivative(r: [f64; 3], lambda: f64) -> [[[f64; 3]; 3]; 3] {
    let (inv_r3, inv_r5, _, _) = powers(r);
    let mut out = [[[0.0; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let mut value = -lambda * kronecker(i, j) * r[k] * inv_r3;
                if i == k {
                    value += r[j] * inv_r3;
                }
                if j == k {
                    value += r[i] * inv_r3;
                }
                value -= 3.0 * r[i] * r[j] * r[k] * inv_r5;
                out[i][j][k] = value;
            }
        }
    }
    out
}

fn second_derivative(r: [f64; 3], lambda: f64) -> [[[[f64; 3]; 3]; 3]; 3] {
    let (inv_r3, inv_r5, inv_r7, _) = powers(r);
    let mut out = [[[[0.0; 3]; 3]; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for m in 0..3 {
                    let mut value = -lambda * kronecker(i, j) * kronecker(k, m) * inv_r3;
                    value += 3.0 * lambda * kronecker(i, j) * r[k] * r[m] * inv_r5;
                    value += (kronecker(i, k) * kronecker(j, m)
                        + kronecker(j, k) * kronecker(i, m))
                        * inv_r3;
                    value -=
                        3.0 * (kronecker(i, k) * r[j] + kronecker(j, k) * r[i]) * r[m] * inv_r5;
                    value -= 3.0
                        * (kronecker(i, m) * r[j] * r[k]
                            + kronecker(j, m) * r[i] * r[k]
                            + kronecker(k, m) * r[i] * r[j])
                        * inv_r5;
                    value += 15.0 * r[i] * r[j] * r[k] * r[m] * inv_r7;
                    out[i][j][k][m] = value;
                }
            }
        }
    }
    out
}

fn third_derivative(r: [f64; 3], lambda: f64) -> [[[[[f64; 3]; 3]; 3]; 3]; 3] {
    let (inv_r3, inv_r5, inv_r7, inv_r9) = powers(r);
    let _ = inv_r3;
    let mut out = [[[[[0.0; 3]; 3]; 3]; 3]; 3];
    for n in 0..3 {
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    for m in 0..3 {
                        let mut value =
                            3.0 * lambda * kronecker(i, j) * kronecker(k, m) * r[n] * inv_r5;
                        value += 3.0
                            * lambda
                            * kronecker(i, j)
                            * (kronecker(n, k) * r[m] + r[k] * kronecker(n, m))
                            * inv_r5;
                        value -= 15.0 * lambda * kronecker(i, j) * r[k] * r[m] * r[n] * inv_r7;
                        value -= 3.0
                            * (kronecker(i, k) * kronecker(j, m)
                                + kronecker(j, k) * kronecker(i, m))
                            * r[n]
                            * inv_r5;
                        let u_n =
                            kronecker(i, k) * kronecker(n, j) + kronecker(j, k) * kronecker(n, i);
                        value -= 3.0 * u_n * r[m] * inv_r5;
                        value -= 3.0
                            * (kronecker(i, k) * r[j] + kronecker(j, k) * r[i])
                            * kronecker(n, m)
                            * inv_r5;
                        value += 15.0
                            * (kronecker(i, k) * r[j] + kronecker(j, k) * r[i])
                            * r[m]
                            * r[n]
                            * inv_r7;
                        let v_n = kronecker(i, m)
                            * (kronecker(n, j) * r[k] + r[j] * kronecker(n, k))
                            + kronecker(j, m) * (kronecker(n, i) * r[k] + r[i] * kronecker(n, k))
                            + kronecker(k, m) * (kronecker(n, i) * r[j] + r[i] * kronecker(n, j));
                        value -= 3.0 * v_n * inv_r5;
                        value += 15.0
                            * (kronecker(i, m) * r[j] * r[k]
                                + kronecker(j, m) * r[i] * r[k]
                                + kronecker(k, m) * r[i] * r[j])
                            * r[n]
                            * inv_r7;
                        value += 15.0
                            * (kronecker(n, i) * r[j] * r[k] * r[m]
                                + r[i] * kronecker(n, j) * r[k] * r[m]
                                + r[i] * r[j] * kronecker(n, k) * r[m]
                                + r[i] * r[j] * r[k] * kronecker(n, m))
                            * inv_r7;
                        value -= 105.0 * r[i] * r[j] * r[k] * r[m] * r[n] * inv_r9;
                        out[i][j][k][m][n] = value;
                    }
                }
            }
        }
    }
    out
}

fn powers(r: [f64; 3]) -> (f64, f64, f64, f64) {
    let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let inv_r2 = 1.0 / r2;
    let inv_r = inv_r2.sqrt();
    let inv_r3 = inv_r * inv_r2;
    let inv_r5 = inv_r3 * inv_r2;
    let inv_r7 = inv_r5 * inv_r2;
    let inv_r9 = inv_r7 * inv_r2;
    (inv_r3, inv_r5, inv_r7, inv_r9)
}

fn kronecker(i: usize, j: usize) -> f64 {
    if i == j { 1.0 } else { 0.0 }
}

fn symmetrize(m: M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = 0.5 * (m[i][j] + m[j][i]);
        }
    }
    out
}

fn length(r: [f64; 3]) -> f64 {
    (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt()
}

fn scale_matrix(m: [[f64; 3]; 3], scale: f64) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = scale * m[i][j];
        }
    }
    out
}

fn cart_from_fractional(cell: M3, s: [f64; 3]) -> [f64; 3] {
    [
        s[0] * cell[0][0] + s[1] * cell[1][0] + s[2] * cell[2][0],
        s[0] * cell[0][1] + s[1] * cell[1][1] + s[2] * cell[2][1],
        s[0] * cell[0][2] + s[1] * cell[1][2] + s[2] * cell[2][2],
    ]
}

fn det3(m: M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn invert_rows(m: M3) -> Result<M3, SaddleError> {
    let d = det3(m);
    if !(d.is_finite() && d.abs() > 0.0) {
        return Err(SaddleError::Invalid(
            "the periodic cell must have positive volume".into(),
        ));
    }
    let mut c = [[0.0; 3]; 3];
    c[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / d;
    c[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / d;
    c[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / d;
    c[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / d;
    c[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / d;
    c[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / d;
    c[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / d;
    c[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / d;
    c[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / d;
    Ok(c)
}

fn invert_acoustic(stiffness: CubicElasticity, n: [f64; 3]) -> [[f64; 3]; 3] {
    let c11 = stiffness.c11;
    let c12 = stiffness.c12;
    let c44 = stiffness.c44;
    let extra = c11 - c12 - 2.0 * c44;
    let mut lambda = [[0.0; 3]; 3];
    for i in 0..3 {
        for k in 0..3 {
            lambda[i][k] = (c12 + c44) * n[i] * n[k];
            if i == k {
                lambda[i][k] += c44 + extra * n[i] * n[i];
            }
        }
    }
    invert_rows(lambda).unwrap_or([[0.0; 3]; 3])
}

fn circle_inverse(stiffness: CubicElasticity, direction: [f64; 3]) -> [[f64; 3]; 3] {
    let radius = length(direction);
    let t = [
        direction[0] / radius,
        direction[1] / radius,
        direction[2] / radius,
    ];
    let seed = if t[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut e1 = cross(t, seed);
    let e1n = length(e1);
    e1 = [e1[0] / e1n, e1[1] / e1n, e1[2] / e1n];
    let e2 = cross(t, e1);
    let nphi = 64;
    let mut acc = [[0.0; 3]; 3];
    for step in 0..nphi {
        let phi = 2.0 * PI * (step as f64 + 0.5) / nphi as f64;
        let (s, c) = phi.sin_cos();
        let n = [
            c * e1[0] + s * e2[0],
            c * e1[1] + s * e2[1],
            c * e1[2] + s * e2[2],
        ];
        let inverse = invert_acoustic(stiffness, n);
        for i in 0..3 {
            for j in 0..3 {
                acc[i][j] += inverse[i][j];
            }
        }
    }
    let dphi = 2.0 * PI / nphi as f64;
    scale_matrix(acc, dphi)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub_vec(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
