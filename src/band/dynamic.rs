//! Dynamic NEB, the image-wise relaxation of Lindgren, Kastlunger and
//! Peterson, J. Chem. Theory Comput. 15, 5787 (2019), doi:10.1021/acs.jctc.9b00633,
//! as ASE 3.29 implements it in `ase.mep.neb.DyNEB`.
//!
//! An image whose largest per-atom force lies under the threshold keeps
//! the force and the energy from the last time it was evaluated. The
//! spring and the tangent still see that image. It is evaluated again
//! when a neighbour moves far enough that the projected force rises
//! back over the threshold.

use ndarray::{Array1, ArrayView1, ArrayView2, s};

use crate::error::SaddleError;

/// Dynamic relaxation of a band. `None` on [`super::BandConfig`] evaluates
/// every image.
///
/// Non-exhaustive: start from [`DynamicNeb::default`] and chain the
/// `with_*` setters. `fmax` is ASE's `fmax` and `scale_fmax` is ASE's
/// `scale_fmax`. The default is ASE's: `fmax` 0.05 and `scale_fmax` 0.
///
/// ```
/// use rgsaddle::DynamicNeb;
///
/// let spec = DynamicNeb::default().with_scale_fmax(6.0);
/// assert_eq!(spec.fmax, 0.05);
/// assert_eq!(spec.scale_fmax, 6.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct DynamicNeb {
    /// An image is eligible to be left in place when the largest
    /// per-atom Euclidean norm of its projected force is strictly below
    /// this value. ASE requires it to equal the optimizer's `fmax`.
    pub fmax: f64,
    /// The threshold at an image is `fmax * (1 + distance * scale_fmax)`.
    /// `distance` is the Euclidean norm of the coordinate difference
    /// from the highest-energy image, with no minimum image. Zero
    /// leaves the threshold uniform.
    pub scale_fmax: f64,
}

with_setters!(DynamicNeb {
    with_fmax => fmax: val f64,
    with_scale_fmax => scale_fmax: val f64,
});

impl Default for DynamicNeb {
    fn default() -> Self {
        Self {
            fmax: 0.05,
            scale_fmax: 0.0,
        }
    }
}

impl DynamicNeb {
    pub(crate) fn check(self) -> Result<(), SaddleError> {
        if !self.fmax.is_finite() {
            return Err(SaddleError::NonFinite("dynamic NEB fmax"));
        }
        if !self.scale_fmax.is_finite() {
            return Err(SaddleError::NonFinite("dynamic NEB scale_fmax"));
        }
        Ok(())
    }
}

/// One interior image after the ASE 3.29 `DyNEB.get_forces` mask.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicImage {
    /// Largest per-atom force norm before any zeroing.
    ///
    /// The optimizer reads `zeroed` and `frozen`. These three numbers
    /// are what the mask compared, kept for the tests of that comparison.
    #[allow(dead_code)]
    pub atom_force: f64,
    /// Euclidean distance from the highest-energy image.
    #[allow(dead_code)]
    pub distance: f64,
    /// `fmax * (1 + distance * scale_fmax)`.
    #[allow(dead_code)]
    pub threshold: f64,
    /// The force row was set to zero. Never set on the highest-energy image.
    pub zeroed: bool,
    /// `set_positions` leaves the image where it is: the force ASE reads
    /// after the mask is strictly below `fmax`.
    pub frozen: bool,
}

/// Apply ASE's DyNEB mask to interior `forces` (the projected force the
/// optimizer would see). `energies` are the interior images only. The
/// highest-energy image is the last interior image that attains the
/// maximum. That is ASE's `NEBState.imax` when the maximum is unique.
/// An exact tie keeps the last image at that energy.
pub(crate) fn dynamic_mask(
    interior: ArrayView2<f64>,
    energies: ArrayView1<f64>,
    forces: &mut ndarray::Array2<f64>,
    spec: DynamicNeb,
) -> Result<Vec<DynamicImage>, SaddleError> {
    spec.check()?;
    let n = interior.nrows();
    if n == 0 || energies.len() != n || forces.dim() != interior.dim() {
        return Err(SaddleError::Shape(
            "dynamic NEB mask needs one energy and one force row per interior image".into(),
        ));
    }
    let dof = interior.ncols();
    if dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(
            "dynamic NEB mask needs 3N coordinates per image".into(),
        ));
    }
    if !interior.iter().all(|v| v.is_finite())
        || !energies.iter().all(|v| v.is_finite())
        || !forces.iter().all(|v| v.is_finite())
    {
        return Err(SaddleError::NonFinite("dynamic NEB mask"));
    }
    let highest = highest_interior(energies);
    let origin = interior.row(highest);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let atom_force = atom_fmax(forces.row(i));
        let distance = euclidean(interior.row(i), origin);
        let threshold = spec.fmax * (1.0 + distance * spec.scale_fmax);
        let under = atom_force < threshold;
        let zeroed = under && i != highest;
        if zeroed {
            forces.row_mut(i).fill(0.0);
        }
        let seen = if zeroed { 0.0 } else { atom_force };
        let frozen = seen < spec.fmax;
        out.push(DynamicImage {
            atom_force,
            distance,
            threshold,
            zeroed,
            frozen,
        });
    }
    Ok(out)
}

/// Last interior image at the highest energy.
fn highest_interior(energies: ArrayView1<f64>) -> usize {
    let mut best = 0;
    for i in 1..energies.len() {
        if energies[i] >= energies[best] {
            best = i;
        }
    }
    best
}

/// Largest per-atom Euclidean norm. ASE's `sqrt((F**2).sum(axis=1)).max()`.
fn atom_fmax(force: ArrayView1<f64>) -> f64 {
    let n_atoms = force.len() / 3;
    let mut max = 0.0;
    for atom in 0..n_atoms {
        let mut sum = 0.0;
        for c in 0..3 {
            let component = force[3 * atom + c];
            sum += component * component;
        }
        let norm = sum.sqrt();
        if norm > max {
            max = norm;
        }
    }
    max
}

fn euclidean(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let mut sum = 0.0;
    for (left, right) in a.iter().zip(b.iter()) {
        let diff = left - right;
        sum += diff * diff;
    }
    sum.sqrt()
}

fn same_row(a: ArrayView1<f64>, b: ArrayView1<f64>) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(left, right)| left == right)
}

/// The surface answer kept for one interior image.
#[derive(Clone, Debug)]
pub(crate) struct ImageAnswer {
    pub position: Array1<f64>,
    pub energy: f64,
    pub gradient: Array1<f64>,
}

/// The images this step will not move, decided at the step's start.
#[derive(Clone, Debug)]
pub(crate) struct DynamicStep {
    pub zeroed: Vec<bool>,
    pub frozen: Vec<bool>,
    /// Largest absolute component of the force the optimizer sees.
    pub visible_max: f64,
}

/// Per-image surface cache and the pin for the step in progress.
#[derive(Clone, Debug, Default)]
pub(crate) struct DynamicRun {
    pub spec: Option<DynamicNeb>,
    /// Geometry the step started at. `None` outside a step.
    pub origin: Option<Array1<f64>>,
    pub step: Option<DynamicStep>,
    pub images: Vec<Option<ImageAnswer>>,
}

impl DynamicRun {
    pub(crate) fn install(&mut self, spec: Option<DynamicNeb>) {
        self.spec = spec;
        self.clear_answers();
    }

    /// Drop the step pin and every cached image answer. The spec stays,
    /// so a reset keeps dynamic NEB on without replaying the old surface.
    pub(crate) fn clear_answers(&mut self) {
        self.origin = None;
        self.step = None;
        self.images.clear();
    }

    pub(crate) fn begin_step(&mut self, origin: Array1<f64>) {
        self.step = None;
        self.origin = Some(origin);
    }

    /// `Some` when this step already chose images to pin, with those
    /// images put back on the step's starting geometry.
    pub(crate) fn pin(&self, x: ArrayView1<f64>) -> Option<Array1<f64>> {
        let step = self.step.as_ref()?;
        let origin = self.origin.as_ref()?;
        let n = step.frozen.len();
        if n == 0 || origin.len() != x.len() || !origin.len().is_multiple_of(n) {
            return None;
        }
        let dof = origin.len() / n;
        let mut y = x.to_owned();
        for (i, &frozen) in step.frozen.iter().enumerate() {
            if !frozen {
                continue;
            }
            let start = i * dof;
            let end = start + dof;
            y.slice_mut(s![start..end])
                .assign(&origin.slice(s![start..end]));
        }
        Some(y)
    }

    /// Put pinned images back on the geometry the step started at.
    pub(crate) fn snap(&self, x: &mut Array1<f64>) {
        let Some(y) = self.pin(x.view()) else {
            return;
        };
        x.assign(&y);
    }

    /// Force rows to zero for the optimizer. The first assembly at the
    /// step's origin records the set; later assemblies of this step
    /// keep it, so a trial geometry does not unfreeze an image that the
    /// position update will put back.
    /// `row_gate` is the convergence force of each unmasked row.
    /// A non-zeroed image whose gate is still above `force_tol` is left
    /// free, so a band tolerance tighter than `fmax` can still be met.
    /// A zeroed image stays pinned: ASE has already taken it out of the
    /// optimizer.
    pub(crate) fn decision(
        &mut self,
        x: ArrayView1<f64>,
        mask: &[DynamicImage],
        visible: f64,
        row_gate: &[f64],
        force_tol: f64,
    ) -> Vec<bool> {
        if let Some(step) = &self.step {
            return step.zeroed.clone();
        }
        let zeroed: Vec<bool> = mask.iter().map(|image| image.zeroed).collect();
        let frozen: Vec<bool> = mask
            .iter()
            .enumerate()
            .map(|(i, image)| {
                let under_tol = row_gate.get(i).is_some_and(|gate| *gate <= force_tol);
                image.frozen && (image.zeroed || under_tol)
            })
            .collect();
        let at_origin = self
            .origin
            .as_ref()
            .is_some_and(|origin| same_row(origin.view(), x));
        if at_origin {
            self.step = Some(DynamicStep {
                zeroed: zeroed.clone(),
                frozen,
                visible_max: visible,
            });
        }
        zeroed
    }

    pub(crate) fn visible_max(&self) -> Option<f64> {
        self.step.as_ref().map(|step| step.visible_max)
    }

    pub(crate) fn recall(
        &self,
        index: usize,
        position: ArrayView1<f64>,
    ) -> Option<(f64, Array1<f64>)> {
        let answer = self.images.get(index)?.as_ref()?;
        if !same_row(answer.position.view(), position) {
            return None;
        }
        Some((answer.energy, answer.gradient.clone()))
    }

    pub(crate) fn remember(
        &mut self,
        index: usize,
        position: ArrayView1<f64>,
        energy: f64,
        gradient: ArrayView1<f64>,
    ) {
        if self.images.len() <= index {
            self.images.resize_with(index + 1, || None);
        }
        self.images[index] = Some(ImageAnswer {
            position: position.to_owned(),
            energy,
            gradient: gradient.to_owned(),
        });
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::{DynamicNeb, dynamic_mask};

    fn row(values: [f64; 3]) -> [f64; 3] {
        values
    }

    #[test]
    fn mask_follows_ase_dyneb() {
        // Three interior images. The middle one is highest. Its
        // neighbour sits 4 length units away along x; the other
        // neighbour sits on top of it.
        let interior = Array2::from_shape_vec(
            (3, 3),
            vec![
                row([0.0, 0.0, 0.0]),
                row([4.0, 0.0, 0.0]),
                row([4.0, 0.0, 0.0]),
            ]
            .into_iter()
            .flatten()
            .collect(),
        )
        .expect("shape");
        let energies = ndarray::arr1(&[1.0, 5.0, 4.0]);
        // Norms 0.5, 2, 4. A 3-4-5 triangle gives the 5 below.
        let mut forces =
            Array2::from_shape_vec((3, 3), vec![0.5, 0.0, 0.0, 0.0, 2.0, 0.0, 3.0, 4.0, 0.0])
                .expect("shape");
        let spec = DynamicNeb::default().with_fmax(1.0).with_scale_fmax(1.0);
        let mask = dynamic_mask(interior.view(), energies.view(), &mut forces, spec).expect("mask");

        // Distance 4 from the highest image, threshold 1 * (1 + 4 * 1) = 5.
        // 0.5 < 5, and this image is not the highest, so the force is cleared.
        assert!(mask[0].zeroed);
        assert!(mask[0].frozen);
        assert_eq!(mask[0].distance, 4.0);
        assert_eq!(mask[0].threshold, 5.0);
        assert!(forces.row(0).iter().all(|v| *v == 0.0));

        // The highest image keeps its force even under the threshold,
        // and a norm of 2 is not under fmax, so it still moves.
        assert!(!mask[1].zeroed);
        assert!(!mask[1].frozen);
        assert_eq!(mask[1].distance, 0.0);
        assert_eq!(forces[(1, 1)], 2.0);

        // Same place as the highest image, so the scale does not loosen
        // the test. Norm 5 is not under fmax 1.
        assert!(!mask[2].zeroed);
        assert!(!mask[2].frozen);
        assert_eq!(mask[2].distance, 0.0);
        assert_eq!(mask[2].threshold, 1.0);
        assert_eq!(forces[(2, 0)], 3.0);
        assert_eq!(forces[(2, 1)], 4.0);
    }

    #[test]
    fn highest_image_under_fmax_is_kept_and_not_moved() {
        let interior = Array2::zeros((1, 3));
        let energies = ndarray::arr1(&[0.0]);
        let mut forces = Array2::from_shape_vec((1, 3), vec![0.2, 0.0, 0.0]).expect("shape");
        let spec = DynamicNeb::default().with_fmax(1.0);
        let mask = dynamic_mask(interior.view(), energies.view(), &mut forces, spec).expect("mask");
        assert!(!mask[0].zeroed);
        assert!(mask[0].frozen);
        assert_eq!(forces[(0, 0)], 0.2);
    }

    #[test]
    fn force_equal_to_fmax_is_not_under_the_test() {
        let interior =
            Array2::from_shape_vec((2, 3), vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]).expect("shape");
        let energies = ndarray::arr1(&[0.0, 1.0]);
        let mut forces =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]).expect("shape");
        let spec = DynamicNeb::default().with_fmax(1.0).with_scale_fmax(0.0);
        let mask = dynamic_mask(interior.view(), energies.view(), &mut forces, spec).expect("mask");
        // Strict `<`, so a force of exactly fmax still moves. The
        // second image is highest and is not zeroed; its norm is 0,
        // which is under fmax, so it stays.
        assert!(!mask[0].zeroed);
        assert!(!mask[0].frozen);
        assert_eq!(forces[(0, 0)], 1.0);
        assert!(!mask[1].zeroed);
        assert!(mask[1].frozen);
    }
}
