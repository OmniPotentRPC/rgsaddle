//! AutoNEB grows a band by inserting images, then climbs.
//!
//! Kolsbjerg, Groves, and Hammer, J. Chem. Phys. 145, 094107 (2016),
//! doi:10.1063/1.4961868.
//!
//! The caller supplies the endpoints and any intermediate guesses.
//! Construction fills the band until it holds `n_simul + 2` images, or
//! `n_max` if that is smaller, by cutting the longest segment.
//! [`AutoNeb::run`] relaxes a local window, inserts one image where the
//! resolution is worst, and repeats until `n_max` images. The last
//! stage arms the climbing image on the window around the highest
//! interior energy and relaxes that window to [`BandConfig::force_tol`].
//!
//! [`ImageInterpolation`] places every new image.
//! [`LinearInterpolation`] is the straight minimum-image segment. An
//! IDPP path implements the same trait; this crate does not ship one.
//! Buildup leaves climbing off. The spring is uniform, because a
//! weighted or Onsager-Machlup spring changes length when an image is
//! inserted. The solid-state block is not part of this loop.
//!
//! `surface_rows` counts force evaluations: one row of
//! [`BandSurface::eval`] is one force call. A surface error discards
//! the window that was in progress and leaves the previous window in
//! place. [`AutoNeb::run`] starts once.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig};
use crate::error::SaddleError;
use crate::spring::SpringKind;

/// Where a new image is placed between two neighbours.
///
/// `n_between` is at least 1. Image `k` in `1..=n_between` sits at
/// fraction `k / (n_between + 1)` from `left` toward `right`. `cell`
/// is the band cell; `None` is free space. The vectors have the same
/// length as `left`.
///
/// [`LinearInterpolation`] steps in the minimum image. An IDPP
/// interpolant minimizes the image-dependent pair potential at those
/// fractions and returns the same shape.
pub trait ImageInterpolation: Sync {
    fn interpolate(
        &self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
        n_between: usize,
        cell: Option<&Cell>,
    ) -> Result<Vec<Array1<f64>>, SaddleError>;
}

/// Straight minimum-image segment between two images.
#[derive(Clone, Copy, Debug, Default)]
pub struct LinearInterpolation;

impl ImageInterpolation for LinearInterpolation {
    fn interpolate(
        &self,
        left: ArrayView1<f64>,
        right: ArrayView1<f64>,
        n_between: usize,
        cell: Option<&Cell>,
    ) -> Result<Vec<Array1<f64>>, SaddleError> {
        if n_between == 0 {
            return Err(SaddleError::Invalid(
                "interpolation needs at least one image".into(),
            ));
        }
        let dof = left.len();
        if right.len() != dof || dof == 0 || !dof.is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "interpolation endpoints must be matching 3N vectors; got {dof} and {}",
                right.len()
            )));
        }
        if !left.iter().chain(right.iter()).all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("interpolation"));
        }
        let mut step = &right - &left;
        if let Some(cell) = cell {
            cell.minimum_image(&mut step);
        }
        let denom = (n_between + 1) as f64;
        let mut images = Vec::with_capacity(n_between);
        for k in 1..=n_between {
            images.push(&left + &step * ((k as f64) / denom));
        }
        Ok(images)
    }
}

/// Index of the segment where the next image goes.
///
/// `lengths[j]` is the distance from image `j` to image `j + 1`.
/// `energies` has one more entry than `lengths`. `span` is the
/// separation of the endpoints. `space_energy_ratio` is the threshold
/// used by ASE's AutoNEB (0.5 there).
///
/// The peak-weighted gap of segment `j` is the absolute value of
/// `(E[j+1] - E[j]) * (E[j+1] + E[j] - 2 E_min) / (2 (E_max - E_min))`.
/// `f_R` is the longest length divided by `span`. `g_R` is the largest
/// weight divided by `(E_max - E_min)`. The longest segment is cut when
/// `g_R` is not positive, `span` is not positive, or `f_R / g_R` exceeds
/// `space_energy_ratio`. Otherwise the largest weight is cut. Ties keep
/// the lower index. A ratio of 0 cuts the longest segment whenever both
/// resolutions are positive. A large ratio cuts the energy gap.
pub fn worst_segment(
    lengths: &[f64],
    energies: &[f64],
    span: f64,
    space_energy_ratio: f64,
) -> Result<usize, SaddleError> {
    if lengths.is_empty() || energies.len() != lengths.len() + 1 {
        return Err(SaddleError::Shape(format!(
            "worst segment needs one energy per image and one length per gap; got {} lengths and {} energies",
            lengths.len(),
            energies.len()
        )));
    }
    if !lengths.iter().chain(energies.iter()).all(|v| v.is_finite()) {
        return Err(SaddleError::NonFinite("worst segment"));
    }
    if !(space_energy_ratio >= 0.0 && space_energy_ratio.is_finite()) {
        return Err(SaddleError::Invalid(
            "space_energy_ratio must be finite and non-negative".into(),
        ));
    }
    let longest = first_max(lengths);
    if !(span > 0.0 && span.is_finite()) {
        return Ok(longest);
    }
    let emin = energies.iter().copied().fold(f64::INFINITY, f64::min);
    let emax = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let enorm = emax - emin;
    if enorm <= 0.0 {
        return Ok(longest);
    }
    let mut weights = Vec::with_capacity(lengths.len());
    for j in 0..lengths.len() {
        let gap = (energies[j + 1] - energies[j]) * (energies[j + 1] + energies[j] - 2.0 * emin)
            / (2.0 * enorm);
        weights.push(gap.abs());
    }
    let f_r = lengths.iter().copied().fold(0.0_f64, f64::max) / span;
    // The second division is the ASE ratio. Both factors have to stay,
    // or the threshold no longer matches that comparison.
    let g_r = weights.iter().copied().fold(0.0_f64, f64::max) / enorm;
    if g_r <= 0.0 || f_r / g_r > space_energy_ratio {
        Ok(longest)
    } else {
        Ok(first_max(&weights))
    }
}

fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, value) in values.iter().enumerate().skip(1) {
        if *value > values[best] {
            best = i;
        }
    }
    best
}

/// AutoNEB controls. Distances and forces are in the caller's units.
#[derive(Clone, Debug)]
pub struct AutoNebConfig {
    /// Band mechanics for every window. Climbing on this config is
    /// not read: buildup leaves it off, and the last stage arms it
    /// when [`AutoNebConfig::climb`] is set.
    pub band: BandConfig,
    /// Free images in a local window. The window pins one image on
    /// each side, so it holds `n_simul + 2` images when the band is
    /// at least that long.
    pub n_simul: usize,
    /// Images at the end, endpoints included.
    pub n_max: usize,
    /// Force stop for every window before the climb.
    pub local_force_tol: f64,
    /// Step cap for every window before the climb.
    pub local_max_steps: usize,
    /// Step cap for the climbing window.
    pub climb_max_steps: usize,
    /// See [`worst_segment`]. Default 0.5.
    pub space_energy_ratio: f64,
    /// Arm the climbing image after the last insertion.
    pub climb: bool,
}

impl Default for AutoNebConfig {
    fn default() -> Self {
        Self {
            band: BandConfig {
                climbing: None,
                ..BandConfig::default()
            },
            n_simul: 3,
            n_max: 7,
            local_force_tol: 1e-1,
            local_max_steps: 100,
            climb_max_steps: 2000,
            space_energy_ratio: 0.5,
            climb: true,
        }
    }
}

impl AutoNebConfig {
    fn validate(&self) -> Result<(), SaddleError> {
        if self.n_simul == 0 {
            return Err(SaddleError::Invalid(
                "AutoNEB n_simul must be at least 1".into(),
            ));
        }
        let Some(window) = self.n_simul.checked_add(2) else {
            return Err(SaddleError::Invalid("AutoNEB window does not fit".into()));
        };
        if self.n_max < window {
            return Err(SaddleError::Invalid(format!(
                "AutoNEB n_max ({}) must be at least n_simul + 2 ({window})",
                self.n_max
            )));
        }
        if self.local_max_steps == 0 || self.climb_max_steps == 0 {
            return Err(SaddleError::Invalid(
                "AutoNEB step caps must be at least 1".into(),
            ));
        }
        if !(self.local_force_tol > 0.0 && self.local_force_tol.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB local force tolerance must be finite and positive".into(),
            ));
        }
        if !(self.band.force_tol > 0.0 && self.band.force_tol.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB force tolerance must be finite and positive".into(),
            ));
        }
        if !(self.space_energy_ratio >= 0.0 && self.space_energy_ratio.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB space_energy_ratio must be finite and non-negative".into(),
            ));
        }
        if self.band.solid.is_some() {
            return Err(SaddleError::Invalid(
                "AutoNEB does not carry the solid-state block".into(),
            ));
        }
        match &self.band.spring {
            SpringKind::Uniform { k } if k.is_finite() => {}
            SpringKind::Uniform { .. } => {
                return Err(SaddleError::Invalid(
                    "AutoNEB spring constant must be finite".into(),
                ));
            }
            SpringKind::Weighted { .. } | SpringKind::OnsagerMachlup { .. } => {
                return Err(SaddleError::Invalid(
                    "AutoNEB needs a uniform spring; other springs change length when an image is inserted"
                        .into(),
                ));
            }
        }
        let method = if self.band.quickmin {
            rgmin::Method::QuickMin
        } else {
            self.band.method.clone()
        };
        crate::band::check_force_driven(&method)
    }
}

/// Result of [`AutoNeb::run`].
#[derive(Clone, Debug)]
pub struct AutoNebReport {
    pub status: BandStatus,
    /// Max absolute projected-force component on the last window.
    pub max_force: f64,
    /// Climbing image in the full band. `None` when climbing is off
    /// or the highest interior image of the window does not lie above
    /// both of that window's ends.
    pub ci_index: Option<usize>,
    /// Rows passed to [`BandSurface::eval`]. One row is one force call.
    pub surface_rows: usize,
    pub n_images: usize,
}

/// Growing band. `I` places inserted images.
pub struct AutoNeb<I: ImageInterpolation = LinearInterpolation> {
    config: AutoNebConfig,
    interpolation: I,
    positions: Array2<f64>,
    energies: Vec<Option<f64>>,
    surface_rows: usize,
    ci_index: Option<usize>,
    started: bool,
}

impl AutoNeb<LinearInterpolation> {
    /// Fill `initial` up to the first window and keep linear insertion
    /// for the rest. `initial` has at least the two endpoints.
    pub fn new(config: AutoNebConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        Self::with_interpolation(config, initial, LinearInterpolation)
    }
}

impl<I: ImageInterpolation> AutoNeb<I> {
    /// Same as [`AutoNeb::new`] with a caller-supplied interpolant.
    pub fn with_interpolation(
        config: AutoNebConfig,
        initial: Array2<f64>,
        interpolation: I,
    ) -> Result<Self, SaddleError> {
        config.validate()?;
        check_initial(&initial, config.n_max)?;
        let n = initial.nrows();
        let mut search = Self {
            config,
            interpolation,
            positions: initial,
            energies: vec![None; n],
            surface_rows: 0,
            ci_index: None,
            started: false,
        };
        search.fill_initial()?;
        Ok(search)
    }

    pub fn positions(&self) -> ArrayView2<'_, f64> {
        self.positions.view()
    }

    /// Energy of each image after the last evaluation that wrote it.
    /// `None` until that image has been evaluated.
    pub fn energies(&self) -> &[Option<f64>] {
        &self.energies
    }

    pub fn climbing_image(&self) -> Option<usize> {
        self.ci_index
    }

    pub fn surface_rows(&self) -> usize {
        self.surface_rows
    }

    /// Relax, insert, and climb. See the module docs for the stages.
    pub fn run<S: BandSurface>(&mut self, surface: &S) -> Result<AutoNebReport, SaddleError> {
        if self.started {
            return Err(SaddleError::Invalid(
                "AutoNEB has already been started".into(),
            ));
        }
        self.started = true;
        let mut last = self.relax(
            surface,
            false,
            self.config.local_force_tol,
            self.config.local_max_steps,
        )?;
        self.ensure_energies(surface)?;
        while self.positions.nrows() < self.config.n_max {
            let gap = self.next_gap()?;
            self.insert_gap(gap, 1)?;
            last = self.relax(
                surface,
                false,
                self.config.local_force_tol,
                self.config.local_max_steps,
            )?;
            self.ensure_energies(surface)?;
        }
        if self.config.climb {
            last = self.relax(
                surface,
                true,
                self.config.band.force_tol,
                self.config.climb_max_steps,
            )?;
        }
        Ok(AutoNebReport {
            status: last.status,
            max_force: last.max_force,
            ci_index: self.ci_index,
            surface_rows: self.surface_rows,
            n_images: self.positions.nrows(),
        })
    }

    fn fill_initial(&mut self) -> Result<(), SaddleError> {
        let target = (self.config.n_simul + 2).min(self.config.n_max);
        while self.positions.nrows() < target {
            let room = target - self.positions.nrows();
            let n_between = if self.positions.nrows() == 2 {
                self.config.n_simul.min(room)
            } else {
                1
            };
            let gap = self.longest_segment();
            self.insert_gap(gap, n_between)?;
        }
        Ok(())
    }

    fn longest_segment(&self) -> usize {
        let n = self.positions.nrows();
        let mut best = 0;
        let mut best_len = -1.0_f64;
        for j in 0..n - 1 {
            let len = segment_length(
                self.positions.row(j),
                self.positions.row(j + 1),
                self.config.band.cell.as_ref(),
            );
            if len > best_len {
                best_len = len;
                best = j;
            }
        }
        best
    }

    fn next_gap(&self) -> Result<usize, SaddleError> {
        let n = self.positions.nrows();
        let mut lengths = Vec::with_capacity(n - 1);
        for j in 0..n - 1 {
            lengths.push(segment_length(
                self.positions.row(j),
                self.positions.row(j + 1),
                self.config.band.cell.as_ref(),
            ));
        }
        let mut energies = Vec::with_capacity(n);
        for energy in &self.energies {
            let Some(energy) = energy else {
                return Err(SaddleError::Invalid(
                    "AutoNEB image energy is missing".into(),
                ));
            };
            energies.push(*energy);
        }
        let span = segment_length(
            self.positions.row(0),
            self.positions.row(n - 1),
            self.config.band.cell.as_ref(),
        );
        worst_segment(&lengths, &energies, span, self.config.space_energy_ratio)
    }

    fn insert_gap(&mut self, after: usize, n_between: usize) -> Result<(), SaddleError> {
        let left = self.positions.row(after).to_owned();
        let right = self.positions.row(after + 1).to_owned();
        let cell = self.config.band.cell;
        let images =
            self.interpolation
                .interpolate(left.view(), right.view(), n_between, cell.as_ref())?;
        let images = checked_images(images, n_between, left.len())?;
        self.positions = splice_rows(&self.positions, after, &images);
        self.energies
            .splice(after + 1..after + 1, std::iter::repeat_n(None, n_between));
        Ok(())
    }

    fn ensure_energies<S: BandSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        if self.energies.iter().all(Option::is_some) {
            return Ok(());
        }
        let n = self.positions.nrows();
        let dof = self.positions.ncols();
        let mut energies = Array1::zeros(n);
        let mut gradients = Array2::zeros((n, dof));
        surface.eval(self.positions.view(), &mut energies, &mut gradients)?;
        self.surface_rows += n;
        if !energies.iter().all(|energy| energy.is_finite()) {
            return Err(SaddleError::NonFinite("AutoNEB energy"));
        }
        self.energies = energies.iter().copied().map(Some).collect();
        Ok(())
    }

    fn relax<S: BandSurface>(
        &mut self,
        surface: &S,
        climb: bool,
        force_tol: f64,
        max_steps: usize,
    ) -> Result<crate::band::BandReport, SaddleError> {
        let window = local_window(self.positions.nrows(), self.config.n_simul, &self.energies);
        let dof = self.positions.ncols();
        let mut sub = Array2::zeros((window.len(), dof));
        for (row, &index) in window.iter().enumerate() {
            sub.row_mut(row).assign(&self.positions.row(index));
        }
        let mut band = self.config.band.clone();
        band.force_tol = force_tol;
        band.climbing = climb.then_some(CiConfig {
            trigger_factor: 1.0,
            trigger_force: f64::INFINITY,
        });
        let mut session = BandSession::new(band, sub)?;
        let mut report = session.step(surface)?;
        self.surface_rows += report.surface_rows;
        while report.status == BandStatus::Running && report.iteration < max_steps {
            report = session.step(surface)?;
            self.surface_rows += report.surface_rows;
        }
        let relaxed = session.positions().to_owned();
        if !relaxed.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("AutoNEB positions"));
        }
        for (row, &index) in window.iter().enumerate() {
            if row == 0 || row + 1 == window.len() {
                continue;
            }
            self.positions.row_mut(index).assign(&relaxed.row(row));
        }
        if let Some(eval) = session.evaluation() {
            for (row, &index) in window.iter().enumerate() {
                let energy = eval.energies[row];
                if !energy.is_finite() {
                    return Err(SaddleError::NonFinite("AutoNEB energy"));
                }
                self.energies[index] = Some(energy);
            }
        } else {
            // The next insertion reads these energies. Drop them so the
            // following whole-band evaluation fills the moved window.
            for &index in &window {
                self.energies[index] = None;
            }
        }
        self.ci_index = climb
            .then_some(report.ci_index)
            .flatten()
            .map(|local| window[local]);
        Ok(report)
    }
}

fn check_initial(initial: &Array2<f64>, n_max: usize) -> Result<(), SaddleError> {
    let (n, dof) = initial.dim();
    if n < 2 || dof == 0 || !dof.is_multiple_of(3) {
        return Err(SaddleError::Shape(format!(
            "AutoNEB needs at least 2 images of 3N dof; got {n} x {dof}"
        )));
    }
    if n > n_max {
        return Err(SaddleError::Invalid(format!(
            "AutoNEB initial band has {n} images and n_max is {n_max}"
        )));
    }
    if !initial.iter().all(|value| value.is_finite()) {
        return Err(SaddleError::NonFinite("AutoNEB positions"));
    }
    Ok(())
}

fn checked_images(
    images: Vec<Array1<f64>>,
    n_between: usize,
    dof: usize,
) -> Result<Vec<Array1<f64>>, SaddleError> {
    if images.len() != n_between {
        return Err(SaddleError::Shape(format!(
            "interpolation returned {} images; expected {n_between}",
            images.len()
        )));
    }
    for image in &images {
        if image.len() != dof {
            return Err(SaddleError::Shape(
                "interpolation image length does not match the band".into(),
            ));
        }
        if !image.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("interpolation"));
        }
    }
    Ok(images)
}

fn segment_length(left: ArrayView1<f64>, right: ArrayView1<f64>, cell: Option<&Cell>) -> f64 {
    let mut step = &right - &left;
    if let Some(cell) = cell {
        cell.minimum_image(&mut step);
    }
    step.dot(&step).sqrt()
}

fn splice_rows(positions: &Array2<f64>, after: usize, rows: &[Array1<f64>]) -> Array2<f64> {
    let n = positions.nrows();
    let dof = positions.ncols();
    let mut out = Array2::zeros((n + rows.len(), dof));
    for i in 0..=after {
        out.row_mut(i).assign(&positions.row(i));
    }
    for (k, row) in rows.iter().enumerate() {
        out.row_mut(after + 1 + k).assign(row);
    }
    for i in (after + 1)..n {
        out.row_mut(i + rows.len()).assign(&positions.row(i));
    }
    out
}

/// Free images are the interior of the returned index run.
///
/// A short band is returned whole. Otherwise the run pins one image
/// on each side of at least `n_simul` free images. Every image that
/// still has no energy is free, so a just-inserted image moves; the
/// window grows past `n_simul` when several are missing. With every
/// energy known, the free set is centered on the highest interior
/// energy and that image is not a pin.
fn local_window(n_images: usize, n_simul: usize, energies: &[Option<f64>]) -> Vec<usize> {
    let pins = n_simul + 2;
    if n_images <= pins {
        return (0..n_images).collect();
    }
    let missing: Vec<usize> = (1..n_images - 1)
        .filter(|&index| energies[index].is_none())
        .collect();
    let (focus_lo, focus_hi) = if let (Some(&lo), Some(&hi)) = (missing.first(), missing.last()) {
        (lo, hi)
    } else {
        let highest = highest_interior(energies);
        (highest, highest)
    };
    let need = focus_hi - focus_lo + 1;
    let interior = need.max(n_simul);
    let spare = interior - need;
    let mut interior_start = focus_lo.saturating_sub(spare / 2);
    if interior_start < 1 {
        interior_start = 1;
    }
    let max_start = (n_images - 1) - interior;
    if interior_start > max_start {
        interior_start = max_start;
    }
    let start = interior_start - 1;
    let end = interior_start + interior;
    (start..=end).collect()
}

fn highest_interior(energies: &[Option<f64>]) -> usize {
    let n = energies.len();
    let mut best_i = n / 2;
    let mut best_e = f64::NEG_INFINITY;
    let mut any = false;
    for (index, energy) in energies
        .iter()
        .enumerate()
        .skip(1)
        .take(n.saturating_sub(2))
    {
        if let Some(energy) = energy
            && *energy > best_e
        {
            best_e = *energy;
            best_i = index;
            any = true;
        }
    }
    if any {
        best_i
    } else if n >= 3 {
        n / 2
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn linear_midpoint_uses_the_minimum_image() {
        let cell = Cell([[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0]]);
        let left = array![0.0, 0.0, 0.0];
        let right = array![9.0, 0.0, 0.0];
        let images = LinearInterpolation
            .interpolate(left.view(), right.view(), 1, Some(&cell))
            .unwrap();
        assert_eq!(images.len(), 1);
        assert_abs_diff_eq!(images[0][0], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(images[0][1], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn energy_gap_wins_when_the_ratio_says_so() {
        let lengths = [1.0, 1.0, 1.0, 1.0];
        let energies = [0.0, 0.0, 10.0, 0.0, 0.0];
        // f_R / g_R is 0.5, so a threshold of 0.5 takes the energy gap
        // and a threshold of 0 takes the first equal-length segment.
        assert_eq!(worst_segment(&lengths, &energies, 4.0, 0.5).unwrap(), 1);
        assert_eq!(worst_segment(&lengths, &energies, 4.0, 0.0).unwrap(), 0);
    }

    #[test]
    fn longest_segment_wins_when_geometry_is_worse() {
        let lengths = [1.0, 1.0, 1.0, 10.0];
        let energies = [0.0, 0.0, 10.0, 0.0, 0.0];
        assert_eq!(worst_segment(&lengths, &energies, 4.0, 0.5).unwrap(), 3);
    }

    #[test]
    fn flat_energies_fall_back_to_the_longest_segment() {
        let lengths = [1.0, 3.0, 2.0];
        let energies = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(worst_segment(&lengths, &energies, 4.0, 0.5).unwrap(), 1);
    }

    #[test]
    fn construction_fills_the_first_window_on_the_longest_segment() {
        let config = AutoNebConfig {
            n_simul: 1,
            n_max: 4,
            ..AutoNebConfig::default()
        };
        let initial = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let search = AutoNeb::new(config, initial).unwrap();
        assert_eq!(search.positions().nrows(), 3);
        assert_abs_diff_eq!(search.positions()[(1, 0)], 1.0, epsilon = 1e-12);
        assert!(search.energies().iter().all(Option::is_none));
    }

    #[test]
    fn weighted_springs_and_a_short_budget_are_refused() {
        let mut config = AutoNebConfig::default();
        config.band.spring = SpringKind::Weighted { ks: vec![1.0, 1.0] };
        let initial = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(matches!(
            AutoNeb::new(config, initial.clone()),
            Err(SaddleError::Invalid(_))
        ));
        let config = AutoNebConfig {
            n_simul: 3,
            n_max: 4,
            ..AutoNebConfig::default()
        };
        assert!(matches!(
            AutoNeb::new(config, initial),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn the_window_keeps_the_peak_and_a_new_image_free() {
        let energies = vec![
            Some(0.0),
            Some(0.2),
            Some(0.4),
            Some(1.0),
            Some(0.5),
            Some(0.1),
            Some(0.0),
        ];
        let window = local_window(7, 3, &energies);
        assert_eq!(window, vec![1, 2, 3, 4, 5]);
        assert!(window[1..window.len() - 1].contains(&3));

        let mut missing = energies;
        missing[5] = None;
        let window = local_window(7, 3, &missing);
        let free = &window[1..window.len() - 1];
        assert!(free.contains(&5), "{window:?}");
    }

    struct Flat;

    impl BandSurface for Flat {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let _ = positions;
            energies.fill(0.0);
            gradients.fill(0.0);
            Ok(())
        }
    }

    #[test]
    fn a_second_run_is_refused_and_the_ends_stay_put() {
        let config = AutoNebConfig {
            n_simul: 1,
            n_max: 4,
            local_max_steps: 5,
            climb_max_steps: 5,
            ..AutoNebConfig::default()
        };
        let initial = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut search = AutoNeb::new(config, initial).unwrap();
        let report = search.run(&Flat).unwrap();
        assert_eq!(report.n_images, 4);
        assert_eq!(report.status, BandStatus::Converged);
        assert_eq!(search.positions()[(0, 0)], 0.0);
        assert_eq!(search.positions()[(3, 0)], 1.0);
        assert!(report.surface_rows > 0);
        assert!(matches!(search.run(&Flat), Err(SaddleError::Invalid(_))));
    }
}
