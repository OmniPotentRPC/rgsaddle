//! AutoNEB grows a band by inserting images, then climbs.
//!
//! Kolsbjerg, Groves, and Hammer, J. Chem. Phys. 145, 094107 (2016),
//! doi:10.1063/1.4961868.
//!
//! The caller supplies the endpoints and any intermediate guesses.
//! Construction fills the band until it holds `n_simul + 2` images, or
//! `n_max` if that is smaller, by cutting the longest segment.
//! [`AutoNeb::run`] relaxes a local window, inserts one image where the
//! resolution is worst, and repeats until `n_max` images. Climbing
//! then runs on the window around the highest interior energy, and
//! the full band is relaxed to [`BandConfig::force_tol`] with the
//! same climbing settings. An unchanged window keeps one
//! [`BandSession`], so its L-BFGS pairs survive a climb that starts
//! on the same images. A new index set, including the full band when
//! that set differs, starts with empty memory.
//!
//! [`ImageInterpolation`] places every new image. The initial fill
//! inserts one midpoint of the longest segment at a time. ASE's
//! AutoNEB, given only the two endpoints, places `n_simul` images in
//! one equal-spaced interpolate. Those two fills meet on a unit
//! segment when `n_simul` is 1 or 3, and they differ for 2.
//! [`LinearInterpolation`] is the straight minimum-image segment. An
//! IDPP path implements the same trait; this crate does not ship one.
//! Buildup leaves climbing off. The spring is uniform for now. A
//! weighted or Onsager-Machlup spring changes length when an image is
//! inserted; zoom-NEB will recompute those springs, and the refusal
//! is temporary. The solid-state block is not part of this loop.
//!
//! `surface_rows` counts force evaluations: one row of
//! [`BandSurface::eval`] is one force call. A surface error discards
//! the window that was in progress and leaves the previous window in
//! place. [`AutoNeb::run`] starts once.
//!
//! [`AutoNebTuning::threads`] splits each evaluation into that many
//! contiguous image blocks. Block `r` of `n` is [`image_block`]. Each
//! block is evaluated on its own thread, and the energy and gradient
//! are stored at the image index after every block has finished.
//! [`worst_segment`] then reads that array from low index to high, so
//! the inserted image does not depend on which block finished first.
//! One thread evaluates the batch as a single call. The surface sees
//! one block per call. A host that tells a whole-band evaluation from
//! an interior evaluation by the row count should leave `threads` at 1.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::band::{BandConfig, BandSession, BandStatus, BandSurface, Cell, CiConfig};
use crate::error::SaddleError;
use crate::spring::SpringKind;

/// Where one new image is placed.
///
/// `insert` returns the coordinates of the image that goes after
/// chain row `after`. `energies` is one entry per chain row; a linear
/// insert ignores it. An IDPP insert minimizes the image-dependent
/// pair potential on that segment.
pub(crate) trait ImageInterpolation: Sync {
    fn insert(
        &self,
        chain: ArrayView2<f64>,
        energies: &[Option<f64>],
        after: usize,
    ) -> Result<Array1<f64>, SaddleError>;
}

/// Straight minimum-image midpoint of the chosen segment.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LinearInterpolation {
    cell: Option<Cell>,
}

impl LinearInterpolation {
    fn new(cell: Option<Cell>) -> Self {
        Self { cell }
    }
}

impl ImageInterpolation for LinearInterpolation {
    fn insert(
        &self,
        chain: ArrayView2<f64>,
        energies: &[Option<f64>],
        after: usize,
    ) -> Result<Array1<f64>, SaddleError> {
        let _ = energies;
        let n = chain.nrows();
        let dof = chain.ncols();
        if n < 2 || after + 1 >= n || dof == 0 || !dof.is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "interpolation needs a segment inside a 3N chain; got {n} x {dof}, after {after}"
            )));
        }
        let left = chain.row(after);
        let right = chain.row(after + 1);
        if !left.iter().chain(right.iter()).all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("interpolation"));
        }
        let mut step = &right - &left;
        if let Some(cell) = &self.cell {
            cell.minimum_image(&mut step);
        }
        Ok(&left + &step * 0.5)
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

/// Contiguous images owned by one rank.
///
/// Ranks `0..n_ranks` cover `0..n_images` without overlap. The first
/// `n_images % n_ranks` ranks own one extra image. A rank past
/// `n_ranks`, a zero rank count, or an empty band returns an empty
/// range.
pub(crate) fn image_block(n_images: usize, rank: usize, n_ranks: usize) -> std::ops::Range<usize> {
    if n_ranks == 0 || rank >= n_ranks || n_images == 0 {
        return 0..0;
    }
    let base = n_images / n_ranks;
    let extra = n_images % n_ranks;
    let start = rank * base + rank.min(extra);
    let len = base + usize::from(rank < extra);
    start..start + len
}

struct OwnedBlock {
    start: usize,
    energies: Array1<f64>,
    gradients: Array2<f64>,
}

/// Worker threads kept for every image evaluation of one search.
///
/// [`eval_images`] builds a pool for a single call. [`AutoNeb`] keeps
/// one pool for the whole run, so a band step does not spawn operating
/// system threads. Each rank still owns [`image_block`], and the block
/// is stored at that image index after every rank has finished.
pub(crate) struct ImageEvalPool {
    threads: usize,
    pool: rayon::ThreadPool,
}

impl ImageEvalPool {
    /// `threads` is the worker count. Zero is an error.
    pub(crate) fn new(threads: usize) -> Result<Self, SaddleError> {
        if threads == 0 {
            return Err(SaddleError::Invalid(
                "image evaluation needs at least one thread".into(),
            ));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| SaddleError::Invalid(format!("image evaluation pool: {err}")))?;
        Ok(Self { threads, pool })
    }

    #[allow(dead_code)]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Evaluate `positions` on this pool. See [`eval_images`].
    pub fn eval<S: BandSurface + ?Sized>(
        &self,
        surface: &S,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        eval_checked(surface, positions, energies, gradients, Some(self))
    }
}

/// Evaluate every image and store the answer at that image's index.
///
/// `threads` is the rank count. Rank `r` evaluates
/// [`image_block`] of `min(threads, n)` on the pool when that count is
/// greater than one. The block is written back only after every rank
/// has finished, in image order, so a slower rank cannot change which
/// slot receives which row. One thread, or a single image, calls
/// [`BandSurface::eval`] once on the whole batch. The stored values
/// match that call when the surface depends only on the row it is
/// given. `threads == 0` is an error. An empty batch is a no-op.
/// There is no SIMD path: each row is scalar `f64` arithmetic.
#[allow(dead_code)]
pub(crate) fn eval_images<S: BandSurface + ?Sized>(
    surface: &S,
    positions: ArrayView2<f64>,
    energies: &mut Array1<f64>,
    gradients: &mut Array2<f64>,
    threads: usize,
) -> Result<(), SaddleError> {
    if threads == 0 {
        return Err(SaddleError::Invalid(
            "image evaluation needs at least one thread".into(),
        ));
    }
    let n = positions.nrows();
    if n > 1 && threads > 1 {
        let pool = ImageEvalPool::new(threads)?;
        return pool.eval(surface, positions, energies, gradients);
    }
    eval_checked(surface, positions, energies, gradients, None)
}

fn eval_checked<S: BandSurface + ?Sized>(
    surface: &S,
    positions: ArrayView2<f64>,
    energies: &mut Array1<f64>,
    gradients: &mut Array2<f64>,
    pool: Option<&ImageEvalPool>,
) -> Result<(), SaddleError> {
    let n = positions.nrows();
    let dof = positions.ncols();
    if energies.len() != n || gradients.dim() != (n, dof) {
        return Err(SaddleError::Shape(format!(
            "image evaluation needs one energy and one gradient row per image; got {n} positions, {} energies, {} x {} gradients",
            energies.len(),
            gradients.nrows(),
            gradients.ncols()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let threads = pool.map(|pool| pool.threads).unwrap_or(1);
    let ranks = threads.min(n);
    if ranks == 1 || pool.is_none() {
        return surface.eval(positions, energies, gradients);
    }
    let pool = pool.expect("pool");
    let blocks: Vec<std::ops::Range<usize>> = (0..ranks)
        .map(|rank| image_block(n, rank, ranks))
        .filter(|range| !range.is_empty())
        .collect();
    let slots: Vec<std::sync::Mutex<Option<Result<OwnedBlock, SaddleError>>>> = (0..blocks.len())
        .map(|_| std::sync::Mutex::new(None))
        .collect();
    pool.pool.scope(|scope| {
        for (slot, range) in slots.iter().zip(blocks.iter()) {
            let range = range.clone();
            scope.spawn(move |_| {
                let rows = range.end - range.start;
                let view = positions.slice(ndarray::s![range.start..range.end, ..]);
                let mut block_energies = Array1::zeros(rows);
                let mut block_gradients = Array2::zeros((rows, dof));
                let result = surface
                    .eval(view, &mut block_energies, &mut block_gradients)
                    .map(|()| OwnedBlock {
                        start: range.start,
                        energies: block_energies,
                        gradients: block_gradients,
                    });
                *slot.lock().unwrap_or_else(|err| err.into_inner()) = Some(result);
            });
        }
    });
    let mut ok = Vec::with_capacity(slots.len());
    for slot in &slots {
        match slot.lock().unwrap_or_else(|err| err.into_inner()).take() {
            Some(Ok(block)) => ok.push(block),
            Some(Err(err)) => return Err(err),
            None => {
                return Err(SaddleError::Invalid(
                    "image evaluation rank did not finish".into(),
                ));
            }
        }
    }
    ok.sort_by_key(|block| block.start);
    debug_assert!(blocks_cover(&ok, n));
    for block in ok {
        let end = block.start + block.energies.len();
        energies
            .slice_mut(ndarray::s![block.start..end])
            .assign(&block.energies);
        gradients
            .slice_mut(ndarray::s![block.start..end, ..])
            .assign(&block.gradients);
    }
    Ok(())
}

fn blocks_cover(blocks: &[OwnedBlock], n: usize) -> bool {
    let mut next = 0;
    for block in blocks {
        if block.start != next {
            return false;
        }
        next += block.energies.len();
    }
    next == n
}

/// Surface that evaluates images on a search's pool.
struct RankedEval<'a, S> {
    surface: &'a S,
    pool: Option<&'a ImageEvalPool>,
}

impl<S: BandSurface> BandSurface for RankedEval<'_, S> {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        match self.pool {
            Some(pool) => pool.eval(self.surface, positions, energies, gradients),
            None => self.surface.eval(positions, energies, gradients),
        }
    }
}

/// AutoNEB controls stored on [`BandConfig::autoneb`].
///
/// Distances and forces are in the caller's units. Climbing uses the
/// parent band's [`BandConfig::climbing`] settings.
#[derive(Clone, Debug)]
pub struct AutoNebTuning {
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
    /// Threads that evaluate images. `1` evaluates the batch on the
    /// calling thread. A larger value uses [`image_block`] and stores
    /// each result by image index. The surface sees one block per call.
    pub threads: usize,
}

impl Default for AutoNebTuning {
    fn default() -> Self {
        Self {
            n_simul: 3,
            n_max: 7,
            local_force_tol: 1e-1,
            local_max_steps: 100,
            climb_max_steps: 2000,
            space_energy_ratio: 0.5,
            climb: true,
            threads: 1,
        }
    }
}

impl AutoNebTuning {
    fn validate(&self, band: &BandConfig) -> Result<(), SaddleError> {
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
        if self.threads == 0 {
            return Err(SaddleError::Invalid(
                "AutoNEB needs at least one image-evaluation thread".into(),
            ));
        }
        if !(self.local_force_tol > 0.0 && self.local_force_tol.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB local force tolerance must be finite and positive".into(),
            ));
        }
        if !(band.force_tol > 0.0 && band.force_tol.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB force tolerance must be finite and positive".into(),
            ));
        }
        if !(self.space_energy_ratio >= 0.0 && self.space_energy_ratio.is_finite()) {
            return Err(SaddleError::Invalid(
                "AutoNEB space_energy_ratio must be finite and non-negative".into(),
            ));
        }
        if band.solid.is_some() {
            return Err(SaddleError::Invalid(
                "AutoNEB does not carry the solid-state block".into(),
            ));
        }
        match &band.spring {
            SpringKind::Uniform { k } if k.is_finite() => {}
            SpringKind::Uniform { .. } => {
                return Err(SaddleError::Invalid(
                    "AutoNEB spring constant must be finite".into(),
                ));
            }
            // Temporary. Zoom-NEB recomputes springs when an image is
            // inserted, which removes this refusal.
            SpringKind::Weighted { .. } | SpringKind::OnsagerMachlup { .. } => {
                return Err(SaddleError::Invalid(
                    "AutoNEB temporarily needs a uniform spring; weighted springs will be recomputed on insertion with zoom-NEB"
                        .into(),
                ));
            }
        }
        let method = if band.quickmin {
            rgmin::Method::QuickMin
        } else {
            band.method.clone()
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
    /// Steps of the final full-band session.
    pub iteration: usize,
}

/// One band session kept while its image indices do not change.
struct OpenWindow {
    indices: Vec<usize>,
    session: BandSession,
}

/// Growing band.
///
/// The interpolant is internal. [`AutoNeb::new`] places each image at
/// the minimum-image midpoint.
pub struct AutoNeb {
    band: BandConfig,
    tuning: AutoNebTuning,
    interpolation: Box<dyn ImageInterpolation + Sync>,
    positions: Array2<f64>,
    energies: Vec<Option<f64>>,
    /// Segment index of each insertion, including the initial fill.
    insertions: Vec<usize>,
    surface_rows: usize,
    ci_index: Option<usize>,
    started: bool,
    pool: Option<ImageEvalPool>,
    open: Option<OpenWindow>,
    /// Times [`BandSession::new`] ran. An unchanged window does not
    /// increment it.
    sessions_opened: usize,
}

impl AutoNeb {
    /// Fill `initial` up to the first window and keep linear insertion
    /// for the rest. `initial` has at least the two endpoints.
    /// [`BandConfig::autoneb`] supplies the tuning; `None` uses
    /// [`AutoNebTuning::default`].
    pub fn new(config: BandConfig, initial: Array2<f64>) -> Result<Self, SaddleError> {
        let interpolation = LinearInterpolation::new(config.cell);
        Self::with_interpolation(config, initial, interpolation)
    }

    /// Same as [`AutoNeb::new`] with a caller-supplied interpolant.
    pub(crate) fn with_interpolation(
        config: BandConfig,
        initial: Array2<f64>,
        interpolation: impl ImageInterpolation + 'static,
    ) -> Result<Self, SaddleError> {
        let tuning = config.autoneb.clone().unwrap_or_default();
        tuning.validate(&config)?;
        check_initial(&initial, tuning.n_max)?;
        let pool = if tuning.threads > 1 {
            Some(ImageEvalPool::new(tuning.threads)?)
        } else {
            None
        };
        let n = initial.nrows();
        let mut search = Self {
            band: config,
            tuning,
            interpolation: Box::new(interpolation),
            positions: initial,
            energies: vec![None; n],
            insertions: Vec::new(),
            surface_rows: 0,
            ci_index: None,
            started: false,
            pool,
            open: None,
            sessions_opened: 0,
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

    /// Segment index of each insertion, initial fill included.
    ///
    /// The index is the image after which the new images were placed,
    /// in the band as it stood before that insertion.
    pub fn insertions(&self) -> &[usize] {
        &self.insertions
    }

    /// Relax, insert, climb, then converge the full band.
    ///
    /// The local windows use [`AutoNebTuning::local_force_tol`]. The
    /// climbing window and the final full band use
    /// [`BandConfig::force_tol`] and the band's climbing settings.
    /// An unchanged window keeps its [`BandSession`]. A different
    /// index set starts a new one, with empty L-BFGS memory.
    pub fn run<S: BandSurface>(&mut self, surface: &S) -> Result<AutoNebReport, SaddleError> {
        if self.started {
            return Err(SaddleError::Invalid(
                "AutoNEB has already been started".into(),
            ));
        }
        self.started = true;
        self.relax(
            surface,
            false,
            false,
            self.tuning.local_force_tol,
            self.tuning.local_max_steps,
        )?;
        self.ensure_energies(surface)?;
        while self.positions.nrows() < self.tuning.n_max {
            let gap = self.next_gap()?;
            self.insert_one(gap)?;
            self.relax(
                surface,
                false,
                false,
                self.tuning.local_force_tol,
                self.tuning.local_max_steps,
            )?;
            self.ensure_energies(surface)?;
        }
        if self.tuning.climb {
            self.relax(
                surface,
                false,
                true,
                self.band.force_tol,
                self.tuning.climb_max_steps,
            )?;
        }
        let last = self.relax(
            surface,
            true,
            self.tuning.climb,
            self.band.force_tol,
            self.tuning.climb_max_steps,
        )?;
        Ok(AutoNebReport {
            status: last.status,
            max_force: last.max_force,
            ci_index: self.ci_index,
            surface_rows: self.surface_rows,
            n_images: self.positions.nrows(),
            iteration: last.iteration,
        })
    }

    fn fill_initial(&mut self) -> Result<(), SaddleError> {
        let target = (self.tuning.n_simul + 2).min(self.tuning.n_max);
        while self.positions.nrows() < target {
            let gap = self.longest_segment();
            self.insert_one(gap)?;
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
                self.band.cell.as_ref(),
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
                self.band.cell.as_ref(),
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
            self.band.cell.as_ref(),
        );
        worst_segment(&lengths, &energies, span, self.tuning.space_energy_ratio)
    }

    fn insert_one(&mut self, after: usize) -> Result<(), SaddleError> {
        let image = self
            .interpolation
            .insert(self.positions.view(), &self.energies, after)?;
        let dof = self.positions.ncols();
        if image.len() != dof {
            return Err(SaddleError::Shape(
                "interpolation image length does not match the band".into(),
            ));
        }
        if !image.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("interpolation"));
        }
        self.positions = splice_rows(&self.positions, after, &[image]);
        self.energies.insert(after + 1, None);
        self.insertions.push(after);
        self.open = None;
        Ok(())
    }

    fn eval_rows<S: BandSurface>(
        &self,
        surface: &S,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        match &self.pool {
            Some(pool) => pool.eval(surface, positions, energies, gradients),
            None => surface.eval(positions, energies, gradients),
        }
    }

    fn ensure_energies<S: BandSurface>(&mut self, surface: &S) -> Result<(), SaddleError> {
        let missing: Vec<usize> = self
            .energies
            .iter()
            .enumerate()
            .filter(|(_, energy)| energy.is_none())
            .map(|(index, _)| index)
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let dof = self.positions.ncols();
        let mut positions = Array2::zeros((missing.len(), dof));
        for (row, &index) in missing.iter().enumerate() {
            positions.row_mut(row).assign(&self.positions.row(index));
        }
        let mut energies = Array1::zeros(missing.len());
        let mut gradients = Array2::zeros((missing.len(), dof));
        self.eval_rows(surface, positions.view(), &mut energies, &mut gradients)?;
        self.surface_rows += missing.len();
        for (row, &index) in missing.iter().enumerate() {
            let energy = energies[row];
            if !energy.is_finite() {
                return Err(SaddleError::NonFinite("AutoNEB energy"));
            }
            self.energies[index] = Some(energy);
        }
        Ok(())
    }

    fn ci_settings(&self) -> CiConfig {
        self.band.climbing.unwrap_or(CiConfig {
            trigger_factor: 0.5,
            trigger_force: 0.0,
        })
    }

    fn relax<S: BandSurface>(
        &mut self,
        surface: &S,
        full: bool,
        climb: bool,
        force_tol: f64,
        max_steps: usize,
    ) -> Result<crate::band::BandReport, SaddleError> {
        let window = if full {
            (0..self.positions.nrows()).collect::<Vec<_>>()
        } else {
            local_window(self.positions.nrows(), self.tuning.n_simul, &self.energies)
        };
        let climbing = climb.then_some(self.ci_settings());
        let reuse = self
            .open
            .as_ref()
            .is_some_and(|open| open.indices == window);
        if reuse {
            let open = self.open.as_mut().expect("open window");
            open.session.set_force_tol(force_tol)?;
            open.session.set_climbing(climbing);
        } else {
            let dof = self.positions.ncols();
            let mut sub = Array2::zeros((window.len(), dof));
            for (row, &index) in window.iter().enumerate() {
                sub.row_mut(row).assign(&self.positions.row(index));
            }
            let mut band = self.band.clone();
            band.force_tol = force_tol;
            band.climbing = climbing;
            band.autoneb = None;
            let mut session = BandSession::new(band, sub)?;
            let last = *window.last().expect("window");
            if let (Some(left), Some(right)) = (self.energies[window[0]], self.energies[last]) {
                session.seed_endpoint_energies([left, right])?;
            }
            self.sessions_opened += 1;
            self.open = Some(OpenWindow {
                indices: window.clone(),
                session,
            });
        }

        let pool = self.pool.as_ref();
        let ranked = RankedEval { surface, pool };
        let (report, rows, relaxed, evaluation) = {
            let open = self.open.as_mut().expect("open window");
            let mut taken = 0usize;
            let mut report = open.session.step(&ranked)?;
            let mut rows = report.surface_rows;
            taken += 1;
            while report.status == BandStatus::Running && taken < max_steps {
                report = open.session.step(&ranked)?;
                rows += report.surface_rows;
                taken += 1;
            }
            let relaxed = open.session.positions().to_owned();
            let evaluation = open.session.evaluation();
            (report, rows, relaxed, evaluation)
        };
        self.surface_rows += rows;
        if !relaxed.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("AutoNEB positions"));
        }
        for (row, &index) in window.iter().enumerate() {
            if row == 0 || row + 1 == window.len() {
                continue;
            }
            self.positions.row_mut(index).assign(&relaxed.row(row));
        }
        if let Some(eval) = evaluation {
            for (row, &index) in window.iter().enumerate() {
                let energy = eval.energies[row];
                if !energy.is_finite() {
                    return Err(SaddleError::NonFinite("AutoNEB energy"));
                }
                self.energies[index] = Some(energy);
            }
        } else {
            // Pins did not move. Drop only the free images so the next
            // evaluation does not repeat a known energy.
            for &index in window.iter().skip(1).take(window.len().saturating_sub(2)) {
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
        let chain = array![[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]];
        let image = LinearInterpolation::new(Some(cell))
            .insert(chain.view(), &[None, None], 0)
            .unwrap();
        assert_abs_diff_eq!(image[0], -0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(image[1], 0.0, epsilon = 1e-12);
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

    fn band_with(tuning: AutoNebTuning) -> BandConfig {
        BandConfig {
            autoneb: Some(tuning),
            ..BandConfig::default()
        }
    }

    #[test]
    fn construction_fills_the_first_window_on_the_longest_segment() {
        let config = band_with(AutoNebTuning {
            n_simul: 1,
            n_max: 4,
            ..AutoNebTuning::default()
        });
        let initial = array![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let search = AutoNeb::new(config, initial).unwrap();
        assert_eq!(search.positions().nrows(), 3);
        assert_abs_diff_eq!(search.positions()[(1, 0)], 1.0, epsilon = 1e-12);
        assert!(search.energies().iter().all(Option::is_none));
    }

    #[test]
    fn weighted_springs_and_a_short_budget_are_refused() {
        let config = BandConfig {
            spring: SpringKind::Weighted { ks: vec![1.0, 1.0] },
            ..BandConfig::default()
        };
        let initial = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let Err(SaddleError::Invalid(message)) = AutoNeb::new(config, initial.clone()) else {
            panic!("weighted spring");
        };
        assert!(message.contains("temporarily"), "{message}");
        let config = band_with(AutoNebTuning {
            n_simul: 3,
            n_max: 4,
            ..AutoNebTuning::default()
        });
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
        let config = band_with(AutoNebTuning {
            n_simul: 1,
            n_max: 4,
            local_max_steps: 5,
            climb_max_steps: 80,
            ..AutoNebTuning::default()
        });
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

    #[test]
    fn image_blocks_partition_the_band() {
        assert_eq!(image_block(7, 0, 2), 0..4);
        assert_eq!(image_block(7, 1, 2), 4..7);
        assert_eq!(image_block(7, 0, 4), 0..2);
        assert_eq!(image_block(7, 1, 4), 2..4);
        assert_eq!(image_block(7, 2, 4), 4..6);
        assert_eq!(image_block(7, 3, 4), 6..7);
        assert_eq!(image_block(7, 4, 4), 0..0);
        assert_eq!(image_block(0, 0, 2), 0..0);
        assert_eq!(image_block(4, 0, 0), 0..0);
        for n in 1..12 {
            for ranks in 1..=n {
                let mut hits = vec![0_u8; n];
                for rank in 0..ranks {
                    for index in image_block(n, rank, ranks) {
                        hits[index] += 1;
                    }
                }
                assert!(
                    hits.iter().all(|&hit| hit == 1),
                    "n={n} ranks={ranks} {hits:?}"
                );
            }
        }
    }

    struct Quadratic;

    impl BandSurface for Quadratic {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            for (i, row) in positions.outer_iter().enumerate() {
                let mut energy = 0.0;
                for (k, &x) in row.iter().enumerate() {
                    energy += x * x;
                    gradients[(i, k)] = 2.0 * x;
                }
                energies[i] = energy;
            }
            Ok(())
        }
    }

    fn same_bits(left: &Array1<f64>, right: &Array1<f64>) -> bool {
        left.iter()
            .zip(right.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    }

    #[test]
    fn parallel_eval_matches_one_thread_bit_for_bit() {
        let positions = array![
            [-1.5, 0.25, 0.0],
            [0.5, -0.5, 1.0],
            [1.25, 0.0, -0.25],
            [-0.75, 0.5, 0.5],
            [0.0, 1.0, -1.0],
        ];
        let n = positions.nrows();
        let dof = positions.ncols();
        let mut serial_e = Array1::zeros(n);
        let mut serial_g = Array2::zeros((n, dof));
        eval_images(
            &Quadratic,
            positions.view(),
            &mut serial_e,
            &mut serial_g,
            1,
        )
        .unwrap();
        for threads in [2, 3, n] {
            let mut energies = Array1::from_elem(n, f64::NAN);
            let mut gradients = Array2::from_elem((n, dof), f64::NAN);
            eval_images(
                &Quadratic,
                positions.view(),
                &mut energies,
                &mut gradients,
                threads,
            )
            .unwrap();
            assert!(same_bits(&serial_e, &energies), "threads={threads}");
            assert!(
                serial_g
                    .iter()
                    .zip(gradients.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "threads={threads}"
            );
        }
        let err = eval_images(
            &Quadratic,
            positions.view(),
            &mut serial_e,
            &mut serial_g,
            0,
        );
        assert!(matches!(err, Err(SaddleError::Invalid(_))));
        let mut short = Array1::zeros(n - 1);
        let err = eval_images(&Quadratic, positions.view(), &mut short, &mut serial_g, 1);
        assert!(matches!(err, Err(SaddleError::Shape(_))));
        let empty = Array2::<f64>::zeros((0, 3));
        let mut empty_e = Array1::zeros(0);
        let mut empty_g = Array2::zeros((0, 3));
        eval_images(&Quadratic, empty.view(), &mut empty_e, &mut empty_g, 4).unwrap();
    }

    #[test]
    fn zero_image_threads_are_refused() {
        let config = band_with(AutoNebTuning {
            threads: 0,
            ..AutoNebTuning::default()
        });
        let initial = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(matches!(
            AutoNeb::new(config, initial),
            Err(SaddleError::Invalid(_))
        ));
    }

    #[test]
    fn three_midpoints_of_a_unit_segment_are_equally_spaced() {
        let config = band_with(AutoNebTuning {
            n_simul: 3,
            n_max: 5,
            ..AutoNebTuning::default()
        });
        let search = AutoNeb::new(config, array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).unwrap();
        for (index, want) in [0.0, 0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
            assert_abs_diff_eq!(search.positions()[(index, 0)], want, epsilon = 1e-12);
        }
    }

    #[test]
    fn two_repeated_midpoints_are_not_equal_thirds() {
        let config = band_with(AutoNebTuning {
            n_simul: 2,
            n_max: 4,
            ..AutoNebTuning::default()
        });
        let search = AutoNeb::new(config, array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).unwrap();
        assert_abs_diff_eq!(search.positions()[(1, 0)], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(search.positions()[(2, 0)], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn an_unchanged_window_keeps_one_session() {
        let config = band_with(AutoNebTuning {
            n_simul: 1,
            n_max: 3,
            local_max_steps: 2,
            climb_max_steps: 40,
            ..AutoNebTuning::default()
        });
        let initial = array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut search = AutoNeb::new(config, initial).unwrap();
        let report = search.run(&Flat).unwrap();
        assert_eq!(report.status, BandStatus::Converged);
        assert_eq!(search.sessions_opened, 1);
    }

    #[test]
    fn an_autoneb_key_turns_tuning_on_and_weighted_springs_stay_temporary() {
        let mut cfg = crate::band::HostConfig::default();
        assert!(cfg.band.autoneb.is_none());
        crate::band::set_i64(&mut cfg, "n_max", 9).unwrap();
        assert_eq!(cfg.band.autoneb.as_ref().unwrap().n_max, 9);
        let err = crate::band::set_i64(&mut cfg, "spring", 1).unwrap_err();
        let SaddleError::Invalid(message) = err else {
            panic!("spring");
        };
        assert!(message.contains("zoom-NEB"), "{message}");
        assert!(crate::band::set_f64(&mut cfg, "not_a_key", 1.0).is_err());
    }

    struct DelayLeft {
        finished: std::sync::Mutex<Vec<f64>>,
    }

    impl BandSurface for DelayLeft {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let tag = positions[(0, 0)];
            if tag < 0.0 {
                std::thread::sleep(std::time::Duration::from_millis(40));
            }
            for (i, row) in positions.outer_iter().enumerate() {
                let mut energy = 0.0;
                for (k, &x) in row.iter().enumerate() {
                    energy += x * x;
                    gradients[(i, k)] = 2.0 * x;
                }
                energies[i] = energy;
            }
            self.finished.lock().unwrap().push(tag);
            Ok(())
        }
    }

    #[test]
    fn a_late_block_does_not_change_the_insertion_index() {
        let positions = array![
            [-2.0, 0.0, 0.0],
            [-0.5, 0.1, 0.0],
            [0.4, -0.2, 0.0],
            [1.5, 0.3, 0.0],
        ];
        let n = positions.nrows();
        let dof = positions.ncols();
        let mut serial_e = Array1::zeros(n);
        let mut serial_g = Array2::zeros((n, dof));
        eval_images(
            &DelayLeft {
                finished: std::sync::Mutex::new(Vec::new()),
            },
            positions.view(),
            &mut serial_e,
            &mut serial_g,
            1,
        )
        .unwrap();
        let delayed = DelayLeft {
            finished: std::sync::Mutex::new(Vec::new()),
        };
        let mut parallel_e = Array1::from_elem(n, f64::NAN);
        let mut parallel_g = Array2::from_elem((n, dof), f64::NAN);
        eval_images(
            &delayed,
            positions.view(),
            &mut parallel_e,
            &mut parallel_g,
            2,
        )
        .unwrap();
        assert!(same_bits(&serial_e, &parallel_e));
        assert!(
            serial_g
                .iter()
                .zip(parallel_g.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let right = image_block(n, 1, 2);
        let left = image_block(n, 0, 2);
        assert_eq!(
            *delayed.finished.lock().unwrap(),
            vec![positions[(right.start, 0)], positions[(left.start, 0)]]
        );
        let mut lengths = Vec::with_capacity(n - 1);
        for j in 0..n - 1 {
            let step = &positions.row(j + 1) - &positions.row(j);
            lengths.push(step.dot(&step).sqrt());
        }
        let span_step = &positions.row(n - 1) - &positions.row(0);
        let span = span_step.dot(&span_step).sqrt();
        let serial: Vec<f64> = serial_e.iter().copied().collect();
        let parallel: Vec<f64> = parallel_e.iter().copied().collect();
        assert_eq!(
            worst_segment(&lengths, &serial, span, 0.5).unwrap(),
            worst_segment(&lengths, &parallel, span, 0.5).unwrap()
        );
    }

    struct HeavyPoly {
        passes: usize,
        blocks: std::sync::Mutex<Vec<std::time::Duration>>,
    }

    impl BandSurface for HeavyPoly {
        fn eval(
            &self,
            positions: ArrayView2<f64>,
            energies: &mut Array1<f64>,
            gradients: &mut Array2<f64>,
        ) -> Result<(), SaddleError> {
            let started = std::time::Instant::now();
            for (i, row) in positions.outer_iter().enumerate() {
                let mut energy = 0.0;
                for (k, &x) in row.iter().enumerate() {
                    let mut acc = 0.0;
                    let mut deriv = 0.0;
                    for pass in 0..self.passes {
                        let z = x + (pass as f64) * 1e-4;
                        acc += z.sin();
                        deriv += z.cos();
                    }
                    energy += acc;
                    gradients[(i, k)] = deriv;
                }
                energies[i] = energy;
            }
            self.blocks.lock().unwrap().push(started.elapsed());
            Ok(())
        }
    }

    fn median_range(
        samples: &mut [std::time::Duration],
    ) -> (
        std::time::Duration,
        std::time::Duration,
        std::time::Duration,
    ) {
        assert!(samples.len() >= 5);
        samples.sort();
        let mid = samples.len() / 2;
        (samples[mid], samples[0], *samples.last().unwrap())
    }

    /// Strong scaling of one persistent pool. The limiter is read from
    /// the measured block times of the widest run, not from an
    /// efficiency threshold. Ignored so it stays out of CI.
    #[test]
    #[ignore = "timing; run alone in release with --test-threads=1"]
    fn parallel_image_eval_strong_scaling() {
        use std::time::Duration;
        let n_images = 16;
        let dof = 32 * 3;
        let passes = 1600;
        let mut positions = Array2::zeros((n_images, dof));
        for i in 0..n_images {
            for k in 0..dof {
                positions[(i, k)] = (i + 1) as f64 * 1e-3 + k as f64 * 1e-4;
            }
        }
        let hardware = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let processors = std::fs::read_to_string("/proc/cpuinfo")
            .map(|text| {
                text.lines()
                    .filter(|line| line.starts_with("processor"))
                    .count()
            })
            .unwrap_or(0);
        let mut counts = vec![1_usize, 2, 4];
        if hardware > 1 && !counts.contains(&hardware) {
            counts.push(hardware);
        }
        let pools: Vec<ImageEvalPool> = counts
            .iter()
            .map(|&threads| ImageEvalPool::new(threads).unwrap())
            .collect();
        let surface = HeavyPoly {
            passes,
            blocks: std::sync::Mutex::new(Vec::new()),
        };
        let mut serial_e = Array1::zeros(n_images);
        let mut serial_g = Array2::zeros((n_images, dof));
        pools[0]
            .eval(&surface, positions.view(), &mut serial_e, &mut serial_g)
            .unwrap();
        for pool in pools.iter().skip(1) {
            let mut energies = Array1::from_elem(n_images, f64::NAN);
            let mut gradients = Array2::from_elem((n_images, dof), f64::NAN);
            pool.eval(&surface, positions.view(), &mut energies, &mut gradients)
                .unwrap();
            assert!(
                same_bits(&serial_e, &energies),
                "threads={}",
                pool.threads()
            );
        }

        let mut buffers: Vec<(Array1<f64>, Array2<f64>)> = counts
            .iter()
            .map(|_| (Array1::zeros(n_images), Array2::zeros((n_images, dof))))
            .collect();
        let mut samples = vec![Vec::with_capacity(5); counts.len()];
        let mut widest_runs: Vec<(Duration, Vec<Duration>)> = Vec::new();
        let widest = counts.len() - 1;
        for _rep in 0..5 {
            for (index, pool) in pools.iter().enumerate() {
                surface.blocks.lock().unwrap().clear();
                let (energies, gradients) = &mut buffers[index];
                let started = std::time::Instant::now();
                pool.eval(&surface, positions.view(), energies, gradients)
                    .unwrap();
                let elapsed = started.elapsed();
                std::hint::black_box((energies.as_slice(), gradients.as_slice()));
                samples[index].push(elapsed);
                if index == widest {
                    widest_runs.push((elapsed, surface.blocks.lock().unwrap().clone()));
                }
            }
        }

        let mut rows: Vec<(usize, f64, Duration, Duration, f64)> = Vec::new();
        for (index, &threads) in counts.iter().enumerate() {
            let (median, low, high) = median_range(&mut samples[index]);
            let efficiency = if threads == 1 {
                1.0
            } else {
                rows[0].1 / (threads as f64 * median.as_secs_f64())
            };
            rows.push((threads, median.as_secs_f64(), low, high, efficiency));
        }
        widest_runs.sort_by_key(|(wall, _)| *wall);
        let chosen = &widest_runs[widest_runs.len() / 2];
        let block_max = chosen.1.iter().copied().max().unwrap_or(Duration::ZERO);
        let block_min = chosen.1.iter().copied().min().unwrap_or(Duration::ZERO);
        let wall = chosen.0.as_secs_f64();
        let overhead = wall - block_max.as_secs_f64();
        let limiter = if block_max > Duration::ZERO && overhead > 0.4 * block_max.as_secs_f64() {
            "scheduling and join"
        } else if block_min > Duration::ZERO
            && block_max.as_secs_f64() > 1.25 * block_min.as_secs_f64()
        {
            "load imbalance"
        } else {
            "per-image arithmetic"
        };

        let n_simul = 3;
        let window = positions.slice(ndarray::s![0..n_simul, ..]).to_owned();
        let mut window_samples = [Vec::new(), Vec::new()];
        let window_pools = [pools[0].threads(), *counts.last().unwrap()];
        for _rep in 0..5 {
            for (slot, &threads) in window_pools.iter().enumerate() {
                let pool = pools.iter().find(|pool| pool.threads() == threads).unwrap();
                let mut energies = Array1::zeros(n_simul);
                let mut gradients = Array2::zeros((n_simul, dof));
                let started = std::time::Instant::now();
                pool.eval(&surface, window.view(), &mut energies, &mut gradients)
                    .unwrap();
                std::hint::black_box((energies.as_slice(), gradients.as_slice()));
                window_samples[slot].push(started.elapsed());
            }
        }
        let (w1, _, _) = median_range(&mut window_samples[0]);
        let (wp, _, _) = median_range(&mut window_samples[1]);
        let speedup = w1.as_secs_f64() / wp.as_secs_f64();

        eprintln!(
            "hardware_threads={hardware} processors={processors} simd=none images={n_images} dof={dof} passes={passes}"
        );
        eprintln!("threads  median_ms  min_ms  max_ms  efficiency");
        for (threads, median, low, high, efficiency) in &rows {
            eprintln!(
                "{threads:<7}  {:>9.3}  {:>6.3}  {:>6.3}  {efficiency:.3}",
                median * 1e3,
                low.as_secs_f64() * 1e3,
                high.as_secs_f64() * 1e3,
            );
        }
        eprintln!(
            "widest blocks: max_ms={:.3} min_ms={:.3} wall_ms={:.3} overhead_ms={:.3} spread={:.3} limiter={limiter}",
            block_max.as_secs_f64() * 1e3,
            block_min.as_secs_f64() * 1e3,
            wall * 1e3,
            overhead * 1e3,
            block_max.as_secs_f64() / block_min.as_secs_f64().max(1e-12),
        );
        eprintln!(
            "autoneb window: free_images={n_simul} (n_simul cap) threads={} median_ms={:.3} threads={} median_ms={:.3} speedup={speedup:.3}",
            window_pools[0],
            w1.as_secs_f64() * 1e3,
            window_pools[1],
            wp.as_secs_f64() * 1e3,
        );
        assert!(rows[0].1 > 0.0);
        assert!(!limiter.is_empty());
    }
}
