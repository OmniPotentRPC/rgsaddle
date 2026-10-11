//! One image evaluator for band methods.
//!
//! Ranks are [`image_block`]: contiguous images, the first
//! `n % ranks` ranks owning one extra image. A pool built with
//! `threads` workers runs those blocks. `RAYON_NUM_THREADS` does not
//! change the partition. One thread calls the surface once on the
//! whole batch. Each block is written back by image index after every
//! rank has finished. There is no cross-thread sum and no SIMD path.

use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::SaddleError;

use super::{ImageId, StencilSurface};

/// Contiguous images owned by one rank.
///
/// Ranks `0..n_ranks` cover `0..n_images` without overlap. The first
/// `n_images % n_ranks` ranks own one extra image.
pub fn image_block(n_images: usize, rank: usize, n_ranks: usize) -> std::ops::Range<usize> {
    if n_ranks == 0 || rank >= n_ranks || n_images == 0 {
        return 0..0;
    }
    let base = n_images / n_ranks;
    let extra = n_images % n_ranks;
    let start = rank * base + rank.min(extra);
    let len = base + usize::from(rank < extra);
    start..start + len
}

struct BlockBuf {
    /// Full-band block, sized at construction.
    wide_e: Array1<f64>,
    wide_g: Array2<f64>,
    /// Shorter block (an interior evaluation). Reallocated only when the length changes.
    fit_e: Array1<f64>,
    fit_g: Array2<f64>,
}

/// Workers kept for the life of one stencil.
pub struct ImagePool {
    threads: usize,
    pool: Option<rayon::ThreadPool>,
    blocks: Vec<Mutex<BlockBuf>>,
    capacity: usize,
    dof: usize,
}

impl ImagePool {
    /// `rows` and `dof` size the scratch buffers. `threads == 1` keeps
    /// no pool and no block buffers.
    pub fn new(threads: usize, rows: usize, dof: usize) -> Result<Self, SaddleError> {
        if threads == 0 {
            return Err(SaddleError::Invalid(
                "image evaluation needs at least one thread".into(),
            ));
        }
        if rows == 0 || dof == 0 {
            return Err(SaddleError::Shape(
                "image evaluation needs at least one row and one coordinate".into(),
            ));
        }
        if threads == 1 {
            return Ok(Self {
                threads,
                pool: None,
                blocks: Vec::new(),
                capacity: rows,
                dof,
            });
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| SaddleError::Invalid(format!("image evaluation pool: {err}")))?;
        let ranks = threads.min(rows);
        let capacity = image_block(rows, 0, ranks).len().max(1);
        let blocks = (0..ranks)
            .map(|rank| {
                let fit = image_block(rows.saturating_sub(2).max(1), rank, ranks)
                    .len()
                    .max(1);
                Mutex::new(BlockBuf {
                    wide_e: Array1::zeros(capacity),
                    wide_g: Array2::zeros((capacity, dof)),
                    fit_e: Array1::zeros(fit),
                    fit_g: Array2::zeros((fit, dof)),
                })
            })
            .collect();
        Ok(Self {
            threads,
            pool: Some(pool),
            blocks,
            capacity,
            dof,
        })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Evaluate `positions` and store each row at its index.
    pub fn eval<S: StencilSurface + ?Sized>(
        &self,
        surface: &S,
        ids: &[ImageId],
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let n = positions.nrows();
        let dof = positions.ncols();
        if ids.len() != n || energies.len() != n || gradients.dim() != (n, dof) {
            return Err(SaddleError::Shape(format!(
                "image evaluation needs one id, energy, and gradient row per image; got {n} positions"
            )));
        }
        if n == 0 {
            return Ok(());
        }
        if self.threads == 1 || n == 1 {
            return surface.eval(ids, positions, energies, gradients);
        }
        let pool = self.pool.as_ref().expect("pool");
        let ranks = self.threads.min(n);
        if ranks > self.blocks.len() || dof != self.dof {
            return Err(SaddleError::Shape(
                "image evaluation pool was built for a smaller band".into(),
            ));
        }
        let ranges: Vec<std::ops::Range<usize>> =
            (0..ranks).map(|rank| image_block(n, rank, ranks)).collect();
        let widest = ranges.iter().map(|range| range.len()).max().unwrap_or(0);
        if widest > self.capacity {
            return Err(SaddleError::Shape(
                "image evaluation block is larger than the scratch buffer".into(),
            ));
        }
        let failed: Mutex<Option<SaddleError>> = Mutex::new(None);
        pool.scope(|scope| {
            for (rank, range) in ranges.iter().enumerate() {
                let range = range.clone();
                let failed = &failed;
                scope.spawn(move |_| {
                    if range.is_empty() {
                        return;
                    }
                    let mut slot = self.blocks[rank]
                        .lock()
                        .unwrap_or_else(|err| err.into_inner());
                    let rows = range.end - range.start;
                    let view = positions.slice(ndarray::s![range.start..range.end, ..]);
                    let (energies, gradients) = block_buffers(&mut slot, rows, dof);
                    let result =
                        surface.eval(&ids[range.start..range.end], view, energies, gradients);
                    if let Err(err) = result
                        && let Ok(mut flag) = failed.lock()
                        && flag.is_none()
                    {
                        *flag = Some(err);
                    }
                });
            }
        });
        if let Some(err) = failed.lock().unwrap_or_else(|err| err.into_inner()).take() {
            return Err(err);
        }
        for (rank, range) in ranges.iter().enumerate() {
            if range.is_empty() {
                continue;
            }
            let slot = self.blocks[rank]
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            let rows = range.len();
            let (block_e, block_g) = if slot.wide_e.len() == rows {
                (slot.wide_e.view(), slot.wide_g.view())
            } else {
                (slot.fit_e.view(), slot.fit_g.view())
            };
            energies
                .slice_mut(ndarray::s![range.clone()])
                .assign(&block_e.slice(ndarray::s![..rows]));
            gradients
                .slice_mut(ndarray::s![range.clone(), ..])
                .assign(&block_g.slice(ndarray::s![..rows, ..]));
        }
        Ok(())
    }
}

fn block_buffers(
    slot: &mut BlockBuf,
    rows: usize,
    dof: usize,
) -> (&mut Array1<f64>, &mut Array2<f64>) {
    if slot.wide_e.len() == rows && slot.wide_g.dim() == (rows, dof) {
        return (&mut slot.wide_e, &mut slot.wide_g);
    }
    if slot.fit_e.len() != rows || slot.fit_g.dim() != (rows, dof) {
        slot.fit_e = Array1::zeros(rows);
        slot.fit_g = Array2::zeros((rows, dof));
    }
    (&mut slot.fit_e, &mut slot.fit_g)
}
