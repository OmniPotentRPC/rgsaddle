//! Chain of images and the mesh operations that do not call a force.
//!
//! [`Chain::evaluate`] owns one block per image. The block size is fixed,
//! so the partition does not depend on the thread count. Each block is
//! written back by image index after the pool joins.
//! Goswami, Gunde, and Jónsson, Front. Chem. (2026),
//! doi:10.3389/fchem.2026.1807063, Sec. 2.3.6, is the arc-length
//! redistribution the Hermite remesh follows.

use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::SaddleError;
use crate::stencil::{ImageId, Surface};

use super::control::Role;

/// Images per block. Independent of the worker count.
pub const EVAL_BLOCK: usize = 1;

/// What a mesh operation changed. No force was called.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshEvent {
    Unchanged,
    Remeshed,
    Inserted { id: ImageId, index: usize },
    Windowed { active: Range<usize> },
}

/// Band geometry operation. Applying one does not evaluate the surface.
#[derive(Clone, Debug)]
pub enum Mesh {
    RemeshLinear,
    RemeshHermite,
    Insert { after: usize },
    WindowAuto { alpha: f64 },
    WindowOffset { n: usize },
}

/// Positions, cached rows, and roles. Image ids survive insert and window.
pub struct Chain {
    ids: Vec<ImageId>,
    x: Array2<f64>,
    e: Vec<f64>,
    g: Array2<f64>,
    roles: Vec<Role>,
    active: Range<usize>,
    next_id: ImageId,
    filled: bool,
}

impl Chain {
    /// One row per image. Ids start at 0 and increase with insertion.
    /// Endpoints are pinned. Interior rows are images.
    pub fn new(x: Array2<f64>) -> Result<Self, SaddleError> {
        let n = x.nrows();
        if n == 0 || x.ncols() == 0 {
            return Err(SaddleError::Shape(
                "a chain needs at least one image and one degree of freedom".into(),
            ));
        }
        if !x.iter().all(|v| v.is_finite()) {
            return Err(SaddleError::NonFinite("chain positions"));
        }
        let dof = x.ncols();
        let mut roles = vec![Role::Image; n];
        if n >= 2 {
            roles[0] = Role::Pinned;
            roles[n - 1] = Role::Pinned;
        } else {
            roles[0] = Role::Walker;
        }
        Ok(Self {
            ids: (0..n as ImageId).collect(),
            x,
            e: vec![0.0; n],
            g: Array2::zeros((n, dof)),
            roles,
            active: 0..n,
            next_id: n as ImageId,
            filled: false,
        })
    }

    pub fn positions(&self) -> ArrayView2<'_, f64> {
        self.x.view()
    }

    pub fn energies(&self) -> &[f64] {
        &self.e
    }

    pub fn gradients(&self) -> ArrayView2<'_, f64> {
        self.g.view()
    }

    pub fn ids(&self) -> &[ImageId] {
        &self.ids
    }

    pub fn roles(&self) -> &[Role] {
        &self.roles
    }

    pub fn active(&self) -> Range<usize> {
        self.active.clone()
    }

    pub fn set_role(&mut self, index: usize, role: Role) -> Result<(), SaddleError> {
        let Some(slot) = self.roles.get_mut(index) else {
            return Err(SaddleError::Shape("role index is outside the chain".into()));
        };
        *slot = role;
        Ok(())
    }

    /// Evaluate every image. Block `i` owns image `i` for any thread count.
    /// The block width is [`EVAL_BLOCK`], not the worker count.
    /// Workers write their own block after the surface call returns, so
    /// the lock is not held across the evaluation.
    pub fn evaluate<S: Surface + ?Sized>(
        &mut self,
        surface: &S,
        pool: &EvalPool,
    ) -> Result<(), SaddleError> {
        let n = self.x.nrows();
        let dof = self.x.ncols();
        let blocks = n.div_ceil(EVAL_BLOCK);
        let positions = self.x.clone();
        let ids = self.ids.clone();
        let mut outs = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let start = b * EVAL_BLOCK;
            let rows = (start + EVAL_BLOCK).min(n) - start;
            outs.push(BlockOut {
                start,
                energy: vec![0.0; rows],
                gradient: Array2::zeros((rows, dof)),
                error: None,
            });
        }
        if pool.threads == 1 || blocks == 1 {
            for block in &mut outs {
                eval_block(
                    surface,
                    &positions,
                    &ids,
                    block.start,
                    &mut block.energy,
                    &mut block.gradient,
                )?;
            }
        } else {
            evaluate_blocks(surface, pool, &positions, &ids, &mut outs, dof);
        }
        let mut energies = vec![0.0; n];
        let mut gradients = Array2::zeros((n, dof));
        for block in outs {
            if let Some(err) = block.error {
                return Err(err);
            }
            for (offset, value) in block.energy.into_iter().enumerate() {
                energies[block.start + offset] = value;
            }
            for (offset, row) in block.gradient.rows().into_iter().enumerate() {
                gradients.row_mut(block.start + offset).assign(&row);
            }
        }
        self.e = energies;
        self.g = gradients;
        self.filled = true;
        Ok(())
    }

    pub fn filled(&self) -> bool {
        self.filled
    }

    pub fn coordinate_slice(&self) -> &[f64] {
        self.x.as_slice().expect("chain positions are contiguous")
    }

    pub fn interior_flat(&self) -> Array1<f64> {
        let n = self.x.nrows();
        if n < 2 {
            return Array1::zeros(0);
        }
        self.x
            .slice(ndarray::s![1..n - 1, ..])
            .iter()
            .copied()
            .collect()
    }

    pub fn all_flat(&self) -> Array1<f64> {
        self.x.iter().copied().collect()
    }

    pub fn write_interior(&mut self, flat: ndarray::ArrayView1<f64>) -> Result<(), SaddleError> {
        let n = self.x.nrows();
        let dof = self.x.ncols();
        if n < 2 || flat.len() != (n - 2) * dof {
            return Err(SaddleError::Shape(
                "interior vector does not match the chain".into(),
            ));
        }
        for i in 1..n - 1 {
            self.x
                .row_mut(i)
                .assign(&flat.slice(ndarray::s![(i - 1) * dof..i * dof]));
        }
        Ok(())
    }

    pub fn write_all(&mut self, flat: ndarray::ArrayView1<f64>) -> Result<(), SaddleError> {
        let n = self.x.nrows();
        let dof = self.x.ncols();
        if flat.len() != n * dof {
            return Err(SaddleError::Shape(
                "band vector does not match the chain".into(),
            ));
        }
        for i in 0..n {
            self.x
                .row_mut(i)
                .assign(&flat.slice(ndarray::s![i * dof..(i + 1) * dof]));
        }
        Ok(())
    }

    /// Replace the geometry. `ids` has one entry per row.
    pub fn replace(
        &mut self,
        x: Array2<f64>,
        ids: Vec<ImageId>,
        next_id: ImageId,
    ) -> Result<(), SaddleError> {
        if x.nrows() != ids.len() || x.nrows() == 0 || x.ncols() == 0 {
            return Err(SaddleError::Shape(
                "replacement chain rows do not match the image ids".into(),
            ));
        }
        if !x.iter().all(|value| value.is_finite()) {
            return Err(SaddleError::NonFinite("chain positions"));
        }
        let n = x.nrows();
        let dof = x.ncols();
        let mut roles = vec![Role::Image; n];
        if n >= 2 {
            roles[0] = Role::Pinned;
            roles[n - 1] = Role::Pinned;
        } else {
            roles[0] = Role::Walker;
        }
        self.ids = ids;
        self.x = x;
        self.e = vec![0.0; n];
        self.g = Array2::zeros((n, dof));
        self.roles = roles;
        self.active = 0..n;
        self.next_id = next_id;
        self.filled = false;
        Ok(())
    }

    pub fn next_id(&self) -> ImageId {
        self.next_id
    }

    pub fn set_ids(&mut self, ids: Vec<ImageId>, next_id: ImageId) -> Result<(), SaddleError> {
        if ids.len() != self.ids.len() {
            return Err(SaddleError::Shape(
                "image id count does not match the chain".into(),
            ));
        }
        self.ids = ids;
        self.next_id = next_id;
        Ok(())
    }
}

struct BlockOut {
    start: usize,
    energy: Vec<f64>,
    gradient: Array2<f64>,
    error: Option<SaddleError>,
}

fn evaluate_blocks<S: Surface + ?Sized>(
    surface: &S,
    pool: &EvalPool,
    positions: &Array2<f64>,
    ids: &[ImageId],
    outs: &mut [BlockOut],
    dof: usize,
) {
    let next = AtomicUsize::new(0);
    let failed = AtomicBool::new(false);
    let slots: Vec<Mutex<BlockOut>> = outs
        .iter_mut()
        .map(|block| Mutex::new(take_block(block, dof)))
        .collect();
    let workers = pool.threads.min(slots.len()).max(1);
    pool.pool.scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|_| {
                loop {
                    if failed.load(Ordering::Acquire) {
                        break;
                    }
                    let b = next.fetch_add(1, Ordering::Relaxed);
                    if b >= slots.len() {
                        break;
                    }
                    let (start, mut energy, mut gradient) = {
                        let mut guard = lock_block(&slots[b]);
                        (
                            guard.start,
                            std::mem::take(&mut guard.energy),
                            std::mem::replace(&mut guard.gradient, Array2::zeros((0, dof))),
                        )
                    };
                    let result =
                        eval_block(surface, positions, ids, start, &mut energy, &mut gradient);
                    let mut guard = lock_block(&slots[b]);
                    guard.energy = energy;
                    guard.gradient = gradient;
                    if let Err(err) = result {
                        guard.error = Some(err);
                        failed.store(true, Ordering::Release);
                    }
                }
            });
        }
    });
    for (slot, dest) in slots.into_iter().zip(outs.iter_mut()) {
        *dest = match slot.into_inner() {
            Ok(block) => block,
            Err(poison) => poison.into_inner(),
        };
    }
}

fn take_block(block: &mut BlockOut, dof: usize) -> BlockOut {
    BlockOut {
        start: block.start,
        energy: std::mem::take(&mut block.energy),
        gradient: std::mem::replace(&mut block.gradient, Array2::zeros((0, dof))),
        error: None,
    }
}

fn lock_block(slot: &Mutex<BlockOut>) -> std::sync::MutexGuard<'_, BlockOut> {
    match slot.lock() {
        Ok(guard) => guard,
        Err(poison) => poison.into_inner(),
    }
}

fn eval_block<S: Surface + ?Sized>(
    surface: &S,
    positions: &Array2<f64>,
    ids: &[ImageId],
    start: usize,
    energy: &mut [f64],
    gradient: &mut Array2<f64>,
) -> Result<(), SaddleError> {
    let rows = energy.len();
    let mut x = Array2::zeros((rows, positions.ncols()));
    for row in 0..rows {
        x.row_mut(row).assign(&positions.row(start + row));
    }
    surface.eval(
        &ids[start..start + rows],
        x.view(),
        energy,
        gradient.view_mut(),
    )
}

impl Mesh {
    pub fn apply(&self, chain: &mut Chain) -> Result<MeshEvent, SaddleError> {
        match self {
            Mesh::RemeshLinear => {
                crate::rtr::reparametrize_equal_arc(&mut chain.x, None);
                chain.filled = false;
                Ok(MeshEvent::Remeshed)
            }
            Mesh::RemeshHermite => {
                hermite_equal_arc(&mut chain.x)?;
                chain.filled = false;
                Ok(MeshEvent::Remeshed)
            }
            Mesh::Insert { after } => insert_midpoint(chain, *after),
            Mesh::WindowAuto { alpha } => window_auto(chain, *alpha),
            Mesh::WindowOffset { n } => window_offset(chain, *n),
        }
    }
}

fn insert_midpoint(chain: &mut Chain, after: usize) -> Result<MeshEvent, SaddleError> {
    if after + 1 >= chain.x.nrows() {
        return Err(SaddleError::Shape("insert needs a following image".into()));
    }
    let dof = chain.x.ncols();
    let mut mid = Array1::zeros(dof);
    for k in 0..dof {
        mid[k] = 0.5 * (chain.x[(after, k)] + chain.x[(after + 1, k)]);
    }
    let n = chain.x.nrows() + 1;
    let mut x = Array2::zeros((n, dof));
    for i in 0..=after {
        x.row_mut(i).assign(&chain.x.row(i));
    }
    x.row_mut(after + 1).assign(&mid);
    for i in (after + 1)..chain.x.nrows() {
        x.row_mut(i + 1).assign(&chain.x.row(i));
    }
    let id = chain.next_id;
    chain.next_id += 1;
    chain.ids.insert(after + 1, id);
    chain.roles.insert(after + 1, Role::Image);
    chain.e.insert(after + 1, 0.0);
    chain.x = x;
    chain.g = Array2::zeros((n, dof));
    chain.filled = false;
    chain.active = 0..n;
    Ok(MeshEvent::Inserted {
        id,
        index: after + 1,
    })
}

fn window_offset(chain: &mut Chain, n: usize) -> Result<MeshEvent, SaddleError> {
    if !chain.filled {
        return Err(SaddleError::Invalid(
            "a window needs cached energies".into(),
        ));
    }
    let peak = peak_index(&chain.e);
    let start = peak.saturating_sub(n);
    let end = (peak + n + 1).min(chain.x.nrows());
    mark_window(chain, start..end, peak);
    Ok(MeshEvent::Windowed {
        active: chain.active.clone(),
    })
}

fn window_auto(chain: &mut Chain, alpha: f64) -> Result<MeshEvent, SaddleError> {
    if !(alpha.is_finite() && (0.0..1.0).contains(&alpha)) {
        return Err(SaddleError::Invalid(
            "window alpha must lie in [0, 1)".into(),
        ));
    }
    if !chain.filled {
        return Err(SaddleError::Invalid(
            "a window needs cached energies".into(),
        ));
    }
    let peak = peak_index(&chain.e);
    let last = chain.e[chain.e.len() - 1];
    let ends = chain.e[0].max(last);
    let span = chain.e[peak] - ends;
    let floor = ends + alpha * span;
    let mut start = peak;
    while start > 0 && chain.e[start - 1] >= floor {
        start -= 1;
    }
    let mut end = peak + 1;
    while end < chain.e.len() && chain.e[end] >= floor {
        end += 1;
    }
    mark_window(chain, start..end, peak);
    Ok(MeshEvent::Windowed {
        active: chain.active.clone(),
    })
}

fn mark_window(chain: &mut Chain, active: Range<usize>, peak: usize) {
    for (i, role) in chain.roles.iter_mut().enumerate() {
        *role = if i + 1 == active.start || i == active.end.saturating_sub(1) {
            Role::Pinned
        } else if i == peak && active.contains(&i) && active.len() > 2 {
            Role::Climb
        } else if active.contains(&i) {
            Role::Image
        } else {
            Role::Pinned
        };
    }
    if active.len() >= 2 {
        chain.roles[active.start] = Role::Pinned;
        chain.roles[active.end - 1] = Role::Pinned;
    }
    chain.active = active;
}

fn peak_index(energies: &[f64]) -> usize {
    let mut best = 0;
    for (i, energy) in energies.iter().enumerate().skip(1) {
        if *energy > energies[best] {
            best = i;
        }
    }
    best
}

/// Equal-arc cubic Hermite. Endpoints stay put. A straight uniform
/// chain is unchanged.
fn hermite_equal_arc(x: &mut Array2<f64>) -> Result<(), SaddleError> {
    let n = x.nrows();
    if n < 3 {
        return Ok(());
    }
    let orig = x.clone();
    let mut s = vec![0.0; n];
    let mut seg = Vec::with_capacity(n - 1);
    for i in 1..n {
        let d = &orig.row(i) - &orig.row(i - 1);
        let length = d.dot(&d).sqrt();
        s[i] = s[i - 1] + length;
        seg.push(d);
    }
    let total = s[n - 1];
    if !(total.is_finite() && total > 1e-15) {
        return Ok(());
    }
    for j in 1..n - 1 {
        let target = total * (j as f64) / ((n - 1) as f64);
        let mut k = 1;
        while k + 1 < n && s[k] < target {
            k += 1;
        }
        let denom = s[k] - s[k - 1];
        let u = if denom > 0.0 {
            (target - s[k - 1]) / denom
        } else {
            0.0
        };
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = -2.0 * u3 + 3.0 * u2;
        let h11 = u3 - u2;
        let p0 = orig.row(k - 1).to_owned();
        let p1 = orig.row(k).to_owned();
        let m0 = seg[k - 1].clone();
        let m1 = seg[k - 1].clone();
        let p = &p0 * h00 + &m0 * h10 + &p1 * h01 + &m1 * h11;
        x.row_mut(j).assign(&p);
    }
    Ok(())
}

/// Persistent worker pool. One chain search keeps one pool.
pub struct EvalPool {
    threads: usize,
    pool: rayon::ThreadPool,
}

impl EvalPool {
    pub fn new(threads: usize) -> Result<Self, SaddleError> {
        if threads == 0 {
            return Err(SaddleError::Invalid(
                "chain evaluation needs at least one thread".into(),
            ));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|err| SaddleError::Invalid(format!("chain evaluation pool: {err}")))?;
        Ok(Self { threads, pool })
    }

    pub fn threads(&self) -> usize {
        self.threads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::Surface;
    use ndarray::{ArrayViewMut2, array};

    struct Quad;

    impl Surface for Quad {
        fn eval(
            &self,
            ids: &[ImageId],
            x: ArrayView2<f64>,
            e: &mut [f64],
            mut g: ArrayViewMut2<f64>,
        ) -> Result<(), SaddleError> {
            for (row, id) in ids.iter().enumerate() {
                let _ = id;
                let v = x[(row, 0)];
                e[row] = v * v;
                g[(row, 0)] = 2.0 * v;
            }
            Ok(())
        }
    }

    #[test]
    fn evaluate_matches_at_one_and_eight_threads() {
        let x = Array2::from_shape_fn((8, 1), |(i, _)| i as f64 - 3.0);
        let mut a = Chain::new(x.clone()).unwrap();
        let mut b = Chain::new(x).unwrap();
        a.evaluate(&Quad, &EvalPool::new(1).unwrap()).unwrap();
        b.evaluate(&Quad, &EvalPool::new(8).unwrap()).unwrap();
        for i in 0..8 {
            assert_eq!(a.energies()[i].to_le_bytes(), b.energies()[i].to_le_bytes());
            assert_eq!(
                a.gradients()[(i, 0)].to_le_bytes(),
                b.gradients()[(i, 0)].to_le_bytes()
            );
        }
    }

    #[test]
    fn a_uniform_line_is_unchanged_by_either_remesh() {
        let mut x = Array2::zeros((5, 1));
        for i in 0..5 {
            x[(i, 0)] = i as f64;
        }
        let original = x.clone();
        let mut chain = Chain::new(x).unwrap();
        Mesh::RemeshLinear.apply(&mut chain).unwrap();
        for i in 0..5 {
            assert_eq!(
                chain.positions()[(i, 0)].to_le_bytes(),
                original[(i, 0)].to_le_bytes()
            );
        }
        Mesh::RemeshHermite.apply(&mut chain).unwrap();
        for i in 0..5 {
            assert_eq!(
                chain.positions()[(i, 0)].to_le_bytes(),
                original[(i, 0)].to_le_bytes()
            );
        }
    }

    #[test]
    fn insert_keeps_the_old_ids_and_adds_one() {
        let x = array![[0.0], [2.0]];
        let mut chain = Chain::new(x).unwrap();
        let event = Mesh::Insert { after: 0 }.apply(&mut chain).unwrap();
        assert_eq!(event, MeshEvent::Inserted { id: 2, index: 1 });
        assert_eq!(chain.ids(), &[0, 2, 1]);
        assert_eq!(
            chain.positions()[(1, 0)].to_le_bytes(),
            (1.0_f64).to_le_bytes()
        );
        assert_eq!(chain.roles()[0], Role::Pinned);
        assert_eq!(chain.roles()[2], Role::Pinned);
    }

    #[test]
    fn offset_window_pins_its_ends_around_the_peak() {
        let x = Array2::zeros((5, 1));
        let mut chain = Chain::new(x).unwrap();
        chain.e = vec![0.0, 1.0, 3.0, 1.0, 0.0];
        chain.filled = true;
        Mesh::WindowOffset { n: 1 }.apply(&mut chain).unwrap();
        assert_eq!(chain.active(), 1..4);
        assert_eq!(chain.roles()[2], Role::Climb);
        assert_eq!(chain.roles()[1], Role::Pinned);
        assert_eq!(chain.roles()[3], Role::Pinned);
    }
}
