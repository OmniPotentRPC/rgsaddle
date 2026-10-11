//! Image indices for one AutoNEB surface call.
//!
//! The band search inserts on the calling thread and evaluates on the
//! pool. Each worker publishes the full-band indices of its block
//! before the callback, and the id book maps those indices to the
//! stable ids. A search that never binds a book keeps the previous
//! row numbering.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::error::SaddleError;

use super::ImageId;

thread_local! {
    static INDICES: RefCell<Option<Vec<usize>>> = const { RefCell::new(None) };
    static BOOK: RefCell<Option<Arc<IdBook>>> = const { RefCell::new(None) };
}

/// Stable ids for the band currently being stepped.
pub struct IdBook {
    pub ids: Mutex<Vec<ImageId>>,
    pub next: AtomicU64,
}

impl IdBook {
    pub fn new(n: usize) -> Arc<Self> {
        Arc::new(Self {
            ids: Mutex::new((0..n as ImageId).collect()),
            next: AtomicU64::new(n as u64),
        })
    }
}

pub(crate) fn bind_book(book: Option<Arc<IdBook>>) {
    BOOK.with(|slot| *slot.borrow_mut() = book);
}

pub(crate) struct BookGuard;

impl Drop for BookGuard {
    fn drop(&mut self) {
        bind_book(None);
        clear_indices();
    }
}

pub(crate) fn bind_for_step(book: Arc<IdBook>) -> BookGuard {
    bind_book(Some(book));
    BookGuard
}

pub(crate) fn set_indices(indices: &[usize]) {
    INDICES.with(|slot| *slot.borrow_mut() = Some(indices.to_vec()));
}

pub(crate) fn clear_indices() {
    INDICES.with(|slot| *slot.borrow_mut() = None);
}

pub(crate) struct IndexGuard;

impl Drop for IndexGuard {
    fn drop(&mut self) {
        clear_indices();
    }
}

pub(crate) fn publish_indices(
    indices: Option<&[usize]>,
    range: std::ops::Range<usize>,
) -> IndexGuard {
    if let Some(all) = indices {
        set_indices(&all[range]);
    } else {
        clear_indices();
    }
    IndexGuard
}

/// Ids for a callback of `nrows` rows.
///
/// Without a bound book the rows are numbered from zero, which is the
/// numbering a one-shot band surface already used. With a book, the
/// published full-band indices select stable ids.
pub fn ids_for_callback(nrows: usize) -> Result<Vec<ImageId>, SaddleError> {
    let book = BOOK.with(|slot| slot.borrow().clone());
    let Some(book) = book else {
        return Ok((0..nrows as ImageId).collect());
    };
    let indices = INDICES.with(|slot| slot.borrow().clone());
    let Some(indices) = indices else {
        return Err(SaddleError::Invalid(
            "AutoNEB evaluation has no image index".into(),
        ));
    };
    if indices.len() != nrows {
        return Err(SaddleError::Shape(
            "AutoNEB image index count does not match the callback rows".into(),
        ));
    }
    let ids = book.ids.lock().unwrap_or_else(|err| err.into_inner());
    let mut out = Vec::with_capacity(nrows);
    for index in indices {
        let Some(id) = ids.get(index).copied() else {
            return Err(SaddleError::Shape(
                "AutoNEB image index is outside the band".into(),
            ));
        };
        out.push(id);
    }
    Ok(out)
}

/// Record one inserted image. `after` is the image the new row follows.
pub(crate) fn note_image_insert(after: usize) {
    let book = BOOK.with(|slot| slot.borrow().clone());
    let Some(book) = book else {
        return;
    };
    let id = book.next.fetch_add(1, Ordering::Relaxed);
    let mut ids = book.ids.lock().unwrap_or_else(|err| err.into_inner());
    let at = after + 1;
    if at <= ids.len() {
        ids.insert(at, id);
    }
}
