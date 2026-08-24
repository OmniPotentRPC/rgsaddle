//! Maturin / PyO3 entry for `rgsaddle`. Typed enums live here so
//! Python does not re-declare C ordinals.

use pyo3::prelude::*;

use crate::ForceGate;

/// pyo3 module. Exposed to Python as `rgsaddle`.
#[pymodule]
fn rgsaddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<ForceGate>()?;
    Ok(())
}
