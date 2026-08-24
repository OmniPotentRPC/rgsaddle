//! Maturin / PyO3 entry for `rgsaddle`. Typed enums live here so
//! Python does not re-declare C ordinals.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ForceGate;

/// Build `enum.IntEnum` from the Rust discriminants.
fn force_gate_int_enum<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let members = PyDict::new(py);
    for (name, gate) in [
        ("L2_NORM", ForceGate::L2Norm),
        ("LINF_NORM", ForceGate::LinfNorm),
        ("MAX_FORCE_ON_ATOM", ForceGate::MaxForceOnAtom),
    ] {
        members.set_item(name, gate.to_abi())?;
    }
    py.import("enum")?
        .getattr("IntEnum")?
        .call1(("ForceGate", members))
}

/// pyo3 module. Exposed to Python as `rgsaddle`.
#[pymodule]
fn rgsaddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("ForceGate", force_gate_int_enum(m.py())?)?;
    Ok(())
}
