//! Maturin / PyO3 entry for `rgsaddle`. Typed enums live here so
//! Python does not re-declare C ordinals.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::{ForceGate, HessUpdate, IrcKind, MinModeKind};

fn mint_int_enum<'py>(
    py: Python<'py>,
    name: &str,
    members: &[(&str, i32)],
) -> PyResult<Bound<'py, PyAny>> {
    let dict = PyDict::new(py);
    for (key, val) in members {
        dict.set_item(*key, *val)?;
    }
    py.import("enum")?.getattr("IntEnum")?.call1((name, dict))
}

/// pyo3 module. Exposed to Python as `rgsaddle`.
#[pymodule]
fn rgsaddle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "ForceGate",
        mint_int_enum(
            py,
            "ForceGate",
            &[
                ("L2_NORM", ForceGate::L2Norm.to_abi()),
                ("LINF_NORM", ForceGate::LinfNorm.to_abi()),
                ("MAX_FORCE_ON_ATOM", ForceGate::MaxForceOnAtom.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "HessUpdate",
        mint_int_enum(
            py,
            "HessUpdate",
            &[
                ("BFGS", HessUpdate::Bfgs.to_abi()),
                ("TS_BFGS", HessUpdate::TsBfgs.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "IrcKind",
        mint_int_enum(
            py,
            "IrcKind",
            &[
                ("GS2", IrcKind::Gs2.to_abi()),
                ("MOROKUMA", IrcKind::Morokuma.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "MinModeKind",
        mint_int_enum(
            py,
            "MinModeKind",
            &[
                ("DIMER", MinModeKind::Dimer.to_abi()),
                ("LANCZOS", MinModeKind::Lanczos.to_abi()),
            ],
        )?,
    )?;
    Ok(())
}
