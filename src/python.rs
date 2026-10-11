//! Maturin / PyO3 entry for `rgsaddle`. Typed enums live here so
//! Python does not re-declare C ordinals.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::band::{BandSession, BandStatus, BandSurface, HostConfig};
use crate::error::SaddleError;
use crate::{
    AutoNeb, CellChart, EigenDevice, ExpandKind, ForceGate, HessUpdate, IrcKind, MinModeKind,
    RestrictedKind,
};
use ndarray::{Array1, Array2, ArrayView2};

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
    m.add(
        "EigenDevice",
        mint_int_enum(
            py,
            "EigenDevice",
            &[
                ("HOST", EigenDevice::Host.to_abi()),
                ("DLPK", EigenDevice::Dlpk.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "ExpandKind",
        mint_int_enum(
            py,
            "ExpandKind",
            &[
                ("LANCZOS", ExpandKind::Lanczos.to_abi()),
                ("GD", ExpandKind::Gd.to_abi()),
                ("JD0", ExpandKind::Jd0.to_abi()),
                ("JD0_ALT", ExpandKind::Jd0Alt.to_abi()),
                ("MJD0", ExpandKind::Mjd0.to_abi()),
                ("MJD0_ALT", ExpandKind::Mjd0Alt.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "CellChart",
        mint_int_enum(
            py,
            "CellChart",
            &[
                ("ENTRIES", CellChart::Entries.to_abi()),
                ("LOG_DEFORM", CellChart::LogDeform.to_abi()),
            ],
        )?,
    )?;
    m.add(
        "RestrictedKind",
        mint_int_enum(
            py,
            "RestrictedKind",
            &[
                ("TRUST_REGION", RestrictedKind::TrustRegion.to_abi()),
                (
                    "MAX_INTERNAL_STEP",
                    RestrictedKind::MaxInternalStep.to_abi(),
                ),
                (
                    "RESTRICTED_ATOMIC_STEP",
                    RestrictedKind::RestrictedAtomicStep.to_abi(),
                ),
            ],
        )?,
    )?;
    m.add_class::<Config>()?;
    Ok(())
}

fn py_invalid(err: SaddleError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

/// Opaque band options. The same string keys as `rgsaddle_config_set_*`.
#[pyclass]
struct Config {
    inner: HostConfig,
}

struct PySurface {
    callback: Py<PyAny>,
}

impl BandSurface for PySurface {
    fn eval(
        &self,
        positions: ArrayView2<f64>,
        energies: &mut Array1<f64>,
        gradients: &mut Array2<f64>,
    ) -> Result<(), SaddleError> {
        let flat: Vec<f64> = positions.iter().copied().collect();
        let nrows = positions.nrows();
        let dof = positions.ncols();
        let answered = Python::attach(|py| {
            let result = self.callback.bind(py).call1((flat, nrows, dof))?;
            let pair: (Vec<f64>, Vec<f64>) = result.extract()?;
            Ok(pair)
        });
        let (energy, gradient) = answered
            .map_err(|err: PyErr| Python::attach(|_py| SaddleError::Surface(err.to_string())))?;
        if energy.len() != energies.len() || gradient.len() != gradients.len() {
            return Err(SaddleError::Shape(
                "python surface returned the wrong shape".into(),
            ));
        }
        for (slot, value) in energies.iter_mut().zip(energy) {
            *slot = value;
        }
        for (slot, value) in gradients.iter_mut().zip(gradient) {
            *slot = value;
        }
        Ok(())
    }
}

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        Self {
            inner: HostConfig::default(),
        }
    }

    fn set_float(&mut self, key: &str, value: f64) -> PyResult<()> {
        crate::band::set_f64(&mut self.inner, key, value).map_err(py_invalid)
    }

    fn set_int(&mut self, key: &str, value: i64) -> PyResult<()> {
        crate::band::set_i64(&mut self.inner, key, value).map_err(py_invalid)
    }

    /// `callback(flat_positions, n_images, dof) -> (energies, gradients)`.
    /// `threads` above 1 calls it from the pool. The return dict carries
    /// status, max_force, ci_index, iteration, calls, n_images, positions.
    fn run_band<'py>(
        &mut self,
        py: Python<'py>,
        n_atoms: usize,
        positions: Vec<f64>,
        callback: Py<PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if n_atoms == 0 || !positions.len().is_multiple_of(3 * n_atoms) {
            return Err(py_invalid(SaddleError::Shape(
                "positions must be n_images * 3 * n_atoms".into(),
            )));
        }
        let n_images = positions.len() / (3 * n_atoms);
        let initial = Array2::from_shape_vec((n_images, 3 * n_atoms), positions)
            .map_err(|err| py_invalid(SaddleError::Shape(err.to_string())))?;
        let surface = PySurface { callback };
        let auto = self.inner.band.autoneb.is_some();
        let outcome = py.detach(|| {
            if auto {
                let mut search = AutoNeb::new(self.inner.band.clone(), initial)?;
                let report = search.run(&surface)?;
                Ok::<_, SaddleError>((
                    report.status == BandStatus::Converged,
                    report.max_force,
                    report.ci_index.map(|index| index as i64).unwrap_or(-1),
                    report.iteration as i64,
                    report.surface_rows as i64,
                    search.positions().iter().copied().collect::<Vec<_>>(),
                ))
            } else {
                let mut session = BandSession::new(self.inner.band.clone(), initial)?;
                let mut report = session.step(&surface)?;
                let mut calls = report.surface_rows;
                while report.status == BandStatus::Running
                    && report.iteration < self.inner.max_steps
                {
                    report = session.step(&surface)?;
                    calls += report.surface_rows;
                }
                Ok((
                    report.status == BandStatus::Converged,
                    report.max_force,
                    report.ci_index.map(|index| index as i64).unwrap_or(-1),
                    report.iteration as i64,
                    calls as i64,
                    session.positions().iter().copied().collect::<Vec<_>>(),
                ))
            }
        });
        let (converged, max_force, ci_index, iteration, calls, final_positions) =
            outcome.map_err(py_invalid)?;
        let dict = PyDict::new(py);
        dict.set_item("converged", converged)?;
        dict.set_item("max_force", max_force)?;
        dict.set_item("ci_index", ci_index)?;
        dict.set_item("iteration", iteration)?;
        dict.set_item("calls", calls)?;
        dict.set_item("n_images", final_positions.len() / (3 * n_atoms))?;
        dict.set_item("positions", final_positions)?;
        Ok(dict)
    }
}

#[cfg(test)]
mod tests {
    use super::Config;
    use pyo3::Python;

    #[test]
    fn config_rejects_an_unknown_key_and_turns_autoneb_on() {
        Python::initialize();
        let mut cfg = Config::new();
        assert!(cfg.set_int("nope", 1).is_err());
        cfg.set_int("autoneb", 1).unwrap();
        cfg.set_int("n_max", 5).unwrap();
        cfg.set_float("force_tol", 1e-2).unwrap();
        assert!(cfg.set_int("spring", 1).is_err());
    }
}
