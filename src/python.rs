//! Maturin / PyO3 entry for `rgsaddle`. Typed enums live here so
//! Python does not re-declare C ordinals.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ndarray::Array1;

use crate::art::ArtSession;
use crate::error::SaddleError;
use crate::minmode::{MinModeSession, PointSurface};
use crate::search::SearchConfig as RustSearch;
use crate::{
    CellChart, EigenDevice, ExpandKind, ForceGate, HessUpdate, IrcKind, MinModeKind, RestrictedKind,
};

fn py_error(err: SaddleError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

struct PySurface {
    callback: Py<PyAny>,
}

impl PointSurface for PySurface {
    fn eval(&self, x: ndarray::ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        Python::attach(|py| {
            let values: Vec<f64> = x.iter().copied().collect();
            let result = self
                .callback
                .call1(py, (values,))
                .map_err(|err| SaddleError::Surface(err.to_string()))?;
            let (energy, gradient): (f64, Vec<f64>) = result
                .extract(py)
                .map_err(|err| SaddleError::Surface(err.to_string()))?;
            if gradient.len() != x.len() {
                return Err(SaddleError::Shape("surface gradient length changed".into()));
            }
            Ok((energy, Array1::from(gradient)))
        })
    }
}

/// Saddle-search configuration. Dimer, Lanczos, and ART nouveau are setters.
#[pyclass]
struct SearchConfig {
    inner: RustSearch,
}

#[pymethods]
impl SearchConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: RustSearch::new(),
        }
    }

    fn set_kind(&mut self, kind: i32) -> PyResult<()> {
        self.inner.set_kind(kind).map_err(py_error)
    }
    fn set_method(&mut self, method: i32) -> PyResult<()> {
        self.inner.set_method(method).map_err(py_error)
    }
    fn set_dr(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_dr(value).map_err(py_error)
    }
    fn set_difference(&mut self, value: i32) -> PyResult<()> {
        self.inner.set_difference(value).map_err(py_error)
    }
    fn set_rotation_tol(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_rotation_tol(value).map_err(py_error)
    }
    fn set_force_tol(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_force_tol(value).map_err(py_error)
    }
    fn set_max_move(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_max_move(value).map_err(py_error)
    }
    fn set_krylov_dim(&mut self, value: i64) -> PyResult<()> {
        self.inner.set_krylov_dim(value).map_err(py_error)
    }
    fn set_eigen_threshold(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_eigen_threshold(value).map_err(py_error)
    }
    fn set_push_step(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_push_step(value).map_err(py_error)
    }
    fn set_activation_distance(&mut self, value: f64) -> PyResult<()> {
        self.inner.set_activation_distance(value).map_err(py_error)
    }
    fn set_perp_steps_push(&mut self, value: i64) -> PyResult<()> {
        self.inner.set_perp_steps_push(value).map_err(py_error)
    }
    fn set_displacement_seed(&mut self, seed: u64) -> PyResult<()> {
        self.inner.set_displacement_seed(seed).map_err(py_error)
    }
    fn set_displacement_amplitude(&mut self, value: f64) -> PyResult<()> {
        self.inner
            .set_displacement_amplitude(value)
            .map_err(py_error)
    }
}

/// One step of a min-mode or ART search.
#[pyclass]
struct SearchReport {
    #[pyo3(get)]
    status: i32,
    #[pyo3(get)]
    max_force: f64,
    #[pyo3(get)]
    curvature: f64,
    #[pyo3(get)]
    pushes: i64,
    #[pyo3(get)]
    climbs: i64,
}

enum SearchBody {
    Min(Box<MinModeSession>),
    Art(Box<ArtSession>),
}

/// The saddle-search session. ART is selected by options on [`SearchConfig`].
#[pyclass]
struct MinMode {
    body: SearchBody,
}

#[pymethods]
impl MinMode {
    #[new]
    fn new(config: &SearchConfig, position: Vec<f64>, mode: Option<Vec<f64>>) -> PyResult<Self> {
        let position = Array1::from(position);
        if config.inner.art_enabled() {
            let direction = mode.map(Array1::from);
            let displacement = config.inner.displacement(direction).map_err(py_error)?;
            let art = config.inner.art_config().map_err(py_error)?;
            let session = ArtSession::new(art, position, displacement).map_err(py_error)?;
            return Ok(Self {
                body: SearchBody::Art(Box::new(session)),
            });
        }
        let Some(mode) = mode else {
            return Err(PyValueError::new_err(
                "a min-mode search needs an initial mode",
            ));
        };
        let session =
            MinModeSession::new(config.inner.mode().clone(), position, Array1::from(mode))
                .map_err(py_error)?;
        Ok(Self {
            body: SearchBody::Min(Box::new(session)),
        })
    }

    fn step(&mut self, surface: Py<PyAny>) -> PyResult<SearchReport> {
        let surface = PySurface { callback: surface };
        match &mut self.body {
            SearchBody::Art(session) => {
                let report = session.step(&surface).map_err(py_error)?;
                Ok(SearchReport {
                    status: match report.status {
                        crate::ArtStatus::Converged => 1,
                        crate::ArtStatus::Exhausted => 2,
                        crate::ArtStatus::Pushing | crate::ArtStatus::Climbing => 0,
                    },
                    max_force: report.max_force,
                    curvature: report.curvature,
                    pushes: report.pushes as i64,
                    climbs: report.climbs as i64,
                })
            }
            SearchBody::Min(session) => {
                let report = session.step(&surface).map_err(py_error)?;
                Ok(SearchReport {
                    status: i32::from(report.status == crate::MinModeStatus::Converged),
                    max_force: report.max_force,
                    curvature: report.curvature,
                    pushes: 0,
                    climbs: report.rotations as i64,
                })
            }
        }
    }

    fn position(&self) -> Vec<f64> {
        match &self.body {
            SearchBody::Art(session) => session.position().iter().copied().collect(),
            SearchBody::Min(session) => session.position().iter().copied().collect(),
        }
    }
}

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
    m.add_class::<SearchConfig>()?;
    m.add_class::<MinMode>()?;
    m.add_class::<SearchReport>()?;
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{MinMode, SearchConfig};
    use pyo3::types::PyAnyMethods;

    #[test]
    fn art_options_step_the_double_well() {
        pyo3::Python::initialize();
        pyo3::Python::attach(|py| {
            let mut config = SearchConfig::new();
            config.set_eigen_threshold(-0.01).unwrap();
            config.set_push_step(0.3).unwrap();
            config.set_activation_distance(0.0).unwrap();
            config.set_perp_steps_push(0).unwrap();
            config.set_displacement_amplitude(0.2).unwrap();
            config.set_max_move(0.3).unwrap();
            assert!(config.set_push_step(0.0).is_err());
            let mut session =
                MinMode::new(&config, vec![1.0, 0.0, 0.0], Some(vec![-1.0, 0.0, 0.0])).unwrap();
            let module = pyo3::types::PyModule::from_code(
                py,
                c"def surface(x):
    a, b, c = x
    e = (a * a - 1.0) ** 2 + 2.0 * b * b + 2.0 * c * c
    return e, [4.0 * a * (a * a - 1.0), 4.0 * b, 4.0 * c]
",
                c"art_surface.py",
                c"art_surface",
            )
            .unwrap();
            let surface = module.getattr("surface").unwrap();
            let report = session.step(surface.unbind()).unwrap();
            let position = session.position();
            assert!(position[0] < 1.0, "{position:?}");
            assert!(report.status == 0 || report.status == 1);
        });
    }
}
