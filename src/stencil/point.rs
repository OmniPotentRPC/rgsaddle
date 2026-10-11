//! Point methods on the same surface as the band.
//!
//! Dimer and Lanczos go through [`MinModeSession`](crate::minmode::MinModeSession).
//! ART nouveau is the `feat/art-nouveau` session and is refused until
//! that branch is on main. The initial mode is the first coordinate
//! axis. A host that wants another seed uses the min-mode session
//! directly.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

use crate::error::SaddleError;
use crate::minmode::{MinModeConfig, MinModeKind, MinModeSession, MinModeStatus, PointSurface};

use super::{ImageId, Phase, PointMethod, StencilReport, StencilSurface};

pub struct PointRun {
    session: MinModeSession,
    id: ImageId,
    total_rows: usize,
    kind: PointMethod,
    matrix: Array2<f64>,
}

impl PointRun {
    pub fn new(
        config: MinModeConfig,
        kind: PointMethod,
        x0: Array1<f64>,
    ) -> Result<Self, SaddleError> {
        if kind == PointMethod::Art {
            return Err(SaddleError::Invalid(
                "ART nouveau lands with rgsaddle feat/art-nouveau".into(),
            ));
        }
        if x0.len() < 3 || !x0.len().is_multiple_of(3) {
            return Err(SaddleError::Shape(format!(
                "a point method needs 3N coordinates; got {}",
                x0.len()
            )));
        }
        let mut mode = Array1::zeros(x0.len());
        mode[0] = 1.0;
        let mut config = config;
        config.kind = match kind {
            PointMethod::Dimer => MinModeKind::Dimer,
            PointMethod::Lanczos => MinModeKind::Lanczos,
            PointMethod::Art => MinModeKind::Dimer,
        };
        let session = MinModeSession::new(config, x0.clone(), mode)?;
        let mut matrix = Array2::zeros((1, x0.len()));
        matrix.row_mut(0).assign(&x0);
        Ok(Self {
            session,
            id: 0,
            total_rows: 0,
            kind,
            matrix,
        })
    }

    pub fn matrix(&self) -> ArrayView2<'_, f64> {
        self.matrix.view()
    }

    pub fn positions(&self) -> ArrayView1<'_, f64> {
        self.session.position()
    }

    pub fn mode(&self) -> ArrayView1<'_, f64> {
        self.session.mode()
    }

    pub fn curvature(&self) -> f64 {
        self.session.curvature()
    }

    pub fn step<S: StencilSurface + ?Sized>(
        &mut self,
        surface: &S,
    ) -> Result<StencilReport, SaddleError> {
        let id = self.id;
        let adapter = PointAdapter { surface, id };
        let phase = match self.kind {
            PointMethod::Dimer | PointMethod::Lanczos | PointMethod::Art => Phase::MinMode,
        };
        let report = self.session.step(&adapter)?;
        self.total_rows += report.evaluations;
        self.matrix.row_mut(0).assign(&self.session.position());
        let status = match report.status {
            MinModeStatus::Converged => crate::band::BandStatus::Converged,
            MinModeStatus::Running => crate::band::BandStatus::Running,
        };
        Ok(StencilReport {
            phase,
            status,
            max_force: report.max_force,
            ci_index: None,
            iteration: report.iteration,
            surface_rows: report.evaluations,
            total_rows: self.total_rows,
            zoom_window: None,
            curvature: Some(report.curvature),
            force_gate: "max-abs-component",
        })
    }
}

struct PointAdapter<'a, S: ?Sized> {
    surface: &'a S,
    id: ImageId,
}

impl<S: StencilSurface + ?Sized> PointSurface for PointAdapter<'_, S> {
    fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let mut positions = Array2::zeros((1, x.len()));
        positions.row_mut(0).assign(&x);
        let mut energies = Array1::zeros(1);
        let mut gradients = Array2::zeros((1, x.len()));
        self.surface
            .eval(&[self.id], positions.view(), &mut energies, &mut gradients)?;
        Ok((energies[0], gradients.row(0).to_owned()))
    }
}
