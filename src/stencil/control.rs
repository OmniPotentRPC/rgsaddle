//! Controllers for a point stencil.
//!
//! The zoom controller stays on `feat/band-methods-zoom`. This module
//! owns the fixed loop and the ART push-then-climb switch.
//! Malek and Mousseau, Phys. Rev. E 62, 7723 (2000),
//! doi:10.1103/PhysRevE.62.7723.

/// What an image is allowed to do. A walker is one centre.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Pinned,
    Image,
    Climb,
    Walker,
}

/// Whether the last step may stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StencilStatus {
    Running,
    Converged,
}

/// The fields a controller may read. Positions stay on the chain.
#[derive(Clone, Copy, Debug)]
pub struct StencilView<'a> {
    pub roles: &'a [Role],
    pub curvature: f64,
    pub max_force: f64,
}

/// Outcome of the previous step, as the controller sees it.
#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub status: StencilStatus,
    pub max_force: f64,
    pub curvature: f64,
}

/// What the builder should do next.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Step,
    Push,
    Climb,
    Stop(StencilStatus),
}

/// One decision per step. The builder stores the spec, not a trait object.
pub trait Controller {
    fn decide(&mut self, view: &StencilView<'_>, last: &Report) -> Action;
}

/// Keep stepping until the report says the gate is met.
#[derive(Clone, Copy, Debug, Default)]
pub struct Fixed;

impl Controller for Fixed {
    fn decide(&mut self, _view: &StencilView<'_>, last: &Report) -> Action {
        if last.status == StencilStatus::Converged {
            Action::Stop(StencilStatus::Converged)
        } else {
            Action::Step
        }
    }
}

/// Ascend along the seed until the curvature drops under the threshold.
#[derive(Clone, Debug)]
pub struct ArtControl {
    threshold: f64,
    push_left: usize,
    climbing: bool,
}

impl ArtControl {
    pub fn new(threshold: f64, push_steps: usize) -> Self {
        Self {
            threshold,
            push_left: push_steps,
            climbing: false,
        }
    }

    pub fn climbing(&self) -> bool {
        self.climbing
    }

    pub fn push_left(&self) -> usize {
        self.push_left
    }
}

impl Controller for ArtControl {
    fn decide(&mut self, view: &StencilView<'_>, last: &Report) -> Action {
        if last.status == StencilStatus::Converged {
            return Action::Stop(StencilStatus::Converged);
        }
        if self.climbing {
            return Action::Climb;
        }
        let open = view.curvature.is_finite() && view.curvature < self.threshold;
        if open || self.push_left == 0 {
            self.climbing = true;
            return Action::Climb;
        }
        self.push_left -= 1;
        Action::Push
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn art_pushes_until_the_curvature_opens_or_the_budget_ends() {
        let mut art = ArtControl::new(0.0, 2);
        let roles = [Role::Walker];
        let running = Report {
            status: StencilStatus::Running,
            max_force: 1.0,
            curvature: 1.0,
        };
        let view = StencilView {
            roles: &roles,
            curvature: 1.0,
            max_force: 1.0,
        };
        assert_eq!(art.decide(&view, &running), Action::Push);
        assert_eq!(art.decide(&view, &running), Action::Push);
        assert_eq!(art.decide(&view, &running), Action::Climb);
        assert!(art.climbing());
        assert_eq!(art.decide(&view, &running), Action::Climb);
    }

    #[test]
    fn a_negative_curvature_climbs_at_once() {
        let mut art = ArtControl::new(0.0, 5);
        let roles = [Role::Walker];
        let view = StencilView {
            roles: &roles,
            curvature: -0.2,
            max_force: 1.0,
        };
        let last = Report {
            status: StencilStatus::Running,
            max_force: 1.0,
            curvature: -0.2,
        };
        assert_eq!(art.decide(&view, &last), Action::Climb);
    }
}
