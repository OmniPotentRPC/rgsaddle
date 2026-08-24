//! Sella TrustRegion: `cons(s) = ||s||`, target `delta`.
//!
//! The restricted-step solver lives in rgmin. This module is the
//! rgsaddle waist. Distinct from IRCTrustRegion
//! (`||(s + d1) * sqrt(m)||`); that constraint is rgmin-4e25.

pub use rgmin::{RestrictedStep, TrustRegion};
