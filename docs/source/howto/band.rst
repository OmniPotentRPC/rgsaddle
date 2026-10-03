Step a NEB band
===============

``BandSession::step`` assembles the tangent, spring and projection forces and advances one host iteration. The fast inertial relaxation engine (``FIRE``) and limited-memory Broyden-Fletcher-Goldfarb-Shanno (``L-BFGS``) methods preserve the endpoints. ``L-BFGS`` computes its direction from the assembled force; summed image energy is not a descent criterion for that force.

Call ``reset`` when the surface changes. The session retains evaluated endpoints and the accepted band while the surface remains valid. A ``Cell`` supplies minimum-image differences for periodic configurations.

``set_rtr(Some(RtrConfig::default()))`` selects the projected trust-region band method. Its model uses the complete displacement after equal-arc redistribution and the radius cap. The projected method is a heuristic for the nonconservative band force; the report does not certify an energy minimum.

``set_force_gate`` chooses the force norm. ``set_highs(true)`` enables rgmin's feasible-step control when the library has the ``highs`` feature. The default ``FIRE`` and ``L-BFGS`` dispatch is unchanged by compiling that feature.
