Step a NEB band
===============

``BandSession::step`` assembles the tangent, spring and projection forces and advances one host iteration. The fast inertial relaxation engine (``FIRE``) and limited-memory Broyden-Fletcher-Goldfarb-Shanno (``L-BFGS``) methods preserve the endpoints. ``L-BFGS`` computes its direction from the assembled force; summed image energy is not a descent criterion for that force.

Call ``reset`` when the surface changes. The session retains evaluated endpoints and the accepted band while the surface remains valid. A ``Cell`` supplies minimum-image differences for periodic configurations.

The projected force of every image with two or more atoms loses its mean Cartesian part, as in eOn when every atom is free. A host with fixed atoms sets ``BandConfig::remove_translation`` to ``false``, or ``RGSADDLE_BAND_KEEP_TRANSLATION`` in the C band config flags, to keep the mean. An image with one atom keeps its force.

The climbing image is the highest interior image, and it climbs only while its energy exceeds both fixed endpoints. A monotonic band keeps the spring force on every image and reports no climbing image.

``set_rtr(Some(RtrConfig::default()))`` selects the projected trust-region band method. Its model uses the complete displacement after equal-arc redistribution and the radius cap. The projected method is a heuristic for the nonconservative band force; the report does not certify an energy minimum.

``set_force_gate`` chooses the force norm. ``set_highs(true)`` enables rgmin's feasible-step control when the library has the ``highs`` feature. The default ``FIRE`` and ``L-BFGS`` dispatch is unchanged by compiling that feature.
