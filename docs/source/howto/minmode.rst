Follow a minimum mode
=====================

``MinModeSession`` estimates a low-curvature direction by dimer rotation or Lanczos, then advances on the reflected force. ``FiniteDifference`` selects forward or central Hessian actions. A surface can supply analytic actions and excluded directions through ``PointSurface``.

The reported force norm must satisfy the selected ``ForceGate``, and the estimated curvature must be negative. If a translation reaches a stationary point, the session measures curvature at that accepted position before reporting convergence. This directional test does not certify the full Hessian index.

``set_kappa`` selects the basin constraint. ``set_highs`` selects the optional feasible-step control. Call ``reset`` at a surface change and use ``set_force_gate`` to select the physical stopping norm. Hosts that require exactly one imaginary mode must perform their full index check.
