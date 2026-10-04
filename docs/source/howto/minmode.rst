Follow a minimum mode
=====================

``MinModeSession`` estimates a low-curvature direction by dimer rotation or Lanczos, then advances on a force switched by the sign of the curvature along it. Below zero the mode component of the force is reversed and the rest kept, ``F - 2 (F.tau) tau``. At zero or above only the reversed mode component remains, ``-(F.tau) tau``, so the walker leaves a minimum basin along the mode instead of relaxing back towards the minimum (Henkelman and Jonsson 1999, Olsen et al. 2004). Both mode kinds use the switch. The session drops its optimizer memory when the branch changes. ``minmode::effective_gradient`` is the switch itself. ``FiniteDifference`` selects forward or central Hessian actions. A surface can supply analytic actions and excluded directions through ``PointSurface``.

The reported force norm must satisfy the selected ``ForceGate``, and the estimated curvature must be negative. If a translation reaches a stationary point, the session measures curvature at that accepted position before reporting convergence. This directional test does not certify the full Hessian index.

``set_kappa`` selects the basin constraint. ``set_highs`` selects the optional feasible-step control. Call ``reset`` at a surface change and use ``set_force_gate`` to select the physical stopping norm. Hosts that require exactly one imaginary mode must perform their full index check.

Keep one session across a host search
-------------------------------------

A host that climbs with its own optimizer keeps one session for the whole search. At each new centre it calls ``set_position`` (``rgsaddle_minmode_set_position``) with the gradient it already has, so the session does not evaluate the centre, then ``estimate_mode`` (``rgsaddle_minmode_estimate``). The previous mode seeds the next rotation, so an unchanged mode costs one evaluation. ``set_mode`` (``rgsaddle_minmode_set_mode``) replaces the seed. All three need ABI minor 5.

Two ``MinModeConfig`` fields set the cost of an estimate:

.. table::

    +------------------------+----------------------------------------------------------------------------------------------------------------+-------------+
    | Field                  | Effect                                                                                                         | Default     |
    +========================+================================================================================================================+=============+
    | ``difference``         | ``Forward``: one gradient per Hessian action, error O(dr). ``Central``: two, error O(dr^2)                     | ``Forward`` |
    +------------------------+----------------------------------------------------------------------------------------------------------------+-------------+
    | ``rotation_angle_tol`` | The dimer stops rotating when an iteration's optimal angle falls under this many radians; 0 turns the test off | 0           |
    +------------------------+----------------------------------------------------------------------------------------------------------------+-------------+

From C both fields sit in ``rgsaddle_minmode_config_t`` and are read only when the config's ``version.minor`` is 5 or more.
