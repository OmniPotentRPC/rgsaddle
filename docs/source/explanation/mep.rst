

Why IRC is a moving sphere on MwRigid
-------------------------------------

The intrinsic reaction coordinate is Fukui's ODE on
mass-weighted configuration space
(`10.1021/j100717a029 <https://doi.org/10.1021/j100717a029>`_,
`10.1021/ar00072a001 <https://doi.org/10.1021/ar00072a001>`_).
In :math:`x_{\mathrm{mw}} = \sqrt{m}\, x`,



.. math::

    \frac{\mathrm{d}x_{\mathrm{mw}}}{\mathrm{d}s}
    = -\frac{g_{\mathrm{mw}}}{\|g_{\mathrm{mw}}\|}.

That space is rgmin ``MwRigid`` (Page--McIver / Eckart,
`10.1063/1.454172 <https://doi.org/10.1063/1.454172>`_).

Gonzalez--Schlegel / Sella discretize the ODE as implicit Euler:
minimize energy on the sphere of radius ``dx`` about the last
accepted point
(`10.1063/1.456010 <https://doi.org/10.1063/1.456010>`_,
`10.1021/acs.jctc.2c00395 <https://doi.org/10.1021/acs.jctc.2c00395>`_).
ORCA's production IRC is Morokuma predictor-corrector on the
same manifold
(`10.1063/1.434152 <https://doi.org/10.1063/1.434152>`_),
not GS2.

The kick needs one extremal Hessian pair. That is
``lowest_eigenpair`` / Lanczos on ``H v``, not a full ELPA spectrum.
ELPA and SLATE pay when a dense ``H`` already exists and
``n >= 512``.
