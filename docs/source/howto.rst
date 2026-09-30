How-to
======

Step a band from C
------------------

Enable ``capi``. The header is ``rgsaddle.h``. ABI major 1. Every wire
struct opens with ``rgsaddle_version_t``. The host:

1. creates a session
2. calls ``step`` until the report says converged, or until host policy stops
3. calls ``reset`` at a surface-epoch boundary
4. frees

There is no run-to-completion C entry point. Units are the caller's.
Positions are unwrapped Cartesian, image-major, stride ``3 * n_atoms``.

Pick tangents, springs, projections
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Kind
     - Variants
   * - ``TangentKind``
     - ``Simple`` (Mills–Jónsson–Schenter), ``Improved`` (Henkelman–Jónsson)
   * - ``SpringKind``
     - ``Uniform { k }``, energy-weighted, Onsager–Machlup
   * - ``ProjectionKind``
     - plain elastic band, ``Neb``, ``Dneb``

Defaults: improved tangent, uniform spring ``k = 5``, NEB projection,
climbing image with trigger factor 0.5.

Minimum-mode instead of a band
------------------------------

``MinModeSession`` refreshes the lowest curvature mode (dimer rotation
or Lanczos on finite-difference Hessian actions), inverts the force
along it, and takes one solver step. Same host loop. ``PointSurface``
evals a single geometry.

Index-1 Newton
--------------

``Index1Session`` takes the Nichols step on a dense Hessian and
updates it with Powell or Bofill. Pass an initial Hessian, or leave
it empty and the first step builds one by central differences. The
trust radius is a max-abs cap on the Cartesian step.

From C, ``rgsaddle_index1_create`` / ``rgsaddle_index1_step`` follow
the same loop. The vector length is ``3 * n_atoms``; a bead polymer
passes ``n_atoms * n_beads``. A host that owns the Hessian calls
``rgsaddle_nichols_step`` with the spectrum (``nmode`` may be smaller
than ``n`` when external modes were dropped) and
``rgsaddle_hessian_bofill`` or ``rgsaddle_hessian_powell`` to update
it. ``rgsaddle_cap_max_abs`` applies the trust cap. The gradient on
the wire is dE/dx.

Do not keep solver history across a potential change
----------------------------------------------------

Call ``reset`` after any model update. That is the documented
boundary. A host that skips it mixes two surfaces in one L-BFGS
memory.
