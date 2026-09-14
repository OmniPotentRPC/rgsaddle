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

Do not keep solver history across a potential change
----------------------------------------------------

Call ``reset`` after any model update. That is the documented
boundary. A host that skips it mixes two surfaces in one L-BFGS
memory.
