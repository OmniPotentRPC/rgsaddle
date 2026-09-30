Reference
=========

Public Rust types
-----------------

.. list-table::
   :header-rows: 1

   * - Type
     - Role
   * - ``BandSession``
     - Outer seam. ``new``, ``step``, ``run``, ``reset``, ``positions``
   * - ``BandConfig``
     - Tangent, spring, projection, climbing, cell, tolerances, method
   * - ``BandSurface``
     - Host potential on a band (energies + gradients per image)
   * - ``BandReport``
     - ``status``, ``max_force``, ``ci_index``, ``iteration``
   * - ``MinModeSession``
     - Dimer or Lanczos lowest-mode search, same stepping contract
   * - ``Index1Session``
     - Nichols index-1 step, trust cap, Powell or Bofill update
   * - ``nichols_step``
     - Displacement from a supplied spectrum (dense or banded)
   * - ``Cell``
     - 3×3 orthorhombic minimum-image wrap
   * - ``SaddleError``
     - Shape, surface, non-finite, solver, ABI

C ABI (feature ``capi``)
------------------------

``include/rgsaddle.h``. ``RGSADDLE_ABI_MAJOR`` is 1. Status codes
include ``RGSADDLE_OK``, ``RGSADDLE_SHAPE``, ``RGSADDLE_SURFACE_FAILED``,
``RGSADDLE_ABI_MISMATCH``. Methods on the wire: FIRE and L-BFGS.
Min-mode kinds: dimer and Lanczos. Index-1 entries:
``rgsaddle_nichols_step``, ``rgsaddle_hessian_powell``,
``rgsaddle_hessian_bofill``, ``rgsaddle_cap_max_abs``, and
``rgsaddle_index1_step``. ``RGSADDLE_ABI_MINOR`` is 3.

Tests that pin the contract
---------------------------

- ``tests/double_well.rs`` — band finds the origin saddle, endpoints
  pinned
- ``tests/minmode_saddle.rs`` — dimer / Lanczos
- ``tests/nichols_saddle.rs`` — Nichols step against i-PI on the same
  Hessian and gradient, Muller-Brown saddles, quadratic saddle
- ``tests/c_abi.rs`` — header and symbols with ``capi``
