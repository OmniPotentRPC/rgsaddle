Architecture
============

The host supplies the potential surface and owns the outer search loop. rgsaddle supplies band force assembly, minimum-mode rotation, restricted index-1 steps, Sella sessions, intrinsic reaction coordinate integration and molecular dynamics mechanics.

Band and minimum-mode translation use rgmin's solver. Dense quadratic restriction is available through ``rgsaddle::trust``, which re-exports rgmin's dense controller. The root ``rgsaddle::TrustRegion`` accepts a spectrum for Sella steps. The projected band trust method is implemented by ``BandRtr``.

``PointSurface`` accepts one coordinate vector. ``BandSurface`` accepts a collection of images. The C callback expresses either request with the same official structure. The library has no Message Passing Interface (MPI) dependency; a host can evaluate a request collectively or distribute image work itself.

Hessian storage and updates live in ``pes`` and ``qn``. ``linalg`` supplies numerical actions and cached direction/action pairs. ``eigensolve`` supplies dense and iterative eigenproblems. Molecular topology enters through ``vocn`` and the equality chart. Periodic differences use linkcell.
