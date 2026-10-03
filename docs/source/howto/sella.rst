Use Sella steppers and sessions
===============================

``SellaMinSession`` uses order-zero quasi-Newton steps. ``SellaSaddleSession`` partitions a rational function optimization (RFO) step into uphill and downhill subspaces. Both retain a Hessian model and a trust radius, and expose one outer step at a time.

``SellaGeom`` selects a named geometry or an equality ``Constraints`` chart. The steppers project onto the tangent space, retract the displacement, and transport vectors. Cartesian sessions use the rigid quotient when internal molecular modes are available. ``InternalPes`` maps host-supplied bonds, angles and dihedrals through the Wilson matrix. ``CellCartesianPes`` and ``CellInternalPes`` include the active cell entries; their energy and gradients are evaluated in the displaced cell.

``TrustRegion`` limits the Euclidean step norm. ``RestrictedAtomicStep`` limits the largest atom displacement and uses that same quantity when updating its radius. ``MaxInternalStep`` limits weighted internal coordinates. The restricted atomic step (RAS) refuses an internal-coordinate packing.

``update_h_ms`` applies a simultaneous multi-secant Hessian update. ``update_hessian_cols`` applies individual updates in column order. ``HessUpdate::TsBfgs`` supports an indefinite Hessian. The retained Sella trajectories are checked by ``tests/sella_gold.rs``.

``SamdSession::new`` uses Euclidean dynamics. ``on`` and ``with_chart`` retain a geometry for every step; ``step_on`` selects a geometry for one step. The velocity-Verlet displacement, transported velocity and stochastic rescaling remain in its tangent space.

``force_match_hessian`` fits pair contributions from Buckingham, Morse, Lennard-Jones and harmonic bond terms. ``EigenDevice::Dlpk`` observes the size, disable and allocation-failure policy. The available eigensolver, QR and projection implementations use the host fallback; selecting that policy does not imply GPU acceleration.
