

Step a NEB band
---------------

``BandSession::step`` assembles eOn-shaped NEB forces and takes
one rgmin ``Solver`` step. ``run`` is a convenience loop over
``step``. Hosts own trust, drift, acquisition, and hybrid MMF
between steps.

``BandSession::reset`` is the model-update boundary: quasi-Newton
history from one surface epoch must not survive onto the next.

Tangents, springs, projections, and the climbing-image force
follow eOn's branch structure. Positions stay unwrapped
Cartesian. A ``Cell`` supplies orthorhombic minimum-image
differences. Neighbor lists are vesin's job.
