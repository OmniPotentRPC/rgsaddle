

Step a minimum-mode search
--------------------------

``MinModeSession::step`` refreshes the lowest curvature mode
(dimer rotation or Lanczos on finite-difference Hessian
actions), inverts the force along it, and takes one rgmin
solver step.

This finds a first-order saddle. Rolling off that saddle is
``IrcSession``, not another min-mode step.
