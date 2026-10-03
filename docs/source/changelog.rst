Changelog
=========

Unreleased
----------

- Retain main's band and minimum-mode solver contracts alongside the Sella, reaction-path, constraint, cell and dynamics interfaces.
- Require curvature measured at a moved stationary point before minimum-mode convergence.
- Update the restricted atomic step radius with the largest atom displacement.
- Preserve the complete cell when reading molecular frames.
- Expose typed C++ and Python controls and the official ABI 1.15 contract.
- Include the retained numerical trajectories, callback-count checks and symbolic, interval and Lean proofs.
