Changelog
=========

Unreleased
----------

- Arm eOn's L-BFGS resets on the band: a two-loop step that reaches the per-atom cap or faces away from the force drops the stored pairs and steps along the force, and an empty memory scales that step by 0.01.
- Climb the highest interior image only while it lies above both fixed endpoints, as eOn does. A monotonic band keeps the spring force on every image and reports no climbing image.
- Remove the mean Cartesian force from each multi-atom band image, as eOn does when every atom is free. ``BandConfig::remove_translation`` and the ``RGSADDLE_BAND_KEEP_TRANSLATION`` band flag keep the mean for hosts with fixed atoms.
- Retain main's band and minimum-mode solver contracts alongside the Sella, reaction-path, constraint, cell and dynamics interfaces.
- Require curvature measured at a moved stationary point before minimum-mode convergence.
- Update the restricted atomic step radius with the largest atom displacement.
- Preserve the complete cell when reading molecular frames.
- Expose typed C++ and Python controls and the official ABI 1.15 contract.
- Include the retained numerical trajectories, callback-count checks and symbolic, interval and Lean proofs.
