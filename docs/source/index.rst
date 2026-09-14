rgsaddle
========

.. raw:: html

   <div class="vi-hero">
     <div class="vi-hero-brand">
       <img class="vi-hero-mark" src="_static/mark.svg" width="64" height="64" alt="" />
       <div>
         <p class="vi-hero-name">rgsaddle</p>
         <p class="vi-hero-tag">Band and minimum-mode saddles over rgmin</p>
       </div>
     </div>
     <p class="vi-hero-tagline">One solver step on an assembled NEB or inverted-mode force. The host owns the loop.</p>
     <div class="vi-hero-pills">
       <span>No optimizer of its own</span>
       <span>eOn NEB branches</span>
       <span>Rust + C ABI</span>
     </div>
     <div class="vi-hero-actions">
       <a class="vi-btn vi-btn-gold" href="getting-started.html">Get started</a>
       <a class="vi-btn vi-btn-ghost" href="reference.html">Reference</a>
     </div>
   </div>

The inner seam is rgmin's ``Solver``: one optimizer step over a force this
crate assembled. The outer seam is ``BandSession::step`` or
``MinModeSession::step``. There is no run-to-completion contract.
``run`` is a loop over ``step``.

Force assembly ports eOn's NEB: Mills–Jónsson–Schenter and
Henkelman–Jónsson tangents, uniform / energy-weighted / Onsager–Machlup
springs, PEB / NEB / DNEB projections, climbing image with the eOn
trigger. Positions are unwrapped Cartesian. A ``Cell`` supplies
orthorhombic minimum-image differences.

Install
-------

.. code:: console

   $ cargo add rgsaddle
   $ cargo add rgsaddle --features capi   # C header and cdylib

First minute
------------

The analytic double well in ``tests/double_well.rs``: minima at
:math:`(\pm 1, 0, 0)`, saddle at the origin, barrier 1. Nine images,
host loop, endpoints stay put.

.. code:: rust

   let mut session = BandSession::new(BandConfig::default(), initial_band(9))?;
   let mut report = session.step(&DoubleWell)?;
   while report.status != BandStatus::Converged {
       report = session.step(&DoubleWell)?;
   }

.. toctree::
   :maxdepth: 1
   :caption: Guides
   :hidden:

   getting-started
   howto
   reference
   explanation
   seat
