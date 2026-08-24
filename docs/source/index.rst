
===============
rgsaddle
===============

:Author: `Rohit Goswami <https://rgoswami.me>`_

.. raw:: html

   <p class="rgsaddle-hero">
     <img class="rgsaddle-hero-logo rgsaddle-hero-logo--light"
          src="_static/rgsaddle-logo-light.webp"
          alt="rgsaddle logo"
          width="420"
          height="84"
          loading="eager" />
     <img class="rgsaddle-hero-logo rgsaddle-hero-logo--dark"
          src="_static/rgsaddle-logo-dark.webp"
          alt="rgsaddle logo"
          width="420"
          height="84"
          loading="eager" />
   </p>

Overview
--------

``rgsaddle`` is band, min-mode, and IRC mechanics over
`rgmin <https://github.com/OmniPotentRPC/rgmin>`_ steppers. It defines no optimizer of its own.

Two seams, stepping at both:

- Inner: rgmin ``Solver`` -- one optimizer step over an assembled force.

- Outer: ``BandSession::step``, ``MinModeSession::step``,
  ``IrcSession::step`` -- assemble the mechanics, take one solver
  step, report. ``run`` is a convenience loop. Hosts own policy
  between steps.

The product path is MEP roll-down: a first-order saddle to both
adjacent minima (``IrcSession``). The band and the dimer are how
you **find** that saddle.

.. code:: bash

    git clone https://github.com/OmniPotentRPC/rgsaddle.git
    cd rgsaddle
    # Build and test on the remote builder, not a laptop.
    cargo test

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/quickstart

.. toctree::
   :maxdepth: 2
   :caption: How-To Guides

   howto/band
   howto/irc
   howto/minmode

.. toctree::
   :maxdepth: 2
   :caption: Explanation

   explanation/mep

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/architecture
   reference/bibliography
   changelog

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing/index

License
-------

MIT.
