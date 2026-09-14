Getting started
===============

A host builds a band, steps until the projected force is under
``force_tol``, and never lets this crate own the loop.

1. Depend on the published crate
--------------------------------

.. code:: toml

   [dependencies]
   rgsaddle = "0.1"
   rgmin = "0.2"
   ndarray = "0.17"

2. Implement ``BandSurface``
----------------------------

One eval fills energies and gradients for every image. The double-well
test surface is the shortest honest example:

.. code:: rust

   impl BandSurface for DoubleWell {
       fn eval(
           &self,
           positions: ArrayView2<f64>,
           energies: &mut Array1<f64>,
           gradients: &mut Array2<f64>,
       ) -> Result<(), SaddleError> {
           // V = (x² − 1)² + 2y² + 2z²
           Ok(())
       }
   }

3. Step
-------

.. code:: rust

   let config = BandConfig {
       force_tol: 1e-3,
       max_move: 0.1,
       ..BandConfig::default()
   };
   let mut session = BandSession::new(config, initial)?;
   loop {
       let report = session.step(&surface)?;
       if report.status == BandStatus::Converged {
           break;
       }
   }

Default stepper is FIRE. The projected band force is not conservative;
rgmin's session L-BFGS currently applies an energy-decrease acceptance
that refuses every NEB step. FIRE matches eOn's velocity NEB stepper.

4. Reset when the surface changes
---------------------------------

``BandSession::reset`` is the model-update boundary. Quasi-Newton
history taken on one surface epoch must not survive onto the next.

See ``tests/double_well.rs`` for the numbers the crate actually
converges.
