

Roll down from a first-order saddle
-----------------------------------

A two-atom double well has a saddle at
the origin and minima at :math:`x = \pm 1`. ``IrcSession`` kicks along
the imaginary mode, then projects each inner step onto the
mass-weighted sphere from `rgmin <https://github.com/OmniPotentRPC/rgmin>`_ (``IrcTrust``). Forward and
reverse leave the col in opposite directions.

What you will need
~~~~~~~~~~~~~~~~~~

A checkout of ``rgsaddle`` with its pinned ``rgmin`` dependency.

The surface
~~~~~~~~~~~

.. code:: rust

    use ndarray::{Array1, ArrayView1};
    use rgsaddle::{PointSurface, SaddleError};

    struct DoubleWell;

    impl PointSurface for DoubleWell {
        fn eval(&self, x: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
            let t = x[0];
            let energy = (t * t - 1.0) * (t * t - 1.0);
            let mut g = Array1::zeros(x.len());
            g[0] = 4.0 * t * (t * t - 1.0);
            Ok((energy, g))
        }
    }

Both branches
~~~~~~~~~~~~~

.. code:: rust

    use ndarray::Array1;
    use rgsaddle::{IrcConfig, IrcDirection, IrcSession};

    let saddle = Array1::zeros(6);
    let masses = Array1::from(vec![1.0, 1.0]);
    let seed = Array1::from(vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0]);
    let mut session = IrcSession::from_surface(
        IrcConfig { dx: 0.2, ..IrcConfig::default() },
        saddle,
        masses,
        seed,
        IrcDirection::Forward,
        &DoubleWell,
    )
    .unwrap();
    let _fwd = session.step(&DoubleWell).unwrap();
    let x_fwd = session.position()[0];
    session.set_direction(IrcDirection::Reverse);
    let _rev = session.step(&DoubleWell).unwrap();
    let x_rev = session.position()[0];
    assert!(x_fwd * x_rev < 0.0);

``from_surface`` estimates the imaginary mode with matrix-free
Lanczos (finite-difference ``H v``). It does not form a dense
Hessian or invoke a dense eigensolver.

The resulting path
~~~~~~~~~~~~~~~~~~

1. Lanczos evaluates Hessian actions at the origin and
   returns the negative-curvature direction.

2. The first ``step`` kicked along that mode onto the mass-weighted sphere of
   radius ``dx``.

3. ``set_direction(Reverse)`` restored the saddle and flipped the
   kick. The two accepted points sit on opposite sides of the col.

The host owns the loop after that. ``run(surface, max_steps)`` is
a convenience over ``step`` until ``at_minimum``. ``reset`` drops quasi-Newton
history at a model-update boundary, as for
``BandSession``.
