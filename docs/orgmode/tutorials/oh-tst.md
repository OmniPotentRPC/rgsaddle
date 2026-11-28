This tutorial takes a newcomer through optimal hyperplanar transition state theory (OH-TST) end to end, in Rust and in Python: seed a plane from a dimer, climb it to the maximum of its free energy, sample it, compute the reactant normalization, get the reversible work to the optimal plane with MBAR, correct for recrossing with a transmission coefficient, and compute a rate. It runs on the same analytic surface as `tutorials/pafi`, so the two tutorials' rates can be checked against each other at the end. `howto/oh-tst` is the condensed reference for every piece this tutorial uses; `explanation/pafi-ohtst` has the theory.


# What you need

A checkout of this repository and Rust 1.88 or newer. `python3` with no extra packages: OH-TST has no Python session either (`howto/python` covers what does), so the Python walkthrough runs the compiled Rust example and reads its answer back, the same pattern `tutorials/pafi` uses.


# The surface

$V(x, y) = (x^2 - 1)^2 + \frac{1}{2} k(x) y^2$, $k(x) = 2.5 + 1.5 x$: the same double well as `tutorials/pafi`, minima at $x = \mp 1$, saddle at the origin, barrier 1.

```rust
struct CurvedWell;

/// `k(x) = 2.5 + 1.5 x`, matching `examples/pafi_tutorial.rs`.
fn stiffness(x: f64) -> f64 {
    2.5 + 1.5 * x
}

impl PointSurface for CurvedWell {
    fn eval(&self, q: ArrayView1<f64>) -> Result<(f64, Array1<f64>), SaddleError> {
        let (x, y) = (q[0], q[1]);
        let k = stiffness(x);
        let well = x * x - 1.0;
        let energy = well * well + 0.5 * k * y * y;
        let dk_dx = 1.5;
        let dvdx = 4.0 * x * well + 0.5 * dk_dx * y * y;
        let dvdy = k * y;
        Ok((energy, array![dvdx, dvdy]))
    }
```

The potential is symmetric under $y \to -y$: the Hessian at $y = 0$ has no $x$-$y$ coupling, and on any plane $x = \text{const}$ the torque vanishes by symmetry. That makes this surface a clean first case, since the plane the dimer seeds is already the optimal hyperplane and there is no rotation to go looking for; `howto/oh-tst` has a surface where the optimizer does turn the plane.


# Seed a plane from a dimer

A minimum-mode (dimer) search finds the saddle and its unstable mode. `Hyperplane::from_min_mode` turns that into a plane: a point and a unit normal, oriented toward the product so the sign convention downstream is fixed.

```rust
let config = MinModeConfig::default().with_force_tol(1.0e-8);
let mut session = MinModeSession::new(config, array![0.05, -0.05], array![1.0, 0.2])?;
let mut converged = false;
for _ in 0..2000 {
    if session.step(&CurvedWell)?.status == MinModeStatus::Converged {
        converged = true;
        break;
    }
}
if !converged {
    return Err(SaddleError::Solver("the dimer did not converge".into()));
}
```

```rust
let seed = Hyperplane::from_min_mode(&session)?.oriented_toward(product.view());
```


# Plane optimisation

`PlaneOptimizer` climbs the plane's free energy by a Riemannian trust region over the sphere of normals and the offset along the normal. Each trial plane is sampled by Langevin dynamics restricted to the plane:

```rust
let sampler = LangevinConfig::default()
    .with_thermal(kt)
    .with_time_step(0.02)
    .with_friction(1.0)
    .with_burn_in(1000)
    .with_stride(5)
    .with_seed(11);
let optimizer_config = PlaneOptimizerConfig::default()
    .with_sampler(sampler.clone().with_records(2000))
    .with_max_iterations(50);
let optimum = PlaneOptimizer::new(&seed, optimizer_config)?.run(&CurvedWell)?;
let turn_degrees = (optimum.plane.normal()[1].atan2(optimum.plane.normal()[0])
    - seed.normal()[1].atan2(seed.normal()[0]))
```

On this surface the optimizer has nothing to do: the seed from the dimer is already at the variational optimum, so the run below reports a turn of under a hundredth of a degree, noise from the stochastic gradient estimates rather than a real rotation. That is worth seeing once on a first run, since most surfaces are not this clean; `howto/oh-tst` runs the optimizer on one that is not symmetric in the transverse direction, where it does turn the plane by a measurable amount.


# Sample the optimal plane

`sample_plane` runs the same restricted Langevin dynamics once more at the optimum, with more records and some configurations retained for the transmission coefficient later:

```rust
let sampled = sample_plane(
    &CurvedWell,
    &optimum.plane,
    None,
    &sampler.clone().with_records(8000).with_retain(256),
```


# The reactant normalization

`basin_density` integrates the mean force on planes parallel to the product direction, spreading out from the reactant minimum, to get $Z_H / Z_R$: the plane through the minimum against the reactant basin as a whole.

```rust
let basin = basin_density(
    &CurvedWell,
    reactant.view(),
    product.view(),
    &BasinConfig::default().with_sampler(sampler.clone().with_records(4000)),
```


# The reversible work, with MBAR

`PlanePath::legs` is the straight path between the reactant plane `basin_density` returns and the optimal plane. `reversible_work_along` integrates the plane free energy's gradient along it; with `with_mbar(true)` and a Gaussian transport map it also solves MBAR across every node's retained configurations for a second, lower-variance estimate of the same work:

```rust
let path = PlanePath::legs(&[basin.plane.clone(), optimum.plane.clone()])?;
let work_config = ReversibleWorkConfig::default()
    .with_sampler(sampler.clone().with_records(2000))
    .with_nodes(9);
let mut integrator = AnalyticSurfaceSampler::new(&CurvedWell, work_config.sampler.clone())
    .with_start(reactant.clone());
let integrated = reversible_work_along(&mut integrator, &path, &work_config)?;
let mbar_config = work_config
    .with_transport(GaussianTransport::default().into())
    .with_mbar(true)
    .with_estimate(WorkEstimate::Mbar);
let mut mapped = AnalyticSurfaceSampler::new(&CurvedWell, mbar_config.sampler.clone())
    .with_start(reactant.clone());
let work = reversible_work_along(&mut mapped, &path, &mbar_config)?;
```


# The rate, then the transmission correction

`ohtst_rate` turns the reactant normalization and the reversible work into `sqrt(kT / 2pi) (Z_H/Z_R) exp(-DeltaA/kT)`, the rate before any dynamical correction. `transmission_coefficient` then shoots trajectories from the plane's retained configurations, forward and backward, to measure how many of the crossings are real: `kappa` below 1 means some trajectories that cross the plane return to where they started instead of committing to the product.

```rust
let rate = ohtst_rate(&basin, &work)?;

let transmission_config = TransmissionConfig::default()
    .with_thermal(kt)
    .with_friction(1.0)
    .with_time_step(0.02)
    .with_steps(400)
    .with_check_every(40)
    .with_shots(sampled.retained.len().min(4096))
    .with_blocks(16)
    .with_seed(23);
let transmission = transmission_coefficient(
    &mut AnalyticSurfaceDynamics::new(&CurvedWell),
    &optimum.plane,
    &sampled.retained,
    &transmission_config,
)?;
```

```rust
let (final_rate, final_error) = rate.transmitted(&transmission);
```


# Run it

```console
$ cargo run --release --features ohtst --example ohtst_tutorial
```

    optimal plane turned -0.0068° from the dimer's seed
    Z_H / Z_R = 3.5112 ± 0.0082
    reversible work reactant plane -> optimum: integration 1.04801 ± 0.00220, BAR 1.04467 ± 0.00207, MBAR 1.04577 ± 0.00079; closed form 1.04581
    k_OH (no transmission correction) = 1.272449e-5 ± 1.1e-7
    kappa = 0.75781 ± 0.02729 (194 reactive, 39 recrossed of 256 shots, 0 unassigned)
    k = kappa * k_OH = 9.642780e-6 ± 3.6e-7

The closed form, 1.04581, is the same analytic barrier `tutorials/pafi` checks its sampled profile against: the reversible work here is the same quantity under a different name, the free energy difference between two planes instead of two points of a potential of mean force. MBAR lands 0.06 standard errors from it, closer than the plain integration (1.0 standard errors) or BAR alone (0.55 standard errors, and with a bigger error bar); `howto/pafi-ohtst-checking-convergence` looks at why MBAR usually wins here.

All 256 shots reached a state (194 reactive), and 39 backward halves recrossed the plane within the window this run checks, giving `kappa = 0.75781 ± 0.02729`: about a quarter of naive transition-state-theory crossings at this plane do not reach the product on the first try. The corrected rate, 9.643e-6, is `kappa` times the bare one, 1.2724e-5.


# Against the PAFI tutorial

`tutorials/pafi` computes a rate on this same surface with no transmission correction, 1.313827e-5, using the reactant's harmonic attempt frequency instead of the sampled reactant normalization `Z_H / Z_R`. That number and `k_OH` here, 1.272449e-5, agree to within 4%: both are transition-state-theory estimates of the same barrier crossing without recrossing, one from a PAFI barrier and a harmonic attempt frequency and the other from a sampled reactant normalization and MBAR work. Neither one knows about recrossing; only the OH-TST side measures it, with `kappa`, and only by paying for the extra shooting.


# From Python

```python
def run_ohtst(repo_root: Path) -> dict[str, float]:
    """Run the compiled OH-TST example and return its JSON result."""
    binary = find_binary(repo_root)
    with tempfile.TemporaryDirectory() as workdir:
        result_path = Path(workdir) / "result.json"
        subprocess.run([str(binary), str(result_path)], check=True)
```

```console
$ cargo build --release --features ohtst --example ohtst_tutorial
$ python3 examples/ohtst_tutorial.py
```

    optimal plane turned -0.0068 degrees from the dimer's seed
    Z_H / Z_R = 3.5112 +/- 0.0082
    reversible work reactant plane -> optimum: MBAR 1.04577 +/- 0.00079, closed form 1.04581
    k_OH (no transmission correction) = 1.272449e-05 +/- 1.1e-07
    kappa = 0.75781 +/- 0.02729
    k = kappa * k_OH = 9.642780e-06 +/- 3.6e-07
    MBAR work is 0.06 standard errors from the closed form

The Rust example's settings are fixed, not read from a config file: there is less to vary here than in the PAFI walkthrough, since a plane optimisation, a basin integration and a transmission run are each already several stages long.


# Next
