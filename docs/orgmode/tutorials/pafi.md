This tutorial takes a newcomer from a checkout to a PAFI free-energy profile and a rate, on an analytic surface, in Rust and in Python. Projected average force integration (PAFI) is Swinburne and Marinica, Phys. Rev. Lett. 120, 135503 (2018), https://doi.org/10.1103/PhysRevLett.120.135503: average the force projected onto a hyperplane normal to a reference path, at fixed arc length, and integrate that average along the path to the potential of mean force. `howto/oh-tst` covers the closely related optimal-hyperplane method, and `explanation/pafi-ohtst` compares the two and gives the equations this tutorial uses without deriving them.


# What you need

A checkout of this repository and Rust 1.88 or newer. `python3` with no extra packages for the Python walkthrough below: it runs the compiled Rust example and reads its answer back, since PAFI has no Python session (`howto/python` covers what does).


# The surface

The surface is an analytic double well with a transverse mode whose stiffness varies along the reaction coordinate:

$$
V(x, y) = (x^2 - 1)^2 + \frac{1}{2} k(x) y^2, \qquad k(x) = 2.5 + 1.5 x.
$$

At fixed $x$ the energy is smallest at $y = 0$, where $V = (x^2 - 1)^2$, the plain double well: minima at $x = \mp 1$, both $V = 0$, and a saddle at the origin, $V = 1$. The valley floor is the straight line $y = 0$, so a path along it has the path tangent and the hyperplane normal both along $x$ everywhere, with no curvature correction to track. The transverse stiffness is $k(-1) = 1$ at the reactant, $k(0) = 2.5$ at the saddle, and $k(+1) = 4$ at the product: stiffer toward the product, softer toward the reactant. Implementing `PointSurface` needs the energy and its Cartesian gradient:

```rust
/// `k(x) = 2.5 + 1.5 x`.
fn stiffness(x: f64) -> f64 {
    2.5 + 1.5 * x
}

struct CurvedWell;

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

Because $k$ varies with $x$ but not with $y$, the transverse coordinate can be integrated out of the Boltzmann weight analytically at any fixed $x$: it is a Gaussian of variance $kT / k(x)$. That gives a closed form for the finite-temperature free energy along the path,

$$
F(x) = (x^2 - 1)^2 - \frac{1}{2} kT \ln\!\frac{2\pi kT}{k(x)},
$$

and so a closed form for the barrier this tutorial checks the sampled one against:

$$
\Delta F = F(0) - F(-1) = 1 + \frac{1}{2} kT \ln\frac{k(0)}{k(-1)}.
$$

```rust
/// Barrier of the exact transverse-Gaussian free energy at `kt`, used only
/// to check the sampled profile: with `y` integrated out analytically,
/// `F(x) = (x² - 1)² - ½ kt ln(2π kt / k(x))`, so
/// `ΔF = F(0) - F(-1) = 1 + ½ kt ln(k(0) / k(-1))`.
fn closed_form_barrier(kt: f64) -> f64 {
    1.0 + 0.5 * kt * (stiffness(0.0) / stiffness(-1.0)).ln()
```

This closed form exists only because the model is this simple. It is not part of PAFI and `rgsaddle::pafi` does not compute it; it is here only so the tutorial has a number to check the sampled barrier against without trusting the sampler to check itself.


# The path and the planes

`ArcPath::from_positions` interpolates the given points with a not-a-knot cubic spline over their summed chord lengths, and `ArcPath::geometry` returns the position, tangent, curvature and jerk at any arc length. Points on a straight line give that line, with zero curvature. 21 points evenly spaced in $x$ from $-1$ to $1$ at $y = 0$ sit exactly on the valley floor:

```rust
fn valley_path(x0: f64, x1: f64, n_images: usize) -> Vec<Array1<f64>> {
    (0..n_images)
        .map(|i| {
            let t = i as f64 / (n_images - 1) as f64;
            let x = x0 + (x1 - x0) * t;
            array![x, 0.0]
        })
        .collect()
```

`zero_temperature_barrier` integrates the potential's own gradient along the path with no sampling, a check of the path itself before spending any sampling budget on it:

    zero-temperature barrier along the path: 0.999998 (potential barrier is 1)

0.999998 against the potential barrier of 1: the 21-point spline already tracks the valley floor closely.


# Sample the planes

`PafiConfig` is non-exhaustive; chain `with_*` setters from `default`:

```rust
impl Default for TutorialConfig {
    fn default() -> Self {
        let kt = 0.1;
        Self {
            kt,
            n_images: 21,
            n_planes: 49,
            pafi: PafiConfig::default()
                .with_thermal(kt)
                .with_mobility(1.0)
                .with_time_step(2.0e-3)
                .with_thermalize(2_000)
                .with_stride(4)
                .with_samples(30_000)
                .with_blocks(24)
                .with_seed(7)
                .with_fix_com(false)
                .with_retain(0),
        }
    }
```

`pafi_profile` takes 49 evenly spaced arc lengths, samples each of their hyperplanes by overdamped Langevin dynamics, and integrates the mean forces to the profile:

```rust
let (profile, planes) = pafi_profile(&surface, &path, &arcs, &config.pafi)?;
```

It samples every plane in parallel on rayon's global pool; `howto/pafi-ohtst-parallel-sampling` times that. The planes are independent of each other: nothing here needs the center-of-mass projection (`with_fix_com(false)`, since the coordinates are a single 2D point, not 3D atoms), and `with_retain(0)` keeps no configurations (while `target_error` stays `None` by default so only one sampling pass runs; `howto/pafi-ohtst-restarting` covers automatic extra rounds and retaining configurations).


# Run it

```console
$ cargo run --release --example pafi_tutorial
```

    zero-temperature barrier along the path: 0.999998 (potential barrier is 1)
    PAFI barrier at kT = 0.1: 1.04418 +/- 0.00068
    closed-form transverse-Gaussian barrier at this kT: 1.04581
    reactant attempt frequency (unit mass): 0.45016
    rate k = nu * exp(-barrier / kT): 1.313827e-5
    largest plane autocorrelation time: 62.00 samples; smallest n_eff: 241.9

The sampled barrier, 1.04418 ± 0.00068, is 2.4 standard errors below the closed form, 1.04581, a difference of 0.0016 or 0.16 percent of the barrier. The worst plane's autocorrelation time is 62 samples and its effective sample count, `samples / (2 tau)`, is 242 out of 30000 recorded: comfortably above the handful that would make the block error untrustworthy (`howto/pafi-ohtst-checking-convergence` shows what that looks like when it is not comfortable).


# The rate

A rate needs a prefactor on top of the barrier. `attempt_frequency` turns a curvature into a frequency, $\nu = \omega / 2\pi$ with $\omega = \sqrt{\lambda / m}$; here $\lambda$ is the curvature of the energy along the path tangent at the reactant, from the exact Hessian of the energy, and the mass is 1 since the surface is already in mass-weighted-like units:

```rust
/// The reactant's attempt frequency along the path tangent at `arc = 0`,
/// from the exact Hessian of the energy, unit mass.
fn reactant_attempt_frequency(path: &ArcPath) -> Result<f64, SaddleError> {
    let geometry = path.geometry(path.bounds().0)?;
    let tangent_norm = geometry.tangent.dot(&geometry.tangent).sqrt();
    let direction = &geometry.tangent / tangent_norm;
    let (x, y) = (geometry.positions[0], geometry.positions[1]);
    let (dx, dy) = (direction[0], direction[1]);
    let hxx = 12.0 * x * x - 4.0;
    let hxy = 1.5 * y;
    let hyy = stiffness(x);
    let lambda = hxx * dx * dx + 2.0 * hxy * dx * dy + hyy * dy * dy;
    attempt_frequency(lambda, 1.0, 1.0)
```

Arrhenius's law then gives the rate directly, with no unit conversion since both the barrier and $kT$ are already in the surface's own energy unit (`rgsaddle::arrhenius` instead fixes eV and kelvin, and is the wrong call here):

```rust
let rate = nu * (-profile.barrier / config.kt).exp();
```

Reactant attempt frequency 0.45016 and barrier 1.04418 at $kT = 0.1$ give rate 1.313827e-5, printed above. `howto/oh-tst` computes the same kind of rate with an additional, independent dynamical correction, the transmission coefficient; `tutorials/oh-tst` compares the two numbers on this exact surface.


# From Python

PAFI is Rust-only: there is no `rgsaddle.pafi` Python module. The Python side of this tutorial drives the compiled Rust example instead, the same way a host language without its own PAFI binding would: write the run's settings to a small JSON file, invoke the binary with that file and an output path, and read the answer back.

```python
def run_pafi(repo_root: Path, config: dict[str, float]) -> dict[str, float]:
    """Run the compiled PAFI example on ``config`` and return its result."""
    binary = find_binary(repo_root)
    with tempfile.TemporaryDirectory() as workdir:
        config_path = Path(workdir) / "config.json"
        result_path = Path(workdir) / "result.json"
        config_path.write_text(json.dumps(config))
        subprocess.run(
            [str(binary), str(config_path), str(result_path)],
            check=True,
        )
```

```console
$ cargo build --release --example pafi_tutorial
$ python3 examples/pafi_tutorial.py
```

    zero-temperature barrier along the path: 0.999998
    PAFI barrier at kT = 0.1: 1.04418 +/- 0.00068
    closed-form transverse-Gaussian barrier: 1.04581
    reactant attempt frequency (unit mass): 0.45016
    rate k = nu * exp(-barrier / kT): 1.313827e-05
    measured barrier is 2.39 standard errors from the closed form

The two runs agree to the last printed digit: the JSON path sets exactly the same `PafiConfig` fields the Rust walkthrough does, down to the time step, so it draws the same chain from the same seed.


# Next
