The band and the minimum-mode search take one rgmin `Solver` step on a force this crate assembled. `Index1Session` takes a restricted-step partitioned RFO step and updates the Hessian with Powell or Bofill. There is no run-to-completion contract. `run` is a loop over `step`.

Force assembly ports eOn's NEB: Mills–Jónsson–Schenter and Henkelman–Jónsson tangents, uniform / energy-weighted / Onsager–Machlup springs, PEB / NEB / DNEB projections, climbing image with the eOn trigger. Positions are unwrapped Cartesian. A `Cell` supplies minimum-image differences.


# Install

```console
$ cargo add rgsaddle
$ cargo add rgsaddle --features capi   # C header and cdylib
```


# First minute

The analytic double well in `tests/double_well.rs`: minima at $(\pm 1, 0, 0)$, saddle at the origin, barrier 1. Nine images, host loop, endpoints stay put.

```rust
let mut session = BandSession::new(BandConfig::default(), initial_band(9))?;
let mut report = session.step(&DoubleWell)?;
while report.status != BandStatus::Converged {
    report = session.step(&DoubleWell)?;
}
```
