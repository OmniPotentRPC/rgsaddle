# Kani null and short-length guards

Every `extern "C"` function in `src/capi.rs` and `src/capi/sessions.rs`
whose signature has both a pointer and a length is covered. The harness
passes a null pointer with a length in `0..=4`, and a non-null dangling
pointer whose length fails the guard (`n <= 0`, `n_atoms < 1`,
`n_images < 3`, or `n_atoms < 2` for `rgsaddle_place_perp_dummy`).
Both calls return before `from_raw_parts`. The unwind bound on each
harness is 1.

Lengths are a `u8` or an `i8`. The same harnesses with an `i64`
nondeterministic length did not finish in 90 seconds, and a hosted
runner was stopped while `cap_max_abs` was still running. That was the
encoding of the length, not an unsupported construct. No callee is
stubbed: the session body is not reached, and replacing it would drop
the real function from the proof. No harness is deferred to a schedule.
A shard that does not report success fails the workflow.

The harness build passes `--no-default-features`. The default `faer`
feature stays off, so `pulp` is not compiled. `pulp` contains global
assembly, which Kani rejects, and it arrives through faer
(`cargo tree -i pulp`). Connect and SQNM call faer for a dense
self-adjoint spectrum. These guards return before that call. With the
feature off those functions still compile, and they panic if reached,
so a harness that entered them would fail.

`cap_max_abs` is two harnesses, null and short. The other 27 entry
points keep both obligations in one harness. That is 29 proofs of 28
functions.

Kani 0.68.0, CBMC 6.11.0, on this machine. The command was
`cargo kani --no-default-features --features capi --harness kani_proofs::abi_guards::<name> --output-format terse`.
The time column is CBMC's `Verification Time`.

| Harness | C function | Result | Time (s) |
|---|---|---|---|
| `band_create_null_and_short` | `rgsaddle_band_create` | success, 0 failed | 4.748 |
| `band_create_solid_null_and_short` | `rgsaddle_band_create_solid` | success, 0 failed | 5.550 |
| `minmode_create_null_and_short` | `rgsaddle_minmode_create` | success, 0 failed | 2.518 |
| `nichols_step_null_and_short` | `rgsaddle_nichols_step` | success, 0 failed | 2.297 |
| `prfo_step_null_and_short` | `rgsaddle_prfo_step` | success, 0 failed | 2.422 |
| `hessian_powell_null_and_short` | `rgsaddle_hessian_powell` | success, 0 failed | 1.726 |
| `hessian_bofill_null_and_short` | `rgsaddle_hessian_bofill` | success, 0 failed | 1.591 |
| `cap_max_abs_null` | `rgsaddle_cap_max_abs` | success, 0 of 1477 failed | 0.474 |
| `cap_max_abs_short` | `rgsaddle_cap_max_abs` | success, 0 of 1478 failed | 0.669 |
| `index1_create_null_and_short` | `rgsaddle_index1_create` | success, 0 failed | 0.901 |
| `irc_create_null_and_short` | `rgsaddle_irc_create` | success, 0 failed | 1.352 |
| `irc_create_from_surface_null_and_short` | `rgsaddle_irc_create_from_surface` | success, 0 failed | 2.363 |
| `sella_min_create_null_and_short` | `rgsaddle_sella_min_create` | success, 0 failed | 0.836 |
| `sella_min_create_on_null_and_short` | `rgsaddle_sella_min_create_on` | success, 0 failed | 0.868 |
| `sella_min_create_internal_null_and_short` | `rgsaddle_sella_min_create_internal` | success, 0 failed | 1.036 |
| `sella_min_create_cell_null_and_short` | `rgsaddle_sella_min_create_cell` | success, 0 failed | 0.929 |
| `sella_min_create_cell_internal_null_and_short` | `rgsaddle_sella_min_create_cell_internal` | success, 0 failed | 1.180 |
| `sella_saddle_create_null_and_short` | `rgsaddle_sella_saddle_create` | success, 0 failed | 0.848 |
| `sella_saddle_create_on_null_and_short` | `rgsaddle_sella_saddle_create_on` | success, 0 failed | 0.917 |
| `sella_saddle_create_internal_null_and_short` | `rgsaddle_sella_saddle_create_internal` | success, 0 failed | 0.998 |
| `sella_saddle_create_cell_null_and_short` | `rgsaddle_sella_saddle_create_cell` | success, 0 failed | 0.949 |
| `sella_saddle_create_cell_internal_null_and_short` | `rgsaddle_sella_saddle_create_cell_internal` | success, 0 failed | 1.110 |
| `constraints_create_null_and_short` | `rgsaddle_constraints_create` | success, 0 failed | 1.020 |
| `samd_create_null_and_short` | `rgsaddle_samd_create` | success, 0 failed | 0.634 |
| `pes_create_null_and_short` | `rgsaddle_pes_create` | success, 0 failed | 1.051 |
| `internal_pes_create_null_and_short` | `rgsaddle_internal_pes_create` | success, 0 failed | 0.918 |
| `internal_pes_create_dummies_null_and_short` | `rgsaddle_internal_pes_create_dummies` | success, 0 failed | 0.956 |
| `internal_pes_create_from_find_null_and_short` | `rgsaddle_internal_pes_create_from_find` | success, 0 failed | 1.749 |
| `place_perp_dummy_null_and_short` | `rgsaddle_place_perp_dummy` | success, 0 failed | 0.622 |

Not harnesses, because the signature has no length:

* Pointer-only entry points, including `rgsaddle_abi_stamp`,
  `rgsaddle_band_step`, and the `free` functions.
* `rgsaddle_internal_pes_n_int`, which writes a count and does not take
  a length.
* A positive length with a live buffer. After the guard the body builds
  a session. That path is outside this contract.

Kani warns that `caller_location` and foreign functions are unsupported.
Those warnings are not a skipped harness. The proofs above finished
with no failed property, so those constructs were not reachable.
