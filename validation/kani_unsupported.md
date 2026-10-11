# Kani harnesses that do not run

The harnesses in `src/kani_proofs.rs` cover every `extern "C"` function in
`src/capi.rs` and `src/capi/sessions.rs` whose signature has both a pointer
and a length (`n`, `n_atoms`, or `n_images`). Each proof passes a null
pointer with `kani::any()` length at most 4, and a non-null dangling
pointer whose length fails the guard (`n <= 0`, `n_atoms < 1`,
`n_images < 3`, or `n_atoms < 2` for `rgsaddle_place_perp_dummy`).
Those branches return before `from_raw_parts`. The unwind bound is 6.

The following are not harnesses.

* A positive length with a live buffer. After the guard the body calls
  `from_raw_parts` and builds a session. That path allocates, updates
  Hessians, and calls the host surface. It is outside the null and
  short-length contract, and Kani is not asked to execute it.
* Pointer-only entry points, including `rgsaddle_abi_stamp`,
  `rgsaddle_band_step`, and the `free` functions. They have no length
  argument.
* `rgsaddle_internal_pes_n_int` writes a count through an output pointer.
  It does not take a length.

`cargo kani --features capi --only-codegen` compiles the crate. CBMC
then checks one harness at a time. The limit is 90 seconds of CBMC
time, after codegen. A harness that does not finish is not a proof.
The job prints `timeout` for that function and continues. A finished
counterexample fails the job.

Kani 0.68.0 on this machine, `timeout 90 cargo kani --features capi --harness kani_proofs::abi_guards::<name>`:

| Harness | Result |
|---|---|
| `band_create_null_and_short` | timeout |
| `band_create_solid_null_and_short` | timeout |
| `minmode_create_null_and_short` | timeout |
| `nichols_step_null_and_short` | timeout |
| `prfo_step_null_and_short` | timeout |
| `hessian_powell_null_and_short` | timeout |
| `hessian_bofill_null_and_short` | timeout |
| `cap_max_abs_null_and_short` | timeout |
| `index1_create_null_and_short` | timeout |
| `irc_create_null_and_short` | timeout |
| `irc_create_from_surface_null_and_short` | success, 2.570 s, 0 of 11604 failed |
| `sella_min_create_null_and_short` | timeout |
| `sella_min_create_on_null_and_short` | timeout |
| `sella_min_create_internal_null_and_short` | timeout |
| `sella_min_create_cell_null_and_short` | timeout |
| `sella_min_create_cell_internal_null_and_short` | timeout |
| `sella_saddle_create_null_and_short` | timeout |
| `sella_saddle_create_on_null_and_short` | timeout |
| `sella_saddle_create_internal_null_and_short` | timeout |
| `sella_saddle_create_cell_null_and_short` | timeout |
| `sella_saddle_create_cell_internal_null_and_short` | timeout |
| `constraints_create_null_and_short` | success, 4.073 s, 0 of 297 failed |
| `samd_create_null_and_short` | success, 0.977 s, 0 of 3682 failed |
| `pes_create_null_and_short` | success, 39.194 s, 0 of 2593 failed |
| `internal_pes_create_null_and_short` | timeout |
| `internal_pes_create_dummies_null_and_short` | timeout |
| `internal_pes_create_from_find_null_and_short` | timeout |
| `place_perp_dummy_null_and_short` | success, 11.411 s, 0 of 1171 failed |

Five harnesses finished with no failed properties. Twenty-three hit the
limit. `rgsaddle_internal_pes_create_from_find` is one of those: an
earlier run left CBMC on the null path for fifteen minutes without a
result. The same limit covers any other harness that grows the same way.

Kani warns that `caller_location`, foreign functions, and atomics are
unsupported. Those warnings are not a skipped harness. A proof is
recorded only when the log says `VERIFICATION:- SUCCESSFUL`.

If the crate later fails to compile under Kani, none of the harnesses
run. The compiler log is the reason. The functions are not silently
dropped from the list above.
