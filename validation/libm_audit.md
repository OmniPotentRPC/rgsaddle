# Libm calls in the stencil

The stencil kernels themselves are arithmetic and square roots.
Three nearby paths call libm. A later ticket can replace them.
This note does not change `src/minmode.rs`.

| Site | Call | Role |
|---|---|---|
| `src/projection.rs`, `dneb_component` | `atan` of the squared perpendicular-norm ratio, scaled by `2/pi` | Trygubenko-Wales switching weight |
| `src/minmode.rs`, dimer rotation | `atan` of the Kastner-Sherwood angle, then `sin` and `cos` of the accepted angle | In-plane dimer step |
| `src/minmode.rs`, `MinModeSession::set_mode` | `acos` of the absolute overlap with the old mode | Retire L-BFGS pairs after a large turn |

`tests/libm_report.rs` prints, for 1000 fixed arguments, the IEEE bits of
`atan` on the DNEB ratio, and of `atan`, `sin`, `cos`, and `acos`.
The arguments start with `-0`, `+0`, the smallest subnormal, and the
smallest positive normal. `acos` arguments lie in `[-1, 1]`.

The verify workflow runs that test on x86_64 glibc, x86_64 musl, and
aarch64 under qemu, and diffs the three logs. A difference is printed
and does not fail the job. The bits are not a correctness oracle.

On this machine the x86_64 glibc log and the aarch64 glibc log matched
on all 1000 rows. The musl log differed on 492 rows: 72 `atan` ratios,
71 `atan`, 29 `sin`, 29 `cos`, and 84 `acos`. The differences are one
ULP on the samples that were inspected. That is a libm difference, not
a change to the stencil arithmetic.
