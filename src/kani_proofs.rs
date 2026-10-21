//! Golden bits for [`crate::minmode::translated_gradient`] and Kani
//! harnesses for the C ABI null and short-length guards.
//!
//! The drift guard has to run under `cargo test`. A `#[cfg(kani)]` module
//! gate on this file would hide that test, and `translated_gradient` has
//! no public wrapper. The harnesses below stay behind `cfg(kani)`.

#[cfg(test)]
mod translated_bits {
    use crate::minmode::translated_gradient;
    use ndarray::Array1;
    use std::path::PathBuf;

    fn rat(token: &str) -> f64 {
        let (num, den) = token.split_once('/').expect("num/den");
        let num: i64 = num.parse().expect(token);
        let den: i64 = den.parse().expect(token);
        num as f64 / den as f64
    }

    fn slash_numbers(line: &str) -> Vec<f64> {
        let mut out = Vec::new();
        let mut rest = line;
        while let Some(start) = rest.find('"') {
            rest = &rest[start + 1..];
            let end = rest.find('"').expect("closing quote");
            let token = &rest[..end];
            if token.contains('/') {
                out.push(rat(token));
            }
            rest = &rest[end + 1..];
        }
        out
    }

    #[test]
    fn translated_gradient_matches_golden_bits() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("formal/golden/stencil.json");
        let text = std::fs::read_to_string(&path).expect("formal/golden/stencil.json");
        let start = text.find("\"translated\"").expect("translated section");
        let mut saw = 0;
        for line in text[start..].lines() {
            if line.trim_start().starts_with(']') {
                break;
            }
            if !line.contains("\"g\"") {
                continue;
            }
            let nums = slash_numbers(line);
            // g, mode, curvature, out. The golden rows are two-dimensional.
            assert_eq!(nums.len(), 7, "{line}");
            let gradient = Array1::from(vec![nums[0], nums[1]]);
            let mode = Array1::from(vec![nums[2], nums[3]]);
            let got = translated_gradient(&gradient, &mode, nums[4]);
            // The ascend branch scales the whole mode by `-parallel`.
            // A zero component times a negative scalar is `-0`, while the
            // rational golden writes `0/1`. IEEE equality still holds.
            for (component, expected) in got.iter().zip(&nums[5..]) {
                assert_eq!(*component, *expected, "{line}");
                if component.to_bits() != expected.to_bits() {
                    assert_eq!(component.to_bits(), (-0.0f64).to_bits(), "{line}");
                    assert_eq!(expected.to_bits(), 0.0f64.to_bits(), "{line}");
                }
            }
            saw += 1;
        }
        assert!(saw >= 3, "translated golden rows");
    }
}

/// Null and short-length guards of the C ABI.
///
/// Each harness quantifies lengths with `u8` (`0..=4`) or `i8` (`<= 0`).
/// An `i64` nondeterministic value left CBMC on the guard for more than
/// 90 seconds, including `rgsaddle_cap_max_abs`, whose body is a few
/// lines. `#[kani::unwind(1)]` is enough on every harness: the guard
/// returns before `from_raw_parts` and before any loop.
///
/// `cap_max_abs` is split into a null-pointer proof and a short-length
/// proof. The other entry points keep both obligations in one harness
/// because that pair finishes in a few seconds.
///
/// The heavy callee is not replaced. On these inputs it is not reached,
/// and a `kani::stub` of it would drop the real function from the proof.
/// There is no scheduled leftover. A harness that does not report
/// `VERIFICATION SUCCESSFUL` fails the shard.
#[cfg(all(kani, feature = "capi"))]
mod abi_guards {
    use crate::capi::{
        RGSADDLE_INVALID_PARAMETER, rgsaddle_band_create, rgsaddle_band_create_solid,
        rgsaddle_cap_max_abs, rgsaddle_constraints_create, rgsaddle_hessian_bofill,
        rgsaddle_hessian_powell, rgsaddle_index1_create, rgsaddle_internal_pes_create,
        rgsaddle_internal_pes_create_dummies, rgsaddle_internal_pes_create_from_find,
        rgsaddle_irc_create, rgsaddle_irc_create_from_surface, rgsaddle_minmode_create,
        rgsaddle_nichols_step, rgsaddle_pes_create, rgsaddle_place_perp_dummy, rgsaddle_prfo_step,
        rgsaddle_samd_create, rgsaddle_sella_min_create, rgsaddle_sella_min_create_cell,
        rgsaddle_sella_min_create_cell_internal, rgsaddle_sella_min_create_internal,
        rgsaddle_sella_min_create_on, rgsaddle_sella_saddle_create,
        rgsaddle_sella_saddle_create_cell, rgsaddle_sella_saddle_create_cell_internal,
        rgsaddle_sella_saddle_create_internal, rgsaddle_sella_saddle_create_on,
    };
    use std::ptr::{null, null_mut};

    /// Lengths the null-pointer contract quantifies over.
    /// A `u8` keeps the solver off the rest of the `i64` domain.
    fn length_upto_4() -> i64 {
        let n: u8 = kani::any();
        kani::assume(n <= 4);
        i64::from(n)
    }

    /// Every non-positive length. An `i8` is the smallest signed domain
    /// that still includes the guard's boundary at 0.
    fn short_nonpositive() -> i64 {
        let n: i8 = kani::any();
        kani::assume(n <= 0);
        i64::from(n)
    }

    fn dangling<T>() -> *const T {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    }

    fn dangling_mut<T>() -> *mut T {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn band_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_band_create(null(), n, n, null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe { rgsaddle_band_create(dangling(), 4, n_atoms, dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn band_create_solid_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_band_create_solid(null(), n, n, null(), null(), 0.0, 0.0) };
        assert!(rejected.is_null());
        let n_images = short_nonpositive();
        // `n_images < 3` returns before the cell buffer is read.
        kani::assume(n_images <= 2);
        let rejected_short = unsafe {
            rgsaddle_band_create_solid(dangling(), n_images, 1, dangling(), dangling(), 0.0, 0.0)
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn minmode_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_minmode_create(null(), n, null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short =
            unsafe { rgsaddle_minmode_create(dangling(), n_atoms, dangling(), dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn nichols_step_null_and_short() {
        let n = length_upto_4();
        let status = unsafe {
            rgsaddle_nichols_step(n, n, null(), null(), null(), null(), 1.0, 0, null_mut())
        };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let n_short = short_nonpositive();
        let status_short = unsafe {
            rgsaddle_nichols_step(
                n_short,
                1,
                dangling(),
                dangling(),
                dangling(),
                dangling(),
                1.0,
                0,
                dangling_mut(),
            )
        };
        assert_eq!(status_short, RGSADDLE_INVALID_PARAMETER);
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn prfo_step_null_and_short() {
        let n = length_upto_4();
        let status = unsafe { rgsaddle_prfo_step(n, null(), null(), null(), 1.0, 0, null_mut()) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let n_short = short_nonpositive();
        let status_short = unsafe {
            rgsaddle_prfo_step(
                n_short,
                dangling(),
                dangling(),
                dangling(),
                1.0,
                0,
                dangling_mut(),
            )
        };
        assert_eq!(status_short, RGSADDLE_INVALID_PARAMETER);
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn hessian_powell_null_and_short() {
        let n = length_upto_4();
        let status = unsafe { rgsaddle_hessian_powell(n, null(), null(), null_mut()) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let n_short = short_nonpositive();
        let status_short =
            unsafe { rgsaddle_hessian_powell(n_short, dangling(), dangling(), dangling_mut()) };
        assert_eq!(status_short, RGSADDLE_INVALID_PARAMETER);
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn hessian_bofill_null_and_short() {
        let n = length_upto_4();
        let status = unsafe { rgsaddle_hessian_bofill(n, null(), null(), null_mut()) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let n_short = short_nonpositive();
        let status_short =
            unsafe { rgsaddle_hessian_bofill(n_short, dangling(), dangling(), dangling_mut()) };
        assert_eq!(status_short, RGSADDLE_INVALID_PARAMETER);
    }

    /// Null `step` is rejected for every length in `0..=4`.
    /// Unwind 1: the guard returns before the step buffer is read.
    #[kani::proof]
    #[kani::unwind(1)]
    fn cap_max_abs_null() {
        let n: u8 = kani::any();
        kani::assume(n <= 4);
        let status = unsafe { rgsaddle_cap_max_abs(i64::from(n), null_mut(), 1.0) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    }

    /// `n <= 0` is rejected before the step buffer is read.
    /// The pointer is non-null and dangling. Unwind 1: no loop runs.
    #[kani::proof]
    #[kani::unwind(1)]
    fn cap_max_abs_short() {
        let n: i8 = kani::any();
        kani::assume(n <= 0);
        let status = unsafe { rgsaddle_cap_max_abs(i64::from(n), dangling_mut(), 1.0) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn index1_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_index1_create(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_index1_create(dangling(), n_atoms, dangling(), dangling(), dangling())
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn irc_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_irc_create(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short =
            unsafe { rgsaddle_irc_create(dangling(), n_atoms, dangling(), dangling(), dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn irc_create_from_surface_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe {
            rgsaddle_irc_create_from_surface(null(), n, null(), null(), null(), None, null_mut())
        };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_irc_create_from_surface(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                None,
                dangling_mut(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_min_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_sella_min_create(null(), n, null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short =
            unsafe { rgsaddle_sella_min_create(dangling(), n_atoms, dangling(), dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_min_create_on_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_sella_min_create_on(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_min_create_on(dangling(), n_atoms, dangling(), dangling(), dangling())
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_min_create_internal_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_sella_min_create_internal(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_min_create_internal(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_min_create_cell_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_sella_min_create_cell(null(), n, null(), null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_min_create_cell(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_min_create_cell_internal_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe {
            rgsaddle_sella_min_create_cell_internal(
                null(),
                n,
                null(),
                null(),
                null(),
                null(),
                null(),
            )
        };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_min_create_cell_internal(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_saddle_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_sella_saddle_create(null(), n, null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short =
            unsafe { rgsaddle_sella_saddle_create(dangling(), n_atoms, dangling(), dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_saddle_create_on_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_sella_saddle_create_on(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_saddle_create_on(dangling(), n_atoms, dangling(), dangling(), dangling())
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_saddle_create_internal_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_sella_saddle_create_internal(null(), n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_saddle_create_internal(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_saddle_create_cell_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_sella_saddle_create_cell(null(), n, null(), null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_saddle_create_cell(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn sella_saddle_create_cell_internal_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe {
            rgsaddle_sella_saddle_create_cell_internal(
                null(),
                n,
                null(),
                null(),
                null(),
                null(),
                null(),
            )
        };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_sella_saddle_create_cell_internal(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                dangling(),
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn constraints_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_constraints_create(null(), n) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe { rgsaddle_constraints_create(dangling(), n_atoms) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn samd_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_samd_create(null(), n, null(), null(), None, null_mut()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_samd_create(
                dangling(),
                n_atoms,
                dangling(),
                dangling(),
                None,
                dangling_mut(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn pes_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_pes_create(n, null(), null(), 0, 0) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe { rgsaddle_pes_create(n_atoms, dangling(), dangling(), 0, 0) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn internal_pes_create_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe { rgsaddle_internal_pes_create(n, null(), null(), null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short =
            unsafe { rgsaddle_internal_pes_create(n_atoms, dangling(), dangling(), dangling()) };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn internal_pes_create_dummies_null_and_short() {
        let n = length_upto_4();
        let rejected =
            unsafe { rgsaddle_internal_pes_create_dummies(n, null(), null(), null(), n, null()) };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_internal_pes_create_dummies(
                n_atoms,
                dangling(),
                dangling(),
                dangling(),
                1,
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn internal_pes_create_from_find_null_and_short() {
        let n = length_upto_4();
        let rejected = unsafe {
            rgsaddle_internal_pes_create_from_find(
                n,
                null(),
                null(),
                n,
                null(),
                n,
                null(),
                n,
                null(),
            )
        };
        assert!(rejected.is_null());
        let n_atoms = short_nonpositive();
        let rejected_short = unsafe {
            rgsaddle_internal_pes_create_from_find(
                n_atoms,
                dangling(),
                dangling(),
                0,
                dangling(),
                0,
                dangling(),
                0,
                dangling(),
            )
        };
        assert!(rejected_short.is_null());
    }

    #[kani::proof]
    #[kani::unwind(1)]
    fn place_perp_dummy_null_and_short() {
        let n = length_upto_4();
        let status = unsafe { rgsaddle_place_perp_dummy(n, null(), 0, 1, null_mut()) };
        assert_eq!(status, RGSADDLE_INVALID_PARAMETER);
        let n_atoms: i8 = kani::any();
        // The guard is `n_atoms < 2`, before either buffer is read.
        kani::assume(n_atoms <= 1);
        let status_short = unsafe {
            rgsaddle_place_perp_dummy(i64::from(n_atoms), dangling(), 0, 1, dangling_mut())
        };
        assert_eq!(status_short, RGSADDLE_INVALID_PARAMETER);
    }
}
