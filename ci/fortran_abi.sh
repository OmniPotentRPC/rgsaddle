#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
build_dir="$work/meson-fortran"

cd "$root"

rustup toolchain install stable --profile minimal
cargo +stable build --locked --features capi

meson setup "$build_dir" -Dfortran=enabled
meson compile -C "$build_dir"

lib_dir="$root/target/debug"
mod_dir="$build_dir/fortran"
fortran_lib_dir="$build_dir/fortran"
test_dir="$root/tests/fortran"
mod_dir="$(dirname "$(find "$build_dir/fortran" -name rgsaddle.mod -print -quit)")"

if [[ -z "$mod_dir" ]]; then
  echo "failed to locate rgsaddle.mod in $build_dir/fortran" >&2
  exit 1
fi

fc="${FC:-gfortran}"
if ! command -v "$fc" >/dev/null 2>&1; then
  echo "missing Fortran compiler: $fc" >&2
  exit 1
fi

declare -a tests=(
  test_neb
  test_dimer
  test_lanczos
  test_pafi
  test_ohtst
)

pass=0
fail=0
mkdir -p "$work/mods"

for name in "${tests[@]}"; do
  exe="$work/$name.x"
  "$fc" -std=f2008 -Wall -Wextra \
    -J"$work/mods" \
    -I"$mod_dir" \
    "$test_dir/support.f90" "$test_dir/$name.f90" \
    -L"$fortran_lib_dir" -lrgsaddle_fortran \
    -L"$lib_dir" -lrgsaddle \
    -Wl,-rpath,"$fortran_lib_dir" -Wl,-rpath,"$lib_dir" \
    -o "$exe"
  if "$exe"; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1))
  fi
done

echo "fortran-tests compiler=$fc passed=$pass failed=$fail total=${#tests[@]}"

if command -v ifx >/dev/null 2>&1; then
  echo "ifx detected; running smoke compile for test_pafi"
  ifx -std18 -I"$mod_dir" "$test_dir/test_pafi.f90" \
    -L"$fortran_lib_dir" -lrgsaddle_fortran \
    -L"$lib_dir" -lrgsaddle \
    -Wl,-rpath,"$fortran_lib_dir" -Wl,-rpath,"$lib_dir" \
    -o "$work/test_pafi_ifx.x"
  "$work/test_pafi_ifx.x"
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
