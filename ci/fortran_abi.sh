#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
build_dir="$work/meson-fortran"
prefix_dir="$work/prefix"

cd "$root"

rustup toolchain install stable --profile minimal
cargo +stable build --locked --features capi

meson setup "$build_dir" -Dfortran=enabled --prefix="$prefix_dir" --libdir=lib
meson compile -C "$build_dir"
meson install -C "$build_dir"

lib_dir="$root/target/debug"
fortran_lib_dir="$prefix_dir/lib"
test_dir="$root/tests/fortran"
mod_dir="$prefix_dir/include/rgsaddle/fortran"

if [[ ! -f "$mod_dir/rgsaddle.mod" ]]; then
  echo "failed to locate installed rgsaddle.mod in $mod_dir" >&2
  exit 1
fi
if [[ ! -f "$prefix_dir/lib/pkgconfig/rgsaddle-fortran.pc" ]]; then
  echo "failed to locate installed rgsaddle-fortran.pc" >&2
  exit 1
fi
if [[ ! -f "$prefix_dir/lib/cmake/rgsaddleFortran/rgsaddleFortranConfig.cmake" ]]; then
  echo "failed to locate installed rgsaddleFortranConfig.cmake" >&2
  exit 1
fi

fc="${FC:-gfortran}"
if ! command -v "$fc" >/dev/null 2>&1; then
  echo "missing Fortran compiler: $fc" >&2
  exit 1
fi

pc_cflags="$(PKG_CONFIG_PATH="$prefix_dir/lib/pkgconfig:$prefix_dir/share/pkgconfig" pkg-config --cflags rgsaddle-fortran)"
pc_libs="$(PKG_CONFIG_PATH="$prefix_dir/lib/pkgconfig:$prefix_dir/share/pkgconfig" pkg-config --libs rgsaddle-fortran)"

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
  # shellcheck disable=SC2086
  "$fc" -std=f2008 -Wall -Wextra \
    -J"$work/mods" \
    $pc_cflags \
    "$test_dir/support.f90" "$test_dir/$name.f90" \
    $pc_libs \
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
  ifx -std18 -I"$mod_dir" "$test_dir/support.f90" "$test_dir/test_pafi.f90" \
    -L"$fortran_lib_dir" -lrgsaddle_fortran \
    -L"$lib_dir" -lrgsaddle \
    -Wl,-rpath,"$fortran_lib_dir" -Wl,-rpath,"$lib_dir" \
    -o "$work/test_pafi_ifx.x"
  "$work/test_pafi_ifx.x"
fi

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
