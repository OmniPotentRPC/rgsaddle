#!/usr/bin/env bash
# Build the Fortran bindings and run the suite.
# gfortran is required and is run with -fcheck=all.
# flang-20 and ifx are used when they are on PATH. ifx compiles the
# module from source; it cannot read a gfortran .mod.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
build_dir="$work/meson-fortran"
prefix_dir="$work/prefix"
export PATH="${HOME}/.local/bin:${PATH}"
export CXX="${CXX:-g++}"

cd "$root"

rustup toolchain install stable --profile minimal
cargo +stable build --locked --features capi

export FFLAGS="${FFLAGS:--std=f2008 -Wall -Wextra -fcheck=all}"
meson setup "$build_dir" -Dfortran=enabled --werror --prefix="$prefix_dir" --libdir=lib
meson compile -C "$build_dir"
meson install -C "$build_dir"

lib_dir="$root/target/debug"
test_dir="$root/tests/fortran"
pc_file="$prefix_dir/lib/pkgconfig/rgsaddle-fortran.pc"
cmake_file="$prefix_dir/lib/cmake/rgsaddleFortran/rgsaddleFortranConfig.cmake"
mod_dir="$prefix_dir/include/rgsaddle/fortran"

if [[ ! -f "$mod_dir/rgsaddle.mod" || ! -f "$pc_file" || ! -f "$cmake_file" ]]; then
  echo "failed to install the Fortran package" >&2
  exit 1
fi
if grep -F "$prefix_dir" "$cmake_file"; then
  echo "CMake config contains the absolute prefix $prefix_dir" >&2
  exit 1
fi
if ! grep -q 'moduledir=\${prefix}/' "$pc_file"; then
  echo "pkg-config moduledir is not prefix-relative:" >&2
  cat "$pc_file" >&2
  exit 1
fi
if ! grep -q -- '-lrgsaddle' "$pc_file"; then
  echo "pkg-config does not name librgsaddle:" >&2
  cat "$pc_file" >&2
  exit 1
fi

moved="$work/moved-prefix"
mv "$prefix_dir" "$moved"
mod_dir="$moved/include/rgsaddle/fortran"
fortran_lib="$moved/lib/librgsaddle_fortran.a"
pc_dir="$moved/lib/pkgconfig:${moved}/share/pkgconfig"
pc_file="$moved/lib/pkgconfig/rgsaddle-fortran.pc"
cmake_file="$moved/lib/cmake/rgsaddleFortran/rgsaddleFortranConfig.cmake"

# Meson writes an absolute prefix= line. pkg-config recomputes it from
# the .pc location only with --define-prefix (the default on Windows,
# off on Linux).
pc_cflags="$(PKG_CONFIG_PATH="$pc_dir" pkg-config --define-prefix --cflags rgsaddle-fortran)"
pc_libs="$(PKG_CONFIG_PATH="$pc_dir" pkg-config --define-prefix --libs rgsaddle-fortran)"
case "$pc_cflags" in
  *"$moved"*) ;;
  *)
    echo "pkg-config cflags did not follow the moved prefix: $pc_cflags" >&2
    exit 1
    ;;
esac
case "$pc_libs" in
  *"-lrgsaddle"*) ;;
  *)
    echo "pkg-config libs missing -lrgsaddle: $pc_libs" >&2
    exit 1
    ;;
esac

declare -a tests=(
  test_neb
  test_dimer
  test_lanczos
  test_pafi
  test_ohtst
  test_handles
)

pass=0
fail=0

compile_and_run() {
  local fc="$1"
  local name="$2"
  shift 2
  local exe="$work/${name}.${fc##*/}.x"
  # shellcheck disable=SC2086
  "$fc" "$@" -o "$exe"
  if "$exe"; then
    pass=$((pass + 1))
  else
    echo "FAIL $fc $name" >&2
    fail=$((fail + 1))
  fi
}

run_assign() {
  local fc="$1"
  local exe="$2"
  local kind rc
  for kind in band minmode pafi ohtst; do
    set +e
    "$exe" "$kind"
    rc=$?
    set -e
    if [[ "$rc" -eq 17 ]]; then
      pass=$((pass + 1))
    else
      echo "FAIL $fc assign $kind rc=$rc" >&2
      fail=$((fail + 1))
    fi
  done
}

link_runtime() {
  printf '%s\n' -L"$lib_dir" -lrgsaddle -Wl,-rpath,"$lib_dir"
}

if ! command -v gfortran >/dev/null 2>&1; then
  echo "missing gfortran" >&2
  exit 1
fi

mkdir -p "$work/gfortran-mods"
for name in "${tests[@]}"; do
  # shellcheck disable=SC2046
  compile_and_run gfortran "$name" \
    -std=f2008 -Wall -Wextra -Werror -fcheck=all \
    -J"$work/gfortran-mods" \
    $pc_cflags \
    "$test_dir/support.f90" "$test_dir/$name.f90" \
    $(link_runtime) \
    $pc_libs
done
# shellcheck disable=SC2046
gfortran -std=f2008 -Wall -Wextra -Werror -fcheck=all \
  -J"$work/gfortran-mods" \
  $pc_cflags \
  "$test_dir/test_assign.f90" \
  $(link_runtime) \
  $pc_libs \
  -o "$work/test_assign.gfortran.x"
run_assign gfortran "$work/test_assign.gfortran.x"

cat >"$work/smoke.f90" <<'EOF'
program smoke
  use, intrinsic :: iso_c_binding
  use rgsaddle
  implicit none
  type(rgsaddle_version_t) :: version
  integer(c_int) :: status
  status = abi_stamp(version)
  if (status /= RGSADDLE_OK) stop 1
  if (version%major /= 1 .or. version%minor /= 47) stop 2
end program smoke
EOF

cmake_src="$work/cmake-consumer"
mkdir -p "$cmake_src"
cp "$work/smoke.f90" "$cmake_src/smoke.f90"
cat >"$cmake_src/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.16)
project(rgsaddle_fortran_consumer LANGUAGES Fortran)
find_package(rgsaddleFortran REQUIRED)
add_executable(smoke smoke.f90)
target_link_libraries(smoke PRIVATE rgsaddle::fortran)
EOF
if cmake -S "$cmake_src" -B "$work/cmake-missing-capi" \
  -DCMAKE_PREFIX_PATH="$moved" \
  -DrgsaddleFortran_LIBRARY="$fortran_lib" \
  -DCMAKE_FIND_ROOT_PATH="$moved" \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY; then
  echo "FAIL CMake package accepted a missing C ABI library" >&2
  fail=$((fail + 1))
else
  pass=$((pass + 1))
fi
cmake -S "$cmake_src" -B "$work/cmake-build" \
  -DCMAKE_PREFIX_PATH="$moved" \
  -DrgsaddleFortran_CAPI_LIBRARY="$lib_dir/librgsaddle.so"
cmake --build "$work/cmake-build"
if LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$work/cmake-build/smoke"; then
  pass=$((pass + 1))
else
  echo "FAIL cmake consumer" >&2
  fail=$((fail + 1))
fi

meson_src="$work/meson-consumer"
mkdir -p "$meson_src"
cp "$work/smoke.f90" "$meson_src/smoke.f90"
cat >"$meson_src/meson.build" <<EOF
project('rgsaddle_fortran_consumer', 'fortran')
dep = dependency('rgsaddle-fortran', required: true)
executable('smoke', 'smoke.f90', dependencies: dep, link_args: ['-L$lib_dir', '-Wl,-rpath,$lib_dir'])
EOF
cat >"$work/pkg-config-define-prefix" <<'EOF'
#!/bin/sh
exec pkg-config --define-prefix "$@"
EOF
chmod +x "$work/pkg-config-define-prefix"
PKG_CONFIG="$work/pkg-config-define-prefix" PKG_CONFIG_PATH="$pc_dir" LIBRARY_PATH="$lib_dir" \
  meson setup "$work/meson-consumer-build" "$meson_src" --wrap-mode=nofallback
LIBRARY_PATH="$lib_dir" meson compile -C "$work/meson-consumer-build"
if LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$work/meson-consumer-build/smoke"; then
  pass=$((pass + 1))
else
  echo "FAIL meson consumer" >&2
  fail=$((fail + 1))
fi

if command -v flang-20 >/dev/null 2>&1; then
  mkdir -p "$work/flang-mods"
  for name in "${tests[@]}"; do
    compile_and_run flang-20 "$name" \
      -std=f2018 \
      -module-dir "$work/flang-mods" \
      "$root/fortran/rgsaddle.f90" \
      "$test_dir/support.f90" "$test_dir/$name.f90" \
      $(link_runtime)
  done
  flang-20 -std=f2018 \
    -module-dir "$work/flang-mods" \
    "$root/fortran/rgsaddle.f90" \
    "$test_dir/test_assign.f90" \
    $(link_runtime) \
    -o "$work/test_assign.flang-20.x"
  run_assign flang-20 "$work/test_assign.flang-20.x"
fi

if command -v ifx >/dev/null 2>&1; then
  mkdir -p "$work/ifx-mods"
  for name in "${tests[@]}"; do
    compile_and_run ifx "$name" \
      -stand f18 \
      -module "$work/ifx-mods" \
      "$root/fortran/rgsaddle.f90" \
      "$test_dir/support.f90" "$test_dir/$name.f90" \
      $(link_runtime)
  done
  ifx -stand f18 \
    -module "$work/ifx-mods" \
    "$root/fortran/rgsaddle.f90" \
    "$test_dir/test_assign.f90" \
    $(link_runtime) \
    -o "$work/test_assign.ifx.x"
  run_assign ifx "$work/test_assign.ifx.x"
fi

# The packaged archive, not only the pkg-config link line.
mkdir -p "$work/archive-mods"
gfortran -std=f2008 -Wall -Wextra -Werror -fcheck=all \
  -J"$work/archive-mods" -I"$mod_dir" \
  "$test_dir/support.f90" "$work/smoke.f90" \
  "$fortran_lib" \
  $(link_runtime) \
  -o "$work/smoke.archive.x"
if "$work/smoke.archive.x"; then
  pass=$((pass + 1))
else
  echo "FAIL static archive smoke" >&2
  fail=$((fail + 1))
fi

echo "fortran-tests passed=$pass failed=$fail"
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
