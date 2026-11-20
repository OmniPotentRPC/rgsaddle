#!/usr/bin/env bash
# Run the Fortran handle lifetime tests under valgrind: the create and
# release loop, the handle status checks, and the plain and container
# assignment probes for every handle type. Any valgrind error, or a
# definitely lost block, exits 1. The assignment probes stop with
# RGSADDLE_HANDLE_NOT_ASSIGNABLE (17) on purpose.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

for tool in gfortran valgrind; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "missing $tool" >&2
    exit 1
  fi
done

cd "$root"
cargo build --locked --features capi

lib_dir="$root/target/debug"
test_dir="$root/tests/fortran"
mkdir -p "$work/mods"

build() {
  local name="$1" std="$2"
  shift 2
  gfortran "-std=$std" -g -O0 -Wall -Wextra -Werror -fcheck=all \
    -J"$work/mods" \
    "$root/fortran/rgsaddle.f90" "$@" \
    -L"$lib_dir" -lrgsaddle -Wl,-rpath,"$lib_dir" \
    -o "$work/$name.x"
}

build test_handle_leaks f2008 "$test_dir/support.f90" "$test_dir/test_handle_leaks.f90"
build test_handles f2008 "$test_dir/support.f90" "$test_dir/test_handles.f90"
build test_assign f2008 "$test_dir/test_assign.f90"
# f08/0011: -std=f2008 rejects intrinsic assignment of a structure with a
# finalized component, which is what the container probe exercises.
build test_handle_container_assign f2018 "$test_dir/support.f90" \
  "$test_dir/test_handle_container_assign.f90"

pass=0
fail=0

under_valgrind() {
  local expected="$1"
  shift
  local rc
  set +e
  RGSADDLE_FORTRAN_VALGRIND=1 valgrind \
    --quiet \
    --error-exitcode=1 \
    --leak-check=full \
    --show-leak-kinds=definite \
    --errors-for-leak-kinds=definite \
    "$@"
  rc=$?
  set -e
  if [[ "$rc" -eq "$expected" ]]; then
    pass=$((pass + 1))
  else
    echo "FAIL valgrind ${*##*/} rc=$rc expected=$expected" >&2
    fail=$((fail + 1))
  fi
}

under_valgrind 0 "$work/test_handle_leaks.x"
under_valgrind 0 "$work/test_handles.x"
for kind in band minmode pafi ohtst; do
  under_valgrind 17 "$work/test_assign.x" "$kind"
  under_valgrind 17 "$work/test_handle_container_assign.x" "$kind"
done

echo "fortran-valgrind passed=$pass failed=$fail"
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
