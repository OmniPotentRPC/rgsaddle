#!/usr/bin/env bash
# Machine-check the formal contracts.
#   1. lake build the library (Mathlib from the pinned cache);
#   2. refuse `sorry`, `admit` or an `axiom` declaration in the sources;
#   3. print the axioms of every theorem and refuse any beyond the three
#      Lean/Mathlib foundations (propext, Classical.choice, Quot.sound).
# SKIP_CACHE=1 skips `lake exe cache get` when the oleans are present.
set -euo pipefail
cd "$(dirname "$0")"
lib=$(sed -n 's/^name = "\(.*\)"$/\1/p' lakefile.toml | head -n 1)

if [[ "${SKIP_CACHE:-0}" != 1 ]]; then
  lake exe cache get 2>&1 | tail -n 2
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
lake build 2>&1 | tee "$work/build.log"

status=0
if grep -rnwE 'sorry|admit' --include='*.lean' --exclude-dir=.lake . ||
  grep -rnE '^\s*axiom\b' --include='*.lean' --exclude-dir=.lake . ; then
  echo "check.sh: sorry, admit or axiom in the sources" >&2
  status=1
fi
if grep -nE "declaration uses 'sorry'" "$work/build.log"; then
  echo "check.sh: the build reports a sorry" >&2
  status=1
fi

{
  echo "import $lib"
  grep -hoE '^(theorem|lemma) [A-Za-z_0-9.]+' "$lib"/*.lean |
    awk -v ns="$lib" '{print "#print axioms " ns "." $2}'
} > "$work/Audit.lean"
lake env lean "$work/Audit.lean" > "$work/axioms.log" 2>&1 || {
  cat "$work/axioms.log"
  echo "check.sh: the axiom audit failed to elaborate" >&2
  exit 1
}
n=$(grep -c "depends on axioms\|does not depend on any axioms" "$work/axioms.log" || true)
bad=$(grep -oE 'depends on axioms: \[[^]]*\]' "$work/axioms.log" |
  sed 's/.*\[//; s/\]//' | tr ',' '\n' | tr -d ' ' |
  grep -vxE 'propext|Classical\.choice|Quot\.sound' | sort -u || true)
if grep -q 'sorryAx' "$work/axioms.log"; then
  bad="sorryAx $bad"
fi
if [[ -n "$bad" ]]; then
  grep -E 'depends on axioms' "$work/axioms.log" >&2
  echo "check.sh: theorems rest on axioms beyond the foundations: $bad" >&2
  status=1
fi
echo "check.sh: $n theorems audited, status $status"
exit "$status"
