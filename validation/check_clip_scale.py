"""Exact normalized scaling and three finite binary64 clipping witnesses."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

from pathlib import Path
import subprocess
import sys

import sympy as sp

x, y, radius, largest = sp.symbols("x y radius largest", positive=True)
length = sp.sqrt((x/largest)**2 + (y/largest)**2)
factor = (radius/largest)/length
assert sp.simplify(factor**2 * (x*x + y*y) - radius**2) == 0
print("EXACT_COMPONENT_SCALED_RADIUS")

result = subprocess.run(
    ["sollya", "--flush", str(Path(__file__).with_name("clip_scale.sollya"))],
    capture_output=True,
    text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
result.check_returncode()
if result.stdout.splitlines() != ["CERTIFIED_FINITE_SCALE_CLIP"] * 3 or result.stderr:
    raise SystemExit("Sollya did not certify all finite-scale clipping witnesses.")
