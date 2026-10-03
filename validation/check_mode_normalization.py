"""Run the normalization identities and require every interval certificate."""

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "mode_normalization.py")], check=True)
result = subprocess.run(
    ["sollya", "--flush", str(root / "mode_normalization.sollya")],
    check=True,
    capture_output=True,
    text=True,
)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
expected = [f"CERTIFIED_MODE_NORM {n}" for n in (3, 75, 512, 4096)]
if lines != expected or result.stderr:
    raise SystemExit("Sollya did not emit exactly the four required certificates")
