"""Require the exact algebra and every interval sign certificate."""

if not __debug__:
    raise SystemExit("Symbolic validation requires Python assertions; disable optimization.")

from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "accepted_point.py")], check=True)
result = subprocess.run(["sollya", "--flush", str(root / "accepted_point.sollya")], capture_output=True, text=True)
print(result.stdout, end="")
if result.stderr:
    print(result.stderr, file=sys.stderr, end="")
result.check_returncode()
expected = ["CERTIFIED_MINIMUM_OPENING_NEGATIVE", "CERTIFIED_MINIMUM_ACCEPTED_POSITIVE", "CERTIFIED_SADDLE_OPENING_POSITIVE", "CERTIFIED_SADDLE_ACCEPTED_NEGATIVE"]
if [line.strip() for line in result.stdout.splitlines() if line.strip()] != expected or result.stderr:
    raise SystemExit("Sollya did not emit exactly the four required sign certificates")
