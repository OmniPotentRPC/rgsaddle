#!/usr/bin/env python3
"""Check the public index-1 ABI against the pinned trajectory."""
import sys

if not __debug__:
    raise SystemExit("validation requires Python assertions; omit -O and PYTHONOPTIMIZE")

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "index1_abi.json"


def run(binary, *args):
    result = subprocess.run(
        [str(binary), *map(str, args)],
        check=True,
        capture_output=True,
    )
    if result.stderr:
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def measure(binary, size, repeats):
    raw = run(binary, "bench", size, repeats).decode().strip()
    fields = raw.split(",")
    assert len(fields) == 5, fields
    n, count, calls = map(int, fields[:3])
    elapsed = float(fields[3])
    assert (n, count, calls) == (size, repeats, 4 * repeats), fields
    assert elapsed > 0, fields
    return {"seconds": elapsed, "checksum": fields[4], "calls": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    golden = json.loads(GOLDEN_PATH.read_text())
    assert golden["cases"] == 128 and golden["steps"] == 768, golden
    assert [row["dimension"] for row in golden["bench"]] == [3, 15, 75, 150, 300]
    args.output.mkdir(parents=True, exist_ok=True)
    trace = run(args.probe.resolve(), "trace")
    (args.output / "trace.bin").write_bytes(trace)
    digest = hashlib.sha256(trace).hexdigest()
    assert trace, "empty trajectory"
    assert len(trace) == golden["trace_bytes"], len(trace)
    assert digest == golden["trace_sha256"], digest
    print(
        f"{golden['cases']} cases, {golden['steps']} steps: "
        f"trace matches ({len(trace)} bytes)",
        flush=True,
    )
    report = {
        "trace_sha256": digest,
        "cases": golden["cases"],
        "steps": golden["steps"],
        "dimensions": {},
    }
    for row in golden["bench"]:
        size = row["dimension"]
        repeats = row["repeats"]
        assert row["calls"] == 4 * repeats, row
        sample = measure(args.probe.resolve(), size, repeats)
        assert sample["checksum"] == row["checksum"], (size, sample["checksum"], row["checksum"])
        assert sample["calls"] == row["calls"], (size, sample["calls"])
        report["dimensions"][str(size)] = {
            "repeats": repeats,
            "seconds": sample["seconds"],
            "checksum": sample["checksum"],
            "calls": sample["calls"],
        }
        label = "repeat" if repeats == 1 else "repeats"
        print(
            f"dimension {size}: checksum matches "
            f"({repeats} {label}, {sample['seconds']:.4f}s)",
            flush=True,
        )
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
