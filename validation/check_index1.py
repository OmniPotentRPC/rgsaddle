#!/usr/bin/env python3
"""Compare public-ABI trajectories and paired index-one timings."""
import sys

if not __debug__:
    raise SystemExit("validation requires Python assertions; omit -O and PYTHONOPTIMIZE")

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess


def run(binary, *args):
    result = subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True)
    if result.stderr:
        raise RuntimeError(result.stderr.decode())
    return result.stdout


def measure(binary, size, repeats):
    fields = run(binary, "bench", size, repeats).decode().strip().split(",")
    assert len(fields) == 5, fields
    n, count, calls = map(int, fields[:3])
    elapsed, checksum = map(float, fields[3:])
    assert (n, count, calls) == (size, repeats, 4 * repeats), fields
    assert elapsed > 0, fields
    return {"seconds": elapsed, "checksum": checksum, "calls": calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = run(args.baseline.resolve(), "trace")
    candidate = run(args.candidate.resolve(), "trace")
    (args.output / "baseline.trace").write_bytes(baseline)
    (args.output / "candidate.trace").write_bytes(candidate)
    assert baseline and candidate, "empty trajectory"
    assert baseline == candidate, "trajectory, Hessian, report, or callback-count mismatch"
    print(f"128 cases, 768 steps: identical traces ({len(candidate)} bytes)", flush=True)
    report = {"trace_sha256": hashlib.sha256(candidate).hexdigest(), "cases": 128,
              "steps": 768, "dimensions": {}}
    for size, repeats in [(3, 1000), (15, 100), (75, 8), (150, 3), (300, 1)]:
        base_samples, candidate_samples = [], []
        measure(args.baseline.resolve(), size, 1)
        measure(args.candidate.resolve(), size, 1)
        for trial in range(7):
            order = [(args.baseline, base_samples), (args.candidate, candidate_samples)]
            if trial % 2:
                order.reverse()
            for binary, samples in order:
                samples.append(measure(binary.resolve(), size, repeats))
            assert base_samples[-1]["checksum"] == candidate_samples[-1]["checksum"], size
        old = statistics.median(sample["seconds"] for sample in base_samples)
        new = statistics.median(sample["seconds"] for sample in candidate_samples)
        report["dimensions"][size] = {"repeats": repeats, "baseline": base_samples,
                                      "candidate": candidate_samples, "median_ratio": new / old}
        (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"dimension {size}: median candidate/baseline = {new / old:.4f}", flush=True)


if __name__ == "__main__":
    main()
