# Copyright (c) rgsaddle developers
"""Figures and numbers for the band and min-mode scaling runs.

Run from the repository root after run.py:

    uv run --no-project --with matplotlib --with numpy \
        python docs/source/figures/band-scaling/plot.py <out-dir>

Reads <out-dir>/runs.csv, prints one row per point and writes
summary.csv to <out-dir> and strong.png, weak.png and serial.png next
to this script. Times are seconds per step. Each point has two arms
of 2 x blocks runs; a pair is two neighbouring runs of one block (A B
and B A), and its ratio is the baseline time over the change time.
The speedup is the median of those ratios, the IQR is that of the
ratios, and a win is a pair whose change ran faster.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
INTERIOR = 16


def quartiles(values):
    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return q2, q1, q3


def load(out: Path):
    points = defaultdict(list)
    with open(out / "runs.csv") as table:
        for row in csv.DictReader(table):
            points[(row["series"], int(row["point"]))].append(row)
    return points


def summarise(rows):
    per_step = {}
    for row in rows:
        key = (int(row["block"]), int(row["slot"]))
        per_step[key] = float(row["seconds"]) / int(row["steps"])
    blocks = sorted({b for b, _ in per_step})
    ratios = []
    for b in blocks:
        ratios.append(per_step[(b, 0)] / per_step[(b, 1)])
        ratios.append(per_step[(b, 3)] / per_step[(b, 2)])
    arm = {
        build: [per_step[(int(r["block"]), int(r["slot"]))]
                for r in rows if r["build"] == build]
        for build in ("baseline", "change")
    }
    first = {build: next(r for r in rows if r["build"] == build)
             for build in arm}
    work = {build: first[build]["rows"] or first[build]["evaluations"]
            for build in arm}
    return {
        "baseline": quartiles(arm["baseline"]),
        "change": quartiles(arm["change"]),
        "speedup": quartiles(ratios),
        "wins": sum(r > 1 for r in ratios),
        "pairs": len(ratios),
        "work": work,
        "args": {build: first[build]["args"] for build in arm},
        "steps": first["change"]["steps"],
    }


def amdahl(threads, times):
    """Least-squares s + p / T; returns (s, p)."""
    a = np.column_stack([np.ones(len(threads)), 1.0 / np.asarray(threads)])
    (s, p), *_ = np.linalg.lstsq(a, np.asarray(times), rcond=None)
    return s, p


def errorbar(ax, x, stats, **style):
    mid = np.array([s[0] for s in stats])
    low = mid - np.array([s[1] for s in stats])
    high = np.array([s[2] for s in stats]) - mid
    ax.errorbar(x, mid, yerr=[low, high], capsize=3, marker="o", **style)


def strong(summary, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, atoms in zip(axes, (64, 512)):
        series = f"strong-{atoms}"
        threads = sorted(t for s, t in summary if s == series)
        stats = [summary[(series, t)]["speedup"] for t in threads]
        errorbar(ax, threads, stats, label="change over serial baseline")
        one = stats[0][0]
        ax.plot(threads, [one * t for t in threads], "k--", lw=1,
                label="linear from one thread")
        bound = [one * INTERIOR / math.ceil(INTERIOR / t) for t in threads]
        ax.plot(threads, bound, "k:", lw=1,
                label="block bound, 16 images")
        ax.set_title(f"Strong scaling, {atoms} atoms, 16 interior images")
        ax.set_xlabel("image threads")
        ax.set_xticks(threads)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("speedup (median of paired ratios, IQR)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "strong.png", dpi=150)


def weak(summary, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    titles = {
        "weak-images": "Weak scaling: 256 atoms, 2T interior images",
        "weak-atoms": "Weak scaling: 8 images, 128 sqrt(T) atoms",
    }
    for ax, series in zip(axes, titles):
        threads = sorted(t for s, t in summary if s == series)
        for build, label in (("baseline", "baseline, one thread"),
                             ("change", "change, T image threads")):
            stats = [tuple(1e3 * v for v in summary[(series, t)][build])
                     for t in threads]
            errorbar(ax, threads, stats, label=label)
        ax.set_title(titles[series])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(threads, [str(t) for t in threads])
        ax.set_xlabel("T (threads of the change)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("ms per band step (median, IQR)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "weak.png", dpi=150)


def serial(summary, out: Path):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels, stats = [], []
    for series in ("rtr", "dimer", "lanczos"):
        for atoms in sorted(a for s, a in summary if s == series):
            labels.append(f"{series}\n{atoms} atoms")
            stats.append(summary[(series, atoms)]["speedup"])
    x = np.arange(len(labels))
    errorbar(ax, x, stats, linestyle="none")
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("speedup, one thread (median, IQR)")
    ax.set_title("Serial runs: RTR band and min-mode searches")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(HERE / "serial.png", dpi=150)


def main() -> None:
    out = Path(sys.argv[1])
    summary = {key: summarise(rows) for key, rows in load(out).items()}
    fields = ["series", "point", "steps", "baseline_ms", "baseline_iqr_ms",
              "change_ms", "change_iqr_ms", "speedup", "speedup_iqr",
              "wins", "pairs", "baseline_work", "change_work"]
    with open(out / "summary.csv", "w", newline="") as table:
        writer = csv.writer(table)
        writer.writerow(fields)
        for (series, point), s in summary.items():
            b, c, r = s["baseline"], s["change"], s["speedup"]
            row = [series, point, s["steps"],
                   f"{1e3 * b[0]:.4f}", f"{1e3 * b[1]:.4f}-{1e3 * b[2]:.4f}",
                   f"{1e3 * c[0]:.4f}", f"{1e3 * c[1]:.4f}-{1e3 * c[2]:.4f}",
                   f"{r[0]:.3f}", f"{r[1]:.3f}-{r[2]:.3f}",
                   s["wins"], s["pairs"],
                   s["work"]["baseline"], s["work"]["change"]]
            writer.writerow(row)
            print(" ".join(map(str, row)))
    for atoms in (64, 512):
        series = f"strong-{atoms}"
        threads = sorted(t for s, t in summary if s == series)
        times = [summary[(series, t)]["change"][0] for t in threads]
        s, p = amdahl(threads, times)
        print(f"{series}: fit s + p/T, s = {1e3 * s:.4f} ms, "
              f"p = {1e3 * p:.4f} ms, serial share {s / (s + p):.3f}")
    strong(summary, out)
    weak(summary, out)
    serial(summary, out)


if __name__ == "__main__":
    main()
