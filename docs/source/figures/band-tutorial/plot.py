"""Figures for the Mueller-Brown band tutorial.

Run from the repository root:

    cargo run --release --example muller_brown_band -- target/muller_brown
    uv run --no-project --with matplotlib --with numpy \
        python docs/source/figures/band-tutorial/plot.py target/muller_brown

Reads the CSV files the example wrote, prints the numbers the tutorial
quotes, and writes mep.png and force_calls.png next to this script.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PREFACTOR = [-200.0, -100.0, -170.0, 15.0]
XX = [-1.0, -1.0, -6.5, 0.7]
XY = [0.0, 0.0, 11.0, 0.6]
YY = [-10.0, -10.0, -6.5, 0.7]
X0 = [1.0, 0.0, -0.5, -1.0]
Y0 = [0.0, 0.5, 1.5, 1.0]
FORCE_TOL = 1e-3
METHODS = {"ci-neb": ("CI-NEB", "tab:blue"), "oci-neb": ("OCI-NEB", "tab:orange")}


def energy(x, y):
    return sum(
        a * np.exp(b * (x - x0) ** 2 + c * (x - x0) * (y - y0) + d * (y - y0) ** 2)
        for a, b, c, d, x0, y0 in zip(PREFACTOR, XX, XY, YY, X0, Y0)
    )


def hessian(x, y, h=1e-5):
    f = energy
    hxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
    hyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
    hxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h**2)
    return np.array([[hxx, hxy], [hxy, hyy]])


def read(path):
    with open(path) as handle:
        return {k: np.array([float(v) for v in col]) for k, *col in zip(*csv.reader(handle))}


def main(data: Path, here: Path) -> None:
    traces = {m: read(data / f"{m}_trace.csv") for m in METHODS}
    bands = {m: read(data / f"{m}_band.csv") for m in METHODS}
    start = read(data / "initial_band.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2))
    gx, gy = np.meshgrid(np.linspace(-1.5, 1.2, 300), np.linspace(-0.4, 2.0, 300))
    surface = left.contourf(gx, gy, np.minimum(energy(gx, gy), 100), levels=30, cmap="viridis")
    fig.colorbar(surface, ax=left, label="energy (clipped at 100)")
    left.plot(start["x"], start["y"], "w--", lw=1, label="start (straight line)")
    for m, (label, colour) in METHODS.items():
        band, trace = bands[m], traces[m]
        ci = int(trace["climbing_image"][-1])
        size = 7 if m == "ci-neb" else 4
        left.plot(band["x"], band["y"], "o-", color=colour, ms=size, label=label)
        right.plot(band["image"], band["energy"], "o-", color=colour, label=label)
        right.plot(ci, band["energy"][ci], "*", color="red", ms=14)
        h = np.linalg.eigvalsh(hessian(band["x"][ci], band["y"][ci]))
        print(
            f"{label}: climbing image {ci} at ({band['x'][ci]:.6f}, {band['y'][ci]:.6f}),"
            f" energy {band['energy'][ci]:.6f}, Hessian eigenvalues {h[0]:.2f} {h[1]:.2f}"
        )
    left.plot(band["x"][ci], band["y"][ci], "*", color="red", ms=14, label="climbing image")
    left.set(xlabel="x", ylabel="y", title="Mueller-Brown, A (top left) to B (bottom right)")
    left.legend(loc="lower left", fontsize=8)
    right.set(xlabel="image", ylabel="energy", title="Energy along the converged band")
    right.legend()
    fig.tight_layout()
    fig.savefig(here / "mep.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m, (label, colour) in METHODS.items():
        t = traces[m]
        calls, steps = t["force_calls"].astype(int), t["iteration"].astype(int)
        ax.semilogy(calls, t["top_gradient"], color=colour, label=f"{label}, highest image")
        settled = np.nonzero(t["top_gradient"] > FORCE_TOL)[0][-1] + 1
        print(
            f"{label}: {steps[-1]} steps, {calls[-1]} force calls; highest image under"
            f" {FORCE_TOL:g} from step {steps[settled]} ({calls[settled]} force calls)"
        )
    t = traces["ci-neb"]
    ax.semilogy(t["force_calls"], t["max_force"], ":", color="tab:blue", label="CI-NEB, band")
    armed = np.nonzero(t["climbing_image"] >= 0)[0][0]
    calls = int(t["force_calls"][armed])
    print(f"climbing image arms at step {int(t['iteration'][armed])} ({calls} force calls)")
    ax.axvline(calls, color="grey", lw=0.8)
    ax.text(calls * 1.05, 1e2, "climbing image arms", fontsize=8, color="grey")
    ax.axhline(FORCE_TOL, color="black", lw=0.8, ls="--", label="force_tol")
    ax.set(xscale="log", ylim=(1e-5, 1e3), xlabel="force calls", ylabel="force")
    ax.set_title("Convergence on Mueller-Brown")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(here / "force_calls.png", dpi=150)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(__file__).resolve().parent)
