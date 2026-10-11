#!/usr/bin/env python3
# Transcribe eOn resamplePath (client/NEBInitialPaths.cpp) at
# 2f84d583dd0c8c5a6a25d35b06fc2a1f3425c051. Arc length is the
# Euclidean chord. Endpoint tangents are the raw chords. An interior
# tangent is the mean of the two chords, normalised and scaled by
# half the chord length from the previous image to the next.

import json
from pathlib import Path

import numpy as np


def cubic_interpolate(p0, t0, p1, t1, f):
    f2 = f * f
    f3 = f2 * f
    h00 = 2 * f3 - 3 * f2 + 1
    h10 = f3 - 2 * f2 + f
    h01 = -2 * f3 + 3 * f2
    h11 = f3 - f2
    return h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1


def resample_path(dense, target_count):
    """Place target_count interior images. The result has target_count + 2 rows."""
    dense = np.asarray(dense, dtype=np.float64)
    n = dense.shape[0]
    if n == target_count + 2:
        return dense.copy()
    arc = np.zeros(n)
    chords = np.zeros_like(dense)
    for i in range(1, n):
        chords[i] = dense[i] - dense[i - 1]
        arc[i] = arc[i - 1] + np.linalg.norm(chords[i])
    total = arc[-1]
    tangents = np.zeros_like(dense)
    for i in range(n):
        if i == 0:
            tangents[i] = dense[1] - dense[0]
        elif i == n - 1:
            tangents[i] = dense[i] - dense[i - 1]
        else:
            d_next = dense[i + 1] - dense[i]
            d_prev = dense[i] - dense[i - 1]
            tangent = 0.5 * (d_next + d_prev)
            local = (arc[i + 1] - arc[i - 1]) / 2.0
            norm = np.linalg.norm(tangent)
            tangents[i] = tangent / norm * local if norm > 0.0 else tangent
    out = np.zeros((target_count + 2, dense.shape[1]))
    out[0] = dense[0]
    out[-1] = dense[-1]
    if total <= 0.0 or target_count == 0:
        return out
    segment = total / (target_count + 1)
    for i in range(1, target_count + 1):
        target = i * segment
        low = 0
        for j in range(1, n):
            if arc[j] >= target:
                low = j - 1
                break
        high = low + 1
        span = arc[high] - arc[low]
        frac = (target - arc[low]) / span if span > 1e-10 else 0.0
        out[i] = cubic_interpolate(
            dense[low], tangents[low], dense[high], tangents[high], frac
        )
    return out


def quarter_circle():
    angles = np.array([0.05, 0.35, 0.55, 0.9, 1.15, 1.45])
    dense = np.column_stack(
        [np.cos(angles), np.sin(angles), np.zeros_like(angles)]
    )
    packed = resample_path(dense, 9)
    return {
        "angles": angles.tolist(),
        "dense": dense.tolist(),
        "packed": packed.tolist(),
        "note": "6 window images packed to 11, cubic Hermite, eOn resamplePath",
    }


def main():
    payload = quarter_circle()
    path = Path(__file__).with_name("hermite_golden.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
