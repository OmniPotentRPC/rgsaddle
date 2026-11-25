#!/usr/bin/env bash
# Benchmark entry point for the full Cu V_iv cell-size campaign.
#
# The Mishin catalogue is generated once for every requested cell.  The
# network assembly writes both the field-free (without correction) and
# vector-dipole (with correction) estimators, so the two columns use the
# same atomistic saddles and statistical samples.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK:?}"' EXIT

export RGSADDLE_REPEATS="${RGSADDLE_REPEATS:-8,10,13}"
export RGSADDLE_TEMPS="${RGSADDLE_TEMPS:-300,500,700}"
export RUST_BACKTRACE=1

cd "${ROOT}"

cargo test --release --test mishin_cu defect_pair_cu_catalogue -- --ignored --exact \
    >"${WORK}/catalogue.log" 2>&1

target="${ROOT}/benchmarks/elastic_dipole_viv_series.csv"
printf '%s\n' \
    'cell_repeats,temperature_K,correction,estimator,V_iv_Omega,sigma_Omega,notes' \
    >"${target}"

IFS=',' read -r -a repeats_values <<<"${RGSADDLE_REPEATS}"
for repeats in "${repeats_values[@]}"; do
    if [[ ! "${repeats}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'invalid RGSADDLE_REPEATS entry: %s\n' "${repeats}" >&2
        exit 2
    fi
    RGSADDLE_ELASTIC_REPEATS="${repeats}" \
        cargo test --release --test mishin_cu defect_pair_network_volume -- --ignored --exact \
        >"${WORK}/volume-${repeats}.log" 2>&1

    python3 - "${ROOT}/benchmarks/defect_pair_network_volume.csv" \
        "${target}" "${repeats}" <<'PY'
import csv
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
repeats = sys.argv[3]
rows = []
with source.open(newline="") as handle:
    for row in csv.DictReader(handle):
        estimator = row.get("estimator", "")
        if estimator.startswith("vector-return:"):
            mode = "without"
        elif estimator.startswith("vector-dipole:"):
            mode = "with"
        else:
            continue
        rows.append(
            {
                "cell_repeats": repeats,
                "temperature_K": row["temperature_K"],
                "correction": mode,
                "estimator": estimator.split(":", 1)[1],
                "V_iv_Omega": row["volume_over_Omega"],
                "sigma_Omega": row["sigma"],
                "notes": row["note"],
            }
        )

with target.open("a", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "cell_repeats",
            "temperature_K",
            "correction",
            "estimator",
            "V_iv_Omega",
            "sigma_Omega",
            "notes",
        ],
    )
    writer.writerows(rows)

if not rows:
    raise SystemExit("network volume produced no with/without correction rows")
PY
done

printf '%s\n' "${ROOT}/benchmarks/elastic_dipole_viv_series.csv"
