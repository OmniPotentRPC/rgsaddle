#!/bin/sh
# Regenerate bench/raw/neb_survey.csv. The rows are generated output.
set -eu
cd "$(dirname "$0")/.."
mkdir -p bench/raw
cargo run --release --example neb_parity
