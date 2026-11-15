#!/usr/bin/env bash
# Full reproduction of the paper from scratch, in a NEW directory (default ../clustaudit_reproduction).
# Same steps on every platform: they are declared once, in scripts/reproduce.py.
#   scripts/run_all_v2.sh [DIR] [options of `reproduce.py from-scratch`, e.g. --skip deep]
# The run stops at the first failed step; DIR/REPRODUCTION_REPORT.md says what was reproduced.
# To check the distributed results without re-running anything: scripts/verify_release.sh
set -euo pipefail
cd "$(dirname "$0")/.."
DIR="${1:-../clustaudit_reproduction}"; shift || true
exec python scripts/reproduce.py from-scratch --out "$DIR" "$@"
