#!/usr/bin/env bash
# Check the distributed results (tests, completeness and integrity of the result files, every
# number of the paper, every table regenerated identically). No experiment is re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
exec python scripts/reproduce.py verify-release "$@"
