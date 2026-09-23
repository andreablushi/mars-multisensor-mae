#!/usr/bin/env bash
# Fetch every published evaluation result that results/ does not hold yet.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src:scripts uv run --all-groups python -c '
from dhub.store import fetched_results

fetched = fetched_results()
for path in fetched:
    print(f"fetched {path}")
print(f"{len(fetched)} results fetched")
'
