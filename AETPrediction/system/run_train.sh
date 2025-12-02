#!/usr/bin/env bash
set -euo pipefail

# Always run from repo root regardless of where script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}/trainer"

python3 train.py "$@"

