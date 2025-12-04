#!/usr/bin/env bash
set -euxo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

python3 -m pip install -r requirements.txt

python3 submission/code/project_run.py
python3 submission/code/evaluate_metrics.py
