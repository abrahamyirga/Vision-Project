#!/usr/bin/env bash
set -euxo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

pip install -r requirements.txt

python submission/code/project_run.py
python submission/code/evaluate_metrics.py
