#!/usr/bin/env bash
set -euxo pipefail

if [[ -n "${VIRTUAL_ENV}" ]]; then
  echo "Using active virtual environment: ${VIRTUAL_ENV}"
fi

pip install -r requirements.txt

python project_run.py
python evaluate_metrics.py
