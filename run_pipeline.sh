#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(pwd)/2026}"
N_WINDOW="${N_WINDOW:-200}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: SOURCE_ROOT does not exist: ${SOURCE_ROOT}" >&2
  exit 1
fi

export SOURCE_ROOT
export N_WINDOW

mkdir -p "${SOURCE_ROOT}/output/LightGBM"

python scripts/script5_export_outputs.py

metrics_path="${SOURCE_ROOT}/output/LightGBM/metrics_snapshot.json"
strategy_path="${SOURCE_ROOT}/output/LightGBM/strategy_params.txt"

if [[ ! -f "${metrics_path}" ]]; then
  echo "ERROR: metrics_snapshot.json missing at ${metrics_path}" >&2
  exit 1
fi

if [[ ! -f "${strategy_path}" ]]; then
  echo "ERROR: strategy_params.txt missing at ${strategy_path}" >&2
  exit 1
fi

as_of_date="$(
  python - <<'PY'
import json
from pathlib import Path
import os
source_root = os.environ.get("SOURCE_ROOT", ".")
path = Path(source_root) / "output" / "LightGBM" / "metrics_snapshot.json"
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("meta", {}).get("eval_base_date_max", ""))
PY
)"

if [[ -z "${as_of_date}" ]]; then
  echo "ERROR: could not determine as_of_date from metrics_snapshot.json" >&2
  exit 1
fi

matched_path="${SOURCE_ROOT}/output/LightGBM/local_matched_games_${as_of_date}.csv"
if [[ ! -f "${matched_path}" ]]; then
  echo "ERROR: local_matched_games output missing at ${matched_path}" >&2
  exit 1
fi

echo "Pipeline outputs ready in ${SOURCE_ROOT}/output/LightGBM"
