#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-$(pwd)/2026}"
LGBM_DIR="${LGBM_DIR:-${SOURCE_ROOT}/LightGBM}"
N_WINDOW="${N_WINDOW:-200}"

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "ERROR: SOURCE_ROOT does not exist: ${SOURCE_ROOT}" >&2
  exit 1
fi

export SOURCE_ROOT
export LGBM_DIR
export N_WINDOW

mkdir -p "${LGBM_DIR}"

metrics_path="${LGBM_DIR}/metrics_snapshot.json"
strategy_path="${LGBM_DIR}/strategy_params.txt"

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
path = Path(os.environ.get("LGBM_DIR", "."))
data = json.loads(path.read_text(encoding="utf-8"))
print(data.get("meta", {}).get("eval_base_date_max", ""))
PY
)"

if [[ -z "${as_of_date}" ]]; then
  echo "ERROR: could not determine as_of_date from metrics_snapshot.json" >&2
  exit 1
fi

matched_path="${LGBM_DIR}/local_matched_games_${as_of_date}.csv"
if [[ ! -f "${matched_path}" ]]; then
  echo "ERROR: local_matched_games output missing at ${matched_path}" >&2
  exit 1
fi
if [[ "$(wc -l < "${matched_path}")" -le 1 ]]; then
  echo "ERROR: local_matched_games output is empty at ${matched_path}" >&2
  exit 1
fi

echo "Pipeline outputs ready in ${LGBM_DIR}"
