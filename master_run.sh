#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/2026}"
LGBM_DIR="${LGBM_DIR:-${SOURCE_ROOT}/LightGBM}"
N_WINDOW="${N_WINDOW:-200}"
STRATEGY_VARIANT="${STRATEGY_VARIANT:-acc}"

export SOURCE_ROOT
export LGBM_DIR
export N_WINDOW
export STRATEGY_VARIANT

python "${SOURCE_ROOT}/src/1_get_data_previous_game_day_2026.py"
python "${SOURCE_ROOT}/src/2_get_data_next_game_day_2026.py"
python "${SOURCE_ROOT}/src/3_predict_games_hybrid_2026.py"
python "${SOURCE_ROOT}/src/4_calculate_betting_statistics_2026.py"
python "${SOURCE_ROOT}/src/5_Isotonic_based_betting_strategy_2026.py"

bash run_pipeline.sh
