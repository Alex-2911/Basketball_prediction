from __future__ import annotations
import json,re,os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
DATE_FMT = '%Y-%m-%d'
STAKE = 100.0
_AGENT_TEAM_ALIASES = {
    'ATL': 'ATL',
    'BOS': 'BOS',
    'BRK': 'BRK',
    'BKN': 'BRK',
    'BROOKLYN NETS': 'BRK',
    'CHO': 'CHO',
    'CHA': 'CHO',
    'CHI': 'CHI',
    'CLE': 'CLE',
    'DAL': 'DAL',
    'DEN': 'DEN',
    'DET': 'DET',
    'GSW': 'GSW',
    'HOU': 'HOU',
    'IND': 'IND',
    'LAC': 'LAC',
    'LAL': 'LAL',
    'MEM': 'MEM',
    'MIA': 'MIA',
    'MIL': 'MIL',
    'MILWAUKEE BUCKS': 'MIL',
    'MIN': 'MIN',
    'NOP': 'NOP',
    'NYK': 'NYK',
    'OKC': 'OKC',
    'ORL': 'ORL',
    'PHI': 'PHI',
    'PHX': 'PHX',
    'POR': 'POR',
    'SAC': 'SAC',
    'SAS': 'SAS',
    'TOR': 'TOR',
    'UTA': 'UTA',
    'UTAH JAZZ': 'UTA',
    'WAS': 'WAS',
    'WASHINGTON WIZARDS': 'WAS',
    'LAKERS': 'LAL',
    'LALAKERS': 'LAL',
    'LA LAKERS': 'LAL',
    'CLIPPERS': 'LAC',
    'LA CLIPPERS': 'LAC',
    'CAVALIERS': 'CLE',
    'CAVS': 'CLE',
    'KNICKS': 'NYK',
    'NEW YORK KNICKS': 'NYK',
    '76ERS': 'PHI',
    'SIXERS': 'PHI',
    'MAGIC': 'ORL',
    'HAWKS': 'ATL',
    'TIMBERWOLVES': 'MIN',
    'WOLVES': 'MIN',
    'NUGGETS': 'DEN',
    'SPURS': 'SAS',
    'SAN ANTONIO SPURS': 'SAS',
    'PISTONS': 'DET',
    'HORNETS': 'CHO',
    'HEAT': 'MIA',
    'PELICANS': 'NOP',
    'BULLS': 'CHI',
    'SUNS': 'PHX',
    'MAVERICKS': 'DAL',
    'WARRIORS': 'GSW',
    'CELTICS': 'BOS',
    'TRAIL BLAZERS': 'POR',
    'BLAZERS': 'POR',
    'RAPTORS': 'TOR',
    'ROCKETS': 'HOU',
    'PACERS': 'IND',
    'KINGS': 'SAC',
    'THUNDER': 'OKC',
    'GRIZZLIES': 'MEM',
    'MEMPHIS GRIZZLIES': 'MEM',
}
frames = []
required_setup_names = [
    'load_combined_predictions',
    'load_bet_log',
    'load_pregame_snapshots',
    'load_script11_decision_artifacts',
    'output_dir',
    'bet_log_path',
    'strategy_shortlist_keys',
]
pregame_snapshot_rows_matched = 0
script11_rows_matched = 0
script11_cols = [
    'script11_artifact_file', 'script11_artifact_path', 'script11_artifact_source_date',
    'script11_engine_state', 'script11_decision', 'script11_stage2_candidate_type',
    'script11_blocked_by', 'script11_rules_passed', 'script11_ev_per_100',
    'script11_prob_used', 'script11_odds_1', 'script11_home_win_rate',
    'script11_pass_hw', 'script11_pass_odds', 'script11_pass_prob', 'script11_pass_ev',
    'script11_canonical_signal', 'script11_allowed_review_type', 'script11_source',
    'script11_params_chosen',
]
agent_training_rows_matched = 0
agent_training_rows_unmatched = 0
agent_training_columns = [
    'agent_training_matched',
    'agent_training_match_method',
    'agent_training_match_score',
    'game_date',
    'created_at',
    'matchup',
    'home_team',
    'away_team',
    'canonical_decision',
    'watchlist_decision',
    'actual_action',
    'decision_class',
    'blocked_by',
    'stake',
    'odds_taken',
    'outcome',
    'result',
    'pnl',
    'avoided_loss_eur',
    'avoided_loss_basis',
    'vibe_class',
    'confidence_tag',
    'recommended_action',
    'recommended_market',
    'lesson',
    'agent_label',
    'homeWinRate',
    'hwrThreshold',
    'oddsHome',
    'oddsAway',
    'probIso',
    'probLiveProxy',
    'probBase',
    'probUsed',
    'marketImpliedRaw',
    'marketImpliedDevig',
    'modelMarketGap',
    'evBasePer100',
    'evLivePer100',
    'rulesPassed',
    'modelMarketGapFlag',
    'stakeCapEur',
    'recommendedConditions',
    'engineState',
    'createdAt',
    'source_file',
    'source_path',
    'source_line_no',
    'source_type',
    'matchup_text_key',
    'matchup_pair_key',
]
agent_export_cols = [
    'date', 'game_date', 'created_at', 'matchup', 'home_team', 'away_team', 'game_key',
    'actual_winner', 'actual_home_win', 'model_selected_side', 'model_selected_team', 'home_team_prob', 'odds_home', 'odds_away',
    'strategy_qualified', 'user_bet_placed', 'user_bet_side', 'user_bet_team', 'user_bet_market', 'user_bet_odds', 'user_stake', 'user_won', 'user_pnl',
    'pregame_snapshot_file', 'pregame_snapshot_source_type', 'pregame_snapshot_available',
    'agent_training_matched', 'agent_training_match_method', 'agent_training_match_score',
    'canonical_decision', 'watchlist_decision', 'actual_action', 'decision_class', 'blocked_by',
    'stake', 'odds_taken', 'outcome', 'result', 'pnl', 'avoided_loss_eur', 'avoided_loss_basis',
    'vibe_class', 'confidence_tag', 'recommended_action', 'recommended_market', 'lesson', 'agent_label',
    'homeWinRate', 'hwrThreshold', 'oddsHome', 'oddsAway', 'probIso', 'probLiveProxy', 'probBase', 'probUsed',
    'marketImpliedRaw', 'marketImpliedDevig', 'modelMarketGap', 'evBasePer100', 'evLivePer100', 'rulesPassed',
    'modelMarketGapFlag', 'stakeCapEur', 'recommendedConditions', 'engineState', 'createdAt', 'source_file', 'source_path', 'source_line_no', 'source_type',
]
OCT_STEM = 'october_2025_walk_forward_replay'
OCT_REPLAY_MODE = 'october_2025_walk_forward_model_only'
replayed_days = 0
days_with_non_empty_prediction_file = 0
missing_prediction_file_days = 0
empty_prediction_file_days = 0
prediction_rows_not_for_game_day_days = 0
days_with_missing_shortlist = 0
no_games_scheduled_days = 0
STAKE = 100.0
required = ["date", "home_team", "away_team", "odds_1", "odds_2"]
prob_candidates = ["prob_used", "prob_live_safe", "prob_iso_oos_time", "prob_iso", "home_team_prob"]
summary_rows = []
cols = [
    "date",
    "home_team",
    "away_team",
    "odds_1",
    "odds_2",
    "home_prob",
    "away_prob",
    "away_ev_100",
    "home_team_won",
    "away_won",
    "away_flat_100_pnl",
    "away_odds_bucket",
    "away_prob_bucket",
    "away_value_label",
]
grid_rows = []

def _normalize_team(team: object) -> str:
    if team is None:
        return ''
    s = str(team).strip().upper()
    if not s or s in {'NAN', 'NONE'}:
        return ''
    aliases = {
        'PHO': 'PHX',
        'PHX': 'PHX',
        'BKN': 'BRK',
        'BRK': 'BRK',
        'CHA': 'CHO',
        'CHO': 'CHO',
        'WSH': 'WAS',
        'WAS': 'WAS',
        'GS': 'GSW',
        'GSW': 'GSW',
        'NO': 'NOP',
        'NOP': 'NOP',
        'NY': 'NYK',
        'NYK': 'NYK',
        'SA': 'SAS',
        'SAS': 'SAS',
        'UTAH': 'UTA',
        'UTA': 'UTA',
        'OKL': 'OKC',
        'OKC': 'OKC',
    }
    return aliases.get(s, s)

def _safe_date(val: object) -> Optional[pd.Timestamp]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in {'0', 'nan', 'NaN', 'None'}:
        return None
    ts = pd.to_datetime(s, errors='coerce')
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()

def _game_key(date_val: object, home_team: object, away_team: object) -> str:
    d = _safe_date(date_val)
    if d is None:
        return ''
    return f'{d.strftime(DATE_FMT)}|{_normalize_team(home_team)}|{_normalize_team(away_team)}'

def _is_blank(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    s = str(value).strip()
    return s == '' or s.lower() in {'nan', 'none', 'null'}

def _first_nonblank(*values: object) -> object:
    for value in values:
        if not _is_blank(value):
            return value
    return None

def _normalize_team_phrase(team: object) -> str:
    if team is None:
        return ''
    s = re.sub(r'[^A-Z0-9 ]+', ' ', str(team).strip().upper())
    s = re.sub(r'\s+', ' ', s).strip()
    if not s:
        return ''
    compact = s.replace(' ', '')
    for key in (s, compact):
        if key in _AGENT_TEAM_ALIASES:
            return _AGENT_TEAM_ALIASES[key]
    tokens = s.split()
    for tok in reversed(tokens):
        if tok in _AGENT_TEAM_ALIASES:
            return _AGENT_TEAM_ALIASES[tok]
    return _normalize_team(s)

def _normalize_matchup_text(text: object) -> str:
    if _is_blank(text):
        return ''
    s = str(text).strip().upper()
    s = s.replace('VS.', ' VS ')
    s = s.replace(' VS ', ' VS ')
    s = s.replace('@', ' @ ')
    s = s.replace(' AT ', ' @ ')
    s = re.sub(r'[^A-Z0-9@ ]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _extract_matchup_teams(text: object) -> tuple[str, str]:
    s = _normalize_matchup_text(text)
    if not s:
        return '', ''
    if ' @ ' in s:
        away_raw, home_raw = [p.strip() for p in s.split(' @ ', 1)]
        return _normalize_team_phrase(home_raw), _normalize_team_phrase(away_raw)
    if ' VS ' in s:
        home_raw, away_raw = [p.strip() for p in s.split(' VS ', 1)]
        return _normalize_team_phrase(home_raw), _normalize_team_phrase(away_raw)
    tokens = [tok for tok in re.findall(r'[A-Z]{2,4}', s) if tok not in {'VS', 'AT'}]
    if len(tokens) >= 2:
        return _normalize_team_phrase(tokens[0]), _normalize_team_phrase(tokens[1])
    return '', ''

def _agent_team_pair_key(home_team: object, away_team: object) -> str:
    home = _normalize_team_phrase(home_team)
    away = _normalize_team_phrase(away_team)
    if not home or not away:
        return ''
    return '|'.join(sorted([home, away]))

def _nested_pick(record: dict, *paths: tuple[str, ...]) -> object:
    sources = [
        record,
        record.get('input_context') if isinstance(record.get('input_context'), dict) else {},
        record.get('decision_taken') if isinstance(record.get('decision_taken'), dict) else {},
        record.get('outcome') if isinstance(record.get('outcome'), dict) else {},
        record.get('agent_label') if isinstance(record.get('agent_label'), dict) else {},
    ]
    for path in paths:
        if isinstance(path, str):
            path = (path,)
        for source in sources:
            current = source
            ok = True
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    ok = False
                    break
                current = current.get(key)
            if ok and not _is_blank(current):
                return current
    return None

def _agent_to_float(value: object) -> float:
    if _is_blank(value):
        return np.nan
    s = str(value).strip().replace(',', '.')
    try:
        return float(s)
    except Exception:
        return np.nan

def _agent_to_text(value: object) -> object:
    if _is_blank(value):
        return pd.NA
    return str(value).strip()

def _normalize_agent_training_record(record: dict, source_path: Path, line_no: int) -> dict:
    date_val = _nested_pick(record, ('date',), ('game_date',), ('replay_date',), ('input_context', 'date'), ('input_context', 'game_date'), ('input_context', 'replay_date'))
    matchup_val = _nested_pick(record, ('matchup',), ('game',), ('input_context', 'matchup'), ('input_context', 'game'))
    home_team_val = _nested_pick(record, ('home_team',), ('homeTeam',), ('input_context', 'home_team'), ('input_context', 'homeTeam'))
    away_team_val = _nested_pick(record, ('away_team',), ('awayTeam',), ('input_context', 'away_team'), ('input_context', 'awayTeam'))
    parsed_home, parsed_away = _extract_matchup_teams(matchup_val)
    home_team = _normalize_team_phrase(_first_nonblank(home_team_val, parsed_home))
    away_team = _normalize_team_phrase(_first_nonblank(away_team_val, parsed_away))
    date_ts = pd.to_datetime(date_val, errors='coerce')
    if pd.notna(date_ts):
        date_ts = pd.Timestamp(date_ts).normalize()
    row = {
        'date': date_ts,
        'game_date': date_ts,
        'created_at': pd.to_datetime(_nested_pick(record, ('created_at',), ('createdAt',), ('input_context', 'created_at'), ('input_context', 'createdAt')), errors='coerce'),
        'matchup': _agent_to_text(_first_nonblank(matchup_val, f'{home_team} vs {away_team}' if home_team and away_team else None)),
        'home_team': home_team,
        'away_team': away_team,
        'game_key': _game_key(date_ts, home_team, away_team),
        'matchup_text_key': _normalize_matchup_text(_first_nonblank(matchup_val, f'{home_team} vs {away_team}' if home_team and away_team else None)),
        'matchup_pair_key': _agent_team_pair_key(home_team, away_team),
        'source_file': source_path.name,
        'source_path': str(source_path),
        'source_line_no': line_no,
        'source_type': 'jsonl',
        'engine_state': _agent_to_text(_nested_pick(record, ('engine_state',), ('input_context', 'engine_state'))),
        'chosen_strategy': _agent_to_text(_nested_pick(record, ('chosen_strategy',), ('input_context', 'chosen_strategy'))),
        'params_used_json': _agent_to_text(_nested_pick(record, ('params_used_json',), ('input_context', 'params_used_json'), ('params_used',), ('input_context', 'params_used'))),
        'canonical_decision': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'canonical_decision'), ('canonical_decision',), ('decision_taken', 'canonical_decision'), ('input_context', 'canonical_decision'))),
        'watchlist_decision': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'watchlist_decision'), ('watchlist_decision',), ('decision_taken', 'watchlist_decision'), ('input_context', 'watchlist_decision'))),
        'actual_action': _agent_to_text(_nested_pick(record, ('actual_action',), ('decision_taken', 'actual_action'), ('input_context', 'actual_action'))),
        'decision_class': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'decision_class'), ('decision_class',), ('input_context', 'decision_class'), ('agent_label', 'decision_class'), ('agent_label', 'ideal_action'))),
        'blocked_by': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'blocked_by'), ('blocked_by',), ('input_context', 'blocked_by'))),
        'stake': _agent_to_float(_nested_pick(record, ('stake',), ('decision_taken', 'stake'), ('input_context', 'stake'), ('input_context', 'stake_cap_eur'))),
        'odds_taken': _agent_to_float(_nested_pick(record, ('input_context', 'market_state', 'odds_taken'), ('odds_taken',), ('decision_taken', 'odds_taken'), ('input_context', 'odds_taken'))),
        'outcome': _agent_to_text(_nested_pick(record, ('outcome', 'result'), ('decision_taken', 'outcome'), ('input_context', 'outcome'), ('result',), ('status',))),
        'result': _agent_to_text(_nested_pick(record, ('outcome', 'result'), ('decision_taken', 'outcome'), ('input_context', 'outcome'), ('result',), ('status',))),
        'pnl': _agent_to_float(_nested_pick(record, ('pnl',), ('decision_taken', 'pnl'), ('outcome', 'pnl'), ('outcome', 'net_profit'), ('input_context', 'pnl'))),
        'avoided_loss_eur': _agent_to_float(_nested_pick(record, ('input_context', 'user_constraints', 'avoided_loss_eur'), ('avoided_loss_eur',), ('avoidedLossEur',), ('input_context', 'avoidedLossEur'), ('input_context', 'avoided_loss_eur'))),
        'avoided_loss_basis': _agent_to_text(_nested_pick(record, ('input_context', 'user_constraints', 'avoided_loss_basis'), ('avoided_loss_basis',), ('avoidedLossBasis',), ('input_context', 'avoidedLossBasis'), ('input_context', 'avoided_loss_basis'))),
        'vibe_class': _agent_to_text(_nested_pick(record, ('input_context', 'vibe_context', 'vibe_class'), ('vibe_class',), ('input_context', 'vibe_class'))),
        'confidence_tag': _agent_to_text(_nested_pick(record, ('input_context', 'vibe_context', 'confidence_tag'), ('confidence_tag',), ('input_context', 'confidence_tag'), ('agent_label', 'confidence'), ('agent_label', 'confidence_tag'))),
        'recommended_action': _agent_to_text(_nested_pick(record, ('input_context', 'market_state', 'recommended_conditions'), ('input_context', 'vibe_context', 'live_conditions'), ('recommended_action',), ('input_context', 'recommended_action'), ('agent_label', 'ideal_action'))),
        'recommended_market': _agent_to_text(_nested_pick(record, ('input_context', 'market_state', 'recommended_market'), ('recommended_market',), ('input_context', 'recommended_market'), ('agent_label', 'allowed_bet_type'))),
        'lesson': _agent_to_text(_nested_pick(record, ('outcome', 'post_game_lesson'), ('lesson',), ('post_game_reflection',), ('input_context', 'post_game_reflection'), ('agent_label', 'why'), ('manual_note',))),
        'homeWinRate': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'home_win_rate'), ('homeWinRate',), ('home_win_rate',), ('input_context', 'homeWinRate'), ('input_context', 'home_win_rate'))),
        'hwrThreshold': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'hwr_threshold'), ('hwrThreshold',), ('home_win_rate_threshold',), ('input_context', 'hwrThreshold'), ('input_context', 'home_win_rate_threshold'))),
        'oddsHome': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'odds_home'), ('oddsHome',), ('odds_1',), ('input_context', 'oddsHome'), ('input_context', 'odds_1'))),
        'oddsAway': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'odds_away'), ('oddsAway',), ('odds_2',), ('input_context', 'oddsAway'), ('input_context', 'odds_2'))),
        'probIso': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'prob_iso'), ('probIso',), ('prob_iso',), ('input_context', 'probIso'), ('input_context', 'prob_iso'))),
        'probLiveProxy': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'prob_live_proxy'), ('probLiveProxy',), ('prob_live_oos_proxy',), ('input_context', 'probLiveProxy'), ('input_context', 'prob_live_oos_proxy'))),
        'probBase': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'prob_base'), ('probBase',), ('prob_base',), ('input_context', 'probBase'), ('input_context', 'prob_base'))),
        'probUsed': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'prob_used'), ('probUsed',), ('prob_used',), ('input_context', 'probUsed'), ('input_context', 'prob_used'))),
        'marketImpliedRaw': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'market_implied_raw'), ('marketImpliedRaw',), ('market_implied_raw',), ('input_context', 'marketImpliedRaw'), ('input_context', 'market_implied_raw'))),
        'marketImpliedDevig': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'market_implied_devig'), ('marketImpliedDevig',), ('market_implied_devig',), ('input_context', 'marketImpliedDevig'), ('input_context', 'market_implied_devig'))),
        'modelMarketGap': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'model_market_gap'), ('modelMarketGap',), ('model_market_gap',), ('input_context', 'modelMarketGap'), ('input_context', 'model_market_gap'))),
        'evBasePer100': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'ev_base_per_100'), ('evBasePer100',), ('EV_base_€_per_100',), ('ev_base_per_100',), ('input_context', 'EV_base_€_per_100'), ('input_context', 'evBasePer100'))),
        'evLivePer100': _agent_to_float(_nested_pick(record, ('input_context', 'model_numbers', 'ev_live_per_100'), ('evLivePer100',), ('EV_live_€_per_100',), ('ev_live_per_100',), ('input_context', 'EV_live_€_per_100'), ('input_context', 'evLivePer100'))),
        'rulesPassed': _agent_to_float(_nested_pick(record, ('input_context', 'rule_state', 'rules_passed'), ('rulesPassed',), ('rules_passed',), ('input_context', 'rulesPassed'), ('input_context', 'rules_passed'))),
        'modelMarketGapFlag': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'blocked_by'), ('modelMarketGapFlag',), ('model_market_gap_flag',), ('input_context', 'modelMarketGapFlag'), ('input_context', 'model_market_gap_flag'))),
        'stakeCapEur': _agent_to_float(_nested_pick(record, ('input_context', 'market_state', 'stake_cap_eur'), ('stakeCapEur',), ('stake_cap_eur',), ('input_context', 'stakeCapEur'), ('input_context', 'stake_cap_eur'))),
        'recommendedConditions': _agent_to_text(_nested_pick(record, ('input_context', 'market_state', 'recommended_conditions'), ('input_context', 'vibe_context', 'live_conditions'), ('recommendedConditions',), ('recommended_conditions',), ('input_context', 'recommendedConditions'), ('input_context', 'recommended_conditions'))),
        'engineState': _agent_to_text(_nested_pick(record, ('input_context', 'rule_state', 'engine_state'), ('engineState',), ('engine_state',), ('input_context', 'engineState'), ('input_context', 'engine_state'))),
        'createdAt': pd.to_datetime(_nested_pick(record, ('createdAt',), ('created_at',), ('input_context', 'createdAt'), ('input_context', 'created_at')), errors='coerce'),
    }
    row['agent_label'] = _agent_to_text(_first_nonblank(row['decision_class'], _nested_pick(record, ('agent_label', 'ideal_action'), ('agent_label', 'decision_class')), row['recommended_action']))
    return row

def load_betting_agent_training_cases(dirs: List[Path]) -> pd.DataFrame:
    search_dirs = [Path(d) for d in dirs if Path(d).exists()]
    if not search_dirs:
        return pd.DataFrame()

    files: List[Path] = []
    latest_files = []
    for base in search_dirs:
        candidate = base / 'betting_agent_training_cases_latest.jsonl'
        if candidate.exists():
            latest_files.append(candidate)
    if latest_files:
        files = [sorted(latest_files)[0]]
    else:
        dated_files: List[Path] = []
        for base in search_dirs:
            dated_files.extend(sorted(base.glob('betting_agent_training_cases_20??-??-??.jsonl')))
        files = sorted({p for p in dated_files if p.exists()})

    if not files:
        return pd.DataFrame()

    rows = []
    for path in files:
        try:
            raw_lines = path.read_text(encoding='utf-8').splitlines()
        except Exception:
            continue
        for line_no, line in enumerate(raw_lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict):
                continue
            rows.append(_normalize_agent_training_record(record, path, line_no))

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce').dt.normalize()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['home_team'] = df['home_team'].astype(str).map(_normalize_team_phrase)
    df['away_team'] = df['away_team'].astype(str).map(_normalize_team_phrase)
    missing_home = df['home_team'].isin({'', 'NAN', 'NONE'})
    missing_away = df['away_team'].isin({'', 'NAN', 'NONE'})
    if missing_home.any() or missing_away.any():
        derived = df['matchup'].apply(_extract_matchup_teams)
        if missing_home.any():
            df.loc[missing_home, 'home_team'] = derived[missing_home].apply(lambda x: x[0])
        if missing_away.any():
            df.loc[missing_away, 'away_team'] = derived[missing_away].apply(lambda x: x[1])
    df['game_key'] = df.apply(lambda r: _game_key(r['date'], r['home_team'], r['away_team']), axis=1)
    df['matchup_text_key'] = df['matchup'].map(_normalize_matchup_text)
    df['matchup_pair_key'] = df.apply(lambda r: _agent_team_pair_key(r['home_team'], r['away_team']), axis=1)
    df['agent_label'] = df['agent_label'].fillna(df['decision_class']).fillna(df['recommended_action'])
    df = df.sort_values(['date', 'matchup', 'source_file', 'source_line_no'], na_position='last').reset_index(drop=True)
    return df
