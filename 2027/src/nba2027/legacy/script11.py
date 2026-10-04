"""Pure Script 11 functions extracted verbatim from the final June notebook. No notebook execution."""
import numpy as np
import pandas as pd
from .odds_utils import compute_market_probs
PROB_COL_LIVE = 'prob_live_safe'
min_EV = 0
prob_clip_lo = 0.35
prob_clip_hi = 0.8
TAIL_N_HIST = 300
LOCAL_TAIL_LADDER = [300, 400, 500]
LOCAL_LADDER_STOP_AT_FIRST_PASS = True
PRINT_LOCAL_LADDER_DEBUG = True
USE_LIVE_OOS_PROXY = True
LIVE_OOS_PROXY_MIN_ROWS = 300
LIVE_OOS_PROXY_RECENT_N = None
LIVE_OOS_PROXY_N_BINS = 25
LIVE_OOS_PROXY_MIN_BIN_N = 25
LIVE_PROXY_BLEND_ALPHA = 0.8
LIVE_PROXY_SMALL_BIN_ALPHA = 0.95
LIVE_PROXY_HARD_MIN_BIN_N = 200
USE_LIVE_SMOOTH_SHRINK = True
LIVE_SHRINK_START = 0.6
LIVE_SHRINK_FACTOR = 0.85
LIVE_SMALL_BIN_MAX_SHRINK_ALPHA = 0.95
GAP_GUARD_MIN = 0.12
UNDERDOG_ODDS_GUARD_MIN = 2.0
UNDERDOG_PROB_GUARD_MIN = 0.6
UNDERDOG_CAP = 0.55
POLICY_STRICT_BLOCK = True
TAU_GAP = 0.08
SUSPICIOUS_ODDS_MIN = 2.0
SUSPICIOUS_PROB_MIN = 0.6
N_WINDOWS = [300, 400, 500]
STABILITY_HITS_NEEDED = 2
MIN_TRADES_PER_WINDOW = 25
WALK_TRAIN_MIN_DAYS = 21
WALK_TEST_DAYS = 21
WALK_STEP_DAYS = 7
MIN_WALK_SPLITS = 4
MIN_TEST_TRADES_TOTAL = 50
MIN_ACTIVE_SPLITS = 3
MIN_TRADES_PER_ACTIVE_SPLIT = 5
USE_SOFT_GATE = True
SOFT_Q = 0.2
MIN_Q_TRADES = 4
TOP_K_CANDIDATES = 400
SCORE_MODE = 'lcb_roi'
LCB_K = 0.5
PROB_COL_HIST = 'prob_iso_oos_time'
FLAT_STAKE_BACKTEST = 100.0
FLAT_STAKE_LIVE = 100.0
LOOKAHEAD_HRS = 36
HOMEWR_MIN_GRID = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
ODDS_MIN_GRID = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
ODDS_MAX_GRID = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
PROB_MIN_GRID = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

def _proxy_attr(proxy, name, default=None):
    if proxy is None:
        return default
    if hasattr(proxy, name):
        return getattr(proxy, name)
    if isinstance(proxy, dict):
        return proxy.get(name, default)
    return default

def _validate_params(params: dict, required=None, name="params"):
    required = required or ["home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold"]
    if params is None:
        raise KeyError(f"{name} is None")
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"{name} missing keys: {missing}. Got: {list(params.keys())}")

def _ensure_datetime(df: pd.DataFrame, col="date") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if col in out.columns and not np.issubdtype(out[col].dtype, np.datetime64):
        out[col] = pd.to_datetime(out[col], errors="coerce")
    return out

def _make_game_key(df: pd.DataFrame, date_col="date", home_col="home_team", away_col="away_team", dst="game_key") -> pd.DataFrame:
    out = _ensure_datetime(df.copy(), date_col)
    out[dst] = (
        out[date_col].dt.strftime("%Y-%m-%d") + "_" +
        out[home_col].astype(str).str.strip() + "_" +
        out[away_col].astype(str).str.strip()
    )
    return out

def _compute_prob_used(df: pd.DataFrame, lo: float, hi: float, src="prob_iso", dst="prob_used") -> pd.DataFrame:
    out = df.copy()
    if src not in out.columns:
        raise KeyError(f"Missing column '{src}' needed to compute '{dst}'.")
    out[dst] = pd.to_numeric(out[src], errors="coerce").clip(lower=float(lo), upper=float(hi))
    return out

def _compute_ev_per_100(df: pd.DataFrame, prob_col="prob_used", odds_col="odds_1", stake_for_ev: float = 100.0, dst="EV_€_per_100") -> pd.DataFrame:
    out = df.copy()
    out[prob_col] = pd.to_numeric(out[prob_col], errors="coerce")
    out[odds_col] = pd.to_numeric(out[odds_col], errors="coerce")
    out[dst] = (out[prob_col] * (out[odds_col] - 1.0) - (1.0 - out[prob_col])) * float(stake_for_ev)
    return out

def _prep_hist_df(hist_df: pd.DataFrame, tail_n: int = None) -> pd.DataFrame:
    if hist_df is None or len(hist_df) == 0:
        return pd.DataFrame()

    df = hist_df.copy()
    df = _ensure_datetime(df, "date")
    if "is_played" in df.columns:
        df = df[df["is_played"] == True].copy()
    df = df.dropna(subset=["date"]).copy()

    need_core = ["date", "home_team", "away_team", "odds_1", "win", "home_win_rate"]
    for c in need_core:
        if c in df.columns:
            df = df[df[c].notna()].copy()

    if all(c in df.columns for c in ["date", "home_team", "away_team"]):
        df = _make_game_key(df, "date", "home_team", "away_team", "game_key")
        df = df.sort_values("date").drop_duplicates(subset="game_key", keep="last").copy()

    tail_n_eff = int(TAIL_N_HIST if tail_n is None else tail_n)
    df = df.sort_values("date").tail(tail_n_eff).copy()

    if PROB_COL_HIST in df.columns and PROB_COL_HIST == "prob_iso_oos_time":
        df = df[df[PROB_COL_HIST].notna()].copy()

    return df.reset_index(drop=True)

def apply_live_smooth_shrink(p_live: float) -> float:
    if pd.isna(p_live):
        return np.nan
    p = float(p_live)
    if p <= float(LIVE_SHRINK_START):
        return p
    p_adj = 0.50 + float(LIVE_SHRINK_FACTOR) * (p - 0.50)
    return float(np.clip(p_adj, 0.0, 1.0))

def _compute_live_prob_path(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    def _series_or_nan(frame: pd.DataFrame, col: str, numeric: bool = True):
        if col in frame.columns:
            s = frame[col]
        else:
            s = pd.Series(np.nan, index=frame.index)
        return pd.to_numeric(s, errors="coerce") if numeric else s

    out = df.copy()
    out["home_team_prob"] = _series_or_nan(out, "home_team_prob", numeric=True)
    out["odds_1"] = _series_or_nan(out, "odds_1", numeric=True)
    out["odds_2"] = _series_or_nan(out, "odds_2", numeric=True)

    proxy_col = _series_or_nan(out, "prob_live_oos_proxy", numeric=True)
    proxy_ready = bool(out["live_oos_proxy_ready"].iloc[0]) if "live_oos_proxy_ready" in out.columns and len(out) else False
    if proxy_ready and proxy_col.notna().any():
        out["prob_live_base"] = proxy_col
        out["live_oos_proxy_used"] = proxy_col.notna()
    else:
        out["prob_live_base"] = out["home_team_prob"]
        out["live_oos_proxy_used"] = False

    out["prob_live_safe_pre_clip"] = pd.to_numeric(out["prob_live_base"], errors="coerce").clip(lower=prob_clip_lo, upper=prob_clip_hi)

    market_raw, market_devig = compute_market_probs(out["odds_1"], out["odds_2"] if "odds_2" in out.columns else None)
    out["market_implied_p_raw"] = pd.to_numeric(market_raw, errors="coerce")
    out["market_implied_p_devig"] = pd.to_numeric(market_devig, errors="coerce")
    p_market = out["market_implied_p_devig"].where(out["market_implied_p_devig"].notna(), out["market_implied_p_raw"])

    out["model_market_gap"] = out["prob_live_safe_pre_clip"] - p_market
    out["model_market_gap_flag"] = pd.to_numeric(out["model_market_gap"], errors="coerce").ge(float(GAP_GUARD_MIN)).fillna(False)

    guard_mask = (
        out["odds_1"].ge(float(UNDERDOG_ODDS_GUARD_MIN))
        & out["prob_live_safe_pre_clip"].ge(float(UNDERDOG_PROB_GUARD_MIN))
        & pd.to_numeric(out["model_market_gap"], errors="coerce").ge(float(GAP_GUARD_MIN))
    )
    out["live_underdog_upscale_guard_triggered"] = guard_mask.fillna(False)

    prob_guarded = pd.to_numeric(out["prob_live_safe_pre_clip"], errors="coerce").copy()
    prob_guarded.loc[guard_mask] = np.minimum(prob_guarded.loc[guard_mask], float(UNDERDOG_CAP))

    prob_blended = prob_guarded.copy()
    blend_mask = p_market.notna()
    if blend_mask.any():
        w = np.exp(-np.abs(pd.to_numeric(out.loc[blend_mask, "model_market_gap"], errors="coerce")) / float(TAU_GAP))
        prob_blended.loc[blend_mask] = (w * prob_guarded.loc[blend_mask]) + ((1.0 - w) * p_market.loc[blend_mask])

    bin_n_series = _series_or_nan(out, "live_oos_proxy_bin_n", numeric=True)
    sufficient_bin_mask = bin_n_series.ge(int(LIVE_PROXY_HARD_MIN_BIN_N)).fillna(False)
    out["live_oos_proxy_bin_confident"] = sufficient_bin_mask
    alpha = pd.Series(1.0, index=out.index, dtype=float)
    if USE_LIVE_SMOOTH_SHRINK:
        alpha.loc[prob_blended > float(LIVE_SHRINK_START)] = float(LIVE_SHRINK_FACTOR)
        gap_mask = out["model_market_gap_flag"].fillna(False)
        alpha.loc[gap_mask & sufficient_bin_mask] = np.minimum(alpha.loc[gap_mask & sufficient_bin_mask], 0.70)
        alpha.loc[gap_mask & (~sufficient_bin_mask)] = np.minimum(alpha.loc[gap_mask & (~sufficient_bin_mask)], float(LIVE_SMALL_BIN_MAX_SHRINK_ALPHA))

    out["prob_base"] = pd.to_numeric(out["prob_live_safe_pre_clip"], errors="coerce")
    out["prob_live_blend_p"] = pd.to_numeric(prob_blended, errors="coerce")
    out["prob_live_shrink_input"] = pd.to_numeric(prob_blended, errors="coerce")
    out["prob_used"] = 0.5 + alpha * (prob_blended - 0.5)
    out["prob_used"] = pd.to_numeric(out["prob_used"], errors="coerce").clip(lower=prob_clip_lo, upper=prob_clip_hi)
    out["live_shrink_alpha"] = alpha
    out["live_shrink_triggered"] = alpha.lt(1.0)

    existing_blocked = out["blocked_by"].astype(str) if "blocked_by" in out.columns else pd.Series("", index=out.index, dtype=object)
    if POLICY_STRICT_BLOCK:
        out["blocked_by"] = np.where(
            out["live_underdog_upscale_guard_triggered"],
            "MODEL_MARKET_GAP",
            np.where(existing_blocked.isin(["", "nan", "None"]), "", existing_blocked),
        )
    else:
        out["blocked_by"] = np.where(existing_blocked.isin(["", "nan", "None"]), "", existing_blocked)

    fallback_series = out["live_oos_proxy_fallback_used"] if "live_oos_proxy_fallback_used" in out.columns else pd.Series(False, index=out.index)
    reason_chain = np.where(fallback_series.fillna(False), "fallback", "proxy")
    reason_chain = np.where(out["model_market_gap_flag"].fillna(False), reason_chain + "|gap_flag", reason_chain)
    reason_chain = np.where(out["live_underdog_upscale_guard_triggered"].fillna(False), reason_chain + "|guard", reason_chain)
    reason_chain = np.where((~out["live_oos_proxy_bin_confident"].fillna(False)) & out["model_market_gap_flag"].fillna(False), reason_chain + "|small_bin_mild_shrink", reason_chain)
    reason_chain = np.where(out["live_shrink_triggered"].fillna(False), reason_chain + "|shrink", reason_chain)
    out["reason_chain"] = reason_chain

    return out

def _evaluate_params_on_window(df_window: pd.DataFrame, params: dict, *,
                               min_ev: float, flat_stake_backtest: float,
                               prob_clip_lo: float, prob_clip_hi: float):
    _validate_params(params, name="params_to_eval")
    if df_window is None or df_window.empty:
        metrics = {
            "n_trades": 0, "win_rate_%": 0.0, "avg_EV_€_per_100": 0.0,
            "profit_€": 0.0, "roi_%": 0.0,
            "prob_thr_eff": round(max(float(params["prob_threshold"]), float(prob_clip_lo)), 3),
        }
        return metrics, pd.DataFrame()

    df = df_window.copy()
    df = _ensure_datetime(df, "date")
    df = _compute_prob_used(df, prob_clip_lo, prob_clip_hi, PROB_COL_HIST, "prob_used")
    df = _compute_ev_per_100(df, "prob_used", "odds_1", 100.0, "EV_€_per_100")
    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df = df.dropna(subset=["home_win_rate", "prob_used", "odds_1", "win", "EV_€_per_100"])

    prob_thr_eff = max(float(params["prob_threshold"]), float(prob_clip_lo))
    mask = (
        (pd.to_numeric(df["home_win_rate"], errors="coerce") >= float(params["home_win_rate_threshold"])) &
        (pd.to_numeric(df["odds_1"], errors="coerce")        >= float(params["odds_min"])) &
        (pd.to_numeric(df["odds_1"], errors="coerce")        <= float(params["odds_max"])) &
        (pd.to_numeric(df["prob_used"], errors="coerce")     >= prob_thr_eff) &
        (pd.to_numeric(df["EV_€_per_100"], errors="coerce")  >  float(min_ev))
    )
    subset = df.loc[mask].copy()
    if subset.empty:
        metrics = {
            "n_trades": 0, "win_rate_%": 0.0, "avg_EV_€_per_100": 0.0,
            "profit_€": 0.0, "roi_%": 0.0, "prob_thr_eff": round(prob_thr_eff, 3),
        }
        return metrics, subset

    subset["pnl"] = np.where(
        subset["win"] == 1,
        float(flat_stake_backtest) * (subset["odds_1"] - 1.0),
        -float(flat_stake_backtest)
    )
    profit = float(subset["pnl"].sum())
    n_trades = int(len(subset))
    total_stake = n_trades * float(flat_stake_backtest)
    roi = (profit / total_stake * 100.0) if total_stake > 0 else 0.0

    metrics = {
        "n_trades": n_trades,
        "win_rate_%": round(float(subset["win"].mean() * 100.0), 2),
        "avg_EV_€_per_100": round(float(subset["EV_€_per_100"].mean()), 2),
        "profit_€": round(profit, 2),
        "roi_%": round(roi, 2),
        "prob_thr_eff": round(prob_thr_eff, 3),
    }
    return metrics, subset

def _make_walk_splits(df: pd.DataFrame, *, train_min_days: int, test_days: int, step_days: int):
    if df is None or df.empty:
        return []
    df = _ensure_datetime(df, "date").sort_values("date").reset_index(drop=True)
    start = df["date"].min().normalize()
    end = df["date"].max().normalize()

    splits = []
    anchor = start + pd.Timedelta(days=train_min_days)

    while True:
        train_end = anchor
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_days)

        if test_start > end:
            break

        train_mask = (df["date"] < train_end)
        test_mask = (df["date"] >= test_start) & (df["date"] < test_end)

        train_df = df.loc[train_mask].copy()
        test_df = df.loc[test_mask].copy()

        if len(test_df) > 0 and len(train_df) > 0:
            splits.append((train_df, test_df))

        anchor = anchor + pd.Timedelta(days=step_days)
        if test_end > end + pd.Timedelta(days=1):
            break

    return splits

def _iter_param_candidates(df_full: pd.DataFrame, *,
                           homewr_grid, odds_min_grid, odds_max_grid, prob_min_grid,
                           min_ev: float, min_trades_train: int):
    df = df_full.copy()
    df = _compute_prob_used(df, prob_clip_lo, prob_clip_hi, PROB_COL_HIST, "prob_used")
    df = _compute_ev_per_100(df, "prob_used", "odds_1", 100.0, "EV_€_per_100")
    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df = df.dropna(subset=["home_win_rate","prob_used","odds_1","win","EV_€_per_100"])

    prob_min_grid_eff = [p for p in prob_min_grid if p >= prob_clip_lo] or [prob_clip_lo]

    cands = []
    for hw in homewr_grid:
        for o_min in odds_min_grid:
            for o_max in odds_max_grid:
                if o_max <= o_min:
                    continue
                for p_min in prob_min_grid_eff:
                    mask = (
                        (pd.to_numeric(df["home_win_rate"], errors="coerce") >= float(hw)) &
                        (pd.to_numeric(df["odds_1"], errors="coerce")        >= float(o_min)) &
                        (pd.to_numeric(df["odds_1"], errors="coerce")        <= float(o_max)) &
                        (pd.to_numeric(df["prob_used"], errors="coerce")     >= float(p_min)) &
                        (pd.to_numeric(df["EV_€_per_100"], errors="coerce")  >  float(min_ev))
                    )
                    sub = df.loc[mask].copy()
                    if len(sub) < int(min_trades_train):
                        continue

                    sub["pnl"] = np.where(
                        sub["win"] == 1,
                        float(FLAT_STAKE_BACKTEST) * (sub["odds_1"] - 1.0),
                        -float(FLAT_STAKE_BACKTEST)
                    )
                    profit = float(sub["pnl"].sum())

                    cands.append({
                        "params": {
                            "home_win_rate_threshold": round(float(hw), 2),
                            "odds_min": round(float(o_min), 2),
                            "odds_max": round(float(o_max), 2),
                            "prob_threshold": round(float(p_min), 2),
                        },
                        "profit": profit,
                        "n": int(len(sub))
                    })
    return cands

def _coverage_gate_ok(test_trade_counts):
    n_splits = len(test_trade_counts)
    total_test_trades = int(np.sum(test_trade_counts)) if test_trade_counts else 0
    active_splits = int(np.sum(np.array(test_trade_counts) >= int(MIN_TRADES_PER_ACTIVE_SPLIT))) if test_trade_counts else 0

    q_trades = float(np.quantile(test_trade_counts, SOFT_Q)) if (USE_SOFT_GATE and len(test_trade_counts) > 0) else None

    hard_ok = (
        n_splits >= int(MIN_WALK_SPLITS) and
        total_test_trades >= int(MIN_TEST_TRADES_TOTAL) and
        active_splits >= int(MIN_ACTIVE_SPLITS)
    )
    if not USE_SOFT_GATE:
        return hard_ok, {
            "splits_used": n_splits,
            "test_trades_total": total_test_trades,
            "active_splits": active_splits,
            "q_trades": q_trades,
        }

    soft_ok = (q_trades is not None and q_trades >= float(MIN_Q_TRADES))
    return (hard_ok and soft_ok), {
        "splits_used": n_splits,
        "test_trades_total": total_test_trades,
        "active_splits": active_splits,
        "q_trades": q_trades,
    }

def find_best_local_params_walk_forward(hist_df_prepped: pd.DataFrame, *,
                                        homewr_grid, odds_min_grid, odds_max_grid, prob_min_grid,
                                        flat_stake_backtest: float, min_ev: float,
                                        min_trades_train: int = 20,
                                        score_mode: str = "lcb_ev",
                                        top_k: int = 200):
    if hist_df_prepped is None or hist_df_prepped.empty:
        return None, None, {"gate_pass": False, "reason": "hist empty"}

    splits = _make_walk_splits(
        hist_df_prepped,
        train_min_days=WALK_TRAIN_MIN_DAYS,
        test_days=WALK_TEST_DAYS,
        step_days=WALK_STEP_DAYS
    )
    if len(splits) == 0:
        return None, None, {"gate_pass": False, "reason": "no splits"}

    candidates = _iter_param_candidates(
        hist_df_prepped,
        homewr_grid=homewr_grid,
        odds_min_grid=odds_min_grid,
        odds_max_grid=odds_max_grid,
        prob_min_grid=prob_min_grid,
        min_ev=min_ev,
        min_trades_train=min_trades_train,
    )
    if not candidates:
        return None, None, {"gate_pass": False, "reason": "no candidate params"}

    candidates = sorted(candidates, key=lambda x: x["profit"], reverse=True)[:int(top_k)]

    best_score = float("-inf")
    best_params_local = None
    best_debug = None

    best_possible_total = 0
    best_possible_active = 0
    best_possible_q = None
    best_possible_profit = float("-inf")

    for cand in candidates:
        params = cand["params"]
        test_trade_counts = []
        test_rois = []
        test_evs = []
        test_profits = []

        for _, test_df in splits:
            metrics_t, _subset_t = _evaluate_params_on_window(
                test_df, params,
                min_ev=min_ev,
                flat_stake_backtest=flat_stake_backtest,
                prob_clip_lo=prob_clip_lo,
                prob_clip_hi=prob_clip_hi,
            )
            test_trade_counts.append(int(metrics_t["n_trades"]))
            test_rois.append(float(metrics_t["roi_%"]))
            test_evs.append(float(metrics_t["avg_EV_€_per_100"]))
            test_profits.append(float(metrics_t["profit_€"]))

        gate_ok, gate_dbg = _coverage_gate_ok(test_trade_counts)

        total_test_trades = int(gate_dbg["test_trades_total"])
        active_splits = int(gate_dbg["active_splits"])
        q_tr = gate_dbg["q_trades"]
        wf_profit_total = float(np.nansum(test_profits))

        best_possible_total = max(best_possible_total, total_test_trades)
        best_possible_active = max(best_possible_active, active_splits)
        if q_tr is not None:
            best_possible_q = q_tr if best_possible_q is None else max(best_possible_q, q_tr)
        best_possible_profit = max(best_possible_profit, wf_profit_total)

        if not gate_ok:
            continue

        if score_mode == "wf_profit":
            score = float(np.nansum(test_profits))
            label = "WF total profit (€)"

        elif score_mode == "lcb_profit":
            mean_profit = float(np.mean(test_profits))
            std_profit = float(np.std(test_profits, ddof=0)) if len(test_profits) > 1 else 0.0
            score = mean_profit - float(LCB_K) * std_profit
            label = f"LCB(mean_test_profit - {LCB_K}*std)"

        elif score_mode == "lcb_roi":
            mean_roi = float(np.mean(test_rois))
            std_roi = float(np.std(test_rois, ddof=0)) if len(test_rois) > 1 else 0.0
            score = mean_roi - float(LCB_K) * std_roi
            label = f"LCB(mean_test_ROI - {LCB_K}*std)"

        elif score_mode == "lcb_ev":
            mean_ev = float(np.mean(test_evs))
            std_ev = float(np.std(test_evs, ddof=0)) if len(test_evs) > 1 else 0.0
            score = mean_ev - float(LCB_K) * std_ev
            label = f"LCB(mean_test_EV - {LCB_K}*std)"

        else:
            raise ValueError(f"Unknown SCORE_MODE: {score_mode}")

        if score > best_score:
            best_score = score
            best_params_local = params
            best_debug = {
                "gate_pass": True,
                "reason": "ok",
                "score_mode": score_mode,
                "score_label": label,
                "score_value": float(score),
                **gate_dbg,
                "wf_test_profit_total": wf_profit_total,
                "top_k": int(top_k),
                "candidates_considered": int(len(candidates)),
            }

    if best_params_local is None:
        return None, None, {
            "gate_pass": False,
            "reason": f"no candidate passed coverage gate (top_k={top_k})",
            "splits_used": int(len(splits)),
            "candidates_considered": int(len(candidates)),
            "top_k": int(top_k),
            "best_possible_test_trades_total_among_checked": best_possible_total,
            "best_possible_active_splits_among_checked": best_possible_active,
            "best_possible_qtrades_among_checked": best_possible_q,
            "best_possible_wf_test_profit_total_among_checked": best_possible_profit,
        }

    metrics_full, subset_full = _evaluate_params_on_window(
        hist_df_prepped, best_params_local,
        min_ev=min_ev,
        flat_stake_backtest=flat_stake_backtest,
        prob_clip_lo=prob_clip_lo,
        prob_clip_hi=prob_clip_hi,
    )
    best_debug.update({
        "full_trades": int(metrics_full["n_trades"]),
        "full_roi": float(metrics_full["roi_%"]),
        "full_profit": float(metrics_full["profit_€"]),
    })
    return best_params_local, subset_full, best_debug

def find_best_local_params_with_fallback(hist_df_raw: pd.DataFrame, *,
                                         local_tail_ladder,
                                         homewr_grid, odds_min_grid, odds_max_grid, prob_min_grid,
                                         flat_stake_backtest: float, min_ev: float,
                                         min_trades_train: int = 20,
                                         score_mode: str = "lcb_ev",
                                         top_k: int = 200,
                                         stop_at_first_pass: bool = True):
    """
    Try LOCAL walk-forward on multiple tail sizes (e.g., 300->400->500).
    Returns:
      best_params_local, best_subset_full, wf_info, hist_df_prepped_local_used
    wf_info includes:
      - local_tail_used
      - local_ladder_attempts (list of debug dicts)
    """
    attempts = []
    if hist_df_raw is None or hist_df_raw.empty:
        return None, None, {
            "gate_pass": False,
            "reason": "hist empty",
            "local_tail_used": None,
            "local_ladder_attempts": attempts,
        }, None

    best_choice = None

    for tail_n in local_tail_ladder:
        hist_local_prepped = _prep_hist_df(hist_df_raw, tail_n=int(tail_n))
        if hist_local_prepped is None or hist_local_prepped.empty:
            attempts.append({
                "tail_n": int(tail_n),
                "rows": 0,
                "date_min": None,
                "date_max": None,
                "gate_pass": False,
                "reason": "empty prepped history",
            })
            continue

        params_local, subset_full, wf_i = find_best_local_params_walk_forward(
            hist_local_prepped,
            homewr_grid=homewr_grid,
            odds_min_grid=odds_min_grid,
            odds_max_grid=odds_max_grid,
            prob_min_grid=prob_min_grid,
            flat_stake_backtest=flat_stake_backtest,
            min_ev=min_ev,
            min_trades_train=min_trades_train,
            score_mode=score_mode,
            top_k=top_k,
        )

        attempts.append({
            "tail_n": int(tail_n),
            "rows": int(len(hist_local_prepped)),
            "date_min": hist_local_prepped["date"].min() if "date" in hist_local_prepped.columns else None,
            "date_max": hist_local_prepped["date"].max() if "date" in hist_local_prepped.columns else None,
            "gate_pass": bool(wf_i.get("gate_pass", False)),
            "reason": wf_i.get("reason"),
            "score_value": wf_i.get("score_value"),
            "splits_used": wf_i.get("splits_used"),
            "test_trades_total": wf_i.get("test_trades_total"),
            "active_splits": wf_i.get("active_splits"),
            "q_trades": wf_i.get("q_trades"),
            "best_possible_test_trades_total": wf_i.get("best_possible_test_trades_total_among_checked"),
            "best_possible_active_splits": wf_i.get("best_possible_active_splits_among_checked"),
            "best_possible_qtrades": wf_i.get("best_possible_qtrades_among_checked"),
        })

        if not wf_i.get("gate_pass", False) or params_local is None:
            continue

        choice = {
            "params": params_local,
            "subset": subset_full,
            "wf": dict(wf_i),
            "prepped": hist_local_prepped,
            "tail_n": int(tail_n),
            "score": float(wf_i.get("score_value", float("-inf"))),
        }

        if stop_at_first_pass:
            choice["wf"]["local_tail_used"] = int(tail_n)
            choice["wf"]["local_ladder_attempts"] = attempts
            return choice["params"], choice["subset"], choice["wf"], choice["prepped"]

        if best_choice is None or choice["score"] > best_choice["score"]:
            best_choice = choice

    if best_choice is not None:
        best_choice["wf"]["local_tail_used"] = int(best_choice["tail_n"])
        best_choice["wf"]["local_ladder_attempts"] = attempts
        return best_choice["params"], best_choice["subset"], best_choice["wf"], best_choice["prepped"]

    wf_info_fail = {
        "gate_pass": False,
        "reason": "no candidate passed coverage gate on any LOCAL tail",
        "local_tail_used": None,
        "local_ladder_attempts": attempts,
    }
    return None, None, wf_info_fail, None

def build_near_miss_watchlist(upcoming_df: pd.DataFrame, params_used: dict = None, *,
                              min_ev: float,
                              top_n: int = 10) -> pd.DataFrame:
    """
    Diagnostic/watchlist table for upcoming games.
    Shows which rules pass/fail and how close each game is to passing.
    Works even when params_used is None (NO_BET mode) by using best_params fallback if available.
    """
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame()

    # pick reference params for diagnostics
    ref_params = params_used
    if ref_params is None:
        ref_params = globals().get("best_params", None)
    if ref_params is None:
        return pd.DataFrame()

    _validate_params(ref_params, name="watchlist params")

    df = _ensure_datetime(upcoming_df, "date").copy()

    # Prob selection = same as live shortlist logic
    df = _compute_live_prob_path(df)

    # EV
    df = _compute_ev_per_100(df, "prob_used", "odds_1", 100.0, "EV_€_per_100")
    df = _make_game_key(df, "date", "home_team", "away_team", "game_key")

    # dedupe same as shortlist
    df = (df.sort_values("EV_€_per_100", ascending=False)
            .drop_duplicates(subset="game_key", keep="first")
            .reset_index(drop=True))

    # effective thresholds
    hw_thr = float(ref_params["home_win_rate_threshold"])
    o_min = float(ref_params["odds_min"])
    o_max = float(ref_params["odds_max"])
    p_thr = max(float(ref_params["prob_threshold"]), float(prob_clip_lo))
    ev_thr = float(min_ev)

    # pass/fail flags
    df["pass_hw"]   = df["home_win_rate"] >= hw_thr
    df["pass_odds"] = df["odds_1"].between(o_min, o_max, inclusive="both")
    df["pass_prob"] = df["prob_used"] >= p_thr
    df["pass_ev"]   = df["EV_€_per_100"] > ev_thr

    # numeric margins (positive = good, negative = missing threshold / outside range)
    df["margin_hw"] = df["home_win_rate"] - hw_thr
    df["margin_prob"] = df["prob_used"] - p_thr
    df["margin_ev"] = df["EV_€_per_100"] - ev_thr

    # odds margin: positive if inside range, otherwise negative distance to nearest bound
    df["margin_odds"] = np.where(
        df["odds_1"] < o_min,
        df["odds_1"] - o_min,
        np.where(df["odds_1"] > o_max, o_max - df["odds_1"], 0.0)
    )

    # count passed rules
    pass_cols = ["pass_hw", "pass_odds", "pass_prob", "pass_ev"]
    df["rules_passed"] = df[pass_cols].sum(axis=1).astype(int)

    # identify first blocking reason (ordered)
    # identify blocking reason
    def _block_reason(row):
        reasons = []
        if POLICY_STRICT_BLOCK and bool(row.get("live_underdog_upscale_guard_triggered", False)):
            reasons.append("MODEL_MARKET_GAP")
        if not bool(row["pass_hw"]):
            reasons.append(f"HWR<{hw_thr:.2f}")
        if not bool(row["pass_odds"]):
            if row["odds_1"] < o_min:
                reasons.append(f"Odds<{o_min:.2f}")
            elif row["odds_1"] > o_max:
                reasons.append(f"Odds>{o_max:.2f}")
            else:
                reasons.append("Odds")
        if not bool(row["pass_prob"]):
            reasons.append(f"Prob<{p_thr:.2f}")
        if not bool(row["pass_ev"]):
            reasons.append(f"EV<={ev_thr:.2f}")

        if len(reasons) == 0:
            return "PASS" if params_used is not None else "PASS_FILTERS_ONLY"
        return " | ".join(dict.fromkeys(reasons))


    df["blocked_by"] = df.apply(_block_reason, axis=1)

    # rank near misses:
    # 1) more rules passed
    # 2) positive EV first
    # 3) higher EV
    # 4) closer probability margin / hwr margin
    df["_is_ev_pos"] = (df["EV_€_per_100"] > 0).astype(int)
    df = df.sort_values(
        by=["rules_passed", "_is_ev_pos", "EV_€_per_100", "margin_prob", "margin_hw"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    # friendly columns
    out_cols = [
        "date", "home_team", "away_team",
        "home_win_rate", "odds_1", "odds_2", "home_team_prob", "prob_iso", "prob_iso_oos_time",
        "prob_live_oos_proxy", "prob_live_safe_pre_clip", "prob_base", "prob_used", "EV_€_per_100",
        "market_implied_p_raw", "market_implied_p_devig", "model_market_gap", "model_market_gap_flag",
        "live_underdog_upscale_guard_triggered", "live_shrink_triggered",
        "live_oos_proxy_ready", "live_oos_proxy_used", "live_oos_proxy_train_rows", "live_oos_proxy_bin_n", "live_oos_proxy_bin_winrate",
        "rules_passed", "blocked_by",
        "margin_hw", "margin_odds", "margin_prob", "margin_ev"
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df[out_cols].head(int(top_n)).copy()

def _get_watchlist_ref_params(params_used, best_params):
    """
    Watchlist reference params:
    - use params_used if engine selected a live strategy
    - otherwise fall back to best_params (GLOBAL grid result)
    Returns (ref_params_dict, ref_source_str)
    """
    ref = None
    ref_source = None

    if params_used is not None:
        ref = params_used.copy()
        ref_source = "params_used"
    elif best_params is not None:
        ref = best_params.copy()
        ref_source = "best_params"
    else:
        return None, None

    # Normalize numeric values (handles np.float64 safely)
    for k in ["home_win_rate_threshold", "odds_min", "odds_max", "prob_threshold"]:
        if k in ref and pd.notna(ref[k]):
            ref[k] = float(ref[k])

    return ref, ref_source

def eval_profit_stability(hist_df_prepped: pd.DataFrame, params: dict, *, min_ev: float):
    out = []
    n_rows_total = 0 if hist_df_prepped is None else len(hist_df_prepped)

    for N in N_WINDOWS:
        N = int(N)

        if hist_df_prepped is None or hist_df_prepped.empty or n_rows_total < N:
            out.append({
                "N": N,
                "usable": False,
                "window_available": False,
                "n_trades": 0,
                "win_rate_%": 0.0,
                "avg_EV_€_per_100": 0.0,
                "profit_€": 0.0,
                "roi_%": 0.0,
                "prob_thr_eff": np.nan,
            })
            continue

        window = hist_df_prepped.tail(N).copy()
        metrics, _ = _evaluate_params_on_window(
            window, params,
            min_ev=min_ev,
            flat_stake_backtest=FLAT_STAKE_BACKTEST,
            prob_clip_lo=prob_clip_lo,
            prob_clip_hi=prob_clip_hi,
        )

        usable = metrics["n_trades"] >= int(MIN_TRADES_PER_WINDOW)
        out.append({"N": N, "usable": usable, "window_available": True, **metrics})

    return out

def choose_profitable_config(hist_df_prepped: pd.DataFrame,
                             global_params: dict,
                             local_params: dict,
                             *,
                             min_ev: float):
    _validate_params(global_params, name="GLOBAL params")
    if local_params is not None:
        _validate_params(local_params, name="LOCAL params")

    g = eval_profit_stability(hist_df_prepped, global_params, min_ev=min_ev)
    l = eval_profit_stability(hist_df_prepped, local_params, min_ev=min_ev) if local_params is not None else None

    def hits(rows):
        return sum(
            1 for r in rows
            if r.get("window_available") and r.get("usable") and r.get("profit_€", 0.0) > 0.0
        )

    def largest_available_N(rows):
        avail = [r["N"] for r in rows if r.get("window_available")]
        return max(avail) if avail else None

    def row_for_N(rows, N):
        return next((r for r in rows if r["N"] == N), None)

    g_hits = hits(g)
    l_hits = hits(l) if l is not None else 0

    g_ok = g_hits >= int(STABILITY_HITS_NEEDED)
    l_ok = (l is not None) and (l_hits >= int(STABILITY_HITS_NEEDED))

    gN = largest_available_N(g)
    lN = largest_available_N(l) if l is not None else None

    g_largest = row_for_N(g, gN) if gN is not None else None
    l_largest = row_for_N(l, lN) if (l is not None and lN is not None) else None

    # --- NO_BET
    if not g_ok and not l_ok:
        return None, {
            "chosen": "NO_BET",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "global_largestN": gN,
            "local_largestN": lN,
            "global_largest": None if g_largest is None else {
                "profit": g_largest["profit_€"], "roi": g_largest["roi_%"], "trades": g_largest["n_trades"]
            },
            "local_largest": None if l_largest is None else {
                "profit": l_largest["profit_€"], "roi": l_largest["roi_%"], "trades": l_largest["n_trades"]
            },
        }

    # --- pick single side if only one passes stability
    if g_ok and not l_ok:
        return global_params.copy(), {
            "chosen": "GLOBAL",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "largestN": gN,
            "largestN_profit": float(g_largest["profit_€"]) if g_largest else np.nan,
            "largestN_roi": float(g_largest["roi_%"]) if g_largest else np.nan,
            "largestN_trades": int(g_largest["n_trades"]) if g_largest else 0,
        }

    if l_ok and not g_ok:
        return local_params.copy(), {
            "chosen": "LOCAL",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "largestN": lN,
            "largestN_profit": float(l_largest["profit_€"]) if l_largest else np.nan,
            "largestN_roi": float(l_largest["roi_%"]) if l_largest else np.nan,
            "largestN_trades": int(l_largest["n_trades"]) if l_largest else 0,
        }

    # --- both pass: compare on largest COMMON window (fair)
    common = []
    if l is not None:
        g_av = {r["N"] for r in g if r.get("window_available")}
        l_av = {r["N"] for r in l if r.get("window_available")}
        common = sorted(list(g_av.intersection(l_av)))
    Ncmp = max(common) if common else (lN if (lN is not None) else gN)

    g_cmp = row_for_N(g, Ncmp)
    l_cmp = row_for_N(l, Ncmp) if l is not None else None

    if l_cmp and g_cmp:
        if float(l_cmp["profit_€"]) > float(g_cmp["profit_€"]):
            return local_params.copy(), {
                "chosen": "LOCAL",
                "compareN": Ncmp,
                "global_hits": int(g_hits),
                "local_hits": int(l_hits),
                "largestN_profit": float(l_cmp["profit_€"]),
                "largestN_roi": float(l_cmp["roi_%"]),
                "largestN_trades": int(l_cmp["n_trades"]),
            }
        if float(l_cmp["profit_€"]) < float(g_cmp["profit_€"]):
            return global_params.copy(), {
                "chosen": "GLOBAL",
                "compareN": Ncmp,
                "global_hits": int(g_hits),
                "local_hits": int(l_hits),
                "largestN_profit": float(g_cmp["profit_€"]),
                "largestN_roi": float(g_cmp["roi_%"]),
                "largestN_trades": int(g_cmp["n_trades"]),
            }

        # tie-breaker on ROI
        if float(l_cmp["roi_%"]) > float(g_cmp["roi_%"]):
            return local_params.copy(), {"chosen": "LOCAL_TIE_ROI", "compareN": Ncmp}
        return global_params.copy(), {"chosen": "GLOBAL_TIE_ROI", "compareN": Ncmp}

    # fallback (should be rare)
    return local_params.copy(), {"chosen": "LOCAL_FALLBACK"}

def choose_profitable_config_dual(hist_df_global: pd.DataFrame,
                                  hist_df_local: pd.DataFrame,
                                  global_params: dict,
                                  local_params: dict,
                                  min_ev: float):
    _validate_params(global_params, name="GLOBAL params")
    if local_params is not None:
        _validate_params(local_params, name="LOCAL params")

    g = eval_profit_stability(hist_df_global, global_params, min_ev=min_ev)
    l = eval_profit_stability(hist_df_local, local_params, min_ev=min_ev) if local_params is not None else None

    def hits(rows):
        return sum(
            1 for r in rows
            if r.get("window_available") and r.get("usable") and r.get("profit_€", 0.0) > 0.0
        )

    def largest_available_N(rows):
        avail = [r["N"] for r in rows if r.get("window_available")]
        return max(avail) if avail else None

    def row_for_N(rows, N):
        return next((r for r in rows if r["N"] == N), None)

    g_hits = hits(g)
    l_hits = hits(l) if l is not None else 0

    g_ok = g_hits >= int(STABILITY_HITS_NEEDED)
    l_ok = (l is not None) and (l_hits >= int(STABILITY_HITS_NEEDED))

    gN = largest_available_N(g)
    lN = largest_available_N(l) if l is not None else None

    g_largest = row_for_N(g, gN) if gN is not None else None
    l_largest = row_for_N(l, lN) if (l is not None and lN is not None) else None

    if not g_ok and not l_ok:
        return None, {
            "chosen": "NO_BET",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "global_largestN": gN,
            "local_largestN": lN,
            "global_largest": None if g_largest is None else {
                "profit": g_largest["profit_€"], "roi": g_largest["roi_%"], "trades": g_largest["n_trades"]
            },
            "local_largest": None if l_largest is None else {
                "profit": l_largest["profit_€"], "roi": l_largest["roi_%"], "trades": l_largest["n_trades"]
            },
        }

    if g_ok and not l_ok:
        return global_params.copy(), {
            "chosen": "GLOBAL",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "largestN": gN,
            "largestN_profit": float(g_largest["profit_€"]) if g_largest else np.nan,
            "largestN_roi": float(g_largest["roi_%"]) if g_largest else np.nan,
            "largestN_trades": int(g_largest["n_trades"]) if g_largest else 0,
        }

    if l_ok and not g_ok:
        return local_params.copy(), {
            "chosen": "LOCAL",
            "global_hits": int(g_hits),
            "local_hits": int(l_hits),
            "largestN": lN,
            "largestN_profit": float(l_largest["profit_€"]) if l_largest else np.nan,
            "largestN_roi": float(l_largest["roi_%"]) if l_largest else np.nan,
            "largestN_trades": int(l_largest["n_trades"]) if l_largest else 0,
        }

    # compare on largest COMMON window
    common = []
    if l is not None:
        g_av = {r["N"] for r in g if r.get("window_available")}
        l_av = {r["N"] for r in l if r.get("window_available")}
        common = sorted(list(g_av.intersection(l_av)))
    Ncmp = max(common) if common else (lN if (lN is not None) else gN)

    g_cmp = row_for_N(g, Ncmp)
    l_cmp = row_for_N(l, Ncmp) if l is not None else None

    if l_cmp and g_cmp:
        if float(l_cmp["profit_€"]) > float(g_cmp["profit_€"]):
            return local_params.copy(), {"chosen": "LOCAL", "compareN": Ncmp}
        if float(l_cmp["profit_€"]) < float(g_cmp["profit_€"]):
            return global_params.copy(), {"chosen": "GLOBAL", "compareN": Ncmp}
        if float(l_cmp["roi_%"]) > float(g_cmp["roi_%"]):
            return local_params.copy(), {"chosen": "LOCAL_TIE_ROI", "compareN": Ncmp}
        return global_params.copy(), {"chosen": "GLOBAL_TIE_ROI", "compareN": Ncmp}

    return local_params.copy(), {"chosen": "LOCAL_FALLBACK"}

def build_flat_shortlist_today(upcoming_df: pd.DataFrame, params_used: dict, *,
                               min_ev: float, flat_stake_live: float):
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame()
    _validate_params(params_used, name="params_used")

    df = _ensure_datetime(upcoming_df, "date")
    df = _compute_live_prob_path(df)
    df = _compute_ev_per_100(df, "prob_used", "odds_1", 100.0, "EV_€_per_100")
    df = _make_game_key(df, "date", "home_team", "away_team", "game_key")

    df = (df.sort_values("EV_€_per_100", ascending=False)
            .drop_duplicates(subset="game_key", keep="first")
            .reset_index(drop=True))

    prob_thr_eff = max(float(params_used["prob_threshold"]), float(prob_clip_lo))
    mask = (
        (pd.to_numeric(df["home_win_rate"], errors="coerce") >= float(params_used["home_win_rate_threshold"])) &
        (pd.to_numeric(df["odds_1"], errors="coerce")        >= float(params_used["odds_min"])) &
        (pd.to_numeric(df["odds_1"], errors="coerce")        <= float(params_used["odds_max"])) &
        (pd.to_numeric(df["prob_used"], errors="coerce")     >= prob_thr_eff) &
        (pd.to_numeric(df["EV_€_per_100"], errors="coerce")  >  float(min_ev)) &
        (~pd.Series(df.get("blocked_by", ""), index=df.index).astype(str).str.contains("MODEL_MARKET_GAP", na=False) if POLICY_STRICT_BLOCK else True)
    )
    picks = df.loc[mask].copy()
    if picks.empty:
        return pd.DataFrame()

    picks["stake_flat"] = float(flat_stake_live)
    picks["potential_profit_if_win"] = (picks["stake_flat"] * (picks["odds_1"] - 1.0)).round(2)
    picks["fair_odds"] = (1.0 / picks["prob_used"]).round(3)
    picks["edge_pct"]  = ((picks["odds_1"] / picks["fair_odds"] - 1.0) * 100.0).round(2)
    picks["EV_€"] = ((picks["prob_used"] * (picks["odds_1"] - 1.0) - (1.0 - picks["prob_used"])) * picks["stake_flat"]).round(2)

    return picks.sort_values("date").reset_index(drop=True)
