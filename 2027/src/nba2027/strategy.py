"""Replay original Script 11 selection on strictly earlier settled history."""
import copy
import pandas as pd
from .legacy import script11 as r
from .legacy.live_oos_proxy import build_live_oos_proxy, apply_live_oos_proxy

def prepare(history, rows, *, now, config):
    config=copy.deepcopy(config)
    history=history.copy()
    if 'result' in history:
        settled=history['result'].eq(history['home_team']) | history['result'].eq(history['away_team'])
        if 'is_played' not in history:history['is_played']=settled
        if 'win' not in history:history['win']=history['result'].eq(history['home_team']).where(settled)
    required=['date','home_team','away_team','is_played','win','home_win_rate','odds_1','prob_iso_oos_time']
    if any(c not in history for c in required):
        raise ValueError('Historical data is missing Script 11 required columns')
    h=history.copy(); dates=pd.to_datetime(h['date'],errors='coerce',utc=True)
    # Date-only historical rows are usable only after the entire date has passed.
    h=h.loc[(dates+pd.Timedelta(days=1)<pd.Timestamp(now)) & h['is_played'].isin([True,1,'True','true','1']) & pd.to_numeric(h['win'],errors='coerce').isin([0,1])].copy()
    h['date']=pd.to_datetime(h['date'],errors='coerce').dt.tz_localize(None)
    if h.empty:return rows,config,{'chosen':'NO_BET','reason':'no strictly prior settled history'}
    h=h.sort_values('date').drop_duplicates(['date','home_team','away_team'],keep='last').reset_index(drop=True)
    calibration_rebuilt=False
    if h['prob_iso_oos_time'].notna().sum()==0:
        from sklearn.isotonic import IsotonicRegression
        import numpy as np
        if 'home_team_prob' not in h:raise ValueError('Cannot reconstruct OOS calibration without raw predictions')
        h['home_team_prob']=pd.to_numeric(h['home_team_prob'],errors='coerce')
        h=h[h['home_team_prob'].between(0,1)].reset_index(drop=True)
        # Keep the legacy 50-row minimum / 10-row folds; exclude same-date outcomes.
        for start in range(50,len(h),10):
            val=h.iloc[start:start+10]
            train=h.iloc[:start]
            train=train[train['date']<val['date'].min()]
            if len(train)<50 or train['win'].nunique()<2:continue
            iso=IsotonicRegression(out_of_bounds='clip').fit(train['home_team_prob'],train['win'])
            h.loc[val.index,'prob_iso_oos_time']=iso.predict(val['home_team_prob'])
        calibration_rebuilt=True
    hist=r._prep_hist_df(h,tail_n=300)
    local,subset,wf,selected=r.find_best_local_params_with_fallback(h,local_tail_ladder=r.LOCAL_TAIL_LADDER,homewr_grid=r.HOMEWR_MIN_GRID,odds_min_grid=r.ODDS_MIN_GRID,odds_max_grid=r.ODDS_MAX_GRID,prob_min_grid=r.PROB_MIN_GRID,flat_stake_backtest=100,min_ev=0,min_trades_train=20,score_mode=r.SCORE_MODE,top_k=r.TOP_K_CANDIDATES,stop_at_first_pass=True)
    eval_hist=selected if selected is not None and not selected.empty else hist
    chosen,info=r.choose_profitable_config_dual(eval_hist,eval_hist,config['watch_reference_params'],local,min_ev=0)
    config['canonical_params']=chosen
    config['baseline_engine_state']=info['chosen']
    proxy=build_live_oos_proxy(h,prob_source_cols=['prob_iso_oos_time'],target_col='win',n_bins=25,min_train_rows=300,min_bin_n=25,use_wilson_lb=True,smoothing_alpha=1,date_col='date',home_col='home_team',away_col='away_team',recent_n=None)
    prepared=[]
    for row in rows:
        frame=pd.DataFrame([row]);frame['date']=row.get('game_date')
        if 'prob_iso' in frame and 'home_team_prob' in frame:
            frame=apply_live_oos_proxy(frame,proxy,in_col='prob_iso',blend_col='prob_iso',blend_alpha=.8,small_bin_alpha=.95,min_bin_n_for_full_proxy=200)
            frame['live_oos_proxy_ready']=bool(proxy.ready)
            prepared.append(frame.iloc[0].to_dict())
        else:prepared.append(row)
    return prepared,config,{'chosen':info,'walk_forward':wf,'calibration_rebuilt':calibration_rebuilt,'same_date_outcomes_excluded':True,'history_rows':len(h),'proxy_ready':bool(proxy.ready),'history_cutoff_utc':now.isoformat()}
