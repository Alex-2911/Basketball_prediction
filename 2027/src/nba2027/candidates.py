"""Offline pregame decision adapter; canonical selection remains frozen at June NO_BET."""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
from .legacy import script11 as rules

ROOT = Path(__file__).resolve().parents[2]

def utc(value):
    t = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if t.tzinfo is None:
        raise ValueError('A timezone-qualified timestamp is required')
    return t.astimezone(timezone.utc)

def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None

def evaluate(rows, *, now, config):
    if now.tzinfo is None:
        raise ValueError('now must be timezone-aware')
    if config.get('execution_enabled') is not False or config.get('paper_only') is not True:
        raise ValueError('This release only supports paper mode with execution disabled')
    keys = []
    for r in rows:
        keys.append((str(r.get('game_date','')), str(r.get('home_team','')).strip(),str(r.get('away_team','')).strip()))
    duplicates = {k for k in keys if keys.count(k)>1}
    outputs = []
    for r,key in zip(rows,keys):
        problems=[]
        values={k:number(r.get(k)) for k in ['home_team_prob','prob_iso','home_win_rate','odds_1','odds_2']}
        if any(v is None for v in values.values()):problems.append('DATA_INCOMPLETE')
        if any(values[k] is not None and not 0<=values[k]<=1 for k in ['home_team_prob','prob_iso','home_win_rate']):problems.append('INVALID_PROBABILITY')
        if any(values[k] is not None and values[k]<=1 for k in ['odds_1','odds_2']):problems.append('INVALID_ODDS')
        if not all(key) or key[1]==key[2]:problems.append('INVALID_GAME_IDENTITY')
        try:
            datetime.strptime(key[0],'%Y-%m-%d')
            start=utc(r['tipoff_utc']); asof=utc(r['as_of_utc'])
            if start<=now:problems.append('GAME_STARTED')
            if start>now+timedelta(hours=config['lookahead_hours']):problems.append('OUTSIDE_PREGAME_WINDOW')
            if asof>now or now-asof>timedelta(hours=config['max_input_age_hours']):problems.append('STALE_OR_FUTURE_INPUT')
            if asof>=start:problems.append('POSTGAME_INPUT')
        except (KeyError,ValueError,TypeError):problems.append('MISSING_OR_INVALID_TIMESTAMPS')
        if r.get('is_played') not in [False,0,'False','false','0']:problems.append('PLAYED_STATUS_NOT_CONFIRMED')
        if key in duplicates:problems.append('DUPLICATE_GAME')
        pproxy=number(r.get('prob_live_oos_proxy'))
        ready=r.get('live_oos_proxy_ready') in [True,1,'true','True','1']
        if ready and (pproxy is None or not 0<=pproxy<=1):problems.append('INVALID_OOS_PROXY')
        result={'game_key':'__'.join(key),'game_date':key[0],'home_team':key[1],'away_team':key[2], 'canonical_decision':'NO_BET','watch_label':'DATA_INCOMPLETE','engine_state':config['baseline_engine_state'],'paper_only':True,'execution_enabled':False,'rules_passed':0,'blocked_by':problems,'as_of_utc':now.isoformat(),'input_as_of_utc':r.get('as_of_utc'),'tipoff_utc':r.get('tipoff_utc')}
        if not problems:
            frame=pd.DataFrame([{**r,**values,'date':key[0],'live_oos_proxy_ready':ready}])
            calc=rules._compute_live_prob_path(frame).iloc[0]
            p=float(calc['prob_used']);ev=100*(p*values['odds_1']-1)
            params=config['watch_reference_params']
            flags=[values['home_win_rate']>=params['home_win_rate_threshold'],params['odds_min']<=values['odds_1']<=params['odds_max'],p>=max(params['prob_threshold'],rules.prob_clip_lo),ev>config['min_ev']]
            guard=bool(calc['live_underdog_upscale_guard_triggered'])
            label='PREGAME_WATCH_ONLY'
            if guard:label='RAW_MODEL_MARKET_GAP_HOME_DOG' if values['home_win_rate']>=.6 and 2<=values['odds_1']<=2.8 and float(calc['prob_base'])>=.6 and p<.55 else 'PREGAME_MARKET_GAP_WATCH'
            elif ev<=0:label='LOW_PRICE_NEGATIVE_EV' if values['home_win_rate']>=.5 and 1.3<=values['odds_1']<=1.7 and p>=.55 else 'NO_VALUE_SKIP'
            elif all(flags):label='PASS_FILTERS_ONLY'
            result.update(prob_base=float(calc['prob_base']),prob_used=p,ev_per_100=ev,rules_passed=sum(flags),watch_label=label,model_market_gap=float(calc['model_market_gap']),model_market_gap_guard=guard)
            selected=config.get('canonical_params')
            if selected is not None:
                canonical_flags=[values['home_win_rate']>=selected['home_win_rate_threshold'],selected['odds_min']<=values['odds_1']<=selected['odds_max'],p>=max(selected['prob_threshold'],rules.prob_clip_lo),ev>config['min_ev']]
                if all(canonical_flags) and not guard:
                    result['canonical_decision']='BET'
                    result['watch_label']='CANONICAL_MODEL_SIGNAL'
            result['blocked_by']=([] if result['canonical_decision']=='BET' else ['ENGINE_NO_BET' if selected is None else 'CANONICAL_FILTERS_FAILED'])+(['MODEL_MARKET_GAP'] if guard else [])+[name for name,ok in zip(['HWR','ODDS','PROBABILITY','EV'],canonical_flags if selected is not None else flags) if not ok]
        outputs.append(result)
    return outputs

def to_case(r):
    return {'input_context':{'game_date':r['game_date'],'home_team':r['home_team'],'away_team':r['away_team'],'as_of_utc':r['as_of_utc'],'model_numbers':{'prob_base':r.get('prob_base'),'prob_used':r.get('prob_used'),'ev_live_per_100':r.get('ev_per_100')},'rule_state':{'canonical_decision':r['canonical_decision'],'watchlist_decision':r['watch_label'],'engine_state':r['engine_state'],'rules_passed':r['rules_passed'],'blocked_by':' | '.join(r['blocked_by'])}},'decision_taken':{'actual_action':'PAPER_OBSERVATION','stake':0},'outcome':{'result':'PENDING','pnl':None},'agent_label':{'decision_class':r['watch_label']}}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--now');ap.add_argument('--history',type=Path);ap.add_argument('--output-dir',type=Path,default=ROOT/'outputs');a=ap.parse_args()
    cfg=json.loads((ROOT/'configs/baseline.json').read_text())
    now=utc(a.now) if a.now else datetime.now(timezone.utc)
    if a.input.suffix=='.parquet':rows=pd.read_parquet(a.input).to_dict('records')
    else:rows=json.loads(a.input.read_text())
    evidence={'chosen':'NO_BET','reason':'frozen June baseline; no history supplied'}
    if a.history:
        from .strategy import prepare
        history=pd.read_parquet(a.history) if a.history.suffix=='.parquet' else pd.read_csv(a.history)
        rows,cfg,evidence=prepare(history,rows,now=now,config=cfg)
    out=evaluate(rows,now=now,config=cfg)
    run=a.output_dir/now.strftime('%Y%m%dT%H%M%S%fZ');run.mkdir(parents=True,exist_ok=False)
    (run/'candidates.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    pd.DataFrame(out).to_csv(run/'candidates.csv',index=False)
    (run/'betting_agent_training_cases_latest.jsonl').write_text(''.join(json.dumps(to_case(r),allow_nan=False)+'\n' for r in out))
    (run/'strategy_evidence.json').write_text(json.dumps(evidence,indent=2,default=str)+'\n')
    (run/'manifest.json').write_text(json.dumps({'history_sha256':hashlib.sha256(a.history.read_bytes()).hexdigest() if a.history else None,'effective_strategy_config':cfg,'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'config_sha256':hashlib.sha256((ROOT/'configs/baseline.json').read_bytes()).hexdigest(),'created_utc':now.isoformat(),'mode':'paper','rows':len(out)},indent=2)+'\n')
    print(json.dumps({'run_dir':str(run),'rows':len(out),'bets':sum(r['canonical_decision']=='BET' for r in out),'execution_enabled':False}))

if __name__=='__main__':main()
