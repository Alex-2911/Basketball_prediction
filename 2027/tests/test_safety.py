import copy,json,unittest
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from nba2027.candidates import evaluate,to_case,ROOT
from nba2027.execution import submit_order
from nba2027.legacy import script11 as r
from nba2027.legacy.script7_cases import _extract_matchup_teams, load_betting_agent_training_cases

class SafetyTests(unittest.TestCase):
 def setUp(self):
  self.cfg=json.loads((ROOT/'configs/baseline.json').read_text());self.now=datetime(2026,10,20,12,tzinfo=timezone.utc)
  self.row={'game_date':'2026-10-20','home_team':'BOS','away_team':'NYK','home_team_prob':.7,'prob_iso':.7,'home_win_rate':.7,'odds_1':1.7,'odds_2':2.2,'tipoff_utc':'2026-10-20T23:00:00Z','as_of_utc':'2026-10-20T11:00:00Z','is_played':False}
 def result(self,row=None,cfg=None):return evaluate([row or self.row],now=self.now,config=cfg or self.cfg)[0]
 def test_frozen_no_bet(self):self.assertEqual(self.result()['canonical_decision'],'NO_BET')
 def test_stale_started_missing_invalid(self):
  for changes,reason in [({'as_of_utc':'2026-10-18T11:00:00Z'},'STALE_OR_FUTURE_INPUT'),({'tipoff_utc':'2026-10-20T11:00:00Z'},'GAME_STARTED'),({'tipoff_utc':None},'MISSING_OR_INVALID_TIMESTAMPS'),({'odds_1':float('nan')},'DATA_INCOMPLETE'),({'home_team_prob':1.2},'INVALID_PROBABILITY')]:
   out=self.result({**self.row,**changes});self.assertIn(reason,out['blocked_by']);self.assertEqual(out['canonical_decision'],'NO_BET')
 def test_duplicates(self):
  out=evaluate([self.row,self.row],now=self.now,config=self.cfg)
  self.assertTrue(all('DUPLICATE_GAME' in x['blocked_by'] for x in out))
 def test_no_execution_even_if_config_changed(self):
  c=copy.deepcopy(self.cfg);c['execution_enabled']=True
  with self.assertRaises(ValueError):self.result(cfg=c)
  with self.assertRaises(RuntimeError):submit_order()
 def test_rule_boundary_and_paper_bet(self):
  c=copy.deepcopy(self.cfg);c['canonical_params']=c['watch_reference_params'];c['baseline_engine_state']='GLOBAL'
  row={**self.row,'home_team_prob':.63,'prob_iso':.63}
  out=self.result(row,c);self.assertEqual(out['canonical_decision'],'BET');self.assertFalse(out['execution_enabled'])
 def test_market_gap_guard(self):
  row={**self.row,'home_team_prob':.8,'prob_iso':.8,'odds_1':2.5,'odds_2':1.5}
  out=self.result(row);self.assertTrue(out['model_market_gap_guard']);self.assertEqual(out['canonical_decision'],'NO_BET')
 def test_strict_ev_zero(self):
  row={**self.row,'home_team_prob':.5,'prob_iso':.5,'odds_1':2,'odds_2':2}
  out=self.result(row);self.assertEqual(out['ev_per_100'],0);self.assertIn('EV',out['blocked_by'])
 def test_june_probability_parity(self):
  row={'home_team_prob':.513,'prob_iso':.475,'home_win_rate':.5,'odds_1':1.51,'odds_2':2.64,'live_oos_proxy_ready':True,'prob_live_oos_proxy':.468,'live_oos_proxy_bin_n':51}
  out=r._compute_live_prob_path(pd.DataFrame([row])).iloc[0]
  self.assertAlmostEqual(float(out['prob_used']),.598,places=3)
  self.assertAlmostEqual(100*(float(out['prob_used'])*1.51-1),-9.66,places=2)
 def test_stability_needs_two_available_windows(self):
  h=pd.DataFrame({'date':pd.date_range('2025-10-01',periods=300),'home_win_rate':.7,'odds_1':1.9,'prob_iso_oos_time':.6,'win':1})
  chosen,info=r.choose_profitable_config_dual(h,h,self.cfg['watch_reference_params'],None,min_ev=0)
  self.assertIsNone(chosen);self.assertEqual(info['chosen'],'NO_BET')
 def test_new_jsonl_script7_compatibility(self):
  import tempfile
  with tempfile.TemporaryDirectory() as td:
   p=Path(td);(p/'betting_agent_training_cases_latest.jsonl').write_text(json.dumps(to_case(self.result()))+'\n')
   out=load_betting_agent_training_cases([p]);self.assertEqual(len(out),1);self.assertEqual(out.iloc[0]['canonical_decision'],'NO_BET')
 def test_script7_halftime_phrase_normalization(self):
  self.assertEqual(_extract_matchup_teams('CLE @ NYK halftime comeback review'),('NYK','CLE'))
if __name__=='__main__':unittest.main()
