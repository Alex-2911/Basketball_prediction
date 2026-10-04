"""Reconcile exported Steadivus cases with separately supplied settled results."""
import argparse,json
from pathlib import Path
import pandas as pd
from nba2027.legacy.script7_cases import load_betting_agent_training_cases

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cases-dir',type=Path,required=True);ap.add_argument('--results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 cases=load_betting_agent_training_cases([a.cases_dir]);results=pd.read_csv(a.results)
 for col in ['date','home_team','away_team','win']:
  if col not in results:raise ValueError('Missing result field: '+col)
 results['date']=pd.to_datetime(results['date'],errors='raise').dt.normalize()
 if results.duplicated(['date','home_team','away_team']).any():raise ValueError('Duplicate settled game results')
 results['win']=pd.to_numeric(results['win'],errors='raise')
 if not results['win'].isin([0,1]).all():raise ValueError('Results must be settled binary home wins')
 merged=cases.merge(results[['date','home_team','away_team','win']],on=['date','home_team','away_team'],how='left',validate='many_to_one')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x') as f:merged.to_csv(f,index=False)
 print(json.dumps({'cases':len(merged),'settled_matches':int(merged['win'].notna().sum()),'unmatched':int(merged['win'].isna().sum())}))
if __name__=='__main__':main()
