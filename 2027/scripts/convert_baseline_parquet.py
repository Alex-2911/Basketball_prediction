"""Materialize the latest baseline tables without concatenating daily snapshots."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/baseline_2026'

def main():
 mappings={'whole_statistics_2026':BASE/'Gathering_Data/Whole_Statistic/nba_games_2026-06-14.csv','combined_predictions_2026':max((BASE/'LightGBM').glob('combined_nba_predictions_acc_*.csv')),'script11_history_2026':BASE/'LightGBM/script11_watchlist_history_latest.csv','script7_reconciliation_2026':BASE/'LightGBM/model_vs_actual_vs_user_bets_latest.csv'}
 results=[]
 for name,source in mappings.items():
  frame=pd.read_csv(source,low_memory=False)
  dest=ROOT/'data/processed'/f'{name}.parquet'
  if dest.exists():raise FileExistsError(dest)
  frame.to_parquet(dest,index=False,compression='zstd')
  loaded=pd.read_parquet(dest);pd.testing.assert_frame_equal(frame,loaded)
  results.append({'table':name,'source':str(source.relative_to(ROOT)),'rows':len(frame),'columns':len(frame.columns),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'parquet_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'parquet_bytes':dest.stat().st_size,'roundtrip_verified':True})
 (ROOT/'configs/parquet_manifest.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
