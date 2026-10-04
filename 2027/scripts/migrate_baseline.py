"""Copy historical references; never run legacy code or mutate the source."""
from pathlib import Path
import hashlib,json,shutil,ast,re
SOURCE=Path('/Users/alexanderrazmyslov/1. Python/1. NBA Script/2026')
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/baseline_2026'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def redact_code(text):
 # Remove embedded credential assignments from executable reference source.
 text=re.sub(r'(?im)^(\s*(?:[\w]*(?:api_?key|token|password|secret)[\w]*)\s*(?::[^=\n]+)?=\s*)[\'\"][^\n]*',r'\1None # credential omitted',text)
 # Provider key formats also excluded from comments/output.
 text=re.sub(r'\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{20,})\b','REDACTED_CREDENTIAL',text)
 return text

def main():
 BASE.mkdir(parents=True,exist_ok=True)
 files=[]
 for f in SOURCE.rglob('*'):
  rel=f.relative_to(SOURCE)
  if not f.is_file() or f.is_symlink():continue
  if any(part.startswith('.') or part in ['venv','__pycache__','sorare_mlb','chrome_profiles','chromedriver','Steadivus Betting Mode.app','iconbuild'] for part in rel.parts):continue
  if f.suffix.lower() not in ['.csv','.json','.jsonl','.xlsx','.py','.ipynb','.md','.txt','.model','.js','.css','.html']:continue
  if f.suffix=='.html' and rel.parts[0]!='steadivus_betting_mode_tool':continue
  if 'manifold_execution_log' in f.name or f.name in ['manifold_api_client.py','tmp_gw29_status.py']:continue
  files.append((f,rel))
 utility=SOURCE/'nba_utils_2026.py'
 if utility.exists():files.append((utility.resolve(),Path('shared_utils/nba_utils_2026.py')))
 entries=[];blobs={};total=0
 for f,rel in files:
  dst=BASE/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  before=digest(f);transformed=False
  if dst.exists():raise RuntimeError('Destination already exists: '+str(rel))
  if f.suffix in ['.py','.ipynb','.js','.html','.md']:
   txt=f.read_text(errors='strict')
   if f.suffix=='.ipynb':
    nb=json.loads(txt)
    for cell in nb['cells']:
     cell['source']=redact_code(''.join(cell.get('source',[]))).splitlines(keepends=True)
     if cell['cell_type']=='code':cell['outputs']=[];cell['execution_count']=None
    nb['metadata'].pop('widgets',None)
    txt=json.dumps(nb,ensure_ascii=False,indent=1)+'\n'
   else:txt=redact_code(txt)
   dst.write_text(txt);transformed=digest(dst)!=before
  else:
   # Copies are independent from 2026, even when historical snapshots repeat.
   shutil.copyfile(f,dst)
  after=digest(f)
  if before!=after:raise RuntimeError('Source changed during migration: '+str(rel))
  copied=digest(dst)
  if not transformed and copied!=before:raise RuntimeError('Copy mismatch: '+str(rel))
  entries.append({'relative_path':str(rel),'source_sha256':before,'copied_sha256':copied,'size_bytes':dst.stat().st_size,'reference_transformed':transformed})
  total+=dst.stat().st_size
 (ROOT/'configs/migration_manifest.json').write_text(json.dumps({'source_root':str(SOURCE),'source_of_truth_evidence':['NEXT_SEASON_SYNC_NOTE.md','June 14 statistical snapshots','June 13 final Script 11 output'],'files':entries,'file_count':len(entries),'bytes':total,'exclusions':['credentials/environment files','browser profiles and binaries','virtual environments/caches','Sorare files','live Manifold client and execution logs','scraped HTML cache','launchers and scheduled jobs'],'note':'Legacy references may contain 2026 paths. Never execute them. Notebook outputs and embedded credential assignments removed.'},indent=2)+'\n')
 print(json.dumps({'copied_files':len(entries),'bytes':total,'transformed_references':sum(e['reference_transformed'] for e in entries)}))
if __name__=='__main__':main()
