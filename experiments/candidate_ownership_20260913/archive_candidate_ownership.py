from pathlib import Path
import hashlib,json,shutil,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent;OUT=BASE/'diagnostics/candidate_ownership_20260913';REPORT=ROOT/'refine-logs/coco-structure/CANDIDATE_OWNERSHIP_RESULTS_20260913.md'
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'COMPLETE.json').read_text());
 for p,h in c['hashes'].items(): assert sha(OUT/p)==h,p
 shutil.copy2(REPORT,OUT/REPORT.name);shutil.copy2(Path(__file__),OUT/Path(__file__).name);shutil.copy2(ROOT/'RESEARCH_FACTS.md',OUT/'RESEARCH_FACTS.md')
 assert all(Path(x).exists() for x in re.findall(r'\]\((C:/[^)]+)\)',REPORT.read_text(encoding='utf8')))
 r=dict(status='COMPLETE',experiment='S054',inference_receipt_sha256=sha(OUT/'COMPLETE.json'),report_sha256=sha(REPORT),verified_images=c['images'],matched_images= c['matched'],scope='15 fixed same-neighbor failures; GT-selected neighbor candidate diagnostic; no training or input edits.',hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='ANALYSIS_RECEIPT.json'})
 (OUT/'ANALYSIS_RECEIPT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps({'status':'COMPLETE','files':len(r['hashes'])}))
if __name__=='__main__':main()
