"""Seal S052 reporting without changing the inference receipt."""
from pathlib import Path
import json,hashlib,shutil,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent
OUT=BASE/'diagnostics/proto_feature_swap_20260912_v2';FAIL=OUT.parent/'proto_feature_swap_20260912'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
    comp=json.loads((OUT/'COMPLETE.json').read_text())
    for p,h in comp['hashes'].items():assert sha(OUT/p)==h,p
    deps=OUT/'execution_dependencies';deps.mkdir(exist_ok=True)
    files=[BASE/n for n in ['crowded_failure_branch_probe.py','subtype_neighbor_probe.py','summarize_subtype_neighbor.py']]
    # Preserve only dependencies which are actually present, recording explicit paths.
    copied=[]
    for p in files:
        if p.exists():shutil.copy2(p,deps/p.name);copied.append(dict(path=str(p),sha256=sha(p)))
    vendor=BASE/'local_readout_runtime_20260912/vendor/ultralytics'
    for name in ['nn/modules/block.py','nn/autobackend.py','models/yolo/segment/predict.py','engine/predictor.py']:
        p=vendor/name;dest=deps/('ultralytics_'+name.replace('/','_'));shutil.copy2(p,dest);copied.append(dict(path=str(p),sha256=sha(p)))
    for name in ['FEATURE_SWAP_PLAN_20260912.md','FEATURE_SWAP_RESULTS_20260912.md']:
        p=ROOT/'refine-logs/coco-structure'/name;shutil.copy2(p,OUT/name)
    for p in [ROOT/'RESEARCH_FACTS.md',Path(__file__)]:shutil.copy2(p,OUT/p.name)
    dump(FAIL/'FAILED.json',dict(status='FAILED_BEFORE_INTERVENTION',reason='Direct prototype call used CPU pre-AutoBackend object with CUDA inputs; active GPU inference copy used in v2.',intervention_rows=0,
        console_log=str(OUT.parent/'proto_feature_swap_20260912.console.log'),console_sha256=sha(OUT.parent/'proto_feature_swap_20260912.console.log'),
        hashes={str(p.relative_to(FAIL)):sha(p) for p in FAIL.rglob('*') if p.is_file() and p.name!='FAILED.json'}))
    report=ROOT/'refine-logs/coco-structure/FEATURE_SWAP_RESULTS_20260912.md'
    missing=[]
    for s in re.findall(r'\]\((C:/[^)]+)\)',report.read_text(encoding='utf8')):
        if Path(s).name!='ANALYSIS_RECEIPT.json' and not Path(s).exists():missing.append(s)
    assert not missing,missing
    assert len(list((OUT/'pairs').glob('*/feature_patch_masks.json')))==44
    receipt=dict(status='COMPLETE',experiment='S052',inference_receipt_sha256=sha(OUT/'COMPLETE.json'),verified_original_files=len(comp['hashes']),
        scope='44 fixed previously explored targets, two fills with seed0 only; image-bootstrap pointwise exploratory CI; fixed original c0/b0. All feature swap paths carry a change, not established origins.',
        source_dependencies=copied,report=str(report),report_sha256=sha(report),failure_receipt=str(FAIL/'FAILED.json'),failure_receipt_sha256=sha(FAIL/'FAILED.json'),
        verification='44 patch masks reconstructed to exactly match recorded target/edit/intersection counts; report links resolve. No new training or automation changes.',
        hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='ANALYSIS_RECEIPT.json'})
    dump(OUT/'ANALYSIS_RECEIPT.json',receipt)
    for p,h in receipt['hashes'].items():assert sha(OUT/p)==h,p
    print(json.dumps(dict(status='COMPLETE',original_hashes=len(comp['hashes']),total_archived_files=len(receipt['hashes']),report=str(report))))
if __name__=='__main__':main()
