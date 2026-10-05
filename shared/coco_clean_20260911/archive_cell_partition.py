"""S053 final report and measurement archive; inference receipt stays immutable."""
from pathlib import Path
import hashlib,json,shutil,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent.parent;OUT=BASE/'diagnostics/feature_cell_partition_20260912'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    comp=json.loads((OUT/'COMPLETE.json').read_text())
    for p,h in comp['hashes'].items():assert sha(OUT/p)==h,p
    report=ROOT/'refine-logs/coco-structure/CELL_PARTITION_RESULTS_20260912.md'
    for name in ['CELL_PARTITION_PLAN_20260912.md','CELL_PARTITION_RESULTS_20260912.md']:shutil.copy2(ROOT/'refine-logs/coco-structure'/name,OUT/name)
    for p in [ROOT/'RESEARCH_FACTS.md',Path(__file__)]:shutil.copy2(p,OUT/p.name)
    missing=[s for s in re.findall(r'\]\((C:/[^)]+)\)',report.read_text(encoding='utf8')) if Path(s).name!='ANALYSIS_RECEIPT.json' and not Path(s).exists()];assert not missing,missing
    assert len(list((OUT/'pairs').glob('*/PATCHES.json')))==15;assert len(list((OUT/'pairs').glob('*/p3_patch_values.npz')))==15
    result=dict(experiment='S053',status='COMPLETE',verified_original_files=len(comp['hashes']),inference_receipt_sha256=sha(OUT/'COMPLETE.json'),report=str(report),report_sha256=sha(report),
        source_S052=str(OUT.parent/'proto_feature_swap_20260912_v2/ANALYSIS_RECEIPT.json'),source_S052_sha256=sha(OUT.parent/'proto_feature_swap_20260912_v2/ANALYSIS_RECEIPT.json'),
        console_sha256=sha(OUT.parent/'feature_cell_partition_20260912.console.log'),scope='15 previously explored images, one fillseed each type, three geometry subsets averaged withinimage, fixedc0b0, no training. PointwiseCI without multiplicitycorrection; directmixed-minus-exclusive unconfirmed.',
        hashes={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='ANALYSIS_RECEIPT.json'})
    (OUT/'ANALYSIS_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for p,h in result['hashes'].items():assert sha(OUT/p)==h,p
    print(json.dumps(dict(status='COMPLETE',original_files=len(comp['hashes']),archived_files=len(result['hashes']),report_links_valid=True)))
if __name__=='__main__':main()
