"""Freeze validated run inputs; never synthesize missing preflight evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    receipts={}
    for name in ['DATA_PIPELINE_PASS.json','CCL_GRADIENT_PASS.json','SMOKE_PAIR_PASS.json']:
        r=json.loads((ROOT/'audits'/name).read_text());assert r['status']=='PASS',name;receipts[name]=r
    audit=json.loads((ROOT/'audits/EXPERIMENT_AUDIT.json').read_text())
    assert audit['launch_blockers_remaining']==[],audit
    assert audit['review_independence']=='same-family' and audit['acceptance_status']=='provisional'
    assert receipts['DATA_PIPELINE_PASS.json']['train_images']==37433
    assert receipts['DATA_PIPELINE_PASS.json']['val_images']==5000
    assert receipts['DATA_PIPELINE_PASS.json']['train_actual']==488870
    cfg=json.loads((ROOT/'train_config.json').read_text())
    assert cfg['epochs']==15 and cfg['save_period']==1 and cfg['batch']==16 and cfg['patience']==0
    census=json.loads((ROOT/'census/COCO2017_CROWDING_CENSUS.json').read_text())
    assert census['status']=='PASS'
    assert census['splits']['train2017']['legacy_manifest_check']['exact_match']
    assert census['splits']['val2017']['scopes']['full']['same_class_bins']['I3_>1']==1523
    assert shutil.disk_usage(ROOT).free>28*2**30,'Insufficient space for all requested epoch checkpoints'
    files=['train_pair.py','ccl_loss.py','run_queue.py','train_config.json','PROTOCOL.md','coco_clean.yaml',
        'weights/yolo26m-seg.pt','official_yolo/train2017.txt','official_yolo/val2017.txt',
        'data/annotations/instances_train2017.json','data/annotations/instances_val2017.json',
        'audits/actual_loader_train2017.json','audits/actual_loader_val2017.json',
        'audits/DATA_PIPELINE_PASS.json','audits/CCL_GRADIENT_PASS.json','audits/SMOKE_PAIR_PASS.json',
        'audits/EXPERIMENT_AUDIT.json','census/COCO2017_CROWDING_CENSUS.json',
        'census/COCO_EVAL_INSTANCE_MANIFEST.csv']
    result=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),seeds=[0,1,2],arms=['baseline','ccl01'],jobs=6,epochs=15,
        save_every_epoch=True,scientific_efficacy='UNTESTED',review_independence='same-family',acceptance_status='provisional',
        frozen_inputs_sha256={str(p):sha(ROOT/p) for p in files},
        gates=list(receipts),remaining_limits=['Official polygon/raster approximations retained identically in both arms',
            'Source identity does not propagate through stock mosaic; CCL uses distinct transformed GT indices',
            'Fullval evaluation must use untouched original COCO JSON and stabilized instance ICI bins',
            'Previously analyzed val2017 is not a new blind test set'])
    (ROOT/'audits/LAUNCH_GATE.json').write_text(json.dumps(result,indent=2))
    print('LAUNCH_GATE_PASS',json.dumps({k:v for k,v in result.items() if k!='frozen_inputs_sha256'}),flush=True)

if __name__=='__main__':main()
