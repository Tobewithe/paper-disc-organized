"""S016: apply the frozen S015 diagnostic to all other-density no-good candidates."""
import argparse,contextlib,io,json,time
from pathlib import Path
import torch
from pycocotools.coco import COCO
from no_candidate_readout_probe import process
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'diagnostics/candidate_lineage300_20260912/gt.csv'
    selected=[r for r in read(source) if float(r['ici'])<=.5+1e-10 and r['official_mask75']=='False' and r['score_mask75']=='False']
    need(len(selected)==1021,'Locked low/other cohort must be1021')
    ids=sorted({int(r['image_id']) for r in selected})
    ref=ROOT/'diagnostics/no_candidate317_20260912/protocol.json';old=json.loads(ref.read_text())
    need(sha(Path(__file__).with_name('no_candidate_readout_probe.py'))==old['script_sha256'],'S015 implementation changed')
    protocol={**old,'images':ids,'target_ids':[int(r['annotation_id']) for r in selected],
        'cohort':'All S014 ICI<=.5 ordinary GT, official mask75 failed and no correct-argmax postconf mask75 candidate',
        'script_sha256':sha(__file__),'source_sha256':sha(source),'s015_protocol_sha256':sha(ref),
        'oracle_implementation_sha256':old['script_sha256'],
        'comparison':'Same explored 300-image source, same source-dependent anchor rule, same60+60solver. No filtering by effect. High/low failures condition on outcomes; no density causal claim.'}
    if a.resume:need(json.loads((a.out/'protocol.json').read_text())==protocol,'Protocol mismatch')
    else:a.out.mkdir(parents=True,exist_ok=False);(a.out/'images').mkdir();write_json(a.out/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    start=time.monotonic()
    for n,iid in enumerate(ids,1):
        path=a.out/'images'/f'{iid}.json'
        doc=json.loads(path.read_text()) if path.exists() else process(gt,iid,[r for r in selected if int(r['image_id'])==iid],a.out,60)
        progress=dict(images=n,total=len(ids),seconds=time.monotonic()-start,last_image=iid,last_seconds=doc['seconds'])
        write_json(a.out/'progress.json',progress)
        if n%10==0 or n==len(ids):print(json.dumps(progress),flush=True)
    allrows={k:[] for k in ['targets','metrics','optimization']}
    for iid in ids:
        doc=json.loads((a.out/'images'/f'{iid}.json').read_text())
        for k in allrows:allrows[k].extend(doc[k])
    for k,v in allrows.items():save_csv(a.out/f'{k}.csv',v)
    write_json(a.out/'COMPLETE.json',dict(status='COMPLETE',network_training=False,oracle=True,images=len(ids),targets=len(selected),
        seconds=time.monotonic()-start,hashes={str(q.relative_to(a.out)):sha(q) for q in a.out.rglob('*') if q.is_file()}))

if __name__=='__main__':main()
