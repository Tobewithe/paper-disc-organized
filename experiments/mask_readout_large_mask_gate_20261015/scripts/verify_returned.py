"""Post-run independent arithmetic and source-selection checks; never starts models."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.stats import hypergeom


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def lines(path):
    with path.open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def rle_digest(rle):
    return hashlib.sha256(json.dumps(rle,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def integer_rle_area(rle):
    # Decode signed 5-bit differential counts in Python, independently of mask.area.
    encoded=rle['counts'];counts=[];pos=0
    while pos<len(encoded):
        value=0;shift=0
        while True:
            char=ord(encoded[pos])-48;pos+=1
            assert 0<=char<=63
            value |= (char&31)<<shift;shift+=5
            if not char&32:
                if char&16:value |= -1<<shift
                break
        if len(counts)>2:value+=counts[-2]
        assert value>=0
        counts.append(value)
    assert sum(counts)==rle['size'][0]*rle['size'][1]
    return sum(counts[1::2])


def close(a,b):
    assert abs(a-b)<1e-12,(a,b)


def bootstrap_independently(rows,target):
    groups=defaultdict(list)
    for row in rows:
        if not target or row['population']=='T':groups[row['image_id']].append(row['delta_iou'])
    values=[sum(groups[i])/len(groups[i]) for i in sorted(groups)]
    rng=np.random.default_rng(1729);draws=[]
    for _ in range(2000):
        sampled=rng.integers(0,len(values),size=len(values))
        draws.append(sum(values[j] for j in sampled)/len(values))
    return dict(mean=sum(values)/len(values),ci95=np.quantile(draws,[.025,.975]).tolist(),images=len(values))


def verify(run,root):
    meta=read(run/'run.json');summary=read(run/'SUMMARY.json');contract=read(run/'EVAL_CONTRACT.json')
    assert meta['execution_status']=='completed' and meta['exit_code']==0
    assert read(run/'CONFIG_AUDIT.json')['status']=='passed'
    assert digest(run/'PROTOCOL.md')==meta['protocol_sha256']
    assert digest(run/'EVAL_CONTRACT.json')==meta['contract_sha256']
    artifacts=read(run/'ARTIFACT_MANIFEST.json')['artifacts']
    for a in artifacts:
        p=run/a['relative_path'];assert p.stat().st_size==a['size_bytes'] and digest(p)==a['sha256'],str(p)
    for rel,h in read(run/'SOURCE_MANIFEST.json')['files'].items():
        assert digest(run/'source_snapshot'/rel)==h
        assert digest(root/rel)==h
    parent=root/'experiments/mask_readout_soft_iou_pilot_20261014/runs/RUN_SOFT_IOU_EVAL_f3177058b874484f9576dd5a3674b9a0'
    assert digest(run/'DAMAGE_PANEL.json')==digest(parent/'DAMAGE_PANEL.json')==contract['damage_panel_sha256']
    assert contract['area_min']==9216 and contract['new_training_steps']==0 and not contract['fresh_95_coverage_guarantee']
    assert read(run/'PANEL_REUSE.json')['current_contract_sha256']==digest(run/'EVAL_CONTRACT.json')
    # Every selected RLE must come from the original corrected source, with all other fields unchanged.
    source_path=(parent/'B_predictions.json').relative_to(root).as_posix()
    with zipfile.ZipFile(run.parents[1]/'RESULTS_BUNDLE.zip') as archive:
        raw=archive.read(source_path)
    assert hashlib.sha256(raw).hexdigest()==contract['input_hashes']['files'][source_path]
    corrected=json.loads(raw);del raw
    parent_indices=read(parent/'B_ROW_MAPPING.json')['original_global_indices']
    source_pos={idx:i for i,idx in enumerate(parent_indices)}
    native=read(run/'NATIVE_SUBSET_predictions.json');mixed=read(run/'MIXED_predictions.json')
    rowmap=read(run/'ROW_MAPPING.json');indices=rowmap['original_global_indices']
    gates=list(lines(run/'GATE_ROWS.jsonl'))
    assert len(native)==len(mixed)==len(indices)==len(gates)
    counts=Counter();choice={}
    for j,(a,b,g,idx) in enumerate(zip(native,mixed,gates,indices)):
        c=corrected[source_pos[idx]]
        assert g['mixed_index']==j and g['global_native_index']==idx
        assert a.keys()==b.keys()==c.keys()
        assert all(a[k]==b[k]==c[k] for k in a if k!='segmentation')
        area=integer_rle_area(a['segmentation']);apply=area>=9216
        assert area==g['native_mask_area'] and apply==g['apply_delta']
        expected=c['segmentation'] if apply else a['segmentation']
        assert b['segmentation']==expected
        assert rle_digest(expected)==g['mixed_rle_sha256']
        assert rle_digest(c['segmentation'])==g['corrected_rle_sha256']
        assert rle_digest(a['segmentation'])==g['native_rle_sha256']
        counts['selected_outputs']+=apply;counts['changed_masks']+=a['segmentation']!=expected
        counts['outputs']+=1;choice[idx]=(area,apply)
    for k,v in counts.items():assert summary['coverage'][k]==v
    old={(r['image_id'],r['annotation_id']):r for r in lines(parent/'B_FIXED_PAIR_EFFECTS.jsonl')}
    effects=list(lines(run/'FIXED_PAIR_EFFECTS.jsonl'))
    for r in effects:
        prev=old[r['image_id'],r['annotation_id']]
        assert r['baseline_output_detection_id']==prev['baseline_output_detection_id']
        area,apply=choice[int(r['baseline_output_detection_id'].rsplit(':',1)[1])]
        assert r['native_mask_area']==area and r['apply_delta']==apply
        close(r['baseline_iou'],prev['baseline_iou'])
        close(r['method_iou'],prev['method_iou'] if apply else prev['baseline_iou'])
        close(r['delta_iou'],r['method_iou']-r['baseline_iou'])
    baseline={(r['image_id'],r['annotation_id']):r for r in lines(run/'BASELINE_SUBSET_GT_LEDGER.jsonl')}
    ledger={(r['image_id'],r['annotation_id']):r for r in lines(run/'MIXED_GT_LEDGER.jsonl')}
    assert baseline.keys()==ledger.keys()
    for key,r in ledger.items():
        oldrow=baseline[key]
        for f in ('is_ignored','gt_category_id','gt_area','task_box75'):assert r[f]==oldrow[f]
        oid=r['bbox75_match_output_id']
        mapped=None if oid is None else rowmap['source_prediction_sha256']+':'+str(indices[int(oid.rsplit(':',1)[1])])
        assert mapped==oldrow['bbox75_match_output_id']
    result=dict(run_id=meta['run_id'],status='passed',artifact_files=len(artifacts),source_files=len(read(run/'SOURCE_MANIFEST.json')['files']),
                verified_output_rows=len(native),verified_fixed_pairs=len(effects),verified_GT=len(ledger),
                pure_python_RLE_area_and_full_source_selection=True,config_audit_sha256=digest(run/'CONFIG_AUDIT.json'),
                run_json_sha256=digest(run/'run.json'),mixed_predictions_sha256=digest(run/'MIXED_predictions.json'))
    if meta['scope']=='full':
        assert indices==parent_indices and len(effects)==2925 and len(native)==185621
        for key,target in [('target',True),('all_bbox75',False)]:
            stat=bootstrap_independently(effects,target);actual=summary[key]
            assert stat['images']==actual['images']
            close(stat['mean'],actual['mean'])
            for a,b in zip(stat['ci95'],actual['ci95']):close(a,b)
            result[key]=stat
        panel=read(run/'DAMAGE_PANEL.json');sample=panel['sample']
        x=sum(ledger[r['image_id'],r['annotation_id']]['task_mask75']=='failure' for r in sample)
        N=22957;n=len(sample);assert n==1024
        low=x;high=N-(n-x)
        while low<high:
            mid=(low+high+1)//2
            if hypergeom.cdf(x,N,mid,n)>=.05:low=mid
            else:high=mid-1
        K=low;bound=K/N
        assert hypergeom.cdf(x,N,K,n)>=.05 and hypergeom.cdf(x,N,K+1,n)<.05
        assert summary['damage']['damaged']==x and summary['damage']['upper_damaged_count']==K
        close(bound,summary['damage']['upper_rate'])
        decision=dict(target_effect=summary['target']['mean']>=.005 and summary['target']['ci95'][0]>0,
                      all_bbox75_guardrail=summary['all_bbox75']['mean']>=-.001,success_damage=bound<=.01)
        assert decision==summary['gates']
        result.update(damaged=x,upper_damaged_count=K,upper_rate=bound,gates=decision,
                      independent_bound_method='SciPy hypergeom CDF inversion, boundary verified',
                      interpretation='Reused development panel; no new independent 95% coverage guarantee')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[3]
    result=verify(args.run.resolve(),root)
    with args.out.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result,ensure_ascii=False))
