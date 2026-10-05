"""Paired summaries of three-region within-image oracle diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from frozen_mechanism_probe import sha,write_json,write_csv


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out
    receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for filename,digest in receipt['hashes'].items():assert sha(out/filename)==digest,filename
    rows=read(out/'readouts.csv');statuses=read(out/'statuses.csv');protocol=json.loads((out/'protocol.json').read_text())
    metrics=['auc_neighbor','auc_background','auc_mixed','coverage','neighbor_fpr','background_fpr','sampled_precision',
             'zero_coverage','zero_neighbor_fpr','zero_background_fpr','train_coverage']
    grouped={}
    for r in rows:
        key=tuple(r[k] for k in ['split','image_id','target_annotation','other_annotation','domain','readout'])
        grouped.setdefault(key,[]).append(r)
    points=[]
    for key,rr in grouped.items():
        assert len(rr)==2 and {r['fold'] for r in rr}=={'0','1'}
        row={k:rr[0][k] for k in ['split','image_id','target_annotation','other_annotation','domain','readout','target_ici','category_id','gt_area']}
        row.update({m:float(np.mean([float(r[m]) for r in rr])) for m in metrics});points.append(row)
    temp={}
    for r in points:
        if r['readout'].startswith('shuffle_proto_s'):
            key=tuple(r[k] for k in ['split','image_id','target_annotation','domain'])
            temp.setdefault(key,[]).append(r)
    for rr in temp.values():
        assert len(rr)==3
        row={k:v for k,v in rr[0].items() if k not in metrics};row['readout']='shuffle_proto_mean'
        row.update({m:float(np.mean([r[m] for r in rr])) for m in metrics});points.append(row)
    summary=[];contrasts=[];counts=[]
    comparisons=[('signed_difference','actual_own'),('foreground_safe','actual_own'),
                 ('oracle_two_logits','actual_own'),('oracle_two_logits','foreground_safe'),
                 ('oracle_proto32','actual_own'),('oracle_proto32','oracle_two_logits'),
                 ('oracle_proto32','oracle_coordinate32'),('oracle_coordinate32','actual_own'),
                 ('oracle_proto32','shuffle_proto_mean')]
    for split,ids in protocol['images'].items():
        lookup={iid:i for i,iid in enumerate(ids)};draw=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
        ss=[r for r in statuses if r['split']==split]
        for status in sorted({r['status'] for r in ss}):
            rr=[r for r in ss if r['status']==status]
            counts.append(dict(split=split,status=status,records=len(rr),images=len({r['image_id'] for r in rr})))
        for domain in ['predicted_crop','expand_each_side_20pct']:
            for group in ['all','high','low']:
                pp=[r for r in points if r['split']==split and r['domain']==domain and
                    (group=='all' or (float(r['target_ici'])>.5+1e-10)==(group=='high'))]
                byarm={}
                for name in sorted({r['readout'] for r in pp}):
                    rr=[r for r in pp if r['readout']==name]
                    byarm[name]={(r['image_id'],r['target_annotation']):r for r in rr}
                    summary.append(dict(split=split,domain=domain,group=group,readout=name,targets=len(rr),images=len({r['image_id'] for r in rr}),
                          **{m:float(np.mean([r[m] for r in rr])) for m in metrics}))
                if not byarm:continue
                for treatment,control in comparisons:
                    if treatment not in byarm or control not in byarm:continue
                    assert set(byarm[treatment])==set(byarm[control])
                    for metric in metrics:
                        count=np.zeros(len(ids));sums=np.zeros(len(ids))
                        for key,r in byarm[treatment].items():
                            k=lookup[int(r['image_id'])];count[k]+=1;sums[k]+=r[metric]-byarm[control][key][metric]
                        den=draw@count;valid=den>0;boot=(draw@sums)[valid]/den[valid]*100
                        contrasts.append(dict(split=split,domain=domain,group=group,treatment=treatment,control=control,metric=metric,
                            n=int(count.sum()),images=int((count>0).sum()),mean_pp=float(sums.sum()/count.sum()*100),
                            ci_low_pp=float(np.quantile(boot,.025)),ci_high_pp=float(np.quantile(boot,.975))))
    # Domain comparisons only among the same targets eligible under both domains.
    domain_contrasts=[]
    for split,ids in protocol['images'].items():
        for name in sorted({r['readout'] for r in points}):
            aa={(r['image_id'],r['target_annotation']):r for r in points if r['split']==split and r['domain']=='predicted_crop' and r['readout']==name}
            bb={(r['image_id'],r['target_annotation']):r for r in points if r['split']==split and r['domain']=='expand_each_side_20pct' and r['readout']==name}
            common=set(aa)&set(bb)
            if common:domain_contrasts.append(dict(split=split,readout=name,paired_targets=len(common),
                 **{m:float(np.mean([bb[k][m]-aa[k][m] for k in common])*100) for m in metrics}))
    write_csv(out/'target_readouts.csv',points);write_csv(out/'summary.csv',summary);write_csv(out/'status_summary.csv',counts)
    write_json(out/'PAIRED_ANALYSIS.json',dict(scope='Within-image GT-informed cross-fit balanced pixel diagnostic, no model training or task AP. No shared interpolation cells across folds, deeper RF overlap remains. Paired image-cluster2000 pointwise intervals, no multiplicity correction; oracle capacity/geometry limitations; targets conditioned on matching and three-region pixel support.',
         contrasts=contrasts,paired_domain_expansion_minus_crop=domain_contrasts,eligibility=counts))
    write_json(out/'SUMMARY_COMPLETE.json',dict(status='COMPLETE',source_sha256=sha(__file__),
          files={p.name:sha(p) for p in [out/'summary.csv',out/'target_readouts.csv',out/'status_summary.csv',out/'PAIRED_ANALYSIS.json']}))
    for r in summary:
        if r['split']=='val' and r['group']=='high' and not r['readout'].startswith('shuffle_proto_s'):print(json.dumps(r),flush=True)


if __name__=='__main__':main()
