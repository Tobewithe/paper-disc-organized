"""Independent arithmetic/receipt checks and descriptive presentation, no inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics/three_region_20260911'
COMMON=HERE/'diagnostics/common_difference_20260911'
REPORT=HERE.parents[1]/'refine-logs/coco-structure'


def read(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def csvwrite(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    checks=[]
    for root,receipt,key in [(OUT,'COMPLETE.json','hashes'),(OUT,'SUMMARY_COMPLETE.json','files'),(COMMON,'COMPLETE.json','hashes')]:
        r=json.loads((root/receipt).read_text());assert r['status']=='COMPLETE'
        for name,digest in r[key].items():
            assert sha(root/name)==digest,(root,name);checks.append(str(root/name))
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['script_sha256']==sha(HERE/'three_region_probe.py')
    assert json.loads((OUT/'SUMMARY_COMPLETE.json').read_text())['source_sha256']==sha(HERE/'summarize_three_region.py')
    rows=read(OUT/'readouts.csv');target=read(OUT/'target_readouts.csv');summary=read(OUT/'summary.csv')
    metrics=['auc_neighbor','auc_background','auc_mixed','coverage','neighbor_fpr','background_fpr','sampled_precision',
             'zero_coverage','zero_neighbor_fpr','zero_background_fpr','train_coverage']
    idfields=['split','image_id','target_annotation','other_annotation','domain','readout']
    groups=defaultdict(list);rawbyid={};residual=0.
    for r in rows:
        for m in metrics:assert 0<=float(r[m])<=1,(m,r)
        residual=max(residual,abs(float(r['auc_mixed'])-(float(r['auc_neighbor'])+float(r['auc_background']))/2))
        den=sum(float(r[m]) for m in ['coverage','neighbor_fpr','background_fpr'])
        expected=float(r['coverage'])/den if den else 0
        residual=max(residual,abs(expected-float(r['sampled_precision'])))
        key=tuple(r[k] for k in idfields);groups[key].append(r)
        rid=tuple(r[k] for k in idfields+['fold']);assert rid not in rawbyid;rawbyid[rid]=r
    for r in rows:
        if r['readout']=='foreground_safe':
            key=tuple('actual_own' if k=='readout' else r[k] for k in idfields+['fold']);control=rawbyid[key]
            for m in ['zero_coverage','zero_neighbor_fpr','zero_background_fpr']:
                assert float(r[m])<=float(control[m])+1e-14,(key,m)
    for r in target:
        if r['readout']=='shuffle_proto_mean':
            rr=[rawbyid[tuple(f'shuffle_proto_s{s}' if k=='readout' else str(f) if k=='fold' else r[k] for k in idfields+['fold'])] for s in range(3) for f in range(2)]
        else:
            rr=groups[tuple(r[k] for k in idfields)]
            assert len(rr)==2 and {q['fold'] for q in rr}=={'0','1'}
        for m in metrics:residual=max(residual,abs(float(r[m])-np.mean([float(q[m]) for q in rr])))
    indexed={}
    for s in summary:
        rr=[r for r in target if r['split']==s['split'] and r['domain']==s['domain'] and r['readout']==s['readout'] and
            (s['group']=='all' or (float(r['target_ici'])>.5+1e-10)==(s['group']=='high'))]
        assert len(rr)==int(s['targets']) and len({r['image_id'] for r in rr})==int(s['images'])
        indexed[(s['split'],s['domain'],s['group'],s['readout'])]={ (r['image_id'],r['target_annotation']):r for r in rr}
        for m in metrics:residual=max(residual,abs(float(s[m])-np.mean([float(r[m]) for r in rr])))
    contrasts=json.loads((OUT/'PAIRED_ANALYSIS.json').read_text())['contrasts'];ci_checked=0;cires=0.
    for c in contrasts:
        key=(c['split'],c['domain'],c['group']);a=indexed[(*key,c['treatment'])];b=indexed[(*key,c['control'])];assert set(a)==set(b)
        delta={k:float(a[k][c['metric']])-float(b[k][c['metric']]) for k in a}
        residual=max(residual,abs(c['mean_pp']/100-np.mean(list(delta.values()))))
        if c['split']=='val' and c['group']=='high' and c['control']=='actual_own' and c['metric'] in ['auc_neighbor','auc_background','auc_mixed','coverage']:
            ids=protocol['images'][c['split']];lookup={str(iid):i for i,iid in enumerate(ids)}
            counts=np.zeros(len(ids));sums=np.zeros(len(ids))
            for (iid,aid),value in delta.items():counts[lookup[iid]]+=1;sums[lookup[iid]]+=value
            draw=np.random.default_rng(20260911).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
            denominator=draw@counts;boot=(draw@sums)[denominator>0]/denominator[denominator>0]*100
            low,high=np.quantile(boot,[.025,.975]);cires=max(cires,abs(low-c['ci_low_pp']),abs(high-c['ci_high_pp']));ci_checked+=1
    assert residual<1e-12 and cires<1e-10,(residual,cires)
    commonrows=read(COMMON/'readouts.csv');cg=defaultdict(list)
    cm=['identity_auc','own_background_auc','neighbor_background_auc','union_background_auc','own_positive','neighbor_positive','background_positive']
    for r in commonrows:
        key=tuple(r[k] for k in idfields);cg[key].append(r)
        assert abs(float(r['union_background_auc'])-(float(r['own_background_auc'])+float(r['neighbor_background_auc']))/2)<1e-12
        assert float(r['algebra_max_abs'])<1e-10
    cp=[]
    for rr in cg.values():
        assert len(rr)==2
        r={k:rr[0][k] for k in idfields+['target_ici']};r.update({m:float(np.mean([float(q[m]) for q in rr])) for m in cm});cp.append(r)
    cs=[]
    for split in ['train','val']:
        for domain in ['predicted_crop','expand_each_side_20pct']:
            for group in ['all','high','low']:
                for name in ['own','neighbor','common_mean','half_difference']:
                    rr=[r for r in cp if r['split']==split and r['domain']==domain and r['readout']==name and
                        (group=='all' or (float(r['target_ici'])>.5+1e-10)==(group=='high'))]
                    if not rr:continue
                    cs.append(dict(split=split,domain=domain,group=group,readout=name,targets=len(rr),images=len({r['image_id'] for r in rr}),**{m:float(np.mean([r[m] for r in rr])) for m in cm}))
    csvwrite(COMMON/'target_readouts.csv',cp);csvwrite(COMMON/'summary.csv',cs)
    # Identity AUC of common and half-difference must agree with direct parent arithmetic.
    for r in cp:
        if r['readout'] not in ['own','half_difference']:continue
        arm='actual_own' if r['readout']=='own' else 'signed_difference'
        prior=indexed[(r['split'],r['domain'],'all',arm)][(r['image_id'],r['target_annotation'])]
        assert abs(r['identity_auc']-float(prior['auc_neighbor']))<1e-12
        assert abs(r['own_background_auc']-float(prior['auc_background']))<1e-12
    write(COMMON/'SUMMARY_COMPLETE.json',dict(status='COMPLETE',post_hoc=True,source_sha256=sha(__file__),files={p.name:sha(p) for p in [COMMON/'target_readouts.csv',COMMON/'summary.csv']}))
    write(OUT/'LOCAL_VERIFICATION.json',dict(status='PASS',type='deterministic file and arithmetic verification, not scientific endorsement',
        hashed_receipt_entries=len(checks),raw_rows=len(rows),target_rows=len(target),summary_rows=len(summary),paired_deltas_checked=len(contrasts),
        recomputed_ci_count=ci_checked,max_arithmetic_abs=float(residual),max_ci_abs_pp=float(cires),foreground_safe_zero_subset=True,
        common_replay=json.loads((COMMON/'COMPLETE.json').read_text()),source_sha256=sha(__file__)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    styles=[('actual_own','Original response','#333333'),('signed_difference','Response difference','#d55e00'),
            ('oracle_two_logits','Two-response GT oracle','#0072b2'),('oracle_proto32','32D prototype GT oracle','#009e73')]
    for ax,domain,label in zip(axes,['predicted_crop','expand_each_side_20pct'],['Normal support','Expand 20% on each side']):
        for arm,text,color in styles:
            r=next(r for r in summary if r['split']=='val' and r['group']=='high' and r['domain']==domain and r['readout']==arm)
            ax.plot(range(3),[100*float(r[m]) for m in ['auc_neighbor','auc_background','auc_mixed']],marker='o',lw=2,label=text,color=color)
        ax.set_title(f"{label}\n{r['targets']} high-ICI targets / {r['images']} images")
        ax.set_xticks(range(3),['Own vs\nneighbor','Own vs\nbackground','Own vs\nmixture']);ax.set_ylim(65,100);ax.grid(axis='y',alpha=.18);ax.set_xlim(-.25,2.25)
    axes[0].set_ylabel('Held-out pixel AUC (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.0))
    fig.suptitle('Frozen COCO diagnostic: relative separation can lose foreground discrimination',fontsize=12)
    fig.subplots_adjust(bottom=.26,top=.79,wspace=.16,left=.07,right=.98)
    dest=REPORT/'figures';dest.mkdir(exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(dest/f'THREE_REGION_20260911.{suffix}',dpi=180)
    print(json.dumps(dict(verification='PASS',raw_rows=len(rows),max_abs=residual,ci_count=ci_checked)))
    for r in cs:
        if r['split']=='val' and r['group']=='high':print(json.dumps(r))


if __name__=='__main__':main()
