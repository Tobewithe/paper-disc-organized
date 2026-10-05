"""S031 fixed image-cluster paired contrasts and label-conflict summaries."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json


def read(p):
    with p.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--previous',type=Path,required=True);a=ap.parse_args()
    run=a.run;receipt=json.loads((run/'COMPLETE.json').read_text())
    for name in ['spatial.csv','labels.csv','solver.csv','witness.csv','pixels.csv']:
        if sha(run/name)!=receipt['hashes'][name]:raise RuntimeError('Changed result '+name)
    rows=read(run/'spatial.csv');labels=read(run/'labels.csv');old=read(a.previous/'spatial.csv')
    targets={int(r['annotation_id']):r for r in read(a.previous/'targets.csv')}
    modes=['original','native_overlap','native_independent','raw_coco'];metrics=['coco_iou','coverage','neighbor','background']
    data={(int(r['annotation_id']),r['mode']):r for r in rows}
    parity=[]
    for row in old:
        if row['mode'] not in ['original','ridge_0.01']:continue
        mode='native_overlap' if row['mode']=='ridge_0.01' else 'original'
        new=data[int(row['annotation_id']),mode]
        for metric in metrics:
            if row[metric]!=new[metric]:raise RuntimeError('S030 replay differed')
        parity.append(1)
    means=[];contrasts=[]
    pairs=[('native_overlap','original'),('native_independent','native_overlap'),('raw_coco','native_independent'),('raw_coco','native_overlap'),('raw_coco','original')]
    for split in ['fit','transfer','combined']:
        aids=sorted(t for t,r in targets.items() if split=='combined' or r['split']==split)
        imgs=sorted({int(targets[t]['image_id']) for t in aids});ii={v:k for k,v in enumerate(imgs)}
        ix=np.array([ii[int(targets[t]['image_id'])] for t in aids]);hi=np.array([targets[t]['high']=='True' for t in aids])
        draws=np.random.default_rng(20260912).multinomial(len(imgs),np.ones(len(imgs))/len(imgs),size=2000)
        arrays={m:np.array([[float(data[t,m][f]) if data[t,m][f] else np.nan for f in metrics] for t in aids]) for m in modes}
        for group,mask in [('all',np.ones(len(aids),bool)),('high',hi),('other',~hi)]:
            for m,vals in arrays.items():means.append(dict(split=split,group=group,mode=m,n=int(mask.sum()),**{f:float(np.nanmean(vals[mask,k])) for k,f in enumerate(metrics)}))
            for left,right in pairs:
                dd=arrays[left]-arrays[right]
                for k,f in enumerate(metrics):
                    good=mask&np.isfinite(dd[:,k]);count=np.bincount(ix[good],minlength=len(imgs)).astype(float)
                    total=np.bincount(ix[good],weights=dd[good,k],minlength=len(imgs))
                    den=np.einsum('bi,i->b',draws,count,optimize=False);num=np.einsum('bi,i->b',draws,total,optimize=False)
                    boot=num[den>0]/den[den>0]
                    contrasts.append(dict(split=split,group=group,comparison=left+'-'+right,metric=f,n=int(good.sum()),delta=float(dd[good,k].mean()),ci95=np.quantile(boot,[.025,.975]).tolist()))
    # Catastrophic means >=20pp rawCOCO decline, defined for description only;
    # every target remains in all primary means, no filtering or replacement.
    catastrophic=[]
    for lab in labels:
        t=int(lab['annotation_id']);before=float(data[t,'original']['coco_iou']);after=float(data[t,'native_overlap']['coco_iou'])
        if before-after>=.20:
            catastrophic.append(dict(annotation_id=t,image_id=int(lab['image_id']),
                iou={m:float(data[t,m]['coco_iou']) for m in modes},
                overlap_deletes_fraction=float(lab['overlap_deleted_own_pixels'])/max(1,int(lab['independent_pixels'])),
                lost_truepositive_rejected_fraction=float(lab['native_lost_tp_rejected_by_overlap'])/max(1,int(lab['native_lost_tp_pixels']))))
    label_summary=[]
    for group in ['all','high','other']:
        q=[r for r in labels if group=='all' or (targets[int(r['annotation_id'])]['high']=='True')==(group=='high')]
        label_summary.append(dict(group=group,n=len(q),changed=sum(int(r['overlap_deleted_own_pixels'])>0 for r in q),
            mean_deleted_fraction=float(np.mean([int(r['overlap_deleted_own_pixels'])/max(1,int(r['independent_pixels'])) for r in q])),
            mean_overlap_independent_iou=float(np.mean([float(r['overlap_vs_independent_iou']) for r in q]))))
    sol=read(run/'solver.csv')
    result=dict(means=means,contrasts=contrasts,catastrophic=catastrophic,label_summary=label_summary,
        prior_spatial_exact_replay=len(parity),cache_y_replay='all160targets2048positionsexact',
        solver_hits_cap={m:sum(r['mode']==m and r['hit_cap']=='True' for r in sol) for m in ['native_independent','raw_coco']},
        source_hashes={n:sha(run/n) for n in ['spatial.csv','labels.csv','pixels.csv','solver.csv']},
        scope='Posthoc160GT-assisted targets23high,2000imageclusterpointwiseCI,noAP/sharedtraining/noinferencegain. '
              'Primaryalltargetsnotcatastrophicselection. Nativeindependentisofficialexistingoption;rawrasterizationalsoexistingtechnique. '
              'Labelchangeiscontrolledforcurrentcoefficientfit,notproofhistoricalnetworkfailurecause.')
    write_json(run/'ANALYSIS.json',result)
    print(json.dumps(dict(main=[r for r in contrasts if r['group']=='all' and r['metric']=='coco_iou'],
        catastrophic=catastrophic,label_summary=label_summary),ensure_ascii=False))


if __name__=='__main__':main()
