"""S033 raw pixel accounting with image-cluster descriptive uncertainty."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from readout_input_probe import sha,write_json
from readout_composition_control import comparison


def read(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def csv_save(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args();run=a.run
    receipt=json.loads((run/'COMPLETE.json').read_text());cfg=json.loads((run/'protocol.json').read_text())
    for filename in ['labels.csv','flows.csv','replay.csv']:
        if sha(run/filename)!=receipt['hashes'][filename]:raise RuntimeError('Changed result '+filename)
    images=cfg['images'];image_index={x:k for k,x in enumerate(images)}
    draws=np.random.default_rng(20260912).multinomial(len(images),np.ones(len(images))/len(images),size=2000)
    label=read(run/'labels.csv');flow=read(run/'flows.csv');by=defaultdict(list)
    for row in flow:by[int(row['annotation_id']),row['comparison']].append(row)
    labels=[];flows=[];empty_label=[]
    common_meta=['image_id','annotation_id','category_id','size','density']
    for row in label:
        if not int(row['raw_area640']):empty_label.append(int(row['annotation_id']));continue
        r={k:int(row[k]) if k in ['image_id','annotation_id','category_id'] else row[k] for k in common_meta}
        r['high']=r['density']=='high';r['matched']=row['matched']=='True'
        area=int(row['raw_area640']);r['raw_area']=area
        for name in ['ind_extra','ind_missing','overlap_deleted']:
            for suffix in ['', '_near1','_near2','_near4','_far2_othergt','_far2_elsewhere','_sharedgt']:
                r[name+suffix]=int(row[name+suffix]);r[name+suffix+'_rate']=int(row[name+suffix])/area
        r['ind_net_rate']=(int(row['ind_area640'])-area)/area
        r['raw_shared_rate']=int(row['raw_shared_pixels'])/area
        r['ind_xor']=r['ind_extra']+r['ind_missing']
        r['ind_xor_near2']=r['ind_extra_near2']+r['ind_missing_near2']
        r['ind_xor_near4']=r['ind_extra_near4']+r['ind_missing_near4']
        r['ind_xor_rate']=r['ind_xor']/area
        labels.append(r)
    for (aid,comp),rr in by.items():
        if len(rr)!=3 or {r['seed'] for r in rr}!={'0','1','2'}:raise RuntimeError('Seed mismatch')
        r={k:int(rr[0][k]) if k in ['image_id','annotation_id','category_id'] else rr[0][k] for k in common_meta}
        r.update(high=r['density']=='high',comparison=comp)
        names=[k for k in rr[0] if k not in common_meta+['coco_area','ici','matched','comparison','seed']]
        for name in names:r[name]=float(np.mean([float(x[name]) for x in rr]))
        for name in names:r[name+'_rate']=r[name]/r['valid_area']
        r['net_area_rate']=(r['after_area']-r['before_area'])/r['valid_area']
        r['shrank']=float(r['after_area']<r['before_area'])
        for name in ['own','same','other','bg']:
            r['net_'+name+'_rate']=r['added_'+name+'_rate']-r['removed_'+name+'_rate']
        flows.append(r)
    def stat(rows,metric,denominator=None):
        q=[r for r in rows if r.get(metric) is not None and (denominator is None or r.get(denominator) is not None)]
        if not q:return dict(n=0,value=None,ci95=None)
        idx=np.array([image_index[r['image_id']] for r in q]);v=np.array([r[metric] for r in q],float)
        den=np.ones(len(q)) if denominator is None else np.array([r[denominator] for r in q],float)
        total=np.bincount(idx,weights=v,minlength=len(images));counts=np.bincount(idx,weights=den,minlength=len(images))
        ns=np.einsum('bi,i->b',draws,total,optimize=False);ds=np.einsum('bi,i->b',draws,counts,optimize=False)
        boot=ns[ds>0]/ds[ds>0]
        return dict(n=len(q),value=float(v.sum()/den.sum()) if den.sum() else None,
            ci95=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None,
            aggregation='pooled numerator/denominator' if denominator else 'target mean; prediction flows average three saved seeds first')
    groups=['all','low','middle','high','nonhigh'];summaries=[]
    for group in groups:
        def keep(r):return group=='all' or (group=='nonhigh' and r['density']!='high') or r['density']==group
        q=[r for r in labels if keep(r)]
        for metric in ['ind_extra_rate','ind_missing_rate','ind_net_rate','ind_xor_rate','overlap_deleted_rate','raw_shared_rate']:
            summaries.append(dict(domain='labels',group=group,metric=metric,**stat(q,metric)))
        for numerator,denom in [('ind_xor_near2','ind_xor'),('ind_xor_near4','ind_xor'),('ind_extra_near2','ind_extra'),('ind_missing_near2','ind_missing')]:
            summaries.append(dict(domain='labels',group=group,metric=numerator+'_fraction',**stat(q,numerator,denom)))
        for comp in cfg['comparisons']:
            q=[r for r in flows if keep(r) and r['comparison']==comp]
            metrics=['shrank','net_area_rate','removed_rate','added_rate']
            for n in ['own','same','other','bg']:
                metrics+=['net_'+n+'_rate','removed_'+n+'_rate','added_'+n+'_rate',
                          'removed_'+n+'_near2_rate','added_'+n+'_near2_rate','removed_'+n+'_far4_rate','added_'+n+'_far4_rate']
            for metric in metrics:summaries.append(dict(domain='flows',comparison=comp,group=group,metric=metric,**stat(q,metric)))
            for prefix in ['removed_own','removed_bg','added_own','added_bg']:
                for tail in ['near2','near4','near_normalized']:
                    summaries.append(dict(domain='flows',comparison=comp,group=group,metric=prefix+'_'+tail+'_fraction',**stat(q,prefix+'_'+tail,prefix)))
            for num,den in [('removed640_ind_extra','removed640'),('added640_ind_missing','added640'),('removed_own_shared','removed_own')]:
                summaries.append(dict(domain='flows',comparison=comp,group=group,metric=num+'_fraction',**stat(q,num,den)))
    composition=[]
    for control in ['nonhigh','low','middle']:
        eligible=lambda r: r['density']=='high' or (r['density']!='high' if control=='nonhigh' else r['density']==control)
        for domain,rr,metrics in [('labels',labels,['ind_extra_rate','ind_missing_rate','ind_xor_rate','raw_shared_rate']),
                                  ('raw_coco-original',[r for r in flows if r['comparison']=='raw_coco-original'],['net_own_rate','net_bg_rate','net_same_rate','net_area_rate'])]:
            q=[r for r in rr if eligible(r)]
            for metric in metrics:
                result=comparison(q,('category_id','size'),metric,images,draws)
                composition.append(dict(domain=domain,contrast='high-'+control,metric=metric,**result))
    size_rows=[]
    for size in ['small','medium','large']:
        for group in ['low','middle','high']:
            q=[r for r in labels if r['size']==size and r['density']==group]
            size_rows.append(dict(domain='labels',size=size,group=group,metric='ind_extra_rate',**stat(q,'ind_extra_rate')))
    result=dict(summaries=summaries,composition=composition,size_groups=size_rows,counts=dict(label_targets=len(labels),
        flow_targets=len(flows)//2,empty_label_targets=empty_label,images=len(images)),
        source_hashes={name:sha(run/name) for name in ['labels.csv','flows.csv','replay.csv']},
        scope='All300exploredtrain2017images. Flow onlyfixedbbox50 validtargets, labels allnative-retainedGT. '
        '2000pointwiseimage-clusterCI after seedmean, noAP/causalshare/densitycausality. '
        'Composition commoncategory-size strata require>=3targetspergroup;minimum group counts as weights, recomputed perbootstrap. '
        'Pooledboundarypixel fractions distinctfrommeanperinstance fractions. Distance thresholds predeclared, not AP optimized.')
    write_json(run/'ANALYSIS.json',result)
    csv_save(run/'label_target_metrics.csv',labels);csv_save(run/'flow_target_metrics.csv',flows)
    print(json.dumps(dict(counts=result['counts'],main=[r for r in summaries if r['group'] in ['all','high'] and (
        r['domain']=='labels' or r.get('comparison')=='raw_coco-original' and r['metric'] in ['net_area_rate','removed_rate','added_rate','net_own_rate','net_bg_rate','net_same_rate','removed_own_near2_fraction','removed_bg_near2_fraction','removed640_ind_extra_fraction','shrank'])]),ensure_ascii=False))


if __name__=='__main__':main()
