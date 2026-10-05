"""S021 matched target comparisons and image-cluster bootstrap."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from analyze_native_labels import sha,read

ARMS=['original','positive_scale_BCE','positive_affine_BCE','native640_free32_BCE','native640_threshold_oracle','free32_norm_restored','free32_double_decode']
CONTRASTS={'affine_minus_original':('positive_affine_BCE','original'),'free_minus_affine':('native640_free32_BCE','positive_affine_BCE'),
    'free_minus_original':('native640_free32_BCE','original'),'threshold_minus_original':('native640_threshold_oracle','original'),
    'free_minus_threshold':('native640_free32_BCE','native640_threshold_oracle'),
    'double_minus_free':('free32_double_decode','native640_free32_BCE'),'norm_restored_minus_free':('free32_norm_restored','native640_free32_BCE')}
FIELDS=['coco_iou','coverage','same_neighbor','background','other_error','valid_iou']

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    receipt=json.loads((out/'COMPLETE.json').read_text());assert receipt['status']=='COMPLETE'
    for name,h in receipt['hashes'].items():assert sha(out/name)==h,name
    rows=read(out/'metrics.csv');ranks=read(out/'ranking.csv');cal=read(out/'calibration.csv');protocol=json.loads((out/'protocol.json').read_text())
    lookup={(int(r['annotation_id']),r['arm']):r for r in rows};ranklookup={int(r['annotation_id']):r for r in ranks}
    assert len(lookup)==len(rows)==len(protocol['target_ids'])*len(ARMS)
    paired=[]
    for aid in protocol['target_ids']:
        arm={k:lookup[(aid,k)] for k in ARMS};ref=arm['original'];q=ranklookup[aid]
        assert float(arm['positive_scale_BCE']['coco_iou'])==float(ref['coco_iou'])
        r={k:ref[k] for k in ['image_id','annotation_id','density','area','parts']}
        for name,(plus,minus) in CONTRASTS.items():
            for field in FIELDS:r[name+'_'+field]=float(arm[plus][field])-float(arm[minus][field]) if arm[plus][field]!='' and arm[minus][field]!='' else None
        for domain in ['neighbor','background','other']:
            x=q['original_auc_'+domain];y=q['free_auc_'+domain]
            r['auc_'+domain+'_delta']=float(y)-float(x) if x!='' and y!='' else None
            r['original_auc_'+domain]=float(x) if x!='' else None;r['free_auc_'+domain]=float(y) if y!='' else None
        r['response_affine_r2']=float(q['response_affine_r2']) if q['response_affine_r2']!='' else None
        paired.append(r)
    with open(out/'PAIRED_TARGETS.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
    ids=sorted({int(r['image_id']) for r in paired});index={v:i for i,v in enumerate(ids)}
    boot=np.random.default_rng(20260912).multinomial(len(ids),np.full(len(ids),1/len(ids)),size=2000)
    def estimate(rr,field):
        num=np.zeros(len(ids));den=np.zeros(len(ids))
        for r in rr:
            if r[field] is not None:num[index[int(r['image_id'])]]+=r[field];den[index[int(r['image_id'])]]+=1
        bd=boot@den;bn=boot@num;ok=bd>0
        return dict(n=int(den.sum()),mean_pp=100*float(num.sum()/den.sum()) if den.sum() else None,ci95_pp=(100*np.quantile(bn[ok]/bd[ok],[.025,.975])).tolist() if ok.any() else None)
    contrasts=[];ranking=[];objective=[];subgroups=[]
    for density in ['high','other']:
        rr=[r for r in paired if r['density']==density]
        for name in CONTRASTS:
            for field in FIELDS:contrasts.append(dict(density=density,contrast=name,metric=field,**estimate(rr,name+'_'+field)))
        for domain in ['neighbor','background','other']:
            ranking.append(dict(density=density,domain=domain,original=estimate(rr,'original_auc_'+domain),free=estimate(rr,'free_auc_'+domain),delta=estimate(rr,'auc_'+domain+'_delta')))
        for arm in ['positive_scale_BCE','positive_affine_BCE']:
            cc=[r for r in cal if r['density']==density and r['arm']==arm]
            objective.append(dict(density=density,arm=arm,n=len(cc),loss_before_mean=float(np.mean([float(r['loss_before']) for r in cc])),
                loss_after_mean=float(np.mean([float(r['loss_after']) for r in cc])),free_loss_mean=float(np.mean([float(r['free32_loss']) for r in cc])),
                free_loss_worse_than_calibration=sum(float(r['free32_loss'])>float(r['loss_after'])+1e-6 for r in cc),
                median_alpha=float(np.median([float(r['alpha']) for r in cc])),median_beta=float(np.median([float(r['beta']) for r in cc])),
                iteration_limit=sum(int(r['iterations'])>=120 for r in cc),alpha_near_zero=sum(r['alpha_near_zero']=='True' for r in cc)))
        for small in [True,False]:
            qq=[r for r in rr if (float(r['area'])<1024)==small]
            subgroups.append(dict(density=density,small=small,**estimate(qq,'free_minus_affine_coco_iou')))
    doc=dict(contrasts=contrasts,ranking=ranking,objective=objective,subgroups=subgroups,
        calibration_explanation='Positive-affine maps preserve input-logit ordering; free-vs-original AUC change tests spatial reordering without a threshold. AUC uses only same-domain eligible targets with both classes.',
        solver_limits='Calibration positive exp parameterization and float64 LBFGS differ from saved free32 float32 solve; achieved finite-budget comparison, not exact nested optimum. No global-optimum claim.',
        numerical={field:dict(median=float(np.median([float(r[field]) for r in ranks])),p90=float(np.quantile([float(r[field]) for r in ranks],.9)),max=float(max(float(r[field]) for r in ranks))) for field in ['coefficient_norm_ratio','free_coefficient_norm','scaled_loss_replay_error','unscaled_loss_replay_error','double_decode_xor','norm_restored_xor','logit_interpolation_max_error','free_interpolation_max_error']},
        bootstrap='2,000 shared image-cluster resamples, pointwise percentile intervals; target-weighted, fixed selected failures, no multiplicity correction.',
        scope=protocol['restrictions'],receipt_files_verified=len(receipt['hashes']),script_sha256=sha(__file__),csv_sha256=sha(out/'metrics.csv'))
    (out/'CALIBRATION_ANALYSIS.json').write_text(json.dumps(doc,indent=2),encoding='utf-8')
    print(json.dumps(dict(iou=[r for r in contrasts if r['metric']=='coco_iou'],ranking=ranking,objective=objective),indent=2))

if __name__=='__main__':main()
