"""Where do failing prediction pixels lie relative to official GT-box mask supervision?"""
import argparse,contextlib,io,json
from pathlib import Path
import numpy as np
import torch
from pycocotools.coco import COCO
from ultralytics.utils import ops
from candidate_lineage_probe import ROOT,read,need,save_csv,write_json,sha
from crossimage_response_experiment import gt_input_regions

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    with contextlib.redirect_stdout(io.StringIO()):gt=COCO(str(ROOT/'data/annotations/instances_val2017.json'))
    ids=json.loads((out/'protocol.json').read_text())['images'];meta={int(r['annotation_id']):r for r in read(out/'targets.csv')}
    rows=[];skipped=[]
    for iid in ids:
        with np.load(ROOT/'diagnostics/full_val_cache_20260911/val'/f'{iid}.npz') as z:item={k:z[k] for k in z.files}
        raster,union,valid=gt_input_regions(gt,iid,item);mapping=dict(zip(map(int,item['mapping_gt']),map(int,item['mapping_pred'])))
        c=torch.tensor(item['coeff'],device='cuda');proto=torch.tensor(item['proto'],device='cuda');boxes=torch.tensor(item['boxes'],device='cuda')
        prediction=ops.process_mask(proto,c,boxes,(640,640),upsample=True).bool()
        h,w=map(int,item['shape']);gain=min(640/h,640/w);rh,rw=round(h*gain),round(w*gain);top,left=round((640-rh)/2-.1),round((640-rw)/2-.1)
        same={cat:torch.stack([mask for aid,mask in raster.items() if gt.anns[aid]['category_id']==cat]).any(0) for cat in {gt.anns[aid]['category_id'] for aid in raster}}
        for aid,j in mapping.items():
            q=gt.anns[aid];own=raster[aid]&valid;area=int(own.sum())
            if not area:
                skipped.append(dict(image_id=iid,annotation_id=aid,reason='zero_valid_area_after_crowd_exclusion'));continue
            pred=prediction[j]&valid;sn=pred&same[q['category_id']]&~own;bg=pred&~union;other=pred&union&~same[q['category_id']]&~own
            x,y,bw,bh=q['bbox'];gtbox=torch.tensor([[x*gain+left,y*gain+top,(x+bw)*gain+left,(y+bh)*gain+top]],device='cuda')
            gt_support=ops.crop_mask(torch.ones((1,640,640),device='cuda'),gtbox)[0].bool()
            proto_support=ops.crop_mask(torch.ones((1,160,160),device='cuda'),gtbox/4)[0].bool()
            # Full-gradient support: any bilinear source prototype cell touched by
            # a prediction-grid pixel that is inside the official GT crop at160.
            influence=torch.nn.functional.interpolate(proto_support[None,None].float(),(640,640),mode='bilinear',align_corners=False)[0,0]>0
            r=meta[aid];row=dict(image_id=iid,annotation_id=aid,ici=float(r['ici']),area=q['area'],official_mask75=r['official_mask75'],
                no_good_postconf_mask75=r['no_good_postconf_mask75'],assignment=r['final_source_assignment'],own_pixels=area)
            for name,mask in [('same_neighbor',sn),('background',bg),('other',other)]:
                count=int(mask.sum());inside=int((mask&gt_support).sum());touch=int((mask&influence).sum())
                row.update({name+'_pixels':count,name+'_inside_gtbox':inside,name+'_outside_gtbox':count-inside,
                    name+'_touch_supervised_proto_cells':touch,name+'_outside_supervised_proto_cells':count-touch})
            rows.append(row)
    save_csv(out/'pixel_supervision_domain.csv',rows)
    summary=[]
    for density in ['high','other']:
        for cohort in ['all_matched','failed_no_good','failed_no_good_own_positive']:
            rr=[r for r in rows if (r['ici']>.5+1e-10)==(density=='high')]
            if cohort!='all_matched':rr=[r for r in rr if r['official_mask75']=='False' and r['no_good_postconf_mask75']=='True']
            if cohort=='failed_no_good_own_positive':rr=[r for r in rr if r['assignment']=='own_gt_positive']
            for region in ['same_neighbor','background']:
                nz=[r for r in rr if r[region+'_pixels']>0]
                summary.append(dict(density=density,cohort=cohort,region=region,n=len(rr),nonzero_errors=len(nz),
                    mean_error_over_gt=100*float(np.mean([r[region+'_pixels']/r['own_pixels'] for r in rr])) if rr else None,
                    outside_gtbox_fraction_instance_mean=100*float(np.mean([r[region+'_outside_gtbox']/r[region+'_pixels'] for r in nz])) if nz else None,
                    outside_supervised_proto_fraction_instance_mean=100*float(np.mean([r[region+'_outside_supervised_proto_cells']/r[region+'_pixels'] for r in nz])) if nz else None,
                    outside_gtbox_fraction_pooled=100*sum(r[region+'_outside_gtbox'] for r in rr)/max(sum(r[region+'_pixels'] for r in rr),1)))
    write_json(out/'SUPERVISION_DOMAIN_ANALYSIS.json',dict(groups=summary,script_sha256=sha(__file__),csv_sha256=sha(out/'pixel_supervision_domain.csv'),
        skipped_zero_valid_area=len(skipped),skipped_targets=skipped,
        scope='Input640grid, original predictions/GT nearest-exact letterbox, crowd excluded for spatial counting. Every cohort including all_matched is restricted to GT with positive valid area; zero-area exclusions individually listed. GT160crop is official coefficient BCE supervision support in this diagnostic. Spatial inclusion only, not proof of missing representation or training history. Coefficient updates affect all spatial pixels globally.'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
