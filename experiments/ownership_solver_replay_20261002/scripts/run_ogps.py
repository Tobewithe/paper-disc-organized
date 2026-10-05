"""Zero-training replay of existing 7O probabilities through fixed ridge."""
import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, r'D:\coco_wire\py')
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from pycocotools.coco import COCO
from ultralytics.utils import ops
import frozen_evaluation as ev
from ogps_solver import (pooled_design_matrix, crop_pool_7o, solve_ogps,
                         solve_ogps_logits, mathematical_self_checks)


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def key(row):
    return tuple(int(row[x]) for x in ('image_id','annotation_id','raw_id'))


@torch.no_grad()
def direct_metrics(x, maps, rows, coco):
    """Declared direct renderer: resize logits into 640-grid box, crop, threshold."""
    device = x['proto'].device
    for i, row in enumerate(rows):
        box = x['boxes'][i].to(device)
        x1 = max(0, min(639, math.floor(float(box[0]))))
        y1 = max(0, min(639, math.floor(float(box[1]))))
        x2 = max(x1+1, min(640, math.ceil(float(box[2]))))
        y2 = max(y1+1, min(640, math.ceil(float(box[3]))))
        gt = torch.as_tensor(coco.annToMask(coco.anns[row['annotation_id']]).astype(bool), device=device)
        padded_gt = ev._padded_gt(gt, x['ratio_pad'], (640,640))
        support = ops.crop_mask(torch.ones((1,640,640),device=device), box[None])[0].bool()
        for arm, logits in maps.items():
            patch = F.interpolate(logits[i:i+1].float(), (y2-y1,x2-x1), mode='bilinear', align_corners=False)[0,0]
            canvas = torch.full((1,640,640), -30., device=device)
            canvas[0,y1:y2,x1:x2] = patch
            binary = ((canvas > 0) & support[None]).byte()
            mask = ev._scale_binary(binary, tuple(x['original_shape']), x['ratio_pad'])[0]
            inter = int((mask & gt).sum())
            row['iou_'+arm] = inter / max(int((mask | gt).sum()),1)
            row['coverage_'+arm] = inter / max(int(gt.sum()),1)
            row['mask75_'+arm] = int(row['iou_'+arm] >= .75)
            row['auc_'+arm],row['fpr_'+arm] = ev.pixel_auc_fpr(canvas[0],padded_gt,support)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0')
    started=time.time()
    save(out/'ENVIRONMENT.json',dict(host=platform.node(),python=sys.version,torch=torch.__version__,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),device=str(device),training=False))
    checks=mathematical_self_checks(device='cpu')
    save(out/'MATH_CHECKS.json',checks)
    paths={k:Path(v) for k,v in cfg['inputs'].items()}
    manifest={k:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for k,p in paths.items()}
    save(out/'INPUT_MANIFEST.json',manifest)
    val=torch.load(paths['ownership'],map_location='cpu',weights_only=False)
    pred=torch.load(paths['predictions'],map_location='cpu',weights_only=False)
    oracle=torch.load(paths['oracle'],map_location='cpu',weights_only=False)
    keys=[tuple(map(int,k)) for k in val['keys']]
    pkeys=[tuple(map(int,k)) for k in pred['keys']]
    omap={key(r):r for r in oracle}
    if len(keys)!=len(set(keys)) or len(omap)!=len(oracle) or keys!=pkeys or set(keys)!=set(omap):
        raise AssertionError('Candidate identity mismatch: no silent intersection')
    if len(keys)!=1346 or len({k[0] for k in keys})!=196:
        raise AssertionError('Unexpected valid historical cohort')
    by_image=defaultdict(list)
    for j,k in enumerate(keys): by_image[k[0]].append(j)
    coco=COCO(cfg['annotations'])
    audit=dict(n_candidates=len(keys),n_images=len(by_image),baseline_iou_max_error=0.,h_max_error=0.,c0_max_error=0.,pooled_base_max_error=0.,label_max_error=0.,identity_delta_max=0.,solver_stationarity_max=0.,max_condition=0.,prepared_sha256={})
    all_rows=[]; saved_coeff=[]; solver_rows=[]
    arm_map={'true_roi':'T_TRUE','base_h':'N_BASEH','wrong_instance':'W_INST','wrong_image':'W_IMAGE'}
    direct_map={'true_roi':'D_TRUE','base_h':'D_BASEH','wrong_instance':'D_INST','wrong_image':'D_IMAGE'}
    with (out/'PER_CANDIDATE.jsonl').open('w',encoding='utf-8') as fh:
        for image_number,(iid,indices) in enumerate(sorted(by_image.items()),1):
            path=Path(cfg['prepared'])/f'{iid:012d}.pt'
            audit['prepared_sha256'][str(iid)]=sha(path)
            x=torch.load(path,map_location='cpu',weights_only=False)
            actual={key(r):i for i,r in enumerate(x['rows'])}
            if set(actual)!={keys[j] for j in indices} or len(actual)!=len(indices):
                raise AssertionError(f'Prepared identity mismatch {iid}')
            # Use prepared row order throughout decoding; assets joined only by permanent key.
            lookup={keys[j]:j for j in indices}
            js=[lookup[key(r)] for r in x['rows']]
            oi=[omap[key(r)] for r in x['rows']]
            c0=x['c0'].float().to(device)
            h=x['h'].float()
            hc=(h-val['h'][js].float()).abs().max().item()
            oc=torch.stack([r['c0'].float() for r in oi])
            cc=(x['c0'].float()-oc).abs().max().item()
            audit['h_max_error']=max(audit['h_max_error'],hc)
            audit['c0_max_error']=max(audit['c0_max_error'],cc)
            if hc>1e-5 or cc>1e-5: raise AssertionError('Feature/coefficient asset mismatch')
            x['proto']=x['proto'].float().to(device)
            x['boxes']=x['boxes'].float().to(device)
            A=pooled_design_matrix(x['proto'],x['boxes'])
            z0=(A.double() @ c0.double().unsqueeze(-1)).squeeze(-1)
            oldbase=val['base'][js].to(device)
            base_error=(z0.reshape(-1,1,8,8).clamp(-30,30)-oldbase.double()).abs().max().item()
            audit['pooled_base_max_error']=max(audit['pooled_base_max_error'],base_error)
            if base_error>2e-5: raise AssertionError(f'Pooled baseline mismatch {base_error}')
            labels=[]
            for i in range(len(js)):
                gt=(x['masks']==int(x['owners'][i])+1).float().to(device)
                labels.append(crop_pool_7o(gt.reshape(1,640,640),x['boxes'][i]))
            gtpool=torch.stack(labels)
            label_error=(gtpool-val['label'][js].to(device)).abs().max().item()
            audit['label_max_error']=max(audit['label_max_error'],label_error)
            if label_error>1e-6: raise AssertionError('GT coarse ownership mismatch')
            identity=solve_ogps_logits(A,c0,z0)['delta'].abs().max().item()
            audit['identity_delta_max']=max(audit['identity_delta_max'],identity)
            if identity>1e-10: raise AssertionError('Unclipped logit identity failed')
            probs={'G_GTsolver':gtpool,'S_BASE':z0.sigmoid()}
            for source,arm in arm_map.items():
                probs[arm]=pred['logits'][source][js].to(device).double().sigmoid()
            coeff={'A':c0,'O':c0+torch.stack([r['delta'].float() for r in oi]).to(device)}
            for arm,q in probs.items():
                sol=solve_ogps(A,c0,q)
                coeff[arm]=sol['coeff'].float()
                diag=sol['diagnostics']
                audit['solver_stationarity_max']=max(audit['solver_stationarity_max'],diag['stationarity_l2'].max().item())
                audit['max_condition']=max(audit['max_condition'],diag['system_condition_2'].max().item())
                for i,j in enumerate(js):
                    solver_rows.append(dict(key=keys[j],arm=arm,**{k:v.reshape(-1)[i].item() for k,v in diag.items()}))
            rr=ev.evaluate_image(x,coeff,coco,chunk_size=4)
            maps={direct_map[k]:v[js].to(device) for k,v in pred['logits'].items()}
            maps['D_GT']=torch.logit(gtpool.clamp(.01,.99))
            direct_metrics(x,maps,rr,coco)
            for i,(row,j) in enumerate(zip(rr,js)):
                row['oracle_matched']=True
                row['box_iou']=float(val['box_iou'][j])
                row['baseline_iou_cached']=float(val['full_iou'][j])
                err=abs(row['iou_A']-row['baseline_iou_cached'])
                audit['baseline_iou_max_error']=max(audit['baseline_iou_max_error'],err)
                if err>1e-6: raise AssertionError(f'Normal baseline mismatch {keys[j]} {err}')
                row['box_good_original_failure']=bool(row['box_iou']>=.75 and row['iou_A']<.75)
                fh.write(json.dumps(row,ensure_ascii=False,allow_nan=True)+'\n')
                all_rows.append(row)
            fh.flush()
            saved_coeff.append(dict(image_id=iid,keys=[keys[j] for j in js],coeff={a:t.cpu() for a,t in coeff.items()}))
            if image_number%10==0 or image_number==1:
                print(json.dumps(dict(images=image_number,total_images=len(by_image),candidates=len(all_rows),elapsed_seconds=round(time.time()-started,1))),flush=True)
    audit['passed']=True
    save(out/'IDENTITY_DECODE_SOLVER_AUDIT.json',audit)
    torch.save(saved_coeff,out/'FROZEN_SOLVED_COEFFICIENTS.pt')
    with (out/'SOLVER_DIAGNOSTICS.jsonl').open('w',encoding='utf-8') as f:
        for r in solver_rows: f.write(json.dumps(r)+'\n')
    from summarize_ogps import summarize
    result=summarize(all_rows,out,seed=cfg['bootstrap_seed'],bootstrap=5000)
    save(out/'COMPLETE.json',dict(completed=True,images=len(by_image),candidates=len(all_rows),elapsed_seconds=time.time()-started,training_updates=0,automatic_next_stage=False))
    print('COMPLETE',flush=True)


if __name__=='__main__': main()
