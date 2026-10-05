"""Frozen A/N/P/PC replay: label domains, actual pixel edits, no training."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path

if '--config' in sys.argv:
    _cfg=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text(encoding='utf-8-sig'))
    sys.path.insert(0,_cfg['source_python'])

import numpy as np
import torch
from torch.nn import functional as F
from pycocotools.coco import COCO
from ultralytics.utils import ops
from ultralytics.utils.loss import v8SegmentationLoss
from online_runtime import FrozenReplay,load_asset,load_index,load_json,dump,sha256,tensor_sha
from native_proto_model import NativeProtoModel
from joint_head import JointCoefficientReadout
from label_replay import LabelReplay
import evaluation_metrics as em

ARMS=('A','N','P','PC'); METHODS=ARMS[1:]
METRICS=('iou','coverage','mask75','auc','fpr')
em.ARMS=ARMS
REGIONS=('agree_fg','missing_overlap','missing_polygon','untrained_fg','extra_fg','agree_bg','untrained_bg')

def identity(row):
    return tuple(row[k] for k in ('split','image_id','annotation_id','branch','raw_id','pyramid_level','target_gt_idx'))

def old_rows(cfg):
    rows={}
    for line in Path(cfg['prior_results']).read_text().splitlines():
        r=json.loads(line);key=identity(r)
        if key in rows:raise AssertionError('Historical duplicate identity')
        rows[key]=r
    return rows

def load_models(replay,cfg):
    models={};hashes={}
    for name in METHODS:
        path=Path(cfg['checkpoints'][name]);ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['mode']==name and ck['epoch']==3
        if not load_json(path.parent/'COMPLETE.json').get('completed'):raise AssertionError('Unfinished checkpoint')
        model=(JointCoefficientReadout(replay.native_cv4,replay.feature_channels,'N',cfg)
               if name=='N' else NativeProtoModel(replay,name))
        before={k:tensor_sha(v) for k,v in model.named_buffers()}
        model.load_state_dict(ck['state_dict'],strict=True)
        model=model.to(replay.device).float().eval().requires_grad_(False)
        if before!={k:tensor_sha(v) for k,v in model.named_buffers()}:raise AssertionError('Frozen BN differs')
        models[name]=model;hashes[name]=sha256(path)
    return models,hashes

def count(v):return int(v.sum())
def div(a,b):return a/b if b else None

def pixel_record(x,k,zs,official,polygon,coco):
    """All pixel decompositions are on the explicitly fixed 640 canvas."""
    device=zs['A'].device;source=x['rows'][k];annotation=coco.anns[int(source['annotation_id'])]
    original=torch.as_tensor(coco.annToMask(annotation).astype(bool),device=device)
    g=em._padded_gt(original,x['ratio_pad'],(640,640))
    t=official.to(device);d=polygon.to(device)
    if (t & ~d).any():raise AssertionError('Official owner mask not subset of independent polygon')
    box=x['target_boxes'][k:k+1].to(device);pred=x['boxes'][k:k+1].to(device)
    u=ops.crop_mask(torch.ones((1,640,640),device=device),box)[0].bool()
    v=ops.crop_mask(torch.ones((1,640,640),device=device),pred)[0].bool()
    area=((box[:,2:]-box[:,:2])/640).prod(1)[0]
    if area<=0:raise AssertionError('Invalid fixed area normalization')
    weight=float(x['segmentation_gain'])/(640*640*float(area))
    # Partition of the predicted-box support, with unsupervised pixels separate.
    regions=dict(agree_fg=g&t&u&v,missing_overlap=g&d&~t&u&v,
        missing_polygon=g&~d&u&v,untrained_fg=g&~u&v,
        extra_fg=~g&t&u&v,agree_bg=~g&~t&u&v,untrained_bg=~g&~u&v)
    cover=sum(r.to(torch.int8) for r in regions.values())
    if not torch.equal(cover,v.to(torch.int8)):raise AssertionError('Output-support partition not exhaustive/exclusive')
    train_regions={'11':g&t&u,'10':g&~t&u,'01':~g&t&u,'00':~g&~t&u}
    a=(zs['A']>0)&v
    # Descriptive two-pixel boundary and initial confidence strata, not methods.
    gf=g.float()[None,None]
    boundary=(F.max_pool2d(gf,5,1,2)[0,0]>.5)&~(-F.max_pool2d(-gf,5,1,2)[0,0]>.5)
    uncertain=zs['A'].abs()<=1
    out=dict(gt_pixels_640=count(g),train_support_pixels=count(u),pred_support_pixels=count(v),
        coco_fg_outside_train=count(g&~u),coco_fg_outside_pred=count(g&~v),
        train_missing_overlap=count(g&d&~t&u),train_missing_polygon=count(g&~d&u),
        train_extra_positive=count(~g&t&u),train_coco_foreground=count(g&u),
        label_disagreement_full=count(g^t),label_disagreement_train=count((g^t)&u),
        original_TP_640=count(a&g),original_FP_640=count(a&~g))
    for name,r in regions.items():
        out['pixels_'+name]=count(r)
        out['base_positive_'+name]=count(a&r)
        out['base_negative_'+name]=count(~a&r)
    original_masks={}
    identity_error=0.
    losses={}
    for arm,z in zs.items():
        z64=z.double();t64=t.double();g64=g.double();sp=F.softplus(z64)
        bce_off=sp-t64*z64;bce_coco=sp-g64*z64
        lo=float(bce_off[u].sum())*weight;lc=float(bce_coco[u].sum())*weight
        lp=float((sp-d.double()*z64)[u].sum())*weight
        label_term=float(((g64-t64)*z64)[u].sum())*weight
        identity_error=max(identity_error,abs((lo-lc)-label_term))
        losses[arm]=(lo,lc)
        out['bce_off_'+arm]=lo;out['bce_coco_'+arm]=lc;out['bce_polygon_'+arm]=lp
        for name,r in train_regions.items():
            out[f'bce_off_{name}_{arm}']=float(bce_off[r].sum())*weight
            out[f'bce_coco_{name}_{arm}']=float(bce_coco[r].sum())*weight
        m=(z>0)&v
        out['input_iou_'+arm]=div(count(m&g),count(m|g))
        out['input_coverage_'+arm]=div(count(m&g),count(g))
        out['input_fpr_'+arm]=div(count(m&~g&v),count(~g&v))
        original_masks[arm]=em._scale_binary(m.float()[None],tuple(x['original_shape']),x['ratio_pad'])[0]
        om=original_masks[arm]
        out['reconstructed_original_iou_'+arm]=div(count(om&original),count(om|original))
        if arm=='A':continue
        deleted=a&~m;added=~a&m;tp_deleted=deleted&g;tp_added=added&g
        out[f'deleted_TP_{arm}']=count(tp_deleted);out[f'added_TP_{arm}']=count(tp_added)
        out[f'deleted_FP_{arm}']=count(deleted&~g);out[f'added_FP_{arm}']=count(added&~g)
        out[f'deleted_TP_boundary_{arm}']=count(tp_deleted&boundary)
        out[f'deleted_TP_uncertain_{arm}']=count(tp_deleted&uncertain)
        out[f'deleted_TP_conflict_{arm}']=count(tp_deleted&~t&u)
        out[f'added_TP_conflict_{arm}']=count(tp_added&~t&u)
        dz=z64-zs['A'].double()
        for name,r in regions.items():
            out[f'deleted_{name}_{arm}']=count(deleted&r)
            out[f'added_{name}_{arm}']=count(added&r)
            out[f'delta_logit_sum_{name}_{arm}']=float(dz[r].sum())
        out[f'deleted_TP_conflict_share_{arm}']=div(out[f'deleted_TP_conflict_{arm}'],count(tp_deleted))
        conflict_den=out['base_positive_missing_overlap']+out['base_positive_missing_polygon']
        out[f'conflict_delete_rate_{arm}']=div(out[f'deleted_TP_conflict_{arm}'],conflict_den)
        out[f'agree_fg_delete_rate_{arm}']=div(out[f'deleted_agree_fg_{arm}'],out['base_positive_agree_fg'])
        both=[out[f'conflict_delete_rate_{arm}'],out[f'agree_fg_delete_rate_{arm}']]
        out[f'deletion_rate_difference_{arm}']=both[0]-both[1] if None not in both else None
        label_delta=float(((g64-t64)*dz)[u].sum())*weight
        actual=(lo-losses['A'][0])-(lc-losses['A'][1])
        identity_error=max(identity_error,abs(actual-label_delta))
        out[f'delta_label_gap_{arm}']=label_delta
        out[f'official_down_coco_up_{arm}']=int(lo<losses['A'][0] and lc>losses['A'][1])
        am=original_masks['A'];mm=original_masks[arm]
        out[f'original_deleted_TP_{arm}']=count(am&~mm&original)
        out[f'original_added_TP_{arm}']=count(~am&mm&original)
    out['loss_identity_absolute_error']=identity_error
    if identity_error>1e-9:raise AssertionError('BCE label-change identity failed')
    return out

def ratio_summary(rows,num_key,den_key,draws=1000):
    groups=em.image_groups(rows)
    if not groups:return dict(pooled=None,image_macro=None,images_defined=0,numerator=0,denominator=0,pooled_image_bootstrap_ci95=[None,None])
    nums=np.array([sum(r[num_key] for r in g) for g in groups],dtype=float)
    dens=np.array([sum(r[den_key] for r in g) for g in groups],dtype=float)
    valid=dens>0;macro=float((nums[valid]/dens[valid]).mean()) if valid.any() else None
    rng=np.random.default_rng(20261003);dist=[]
    for _ in range(draws):
        ix=rng.integers(0,len(groups),len(groups));de=dens[ix].sum()
        if de:dist.append(nums[ix].sum()/de)
    return dict(pooled=div(float(nums.sum()),float(dens.sum())),
        image_macro=macro,images_defined=int(valid.sum()),numerator=float(nums.sum()),denominator=float(dens.sum()),
        pooled_image_bootstrap_ci95=np.quantile(dist,[.025,.975]).tolist() if dist else [None,None])

def summarize(rows,out,population,audit):
    tables={};per_image=[]
    preds={'all':lambda r:True,'original_failure':lambda r:r['mask75_A']==0,
        'original_success':lambda r:r['mask75_A']==1,'box_good_mask_bad':lambda r:r['box_good_mask_bad']}
    for group,pred in preds.items():
        rs=[r for r in rows if pred(r)];groups=em.image_groups(rs)
        if not rs:continue
        table=dict(images=len(groups),candidates=len(rs),arms={},label_counts={})
        for k in ('train_missing_overlap','train_missing_polygon','train_extra_positive','train_coco_foreground',
                  'coco_fg_outside_train','coco_fg_outside_pred','label_disagreement_train','gt_pixels_640'):
            table['label_counts'][k]=sum(r[k] for r in rs)
        for arm in METHODS:
            comparisons={m:em.paired(groups,arm,'A',m,20261003,1000) for m in
                (*METRICS,'input_iou','input_coverage','input_fpr','bce_off','bce_coco','bce_polygon')}
            changes={k:sum(r[k+'_'+arm] for r in rs) for k in
                ('deleted_TP','added_TP','deleted_FP','added_FP','original_deleted_TP','original_added_TP',
                 'deleted_TP_boundary','deleted_TP_uncertain','deleted_TP_conflict','added_TP_conflict')}
            reg={name:dict(pixels=sum(r['pixels_'+name] for r in rs),
                base_positive=sum(r['base_positive_'+name] for r in rs),
                deleted=sum(r[f'deleted_{name}_{arm}'] for r in rs),
                added=sum(r[f'added_{name}_{arm}'] for r in rs),
                delta_logit_sum=sum(r[f'delta_logit_sum_{name}_{arm}'] for r in rs)) for name in REGIONS}
            # A weighted pixel ratio is not a candidate mean; resample whole images.
            fraction=ratio_summary(rs,f'deleted_TP_conflict_{arm}',f'deleted_TP_{arm}')
            conflict_base=sum(r['base_positive_missing_overlap']+r['base_positive_missing_polygon'] for r in rs)
            agree_base=sum(r['base_positive_agree_fg'] for r in rs)
            rates=dict(conflict=div(changes['deleted_TP_conflict'],conflict_base),
                agreement=div(reg['agree_fg']['deleted'],agree_base),
                conflict_base_TP=conflict_base,agreement_base_TP=agree_base)
            off=comparisons['bce_off']['image_macro']['delta'];co=comparisons['bce_coco']['image_macro']['delta']
            decision=('label_conflict_is_an_important_clue_not_causal_proof' if off<0<co and (fraction['pooled'] or 0)>=.5
                else 'label_conflict_not_established_as_primary_direct_damage_explanation')
            table['arms'][arm]=dict(comparisons=comparisons,pixel_changes=changes,regions=reg,
                conflict_deleted_TP_share=fraction,TP_deletion_rates=rates,decision=decision,
                official_down_coco_up_candidates=sum(r[f'official_down_coco_up_{arm}'] for r in rs),
                repair=sum(r['mask75_A']==0 and r[f'mask75_{arm}']==1 for r in rs),
                damage=sum(r['mask75_A']==1 and r[f'mask75_{arm}']==0 for r in rs))
        tables[group]=table
        for g in groups:
            item=dict(image_id=g[0]['image_id'],split='dev',group=group,candidates=len(g))
            for arm in ARMS:
                for m in (*METRICS,'input_iou','input_coverage','bce_off','bce_coco','bce_polygon'):
                    item[m+'_'+arm]=em.avg([r[m+'_'+arm] for r in g])
            for arm in METHODS:
                for name in ('deleted_TP','added_TP','deleted_TP_conflict','added_TP_conflict'):
                    item[name+'_'+arm]=sum(r[name+'_'+arm] for r in g)
            per_image.append(item)
    em.append_rows(out/'PER_IMAGE.jsonl',per_image)
    summary=dict(population=population,audit=audit,tables=tables,bootstrap_draws=1000,
        study_scope='frozen-output explanatory diagnostic; no training or causal intervention')
    dump(out/'SUMMARY.json',em.clean(summary))
    lines=['# Frozen supervision/output domain audit','',
        'Same fixed A/N/P/PC epoch-3 outputs; no training, assignment changes or calibration. Current 640 label projection uses nearest, exactly as the prior metric implementation.','',
        '| Arm − A | Original macro IoU (pp) | Official BCE Δ | COCO-label BCE Δ | Polygon BCE Δ | Removed TP on conflict (%) [image-bootstrap CI] | Repair/damage |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,value in tables['all']['arms'].items():
        c=value['comparisons'];f=value['conflict_deleted_TP_share'];ci=f['pooled_image_bootstrap_ci95']
        fraction_text='undefined' if f['pooled'] is None else f"{100*f['pooled']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}]"
        lines.append(f"| {arm} | {100*c['iou']['image_macro']['delta']:+.4f} | {c['bce_off']['image_macro']['delta']:+.6f} | {c['bce_coco']['image_macro']['delta']:+.6f} | {c['bce_polygon']['image_macro']['delta']:+.6f} | {fraction_text} | {value['repair']}/{value['damage']} |")
    lines+=['','Scope and interpretation:','',
        '- T is official overlap ownership; D is the SAME converted polygons rasterized independently before overlap sorting; G is original COCO projected to the fixed 640 grid; U is the GT training box, V the predicted output box.',
        '- Pixel edit counts partition V into shared foreground, missing-overlap, missing-polygon, untrained foreground, extra foreground, shared background and untrained background. GT-box-exterior pixels are unsupervised, not negative training labels.',
        '- BCE comparisons use the identical logits and U/area/gain. Original-image IoU uses the unchanged official decoder and original annToMask, and is checked against prior per-candidate rows.',
        '- Exact identity checked: BCE_off − BCE_COCO = sum(weight * (G−T) * logit); it separates label effects on the scalar objective, not the causal effect of training with different labels.',
        '- The >=50% removed-TP conflict share plus official macro BCE down / COCO macro BCE up is a predeclared triage rule, not a universal scientific threshold. Failing it does not prove labels irrelevant. Always inspect region rates, area and boundary/confidence strata.',
        '- These previously viewed development images provide exploratory explanation only. No new blind test, AP, model, optimizer, gate, or automatic label retraining. Local-view outputs are not included because that intervention did not train on these labels.',
        '- S031–S033 already studied label semantics. This audit only connects the CURRENT official candidate identities and frozen updates to those hypotheses; no old-version numerical results are transferred.','']
    for arm in METHODS:lines.append(f"{arm}: {tables['all']['arms'][arm]['decision']}")
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')

@torch.no_grad()
def run(args):
    started=time.monotonic();cfg=load_json(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if os.name=='nt' or not torch.cuda.is_available():raise RuntimeError('Authorized server GPU only')
    if (out/'PER_CANDIDATE.jsonl').exists():raise RuntimeError('Run already contains outputs')
    deadline=started+(180 if args.smoke else cfg['max_evaluation_seconds'])
    items=load_index(cfg)['dev'];assert len(items)==256
    if [int(x['image_id']) for x in items]!=load_json(cfg['split'])['dev']:raise AssertionError('Image list differs')
    expected=old_rows(cfg)
    if args.smoke:items=[e for e in items if e['n']][:2]
    population=dict(planned_images=len(items),effective_images=sum(bool(e['n']) for e in items),
        candidates=sum(e['n'] for e in items),no_positive_images=[e['image_id'] for e in items if not e['n']])
    replay=FrozenReplay(cfg);models,hashes=load_models(replay,cfg);labels=LabelReplay(cfg)
    buffers={a:{k:tensor_sha(v) for k,v in m.named_buffers()} for a,m in models.items()}
    coco=COCO(cfg['annotations_train']);rows=[];seen=set();max_metric_error=0.;max_loss_error=0.;label_audits=[]
    dump(out/'ENVIRONMENT.json',replay.import_info)
    inputs=dict(weights_sha256=replay.weights_sha256,checkpoints_sha256=hashes,prior_results_sha256=sha256(cfg['prior_results']),
        config_sha256=sha256(args.config),index_sha256=sha256(Path(cfg['cache'])/'INDEX.json'))
    dump(out/'INPUT_MANIFEST.json',inputs)
    for position,entry in enumerate(items):
        if time.monotonic()>deadline:raise TimeoutError('Fixed audit budget reached')
        if not entry['n']:continue
        x=load_asset(cfg,entry['image_id'],True)
        identity_list=[identity(r) for r in x['rows']]
        if len(set(identity_list))!=len(identity_list) or seen.intersection(identity_list):raise AssertionError('Duplicate identity')
        lab=labels.replay(x);label_audits.append(lab['audit'])
        features=replay.replay([x]);device=replay.device
        variants={'A':(x['c0'].to(device),x['proto'].to(device))}
        inp={k:x[k] for k in ('image_id','raw_ids','c0')}
        variants['N']=(models['N'](features,[inp])[0],variants['A'][1])
        for arm in ('P','PC'):
            q=models[arm](features,[inp]);variants[arm]=(q['coefficients'][0],q['prototypes'][0])
        torch.testing.assert_close(variants['P'][0],variants['A'][0],atol=0,rtol=0)
        result=None
        for arm,(c,p) in variants.items():
            gx=dict(x,proto=p,c0=x['c0'].to(device),boxes=x['boxes'].to(device))
            metrics=em.evaluate_image(gx,{'A':gx['c0'],arm:c},coco,chunk_size=4)
            if arm=='A':result=metrics
            else:
                for r,m in zip(result,metrics):
                    for metric in METRICS:r[metric+'_'+arm]=m[metric+'_'+arm]
            for r,m in zip(result,metrics):
                old=expected[identity(r)]
                for metric in METRICS:
                    a,b=m[metric+'_'+arm],old[metric+'_'+arm]
                    if em.finite(a)!=em.finite(b):raise AssertionError('Metric definedness mismatch')
                    if em.finite(a):
                        err=abs(a-b);max_metric_error=max(max_metric_error,err)
                        if err>1e-12:raise AssertionError(f'{arm} {metric} frozen result differs: {err}')
        for k,row in enumerate(result):
            lo=k//4*4;hi=min(lo+4,len(result))
            zs={a:em._logits(p,c[lo:hi],(640,640))[k-lo] for a,(c,p) in variants.items()}
            detail=pixel_record(x,k,zs,lab['official'][k],lab['polygon'][k],coco)
            row.update(detail)
            for arm in ARMS:
                if abs(row['reconstructed_original_iou_'+arm]-row['iou_'+arm])>1e-12:
                    raise AssertionError('Pixel-edit masks do not reproduce normal original-image IoU')
            if args.smoke:
                # Independent official implementation confirms normalization.
                b=x['target_boxes'][k:k+1].to(device);area=((b[:,2:]-b[:,:2])/640).prod(1)
                for a,(c,p) in variants.items():
                    p640=F.interpolate(p[None],(640,640),mode='bilinear',align_corners=False)[0]
                    official=v8SegmentationLoss.single_mask_loss(lab['official'][k:k+1].to(device).float(),c[k:k+1],p640,b,area)*x['segmentation_gain']
                    err=abs(float(official)-row['bce_off_'+a]);max_loss_error=max(max_loss_error,err)
                    if err>3e-5+3e-5*abs(float(official)):raise AssertionError('Reference loss normalization mismatch')
            row['box_good_mask_bad']=expected[identity(row)]['box_good_mask_bad']
        em.append_rows(out/'PER_CANDIDATE.jsonl',result);rows.extend(result);seen.update(identity_list)
        if position%20==0 or position+1==len(items):
            progress=dict(images=position+1,candidates=len(rows),elapsed_s=time.monotonic()-started)
            dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
    if len(rows)!=population['candidates']:raise AssertionError('Population incomplete')
    if not args.smoke and seen!=set(expected):raise AssertionError('Expected full population not evaluated')
    if time.monotonic()>deadline:raise TimeoutError('Fixed audit budget reached before summarization')
    replay.assert_unchanged()
    for a,m in models.items():
        if buffers[a]!={k:tensor_sha(v) for k,v in m.named_buffers()}:raise AssertionError('Buffers changed')
    audit=dict(inputs=inputs,smoke=args.smoke,no_parameter_updates=True,normal_metrics_max_error=max_metric_error,
        official_loss_max_abs_error=max_loss_error if args.smoke else None,
        official_implementation_checked_in_this_run=args.smoke,
        loss_identity_max_abs_error=max(r['loss_identity_absolute_error'] for r in rows),
        label_replay=label_audits,source_and_buffers_unchanged=True,scope='same-domain frozen-output audit')
    dump(out/'AUDIT.json',em.clean(audit));summarize(rows,out,population,audit)
    if time.monotonic()>deadline:raise TimeoutError('Fixed audit budget reached during summarization')
    dump(out/'COMPLETE.json',dict(completed=True,passed=True,candidates=len(rows),elapsed_s=time.monotonic()-started,
        no_training=True,automatic_followup=False))

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    try:run(args)
    except BaseException as exc:
        dump(Path(args.out)/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        dump(Path(args.out)/'COMPLETE.json',dict(completed=False,passed=False));raise

if __name__=='__main__':main()
